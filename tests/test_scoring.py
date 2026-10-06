import copy
import gc
import logging
import time

import pytest
import support
import torch

from conftest import SPLITS, shared
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.compute import single
from graphspec.infra.timing import timed
from graphspec.schema.decisions import RECORDED
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import pack_batch
from tuning.families.heads import DEPTH, DEPTHS, NONE, REALISATION, realisation
from tuning.families.models import build as network
from tuning.families.periods import blocks as windows
from tuning.fits.guards import captured

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("scoring")
COUNTS = (1, 2, 8, 50)
CHECKED = COUNTS[1:]
SHIPPED = ("node", "vector", "action_nodes", "actor_edge", "set_edge")
SPOTS = 52
STEPS = 32
ROUNDS, MARGIN = (12, 28), 1.5


@timed("scoring.schema")
def schema():
    stated = synthetic.decision_schema()
    kind = copy.deepcopy(stated["tasks"]["kinds"]["show"])
    kind["target"].update(
        actions=dict(go=dict(roles=dict(to=dict(sort="node", kinds=["spot"], count=[1, 1])), parameters=[])),
        series=["gain"],
        utility=dict(name="weighted", better="higher", weights=dict(gain=1.0)),
    )
    stated["columns"] = dict(load=stated["columns"]["load"])
    stated["nodes"] = dict(agent=stated["nodes"]["agent"], spot=stated["nodes"]["spot"])
    stated["tasks"]["references"] = dict(agents=stated["tasks"]["references"]["agents"])
    stated["tasks"]["kinds"] = dict(pick=kind)
    return stated


def graph(count):
    started = time.perf_counter()
    spots, agents, moment = ["s%02d" % at for at in range(SPOTS)], ["a0", "a1"], STEPS - 1
    options = [
        synthetic.offered("o%02d" % at, [synthetic.acting("go", "a0", dict(to=[spots[at]]), {})], "set", RECORDED)
        for at in range(count)
    ]
    for at, option in enumerate(options):
        option.update(chosen=at == 0, outcome=dict(gain=0.5 * at + 0.1 * count))
    tasks = dict(
        key=[1],
        kind=["pick"],
        time=[moment],
        split=["validation" if count == 1 else "train"],
        weight=[None],
        target=[dict(options=options, concurrent=[], next={}, ended=True, discount=None)],
        known=[moment + 1],
        references=dict(agents=[["a0"]]),
        signal=[[0.5]],
        signal_observed=[[True]],
    )
    ticks = [tick for tick in range(STEPS) for _ in spots]
    loads = dict(
        key=spots * STEPS,
        since=ticks,
        until=[tick + 1 if tick + 1 < STEPS else None for tick in ticks],
        load=[0.1 * at + tick for tick in range(STEPS) for at in range(SPOTS)],
    )
    chained = len(spots) - 1
    near = dict(source=spots[:-1], target=spots[1:], present=[True] * chained, since=[0] * chained, until=[None] * chained)
    base = dict(source=agents, target=[spots[0], spots[SPOTS // 2]], present=[True] * 2, since=[0] * 2, until=[None] * 2)
    weighed = dict(key=agents, since=[0, 0], until=[None, None], load=[1.0, 2.0])
    result = dict(
        name="k%d" % count,
        unit="tick",
        columns={},
        nodes=dict(agent=agents, spot=spots),
        values=[
            dict(form="state", parameters={}, rows=weighed, kind="agent"),
            dict(form="state", parameters={}, rows=loads, kind="spot"),
        ],
        edges=[
            dict(form="state", parameters={}, rows=near, table="near", ends=("spot", "spot")),
            dict(form="state", parameters={}, rows=base, table="base", ends=("agent", "spot")),
        ],
        tasks=tasks,
    )
    LOG.debug("scoring graph exit options=%d seconds=%.4f", count, time.perf_counter() - started)
    return result


@timed("scoring.made")
def made(root):
    case = dict(schema=schema(), graphs=[graph(count) for count in COUNTS])
    build(synthetic.stage(root, "scoring", case), root / "bank", 1, ("train",))


@timed("scoring.configured")
def configured(held, name, history):
    deep = {DEPTH: max(DEPTHS)} if name != NONE and realisation(name).structure else {}
    values = dict(view="snapshot", **{REALISATION: name}, **deep)
    graph = {"time/history": history, "time/steps": "changes", **support.several(held.registry)}
    return support.configuration(held.registry, values, graph, 5, SPLITS)


@timed("scoring.packed")
def packed(held, cfg):
    history = cfg["graph"]["time/history"]
    blocks = sorted(block for partition in (0, 1) for block in held.blocks(partition, history, cfg))
    names = {entry["index"]: entry["name"] for entry in held.viewed(cfg).graphs}
    return {names[block[0]]: pack_batch(held, windows(held, [block], cfg), cfg, "cpu") for block in blocks}


@pytest.fixture(scope="module")
def prepared():
    started = time.perf_counter()
    held = Deriver(shared("scoring", made) / "bank", "cpu", support.SEED, RESIDENT)
    plain = packed(held, configured(held, NONE, STEPS))
    nets, batches, acting = {}, {}, {}
    for name in SHIPPED:
        cfg = configured(held, name, STEPS)
        torch.manual_seed(5)
        nets[name] = network(cfg, held.fitted(cfg)).eval()
        if realisation(name).actor not in acting:
            acting[realisation(name).actor] = packed(held, cfg)
        batches[name] = acting[realisation(name).actor]
    found = dict(plain=plain, nets=nets, batches=batches)
    LOG.info("scoring prepared exit seconds=%.3f", time.perf_counter() - started)
    yield found
    found.clear()


def clocked(make):
    gc.disable()
    try:
        clock = time.perf_counter()
        make()
        return time.perf_counter() - clock
    finally:
        gc.enable()


def sampled(found, rounds, held, caught):
    started = time.perf_counter()
    nets, batches, body = held["nets"], held["batches"], held["nets"]["node"]
    measured = [(("alone", name), lambda name=name: nets[name](batches[name]["k1"])) for name in SHIPPED]
    for count in CHECKED:
        graph = "k%d" % count
        measured.append(("history", lambda graph=graph: captured(body, held["plain"][graph])))
        for name in SHIPPED:
            b, (features, members) = batches[name][graph], caught[name, count]
            measured.append(((name, count), lambda name=name, b=b: nets[name](b)))
            measured.append((("head", name, count), lambda name=name, b=b, f=features, m=members: nets[name].head(f, b, m)))
    for repeat in range(rounds):
        for key, make in measured[:: 1 if repeat % 2 else -1]:
            found.setdefault(key, []).append(clocked(make))
    LOG.debug("scoring sampled exit rounds=%d seconds=%.4f", rounds, time.perf_counter() - started)


def estimated(found):
    started = time.perf_counter()
    least = {key: min(values) for key, values in found.items()}
    bound = {(name, count): least["history"] + least["head", name, count] for name in SHIPPED for count in CHECKED}
    ratios = {key: least[key] / bound[key] for key in bound}
    wrong = {(name, count): count * least["alone", name] / bound[name, count] for name, count in bound}
    LOG.debug("scoring estimated exit samples=%d seconds=%.4f", len(found["alone", SHIPPED[0]]), time.perf_counter() - started)
    return least, ratios, wrong


def test_k_options_take_one_encoding_of_the_history_and_k_of_the_last_step_and_one_encoding_per_option_does_not(prepared):
    started = time.perf_counter()
    held, found, settled = prepared, {}, False
    with torch.no_grad(), single():
        caught = {(name, count): captured(held["nets"][name], held["batches"][name]["k%d" % count]) for name in SHIPPED for count in CHECKED}
        for name in SHIPPED:
            for b in held["batches"][name].values():
                held["nets"][name](b)
        for rounds in ROUNDS:
            if not settled:
                sampled(found, rounds, held, caught)
                least, ratios, wrong = estimated(found)
                over, under = max(ratios.values()), min(wrong.values())
                settled = over <= MARGIN < under
    largest = {count: round(max(ratios[name, count] for name in SHIPPED), 3) for count in CHECKED}
    smallest = {count: round(min(wrong[name, count] for name in SHIPPED), 3) for count in CHECKED}
    shown = (
        "by K the largest ratio %s and the smallest per option %s; the bound is one encoding of the history, %.3f ms, "
        "plus the head scoring the K options alone; by realisation and K the ratios to it of K options and of K times one: %s"
    ) % (
        largest,
        smallest,
        1e3 * least["history"],
        {"%s %d" % key: (round(ratios[key], 3), round(wrong[key], 3)) for key in ratios},
    )
    LOG.info("scoring of options exit rounds=%d seconds=%.3f; %s", len(found["alone", SHIPPED[0]]), time.perf_counter() - started, shown)
    assert over <= MARGIN, "scoring K options took up to %.4f times the bound, above %.1f; %s" % (over, MARGIN, shown)
    assert under > MARGIN, "scoring each option alone took only %.4f times the bound, not above %.1f; %s" % (under, MARGIN, shown)


def test_all_options_of_a_decision_read_one_encoding_of_the_history_whatever_their_number(prepared):
    started = time.perf_counter()
    held = prepared
    for name in SHIPPED:
        net, seen = held["nets"][name], []
        hooks = [
            net.input.register_forward_hook(lambda module, given, output: seen.append(("input", len(given[0])))),
            net.spatial[0].register_forward_hook(lambda module, given, output: seen.append(("spatial", given[1].shape[1]))),
        ]
        try:
            with torch.no_grad():
                for count in CHECKED:
                    net(held["batches"][name]["k%d" % count])
                    captured(net, held["plain"]["k%d" % count])
        finally:
            for hook in hooks:
                hook.remove()
        assert len(seen) == 4 * len(CHECKED) and len(set(seen)) == 2 and seen[:2] * 2 * len(CHECKED) == seen, (name, seen)
    LOG.info("one encoding of the history exit seconds=%.3f", time.perf_counter() - started)
