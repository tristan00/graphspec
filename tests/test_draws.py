import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest
import support
import torch
from keyed import Keys, Spaces, dense_keyed, differing, labelled, rows, through, variant_part

from conftest import SPLITS
from graphspec.budgets import GraphLimits
from graphspec.engine.segments import Derivation
from graphspec.infra import compute
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import derive, pipeline
from graphspec.operations import parameters as specs
from graphspec.operations.rows import FACTS
from graphspec.operations.state import ENTITIES
from graphspec.views.graph import LABELS, TASKS, Dense, verify_dense
from tuning.bank.bank import Deriver, grouped
from tuning.batches import inputs

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("draws")
GROUPS = ("coupling", "events", "mixed", "neighbors", "walks")
SAMPLED, PAIRED = 2, 8
LIMITS = GraphLimits("cpu")
DRAWN = {
    "time/periods": "every",
    "time/period_length": 3,
    "nodes/place/filter": "sample",
    "nodes/place/sample": 0.6,
    "nodes/item/filter": "sample",
    "nodes/item/sample": 0.7,
    "derive/hops": 2,
    "derive/rewiring": "sdrf",
    "derive/rewiring_budget": 3,
    "edges/visits/drop": 0.3,
    "edges/visits/persistence": 2,
    "edges/near/cap": 1,
    "edges/near/cap_order": "random",
    "edges/hop:2/drop": 0.3,
    "edges/rewire/drop": 0.2,
    "negatives/of": "link:visits",
    "negatives/rule": "degree",
    "negatives/count": 2,
}
NAMES = support.SEVEN + ("graphspec.negatives",)
TIED = {
    "derive/project:visits:sources:person:person": True,
    "derive/project_cap": 2,
    "derive/hops": 2,
    "derive/diffusion": "ppr",
    "derive/diffusion_top_k": 1,
    "edges/visits/cap": 1,
    "edges/visits/cap_order": "recent",
    "edges/hop:2/cap": 1,
    "edges/hop:2/cap_order": "recent",
    "relation_forms/parallel": "last",
}
WALKS = {"walks/bias": 0.5, "walks/fanout": 3, "walks/window": 6}
CAPS = {"coupling/scheme": "window", "coupling/window": 4, "coupling/cap": 1, "events/cap": 1}
JOINED = {"graphs/join": "single", "graphs/join_size": 2}
COPIES = 40
CROSSING = (
    {"derive/rewiring": "fosr", "derive/rewiring_budget": 3},
    {"derive/diffusion": "ppr", "derive/diffusion_top_k": 1},
    {"derive/diffusion": "heat", "derive/diffusion_top_k": 1},
)
GENERATORS = (
    (np.random, ("default_rng", "Generator", "RandomState", "seed", "random", "rand", "choice", "permutation")),
    (torch, ("Generator", "manual_seed", "rand", "randn", "randint", "randperm", "multinomial", "bernoulli")),
)


class Unread:
    max_facts, max_working_bytes = LIMITS.max_facts, LIMITS.max_working_bytes

    @property
    def device(self):
        raise AssertionError("a derivation read the device")


class Held:
    def __init__(self, arrays):
        self.arrays = arrays

    def numpy(self, key):
        return np.asarray(self.arrays[key])

    def column(self, key, index):
        return np.ascontiguousarray(self.numpy(key)[:, index])


def stated(registry, values, graph):
    started = time.perf_counter()
    entries, rest = support.pipelined(registry, values, NAMES)
    assert rest == {}
    result = pipeline.resolved(registry, entries), specs.configured(
        specs.settings(registry, GROUPS), dict(WALKS, **CAPS, **graph)
    )
    LOG.debug("stated exit settings=%d seconds=%.4f", len(graph), time.perf_counter() - started)
    return result


def permuted(graph, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    arrays = dict(graph.arrays)
    place = rng.permutation(len(graph.kinds))
    for name in ("kinds", "entity_token", "entity_connected"):
        arrays[name] = np.empty_like(np.asarray(arrays[name]))
        arrays[name][place] = np.asarray(graph.arrays[name])
    for name in ENTITIES:
        held = np.asarray(arrays[name])
        arrays[name] = np.where(held >= 0, place[np.maximum(held, 0)], -1).astype(held.dtype)
    keys = [None] * len(place)
    for entity, key in enumerate(graph.keys):
        keys[place[entity]] = key
    for table, reference in (("x_table", "obs_ref"), ("value_table", "fact_ref")):
        rows = rng.permutation(len(arrays[table]))
        arrays[table] = np.empty_like(np.asarray(arrays[table]))
        arrays[table][rows] = np.asarray(graph.arrays[table])
        held = np.asarray(arrays[reference])
        arrays[reference] = np.where(held >= 0, rows[np.maximum(held, 0)], -1).astype(held.dtype)
    order = np.lexsort((arrays["obs_entity"], arrays["obs_step"]))
    for name in ("obs_entity", "obs_step", "obs_ref", "obs_anchor"):
        arrays[name] = np.asarray(arrays[name])[order]
    step, present = np.asarray(arrays["fact_step"]), np.asarray(arrays["fact_present"])
    edge = labelled(step, present, arrays["fact_code"], arrays["fact_source"], arrays["fact_target"])
    order = np.lexsort((np.arange(len(edge)), rng.random(len(edge))[edge], ~present, step))
    for name in FACTS:
        arrays[name] = np.asarray(arrays[name])[order]
    result = Dense(graph.name, keys, graph.steps, **arrays)
    LOG.debug("permuted exit graph=%s seconds=%.4f", graph.name, time.perf_counter() - started)
    return result


def crossing(graph, registry, copies):
    started = time.perf_counter()
    entities = len(graph.kinds)
    shift = np.arange(copies, dtype=np.int64)[:, None] * entities
    arrays = {name: np.asarray(value) for name, value in graph.arrays.items()}
    bridge = (arrays["kinds"][arrays["obs_entity"]] == registry.kind("place")) & ~arrays["obs_anchor"]
    linked = (arrays["obs_entity"][bridge][None, :] + shift[:-1]).reshape(-1)
    bridges = len(linked)
    joined = dict(
        source=np.r_[(arrays["fact_source"][None, :] + shift).reshape(-1), linked],
        target=np.r_[(arrays["fact_target"][None, :] + shift).reshape(-1), linked + entities],
        code=np.r_[np.tile(arrays["fact_code"], copies), np.full(bridges, registry.code("near"))],
        step=np.r_[np.tile(arrays["fact_step"], copies), np.tile(arrays["obs_step"][bridge], copies - 1)],
        present=np.r_[np.tile(arrays["fact_present"], copies), np.ones(bridges, bool)],
        ref=np.r_[np.tile(arrays["fact_ref"], copies), np.full(bridges, -1)],
    )
    facts = np.lexsort((~joined["present"], joined["step"]))
    entity, step = (arrays["obs_entity"][None, :] + shift).reshape(-1), np.tile(arrays["obs_step"], copies)
    observations = np.lexsort((entity, step))
    count = len(arrays["task_key"])
    tasks = np.argsort(np.tile(arrays["task_step"], copies), kind="stable")
    placed = np.argsort(tasks)
    owner = placed[(arrays["member_task"][None, :] + np.arange(copies)[:, None] * count).reshape(-1)]
    members = np.argsort(owner, kind="stable")
    ends = {}
    for name in ("member_source", "member_target"):
        values = np.tile(arrays[name], copies)
        ends[name] = np.where(values >= 0, values + np.repeat(shift[:, 0], len(arrays[name])), -1)
    keys = arrays["task_key"][None, :] + np.arange(copies)[:, None] * (int(arrays["task_key"].max(initial=-1)) + 1)
    copied = dict(
        kinds=np.tile(arrays["kinds"], copies),
        entity_token=np.tile(arrays["entity_token"], copies)
        + np.repeat(np.arange(copies, dtype=np.uint64), entities) * np.uint64(0x9E3779B97F4A7C15),
        entity_connected=np.tile(arrays["entity_connected"], copies),
        obs_entity=entity[observations],
        obs_step=step[observations],
        obs_ref=np.tile(arrays["obs_ref"], copies)[observations],
        obs_anchor=np.tile(arrays["obs_anchor"], copies)[observations],
        **{"fact_" + name: values[facts] for name, values in joined.items()},
        **{name: np.concatenate([arrays[name]] * copies)[tasks] for name in TASKS if name != "task_key"},
        task_key=keys.reshape(-1)[tasks],
        member_task=owner[members],
        **{name: np.tile(arrays[name], copies)[members] for name in ("member_name", "member_place", "member_code")},
        **{name: values[members] for name, values in ends.items()},
        **{name: arrays[name][:0] for name in LABELS},
    )
    arrays.update({name: value.astype(arrays[name].dtype) for name, value in copied.items()})
    result = Dense(graph.name, [], graph.steps, **arrays)
    verify_dense(result, registry)
    LOG.debug("crossing exit entities=%d seconds=%.4f", len(result.kinds), time.perf_counter() - started)
    return result


def crossed(registry, values):
    started = time.perf_counter()
    entries, rest = support.pipelined(registry, values, ("graphspec.derive",))
    assert rest == {}
    steps = pipeline.resolved(registry, entries)
    result = steps, specs.configured(specs.settings(registry, GROUPS), dict(WALKS, **CAPS))
    LOG.debug("crossed exit seconds=%.4f", time.perf_counter() - started)
    return result


def components(graph, registry):
    started = time.perf_counter()
    held = derive.derivable(graph, registry)
    facts = np.searchsorted(graph.fact_step, np.arange(graph.steps + 1))
    observed = np.searchsorted(graph.obs_step, np.arange(graph.steps + 1))
    sizes = []
    for step in range(graph.steps):
        span = np.arange(facts[step], facts[step + 1])
        rows = span[held[span]]
        u, v = derive.undirected(graph.fact_source[rows], graph.fact_target[rows], len(graph.kinds))
        low, high = observed[step], observed[step + 1]
        present = np.sort(graph.obs_entity[low:high][~graph.obs_anchor[low:high]])
        sizes += [len(members) for members, _, _ in derive.component_edges(present, u, v)]
    LOG.debug("components exit components=%d seconds=%.4f", len(sizes), time.perf_counter() - started)
    return sizes


def variant(members, registry, steps, spec, cuts, limits):
    started = time.perf_counter()
    derivation = Derivation.begun(members, {}, registry, steps, spec, GROUPS, support.SEED, limits, None, 0)
    for high in cuts + [sum(member.steps for member in members)]:
        derivation = derivation.continued(high, None)
    result = derivation.derived(), derivation.arrays()
    LOG.info("variant exit graphs=%d cuts=%d seconds=%.3f", len(members), len(cuts), time.perf_counter() - started)
    return result


def keyed(found, registry):
    started = time.perf_counter()
    dense, arrays = found
    keys = Keys(dense, registry)
    held = variant_part(arrays, keys, list(arrays))
    result = dict(dense_keyed(dense, keys), **{"variant/" + name: value for name, value in held.items()})
    LOG.debug("keyed exit arrays=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def derived(members, registry, steps, spec, cuts, limits):
    started = time.perf_counter()
    result = keyed(variant(members, registry, steps, spec, cuts, limits), registry)
    LOG.debug("derived exit graphs=%d seconds=%.4f", len(members), time.perf_counter() - started)
    return result


def levels(tree):
    started = time.perf_counter()
    found = []
    while tree is not None:
        found.append(("x", tree["x"].segments[0][2][0][1]))
        found += [(name, tree[name]) for name in ("mask", "gap", "relation") if name in tree]
        tree = tree.get("child")
    LOG.debug("levels exit arrays=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def sampled(found, registry, order):
    started = time.perf_counter()
    dense, arrays = found
    keys = Keys(dense, registry)
    spaces, held = Spaces(arrays, keys), Held(arrays)
    spec = {"neighbors/order": order, "neighbors/decay": 0.5}
    window = SimpleNamespace(graph=0, held=held, start=0, registry=registry, cfg=dict(graph=spec), seed=support.SEED)
    owner, kinds = arrays["member_row"], arrays["row_kind"]
    pair = inputs.paired(registry, kinds, owner, arrays["member_name"])
    b = dict(
        member_row=owner,
        member_entity=arrays["member_entity"],
        row_steps=arrays["row_step"].astype(np.int64),
        row_kind=kinds,
        pair=pair,
        partner=inputs.partnered(pair, len(owner)),
    )
    history = inputs.history_inputs(window, b, SAMPLED, True, None)
    cutoffs = b["row_steps"][owner] * 2 + 1
    tree = inputs.tree(
        0, held, b["member_entity"].astype(np.int64), cutoffs, 0, SAMPLED, 2, registry, spec, support.SEED
    )
    x, task = (piece[1] for piece in history["history_x"].segments[0][2])
    paired = inputs.history_inputs(window, b, PAIRED, False, None)
    result = dict(
        history=rows(
            spaces.member,
            through(keys.entities, history["history_ids"]),
            through(spaces.observations, x),
            through(keys.tasks, task),
            *(history[name] for name in ("history_mask", "history_gap", "history_relation", "occurrence", "structure")),
        ),
        common=rows(
            spaces.rows,
            through(keys.entities, paired["common_ids"]),
            *(paired[name] for name in ("common_frequency", "common_stats", "pair_gap", "pair_mask")),
        ),
        tree=rows(
            spaces.member,
            *(
                (through(spaces.observations, value) if name == "x" else value).reshape(len(owner), -1)
                for name, value in levels(tree)
            ),
        ),
    )
    full, shared = int(history["history_mask"].all(1).sum()), int(((paired["common_ids"] >= 0).sum(1) > 1).sum())
    LOG.info(
        "sampled exit order=%s full=%d shared=%d seconds=%.3f", order, full, shared, time.perf_counter() - started
    )
    return result, full, shared


@pytest.fixture(scope="module")
def drawn(bank):
    opened = Deriver(bank, "cpu", support.SEED, RESIDENT)
    fitted = support.fitted(opened)
    registry = pipeline.grown(fitted, stated(fitted, DRAWN, {})[0], None)
    views = [support.shown(opened, index).dense() for index in range(4)]
    cfg = support.configuration(
        opened.registry, dict(view="snapshot"), dict(support.several(opened.registry), **JOINED), 5, SPLITS
    )
    steps, spec = stated(fitted, DRAWN, {})
    reference = derived(views[:1], registry, steps, spec, [], LIMITS)
    return registry, views, reference, (opened.viewed(cfg).graphs, opened.groups(cfg)), fitted


@pytest.fixture(scope="module")
def unlinked(drawn):
    registry, views, _, _, fitted = drawn
    moved = [permuted(graph, seed) for seed, graph in enumerate(views)]
    separate = dict(JOINED, **{"graphs/connect/" + kind: False for kind in registry.nodes})
    steps, spec = stated(fitted, DRAWN, separate)
    whole = variant(views[:2], registry, steps, spec, [], LIMITS)
    turned = variant(moved[1::-1], registry, steps, spec, [5, views[1].steps + 2], LIMITS)
    return moved, whole, turned


def test_permuting_entities_and_graphs_leaves_every_derived_array_equal_after_sorting_by_key(drawn, unlinked):
    started = time.perf_counter()
    registry, views, reference, (entries, groups), fitted = drawn
    moved, whole, turned = unlinked
    graphs = views[:2]
    steps, spec = stated(fitted, DRAWN, {})
    assert differing(derived(moved[:1], registry, steps, spec, [4, 9], LIMITS), reference) == []
    assert len(reference["variant/walks"]) and registry.code("rewire") in reference["facts"][:, 2]
    assert len(reference["variant/candidates"]) and "negatives" in registry.by_candidate
    assert differing(keyed(turned, registry), keyed(whole, registry)) == []
    steps, spec = stated(fitted, DRAWN, dict(JOINED, **{"graphs/connect_fraction": 0.5}))
    linked = derived(graphs, registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[:2], registry, steps, spec, [graphs[0].steps + 3], LIMITS), linked) == []
    assert set(linked["entities"][:, -1].tolist()) == {0, 1}
    order = [2, 0, 3, 1]
    keys = [(0, 1)] * len(entries)
    turned = grouped([dict(entries[old], index=new) for new, old in enumerate(order)], keys, 2)
    assert grouped(entries, keys, 2) == groups
    assert [[entries[order[new]]["name"] for new in group] for group in turned] == [
        [entries[index]["name"] for index in group] for group in groups
    ]
    last = derived([views[index] for index in groups[-1]], registry, steps, spec, [], LIMITS)
    placed = [moved[order[new]] for new in turned[-1]]
    assert differing(derived(placed, registry, steps, spec, [5], LIMITS), last) == []
    steps, spec = stated(fitted, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    fosr = derived(graphs[1:], registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[1:2], registry, steps, spec, [7], LIMITS), fosr) == []
    steps, spec = stated(fitted, TIED, {})
    tied = derived(graphs[:1], registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[:1], registry, steps, spec, [4], LIMITS), tied) == []
    LOG.info("permuted draws exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.parametrize("order", ["recent", "uniform"])
def test_the_neighbours_a_batch_reads_are_equal_when_entities_are_renumbered_or_graphs_reordered(
    drawn, unlinked, order
):
    registry = drawn[0]
    _, whole, turned = unlinked
    found, full, shared = sampled(whole, registry, order)
    assert full and shared and differing(sampled(turned, registry, order)[0], found) == []


def test_no_draw_reads_a_random_generator_or_a_device(drawn, monkeypatch):
    registry, views, reference, _, fitted = drawn
    for module, names in GENERATORS:
        for name in names:
            monkeypatch.setattr(module, name, lambda *args, name=name, **kwargs: pytest.fail("a draw used " + name))
    steps, spec = stated(fitted, DRAWN, {})
    assert differing(derived(views[:1], registry, steps, spec, [6], Unread()), reference) == []


def test_the_case_of_the_gpu_check_gives_dense_algebra_the_rows_from_which_it_runs_on_the_gpu(drawn):
    started = time.perf_counter()
    views, fitted = drawn[1], drawn[4]
    graph = crossing(views[0], fitted, COPIES)
    for values in CROSSING:
        steps, _ = crossed(fitted, values)
        assert [step.name for step in steps] == ["graphspec.derive"]
    sizes = components(graph, fitted)
    assert max(sizes) >= compute.GPU_FROM > max(components(views[0], fitted))
    LOG.info("crossing exit largest=%d seconds=%.3f", max(sizes), time.perf_counter() - started)


@pytest.mark.gpu
def test_the_arrays_of_every_draw_are_equal_on_cpu_and_on_gpu(drawn, monkeypatch):
    registry, views, reference, _, fitted = drawn
    steps, spec = stated(fitted, DRAWN, {})
    assert differing(derived(views[:1], registry, steps, spec, [], GraphLimits("cuda")), reference) == []
    steps, spec = stated(fitted, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    on = [derived(views[1:2], registry, steps, spec, [], GraphLimits(device)) for device in ("cpu", "cuda")]
    assert differing(on[1], on[0]) == []
    graph, used, chosen = crossing(views[0], fitted, COPIES), [], compute.device_for
    monkeypatch.setattr(compute, "device_for", lambda size, device: used.append(chosen(size, device)) or used[-1])
    for values in CROSSING:
        steps, spec = crossed(fitted, values)
        grown = pipeline.grown(fitted, steps, None)
        on = [derived([graph], grown, steps, spec, [], GraphLimits(device)) for device in ("cpu", "cuda")]
        assert differing(on[1], on[0]) == []
    assert "cuda" in used
