import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pytest
import registered
import support
from support import SEED, configuration

from graphspec import (
    ABOUT,
    ENCODERS,
    FORMS,
    HELD,
    NEGATIVES,
    SEES,
    SEVERAL,
    SORTS,
    STEPS,
    STORAGE,
    TIES,
    TYPES,
    registries,
)
from graphspec.build.forms import Form
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.processes import processes
from graphspec.infra.processes.processes import guarded, parallel, spawned
from graphspec.infra.timing import timed
from graphspec.operations.negatives import Rule
from graphspec.registries import carried, restored
from graphspec.schema.schema import VERSION
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.bank.variants import prepare
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("workers")
MAKERS, SITES = ("m0", "m1", "m2", "m3"), ("s0", "s1", "s2", "s3", "s4", "s5")
TIMES = (1, 2, 3, 4, 5, 6, 7, 8)
SPLITS = dict(trained=["alpha"], scored=["omega"], sees=dict(alpha=["everything"], omega=[]))
GRAPH = {
    "time/history": 4,
    "time/steps": "own",
    "time/several": "middle",
    "sets/bundle/storage": "stored",
}
ENTRIES = (
    (TYPES, "count", registered.Count()),
    (ENCODERS, "halved", registered.Halved()),
    (HELD, "counted", registered.Counted()),
    (TIES, "rank", registered.Rank()),
    (SEVERAL, "middle", registered.middle),
    (ABOUT, "kept", registered.Kept()),
    (FORMS, "spanned", Form(registered.spanned, False)),
    (SORTS, "triple", registered.TRIPLE),
    (STEPS, "own", registered.Own()),
    (SEES, "everything", registered.everything),
    (STORAGE, "stored", registered.STORED),
    (NEGATIVES, "parity", registered.PARITY),
)
VALUE = dict(type="number", encoder="value")


@timed("workers.stated")
def stated():
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="rank"),
        observers=False,
        columns=dict(size=dict(type="count", encoder="halved"), load=VALUE, level=VALUE, mass=VALUE),
        graph=dict(columns=[]),
        nodes=dict(
            maker=dict(
                columns=["size"],
                holds=dict(nodes=dict(part=dict(columns=["mass"])), edges={}, encoder="counted"),
            ),
            site=dict(columns=["load"], holds=None),
        ),
        sets=dict(bundle=dict(kinds=["site"], roles=["one", "two"], members=[], columns=["level"])),
        edges=dict(
            serves=synthetic.edge("maker", "site", []),
            note=dict(
                endpoints=[["bundle", "maker"]], directed=True, columns=[], relation="declared", about="kept"
            ),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(unit=dict(sort="node", kinds=["maker"]), trio=dict(sort="triple", kinds=["site"])),
            kinds=dict(
                ask=dict(
                    references=["unit", "trio"],
                    columns=[],
                    target=dict(kind="number", series="out", about="unit", utility=synthetic.valued(["out"])),
                    weight=False,
                    absent="stop",
                    event=None,
                    matures=None,
                )
            ),
        ),
    )


@timed("workers.lasting")
def lasting(kind, keys, column, values, seq):
    rows = dict(key=list(keys), since=[0] * len(keys), until=[None] * len(keys), seq=seq, **{column: values})
    return dict(form="state", parameters={}, rows=rows, kind=kind)


@timed("workers.graph")
def graph(place):
    makers = dict(
        key=list(MAKERS) + ["m0"],
        at=[0, 0, 0, 0, 3],
        seq=[0, 1, 2, 3, 4],
        size=[1 + at + place for at in range(4)] + [9],
    )
    values = [
        dict(form="spanned", parameters=dict(length=40), rows=makers, kind="maker"),
        lasting("site", SITES, "load", [1.0 + at for at in range(6)], list(range(10, 16))),
        lasting("bundle", ("b0", "b1"), "level", [0.5, 1.5 + place], [20, 21]),
    ]
    serves = dict(
        source=list(MAKERS),
        target=list(SITES[:4]),
        present=[True] * 4,
        since=[0] * 4,
        until=[None] * 4,
        seq=list(range(30, 34)),
    )
    notes = dict(source=["b0"], target=["m1"], present=[True], since=[0], until=[None], seq=[40])
    edges = [
        dict(form="state", parameters={}, rows=serves, table="serves", ends=("maker", "site")),
        dict(form="state", parameters={}, rows=notes, table="note", ends=("bundle", "maker")),
    ]
    count = len(TIMES)
    tasks = dict(
        key=list(range(count)),
        kind=["ask"] * count,
        time=list(TIMES),
        split=["alpha" if at % 2 == 0 else "omega" for at in range(count)],
        weight=[None] * count,
        references=dict(
            unit=[MAKERS[at % 4] for at in range(count)],
            trio=[[SITES[(at + step) % 6] for step in range(3)] for at in range(count)],
        ),
        target=[0.5 * at + place for at in range(count)],
        known=[None] * count,
    )
    parts = dict(owner=[], kind=[], key=[], mass=[])
    for number, owner in enumerate(MAKERS):
        for at in range(1 + number):
            parts["owner"].append(owner)
            parts["kind"].append("part")
            parts["key"].append("p%d" % at)
            parts["mass"].append(1.0 + at)
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(maker=list(MAKERS), site=list(SITES), bundle=["b0", "b1"]),
        values=values,
        edges=edges,
        sets=dict(bundle=dict(set=["b0", "b0", "b1", "b1"], member=list(SITES[:4]), role=["one", "two"] * 2)),
        held=dict(maker=dict(nodes=parts, edges=dict(owner=[], source=[], target=[], relation=[]))),
        tasks=tasks,
    )


@pytest.fixture(scope="module")
def entries():
    for names, name, entry in ENTRIES:
        names.register(name, entry)
    yield ENTRIES
    for names, name, _ in ENTRIES:
        del names.held[name]


@timed("workers.counted")
def counted(pools, *arguments, **named):
    pools.append(named)
    return ProcessPoolExecutor(*arguments, **named)


@pytest.fixture(scope="module")
def spread(entries, tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("workers")
    case = dict(schema=stated(), graphs=[graph(place) for place in range(2)])
    pools = []
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(processes, "ProcessPoolExecutor", partial(counted, pools))
        build(synthetic.stage(root, registered.NAME, case), root / "bank", 2, ("alpha",))
    LOG.info("bank of workers exit seconds=%.3f", time.perf_counter() - started)
    return root / "bank", pools


def test_an_entry_registered_in_every_registry_reaches_the_workers_of_a_build_and_a_fit(
    spread, tmp_path, statement
):
    bank, pools = spread
    assert [pool["max_workers"] for pool in pools] == [2]
    held = Deriver(bank, "cpu", SEED, RESIDENT)
    assert held.manifest["spec"]["encoding"]["states"] is None
    splits = copy.deepcopy(SPLITS)
    cfg = dict(configuration(held.registry, dict(view="snapshot"), GRAPH, 5, splits), epochs=1)
    prepare(bank, cfg, 1, "cpu", SEED)
    assert held.fitted(cfg).encoding.states["size"] == dict(most=5.0)
    blocks = held.blocks(0, 4, cfg)
    given = state(dict(statement, workers=2, **splits))
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], given)
    LOG.info("fit with two workers exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert np.isfinite(saved["measures"]["mse"]["value"]) and saved["holdout_rows"] == len(held.members(cfg, 1))


def test_a_worker_holds_exactly_the_entries_its_caller_holds_or_stops_naming_one_whose_code_differs(
    entries, monkeypatch
):
    from importlib import import_module

    from graphspec.registries import REGISTRIES

    assert all(any(held is names for held in vars(import_module(names.module)).values()) for names in REGISTRIES)
    keyed = {(NEGATIVES.label, "parity")}
    with registries.keyed(keyed):
        started = spawned(registered.ready, (), carried, restored)
    assert registries.READ.held is None
    monkeypatch.setattr(registries.READ, "held", None)
    for names in REGISTRIES:
        monkeypatch.setattr(names, "held", support.entries(names))
    monkeypatch.setattr(processes, "FAILED", [])
    NEGATIVES.held["parity"] = Rule(registered.parities, registered.parity, 9)
    SEVERAL.held["stale"] = registered.middle
    del STEPS.held["changes"]
    started["initializer"](*started["initargs"])
    assert guarded(abs, -1) == 1 and registries.READ.held == keyed and registries.READ.frame is None
    assert NEGATIVES.named("parity").version == 1 and "stale" not in SEVERAL.names() and "changes" in STEPS.names()
    assert all(names.named(name) is names.shipped[name] for names in REGISTRIES for name in names.shipped)
    monkeypatch.setattr(registered.Kept, "version", 2)
    started["initializer"](*started["initargs"])
    with pytest.raises(
        ValueError,
        match="the rule for an edge about an edge kept is at version 1 in the process that called and at 2 in this "
        "worker, whose code differs",
    ):
        guarded(abs, -1)


def test_an_entry_that_cannot_reach_a_worker_stops_naming_its_registry_and_name(entries, monkeypatch):
    class Local:
        version = 1

    SEES.register("local", Local())
    try:
        with pytest.raises(
            ValueError,
            match="the rule of what a split may see local is registered from outside the library and cannot be "
            "passed to the processes the library starts",
        ):
            parallel(abs, [1, 2], 2, registered.ready, (), carried, restored)
    finally:
        del SEES.held["local"]
    registered.Vanished = type("Vanished", (), {"__module__": registered.__name__, "version": 1})
    SEES.register("vanished", registered.Vanished())
    try:
        started = spawned(registered.ready, (), carried, restored)
    finally:
        del SEES.held["vanished"]
        del registered.Vanished
    assert {
        (label, name)
        for label, _, held in started["initargs"][0][0]
        if held is not None
        for name, pickled, _ in held
        if pickled is not None
    } == {(names.label, name) for names, name, _ in ENTRIES} | {(SEES.label, "vanished")}
    monkeypatch.setattr(processes, "FAILED", [])
    monkeypatch.setattr(SEES, "held", support.entries(SEES))
    started["initializer"](*started["initargs"])
    with pytest.raises(AttributeError, match="Vanished"):
        guarded(abs, -1)
