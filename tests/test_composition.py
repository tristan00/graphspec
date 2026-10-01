import copy
import gc
import logging
import time

import pytest
import support
import torch

from conftest import shared
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import batches, pack_batch
from tuning.families.model_spec import VIEWS, views
from tuning.families.models import build as network
from tuning.families.periods import blocks as period_blocks

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("composition")
USERS, ITEMS = ("u0", "u1", "u2", "u3"), ("i0", "i1", "i2", "i3", "i4")
TIMES, CHANGED = (1, 2, 3, 4, 5, 6), 3
SPLITS = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
ROUNDING = 1e-5
GRAPH = {"time/history": 2, "time/context": 1, "time/periods": "every", "time/period_length": 2}
SPECS = [dict(view=view) for view in VIEWS] + [{"view": "snapshot", "snapshot/temporal": "graph_sequence"}]
OPENED = {}


@timed("composition.schema")
def schema():
    number = dict(type="number", encoder="value")
    kinds = dict(
        score=dict(
            references=["who"],
            columns=[],
            target=dict(kind="number", series="score", about="who", utility=synthetic.valued(["score"])),
            weight=False,
            absent="stop",
            event=None,
            matures=None,
        ),
        visit=dict(
            references=["who", "at"], columns=[], target=None, weight=False, absent="stop", event=["at", "who"], matures=None
        ),
    )
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(load=number, price=number),
        graph=dict(columns=[]),
        nodes=dict(user=dict(columns=["load"], holds=None), item=dict(columns=["price"], holds=None)),
        sets={},
        edges=dict(buys=synthetic.edge("user", "item", []), near=synthetic.edge("item", "item", [], directed=False)),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=1,
            references=dict(who=dict(sort="node", kinds=["user"]), at=dict(sort="node", kinds=["item"])),
            kinds=kinds,
        ),
    )


@timed("composition.graph")
def graph(changed):
    tasks = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], signal=[], signal_observed=[])
    references = dict(who=[], at=[])
    for at, moment in enumerate(TIMES):
        for kind in ("score", "visit"):
            tasks["key"].append(len(tasks["key"]))
            tasks["kind"].append(kind)
            tasks["time"].append(moment)
            tasks["split"].append(None if kind == "visit" else "train" if at < CHANGED else "validation")
            tasks["weight"].append(None)
            tasks["known"].append(None)
            tasks["signal"].append([0.1 * at])
            tasks["signal_observed"].append([True])
            tasks["target"].append(None if kind == "visit" else float(at) + 0.5 * (at % 2))
            references["who"].append(USERS[min(at, CHANGED)])
            moved = 2 * (changed and at == CHANGED)
            references["at"].append(ITEMS[(at + moved) % len(ITEMS)] if kind == "visit" else None)
    users = dict(key=list(USERS), since=[0] * 4, until=[None] * 4, load=[1.0 + at for at in range(4)])
    items = dict(
        key=list(ITEMS) + ["i1"],
        since=[0] * 5 + [5],
        until=[None, 3, None, None, None, None],
        price=[2.0 * at for at in range(5)] + [3.0],
    )
    edges = [("u0", "i2", 1), ("u1", "i2", 2), ("u2", "i0", 2)] + ([("u3", "i4", TIMES[CHANGED] + 1)] if changed else [])
    buys = dict(
        source=[edge[0] for edge in edges],
        target=[edge[1] for edge in edges],
        present=[True] * len(edges),
        since=[edge[2] for edge in edges],
        until=[None] * len(edges),
    )
    near = dict(source=["i0", "i3"], target=["i2", "i4"], present=[True] * 2, since=[0, 0], until=[None] * 2)
    return dict(
        name="g0",
        unit="tick",
        columns={},
        nodes=dict(user=list(USERS), item=list(ITEMS)),
        values=[
            dict(form="state", parameters={}, rows=users, kind="user"),
            dict(form="state", parameters={}, rows=items, kind="item"),
        ],
        edges=[
            dict(form="state", parameters={}, table="buys", ends=("user", "item"), rows=buys),
            dict(form="state", parameters={}, table="near", ends=("item", "item"), rows=near),
        ],
        tasks=dict(tasks, references=references),
    )


@timed("composition.made")
def made(root):
    for name in ("stated", "changed"):
        build(synthetic.stage(root, name, dict(schema=schema(), graphs=[graph(name == "changed")])), root / name, 1, ("train",))


@pytest.fixture(scope="module")
def banks():
    root = shared("composition", made)
    yield {name: root / name for name in ("stated", "changed")}
    OPENED.clear()
    gc.collect()


@timed("composition.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", RESIDENT)
    return OPENED[name]


@timed("composition.predicted")
def predicted(held, cfg, net, together):
    blocks = period_blocks(held, [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)], cfg)
    groups = batches(blocks, len(blocks), held.sizes(cfg), None, frame_budget=1024) if together else [[b] for b in blocks]
    found = {}
    with torch.inference_mode():
        for group in groups:
            b = pack_batch(held, group, cfg, "cpu")
            rows = b["rows"].tolist()
            found.update(zip([held.rows[row]["time"] for row in rows], net(b).tolist()[: len(rows)]))
    return found


@pytest.mark.parametrize("values", SPECS, ids=["-".join(str(value) for value in spec.values()) for spec in SPECS])
def test_a_prediction_depends_neither_on_the_windows_beside_it_nor_on_a_fact_after_its_time(banks, values):
    started = time.perf_counter()
    stated, changed = opened(banks, "stated"), opened(banks, "changed")
    assert values["view"] in views(stated.registry)
    cfg = support.configuration(stated.registry, values, GRAPH, 5, copy.deepcopy(SPLITS))
    torch.manual_seed(5)
    net = network(cfg, stated.fitted(cfg)).eval()
    together, alone, later = predicted(stated, cfg, net, True), predicted(stated, cfg, net, False), predicted(changed, cfg, net, True)
    assert sorted(together) == sorted(alone) == sorted(later) == list(TIMES)
    beside = [moment for moment in TIMES if abs(together[moment] - alone[moment]) > ROUNDING]
    assert beside == [], "predictions that depend on the windows packed beside them, at %s" % beside
    read = [moment for moment in TIMES[: CHANGED + 1] if abs(together[moment] - later[moment]) > ROUNDING]
    assert read == [], "predictions that read a fact after their time, at %s" % read
    LOG.info("composition exit spec=%s seconds=%.3f", values, time.perf_counter() - started)
