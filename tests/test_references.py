import copy
import logging
import time

import pytest
import torch

from conftest import shared
from graphspec import spec as specs
from graphspec import synthetic
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema import VERSION, Schema
from graphspec.transform import materialize
from graphspec.truth import Graph
from tuning.batching import batches, pack_batch
from tuning.build import build
from tuning.data import Deriver
from tuning.inputs import couples
from tuning.model_spec import VIEWS, configuration, views
from tuning.models import build as network

LOG = logging.getLogger(__name__)
SPLITS = dict(trained=["train"], scored=["test"], sees=dict(train=[], test=[]))
MAKERS, SITES = ("m0", "m1", "m2", "m3"), ("s0", "s1", "s2", "s3")
TIMES, THIRD, KIND, VALUE = 4, 1, 3, 2
FEW, MANY = 5, 50
EDGES = ("snapshot", "event_graph", "memory", "projection", "sequence", "tree", "walks")
SHARED = dict(
    snapshot="plain",
    signal="plain",
    affinity="plain",
    memory="neighbors",
    projection="neighbors",
    sequence="neighbors",
    tree="neighbors",
)
FAMILIES = [
    pytest.param(family, marks=pytest.mark.xdist_group("references-" + SHARED.get(family, family)))
    for family in VIEWS
]
NETS, HELD, OPENED = {}, {}, {}


@timed("references.stated")
def stated(order):
    number = dict(type="number", encoder="value")
    kinds = {
        kind: dict(
            references=list(order),
            columns=["ask"],
            target=dict(kind="number", series="out", about=None),
            weight=False,
            absent="stop",
            event=["r0", "r1"],
            matures=None,
        )
        for kind in ("up", "down")
    }
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(size=number, load=number, weight=number, ask=number, era=number),
        graph=dict(columns=["era"]),
        nodes=dict(maker=dict(columns=["size"], holds=None), site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(
            serves=synthetic.edge("maker", "site", ["weight"]),
            beside=synthetic.edge("site", "site", [], directed=False),
            links=synthetic.edge("maker", "site", ["weight"], relation=dict(column="verb")),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=1,
            references=dict(
                r0=dict(sort="node", kinds=["maker"]),
                r1=dict(sort="node", kinds=["site"]),
                r2=dict(sort="node", kinds=["maker"]),
                r3=dict(sort="path", kinds=["maker", "site"]),
                r4=dict(sort="graph", kinds=[]),
                r5=dict(sort="edge", kinds=["maker"]),
            ),
            kinds=kinds,
        ),
    )


@timed("references.graph")
def graph(name, changed=False, era=1.0, values=FEW):
    pairs = [(MAKERS[at], SITES[at]) for at in range(4)] + [(MAKERS[at], SITES[(at + 1) % 4]) for at in range(4)]
    links = [(MAKERS[at % 4], SITES[at // 4 % 4], "v%02d" % (at % values), 0) for at in range(max(values, 16))]
    links.append((MAKERS[VALUE], SITES[(VALUE + 1) % 4], "v%02d" % (2 if changed else 1), VALUE))
    return dict(
        name=name,
        unit="tick",
        columns=dict(era=era),
        nodes=dict(maker=list(MAKERS), site=list(SITES)),
        values=[
            dict(
                form="state",
                parameters={},
                kind="maker",
                rows=dict(
                    key=list(MAKERS) * 2,
                    since=[0] * 4 + [3] * 4,
                    until=[3] * 4 + [None] * 4,
                    size=[1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
                ),
            ),
            dict(
                form="state",
                parameters={},
                kind="site",
                rows=dict(key=list(SITES), since=[0] * 4, until=[None] * 4, load=[2.0, 3.0, 5.0, 7.0]),
            ),
        ],
        edges=[
            dict(
                form="state",
                parameters={},
                table="serves",
                ends=("maker", "site"),
                rows=dict(
                    source=[pair[0] for pair in pairs],
                    target=[pair[1] for pair in pairs],
                    present=[True] * 8,
                    since=[0] * 4 + [1, 2, 3, 4],
                    until=[None] * 8,
                    weight=[1.0 + at for at in range(8)],
                ),
            ),
            dict(
                form="state",
                parameters={},
                table="beside",
                ends=("site", "site"),
                rows=dict(source=["s0", "s2"], target=["s1", "s3"], present=[True] * 2, since=[0, 0], until=[None] * 2),
            ),
            dict(
                form="state",
                parameters={},
                table="links",
                ends=("maker", "site"),
                rows=dict(
                    source=[link[0] for link in links],
                    target=[link[1] for link in links],
                    present=[True] * len(links),
                    since=[link[3] for link in links],
                    until=[None] * len(links),
                    relation=[link[2] for link in links],
                    weight=[0.5 + at for at in range(len(links))],
                ),
            ),
        ],
        tasks=dict(
            key=list(range(TIMES)),
            kind=["down" if changed and at == KIND else "up" for at in range(TIMES)],
            time=list(range(TIMES)),
            split=["train" if at < 3 else "test" for at in range(TIMES)],
            weight=[None] * TIMES,
            ask=[0.25 * at for at in range(TIMES)],
            references=dict(
                r0=[MAKERS[at % 4] for at in range(TIMES)],
                r1=[SITES[(at + 1) % 4] for at in range(TIMES)],
                r2=[MAKERS[(at + (3 if changed and at == THIRD else 2)) % 4] for at in range(TIMES)],
                r3=[
                    [(MAKERS[at % 2], "serves", SITES[at % 2]), (SITES[at % 2], "beside", SITES[1 - at % 2])]
                    for at in range(TIMES)
                ],
                r5=[(MAKERS[(at + 1) % 4], "serves", SITES[(at + 1) % 4]) for at in range(TIMES)],
            ),
            target=[0.5 * at + len(name) for at in range(TIMES)],
            known=[None] * TIMES,
            signal=[[0.1 * at] for at in range(TIMES)],
            signal_observed=[[True] for _ in range(TIMES)],
        ),
    )


NAMES = ("r0", "r1", "r2", "r3", "r4", "r5")
CASES = dict(
    stated=lambda: dict(schema=stated(NAMES), graphs=[graph("main"), graph("era")]),
    changed=lambda: dict(schema=stated(NAMES), graphs=[graph("main", changed=True), graph("era", era=2.0)]),
    reordered=lambda: dict(schema=stated(("r4", "r5", "r2", "r0", "r3", "r1")), graphs=[graph("main")]),
    wide=lambda: dict(schema=stated(NAMES), graphs=[graph("main", values=MANY)]),
)


def made(root):
    started = time.perf_counter()
    for name, case in CASES.items():
        build(synthetic.stage(root, name, case()), root / name, 1, ("train",))
    LOG.info("banks of references exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture(scope="module")
def banks():
    root = shared("references", made)
    return {name: root / name for name in CASES}


@timed("references.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", RESIDENT)
    return OPENED[name]


@timed("references.predicted")
def predicted(banks, family, name):
    if (family, name) not in HELD:
        held = opened(banks, name)
        cfg = configuration(held.registry, dict(view=family), {"time/history": 2}, 5, copy.deepcopy(SPLITS))
        if family not in NETS:
            torch.manual_seed(5)
            NETS[family] = network(cfg, held.fitted(cfg)).eval()
        blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
        found = {}
        with torch.inference_mode():
            for graph in sorted({block[0] for block in blocks}):
                chosen = [block for block in blocks if block[0] == graph]
                for group in batches(chosen, len(chosen), held.sizes(cfg), frame_budget=1024):
                    b = pack_batch(held, group, cfg, "cpu")
                    rows = [held.rows[row]["graph"] + "/%d" % held.rows[row]["ticket"] for row in b["rows"]]
                    found.update(zip(rows, NETS[family](b).tolist()))
        HELD[family, name] = found
    return HELD[family, name]


@timed("references.compared")
def compared(banks, family, graph):
    stated, changed = predicted(banks, family, "stated"), predicted(banks, family, "changed")
    rows = sorted(row for row in stated if row.startswith(graph + "/"))
    assert len(rows) == TIMES and sorted(changed) == sorted(stated)
    return [stated[row] for row in rows], [changed[row] for row in rows]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name(banks, family):
    registry = opened(banks, "stated").registry
    assert family in views(registry) and couples(registry) == {0: (0, 1), 1: (0, 1)}
    stated, changed = compared(banks, family, "main")
    assert stated[:THIRD] == changed[:THIRD] and stated[THIRD] != changed[THIRD]
    reordered, stated = predicted(banks, family, "reordered"), predicted(banks, family, "stated")
    assert len(reordered) == TIMES and reordered == {row: stated[row] for row in reordered}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_kind_of_a_task_reaches_every_family(banks, family):
    stated, changed = compared(banks, family, "main")
    assert stated[KIND] != changed[KIND]


@pytest.mark.parametrize("family", [family for family in FAMILIES if family.values[0] in EDGES])
def test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges(banks, family):
    stated, changed = compared(banks, family, "main")
    assert stated[:THIRD] == changed[:THIRD] and stated[VALUE] != changed[VALUE]


@timed("references.sizes")
def sizes(banks, name):
    held = opened(banks, name)
    cfg = configuration(held.registry, dict(view="snapshot"), {"time/history": 2}, 5, copy.deepcopy(SPLITS))
    registry = held.fitted(cfg)
    space = specs.space(registry, specs.GROUPS)
    spec = specs.configured(space, {"time/history": 2})
    arrays = materialize(Graph.load(held.view(0, cfg)).dense(), registry, spec, specs.GROUPS, held.limits())
    weights = {}
    for family in views(registry):
        torch.manual_seed(5)
        chosen = configuration(held.registry, dict(view=family), {"time/history": 2}, 5, copy.deepcopy(SPLITS))
        weights[family] = sum(
            parameter.numel()
            for module in network(chosen, registry).modules()
            if not (isinstance(module, torch.nn.Embedding) and module.num_embeddings == registry.relation_slots)
            for parameter in module.parameters(recurse=False)
        )
    return dict(values=len(registry.groups["links"]), parameters=len(space), arrays=sorted(arrays), weights=weights)


@pytest.mark.xdist_group("references-plain")
def test_a_relation_taken_from_a_column_has_the_parameters_arrays_and_weights_of_its_table(banks):
    few, many = sizes(banks, "stated"), sizes(banks, "wide")
    assert few.pop("values") == FEW and many.pop("values") == MANY and few == many
    assert "relations/links/keep" in [parameter.name for parameter in specs.space(opened(banks, "wide").registry, ())]
    assert "attributes_links" in few["arrays"] and set(few["weights"]) == set(VIEWS)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_attributes_of_a_graph_reach_every_family(banks, family):
    stated, changed = compared(banks, family, "era")
    assert all(left != right for left, right in zip(stated, changed))
    mapping = CASES["stated"]()["schema"]
    mapping["tasks"]["kinds"]["up"]["columns"] = ["ask", "era"]
    with pytest.raises(ValueError, match="the columns era are columns of the graph and of a kind of task"):
        Schema(mapping)
