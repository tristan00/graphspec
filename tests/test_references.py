import copy
import gc
import logging
import time

import pytest
import support
import torch

from conftest import shared
from graphspec.engine.transform import materialize
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.schema.schema import VERSION, Schema
from graphspec.sources import synthetic
from graphspec.views.graph import Graph
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import batches, pack_batch
from tuning.batches.inputs import couples
from tuning.families.event_graph import KINDS
from tuning.families.model_spec import VIEWS, views
from tuning.families.models import build as network

LOG = logging.getLogger(__name__)
configuration = support.configuration
SPLITS = dict(trained=["train"], scored=["test"], sees=dict(train=[], test=[]))
MAKERS, SITES = ("m0", "m1", "m2", "m3"), ("s0", "s1", "s2", "s3")
TIMES, THIRD, KIND, VALUE = 4, 1, 3, 2
FEW, MANY, CROWDED = 5, 50, (0, 1, 1, 2)
EDGES = ("snapshot", "event_graph", "memory", "projection", "sequence", "tree", "walks")
NETS, HELD, OPENED = {}, {}, {}
pytestmark = pytest.mark.xdist_group("references")


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
def graph(name, changed=False, era=1.0, values=FEW, times=tuple(range(TIMES))):
    pairs = [(MAKERS[at], SITES[at]) for at in range(4)] + [(MAKERS[at], SITES[(at + 1) % 4]) for at in range(4)]
    links = [(MAKERS[at % 4], SITES[at // 4 % 4], "v%02d" % (at % values), 0) for at in range(max(values, 8))]
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
            time=list(times),
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
    stated=lambda: dict(schema=stated(NAMES), graphs=[graph("main"), graph("era", times=CROWDED)]),
    changed=lambda: dict(
        schema=stated(NAMES), graphs=[graph("main", changed=True), graph("era", era=2.0, times=CROWDED)]
    ),
    reordered=lambda: dict(schema=stated(("r4", "r5", "r2", "r0", "r3", "r1")), graphs=[graph("main")]),
)
WIDE = dict(wide=lambda: dict(schema=stated(NAMES), graphs=[graph("main", values=MANY)]))


def made(cases):
    def make(root):
        started = time.perf_counter()
        for name, case in cases.items():
            build(synthetic.stage(root, name, case()), root / name, 1, ("train",))
        LOG.info("banks of references exit banks=%d seconds=%.3f", len(cases), time.perf_counter() - started)

    return make


@pytest.fixture(scope="module")
def banks():
    root = shared("references", made(CASES))
    yield {name: root / name for name in CASES}
    OPENED.clear()
    gc.collect()


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


@pytest.mark.parametrize("family", VIEWS)
def test_every_family_reads_the_references_of_a_task_as_a_set_keyed_by_name(banks, family):
    registry = opened(banks, "stated").registry
    assert family in views(registry) and couples(registry) == {0: (0, 1), 1: (0, 1)}
    stated, changed = compared(banks, family, "main")
    assert stated[:THIRD] == changed[:THIRD] and stated[THIRD] != changed[THIRD]
    reordered, stated = predicted(banks, family, "reordered"), predicted(banks, family, "stated")
    assert len(reordered) == TIMES and reordered == {row: stated[row] for row in reordered}


@pytest.mark.parametrize("family", VIEWS)
def test_the_kind_of_a_task_reaches_every_family(banks, family):
    stated, changed = compared(banks, family, "main")
    assert stated[KIND] != changed[KIND]


@pytest.mark.parametrize("family", [family for family in VIEWS if family in EDGES])
def test_the_value_of_a_relation_taken_from_a_column_reaches_every_family_that_reads_edges(banks, family):
    stated, changed = compared(banks, family, "main")
    assert stated[:THIRD] == changed[:THIRD] and stated[VALUE] != changed[VALUE]


def test_the_kinds_of_event_and_the_relations_never_share_a_slot(banks):
    held = opened(banks, "stated")
    registry = held.registry
    stated = {"time/history": 2, "time/context": 1}
    cfg = configuration(registry, dict(view="event_graph"), stated, 5, copy.deepcopy(SPLITS))
    torch.manual_seed(5)
    net, found = network(cfg, held.fitted(cfg)).eval(), []
    net.layers[0].register_forward_pre_hook(lambda module, args: found.append(args[2]))
    blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
    with torch.inference_mode():
        for group in batches(blocks, len(blocks), held.sizes(cfg), frame_budget=1024):
            net(pack_batch(held, group, cfg, "cpu"))
    slots = set(torch.cat(found).tolist())
    assert registry.relations > 4 and registry.marker("period_context") in slots
    kinds = slots - {registry.marker("period_context")}
    assert kinds and min(kinds) >= registry.relation_slots and max(kinds) < registry.relation_slots + KINDS


@timed("references.sizes")
def sizes(banks, name):
    held = opened(banks, name)
    cfg = configuration(held.registry, dict(view="snapshot"), {"time/history": 2}, 5, copy.deepcopy(SPLITS))
    registry = held.fitted(cfg)
    space = support.parameters(registry, specs.GROUPS)
    steps, spec = support.steps(registry, {"time/history": 2}, specs.GROUPS)
    arrays = materialize(
        Graph.load(held.view(0, cfg)).dense(), registry, steps, spec, specs.GROUPS, held.limits(), None
    )
    weights = {}
    for family in views(registry):
        torch.manual_seed(5)
        chosen = configuration(held.registry, dict(view=family), {"time/history": 2}, 5, copy.deepcopy(SPLITS))
        weights[family] = sum(
            parameter.numel()
            for module in network(chosen, registry).modules()
            if not (
                isinstance(module, torch.nn.Embedding)
                and module.num_embeddings in (registry.relation_slots, registry.relation_slots + KINDS)
            )
            for parameter in module.parameters(recurse=False)
        )
    return dict(values=len(registry.groups["links"]), parameters=len(space), arrays=sorted(arrays), weights=weights)


def test_a_relation_taken_from_a_column_has_the_parameters_arrays_and_weights_of_its_table(banks):
    banks = dict(banks, wide=shared("references-wide", made(WIDE)) / "wide")
    few, many = sizes(banks, "stated"), sizes(banks, "wide")
    assert few.pop("values") == FEW and many.pop("values") == MANY and few == many
    assert "relations/links/keep" in [
        parameter.name for parameter in support.parameters(opened(banks, "wide").registry, ())
    ]
    assert "attributes_links" in few["arrays"] and set(few["weights"]) == set(VIEWS)


def test_a_frame_of_a_sequence_over_graphs_reads_every_task_seen_as_an_event_at_its_step(banks):
    held = opened(banks, "stated")
    assert set(views(held.registry)) == set(VIEWS)
    stated = dict(view="snapshot", **{"snapshot/temporal": "graph_sequence"})
    cfg = configuration(held.registry, stated, {"time/history": 2}, 5, copy.deepcopy(SPLITS))
    torch.manual_seed(5)
    net = network(cfg, held.fitted(cfg))
    era = [graph["index"] for graph in held.viewed(cfg).graphs if graph["name"] == "era"]
    blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg) if block[0] in era]
    found = [pack_batch(held, group, cfg, "cpu") for group in batches(blocks, 1, held.sizes(cfg), frame_budget=1024)]
    b = next(batch for batch in found if torch.bincount(batch["frame_event"]).max() == 2)
    frame = int(torch.bincount(b["frame_event"]).argmax())
    second = (b["frame_event"] == frame).nonzero().flatten()[1]
    moved = dict(b, frame_task=b["frame_task"].clone())
    moved["frame_task"][second] += 1
    predicted, changed = net(b), net(moved)
    rows = (b["row_frames"][b["selected"]] == frame).nonzero().flatten()
    assert len(rows) == 2 and torch.isfinite(predicted).all() and (predicted[rows] != changed[rows]).all()
    sum(net(batch).sum() for batch in found).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in net.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("family", VIEWS)
def test_the_attributes_of_a_graph_reach_every_family(banks, family):
    stated, changed = compared(banks, family, "era")
    assert all(left != right for left, right in zip(stated, changed))
    mapping = CASES["stated"]()["schema"]
    mapping["tasks"]["kinds"]["up"]["columns"] = ["ask", "era"]
    with pytest.raises(ValueError, match="the columns era are columns of the graph and of a kind of task"):
        Schema(mapping)
