import copy
import logging
import time

import numpy as np
import pytest

from conftest import Inline
from graphspec import synthetic
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema import VERSION, Schema
from graphspec.truth import Facts, Graph, inputs
from graphspec.view import finished, sequences, view
from tuning.build import build
from tuning.data import Deriver
from tuning.model_spec import configuration
from tuning.statement import state
from tuning.training import fit

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("small")
MAKERS, SITES = ("m0", "m1", "m2", "m3"), ("s0", "s1", "s2", "s3")
TIMES = (1, 1, 1, 1, 2, 3, 3, 4)
SPLIT = ("train", "train", "train", "train", "train", "validation", "validation", "validation")
HIDDEN = dict(trained=["train"], scored=["validation"], sees=dict(train=["without_targets"], validation=[]))
VALUE = dict(type="number", encoder="value")


@timed("small.stated")
def stated(edged):
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(size=VALUE, load=VALUE, **(dict(weight=VALUE) if edged else {})),
        graph=dict(columns=[]),
        nodes=dict(maker=dict(columns=["size"], holds=None), site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(serves=synthetic.edge("maker", "site", ["weight"])) if edged else {},
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(
                unit=dict(sort="node", kinds=["maker"]),
                **(dict(link=dict(sort="edge", kinds=["maker"])) if edged else {}),
            ),
            kinds=dict(
                ask=dict(
                    references=["unit", "link"] if edged else ["unit"],
                    columns=[],
                    target=dict(kind="number", series="outcome", about="link" if edged else "unit"),
                    weight=True,
                    absent="stop",
                    event=None,
                    matures=None,
                )
            ),
        ),
    )


@timed("small.graph")
def graph(place, edged):
    count = len(TIMES)
    values = [
        dict(
            form="state",
            parameters={},
            kind="maker",
            rows=dict(
                key=list(MAKERS), since=[0] * 4, until=[None] * 4, size=[1.0 + at + place for at in range(4)]
            ),
        ),
        dict(
            form="state",
            parameters={},
            kind="site",
            rows=dict(key=list(SITES), since=[0] * 4, until=[None] * 4, load=[2.0 + at for at in range(4)]),
        ),
    ]
    edges = []
    if edged:
        rows = dict(
            source=list(MAKERS),
            target=list(SITES),
            present=[True] * 4,
            weight=[1.0 + at for at in range(4)],
            since=[0] * 4,
            until=[None] * 4,
        )
        edges.append(dict(form="state", parameters={}, rows=rows, table="serves", ends=("maker", "site")))
    references = dict(unit=[MAKERS[at % 4] for at in range(count)])
    if edged:
        references["link"] = [(MAKERS[at % 4], "serves", SITES[at % 4]) for at in range(count)]
    tasks = dict(
        key=list(range(count)),
        kind=["ask"] * count,
        time=list(TIMES),
        split=list(SPLIT),
        weight=[1.0 + 0.5 * at for at in range(count)],
        references=references,
        target=[0.5 * at + place for at in range(count)],
        known=[None] * count,
    )
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(maker=list(MAKERS), site=list(SITES)),
        values=values,
        edges=edges,
        tasks=tasks,
    )


@timed("small.case")
def case(edged):
    return dict(schema=stated(edged), graphs=[graph(place, edged) for place in range(2)])


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("small")
    result = {}
    for name, edged in (("bare", False), ("edged", True)):
        build(synthetic.stage(root, name, case(edged)), root / name, 1, ("train",))
        result[name] = root / name
    LOG.info("small banks exit seconds=%.3f", time.perf_counter() - started)
    return result


@timed("small.refused")
def refused(root, name, changed):
    held = case(True)
    changed(held)
    build(synthetic.stage(root, name, held), root / (name + ".bank"), 1, ("train",))


@pytest.fixture
def packed(monkeypatch):
    import torch

    started = time.perf_counter()
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr("tuning.packing.Packer", Inline)
    LOG.debug("packed exit seconds=%.4f", time.perf_counter() - started)


@pytest.mark.parametrize("name", ["bare", "edged"])
def test_a_view_without_any_edge_completes_one_fit_on_cpu(banks, packed, tmp_path, statement, name):
    splits = copy.deepcopy(HIDDEN)
    held = Deriver(banks[name], "cpu", RESIDENT)
    cfg = dict(configuration(held.registry, dict(view="snapshot"), {"time/history": 4}, 5, splits), epochs=1)
    trained = [entry for entry in held.viewed(cfg).graphs if entry["split"] == held.splits.index("train")]
    assert trained and all(entry["facts"] == 0 for entry in trained)
    shown = [Graph.load(held.view(entry["index"], cfg)) for entry in held.viewed(cfg).graphs]
    assert sum(len(view.fact_start) for view in shown) == (0 if name == "bare" else 8)
    blocks = held.blocks(0, 4, cfg)
    started = time.perf_counter()
    given = state(dict(statement, **splits))
    fit(banks[name], tmp_path, cfg, blocks, 16, started + statement["seconds"], given)
    LOG.info("fit without edges exit bank=%s seconds=%.3f", name, time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert np.isfinite(saved["raw_val_mse"]) and saved["holdout_rows"] == len(held.members(cfg, 1))


def test_a_value_beyond_sixteen_bits_stops_the_view_naming_the_node_the_column_and_the_encoder(banks):
    held = Deriver(banks["edged"], "cpu", RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), {}, 5, copy.deepcopy(HIDDEN))
    registry, facts = held.fitted(cfg), Facts.load(held.truth(1))
    values = np.array(facts.obs_values)
    values[np.asarray(facts.obs_entity) == facts.keys.index("m2"), registry.encoding.layout.index("size")] = 1e6
    large = facts.replace(obs_values=values)
    shared = sequences(large, registry, "tasks", held.looking(cfg)[0])[0]
    shown = view(large, registry, shared, None)
    with pytest.raises(
        ValueError,
        match=r"the node m2 of the kind maker in the view of graph g1 holds the value 1000000\.0 of the column "
        r"size, encoded by value, which gives 1000000\.0; model inputs are stored at 16 bits, which hold "
        r"magnitudes up to 65504",
    ):
        finished(shown, registry, inputs(registry, large.kinds, large.arrays))


@pytest.mark.parametrize("weight", [float("inf"), float("nan"), 1e39, 0.0])
def test_a_weight_that_is_not_finite_at_float32_or_not_above_zero_stops_naming_the_task(tmp_path, weight):
    def weighted(held):
        held["graphs"][0]["tasks"]["weight"][3] = weight

    with pytest.raises(ValueError, match="the task 3 of the kind ask of graph g0 states the weight"):
        refused(tmp_path, "weighted", weighted)


def test_a_column_that_nothing_uses_stops_at_the_schema_naming_it():
    mapping = stated(True)
    mapping["columns"]["spare"] = VALUE
    with pytest.raises(ValueError, match="the columns spare are declared and used by no kind of node"):
        Schema(mapping)


def test_a_missing_reference_names_the_task_the_reference_and_what_the_kind_requires(tmp_path):
    def missing(held):
        held["graphs"][1]["tasks"]["references"]["unit"][5] = None

    with pytest.raises(
        ValueError,
        match="the kind ask requires of each of its tasks exactly 1 member under the reference unit, which is "
        "of the sort node, and the task 5 of graph g1 names 0",
    ):
        refused(tmp_path, "missing", missing)
