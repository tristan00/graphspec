import logging
import time

import numpy as np
import pytest

from conftest import inline
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.targets import TARGETS, valued
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("classes")
SITES = ("s0", "s1", "s2", "s3", "s4", "s5")
GRADES = ["low", "mid", "high"]
TAGS = ["red", "green", "blue"]
TIMES = (1, 2, 3, 4, 5, 6, 7, 8)
KINDS = dict(level=("number", "size"), grade=("class", GRADES), tags=("classes", TAGS))


@timed("classes.aimed")
def aimed(kind, series):
    return dict(kind=kind, series=series, about=None, utility=synthetic.valued([series] if kind == "number" else series))


@timed("classes.schema")
def schema(kinds):
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties=None),
        observers=False,
        columns=dict(load=dict(type="number", encoder="value")),
        graph=dict(columns=[]),
        nodes=dict(site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(link=synthetic.edge("site", "site", [])),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(at=dict(sort="node", kinds=["site"])),
            kinds={
                name: dict(
                    references=["at"],
                    columns=[],
                    target=aimed(*kinds[name]),
                    weight=False,
                    absent="stop",
                    event=None,
                    matures=None,
                )
                for name in kinds
            },
        ),
    )


@timed("classes.stated")
def stated(kind, at, place):
    if kind == "level":
        return 0.5 * at + place
    if kind == "grade":
        return GRADES[(at + place) % 3]
    return [name for bit, name in enumerate(TAGS) if (at * place) % 8 >> bit & 1]


@timed("classes.graph")
def graph(place):
    kinds = list(KINDS)
    values = dict(key=list(SITES), since=[0] * 6, until=[None] * 6, load=[1.0 + at + place for at in range(6)])
    links = dict(
        source=list(SITES[:-1]), target=list(SITES[1:]), present=[True] * 5, since=[0] * 5, until=[None] * 5
    )
    kind = [kinds[at % len(kinds)] for at in range(len(TIMES))]
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(site=list(SITES)),
        values=[dict(form="state", parameters={}, rows=values, kind="site")],
        edges=[dict(form="state", parameters={}, rows=links, table="link", ends=("site", "site"))],
        tasks=dict(
            key=list(range(len(TIMES))),
            kind=kind,
            time=list(TIMES),
            split=["train" if (at // len(kinds)) % 2 == 0 else "validation" for at in range(len(TIMES))],
            weight=[None] * len(TIMES),
            references=dict(at=[SITES[at % 6] for at in range(len(TIMES))]),
            target=[stated(kind[at], at, place) for at in range(len(TIMES))],
            known=[None] * len(TIMES),
        ),
    )


@pytest.fixture(scope="module")
def classes(tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("classes")
    case = dict(schema=schema(KINDS), graphs=[graph(place) for place in range(2)])
    build(synthetic.stage(root, "classes", case), root / "bank", 1, ("train",))
    LOG.info("bank of classes exit seconds=%.3f", time.perf_counter() - started)
    return root / "bank", case


@timed("classes.read")
def read(held):
    found = {}
    kinds = held.registry.schema.tasks["kinds"]
    for row in held.rows:
        found.setdefault((row["graph"], row["kind"], row["ticket"]), []).append(row)
    return {key: TARGETS.named(kinds[key[1]]["target"]["kind"]).read(rows) for key, rows in found.items()}


def test_every_kind_over_classes_reads_back_as_stated_with_its_classes_as_its_series(classes):
    bank, case = classes
    held = Deriver(bank, "cpu", RESIDENT)
    schema = held.registry.schema
    assert schema.targets() == tuple(kind for kind, _ in KINDS.values())
    assert schema.series() == ("size", *GRADES, *TAGS) and schema.target_columns() == dict(series="STRING")
    wanted = {
        (graph["name"], kind, key): target
        for graph in case["graphs"]
        for kind, key, target in zip(graph["tasks"]["kind"], graph["tasks"]["key"], graph["tasks"]["target"])
    }
    assert read(held) == wanted and [] in wanted.values()
    assert [entry["name"] for entry in held.split["series"]] == ["size"]
    classes = len(schema.series()) - 1
    assert held.center[1:].tolist() == [0.0] * classes and held.scale[1:].tolist() == [1.0] * classes
    grade = [row for row in held.rows if row["kind"] == "grade"]
    assert len(grade) == 3 * sum(kind == "grade" for _, kind, _ in wanted) and {row["y"] for row in grade} == {0.0, 1.0}


def test_a_class_is_one_of_at_least_two_stated_classes_and_holds_one_point_for_each():
    where = "the target of the kind of task grade"
    with pytest.raises(ValueError, match="the target of the kind of task grade names at least 2 classes"):
        Schema(schema(dict(grade=("class", ["low"]))))
    kind = TARGETS.named("class")
    with pytest.raises(ValueError, match="states its target as one of the classes low, mid, high, found 'top'"):
        kind.staged("top", tuple(GRADES), where)
    rows = dict(task=[0, 0, 0], graph=["g0"] * 3, y=[1.0, 1.0, 0.0], series=GRADES)
    with pytest.raises(ValueError, match="state 1 for the one class of a task and 0 for every other"):
        kind.points(rows, tuple(GRADES), where)
    with pytest.raises(ValueError, match="a task of graph g1 holds under .* one point for every class"):
        kind.points(dict(rows, graph=["g1"] * 3, series=["low", "low", "high"]), tuple(GRADES), where)
    assert kind.points(dict(rows, y=[0.0, 1.0, 0.0]), tuple(GRADES), where) == dict(series=GRADES)
    assert kind.read([dict(series=name, y=y) for name, y in zip(GRADES, [0.0, 1.0, 0.0])]) == "mid"
    target = Schema(schema(dict(grade=("class", GRADES)))).tasks["kinds"]["grade"]["target"]
    target = dict(target, utility=dict(target["utility"], weights=dict(low=1.0, mid=4.0, high=0.0)))
    tasks, values = valued(target, dict(task=[3, 3, 3], series=GRADES, y=[0.25, 0.5, 0.25]))
    assert tasks.tolist() == [3] and values.tolist() == [2.25]


def test_a_class_is_predicted_by_scores_its_loss_normalises_per_task_and_its_measures_read_probabilities():
    import torch
    from torch.nn import functional as F

    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    scores = torch.tensor([2.0, 0.0, -1.0, 0.5, 0.5, 3.0, 1.0, 1.0, 1.0])
    row, target = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]), torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 1])
    weight = torch.tensor([1.0] * 3 + [3.0] * 3 + [1.0] * 3)
    batch = dict(target=target, weight=weight, point_row=row, tasks=3)
    expected = F.cross_entropy(scores.view(3, 3), torch.tensor([0, 1, 2]), reduction="none")
    found = LOSSES.named("entropy")(scores, batch, None)
    assert float(found) == pytest.approx(float((expected * torch.tensor([1.0, 3.0, 1.0])).sum() / 5))
    probability = predicted(("class",), np.zeros(9, np.int64), row.numpy(), scores.numpy())
    assert np.allclose(np.bincount(row.numpy(), probability), 1.0)
    assert np.allclose(probability[:3], F.softmax(scores[:3], 0).numpy())
    held = scored(target.numpy(), probability, row.numpy(), np.tile(np.arange(3), 3), GRADES, np.ones(3))
    metrics = measured(["accuracy", "entropy"], held, dict(low="class", mid="class", high="class"))
    assert metrics["accuracy"]["value"] == pytest.approx(1 / 3)
    assert metrics["accuracy"]["series"] == pytest.approx(dict(low=2 / 3, mid=2 / 3, high=1 / 3))
    assert metrics["entropy"]["value"] == pytest.approx(float(expected.mean()))
    assert sum(metrics["entropy"]["series"].values()) == pytest.approx(metrics["entropy"]["value"])


def test_a_set_of_classes_is_one_decision_per_class_by_its_loss_and_its_measure():
    import torch
    from torch.nn import functional as F

    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    where, kind = "the target of the kind of task tags", TARGETS.named("classes")
    for wrong in (["red", "red"], ["pink"], "red"):
        with pytest.raises(ValueError, match="a list of distinct classes of red, green, blue, possibly none"):
            kind.staged(wrong, tuple(TAGS), where)
    assert [row["y"] for row in kind.staged(["blue", "red"], tuple(TAGS), where)] == [1.0, 0.0, 1.0]
    rows = dict(task=[0, 0, 0], graph=["g0"] * 3, y=[1.0, 0.5, 0.0], series=TAGS)
    with pytest.raises(ValueError, match="state 1 for every class of the set of a task and 0 for every other"):
        kind.points(rows, tuple(TAGS), where)
    assert kind.read([dict(series=name, y=y) for name, y in zip(TAGS, [1.0, 0.0, 1.0])]) == ["red", "blue"]
    scores = torch.tensor([2.0, -1.0, 0.0, 0.5, 3.0, -2.0])
    target, weight = torch.tensor([1.0, 0, 0, 1, 1, 0]), torch.tensor([1.0] * 3 + [2.0] * 3)
    batch = dict(target=target, weight=weight, point_row=torch.tensor([0, 0, 0, 1, 1, 1]), tasks=2)
    error = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    assert float(LOSSES.named("binary")(scores, batch, None)) == pytest.approx(float((weight * error).sum() / 9))
    probability = predicted(("classes",), np.zeros(6, np.int64), np.repeat([0, 1], 3), scores.numpy())
    assert np.allclose(probability, torch.sigmoid(scores).numpy())
    held = scored(target.numpy(), probability, np.repeat([0, 1], 3), np.tile(np.arange(3), 2), TAGS, np.ones(3))
    found = measured(["binary"], held, dict.fromkeys(TAGS, "classes"))["binary"]
    assert found["value"] == pytest.approx(float(error.mean()))
    assert found["series"]["blue"] == pytest.approx(float(error[[2, 5]].mean()))


def test_a_bank_of_a_number_and_kinds_over_classes_completes_one_fit(classes, tmp_path, statement, monkeypatch):
    bank = classes[0]
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    held = Deriver(bank, "cpu", RESIDENT)
    losses = {"number": "squared", "class": "entropy", "classes": "binary"}
    given = dict(statement, loss=losses, measures=["entropy", "accuracy", "mse", "binary"])
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of kinds over classes exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20 and saved["loss"] == given["loss"]
    measures = saved["measures"]
    assert list(measures["accuracy"]["series"]) == GRADES and list(measures["mse"]["series"]) == ["size"]
    assert list(measures["binary"]["series"]) == TAGS and 0 <= measures["accuracy"]["value"] <= 1
    assert np.isfinite([measures[name]["value"] for name in ("entropy", "binary", "mse")]).all()
    with np.load(tmp_path / "predictions.npz") as found:
        aim, task, prediction = found["aim"], held.task_of[found["row_index"]], found["prediction"]
        assert set(aim.tolist()) == set(range(len(KINDS)))
        chosen = aim == list(KINDS).index("grade")
        assert np.allclose(np.bincount(task[chosen], prediction[chosen])[np.unique(task[chosen])], 1.0)
        chosen = aim != list(KINDS).index("level")
        assert ((prediction[chosen] > 0) & (prediction[chosen] < 1)).all()
