import logging
import time

import numpy as np
import pytest
import support

from conftest import inline
from graphspec.facts.stored import Facts
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
TREES = ["oak", "elm", "ash"]
SHARES = (0.5, 0.25, 0.25)
TIMES = (1, 2, 3, 4, 5, 6, 7, 8)
FLAGS = dict(red="reddish", green="greenish", blue="bluish")
PARTS = dict(oak="oaks", elm="elms", ash="ashes")
KINDS = dict(
    level=("number", "size", None),
    grade=("class", GRADES, "tier"),
    tags=("classes", TAGS, FLAGS),
    mix=("proportions", TREES, PARTS),
)
COLUMNS = dict(
    load=dict(type="number", encoder="value"),
    tier=dict(type="category", encoder="onehot"),
    **{name: dict(type="flag", encoder="value") for name in FLAGS.values()},
    **{name: dict(type="number", encoder="value") for name in PARTS.values()},
)


@timed("classes.aimed")
def aimed(kind, series):
    return dict(kind=kind, series=series, about=None, utility=synthetic.valued([series] if kind == "number" else series))


@timed("classes.schema")
def schema(kinds):
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties=None),
        observers=False,
        columns=COLUMNS,
        graph=dict(columns=[]),
        nodes=dict(site=dict(columns=list(COLUMNS), holds=None)),
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
                    target=aimed(*kinds[name][:2]),
                    weight=False,
                    absent="stop",
                    event=None,
                    matures=None if kinds[name][2] is None else dict(reference="at", column=kinds[name][2]),
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
    if kind == "tags":
        return [name for bit, name in enumerate(TAGS) if (at * place) % 8 >> bit & 1]
    return {name: SHARES[(at + place + position) % 3] for position, name in enumerate(TREES)}


@timed("classes.graph")
def graph(place):
    kinds = list(KINDS)
    values = dict(key=list(SITES), since=[0] * 6, until=[None] * 6, **{name: [None] * 6 for name in COLUMNS})
    values["load"] = [1.0 + at + place for at in range(6)]
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
            known=[moment + 1 for moment in TIMES],
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
    assert schema.targets() == tuple(kind for kind, _, _ in KINDS.values())
    assert schema.series() == ("size", *GRADES, *TAGS, *TREES) and schema.target_columns() == dict(series="STRING")
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
    written = support.written(bank.parent / "classes.staging", schema)
    for entry in held.manifest["graphs"]:
        truth = Facts.load(held.truth(entry["index"]))
        found = truth.stated(held.registry, held.splits, held.manifest["observers"])
        assert found["targets"] == written[truth.name]["targets"]
        named = [schema.series()[code] for code in truth.target_series.tolist()]
        assert named == [held.rows[row]["series"] for row in truth.target_row.tolist()]


def test_a_class_is_one_of_at_least_two_stated_classes_and_holds_one_point_for_each():
    where = "the target of the kind of task grade"
    with pytest.raises(ValueError, match="the target of the kind of task grade names at least 2 classes"):
        Schema(schema(dict(grade=("class", ["low"], None))))
    kind = TARGETS.named("class")
    with pytest.raises(ValueError, match="states its target as one of the classes low, mid, high, found 'top'"):
        kind.staged("top", dict(series=tuple(GRADES)), where)
    rows = dict(task=[0, 0, 0], graph=["g0"] * 3, y=[1.0, 1.0, 0.0], series=GRADES)
    with pytest.raises(ValueError, match="state 1 for the one class of a task and 0 for every other"):
        kind.points(rows, dict(series=tuple(GRADES)), where)
    with pytest.raises(ValueError, match="a task of graph g1 holds under .* one point for every class"):
        kind.points(dict(rows, graph=["g1"] * 3, series=["low", "low", "high"]), dict(series=tuple(GRADES)), where)
    assert kind.points(dict(rows, y=[0.0, 1.0, 0.0]), dict(series=tuple(GRADES)), where) == dict(series=GRADES)
    assert kind.read([dict(series=name, y=y) for name, y in zip(GRADES, [0.0, 1.0, 0.0])]) == "mid"
    target = Schema(schema(dict(grade=KINDS["grade"]))).tasks["kinds"]["grade"]["target"]
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
    probability = predicted(("class",), np.zeros(9, np.int64), row.numpy(), scores.numpy(), {})
    assert np.allclose(np.bincount(row.numpy(), probability), 1.0)
    assert np.allclose(probability[:3], F.softmax(scores[:3], 0).numpy())
    held = scored(target.numpy(), probability, row.numpy(), np.tile(np.arange(3), 3), GRADES, np.ones(3), {})
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
            kind.staged(wrong, dict(series=tuple(TAGS)), where)
    assert [row["y"] for row in kind.staged(["blue", "red"], dict(series=tuple(TAGS)), where)] == [1.0, 0.0, 1.0]
    rows = dict(task=[0, 0, 0], graph=["g0"] * 3, y=[1.0, 0.5, 0.0], series=TAGS)
    with pytest.raises(ValueError, match="state 1 for every class of the set of a task and 0 for every other"):
        kind.points(rows, dict(series=tuple(TAGS)), where)
    assert kind.read([dict(series=name, y=y) for name, y in zip(TAGS, [1.0, 0.0, 1.0])]) == ["red", "blue"]
    scores = torch.tensor([2.0, -1.0, 0.0, 0.5, 3.0, -2.0])
    target, weight = torch.tensor([1.0, 0, 0, 1, 1, 0]), torch.tensor([1.0] * 3 + [2.0] * 3)
    batch = dict(target=target, weight=weight, point_row=torch.tensor([0, 0, 0, 1, 1, 1]), tasks=2)
    error = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    assert float(LOSSES.named("binary")(scores, batch, None)) == pytest.approx(float((weight * error).sum() / 9))
    probability = predicted(("classes",), np.zeros(6, np.int64), np.repeat([0, 1], 3), scores.numpy(), {})
    assert np.allclose(probability, torch.sigmoid(scores).numpy())
    held = scored(target.numpy(), probability, np.repeat([0, 1], 3), np.tile(np.arange(3), 2), TAGS, np.ones(3), {})
    found = measured(["binary"], held, dict.fromkeys(TAGS, "classes"))["binary"]
    assert found["value"] == pytest.approx(float(error.mean()))
    assert found["series"]["blue"] == pytest.approx(float(error[[2, 5]].mean()))


def test_a_proportion_vector_states_a_share_of_every_class_and_is_predicted_and_measured_as_one():
    import torch

    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    where, kind = "the target of the kind of task mix", TARGETS.named("proportions")
    for wrong in (dict(oak=0.5, elm=0.5), dict(oak=0.5, elm=0.25, ash=0.5), dict(oak=1.5, elm=-0.5, ash=0.0)):
        with pytest.raises(ValueError, match="a share of at least 0 for every class of oak, elm, ash, summing to 1"):
            kind.staged(wrong, dict(series=tuple(TREES)), where)
    rows = dict(task=[0, 0, 0], graph=["g0"] * 3, y=[0.5, 0.25, 0.5], series=TREES)
    with pytest.raises(ValueError, match="state shares of at least 0 that sum to 1 within 1e-06 over the classes"):
        kind.points(rows, dict(series=tuple(TREES)), where)
    assert kind.read([dict(series=name, y=y) for name, y in zip(TREES, SHARES)]) == dict(zip(TREES, SHARES))
    scores, row = torch.tensor([1.0, 0.0, -1.0, 2.0, 2.0, 0.0]), torch.tensor([0, 0, 0, 1, 1, 1])
    target = torch.tensor([0.5, 0.25, 0.25, 0.0, 1.0, 0.0])
    batch = dict(target=target, weight=torch.ones(6), point_row=row, tasks=2)
    logged = torch.cat([torch.log_softmax(scores[:3], 0), torch.log_softmax(scores[3:], 0)])
    crossed = -(target * logged)
    assert float(LOSSES.named("entropy")(scores, batch, None)) == pytest.approx(float(crossed.sum() / 2))
    probability = predicted(("proportions",), np.zeros(6, np.int64), row.numpy(), scores.numpy(), {})
    assert np.allclose(probability, logged.exp().numpy())
    held = scored(target.numpy(), probability, row.numpy(), np.tile(np.arange(3), 2), TREES, np.ones(3), {})
    found = measured(["divergence", "entropy"], held, dict.fromkeys(TREES, "proportions"))
    stated = target.numpy()
    divergence = np.where(stated > 0, stated * np.log(np.where(stated > 0, stated, 1) / probability), 0)
    assert found["divergence"]["value"] == pytest.approx(divergence.sum() / 2)
    assert found["divergence"]["series"]["oak"] == pytest.approx(divergence[[0, 3]].sum() / 2)
    assert found["entropy"]["value"] == pytest.approx(float(crossed.sum() / 2))


def test_what_matures_of_a_kind_over_classes_is_a_column_or_a_column_per_class_of_its_own_type():
    where = "what matures of the kind of task %s is "
    refused = (
        (dict(tags=("classes", TAGS, "reddish")), where % "tags" + "a column for every class of red, green, blue"),
        (dict(tags=("classes", TAGS, dict(FLAGS, blue="oaks"))), where % "tags" + "a column of the type flag"),
        (dict(mix=("proportions", TREES, dict(PARTS, ash="tier"))), where % "mix" + "a column of the type number"),
        (dict(grade=("class", GRADES, dict.fromkeys(GRADES, "tier"))), where % "grade" + "one column, found"),
        (dict(grade=("class", GRADES, "load")), where % "grade" + "a column of the type category"),
        (dict(course=("path", ["near"], "load")), "only a target of a kind that matures into the values of columns"),
    )
    for kinds, message in refused:
        with pytest.raises(ValueError, match=message):
            Schema(schema(kinds))
    stated = Schema(schema(KINDS))
    assert Schema(stated.mapping()).mapping() == stated.mapping() == schema(KINDS)
    assert stated.ripened() == dict(tier=GRADES)
    assert "matures as reddish for red, greenish for green, bluish for blue of at" in stated.text(None)


def test_a_bank_of_a_number_and_kinds_over_classes_completes_one_fit(classes, tmp_path, statement, monkeypatch):
    bank = classes[0]
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    held = Deriver(bank, "cpu", RESIDENT)
    losses = {"number": "squared", "class": "entropy", "classes": "binary", "proportions": "entropy"}
    given = dict(statement, loss=losses, measures=["entropy", "accuracy", "mse", "binary", "divergence"])
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of kinds over classes exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20 and saved["loss"] == given["loss"]
    measures = saved["measures"]
    assert list(measures["accuracy"]["series"]) == GRADES and list(measures["mse"]["series"]) == ["size"]
    assert list(measures["binary"]["series"]) == TAGS and 0 <= measures["accuracy"]["value"] <= 1
    assert list(measures["entropy"]["series"]) == GRADES + TREES and list(measures["divergence"]["series"]) == TREES
    assert np.isfinite([measures[name]["value"] for name in ("entropy", "binary", "mse", "divergence")]).all()
    with np.load(tmp_path / "predictions.npz") as found:
        aim, task, prediction = found["aim"], held.task_of[found["row_index"]], found["prediction"]
        assert set(aim.tolist()) == set(range(len(KINDS)))
        for name in ("grade", "mix"):
            chosen = aim == list(KINDS).index(name)
            assert np.allclose(np.bincount(task[chosen], prediction[chosen])[np.unique(task[chosen])], 1.0)
        chosen = aim != list(KINDS).index("level")
        assert ((prediction[chosen] > 0) & (prediction[chosen] < 1)).all()


@timed("classes.observed")
def observed(held, cfg, name, moment, key, columns):
    from graphspec.views.graph import Graph

    (entry,) = [entry for entry in held.viewed(cfg).graphs if held.manifest["graphs"][entry["source"]]["name"] == name]
    view = Graph.load(held.view(entry["index"], cfg))
    dense, layout = view.dense(), held.registry.encoding.layout
    step = int(np.flatnonzero(np.asarray(view.step_time) == moment)[0])
    rows = (dense.obs_step == step) & (dense.obs_entity == view.keys.index(key))
    (values,) = np.asarray(view.obs_values)[dense.obs_ref[rows]]
    return [values[layout.index(column)] for column in columns]


def test_every_kind_over_classes_matures_into_its_own_form_once_known(classes, monkeypatch):
    bank, case = classes
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    held = Deriver(bank, "cpu", RESIDENT)
    tiers = held.registry.encoding.dictionaries["tier"]
    assert tiers == sorted(GRADES)
    for place, graph in enumerate(case["graphs"]):
        name, target = graph["name"], graph["tasks"]["target"]
        assert np.isnan(observed(held, cfg, name, 2, "s1", ["tier"])).all()
        assert [tiers[int(code)] for code in observed(held, cfg, name, 3, "s1", ["tier"])] == [target[1]]
        assert np.isnan(observed(held, cfg, name, 8, "s5", ["tier"])).all()
        assert observed(held, cfg, name, 4, "s2", FLAGS.values()) == [float(tag in target[2]) for tag in TAGS]
        assert np.isnan(observed(held, cfg, name, 3, "s2", FLAGS.values())).all()
        assert observed(held, cfg, name, 5, "s3", PARTS.values()) == [target[3][tree] for tree in TREES]
    assert case["graphs"][1]["tasks"]["target"][2] == ["green"]


def test_a_memory_reads_a_window_alone_that_holds_no_event(classes):
    import torch

    from conftest import SPLITS
    from tuning.batches.batching import pack_batch
    from tuning.families.models import build as network

    started = time.perf_counter()
    held = Deriver(classes[0], "cpu", RESIDENT)
    cfg = support.configuration(held.registry, dict(view="memory"), {"time/history": 2}, 5, SPLITS)
    blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
    ticks = {graph: np.asarray(held.arrays(graph, cfg)["events"])[:, 2] for graph in {block[0] for block in blocks}}
    quiet = [block for block in blocks if not ((ticks[block[0]] >= 2 * block[1]) & (ticks[block[0]] < 2 * block[2])).any()]
    assert quiet
    b = pack_batch(held, quiet[:1], cfg, "cpu")
    torch.manual_seed(5)
    with torch.inference_mode():
        predicted = network(cfg, held.fitted(cfg)).eval()(b)
    assert len(b["events"]) == 0 and len(predicted) == len(b["series"]) and torch.isfinite(predicted).all()
    LOG.info("quiet window exit blocks=%d seconds=%.3f", len(quiet), time.perf_counter() - started)
