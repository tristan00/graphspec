import gc
import logging
import time

import numpy as np
import pytest
import support
import torch

from conftest import SPLITS, inline, shared
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.targets import TARGETS, valued
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import batches, pack_batch
from tuning.families.model_spec import VIEWS, views
from tuning.families.models import build as network
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("outcomes")
USERS, ITEMS = ("u0", "u1", "u2", "u3"), ("i0", "i1", "i2")
TIMES, CHANGED = (1, 2, 3, 4, 5, 6), ("g0", 4)
TOKENS = ["a", "b", "c"]
KINDS = dict(
    spell=dict(kind="sequence", series=TOKENS, length=3),
    wait=dict(kind="survival", series="churn", width=2, intervals=3),
)
LOSSES = dict(sequence="tokens", survival="hazard")
MEASURES = ["brier", "concordance", "tokens", "token_accuracy", "exact"]
OPENED, HELD, NETS = {}, {}, {}


@timed("outcomes.schema")
def schema(kinds):
    number = dict(type="number", encoder="value")
    asked = {
        name: dict(
            references=["who"],
            columns=[],
            target=dict(stated, about=None, utility=synthetic.valued(stated["series"] if name == "spell" else ["churn"])),
            weight=False,
            absent="stop",
            event=None,
            matures=None,
        )
        for name, stated in kinds.items()
    }
    asked["visit"] = dict(
        references=["who", "at"], columns=[], target=None, weight=False, absent="stop", event=["at", "who"], matures=None
    )
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(load=number, price=number),
        graph=dict(columns=[]),
        nodes=dict(user=dict(columns=["load"], holds=None), item=dict(columns=["price"], holds=None)),
        sets={},
        edges=dict(buys=synthetic.edge("user", "item", [])),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=1,
            references=dict(who=dict(sort="node", kinds=["user"]), at=dict(sort="node", kinds=["item"])),
            kinds=asked,
        ),
    )


@timed("outcomes.stated")
def stated(kind, at, place, changed):
    if kind == "wait":
        return dict(duration=1 + (3 * at + place + 4 * changed) % 8, happened=(at + place) % 3 != 0)
    if changed:
        return ["c", "c", "a"]
    return [TOKENS[(2 * at + place + step) % 3] for step in range((at + place) % 4)]


@timed("outcomes.graph")
def graph(place, changed):
    names = list(KINDS) + ["visit"]
    tasks = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], signal=[], signal_observed=[])
    references = dict(who=[], at=[])
    for at, moment in enumerate(TIMES):
        for kind in names:
            tasks["key"].append(len(tasks["key"]))
            tasks["kind"].append(kind)
            tasks["time"].append(moment)
            tasks["split"].append(None if kind == "visit" else "train" if at < len(TIMES) // 2 else "validation")
            tasks["weight"].append(None)
            tasks["known"].append(None)
            tasks["signal"].append([0.1 * at + place])
            tasks["signal_observed"].append([True])
            touched = ("g%d" % place, at) == CHANGED and changed
            tasks["target"].append(None if kind == "visit" else stated(kind, at, place, touched))
            references["who"].append(USERS[(at + place) % len(USERS)])
            references["at"].append(ITEMS[at % len(ITEMS)] if kind == "visit" else None)
    users = dict(key=list(USERS), since=[0] * 4, until=[None] * 4, load=[1.0 + at + place for at in range(4)])
    items = dict(key=list(ITEMS), since=[0] * 3, until=[None] * 3, price=[2.0 * at - place for at in range(3)])
    buys = dict(source=["u0", "u1"], target=["i1", "i2"], present=[True] * 2, since=[1, 2], until=[None] * 2)
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(user=list(USERS), item=list(ITEMS)),
        values=[
            dict(form="state", parameters={}, rows=users, kind="user"),
            dict(form="state", parameters={}, rows=items, kind="item"),
        ],
        edges=[dict(form="state", parameters={}, table="buys", ends=("user", "item"), rows=buys)],
        tasks=dict(tasks, references=references),
    )


@timed("outcomes.case")
def case(changed):
    return dict(schema=schema(KINDS), graphs=[graph(place, changed) for place in range(2)])


@timed("outcomes.made")
def made(root):
    started = time.perf_counter()
    for name in ("stated", "changed"):
        build(synthetic.stage(root, name, case(name == "changed")), root / name, 1, ("train",))
    LOG.info("banks of outcomes exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture(scope="module")
def banks():
    root = shared("outcomes", made)
    yield {name: root / name for name in ("stated", "changed")}
    OPENED.clear()
    gc.collect()


@timed("outcomes.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", RESIDENT)
    return OPENED[name]


@timed("outcomes.read")
def read(held):
    found = {}
    kinds = held.registry.schema.tasks["kinds"]
    for row in held.rows:
        found.setdefault((row["graph"], row["kind"], row["ticket"]), []).append(row)
    return {key: TARGETS.named(kinds[key[1]]["target"]["kind"]).read(rows) for key, rows in found.items()}


def test_every_kind_of_2_13_reads_back_as_stated(banks):
    held = opened(banks, "stated")
    schema, given = held.registry.schema, case(False)
    assert schema.targets() == ("sequence", "survival") and Schema(schema.mapping()).mapping() == schema.mapping()
    columns = dict(series="STRING", place="INT32", tau="FLOAT", duration="INT64", happened="BOOL")
    assert schema.target_columns() == columns and schema.series() == (*TOKENS, "churn")
    wanted = {
        (graph["name"], kind, key): target
        for graph in given["graphs"]
        for kind, key, target in zip(graph["tasks"]["kind"], graph["tasks"]["key"], graph["tasks"]["target"])
        if target is not None
    }
    assert read(held) == wanted and [] in wanted.values() and ["c", "a", "b"] in wanted.values()
    waits = [target for (_, kind, _), target in wanted.items() if kind == "wait"]
    assert {target["happened"] for target in waits} == {True, False} and max(row["duration"] for row in waits) > 6
    assert held.split["series"] == [] and held.scale.tolist() == [1.0] * 4
    assert np.isnan(held.fields["tau"][held.aims == 0]).all() and (held.fields["tau"][held.aims == 1] > 0).all()
    written = support.written(banks["stated"].parent / "stated.staging", schema)
    for entry in held.manifest["graphs"]:
        truth = Facts.load(held.truth(entry["index"]))
        found = truth.stated(held.registry, held.splits, held.manifest["observers"])
        assert found["targets"] == written[truth.name]["targets"]


def test_a_sequence_states_its_tokens_and_its_greatest_length_and_holds_a_point_per_token_and_place():
    where, kind = "the target of the kind of task spell", TARGETS.named("sequence")
    target = Schema(schema(KINDS)).tasks["kinds"]["spell"]["target"]
    assert target["length"] == 3 and target["series"] == tuple(TOKENS)
    for wrong in (["a", "b", "c", "a"], ["d"], "ab", [1]):
        with pytest.raises(ValueError, match="as a list of at most 3 tokens of a, b, c, possibly none"):
            kind.staged(wrong, target, where)
    rows = kind.staged(["b", "a"], target, where)
    assert [(row["place"], row["series"]) for row in rows[:4]] == [(0, "a"), (0, "b"), (0, "c"), (1, "a")]
    assert [row["y"] for row in rows] == [0, 1, 0, 1, 0, 0, 0, 0, 0] and kind.read(rows[::-1]) == ["b", "a"]
    points = dict(task=[0] * 9, graph=["g0"] * 9, **{name: [row[name] for row in rows] for name in rows[0]})
    assert kind.points(points, target, where) == dict(series=points["series"], place=points["place"])
    refused = (
        (dict(points, place=[3] + points["place"][1:]), "state their place, a whole number from 0 below 3"),
        (dict(points, series=["d"] + points["series"][1:]), "names the token 'd', which its kind does not state"),
        (dict(points, place=[1] + points["place"][1:]), "a task of graph g0 holds under .* each once"),
        (dict(points, y=[1] + points["y"][1:]), "state 1 for the token at each place of the sequence of a task"),
        (dict(points, y=[0] * 6 + [1, 0, 0]), "the places that hold a token coming before those that hold none"),
    )
    for given, message in refused:
        with pytest.raises(ValueError, match=message):
            kind.points(given, target, where)
    restated = (
        (dict(kind="sequence", series=TOKENS), "lacks length"),
        (dict(kind="sequence", series=TOKENS, length=0), "the greatest length of the target of .* at least 1, found 0"),
        (dict(kind="sequence", series=[], length=2), "names its series as a list of distinct names"),
    )
    for given, message in restated:
        with pytest.raises(ValueError, match=message):
            Schema(schema(dict(spell=given)))
    with pytest.raises(ValueError, match="only a target of a kind that matures into the values of columns"):
        held = schema(KINDS)
        held["tasks"]["kinds"]["spell"]["matures"] = dict(reference="who", column="load")
        Schema(held)
    tasks, values = valued(target, dict(task=[2] * 3, series=TOKENS, y=[0.5, 0.25, 0.0]))
    assert tasks.tolist() == [2] and values.tolist() == [0.75]


def test_a_sequence_is_predicted_place_by_place_against_its_end_and_scored_per_token():
    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    scores = torch.tensor([0.5, 2.0, -1.0, -0.5, 1.0, 0.2, 0.3, 1.5])
    target = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    place, row = torch.tensor([0.0, 0.0, 1.0, 1.0] * 2), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    weight = torch.tensor([1.0] * 4 + [3.0] * 4)
    batch = {"target": target, "weight": weight, "point_row": row, "tasks": 2, "field/place": place}
    logged = [torch.log_softmax(torch.cat((scores[at : at + 2], torch.zeros(1))), 0) for at in range(0, 8, 2)]
    parts = torch.stack([-logged[0][1], -logged[1][2], -logged[2][0], -logged[3][0]])
    expected = float((parts * torch.tensor([1.0, 1.0, 3.0, 3.0])).sum() / 8)
    assert float(LOSSES.named("tokens")(scores, batch, None)) == pytest.approx(expected)
    fields = dict(place=place.numpy())
    probability = predicted(("sequence",), np.zeros(8, np.int64), row.numpy(), scores.numpy(), fields)
    assert np.allclose(probability, torch.cat([found.exp()[:2] for found in logged]).numpy())
    series = np.tile([0, 1], 4)
    held = scored(target.numpy(), probability, row.numpy(), series, ["x", "y"], np.ones(2), fields)
    found = measured(["tokens", "token_accuracy", "exact"], held, dict(x="sequence", y="sequence"))
    assert found["tokens"]["value"] == pytest.approx(float(parts.mean()))
    assert found["tokens"]["series"] == pytest.approx(dict(x=float(parts[2:].sum() / 4), y=float(parts[0] / 4)))
    assert found["token_accuracy"]["value"] == 0.75 and found["token_accuracy"]["series"] == dict(x=0.75, y=0.75)
    assert found["exact"]["value"] == 0.5 and found["exact"]["series"] == dict(x=0.5, y=0.5)


def test_a_time_to_an_event_states_its_duration_and_outcome_and_holds_a_point_per_interval():
    where, kind = "the target of the kind of task wait", TARGETS.named("survival")
    target = Schema(schema(KINDS)).tasks["kinds"]["wait"]["target"]
    assert (target["width"], target["intervals"], target["series"]) == (2, 3, ("churn",))
    for wrong in (dict(duration=0, happened=True), dict(duration=2.5, happened=True), dict(duration=2), [3, True]):
        with pytest.raises(ValueError, match="as its duration, a whole number of at least 1 of the unit of its graph"):
            kind.staged(wrong, target, where)
    for stated_target, labels in ((dict(duration=4, happened=True), [0, 1, 0]), (dict(duration=9, happened=True), [0] * 3)):
        rows = kind.staged(stated_target, target, where)
        assert [row["tau"] for row in rows] == [2.0, 4.0, 6.0] and [row["y"] for row in rows] == labels
        assert kind.read(rows) == stated_target
    rows = kind.staged(dict(duration=3, happened=False), target, where)
    points = dict(task=[0] * 3, graph=["g1"] * 3, **{name: [row[name] for row in rows] for name in rows[0]})
    named = kind.inputs + kind.outcomes
    assert kind.inputs == ("tau",) and kind.outcomes == ("duration", "happened")
    assert kind.points(points, target, where) == dict(series=["churn"] * 3, **{name: points[name] for name in named})
    refused = (
        (dict(points, duration=[3, 3, None]), "state the duration of their case, a whole number of at least 1"),
        (dict(points, tau=[2.0, 4.0, 8.0]), "a task of graph g1 holds under .* 3 intervals of 2, by the end"),
        (dict(points, happened=[False, True, False]), "a task of graph g1 holds under .* one duration and one outcome"),
        (dict(points, y=[0.0, 1.0, 0.0]), "state 1 for the interval in which the event of their task happened"),
    )
    for given, message in refused:
        with pytest.raises(ValueError, match=message):
            kind.points(given, target, where)
    restated = (
        (dict(KINDS["wait"], width=0), "the width of the intervals of the target of .* at least 1, found 0"),
        ({key: KINDS["wait"][key] for key in ("kind", "series", "width")}, "lacks intervals"),
    )
    for given, message in restated:
        with pytest.raises(ValueError, match=message):
            Schema(schema(dict(wait=given)))
    ordered = schema(KINDS)
    ordered["time"] = dict(model="order", units=[], ties="key")
    with pytest.raises(ValueError, match="is a time to an event, which counts on a clock, and the time of the sch"):
        Schema(ordered)
    with pytest.raises(ValueError, match="only a target of a kind that matures into the values of columns"):
        held = schema(KINDS)
        held["tasks"]["kinds"]["wait"]["matures"] = dict(reference="who", column="load")
        Schema(held)
    tasks, values = valued(target, dict(task=[5] * 3, series=["churn"] * 3, y=[0.25, 0.5, 0.0]))
    assert tasks.tolist() == [5] and values.tolist() == [0.75]


@timed("outcomes.agreed")
def agreed(cases, risk, horizon):
    lasted, happened = cases[:, 0], cases[:, 1] == 1
    pairs = [
        (first, second)
        for first in range(len(cases))
        for second in range(len(cases))
        if happened[first] and lasted[first] <= horizon and lasted[second] > lasted[first]
    ]
    found = sum(1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0 for i, j in pairs)
    return found / len(pairs)


@timed("outcomes.kept")
def kept(cases, moment):
    lasted, happened, found = cases[:, 0], cases[:, 1] == 1, 1.0
    for at in sorted(set(lasted.tolist())):
        if at <= moment:
            found *= 1 - np.sum((lasted == at) & ~happened) / np.sum(lasted >= at)
    return found


def test_a_time_to_an_event_is_a_likelihood_over_intervals_that_reads_cases_ended_before_their_event():
    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    cases = np.asarray([[3, 1], [4, 0], [9, 1], [1, 1], [4, 1]], np.float64)
    tau, aims = np.asarray([2.0, 4.0, 6.0]), np.zeros(15, np.int64)
    labels = ((cases[:, 1:] == 1) & (tau - 2 < cases[:, :1]) & (cases[:, :1] <= tau)).reshape(-1)
    scores = torch.tensor([0.5, -1.0, 2.0, -0.5, 0.0, 1.0, -2.0, -1.5, -1.0, 1.5, 0.0, 0.0, 0.0, 0.3, 0.1])
    weight = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]).repeat_interleave(3)
    fields = {"tau": np.tile(tau, 5), "duration": cases[:, 0].repeat(3), "happened": cases[:, 1].repeat(3)}
    target, row = torch.tensor(labels, dtype=torch.float32), torch.arange(5).repeat_interleave(3)
    batch = dict(target=target, weight=weight, point_row=row, tasks=5)
    batch.update({"field/" + key: torch.tensor(value) for key, value in fields.items()})
    seen = (torch.tensor(fields["tau"]) <= torch.tensor(fields["duration"])) | (target == 1)
    error = torch.nn.functional.binary_cross_entropy_with_logits(scores, target, reduction="none")
    expected = float((weight * error * seen).sum() / 6)
    assert float(LOSSES.named("hazard")(scores, batch, None)) == pytest.approx(expected)
    hazard = torch.sigmoid(scores).numpy().reshape(5, 3)
    chance = hazard * np.c_[np.ones(5), np.cumprod(1 - hazard, 1)[:, :-1]]
    probability = predicted(("survival",), aims, row.numpy(), scores.numpy(), fields)
    assert np.allclose(probability, chance.reshape(-1))
    order = np.arange(14, -1, -1)
    shuffled = {key: value[order] for key, value in fields.items()}
    turned = predicted(("survival",), aims, row.numpy()[order], scores.numpy()[order], shuffled)
    assert np.allclose(turned, chance.reshape(-1)[order])
    held = scored(labels, probability, row.numpy(), aims, ["churn"], np.ones(1), fields)
    found = measured(["concordance", "brier"], held, dict(churn="survival"))
    assert found["concordance"]["value"] == pytest.approx(agreed(cases, chance.sum(1), 6.0))
    assert found["concordance"]["series"]["churn"] == found["concordance"]["value"]
    parts = []
    for (moment, happened), survives in zip(cases, 1 - np.cumsum(chance, 1)):
        for end, alive in zip(tau, survives):
            if happened and moment <= end:
                parts.append(alive**2 / kept(cases, moment - 1))
            else:
                parts.append((1 - alive) ** 2 / kept(cases, end) if moment > end else 0.0)
    assert found["brier"]["value"] == pytest.approx(np.mean(parts)) == found["brier"]["series"]["churn"]
    rng = np.random.default_rng(3)
    many = np.c_[rng.integers(1, 9, 60), rng.integers(0, 2, 60)].astype(np.float64)
    values, owner = (rng.integers(0, 4, 60) - 2.0).repeat(3), np.arange(60).repeat(3)
    fields = {"tau": np.tile(tau, 60), "duration": many[:, 0].repeat(3), "happened": many[:, 1].repeat(3)}
    chance = predicted(("survival",), np.zeros(180, np.int64), owner, values, fields)
    held = scored(np.zeros(180), chance, owner, np.zeros(180, np.int64), ["churn"], np.ones(1), fields)
    risk = np.bincount(owner, chance)
    assert len(np.unique(risk)) == 4
    found = measured(["concordance"], held, dict(churn="survival"))["concordance"]["value"]
    assert found == pytest.approx(agreed(many, risk, 6.0))


def test_a_bank_of_the_kinds_of_2_13_completes_one_fit(banks, tmp_path, statement, monkeypatch):
    bank = banks["stated"]
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    held = opened(banks, "stated")
    given = dict(statement, loss=LOSSES, measures=MEASURES)
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of the kinds of 2.13 exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20 and saved["loss"] == LOSSES
    found = saved["measures"]
    assert list(found["exact"]["series"]) == TOKENS and np.isfinite(found["tokens"]["value"])
    assert list(found["brier"]["series"]) == ["churn"] and 0 < found["brier"]["value"]
    assert all(0 <= found[name]["value"] <= 1 for name in ("token_accuracy", "exact", "concordance"))
    with np.load(tmp_path / "predictions.npz") as stored:
        task, prediction, aim = held.task_of[stored["row_index"]], stored["prediction"], stored["aim"]
        place, tau = stored["field/place"], stored["field/tau"]
        assert set(aim.tolist()) == {0, 1} and ((prediction > 0) & (prediction < 1)).all()
        assert np.isnan(place[aim == 1]).all() and np.isnan(tau[aim == 0]).all()
        total = np.bincount(task[aim == 0] * 3 + place[aim == 0].astype(np.int64), prediction[aim == 0])
        assert (total[total > 0] < 1).all()
        total = np.bincount(task[aim == 1], prediction[aim == 1])
        assert (total[np.unique(task[aim == 1])] < 1).all()


def test_the_batch_a_model_reads_holds_the_inputs_of_a_sequence_and_a_time_to_an_event_and_no_outcome(banks):
    held = opened(banks, "stated")
    cfg = support.configuration(held.registry, dict(view="snapshot"), {"time/history": 2}, 5, SPLITS)
    assert held.registry.schema.targets() == ("sequence", "survival") and sorted(held.inputs) == ["place", "tau"]
    assert support.outcomeless(held, cfg) == (["field/place", "field/tau"], True)


@timed("outcomes.predicted")
def predicted(banks, family, name):
    if (family, name) not in HELD:
        held = opened(banks, name)
        cfg = support.configuration(held.registry, dict(view=family), {"time/history": 2}, 5, SPLITS)
        if family not in NETS:
            torch.manual_seed(5)
            NETS[family] = network(cfg, held.fitted(cfg)).eval()
        blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
        found = {}
        with torch.inference_mode():
            for graph in sorted({block[0] for block in blocks}):
                chosen = [block for block in blocks if block[0] == graph]
                for group in batches(chosen, len(chosen), held.sizes(cfg), None, frame_budget=1024):
                    b = pack_batch(held, group, cfg, "cpu")
                    found.update(zip(b["rows"].tolist(), NETS[family](b).tolist()))
        HELD[family, name] = found
    return HELD[family, name]


@pytest.mark.parametrize("family", VIEWS)
def test_changing_only_a_target_of_the_kinds_of_2_13_leaves_what_every_family_reads_unchanged(banks, family):
    stated_rows, changed_rows = opened(banks, "stated").rows, opened(banks, "changed").rows
    assert family in views(opened(banks, "stated").registry)
    held, changed = predicted(banks, family, "stated"), predicted(banks, family, "changed")
    assert sorted(held) == sorted(changed) == list(range(len(stated_rows)))
    assert held == pytest.approx(changed, rel=1e-6, abs=1e-6)
    tasks = {}
    for row, value in held.items():
        tasks.setdefault((stated_rows[row]["graph"], stated_rows[row]["ticket"]), set()).add(round(value, 6))
    assert all(len(values) > 1 for values in tasks.values())
    identity = ("graph", "kind", "ticket", "series", "place", "tau")
    assert [[row.get(name) for name in identity] for row in stated_rows] == [
        [row.get(name) for name in identity] for row in changed_rows
    ]
    assert {row["kind"] for row, other in zip(stated_rows, changed_rows) if row != other} == set(KINDS)
