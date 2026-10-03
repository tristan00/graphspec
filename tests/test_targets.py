import copy
import logging
import time

import numpy as np
import pytest
import support

from conftest import SPLITS, inline
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.targets import TARGETS, UTILITIES, valued
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("targets")
SITES = ("s0", "s1", "s2", "s3", "s4", "s5")
VALUE = dict(type="number", encoder="value")
TIMES = (1, 2, 3, 4, 5, 6, 7, 8)
LATE = (1 << 24) + 1


@timed("targets.asked")
def asked(target):
    return dict(references=["at"], columns=[], target=target, weight=False, absent="stop", event=None, matures=None)


@timed("targets.aimed")
def aimed(kind, series):
    return dict(kind=kind, series=series, about=None, utility=synthetic.valued([series] if kind == "number" else series))


@timed("targets.mixed")
def mixed(kinds):
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties=None),
        observers=False,
        columns=dict(load=VALUE),
        graph=dict(columns=[]),
        nodes=dict(site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(link=synthetic.edge("site", "site", [])),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(at=dict(sort="node", kinds=["site"])),
            kinds={name: asked(target) for name, target in kinds.items()},
        ),
    )


@timed("targets.graph")
def graph(place, stated):
    kinds = list(stated)
    values = [
        dict(
            form="state",
            parameters={},
            rows=dict(
                key=list(SITES), since=[0] * 6, until=[None] * 6, load=[1.0 + at + place for at in range(6)]
            ),
            kind="site",
        )
    ]
    links = dict(
        source=list(SITES[:-1]), target=list(SITES[1:]), present=[True] * 5, since=[0] * 5, until=[None] * 5
    )
    edges = [dict(form="state", parameters={}, rows=links, table="link", ends=("site", "site"))]
    kind = [kinds[at % len(kinds)] for at in range(len(TIMES))]
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(site=list(SITES)),
        values=values,
        edges=edges,
        tasks=dict(
            key=list(range(len(TIMES))),
            kind=kind,
            time=list(TIMES),
            split=["train" if (at // len(kinds)) % 2 == 0 else "validation" for at in range(len(TIMES))],
            weight=[None] * len(TIMES),
            references=dict(at=[SITES[at % 6] for at in range(len(TIMES))]),
            target=[stated[kind[at]](at, place) for at in range(len(TIMES))],
            known=[None] * len(TIMES),
        ),
    )


@timed("targets.number")
def number(at, place):
    return 0.5 * at + place + 0.1


@timed("targets.course")
def course(at, place):
    return [[1, "near", 1.0 + at + place], [LATE, "near", 2.0 * at - place], [1, "far", 0.25 * at * at]]


@timed("targets.case")
def case(kinds, targets):
    return dict(schema=mixed(kinds), graphs=[graph(place, targets) for place in range(2)])


@timed("targets.pointwise")
def pointwise(held, stated):
    given, seen, result = {}, {}, []
    for graph in stated["graphs"]:
        tasks = graph["tasks"]
        for kind, key, target in zip(tasks["kind"], tasks["key"], tasks["target"]):
            given[graph["name"], key] = [(0, target)] if kind == "level" else [(tau, y) for tau, _, y in target]
    for row in held.rows:
        key = (row["graph"], row["ticket"])
        seen[key] = seen.get(key, -1) + 1
        result.append(given[key][seen[key]])
    return result


@timed("targets.read")
def read(held, name):
    found = {}
    kinds = held.registry.schema.tasks["kinds"]
    for row in held.rows:
        found.setdefault((row["graph"], row["kind"], row["ticket"]), []).append(row)
    return {
        key: TARGETS.named(kinds[key[1]]["target"]["kind"]).read(rows)
        for key, rows in found.items()
        if key[0] == name
    }


@timed("targets.share")
def share(at, place):
    return [(at % 4) / 4, 1.0 + at]


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    import shares

    from tuning import HEADS, LOSSES, MEASURES

    entries = (
        (TARGETS, "share", shares.Share()),
        (UTILITIES, "expected", shares.Expected()),
        (HEADS, "share", shares.HEAD),
        (LOSSES, "crossed", shares.Crossed()),
        (MEASURES, "absolute", shares.Absolute()),
    )
    for names, name, entry in entries:
        names.register(name, entry)
    try:
        started = time.perf_counter()
        root = tmp_path_factory.mktemp("share")
        target = dict(kind="share", series="part", about=None, utility=dict(name="expected", better="higher"))
        stated = dict(schema=mixed(dict(part=target)), graphs=[graph(0, dict(part=share))])
        build(synthetic.stage(root, "share", stated), root / "bank", 1, ("train",))
        LOG.info("bank of a registered kind of target exit seconds=%.3f", time.perf_counter() - started)
        yield root / "bank", stated
    finally:
        for names, name, _ in entries:
            del names.held[name]


@pytest.fixture(scope="module")
def numbered(tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("mixed")
    kinds = dict(level=aimed("number", "size"), course=aimed("path", ["near", "far"]))
    stated = case(kinds, dict(level=number, course=course))
    build(synthetic.stage(root, "mixed", stated), root / "bank", 1, ("train",))
    LOG.info("bank of mixed targets exit seconds=%.3f", time.perf_counter() - started)
    return root / "bank", stated


def restated(**target):
    held = copy.deepcopy(synthetic.varied("task")["schema"])
    held["tasks"]["kinds"]["buy"]["target"] = dict(held["tasks"]["kinds"]["buy"]["target"], **target)
    return held


def test_kinds_of_target_and_utilities_are_registered_and_a_target_names_them():
    assert TARGETS.names()[:2] == ("number", "path") and UTILITIES.names()[0] == "weighted"
    schema = Schema(restated())
    assert schema.tasks["kinds"]["buy"]["target"]["utility"] == dict(
        name="weighted", better="higher", weights=dict(outcome=1.0)
    )
    assert schema.targets() == ("number",) and schema.target_columns() == {}
    assert Schema(synthetic.varied("path")["schema"]).target_columns() == dict(tau="INT64", series="STRING")
    with pytest.raises(KeyError, match="no kind of target is registered under the name 'unheard'"):
        Schema(restated(kind="unheard"))
    with pytest.raises(ValueError, match="the target of the kind of task buy names its one series without colons"):
        Schema(restated(series=["outcome"]))
    with pytest.raises(KeyError, match="no utility is registered under the name 'profit'"):
        Schema(restated(utility=dict(name="profit", better="higher")))
    with pytest.raises(ValueError, match="states which value is better as one of higher, lower, found 'more'"):
        Schema(restated(utility=dict(name="weighted", better="more", weights=dict(outcome=1.0))))
    with pytest.raises(ValueError, match="weighs every series of its target, outcome, by a finite number"):
        Schema(restated(utility=dict(name="weighted", better="higher", weights=dict(other=1.0))))
    with pytest.raises(ValueError, match="holds fields it does not know: rate"):
        Schema(restated(utility=dict(name="weighted", better="lower", weights=dict(outcome=1.0), rate=2)))


def test_the_utility_of_a_target_is_one_number_per_task():
    path = Schema(synthetic.varied("path")["schema"]).tasks["kinds"]["buy"]["target"]
    stated = dict(path, utility=dict(path["utility"], weights=dict(level=2.0, trend=-1.0)))
    points = dict(task=[4, 4, 9, 4], series=["level", "trend", "level", "level"], y=[1.0, 3.0, 5.0, 0.5])
    tasks, values = valued(stated, points)
    assert tasks.tolist() == [4, 9] and values.tolist() == [2.0 - 3.0 + 1.0, 10.0]
    number = Schema(restated()).tasks["kinds"]["buy"]["target"]
    tasks, values = valued(number, dict(task=[1, 0], series=["outcome", "outcome"], y=[2.5, -1.0]))
    assert tasks.tolist() == [0, 1] and np.array_equal(values, [-1.0, 2.5])


def test_a_path_states_each_tau_a_whole_number_of_at_least_1_that_64_bits_hold():
    started = time.perf_counter()
    kind, where, longest = TARGETS.named("path"), "the graph g0, the task 3", (1 << 63) - 1
    target = Schema(synthetic.varied("path")["schema"]).tasks["kinds"]["buy"]["target"]
    for tau in (1.5, 2.0, 0, True, "1", None):
        with pytest.raises(ValueError, match="each its tau, a whole number of at least 1 of the unit of its graph,"):
            kind.staged([[1, "level", 1.0], [tau, "trend", 2.0]], target, where)
    assert kind.staged([[longest, "level", 1.0]], target, where) == [dict(tau=longest, series="level", y=1.0)]
    with pytest.raises(
        ValueError,
        match="the graph g0, the task 3 states 9223372036854775808 as the tau of a point, past 9223372036854775807, "
        "the greatest whole number of 64 bits$",
    ):
        kind.staged([[longest + 1, "level", 1.0]], target, where)
    where, points = "the target of the kind of task buy", dict(task=[0, 0], graph=["g0"] * 2, y=[1.0, 2.0])
    points["series"] = ["level", "trend"]
    for tau in ([1, 0], [1, None], [1, 2.5]):
        with pytest.raises(ValueError, match="the points of %s state their tau, a whole number of at least 1$" % where):
            kind.points(dict(points, tau=tau), target, where)
    assert kind.points(dict(points, tau=[longest, 1]), target, where) == dict(series=points["series"], tau=[longest, 1])
    LOG.info("whole times of a path exit seconds=%.3f", time.perf_counter() - started)


def test_a_statement_names_a_loss_for_every_kind_of_target_of_the_bank_and_measures_that_read_each(
    numbered, tmp_path, statement
):
    bank, config = numbered[0], dict(seed=statement["seed"], splits=SPLITS, pipeline=[])
    both = dict(number="squared", path="balanced")
    refused = (
        (
            ValueError,
            dict(),
            "names no loss for the kind of target path of the bank; the losses that read it are squared, balanced$",
        ),
        (
            ValueError,
            dict(loss=dict(both, course="squared")),
            "names a loss for the kind of target course, which the bank does not hold; it holds number, path$",
        ),
        (KeyError, dict(loss=dict(both, path="absolute")), "no loss is registered under the name 'absolute'"),
        (
            ValueError,
            dict(loss=both, measures=["r2"]),
            "no measure the statement names reads the kind of target path of the bank; the measures that read it "
            "are mse, balanced$",
        ),
    )
    for error, changes, message in refused:
        with pytest.raises(error, match=message):
            fit(bank, tmp_path, config, [], 8, time.perf_counter() + 30, state(dict(statement, **changes)))
    assert not list(tmp_path.iterdir())


def test_a_number_and_a_path_in_one_bank_read_back_as_stated_and_are_scaled_per_series(numbered):
    bank, stated = numbered
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    schema = held.registry.schema
    assert schema.targets() == ("number", "path") and schema.series() == ("size", "near", "far")
    assert schema.measured() == dict(size="number", near="path", far="path")
    for graph in stated["graphs"]:
        tasks = graph["tasks"]
        wanted = {
            (graph["name"], kind, key): target
            for kind, key, target in zip(tasks["kind"], tasks["key"], tasks["target"])
        }
        assert read(held, graph["name"]) == wanted
    back, meant = support.read_back(held), support.meant(stated)
    assert list(back) == list(meant) == ["g0", "g1"]
    for name, found in back.items():
        assert support.differing(found, meant[name], schema) == [] and len(found["targets"]["y"]) == 16, name
    trained = held.split["indices"]["train"]
    assert [entry["name"] for entry in held.split["series"]] == ["size", "near", "far"]
    for entry in held.split["series"]:
        values = [held.rows[row]["y"] for row in trained if held.rows[row]["series"] == entry["name"]]
        assert entry["mean"] == pytest.approx(np.mean(values)) and entry["sd"] == pytest.approx(np.std(values, ddof=1))
    path, tau = np.asarray([row["kind"] == "course" for row in held.rows]), held.fields["tau"]
    assert np.array_equal(held.aims, path.astype(np.int64)) and tau.dtype == np.int64 and (tau[~path] == 0).all()
    assert held.targets.dtype == np.float64 and LATE in tau.tolist()
    assert list(zip(tau.tolist(), held.targets.tolist())) == pointwise(held, stated)
    assert any(float(np.float32(y)) != y for _, y in pointwise(held, stated))


def test_a_head_gives_every_point_the_prediction_of_the_head_of_its_kind_of_target():
    from types import SimpleNamespace

    import torch

    from tuning.families.layers import Head

    schema = SimpleNamespace(targets=lambda: ("number", "path"), series=lambda: ("size", "near", "far"))
    torch.manual_seed(3)
    head = Head(5, {"hidden": 4, "time/encoding": "log_cosine"}, SimpleNamespace(schema=schema), 2)
    features = torch.randn(3, 5)
    b = {
        "point_row": torch.tensor([0, 1, 1, 2, 0]),
        "series": torch.tensor([1, 0, 2, 0, 1]),
        "aim": torch.tensor([1, 0, 1, 0, 1]),
        "field/tau": torch.tensor([1.0, float("nan"), 2.0, float("nan"), 3.0]),
    }
    found = head(features, b, torch.zeros(0, 2))
    h, series = head.task(features)[b["point_row"]], head.series(b["series"])
    assert found.shape == (5,) and torch.isfinite(found).all()
    for code in (0, 1):
        points = (b["aim"] == code).nonzero().flatten()
        torch.testing.assert_close(found[points], head.kinds[code](h[points], series[points], b, points))


def test_a_bank_of_numbers_and_paths_completes_one_fit_with_a_head_per_kind_of_target(
    numbered, tmp_path, statement, monkeypatch
):
    bank, stated = numbered
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    given = dict(statement, loss=dict(number="balanced", path="balanced"), measures=["balanced", "mse"])
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of mixed targets exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["loss"] == given["loss"]
    assert sorted(saved["measures"]["balanced"]["series"]) == ["far", "near", "size"]
    assert saved["holdout_rows"] == len(held.members(cfg, 1)) and saved["statement"] == given
    with np.load(tmp_path / "predictions.npz") as found:
        aim, tau = found["aim"], found["field/tau"]
        assert set(aim.tolist()) == {0, 1} and tau.dtype == np.int64 and (tau[aim == 0] == 0).all()
        assert set(tau[aim == 1].tolist()) == {1, LATE}
        assert np.isfinite(found["prediction"]).all() and len(aim) == saved["holdout_rows"]
        rows, prediction, size = pointwise(held, stated), found["prediction"], found["series"] == 0
        aims = np.asarray([rows[row][1] for row in found["row_index"].tolist()])
        assert found["target"].dtype == np.float64 and np.array_equal(found["target"], aims) and size.any()
        assert saved["measures"]["mse"]["series"]["size"] == float(((prediction - aims)[size] ** 2).mean())


def test_the_batch_a_model_reads_holds_the_inputs_of_a_path_and_of_a_registered_kind_and_no_outcome(
    numbered, shared, monkeypatch
):
    started = time.perf_counter()
    for (bank, _), kinds, read in ((numbered, ("number", "path"), ["field/tau"]), (shared, ("share",), ["field/trials"])):
        held = Deriver(bank, "cpu", support.SEED, RESIDENT)
        assert held.registry.schema.targets() == kinds
        assert support.outcomeless(held, inline(bank, ("snapshot",), monkeypatch)["snapshot"]) == (read, True)
    LOG.info("outcomes of numbers, paths and shares exit seconds=%.3f", time.perf_counter() - started)


def test_a_kind_of_target_registered_by_a_test_builds_reads_back_and_completes_one_fit(
    shared, tmp_path, statement, monkeypatch
):
    bank, stated = shared
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    schema = held.registry.schema
    assert schema.targets() == ("share",) and schema.target_columns() == dict(trials="FLOAT")
    tasks = stated["graphs"][0]["tasks"]
    assert read(held, "g0") == {("g0", "part", key): target for key, target in zip(tasks["key"], tasks["target"])}
    back, meant = support.read_back(held), support.meant(stated)
    assert list(back) == ["g0"] and support.differing(back["g0"], meant["g0"], schema) == []
    assert held.split["series"] == [] and held.center.tolist() == [0.0] and held.scale.tolist() == [1.0]
    points = dict(
        task=[row["ticket"] for row in held.rows],
        series=[row["series"] for row in held.rows],
        y=[row["y"] for row in held.rows],
        trials=held.fields["trials"],
    )
    keys, values = valued(schema.tasks["kinds"]["part"]["target"], points)
    assert values.tolist() == [(at % 4) / 4 * (1.0 + at) for at in keys.tolist()]
    with pytest.raises(ValueError, match="a share of the target of the kind of task part lies outside 0 and 1"):
        TARGETS.named("share").points(
            dict(task=[0], graph=["g0"], y=[1.5], trials=[2.0]), dict(series=("part",)), "the target of the kind of task part"
        )
    cfg = inline(bank, ("snapshot",), monkeypatch)["snapshot"]
    given = dict(statement, loss=dict(share="crossed"), measures=["absolute"])
    refused = (
        (dict(), "names no loss for the kind of target share of the bank; the losses that read it are crossed$"),
        (
            dict(loss=dict(share="squared")),
            "names the loss squared for the kind of target share, which it does not read; the losses that read it are "
            "crossed$",
        ),
        (
            dict(loss=given["loss"], measures=["absolute", "mse"]),
            "names the measure mse, which reads targets of the kinds number, path, decision and no kind of target of "
            "the bank, which holds share$",
        ),
    )
    for changes, message in refused:
        with pytest.raises(ValueError, match=message):
            fit(bank, tmp_path, cfg, [], 16, time.perf_counter() + 30, state(dict(statement, **changes)))
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of a registered kind of target exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["loss"] == dict(share="crossed")
    assert list(saved["measures"]) == ["absolute"] and list(saved["measures"]["absolute"]["series"]) == ["part"]
    with np.load(tmp_path / "predictions.npz") as found:
        assert ((found["prediction"] > 0) & (found["prediction"] < 1)).all()
        assert found["field/trials"].tolist() == held.fields["trials"][found["row_index"]].tolist()
