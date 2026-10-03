import numpy as np
import pyarrow as pa
import pytest
from support import configuration

from conftest import SPLITS
from graphspec.build.output import Stage
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, write_json
from graphspec.schema.schema import TARGET, Schema
from graphspec.sources.synthetic import Counter, graph_rows, uniform
from tuning.bank.bank import Deriver
from tuning.bank.build import build

SETTINGS = dict(entities=40, steps=12, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
SHAPE = uniform(4, 2, "path")


def staged_points(root, name, edit):
    tables = graph_rows(np.random.default_rng(3), 0, "graph_000", SHAPE, SETTINGS, Counter())
    tables[(TARGET, None, None)] = edit(tables[(TARGET, None, None)])
    with Stage(root / name) as stage:
        stage.write(tables)
    task = root / (name + ".task.json")
    write_json(task, dict(name=name, staging=name, schema=SHAPE["schema"]))
    return task


def test_path_bank_records_one_row_per_point_and_splits_by_task(path_bank):
    manifest = read_json(path_bank / "manifest.json")
    assert Schema(manifest["spec"]["schema"]).targets() == ("path",) and manifest["rows"] == 208
    supervision = read_json(path_bank / "supervision.json")
    rows = supervision["rows"]
    assert all(row["tau"] >= 1 for row in rows)
    opened = Deriver(path_bank, "cpu", RESIDENT)
    assert opened.aims.tolist() == [0] * 208 and len(opened.fields["tau"]) == 208
    assert opened.series_names == list(opened.registry.schema.series()) == ["level", "trend"]
    assert np.bincount(opened.series).tolist() == [104, 104]
    for code, entry in enumerate(supervision["split"]["series"]):
        train = [row["y"] for row in rows if row["series"] == entry["name"]]
        train = np.asarray(train)[
            np.isin(np.flatnonzero(opened.series == code), supervision["split"]["indices"]["train"])
        ]
        assert entry["mean"] == pytest.approx(train.mean()) and entry["sd"] == pytest.approx(
            train.std(ddof=1)
        )
    cfg = configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    shown = opened.viewed(cfg)
    for index in range(4):
        points, places = shown.labels[index]
        steps = shown.row_steps[index][places]
        keys = np.stack((steps, opened.series[points], opened.fields["tau"][points]), 1)
        assert all(tuple(a) < tuple(b) for a, b in zip(keys[:-1], keys[1:]))
        assert len(set(opened.partitions(cfg)[points].tolist())) == 1
    tasks = {(row["graph"], row["key"]) for row in rows}
    assert len(set(opened.task_of.tolist())) == len(tasks)


def test_path_supervision_rejects_duplicate_tau(tmp_path):
    def duplicated(table):
        tau = table.column("tau").to_numpy().copy()
        tau[1] = tau[0]
        return table.set_column(table.schema.get_field_index("tau"), "tau", pa.array(tau, pa.int64()))

    with pytest.raises(ValueError, match="graph_000 has two points with one task, one series and one tau"):
        build(staged_points(tmp_path, "twice", duplicated), tmp_path / "bank", 1, ("train",))


def test_path_supervision_rejects_nonpositive_tau(tmp_path):
    def flattened(table):
        tau = table.column("tau").to_numpy().copy()
        tau[2] = 0
        return table.set_column(table.schema.get_field_index("tau"), "tau", pa.array(tau, pa.int64()))

    with pytest.raises(ValueError, match="state their tau, a whole number of at least 1$"):
        build(staged_points(tmp_path, "zero", flattened), tmp_path / "bank", 1, ("train",))


def test_path_targets_require_a_series(tmp_path):
    with pytest.raises(ValueError, match="target lacks the columns series"):
        build(
            staged_points(tmp_path, "unnamed", lambda table: table.drop_columns(["series"])),
            tmp_path / "bank",
            1,
            ("train",),
        )


def test_targets_without_tau_stay_scalar(bank):
    assert Schema(read_json(bank / "manifest.json")["spec"]["schema"]).targets() == ("number",)
    assert Deriver(bank, "cpu", RESIDENT).fields == {}


def expected_points(opened, block):
    graph, _, stop, moment, partition, _, _ = block
    cfg = configuration(opened.registry, dict(view="snapshot"), {}, 5, SPLITS)
    shown = opened.viewed(cfg)
    rows, places = shown.labels[graph]
    steps = shown.row_steps[graph][places]
    chosen = rows[(steps >= moment) & (steps < stop)]
    return sorted(chosen[opened.partitions(cfg)[chosen] == partition].tolist())


def test_path_windows_carry_every_point_of_their_step(path_bank):
    from tuning.batches.batching import recipe_batch
    from tuning.batches.inputs import recipe

    opened = Deriver(path_bank, "cpu", RESIDENT)
    cfg = configuration(opened.registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS)
    for partition, indices in ((0, "train"), (1, "validation")):
        blocks = opened.blocks(partition, 4, cfg)
        covered = []
        for block in blocks:
            b = recipe(opened, block, cfg)
            rows = b["rows"].value
            assert sorted(rows.tolist()) == expected_points(opened, block) and len(rows) > 0
            tau = b["field/tau"]
            assert np.array_equal(tau, opened.fields["tau"][rows]) and tau.dtype == np.float32
            assert b["tasks"] == int(b["selected"].sum()) == 1
            assert np.all(b["point_row"] == 0) and len(b["point_row"]) == len(rows)
            covered.extend(rows.tolist())
        assert sorted(covered) == opened.split["indices"][indices]
    full = [block for block in opened.blocks(0, 4, cfg) if block[2] - block[1] == 4]
    b = recipe_batch(opened, full[:3], cfg)
    assert b["tasks"] == 3 and b["point_row"].tolist() == sorted(b["point_row"].tolist())
    assert set(b["point_row"].tolist()) == {0, 1, 2} and len(b["field/tau"]) == len(b["rows"].value)


def test_path_windows_join_like_the_windows_of_numbers(path_bank):
    from tuning.bank.variants import prepare
    from tuning.batches.inputs import recipe

    opened = Deriver(path_bank, "cpu", RESIDENT)
    single = configuration(
        opened.registry,
        dict(view="snapshot"),
        {"time/history": 4, "graphs/join": "single", "graphs/join_size": 2},
        5,
        SPLITS,
    )
    prepare(path_bank, single, 1, "cpu")
    opened = Deriver(path_bank, "cpu", RESIDENT)
    assert opened.groups(single) == [[0, 2], [1, 3]]
    for partition in (0, 1):
        blocks = opened.blocks(partition, 4, single)
        assert len(blocks) == len(
            opened.blocks(
                partition,
                4,
                configuration(opened.registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS),
            )
        )
        covered = []
        for block in blocks:
            rows = recipe(opened, block, single)["rows"].value
            covered.extend(rows.tolist())
            group, _, stop, moment, _, _, _ = block
            members = opened.groups(single)[group]
            offsets = np.r_[0, np.cumsum([opened.viewed(single).graphs[m]["steps"] for m in members])]
            member = int(np.searchsorted(offsets, moment, "right") - 1)
            local = (members[member], 0, stop - offsets[member], moment - offsets[member], partition, 0, 0)
            assert sorted(rows.tolist()) == expected_points(opened, local)
        assert sorted(covered) == opened.split["indices"][("train", "validation")[partition]]


def balanced(predictions, targets, weights, point_row, series, tasks, count):
    from types import SimpleNamespace

    from tuning.fits.losses import LOSSES

    held = SimpleNamespace(center=np.zeros(count, np.float32), scale=np.ones(count, np.float32))
    batch = dict(target=targets, weight=weights, point_row=point_row, series=series, tasks=tasks)
    return LOSSES.named("balanced")(predictions, batch, held)


def test_the_balanced_loss_averages_within_task_and_series_then_tasks_then_series():
    import torch

    predictions = torch.tensor([1.0, 3.0, 1.0, 2.0, 4.0, 2.0])
    targets = torch.zeros(6)
    point_row = torch.tensor([0, 1, 1, 2, 0, 0])
    series = torch.tensor([0, 0, 0, 0, 1, 1])
    first, second = (1 + 5 + 4) / 3, (16 + 4) / 2
    value = balanced(predictions, targets, torch.ones(6), point_row, series, 3, 2)
    assert float(value) == pytest.approx((first + second) / 2)
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    weighted = balanced(predictions, targets, weights, point_row, series, 3, 2)
    assert float(weighted) == pytest.approx(((2 * 1 + 5 + 4) / 4 + (16 + 4) / 2) / 2)


def test_the_balanced_measure_is_the_balanced_loss_on_the_points_scored():
    from tuning.measures.measures import measured, scored

    targets = np.array([2.0, 1.0, -1.0, 4.0, 2.0])
    predictions = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    owners = np.array([0, 1, 1, 0, 1])
    series = np.array([0, 0, 0, 1, 1])
    metrics = measured(
        ["balanced", "mse", "r2"],
        scored(targets, predictions, owners, series, ["a", "b"], np.array([1.0, 2.0]), {}),
        dict(a="number", b="path"),
    )
    mse = (1.0 * 1 + 0.5 * 0 + 0.5 * 4) / 2
    assert metrics["balanced"]["series"] == pytest.approx(dict(a=mse, b=2.5))
    assert metrics["balanced"]["value"] == pytest.approx((mse + 2.5) / 2)
    assert metrics["mse"]["series"] == pytest.approx(dict(a=5 / 3, b=10.0))
    assert metrics["mse"]["value"] == pytest.approx(5.0) and set(metrics["mse"]) == {"value", "series"}
    assert metrics["r2"]["value"] == pytest.approx(1 - (5 / 3) / np.var(targets[:3]))
    assert list(metrics["r2"]["series"]) == ["a"]
    unscored = scored(targets, predictions, owners, series, ["a", "b", "c"], np.ones(3), {})
    with pytest.raises(ValueError, match="the points scored hold no point of the series c, which the measure mse reads"):
        measured(["mse"], unscored, dict(a="number", b="path", c="number"))


def test_validation_loss_matches_the_training_loss_on_the_same_points():
    import torch

    from tuning.measures.measures import measured, scored

    rng = np.random.default_rng(4)
    owners = np.repeat(np.arange(6), 4)
    series = np.tile(np.array([0, 0, 1, 1]), 6)
    targets, predictions = rng.normal(size=24), rng.normal(size=24)
    center, scale = np.array([0.3, -0.2]), np.array([1.5, 0.4])
    found = measured(
        ["balanced"], scored(targets, predictions, owners, series, ["a", "b"], scale, {}), dict(a="path", b="path")
    )

    def standard(values):
        return torch.tensor((values - center[series]) / scale[series])

    trained = balanced(
        standard(predictions),
        standard(targets),
        torch.ones(24, dtype=torch.float64),
        torch.tensor(owners),
        torch.tensor(series),
        6,
        2,
    )
    assert float(trained) == pytest.approx(found["balanced"]["value"])


def test_objective_ranks_by_the_first_measure_and_blends_its_series_by_their_weights(path_bank):
    from tuning.measures import objective

    names = Deriver(path_bank, "cpu", RESIDENT).series_names
    assert objective.weights(names, None) is None
    assert objective.weights(names, ["trend=2"]) == {"level": 0.0, "trend": 2.0}
    chosen = objective.weights(names, ["trend=3", "level=1"])
    result = dict(measures=dict(balanced=dict(value=4.0, series=dict(level=2.0, trend=6.0)), mse=dict(value=9.0)))
    plan = dict(objective=chosen, measures=["balanced", "mse"])
    assert objective.value(result, plan) == pytest.approx(5.0)
    assert objective.value(result, dict(plan, objective=None)) == 4.0
    assert [entry["key"] for entry in objective.measures(plan)] == [
        "mse",
        "measure/mse",
        "series/level/loss",
        "series/trend/loss",
    ]
    assert objective.measures(plan)[0]["label"] == "Validation loss"
    assert objective.columns(plan) == ["series/level/loss", "series/trend/loss"]
    assert objective.scores(result, plan) == {"measure/mse": 9.0, "series/level/loss": 2.0, "series/trend/loss": 6.0}
    assert objective.scores(dict(error="x"), plan) == {} and objective.columns(dict(objective=None)) == []
    with pytest.raises(ValueError, match="ranks its trials by the first measure .* r2 states that higher is better"):
        objective.measures(dict(plan, measures=["r2"]))
    earlier = dict(series=dict(level=dict(val_loss=2.0), trend=dict(val_loss=6.0)))
    assert objective.value(earlier, dict(objective=chosen)) == pytest.approx(5.0)
    assert objective.value(dict(raw_val_mse=0.5), dict(objective=None)) == 0.5
    assert objective.scores(earlier, dict(objective=chosen)) == {"series/level/loss": 2.0, "series/trend/loss": 6.0}
    for given in (["other=1"], ["level=1", "level=2"], ["level=0"], ["level=-1", "trend=2"]):
        with pytest.raises(ValueError, match="objective"):
            objective.weights(names, given)


@pytest.mark.parametrize(
    "model",
    [
        {"view": name}
        for name in (
            "snapshot",
            "mixed",
            "event_graph",
            "memory",
            "projection",
            "sequence",
            "tree",
            "walks",
            "affinity",
            "signal",
        )
    ],
)
def test_path_heads_emit_one_value_per_point(path_bank, model):
    import torch

    from tuning.batches.batching import pack_batch
    from tuning.families.models import build
    from tuning.families.periods import blocks as period_blocks
    from tuning.fits.losses import Summed

    opened = Deriver(path_bank, "cpu", RESIDENT)
    loss = Summed(dict(path="balanced"), opened.registry.schema.targets())
    cfg = configuration(opened.registry, dict(model, hidden=8), {"time/history": 2}, 11, SPLITS)
    assert cfg["model"]["time/encoding"] == "log_cosine"
    full = [block for block in opened.blocks(0, 2, cfg) if block[2] - block[1] == 2]
    batch = pack_batch(opened, period_blocks(opened, full[:2], cfg), cfg, "cpu")
    net = build(cfg, opened.fitted(cfg))
    prediction = net(batch)
    assert prediction.shape == batch["series"].shape == batch["field/tau"].shape and "target" not in batch
    assert len(prediction) > batch["tasks"] == 2
    assert torch.isfinite(prediction).all()
    value = loss(prediction, batch, opened)
    value.backward()
    assert torch.isfinite(value) and net.head.kinds[0].time.frequency.grad is not None
    assert net.head.series.weight.grad is not None and set(batch["series"].tolist()) == {0, 1}
