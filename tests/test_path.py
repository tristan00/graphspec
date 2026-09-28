import numpy as np
import pyarrow as pa
import pytest

from graphspec.infra.files import read_json, write_json
from graphspec.infra.staging import Stage
from graphspec.synthetic import TARGET, Counter, generate, graph_rows
from tuning.build import build
from tuning.data import Bank

SETTINGS = dict(
    entities=40, steps=12, features=8, signals=2, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3
)


@pytest.fixture(scope="session")
def path_bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("path")
    task = generate(root, "tiny", 4, "path", 1, **SETTINGS)
    build(task, root / "bank", 1)
    return root / "bank"


def staged_points(root, name, edit):
    tables = graph_rows(np.random.default_rng(3), 0, "graph_000", SETTINGS, Counter(), "path")
    tables[(TARGET, None, None)] = edit(tables[(TARGET, None, None)])
    with Stage(root / name) as stage:
        stage.write(tables)
    task = root / (name + ".task.json")
    write_json(task, dict(name=name, staging=name, target=TARGET))
    return task


def test_path_bank_records_one_row_per_point_and_splits_by_query(path_bank):
    manifest = read_json(path_bank / "manifest.json")
    assert manifest["target"]["level"] == "path" and manifest["rows"] == 208
    supervision = read_json(path_bank / "supervision.json")
    rows = supervision["rows"]
    assert all(row["tau"] >= 1 for row in rows)
    opened = Bank(path_bank)
    assert opened.level == "path" and len(opened.taus) == 208
    assert opened.series_names == opened.registry.extras["series"] == ["level", "trend"]
    assert np.bincount(opened.series).tolist() == [104, 104]
    for code, entry in enumerate(supervision["split"]["series"]):
        train = [row["y"] for row in rows if row["series"] == entry["name"]]
        train = np.asarray(train)[
            np.isin(np.flatnonzero(opened.series == code), supervision["split"]["train_indices"])
        ]
        assert entry["mean"] == pytest.approx(train.mean()) and entry["sd"] == pytest.approx(
            train.std(ddof=1)
        )
    for index in range(4):
        steps, points = opened.row_steps[index], opened.row_index[index]
        keys = np.stack((steps, opened.series[points], opened.taus[points]), 1)
        assert all(tuple(a) < tuple(b) for a, b in zip(keys[:-1], keys[1:]))
        assert len(set(opened.partitions[points].tolist())) == 1
    queries = {(row["graph"], row["step"]) for row in rows}
    assert len(set(opened.query_of.tolist())) == len(queries)


def test_path_supervision_rejects_duplicate_tau(tmp_path):
    def duplicated(table):
        tau = table.column("tau").to_numpy().copy()
        tau[1] = tau[0]
        return table.set_column(table.schema.get_field_index("tau"), "tau", pa.array(tau, pa.float32()))

    with pytest.raises(ValueError, match="graph_000 has two points with one query, one series and one tau"):
        build(staged_points(tmp_path, "twice", duplicated), tmp_path / "bank", 1)


def test_path_supervision_rejects_nonpositive_tau(tmp_path):
    def flattened(table):
        tau = table.column("tau").to_numpy().copy()
        tau[2] = 0.0
        return table.set_column(table.schema.get_field_index("tau"), "tau", pa.array(tau, pa.float32()))

    with pytest.raises(ValueError, match="strictly positive"):
        build(staged_points(tmp_path, "zero", flattened), tmp_path / "bank", 1)


def test_path_targets_require_a_series(tmp_path):
    with pytest.raises(ValueError, match="requires a series STRING column"):
        build(
            staged_points(tmp_path, "unnamed", lambda table: table.drop_columns(["series"])),
            tmp_path / "bank",
            1,
        )


def test_targets_without_tau_stay_scalar(bank):
    assert read_json(bank / "manifest.json")["target"]["level"] == "query"


def expected_points(opened, block):
    graph, _, stop, moment, partition, _, _ = block
    rows, steps = opened.row_index[graph], opened.row_steps[graph]
    chosen = rows[(steps >= moment) & (steps < stop)]
    return sorted(chosen[opened.partitions[chosen] == partition].tolist())


def test_path_windows_carry_every_point_of_their_step(path_bank):
    from tuning.batching import recipe_batch
    from tuning.inputs import recipe
    from tuning.model_spec import configuration

    opened = Bank(path_bank)
    cfg = configuration(opened.registry, dict(view="snapshot"), {"time/history": 4}, 5)
    for partition, indices in ((0, "train_indices"), (1, "validation_indices")):
        blocks = opened.blocks(partition, 4, cfg)
        covered = []
        for block in blocks:
            b = recipe(opened, block, cfg)
            rows = b["rows"].value
            assert sorted(rows.tolist()) == expected_points(opened, block) and len(rows) > 0
            assert np.array_equal(b["tau"], opened.taus[rows]) and b["tau"].dtype == np.float32
            assert b["queries"] == int(b["selected"].sum()) == 1
            assert np.all(b["point_query"] == 0) and len(b["point_query"]) == len(rows)
            covered.extend(rows.tolist())
        assert sorted(covered) == opened.split[indices]
    full = [block for block in opened.blocks(0, 4, cfg) if block[2] - block[1] == 4]
    b = recipe_batch(opened, full[:3], cfg)
    assert b["queries"] == 3 and b["point_query"].tolist() == sorted(b["point_query"].tolist())
    assert set(b["point_query"].tolist()) == {0, 1, 2} and len(b["tau"]) == len(b["rows"].value)


def test_path_windows_join_like_query_windows(path_bank):
    from tuning.inputs import recipe
    from tuning.model_spec import configuration
    from tuning.variants import prepare

    opened = Bank(path_bank)
    single = configuration(
        opened.registry,
        dict(view="snapshot"),
        {"time/history": 4, "graphs/join": "single", "graphs/join_size": 2},
        5,
    )
    prepare(path_bank, single, 1)
    opened = Bank(path_bank)
    assert opened.groups(single) == [[0, 2], [1, 3]]
    for partition in (0, 1):
        blocks = opened.blocks(partition, 4, single)
        assert len(blocks) == len(
            opened.blocks(
                partition, 4, configuration(opened.registry, dict(view="snapshot"), {"time/history": 4}, 5)
            )
        )
        covered = []
        for block in blocks:
            rows = recipe(opened, block, single)["rows"].value
            covered.extend(rows.tolist())
            group, _, stop, moment, _, _, _ = block
            members = opened.groups(single)[group]
            offsets = np.r_[0, np.cumsum([opened.manifest["graphs"][m]["steps"] for m in members])]
            member = int(np.searchsorted(offsets, moment, "right") - 1)
            local = (members[member], 0, stop - offsets[member], moment - offsets[member], partition, 0, 0)
            assert sorted(rows.tolist()) == expected_points(opened, local)
        assert sorted(covered) == opened.split[("train_indices", "validation_indices")[partition]]


def test_path_loss_averages_within_query_and_series_then_queries_then_series():
    import torch

    from tuning.training import path_loss

    predictions = torch.tensor([1.0, 3.0, 1.0, 2.0, 4.0, 2.0])
    targets = torch.zeros(6)
    point_query = torch.tensor([0, 1, 1, 2, 0, 0])
    series = torch.tensor([0, 0, 0, 0, 1, 1])
    first, second = (1 + 5 + 4) / 3, (16 + 4) / 2
    value = path_loss(predictions, targets, point_query, series, 3, 2)
    assert float(value) == pytest.approx((first + second) / 2)


def test_path_score_weights_queries_equally_per_series_and_measures_persistence_skill():
    from tuning.training import path_score, score

    targets = np.array([2.0, 1.0, -1.0, 4.0, 2.0])
    predictions = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    queries = np.array([0, 1, 1, 0, 1])
    series = np.array([0, 0, 0, 1, 1])
    metrics = path_score(targets, predictions, queries, series, ["a", "b"], np.array([1.0, 2.0]))
    mse = (1.0 * 1 + 0.5 * 0 + 0.5 * 4) / 2
    persistence = (1.0 * 4 + 0.5 * 1 + 0.5 * 1) / 2
    first, second = metrics["series"]["a"], metrics["series"]["b"]
    assert first["raw_val_mse"] == pytest.approx(mse) and first["val_loss"] == pytest.approx(mse)
    assert first["val_r2"] == pytest.approx(1 - mse / persistence)
    assert second["raw_val_mse"] == pytest.approx(10) and second["val_loss"] == pytest.approx(2.5)
    assert second["val_r2"] == pytest.approx(0) and second["queries"] == 2 and second["points"] == 2
    assert metrics["val_loss"] == pytest.approx((mse + 2.5) / 2)
    assert score(targets[:3], predictions[:3])["raw_val_mse"] == pytest.approx(5 / 3)
    with pytest.raises(ValueError, match="persistence"):
        path_score(np.zeros(5), predictions, queries, series, ["a", "b"], np.ones(2))


def test_objective_blends_series_losses_by_their_weights(path_bank):
    from tuning import objective

    opened = Bank(path_bank)
    assert objective.weights(opened, None) == {"level": 1.0, "trend": 1.0}
    assert objective.weights(opened, ["trend=2"]) == {"level": 0.0, "trend": 2.0}
    chosen = objective.weights(opened, ["trend=3", "level=1"])
    result = dict(series=dict(level=dict(val_loss=2.0), trend=dict(val_loss=6.0)))
    assert objective.value(result, dict(objective=chosen)) == pytest.approx(5.0)
    assert objective.value(dict(raw_val_mse=0.5), dict(objective=None)) == 0.5
    plan = dict(objective=chosen)
    assert [entry["key"] for entry in objective.measures(plan)][2:4] == [
        "series/level/skill",
        "series/level/loss",
    ]
    assert objective.columns(plan) == ["series/level/skill", "series/trend/skill"]
    full = dict(series=dict(level=dict(val_loss=2.0, val_r2=0.5), trend=dict(val_loss=6.0, val_r2=-0.1)))
    assert objective.scores(full, plan)["series/trend/skill"] == -0.1
    assert objective.scores(dict(error="x"), plan) == {} and objective.columns(dict(objective=None)) == []
    for given in (["other=1"], ["level=1", "level=2"], ["level=0"], ["level=-1", "trend=2"]):
        with pytest.raises(ValueError, match="objective"):
            objective.weights(opened, given)


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

    from tuning.batching import pack_batch
    from tuning.model_spec import configuration
    from tuning.models import build
    from tuning.training import loss
    from tuning.turns import blocks as turn_blocks

    opened = Bank(path_bank)
    cfg = configuration(opened.registry, dict(model, hidden=8), {"time/history": 2}, 11)
    assert cfg["model"]["time/encoding"] == "log_cosine"
    full = [block for block in opened.blocks(0, 2, cfg) if block[2] - block[1] == 2]
    batch = pack_batch(opened, turn_blocks(opened, full[:2], cfg), cfg, "cpu")
    net = build(cfg, opened.registry, len(opened.categories))
    prediction = net(batch)
    assert prediction.shape == batch["target"].shape == batch["tau"].shape
    assert len(prediction) > batch["queries"] == 2
    assert torch.isfinite(prediction).all()
    value = loss(prediction, batch, opened)
    value.backward()
    assert torch.isfinite(value) and net.reward.time.frequency.grad is not None
    assert net.reward.series.weight.grad is not None and set(batch["series"].tolist()) == {0, 1}
