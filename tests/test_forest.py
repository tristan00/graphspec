import numpy as np
import pytest

from graphspec.infra.files import read_json
from tuning import selector, space, surrogates
from tuning.data import Bank
from tuning.statement import state
from tuning.surrogates import COMPLETION_WEIGHT, Forest, log_expected_improvement


def test_log_expected_improvement_matches_the_closed_form_and_stays_finite_far_away():
    mean = np.array([0.5, 1.0, 1.5, 9.0, 1.0])
    deviation = np.array([0.2, 0.2, 0.2, 0.2, 0.0])
    values = log_expected_improvement(1.0, mean, deviation)
    assert np.isfinite(values).all()
    assert values[0] > values[1] > values[2] > values[3]
    assert values[0] == pytest.approx(np.log(0.5 * 0.99379 + 0.2 * 0.0175283), abs=1e-3)
    assert values[4] < -10


def test_forest_prefers_the_low_loss_region_and_reports_disagreement():
    rng = np.random.default_rng(0)
    x = rng.random((80, 6))
    losses = (x[:, 0] - 0.2) ** 2 + 0.05 * rng.random(80)
    model = Forest(x, losses, [True] * 80, 1)
    candidates = rng.random((300, 6))
    index, value = model.choose(candidates)
    assert abs(candidates[index, 0] - 0.2) < 0.15
    assert np.isfinite(value)
    assert model.scores["acquisition"].argmax() == index
    assert model.diagnostics["completion_probability"] is None
    assert model.diagnostics["uncertainty"] >= 0
    assert model.best == pytest.approx(losses.min())
    assert "completion_probability" not in model.scores
    assert (model.scores["predicted_mse"] > 0).all()
    assert ((model.scores["beat_probability"] >= 0) & (model.scores["beat_probability"] <= 1)).all()
    assert model.scores["beat_probability"][index] > np.median(model.scores["beat_probability"])
    scaled = Forest(x, losses * 1000, [True] * 80, 1)
    scaled.choose(candidates)
    assert np.corrcoef(scaled.scores["log_improvement"], model.scores["log_improvement"])[0, 1] > 0.95
    assert np.allclose(scaled.scores["predicted_mse"], model.scores["predicted_mse"] * 1000, rtol=0.25)
    with pytest.raises(ValueError, match="positive"):
        Forest(x, losses - losses.min(), [True] * 80, 1)


def test_failed_trials_train_the_completion_forest():
    x = np.linspace(0, 1, 40)[:, None]
    completed = x[:, 0] < 0.6
    losses = np.where(completed, (x[:, 0] - 0.3) ** 2, np.nan)
    model = Forest(x, losses, completed, 3)
    index, value = model.choose(x)
    assert x[index, 0] < 0.6
    assert 0 <= model.diagnostics["completion_probability"] <= 1
    assert np.allclose(
        model.scores["acquisition"],
        model.scores["log_improvement"]
        + COMPLETION_WEIGHT * np.log(np.maximum(model.scores["completion_probability"], 1e-9)),
    )
    assert COMPLETION_WEIGHT > 1
    assert model.scores["completion_probability"][x[:, 0] > 0.9].mean() < 0.3
    with pytest.raises(ValueError, match="two completed"):
        Forest(x, losses, [False] * 40, 3)


@pytest.mark.xdist_group("candidates")
def test_forest_suggest_records_ranked_candidates_and_fixed_parameters(bank, candidate_pool, monkeypatch):
    registry = Bank(bank).registry
    observations, failures = [], []
    for index in range(6):
        config, _ = selector.random_suggest(registry, index, index + 1, {"epochs": 2}, 600)
        entry = dict(config=config, status="complete", trial=str(index), objective=float(index + 1))
        observations.append(entry)
        failed = dict(entry, status="failed", trial="failed-" + str(index), objective=None)
        failures.append(failed)
    monkeypatch.setattr(space, "CANDIDATES", 20)
    bounds = {"epochs": {"low": 1, "high": 3}}
    encoding = space.Space(registry, bounds, 600, ())
    progress = []
    config, selection = surrogates.suggest_forest(
        registry, observations, failures, 4, 9, bounds, ["model/view"], "cpu", 600, progress=progress.append
    )
    assert selection["sampler"] == "SMAC"
    assert selection["phase"] == "forest"
    assert selection["settings"]["model"] == "random forest"
    assert selection["settings"]["transform"] == "log"
    assert selection["settings"]["completion_weight"] == COMPLETION_WEIGHT
    assert selection["settings"]["candidates"] == 20
    assert config["model"]["view"] == selection["fixed_params"]["model/view"]
    assert config["epochs"] <= 3
    assert selection["observations"] == selection["failed_observations"] == 6
    assert selection["diagnostics"]["best_observed_mse"] == 1.0
    ranked = selection["diagnostics"]["candidates"]
    assert len(ranked) == 20
    assert [c["rank"]["acquisition"] for c in ranked] == list(range(1, 21))
    assert sorted(c["rank"]["predicted_mse"] for c in ranked) == list(range(1, 21))
    assert min(ranked, key=lambda c: c["predicted_mse"])["rank"]["predicted_mse"] == 1
    assert max(ranked, key=lambda c: c["uncertainty"])["rank"]["uncertainty"] == 1
    assert ranked[0]["acquisition"] >= ranked[-1]["acquisition"]
    assert ranked[0]["config"] == config
    assert 0 <= ranked[0]["completion_probability"] <= 1
    assert sum(selection["diagnostics"]["distribution"]["acquisition"]["counts"]) == 20
    importance = selection["diagnostics"]["importance"]
    assert 0 < len(importance["score"]) <= 20 and 0 < len(importance["completion"]) <= 20
    assert importance["score"][0]["value"] >= importance["score"][-1]["value"] > 0
    assert all(entry["name"] in encoding.parameters for entry in importance["score"])
    assert [entry["stage"] for entry in progress] == [
        "Fitting models",
        "Ranking candidates",
        "Proposal ready",
    ]
    repeated, _ = surrogates.suggest_forest(registry, observations, failures, 4, 9, bounds, ["model/view"], "cpu", 600)
    assert repeated == config


def test_startup_counts_toward_the_forest_selector(bank, tmp_path, monkeypatch, statement):
    registry = Bank(bank).registry
    config, metadata = selector.random_suggest(registry, 1, 2, {}, 600)
    observations = [dict(config=config, status="complete", trial="good-" + str(i)) for i in range(40)]
    failures = [dict(config=config, status="failed", trial="bad-" + str(i)) for i in range(60)]
    monkeypatch.setattr(selector, "load_history", lambda *args: (observations, failures, [], []))
    monkeypatch.setattr(surrogates, "suggest_forest", lambda *args, **kwargs: (config, dict(metadata, phase="forest")))
    plan = dict(
        selector="forest",
        random_trials=100,
        mix={"forest": 100},
        seconds=1200,
        random_params=[],
        seed=5,
        trials=[],
        bounds={},
        bank_manifest_sha256="bank",
        supervision_sha256="labels",
    )
    selector.append_trial(tmp_path, plan, registry, state(dict(statement, seconds=1200)))
    assert plan["trials"][-1]["selector"] == "forest"
    assert read_json(tmp_path / plan["trials"][-1]["directory"] / "selection.json")["phase"] == "forest"
