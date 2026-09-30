import copy

import numpy as np
import pytest
import torch

from graphspec.infra.files import read_json
from tuning.bank.bank import Bank
from tuning.searches import selector, space, surrogates
from tuning.searches.surrogates import Search
from tuning.statement import state

pytestmark = pytest.mark.xdist_group("candidates")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_remaining_parameter_changes_with_the_fixed_parameter():
    x, losses = [], []
    for view in (0, 1):
        for value in np.linspace(0, 1, 11):
            x.append([view, 1 - view, value])
            losses.append((value - (0.2 if view == 0 else 0.8)) ** 2)
    search = Search(x, losses, [True] * len(x), "cpu")
    selected = []
    for view in (0, 1):
        candidates = np.array([[view, 1 - view, value] for value in np.linspace(0, 1, 101)])
        index, _ = search.choose(candidates)
        assert np.isfinite(search.diagnostics["predicted_mse"])
        assert search.diagnostics["uncertainty"] >= 0
        assert search.diagnostics["completion_probability"] is None
        selected.append(candidates[index, -1])
    assert selected[0] < 0.4
    assert selected[1] > 0.6


def test_failed_trials_train_feasibility_without_fake_objective_values():
    x = np.linspace(0, 1, 16)[:, None]
    completed = x[:, 0] < 0.6
    losses = np.where(completed, (x[:, 0] - 0.3) ** 2, np.nan)
    search = Search(x, losses, completed, "cpu")
    objective, feasibility = search.acquisition.model.models
    assert len(objective.train_inputs[0]) == sum(completed)
    assert len(feasibility.train_inputs[0]) == len(x)
    assert torch.isfinite(objective.train_targets).all()
    assert torch.isfinite(feasibility.train_targets).all()
    index, value = search.choose(x)
    assert x[index, 0] < 0.6
    assert np.isfinite(value)
    assert 0 <= search.diagnostics["completion_probability"] <= 1
    assert search.scores["acquisition"].argmax() == index
    assert search.best == pytest.approx(losses[completed].min())
    assert ((search.scores["beat_probability"] >= 0) & (search.scores["beat_probability"] <= 1)).all()
    assert np.allclose(
        search.scores["acquisition"],
        search.scores["log_improvement"] + np.log(search.scores["completion_probability"]),
        atol=1e-6,
    )


def test_history_and_candidates_preserve_parameters_and_condition_on_random_values(
    bank, candidate_pool, monkeypatch
):
    registry = Bank(bank).registry
    observations, failures = [], []
    for index, view in enumerate(("snapshot", "mixed", "sequence", "memory")):
        config, _ = selector.random_suggest(registry, index, index + 1, {"view": view, "epochs": 4}, 600)
        config["max_seconds"] = 600 + 400 * index
        entry = dict(config=config, status="complete", trial=str(index), objective=float(index + 1))
        observations.append(entry)
        failed = copy.deepcopy(entry)
        failed.update(status="timeout", trial="failed-" + str(index), objective=None)
        failures.append(failed)
    original = copy.deepcopy(observations + failures)
    seen = []

    class InspectSearch:
        def __init__(self, x, losses, completed, device):
            assert device == "cpu"
            assert len(x) == 8
            assert completed == [True] * 4 + [False] * 4
            assert np.allclose(x[:4, -2], [0, 1 / 3, 2 / 3, 1])
            assert np.array_equal(x[:4, -2:], x[4:, -2:])

        def choose(self, candidates):
            seen.append(candidates)
            assert np.all(candidates[:, :10] == candidates[0, :10])
            for name in ("epochs", "max_seconds"):
                offset = sum(
                    len(p.options) if p.options else 2
                    for key, p in encoding.parameters.items()
                    if key in preceding[name]
                )
                assert np.all(candidates[:, offset : offset + 2] == candidates[0, offset : offset + 2])
            self.diagnostics = dict(predicted_mse=0.2, uncertainty=0.1, completion_probability=0.8)
            self.scores = {
                key: np.arange(len(candidates), dtype=float) for key in ("acquisition", "predicted_mse")
            }
            self.best = 1.0
            return 0, 0.0

    monkeypatch.setattr(surrogates, "Search", InspectSearch)
    monkeypatch.setattr(space, "CANDIDATES", 12)
    bounds = {"epochs": {"low": 1, "high": 3}, "hidden": 37}
    encoding = space.Space(registry, bounds, 600, [entry["config"] for entry in original])
    names = list(encoding.parameters)
    preceding = {name: names[: names.index(name)] for name in ("epochs", "max_seconds")}
    random_params = ["model/view", "lr", "epochs", "max_seconds"]
    progress = []
    config, selection = surrogates.suggest_botorch(
        registry, observations, failures, 42, 7, bounds, random_params, "cpu", 600, progress=progress.append
    )
    assert selection["observations"] == selection["failed_observations"] == 4
    assert selection["excluded"] == []
    assert selection["training_trials"] == [entry["trial"] for entry in original]
    assert observations + failures == original
    assert config["model"]["view"] == selection["fixed_params"]["model/view"]
    assert config["lr"] == selection["fixed_params"]["lr"]
    assert config["epochs"] == selection["fixed_params"]["epochs"]
    assert config["max_seconds"] == selection["fixed_params"]["max_seconds"]
    assert config["max_seconds"] == 600
    assert config["model"]["hidden"] == 37
    assert config["epochs"] <= 3
    assert len(seen) == 1
    assert len(seen[0]) == 12
    assert selection["diagnostics"]["best_observed_mse"] == 1.0
    assert [c["acquisition"] for c in selection["diagnostics"]["candidates"]] == list(range(11, -1, -1))
    assert selection["diagnostics"]["candidates"][0]["rank"] == {"acquisition": 1, "predicted_mse": 12}
    assert config in [c["config"] for c in selection["diagnostics"]["candidates"]]
    assert sum(selection["diagnostics"]["distribution"]["acquisition"]["counts"]) == 12
    assert selection["settings"]["device"] == "cpu"
    assert [entry["stage"] for entry in progress] == [
        "Fitting models",
        "Ranking candidates",
        "Proposal ready",
    ]
    assert progress[-1]["stage"] == "Proposal ready"
    repeated, repeated_selection = surrogates.suggest_botorch(
        registry, observations, failures, 42, 7, bounds, random_params, "cpu", 600
    )
    assert repeated == config
    assert repeated_selection["fixed_params"] == selection["fixed_params"]


def test_random_view_covers_snapshot_and_other_choices(bank):
    registry = Bank(bank).registry
    encoding = space.Space(registry, {}, 600, ())
    parameter = encoding.proposal_parameters["model/view"]
    values = {parameter.uniform(np.random.default_rng(seed)) for seed in range(200)}
    assert values == set(parameter.options)
    assert "snapshot" in values
    with pytest.raises(ValueError, match="unknown parameters"):
        encoding.random_names(["model/typo"])
    assert encoding.random_names(["model/view"]) == ["model/view"]


@pytest.mark.parametrize("finished,expected", [(99, "random"), (100, "botorch")])
def test_startup_counts_tpe_and_failed_trials(bank, tmp_path, monkeypatch, statement, finished, expected):
    registry = Bank(bank).registry
    config, metadata = selector.random_suggest(registry, 1, 2, {}, 600)
    observations = [dict(config=config, status="complete", trial="good-" + str(i)) for i in range(40)]
    failures = [dict(config=config, status="failed", trial="bad-" + str(i)) for i in range(finished - 40)]
    monkeypatch.setattr(selector, "load_history", lambda *args: (observations, failures, [], []))
    calls = []

    def propose(*args, progress=None):
        calls.append(args[-3:])
        config["max_seconds"] = 843
        return config, dict(metadata, phase="botorch")

    monkeypatch.setattr(surrogates, "suggest_botorch", propose)
    plan = dict(
        selector="botorch",
        random_trials=100,
        mix={"botorch": 100},
        seconds=1200,
        random_params=["model/view"],
        seed=5,
        trials=[],
        bounds={},
        bank_manifest_sha256="bank",
        supervision_sha256="labels",
    )
    selector.append_trial(tmp_path, plan, registry, state(dict(statement, seconds=1200)))
    item = plan["trials"][-1]
    assert item["selector"] == expected
    selection = read_json(tmp_path / item["directory"] / "selection.json")
    assert selection["startup_trials_finished"] == finished
    assert calls == ([(["model/view"], "cpu", 1200)] if expected == "botorch" else [])
    saved = read_json(tmp_path / item["directory"] / "config.json")
    assert saved["max_seconds"] == (843 if expected == "botorch" else 1200)


def test_acquisition_conditions_remaining_choices_on_epochs_and_runtime():
    inputs, losses = [], []
    for epochs in (0.0, 1.0):
        for runtime in (0.0, 1.0):
            for value in np.linspace(0, 1, 11):
                inputs.append([epochs, runtime, value])
                losses.append((value - (0.1 + 0.35 * epochs + 0.35 * runtime)) ** 2)
    search = Search(inputs, losses, [True] * len(inputs), "cpu")
    for epochs in (0.0, 1.0):
        for runtime in (0.0, 1.0):
            candidates = np.array([[epochs, runtime, value] for value in np.linspace(0, 1, 101)])
            index, _ = search.choose(candidates)
            assert abs(candidates[index, -1] - (0.1 + 0.35 * epochs + 0.35 * runtime)) < 0.15


def test_unknown_historical_runtime_stays_unknown(bank):
    registry = Bank(bank).registry
    config, _ = selector.random_suggest(registry, 1, 2, {}, 600)
    unknown = copy.deepcopy(config)
    del unknown["max_seconds"]
    encoding = space.Space(registry, {}, 600, [dict(config, max_seconds=1200)])
    encoded = encoding.encode([config, unknown])
    assert encoded[0, -1] == 1
    assert encoded[1, -1] == 0


def test_inactive_parameters_are_distinct_from_active_minimum_values(bank):
    registry = Bank(bank).registry
    config, _ = selector.random_suggest(registry, 1, 2, {"view": "mixed"}, 600)
    other = copy.deepcopy(config)
    other["model"]["snapshot/layers"] = 1
    encoding = space.Space(registry, {}, 600, ())
    encoded = encoding.encode([config, other])
    assert not np.array_equal(encoded[0], encoded[1])


def test_selector_mix_rolls_each_trial_after_startup(bank, tmp_path, monkeypatch, statement):
    registry = Bank(bank).registry
    config, metadata = selector.random_suggest(registry, 1, 2, {}, 600)
    observations = [dict(config=config, status="complete", trial="good-" + str(i)) for i in range(20)]
    monkeypatch.setattr(selector, "load_history", lambda *args: (observations, [], [], []))
    monkeypatch.setattr(selector, "suggest", lambda *args, **kwargs: (config, dict(metadata, phase="tpe")))
    monkeypatch.setattr(surrogates, "suggest_forest", lambda *args, **kwargs: (config, dict(metadata, phase="forest")))
    rolls = iter(np.random.default_rng(0).random(40))
    monkeypatch.setattr(selector, "roll", lambda: next(rolls))
    plan = dict(
        selector="forest",
        random_trials=10,
        mix={"random": 80, "forest": 10, "tpe": 10},
        seconds=600,
        random_params=[],
        seed=5,
        trials=[],
        bounds={},
        bank_manifest_sha256="bank",
        supervision_sha256="labels",
    )
    for _ in range(40):
        selector.append_trial(tmp_path, plan, registry, state(dict(statement, seconds=600)))
    chosen = [item["selector"] for item in plan["trials"]]
    expected = [
        "random" if value < 0.8 else "forest" if value < 0.9 else "tpe"
        for value in np.random.default_rng(0).random(40)
    ]
    assert chosen == expected
    assert set(chosen) == {"random", "forest", "tpe"}
    selection = read_json(tmp_path / plan["trials"][-1]["directory"] / "selection.json")
    assert selection["mix"] == plan["mix"]
    with pytest.raises(ValueError, match="sum to 100"):
        selector.check_mix({"random": 50, "tpe": 40})
    with pytest.raises(ValueError, match="unknown selectors"):
        selector.check_mix({"random": 50, "grid": 50})
