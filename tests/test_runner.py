import subprocess
import time
from pathlib import Path

import pytest

from graphspec.infra.files import JsonCache, read_json, sha256_json, write_json
from tuning import run as runner
from tuning.budget import FitBudget
from tuning.records import VERSION, Outcome, Run, Status
from tuning.statement import state as stated


def test_fit_progress_survives_failure_with_partial_timings(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("tuning.budget.time.perf_counter", lambda: clock[0])
    with pytest.raises(RuntimeError, match="failed batch"):
        with FitBudget(tmp_path, 100, 300) as budget:
            clock[0] = 13.0
            budget.timings["prepare"] = 3.0
            budget.pulse("training")
            clock[0] = 17.0
            budget.timings["train_compute"] = 2.5
            budget.pulse("training")
            saved = read_json(tmp_path / "fit_progress.json")
            assert saved["phase"] == "training"
            assert saved["phase_seconds"] == {"preparation": 3.0, "training": 4.0}
            clock[0] = 19.0
            raise RuntimeError("failed batch")
    saved = read_json(tmp_path / "fit_progress.json")
    assert saved["phase_seconds"] == {"preparation": 3.0, "training": 6.0}
    assert saved["timings"] == {"prepare": 3.0, "train_compute": 2.5}


def test_preparation_failure_preserves_its_stage_and_elapsed_time(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("tuning.budget.time.perf_counter", lambda: clock[0])
    with pytest.raises(MemoryError):
        with FitBudget(tmp_path, 100, 300):
            clock[0] = 14.0
            raise MemoryError()
    progress = read_json(tmp_path / "fit_progress.json")
    assert progress["phase"] == "preparation"
    assert progress["phase_seconds"] == {"preparation": 4.0}
    assert progress["timings"] == {}


def test_active_measurement_is_saved_before_it_finishes(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("tuning.budget.time.perf_counter", lambda: clock[0])
    with pytest.raises(MemoryError):
        with FitBudget(tmp_path, 100, 300) as budget:
            budget.pulse("training")
            with budget.measure("train_compute"):
                clock[0] = 15.0
                budget.pulse()
                assert read_json(tmp_path / "fit_progress.json")["timings"]["train_compute"] == 5.0
                clock[0] = 16.0
                raise MemoryError()
    progress = read_json(tmp_path / "fit_progress.json")
    assert progress["timings"]["train_compute"] == 6.0
    runner.mark_timeout(tmp_path, 100, 10, {}, "deadline")
    outcome = read_json(tmp_path / "timeout.json")
    assert outcome["stage"] == "training"
    assert outcome["timings"] == progress["timings"]
    assert outcome["fit_phases"] == progress["phase_seconds"]


def plan_at(directory):
    from tuning.selector import seeds

    items = []
    for position, name in enumerate(("opaque-first", "opaque-second")):
        trial = directory / name
        trial.mkdir()
        write_json(
            trial / "config.json",
            dict(
                seed=seeds(5, position)[1],
                model={"view": "sequence"},
                graph={},
                epochs=1,
                max_seconds=10,
            ),
        )
        write_json(trial / "selection.json", {"graph_summary": {}})
        items.append(
            dict(
                directory=name,
                model=sha256_json({"view": "sequence"}),
                graph=sha256_json({}),
                selector="random",
            )
        )
    plan = dict(
        version=VERSION,
        bank="unused",
        trials=items,
        limit=2,
        seconds=10,
        selector="random",
        train_rows=2,
        holdout_rows=2,
        objective=None,
        workers=1,
        variant_bytes=100,
    )
    write_json(directory / "plan.json", plan)
    return plan


def test_trial_states_do_not_depend_on_directory_naming_or_missing_metrics(tmp_path):
    plan = plan_at(tmp_path)
    first, second = (tmp_path / item["directory"] for item in plan["trials"])
    (first / "worker.log").write_text("started")
    write_json(second / "timeout.json", {"error": "deadline"})
    trials = Run(tmp_path, JsonCache()).trials()
    assert [trial.index for trial in trials] == [0, 1]
    assert [trial.status for trial in trials] == [Status.INTERRUPTED, Status.TIMEOUT]
    write_json(second / "failure.json", {"error": "also failed"})
    with pytest.raises(ValueError, match="conflicting"):
        Outcome.read(second, JsonCache())


def patch_execution(monkeypatch, directory, events, cleanup_error=False):
    def layout(*args):
        return directory / "blocks.json", 1

    class Worker:
        pid = 42

        def __init__(self, module, arguments, stream):
            self.trial = Path(arguments[arguments.index("--output") + 1])

        def __enter__(self):
            events.append(("start", self.trial.name))
            return self

        def wait(self, timeout):
            if self.trial.name == "opaque-first":
                raise subprocess.TimeoutExpired("owned-worker", timeout)
            write_json(self.trial / "result.json", {"status": "complete"})
            return 0

        def __exit__(self, *exception):
            events.append(("joined", self.trial.name))
            if self.trial.name == "opaque-first":
                assert (self.trial / "timeout.json").exists()
            if cleanup_error:
                raise RuntimeError("OS refused process termination")

    class Deriver:
        def __init__(self, path, device, resident):
            assert device == "cpu"

        def stamp(self, config):
            return directory

    monkeypatch.setattr(runner, "WorkerProcess", Worker)
    monkeypatch.setattr(
        "tuning.budget.wait_worker",
        lambda process, deadline, directory, preparation: process.wait(
            timeout=max(0.0, deadline - time.perf_counter())
        ),
    )
    monkeypatch.setattr(runner, "training_layout", layout)
    monkeypatch.setattr(runner, "snapshot", lambda *args: None)
    monkeypatch.setattr(runner, "report", lambda *args: [])
    monkeypatch.setattr("tuning.data.Deriver", Deriver)
    monkeypatch.setattr("tuning.variants.prune", lambda *args: None)


def test_timeout_is_recorded_then_joined_before_the_next_trial(tmp_path, monkeypatch, statement):
    statement = dict(statement, seconds=10)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events)
    runner.execute(tmp_path, plan, False, None, stated(statement))
    assert events == [
        ("start", "opaque-first"),
        ("joined", "opaque-first"),
        ("start", "opaque-second"),
        ("joined", "opaque-second"),
    ]
    assert read_json(tmp_path / "runner.json")["status"] == "complete"
    assert not (tmp_path / "running.json").exists()
    assert Outcome.read(tmp_path / "opaque-first", JsonCache()).status == Status.TIMEOUT
    assert Outcome.read(tmp_path / "opaque-second", JsonCache()).status == Status.COMPLETE


def test_the_seconds_of_the_statement_control_deadline_and_all_terminal_records(
    tmp_path, monkeypatch, statement
):
    from tuning.selector import seeds

    statement = dict(statement, seconds=45)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events)
    for item in plan["trials"]:
        path = tmp_path / item["directory"] / "config.json"
        write_json(path, dict(read_json(path), max_seconds=45))
    deadlines = []

    def wait(process, deadline, directory, preparation):
        assert read_json(tmp_path / "running.json")["max_seconds"] == 45
        deadlines.append(deadline - time.perf_counter())
        return process.wait(timeout=1)

    monkeypatch.setattr("tuning.budget.wait_worker", wait)
    runner.execute(tmp_path, plan, False, None, stated(statement))
    assert all(44 < deadline <= 45 for deadline in deadlines) and len(deadlines) == 2
    assert read_json(tmp_path / "opaque-first" / "timeout.json")["budget"] == 45
    assert read_json(tmp_path / "opaque-first" / "trial.json")["budget"] == 45
    assert read_json(tmp_path / "opaque-second" / "trial.json")["budget"] == 45
    for position, item in enumerate(plan["trials"]):
        written = read_json(tmp_path / item["directory"] / "statement.json")
        assert written == dict(statement, seed=seeds(5, position)[1])
        assert stated(written).seconds == 45


@pytest.mark.parametrize(
    "change,error,message",
    [
        (dict(max_seconds=45), ValueError, "config.json of opaque-first holds max_seconds 45"),
        (dict(seed=5), ValueError, "config.json of opaque-first holds seed 5"),
        ("max_seconds", KeyError, "config.json of opaque-first lacks max_seconds"),
        ("seed", KeyError, "config.json of opaque-first lacks seed"),
    ],
)
def test_a_trial_whose_copy_differs_from_the_statement_stops_naming_the_field(
    tmp_path, monkeypatch, statement, change, error, message
):
    statement = dict(statement, seconds=10)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events)
    path = tmp_path / "opaque-first" / "config.json"
    config = read_json(path)
    if isinstance(change, dict):
        config.update(change)
    else:
        del config[change]
    write_json(path, config)
    with pytest.raises(error, match=message):
        runner.execute(tmp_path, plan, False, None, stated(statement))
    assert events == []
    assert read_json(path) == config
    assert not (tmp_path / "opaque-first" / "statement.json").exists()


def test_cleanup_failure_preserves_timeout_and_stops_before_another_trial(tmp_path, monkeypatch, statement):
    statement = dict(statement, seconds=10)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events, cleanup_error=True)
    with pytest.raises(RuntimeError, match="OS refused"):
        runner.execute(tmp_path, plan, False, None, stated(statement))
    assert events == [("start", "opaque-first"), ("joined", "opaque-first")]
    assert Outcome.read(tmp_path / "opaque-first", JsonCache()).status == Status.TIMEOUT
    state = read_json(tmp_path / "runner.json")
    assert state["status"] == "failed" and "OS refused" in state["error"]


def test_expired_preparation_never_launches_a_worker(tmp_path, monkeypatch, statement):
    statement = dict(statement, seconds=10)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events)
    runner.execute_trial(
        tmp_path, plan, plan["trials"][0], 0, time.perf_counter() - 11, {}, stated(statement)
    )
    assert events == []
    assert Outcome.read(tmp_path / "opaque-first", JsonCache()).status == Status.TIMEOUT


def test_worker_deadline_exit_becomes_timeout_and_continues(tmp_path, monkeypatch, statement):
    statement = dict(statement, seconds=10)
    plan = plan_at(tmp_path)
    events = []
    patch_execution(monkeypatch, tmp_path, events)
    monkeypatch.setattr(runner.WorkerProcess, "wait", lambda self, timeout: runner.TIMEOUT_EXIT_CODE)
    monkeypatch.setattr(
        runner.WorkerProcess, "__exit__", lambda self, *args: events.append(("joined", self.trial.name))
    )
    runner.execute(tmp_path, plan, False, None, stated(statement))
    assert len(events) == 4
    assert all(
        Outcome.read(tmp_path / item["directory"], JsonCache()).status == Status.TIMEOUT
        for item in plan["trials"]
    )
    assert read_json(tmp_path / "runner.json")["status"] == "complete"


def test_unfinished_trials_are_discarded_instead_of_continued(tmp_path):
    plan = plan_at(tmp_path)
    first, second = (tmp_path / item["directory"] for item in plan["trials"])
    (first / "worker.log").write_text("started")
    write_json(second / "result.json", {"status": "complete"})
    runner.discard_unfinished(tmp_path, plan)
    assert [item["directory"] for item in plan["trials"]] == ["opaque-second"]
    assert not first.exists()
    assert second.exists()
