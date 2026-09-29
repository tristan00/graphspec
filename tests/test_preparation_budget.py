import subprocess

import pytest

from graphspec.infra.files import read_json, write_json
from tuning import budget


def test_parent_stops_preparation_even_when_heartbeat_stalls(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(budget.time, "perf_counter", lambda: clock[0])
    write_json(
        tmp_path / "fit_progress.json", dict(graph_preparation=dict(started=10, deadline=310, finished=None))
    )

    class Worker:
        def wait(self, timeout):
            clock[0] += timeout
            raise subprocess.TimeoutExpired("worker", timeout)

    with pytest.raises(budget.PreparationTimeout, match="300-second"):
        budget.wait_worker(Worker(), 1210, tmp_path, 300)
    assert clock[0] == 310


def test_finished_preparation_does_not_limit_training(tmp_path, monkeypatch):
    clock = [500.0]
    monkeypatch.setattr(budget.time, "perf_counter", lambda: clock[0])
    write_json(
        tmp_path / "fit_progress.json", dict(graph_preparation=dict(started=10, deadline=310, finished=250))
    )

    class Worker:
        def wait(self, timeout):
            return 0

    assert budget.wait_worker(Worker(), 1210, tmp_path, 300) == 0


def test_trial_deadline_can_expire_before_preparation_limit(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(budget.time, "perf_counter", lambda: clock[0])
    write_json(
        tmp_path / "fit_progress.json", dict(graph_preparation=dict(started=10, deadline=310, finished=None))
    )

    class Worker:
        def wait(self, timeout):
            clock[0] += timeout
            raise subprocess.TimeoutExpired("worker", timeout)

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        budget.wait_worker(Worker(), 15, tmp_path, 300)
    assert not isinstance(caught.value, budget.PreparationTimeout)
    assert clock[0] == 15


def test_worker_publishes_stage_deadline_and_rejects_late_completion(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(budget.time, "perf_counter", lambda: clock[0])
    with pytest.raises(TimeoutError, match="graph preparation"):
        with budget.FitBudget(tmp_path, 1210, 300) as recorder:
            with recorder.measure("graph_prepare"):
                saved = read_json(tmp_path / "fit_progress.json")
                assert saved["graph_preparation"] == dict(started=10, deadline=310, finished=None)
                clock[0] = 311
    saved = read_json(tmp_path / "fit_progress.json")
    assert saved["graph_preparation"]["finished"] == 311
    assert saved["timeout_reason"] == "graph preparation exceeded its 300-second limit"
    assert saved["timings"]["graph_prepare"] == 301
