import threading

import pytest

from graphspec.infra.files import read_json
from graphspec.infra.timing import Timeline, measure


def reconciled(snapshot):
    nodes = {tuple(node["path"]): node for node in snapshot["nodes"]}
    for path, node in nodes.items():
        children = sum(value["seconds"] for key, value in nodes.items() if key[:-1] == path)
        assert node["seconds"] == pytest.approx(node["self_seconds"] + children)
        assert node["self_seconds"] >= -1e-9
    return nodes


def test_nested_active_and_failed_steps_reconcile(tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("graphspec.infra.timing.time.perf_counter", lambda: clock[0])
    with pytest.raises(ValueError):
        with Timeline(tmp_path / "timing.json") as timeline, measure("fit"):
            clock[0] = 2
            with measure("training"):
                for _ in range(2):
                    with measure("train_compute"):
                        clock[0] += 2
                    clock[0] += 1
                with measure("train_pack"):
                    clock[0] += 3
                    snapshot = timeline.snapshot()
                    nodes = reconciled(snapshot)
                    assert nodes[("main", "fit", "training", "train_pack")]["active"] == 1
                    assert nodes[("main", "fit", "training", "train_compute")]["calls"] == 2
                    raise ValueError()
    nodes = reconciled(read_json(tmp_path / "timing.json"))
    assert not any(node["active"] for node in nodes.values())
    assert nodes[("main", "fit")]["seconds"] == 10
    assert nodes[("main", "fit", "training", "train_pack")]["last_error"] == "ValueError"
    assert nodes[("main", "fit", "training", "train_compute")]["failures"] == 0


def test_background_work_has_a_separate_clock():
    with Timeline() as timeline, measure("fit"):

        def background():
            with measure("prefetch"):
                pass

        thread = threading.Thread(target=background)
        thread.start()
        thread.join()
    nodes = reconciled(timeline.snapshot())
    assert ("background", "prefetch") in nodes
    assert ("main", "fit", "prefetch") not in nodes


def test_recorder_reuse_preserves_totals_and_restarts_snapshots(tmp_path):
    timeline = Timeline(tmp_path / "timing.json")
    for _ in range(2):
        with timeline, measure("graph_job"):
            assert not timeline.stopped.is_set()
    nodes = reconciled(read_json(tmp_path / "timing.json"))
    assert nodes[("main", "graph_job")]["calls"] == 2
