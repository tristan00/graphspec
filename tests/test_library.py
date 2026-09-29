import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from support import VALUES, asked, stated

from graphspec import GraphTables, Stage, write_task
from graphspec.infra.runtime import heavy_job, output_path
from graphspec.staged import Staged
from tuning.build import build, open_source


def test_output_cannot_be_inside_source_tree():
    with pytest.raises(ValueError, match="outside the code"):
        output_path(Path(__file__).parents[1] / "artifacts" / "new-run")


def test_missing_staging_does_not_read_database(tmp_path):
    schema = stated(dict(client=["value"]), {}, VALUES, asked(["client"])).mapping()
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "task.json", dict(staging="missing", database="other.lbug", schema=schema))
    with pytest.raises(ValueError, match="states no schema"):
        open_source(tmp_path / "task.json", dict(staging="missing"))


def test_public_graph_writer_builds_without_database_and_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ladybug", None)
    schema = stated(
        dict(customer=["value", "other"], organization=["value"]),
        dict(member=dict(endpoints=[("customer", "organization")], directed=True, columns=[])),
        dict(value=dict(type="number", encoder="value"), other=dict(type="number", encoder="standard")),
        asked(["customer"]),
    )
    graph = GraphTables(0, "different", "step", schema)
    when = dict(since=0, until=None, seq=[0, 1, 2, 3])
    graph.nodes("customer", ["a", "b", "c", "d"])
    graph.nodes("organization", ["shared"])
    graph.observe("customer", ["a", "b", "c", "d"], when, value=np.ones(4), other=[1, 2, None, 4])
    graph.observe("organization", ["shared"], dict(when, seq=4), value=np.ones(1))
    with pytest.raises(TypeError, match="when|present"):
        graph.edges("member", "customer", "organization", ["a", "b", "c", "d"], ["shared"] * 4)
    graph.edges(
        "member", "customer", "organization", ["a", "b", "c", "d"], ["shared"] * 4, when, present=True
    )
    graph.tasks(
        "outcome",
        [10, 11, 12, 13],
        0,
        dict(holder=("customer", ["a", "b", "c", "d"])),
        ["train", "train", "validation", "validation"],
        [0, 1, 0, 1],
        [None] * 4,
        None,
        None,
    )
    with Stage(tmp_path / "source") as stage:
        graph.write(stage)
    before = {p.name: p.read_bytes() for p in (tmp_path / "source").iterdir()}
    task = write_task(tmp_path / "task.json", "different", tmp_path / "source", schema)
    manifest = build(task, tmp_path / "bank", 1, ("train",))
    assert manifest["spec"]["extras"]["target"] == "number" and manifest["rows"] == 4
    assert manifest["split_rows"] == dict(train=2, validation=2)
    assert manifest["spec"]["nodes"] == ["customer", "organization"]
    assert "widths" not in manifest["spec"] and manifest["spec"]["encoding"]["states"] is None
    Staged(tmp_path / "source", schema)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "source").iterdir()}


def test_infra_does_not_import_graph_logic():
    root = Path(__file__).parents[1] / "graphspec" / "infra"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(
                name == "graphspec" or name.startswith(("graphspec.staged", "tuning")) for name in names
            ), path


def test_heavy_jobs_are_exclusive_across_processes(monkeypatch):
    from graphspec.infra import runtime

    name = "heavy-exclusion-%d" % os.getpid()
    held = runtime.singleton
    monkeypatch.setattr(runtime, "singleton", lambda given, message: held(name, message))
    program = (
        "from graphspec.infra import runtime\n"
        "held = runtime.singleton\n"
        "runtime.singleton = lambda given, message: held(%r, message)\n"
        "with runtime.heavy_job():\n    pass"
    ) % name
    with heavy_job():
        result = subprocess.run(
            [sys.executable, "-B", "-c", program], capture_output=True, text=True, timeout=4
        )
    assert result.returncode != 0 and "another graph build or training job" in result.stderr


def test_the_heavy_job_of_production_requests_the_singleton_named_heavy(monkeypatch):
    from contextlib import contextmanager

    from graphspec.infra import runtime

    requested = []

    @contextmanager
    def recorder(name, message):
        requested.append((name, message))
        yield

    monkeypatch.setattr(runtime, "singleton", recorder)
    monkeypatch.setattr(runtime.ACTIVE, "held", False, raising=False)
    with runtime.heavy_job():
        assert runtime.ACTIVE.held
        with runtime.heavy_job():
            pass
    assert [name for name, _ in requested] == ["heavy"]
    assert "another graph build or training job is running" in requested[0][1]


def test_database_export_and_staging_read_the_same_graph(bank, tmp_path):
    pytest.importorskip("ladybug")
    from graphspec.database import export
    from graphspec.infra.files import read_json, write_json
    from graphspec.truth import STORED, Facts

    export(bank.parent / "tiny.staging", tmp_path / "tiny.lbug")
    task = tmp_path / "task.json"
    schema = read_json(bank.parent / "tiny.task.json")["schema"]
    write_json(task, dict(name="tiny", database="tiny.lbug", schema=schema))
    build(task, tmp_path / "database-bank", 1, ("train",))
    for index in range(4):
        left = Facts.load(bank / "truth" / ("graph_%04d" % index))
        right = Facts.load(tmp_path / "database-bank" / "truth" / ("graph_%04d" % index))
        assert all(np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True) for name in STORED)
