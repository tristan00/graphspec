import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from graphspec import GraphTables, Stage, write_task
from graphspec.infra.runtime import heavy_job, output_path
from graphspec.staged import Staged
from tuning.build import build, open_source


def test_output_cannot_be_inside_source_tree():
    with pytest.raises(ValueError, match="outside the code"):
        output_path(Path(__file__).parents[1] / "artifacts" / "new-run")


def test_missing_staging_does_not_read_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "task.json", dict(staging="missing", database="other.lbug"))


def test_public_graph_writer_builds_without_database_and_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ladybug", None)
    graph = GraphTables(0, "different", 4)
    graph.nodes("customer", ["a", "b", "c", "d"])
    graph.nodes("organization", ["shared"])
    graph.observe("customer", ["a", "b", "c", "d"], np.ones((4, 4)))
    graph.observe("organization", ["shared"], np.ones((1, 4)))
    graph.edges("member", "customer", "organization", ["a", "b", "c", "d"], ["shared"] * 4)
    graph.labels(
        "customer", ["a", "b", "c", "d"], [0, 1, 0, 1], ["train", "train", "validation", "validation"]
    )
    with Stage(tmp_path / "source") as stage:
        graph.write(stage)
    before = {p.name: p.read_bytes() for p in (tmp_path / "source").iterdir()}
    task = write_task(tmp_path / "task.json", "different", tmp_path / "source", "outcome")
    manifest = build(task, tmp_path / "bank", 1)
    assert manifest["target"]["level"] == "node" and manifest["rows"] == 4
    assert manifest["spec"]["nodes"] == ["customer", "organization"]
    Staged(tmp_path / "source")
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


def test_database_export_and_staging_read_the_same_graph(bank, tmp_path):
    pytest.importorskip("ladybug")
    from graphspec.database import export
    from graphspec.infra.files import write_json
    from graphspec.truth import ARRAYS, Graph

    export(bank.parent / "tiny.staging", tmp_path / "tiny.lbug")
    task = tmp_path / "task.json"
    write_json(task, dict(name="tiny", database="tiny.lbug", target="outcome"))
    build(task, tmp_path / "database-bank", 1)
    for index in range(4):
        left = Graph.load(bank / "truth" / ("graph_%04d" % index))
        right = Graph.load(tmp_path / "database-bank" / "truth" / ("graph_%04d" % index))
        assert all(np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True) for name in ARRAYS)
