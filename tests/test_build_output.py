import sqlite3
import sys

import numpy as np
import pyarrow as pa
import pytest
from test_builder import schema

from graphspec import Builder, BuildPlan, Stage, inspect
from graphspec.budgets import GraphCapacityError
from graphspec.build.output import manifest
from graphspec.build.staged import Unit
from graphspec.facts.stored import STORED, Facts
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import Schema
from graphspec.sources import synthetic
from tuning import build_bank
from tuning.bank.bank import Bank

pytestmark = pytest.mark.extended


@timed("output_test.source")
def source(root):
    mapping = schema()
    mapping["columns"]["shade"] = dict(type="category", encoder="onehot")
    mapping["nodes"] = {kind: dict(columns=["a", "b", "shade"], holds=None) for kind in ("zeta", "node")}
    mapping["edges"] = {name: synthetic.edge("node", "node", []) for name in ("zlink", "alink")}
    builder = Builder({}, root).schema(mapping)
    for graph in ("one", "two", "three"):
        builder.graph(graph, "tick")
        for kind in mapping["nodes"]:
            for at in range(12):
                builder.node(kind, "n%d" % at, 0, graph=graph, known=None, a=at, b=at + 1,
                             shade="red" if at % 2 else "blue")
        for at in range(4):
            builder.event("alink", ("node", "n0"), ("node", "n1"), at, graph=graph, known=None)
            builder.task("level", at, at + 1, dict(at="n0"), "train" if at < 2 else "validation",
                         float(at), None, graph=graph)
    return builder


def test_one_and_three_physical_shards_have_every_equal_array_and_fingerprint(tmp_path, monkeypatch):
    from graphspec.build import writing

    builder = source(tmp_path)
    writes = []
    original = Stage.write

    @timed("output_test.record_write")
    def tracked(stage, tables):
        writes.extend(table.nbytes for table in tables.values())
        return original(stage, tables)

    monkeypatch.setattr(Stage, "write", tracked)
    plans = [BuildPlan(16 << 20, 1, 1), BuildPlan(16 << 20, 2, 3)]
    banks, manifests = [], []
    for number, plan in enumerate(plans):
        path = tmp_path / ("bank%d" % number)
        build_bank(builder, path, plan.workers, ("train",), plan=plan)
        banks.append(Bank(path))
        described = inspect(path)
        assert '"node": 12' in described and '"train": 2' in described and "skipped references: 0" in described
        staged = path.parent / (path.name + "-source.staging")
        manifests.append(read_json(staged / "build.json"))
        for graph in manifests[-1]["graphs"]:
            entries = read_json(staged / graph["directory"] / "stage.json")
            assert all(len(entry["files"]) == plan.shards for entry in entries)
            with sqlite3.connect(staged / graph["key_file"]) as keys:
                assert keys.execute("SELECT COUNT(*) FROM nodes WHERE key='n0'").fetchone()[0] == 2
        assert list(manifests[-1]["schema"]["nodes"]) == ["node", "zeta"]
        assert list(manifests[-1]["schema"]["edges"]) == ["alink", "zlink"]
        assert manifests[-1]["vocabularies"]["shade"] == ["blue", "red"]
    assert max(writes) < min(plan.worker_bytes // 4 for plan in plans)
    assert manifests[0]["fingerprint"] == manifests[1]["fingerprint"]
    assert manifests[0]["source_fingerprint"] == manifests[1]["source_fingerprint"]
    for entry in banks[0].manifest["graphs"]:
        before, after = (Facts.load(bank.truth(entry["index"])) for bank in banks)
        assert before.keys == after.keys
        for name in STORED:
            np.testing.assert_equal(np.asarray(before.arrays[name]), np.asarray(after.arrays[name]), err_msg=name)
    assert banks[0].rows == banks[1].rows
    repeated = builder.stage(tmp_path, "again", plans[0])
    assert read_json(repeated.parent / "again.staging" / "build.json")["fingerprint"] == manifests[0]["fingerprint"]
    with pytest.raises(GraphCapacityError, match="metadata budget"):
        builder.stage(tmp_path, "tiny", BuildPlan(4096, 1))
    with pytest.raises(GraphCapacityError, match="row 0"):
        list(writing.batches([dict(value="x" * 100)], 8, "oversize"))
    keys = writing.Keys(tmp_path / "vocabulary.sqlite3", Schema(builder.records()["schema"]), 4096, None)
    try:
        keys.labels("shade", ("value%d" % number for number in range(100)))
        with pytest.raises(GraphCapacityError, match="fitted vocabularies"):
            keys.vocabularies(4096)
    finally:
        keys.close()


def test_reuse_checks_source_and_every_immutable_file(tmp_path):
    builder, plan = source(tmp_path), BuildPlan(16 << 20, 1)
    task = builder.stage(tmp_path, "kept", plan)
    folder = tmp_path / "kept.staging"
    entry = read_json(folder / "stage.json")[0]
    file = folder / entry["files"][0]
    before = file.stat().st_mtime_ns, sha256_file(file)
    assert builder.stage(tmp_path, "kept", plan, reuse=True) == task
    assert (file.stat().st_mtime_ns, sha256_file(file)) == before
    task_record = read_json(task)
    for field, value in (("schema", {}), ("staging", "another.staging")):
        write_json(task, dict(task_record, **{field: value}))
        with pytest.raises(ValueError, match="task record changed"):
            builder.stage(tmp_path, "kept", plan, reuse=True)
    write_json(task, task_record)
    with pytest.raises(FileExistsError, match="reuse"):
        builder.stage(tmp_path, "kept", plan)
    builder.node("node", "later", 5, graph="one", known=None, a=1)
    with pytest.raises(ValueError, match="does not match"):
        builder.stage(tmp_path, "kept", plan, reuse=True)
    with file.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="staged file changed"):
        source(tmp_path).stage(tmp_path, "kept", plan, reuse=True)


def test_oversize_rows_stop_before_source_serialization_and_manifest_edits_block_reuse(tmp_path, monkeypatch):
    from graphspec.build import writing

    builder, plan = source(tmp_path), BuildPlan(16 << 20, 1)
    builder.graphs["one"]["nodes"]["node"].append("x" * plan.memory_bytes)

    @timed("output_test.unserialized")
    def refused(value):
        raise AssertionError("oversize source was serialized before capacity checks")

    with monkeypatch.context() as patched:
        patched.setattr(writing, "fingerprint", refused)
        with pytest.raises(GraphCapacityError, match="nodes of node"):
            builder.stage(tmp_path, "oversized", plan)
    builder = source(tmp_path)
    builder.stage(tmp_path, "manifest", plan)
    folder = tmp_path / "manifest.staging"
    entries = read_json(folder / "stage.json")
    entries[0]["rows"] += 1
    write_json(folder / "stage.json", entries)
    with pytest.raises(ValueError, match="table manifest changed"):
        builder.stage(tmp_path, "manifest", plan, reuse=True)


def test_builder_and_cli_inspection_show_schema_counts_splits_and_skips(tmp_path, monkeypatch, capsys):
    from tuning.__main__ import main

    monkeypatch.setattr("graphspec.infra.log.configure", lambda: None)
    builder = source(tmp_path)
    shown = inspect(builder)
    assert "time: one clock" in shown and '"node": 12' in shown
    assert '"zlink": 0' in shown
    assert '"train": 2' in shown and '"validation": 2' in shown and "skipped references: 0" in shown
    script = tmp_path / "builder.py"
    script.write_text("from test_build_output import source\nbuilder = source(sources)\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["graphspec", "inspect", str(script), "--sources", str(tmp_path)])
    main()
    assert capsys.readouterr().out.strip() == shown.strip()
    task = builder.stage(tmp_path, "inspected", BuildPlan(16 << 20, 1))
    assert "skipped references: 0" in inspect(task)
    output = tmp_path / "cli-bank"
    monkeypatch.setattr(sys, "argv", ["graphspec", "build", "--script", str(script), "--sources", str(tmp_path),
                                     "--output", str(output), "--trained", "train", "--workers", "1",
                                     "--memory-bytes", str(16 << 20)])
    main()
    bank = Bank(output)
    assert len(bank.rows) == 12
    assert {row["graph"] for row in bank.rows} == {"one", "two", "three"}
    assert all(graph["inspection"]["tasks"]["level"] == dict(splits=dict(train=2, validation=2), without_split=0)
               for graph in bank.manifest["graphs"])
    assert '"node": 12' in inspect(output) and "skipped references: 0" in inspect(output)


def test_endpoint_patches_apply_to_file_rows_before_row_group_slices(tmp_path):
    with Stage(tmp_path / "patched") as stage:
        for start in (0, 2):
            stage.write({("observation", None, None): pa.table(dict(id=[start, start + 1], until=pa.array([None, None], pa.int64())))})
    directory = tmp_path / "patched"
    entries = read_json(directory / "stage.json")
    entries[0]["patches"] = [dict(file=entries[0]["files"][0], row=2, column="until", value=9)]
    write_json(directory / "stage.json", entries)
    unit = Unit(directory, manifest(directory)[0])
    assert unit.read().column("until").to_pylist() == [None, None, 9, None]
    assert unit.rows_between(1, 4).column("until").to_pylist() == [None, 9, None]
    for file in unit.files:
        file.close()
    entries[0]["patches"] = None
    write_json(directory / "stage.json", entries)
    with pytest.raises(ValueError, match="patches are a list"):
        manifest(directory)


def test_merge_rejects_duplicate_shards_and_incompatible_schemas(tmp_path):
    for name, values in (("first", [1]), ("second", ["different"])):
        with Stage(tmp_path / name) as stage:
            stage.write({("node", None, None): pa.table(dict(id=values))})
    with pytest.raises(ValueError, match="shard twice"):
        Stage.merge(tmp_path, [tmp_path / "first", tmp_path / "first"])
    with pytest.raises(ValueError, match="incompatible schemas"):
        Stage.merge(tmp_path, [tmp_path / "first", tmp_path / "second"])
