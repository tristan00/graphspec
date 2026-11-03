from contextlib import closing

import pytest

from graphspec import Builder, BuildPlan
from graphspec.build.staged import Staged
from graphspec.facts.stored import Facts
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.time import ALWAYS
from graphspec.sources import synthetic

pytestmark = pytest.mark.builder


@timed("append_test.schema")
def schema():
    return dict(
        version=VERSION, time=dict(model="clocks", units=["tick"], ties=None), observers=True,
        columns=dict(a=dict(type="number", encoder="value"), b=dict(type="number", encoder="value")),
        graph=dict(columns=[]), nodes=dict(node=dict(columns=["a", "b"], holds=None)), sets={},
        edges=dict(link=synthetic.edge("node", "node", [])),
        tasks=dict(key="ticket", grouping=None, signals=0, references=dict(at=dict(sort="node", kinds=["node"])),
                   kinds=dict(level=synthetic.kind(["at"], [], dict(kind="number", series="level", about=None,
                                                                  utility=synthetic.valued(["level"])), None))),
    )


@timed("append_test.before")
def before(path):
    builder = Builder({}, path).schema(schema()).graph("one", "tick")
    builder.node("node", "z", None, known=None, a=1)
    builder.change("node", "z", 0, known=None, seen=["alice"], b=2)
    builder.change("node", "z", 1, known=None, seen=["bob"], b=3)
    builder.edge("link", ("node", "z"), ("node", "z"), 0, known=None)
    builder.task("level", 1, 1, dict(at="z"), "train", 1, None)
    builder.task("level", 2, 2, dict(at="z"), "train", 2, None)
    builder.task("level", 3, 3, dict(at="z"), "validation", None, None)
    return builder


@timed("append_test.after")
def after(builder):
    builder.graphs["one"]["values"][0]["rows"]["replaced"][0] = 4
    builder.graphs["one"]["edges"][0]["rows"]["replaced"][0] = 5
    builder.node("node", "z", None, known=4, a=9)
    builder.edge("link", ("node", "z"), ("node", "z"), 0, known=5, present=False)
    builder.node("node", "a", 4, known=None, a=4, b=None)
    builder.change("node", "z", 4, known=None, seen=["alice"], b=None)
    builder.change("node", "z", 5, known=None, seen=["bob"], b=5)
    builder.event("link", ("node", "a"), ("node", "z"), 5, known=None)
    builder.task("level", 3, 3, dict(at="z"), "validation", 3, 5)
    builder.task("level", 4, 6, dict(at="a"), "validation", 4, None)
    return builder


@timed("append_test.semantic")
def semantic(tables):
    nodes = {row["entity"]: (row["kind"], row["key"]) for row in tables["nodes"].to_pylist()}
    result = {}
    for kind, table in tables["observations"].items():
        rows = []
        for row in table.to_pylist():
            row["entity"] = nodes[row["entity"]]
            rows.append(row)
        result["obs:" + kind] = sorted(rows, key=repr)
    for kind, table in tables["edges"].items():
        rows = []
        for row in table.to_pylist():
            row["source"], row["target"] = nodes[row["source"]], nodes[row["target"]]
            rows.append(row)
        result["edge:" + kind] = sorted(rows, key=repr)
    result["tasks"] = [{name: value for name, value in row.items() if name != "id"} for row in tables["tasks"].to_pylist()]
    result["targets"] = [{name: value for name, value in row.items() if name != "id"} for row in tables["targets"].to_pylist()]
    return result


def test_later_staged_append_preserves_prefix_and_matches_whole_with_observer_scopes(tmp_path, monkeypatch):
    from graphspec.build import appending

    plan = BuildPlan(16 << 20, 2)
    builder = before(tmp_path)
    task = builder.stage(tmp_path, "split", plan)
    stage = tmp_path / "split.staging"
    original = {path.relative_to(stage): sha256_file(path) for path in stage.rglob("*.parquet")}
    after(builder)
    write = appending.write_graph
    supplied = []

    @timed("append_test.recorded_write")
    def record(schema_mapping, graph, *args, **kwargs):
        supplied.append(graph)
        return write(schema_mapping, graph, *args, **kwargs)

    monkeypatch.setattr(appending, "write_graph", record)
    builder.append(task, plan)
    assert len(supplied) == 1
    assert supplied[0]["tasks"]["key"] == [3, 4]
    assert all(max(ALWAYS if moment is None else moment, ALWAYS if known is None else known) >= 4
               for statement in supplied[0]["values"] + supplied[0]["edges"]
               for moment, known in zip(statement["rows"].get("since", statement["rows"].get("at")), statement["rows"]["known"]))
    assert {path: sha256_file(stage / path) for path in original} == original
    builder.stage(tmp_path, "whole", plan)
    with closing(Staged(stage, Schema(schema()))) as split, closing(Staged(tmp_path / "whole.staging", Schema(schema()))) as whole:
        assert semantic(split.reader().graph(0)) == semantic(whole.reader().graph(0))
    assert read_json(stage / "build.json")["fingerprint"] == read_json(tmp_path / "whole.staging" / "build.json")["fingerprint"]


def test_append_refuses_retroactive_feature_or_task_changes_without_mutating_staging(tmp_path, monkeypatch):
    from graphspec.build import appending

    plan = BuildPlan(16 << 20, 1)
    builder = before(tmp_path)
    task = builder.stage(tmp_path, "split", plan)
    root = tmp_path / "split.staging"
    digest = sha256_file(root / "build.json")
    stage_digest = sha256_file(root / "stage.json")
    builder.graphs["one"]["values"][0]["rows"]["column:a"][0] = 9
    with pytest.raises(ValueError, match="existing fact"):
        builder.append(task, plan)
    assert sha256_file(root / "build.json") == digest
    builder.graphs["one"]["values"][0]["rows"]["column:a"][0] = 1
    builder.graphs["one"]["tasks"]["split"][0] = "validation"
    with pytest.raises(ValueError, match="inputs of an existing task"):
        builder.append(task, plan)
    assert sha256_file(root / "build.json") == digest
    builder.graphs["one"]["tasks"]["split"][0] = "train"
    builder.task("level", 4, 4, dict(at="missing"), "validation", 4, None)
    with pytest.raises(ValueError, match="names the node missing"):
        builder.append(task, plan)
    assert sha256_file(root / "build.json") == digest
    assert not (root / "append_00000001").exists()
    builder.graphs["one"]["tasks"]["references"]["at"][-1] = "z"
    original_write = appending.write_json

    @timed("append_test.failed_publish")
    def failed_publish(path, value):
        if path == root / "build.json":
            raise OSError("publication refused")
        return original_write(path, value)

    monkeypatch.setattr(appending, "write_json", failed_publish)
    with pytest.raises(OSError, match="publication refused"):
        builder.append(task, plan)
    assert sha256_file(root / "stage.json") == stage_digest
    assert sha256_file(root / "build.json") == digest
    assert not (root / "append_00000001").exists()
    monkeypatch.setattr(appending, "write_json", original_write)
    builder.append(task, plan)
    assert read_json(root / "build.json")["generation"] == 1


@pytest.mark.parametrize("escape", (False, True))
def test_truth_layers_refuse_cycles_and_paths_outside_the_bank(tmp_path, escape):
    root = tmp_path / "bank"
    one, two = root / "truth" / "one", root / "truth" / "two"
    one.mkdir(parents=True)
    two.mkdir()
    write_json(root / "manifest.json", {})
    plan = dict(rows="identity", patches=[], replace={})
    write_json(one / "graph.json", dict(append=dict(base="../../../outside" if escape else "../two", delta="../two", plan=plan)))
    write_json(two / "graph.json", dict(append=dict(base="../one", delta="../one", plan=plan)))
    with pytest.raises(ValueError, match="lineage" if escape else "cycle"):
        Facts.load(one)
