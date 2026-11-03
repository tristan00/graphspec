import numpy as np
import pytest

from graphspec import Builder, BuildPlan
from graphspec.build import records
from graphspec.build.writing import canonical_records
from graphspec.facts.stored import STORED, Facts
from graphspec.infra.timing import timed
from graphspec.schema.schema import VALUE, VERSION
from graphspec.schema.targets import TARGETS
from graphspec.sources import synthetic
from tuning import build_bank
from tuning.bank.bank import Bank
from tuning.bank.build import build

pytestmark = [pytest.mark.xdist_group("fixture"), pytest.mark.builder]
PLAN = BuildPlan(64 * 1024 * 1024, 1)


@timed("builder_test.rebuilt")
def rebuilt(bank, sources):
    schema = bank.registry.schema
    builder = Builder({}, sources).schema(schema.mapping())
    for entry in bank.manifest["graphs"]:
        truth = Facts.load(bank.truth(entry["index"]))
        stated = truth.stated(bank.registry, bank.splits, bank.manifest["observers"])
        builder.graph(truth.name, **stated["graph"])
        for kind, keys in stated["nodes"].items():
            observed = stated["observations"][kind]
            for key in keys:
                rows = [at for at, value in enumerate(observed["key"]) if value == key]
                assert rows
                for at in rows:
                    builder.node(
                        kind, key, observed["since"][at], observed["until"][at], graph=truth.name,
                        known=observed["known"][at], replaced=observed["replaced"][at],
                        seen=observed["observers"][at], seq=observed["seq"][at],
                        **{name: observed[VALUE + name][at] for name in schema.nodes[kind]},
                    )
        for table, rows in stated["edges"].items():
            kinds = {key: kind for kind, keys in stated["nodes"].items() for key in keys}
            for at in range(len(rows["source"])):
                source, target = rows["source"][at], rows["target"][at]
                builder.edge(
                    table, (kinds[source], source), (kinds[target], target), rows["since"][at], rows["until"][at],
                    graph=truth.name, known=rows["known"][at], replaced=rows["replaced"][at],
                    seen=rows["observers"][at], seq=rows["seq"][at], present=rows["present"][at],
                    **{name: rows[VALUE + name][at] for name in schema.edges[table]["columns"]},
                )
        tasks = stated["tasks"]
        for at, key in enumerate(tasks["key"]):
            kind = tasks["kind"][at]
            aim = schema.tasks["kinds"][kind]["target"]
            rows = [row for row in bank.rows if row["graph"] == truth.name and row[schema.tasks["key"]] == key]
            target = TARGETS.named(aim["kind"]).read(rows) if rows else None
            known = rows[0]["known"] if rows and "known" in rows[0] else None
            builder.task(
                kind, key, tasks["time"][at],
                {name: tasks["references"][name][at][0][0] for name in schema.tasks["kinds"][kind]["references"]},
                tasks["split"][at], target, known, graph=truth.name, weight=tasks["weight"][at],
                grouping=tasks["grouping"][at], signal=np.asarray(truth.task_signals)[at, :, 0].tolist(),
                signal_observed=np.asarray(truth.task_signals)[at, :, 1].astype(bool).tolist(),
                **{name: tasks[VALUE + name][at] for name in schema.tasks["kinds"][kind]["columns"]},
            )
    return builder


def test_calls_reproduce_every_truth_array_and_target_of_the_ground_fixture(grounded, tmp_path):
    import support

    original = Bank(grounded["query"] / "bank")
    builder = rebuilt(original, tmp_path)
    reference = canonical_records(builder.records())
    build(records.stage(tmp_path, "canonical", reference), tmp_path / "expected", 1, ("train",))
    expected = Bank(tmp_path / "expected")
    build_bank(builder, tmp_path / "calls", 1, ("train",), plan=PLAN)
    made = Bank(tmp_path / "calls")
    assert made.rows == original.rows
    for entry in original.manifest["graphs"]:
        before, after = Facts.load(original.truth(entry["index"])), Facts.load(made.truth(entry["index"]))
        actual = after.stated(made.registry, made.splits, made.manifest["observers"])
        prior = before.stated(original.registry, original.splits, original.manifest["observers"])
        prior["tasks"] = support.normal(prior["tasks"], original.registry.schema)
        assert support.differing(actual, prior, original.registry.schema) == []
        wanted = Facts.load(expected.truth(entry["index"]))
        assert wanted.keys == after.keys
        for name in STORED:
            np.testing.assert_equal(np.asarray(after.arrays[name]), np.asarray(wanted.arrays[name]), err_msg=name)


@timed("builder_test.schema")
def schema(timeless=False):
    return dict(
        version=VERSION, time=dict(model="none" if timeless else "clock", units=[] if timeless else ["tick"], ties=None),
        observers=False, columns=dict(a=dict(type="number", encoder="value"), b=dict(type="number", encoder="value")),
        graph=dict(columns=[]), nodes=dict(node=dict(columns=["a", "b"], holds=None)), sets={},
        edges=dict(link=synthetic.edge("node", "node", [])),
        tasks=dict(key="ticket", grouping=None, signals=0, references=dict(at=dict(sort="node", kinds=["node"])),
                   kinds=dict(level=synthetic.kind(["at"], [], dict(kind="number", series="level", about=None,
                                                                  utility=synthetic.valued(["level"])), None))),
    )


def test_changes_end_per_column_and_events_hold_for_one_instant(tmp_path):
    builder = Builder({}, tmp_path).schema(schema()).graph("one", "tick")
    builder.node("node", "n", 0, known=None, a=1)
    builder.change("node", "n", 0, known=None, b=2)
    builder.change("node", "n", 3, known=None, b=None)
    builder.event("link", ("node", "n"), ("node", "n"), 2, known=None)
    for at in range(1, 5):
        builder.task("level", at, at, dict(at="n"), "train" if at < 3 else "validation", float(at), None)
    build_bank(builder, tmp_path / "calls", 1, ("train",), plan=PLAN)
    held = Bank(tmp_path / "calls")
    truth = Facts.load(held.truth(0))
    values = truth.stated(held.registry, held.splits, held.manifest["observers"])
    assert truth.fact_since.tolist() == [2] and truth.fact_until.tolist() == [3]
    rows = values["observations"]["node"]
    changed = [at for at, columns in enumerate(rows["fact:columns"]) if columns == ["b"]]
    assert [rows["until"][at] for at in changed] == [3, None]
    assert [rows["column:b"][at] for at in changed] == [2, None]


def test_timeless_calls_take_no_times_and_invalid_local_fields_stop_before_staging(tmp_path):
    builder = Builder({}, tmp_path).schema(schema(True)).graph("one", None)
    builder.node(kind="node", key="n", a=1, b=None)
    builder.task("level", 1, references=dict(at="n"), split="train", target=1.0, known=None)
    builder.task("level", 2, references=dict(at="n"), split="train", target=2.0, known=None)
    build(builder.stage(tmp_path, "timeless", PLAN), tmp_path / "bank", 1, ("train",))
    with pytest.raises(ValueError, match="no times"):
        builder.node("node", "n", 3, a=2)
    with pytest.raises(ValueError, match="unknown"):
        builder.node("node", "n", typo=2)
    with pytest.raises(ValueError, match="undeclared node"):
        builder.edge("link", ("node", "n"), ("node", "missing"))
    with pytest.raises(ValueError, match="distinct integer key"):
        builder.task("level", 1, references=dict(at="n"), split="train", target=3, known=None)
    assert len(builder.records()["graphs"][0]["tasks"]["key"]) == 2


def test_explicit_feature_values_preserve_method_and_record_metadata_names(tmp_path):
    mapping = schema()
    node_columns = ["graph", "key", "known", "since", "values", "plain"]
    edge_columns = ["graph", "source", "target", "key", "values"]
    task_columns = ["at", "target", "key", "known", "values"]
    mapping["columns"] = {name: dict(type="number", encoder="value")
                          for name in sorted(set(node_columns + edge_columns + task_columns))}
    mapping["nodes"]["node"]["columns"] = node_columns
    mapping["edges"]["link"]["columns"] = edge_columns
    mapping["tasks"]["kinds"]["level"]["columns"] = task_columns
    builder = Builder({}, tmp_path).schema(mapping).graph("one", "tick")
    node_values = {name: float(at + 10) for at, name in enumerate(node_columns)}
    edge_values = {name: float(at + 20) for at, name in enumerate(edge_columns)}
    task_values = {name: float(at + 30) for at, name in enumerate(task_columns)}
    builder.node("node", "n", 0, known=None, values=node_values)
    builder.change("node", "n", 3, known=None, values=dict(graph=99.0))
    builder.edge("link", ("node", "n"), ("node", "n"), 0, 2, known=None, values=edge_values)
    builder.event("link", ("node", "n"), ("node", "n"), 2, known=None, values=edge_values)
    for at in range(1, 5):
        builder.task("level", at, at, dict(at="n"), "train" if at < 3 else "validation",
                     float(at), None, values=task_values)
    build_bank(builder, tmp_path / "reserved", 1, ("train",), plan=PLAN)
    bank = Bank(tmp_path / "reserved")
    truth = Facts.load(bank.truth(0))
    found = truth.stated(bank.registry, bank.splits, bank.manifest["observers"])
    assert truth.keys == ["n"] and truth.task_key.tolist() == [1, 2, 3, 4]
    assert truth.task_time.tolist() == [1, 2, 3, 4] and truth.target_y.tolist() == [1, 2, 3, 4]
    assert found["observations"]["node"]["since"] == [0, 3]
    for name, value in node_values.items():
        assert found["observations"]["node"][VALUE + name][0] == value
    for name, value in edge_values.items():
        assert found["edges"]["link"][VALUE + name] == [value, value]
    for name, value in task_values.items():
        assert found["tasks"][VALUE + name] == [value] * 4
    with pytest.raises(ValueError, match="columns twice: plain"):
        builder.node("node", "n", 4, known=None, values=dict(plain=1), plain=2)


def test_timeless_graph_attributes_can_use_the_unit_parameter_name(tmp_path):
    mapping = schema(True)
    mapping["columns"]["unit"] = dict(type="number", encoder="value")
    mapping["graph"]["columns"] = ["unit"]
    builder = Builder({}, tmp_path).schema(mapping).graph("one", None, values=dict(unit=7))
    builder.node("node", "n", a=1, b=2)
    for key in (1, 2):
        builder.task("level", key, references=dict(at="n"), split="train", target=float(key), known=None)
    build_bank(builder, tmp_path / "timeless-unit", 1, ("train",), plan=PLAN)
    bank = Bank(tmp_path / "timeless-unit")
    truth = Facts.load(bank.truth(0))
    assert truth.unit is None
    assert truth.stated(bank.registry, bank.splits, bank.manifest["observers"])["graph"] == dict(unit=None, values=dict(unit=7))


def test_changes_keep_independent_observer_histories_and_canonicalize_scopes(tmp_path):
    mapping = schema()
    mapping["observers"] = True
    mapping["time"]["ties"] = "sequence"
    builder = Builder({}, tmp_path).schema(mapping).graph("one", "tick")
    builder.change("node", "n", 0, known=None, seen=["alice"], seq=0, a=1)
    builder.change("node", "n", 0, known=None, seen=["bob"], seq=1, a=2)
    builder.change("node", "n", 2, known=None, seen=["bob"], seq=1, a=3)
    builder.change("node", "n", 0, known=None, seen=["bob", "alice"], seq=2, b=4)
    builder.change("node", "n", 3, known=None, seen=["alice", "bob"], seq=2, b=5)
    for key in (1, 2):
        builder.task("level", key, key, dict(at="n"), "train", float(key), None, observer="alice")
    assert len(builder.records()["graphs"][0]["values"]) == 3
    build_bank(builder, tmp_path / "observers", 1, ("train",), plan=PLAN)
    bank = Bank(tmp_path / "observers")
    truth = Facts.load(bank.truth(0))
    found = truth.stated(bank.registry, bank.splits, bank.manifest["observers"])["observations"]["node"]
    rows = list(zip(found["observers"], found["since"], found["until"], found[VALUE + "a"], found[VALUE + "b"]))
    assert sorted(rows) == sorted([(["alice"], 0, None, 1, None), (["bob"], 0, 2, 2, None),
                                  (["alice", "bob"], 0, 3, None, 4), (["bob"], 2, None, 3, None),
                                  (["alice", "bob"], 3, None, None, 5)])
    with pytest.raises(ValueError, match="observer names"):
        builder.change("node", "n", 4, known=None, seen=["alice", "alice"], a=6)
