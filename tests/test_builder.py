import numpy as np
import pytest

from graphspec import Builder
from graphspec.facts.stored import STORED, Facts
from graphspec.infra.timing import timed
from graphspec.schema.schema import VALUE, VERSION
from graphspec.schema.targets import TARGETS
from graphspec.sources import synthetic
from tuning import build_bank
from tuning.bank.bank import Bank
from tuning.bank.build import build

pytestmark = pytest.mark.xdist_group("fixture")


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
    original = Bank(grounded["query"] / "bank")
    builder = rebuilt(original, tmp_path)
    build_bank(builder, tmp_path / "calls", 1, ("train",))
    made = Bank(tmp_path / "calls")
    assert made.rows == original.rows
    for entry in original.manifest["graphs"]:
        before, after = Facts.load(original.truth(entry["index"])), Facts.load(made.truth(entry["index"]))
        assert before.keys == after.keys
        for name in STORED:
            np.testing.assert_equal(np.asarray(after.arrays[name]), np.asarray(before.arrays[name]), err_msg=name)


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
    build_bank(builder, tmp_path / "calls", 1, ("train",))
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
    build(builder.stage(tmp_path, "timeless"), tmp_path / "bank", 1, ("train",))
    with pytest.raises(ValueError, match="no times"):
        builder.node("node", "n", 3, a=2)
    with pytest.raises(ValueError, match="unknown"):
        builder.node("node", "n", typo=2)
    with pytest.raises(ValueError, match="undeclared node"):
        builder.edge("link", ("node", "n"), ("node", "missing"))
    with pytest.raises(ValueError, match="distinct integer key"):
        builder.task("level", 1, references=dict(at="n"), split="train", target=3, known=None)
    assert len(builder.records()["graphs"][0]["tasks"]["key"]) == 2
