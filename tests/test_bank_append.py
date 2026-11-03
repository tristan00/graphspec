import copy
import importlib

import pyarrow as pa
import pytest

from graphspec import Builder, BuildPlan
from graphspec.budgets import GraphCapacityError
from graphspec.descriptions.describe import recorded
from graphspec.facts.stored import Facts
from graphspec.infra.files import read_json, sha256_file
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION
from graphspec.sources import synthetic
from tuning import append_bank
from tuning.bank.bank import Bank
from tuning.bank.build import build

append_module = importlib.import_module("tuning.bank.append")
PLAN = BuildPlan(64 * 1024 * 1024, 1)
pytestmark = pytest.mark.extended


@timed("bank_append_test.schema")
def schema():
    return dict(version=VERSION, time=dict(model="clocks", units=["tick"], ties=None), observers=False,
                columns=dict(a=dict(type="number", encoder="value"), b=dict(type="number", encoder="value"),
                             category=dict(type="category", encoder="onehot")), graph=dict(columns=[]),
                nodes=dict(node=dict(columns=["a", "b", "category"], holds=None)), sets={},
                edges=dict(link=synthetic.edge("node", "node", [], relation=dict(column="predicate")),
                           fixed=synthetic.edge("node", "node", [])),
                tasks=dict(key="ticket", grouping=None, signals=0, references=dict(at=dict(sort="node", kinds=["node"])),
                           kinds=dict(level=synthetic.kind(["at"], [], dict(kind="number", series="level", about=None,
                                                                          utility=synthetic.valued(["level"])), None))))


@timed("bank_append_test.source")
def source(root, complete):
    builder = Builder({}, root).schema(schema()).graph("one", "tick")
    builder.change("node", "n", 0, known=None, replaced=4 if complete else None, a=1)
    builder.change("node", "n", 0, known=None, b=2, category="z")
    builder.edge(("link", "z"), ("node", "n"), ("node", "n"), 0, 4 if complete else None, known=None)
    for key in (3, 1, 2):
        builder.task("level", key, 0 if key == 3 else key, dict(at="n"), "train" if key < 3 else "unlabelled",
                     None if key == 3 and not complete else float(key), 6 if key == 3 and complete else None)
    builder.task("level", 6, 2, dict(at="n"), None, None, None)
    if complete:
        builder.node("node", "n", 0, known=4, a=9)
        builder.change("node", "n", 4, known=None, b=7, category="a")
        builder.node("node", "m", 4, known=None, a=3, b=4, category="a")
        builder.edge(("link", "a"), ("node", "n"), ("node", "m"), 4, known=None)
        builder.task("level", 4, 5, dict(at="m"), "validation", 4.0, None)
        builder.graph("unlabelled", "tick")
        builder.node("node", "u", 0, graph="unlabelled", known=None, a=1, b=2, category="a")
    return builder


@timed("bank_append_test.decisions")
def decisions(root, complete):
    mapping = synthetic.decision_schema()
    mapping["columns"] = dict(load=mapping["columns"]["load"])
    mapping["nodes"] = {name: mapping["nodes"][name] for name in ("agent", "mark")}
    mapping["edges"] = {}
    mapping["tasks"]["signals"] = 0
    mapping["tasks"]["references"] = dict(agents=mapping["tasks"]["references"]["agents"])
    mapping["tasks"]["kinds"] = dict(turn=mapping["tasks"]["kinds"]["turn"])
    mapping["tasks"]["kinds"]["turn"]["target"]["actions"] = dict(
        put=dict(roles=dict(made=dict(sort="new", kinds=["mark"], count=[1, 1])), parameters=[]))
    builder = Builder({}, root).schema(mapping).graph("decision", "tick")
    builder.node("agent", "a", 0, known=None, load=1)
    builder.node("agent", "b", 0, known=None, load=2)
    builder.node("mark", "existing", 0, known=None)
    for key in range(1, 5 if complete else 4):
        actions = [] if key < 4 else [synthetic.acting("put", "a", dict(made=["fresh"]), {})]
        option = synthetic.offered("wait" if key < 4 else "put", actions, "set", "recorded")
        option["chosen"] = True
        option["outcome"] = None if key == 3 and not complete else dict(gain=float(key), cost=float(key * 2))
        actor = "b" if key == 3 else "a"
        following = dict(a=2) if key == 1 else dict(a=4) if key == 2 and complete else {}
        target = dict(options=[option], concurrent=[], next=following, ended=False, discount=0.9)
        known = 6 if key == 3 and complete else None if key == 3 else key + 1
        builder.task("turn", key, key, dict(agents=[actor]), "train" if key < 3 else "validation", target, known)
    return builder


@timed("bank_append_test.semantic")
def semantic(bank, index):
    truth = Facts.load(bank.truth(index))
    result = truth.stated(bank.registry, bank.splits, bank.manifest["observers"])
    result["targets"]["row"] = [bank.rows[row] for row in result["targets"]["row"]]
    for rows in list(result["observations"].values()) + list(result["edges"].values()):
        columns = list(rows)
        ordered = sorted(range(len(rows[columns[0]])), key=lambda at: repr([rows[column][at] for column in columns]))
        for column in columns:
            rows[column] = [rows[column][at] for at in ordered]
    return result


@timed("bank_append_test.labels")
def labels(bank, indices=None):
    rows = bank.rows if indices is None else [bank.rows[index] for index in indices]
    return sorted(rows, key=lambda row: (row["graph"], row[bank.registry.schema.tasks["key"]], row["series"], repr(row)))


def test_later_facts_completion_vocabularies_and_new_graph_equal_whole_without_reparsing_prefix(tmp_path, monkeypatch):
    task = source(tmp_path, False).stage(tmp_path, "prefix", PLAN)
    root = tmp_path / "bank"
    build(task, root, 1, ("train",), plan=PLAN)
    before = Bank(root)
    old_semantic = semantic(before, 0)
    original = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    category = copy.deepcopy(before.registry.encoding.dictionaries)
    relation = before.registry.code("z")
    complete = source(tmp_path, True)
    complete.append(task, PLAN)
    seen, original_parsed = [], append_module.parsed

    @timed("bank_append_test.delta_parsed")
    def delta_parsed(registry, splits, observers, tables, rows, series, name):
        if name == "one":
            observed = [max(since, since if known is None else known) for table in tables["observations"].values()
                        for since, known in zip(table.column("since").to_pylist(), table.column("known").to_pylist())]
            facts = [max(since, since if known is None else known) for table in tables["edges"].values()
                     for since, known in zip(table.column("since").to_pylist(), table.column("known").to_pylist())]
            assert all(value > 3 for value in observed)
            assert all(value > 3 for value in facts)
            seen.append((len(observed), len(facts)))
        return original_parsed(registry, splits, observers, tables, rows, series, name)

    monkeypatch.setattr(append_module, "parsed", delta_parsed)
    append_module.append(task, root, 1)
    assert seen == [(4, 1)]
    assert all(sha256_file(path) == digest for path, digest in original.items() if path.name != "manifest.json")
    made = Bank(root)
    assert before.source != made.source and before.cache != made.cache
    assert before.truth(0) == root / "truth" / "graph_0000"
    assert semantic(before, 0) == old_semantic
    assert made.manifest["generation"] == 1 and len(made.manifest["graphs"]) == 2
    assert made.registry.encoding.dictionaries["category"] == category["category"] + ["a"]
    assert made.registry.code("z") == relation
    task_counts = made.manifest["graphs"][0]["inspection"]["tasks"]["level"]
    assert task_counts["splits"]["unlabelled"] == 1 and task_counts["without_split"] == 1
    assert made.rows[:len(before.rows)] == before.rows
    whole_task = complete.stage(tmp_path, "whole", PLAN)
    build(whole_task, tmp_path / "whole-bank", 1, ("train",), plan=PLAN)
    whole = Bank(tmp_path / "whole-bank")
    assert made.rows != whole.rows and labels(made) == labels(whole)
    assert made.split["series"] == whole.split["series"]
    for split in made.splits:
        assert labels(made, made.split["indices"][split]) == labels(whole, whole.split["indices"][split])
    assert made.manifest["source_fingerprint"] == whole.manifest["source_fingerprint"]
    for index in (0, 1):
        assert semantic(made, index) == semantic(whole, index)
        truth = Facts.load(made.truth(index))
        assert all(made.rows[int(row)]["ticket"] == int(truth.task_key[int(task)])
                   for row, task in zip(truth.target_row, truth.target_task))
    assert sorted(Facts.load(made.truth(0)).target_y.tolist()) == [1, 2, 3, 4]
    described = recorded(root, made.manifest)
    expected = recorded(whole.directory, whole.manifest)
    assert described["graphs"] == expected["graphs"]
    immutable_generation = {path: sha256_file(path) for path in (root / "generations" / "1").rglob("*") if path.is_file()}
    complete.change("node", "n", 7, graph="one", known=None, b=9)
    complete.task("level", 5, 8, dict(at="n"), "validation", 5.0, None, graph="one")
    complete.append(task, PLAN)
    append_module.append(task, root, 1)
    twice = Bank(root)
    assert twice.manifest["generation"] == 2 and twice.rows[:4] == made.rows
    assert all(sha256_file(path) == digest for path, digest in immutable_generation.items())
    assert sorted(Facts.load(twice.truth(0)).target_y.tolist()) == [1, 2, 3, 4, 5]
    assert sorted(Facts.load(made.truth(0)).target_y.tolist()) == [1, 2, 3, 4]


def test_failed_generation_keeps_bank_readable_and_pending_journal_retries(tmp_path, monkeypatch):
    task = source(tmp_path, False).stage(tmp_path, "prefix", PLAN)
    root = tmp_path / "bank"
    build(task, root, 1, ("train",), plan=PLAN)
    before = read_json(root / "manifest.json")
    source(tmp_path, True).append(task, PLAN)
    original_write = append_module.write_json

    @timed("bank_append_test.failed_write")
    def failed_write(path, value):
        if path == root / "manifest.json":
            raise OSError("publication refused")
        return original_write(path, value)

    monkeypatch.setattr(append_module, "write_json", failed_write)
    with pytest.raises(OSError, match="publication refused"):
        append_bank(task, root, 1)
    assert read_json(root / "manifest.json") == before and len(Bank(root).rows) == 2
    assert not (root / "generations" / "1").exists()
    monkeypatch.setattr(append_module, "write_json", original_write)
    append_bank(task, root, 1)
    assert len(Bank(root).rows) == 4
    with pytest.raises(ValueError, match="source fingerprint"):
        append_module.append(task, root, 1)


def test_bank_budget_stops_before_source_materialization(tmp_path, monkeypatch):
    task = source(tmp_path, False).stage(tmp_path, "prefix", PLAN)
    build_module = importlib.import_module("tuning.bank.build")

    @timed("bank_append_test.forbidden_open")
    def forbidden_open(*args):
        raise AssertionError("source was materialized")

    monkeypatch.setattr(build_module, "open_source", forbidden_open)
    with pytest.raises(GraphCapacityError, match="conservative working bytes"):
        build(task, tmp_path / "bank", 1, ("train",), plan=BuildPlan(4096, 1))


def test_decision_continuity_completion_and_prior_creation_context_equal_whole(tmp_path, monkeypatch):
    task = decisions(tmp_path, False).stage(tmp_path, "prefix", PLAN)
    root = tmp_path / "bank"
    build(task, root, 1, ("train",), plan=PLAN)
    complete = decisions(tmp_path, True)
    complete.append(task, PLAN)
    original_parse, seen = append_module.parse, []

    @timed("bank_append_test.prior_context")
    def prior_context(registry, tables, parts, first_seen=None, decision_links=True):
        assert not len(parts["seen"]["entity"])
        assert first_seen[(registry.kind("mark"), "existing")] == 0
        roles = tables["decisions"]["role"]
        wrong = roles.set_column(roles.schema.get_field_index("key"), "key", pa.array(["existing"] * roles.num_rows))
        bad = dict(tables, decisions=dict(tables["decisions"], role=wrong))
        with pytest.raises(ValueError, match="observed at or before"):
            original_parse(registry, bad, parts, first_seen, decision_links)
        seen.append(True)
        return original_parse(registry, tables, parts, first_seen, decision_links)

    monkeypatch.setattr(append_module, "parse", prior_context)
    append_module.append(task, root, 1)
    made = Bank(root)
    whole_task = complete.stage(tmp_path, "whole", PLAN)
    build(whole_task, tmp_path / "whole-bank", 1, ("train",), plan=PLAN)
    whole = Bank(tmp_path / "whole-bank")
    assert seen == [True] and made.rows == whole.rows
    assert semantic(made, 0) == semantic(whole, 0)
    truth = Facts.load(made.truth(0))
    assert truth.next_task.tolist() == [0, 1] and truth.next_other.tolist() == [1, 3]
