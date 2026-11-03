import json
from dataclasses import FrozenInstanceError

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from graphspec import TABLES, Builder, BuildPlan, Frame, align, compute, expand, select, source, table, union
from graphspec.build.expression_types import EXTENDED_TIME, NEVER_TIME, Kind
from graphspec.build.streams import capacity
from graphspec.build.table_types import ACTIVE

pytestmark = pytest.mark.extended
PLAN = BuildPlan(4 << 20, 1)


def records(tmp_path, name="records", rows=None, columns=None, through=None):
    rows = [dict(key="a", x=1), dict(key="b", x=3)] if rows is None else rows
    columns = dict(key="identifier", x="number") if columns is None else columns
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(rows), encoding="utf-8")
    return source(path.name, "json", columns, through=through)


def test_select_compute_materialize_whole_inputs_preserve_marks_and_reset_row_positions(tmp_path):
    original = records(tmp_path, rows=[dict(key="a", x=-1, keep=False), dict(key="b", x=2, keep=True),
                                       dict(key="c", x=None, keep=False), dict(key="d", x=4, keep=True)],
                       columns=dict(key="identifier", x=dict(type="number", missing="omit", outcome="never"), keep="flag"))
    chosen = select(original, where="keep and x > 0")
    scored = compute(chosen, columns=dict(fraction="x / sum(x)", position="row()"), outcomes=dict(fraction="never"))
    builder = Builder(dict(original=original, chosen=chosen, scored=scored), tmp_path)
    result = builder.frame("scored", PLAN)
    assert result.table.to_pylist() == [
        dict(key="b", x=2.0, keep=True, fraction=1 / 3, position=0.0),
        dict(key="d", x=4.0, keep=True, fraction=2 / 3, position=1.0),
    ]
    assert result.provenance["original"].to_pylist() == [[1], [3]]
    assert result.provenance["chosen"].to_pylist() == [[0], [1]]
    assert result.provenance["scored"].to_pylist() == [[0], [1]]
    assert result.outcomes["fraction"] == (("original", "never"), ("scored", "never"))
    assert ("original", "x") in result.lineage["fraction"]
    assert pa.concat_tables(builder.read("scored", PLAN)).equals(result.table)


def test_compute_refuses_collisions_and_does_not_read_new_sibling_columns(tmp_path):
    original = records(tmp_path)
    for columns, message in ((dict(x="x + 1"), "collide.*x"),
                             (dict(a="x + 1", b="a + 1"), "columns.b.*unknown column a")):
        changed = compute(original, columns=columns)
        builder = Builder(dict(original=original, changed=changed), tmp_path)
        with pytest.raises(ValueError, match=message):
            list(builder.read("changed", PLAN))


def test_align_and_union_follow_declared_table_and_column_order(tmp_path):
    first = records(tmp_path, "first")
    other = records(tmp_path, "other", rows=[dict(label="low"), dict(label="high")], columns=dict(label="category"))
    side = align([first, other])
    second = records(tmp_path, "second", rows=[dict(x=7, key="c")], columns=dict(x="number", key="identifier"))
    appended = union([first, second])
    builder = Builder(dict(first=first, other=other, side=side, second=second, appended=appended), tmp_path)
    aligned = builder.frame("side", PLAN)
    assert aligned.table.column_names == ["key", "x", "label"]
    assert aligned.table.to_pylist() == [dict(key="a", x=1.0, label="low"), dict(key="b", x=3.0, label="high")]
    assert aligned.provenance["first"].to_pylist() == aligned.provenance["other"].to_pylist() == [[0], [1]]
    result = builder.frame("appended", PLAN)
    assert result.table.to_pylist() == [dict(key="a", x=1.0), dict(key="b", x=3.0), dict(key="c", x=7.0)]
    assert result.kinds["key"] == Kind("identifier")
    assert result.provenance["first"].to_pylist() == [[0], [1], []]
    assert result.provenance["second"].to_pylist() == [[], [], [0]]


def test_align_and_union_refuse_incompatible_inputs(tmp_path):
    first = records(tmp_path, "first")
    short = records(tmp_path, "short", rows=[dict(other=1)], columns=dict(other="number"))
    labels = records(tmp_path, "labels", rows=[dict(key="c", x="known")], columns=dict(key="identifier", x="category"))
    for declaration, message in ((align([first, short]), "equal input row counts"),
                                 (align([first, first]), "collide"), (union([first, short]), "same columns"),
                                 (union([first, labels]), "incompatible semantic types")):
        with pytest.raises(ValueError, match=message):
            list(Builder(dict(result=declaration), tmp_path).read("result", PLAN))


def test_union_preserves_exact_extended_time_metadata(tmp_path):
    first = records(tmp_path, rows=[dict(moment=(1 << 63) - 1)], columns=dict(moment=dict(type="time", unit="turn")))
    finite = compute(first, columns=dict(ends="moment"))
    unbounded = compute(first, columns=dict(ends="never"))
    combined = union([finite, unbounded])
    result = Builder(dict(records=first, finite=finite, unbounded=unbounded, combined=combined), tmp_path).frame("combined", PLAN)
    assert result.kinds["ends"] == Kind("time", unit="turn", arrow=EXTENDED_TIME)
    assert result.table["ends"].to_pylist() == [(1 << 63) - 1, NEVER_TIME]
    assert result.table["moment"].type == pa.int64()


def test_declarations_are_lazy_immutable_and_scripts_export_table_values(tmp_path):
    original = source("not-read.json", "json", dict(x="number"))
    columns = dict(next="x + 1")
    declaration = compute(original, columns=columns)
    columns["next"] = "x + 100"
    declaration.parameters["columns"]["next"] = "changed"
    assert declaration.parameters == dict(columns=dict(next="x + 1"))
    with pytest.raises(FrozenInstanceError):
        declaration.operation = "select"
    records(tmp_path)
    script = tmp_path / "tables.py"
    script.write_text("from graphspec import source, compute\n"
                      "records = source('records.json', 'json', dict(key='identifier', x='number'))\n"
                      "scored = compute(records, columns=dict(next='x + 1'))\n", encoding="utf-8")
    builder = Builder(script, tmp_path)
    assert pa.concat_tables(builder.read("scored", PLAN))["next"].to_pylist() == [2.0, 4.0]


def test_source_and_derived_reads_share_static_expanded_time_units_without_reparsing(tmp_path):
    original = records(tmp_path, rows=[dict(clock=1, events=[dict(at=1)])],
                       columns=dict(clock=dict(type="time", unit="day", at="start"), events=dict(type="list", of="nested")))
    events = expand(original, column="events", columns=dict(at=dict(type="time", unit="second")))
    builder = Builder(dict(original=original, events=events), tmp_path)
    assert pa.concat_tables(builder.read("original", PLAN))["clock"].to_pylist() == [86400]
    result = builder.frame("events", PLAN)
    assert result.unit == "second"
    assert result.table.to_pylist() == [dict(clock=86400, at=1, place=0.0)]


def test_table_completeness_is_scalar_and_carried_from_original_source(tmp_path):
    original = records(tmp_path, through="sum(x)")
    empty = select(original, where="False")
    result = Builder(dict(original=original, empty=empty), tmp_path).frame("empty", PLAN)
    assert result.table.num_rows == 0 and result.through == "sum(x)" and result.complete.as_py() == 4
    varying = source(original.place, "json", original.columns, through="x")
    with pytest.raises(ValueError, match="one scalar.*row-varying"):
        Builder(dict(varying=varying), tmp_path).frame("varying", PLAN)
    with pytest.raises(ValueError, match="one scalar"):
        Frame(pa.table(dict(x=[1.0])), dict(x=Kind("number")), complete=pa.array([1]))


def test_frame_metadata_is_deeply_read_only(tmp_path):
    original = records(tmp_path, columns=dict(key="identifier", x=dict(type="number", outcome="never")))
    result = Builder(dict(original=original), tmp_path).frame("original", PLAN)
    for mapping in (result.kinds, result.outcomes, result.lineage, result.provenance):
        with pytest.raises(TypeError):
            mapping["injected"] = None
    marks = {"x": [["input", "never"]]}
    frame = Frame(pa.table(dict(x=[1.0])), dict(x=Kind("number")), outcomes=marks)
    marks["x"][0][1] = "changed"
    assert frame.outcomes["x"] == (("input", "never"),)


def test_custom_table_callback_receives_iterators_parameters_and_opaque_input_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(TABLES, "held", {name: TABLES.named(name) for name in TABLES.names()})
    called = []

    def reversed_rows(tables, parameters):
        called.append(dict(parameters))
        for part in tables[0]:
            yield pc.take(part, pa.array(list(reversed(range(part.num_rows))), pa.int64()))

    reversed_rows.version = 1
    TABLES.register("reversed_rows", reversed_rows)
    original = records(tmp_path)
    changed = table([original], name="reversed_rows", columns=original.columns, label="explicit")
    result = Builder(dict(original=original, changed=changed), tmp_path).frame("changed", PLAN)
    assert result.table.to_pylist() == [dict(key="b", x=3.0), dict(key="a", x=1.0)]
    assert called == [dict(label="explicit")]
    assert set(result.provenance) == {"changed"}
    assert result.lineage["x"] == (("changed", ("custom", "reversed_rows", ("original",))),)


def test_custom_table_output_times_are_already_in_graph_units(tmp_path, monkeypatch):
    monkeypatch.setattr(TABLES, "held", {name: TABLES.named(name) for name in TABLES.names()})

    def passed(tables, parameters):
        yield from tables[0]

    passed.version = 1
    TABLES.register("passed", passed)
    columns = dict(day=dict(type="time", unit="day", at="end"), second=dict(type="time", unit="second"))
    original = records(tmp_path, rows=[dict(day=0, second=1)], columns=columns)
    changed = table([original], name="passed", columns=columns)
    result = Builder(dict(original=original, changed=changed), tmp_path).frame("changed", PLAN)
    assert result.table.to_pylist() == [dict(day=86399, second=1)]
    assert result.kinds["day"] == Kind("time", unit="second")


@pytest.mark.parametrize("declared,valid,invalid,message", [
    (dict(type="category", values=["known"]), "known", "unknown", "outside values"),
    (dict(type="ordered", values=["known"]), 0.0, 1.0, "outside the declared positions"),
])
def test_custom_table_outputs_honor_declared_domains_without_missing_rules(tmp_path, monkeypatch, declared, valid, invalid, message):
    monkeypatch.setattr(TABLES, "held", {name: TABLES.named(name) for name in TABLES.names()})

    def constrained(tables, parameters):
        yield pa.table(dict(x=[valid]))
        yield pa.table(dict(x=[invalid]))

    constrained.version = 1
    TABLES.register("constrained", constrained)
    original = records(tmp_path)
    changed = table([original], name="constrained", columns=dict(x=declared))
    with pytest.raises(ValueError, match="table changed: column x row 2: .*" + message):
        Builder(dict(original=original, changed=changed), tmp_path).frame("changed", PLAN)


@pytest.mark.parametrize("failure,message", [("missing", "missing rule"), ("large", "budget"), ("schema", "one typed schema")])
def test_custom_table_outputs_close_on_validation_and_budget_errors(tmp_path, monkeypatch, failure, message):
    monkeypatch.setattr(TABLES, "held", {name: TABLES.named(name) for name in TABLES.names()})
    closed = []

    def broken(tables, parameters):
        try:
            if failure == "missing":
                yield pa.table(dict(x=pa.array([None], pa.float64())))
            elif failure == "large":
                yield pa.table(dict(x=["x" * 4096] * 32))
            else:
                yield pa.table(dict(x=[1.0]))
                yield pa.table(dict(x=[2.0], extra=[3.0]))
        finally:
            closed.append(capacity())

    broken.version = 1
    TABLES.register("broken", broken)
    original = records(tmp_path)
    columns = {"*": "number"} if failure == "schema" else dict(x="text" if failure == "large" else "number")
    changed = table([original], name="broken", columns=columns)
    with pytest.raises(ValueError, match=message):
        list(Builder(dict(original=original, changed=changed), tmp_path).read("changed", BuildPlan(65536, 1)))
    assert closed == [65536]
    with pytest.raises(ValueError, match="explicit BuildPlan"):
        capacity()
    assert ACTIVE.get() is None


def test_interleaved_source_and_table_reads_do_not_leak_working_contexts(tmp_path):
    original = records(tmp_path)
    changed = compute(original, columns=dict(next="x + 1"))
    builder = Builder(dict(original=original, changed=changed), tmp_path)
    raw, derived = builder.read("original", PLAN), builder.read("changed", PLAN)
    assert next(raw)["x"].to_pylist() == [1.0, 3.0]
    assert next(derived)["next"].to_pylist() == [2.0, 4.0]
    assert ACTIVE.get() is None
    with pytest.raises(ValueError, match="explicit BuildPlan"):
        capacity()
    raw.close()
    derived.close()


def test_materializing_operations_refuse_retained_input_and_result_above_budget(tmp_path):
    original = records(tmp_path, rows=[dict(x=1.0)] * 1000, columns=dict(x="number"))
    changed = compute(original, columns=dict(next="x + 1"))
    with pytest.raises(ValueError, match="budget"):
        list(Builder(dict(original=original, changed=changed), tmp_path).read("changed", BuildPlan(16384, 1)))


def test_source_completeness_evaluation_reserves_the_fresh_source_frame(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("x\n" + "1.0\n" * 100, encoding="utf-8")
    original = source(path.name, "csv", dict(x="number"), through="sum(x)")
    with pytest.raises(ValueError, match="expression table working storage"):
        Builder(dict(records=original), tmp_path).frame("records", BuildPlan(13500, 1))
    assert Builder(dict(records=original), tmp_path).frame("records", BuildPlan(14250, 1)).complete.as_py() == 100
