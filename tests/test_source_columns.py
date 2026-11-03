import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphspec import TYPES, Builder, BuildPlan, inspect, source
from graphspec.schema.columns import convert
from graphspec.schema.schema import Category, Flag, Number

pytestmark = pytest.mark.builder
PLAN = BuildPlan(2 << 20, 1)


def builder(tmp_path, rows, columns):
    path = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return Builder(dict(records=source(path.name, "parquet", columns)), tmp_path)


def test_scalar_types_are_converted_and_core_bank_types_are_preserved(tmp_path):
    held = builder(tmp_path, [dict(id=12, amount="2.5", level="high", label=3, active="Y", words="free text")],
                   dict(id="identifier", amount="number", level=dict(type="ordered", values=["low", "high"]),
                        label=dict(type="category", values=["3"]), active=dict(type="flag", true=["Y"], false=["N"]),
                        words="text"))
    table = pa.concat_tables(held.read("records", PLAN))
    assert table.to_pylist() == [dict(id="12", amount=2.5, level=1.0, label="3", active=True, words="free text")]
    assert table["amount"].type == table["level"].type == pa.float64()
    assert isinstance(TYPES.named("number"), Number)
    assert isinstance(TYPES.named("flag"), Flag)
    assert isinstance(TYPES.named("category"), Category)
    assert TYPES.named("number").written == pa.float32()


@pytest.mark.parametrize("reader,delimiter,unit", [("csv", ",", "minute"), ("delimited", "|", "turn")])
def test_declared_delimited_scalars_preserve_script_counts_flags_nulls_and_lexemes(tmp_path, reader, delimiter, unit):
    from graphspec.build.readers import batches

    path = tmp_path / "records.csv"
    rows = [("key", "moment", "active", "mapped", "words", "amount"),
            ("0007", "12", "true", "Y", "00012", "0"),
            ("0008", "9007199254740993", "0", "N", "TRUE", "2.5")]
    path.write_text("\n".join(delimiter.join(row) for row in rows) + "\n", encoding="utf-8")
    parameters = {} if reader == "csv" else dict(delimiter=delimiter, names=None)
    columns = dict(key="identifier", moment=dict(type="time", unit=unit), active="flag",
                   mapped=dict(type="flag", true=["Y"], false=["N"]), words="text")
    columns["*"] = dict(type="number", nulls=[0], missing="omit")
    declaration = source(path.name, reader, columns, **parameters)
    raw = pa.concat_tables(batches(declaration, tmp_path, PLAN))
    assert raw["moment"].to_pylist() == ["12", "9007199254740993"]
    assert raw["active"].to_pylist() == ["true", "0"]
    assert raw["amount"].to_pylist() == ["0", "2.5"]
    held = Builder(dict(records=declaration), tmp_path)
    assert pa.concat_tables(held.read("records", PLAN)).to_pylist() == [
        dict(key="0007", moment=12, active=True, mapped=True, words="00012", amount=None),
        dict(key="0008", moment=9007199254740993, active=False, mapped=False, words="TRUE", amount=2.5),
    ]
    columns["*"] = dict(type="number", nulls=[0])
    held.declarations["records"] = source(path.name, reader, columns, **parameters)
    with pytest.raises(ValueError, match="column amount.*row 1.*without.*missing rule"):
        list(held.read("records", PLAN))


@pytest.mark.parametrize("kind,value,message", [
    (dict(type="time", unit="day"), "2020-01-01", "requires format"),
    (dict(type="time", unit="turn"), "1.5", "whole int64"),
    (dict(type="time", unit="turn"), "9223372036854775808", "whole int64"),
    ("flag", "yes", "explicit true and false"),
])
def test_delimited_scalar_adaptation_refuses_unstated_interpretations(tmp_path, kind, value, message):
    path = tmp_path / "records.csv"
    path.write_text("value\n" + value + "\n", encoding="utf-8")
    held = Builder(dict(records=source(path.name, "csv", dict(value=kind))), tmp_path)
    with pytest.raises(ValueError, match="column value row 1.*" + message):
        list(held.read("records", PLAN))


@pytest.mark.parametrize("statement,values,message", [
    ("number", [True], "row 1.*number"),
    ("number", ["oops"], "row 1.*number"),
    ("number", [float("inf")], "row 1.*finite"),
    ("flag", ["true"], "row 1.*declared"),
    (dict(type="flag", true="yes"), ["yes"], "both true and false"),
    (dict(type="flag", true=[1], false=[1]), [1], "overlap"),
    (dict(type="ordered", values=["a", "a"]), ["a"], "distinct"),
    (dict(type="ordered", values=["a", "b"]), ["c"], "row 1.*outside"),
    ("ordered", ["a"], "requires values"),
    (dict(type="category", values=["a"]), ["b"], "row 1.*outside"),
    ("text", [2], "row 1.*text"),
    (dict(type="number", imaginary=True), [2], "does not take imaginary"),
])
def test_types_refuse_undeclared_interpretations(statement, values, message):
    with pytest.raises(ValueError, match=message):
        convert(pa.array(values), statement)


def test_wildcard_types_apply_only_to_unnamed_columns(tmp_path):
    held = builder(tmp_path, [dict(key=2, a="3.5", b="4")], {"key": "identifier", "*": "number"})
    assert pa.concat_tables(held.read("records", PLAN)).to_pylist() == [dict(key="2", a=3.5, b=4.0)]
    held.declarations["records"] = source("records.parquet", "parquet", {"key": None, "*": "number"})
    with pytest.raises(ValueError, match="source records.*column key.*no stated type"):
        list(held.read("records", PLAN))


@pytest.mark.parametrize("columns", [{}, {"amount": None}, {"amount": {}}, {"*": None}])
def test_unstated_type_refuses_typed_read_but_inspect_proposes_without_mutation(tmp_path, columns):
    held = builder(tmp_path, [dict(amount=2.5)], columns)
    report = inspect(held, plan=PLAN)
    assert "amount" in report and "proposed" in report and "number" in report
    assert held.declarations["records"].columns == columns
    with pytest.raises(ValueError, match="column amount.*no stated type"):
        list(held.read("records", PLAN))


def test_inspect_reads_all_batches_and_reports_missing_without_a_rule(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("amount,day,kind\n2,2020-01-01,a\n,2020-01-02,b\n", encoding="utf-8")
    held = Builder(dict(records=source(path.name, "csv", {"*": None})), tmp_path)
    report = inspect(held, plan=PLAN)
    assert '"missing": 1' in report and '"unit": "day"' in report and '"rows": 2' in report
    with pytest.raises(ValueError, match="BuildPlan"):
        inspect(held)


def test_missing_values_refuse_typed_read_with_source_column_and_row(tmp_path):
    held = builder(tmp_path, [dict(amount=1), dict(amount=None)], dict(amount="number"))
    with pytest.raises(ValueError, match="source records.*records.parquet.*column amount.*row 2.*without.*rule"):
        list(held.read("records", PLAN))


@pytest.mark.parametrize("columns,rows,message", [
    (dict(amount=None), [dict(amount=1)], "column amount.*no stated type"),
    (dict(amount="number"), [dict(amount=None)], "column amount.*row 1.*without.*rule"),
])
def test_source_type_errors_stop_stage_before_graph_output(tmp_path, columns, rows, message):
    held = builder(tmp_path, rows, columns)
    with pytest.raises(ValueError, match=message):
        held.stage(tmp_path / "output", "refused", PLAN)
    assert not (tmp_path / "output").exists()


def test_typed_errors_keep_global_rows_across_reader_batches(tmp_path, monkeypatch):
    from graphspec import READERS

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})

    def split(place, columns, parameters):
        yield pa.table(dict(amount=["1", "2"]))
        yield pa.table(dict(amount=["bad"]))

    READERS.register("split_rows", split)
    held = Builder(dict(records=source("input", "split_rows", dict(amount="number"))), tmp_path)
    with pytest.raises(ValueError, match="column amount row 3"):
        list(held.read("records", PLAN))


def test_custom_type_is_called_with_raw_values_and_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})
    calls = []

    def scaled(values, parameters):
        calls.append((values.type, dict(parameters)))
        return pa.array([float(value) * parameters["scale"] for value in values.to_pylist()], pa.float64())

    scaled.version, scaled.gives = 1, "number"
    TYPES.register("scaled", scaled)
    held = builder(tmp_path, [dict(amount="3")], dict(amount=dict(type="scaled", scale=2)))
    assert pa.concat_tables(held.read("records", PLAN))["amount"].to_pylist() == [6.0]
    assert calls == [(pa.string(), dict(scale=2))]


def test_custom_types_preserve_input_nulls_for_the_missing_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})

    def implicit_fill(values, parameters):
        return pa.array([0.0 if value is None else float(value) for value in values.to_pylist()])

    implicit_fill.version, implicit_fill.gives = 1, "number"
    TYPES.register("implicit_fill", implicit_fill)
    held = builder(tmp_path, [dict(amount="1"), dict(amount=None)],
                   dict(amount=dict(type="implicit_fill", missing="omit")))
    with pytest.raises(ValueError, match="column amount row 2.*cannot fill.*missing rule"):
        list(held.read("records", PLAN))


def test_custom_type_produced_nan_requires_an_explicit_missing_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})

    def missing_number(values, parameters):
        return pa.array([float("nan")] * len(values))

    missing_number.version, missing_number.gives = 1, "number"
    TYPES.register("missing_number", missing_number)
    held = builder(tmp_path, [dict(amount=1)], dict(amount="missing_number"))
    with pytest.raises(ValueError, match="column amount.*row 1.*without.*missing rule"):
        list(held.read("records", PLAN))
    held.declarations["records"] = source("records.parquet", "parquet", dict(amount=dict(type="missing_number", missing="omit")))
    assert pa.concat_tables(held.read("records", PLAN))["amount"].to_pylist() == [None]


@pytest.mark.parametrize("name,width", [("point", 1), ("point", 4), ("vector", 0)])
def test_custom_types_obey_shipped_fixed_width_shapes(name, width, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {key: TYPES.named(key) for key in TYPES.names()})

    def malformed(values, parameters):
        return pa.array([[1.0] * width] * len(values), type=pa.list_(pa.float64(), width))

    malformed.version, malformed.gives = 1, name
    TYPES.register("malformed", malformed)
    with pytest.raises(ValueError, match="Arrow representation"):
        convert(pa.array([1]), "malformed")


def test_zero_width_native_lists_preserve_empty_and_missing_rows():
    from graphspec.schema.missing import normalize

    values = pa.array([[], None], type=pa.list_(pa.float64(), 0))
    result = convert(normalize(values, {}), dict(type="list", of="number"))
    assert result.to_pylist() == [[], None]
    assert result.type == pa.list_(pa.float64())


@pytest.mark.parametrize("result,gives,message", [
    ([2], "number", "Arrow array"),
    (pa.array([2.0, 3.0]), "number", "input length"),
    (pa.array([2]), "number", "Arrow representation"),
    (pa.array([2.0]), "unknown", "shipped column type"),
])
def test_custom_type_return_contract_is_checked(result, gives, message, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})

    def invalid(values, parameters):
        return result

    invalid.version, invalid.gives = 1, gives
    TYPES.register("invalid", invalid)
    with pytest.raises(ValueError, match=message):
        convert(pa.array([1]), "invalid")


def test_declared_sources_share_finest_time_unit(tmp_path):
    (tmp_path / "coarse.json").write_text(json.dumps([dict(day=0)]), encoding="utf-8")
    (tmp_path / "fine.json").write_text(json.dumps([dict(second=1)]), encoding="utf-8")
    held = Builder(dict(coarse=source("coarse.json", "json", dict(day=dict(type="time", unit="day", at="end"))),
                        fine=source("fine.json", "json", dict(second=dict(type="time", unit="second")))), tmp_path)
    assert pa.concat_tables(held.read("coarse", PLAN))["day"].to_pylist() == [86399]
    held.declarations["coarse"] = source("coarse.json", "json", dict(day=dict(type="time", unit="day")))
    with pytest.raises(ValueError, match="coarse.day.*requires at.*fine.second"):
        list(held.read("coarse", PLAN))


def test_custom_temporal_types_negotiate_units_through_nested_list_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})
    calls = []

    def calendar_day(values, parameters):
        calls.append(dict(parameters))
        return TYPES.named("time")(values, parameters)

    def calendar_list(values, parameters):
        return TYPES.named("list")(values, parameters)

    calendar_day.version, calendar_day.gives = 1, "time"
    calendar_list.version, calendar_list.gives = 1, "list"
    TYPES.register("calendar_day", calendar_day)
    TYPES.register("calendar_list", calendar_list)
    day = dict(type="calendar_day", unit="day", at="end")
    columns = dict(moment=day, moments=dict(type="list", of=dict(type="calendar_list", of=day)),
                   second=dict(type="time", unit="second"))
    held = builder(tmp_path, [dict(moment=0, moments=[[0, 1]], second=1)], columns)
    assert pa.concat_tables(held.read("records", PLAN)).to_pylist() == [
        dict(moment=86399, moments=[[86399, 172799]], second=1)
    ]
    assert calls == [dict(unit="day", at="end"), dict(unit="day", at="end")]
    held.declarations["records"] = source("records.parquet", "parquet", dict(moment=dict(type="calendar_day")))
    with pytest.raises(ValueError, match="records.moment.*requires unit"):
        list(held.read("records", PLAN))


def test_temporal_wildcards_invoke_each_reader_once_per_operation(tmp_path, monkeypatch):
    from graphspec import READERS

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    calls, closed = [], []

    def once(place, columns, parameters):
        calls.append(place.name)
        try:
            yield pa.table(dict(moment=[0]))
            yield pa.table(dict(moment=[1]))
        finally:
            closed.append(place.name)

    READERS.register("once", once)
    declarations = dict(coarse=source("coarse", "once", {"*": dict(type="time", unit="day", at="end")}),
                        fine=source("fine", "once", {"*": dict(type="time", unit="second")}))
    held = Builder(declarations, tmp_path)
    held.check_sources(PLAN)
    assert calls == closed == ["coarse", "fine"]
    calls.clear()
    closed.clear()
    assert pa.concat_tables(held.read(declarations["coarse"], PLAN))["moment"].to_pylist() == [86399, 172799]
    assert calls == closed == ["coarse", "fine"]
    calls.clear()
    closed.clear()
    iterator = held.read("coarse", PLAN)
    assert next(iterator)["moment"].to_pylist() == [86399]
    iterator.close()
    assert calls == closed == ["coarse", "fine"]


def test_unused_temporal_wildcard_does_not_set_the_graph_unit(tmp_path):
    held = builder(tmp_path, [dict(moment=2)],
                   {"moment": "number", "*": dict(type="time", unit="day", at="end")})
    assert pa.concat_tables(held.read("records", PLAN))["moment"].to_pylist() == [2.0]


def test_temporal_lookahead_closes_every_reader_after_unit_failure(tmp_path, monkeypatch):
    from graphspec import READERS

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    closed = []

    def records(place, columns, parameters):
        try:
            yield pa.table(dict(moment=[0]))
        finally:
            closed.append(place.name)

    READERS.register("unit_failure", records)
    held = Builder(dict(coarse=source("coarse", "unit_failure", {"*": dict(type="time", unit="day")}),
                        fine=source("fine", "unit_failure", {"*": dict(type="time", unit="second")})), tmp_path)
    with pytest.raises(ValueError, match="coarse.moment.*requires at.*fine.moment"):
        held.check_sources(PLAN)
    assert closed == ["coarse", "fine"]


def test_temporal_lookahead_refuses_unbounded_retained_batches(tmp_path, monkeypatch):
    from graphspec import READERS

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    calls, closed = [], []

    def records(place, columns, parameters):
        calls.append(place.name)
        try:
            yield pa.table(dict(moment=list(range(128))))
        finally:
            closed.append(place.name)

    READERS.register("bounded_lookahead", records)
    held = Builder({name: source(name, "bounded_lookahead", {"*": dict(type="time", unit="day")})
                    for name in ("first", "second", "third")}, tmp_path)
    with pytest.raises(ValueError, match="lookahead.*BuildPlan"):
        held.check_sources(BuildPlan(8192, 1))
    assert calls == closed == ["first", "second"]
