import pyarrow as pa
import pyarrow.compute as pc
import pytest

from graphspec import FUNCTIONS, Builder, BuildPlan, evaluate, source
from graphspec.build.expression_types import EXTENDED_TIME, NEVER_TIME, Function, Kind

pytestmark = pytest.mark.builder
PLAN = BuildPlan(1 << 20, 1)
TABLE = pa.table(dict(x=pa.array([-5, 5, None], pa.float64()), flag=[False, True, None]))
COLUMNS = dict(x="number", flag="flag")


@pytest.mark.parametrize("expression,expected", [
    ("x + 2", [-3.0, 7.0, None]),
    ("x - 2", [-7.0, 3.0, None]),
    ("x * 2", [-10.0, 10.0, None]),
    ("x / 2", [-2.5, 2.5, None]),
    ("x // 2", [-3.0, 2.0, None]),
    ("x % 2", [1.0, 1.0, None]),
    ("x ** 2", [25.0, 25.0, None]),
    ("-x", [5.0, -5.0, None]),
    ("+x", [-5.0, 5.0, None]),
    ("x > 0", [False, True, None]),
    ("-6 <= x < 5", [True, False, None]),
    ("x in [-5, 0]", [True, False, None]),
    ("x not in [-5, 0]", [False, True, None]),
    ("x is missing", [False, False, True]),
    ("x is not missing", [True, True, False]),
    ("x + missing", [None, None, None]),
])
def test_expression_operators_equal_literal_rows(expression, expected):
    assert evaluate(expression, TABLE, COLUMNS, plan=PLAN).values.to_pylist() == expected


def test_logic_propagates_missing_even_when_one_operand_decides_boolean_result():
    table = pa.table(dict(a=[False, True, None], b=[None, False, True]))
    columns = dict(a="flag", b="flag")
    assert evaluate("a and b", table, columns, plan=PLAN).values.to_pylist() == [None, False, None]
    assert evaluate("a or b", table, columns, plan=PLAN).values.to_pylist() == [None, True, None]
    assert evaluate("not a", table, columns, plan=PLAN).values.to_pylist() == [True, False, None]


@pytest.mark.parametrize("expression,invalid,message", [
    ("round(x, missing)", "round(missing, 1.5)", "digits"),
    ("bucket(x, missing)", "bucket(missing, 0)", "width"),
    ("bin(x, missing)", "bin(missing, [0, 0])", "increasing"),
])
def test_missing_numeric_parameters_propagate_without_accepting_malformed_present_literals(expression, invalid, message):
    assert evaluate(expression, TABLE, COLUMNS, plan=PLAN).values.to_pylist() == [None, None, None]
    assert evaluate(expression, TABLE.slice(0, 0), COLUMNS, plan=PLAN).values.to_pylist() == []
    with pytest.raises(ValueError, match=message):
        evaluate(invalid, TABLE, COLUMNS, plan=PLAN)


def test_conditionals_evaluate_only_selected_rows_but_validate_both_branches():
    table = pa.table(dict(label=["yes", "absent"], keep=[True, False], x=pa.array([-1, 4], pa.float64())))
    columns = dict(label="category", keep="flag", x="number")
    result = evaluate("{'yes': 10}[label] if keep else sqrt(x)", table, columns, plan=PLAN)
    assert result.values.to_pylist() == [10.0, 2.0]
    with pytest.raises(ValueError, match="unknown column absent"):
        evaluate("x if True else absent", table, columns, plan=PLAN)
    with pytest.raises(ValueError, match="types"):
        evaluate("x if True else keep", table, columns, plan=PLAN)
    assert evaluate("x if missing else 1", table, columns, plan=PLAN).values.to_pylist() == [None, None]
    assert evaluate("sum(x) if count(x) > 1 else missing", table, columns, plan=PLAN, whole=True).values.as_py() == 3.0


def test_conditional_aggregate_reduces_original_whole_scope():
    table = pa.table(dict(x=pa.array([1, 2, 3], pa.float64()), keep=[True, False, True]))
    result = evaluate("sum(x) if keep else 0", table, dict(x="number", keep="flag"), plan=PLAN, whole=True)
    assert result.values.to_pylist() == [6.0, 0.0, 6.0]


def test_reserved_and_nonidentifier_column_names_are_explicit():
    table = pa.table({"order date": [1.0, 2.0], "missing": [3.0, 4.0], "year": [5.0, 6.0]})
    columns = {name: "number" for name in table.column_names}
    result = evaluate("column('order date') + column('missing') + year", table, columns, plan=PLAN)
    assert result.values.to_pylist() == [9.0, 12.0]
    assert evaluate("missing", table, columns, plan=PLAN).values.as_py() is None


@pytest.mark.parametrize("expression", [
    "__import__('os')", "x.__class__", "x.sum()", "[x for x in [1]]", "(lambda: 1)()",
    "f'{x}'", "(x := 1)", "[1][0]", "sum(*[x])", "x is None", "None",
])
def test_expressions_cannot_execute_python_or_use_undeclared_syntax(expression):
    with pytest.raises(ValueError, match="declaration blocked field key"):
        evaluate(expression, TABLE, COLUMNS, plan=PLAN, declaration="blocked", field="key")


@pytest.mark.parametrize("expression", ["{1: 'one', 1.0: 'again'}[x]", "{1: 'one', True: 'yes'}[x]",
                                        "{True: 'yes'}[x]"])
def test_lookup_keys_keep_declared_types_and_never_overwrite_duplicates(expression):
    with pytest.raises(ValueError):
        evaluate(expression, TABLE, COLUMNS, plan=PLAN)
    assert evaluate("{5: 'yes', -5: 'no'}[x]", TABLE, COLUMNS, plan=PLAN).values.to_pylist() == ["no", "yes", None]


def test_custom_functions_receive_arrays_and_literal_arguments_once(monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    calls = []

    def add_amount(values, amount):
        calls.append((values.to_pylist(), amount))
        return pc.add(values, amount)

    add_amount.version, add_amount.reads, add_amount.gives, add_amount.aggregate = 1, ("number", "number"), "number", False
    FUNCTIONS.register("add_amount", add_amount)
    assert evaluate("add_amount(x, 2)", TABLE, COLUMNS, plan=PLAN).values.to_pylist() == [-3.0, 7.0, None]
    assert calls == [([-5.0, 5.0], 2)]


@pytest.mark.parametrize("form", ["callback", "wrapper", "reserved"])
def test_custom_functions_cannot_opt_out_of_missing_propagation(monkeypatch, form):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})

    def fill(values):
        return pc.fill_null(values, 99.0)

    fill.version, fill.reads, fill.gives, fill.aggregate, fill.nulls = 1, ("number",), "number", False, "read"
    entry = Function(fill, ("number",), "number", nulls="read") if form == "wrapper" else fill
    name = "coalesce" if form == "reserved" else "custom_fill"
    if form == "reserved":
        FUNCTIONS.held[name] = entry
    else:
        FUNCTIONS.register(name, entry)
    assert evaluate(name + "(x)", TABLE, COLUMNS, plan=PLAN).values.to_pylist() == [-5.0, 5.0, None]


@pytest.mark.parametrize("kind,output", [
    (Kind("number"), pa.array([1.0, float("nan")])),
    (Kind("number"), pa.array([1.0, float("inf")])),
    (Kind("duration", unit="day"), pa.array([1.0, float("inf")])),
    (Kind("vector", width=2), pa.array([[1.0, 2.0], [1.0, float("inf")]], type=pa.list_(pa.float64(), 2))),
])
def test_function_outputs_enforce_finite_values_with_original_row_context(monkeypatch, kind, output):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})

    def invalid(values):
        return output

    FUNCTIONS.register("nonfinite", Function(invalid, ("number",), kind))
    with pytest.raises(ValueError, match="declaration facts field score.*row 12.*finite"):
        evaluate("nonfinite(x)", TABLE, COLUMNS, plan=PLAN, declaration="facts", field="score", offset=10)


def test_custom_function_can_return_missing_as_an_arrow_null(monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})

    def nullable(values):
        return pa.array([None] * len(values), pa.float64())

    FUNCTIONS.register("nullable", Function(nullable, ("number",), "number"))
    assert evaluate("nullable(x)", TABLE, COLUMNS, plan=PLAN).values.to_pylist() == [None, None, None]


def test_custom_function_does_not_read_a_missing_literal_parameter(monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    calls = []

    def shifted(values, amount):
        calls.append(True)
        return pc.add(values, float(amount))

    shifted.version, shifted.reads, shifted.gives, shifted.aggregate = 1, ("number", "number"), "number", False
    FUNCTIONS.register("shifted", shifted)
    assert evaluate("shifted(x, missing)", TABLE, COLUMNS, plan=PLAN).values.to_pylist() == [None, None, None]
    assert evaluate("shifted(x, missing)", TABLE.slice(0, 0), COLUMNS, plan=PLAN).values.to_pylist() == []
    assert calls == []


@pytest.mark.parametrize("form", ["scalar", "length", "dtype", "metadata"])
def test_custom_function_contract_errors_name_declaration_and_field(monkeypatch, form):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})

    def invalid(values):
        if form == "scalar":
            return pa.scalar(1.0)
        return pa.array([1] if form == "length" else [1, 2], pa.int64())

    invalid.version, invalid.reads, invalid.gives, invalid.aggregate = 1, ("number",), "number", False
    if form == "metadata":
        invalid.aggregate = "false"
    FUNCTIONS.register("invalid", invalid)
    with pytest.raises(ValueError, match="declaration facts field target"):
        evaluate("invalid(x)", TABLE, COLUMNS, plan=PLAN, declaration="facts", field="target")


def test_nested_scalar_calls_keep_explicit_window_scope():
    later = pa.table(dict(value=["2", "bad", "4"], keep=[True, False, True]))
    result = evaluate("sum(number(later.value), where=later.keep)", pa.table(dict(key=["a"])), dict(key="identifier"),
                      plan=PLAN, whole=True, namespaces={"later": (later, dict(value="text", keep="flag"))})
    assert result.values.as_py() == 6.0
    with pytest.raises(ValueError, match="only inside an aggregate"):
        evaluate("later.value", pa.table(dict(key=["a"])), dict(key="identifier"), plan=PLAN,
                 namespaces={"later": (later, dict(value="text", keep="flag"))})


def test_unrelated_main_through_array_does_not_constrain_window_reductions():
    result = evaluate("sum(later.x)", pa.table(dict(key=["a", "b"])), dict(key="identifier"),
                      plan=PLAN, whole=True, through=pa.array([5, 5]),
                      namespaces={"later": (pa.table(dict(x=[1.0, 2.0, 3.0])), dict(x="number"))})
    assert result.values.as_py() == 6.0


def test_filtered_aggregate_errors_keep_original_source_rows_and_offset():
    table = pa.table(dict(tau=[0.0, 1.0, 2.0, 2.5], y=[1.0] * 4, keep=[False, True, False, True]))
    with pytest.raises(ValueError, match="declaration facts field target.*row 21"):
        evaluate("points(tau, 'signal', y, where=keep)", table, dict(tau="number", y="number", keep="flag"),
                 plan=PLAN, whole=True, offset=17, declaration="facts", field="target")


def test_reader_output_time_metadata_uses_graph_units_once(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("day,second,lag,amount\n0,1,2,3\n", encoding="utf-8")
    columns = dict(day=dict(type="time", unit="day", at="end"), second=dict(type="time", unit="second"),
                   lag=dict(type="duration", unit="minute"), amount="number")
    builder = Builder(dict(records=source(path.name, "csv", columns)), tmp_path)
    table = pa.concat_tables(builder.read("records", PLAN))
    assert table["day"].to_pylist() == [86399]
    result = evaluate("day + lag", table, builder.declarations["records"].columns, plan=PLAN, unit="second")
    assert result.type == "time" and result.unit == "second" and result.values.to_pylist() == [86519]
    result = evaluate("day - second", table, columns, plan=PLAN, unit="second")
    assert result.type == "duration" and result.unit == "second" and result.values.to_pylist() == [86398.0]
    assert evaluate("day + missing", table, columns, plan=PLAN, unit="second").values.to_pylist() == [None]


def test_native_timestamp_columns_cannot_be_misread_as_graph_unit_counts():
    table = pa.table(dict(moment=pa.array([1000], pa.int64()).cast(pa.timestamp("ms"))))
    with pytest.raises(ValueError, match="already-typed.*graph unit"):
        evaluate("moment", table, dict(moment=dict(type="time", unit="second")), plan=PLAN, unit="second")


def test_custom_callbacks_receive_finite_extended_times_as_int64_and_refuse_never(monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    seen = []

    def identity(values):
        seen.append(values.type)
        return values

    identity.version, identity.reads, identity.gives, identity.aggregate = 1, ("time",), "time", False
    FUNCTIONS.register("finite_time", identity)
    table = pa.table(dict(moment=[(1 << 63) - 1, -(1 << 63)], keep=[True, True]))
    columns = dict(moment=dict(type="time", unit="turn"), keep="flag")
    result = evaluate("finite_time(moment if keep else never)", table, columns, plan=PLAN, unit="turn")
    assert result.values.type == pa.int64() and result.values.to_pylist() == table["moment"].to_pylist()
    assert seen == [pa.int64()]
    with pytest.raises(ValueError, match="row 1.*custom function.*never.*int64"):
        evaluate("finite_time(never)", table, columns, plan=PLAN, unit="turn")
    result = evaluate("never", table, columns, plan=PLAN, unit="turn")
    assert result.values.type == EXTENDED_TIME and result.values.as_py() == NEVER_TIME
    assert evaluate("text(never)", table, columns, plan=PLAN, unit="turn").values.to_pylist() == ["never", "never"]
    with pytest.raises(ValueError, match="never.*finite number"):
        evaluate("number(never)", table, columns, plan=PLAN, unit="turn")


def test_custom_callbacks_preserve_list_rows_and_reject_nested_never(monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    seen = []

    def lengths(values):
        seen.append(values.type)
        return pc.list_value_length(values).cast(pa.float64())

    lengths.version, lengths.reads, lengths.gives, lengths.aggregate = 1, ("list",), "number", False
    FUNCTIONS.register("time_lengths", lengths)
    table = pa.table(dict(moment=[(1 << 63) - 1, -(1 << 63)]))
    columns = dict(moment=dict(type="time", unit="turn"))
    result = evaluate("time_lengths(list(coalesce(moment, never)))", table, columns, plan=PLAN, whole=True, unit="turn")
    assert result.values.as_py() == 2.0 and seen == [pa.list_(pa.int64())]
    counts = pa.table(dict(x=pa.array([1.0, 2.0])))
    assert evaluate("sqrt(sum(x) + 1)", counts, dict(x="number"), plan=PLAN, whole=True).values.as_py() == 2.0
    assert evaluate("x + sum(x)", counts, dict(x="number"), plan=PLAN, whole=True).values.to_pylist() == [4.0, 5.0]
    with pytest.raises(ValueError, match="row 1.*list member.*never.*int64"):
        evaluate("time_lengths(list(never))", table, columns, plan=PLAN, whole=True, unit="turn")
    with pytest.raises(ValueError, match="row 1.*list member.*never.*int64"):
        evaluate("time_lengths([[never]])", table, columns, plan=PLAN, unit="turn")


def test_never_composes_with_literal_lists_and_lookups():
    table = pa.table(dict(moment=[(1 << 63) - 1], key=["done"]))
    columns = dict(moment=dict(type="time", unit="turn"), key="category")
    assert evaluate("moment in [never]", table, columns, plan=PLAN, unit="turn").values.to_pylist() == [False]
    result = evaluate("{'done': never}[key]", table, columns, plan=PLAN, unit="turn")
    assert result.values.type == EXTENDED_TIME and result.values.to_pylist() == [NEVER_TIME]


def test_explicit_plan_bounds_work_and_unresolved_types_are_errors():
    with pytest.raises(ValueError, match="budget"):
        evaluate("x + 1", pa.table(dict(x=[1.0] * 100)), dict(x="number"), plan=BuildPlan(4096, 1))
    with pytest.raises(ValueError, match="no stated type"):
        evaluate("x", TABLE, dict(flag="flag"), plan=PLAN)
    with pytest.raises(ValueError, match="explicit graph unit"):
        evaluate("time('2020-01-01')", TABLE, COLUMNS, plan=PLAN)
