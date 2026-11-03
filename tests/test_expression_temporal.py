from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pyarrow as pa
import pytest

from graphspec import BuildPlan
from graphspec.build.expression_temporal import arithmetic, span_kind
from graphspec.build.expression_types import (
    EXTENDED_TIME,
    FUNCTIONS,
    NEVER_TIME,
    UNSTATED,
    Call,
    Kind,
    Value,
    calling,
)
from graphspec.build.streams import reading

pytestmark = pytest.mark.builder
PLAN = BuildPlan(2 << 20, 1)


def invoke(name, *arguments, rows=3, unit="second", through=UNSTATED):
    with reading(PLAN), calling(Call(unit, rows, (), {}, through=through)):
        return FUNCTIONS.named(name)(*arguments)


def operate(op, left, left_kind, right, right_kind, unit, rows=1):
    left = pa.array(left) if isinstance(left, list) else left
    right = pa.array(right) if isinstance(right, list) else right
    with reading(PLAN), calling(Call(unit, rows, (), {})):
        return arithmetic(op, Value(left, left_kind), Value(right, right_kind), unit)


@pytest.mark.parametrize("name,arguments,expected", [
    ("matches", (r"\d+",), [True, True, None]),
    ("extract", (r"(\d+)",), ["12", "34", None]),
    ("replace", (r"\d+", "#"), [" AbC# ", "xy#", None]),
    ("lower", (), [" abc12 ", "xy34", None]),
    ("upper", (), [" ABC12 ", "XY34", None]),
    ("strip", (), ["AbC12", "xy34", None]),
    ("length", (), [7.0, 4.0, None]),
    ("left", (2,), [" A", "xy", None]),
    ("right", (2,), ["2 ", "34", None]),
    ("split", ("b",), [[" A", "C12 "], ["xy34"], None]),
    ("concat", ("!", "?"), [" AbC12 !?", "xy34!?", None]),
])
def test_every_text_function_matches_literal_values(name, arguments, expected):
    result = invoke(name, pa.array([" AbC12 ", "xy34", None]), *arguments)
    assert result.to_pylist() == expected


def test_text_functions_use_unicode_and_dynamic_patterns_without_filling_nulls():
    assert invoke("upper", pa.array(["straße", "é", None])).to_pylist() == ["STRASSE", "É", None]
    assert invoke("length", pa.array(["é", "猫", None])).to_pylist() == [1.0, 1.0, None]
    assert invoke("matches", pa.array(["abc", "123", None]), pa.array(["^a", "[a-z]", "["])).to_pylist() == [True, False, None]
    assert invoke("concat", pa.array(["a", None, "b"]), pa.array(["1", "2", None])).to_pylist() == ["a1", None, None]
    assert invoke("right", pa.array(["abc", "xyz", None]), 0).to_pylist() == ["", "", None]


def test_extract_has_explicit_capture_and_no_match_semantics():
    values = pa.array(["a12", "abc", None])
    assert invoke("extract", values, r"\d+").to_pylist() == ["12", None, None]
    assert invoke("extract", values, r"a(\d+)").to_pylist() == ["12", None, None]
    with pytest.raises(ValueError, match="zero or one capture"):
        invoke("extract", pa.array([], type=pa.string()), "(a)(b)", rows=0)
    with pytest.raises(ValueError, match="invalid pattern"):
        invoke("matches", pa.array([], type=pa.string()), "[", rows=0)


@pytest.mark.parametrize("name,arguments,message", [
    ("matches", (pa.array(["ok", "[", "ok"]),), "row 2.*invalid pattern"),
    ("left", (1.5,), "nonnegative whole number"),
    ("right", (-1,), "nonnegative whole number"),
    ("split", ("",), "separator is nonempty"),
    ("replace", ("(a)", r"\2"), "invalid replacement"),
])
def test_text_parameter_errors_name_the_row(name, arguments, message):
    with pytest.raises(ValueError, match=message):
        invoke(name, pa.array(["a", "b", "c"]), *arguments)
    with pytest.raises(ValueError, match="row 1.*reads text"):
        invoke("lower", pa.array([1, 2, 3]))


@pytest.mark.parametrize("rows", [[], [None]])
def test_literal_replacement_is_validated_without_present_text(rows):
    with pytest.raises(ValueError, match="invalid replacement"):
        invoke("replace", pa.array(rows, pa.string()), "(a)", r"\2", rows=len(rows))


@pytest.mark.parametrize("name", ["left", "right"])
@pytest.mark.parametrize("rows", [[], [None]])
def test_literal_text_counts_are_validated_without_present_text(name, rows):
    for value in (-1, 1.5):
        with pytest.raises(ValueError, match="nonnegative whole number"):
            invoke(name, pa.array(rows, pa.string()), value, rows=len(rows))
    assert invoke(name, pa.array(rows, pa.string()), None, rows=len(rows)).to_pylist() == rows
    assert invoke(name, pa.array(["abc"]), None, rows=1).to_pylist() == [None]


def test_dynamic_text_count_errors_preserve_their_row():
    with pytest.raises(ValueError, match="row 2.*nonnegative whole number"):
        invoke("left", pa.array(["abc", "def", "ghi"]), pa.array([1.0, 1.5, 2.0]))


@pytest.mark.parametrize("name,expected", [
    ("year", [1970.0, 1969.0, 2024.0, None]),
    ("month", [1.0, 12.0, 2.0, None]),
    ("day", [1.0, 31.0, 29.0, None]),
    ("weekday", [3.0, 2.0, 3.0, None]),
    ("hour", [0.0, 23.0, 12.0, None]),
    ("minute", [0.0, 59.0, 30.0, None]),
])
def test_every_date_part_matches_literal_values(name, expected):
    result = invoke(name, pa.array([0, -1, 1709209800, None], pa.int64()), rows=4)
    assert result.type == pa.float64()
    assert result.to_pylist() == expected


def test_date_parts_honor_week_origin_and_calendar_count_units():
    values = pa.array([0, 1, -1], pa.int64())
    assert invoke("year", values, unit="week").to_pylist() == [1969.0, 1970.0, 1969.0]
    assert invoke("day", values, unit="week").to_pylist() == [29.0, 5.0, 22.0]
    assert invoke("weekday", values, unit="week").to_pylist() == [0.0, 0.0, 0.0]
    assert invoke("month", pa.array([0, 1, 649]), unit="month").to_pylist() == [1.0, 2.0, 2.0]
    assert invoke("year", pa.array([1969, 1970, 2024]), unit="year").to_pylist() == [1969.0, 1970.0, 2024.0]


def test_native_timestamp_dates_are_utc_and_floored_to_graph_precision():
    values = pa.array([datetime(1970, 1, 1, 1, 30, tzinfo=timezone(timedelta(hours=2)))])
    assert invoke("day", values, rows=1).to_pylist() == [31.0]
    assert invoke("hour", values, rows=1).to_pylist() == [23.0]
    assert invoke("minute", values, rows=1).to_pylist() == [30.0]
    assert invoke("hour", values, rows=1, unit="day").to_pylist() == [0.0]


@pytest.mark.parametrize("unit", [None, "turn"])
def test_dates_refuse_clockless_or_custom_units(unit):
    with pytest.raises(ValueError, match="calendar function.*unit"):
        invoke("year", pa.array([0]), rows=1, unit=unit)
    with pytest.raises(ValueError, match="calendar function.*unit"):
        invoke("time", "1970-01-01", rows=1, unit=unit)


def test_time_literal_iso_dates_offsets_and_fractional_seconds():
    values = pa.array(["1970-01-01", "1970-01-01T00:01:59", "1970-01-01 01:00:00+01:00",
                       "1969-12-31T23:59:59.999999Z", None])
    result = invoke("time", values, unit="minute", rows=5)
    assert result.type == pa.int64()
    assert result.to_pylist() == [0, 1, 0, -1, None]
    with pytest.raises(ValueError, match="row 2"):
        invoke("time", pa.array(["2024-02-29", "2025-02-29"]), rows=2)
    with pytest.raises(ValueError, match="row 1.*ISO 8601"):
        invoke("time", "01/02/2024", rows=1)


def test_interval_endpoints_preserve_open_ends_without_calendar_guessing():
    values = pa.array([[1, 5], [None, 8], [3, None], None], pa.list_(pa.int64(), 2))
    assert invoke("start", values, rows=4, unit="turn").to_pylist() == [1, None, 3, None]
    assert invoke("end", values, rows=4, unit="turn").to_pylist() == [5, 8, None, None]


def test_span_preserves_its_explicit_unit_and_fractional_count():
    values = pa.array([1.0, 2.5, None])
    assert invoke("span", values, "month").to_pylist() == [1.0, 2.5, None]
    assert span_kind((Value(values, Kind("number")), Value("month", Kind("category"))), {}) == Kind("duration", unit="month")
    with pytest.raises(ValueError, match="literal name"):
        invoke("span", values, pa.array(["day", "day", "day"]))


def test_met_uses_explicit_source_completeness_for_missing_events():
    values, deadlines = pa.array([3, 8, None, None, None]), pa.array([5, 5, 5, 8, None])
    result = invoke("met", values, deadlines, rows=5, through=pa.array([10, 10, 5, 7, 10]))
    assert result.to_pylist() == [True, False, False, None, None]
    with pytest.raises(ValueError, match="row 1.*explicit through"):
        invoke("met", None, 5, rows=1)
    assert invoke("met", None, 5, rows=1, through=None).to_pylist() == [None]


def test_time_integer_arithmetic_keeps_values_above_float_precision():
    large = (1 << 53) + 1
    kind = Kind("time", unit="second")
    assert operate("+", [large], kind, 1, Kind("number"), "second").values.to_pylist() == [large + 1]
    assert operate("-", [large], kind, 1, Kind("number"), "second").values.to_pylist() == [large - 1]
    assert operate("<", [large], kind, [large + 1], kind, "second").values.to_pylist() == [True]
    result = operate("-", [large + 1], kind, [large], kind, "second")
    assert result.kind == Kind("duration", unit="second") and result.values.to_pylist() == [1.0]
    kind = Kind("time", unit="year")
    assert operate("+", [large], kind, 1, Kind("number"), "year").values.to_pylist() == [large + 1]


def test_fixed_duration_arithmetic_is_exact_at_time_boundaries():
    minute = Kind("time", unit="minute")
    second = Kind("duration", unit="second")
    assert operate("+", [0], minute, [120.0], second, "minute").values.to_pylist() == [2]
    with pytest.raises(ValueError, match="row 2.*90.0.*not a whole.*minute"):
        operate("+", [0, 0], minute, [60.0, 90.0], second, "minute", rows=2)
    result = operate("+", [1.0], Kind("duration", unit="minute"), [30.0], second, "second")
    assert result.kind == Kind("duration", unit="minute") and result.values.to_pylist() == [1.5]
    result = operate("/", [60.0], second, [2.0], Kind("duration", unit="minute"), "second")
    assert result.kind == Kind("number") and result.values.to_pylist() == [0.5]
    assert operate("*", 3, Kind("number"), [2.5], second, "second").values.to_pylist() == [7.5]
    assert operate("/", [5.0], second, 2, Kind("number"), "second").values.to_pylist() == [2.5]


def test_calendar_duration_shifts_preserve_day_and_refuse_impossible_dates():
    day = Kind("time", unit="day")
    month = Kind("duration", unit="month")
    assert operate("+", [19751], day, [1.0], month, "day").values.to_pylist() == [19782]
    with pytest.raises(ValueError, match="row 1.*day.*range"):
        operate("+", [19753], day, [1.0], month, "day")
    with pytest.raises(ValueError, match="row 1.*day.*range"):
        operate("+", [19782], day, [1.0], Kind("duration", unit="year"), "day")
    with pytest.raises(ValueError, match="row 1.*whole number of months"):
        operate("+", [19751], day, [0.5], month, "day")
    with pytest.raises(ValueError, match="cannot be compared as fixed lengths"):
        operate("<", [1.0], month, [31.0], Kind("duration", unit="day"), "day")


def test_fractional_years_move_by_exact_whole_calendar_months():
    year = Kind("duration", unit="year")
    assert operate("+", [0], Kind("time", unit="month"), [0.5], year, "month").values.to_pylist() == [6]
    assert operate("+", [19751], Kind("time", unit="day"), [0.5], year, "day").values.to_pylist() == [19933]
    with pytest.raises(ValueError, match="row 1.*whole number of months"):
        operate("+", [19751], Kind("time", unit="day"), [0.1], year, "day")
    with pytest.raises(ValueError, match="row 1.*whole count of graph unit"):
        operate("+", [2024], Kind("time", unit="year"), [0.5], year, "year")


def test_custom_duration_units_convert_only_to_themselves():
    turn = Kind("time", unit="turn")
    assert operate("+", [4], turn, [2.0], Kind("duration", unit="turn"), "turn").values.to_pylist() == [6]
    with pytest.raises(ValueError, match="row 1.*no fixed conversion.*turn"):
        operate("+", [4], turn, [2.0], Kind("duration", unit="day"), "turn")


def test_evaluator_uses_text_type_metadata_and_typed_time_values():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(label=["abc", "xyz", None], moment=[0, 86400, None]))
    columns = dict(label="category", moment=dict(type="time", unit="day", at="end"))
    result = evaluate("matches(label, '^a')", table, columns, plan=PLAN, unit="second")
    assert result.kind == Kind("flag") and result.values.to_pylist() == [True, False, None]
    result = evaluate("day(moment)", table, columns, plan=PLAN, unit="second")
    assert result.values.to_pylist() == [1.0, 2.0, None]
    result = evaluate("day(time('1970-01-02'))", table, columns, plan=PLAN, unit="second")
    assert result.values.to_pylist() == [2.0, 2.0, 2.0]


def test_evaluator_preserves_large_time_values_and_declared_duration_units():
    from graphspec.build.expressions import evaluate

    large = (1 << 53) + 1
    table = pa.table(dict(moment=[large], wait=[120.0]))
    columns = dict(moment=dict(type="time", unit="minute"), wait=dict(type="duration", unit="second"))
    result = evaluate("moment + span(120, 'second')", table, columns, plan=PLAN, unit="minute")
    assert result.kind == Kind("time", unit="minute") and result.values.to_pylist() == [large + 2]
    result = evaluate("moment + wait", table, columns, plan=PLAN, unit="minute")
    assert result.values.to_pylist() == [large + 2]


def test_evaluator_temporal_errors_name_expression_field_and_global_row():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=[0, 0], wait=[60.0, 90.0], label=["a", "b"]))
    columns = dict(moment=dict(type="time", unit="minute"), wait=dict(type="duration", unit="second"), label="text")
    with pytest.raises(ValueError, match="records.*deadline.*row 12"):
        evaluate("moment + wait", table, columns, plan=PLAN, unit="minute", declaration="records", field="deadline", offset=10)
    with pytest.raises(ValueError, match="records.*marked.*pattern"):
        evaluate("matches(label, '[')", table, columns, plan=PLAN, unit="minute", declaration="records", field="marked")
    with pytest.raises(ValueError, match="year.*time"):
        evaluate("year(wait)", table, columns, plan=PLAN, unit="minute")


def test_ordered_columns_serve_numeric_counts_and_temporal_arithmetic():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(grade=[1.0, 2.0, 0.0], label=["abc", "xyz", "def"], moment=pa.array([4, 4, 4], pa.int64())))
    columns = dict(grade=dict(type="ordered", values=["none", "one", "two"]), label="text",
                   moment=dict(type="time", unit="day"))
    for expression, expected in (("left(label, grade)", ["a", "xy", ""]),
                                 ("right(label, grade)", ["c", "yz", ""]),
                                 ("span(grade, 'day')", [1.0, 2.0, 0.0]),
                                 ("moment + grade", [5, 6, 4]),
                                 ("grade + moment", [5, 6, 4]),
                                 ("span(2, 'day') * grade", [2.0, 4.0, 0.0])):
        assert evaluate(expression, table, columns, plan=PLAN, unit="day").values.to_pylist() == expected


@pytest.mark.parametrize("expression,expected", [
    ("moment < never", [True, True, True, None]),
    ("moment <= never", [True, True, True, None]),
    ("moment == never", [False, False, False, None]),
    ("moment != never", [True, True, True, None]),
    ("never > moment", [True, True, True, None]),
    ("never >= moment", [True, True, True, None]),
])
def test_never_orders_after_every_finite_int64_time(expression, expected):
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=pa.array([1 << 62, (1 << 62) + 1, (1 << 63) - 1, None], pa.int64())))
    result = evaluate(expression, table, dict(moment=dict(type="time", unit="second")), plan=PLAN, unit="second")
    assert result.values.to_pylist() == expected


def test_never_promotes_conditionals_and_coalesce_without_changing_finite_times():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=pa.array([1 << 62, (1 << 62) + 1, (1 << 63) - 1, None], pa.int64()),
                          choose=[False, True, False, True]))
    columns = dict(moment=dict(type="time", unit="second"), choose="flag")
    kind = Kind("time", unit="second", arrow=EXTENDED_TIME)
    result = evaluate("coalesce(moment, never)", table, columns, plan=PLAN, unit="second")
    assert result.kind == kind
    assert result.values.to_pylist() == [Decimal(1 << 62), Decimal((1 << 62) + 1), Decimal((1 << 63) - 1), Decimal(NEVER_TIME)]
    result = evaluate("never if choose else moment", table, columns, plan=PLAN, unit="second")
    assert result.kind == kind
    assert result.values.to_pylist() == [Decimal(1 << 62), Decimal(NEVER_TIME), Decimal((1 << 63) - 1), Decimal(NEVER_TIME)]
    finite = evaluate("moment", table, columns, plan=PLAN, unit="second")
    assert finite.values.type == pa.int64()


def test_never_is_preserved_by_supported_time_aggregates():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=pa.array([1 << 62, (1 << 63) - 1, None], pa.int64()), by=[2.0, 1.0, 0.0]))
    columns = dict(moment=dict(type="time", unit="second"), by="number")
    for expression, expected in (
        ("min(coalesce(moment, never))", Decimal(1 << 62)),
        ("max(coalesce(moment, never))", Decimal(NEVER_TIME)),
        ("first(coalesce(moment, never), by=by)", Decimal(NEVER_TIME)),
        ("last(coalesce(moment, never), by=by)", Decimal(1 << 62)),
        ("list(coalesce(moment, never))", [Decimal(1 << 62), Decimal((1 << 63) - 1), Decimal(NEVER_TIME)]),
        ("distinct(coalesce(moment, never))", [Decimal(1 << 62), Decimal((1 << 63) - 1), Decimal(NEVER_TIME)]),
    ):
        result = evaluate(expression, table, columns, plan=PLAN, unit="second", whole=True)
        assert result.values.as_py() == expected


def test_never_preserves_valid_shifts_and_does_not_hide_finite_overflow():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=pa.array([(1 << 63) - 1], pa.int64())))
    columns = dict(moment=dict(type="time", unit="second"))
    for expression in ("never + 1", "never - 1"):
        result = evaluate(expression, table, columns, plan=PLAN, unit="second")
        assert result.kind == Kind("time", unit="second", arrow=EXTENDED_TIME)
        assert result.values.as_py() == Decimal(NEVER_TIME)
    for expression, unit in (("never + span(2, 'minute')", "second"),
                             ("never - span(0.5, 'year')", "day"),
                             ("never + span(1, 'turn')", "turn")):
        result = evaluate(expression, table, columns, plan=PLAN, unit=unit)
        assert result.values.to_pylist() == [Decimal(NEVER_TIME)]
    result = evaluate("coalesce(moment, never) - moment", table, columns, plan=PLAN, unit="second")
    assert result.kind == Kind("duration", unit="second") and result.values.to_pylist() == [0.0]
    with pytest.raises(ValueError, match="row 1.*outside int64"):
        evaluate("coalesce(moment, never) + 1", table, columns, plan=PLAN, unit="second")


@pytest.mark.parametrize("expression,unit,message", [
    ("never + span(90, 'second')", "minute", "whole count"),
    ("never + span(0.1, 'year')", "day", "whole number of months"),
    ("never + span(0.5, 'month')", "day", "whole number of months"),
    ("never + span(1, 'month')", "turn", "calendar function"),
    ("never + span(1, 'day')", "turn", "no fixed conversion"),
])
def test_never_does_not_bypass_duration_unit_or_count_rules(expression, unit, message):
    from graphspec.build.expressions import evaluate

    with pytest.raises(ValueError, match="row 1.*" + message):
        evaluate(expression, pa.table(dict(value=[1.0])), dict(value="number"), plan=PLAN, unit=unit)


@pytest.mark.parametrize("expression", ["moment - never", "never - moment", "never - never"])
def test_never_differences_refuse_nonfinite_durations(expression):
    from graphspec.build.expressions import evaluate

    with pytest.raises(ValueError, match="row 1.*no finite duration"):
        evaluate(expression, pa.table(dict(moment=pa.array([0], pa.int64()))),
                 dict(moment=dict(type="time", unit="second")), plan=PLAN, unit="second")


def test_never_has_no_calendar_parts_and_met_uses_infinite_deadlines():
    from graphspec.build.expressions import evaluate

    table = pa.table(dict(moment=pa.array([(1 << 62) + 1, (1 << 63) - 1, None], pa.int64())))
    columns = dict(moment=dict(type="time", unit="second"))
    with pytest.raises(ValueError, match="row 1.*never.*finite calendar date"):
        evaluate("year(never)", table, columns, plan=PLAN, unit="second")
    result = evaluate("met(moment, never)", table, columns, plan=PLAN, unit="second", through=(1 << 63) - 1)
    assert result.values.to_pylist() == [True, True, None]
    result = evaluate("met(never, moment)", table, columns, plan=PLAN, unit="second")
    assert result.values.to_pylist() == [False, False, None]
    through = pa.array([Decimal(NEVER_TIME)] * 3, EXTENDED_TIME)
    result = evaluate("met(moment, never)", table, columns, plan=PLAN, unit="second", through=through)
    assert result.values.to_pylist() == [True, True, False]
