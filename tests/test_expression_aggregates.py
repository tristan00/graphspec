import pyarrow as pa
import pytest

from graphspec import BuildPlan
from graphspec.build.aggregates import POINTS, SHARES
from graphspec.build.expressions import evaluate

pytestmark = pytest.mark.extended
PLAN = BuildPlan(1 << 20, 1)
TABLE = pa.table(dict(
    x=pa.array([1, 3, 5, 3], type=pa.float64()),
    name=["b", "a", "b", "c"],
    flag=[False, True, True, False],
    order=pa.array([2, 1, 1, 3], type=pa.float64()),
    tau=pa.array([3, 1, 2, 4], type=pa.int64()),
    weight=pa.array([1, 1, 2, 0], type=pa.float64()),
))
COLUMNS = dict(x="number", name="category", flag="flag", order="number", tau=dict(type="time", unit="day"), weight="number")
EXAMPLES = [
    ("count(x)", 4.0),
    ("sum(x)", 12.0),
    ("mean(x)", 3.0),
    ("min(x)", 1.0),
    ("max(x)", 5.0),
    ("deviation(x)", 1.4142135623730951),
    ("any(flag)", True),
    ("all(flag)", False),
    ("distinct(name)", ["b", "a", "c"]),
    ("list(name, by=order)", ["a", "b", "b", "c"]),
    ("first(name, by=order)", "a"),
    ("last(name, by=order)", "c"),
    ("points(tau, 'speed', x)", [
        dict(tau=3, series="speed", y=1.0), dict(tau=1, series="speed", y=3.0),
        dict(tau=2, series="speed", y=5.0), dict(tau=4, series="speed", y=3.0),
    ]),
    ("shares(name, weight)", [dict(name="b", weight=0.75), dict(name="a", weight=0.25), dict(name="c", weight=0.0)]),
]


def evaluated(text, table=TABLE, columns=COLUMNS, **parameters):
    return evaluate(text, table, columns, plan=PLAN, whole=True, unit="day", **parameters)


@pytest.mark.parametrize("text,expected", EXAMPLES)
def test_every_shipped_aggregate_equals_its_literal_values(text, expected):
    result = evaluated(text)
    assert isinstance(result.values, pa.Scalar)
    assert result.values.as_py() == (pytest.approx(expected) if type(expected) is float else expected)


@pytest.mark.parametrize("text,expected", EXAMPLES)
def test_empty_aggregates_are_typed_missing_except_count(text, expected):
    result = evaluated(text, TABLE.slice(0, 0))
    assert isinstance(result.values, pa.Scalar)
    assert result.values.as_py() == (0.0 if text == "count(x)" else None)
    assert result.values.type == evaluated(text).values.type


@pytest.mark.parametrize("text,expected", EXAMPLES)
def test_selected_missing_input_propagates_through_every_aggregate(text, expected):
    columns = []
    for column in TABLE.columns:
        values = column.to_pylist()
        values[1] = None
        columns.append(pa.array(values, type=column.type))
    table = pa.Table.from_arrays(columns, names=TABLE.column_names)
    assert evaluated(text, table).values.as_py() is None


def test_aggregates_need_explicit_whole_table_scope_and_ignore_arrow_chunk_boundaries():
    with pytest.raises(ValueError, match="whole"):
        evaluate("sum(x)", TABLE, COLUMNS, plan=PLAN)
    chunked = pa.concat_tables([TABLE.slice(0, 1), TABLE.slice(1, 2), TABLE.slice(3)])
    assert chunked["x"].num_chunks == 3
    assert evaluated("sum(x)", chunked).values.as_py() == 12.0
    assert evaluated("x / sum(x)", chunked).values.to_pylist() == pytest.approx([1 / 12, 0.25, 5 / 12, 0.25])
    assert evaluated("sum(x) + max(x)", chunked).values.as_py() == 17.0


def test_aggregate_literals_broadcast_to_the_full_scope():
    assert evaluated("count(1)").values.as_py() == 4.0
    assert evaluated("sum(2)").values.as_py() == 8.0
    assert evaluated("list('same')").values.as_py() == ["same", "same", "same", "same"]
    assert evaluated("deviation(2)").values.as_py() == 0.0


def test_where_filters_before_strict_missing_and_missing_predicates_do_not_hold():
    table = pa.table(dict(x=pa.array([None, 2, 4], type=pa.float64()), keep=[False, True, None]))
    columns = dict(x="number", keep="flag")
    assert evaluated("sum(x, where=keep)", table, columns).values.as_py() == 2.0
    assert evaluated("count(x, where=x is not missing)", table, columns).values.as_py() == 2.0
    assert evaluated("sum(x, where=False)", table, columns).values.as_py() is None
    assert evaluated("count(x, where=False)", table, columns).values.as_py() == 0.0


def test_ordered_aggregates_keep_source_order_for_ties_and_filter_by_together():
    table = pa.table(dict(value=["a", "b", "c", "d"], by=pa.array([2, 1, 1, 2], type=pa.float64())))
    columns = dict(value="text", by="number")
    assert evaluated("list(value)", table, columns).values.as_py() == ["a", "b", "c", "d"]
    assert evaluated("list(value, by=by)", table, columns).values.as_py() == ["b", "c", "a", "d"]
    assert evaluated("first(value, by=by)", table, columns).values.as_py() == "b"
    assert evaluated("last(value, by=by)", table, columns).values.as_py() == "d"
    assert evaluated("list(value, by=by, where=value != 'b')", table, columns).values.as_py() == ["c", "a", "d"]
    missing_order = table.set_column(1, "by", pa.array([2, None, 1, 2], type=pa.float64()))
    assert evaluated("first(value, by=by)", missing_order, columns).values.as_py() is None


def test_aggregates_preserve_time_and_duration_units():
    result = evaluated("min(tau)")
    assert result.type == "time" and result.unit == "day" and result.values.as_py() == 1
    columns = dict(COLUMNS, x=dict(type="duration", unit="minute"))
    result = evaluated("sum(x)", columns=columns)
    assert result.type == "duration" and result.unit == "minute" and result.values.as_py() == 12.0


def test_points_and_shares_have_stable_schemas_and_explicit_semantic_kinds():
    points = evaluated("points(tau, 'speed', x)")
    shares = evaluated("shares(name, weight)")
    assert points.type == "points" and points.values.type == POINTS
    assert shares.type == "shares" and shares.values.type == SHARES
    other = TABLE.set_column(1, "name", pa.array(["new", "new", "other", "other"]))
    assert evaluated("shares(name, weight)", other).values.type == SHARES
    table = pa.table(dict(tau=pa.array([-1, 0, 2], type=pa.float64()), y=pa.array([2, 3, 4], type=pa.float64())))
    assert evaluated("points(tau, 'x', y)", table, dict(tau="number", y="number")).values.as_py() == [
        dict(tau=-1, series="x", y=2.0), dict(tau=0, series="x", y=3.0), dict(tau=2, series="x", y=4.0)
    ]


@pytest.mark.parametrize("weights,message", [([0.0, 0.0], "positive total"), ([1.0, -1.0], "nonnegative")])
def test_shares_refuse_nonpositive_totals_and_negative_weights(weights, message):
    table = pa.table(dict(name=["a", "a"], weight=weights))
    with pytest.raises(ValueError, match=message):
        evaluated("shares(name, weight)", table, dict(name="category", weight="number"))


def test_points_refuse_fractional_counts_and_numeric_reductions_refuse_overflow():
    table = pa.table(dict(x=[1.0, 1.5]))
    with pytest.raises(ValueError, match="row 2.*whole int64"):
        evaluated("points(x, 'x', x)", table, dict(x="number"))
    with pytest.raises(ValueError, match="finite"):
        evaluated("sum(x)", pa.table(dict(x=[1e308, 1e308])), dict(x="number"))


def test_points_keep_exact_finite_counts_after_time_promotion():
    table = pa.table(dict(moment=pa.array([-(1 << 63), (1 << 63) - 1, (1 << 53) + 1], type=pa.int64()), y=[1.0, 2.0, 3.0]))
    columns = dict(moment=dict(type="time", unit="day"), y="number")
    result = evaluated("points(coalesce(moment, never), 'value', y)", table, columns)
    assert result.values.type == POINTS
    assert result.values.as_py() == [
        dict(tau=-9223372036854775808, series="value", y=1.0),
        dict(tau=9223372036854775807, series="value", y=2.0),
        dict(tau=9007199254740993, series="value", y=3.0),
    ]


def test_points_refuse_never_after_time_promotion():
    table = pa.table(dict(moment=pa.array([1, None], type=pa.int64()), y=[1.0, 2.0]))
    with pytest.raises(ValueError, match="row 2.*points tau.*int64"):
        evaluated("points(coalesce(moment, never), 'value', y)", table,
                  dict(moment=dict(type="time", unit="day"), y="number"))


def test_lookup_uses_its_literal_mapping_and_stops_on_an_absent_key():
    table = pa.table(dict(grade=["L", None, "H"]))
    result = evaluated("{'L': 'low', 'H': 'high'}[grade]", table, dict(grade="category"))
    assert result.values.to_pylist() == ["low", None, "high"]
    with pytest.raises(ValueError, match="row 3.*key.*'X'"):
        evaluated("{'L': 'low', 'H': 'high'}[grade]", pa.table(dict(grade=["L", "H", "X"])), dict(grade="category"))


def test_explicit_window_namespaces_reduce_their_own_rows_only():
    later = pa.table(dict(x=pa.array([2, 8], type=pa.float64())))
    result = evaluated("sum(later.x) + x", namespaces=dict(later=(later, dict(x="number"))))
    assert result.values.to_pylist() == [11.0, 13.0, 15.0, 13.0]
    with pytest.raises(ValueError, match="aggregate"):
        evaluated("later.x + x", namespaces=dict(later=(later, dict(x="number"))))


def test_aggregate_scratch_is_refused_before_exceeding_the_plan():
    table = pa.table(dict(value=pa.array(range(100), type=pa.float64())))
    with pytest.raises(ValueError, match="budget"):
        evaluate("distinct(value)", table, dict(value="number"), plan=BuildPlan(4096, 1), whole=True)
