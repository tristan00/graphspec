from decimal import Decimal

import pyarrow as pa
import pytest

from graphspec import FUNCTIONS, TABLES, BuildPlan
from graphspec.build.aggregates import POINTS, SHARES
from graphspec.build.expression_types import EXTENDED_TIME, NEVER_TIME, Kind
from graphspec.build.table_types import Context, Frame, working

pytestmark = pytest.mark.builder
PLAN = BuildPlan(1 << 20, 1)


def frame(data, kinds, name="input", outcomes=None, unit=None):
    table = pa.table(data)
    return Frame(table, kinds, unit=unit, through="max(at)", complete=pa.scalar(17, pa.int64()),
                 outcomes=outcomes or {}, lineage={column: ((name, column),) for column in table.column_names},
                 provenance={name: pa.array([[row] for row in range(table.num_rows)], pa.list_(pa.int64()))})


def run(operation, *frames, **parameters):
    entry = TABLES.named(operation)
    entry.validate(parameters)
    context = Context(operation, PLAN, frames[0].unit)
    with working(context):
        return entry.apply(frames, parameters, context)


def test_group_keeps_first_seen_groups_null_keys_and_full_group_aggregates():
    source = frame(dict(key=["b", "a", "b", "a", None], x=[1.0, 4.0, 3.0, None, 8.0]),
                   dict(key=Kind("identifier"), x=Kind("number")), outcomes=dict(x=(("input", "never"),)))
    result = run("group", source, by=["key"], columns=dict(total="sum(x)", present="sum(x, where=x is not missing)",
                                                        positions="list(row())", magnitude="abs(sum(x))"))
    assert result.table.to_pylist() == [
        dict(key="b", total=4.0, present=4.0, positions=[0.0, 2.0], magnitude=4.0),
        dict(key="a", total=None, present=4.0, positions=[1.0, 3.0], magnitude=None),
        dict(key=None, total=8.0, present=8.0, positions=[4.0], magnitude=8.0),
    ]
    assert result.provenance["input"].to_pylist() == [[0, 2], [1, 3], [4]]
    assert result.through == source.through and result.complete.equals(source.complete)
    assert result.outcomes["total"] == (("input", "never"),)
    assert ("input", "x") in result.lineage["total"]


def test_group_spans_input_chunks_and_has_explicit_empty_group_behavior():
    source = frame(dict(key=["b", "a", "b"], x=[1.0, 2.0, 3.0]), dict(key=Kind("identifier"), x=Kind("number")))
    chunked = Frame(pa.concat_tables([source.table.slice(0, 1), source.table.slice(1)]), source.kinds,
                    unit=source.unit, provenance=source.provenance)
    assert run("group", chunked, by=["key"], columns=dict(total="sum(x)")).table.to_pylist() == [
        dict(key="b", total=4.0), dict(key="a", total=2.0)
    ]
    empty = frame(dict(key=pa.array([], pa.string()), x=pa.array([], pa.float64())), source.kinds)
    grouped = run("group", empty, by=[], columns=dict(count="count(x)", average="mean(x)",
                                                     filled="coalesce(mean(x), 0)", conditional="sum(x) if count(x) > 0 else 0"))
    assert grouped.table.to_pylist() == [dict(count=0.0, average=None, filled=0.0, conditional=0.0)]
    named = run("group", empty, by=["key"], columns=dict(total="sum(x)"))
    assert named.table.num_rows == 0 and named.table.column_names == ["key", "total"]
    assert named.kinds["total"] == Kind("number")


def test_empty_named_groups_infer_types_without_calling_custom_aggregates(monkeypatch):
    calls = []

    def reduce(values):
        calls.append(len(values))
        return pa.scalar(0.0)

    reduce.version, reduce.reads, reduce.gives, reduce.aggregate = 1, ("number",), "number", True
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    FUNCTIONS.register("table_empty_reduce", reduce)
    source = frame(dict(key=pa.array([], pa.string()), x=pa.array([], pa.float64())),
                   dict(key=Kind("identifier"), x=Kind("number")))
    assert run("group", source, by=["key"], columns=dict(total="table_empty_reduce(x)")).table.num_rows == 0
    assert calls == []


def test_group_keeps_extended_times_and_points_shares_semantic_metadata():
    source = frame(dict(key=["a", "b", "a"], moment=pa.array([(1 << 63) - 1, 1, None], pa.int64()),
                        label=["blue", "red", "blue"], weight=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0]),
                   dict(key=Kind("identifier"), moment=Kind("time", unit="turn"), label=Kind("category"),
                        weight=Kind("number"), y=Kind("number")), unit="turn")
    result = run("group", source, by=["key"], columns=dict(times="list(coalesce(moment, never))",
                                                         path="points(moment, 'signal', y)",
                                                         shares="shares(label, weight)"))
    assert result.table.to_pylist() == [
        dict(key="a", times=[Decimal(9223372036854775807), Decimal(NEVER_TIME)], path=None,
             shares=[dict(name="blue", weight=1.0)]),
        dict(key="b", times=[Decimal(1)], path=[dict(tau=1, series="signal", y=20.0)],
             shares=[dict(name="red", weight=1.0)]),
    ]
    assert result.kinds["times"].of == Kind("time", unit="turn", arrow=EXTENDED_TIME)
    assert result.kinds["path"] == Kind("points", arrow=POINTS)
    assert result.kinds["shares"] == Kind("shares", arrow=SHARES)


def test_pivot_keeps_first_seen_group_and_name_order_and_empty_cell_semantics():
    source = frame(dict(owner=["b", "a", "b", "a"], sensor=["z", "x", "x", "x"], amount=[-1.0, 2.0, 3.0, 4.0]),
                   dict(owner=Kind("identifier"), sensor=Kind("identifier"), amount=Kind("number")))
    result = run("pivot", source, by=["owner"], names="sensor", value="abs(sum(amount))")
    assert result.table.column_names == ["owner", "z", "x"]
    assert result.table.to_pylist() == [dict(owner="b", z=1.0, x=3.0), dict(owner="a", z=None, x=6.0)]
    assert result.provenance["input"].to_pylist() == [[0, 2], [1, 3]]
    counted = run("pivot", source, by=["owner"], names="sensor", value="count(amount)")
    assert counted.table.to_pylist() == [dict(owner="b", z=1.0, x=1.0), dict(owner="a", z=0.0, x=2.0)]
    listed = run("pivot", source, by=["owner"], names="sensor", value="list(amount)")
    assert listed.table.to_pylist() == [dict(owner="b", z=[-1.0], x=[3.0]), dict(owner="a", z=None, x=[2.0, 4.0])]
    empty = Frame(source.table.slice(0, 0), source.kinds, unit=source.unit)
    result = run("pivot", empty, by=["owner"], names="sensor", value="sum(amount)")
    assert result.table.num_rows == 0 and result.table.schema == pa.schema([("owner", pa.string())])


@pytest.mark.parametrize("labels,message", [([None], "row 1.*names"), ([""], "row 1.*names"),
                                            (["amount"], "collide"), (["owner"], "collide")])
def test_pivot_names_are_explicit_and_cannot_overwrite_any_input_column(labels, message):
    source = frame(dict(owner=["a"], sensor=pa.array(labels, pa.string()), amount=[1.0]),
                   dict(owner=Kind("identifier"), sensor=Kind("identifier"), amount=Kind("number")))
    with pytest.raises(ValueError, match=message):
        run("pivot", source, by=["owner"], names="sensor", value="sum(amount)")


def test_pivot_refuses_numeric_name_columns_even_when_empty():
    source = frame(dict(owner=pa.array([], pa.string()), sensor=pa.array([], pa.float64()), amount=pa.array([], pa.float64())),
                   dict(owner=Kind("identifier"), sensor=Kind("number"), amount=Kind("number")))
    with pytest.raises(ValueError, match="names column sensor.*text"):
        run("pivot", source, by=["owner"], names="sensor", value="sum(amount)")


def test_unpivot_uses_row_then_column_order_and_preserves_missing_values_and_marks():
    source = frame(dict(moment=pa.array([0, 1], pa.int64()), left=[10.0, 20.0], right=[11.0, None]),
                   dict(moment=Kind("time", unit="day"), left=Kind("number"), right=Kind("number")), unit="day",
                   outcomes=dict(left=(("input", "moment + 1"),), right=(("input", "moment + 2"),)))
    result = run("unpivot", source, keep=["moment"], name="series", value="reading")
    assert result.table.to_pylist() == [
        dict(moment=0, series="left", reading=10.0), dict(moment=0, series="right", reading=11.0),
        dict(moment=1, series="left", reading=20.0), dict(moment=1, series="right", reading=None),
    ]
    assert result.kinds["series"] == Kind("identifier") and result.kinds["reading"] == Kind("number")
    assert result.provenance["input"].to_pylist() == [[0], [0], [1], [1]]
    assert result.outcomes["reading"] == (("input", "moment + 1"), ("input", "moment + 2"))
    assert result.through == source.through and result.complete.equals(source.complete)
    empty = Frame(source.table.slice(0, 0), source.kinds, unit="day")
    result = run("unpivot", empty, keep=["moment"], name="series", value="reading")
    assert result.table.num_rows == 0 and result.table.schema == pa.schema([
        ("moment", pa.int64()), ("series", pa.string()), ("reading", pa.float64())
    ])


def test_unpivot_preserves_semantic_point_records_and_refuses_mixed_types():
    source = frame(dict(key=["a"], left=pa.array([[dict(tau=1, series="x", y=2.0)]], POINTS),
                        right=pa.array([None], POINTS)),
                   dict(key=Kind("identifier"), left=Kind("points", arrow=POINTS), right=Kind("points", arrow=POINTS)))
    result = run("unpivot", source, keep=["key"], name="series", value="path")
    assert result.kinds["path"] == Kind("points", arrow=POINTS)
    assert result.table["path"].type == POINTS
    assert result.table.to_pylist() == [dict(key="a", series="left", path=[dict(tau=1, series="x", y=2.0)]),
                                       dict(key="a", series="right", path=None)]
    mixed = frame(dict(left=[1.0], right=["x"]), dict(left=Kind("number"), right=Kind("category")))
    with pytest.raises(ValueError, match="one exact declared type"):
        run("unpivot", mixed, keep=[], name="series", value="value")


def test_reshape_refuses_collisions_unstated_fields_and_nonaggregate_group_columns():
    source = frame(dict(key=["a"], amount=[1.0]), dict(key=Kind("identifier"), amount=Kind("number")))
    with pytest.raises(ValueError, match="collide"):
        run("group", source, by=["key"], columns=dict(amount="sum(amount)"))
    with pytest.raises(ValueError, match="aggregate expression"):
        run("group", source, by=["key"], columns=dict(total="amount"))
    with pytest.raises(ValueError, match="no column"):
        run("group", source, by=["unknown"], columns=dict(total="sum(amount)"))
    with pytest.raises(ValueError, match="collide"):
        run("unpivot", source, keep=["key"], name="series", value="amount")
