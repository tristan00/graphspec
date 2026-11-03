from decimal import Decimal

import pyarrow as pa
import pytest

from graphspec import TABLES, BuildPlan
from graphspec.build.aggregates import POINTS
from graphspec.build.expression_types import EXTENDED_TIME, NEVER_TIME, Kind
from graphspec.build.table_types import Context, Frame, working

pytestmark = pytest.mark.extended
PLAN = BuildPlan(1 << 20, 1)


def frame(data, kinds, name="input", outcomes=None, unit=None, through="max(at)", complete=17):
    table = pa.table(data)
    return Frame(table, kinds, unit=unit, through=through, complete=pa.scalar(complete, pa.int64()),
                 outcomes=outcomes or {}, lineage={column: ((name, column),) for column in table.column_names},
                 provenance={name: pa.array([[row] for row in range(table.num_rows)], pa.list_(pa.int64()))})


def run(operation, *frames, **parameters):
    entry = TABLES.named(operation)
    entry.validate(parameters)
    context = Context(operation, PLAN, frames[0].unit)
    with working(context):
        return entry.apply(frames, parameters, context)


def test_expand_list_preserves_member_order_missing_members_and_original_positions():
    source = frame(dict(key=["a", "b", "c"], items=pa.array([[1.0, None, 3.0], [], [4.0]], pa.list_(pa.float64()))),
                   dict(key=Kind("identifier"), items=Kind("list", of=Kind("number"))),
                   outcomes=dict(items=(("input", "never"),)))
    result = run("expand", source, column="items", into="amount")
    assert result.table.to_pylist() == [dict(key="a", amount=1.0, place=0.0), dict(key="a", amount=None, place=1.0),
                                       dict(key="a", amount=3.0, place=2.0), dict(key="c", amount=4.0, place=0.0)]
    assert "items" not in result.table.column_names
    assert result.provenance["input"].to_pylist() == [[0], [0], [0], [2]]
    assert result.outcomes["amount"] == (("input", "never"),)
    assert result.through == source.through and result.complete.equals(source.complete)
    empty = Frame(source.table.slice(1, 1), source.kinds, unit=None)
    result = run("expand", empty, column="items", into="amount")
    assert result.table.num_rows == 0 and result.kinds["amount"] == Kind("number")


def test_expand_count_uses_whole_counts_and_adds_no_implicit_place():
    source = frame(dict(key=["a", "b", "c"], count=[2.0, 0.0, 1.0]), dict(key=Kind("identifier"), count=Kind("number")))
    result = run("expand", source, count="count", into="index")
    assert result.table.to_pylist() == [dict(key="a", count=2.0, index=0.0), dict(key="a", count=2.0, index=1.0),
                                       dict(key="c", count=1.0, index=0.0)]
    assert result.table.column_names == ["key", "count", "index"]
    constant = run("expand", source, count="2", into="arm")
    assert constant.table["arm"].to_pylist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_record_expansion_reuses_source_types_missing_rules_aliases_and_outcome_marks():
    records = pa.array([[dict(name="x", score="NA", when="1970-01-01T00:00:02", accepted=True),
                         dict(name="y", score="3.5", when="1970-01-01T00:00:05", accepted=False)], []])
    source = frame(dict(key=["a", "b"], moment=pa.array([0, 10], pa.int64()), records=records),
                   dict(key=Kind("identifier"), moment=Kind("time", unit="second"),
                        records=Kind("list", of=Kind("nested", arrow=records.type.value_type))), unit="second")
    result = run("expand", source, column="records", columns=dict(
        name="identifier", score=dict(type="number", nulls=["NA"], missing=dict(name="value", value=0), outcome="recorded"),
        when=dict(type="time", unit="second", format="%Y-%m-%dT%H:%M:%S", outcome="recorded"), accepted="flag",
    ), names=dict(name="item", when="recorded", place="rank"))
    assert result.table.to_pylist() == [dict(key="a", moment=0, item="x", score=0.0, recorded=2, accepted=True, rank=0.0),
                                       dict(key="a", moment=0, item="y", score=3.5, recorded=5, accepted=False, rank=1.0)]
    assert result.kinds["recorded"] == Kind("time", unit="second")
    assert result.outcomes["score"] == result.outcomes["recorded"] == (("expand", "recorded"),)
    assert result.provenance["input"].to_pylist() == [[0], [0]]


def test_expand_preserves_extended_time_and_semantic_point_member_types():
    time_kind = Kind("time", unit="turn", arrow=EXTENDED_TIME)
    time_type = pa.list_(EXTENDED_TIME)
    source = frame(dict(key=["a"], moments=pa.array([[Decimal((1 << 63) - 1), Decimal(NEVER_TIME)]], time_type)),
                   dict(key=Kind("identifier"), moments=Kind("list", of=time_kind, arrow=time_type)), unit="turn")
    result = run("expand", source, column="moments", into="moment")
    assert result.kinds["moment"] == time_kind
    assert result.table["moment"].to_pylist() == [Decimal(9223372036854775807), Decimal(NEVER_TIME)]
    values = pa.array([[[dict(tau=1, series="x", y=2.0)], None]], pa.list_(POINTS))
    source = frame(dict(paths=values), dict(paths=Kind("list", of=Kind("points", arrow=POINTS), arrow=values.type)))
    result = run("expand", source, column="paths", into="path")
    assert result.kinds["path"] == Kind("points", arrow=POINTS)
    assert result.table.to_pylist() == [dict(path=[dict(tau=1, series="x", y=2.0)], place=0.0), dict(path=None, place=1.0)]


def test_generated_place_can_be_renamed_without_overwriting_outer_values():
    source = frame(dict(key=["a"], place=pa.array([[4.0, 5.0]], pa.list_(pa.float64(), 2)),
                        items=pa.array([["x", "y"]], pa.list_(pa.string()))),
                   dict(key=Kind("identifier"), place=Kind("point", width=2), items=Kind("list", of=Kind("identifier"))))
    with pytest.raises(ValueError, match="collide"):
        run("expand", source, column="items", into="item")
    result = run("expand", source, column="items", into="item", names=dict(place="rank"))
    assert result.table.to_pylist() == [dict(key="a", place=[4.0, 5.0], item="x", rank=0.0),
                                       dict(key="a", place=[4.0, 5.0], item="y", rank=1.0)]
    with pytest.raises(ValueError, match="collide"):
        run("expand", source, column="items", into="items", names=dict(place="rank"))


def test_unknown_list_structure_and_missing_record_fields_refuse_with_outer_rows():
    source = frame(dict(items=pa.array([[1.0], None], pa.list_(pa.float64()))),
                   dict(items=Kind("list", of=Kind("number"))))
    with pytest.raises(ValueError, match="row 2.*nonmissing list"):
        run("expand", source, column="items", into="item")
    records = pa.array([[dict(name="x"), dict(name="y")], [{}]])
    source = frame(dict(records=records), dict(records=Kind("list", of=Kind("nested", arrow=records.type.value_type))))
    with pytest.raises(ValueError, match="row 2.*member 0.*name.*missing"):
        run("expand", source, column="records", columns=dict(name="identifier"))


@pytest.mark.parametrize("count", [None, -1.0, 1.5])
def test_expand_refuses_missing_negative_or_fractional_counts(count):
    source = frame(dict(count=pa.array([count], pa.float64())), dict(count=Kind("number")))
    with pytest.raises(ValueError, match="row 1.*nonmissing nonnegative whole count"):
        run("expand", source, count="count", into="index")


def trees():
    roots = pa.array([
        dict(form="A", leaf=False, children=[dict(form="B", leaf=True, children=[]),
                                            dict(form="C", leaf=False, children=[dict(form="D", leaf=True, children=[])])]),
        dict(form="E", leaf=True, children=[]),
    ])
    return frame(dict(program=["p", "q"], root=roots),
                 dict(program=Kind("identifier"), root=Kind("nested", arrow=roots.type)))


def test_nested_preorder_paths_are_scoped_per_outer_record_and_retain_sibling_positions():
    source = trees()
    result = run("nested", source, record="root", children="children", columns=dict(form="category", leaf="flag"), depth=None)
    assert result.table.to_pylist() == [
        dict(program="p", form="A", leaf=False, path="0", parent=None, place=0.0, depth=0.0),
        dict(program="p", form="B", leaf=True, path="0.0", parent="0", place=0.0, depth=1.0),
        dict(program="p", form="C", leaf=False, path="0.1", parent="0", place=1.0, depth=1.0),
        dict(program="p", form="D", leaf=True, path="0.1.0", parent="0.1", place=0.0, depth=2.0),
        dict(program="q", form="E", leaf=True, path="0", parent=None, place=0.0, depth=0.0),
    ]
    assert "root" not in result.table.column_names
    assert result.provenance["input"].to_pylist() == [[0], [0], [0], [0], [1]]
    assert result.through == source.through and result.complete.equals(source.complete)


def test_nested_depth_limits_and_aliases_are_explicit_and_empty_schema_is_typed():
    source = trees()
    result = run("nested", source, record="root", children="children", columns=dict(form="category", leaf="flag"), depth=1,
                 names=dict(form="shape", path="key", parent="owner", place="rank", depth="level"))
    assert result.table.to_pylist() == [
        dict(program="p", shape="A", leaf=False, key="0", owner=None, rank=0.0, level=0.0),
        dict(program="p", shape="B", leaf=True, key="0.0", owner="0", rank=0.0, level=1.0),
        dict(program="p", shape="C", leaf=False, key="0.1", owner="0", rank=1.0, level=1.0),
        dict(program="q", shape="E", leaf=True, key="0", owner=None, rank=0.0, level=0.0),
    ]
    empty = Frame(source.table.slice(0, 0), source.kinds, unit=source.unit)
    result = run("nested", empty, record="root", children="children", columns=dict(form="category", leaf="flag"), depth=None)
    assert result.table.num_rows == 0
    assert result.kinds["form"] == Kind("category") and result.kinds["parent"] == Kind("identifier")


@pytest.mark.parametrize("record", [dict(form="A", leaf=True), dict(form="A", leaf=True, children=None),
                                   dict(form="A", leaf=True, children="unknown")])
def test_nested_leaves_require_explicit_empty_children(record):
    values = pa.array([record])
    source = frame(dict(root=values), dict(root=Kind("nested", arrow=values.type)))
    with pytest.raises(ValueError, match="row 1.*children.*path 0.*explicit nonmissing list"):
        run("nested", source, record="root", children="children", columns=dict(form="category", leaf="flag"), depth=None)


def test_nested_honors_an_explicit_children_replacement_before_traversal():
    roots = pa.array([dict(form="A"), dict(form="B", children=[])])
    source = frame(dict(root=roots), dict(root=Kind("nested", arrow=roots.type)))
    result = run("nested", source, record="root", children="children", columns=dict(
        form="category", children=dict(type="list", of="nested", missing=dict(name="value", value=[])),
    ), depth=None, names=dict(children="offspring"))
    assert result.table.to_pylist() == [dict(form="A", offspring=[], path="0", parent=None, place=0.0, depth=0.0),
                                       dict(form="B", offspring=[], path="0", parent=None, place=0.0, depth=0.0)]
    assert result.kinds["offspring"].name == "list" and result.kinds["offspring"].of.name == "nested"


def test_nested_refuses_null_roots_and_names_the_path_of_missing_fields():
    source = trees()
    null = Frame(pa.table(dict(program=["p"], root=pa.array([None], source.table["root"].type))), source.kinds, unit=None)
    with pytest.raises(ValueError, match="row 1.*root path 0.*nonmissing record"):
        run("nested", null, record="root", children="children", columns=dict(form="category"), depth=None)
    roots = pa.array([dict(form="A", children=[dict(children=[])])])
    source = frame(dict(root=roots), dict(root=Kind("nested", arrow=roots.type)))
    with pytest.raises(ValueError, match="row 1.*path 0.0.*form.*missing"):
        run("nested", source, record="root", children="children", columns=dict(form="category"), depth=None)


def test_nested_reads_only_declared_fields_and_preserves_native_timestamp_error_rows():
    roots = pa.StructArray.from_arrays([
        pa.array(["A", "B"]), pa.array([[], []], pa.list_(pa.null())),
        pa.array([0, (1 << 63) - 1], pa.timestamp("s")),
    ], names=["form", "children", "moment"])
    source = frame(dict(root=roots), dict(root=Kind("nested", arrow=roots.type)), unit="day")
    result = run("nested", source, record="root", children="children", columns=dict(form="category"), depth=None)
    assert result.table["form"].to_pylist() == ["A", "B"]
    with pytest.raises(ValueError, match="row 2.*path 0.*moment"):
        run("nested", source, record="root", children="children", columns=dict(moment=dict(type="time", unit="day")), depth=None)


def test_cut_preserves_flat_order_zero_length_groups_and_both_row_origins():
    flat = frame(dict(value=[10.0, 11.0, 12.0]), dict(value=Kind("number")), name="flat")
    sizes = frame(dict(size=[2.0, 0.0, 1.0]), dict(size=Kind("number")), name="sizes", through="max(size)", complete=23,
                  outcomes=dict(size=(("sizes", "never"),)))
    result = run("cut", flat, sizes, count="size", into="group", names=dict(place="position"))
    assert result.table.to_pylist() == [dict(value=10.0, group=0.0, position=0.0),
                                       dict(value=11.0, group=0.0, position=1.0), dict(value=12.0, group=2.0, position=0.0)]
    assert result.provenance["flat"].to_pylist() == [[0], [1], [2]]
    assert result.provenance["sizes"].to_pylist() == [[0], [0], [2]]
    assert result.outcomes["group"] == (("sizes", "never"),)
    assert result.through == flat.through and result.complete.equals(flat.complete)
    empty = Frame(flat.table.slice(0, 0), flat.kinds, unit=None)
    zero = frame(dict(size=[0.0, 0.0]), dict(size=Kind("number")), name="sizes")
    result = run("cut", empty, zero, count="size", into="group")
    assert result.table.num_rows == 0 and result.table.column_names == ["value", "group", "place"]


def test_cut_refuses_lengths_that_do_not_partition_all_rows_and_names_that_collide():
    flat = frame(dict(value=[1.0, 2.0, 3.0]), dict(value=Kind("number")))
    sizes = frame(dict(size=[2.0, 2.0]), dict(size=Kind("number")))
    with pytest.raises(ValueError, match="sum to 4.*3 rows"):
        run("cut", flat, sizes, count="size", into="group")
    with pytest.raises(ValueError, match="collide"):
        run("cut", flat, sizes, count="size", into="value")


def test_expansion_refuses_excessive_output_before_constructing_rows():
    source = frame(dict(key=["a"]), dict(key=Kind("identifier")))
    with pytest.raises(ValueError, match="budget"):
        run("expand", source, count="1000000", into="index")
