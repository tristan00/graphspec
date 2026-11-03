from decimal import Decimal

import pyarrow as pa
import pytest

from graphspec import BuildPlan
from graphspec.build.aggregates import POINTS, SHARES
from graphspec.build.expression_types import EXTENDED_TIME, FUNCTIONS, NEVER_TIME, Kind
from graphspec.build.expressions import evaluate
from graphspec.build.table_relations import Closure, Join, Order
from graphspec.build.table_types import Context, Frame

pytestmark = pytest.mark.extended
PLAN = BuildPlan(4 << 20, 1)


def frame(data, kinds, name="source", unit=None, through=None, outcomes=None, complete=None):
    table = pa.table(data)
    return Frame(table, kinds, unit, through, outcomes or {},
                 {column: ((name, column),) for column in kinds},
                 {name: pa.array([[row] for row in range(table.num_rows)], pa.list_(pa.int64()))}, complete)


def applied(entry, frames, parameters, plan=PLAN):
    return entry.apply(frames, parameters, Context("made_" + type(entry).__name__.lower(), plan, frames[0].unit))


def test_aligned_other_namespace_is_rowwise_in_predicates_and_branches():
    table = pa.table(dict(value=[1.0, 5.0, 3.0]))
    other = pa.table(dict(limit=[2.0, 4.0, None], label=["a", "b", "c"]))
    namespace = dict(other=(other, dict(limit=Kind("number"), label=Kind("category"))))
    result = evaluate("value < other.limit", table, dict(value=Kind("number")), plan=PLAN, namespaces=namespace)
    assert result.values.to_pylist() == [True, False, None]
    result = evaluate("other.label if value < 3 else 'large'", table, dict(value=Kind("number")),
                      plan=PLAN, namespaces=namespace)
    assert result.values.to_pylist() == ["a", "large", "large"]


def test_other_namespace_aggregates_stay_in_the_main_row_scope():
    table = pa.table(dict(value=[1.0, 5.0, 3.0]))
    other = pa.table(dict(value=[2.0, 4.0, 6.0]))
    result = evaluate("sum(value + other.value)", table, dict(value="number"), plan=PLAN, whole=True,
                      namespaces=dict(other=(other, dict(value="number"))))
    assert result.values.as_py() == 21.0
    current = pa.table(dict(value=[0.0]))
    other = pa.table(dict(bump=[10.0]))
    later = pa.table(dict(value=[1.0, 2.0]))
    result = evaluate("sum(later.value + other.bump)", current, dict(value="number"), plan=PLAN, whole=True,
                      namespaces=dict(other=(other, dict(bump="number")), later=(later, dict(value="number"))))
    assert result.values.as_py() == 23.0


def test_other_namespace_refuses_unaligned_rows_without_relaxing_window_rules():
    table = pa.table(dict(value=[1.0, 2.0]))
    with pytest.raises(ValueError, match="other namespace.*aligned row"):
        evaluate("1", table, dict(value="number"), plan=PLAN,
                 namespaces=dict(other=(table.slice(0, 1), dict(value="number"))))
    with pytest.raises(ValueError, match="only inside an aggregate"):
        evaluate("later.value", table, dict(value="number"), plan=PLAN,
                 namespaces=dict(later=(table, dict(value="number"))))


def test_exact_kind_metadata_preserves_extended_time_and_compound_shapes():
    moments = pa.array([Decimal((1 << 63) - 1), Decimal(NEVER_TIME)], EXTENDED_TIME)
    times = pa.array([[Decimal((1 << 63) - 1)], [Decimal(NEVER_TIME)]], pa.list_(EXTENDED_TIME))
    points = pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64(), 2))
    records = pa.array([dict(name="a"), dict(name="b")])
    table = pa.table(dict(moment=moments, times=times, point=points, record=records))
    moment = Kind("time", unit="second", arrow=EXTENDED_TIME)
    kinds = dict(moment=moment, times=Kind("list", of=moment), point=Kind("point", width=2),
                 record=Kind("nested", arrow=records.type))
    assert evaluate("moment < never", table, kinds, plan=PLAN, unit="second").values.to_pylist() == [True, False]
    for name in ("times", "point", "record"):
        result = evaluate(name, table, kinds, plan=PLAN, unit="second")
        assert result.kind == kinds[name]
        assert result.values.type == table[name].type
        assert result.values.to_pylist() == table[name].to_pylist()


def test_exact_kind_metadata_refuses_incompatible_physical_types_and_units():
    table = pa.table(dict(value=pa.array([0], pa.int64())))
    with pytest.raises(ValueError, match="exact Arrow representation"):
        evaluate("value", table, dict(value=Kind("number")), plan=PLAN)
    with pytest.raises(ValueError, match="explicit unit"):
        evaluate("value", table, dict(value=Kind("time", unit="day")), plan=PLAN, unit="second")


@pytest.mark.parametrize("kind,values,message", [
    (Kind("number", arrow=pa.int64()), pa.array([1], pa.int64()), "invalid Arrow representation"),
    (Kind("point", width=2, arrow=pa.list_(pa.float64(), 3)), pa.array([[1.0, 2.0, 3.0]], pa.list_(pa.float64(), 3)), "valid width"),
    (Kind("vector", width=3, arrow=pa.list_(pa.float64(), 2)), pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2)), "valid width"),
    (Kind("points", arrow=pa.int64()), pa.array([1], pa.int64()), "invalid Arrow representation"),
    (Kind("shares", arrow=pa.int64()), pa.array([1], pa.int64()), "invalid Arrow representation"),
])
def test_exact_kind_overrides_cannot_bypass_their_logical_representations(kind, values, message):
    with pytest.raises(ValueError, match=message):
        evaluate("value", pa.table(dict(value=values)), dict(value=kind), plan=PLAN)
    with pytest.raises(ValueError, match=message):
        Frame(pa.table(dict(value=values)), dict(value=kind))


def test_exact_kind_metadata_retains_points_and_shares_records():
    points = pa.array([[dict(tau=1, series="x", y=2.0)]], POINTS)
    shares = pa.array([[dict(name="a", weight=1.0)]], SHARES)
    table = pa.table(dict(points=points, shares=shares))
    kinds = dict(points=Kind("points", arrow=POINTS), shares=Kind("shares", arrow=SHARES))
    for name in kinds:
        result = evaluate(name, table, kinds, plan=PLAN)
        assert result.kind == kinds[name] and result.values.type == table[name].type
        assert result.values.to_pylist() == table[name].to_pylist()


def test_expression_positions_preserve_input_row_identity_and_contextual_errors():
    table = pa.table(dict(value=[1.0, -1.0, 4.0]))
    kinds = dict(value=Kind("number"))
    result = evaluate("row()", table, kinds, plan=PLAN, positions=[5, 2, 5], offset=10)
    assert result.values.to_pylist() == [15.0, 12.0, 15.0]
    result = evaluate("row() if value > 0 else 0", table, kinds, plan=PLAN, positions=[5, 2, 5])
    assert result.values.to_pylist() == [5.0, 0.0, 5.0]
    with pytest.raises(ValueError, match="row 13"):
        evaluate("sqrt(value)", table, kinds, plan=PLAN, positions=[5, 2, 5], offset=10)
    with pytest.raises(ValueError, match="input-row position"):
        evaluate("row()", table, kinds, plan=PLAN, positions=[0, 1])


def test_ordered_neighbor_maps_read_whole_table_values_without_changing_row_identity():
    table = pa.table(dict(value=[1.0, 4.0, 2.0], keep=[False, True, True]))
    kinds = dict(value=Kind("number"), keep=Kind("flag"))
    neighbors = dict(previous=[None, 2, 0], following=[2, None, 1])
    result = evaluate("previous(row())", table, kinds, plan=PLAN, ordered=True, positions=[5, 2, 8], neighbors=neighbors)
    assert result.values.to_pylist() == [None, 8.0, 5.0]
    result = evaluate("following(value) if keep else missing", table, kinds, plan=PLAN, ordered=True, neighbors=neighbors)
    assert result.values.to_pylist() == [None, None, 4.0]
    result = evaluate("value / sum(value)", table, kinds, plan=PLAN, whole=True, ordered=True, neighbors=neighbors)
    assert result.values.to_pylist() == pytest.approx([1 / 7, 4 / 7, 2 / 7])
    with pytest.raises(ValueError, match="neighbors maps each row"):
        evaluate("previous(value)", table, kinds, plan=PLAN, ordered=True,
                 neighbors=dict(previous=[None, 9, 0], following=[2, None, 1]))


def test_scalar_aggregate_compositions_and_empty_scalar_conditionals_keep_scalar_shape():
    table = pa.table(dict(value=[-1.0, -3.0]))
    kinds = dict(value=Kind("number"))
    for text, expected in (("abs(sum(value))", 4.0), ("coalesce(mean(value), 0)", -2.0),
                           ("sum(value) if count(value) > 0 else 0", -4.0)):
        result = evaluate(text, table, kinds, plan=PLAN, whole=True)
        assert isinstance(result.values, pa.Scalar) and result.values.as_py() == expected
    empty = table.slice(0, 0)
    for text, expected in (("abs(count(value))", 0.0), ("coalesce(mean(value), 0)", 0.0),
                           ("sum(value) if count(value) > 0 else 0", 0.0)):
        result = evaluate(text, empty, kinds, plan=PLAN, whole=True)
        assert isinstance(result.values, pa.Scalar) and result.values.as_py() == expected
    assert evaluate("abs(2)", table, kinds, plan=PLAN).values.to_pylist() == [2.0, 2.0]
    assert evaluate("value + abs(sum(value))", table, kinds, plan=PLAN, whole=True).values.to_pylist() == [3.0, 1.0]


def test_aggregate_derived_custom_function_arguments_remain_arrow_arrays(monkeypatch):
    received = []

    def shifted(values, amount):
        received.append((values, amount))
        return pa.array([value + amount for value in values.to_pylist()], pa.float64())

    shifted.version, shifted.reads, shifted.gives, shifted.aggregate = 1, ("number", "number"), "number", False
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    FUNCTIONS.register("table_shifted_aggregate", shifted)
    table = pa.table(dict(value=[1.0, 3.0]))
    result = evaluate("table_shifted_aggregate(sum(value), 1 + 1)", table, dict(value="number"), plan=PLAN, whole=True)
    assert result.values.as_py() == 6.0
    assert isinstance(received[0][0], pa.Array) and received[0][0].to_pylist() == [4.0]
    assert received[0][1] == 2.0


def test_join_keeps_left_rows_then_right_match_order_and_explicit_missing_matches():
    left = frame(dict(key=["a", "b", "a", "c", None], amount=[2.0, 7.0, 3.0, 1.0, 9.0]),
                 dict(key=Kind("identifier"), amount=Kind("number")), name="left")
    right = frame(dict(key=["a", "a", "b", None], points=[10.0, 11.0, 20.0, 99.0]),
                  dict(key=Kind("identifier"), points=Kind("number")), name="right")
    parameters = dict(on=dict(key="key"), take=dict(score="points"), unmatched="keep")
    result = applied(Join(), [left, right], parameters)
    assert result.table.to_pylist() == [
        dict(key="a", amount=2.0, score=10.0), dict(key="a", amount=2.0, score=11.0),
        dict(key="b", amount=7.0, score=20.0), dict(key="a", amount=3.0, score=10.0),
        dict(key="a", amount=3.0, score=11.0), dict(key="c", amount=1.0, score=None),
        dict(key=None, amount=9.0, score=None),
    ]
    assert result.provenance["left"].to_pylist() == [[0], [0], [1], [2], [2], [3], [4]]
    assert result.provenance["right"].to_pylist() == [[0], [1], [2], [0], [1], [], []]
    dropped = applied(Join(), [left, right], dict(parameters, unmatched="drop"))
    assert dropped.table.to_pylist() == result.table.slice(0, 5).to_pylist()
    with pytest.raises(ValueError, match="row 4.*unmatched is stop"):
        applied(Join(), [left, right], dict(parameters, unmatched="stop"))


def test_join_inequalities_and_row_identity_use_aligned_candidate_rows():
    left = frame(dict(value=[3.0, 1.0]), dict(value=Kind("number")), name="left")
    right = frame(dict(limit=[2.0, 4.0], label=["small", "large"]),
                  dict(limit=Kind("number"), label=Kind("category")), name="right")
    parameters = dict(on={}, take=dict(matched="label"), unmatched="drop", where="value < other.limit")
    result = applied(Join(), [left, right], parameters)
    assert result.table.to_pylist() == [dict(value=3.0, matched="large"), dict(value=1.0, matched="small"),
                                       dict(value=1.0, matched="large")]
    result = applied(Join(), [left, right], dict(parameters, where="row() == 1"))
    assert result.table.to_pylist() == [dict(value=1.0, matched="small"), dict(value=1.0, matched="large")]


def test_join_predicate_aggregates_reduce_the_whole_candidate_pair_table():
    left = frame(dict(key=["a", "b"]), dict(key=Kind("identifier")), name="left")
    right = frame(dict(key=["a", "a", "b"], score=[10.0, 20.0, 100.0]),
                  dict(key=Kind("identifier"), score=Kind("number")), name="right")
    result = applied(Join(), [left, right], dict(on=dict(key="key"), take=dict(score="score"), unmatched="keep",
                                               where="other.score > mean(other.score)"))
    assert result.table.to_pylist() == [dict(key="a", score=None), dict(key="b", score=100.0)]


def test_join_window_has_closed_bounds_exact_spans_and_a_distinct_never():
    kind = Kind("time", unit="minute", arrow=EXTENDED_TIME)
    left = frame(dict(at=pa.array([Decimal(10), Decimal(30), Decimal(NEVER_TIME)], EXTENDED_TIME)),
                 dict(at=kind), name="left", unit="minute")
    right = frame(dict(moment=pa.array([Decimal(value) for value in (8, 10, 12, 13, 30, NEVER_TIME)], EXTENDED_TIME),
                       tag=["before", "same", "after", "late", "second", "never"]),
                  dict(moment=kind, tag=Kind("category")), name="right", unit="minute")
    parameters = dict(on={}, take=dict(matched="tag"), unmatched="stop",
                      window=dict(at="at", other="moment", before="span(120, 'second')", after=2))
    result = applied(Join(), [left, right], parameters)
    assert result.table["matched"].to_pylist() == ["before", "same", "after", "second", "never"]
    assert result.table["at"].to_pylist() == [Decimal(10), Decimal(10), Decimal(10), Decimal(30), Decimal(NEVER_TIME)]
    with pytest.raises(ValueError, match="row 1.*whole count.*minute"):
        applied(Join(), [left, right], dict(parameters, window=dict(parameters["window"], before="span(90, 'second')")))


def test_join_window_uses_calendar_months_and_missing_times_match_nothing():
    kind = Kind("time", unit="day")
    left = frame(dict(at=pa.array([19751, None], pa.int64())), dict(at=kind), unit="day")
    right = frame(dict(moment=pa.array([19751, 19782, 19783, None], pa.int64()), tag=["same", "leap", "later", "missing"]),
                  dict(moment=kind, tag=Kind("category")), name="right", unit="day")
    parameters = dict(on={}, take=dict(matched="tag"), unmatched="keep",
                      window=dict(at="at", other="moment", before=0, after="span(1, 'month')"))
    assert applied(Join(), [left, right], parameters).table.to_pylist() == [
        dict(at=19751, matched="same"), dict(at=19751, matched="leap"), dict(at=None, matched=None),
    ]


def test_join_preserves_through_and_deferred_outcome_metadata_and_refuses_collisions():
    left = frame(dict(key=["a"]), dict(key=Kind("identifier")), name="left", through="max(at)", complete=50)
    right = frame(dict(key=["a"], amount=[2.0]), dict(key=Kind("identifier"), amount=Kind("number")),
                  name="right", outcomes=dict(amount=(("right", "known"),)))
    parameters = dict(on=dict(key="key"), take=dict(total="amount"), unmatched="stop", outcomes=dict(total="time('1970-01-01')"))
    result = applied(Join(), [left, right], parameters)
    assert result.through == "max(at)" and result.complete == 50
    assert result.outcomes["total"] == (("right", "known"), ("made_join", "time('1970-01-01')"))
    assert ("right", "amount") in result.lineage["total"]
    with pytest.raises(ValueError, match="collide.*key"):
        applied(Join(), [left, right], dict(parameters, take=dict(key="amount"), outcomes={}))


def test_join_empty_right_keeps_typed_nulls_or_drops_rows_by_the_stated_policy():
    left = frame(dict(key=["a"]), dict(key=Kind("identifier")))
    right = frame(dict(key=pa.array([], pa.string()), value=pa.array([], pa.float64())),
                  dict(key=Kind("identifier"), value=Kind("number")), name="right")
    parameters = dict(on=dict(key="key"), take=dict(value="value"), unmatched="keep")
    result = applied(Join(), [left, right], parameters)
    assert result.table.to_pylist() == [dict(key="a", value=None)]
    assert result.table["value"].type == pa.float64()
    assert applied(Join(), [left, right], dict(parameters, unmatched="drop")).table.num_rows == 0


def test_order_annotates_sorted_runs_and_returns_the_original_unsorted_rows():
    source = frame(dict(user=["a", "b", "a", "a", "a", "b"], minute=pa.array([40, 5, 0, 10, 10, 50], pa.int64()),
                        key=["a40", "b5", "a0", "a10first", "a10second", "b50"], amount=[4.0, 2.0, 1.0, 3.0, 5.0, 6.0]),
                   dict(user=Kind("identifier"), minute=Kind("time", unit="minute"), key=Kind("identifier"), amount=Kind("number")),
                   unit="minute", through="max(minute)", outcomes=dict(amount=(("source", "minute"),)))
    result = applied(Order(), [source], dict(within=["user"], by=["minute"], gap=20,
                                            columns=dict(before="previous(key)", after="following(key)", identity="row()",
                                                         share="amount / sum(amount)")))
    assert result.table["key"].to_pylist() == source.table["key"].to_pylist()
    assert result.table["run"].to_pylist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert result.table["place"].to_pylist() == [0.0, 0.0, 0.0, 1.0, 2.0, 0.0]
    assert result.table["before"].to_pylist() == [None, None, None, "a0", "a10first", None]
    assert result.table["after"].to_pylist() == [None, None, "a10first", "a10second", None, None]
    assert result.table["identity"].to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert result.table["share"].to_pylist() == pytest.approx([4 / 21, 2 / 21, 1 / 21, 3 / 21, 5 / 21, 6 / 21])
    assert result.through == "max(minute)" and result.outcomes["amount"] == source.outcomes["amount"]
    assert result.outcomes["share"] == source.outcomes["amount"]
    assert result.provenance["source"].equals(source.provenance["source"])


def test_order_gap_respects_exact_duration_thresholds_and_never_runs():
    kind = Kind("time", unit="minute", arrow=EXTENDED_TIME)
    source = frame(dict(moment=pa.array([Decimal(12), Decimal(0), Decimal(NEVER_TIME), Decimal(2), Decimal(NEVER_TIME)], EXTENDED_TIME)),
                   dict(moment=kind), unit="minute")
    result = applied(Order(), [source], dict(within=[], by=["moment"], gap="span(120, 'second')"))
    assert result.table["run"].to_pylist() == [1.0, 0.0, 2.0, 0.0, 2.0]
    assert result.table["place"].to_pylist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="whole count.*minute"):
        applied(Order(), [source], dict(within=[], by=["moment"], gap="span(90, 'second')"))


def test_order_never_does_not_hide_an_invalid_calendar_gap_boundary():
    kind = Kind("time", unit="day", arrow=EXTENDED_TIME)
    source = frame(dict(moment=pa.array([Decimal(19753), Decimal(NEVER_TIME)], EXTENDED_TIME)), dict(moment=kind), unit="day")
    with pytest.raises(ValueError, match="row 2.*day.*range"):
        applied(Order(), [source], dict(within=[], by=["moment"], gap="span(1, 'month')"))


def test_order_renames_generated_columns_and_refuses_missing_sort_values_or_overwrites():
    source = frame(dict(value=[2.0, 1.0]), dict(value=Kind("number")))
    result = applied(Order(), [source], dict(within=[], by=["value"], gap=None,
                                            names=dict(run="session", place="rank"), columns=dict(before="previous(value)")))
    assert result.table.to_pylist() == [dict(value=2.0, session=0.0, rank=1.0, before=1.0),
                                       dict(value=1.0, session=0.0, rank=0.0, before=None)]
    with pytest.raises(ValueError, match="collide.*value"):
        applied(Order(), [source], dict(within=[], by=["value"], gap=None, columns=dict(value="1")))
    missing = frame(dict(value=pa.array([1.0, None], pa.float64())), dict(value=Kind("number")))
    with pytest.raises(ValueError, match="row 2.*by column value is missing"):
        applied(Order(), [missing], dict(within=[], by=["value"], gap=None))


def test_order_empty_inputs_keep_typed_annotations_and_validate_independent_new_columns():
    source = frame(dict(value=pa.array([], pa.float64())), dict(value=Kind("number")))
    result = applied(Order(), [source], dict(within=[], by=["value"], gap=None, columns=dict(before="previous(value)", one="1")))
    assert result.table.column_names == ["value", "run", "place", "before", "one"]
    assert all(field.type == pa.float64() for field in result.table.schema)
    assert result.table.num_rows == 0
    with pytest.raises(ValueError, match="unknown column added"):
        applied(Order(), [source], dict(within=[], by=["value"], gap=None, columns=dict(added="1", dependent="added + 1")))


def test_closure_uses_stable_breadth_first_shortest_paths_and_records_the_path_rows():
    source = frame(dict(source=["a", "a", "b", "c", "d", "a"], target=["b", "c", "d", "d", None, "b"]),
                   dict(source=Kind("identifier"), target=Kind("identifier")), through="never")
    parameters = dict(source="source", target="target", most=None)
    result = applied(Closure(), [source], parameters)
    assert result.table.to_pylist() == [dict(source="a", target="b", steps=1.0), dict(source="a", target="c", steps=1.0),
                                       dict(source="a", target="d", steps=2.0), dict(source="b", target="d", steps=1.0),
                                       dict(source="c", target="d", steps=1.0)]
    assert result.provenance["source"].to_pylist() == [[0], [1], [0, 2], [2], [3]]
    assert result.through == "never"
    limited = applied(Closure(), [source], dict(parameters, most=1, names=dict(steps="hops")))
    assert limited.table.to_pylist() == [dict(source="a", target="b", hops=1.0), dict(source="a", target="c", hops=1.0),
                                        dict(source="b", target="d", hops=1.0), dict(source="c", target="d", hops=1.0)]
    assert applied(Closure(), [source], dict(parameters, most=0)).table.num_rows == 0


def test_closure_shortest_steps_include_later_direct_edges():
    source = frame(dict(source=["a", "b", "a"], target=["b", "c", "c"]),
                   dict(source=Kind("identifier"), target=Kind("identifier")))
    assert applied(Closure(), [source], dict(source="source", target="target", most=None)).table.to_pylist() == [
        dict(source="a", target="b", steps=1.0), dict(source="a", target="c", steps=1.0), dict(source="b", target="c", steps=1.0),
    ]


@pytest.mark.parametrize("most", [None, 0, 1])
def test_closure_refuses_cycles_even_when_the_step_limit_would_hide_them(most):
    source = frame(dict(source=["a", "b"], target=["b", "a"]), dict(source=Kind("identifier"), target=Kind("identifier")))
    with pytest.raises(ValueError, match="closure contains a cycle"):
        applied(Closure(), [source], dict(source="source", target="target", most=most))


def test_closure_refuses_missing_sources_and_generated_name_collisions():
    source = frame(dict(source=pa.array([None], pa.string()), target=pa.array([None], pa.string())),
                   dict(source=Kind("identifier"), target=Kind("identifier")))
    with pytest.raises(ValueError, match="row 1.*source is missing"):
        applied(Closure(), [source], dict(source="source", target="target", most=None))
    source = frame(dict(source=["a"], target=["b"], steps=[1.0]),
                   dict(source=Kind("identifier"), target=Kind("identifier"), steps=Kind("number")))
    with pytest.raises(ValueError, match="collide.*steps"):
        applied(Closure(), [source], dict(source="source", target="target", most=None))


def test_join_fanout_is_refused_within_the_explicit_memory_budget():
    left = frame(dict(key=["a"] * 100), dict(key=Kind("identifier")), name="left")
    right = frame(dict(key=["a"] * 100), dict(key=Kind("identifier")), name="right")
    with pytest.raises(ValueError, match="budget"):
        applied(Join(), [left, right], dict(on=dict(key="key"), take={}, unmatched="stop"), BuildPlan(256 << 10, 1))
