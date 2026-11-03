import pyarrow as pa
import pytest

from graphspec import BuildPlan
from graphspec.build.expressions import evaluate

pytestmark = pytest.mark.builder
PLAN = BuildPlan(1 << 20, 1)
TABLE = pa.table(dict(
    x=pa.array([-1.5, 0, 2, None], type=pa.float64()),
    y=pa.array([2, 2, -3, 4], type=pa.float64()),
    spare=pa.array([None, 5, None, None], type=pa.float64()),
    text=["1", "-2.5", "0", None],
    truth=["true", "false", "true", None],
    flag=[True, False, True, None],
))
COLUMNS = dict(x="number", y="number", spare="number", text="text", truth="text", flag="flag")
EXAMPLES = [
    ("abs(x)", [1.5, 0.0, 2.0, None]),
    ("exp(x)", [0.22313016014842982, 1.0, 7.38905609893065, None]),
    ("log(abs(x) + 1)", [0.9162907318741551, 0.0, 1.0986122886681098, None]),
    ("sqrt(abs(x))", [1.224744871391589, 0.0, 1.4142135623730951, None]),
    ("round(x)", [-2.0, 0.0, 2.0, None]),
    ("floor(x)", [-2.0, 0.0, 2.0, None]),
    ("ceil(x)", [-1.0, 0.0, 2.0, None]),
    ("least(x, y)", [-1.5, 0.0, -3.0, None]),
    ("greatest(x, y)", [2.0, 2.0, 2.0, None]),
    ("bin(x, [-1, 0, 2])", [0.0, 2.0, 3.0, None]),
    ("bucket(x, 2)", [-1.0, 0.0, 1.0, None]),
    ("coalesce(x, spare, 7)", [-1.5, 0.0, 2.0, 7.0]),
    ("number(text)", [1.0, -2.5, 0.0, None]),
    ("flag(truth)", [True, False, True, None]),
    ("category(x)", ["-1.5", "0", "2", None]),
    ("text(x)", ["-1.5", "0", "2", None]),
    ("vector(x, y)", [[-1.5, 2.0], [0.0, 2.0], [2.0, -3.0], None]),
    ("point(x, y)", [[-1.5, 2.0], [0.0, 2.0], [2.0, -3.0], None]),
    ("row()", [0.0, 1.0, 2.0, 3.0]),
    ("previous(x)", [None, -1.5, 0.0, 2.0]),
    ("following(x)", [0.0, 2.0, None, None]),
]


def evaluated(text, table=TABLE, columns=COLUMNS, **parameters):
    return evaluate(text, table, columns, plan=PLAN, **parameters)


@pytest.mark.parametrize("text,expected", EXAMPLES)
def test_every_shipped_scalar_function_equals_literal_values(text, expected):
    result = evaluated(text, ordered=text.startswith(("previous(", "following(")))
    assert isinstance(result.values, (pa.Array, pa.ChunkedArray))
    assert result.values.to_pylist() == expected


def test_round_uses_even_ties_and_explicit_decimal_places():
    values = pa.table(dict(x=pa.array([1.25, 1.35, -1.25, -1.35], type=pa.float64())))
    assert evaluated("round(x, 1)", values, dict(x="number")).values.to_pylist() == [1.2, 1.4, -1.2, -1.4]
    values = pa.table(dict(x=pa.array([15, 25, -15, -25], type=pa.float64())))
    assert evaluated("round(x, -1)", values, dict(x="number")).values.to_pylist() == [20.0, 20.0, -20.0, -20.0]


def test_bin_and_bucket_state_their_edge_and_negative_boundary_behavior():
    values = pa.table(dict(x=pa.array([-2, -1, -0.5, 0, 1, 2], type=pa.float64())))
    assert evaluated("bin(x, [-1, 0, 2])", values, dict(x="number")).values.to_pylist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    assert evaluated("bucket(x, 1)", values, dict(x="number")).values.to_pylist() == [-2.0, -1.0, -1.0, 0.0, 1.0, 2.0]


def test_casts_handle_flags_numbers_and_canonical_whole_number_names():
    assert evaluated("number(flag)").values.to_pylist() == [1.0, 0.0, 1.0, None]
    assert evaluated("flag(x)").values.to_pylist() == [True, False, True, None]
    assert evaluated("category(flag)").values.to_pylist() == ["True", "False", "True", None]
    assert evaluated("coalesce(missing, x, 0)").values.to_pylist() == [-1.5, 0.0, 2.0, 0.0]


def test_vectors_and_points_keep_fixed_width_and_whole_row_missing():
    vector = evaluated("vector(x, y, 5)")
    point = evaluated("point(x, y, 5)")
    assert vector.type == "vector" and vector.values.type == pa.list_(pa.float64(), 3)
    assert point.type == "point" and point.values.type == pa.list_(pa.float64(), 3)
    assert point.values.to_pylist() == [[-1.5, 2.0, 5.0], [0.0, 2.0, 5.0], [2.0, -3.0, 5.0], None]


@pytest.mark.parametrize("text,message", [
    ("round(x, 1.5)", "whole"),
    ("bin(x, [0, 0])", "strictly increasing"),
    ("bin(x, [2, 1])", "strictly increasing"),
    ("bucket(x, 0)", "positive"),
    ("bucket(x, -1)", "positive"),
    ("hash(x, salt='')", "salt"),
    ("previous(x)", "ordered"),
    ("following(x)", "ordered"),
])
def test_malformed_parameters_refuse_even_an_empty_table(text, message):
    with pytest.raises(ValueError, match=message):
        evaluated(text, TABLE.slice(0, 0))


@pytest.mark.parametrize("text", ["vector()", "point(x)", "point(x, y, x, y)", "least(x)", "greatest(x)", "hash(salt='x')"])
def test_scalar_functions_refuse_wrong_argument_counts(text):
    with pytest.raises((ValueError, TypeError)):
        evaluated(text)


@pytest.mark.parametrize("text,values", [
    ("log(x)", [1.0, 0.0]),
    ("sqrt(x)", [1.0, -1.0]),
    ("exp(x)", [1.0, 1000.0]),
    ("number(x)", ["1", "invalid"]),
    ("flag(x)", ["true", "yes"]),
])
def test_bad_scalar_values_name_the_declaration_field_and_global_row(text, values):
    kind = "text" if type(values[0]) is str else "number"
    with pytest.raises(ValueError, match="records.*score.*row 12"):
        evaluated(text, pa.table(dict(x=values)), dict(x=kind), declaration="records", field="score", offset=10)


def test_hash_has_canonical_numeric_identity_and_preserves_nulls_order_and_salt():
    table = pa.table(dict(x=pa.array([1, 1, None], type=pa.float64()), name=["a", "b", "c"]))
    columns = dict(x="number", name="category")
    actual = evaluated("hash(x, salt='numbers')", table, columns).values.to_pylist()
    literal = evaluated("hash(1, salt='numbers')", table, columns).values.to_pylist()
    assert literal == [0.9771984298860562, 0.9771984298860562, 0.9771984298860562]
    assert actual == [literal[0], literal[0], None]
    assert all(0 <= value < 1 for value in literal)
    assert evaluated("hash(1.0, salt='numbers')", table, columns).values.to_pylist() == literal
    assert evaluated("hash(1, salt='other')", table, columns).values.to_pylist() != literal
    forward = evaluated("hash(x, name, salt='pair')", table, columns).values.to_pylist()
    backward = evaluated("hash(name, x, salt='pair')", table, columns).values.to_pylist()
    assert forward != backward and forward[-1] is None and backward[-1] is None
    assert forward == evaluated("hash(x, name, salt='pair')", table, columns).values.to_pylist()
    reference = pa.table(dict(x=pa.array([1.0, -2.5, 0.0], type=pa.float64())))
    assert evaluated("hash(x, salt='numbers')", reference, dict(x="number")).values.to_pylist() == [
        0.9771984298860562, 0.7170609916089675, 0.016864227814941124
    ]


def test_hash_positional_and_keyword_salts_have_the_same_literal_results():
    table = pa.table(dict(x=pa.array([1.0, -2.5, 0.0], type=pa.float64()), name=["a", "b", "c"]))
    columns = dict(x="number", name="category")
    expected = [0.9771984298860562, 0.7170609916089675, 0.016864227814941124]
    assert evaluated("hash(x, 'numbers')", table, columns).values.to_pylist() == expected
    assert evaluated("hash(x, salt='numbers')", table, columns).values.to_pylist() == expected
    assert evaluated("hash(x, name, 'pair')", table, columns).values.to_pylist() == evaluated(
        "hash(x, name, salt='pair')", table, columns
    ).values.to_pylist()
    assert evaluated("hash(x, 'literal key', salt='pair')", table, columns).values.to_pylist() == evaluated(
        "hash(x, 'literal key', 'pair')", table, columns
    ).values.to_pylist()
    assert evaluated("hash(x, 'numbers', salt='numbers')", table, columns).values.to_pylist() == evaluated(
        "hash(x, 'numbers', 'numbers')", table, columns
    ).values.to_pylist()


@pytest.mark.parametrize("text", [
    "hash()", "hash(x)", "hash('numbers')", "hash(x, truth)", "hash(x, lower('numbers'))",
    "hash(x, 3)", "hash(x, missing)", "hash(x, '')", "hash(x, salt=truth)",
    "hash(x, salt=3)", "hash(x, salt=missing)", "hash(x, salt='a', salt='b')",
])
def test_hash_salts_are_explicit_nonempty_text_literals_and_never_duplicate_keywords(text):
    for table in (TABLE, TABLE.slice(0, 0)):
        with pytest.raises(ValueError, match="hash|salt|keyword"):
            evaluated(text, table)


@pytest.mark.parametrize("function", ["text", "category"])
def test_promoted_finite_times_keep_exact_text_and_category_values(function):
    table = pa.table(dict(moment=pa.array([0, (1 << 53) + 1, (1 << 63) - 1, -(1 << 63)], type=pa.int64())))
    columns = dict(moment=dict(type="time", unit="turn"))
    expected = ["0", "9007199254740993", "9223372036854775807", "-9223372036854775808"]
    assert evaluated(function + "(moment)", table, columns, unit="turn").values.to_pylist() == expected
    assert evaluated(function + "(coalesce(moment, never))", table, columns, unit="turn").values.to_pylist() == expected
    assert evaluated(function + "(never)", table, columns, unit="turn").values.to_pylist() == ["never"] * 4


def test_promoted_finite_times_keep_hash_identity_and_never_has_an_explicit_token():
    table = pa.table(dict(moment=pa.array([0, (1 << 53) + 1, (1 << 63) - 1, -(1 << 63)], type=pa.int64())))
    columns = dict(moment=dict(type="time", unit="turn"))
    original = evaluated("hash(moment, salt='times')", table, columns, unit="turn").values.to_pylist()
    promoted = evaluated("hash(coalesce(moment, never), salt='times')", table, columns, unit="turn").values.to_pylist()
    assert promoted == original
    never = evaluated("hash(never, salt='times')", table, columns, unit="turn").values.to_pylist()
    named = evaluated("hash('never', salt='times')", table, columns, unit="turn").values.to_pylist()
    assert all(0 <= value < 1 for value in never) and never != named
    assert never == evaluated("hash(never, 'times')", table, columns, unit="turn").values.to_pylist()
    assert evaluated("flag(coalesce(moment, never))", table, columns, unit="turn").values.to_pylist() == [False, True, True, True]


@pytest.mark.parametrize("function", ["number", "flag"])
def test_never_cannot_be_coerced_to_a_finite_number_or_flag(function):
    with pytest.raises(ValueError, match="never"):
        evaluated(function + "(never)", unit="turn")


def test_row_and_neighbors_in_conditionals_read_the_original_ordered_scope():
    table = pa.table(dict(x=pa.array([10, 20, 30, 40, 50], type=pa.float64()), choose=[True, False, True, False, True]))
    columns = dict(x="number", choose="flag")
    assert evaluated("row() if choose else -1", table, columns, offset=100).values.to_pylist() == [100.0, -1.0, 102.0, -1.0, 104.0]
    result = evaluated("previous(x) if choose else 0", table, columns, ordered=True)
    assert result.values.to_pylist() == [None, 0.0, 20.0, 0.0, 40.0]
    result = evaluated("following(x) if choose else 0", table, columns, ordered=True)
    assert result.values.to_pylist() == [20.0, 0.0, 40.0, 0.0, None]


def test_coalesce_refuses_incompatible_types_instead_of_stringifying():
    with pytest.raises(ValueError):
        evaluated("coalesce(x, 'unknown')")
