import pyarrow as pa
import pyarrow.compute as pc
import pytest

from graphspec import MISSING, READERS, TYPES, Builder, BuildPlan, source
from graphspec.schema.columns import convert
from graphspec.schema.missing import apply, normalize, validate
from graphspec.schema.temporal import context

pytestmark = pytest.mark.extended
PLAN = BuildPlan(1 << 20, 1)


def filled(values, statement, offset=0):
    typed = convert(normalize(values, statement), statement)
    return apply(typed, statement, "value", offset, lambda raw: convert(raw, statement))


def test_list_members_are_converted_without_losing_empty_lists_or_missing_rows():
    values = pa.array([["1", "2"], [], None, ["3", None]])
    result = convert(values, dict(type="list", of="number"))
    assert result.type == pa.list_(pa.float64())
    assert result.to_pylist() == [[1.0, 2.0], [], None, [3.0, None]]
    assert convert(pa.array([], type=pa.list_(pa.int32())), dict(type="list", of="number")).type == result.type


def test_declared_member_missing_rules_apply_and_report_the_outer_source_row(tmp_path, monkeypatch):
    values = pa.array([[], ["1", None], None, ["NA", "2"]])
    member = dict(type="number", nulls=["NA"], missing=dict(name="value", value="0"))
    result = convert(values, dict(type="list", of=member))
    assert result.to_pylist() == [[], [1.0, 0.0], None, [0.0, 2.0]]
    assert convert(values, dict(type="list", of=dict(member, missing="omit"))).to_pylist() == [
        [], [1.0, None], None, [None, 2.0]
    ]

    def reader(place, columns, parameters):
        yield pa.table(dict(value=pa.array([[], ["1"]], type=values.type)))
        yield pa.table(dict(value=values))

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    READERS.register("member_missing", reader)
    declaration = source("lists", "member_missing", dict(value=dict(type="list", of=dict(member, missing="stop"))))
    with pytest.raises(ValueError, match="source records.*column value row 4.*stop"):
        list(Builder(dict(records=declaration), tmp_path).read("records", PLAN))


def test_nested_records_and_lists_of_records_preserve_recursive_content():
    records = pa.array([dict(name="a", child=dict(keys=[1, 2])), None])
    assert convert(records, "nested").equals(records)
    values = pa.array([[dict(name="a", child=dict(keys=[1, 2]))], [], None])
    assert convert(values, dict(type="list", of="nested")).equals(values)
    recursive = pa.array([[["1", "2"], []], None])
    result = convert(recursive, dict(type="list", of=dict(type="list", of="number")))
    assert result.to_pylist() == [[[1.0, 2.0], []], None]
    assert result.type == pa.list_(pa.list_(pa.float64()))


@pytest.mark.parametrize("name,width", [("vector", 4), ("point", 2), ("point", 3)])
def test_vectors_and_points_have_fixed_width_float_values(name, width):
    values = pa.array([list(range(width)), None, [None] * width])
    result = convert(values, dict(type=name, width=width))
    assert result.type == pa.list_(pa.float64(), width)
    assert result.to_pylist() == [[float(value) for value in range(width)], None, [None] * width]
    assert len(result) == 3


@pytest.mark.parametrize("statement", [dict(type="list", of="number"), dict(type="vector", width=2)])
def test_text_composites_require_an_explicit_separator_even_when_empty(statement):
    for values in (pa.array(["1|2"]), pa.array([], type=pa.string())):
        with pytest.raises(ValueError, match="separator"):
            convert(values, statement)
    result = convert(pa.array(["1|2", None]), dict(statement, separator="|"))
    assert result.to_pylist() == [[1.0, 2.0], None]


@pytest.mark.parametrize("reader", ["csv", "delimited"])
def test_delimited_list_members_use_declared_scalar_types_and_numeric_missing_tokens(tmp_path, reader):
    place = tmp_path / "lists.csv"
    place.write_text(
        "turns,flags,amounts,filled,groups\n1|2,true|0,0|1,0|1,1|2;3|4\n3|4,false|1,2|0,2|0,5;6\n",
        encoding="utf-8",
    )
    clock = dict(type="time", unit="turn")
    columns = dict(
        turns=dict(type="list", of=clock, separator="|"),
        flags=dict(type="list", of="flag", separator="|"),
        amounts=dict(type="list", of=dict(type="number", nulls=[0], missing="omit"), separator="|"),
        filled=dict(type="list", of=dict(type="number", nulls=[0], missing=dict(name="value", value=7)), separator="|"),
        groups=dict(type="list", of=dict(type="list", of=clock, separator="|"), separator=";"),
    )
    parameters = dict(delimiter=",", names=None) if reader == "delimited" else {}
    declaration = source(place, reader, columns, **parameters)
    result = pa.concat_tables(Builder(dict(records=declaration), tmp_path).read("records", PLAN))
    assert result.to_pylist() == [
        dict(turns=[1, 2], flags=[True, False], amounts=[None, 1.0], filled=[7.0, 1.0], groups=[[1, 2], [3, 4]]),
        dict(turns=[3, 4], flags=[False, True], amounts=[2.0, None], filled=[2.0, 7.0], groups=[[5], [6]]),
    ]
    with context("turn"), pytest.raises(ValueError, match="format"):
        convert(pa.array(["1|2"]), columns["turns"])


def test_delimited_custom_list_member_types_keep_the_raw_values_and_null_tokens(tmp_path, monkeypatch):
    seen = []

    class Shift:
        version = 1
        gives = "number"

        def __call__(self, values, parameters):
            seen.append((values.type, values.to_pylist()))
            return pc.add(values.cast(pa.float64()), 1.0)

    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})
    TYPES.register("raw_member_shift", Shift())
    place = tmp_path / "custom.csv"
    place.write_text("values\n01|0\n", encoding="utf-8")
    declaration = source(
        place, "csv", dict(values=dict(type="list", of=dict(type="raw_member_shift", nulls=[1]), separator="|"))
    )
    result = pa.concat_tables(Builder(dict(records=declaration), tmp_path).read("records", PLAN))
    assert result.to_pylist() == [dict(values=[2.0, 1.0])]
    assert seen == [(pa.string(), ["01", "0"])]


@pytest.mark.parametrize(
    "statement,message",
    [
        (dict(type="list"), "of"),
        (dict(type="list", of=1), "of"),
        (dict(type="list", of="number", separator=""), "separator"),
        (dict(type="vector"), "width"),
        (dict(type="vector", width=True), "width"),
        (dict(type="vector", width=0), "width"),
        (dict(type="point", width=4), "width"),
        (dict(type="nested", separator="|"), "separator"),
    ],
)
def test_compound_parameters_are_validated_on_empty_arrays(statement, message):
    with pytest.raises(ValueError, match=message):
        convert(pa.array([], type=pa.null()), statement)


def test_bad_width_record_and_list_member_report_their_source_row():
    with pytest.raises(ValueError, match="row 2.*width.*3.*2"):
        convert(pa.array([[1, 2, 3], [1, 2]]), dict(type="vector", width=3))
    with pytest.raises(ValueError, match="row 2.*record"):
        convert(pa.array([None, "not a record"]), "nested")
    with pytest.raises(ValueError, match="row 3.*member"):
        convert(pa.array([[], ["1", "2"], ["bad"]]), dict(type="list", of="number"))


def test_list_uses_registered_member_types(monkeypatch):
    class Doubled:
        version = 1
        gives = "number"

        def __call__(self, values, parameters):
            return pc.multiply(values.cast(pa.float64()), parameters["factor"])

    monkeypatch.setattr(TYPES, "held", {name: TYPES.named(name) for name in TYPES.names()})
    TYPES.register("doubled_member", Doubled())
    result = convert(pa.array([[1, 2], None]), dict(type="list", of=dict(type="doubled_member", factor=2)))
    assert result.to_pylist() == [[2.0, 4.0], None]


def test_native_null_nan_and_explicit_tokens_normalize_before_number_conversion():
    result = filled(pa.array([1.0, float("nan"), None, -1]), dict(type="number", nulls=[-1], missing="omit"))
    assert result.to_pylist() == [1.0, None, None, None]
    result = filled(pa.array(["2", "NA", "?", None]), dict(type="number", nulls=["NA", "?"], missing="omit"))
    assert result.to_pylist() == [2.0, None, None, None]
    assert len(result) == 4
    result = filled(pa.array([[1.0, float("nan")]]), dict(type="vector", width=2))
    assert result.to_pylist() == [[1.0, None]]
    dictionary = pa.array([1.0, float("nan"), None]).dictionary_encode()
    assert normalize(dictionary, {}).to_pylist() == [1.0, None, None]


def test_native_missing_normalization_preserves_scalar_range_and_token_error_rows():
    values = pa.array([0, (1 << 63) - 1], type=pa.timestamp("s"))
    assert normalize(values, {}).equals(values)
    with pytest.raises(ValueError, match="row 2.*missing tokens"):
        normalize(values, dict(nulls=["unknown"]))


@pytest.mark.parametrize("kind", [pa.list_(pa.float64()), pa.large_list(pa.float64()), pa.list_(pa.float64(), 2)])
def test_native_composite_missing_values_keep_slice_offsets_and_parent_masks(kind):
    values = pa.array([[9, 9], [float("nan"), 2], None, [3, float("nan")], [8, 8]], type=kind).slice(1, 3)
    result = normalize(values, {})
    assert result.type == kind
    assert result.is_null().equals(values.is_null())
    assert result.to_pylist() == [[None, 2.0], None, [3.0, None]]
    record = pa.StructArray.from_arrays([values], names=["members"], mask=pa.array([False, True, False]))
    nested = normalize(pa.chunked_array([record.slice(0, 1), record.slice(1)]), {})
    assert nested.to_pylist() == [dict(members=[None, 2.0]), None, dict(members=[3.0, None])]


def test_dictionary_missing_values_keep_the_dictionary_and_make_indices_null():
    dictionary = pa.array(["a", None, "b"])
    values = pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 2], type=pa.int8()), dictionary)
    result = normalize(values, {})
    assert result.type == values.type
    assert result.dictionary.equals(dictionary)
    assert result.indices.to_pylist() == [0, None, None, 2]
    assert result.to_pylist() == ["a", None, None, "b"]


@pytest.mark.parametrize("missing,message", [(None, "without a missing rule"), ("stop", "stop")])
def test_missing_without_a_rule_or_with_stop_names_column_and_global_row(missing, message):
    statement = dict(type="number")
    if missing is not None:
        statement["missing"] = missing
    with pytest.raises(ValueError, match="column value row 12.*" + message):
        filled(pa.array([1.0, None]), statement, 10)


def test_value_replacements_are_converted_with_the_declared_column_type():
    result = filled(pa.array([1.0, None]), dict(type="number", missing=dict(name="value", value="4.5")))
    assert result.to_pylist() == [1.0, 4.5]
    statement = dict(type="vector", width=2, missing=dict(name="value", value=["3", "4"]))
    result = filled(pa.array([[1, 2], None]), statement)
    assert result.type == pa.list_(pa.float64(), 2)
    assert result.to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="column value row 12.*invalid missing replacement"):
        filled(pa.array([1.0, None]), dict(type="number", missing=dict(name="value", value="bad")), 10)


@pytest.mark.parametrize("values,statement,replacement", [
    (pa.array([dict(a=1), None]), dict(type="nested"), dict(a=2, b=3)),
    (pa.array([[dict(inner=dict(a=1))], None]), dict(type="list", of="nested"), [dict(inner=dict(a=2, b=3))]),
    (pa.nulls(2), dict(type="nested"), dict(b=3)),
])
def test_missing_replacements_never_discard_record_fields(values, statement, replacement):
    statement = dict(statement, missing=dict(name="value", value=replacement))
    with pytest.raises(ValueError, match="invalid missing replacement.*discard record fields b"):
        filled(values, statement)


def test_nested_missing_replacement_can_leave_existing_fields_null():
    values = pa.array([dict(a=1, b=2), None])
    result = filled(values, dict(type="nested", missing=dict(name="value", value=dict(a=3))))
    assert result.to_pylist() == [dict(a=1, b=2), dict(a=3, b=None)]


def test_registered_missing_rule_receives_typed_values_and_parameters(monkeypatch):
    seen = []

    class Fill:
        version = 1

        def __call__(self, values, parameters):
            seen.append((values.type, values.to_pylist(), parameters))
            return pc.fill_null(values, parameters["replacement"])

    monkeypatch.setattr(MISSING, "held", {name: MISSING.named(name) for name in MISSING.names()})
    MISSING.register("custom_fill", Fill())
    result = filled(
        pa.array(["2", "unknown"]),
        dict(type="number", nulls=["unknown"], missing=dict(name="custom_fill", replacement=9.0)),
    )
    assert result.to_pylist() == [2.0, 9.0]
    assert seen == [(pa.float64(), [2.0, None], dict(replacement=9.0))]
    result = filled(pa.array([2.0, None]), dict(type="number", missing=dict(name="custom_fill", replacement=float("nan"))))
    assert result.to_pylist() == [2.0, None]


@pytest.mark.parametrize(
    "statement,message",
    [
        (dict(nulls="NA"), "nulls"),
        (dict(missing=None), "missing"),
        (dict(missing={}), "name"),
        (dict(missing=dict(name="value")), "value"),
        (dict(missing=dict(name="omit", value=1)), "parameters"),
    ],
)
def test_missing_parameters_are_explicit_even_without_missing_data(statement, message):
    with pytest.raises(ValueError, match=message):
        validate(statement)


def test_a_missing_rule_cannot_change_present_values_or_the_column_length(monkeypatch):
    class Changes:
        version = 1

        def __call__(self, values, parameters):
            return pa.array([9.0, 9.0]) if parameters["same_length"] else pa.array([9.0])

    monkeypatch.setattr(MISSING, "held", {name: MISSING.named(name) for name in MISSING.names()})
    MISSING.register("changes", Changes())
    for same_length, message in ((True, "present"), (False, "length")):
        statement = dict(type="number", missing=dict(name="changes", same_length=same_length))
        with pytest.raises(ValueError, match=message):
            filled(pa.array([1.0, None]), statement)


@pytest.mark.parametrize("values,statement,replacement", [
    (pa.array(["allowed", None]), dict(type="category", values=["allowed"]), "outside"),
    (pa.array(["first", None]), dict(type="ordered", values=["first", "last"]), 0.5),
    (pa.array(["first", None]), dict(type="ordered", values=["first", "last"]), 2.0),
    (pa.array([1.0, None]), dict(type="number"), float("inf")),
    (pa.array([["allowed"], None]), dict(type="list", of=dict(type="category", values=["allowed"])), ["outside"]),
])
def test_registered_missing_rules_preserve_the_declared_value_domain(monkeypatch, values, statement, replacement):
    class Fill:
        version = 1

        def __call__(self, values, parameters):
            return pc.fill_null(values, pa.scalar(parameters["value"], type=values.type))

    monkeypatch.setattr(MISSING, "held", {name: MISSING.named(name) for name in MISSING.names()})
    MISSING.register("domain_fill", Fill())
    with pytest.raises(ValueError):
        filled(values, dict(statement, missing=dict(name="domain_fill", value=replacement)))


def test_source_missing_errors_name_source_column_and_row_across_batches(tmp_path, monkeypatch):
    def reader(place, columns, parameters):
        yield pa.table(dict(value=[1.0, 2.0]))
        yield pa.table(dict(value=[3.0, None]))

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    READERS.register("missing_batches", reader)
    builder = Builder(dict(records=source("values", "missing_batches", dict(value="number"))), tmp_path)
    with pytest.raises(ValueError, match="source records.*column value row 4.*missing"):
        list(builder.read("records", PLAN))


def test_source_keeps_typed_vectors_and_omitted_values_in_their_original_rows(tmp_path, monkeypatch):
    def reader(place, columns, parameters):
        yield pa.table(dict(key=[1, 2], value=[[1, 2], None]))

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    READERS.register("vector_rows", reader)
    declaration = source(
        "vectors", "vector_rows", dict(key="identifier", value=dict(type="vector", width=2, missing="omit"))
    )
    batches = list(Builder(dict(records=declaration), tmp_path).read("records", PLAN))
    result = pa.concat_tables(batches)
    assert result.schema.field("value").type == pa.list_(pa.float64(), 2)
    assert result.to_pylist() == [dict(key="1", value=[1.0, 2.0]), dict(key="2", value=None)]
