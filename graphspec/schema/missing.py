import math
from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .fields import ColumnValueError

MISSING = Names("missing rule", True)


class Omit:
    version = 1
    fields = frozenset()
    required = ()

    @timed("missing.Omit")
    def __call__(self, values, parameters):
        return values


class Stop:
    version = 1
    fields = frozenset()
    required = ()

    @timed("missing.Stop")
    def __call__(self, values, parameters):
        if values.null_count:
            raise ColumnValueError(values.is_null().to_pylist().index(True) + 1, "the rule stop refuses a missing value")
        return values


class Value:
    version = 1
    fields = frozenset(("value",))
    required = ("value",)

    @timed("missing.Value")
    def __call__(self, values, parameters):
        return pc.fill_null(values, pa.scalar(parameters["value"], type=values.type))


@timed("missing.rule")
def rule(statement):
    if type(statement) is str:
        name, parameters = statement, {}
    elif isinstance(statement, Mapping) and type(statement.get("name")) is str:
        name = statement["name"]
        parameters = {key: value for key, value in statement.items() if key != "name"}
    else:
        raise ValueError("missing states a registered rule name or a mapping with name")
    entry = MISSING.named(name)
    if not callable(entry):
        raise ValueError("the missing rule %s is callable" % name)
    allowed = getattr(entry, "fields", None)
    if allowed is not None:
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError("the missing rule %s has no parameters %s" % (name, ", ".join(sorted(unknown))))
    for field in getattr(entry, "required", ()):
        if field not in parameters:
            raise ValueError("the missing rule %s requires %s" % (name, field))
    check = getattr(entry, "validate", None)
    if check is not None:
        check(parameters)
    return entry, parameters


@timed("missing.validate")
def validate(statement):
    if not isinstance(statement, Mapping):
        return
    if "nulls" in statement and not isinstance(statement["nulls"], (list, tuple)):
        raise ValueError("nulls states a list of values that mean missing")
    if "missing" in statement:
        rule(statement["missing"])


@timed("missing.normalize")
def normalize(values, statement):
    validate(statement)
    tokens = statement.get("nulls", ()) if isinstance(statement, Mapping) else ()
    if not tokens:
        return native(values)
    absent = []
    for row, scalar in enumerate(values, 1):
        try:
            value = scalar.as_py()
            absent.append(value is None or isinstance(value, float) and math.isnan(value) or any(
                value == token for token in tokens
            ))
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(row, "cannot compare missing tokens: %s" % error) from error
    return native(pc.if_else(pa.array(absent, type=pa.bool_()), None, values) if any(absent) else values)


@timed("missing.native")
def native(values):
    if isinstance(values, pa.ChunkedArray):
        return pa.chunked_array([native(chunk) for chunk in values.chunks], type=values.type)
    kind = values.type
    if pa.types.is_floating(kind):
        return pc.if_else(pc.fill_null(pc.is_nan(values), False), None, values)
    if pa.types.is_dictionary(kind):
        dictionary = native(values.dictionary)
        absent = pc.fill_null(pc.take(dictionary.is_null(), values.indices), True)
        indices = pc.if_else(absent, None, values.indices)
        return pa.DictionaryArray.from_arrays(indices, dictionary, ordered=kind.ordered)
    if pa.types.is_struct(kind) and kind.num_fields:
        return pa.StructArray.from_arrays(
            [native(values.field(index)) for index in range(kind.num_fields)], fields=list(kind), mask=values.is_null()
        )
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        start, stop = values.offsets[0].as_py(), values.offsets[-1].as_py()
        offsets = pc.subtract(values.offsets, pa.scalar(start, values.offsets.type))
        members = native(values.values.slice(start, stop - start))
        array = pa.ListArray if pa.types.is_list(kind) else pa.LargeListArray
        return array.from_arrays(offsets, members, type=kind, mask=values.is_null())
    if pa.types.is_fixed_size_list(kind):
        if kind.list_size == 0:
            return values
        members = native(values.values.slice(values.offset * kind.list_size, len(values) * kind.list_size))
        return pa.FixedSizeListArray.from_arrays(members, type=kind, mask=values.is_null())
    return values


@timed("missing.compatible")
def compatible(source, target):
    if pa.types.is_struct(source) and pa.types.is_struct(target):
        lost = set(source.names) - set(target.names)
        if lost:
            raise ValueError("a missing replacement would discard record fields %s" % ", ".join(sorted(lost)))
        for field in source:
            compatible(field.type, target.field(field.name).type)
    lists = (pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list)
    if any(check(source) for check in lists) and any(check(target) for check in lists):
        compatible(source.value_type, target.value_type)


@timed("missing.resolve")
def resolve(values, statement, converter):
    from .columns import held

    validate(statement)
    first = values.is_null().to_pylist().index(True) + 1 if values.null_count else None
    if not isinstance(statement, Mapping) or "missing" not in statement:
        if first is not None:
            raise ColumnValueError(first, "has a missing value without a missing rule")
        return values
    entry, parameters = rule(statement["missing"])
    if isinstance(entry, Value):
        try:
            replacement = converter(pa.array([parameters["value"]]))
            if replacement.type != values.type:
                compatible(replacement.type, values.type)
                replacement = replacement.cast(values.type, safe=True)
        except Exception as error:
            message = "has an invalid missing replacement: %s" % error
            if first is not None:
                raise ColumnValueError(first, message) from error
            raise ValueError(message) from error
        parameters = dict(value=replacement[0].as_py())
    result = entry(values, parameters)
    if not isinstance(result, (pa.Array, pa.ChunkedArray)) or len(result) != len(values):
        raise ValueError("missing rule returns an Arrow array of the same length")
    if result.type != values.type:
        raise ValueError("missing rule returns values of type %s" % values.type)
    before = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    result = normalize(result, {})
    result = result.combine_chunks() if isinstance(result, pa.ChunkedArray) else result
    present = before.is_valid()
    if not before.filter(present).equals(result.filter(present)):
        for index, (original, replacement) in enumerate(zip(before, result, strict=True), 1):
            if original.is_valid and not original.equals(replacement):
                raise ColumnValueError(index, "missing rule changes a value that is present")
    return held(result, statement)


@timed("missing.apply")
def apply(values, statement, column, offset, converter):
    try:
        return resolve(values, statement, converter)
    except ColumnValueError as error:
        raise ValueError("column %s row %d: %s" % (column, offset + error.row, error.detail)) from error
    except Exception as error:
        raise ValueError("column %s: %s" % (column, error)) from error


MISSING.register("omit", Omit())
MISSING.register("stop", Stop())
MISSING.register("value", Value())
