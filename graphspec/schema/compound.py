from bisect import bisect_right
from collections.abc import Mapping
from functools import partial

import pyarrow as pa

from graphspec.infra.timing import timed

from .fields import ColumnValueError


@timed("compound.separator")
def separator(parameters):
    if "separator" in parameters and (type(parameters["separator"]) is not str or not parameters["separator"]):
        raise ValueError("separator is a nonempty string")


@timed("compound.rows")
def rows(values, parameters):
    kind = values.type
    if pa.types.is_dictionary(kind):
        kind = kind.value_type
    if (pa.types.is_string(kind) or pa.types.is_large_string(kind)) and "separator" not in parameters:
        raise ValueError("a list, vector or point stored as text requires separator")
    result = []
    for index, value in enumerate(values.to_pylist(), 1):
        if value is None:
            result.append(None)
        elif type(value) is str:
            if "separator" not in parameters:
                raise ColumnValueError(index, "a list, vector or point stored as text requires separator")
            result.append(value.split(parameters["separator"]))
        elif isinstance(value, (list, tuple)):
            result.append(list(value))
        else:
            raise ColumnValueError(index, "a list, vector or point holds a list of values")
    return result


@timed("compound.members")
def members(values, held, statement):
    from .columns import completed, convert, prepared
    from .missing import resolve

    offsets, flattened = [0], []
    for row in held:
        if row is not None:
            flattened.extend(row)
        offsets.append(len(flattened))
    kind = values.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        member_type = kind.value_type
    elif pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_dictionary(kind):
        member_type = pa.string()
    else:
        member_type = pa.null()
    try:
        converted = completed(convert(prepared(pa.array(flattened, type=member_type), statement), statement), statement)
        if isinstance(statement, Mapping) and "missing" in statement:
            converted = resolve(converted, statement, partial(convert, given=statement))
    except ColumnValueError as error:
        raise ColumnValueError(bisect_right(offsets, error.row - 1), "list member %s" % error) from error
    if isinstance(converted, pa.ChunkedArray):
        converted = converted.combine_chunks()
    return converted, offsets


class List:
    version = 1
    gives = "list"
    fields = frozenset(("of", "separator"))
    required = ("of",)

    @timed("compound.List.validate")
    def validate(self, parameters):
        separator(parameters)
        member = parameters["of"]
        if not (type(member) is str and member) and not (
            isinstance(member, Mapping) and type(member.get("type")) is str and member["type"]
        ):
            raise ValueError("list of states a registered column type or a mapping with type")

    @timed("compound.List")
    def __call__(self, values, parameters):
        held = rows(values, parameters)
        converted, offsets = members(values, held, parameters["of"])
        return pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()), converted, mask=pa.array([row is None for row in held], type=pa.bool_())
        )


class Vector:
    version = 1
    gives = "vector"
    fields = frozenset(("width", "separator"))
    required = ("width",)

    @timed("compound.Vector.validate")
    def validate(self, parameters):
        separator(parameters)
        if type(parameters["width"]) is not int or parameters["width"] < 1:
            raise ValueError("vector width is a positive whole number")

    @timed("compound.Vector")
    def __call__(self, values, parameters):
        held, width = rows(values, parameters), parameters["width"]
        for index, row in enumerate(held, 1):
            if row is not None and len(row) != width:
                raise ColumnValueError(index, "%s width is %d, found %d values" % (self.gives, width, len(row)))
        converted, _ = members(values, [row if row is not None else [None] * width for row in held], "number")
        return pa.FixedSizeListArray.from_arrays(
            converted, width, mask=pa.array([row is None for row in held], type=pa.bool_())
        )


class Point(Vector):
    version = 1
    gives = "point"

    @timed("compound.Point.validate")
    def validate(self, parameters):
        separator(parameters)
        if type(parameters["width"]) is not int or parameters["width"] not in (2, 3):
            raise ValueError("point width is 2 or 3")


class Nested:
    version = 1
    gives = "nested"
    fields = frozenset()
    required = ()

    @timed("compound.Nested.validate")
    def validate(self, parameters):
        return None

    @timed("compound.Nested")
    def __call__(self, values, parameters):
        if pa.types.is_null(values.type):
            return pa.nulls(len(values), type=pa.struct([]))
        if not pa.types.is_struct(values.type):
            for index, value in enumerate(values.to_pylist(), 1):
                if value is not None:
                    raise ColumnValueError(index, "nested holds a record, whose fields may be records and lists")
            raise ValueError("nested holds records as an Arrow struct")
        return values
