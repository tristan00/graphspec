import bisect
import json
import math
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra import hashing
from graphspec.infra.timing import timed
from graphspec.schema.fields import ColumnValueError

from .expression_types import FUNCTIONS, NEVER_TIME, Function, Kind, array, call, common, datatype
from .streams import checked

NUMBERS = ("number", "ordered")


class Numeric:
    @timed("expression.Numeric")
    def __init__(self, name):
        self.name = name

    @timed("expression.Numeric.call")
    def __call__(self, *values):
        if self.name == "round" and len(values) == 2 and values[1] is not None \
                and not isinstance(values[1], (pa.Array, pa.ChunkedArray)):
            if type(values[1]) not in (int, float) or not math.isfinite(values[1]) or values[1] != int(values[1]):
                raise ValueError("round digits is a whole number")
        rows = [array(value).to_pylist() for value in values]
        result = []
        for row, arguments in enumerate(zip(*rows, strict=True), 1):
            if any(value is None for value in arguments):
                result.append(None)
                continue
            try:
                if self.name == "least":
                    value = min(arguments)
                elif self.name == "greatest":
                    value = max(arguments)
                elif self.name == "round":
                    digits = 0 if len(arguments) == 1 else arguments[1]
                    if not math.isfinite(digits) or digits != int(digits):
                        raise ValueError("round digits is a whole number")
                    value = round(arguments[0], int(digits))
                elif self.name == "abs":
                    value = abs(arguments[0])
                else:
                    value = getattr(math, self.name)(arguments[0])
                if not math.isfinite(value):
                    raise ValueError("%s produces a finite number" % self.name)
                result.append(float(value))
            except (ValueError, OverflowError, TypeError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.float64())


@timed("expression.bin")
def binned(values, edges):
    if edges is None:
        return pa.nulls(len(array(values)), pa.float64())
    if not isinstance(edges, list) or any(type(value) not in (int, float) or not math.isfinite(value) for value in edges) \
            or any(one >= other for one, other in zip(edges, edges[1:])):
        raise ValueError("bin edges is a literal list of strictly increasing finite numbers")
    return pa.array([None if value is None else float(bisect.bisect_right(edges, value))
                     for value in array(values).to_pylist()], pa.float64())


@timed("expression.bucket")
def bucket(values, widths):
    if widths is not None and not isinstance(widths, (pa.Array, pa.ChunkedArray)) and (type(widths) not in (int, float)
                                                               or not math.isfinite(widths) or widths <= 0):
        raise ValueError("bucket width is positive and finite")
    result = []
    for row, (value, width) in enumerate(zip(array(values), array(widths), strict=True), 1):
        value, width = value.as_py(), width.as_py()
        if value is None or width is None:
            result.append(None)
        elif not math.isfinite(width) or width <= 0:
            raise ColumnValueError(row, "bucket width is positive and finite")
        else:
            result.append(float(math.floor(value / width)))
    return pa.array(result, pa.float64())


@timed("expression.coalesced_kind")
def coalesced_kind(arguments, keywords):
    kind = Kind("missing")
    for argument in arguments:
        kind = common(kind, argument.kind)
    return kind


@timed("expression.coalesce")
def coalesce(*values):
    kind = coalesced_kind(call().arguments, {})
    values = [array(value, type=datatype(kind)) for value in values]
    return pc.coalesce(*values)


class Cast:
    @timed("expression.Cast")
    def __init__(self, name):
        self.name = name

    @timed("expression.Cast.call")
    def __call__(self, values):
        result = []
        for row, value in enumerate(array(values).to_pylist(), 1):
            if value is None:
                result.append(None)
                continue
            try:
                if call().arguments[0].kind.name == "time" and value == NEVER_TIME:
                    if self.name not in ("category", "text"):
                        raise ValueError("never cannot be converted to a finite number or flag")
                    value = "never"
                elif self.name == "number":
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("number produces a finite value")
                elif self.name == "flag":
                    if type(value) is str:
                        if value not in ("true", "false"):
                            raise ValueError("flag text is true or false")
                        value = value == "true"
                    elif type(value) in (bool, int, float, Decimal):
                        value = value != 0
                    else:
                        raise ValueError("flag reads a number, flag or true/false text")
                elif isinstance(value, (float, Decimal)) and value == int(value):
                    value = str(int(value))
                elif type(value) in (str, int, float, bool):
                    value = str(value)
                else:
                    raise ValueError("%s reads a scalar value" % self.name)
                result.append(value)
            except (ValueError, OverflowError, TypeError) as error:
                raise ColumnValueError(row, str(error)) from error
        kind = Kind(self.name)
        return pa.array(result, datatype(kind))


@timed("expression.vector_kind")
def vector_kind(arguments, keywords):
    return Kind("vector", width=len(arguments))


@timed("expression.point_kind")
def point_kind(arguments, keywords):
    return Kind("point", width=len(arguments))


@timed("expression.vector")
def vector(*values):
    columns = [array(value).to_pylist() for value in values]
    checked(call().rows * len(values) * 32, "constructing expression vectors")
    result = [None if any(value is None for value in row) else list(row) for row in zip(*columns, strict=True)]
    return pa.array(result, pa.list_(pa.float64(), len(values)))


@timed("expression.row")
def row():
    context = call()
    values = context.indices if context.indices is not None else range(context.offset, context.offset + context.rows)
    return pa.array(values, pa.float64())


@timed("expression.hash")
def hashed(*values, salt):
    if type(salt) is not str or not salt:
        raise ValueError("hash states a nonempty literal salt")
    columns = []
    for value, argument in zip(values, call().arguments, strict=True):
        column = array(value).to_pylist()
        if any(isinstance(item, (list, dict)) for item in column if item is not None):
            raise ValueError("hash reads scalar keys")
        if argument.kind.name in ("number", "ordered"):
            column = [None if item is None else float(item) for item in column]
        elif argument.kind.name == "time":
            column = [None if item is None else {"time": "never"} if item == NEVER_TIME else int(item) for item in column]
        columns.append(np.asarray([hashing.token(json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False))
                                   for item in column], dtype=np.uint64))
    checked(call().rows * (len(columns) + 2) * 32, "hashing expression rows")
    return pa.array(hashing.unit(0, salt, *columns), pa.float64())


@timed("expression.same_kind")
def same_kind(arguments, keywords):
    return arguments[0].kind


class Shift:
    @timed("expression.Shift")
    def __init__(self, direction):
        self.direction = direction

    @timed("expression.Shift.call")
    def __call__(self, values):
        if not call().ordered:
            raise ValueError("previous and following require an explicitly ordered context")
        values = array(values)
        if call().neighbors is None:
            positions = [index + self.direction if 0 <= index + self.direction < len(values) else None
                         for index in range(len(values))]
        else:
            positions = call().neighbors["previous" if self.direction < 0 else "following"]
            if len(positions) != len(values):
                raise ValueError("ordered neighbor positions align with the complete ordered table")
        return pc.take(values, pa.array(positions, pa.int64()))


for _name in ("abs", "exp", "log", "sqrt", "floor", "ceil"):
    FUNCTIONS.register(_name, Function(Numeric(_name), (NUMBERS,), "number"))
FUNCTIONS.register("round", Function(Numeric("round"), (NUMBERS, NUMBERS), "number", minimum=1))
for _name in ("least", "greatest"):
    FUNCTIONS.register(_name, Function(Numeric(_name), (NUMBERS,), "number", minimum=2, maximum=None))
FUNCTIONS.register("bin", Function(binned, (NUMBERS, "list"), "number"))
FUNCTIONS.register("bucket", Function(bucket, (NUMBERS, NUMBERS), "number"))
FUNCTIONS.register("coalesce", Function(coalesce, ("any",), coalesced_kind, minimum=2, maximum=None, nulls="read"))
for _name in ("number", "flag", "category", "text"):
    FUNCTIONS.register(_name, Function(Cast(_name), ("any",), _name))
FUNCTIONS.register("vector", Function(vector, (NUMBERS,), vector_kind, minimum=1, maximum=None))
FUNCTIONS.register("point", Function(vector, (NUMBERS,), point_kind, minimum=2, maximum=3))
FUNCTIONS.register("row", Function(row, (), "number"))
FUNCTIONS.register("hash", Function(hashed, ("any",), "number", minimum=1, maximum=None, keywords=("salt",)))
FUNCTIONS.register("previous", Function(Shift(-1), ("any",), same_kind, nulls="read", version=2))
FUNCTIONS.register("following", Function(Shift(1), ("any",), same_kind, nulls="read", version=2))
