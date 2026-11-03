import math
from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.temporal import whole

from .expression_types import FUNCTIONS, Function, Kind, datatype
from .streams import checked

POINTS = pa.list_(pa.struct([("tau", pa.int64()), ("series", pa.string()), ("y", pa.float64())]))
SHARES = pa.list_(pa.struct([("name", pa.string()), ("weight", pa.float64())]))
NUMBERS = ("number", "ordered", "duration")
NAMES = ("category", "identifier", "text")


@timed("aggregates.absent")
def absent(*values):
    return not len(values[0]) or any(value.null_count for value in values if value is not None)


@timed("aggregates.scalar")
def scalar(values):
    values = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    return pa.ListArray.from_arrays(pa.array([0, len(values)], type=pa.int32()), values)[0]


@timed("aggregates.ordered")
def ordered(values, by):
    if by is None:
        return values
    if len(values) != len(by):
        raise ValueError("an aggregate orders values by one value for each row")
    checked(values.nbytes + by.nbytes + len(values) * 32, "aggregate ordering")
    keys = pa.table(dict(value=by, row=pa.array(range(len(values)), type=pa.int64())))
    indices = pc.sort_indices(keys, sort_keys=[("value", "ascending"), ("row", "ascending")])
    return pc.take(values, indices)


@timed("aggregates.count")
def count(values):
    return pa.scalar(None if values.null_count else float(len(values)), type=pa.float64())


@timed("aggregates.reduced")
def reduced(values, operation, name):
    if absent(values):
        return pa.scalar(None, type=pa.float64())
    result = operation(values, skip_nulls=False)
    if not math.isfinite(result.as_py()):
        raise ValueError("%s gives a finite value" % name)
    return result.cast(pa.float64())


@timed("aggregates.total")
def total(values):
    return reduced(values, pc.sum, "sum")


@timed("aggregates.average")
def average(values):
    return reduced(values, pc.mean, "mean")


@timed("aggregates.deviation")
def deviation(values):
    return reduced(values, pc.stddev, "deviation")


@timed("aggregates.minimum")
def minimum(values):
    return pa.scalar(None, type=values.type) if absent(values) else pc.min(values, skip_nulls=False)


@timed("aggregates.maximum")
def maximum(values):
    return pa.scalar(None, type=values.type) if absent(values) else pc.max(values, skip_nulls=False)


@timed("aggregates.exists")
def exists(values):
    return pa.scalar(None, type=pa.bool_()) if absent(values) else pc.any(values, skip_nulls=False)


@timed("aggregates.every")
def every(values):
    return pa.scalar(None, type=pa.bool_()) if absent(values) else pc.all(values, skip_nulls=False)


@timed("aggregates.key")
def key(value):
    if isinstance(value, list):
        return (list, tuple(key(member) for member in value))
    if isinstance(value, dict):
        return (dict, tuple((name, key(member)) for name, member in value.items()))
    return (type(value), value)


@timed("aggregates.distinct")
def distinct(values):
    if absent(values):
        return pa.scalar(None, type=pa.list_(values.type))
    checked(values.nbytes * 4 + len(values) * 96, "aggregate distinct values")
    seen, indices = set(), []
    for index, value in enumerate(values):
        try:
            token = key(value.as_py())
            if token not in seen:
                seen.add(token)
                indices.append(index)
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(index + 1, "distinct cannot read this value: %s" % error) from error
    return scalar(pc.take(values, pa.array(indices, type=pa.int64())))


@timed("aggregates.listed")
def listed(values, by=None):
    return pa.scalar(None, type=pa.list_(values.type)) if absent(values, by) else scalar(ordered(values, by))


@timed("aggregates.first")
def first(values, by=None):
    return pa.scalar(None, type=values.type) if absent(values, by) else ordered(values, by)[0]


@timed("aggregates.last")
def last(values, by=None):
    return pa.scalar(None, type=values.type) if absent(values, by) else ordered(values, by)[-1]


@timed("aggregates.points")
def points(tau, series, y):
    if absent(tau, series, y):
        return pa.scalar(None, type=POINTS)
    if len(tau) != len(series) or len(tau) != len(y):
        raise ValueError("points reads aligned tau, series and y values")
    checked(tau.nbytes + series.nbytes + y.nbytes + len(tau) * 96, "aggregate points")
    times, names, amounts = [], [], []
    for row, (moment, name, value) in enumerate(zip(tau, series, y, strict=True), 1):
        moment, name, value = moment.as_py(), name.as_py(), value.as_py()
        try:
            moment = whole(moment)
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(row, "points tau is a whole int64 count") from error
        if not -(1 << 63) <= moment < (1 << 63):
            raise ColumnValueError(row, "points tau is a whole int64 count")
        if type(name) is not str:
            raise ColumnValueError(row, "points series is a name")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ColumnValueError(row, "points y is a finite number")
        times.append(moment)
        names.append(name)
        amounts.append(float(value))
    return scalar(pa.StructArray.from_arrays(
        [pa.array(times, pa.int64()), pa.array(names, pa.string()), pa.array(amounts, pa.float64())],
        fields=list(POINTS.value_type),
    ))


@timed("aggregates.shares")
def shares(names, weights):
    if absent(names, weights):
        return pa.scalar(None, type=SHARES)
    if len(names) != len(weights):
        raise ValueError("shares reads one weight for each name")
    checked(names.nbytes * 4 + weights.nbytes + len(names) * 96, "aggregate shares")
    totals = {}
    for row, (name, weight) in enumerate(zip(names, weights, strict=True), 1):
        name, weight = name.as_py(), weight.as_py()
        if type(name) is not str:
            raise ColumnValueError(row, "shares names each class as text")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ColumnValueError(row, "shares weights are finite nonnegative numbers")
        totals[name] = totals.get(name, 0.0) + weight
        if not math.isfinite(totals[name]):
            raise ColumnValueError(row, "shares weights for %r exceed a finite total" % name)
    try:
        total_weight = math.fsum(totals.values())
    except OverflowError as error:
        raise ValueError("shares weights have a finite total") from error
    if total_weight <= 0:
        raise ValueError("shares weights have a positive total")
    return scalar(pa.StructArray.from_arrays(
        [pa.array(list(totals), pa.string()), pa.array([weight / total_weight for weight in totals.values()], pa.float64())],
        fields=list(SHARES.value_type),
    ))


@timed("aggregates.lookup")
def lookup(mapping, keys):
    if not isinstance(mapping, Mapping):
        raise ValueError("a lookup states its literal mapping")
    options = pa.array(list(mapping.values()))
    checked(options.nbytes + keys.nbytes * 4 + len(keys) * 32, "literal lookup")
    positions = {key: index for index, key in enumerate(mapping)}
    indices = []
    for row, value in enumerate(keys, 1):
        try:
            value = value.as_py()
            if value is None:
                indices.append(None)
            elif value in positions:
                indices.append(positions[value])
            else:
                raise ColumnValueError(row, "lookup has no value for key %r" % (value,))
        except (TypeError, OverflowError) as error:
            raise ColumnValueError(row, "lookup requires a scalar key: %s" % error) from error
    return pc.take(options, pa.array(indices, type=pa.int64()))


@timed("aggregates.same_kind")
def same_kind(arguments, keywords):
    return arguments[0].kind


@timed("aggregates.numeric_kind")
def numeric_kind(arguments, keywords):
    kind = arguments[0].kind
    return kind if kind.name == "duration" else Kind("number")


@timed("aggregates.list_kind")
def list_kind(arguments, keywords):
    kind = arguments[0].kind
    return Kind("list", of=kind, arrow=pa.list_(datatype(kind)))


for _name, _callback, _reads, _gives, _keywords in (
    ("count", count, ("any",), Kind("number"), ()),
    ("sum", total, (NUMBERS,), numeric_kind, ()),
    ("mean", average, (NUMBERS,), numeric_kind, ()),
    ("min", minimum, ("any",), same_kind, ()),
    ("max", maximum, ("any",), same_kind, ()),
    ("deviation", deviation, (NUMBERS,), numeric_kind, ()),
    ("any", exists, ("flag",), Kind("flag"), ()),
    ("all", every, ("flag",), Kind("flag"), ()),
    ("distinct", distinct, ("any",), list_kind, ()),
    ("list", listed, ("any",), list_kind, ("by",)),
    ("first", first, ("any",), same_kind, ("by",)),
    ("last", last, ("any",), same_kind, ("by",)),
    ("points", points, (("number", "duration", "time"), NAMES, NUMBERS), Kind("points", arrow=POINTS), ()),
    ("shares", shares, (NAMES, NUMBERS), Kind("shares", arrow=SHARES), ()),
):
    FUNCTIONS.register(_name, Function(_callback, _reads, _gives, aggregate=True, keywords=_keywords))
