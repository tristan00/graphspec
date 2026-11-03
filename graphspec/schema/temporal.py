import logging
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Real

import pyarrow as pa

from graphspec.infra.timing import timed

from .fields import ColumnValueError

LOG = logging.getLogger(__name__)
FIXED = dict(second=1, minute=60, hour=3600, day=86400, week=604800)
RESOLUTION = dict(s=1, ms=1000, us=1000000, ns=1000000000)
CALENDAR = (*FIXED, "month", "year")
BEFORE = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
EPOCH = datetime(1970, 1, 1)
UNSET = object()
CURRENT = ContextVar("source_time_unit", default=UNSET)


@contextmanager
def context(graph_unit):
    started = time.perf_counter()
    token = CURRENT.set(graph_unit)
    try:
        yield
    finally:
        CURRENT.reset(token)
        LOG.debug("context exit unit=%r seconds=%.6f", graph_unit, time.perf_counter() - started)


@timed("temporal.unit")
def unit(columns):
    columns = tuple(columns)
    if not columns:
        return None
    for name, parameters in columns:
        try:
            Time().validate(parameters)
        except ValueError as error:
            raise ValueError("column %s: %s" % (name, error)) from error
    units = {parameters["unit"] for _, parameters in columns}
    if len(units) > 1 and any(value not in CALENDAR for value in units):
        raise ValueError("incompatible units of time: " + ", ".join(
            "%s=%r" % (name, parameters["unit"]) for name, parameters in columns
        ))
    finest = min(units, key=CALENDAR.index) if units <= set(CALENDAR) else next(iter(units))
    for name, parameters in columns:
        try:
            aligned(parameters, finest)
        except ValueError as error:
            raise ValueError("column %s: %s; graph unit %r from %s" % (
                name, error, finest,
                ", ".join(other for other, stated in columns if stated["unit"] == finest),
            )) from error
    return finest


@timed("temporal.aligned")
def aligned(parameters, graph_unit):
    own = parameters["unit"]
    if own != graph_unit and (own not in CALENDAR or graph_unit not in CALENDAR):
        raise ValueError("time unit %r cannot meet graph unit %r" % (own, graph_unit))
    if own != graph_unit and CALENDAR.index(own) < CALENDAR.index(graph_unit):
        raise ValueError("graph unit %r is coarser than column unit %r" % (graph_unit, own))
    if own == graph_unit and "at" in parameters:
        raise ValueError("at is refused for a time of the graph's finest unit %r" % own)
    if own != graph_unit and "at" not in parameters:
        raise ValueError("time unit %r coarser than %r requires at=start or at=end" % (own, graph_unit))


@timed("temporal.whole")
def whole(value):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("a numeric time counts whole units, found %r" % (value,))
    try:
        integer = int(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("a numeric time counts finite whole units, found %r" % (value,)) from error
    if integer != value:
        raise ValueError("a numeric time counts whole units, found %r" % (value,))
    return integer


@timed("temporal.checked")
def checked(value):
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError("time %r is outside int64" % value)
    return value


@timed("temporal.days")
def days(year, month):
    previous = year - 1
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 365 * previous + previous // 4 - previous // 100 + previous // 400 + BEFORE[month - 1] + (
        int(leap and month > 2)
    ) - 719162


@timed("temporal.bounds")
def bounds(value, own):
    if own in FIXED:
        start = value * FIXED[own] + (-259200 if own == "week" else 0)
        return start, start + FIXED[own]
    if own == "year":
        return days(value, 1) * 86400, days(value + 1, 1) * 86400
    year, month = divmod(value, 12)
    year, month = year + 1970, month + 1
    return days(year, month) * 86400, days(year + (month == 12), month % 12 + 1) * 86400


@timed("temporal.count")
def count(value, parameters):
    own = parameters["unit"]
    if isinstance(value, str):
        if "format" not in parameters:
            raise ValueError("a text time requires format, found %r" % value)
        if own not in CALENDAR:
            raise ValueError("a text time cannot count unit %r" % own)
        value = datetime.strptime(value, parameters["format"])
    if isinstance(value, date):
        if own not in CALENDAR:
            raise ValueError("a timestamp cannot count unit %r" % own)
        if not isinstance(value, datetime):
            value = datetime.combine(value, datetime.min.time())
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if own == "year":
            return value.year
        if own == "month":
            return (value.year - 1970) * 12 + value.month - 1
        elapsed = value - EPOCH
        seconds = elapsed.days * 86400 + elapsed.seconds
        return (seconds + (259200 if own == "week" else 0)) // FIXED[own]
    return whole(value)


@timed("temporal.convert")
def convert(value, parameters, graph_unit):
    if value is None:
        return None
    value = count(value, parameters)
    own = parameters["unit"]
    if own == graph_unit:
        return checked(value)
    if graph_unit == "month":
        return checked((value - 1970) * 12 + (11 if parameters["at"] == "end" else 0))
    start, stop = bounds(value, own)
    origin = -259200 if graph_unit == "week" else 0
    step = FIXED[graph_unit]
    return checked((stop - origin - 1) // step if parameters["at"] == "end" else (
        start - origin + step - 1
    ) // step)


@timed("temporal.formatting")
def formatting(kind, parameters):
    while pa.types.is_dictionary(kind):
        kind = kind.value_type
    if (pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_string_view(kind)) \
            and "format" not in parameters:
        raise ValueError("a text time requires format")


class Time:
    version = 1
    fields = frozenset(("unit", "format", "at"))
    required = ("unit",)

    @timed("temporal.Time.validate")
    def validate(self, parameters):
        if "unit" not in parameters:
            raise ValueError("time requires unit, including None for an order without a clock")
        own = parameters["unit"]
        if own is not None and (type(own) is not str or not own):
            raise ValueError("time unit is a nonempty name or None")
        if "format" in parameters and (type(parameters["format"]) is not str or not parameters["format"]):
            raise ValueError("time format is a nonempty string")
        if "format" in parameters and own not in CALENDAR:
            raise ValueError("a time format requires a calendar unit, found %r" % own)
        if "at" in parameters and parameters["at"] not in ("start", "end"):
            raise ValueError("time at is start or end")

    @timed("temporal.Time.__call__")
    def __call__(self, values, parameters):
        self.validate(parameters)
        formatting(values.type, parameters)
        graph_unit = CURRENT.get()
        if graph_unit is UNSET:
            raise ValueError("time conversion requires the graph's negotiated unit")
        aligned(parameters, graph_unit)
        result = []
        for row, scalar in enumerate(values, 1):
            try:
                value = scalar.as_py()
                result.append(convert(value, parameters, graph_unit))
            except (TypeError, ValueError, OverflowError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.int64())


class Duration:
    version = 1
    fields = frozenset(("unit",))
    required = ("unit",)

    @timed("temporal.Duration.validate")
    def validate(self, parameters):
        if "unit" not in parameters or type(parameters["unit"]) is not str or not parameters["unit"]:
            raise ValueError("duration requires a nonempty unit name")

    @timed("temporal.Duration.__call__")
    def __call__(self, values, parameters):
        self.validate(parameters)
        result = []
        for row, scalar in enumerate(values, 1):
            try:
                while pa.types.is_dictionary(scalar.type) and scalar.is_valid:
                    scalar = scalar.value
                if not scalar.is_valid:
                    result.append(None)
                    continue
                if pa.types.is_duration(scalar.type):
                    if parameters["unit"] not in FIXED:
                        raise ValueError("a native duration has no fixed conversion to %r" % parameters["unit"])
                    value = scalar.value / (RESOLUTION[scalar.type.unit] * FIXED[parameters["unit"]])
                else:
                    value = scalar.as_py()
                if isinstance(value, bool) or not isinstance(value, (Real, Decimal, str)):
                    raise ValueError("duration is a numeric count of its unit, found %r" % (value,))
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("duration is finite, found %r" % value)
                result.append(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.float64())


class Interval(Time):
    version = 1

    @timed("temporal.Interval.__call__")
    def __call__(self, values, parameters):
        self.validate(parameters)
        kind = values.type
        while pa.types.is_dictionary(kind):
            kind = kind.value_type
        if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind) \
                or pa.types.is_list_view(kind) or pa.types.is_large_list_view(kind):
            formatting(kind.value_type, parameters)
        graph_unit = CURRENT.get()
        if graph_unit is UNSET:
            raise ValueError("interval conversion requires the graph's negotiated unit")
        aligned(parameters, graph_unit)
        result = []
        for row, scalar in enumerate(values, 1):
            try:
                value = scalar.as_py()
                if value is None:
                    result.append(None)
                    continue
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError("an interval holds exactly two times, found %r" % (value,))
                result.append([
                    None if isinstance(endpoint, float) and math.isnan(endpoint) else convert(endpoint, parameters, graph_unit)
                    for endpoint in value
                ])
            except (TypeError, ValueError, OverflowError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.list_(pa.int64(), 2))
