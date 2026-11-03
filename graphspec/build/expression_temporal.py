import math
import operator
import re
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.temporal import CALENDAR, EPOCH, FIXED, bounds, checked, count, whole

from .expression_types import (
    EXTENDED_TIME,
    FUNCTIONS,
    NEVER_TIME,
    UNSTATED,
    Function,
    Kind,
    Value,
    array,
    call,
)

ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?")
COMPARISONS = {"<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne,
               ">": operator.gt, ">=": operator.ge}
TEXT = ("text", "category", "identifier")
NUMBERS = ("number", "ordered")


@timed("expression_temporal.regex")
def regex(pattern, extract=False):
    try:
        result = re.compile(pattern)
    except re.error as error:
        raise ValueError("invalid pattern %r: %s" % (pattern, error)) from error
    if extract and result.groups > 1:
        raise ValueError("extract requires zero or one capture group")
    return result


@timed("expression_temporal.width")
def width(value, name):
    try:
        result = whole(value)
    except ValueError as error:
        raise ValueError("%s count is a nonnegative whole number, found %r" % (name, value)) from error
    if result < 0:
        raise ValueError("%s count is a nonnegative whole number, found %r" % (name, value))
    return result


@timed("expression_temporal.text_value")
def text_value(name, values):
    text = values[0]
    if type(text) is not str:
        raise ValueError("%s reads text, found %r" % (name, text))
    if name in ("matches", "extract", "replace"):
        pattern = regex(values[1], name == "extract")
        if name == "replace":
            return pattern.sub(values[2], text)
        match = pattern.search(text)
        if name == "matches":
            return match is not None
        return None if match is None else match.group(1 if pattern.groups else 0)
    if name in ("lower", "upper", "strip"):
        return getattr(text, name)()
    if name == "length":
        return float(len(text))
    if name in ("left", "right"):
        size = width(values[1], name)
        return "" if size == 0 else text[:size] if name == "left" else text[-size:]
    if name == "split":
        if not values[1]:
            raise ValueError("split separator is nonempty")
        return text.split(values[1])
    if name == "concat":
        return "".join(values)
    raise ValueError("no text function %s" % name)


class Text:
    @timed("expression_temporal.Text")
    def __init__(self, name):
        self.name = name

    @timed("expression_temporal.Text.call")
    def __call__(self, *arguments):
        if self.name in ("matches", "extract", "replace") and isinstance(arguments[1], str):
            pattern = regex(arguments[1], self.name == "extract")
            if self.name == "replace" and isinstance(arguments[2], str):
                try:
                    pattern.sub(arguments[2], "")
                except re.error as error:
                    raise ValueError("invalid replacement %r: %s" % (arguments[2], error)) from error
        if self.name == "split" and isinstance(arguments[1], str) and arguments[1] == "":
            raise ValueError("split separator is nonempty")
        if self.name in ("left", "right") and arguments[1] is not None \
                and not isinstance(arguments[1], (pa.Array, pa.ChunkedArray)):
            width(arguments[1], self.name)
        columns = [array(value).to_pylist() for value in arguments]
        result = []
        for row, values in enumerate(zip(*columns, strict=True), 1):
            try:
                result.append(None if any(value is None for value in values) else text_value(self.name, values))
            except (TypeError, ValueError, OverflowError, re.error) as error:
                raise ColumnValueError(row, str(error)) from error
        dtype = pa.bool_() if self.name == "matches" else pa.float64() if self.name == "length" else (
            pa.list_(pa.string()) if self.name == "split" else pa.string()
        )
        return pa.array(result, dtype)


@timed("expression_temporal.calendar")
def calendar(unit):
    if unit not in CALENDAR:
        raise ValueError("a calendar function cannot read time unit %r" % unit)


@timed("expression_temporal.time_count")
def time_count(value):
    value = whole(value)
    return value if value == NEVER_TIME else checked(value)


@timed("expression_temporal.instant")
def instant(value, unit):
    calendar(unit)
    if isinstance(value, date):
        if not isinstance(value, datetime):
            value = datetime.combine(value, datetime.min.time())
        value = value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo is not None else value
        value = count(value, dict(unit=unit))
    value = time_count(value)
    if value == NEVER_TIME:
        raise ValueError("never has no finite calendar date")
    return EPOCH + timedelta(seconds=bounds(value, unit)[0])


class Part:
    @timed("expression_temporal.Part")
    def __init__(self, name):
        self.name = name

    @timed("expression_temporal.Part.call")
    def __call__(self, values):
        unit = call().unit
        calendar(unit)
        result = []
        for row, scalar in enumerate(array(values), 1):
            try:
                if not scalar.is_valid:
                    result.append(None)
                    continue
                value = instant(scalar.as_py(), unit)
                result.append(float(value.weekday() if self.name == "weekday" else getattr(value, self.name)))
            except (TypeError, ValueError, OverflowError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.float64())


class Endpoint:
    @timed("expression_temporal.Endpoint")
    def __init__(self, index):
        self.index = index

    @timed("expression_temporal.Endpoint.call")
    def __call__(self, values):
        result = []
        for row, scalar in enumerate(array(values), 1):
            try:
                value = scalar.as_py()
                if value is not None and (not isinstance(value, list) or len(value) != 2):
                    raise ValueError("start and end read an interval of exactly two times")
                result.append(None if value is None or value[self.index] is None else checked(whole(value[self.index])))
            except (TypeError, ValueError, OverflowError) as error:
                raise ColumnValueError(row, str(error)) from error
        return pa.array(result, pa.int64())


@timed("expression_temporal.time_kind")
def time_kind(arguments, keywords):
    return Kind("time", unit=call().unit)


@timed("expression_temporal.time_literal")
def time_literal(values):
    unit = call().unit
    calendar(unit)
    result = []
    for row, value in enumerate(array(values).to_pylist(), 1):
        try:
            if value is None:
                result.append(None)
                continue
            if not isinstance(value, str) or ISO.fullmatch(value) is None:
                raise ValueError("time needs an ISO 8601 date or date and time, found %r" % value)
            result.append(checked(count(datetime.fromisoformat(value), dict(unit=unit))))
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(row, str(error)) from error
    return pa.array(result, pa.int64())


@timed("expression_temporal.span_kind")
def span_kind(arguments, keywords):
    unit = arguments[1].values
    if isinstance(unit, pa.Scalar):
        unit = unit.as_py()
    if type(unit) is not str or not unit:
        raise ValueError("span unit is one nonempty literal name")
    return Kind("duration", unit=unit)


@timed("expression_temporal.span")
def span(values, unit):
    if isinstance(unit, pa.Scalar):
        unit = unit.as_py()
    if type(unit) is not str or not unit:
        raise ValueError("span unit is one nonempty literal name")
    result = []
    for row, value in enumerate(array(values).to_pylist(), 1):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ColumnValueError(row, "span counts finite numbers")
        result.append(value)
    return pa.array(result, pa.float64())


@timed("expression_temporal.fixed")
def fixed(value, own, target):
    if own == target:
        return Fraction(str(value))
    if own not in FIXED or target not in FIXED:
        raise ValueError("duration unit %r has no fixed conversion to %r" % (own, target))
    return Fraction(str(value)) * FIXED[own] / FIXED[target]


@timed("expression_temporal.shift")
def shift(value, duration, own, unit):
    value = time_count(value)
    if own == unit:
        amount = Fraction(str(duration))
        if amount.denominator != 1:
            raise ValueError("duration %r %s is not a whole count of graph unit %r" % (duration, own, unit))
        return NEVER_TIME if value == NEVER_TIME else checked(value + int(amount))
    if own in ("month", "year"):
        calendar(unit)
        months = Fraction(str(duration)) * (12 if own == "year" else 1)
        if months.denominator != 1:
            raise ValueError("calendar duration %r %s is not a whole number of months" % (duration, own))
        months = int(months)
        if unit in ("month", "year"):
            if unit == "year" and months % 12:
                raise ValueError("calendar duration %r %s reaches no whole time of graph unit %r" % (duration, own, unit))
            return NEVER_TIME if value == NEVER_TIME else checked(value + (months // 12 if unit == "year" else months))
        if value == NEVER_TIME:
            return NEVER_TIME
        current = instant(value, unit)
        year, month = divmod(current.year * 12 + current.month - 1 + months, 12)
        moved = current.replace(year=year, month=month + 1)
        result = count(moved, dict(unit=unit))
        if instant(result, unit) != moved:
            raise ValueError("calendar duration %r %s reaches no whole time of graph unit %r" % (duration, own, unit))
        return checked(result)
    amount = fixed(duration, own, unit)
    if amount.denominator != 1:
        raise ValueError("duration %r %s is not a whole count of graph unit %r" % (duration, own, unit))
    return NEVER_TIME if value == NEVER_TIME else checked(value + int(amount))


@timed("expression_temporal.arithmetic")
def arithmetic(op, left, right, unit):
    a, b = left.kind, right.kind
    a = Kind("number") if a.name == "ordered" else a
    b = Kind("number") if b.name == "ordered" else b
    if a.name == b.name == "time":
        if op not in ("-", *COMPARISONS):
            raise ValueError("two times support subtraction and comparison, found %s" % op)
        kind = Kind("flag") if op in COMPARISONS else Kind("duration", unit=unit)
    elif {a.name, b.name} == {"time", "number"} and op in COMPARISONS:
        kind = Kind("flag")
    elif a.name == "time" and b.name in ("number", "duration") and op in ("+", "-"):
        kind = Kind("time", unit=unit, arrow=a.arrow)
    elif b.name == "time" and a.name in ("number", "duration") and op == "+":
        return arithmetic(op, right, left, unit)
    elif a.name == b.name == "duration" and op in ("+", "-", "/", *COMPARISONS):
        if op in COMPARISONS and (a.unit in ("month", "year") or b.unit in ("month", "year")):
            raise ValueError("duration units %r and %r cannot be compared as fixed lengths" % (a.unit, b.unit))
        kind = Kind("flag") if op in COMPARISONS else Kind("number") if op == "/" else a
    elif a.name == "duration" and b.name == "number" and op in ("*", "/"):
        kind = a
    elif a.name == "number" and b.name == "duration" and op == "*":
        return arithmetic(op, right, left, unit)
    else:
        raise ValueError("%s is not defined between %s and %s" % (op, a.name, b.name))
    result = []
    columns = array(left.values).to_pylist(), array(right.values).to_pylist()
    for row, (one, other) in enumerate(zip(*columns, strict=True), 1):
        try:
            if one is None or other is None:
                result.append(None)
                continue
            if a.name == b.name == "time":
                one, other = time_count(one), time_count(other)
                if op not in COMPARISONS and NEVER_TIME in (one, other):
                    raise ValueError("a difference involving never has no finite duration")
                value = COMPARISONS[op](one, other) if op in COMPARISONS else float(one - other)
            elif op in COMPARISONS and "number" in (a.name, b.name) and "time" in (a.name, b.name):
                value = COMPARISONS[op](time_count(one) if a.name == "time" else one,
                                        time_count(other) if b.name == "time" else other)
            elif a.name == "time":
                value = shift(one, other if op == "+" else -other, b.unit if b.name == "duration" else unit, unit)
            elif b.name == "duration":
                converted = fixed(other, b.unit, a.unit)
                original = Fraction(str(one))
                if op in COMPARISONS:
                    value = COMPARISONS[op](original, converted)
                elif op == "+":
                    value = float(original + converted)
                elif op == "-":
                    value = float(original - converted)
                else:
                    value = float(original / converted)
            else:
                value = one * other if op == "*" else one / other
            if kind.name in ("number", "duration") and not math.isfinite(value):
                raise ValueError("temporal arithmetic produces a finite number")
            result.append(value)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            raise ColumnValueError(row, str(error)) from error
    dtype = (EXTENDED_TIME if kind.arrow == EXTENDED_TIME else pa.int64()) if kind.name == "time" else (
        pa.bool_() if kind.name == "flag" else pa.float64()
    )
    return Value(pa.array(result, dtype), kind)


@timed("expression_temporal.met")
def met(values, deadlines):
    context = call()
    times, ends = array(values).to_pylist(), array(deadlines).to_pylist()
    result = []
    through = None
    for row, (value, deadline) in enumerate(zip(times, ends, strict=True), 1):
        try:
            if deadline is None:
                result.append(None)
            elif value is not None:
                result.append(time_count(value) <= time_count(deadline))
            else:
                if through is None:
                    if context.through is UNSTATED:
                        raise ValueError("met of a missing time requires the source's explicit through")
                    through = array(context.through).to_pylist()
                result.append(None if through[row - 1] is None or time_count(through[row - 1]) < time_count(deadline) else False)
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(row, str(error)) from error
    return pa.array(result, pa.bool_())


for _name, _reads, _gives in (
    ("matches", ("text", "text"), "flag"),
    ("extract", ("text", "text"), "category"),
    ("replace", ("text", "text", "text"), "category"),
    ("lower", ("text",), "category"),
    ("upper", ("text",), "category"),
    ("strip", ("text",), "category"),
    ("length", ("text",), "number"),
    ("left", ("text", "number"), "category"),
    ("right", ("text", "number"), "category"),
    ("split", ("text", "text"), Kind("list", of=Kind("category"))),
):
    FUNCTIONS.register(_name, Function(Text(_name), tuple(
        TEXT if value == "text" else NUMBERS if value == "number" else value for value in _reads
    ), _gives))
FUNCTIONS.register("concat", Function(Text("concat"), (TEXT,), "category", minimum=2, maximum=None))
for _name in ("year", "month", "day", "weekday", "hour", "minute"):
    FUNCTIONS.register(_name, Function(Part(_name), ("time",), "number"))
FUNCTIONS.register("start", Function(Endpoint(0), ("interval",), time_kind))
FUNCTIONS.register("end", Function(Endpoint(1), ("interval",), time_kind))
FUNCTIONS.register("time", Function(time_literal, (TEXT,), time_kind))
FUNCTIONS.register("span", Function(span, (NUMBERS, TEXT), span_kind))
FUNCTIONS.register("met", Function(met, ("time", "time"), "flag", nulls="read"))
