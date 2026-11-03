import copy
import logging
import math
import re
import time
from bisect import bisect_right
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed

from .compound import List, Nested, Point, Vector
from .fields import ColumnValueError
from .schema import TYPES
from .temporal import Duration, Interval, Time

LOG = logging.getLogger(__name__)
COMMON = frozenset(("type", "nulls", "missing", "outcome"))
SHIPPED = frozenset(("number", "flag", "category", "ordered", "identifier", "text", "time", "duration",
                     "interval", "list", "vector", "point", "nested"))
NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
BOOLEANS = {"true": True, "True": True, "TRUE": True, "1": True,
            "false": False, "False": False, "FALSE": False, "0": False}
SCALAR_TEXT = ContextVar("delimited_scalar_values", default=False)


@contextmanager
def scalar_context(enabled):
    started = time.perf_counter()
    token = SCALAR_TEXT.set(enabled)
    try:
        yield
    finally:
        SCALAR_TEXT.reset(token)
        LOG.debug("scalar context exit enabled=%s seconds=%.6f", enabled, time.perf_counter() - started)


@timed("columns.prepared")
def prepared(values, given):
    from .missing import normalize

    declared = statement(given)
    values = normalize(values, declared)
    return normalize(lexical(values, declared), declared) if SCALAR_TEXT.get() else values


@timed("columns.completed")
def completed(values, given):
    from .missing import normalize

    declared = statement(given)
    return normalize(values, declared) if declared["type"] in ("number", "duration") else values


@timed("columns.statement")
def statement(given):
    if given is None:
        return {}
    if type(given) is str:
        return dict(type=given)
    if not isinstance(given, Mapping):
        raise ValueError("a column states its type by name or a mapping")
    return copy.deepcopy(dict(given))


@timed("columns.checked")
def checked(given):
    from .missing import validate

    declared = statement(given)
    name = declared.get("type")
    if type(name) is not str or not name:
        raise ValueError("the column has no stated type")
    entry = TYPES.named(name)
    parameters = {key: value for key, value in declared.items() if key not in COMMON}
    fields = getattr(entry, "fields", None)
    if fields is not None and set(parameters) - set(fields):
        raise ValueError("the type %s does not take %s" % (name, ", ".join(sorted(set(parameters) - set(fields)))))
    required = set(getattr(entry, "required", ())) - set(parameters)
    if required:
        raise ValueError("the type %s requires %s" % (name, ", ".join(sorted(required))))
    validate(declared)
    validator = getattr(entry, "validate", None)
    if validator is not None:
        validator(parameters)
    if not callable(getattr(entry, "convert", entry)):
        raise ValueError("the type %s has no source conversion" % name)
    if name not in SHIPPED:
        gives = getattr(entry, "gives", None)
        if type(gives) is not str or gives not in SHIPPED:
            raise ValueError("the type %s states gives as a shipped column type" % name)
    return declared, entry, parameters


@timed("columns.representation")
def representation(name, kind):
    if name in ("number", "ordered", "duration"):
        return kind == pa.float64()
    if name == "flag":
        return kind == pa.bool_()
    if name in ("category", "identifier", "text"):
        return kind == pa.string()
    if name == "time":
        return kind == pa.int64()
    if name == "interval":
        return pa.types.is_fixed_size_list(kind) and kind.list_size == 2 and kind.value_type == pa.int64()
    if name == "list":
        return pa.types.is_list(kind) or pa.types.is_large_list(kind)
    if name in ("vector", "point"):
        return pa.types.is_fixed_size_list(kind) and kind.value_type == pa.float64() and (
            kind.list_size in (2, 3) if name == "point" else kind.list_size > 0
        )
    return name == "nested" and pa.types.is_struct(kind)


@timed("columns.held")
def held(values, given):
    declared = statement(given)
    stated = declared["type"]
    name = stated if stated in SHIPPED else TYPES.named(stated).gives
    if not representation(name, values.type):
        raise ValueError("the type %s gives %s, which requires its Arrow representation, found %s"
                         % (stated, name, values.type))
    if name == "list" and stated == "list":
        offsets = [0]
        for size in pc.list_value_length(values).fill_null(0).to_pylist():
            offsets.append(offsets[-1] + size)
        try:
            held(pc.list_flatten(values), declared["of"])
        except ColumnValueError as error:
            raise ColumnValueError(bisect_right(offsets, error.row - 1), "list member %s" % error) from error
        return values
    allowed = set(declared["values"]) if stated == "category" and "values" in declared else None
    if name not in ("number", "duration", "ordered", "vector", "point") and allowed is None:
        return values
    for index, value in enumerate(values.to_pylist(), 1):
        if value is None:
            continue
        if name in ("number", "duration") and not math.isfinite(value):
            raise ColumnValueError(index, "needs a finite %s, found %r" % (name, value))
        if name == "ordered" and (not math.isfinite(value) or value != int(value) or value < 0
                                   or stated == "ordered" and value >= len(declared["values"])):
            raise ColumnValueError(index, "ordered code %r is outside the declared positions" % value)
        if allowed is not None and value not in allowed:
            raise ColumnValueError(index, "has category %r outside values" % value)
        if name in ("vector", "point"):
            if stated == name and len(value) != declared["width"]:
                raise ColumnValueError(index, "%s width is %d, found %d values" % (name, declared["width"], len(value)))
            if any(member is not None and not math.isfinite(member) for member in value):
                raise ColumnValueError(index, "%s members are finite numbers or missing" % name)
    return values


def convert(values, given):
    from .missing import normalize

    started = time.perf_counter()
    declared, entry, parameters = checked(given)
    name = declared["type"]
    try:
        result = getattr(entry, "convert", entry)(values, parameters)
        if not isinstance(result, (pa.Array, pa.ChunkedArray)) or len(result) != len(values):
            raise ValueError("a column type returns an Arrow array of the input length")
        gives = name if name in SHIPPED else entry.gives
        if not representation(gives, result.type):
            raise ValueError("the type %s gives %s, which requires its Arrow representation, found %s"
                             % (name, gives, result.type))
        result = normalize(result, {})
        for index, (absent, present) in enumerate(zip(values.is_null().to_pylist(), result.is_valid().to_pylist(), strict=True), 1):
            if absent and present:
                raise ColumnValueError(index, "type conversion cannot fill a missing value; state a missing rule")
        return held(result, declared)
    finally:
        LOG.debug("convert exit type=%s values=%d seconds=%.6f", name, len(values), time.perf_counter() - started)


@timed("columns.number")
def number(values, parameters):
    result = []
    for index, value in enumerate(values.to_pylist(), 1):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ColumnValueError(index, "needs a number, found %r" % (value,))
        try:
            value = float(value)
        except (ValueError, OverflowError) as error:
            raise ColumnValueError(index, "needs a number, found %r" % (value,)) from error
        if not math.isfinite(value):
            raise ColumnValueError(index, "needs a finite number, found %r" % (value,))
        result.append(value)
    return pa.array(result, pa.float64())


@timed("columns.lexical")
def lexical(values, given):
    declared = statement(given)
    kind = values.type.value_type if pa.types.is_dictionary(values.type) else values.type
    if not (pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_string_view(kind)):
        return values
    name = declared["type"]
    flag = name == "flag" and "true" not in declared and "false" not in declared
    clock = name == "time" and "format" not in declared
    if not flag and not clock:
        return values
    result = []
    for row, value in enumerate(values.to_pylist(), 1):
        if value is None:
            result.append(None)
        elif flag:
            if value not in BOOLEANS:
                raise ColumnValueError(row, "a delimited flag needs true/false, 1/0, or explicit true and false values")
            result.append(BOOLEANS[value])
        else:
            if NUMERIC.fullmatch(value.strip()) is None:
                raise ColumnValueError(row, "a text time requires format; found %r" % value)
            try:
                count = Decimal(value)
                valid = count.is_finite() and count == count.to_integral_value() and -(1 << 63) <= count < (1 << 63)
            except ArithmeticError as error:
                raise ColumnValueError(row, "a numeric time counts whole int64 units, found %r" % value) from error
            if not valid:
                raise ColumnValueError(row, "a numeric time counts whole int64 units, found %r" % value)
            result.append(int(count))
    return pa.array(result, pa.bool_() if flag else pa.int64())


@timed("columns.names")
def names(values):
    result = []
    for index, value in enumerate(values.to_pylist(), 1):
        if value is not None and not isinstance(value, (str, int, float, bool, Decimal)):
            raise ColumnValueError(index, "needs a name, found %r" % (value,))
        result.append(None if value is None else str(value))
    return result


@timed("columns.vocabulary")
def vocabulary(parameters):
    if "values" in parameters:
        values = parameters["values"]
        if not isinstance(values, (list, tuple)) or any(type(value) is not str for value in values) \
                or len(set(values)) != len(values):
            raise ValueError("values states a list of distinct names")


@timed("columns.category")
def category(values, parameters):
    vocabulary(parameters)
    result = names(values)
    allowed = set(parameters["values"]) if "values" in parameters else None
    for index, value in enumerate(result, 1):
        if value is not None and allowed is not None and value not in allowed:
            raise ColumnValueError(index, "has category %r outside values" % (value,))
    return pa.array(result, pa.string())


@timed("columns.flag_values")
def flag_values(parameters):
    if bool("true" in parameters) != bool("false" in parameters):
        raise ValueError("a flag states both true and false values")
    result = {}
    for name in ("true", "false"):
        given = parameters.get(name, [])
        if not isinstance(given, (list, tuple)):
            given = [given]
        if any(value is None or not isinstance(value, (str, int, float, bool)) for value in given):
            raise ValueError("a flag states true and false as scalar values or lists of them")
        result[name] = given
    if any(value in result["false"] for value in result["true"]):
        raise ValueError("the true and false values of a flag overlap")
    return result


@timed("columns.flag")
def flag(values, parameters):
    tokens = flag_values(parameters)
    result = []
    for index, value in enumerate(values.to_pylist(), 1):
        if value is None or type(value) is bool:
            result.append(value)
        elif value in tokens["true"]:
            result.append(True)
        elif value in tokens["false"]:
            result.append(False)
        else:
            raise ColumnValueError(index, "needs a flag or a declared true or false value, found %r" % (value,))
    return pa.array(result, pa.bool_())


class Ordered:
    version = 1
    fields = frozenset(("values",))
    required = ("values",)

    @timed("columns.Ordered.validate")
    def validate(self, parameters):
        vocabulary(parameters)

    @timed("columns.Ordered.convert")
    def __call__(self, values, parameters):
        result = category(values, parameters).to_pylist()
        positions = {value: index for index, value in enumerate(parameters["values"])}
        return pa.array([None if value is None else float(positions[value]) for value in result], pa.float64())


class Identifier:
    version = 1
    fields = frozenset()

    @timed("columns.Identifier.convert")
    def __call__(self, values, parameters):
        return pa.array(names(values), pa.string())


class Text:
    version = 1
    fields = frozenset()

    @timed("columns.Text.convert")
    def __call__(self, values, parameters):
        result = values.to_pylist()
        for index, value in enumerate(result, 1):
            if value is not None and type(value) is not str:
                raise ColumnValueError(index, "needs text, found %r" % (value,))
        return pa.array(result, pa.string())


@timed("columns.proposed")
def proposed(values, name=""):
    kind = values.type
    if pa.types.is_dictionary(kind):
        return proposed(values.cast(kind.value_type), name)
    if pa.types.is_boolean(kind):
        return "flag"
    if pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind):
        return "identifier" if name.lower() in ("id", "key") or name.lower().endswith("_id") else "number"
    if pa.types.is_timestamp(kind) or pa.types.is_date(kind):
        return dict(type="time", unit="day" if pa.types.is_date(kind) else "second")
    if pa.types.is_duration(kind):
        return dict(type="duration", unit="second")
    if pa.types.is_struct(kind):
        return "nested"
    if pa.types.is_fixed_size_list(kind) and (pa.types.is_floating(kind.value_type) or pa.types.is_integer(kind.value_type)):
        return dict(type="vector", width=kind.list_size)
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        return dict(type="list", of=proposed(pa.array([], kind.value_type)))
    if pa.types.is_null(kind):
        return None
    found = [value for value in values.to_pylist() if value is not None]
    if not found:
        return None
    if not all(type(value) is str for value in found):
        return None
    if name.lower() in ("id", "key") or name.lower().endswith("_id"):
        return "identifier"
    if all(NUMERIC.fullmatch(value) for value in found):
        return "number"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in found):
        return dict(type="time", unit="day", format="%Y-%m-%d")
    return "text" if any(len(value) > 128 or any(character.isspace() for character in value) for value in found) else "category"


@timed("columns.temporal_columns")
def temporal_columns(given, name):
    declared = statement(given)
    kind = declared.get("type")
    if type(kind) is str and kind not in SHIPPED:
        try:
            kind = getattr(TYPES.named(kind), "gives", None)
        except KeyError as error:
            raise ValueError("column %s: %s" % (name, error)) from error
    if kind in ("time", "interval"):
        return [(name, {key: declared[key] for key in Time.fields if key in declared})]
    if kind == "list" and "of" in declared:
        return temporal_columns(declared["of"], name + "[]")
    return []


for _name, _entry in (("ordered", Ordered()), ("identifier", Identifier()), ("text", Text()),
                      ("time", Time()), ("duration", Duration()), ("interval", Interval()),
                      ("list", List()), ("vector", Vector()), ("point", Point()), ("nested", Nested())):
    TYPES.register(_name, _entry)
