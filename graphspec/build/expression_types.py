import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .streams import checked

LOG = logging.getLogger(__name__)
FUNCTIONS = Names("expression function", True)
UNSTATED = object()
CURRENT = ContextVar("expression_function_call", default=None)
NEVER_TIME = 1 << 63
EXTENDED_TIME = pa.decimal128(20, 0)


@dataclass(frozen=True)
class Kind:
    name: str
    unit: object = None
    of: object = None
    width: object = None
    arrow: object = None


@dataclass(frozen=True)
class Value:
    values: object
    kind: Kind

    @property
    @timed("expression.Value.type")
    def type(self):
        return self.kind.name

    @property
    @timed("expression.Value.unit")
    def unit(self):
        return self.kind.unit


@dataclass(frozen=True)
class Call:
    unit: object
    rows: int
    arguments: tuple
    keywords: dict
    offset: int = 0
    ordered: bool = False
    through: object = UNSTATED
    indices: object = None


class Function:
    @timed("expression.Function")
    def __init__(self, callback, reads, gives, aggregate=False, version=1, minimum=None, maximum=UNSTATED,
                 keywords=(), nulls="propagate"):
        self.callback, self.reads, self.gives = callback, tuple(reads), gives
        self.aggregate, self.version = aggregate, version
        self.minimum = len(self.reads) if minimum is None else minimum
        self.maximum = len(self.reads) if maximum is UNSTATED else maximum
        self.keywords, self.nulls = tuple(keywords), nulls

    @timed("expression.Function.call")
    def __call__(self, *arguments, **keywords):
        return self.callback(*arguments, **keywords)


@timed("expression.call")
def call():
    result = CURRENT.get()
    if result is None:
        raise ValueError("an expression function requires its evaluation context")
    return result


@contextmanager
def calling(context):
    started = time.perf_counter()
    token = CURRENT.set(context)
    try:
        yield
    finally:
        CURRENT.reset(token)
        LOG.debug("expression call exit rows=%d seconds=%.6f", context.rows, time.perf_counter() - started)


@timed("expression.datatype")
def datatype(kind):
    if kind.arrow is not None:
        return kind.arrow
    if kind.name in ("number", "ordered", "duration"):
        return pa.float64()
    if kind.name in ("category", "identifier", "text"):
        return pa.string()
    if kind.name == "flag":
        return pa.bool_()
    if kind.name == "time":
        return pa.int64()
    if kind.name == "interval":
        return pa.list_(pa.int64(), 2)
    if kind.name == "list" and kind.of is not None:
        return pa.list_(datatype(kind.of))
    if kind.name in ("vector", "point") and type(kind.width) is int and kind.width > 0:
        return pa.list_(pa.float64(), kind.width)
    if kind.name == "missing":
        return pa.null()
    raise ValueError("the expression type %s needs its Arrow representation" % kind.name)


@timed("expression.common")
def common(left, right):
    if left.name == "missing":
        return right
    if right.name == "missing":
        return left
    if {left.name, right.name} <= {"number", "ordered"}:
        return Kind("number")
    if {left.name, right.name} <= {"category", "identifier", "text"}:
        return Kind("text" if "text" in (left.name, right.name) else "category")
    if left.name == right.name == "list":
        return Kind("list", of=common(left.of, right.of))
    if left.name == right.name == "time" and left.unit == right.unit \
            and EXTENDED_TIME in (datatype(left), datatype(right)):
        return Kind("time", unit=left.unit, arrow=EXTENDED_TIME)
    if left.name == right.name and left.unit == right.unit and left.width == right.width and datatype(left) == datatype(right):
        return left
    raise ValueError("expression types %s and %s do not agree" % (left, right))


@timed("expression.array")
def array(value, rows=None, type=None):
    if isinstance(value, pa.ChunkedArray):
        checked(value.nbytes * 4, "combining expression argument chunks")
        result = value.combine_chunks()
    elif isinstance(value, pa.Array):
        result = value
    else:
        scalar = value if isinstance(value, pa.Scalar) else pa.scalar(value, type=type)
        count = call().rows if rows is None else rows
        checked(max(1, pa.array([scalar.as_py()], type=scalar.type).nbytes) * count * 4,
                "broadcasting an expression literal")
        result = pa.array([], type=scalar.type) if count == 0 else pa.repeat(scalar, count)
    if rows is not None and len(result) != rows:
        raise ValueError("an expression argument has %d rows, expected %d" % (len(result), rows))
    if type is not None and result.type != type:
        checked(result.nbytes * 4, "casting an expression argument")
        result = result.cast(type, safe=True)
    return result
