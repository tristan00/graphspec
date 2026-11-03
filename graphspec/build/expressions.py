import ast
import logging
import math
import operator
import time
from bisect import bisect_right
from collections.abc import Mapping
from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.registries import version
from graphspec.schema.columns import SHIPPED, held, representation, statement
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.schema import TYPES

from . import aggregates, expression_temporal
from . import expression_scalars as expression_scalars
from .expression_types import (
    EXTENDED_TIME,
    FUNCTIONS,
    NEVER_TIME,
    UNSTATED,
    Call,
    Function,
    Kind,
    Value,
    array,
    calling,
    common,
    datatype,
)
from .planning import stated
from .streams import checked, reading

LOG = logging.getLogger(__name__)
BINARY = {ast.Add: ("+", operator.add), ast.Sub: ("-", operator.sub), ast.Mult: ("*", operator.mul),
          ast.Div: ("/", operator.truediv), ast.FloorDiv: ("//", operator.floordiv), ast.Mod: ("%", operator.mod),
          ast.Pow: ("**", operator.pow)}
COMPARE = {ast.Eq: ("==", operator.eq), ast.NotEq: ("!=", operator.ne), ast.Lt: ("<", operator.lt),
           ast.LtE: ("<=", operator.le), ast.Gt: (">", operator.gt), ast.GtE: (">=", operator.ge)}
NUMBERS = ("number", "ordered")
NAMES = ("category", "identifier", "text")
ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.List, ast.Dict, ast.Subscript, ast.Call,
           ast.keyword, ast.Attribute, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
           ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)


@timed("expressions.literal")
def literal(node, unit=UNSTATED):
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
        value = node.value
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric literals are finite")
        name = "flag" if type(value) is bool else "category" if type(value) is str else "number"
        return Value(pa.scalar(value), Kind(name))
    if isinstance(node, ast.Name) and node.id == "missing":
        return Value(pa.scalar(None), Kind("missing"))
    if isinstance(node, ast.Name) and node.id == "never":
        if unit is UNSTATED:
            raise ValueError("never requires an explicit graph unit")
        return Value(pa.scalar(Decimal(NEVER_TIME), EXTENDED_TIME), Kind("time", unit=unit, arrow=EXTENDED_TIME))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        held = literal(node.operand, unit)
        if held.type not in NUMBERS:
            raise ValueError("a signed literal is a number")
        value = held.values.as_py()
        return Value(pa.scalar(-value if isinstance(node.op, ast.USub) else value), held.kind)
    if isinstance(node, ast.List):
        values, kind = [], Kind("missing")
        for item in node.elts:
            held = literal(item, unit)
            kind = common(kind, held.kind)
            values.append(held.values.as_py())
        return Value(pa.scalar(values, pa.list_(datatype(kind))), Kind("list", of=kind))
    raise ValueError("this position requires a scalar or list literal; use missing for a null")


@timed("expressions.mapping")
def mapping(node, unit=UNSTATED):
    result, keys, values = {}, Kind("missing"), Kind("missing")
    for key, value in zip(node.keys, node.values, strict=True):
        key, value = literal(key, unit), literal(value, unit)
        if key.type not in (*NUMBERS, *NAMES, "flag", "time"):
            raise ValueError("a lookup key is a nonmissing scalar literal")
        keys, values = common(keys, key.kind), common(values, value.kind)
        found = key.values.as_py()
        if found in result:
            raise ValueError("a lookup repeats the key %r" % found)
        result[found] = value.values.as_py()
    if not result:
        raise ValueError("a lookup states at least one key and value")
    return result, keys, values


@timed("expressions.kind")
def column_kind(given, physical, unit):
    if isinstance(given, Kind):
        if given.name not in (*SHIPPED, "missing", "points", "shares"):
            raise ValueError("a column Kind names a shipped expression representation")
        if physical != datatype(given):
            raise ValueError("a column Kind requires its exact Arrow representation %s, found %s" % (datatype(given), physical))
        if given.name == "time" and physical == EXTENDED_TIME:
            valid = True
        elif given.name in ("points", "shares"):
            valid = physical == (aggregates.POINTS if given.name == "points" else aggregates.SHARES)
        elif given.name == "missing":
            valid = pa.types.is_null(physical)
        else:
            valid = representation(given.name, physical)
        if not valid:
            raise ValueError("column Kind %s has an invalid Arrow representation %s" % (given.name, physical))
        if given.name in ("time", "interval") and (unit is UNSTATED or given.unit != unit):
            raise ValueError("a temporal column Kind states the graph's explicit unit")
        if given.name == "duration" and (type(given.unit) is not str or not given.unit):
            raise ValueError("a duration column Kind states its unit")
        if given.name in ("vector", "point") and (type(given.width) is not int or given.width < 1
                                                  or physical.list_size != given.width
                                                  or given.name == "point" and given.width not in (2, 3)):
            raise ValueError("a vector or point column Kind states its exact valid width")
        if given.name == "list":
            if not (pa.types.is_list(physical) or pa.types.is_large_list(physical)) or given.of is None:
                raise ValueError("a list column Kind states its member type")
            column_kind(given.of, physical.value_type, unit)
        if given.name == "nested" and not pa.types.is_struct(physical):
            raise ValueError("a nested column Kind holds an Arrow struct")
        return given
    declared = statement(given)
    name = declared.get("type")
    if type(name) is not str or not name:
        raise ValueError("the column has no stated type")
    if name not in SHIPPED:
        name = getattr(TYPES.named(name), "gives", None)
    if name not in SHIPPED:
        raise ValueError("the column states a registered source type with a shipped representation")
    nullable = pa.types.is_null(physical)
    if name in ("number", "ordered", "duration") and not (nullable or pa.types.is_integer(physical)
                                                           or pa.types.is_floating(physical) or pa.types.is_decimal(physical)):
        raise ValueError("evaluate reads already-typed numeric columns, found %s" % physical)
    if name == "flag" and not (nullable or pa.types.is_boolean(physical)):
        raise ValueError("evaluate reads already-typed flag columns")
    if name in NAMES and not (nullable or pa.types.is_string(physical) or pa.types.is_large_string(physical)):
        raise ValueError("evaluate reads already-typed text/name columns")
    if name in ("time", "interval"):
        if unit is UNSTATED:
            raise ValueError("time expressions require an explicit graph unit, including None for an order")
        expected = pa.int64() if name == "time" else pa.list_(pa.int64(), 2)
        if not nullable and physical != expected:
            raise ValueError("evaluate reads already-typed %s in the graph unit, found %s" % (name, physical))
        return Kind(name, unit=unit)
    if name == "duration":
        if "unit" not in declared:
            raise ValueError("a duration column requires its stated unit")
        return Kind(name, unit=declared["unit"])
    if name == "list":
        if "of" not in declared or not (pa.types.is_list(physical) or pa.types.is_large_list(physical)):
            raise ValueError("a list column states of and holds an Arrow list")
        return Kind(name, of=column_kind(declared["of"], physical.value_type, unit))
    if name in ("vector", "point"):
        width = declared.get("width")
        if type(width) is not int or width < 1 or name == "point" and width not in (2, 3):
            raise ValueError("the %s column states its valid width" % name)
        return Kind(name, width=width)
    return Kind(name, arrow=physical if name == "nested" else None)


@timed("expressions.accepts")
def accepts(expected, actual):
    return actual == "missing" or expected in ("any", "*") or actual in (expected if isinstance(expected, tuple) else (expected,))


@timed("expressions.scalar")
def scalar(value):
    return isinstance(value.values, pa.Scalar)


@timed("expressions.rows")
def rows(value, count):
    return array(value.values, rows=count)


@timed("expressions.raw")
def raw(value):
    return value.values.as_py() if scalar(value) else value.values


@timed("expressions.literal_argument")
def literal_argument(node):
    if isinstance(node, (ast.Constant, ast.List)) or isinstance(node, ast.Name) and node.id in ("missing", "never"):
        return True
    if isinstance(node, (ast.Call, ast.Name, ast.Attribute)):
        return False
    return all(literal_argument(value) for value in ast.iter_child_nodes(node))


@timed("expressions.checked_output")
def checked_output(value, kind):
    values = pa.repeat(value, 1) if isinstance(value, pa.Scalar) else value
    if kind.name == "time" and datatype(kind) == EXTENDED_TIME:
        for row, value in enumerate(values.to_pylist(), 1):
            if value is not None and (value != int(value) or not -(1 << 63) <= value <= NEVER_TIME):
                raise ColumnValueError(row, "extended times are finite int64 counts or never")
    elif kind.name == "list":
        offsets = [0]
        for size in pc.list_value_length(values).fill_null(0).to_pylist():
            offsets.append(offsets[-1] + size)
        try:
            checked_output(pc.list_flatten(values), kind.of)
        except ColumnValueError as error:
            raise ColumnValueError(bisect_right(offsets, error.row - 1), "list member %s" % error) from error
    elif kind.name == "ordered":
        for row, value in enumerate(values.to_pylist(), 1):
            if value is not None and (not math.isfinite(value) or value < 0 or value != int(value)):
                raise ColumnValueError(row, "ordered values are finite nonnegative whole positions")
    elif kind.name in SHIPPED:
        parameters = dict(type=kind.name)
        if kind.name in ("vector", "point"):
            parameters["width"] = kind.width
        held(values, parameters)
    elif kind.name in ("points", "shares"):
        flattened = pc.list_flatten(values)
        field = "y" if kind.name == "points" else "weight"
        held(pc.struct_field(flattened, field), dict(type="number"))
    return values


@timed("expressions.custom_argument")
def custom_argument(value, kind):
    if kind.name == "list":
        if value is None:
            return None
        scalar_value = not isinstance(value, (pa.Array, pa.ChunkedArray))
        values = pa.array([value], datatype(kind)) if scalar_value else array(value)
        offsets = [0]
        for size in pc.list_value_length(values).fill_null(0).to_pylist():
            offsets.append(offsets[-1] + size)
        try:
            members = custom_argument(pc.list_flatten(values), kind.of)
        except ColumnValueError as error:
            raise ColumnValueError(bisect_right(offsets, error.row - 1), "list member %s" % error.detail) from error
        converted = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), members, mask=values.is_null())
        return converted[0].as_py() if scalar_value else converted
    if kind.name == "time" and datatype(kind) == EXTENDED_TIME:
        if isinstance(value, (pa.Array, pa.ChunkedArray)):
            for row, scalar_value in enumerate(value, 1):
                if scalar_value.is_valid and scalar_value.as_py() == NEVER_TIME:
                    raise ColumnValueError(row, "a custom function cannot read never as a finite int64 time")
            return value.cast(pa.int64(), safe=True)
        if value is not None:
            if value == NEVER_TIME:
                raise ColumnValueError(1, "a custom function cannot read never as a finite int64 time")
            return int(value)
    return value


class Evaluator:
    @timed("expressions.Evaluator")
    def __init__(self, table, columns, whole, unit, offset, ordered, through, namespaces, positions=None, neighbors=None):
        if not isinstance(table, pa.Table) or not isinstance(columns, Mapping):
            raise ValueError("evaluate takes an already-typed Arrow table and its column declarations")
        if type(whole) is not bool or type(ordered) is not bool:
            raise ValueError("whole and ordered are explicit booleans")
        if type(offset) is not int or offset < 0:
            raise ValueError("row offset is a nonnegative whole number")
        if isinstance(positions, (pa.Array, pa.ChunkedArray)):
            positions = positions.to_pylist()
        if positions is not None and (not isinstance(positions, (list, tuple)) or len(positions) != table.num_rows
                                      or any(type(value) is not int or value < 0 for value in positions)):
            raise ValueError("expression positions states one nonnegative whole input-row position per row")
        self.positions = tuple(range(table.num_rows)) if positions is None else tuple(positions)
        if neighbors is not None:
            if not ordered or not isinstance(neighbors, Mapping) or set(neighbors) != {"previous", "following"}:
                raise ValueError("ordered neighbors states previous and following row maps")
            neighbors = {name: list(values) for name, values in neighbors.items()}
            if any(len(values) != table.num_rows or any(
                value is not None and (type(value) is not int or not 0 <= value < table.num_rows) for value in values
            ) for values in neighbors.values()):
                raise ValueError("ordered neighbors maps each row to an existing row or missing")
        self.neighbors = neighbors
        self.tables, self.columns = {None: table}, {None: dict(columns)}
        for name, entry in (namespaces or {}).items():
            if name not in ("earlier", "later", "other") or not isinstance(entry, (tuple, list)) or len(entry) != 2 \
                    or not isinstance(entry[0], pa.Table) or not isinstance(entry[1], Mapping):
                raise ValueError("namespaces states earlier/later or aligned other as (Arrow table, column declarations)")
            if name == "other" and entry[0].num_rows != table.num_rows:
                raise ValueError("the other namespace has exactly one aligned row per main row")
            self.tables[name], self.columns[name] = entry[0], dict(entry[1])
        self.whole, self.unit, self.offset, self.ordered, self.through = whole, unit, offset, ordered, through
        self.kinds, self.inferred, self.reductions = {}, {}, {}
        checked(sum(value.nbytes * 4 + value.num_rows * 64 for value in self.tables.values()), "expression table working storage")

    @timed("expressions.column_kind")
    def column(self, name, scope=None):
        if scope not in self.tables:
            raise ValueError("the expression has no resolved %s rows" % scope)
        table, columns = self.tables[scope], self.columns[scope]
        if name not in table.column_names:
            raise ValueError("unknown column %s%s" % ("" if scope is None else scope + ".", name))
        key = scope, name
        if key not in self.kinds:
            self.kinds[key] = column_kind(columns[name] if name in columns else columns.get("*"), table[name].type, self.unit)
        return self.kinds[key]

    @timed("expressions.context")
    def context(self, arguments=(), keywords=None, ids=(), scope=None, deadline=False):
        through = self.through
        if isinstance(through, (pa.Array, pa.ChunkedArray)):
            if not deadline:
                through = UNSTATED
            elif len(through) != self.tables[scope].num_rows:
                raise ValueError("through has one value per expression row")
            else:
                through = pc.take(through, pa.array(ids, pa.int64()))
        offset = self.offset if scope is None else 0
        return Call(self.unit, len(ids), tuple(arguments), keywords or {}, offset, self.ordered, through,
                    tuple(self.position(value, scope) for value in ids), self.neighbors if scope is None else None)

    @timed("expressions.position")
    def position(self, index, scope=None):
        if scope is not None:
            return index
        return self.offset + (self.positions[index] if 0 <= index < len(self.positions) else index)

    @timed("expressions.function")
    def function(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("an expression calls only a registered function by its name")
        entry = FUNCTIONS.named(node.func.id)
        if not callable(entry) or version(entry) is None or not isinstance(getattr(entry, "reads", None), tuple) \
                or type(getattr(entry, "aggregate", None)) is not bool or not hasattr(entry, "gives"):
            raise ValueError("function %s states reads, gives, aggregate and a version" % node.func.id)
        if not isinstance(entry, Function) and type(entry.gives) is not str:
            raise ValueError("a custom function states gives as a registered type name")
        return entry

    @timed("expressions.derived")
    def derived(self, node):
        return any(isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                   and value.func.id != "column" and self.function(value).aggregate for value in ast.walk(node))

    @timed("expressions.varying")
    def varying(self, node, inside=False):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "column" or node.func.id in ("row", "previous", "following") and not inside:
                return not inside
            inside = inside or self.function(node).aggregate
            children = [*node.args, *(keyword.value for keyword in node.keywords)]
        elif isinstance(node, ast.Name):
            return not inside and node.id not in ("missing", "never")
        elif isinstance(node, ast.Attribute):
            return not inside
        else:
            children = list(ast.iter_child_nodes(node))
        return any(self.varying(child, inside) for child in children)

    @timed("expressions.result_kind")
    def result_kind(self, entry, arguments, keywords):
        with calling(self.context(arguments, keywords)):
            kind = entry.gives(arguments, keywords) if callable(entry.gives) else entry.gives
        if type(kind) is str:
            if kind not in SHIPPED and kind != "missing":
                raise ValueError("a function gives a registered column type, found %r" % kind)
            kind = Kind(kind, unit=self.unit if kind in ("time", "interval", "duration") else None)
        if not isinstance(kind, Kind):
            raise ValueError("a function resolves its result type explicitly")
        if kind.name in ("time", "interval", "duration") and kind.unit is UNSTATED:
            raise ValueError("a temporal function requires an explicit graph unit")
        datatype(kind)
        return kind

    @timed("expressions.operation_kind")
    def operation_kind(self, symbol, left, right):
        if "missing" in (left.name, right.name):
            other = right if left.name == "missing" else left
            return Kind("flag") if symbol in tuple(value[0] for value in COMPARE.values()) else other
        if left.name in ("time", "duration") or right.name in ("time", "duration"):
            with calling(self.context(ids=())):
                return expression_temporal.arithmetic(symbol, Value(pa.array([], datatype(left)), left),
                                                      Value(pa.array([], datatype(right)), right), self.unit).kind
        if symbol in tuple(value[0] for value in COMPARE.values()):
            common(left, right)
            return Kind("flag")
        if left.name not in NUMBERS or right.name not in NUMBERS:
            raise ValueError("arithmetic reads numbers or declared temporal values")
        return Kind("number")

    @timed("expressions.infer")
    def infer(self, node, aggregate=False):
        key = id(node), aggregate
        if key in self.inferred:
            return self.inferred[key]
        result = None
        if isinstance(node, (ast.Constant, ast.List)) or isinstance(node, ast.Name) and node.id in ("missing", "never"):
            result = literal(node, self.unit)
        elif isinstance(node, ast.Name):
            result = Value(None, self.column(node.id))
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in ("earlier", "later", "other"):
                raise ValueError("an attribute reads an earlier, later or aligned other column")
            if node.value.id != "other" and not aggregate:
                raise ValueError("earlier/later columns are read only inside an aggregate")
            result = Value(None, self.column(node.attr, node.value.id))
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Dict):
                raise ValueError("a subscript reads a literal lookup mapping")
            _, keys, kind = mapping(node.value, self.unit)
            common(keys, self.infer(node.slice, aggregate).kind)
            result = Value(None, kind)
        elif isinstance(node, ast.BinOp):
            left, right = self.infer(node.left, aggregate), self.infer(node.right, aggregate)
            result = Value(None, self.operation_kind(BINARY[type(node.op)][0], left.kind, right.kind))
        elif isinstance(node, ast.UnaryOp):
            value = self.infer(node.operand, aggregate)
            if isinstance(node.op, ast.Not):
                if value.type not in ("flag", "missing"):
                    raise ValueError("not reads a flag")
                result = Value(None, Kind("flag"))
            else:
                if value.type not in (*NUMBERS, "duration", "missing"):
                    raise ValueError("a signed value is a number or duration")
                result = Value(None, value.kind)
        elif isinstance(node, ast.BoolOp):
            if any(self.infer(value, aggregate).type not in ("flag", "missing") for value in node.values):
                raise ValueError("and/or read flags")
            result = Value(None, Kind("flag"))
        elif isinstance(node, ast.Compare):
            left = self.infer(node.left, aggregate)
            for operation, right_node in zip(node.ops, node.comparators, strict=True):
                right = self.infer(right_node, aggregate)
                if isinstance(operation, (ast.Is, ast.IsNot)):
                    if not isinstance(right_node, ast.Name) or right_node.id != "missing":
                        raise ValueError("is and is not compare with missing")
                elif isinstance(operation, (ast.In, ast.NotIn)):
                    if not scalar(right) or right.type != "list":
                        raise ValueError("in and not in read a literal list")
                    common(left.kind, right.kind.of)
                else:
                    self.operation_kind(COMPARE[type(operation)][0], left.kind, right.kind)
                left = right
            result = Value(None, Kind("flag"))
        elif isinstance(node, ast.IfExp):
            if self.infer(node.test, aggregate).type not in ("flag", "missing"):
                raise ValueError("a conditional predicate is a flag")
            result = Value(None, common(self.infer(node.body, aggregate).kind, self.infer(node.orelse, aggregate).kind))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "column":
                if len(node.args) != 1 or node.keywords:
                    raise ValueError("column takes one literal column name")
                name = literal(node.args[0], self.unit)
                if name.type not in NAMES:
                    raise ValueError("column takes one literal column name")
                result = Value(None, self.column(name.values.as_py()))
            else:
                if isinstance(node.func, ast.Name) and node.func.id == "hash" \
                        and not any(keyword.arg == "salt" for keyword in node.keywords):
                    if len(node.args) < 2:
                        raise ValueError("hash requires at least one key and an explicit final positional salt or salt=")
                    salt = node.args[-1]
                    if not isinstance(salt, ast.Constant) or type(salt.value) is not str or not salt.value:
                        raise ValueError("hash salt is a nonempty string literal")
                    node.args = node.args[:-1]
                    node.keywords.append(ast.keyword(arg="salt", value=salt))
                entry = self.function(node)
                if entry.aggregate and (not self.whole or aggregate):
                    raise ValueError("an aggregate needs whole=True on a complete table and cannot nest another aggregate")
                if node.func.id in ("previous", "following") and not self.ordered:
                    raise ValueError("previous and following require an explicitly ordered context")
                arguments = [self.infer(value, aggregate or entry.aggregate) for value in node.args]
                keywords = {}
                for keyword in node.keywords:
                    if keyword.arg is None or keyword.arg in keywords:
                        raise ValueError("function keyword names are explicit and distinct")
                    if keyword.arg not in getattr(entry, "keywords", ()) and not (entry.aggregate and keyword.arg == "where"):
                        raise ValueError("function %s does not take %s" % (node.func.id, keyword.arg))
                    keywords[keyword.arg] = self.infer(keyword.value, aggregate or entry.aggregate)
                minimum, maximum = getattr(entry, "minimum", len(entry.reads)), getattr(entry, "maximum", len(entry.reads))
                if len(arguments) < minimum or maximum is not None and len(arguments) > maximum:
                    raise ValueError("function %s has the wrong number of arguments" % node.func.id)
                for index, argument in enumerate(arguments):
                    expected = entry.reads[min(index, len(entry.reads) - 1)]
                    if not accepts(expected, argument.type):
                        raise ValueError("function %s argument %d reads %r, found %s" % (node.func.id, index + 1, expected, argument.type))
                if "where" in keywords and keywords["where"].type not in ("flag", "missing"):
                    raise ValueError("an aggregate where is a flag")
                if node.func.id == "hash" and "salt" not in keywords:
                    raise ValueError("hash requires an explicit salt")
                if node.func.id == "hash" and (not scalar(keywords["salt"]) or keywords["salt"].type not in NAMES
                                                or not keywords["salt"].values.as_py()):
                    raise ValueError("hash salt is a nonempty string literal")
                result = Value(None, self.result_kind(entry, arguments, keywords))
        if result is None:
            raise ValueError("unsupported expression form %s" % type(node).__name__)
        self.inferred[key] = result
        return result

    @timed("expressions.column_values")
    def column_values(self, name, ids, scope=None, current=None):
        kind = self.column(name, scope)
        values = self.tables[scope][name]
        if current in ("earlier", "later") and scope in (None, "other"):
            if len(values) != 1:
                raise ValueError("an unqualified column in window rows requires one current row")
            return Value(values[0], kind)
        values = pc.take(values, pa.array(ids, pa.int64()))
        if values.type != datatype(kind):
            values = values.cast(datatype(kind), safe=True)
        return Value(values, kind)

    @timed("expressions.binary")
    def binary(self, operation, left, right, ids, scope):
        symbol, function = operation
        kind = self.operation_kind(symbol, left.kind, right.kind)
        constant = scalar(left) and scalar(right)
        size = 1 if constant else len(ids)
        with calling(self.context((left, right), ids=ids[:size] if not constant else (0,), scope=scope)):
            if left.type == "missing" or right.type == "missing":
                result = pa.nulls(size, datatype(kind))
                return Value(result[0] if constant else result, kind)
            if left.type in ("time", "duration") or right.type in ("time", "duration"):
                result = expression_temporal.arithmetic(symbol, left, right, self.unit)
                return Value(result.values[0] if constant else result.values, kind)
            result = []
            for row, (one, other) in enumerate(zip(rows(left, size).to_pylist(), rows(right, size).to_pylist(), strict=True), 1):
                if one is None or other is None:
                    result.append(None)
                    continue
                try:
                    value = function(float(one), float(other)) if kind.name == "number" else function(one, other)
                    if kind.name == "number" and (isinstance(value, complex) or not math.isfinite(value)):
                        raise ValueError("arithmetic produces a finite real number")
                    result.append(value)
                except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
                    raise ColumnValueError(row, str(error)) from error
            result = pa.array(result, datatype(kind))
            return Value(result[0] if constant else result, kind)

    @timed("expressions.logic")
    def logic(self, values, conjunction, ids, scope):
        constant = all(scalar(value) for value in values)
        size = 1 if constant else len(ids)
        with calling(self.context(values, ids=ids[:size], scope=scope)):
            columns = [rows(value, size).to_pylist() for value in values]
        result = [None if any(value is None for value in row) else all(row) if conjunction else any(row)
                  for row in zip(*columns, strict=True)]
        array_result = pa.array(result, pa.bool_())
        return Value(array_result[0] if constant else array_result, Kind("flag"))

    @timed("expressions.invoke")
    def invoke(self, node, ids, scope, aggregate=False):
        entry = self.function(node)
        if entry.aggregate and id(node) in self.reductions:
            return self.reductions[id(node)]
        original_ids = ids
        if entry.aggregate:
            names = {value.value.id for value in ast.walk(node) if isinstance(value, ast.Attribute)
                     and isinstance(value.value, ast.Name) and value.value.id in ("earlier", "later")}
            if len(names) > 1:
                raise ValueError("one aggregate reads one earlier/later row scope")
            scope = next(iter(names)) if names else None
            ids = list(range(self.tables[scope].num_rows))
        elif node.func.id in ("previous", "following"):
            ids = list(range(self.tables[scope].num_rows))
        keywords_nodes = {keyword.arg: keyword.value for keyword in node.keywords}
        if entry.aggregate and "where" in keywords_nodes:
            predicate = self.run(keywords_nodes.pop("where"), ids, scope, True)
            with calling(self.context(ids=ids, scope=scope)):
                ids = [position for position, keep in zip(ids, rows(predicate, len(ids)).to_pylist(), strict=True) if keep is True]
        arguments = [self.run(value, ids, scope, aggregate or entry.aggregate) for value in node.args]
        keywords = {name: self.run(value, ids, scope, aggregate or entry.aggregate) for name, value in keywords_nodes.items()}
        kind = self.infer(node, aggregate).kind
        folded = not entry.aggregate and node.func.id not in ("row", "previous", "following") \
            and all(scalar(value) for value in (*arguments, *keywords.values())) and self.derived(node) \
            and not (node.func.id == "met" and isinstance(self.through, (pa.Array, pa.ChunkedArray)))
        if folded:
            ids = ids[:1] if ids else [0]
        with calling(self.context(arguments, keywords, ids, scope, node.func.id == "met")):
            arrays = [rows(value, len(ids)) for value in arguments]
            keyword_arrays = {name: rows(value, len(ids)) for name, value in keywords.items()}
            if entry.aggregate:
                if any(value.null_count for value in (*arrays, *keyword_arrays.values())):
                    result = pa.scalar(None, datatype(kind))
                else:
                    try:
                        if FUNCTIONS.shipped.get(node.func.id) is not entry:
                            arrays = [custom_argument(value, argument.kind) for value, argument in zip(arrays, arguments, strict=True)]
                            keyword_arrays = {name: custom_argument(value, keywords[name].kind)
                                              for name, value in keyword_arrays.items()}
                        result = entry(*arrays, **keyword_arrays)
                    except ColumnValueError as error:
                        position = ids[error.row - 1] if 0 < error.row <= len(ids) else error.row - 1
                        absolute = ColumnValueError(self.position(position, scope) + 1, error.detail)
                        absolute.absolute = True
                        raise absolute from error
                if not isinstance(result, pa.Scalar):
                    raise ValueError("an aggregate function returns exactly one Arrow scalar")
            else:
                valid = [True] * len(ids)
                reads_missing = node.func.id in ("coalesce", "met", "previous", "following") \
                    and FUNCTIONS.shipped.get(node.func.id) is entry
                if not reads_missing:
                    for value in (*arrays, *keyword_arrays.values()):
                        valid = [previous and present for previous, present in zip(valid, value.is_valid().to_pylist(), strict=True)]
                positions = [index for index, present in enumerate(valid) if present]
                selected = pa.array(positions, pa.int64())
                passed = [raw(value) if scalar(value) and literal_argument(argument) else pc.take(array_value, selected)
                          for value, argument, array_value in zip(arguments, node.args, arrays, strict=True)]
                options = {name: raw(value) if scalar(value) and literal_argument(keywords_nodes[name])
                           else pc.take(keyword_arrays[name], selected) for name, value in keywords.items()}
                active = [ids[position] for position in positions]
                with calling(self.context(arguments, keywords, active, scope, node.func.id == "met")):
                    try:
                        shipped = FUNCTIONS.shipped.get(node.func.id) is entry
                        if not shipped and active:
                            passed = [custom_argument(value, argument.kind) for value, argument in zip(passed, arguments, strict=True)]
                            options = {name: custom_argument(value, keywords[name].kind) for name, value in options.items()}
                        computed = entry(*passed, **options) if shipped or active else pa.array([], datatype(kind))
                    except ColumnValueError as error:
                        raise ColumnValueError(positions[error.row - 1] + 1, error.detail) from error
                if not isinstance(computed, (pa.Array, pa.ChunkedArray)) or len(computed) != len(active):
                    raise ValueError("a nonaggregate function returns an Arrow array of its input row count")
                indices, index = [], 0
                for present in valid:
                    indices.append(index if present else None)
                    index += int(present)
                result = pc.take(computed, pa.array(indices, pa.int64()))
            if result.type != datatype(kind):
                raise ValueError("function %s returns %s, its declared %s requires %s"
                                 % (node.func.id, result.type, kind.name, datatype(kind)))
            try:
                output = checked_output(result, kind)
            except ColumnValueError as error:
                if entry.aggregate:
                    raise ValueError("aggregate %s returned an invalid value: %s" % (node.func.id, error.detail)) from error
                position = ids[error.row - 1] if 0 < error.row <= len(ids) else error.row - 1
                absolute = ColumnValueError(self.position(position, scope) + 1, error.detail)
                absolute.absolute = True
                raise absolute from error
            checked(output.nbytes * 4, "expression function output")
        value = Value(result[0] if folded else result, kind)
        if entry.aggregate:
            self.reductions[id(node)] = value
        elif node.func.id in ("previous", "following"):
            value = Value(pc.take(result, pa.array(original_ids, pa.int64())), kind)
        return value

    @timed("expressions.run")
    def run(self, node, ids, scope=None, aggregate=False):
        inferred = self.infer(node, aggregate)
        if inferred.values is not None:
            return inferred
        try:
            if isinstance(node, ast.Name):
                return self.column_values(node.id, ids, current=scope)
            if isinstance(node, ast.Attribute):
                return self.column_values(node.attr, ids, node.value.id, scope)
            if isinstance(node, ast.Call):
                if node.func.id == "column":
                    return self.column_values(literal(node.args[0], self.unit).values.as_py(), ids, current=scope)
                return self.invoke(node, ids, scope, aggregate)
            if isinstance(node, ast.Subscript):
                options, _, kind = mapping(node.value, self.unit)
                keys = self.run(node.slice, ids, scope, aggregate)
                constant = scalar(keys)
                with calling(self.context(ids=ids, scope=scope)):
                    result = aggregates.lookup(options, rows(keys, 1 if constant else len(ids)))
                if result.type != datatype(kind):
                    result = result.cast(datatype(kind), safe=True)
                return Value(result[0] if constant else result, kind)
            if isinstance(node, ast.BinOp):
                return self.binary(BINARY[type(node.op)], self.run(node.left, ids, scope, aggregate),
                                   self.run(node.right, ids, scope, aggregate), ids, scope)
            if isinstance(node, ast.UnaryOp):
                value = self.run(node.operand, ids, scope, aggregate)
                if isinstance(node.op, ast.Not):
                    return Value(pc.invert(value.values), Kind("flag"))
                return self.binary(("*", operator.mul), value,
                                   Value(pa.scalar(-1 if isinstance(node.op, ast.USub) else 1), Kind("number")), ids, scope)
            if isinstance(node, ast.BoolOp):
                return self.logic([self.run(value, ids, scope, aggregate) for value in node.values],
                                  isinstance(node.op, ast.And), ids, scope)
            if isinstance(node, ast.Compare):
                left, comparisons = self.run(node.left, ids, scope, aggregate), []
                for operation, right_node in zip(node.ops, node.comparators, strict=True):
                    right = self.run(right_node, ids, scope, aggregate)
                    if isinstance(operation, (ast.Is, ast.IsNot)):
                        result = pc.is_null(left.values)
                        if isinstance(operation, ast.IsNot):
                            result = pc.invert(result)
                        comparisons.append(Value(result, Kind("flag")))
                    elif isinstance(operation, (ast.In, ast.NotIn)):
                        constant = scalar(left)
                        with calling(self.context(ids=ids, scope=scope)):
                            values = rows(left, 1 if constant else len(ids)).to_pylist()
                        options = right.values.as_py()
                        result = pa.array([None if value is None else (value not in options if isinstance(operation, ast.NotIn)
                                                                      else value in options) for value in values], pa.bool_())
                        comparisons.append(Value(result[0] if constant else result, Kind("flag")))
                    else:
                        comparisons.append(self.binary(COMPARE[type(operation)], left, right, ids, scope))
                    left = right
                return comparisons[0] if len(comparisons) == 1 else self.logic(comparisons, True, ids, scope)
            if isinstance(node, ast.IfExp):
                predicate = self.run(node.test, ids, scope, aggregate)
                if scalar(predicate) and not self.varying(node):
                    selected = predicate.values.as_py()
                    if selected is None:
                        return Value(pa.scalar(None, datatype(inferred.kind)), inferred.kind)
                    branch = self.run(node.body if selected else node.orelse, ids, scope, aggregate)
                    return Value(branch.values.cast(datatype(inferred.kind), safe=True), inferred.kind)
                with calling(self.context(ids=ids, scope=scope)):
                    flags = rows(predicate, len(ids)).to_pylist()
                parts = []
                for truth, branch in ((True, node.body), (False, node.orelse)):
                    selected = [value for value, flag in zip(ids, flags, strict=True) if flag is truth]
                    if selected:
                        value = self.run(branch, selected, scope, aggregate)
                        with calling(self.context(ids=selected, scope=scope)):
                            values = rows(value, len(selected)).cast(datatype(inferred.kind), safe=True)
                        positions = {value: index for index, value in enumerate(selected)}
                        parts.append(pc.take(values, pa.array([positions.get(value) for value in ids], pa.int64())))
                    else:
                        parts.append(pa.nulls(len(ids), datatype(inferred.kind)))
                return Value(pc.if_else(pa.array(flags, pa.bool_()), *parts), inferred.kind)
        except ColumnValueError as error:
            if getattr(error, "absolute", False):
                raise
            position = ids[error.row - 1] if ids and 0 < error.row <= len(ids) else error.row - 1
            result = ColumnValueError(self.position(position, scope) + 1, error.detail)
            result.absolute = True
            raise result from error
        raise ValueError("unsupported expression form %s" % type(node).__name__)


def evaluate(text, table, columns, *, plan, whole=False, unit=UNSTATED, declaration="expression", field="value",
             offset=0, ordered=False, through=UNSTATED, namespaces=None, positions=None, neighbors=None):
    started = time.perf_counter()
    try:
        with reading(stated(plan)):
            if type(text) is not str or not text:
                raise ValueError("an expression is a nonempty string")
            checked(len(text.encode("utf-8")) * 64, "parsing an expression")
            tree = ast.parse(text, mode="eval")
            nodes = list(ast.walk(tree))
            if len(nodes) > 4096 or any(not isinstance(node, ALLOWED) for node in nodes):
                raise ValueError("the expression uses an unsupported or excessively large Python syntax form")
            evaluator = Evaluator(table, columns, whole, unit, offset, ordered, through, namespaces, positions, neighbors)
            evaluator.infer(tree.body)
            result = evaluator.run(tree.body, list(range(table.num_rows)))
            return result
    except Exception as error:
        raise ValueError("declaration %s field %s: %s" % (declaration, field, error)) from error
    finally:
        LOG.debug("evaluate exit declaration=%s field=%s seconds=%.6f", declaration, field, time.perf_counter() - started)
