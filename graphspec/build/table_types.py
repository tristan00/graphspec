import ast
import logging
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property, partial
from types import MappingProxyType

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.columns import completed, convert, prepared
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.missing import resolve
from graphspec.schema.temporal import context as temporal_context

from .expression_types import FUNCTIONS, UNSTATED, Kind, Value, array
from .expressions import ALLOWED, Evaluator, column_kind, evaluate
from .planning import BuildPlan, stated
from .streams import checked, reading

LOG = logging.getLogger(__name__)
TABLES = Names("table operation", True)
ACTIVE = ContextVar("table_operation_context", default=None)


@timed("tables.frozen")
def frozen(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class Frame:
    table: pa.Table
    kinds: dict
    unit: object = UNSTATED
    through: object = None
    outcomes: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    complete: object = UNSTATED
    positions: object = None

    @timed("tables.Frame")
    def __post_init__(self):
        if not isinstance(self.table, pa.Table) or len(set(self.table.column_names)) != self.table.num_columns:
            raise ValueError("a table frame holds an Arrow table with distinct column names")
        if set(self.kinds) != set(self.table.column_names) or any(not isinstance(kind, Kind) for kind in self.kinds.values()):
            raise ValueError("a table frame states the exact Kind of every column")
        for name, kind in self.kinds.items():
            column_kind(kind, self.table[name].type, self.unit)
        if set(self.outcomes) - set(self.kinds) or set(self.lineage) - set(self.kinds):
            raise ValueError("table metadata names existing columns")
        if any(len(values) != self.table.num_rows or values.type != pa.list_(pa.int64()) for values in self.provenance.values()):
            raise ValueError("row provenance holds a list of parent row numbers for each output row")
        if self.positions is not None and len(self.positions) != self.table.num_rows:
            raise ValueError("table row positions align with its rows")
        if isinstance(self.complete, (pa.Array, pa.ChunkedArray)):
            raise ValueError("table completeness is one scalar, not row-varying values")
        for name in ("kinds", "outcomes", "lineage", "provenance"):
            object.__setattr__(self, name, frozen(getattr(self, name)))
        object.__setattr__(self, "through", frozen(self.through))
        if self.positions is not None and not isinstance(self.positions, (pa.Array, pa.ChunkedArray)):
            object.__setattr__(self, "positions", tuple(self.positions))

    @cached_property
    @timed("tables.Frame.nbytes")
    def nbytes(self):
        metadata = len(repr((self.kinds, self.outcomes, self.lineage, self.through)).encode("utf-8"))
        complete = pa.repeat(self.complete, 1).nbytes if isinstance(self.complete, pa.Scalar) else 0
        return self.table.nbytes + self.table.schema.serialize().size + metadata + complete \
            + sum(value.nbytes for value in self.provenance.values()) + (0 if self.positions is None else len(self.positions) * 8)


@timed("tables.guard")
def guard(size, operation):
    context = ACTIVE.get()
    if context is None:
        checked(size, operation)
    else:
        context.check(size, operation)


@contextmanager
def working(context):
    started = time.perf_counter()
    token = ACTIVE.set(context)
    try:
        with reading(context.plan):
            yield
    finally:
        ACTIVE.reset(token)
        LOG.debug("table context exit name=%s seconds=%.6f", context.name, time.perf_counter() - started)


@timed("tables.column")
def column(frame, name):
    if type(name) is not str or name not in frame.kinds:
        raise ValueError("the table has no column %r" % name)
    return frame.kinds[name]


@timed("tables.named")
def named(value, where):
    if type(value) is not str or not value:
        raise ValueError("%s states a nonempty column name" % where)
    return value


@timed("tables.fresh")
def fresh(frame, names):
    names = list(names)
    if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("new column names are distinct nonempty strings")
    collision = set(names) & set(frame.kinds)
    if collision:
        raise ValueError("new columns collide with input columns: " + ", ".join(sorted(collision)))
    return names


@timed("tables.generated")
def generated(frame, defaults, names=None):
    names = {} if names is None else names
    if not isinstance(names, Mapping) or set(names) - set(defaults):
        raise ValueError("names renames only columns this operation generates")
    result = {name: names.get(name, name) for name in defaults}
    fresh(frame, result.values())
    return result


@timed("tables.broadcast")
def broadcast(value, count):
    return array(value.values, rows=count)


@timed("tables.indices")
def indices(values, size):
    values = values.to_pylist() if isinstance(values, (pa.Array, pa.ChunkedArray)) else list(values)
    if any(value is not None and (type(value) is not int or not 0 <= value < size) for value in values):
        raise ValueError("row indices refer to existing input rows or an explicit missing row")
    return values


@timed("tables.take")
def take(frame, selected):
    selected = indices(selected, frame.table.num_rows)
    sizes = {}
    for value in set(selected) - {None}:
        sizes[value] = frame.table.slice(value, 1).nbytes + sum(column.slice(value, 1).nbytes for column in frame.provenance.values())
    guard(sum(sizes.get(value, 0) + 16 for value in selected) * 4, "taking table rows")
    chosen = pa.array(selected, pa.int64())
    positions = frame.positions if frame.positions is not None else pa.array(range(frame.table.num_rows), pa.int64())
    result = Frame(pc.take(frame.table, chosen), frame.kinds, frame.unit, frame.through, frame.outcomes, frame.lineage,
                   {name: pc.take(values, chosen) for name, values in frame.provenance.items()},
                   frame.complete,
                   pc.take(positions, chosen))
    guard(result.nbytes * 4, "taken table rows")
    return result


@timed("tables.carry")
def carry(frame, table, kinds, indices=None, outcomes=None, lineage=None, provenance=None):
    if indices is not None:
        parent = take(frame, indices)
    else:
        parent = frame
    if provenance is None:
        if parent.table.num_rows != table.num_rows:
            raise ValueError("a changed row count states its row provenance")
        provenance = parent.provenance
    return Frame(table, kinds, frame.unit, frame.through,
                 {name: marks for name, marks in frame.outcomes.items() if name in kinds} if outcomes is None else outcomes,
                 {name: origins for name, origins in frame.lineage.items() if name in kinds} if lineage is None else lineage,
                 provenance, parent.complete)


@timed("tables.combine_provenance")
def combine_provenance(frames, index_groups):
    if len(frames) != len(index_groups) or not frames:
        raise ValueError("provenance aligns one row-group mapping with each input table")
    count = len(index_groups[0])
    if any(len(groups) != count for groups in index_groups):
        raise ValueError("provenance mappings have one entry per output row")
    guard(sum(frame.nbytes for frame in frames) * 8 + count * 16, "gathering table provenance")
    result = {}
    for frame, groups in zip(frames, index_groups, strict=True):
        for name, values in frame.provenance.items():
            held = values.to_pylist()
            output = result.setdefault(name, [[] for _ in range(count)])
            for position, group in enumerate(groups):
                for row in indices(group, frame.table.num_rows):
                    if row is not None and held[row] is not None:
                        output[position].extend(held[row])
    guard(sum(sum(len(row) * 8 + 8 for row in rows) for rows in result.values()) * 8, "table provenance result")
    return {name: pa.array(values, pa.list_(pa.int64())) for name, values in result.items()}


@timed("tables.references")
def references(text):
    tree = ast.parse(text, mode="eval")
    functions = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    namespaces = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in functions | namespaces and node.id not in ("missing", "never"):
            result.append((None, node.id))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            result.append((node.value.id, node.attr))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "column" \
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            result.append((None, node.args[0].value))
    return tuple(dict.fromkeys(result))


@timed("tables.dependencies")
def dependencies(frame, text, namespace=None):
    frames = dict(namespace or {})
    frames[None] = frame
    lineage, outcomes = [], []
    for scope, name in references(text):
        if scope not in frames:
            raise ValueError("the expression has no table namespace %r" % scope)
        parent = frames[scope]
        column(parent, name)
        lineage.extend(parent.lineage.get(name, ()))
        outcomes.extend(parent.outcomes.get(name, ()))
    return tuple(dict.fromkeys(lineage)), tuple(dict.fromkeys(outcomes))


@timed("tables.aggregate_expression")
def aggregate_expression(node, inside=False):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "column":
            return inside, False
        entry = FUNCTIONS.named(node.func.id)
        if entry.aggregate:
            return True, True
        if node.func.id in ("row", "previous", "following"):
            return inside, False
        children = [*node.args, *(keyword.value for keyword in node.keywords)]
    elif isinstance(node, ast.Name):
        return inside or node.id in ("missing", "never"), False
    elif isinstance(node, ast.Attribute):
        return inside, False
    else:
        children = list(ast.iter_child_nodes(node))
    results = [aggregate_expression(child, inside) for child in children]
    return all(valid for valid, _ in results), any(found for _, found in results)


class Context:
    @timed("tables.Context")
    def __init__(self, name, plan, unit=UNSTATED, retained=0):
        self.name, self.plan, self.unit, self.retained = name, stated(plan), unit, retained

    @timed("tables.Context.check")
    def check(self, size, operation):
        with reading(self.plan):
            checked(self.retained + size, "%s of table %s" % (operation, self.name))

    @timed("tables.Context.budget")
    def budget(self):
        remaining = self.plan.worker_bytes - self.retained
        self.check(4096, "expression minimum working storage")
        return BuildPlan(remaining, 1)

    @timed("tables.Context.evaluate")
    def evaluate(self, frame, text, field, ordered=False, namespace=None, positions=None, neighbors=None):
        return evaluate(text, frame.table, frame.kinds, plan=self.budget(), whole=True, unit=self.unit,
                        declaration=self.name, field=field, ordered=ordered, through=frame.complete,
                        positions=frame.positions if positions is None else positions,
                        neighbors=neighbors,
                        namespaces={name: (value.table, value.kinds) for name, value in (namespace or {}).items()})

    @timed("tables.Context.infer")
    def infer(self, frame, text, field, aggregate=False, ordered=False, namespace=None):
        try:
            with reading(self.budget()):
                checked(len(text.encode("utf-8")) * 64, "parsing a table expression")
                tree = ast.parse(text, mode="eval")
                nodes = list(ast.walk(tree))
                if len(nodes) > 4096 or any(not isinstance(node, ALLOWED) for node in nodes):
                    raise ValueError("the table expression uses unsupported syntax")
                if aggregate and aggregate_expression(tree.body) != (True, True):
                    raise ValueError("this column requires an aggregate expression without freestanding row values")
                evaluator = Evaluator(frame.table, frame.kinds, True, self.unit, 0, ordered, frame.complete,
                                      {name: (value.table, value.kinds) for name, value in (namespace or {}).items()})
                return evaluator.infer(tree.body).kind
        except Exception as error:
            raise ValueError("declaration %s field %s: %s" % (self.name, field, error)) from error

    @timed("tables.Context.constant")
    def constant(self, text, field):
        if type(text) is not str:
            if type(text) not in (int, float):
                raise ValueError("declaration %s field %s states a number or constant expression" % (self.name, field))
            return Value(pa.scalar(text), Kind("number"))
        tree = ast.parse(text, mode="eval")
        if references(text) or any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                  and (node.func.id in ("row", "previous", "following") or FUNCTIONS.named(node.func.id).aggregate)
                                  for node in ast.walk(tree)):
            raise ValueError("declaration %s field %s requires a constant expression" % (self.name, field))
        value = evaluate(text, pa.table({"_": [None]}), {}, plan=self.budget(), whole=True, unit=self.unit,
                         declaration=self.name, field=field)
        return Value(value.values[0], value.kind) if isinstance(value.values, (pa.Array, pa.ChunkedArray)) else value


@timed("tables.typed")
def typed(values, declaration, context, column):
    try:
        with reading(context.budget()), temporal_context(context.unit):
            converted = completed(convert(prepared(values, declaration), declaration), declaration)
            converted = resolve(converted, declaration, partial(convert, given=declaration))
            return Value(converted, column_kind(declaration, converted.type, context.unit))
    except ColumnValueError as error:
        raise ColumnValueError(error.row, "column %s: %s" % (column, error.detail)) from error
    except Exception as error:
        raise ValueError("column %s: %s" % (column, error)) from error
