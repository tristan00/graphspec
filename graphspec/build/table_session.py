import ast
import logging
import time
from functools import partial

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.schema.columns import checked as checked_column
from graphspec.schema.columns import convert, held, statement, temporal_columns
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.missing import normalize, resolve
from graphspec.schema.temporal import checked as checked_time
from graphspec.schema.temporal import context as temporal_context
from graphspec.schema.temporal import whole

from .expression_types import FUNCTIONS, NEVER_TIME, UNSTATED
from .expressions import checked_output, column_kind, custom_argument
from .planning import stated
from .readers import Source, SourceSession, typed
from .streams import reading
from .table_types import TABLES, Context, Frame, references, working
from .tables import Table

LOG = logging.getLogger(__name__)


class Unread(ValueError):
    pass


@timed("table_session.collect")
def collect(declarations, target=None, name=None):
    names, nodes, active = {}, {}, set()
    for label, declaration in declarations.items():
        if isinstance(declaration, (Source, Table)):
            names.setdefault(id(declaration), label)
    if target is not None and name is not None:
        names[id(target)] = name
    pending = [(value, names[id(value)], False) for value in declarations.values() if isinstance(value, (Source, Table))]
    if target is not None:
        pending.append((target, name or names.get(id(target), str(target.place) if isinstance(target, Source) else "table"), False))
    while pending:
        declaration, label, exiting = pending.pop()
        identity = id(declaration)
        if exiting:
            active.remove(identity)
            nodes[identity] = declaration
            continue
        if identity in nodes:
            continue
        if identity in active:
            raise ValueError("table declarations form a cycle at %s" % label)
        if not isinstance(declaration, (Source, Table)):
            raise ValueError("table inputs are source or table declarations")
        names.setdefault(identity, label)
        active.add(identity)
        pending.append((declaration, names[identity], True))
        if isinstance(declaration, Table):
            pending.extend((value, "%s.inputs.%d" % (names[identity], index), False)
                           for index, value in reversed(tuple(enumerate(declaration.inputs))))
    if len(set(names.values())) != len(names):
        raise ValueError("table declaration names collide with an implicit input declaration")
    sources = {name: nodes[identity] for identity, name in names.items() if isinstance(nodes.get(identity), Source)}
    temporal = []
    for identity, declaration in nodes.items():
        if isinstance(declaration, Table):
            columns = declaration.columns
            if declaration.operation in ("expand", "nested"):
                columns = declaration.parameters.get("columns")
            for column, given in (columns or {}).items():
                temporal.extend(temporal_columns(given, "%s.%s" % (names[identity], column)))
    return names, nodes, sources, tuple(temporal)


@timed("table_session.constant")
def constant(text):
    if references(text):
        return False
    return not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and (node.func.id in ("row", "previous", "following") or FUNCTIONS.named(node.func.id).aggregate)
                   for node in ast.walk(ast.parse(text, mode="eval")))


@timed("table_session.completeness")
def completeness(frame, context):
    if frame.through is None:
        return UNSTATED
    if type(frame.through) is not str or not frame.through:
        raise ValueError("source through states a nonempty time expression")
    value = context.constant(frame.through, "through") if constant(frame.through) \
        else context.evaluate(frame, frame.through, "through")
    if value.type not in ("time", "number", "missing"):
        raise ValueError("source through gives a time or whole count in the graph unit")
    scalar = value.values
    if isinstance(scalar, (pa.Array, pa.ChunkedArray)):
        unique = pc.unique(scalar)
        if len(unique) != 1:
            raise ValueError("source through must resolve to one scalar, not row-varying completeness")
        scalar = unique[0]
    if not scalar.is_valid:
        return pa.scalar(None, pa.int64())
    count = whole(scalar.as_py())
    if count == NEVER_TIME and value.type == "time":
        return scalar
    return pa.scalar(checked_time(count), pa.int64())


@timed("table_session.replacement")
def replacement(values, declared, kind, unit):
    if kind.name in ("time", "interval"):
        declared = dict(declared, type=kind.name, unit=unit)
        declared.pop("at", None)
    return convert(values, declared)


@timed("table_session.canonical")
def canonical(table, columns, context, offset):
    missing = set(columns) - {"*"} - set(table.column_names)
    if missing:
        raise ValueError("custom table output lacks columns %s" % ", ".join(sorted(missing)))
    arrays, kinds, outcomes = [], {}, {}
    for column in table.column_names:
        try:
            given = columns[column] if column in columns else columns.get("*")
            declared, _, _ = checked_column(given)
            kind = column_kind(declared, table[column].type, context.unit)
            with temporal_context(context.unit):
                values = normalize(table[column], declared)
                checked_output(values, kind)
                held(values, declared)
                values = resolve(values, declared, partial(replacement, declared=declared, kind=kind, unit=context.unit))
            checked_output(values, kind)
            held(values, declared)
            arrays.append(values)
            kinds[column] = kind
            if "outcome" in declared:
                outcomes[column] = ((context.name, declared["outcome"]),)
        except ColumnValueError as error:
            raise ValueError("column %s row %d: %s" % (column, offset + error.row, error.detail)) from error
        except Exception as error:
            raise ValueError("column %s: %s" % (column, error)) from error
    return pa.Table.from_arrays(arrays, names=table.column_names), kinds, outcomes


class Session:
    @timed("table_session.Session")
    def __init__(self, declarations, sources, plan, target=None, name=None, aliases=None):
        self.plan = stated(plan)
        self.aliases = dict(aliases or {})
        self.names, self.nodes, source_declarations, temporal = collect(declarations, target, name)
        self.readers = SourceSession(source_declarations, sources, self.plan, temporal)
        self.frames, self.pending = {}, set()
        self.sources = sources
        self.unit = UNSTATED

    @timed("table_session.enter")
    def __enter__(self):
        return self

    @timed("table_session.exit")
    def __exit__(self, kind, value, traceback):
        try:
            self.readers.close()
        finally:
            self.frames.clear()

    @timed("table_session.retained")
    def retained(self):
        return sum(frame.nbytes * 4 for frame in self.frames.values()) + self.readers.retained * 4

    @timed("table_session.context")
    def context(self, declaration):
        if self.unit is UNSTATED:
            self.unit = self.readers.unit()
        return Context(self.names[id(declaration)], self.plan, self.unit, self.retained())

    @timed("table_session.source")
    def source(self, declaration, context):
        iterator = typed(declaration, self.sources, self.plan, context.name, self.readers.declarations, self.readers)
        parts, size = [], 0
        try:
            for part in iterator:
                size += part.nbytes
                context.retained = self.retained()
                context.check(size * 4, "materializing a complete source")
                parts.append(part)
        finally:
            iterator.close()
        table = pa.concat_tables(parts)
        columns = declaration.columns
        kinds, outcomes, lineage = {}, {}, {}
        for name in table.column_names:
            given = columns[name] if name in columns else columns.get("*")
            kinds[name] = column_kind(given, table[name].type, self.unit)
            lineage[name] = ((context.name, name),)
            declared = statement(given)
            if "outcome" in declared:
                outcomes[name] = ((context.name, declared["outcome"]),)
        frame = Frame(table, kinds, self.unit, declaration.through, outcomes, lineage)
        context.check(frame.nbytes * 4, "complete source frame")
        retained = context.retained
        context.retained += frame.nbytes * 4
        try:
            complete = completeness(frame, context)
        finally:
            context.retained = retained
        return Frame(table, kinds, self.unit, declaration.through, outcomes, lineage, complete=complete)

    @timed("table_session.custom")
    def custom(self, declaration, frames, context):
        entry = TABLES.named(declaration.operation)
        inputs, retained, input_bytes = [], context.retained, 0
        iterator, parts, size, count, schema, kinds, outcomes = None, [], 0, 0, None, None, None
        end = object()
        try:
            for frame in frames:
                arrays = []
                for name in frame.table.column_names:
                    try:
                        with reading(context.budget()):
                            arrays.append(custom_argument(frame.table[name], frame.kinds[name]))
                    except ColumnValueError as error:
                        raise ValueError("custom table input column %s row %d: %s" % (name, error.row, error.detail)) from error
                given = pa.Table.from_arrays(arrays, names=frame.table.column_names)
                context.check(given.nbytes * 4, "custom callback inputs")
                input_bytes += given.nbytes * 4
                context.retained = retained + input_bytes
                inputs.append(iter((given,)))
            with reading(context.budget()), temporal_context(context.unit):
                iterator = iter(entry(inputs, declaration.parameters))
            while True:
                with reading(context.budget()), temporal_context(context.unit):
                    part = next(iterator, end)
                if part is end:
                    break
                if not isinstance(part, pa.Table):
                    raise ValueError("a custom table callback yields Arrow tables")
                context.check(part.nbytes * 8, "custom table output batch")
                part, current_kinds, current_outcomes = canonical(part, declaration.columns, context, count)
                if schema is None:
                    schema, kinds, outcomes = part.schema, current_kinds, current_outcomes
                elif not schema.equals(part.schema, check_metadata=False) or kinds != current_kinds:
                    raise ValueError("a custom table callback keeps one typed schema across batches")
                size += part.nbytes
                count += part.num_rows
                parts.append(part)
                context.retained = retained + input_bytes + size * 4
            if schema is None:
                raise ValueError("a custom table callback yields its schema even for an empty result")
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    with reading(self.plan), temporal_context(context.unit):
                        close()
            finally:
                context.retained = retained
        parents = tuple(self.names[id(parent)] for parent in declaration.inputs)
        opaque = ((context.name, ("custom", declaration.operation, parents)),)
        return Frame(pa.concat_tables(parts), kinds, self.unit, frames[0].through, outcomes,
                     {name: opaque for name in kinds}, complete=frames[0].complete)

    @timed("table_session.get")
    def get(self, declaration):
        declaration = self.aliases.get(id(declaration), declaration)
        identity = id(declaration)
        if identity in self.frames:
            return self.frames[identity]
        if identity in self.pending:
            raise ValueError("table dependencies form a cycle")
        self.pending.add(identity)
        name = self.names[identity]
        try:
            context = self.context(declaration)
            if isinstance(declaration, Source):
                with working(context):
                    result = self.source(declaration, context)
            else:
                frames = [self.get(parent) for parent in declaration.inputs]
                context.retained = self.retained()
                entry = TABLES.named(declaration.operation)
                with working(context):
                    result = entry.apply(frames, declaration.parameters, context) if hasattr(entry, "apply") \
                        else self.custom(declaration, frames, context)
            if not isinstance(result, Frame):
                raise ValueError("a shipped table operation returns one typed Frame")
            context.retained = self.retained()
            context.check(result.nbytes * 4 + result.table.num_rows * 80, "retaining table result and row provenance")
            rows = pa.array(range(result.table.num_rows), pa.int64())
            origins = dict(result.provenance)
            origins[context.name] = pa.ListArray.from_arrays(pa.array(range(result.table.num_rows + 1), pa.int32()), rows)
            result = Frame(result.table, result.kinds, self.unit, result.through, result.outcomes, result.lineage,
                           origins, result.complete, rows)
            context.check(result.nbytes * 4, "retaining complete table")
            self.frames[identity] = result
            return result
        except Unread:
            raise
        except Exception as error:
            where = "table %s" % name if isinstance(declaration, Table) else "source %s (%s)" % (name, declaration.place)
            text = str(error)
            raise Unread(text if text.startswith(where) else "%s: %s" % (where, text)) from error
        finally:
            self.pending.remove(identity)


def batches(declaration, declarations, sources, plan, name=None):
    started = time.perf_counter()
    count = 0
    try:
        with Session(declarations, sources, plan, declaration, name) as session:
            frame = session.get(declaration)
            if not frame.table.num_rows:
                yield frame.table
            else:
                for batch in frame.table.to_batches(max_chunksize=65536):
                    table = pa.Table.from_batches([batch])
                    count += table.num_rows
                    yield table
    finally:
        LOG.debug("table batches exit name=%s rows=%d seconds=%.6f", name, count, time.perf_counter() - started)
