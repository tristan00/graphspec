import copy
import logging
import runpy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.columns import checked as checked_column
from graphspec.schema.columns import (
    completed,
    convert,
    prepared,
    proposed,
    scalar_context,
    statement,
    temporal_columns,
)
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.missing import apply, normalize
from graphspec.schema.temporal import context, unit

from .planning import BuildPlan, stated
from .streams import checked, reading, selected, yielded

LOG = logging.getLogger(__name__)
READERS = Names("reader", False)


@timed("readers.namespace")
def namespace(path, sources):
    return runpy.run_path(str(Path(path).resolve()), init_globals=dict(sources=Path(sources).resolve()))


@timed("readers.declarations")
def declarations(namespace):
    from .database import Database
    from .mappings import Mapping
    from .tables import Table

    result = {name: value for name, value in namespace.items() if isinstance(value, (Source, Table, Mapping, Database))}
    if not result:
        raise ValueError("a declaration script exports source, table or mapping declarations")
    return result


@dataclass(frozen=True, init=False, eq=False)
class Source:
    place: str | Path
    reader: str
    _columns: dict
    _parameters: dict
    _through: object

    @timed("readers.source")
    def __init__(self, place, reader, columns, through=None, **parameters):
        if not isinstance(place, (str, Path)) or not str(place):
            raise ValueError("a source states its place as a nonempty path or connection string")
        if type(reader) is not str or not reader:
            raise ValueError("a source states the name of its reader")
        if not isinstance(columns, Mapping) or any(type(name) is not str or not name for name in columns):
            raise ValueError("a source states a mapping of column names to types")
        object.__setattr__(self, "place", place)
        object.__setattr__(self, "reader", reader)
        object.__setattr__(self, "_columns", copy.deepcopy(dict(columns)))
        object.__setattr__(self, "_parameters", copy.deepcopy(parameters))
        object.__setattr__(self, "_through", copy.deepcopy(through))

    @property
    @timed("readers.columns")
    def columns(self):
        return copy.deepcopy(self._columns)

    @property
    @timed("readers.parameters")
    def parameters(self):
        return copy.deepcopy(self._parameters)

    @property
    @timed("readers.through")
    def through(self):
        return copy.deepcopy(self._through)


@timed("readers.declared")
def source(place, reader, columns, through=None, **parameters):
    return Source(place, reader, columns, through, **parameters)


def batches(declaration, sources, plan, name=None):
    started = time.perf_counter()
    where = "source %s" % (name if name is not None else declaration.place)
    count = 0
    iterator, parts = None, None
    end = object()
    try:
        reader = READERS.named(declaration.reader)
        place = declaration.place
        if getattr(reader, "file", True):
            place = Path(place)
            if not place.is_absolute():
                place = Path(sources) / place
            place = place.resolve()
        schema = None
        columns = tuple(declaration.columns) or ("*",)
        with reading(plan):
            iterator = iter(reader(place, columns, declaration.parameters))
        while True:
            with reading(plan):
                table = next(iterator, end)
                if table is end:
                    break
                if not isinstance(table, pa.Table):
                    raise ValueError("a reader yields pyarrow.Table batches")
                table = selected(table, columns)
                if schema is None:
                    schema = table.schema
                elif not schema.equals(table.schema, check_metadata=False):
                    raise ValueError("a reader yields batches of one schema")
                parts = yielded(table, columns)
            while True:
                with reading(plan):
                    batch = next(parts, end)
                if batch is end:
                    break
                count += batch.num_rows
                yield batch
        if schema is None:
            raise ValueError("a reader yields its schema even when it has no records")
    except Exception as error:
        raise ValueError("%s: %s" % (where, error)) from error
    finally:
        for pending in (parts, iterator):
            close = getattr(pending, "close", None)
            if close is not None:
                with reading(plan):
                    close()
        LOG.debug("batches exit source=%s rows=%d seconds=%.6f", name, count, time.perf_counter() - started)


class SourceSession:
    @timed("readers.SourceSession")
    def __init__(self, declarations, sources, plan, temporal=()):
        self.declarations = {name: declaration for name, declaration in declarations.items() if isinstance(declaration, Source)}
        self.sources, self.plan = sources, stated(plan)
        self.iterators, self.initial, self.consumed = {}, {}, set()
        self.retained, self.resolved, self.graph_unit = 0, False, None
        self.temporal = tuple(temporal)

    @timed("readers.SourceSession.enter")
    def __enter__(self):
        return self

    @timed("readers.SourceSession.exit")
    def __exit__(self, kind, value, traceback):
        self.close()

    @timed("readers.SourceSession.close")
    def close(self):
        errors = []
        for iterator in tuple(self.iterators.values()):
            try:
                iterator.close()
            except Exception as error:
                errors.append(error)
        self.iterators.clear()
        self.initial.clear()
        self.retained = 0
        if errors:
            raise ExceptionGroup("closing source readers", errors)

    @timed("readers.SourceSession.open")
    def open(self, name):
        if name in self.consumed:
            raise ValueError("source %s was already read in this operation" % name)
        if name not in self.iterators:
            remaining = self.plan.worker_bytes - self.retained * 4
            if remaining < 4096:
                raise ValueError("source lookahead leaves less than 4096 bytes for source %s; increase the BuildPlan budget" % name)
            iterator = batches(self.declarations[name], self.sources, BuildPlan(remaining, 1), name)
            self.iterators[name] = iterator
            table = next(iterator)
            self.initial[name] = table
            self.retained += table.nbytes
            with reading(self.plan):
                checked(self.retained * 4, "source lookahead")
        return self.initial[name]

    @timed("readers.SourceSession.unit")
    def unit(self):
        if not self.resolved:
            self.graph_unit = time_unit(self)
            self.resolved = True
        return self.graph_unit

    def rows(self, name):
        started = time.perf_counter()
        table = self.open(name)
        iterator = self.iterators[name]
        self.initial.pop(name)
        self.retained -= table.nbytes
        self.consumed.add(name)
        try:
            yield table
            del table
            yield from iterator
        finally:
            self.iterators.pop(name, None)
            iterator.close()
            LOG.debug("session rows exit source=%s seconds=%.6f", name, time.perf_counter() - started)


@timed("readers.time_unit")
def time_unit(session):
    columns = list(session.temporal)
    for name, declaration in session.declarations.items():
        declared = declaration.columns
        for column, given in declared.items():
            if column == "*" and temporal_columns(given, name):
                for unlisted in session.open(name).column_names:
                    if unlisted not in declared:
                        columns.extend(temporal_columns(given, "%s.%s" % (name, unlisted)))
                continue
            columns.extend(temporal_columns(given, "%s.%s" % (name, column)))
    return unit(columns)


def typed(declaration, sources, plan, name=None, declarations=None, session=None, temporal=()):
    started = time.perf_counter()
    declarations = declarations if declarations is not None else {name if name is not None else str(declaration.place): declaration}
    if name is None:
        name = next((key for key, value in declarations.items() if value is declaration), None)
    source_name = name if name is not None else str(declaration.place)
    where = "source %s (%s)" % (source_name, declaration.place)
    owns_session = session is None
    if owns_session:
        session = SourceSession(declarations, sources, plan, temporal)
    columns = declaration.columns
    offset, schema, iterator = 0, None, None
    try:
        graph_unit = session.unit()
        iterator = session.rows(source_name)
        for table in iterator:
            result = []
            with reading(plan), context(graph_unit), scalar_context(declaration.reader in ("csv", "delimited")):
                for column in table.column_names:
                    given = columns[column] if column in columns else columns.get("*")
                    try:
                        declared, _, _ = checked_column(given)
                        values = completed(convert(prepared(table[column], declared), declared), declared)
                        values = apply(values, declared, column, offset, partial(convert, given=declared))
                        result.append(values)
                    except ColumnValueError as error:
                        raise ValueError("%s column %s row %d: %s" %
                                         (where, column, offset + error.row, error.detail)) from error
                    except Exception as error:
                        raise ValueError("%s column %s: %s" % (where, column, error)) from error
                converted = pa.Table.from_arrays(result, names=table.column_names).replace_schema_metadata(table.schema.metadata)
                checked((table.nbytes + converted.nbytes) * 2 + session.retained * 4, "typed source batch and lookahead")
                if schema is None:
                    schema = converted.schema
                elif not schema.equals(converted.schema, check_metadata=False):
                    raise ValueError("%s: converted columns must keep one Arrow schema across batches" % where)
            offset += converted.num_rows
            yield converted
    finally:
        try:
            if iterator is not None:
                iterator.close()
        finally:
            if owns_session:
                session.close()
        LOG.debug("typed exit source=%s rows=%d seconds=%.6f", source_name, offset, time.perf_counter() - started)


def proposals(declaration, sources, plan, name=None):
    started = time.perf_counter()
    columns, result, rows = declaration.columns, {}, 0
    iterator = batches(declaration, sources, plan, name)
    try:
        for table in iterator:
            rows += table.num_rows
            for column in table.column_names:
                given = statement(columns[column] if column in columns else columns.get("*"))
                values = normalize(table[column], given)
                entry = result.setdefault(column, dict(type=given.get("type"), proposed=None, missing=0))
                entry["missing"] += values.null_count
                if not given.get("type"):
                    proposal = proposed(values, column)
                    if entry["proposed"] is None:
                        entry["proposed"] = proposal
                    elif proposal is not None and entry["proposed"] != proposal:
                        entry["proposed"] = "text" if pa.types.is_string(values.type) else None
        return dict(place=str(declaration.place), rows=rows, columns=result)
    finally:
        iterator.close()
        LOG.debug("proposals exit source=%s rows=%d seconds=%.6f", name, rows, time.perf_counter() - started)
