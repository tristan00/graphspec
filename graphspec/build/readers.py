import copy
import logging
import runpy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .streams import reading, selected, yielded

LOG = logging.getLogger(__name__)
READERS = Names("reader", False)


@timed("readers.namespace")
def namespace(path, sources):
    return runpy.run_path(str(Path(path).resolve()), init_globals=dict(sources=Path(sources).resolve()))


@timed("readers.declarations")
def declarations(namespace):
    result = {name: value for name, value in namespace.items() if isinstance(value, Source)}
    if not result:
        raise ValueError("a declaration script exports source declarations")
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
        if not isinstance(columns, Mapping) or not columns or any(type(name) is not str or not name for name in columns):
            raise ValueError("a source states a nonempty mapping of column names to types")
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
        columns = tuple(declaration.columns)
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
