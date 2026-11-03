import bz2
import gzip
import logging
import lzma
import tarfile
import time
import zipfile
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path

from compression import zstd

from graphspec.infra.timing import timed

from .planning import stated

LOG = logging.getLogger(__name__)
CURRENT = ContextVar("source_read_budget", default=None)
COMPRESSED = {".gz": gzip.GzipFile, ".bz2": bz2.BZ2File, ".xz": lzma.LZMAFile, ".zst": zstd.ZstdFile}


@contextmanager
def reading(plan):
    started = time.perf_counter()
    token = CURRENT.set(stated(plan).worker_bytes)
    try:
        yield
    finally:
        CURRENT.reset(token)
        LOG.debug("source budget exit seconds=%.6f", time.perf_counter() - started)


@timed("streams.capacity")
def capacity():
    limit = CURRENT.get()
    if limit is None:
        raise ValueError("read a source with an explicit BuildPlan")
    return limit


@timed("streams.checked")
def checked(size, operation):
    limit = capacity()
    if size > limit:
        raise ValueError("%s needs %d bytes, above the source worker budget of %d bytes" % (operation, size, limit))


@timed("streams.loaded")
def loaded(stream, operation):
    limit = capacity() // 4
    result = stream.read(limit + 1)
    checked(len(result) * 4, operation)
    return result


@timed("streams.decoded")
def decoded(stream, name, stack):
    while Path(name).suffix.lower() in COMPRESSED:
        suffix = Path(name).suffix.lower()
        kind = COMPRESSED[suffix]
        stream = stack.enter_context(kind(fileobj=stream, mode="rb") if suffix == ".gz" else kind(stream, "rb"))
        name = name[:-len(suffix)]
    return stream, name


@contextmanager
def opened(place, member=None):
    started = time.perf_counter()
    with ExitStack() as stack:
        stream, name = decoded(stack.enter_context(Path(place).open("rb")), str(place), stack)
        suffix = Path(name).suffix.lower()
        if suffix in (".zip", ".tar"):
            if type(member) is not str or not member:
                raise ValueError("the archive %s needs the name of its member" % place)
            if suffix == ".zip":
                archive = stack.enter_context(zipfile.ZipFile(stream))
                if archive.getinfo(member).is_dir():
                    raise ValueError("the archive member %s is a directory" % member)
                stream = stack.enter_context(archive.open(member))
            else:
                archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r:*"))
                info = archive.getmember(member)
                if not info.isfile():
                    raise ValueError("the archive member %s is not a regular file" % member)
                stream = stack.enter_context(archive.extractfile(info))
            stream, name = decoded(stream, member, stack)
        elif member is not None:
            raise ValueError("a member is read from a zip or tar archive, found %s" % place)
        try:
            yield stream
        finally:
            LOG.debug("opened exit place=%s member=%s seconds=%.6f", place, member, time.perf_counter() - started)


@timed("streams.selected")
def selected(table, columns):
    missing = [name for name in columns if name != "*" and name not in table.column_names]
    if missing:
        raise ValueError("the source lacks the declared columns %s" % ", ".join(missing))
    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError("the source repeats a column name")
    return table if "*" in columns else table.select(columns)


def yielded(table, columns):
    started = time.perf_counter()
    table = selected(table, columns)
    limit = capacity() // 4
    offset = 0
    try:
        while offset < table.num_rows:
            size = min(65536, table.num_rows - offset)
            part = table.slice(offset, size)
            while part.nbytes > limit and size > 1:
                size = max(1, size // 2)
                part = table.slice(offset, size)
            checked(part.nbytes * 4, "one source batch")
            yield part
            offset += size
        if not table.num_rows:
            yield table
    finally:
        LOG.debug("yielded exit rows=%d seconds=%.6f", table.num_rows, time.perf_counter() - started)
