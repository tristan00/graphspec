import json
import logging
import re
import sqlite3
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path

import pyarrow as pa

from graphspec.build.readers import READERS
from graphspec.build.streams import COMPRESSED, capacity, checked, loaded, opened, yielded
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)


@timed("databases.quoted")
def quoted(name):
    if type(name) is not str or not name or "\x00" in name:
        raise ValueError("a database identifier is a nonempty string without null characters")
    return '"' + name.replace('"', '""') + '"'


@timed("databases.statement")
def statement(columns, parameters, qualified):
    if set(parameters) not in ({"table"}, {"query"}):
        raise ValueError("a database reader states exactly one of table and query")
    names = "*" if "*" in columns else ", ".join(quoted(name) for name in columns)
    if "table" in parameters:
        table = parameters["table"]
        if type(table) is not str or not table:
            raise ValueError("a database table is named by a nonempty string")
        table = ".".join(quoted(name) for name in table.split(".")) if qualified else quoted(table)
        return "SELECT %s FROM %s" % (names, table)
    query = parameters["query"]
    if type(query) is not str or not query.strip():
        raise ValueError("a database query is a nonempty string")
    return "SELECT %s FROM (%s) AS graphspec_source" % (names, query.strip().removesuffix(";"))


@timed("databases.bytesize")
def bytesize(value):
    size, pending = 0, [value]
    while pending:
        item = pending.pop()
        size += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return size


@timed("databases.arrow")
def arrow(rows, names, schema=None):
    checked(bytesize(rows) * 3, "database row conversion")
    arrays = [pa.array(values, type=None if schema is None else schema.field(name).type)
              for name, values in zip(names, zip(*rows))]
    result = pa.Table.from_arrays(arrays, names=names) if schema is None else pa.Table.from_arrays(arrays, schema=schema)
    checked(result.nbytes, "database batch")
    return result


@timed("databases.datatype")
def datatype(name, postgres):
    name = name.lower().strip()
    if postgres and name.endswith("[]"):
        return pa.list_(datatype(name[:-2], True))
    if postgres and name in ("bool", "boolean"):
        return pa.bool_()
    if name in ("int", "integer", "smallint", "bigint", "int2", "int4", "int8"):
        return pa.int64()
    if name in ("real", "float", "double", "double precision", "float4", "float8"):
        return pa.float64()
    if name in ("text", "string", "name") or name.startswith(("varchar", "character", "char(")):
        return pa.string()
    if name in ("blob", "bytea"):
        return pa.binary()
    if not postgres:
        return pa.null()
    if name == "date":
        return pa.date32()
    if name.startswith("timestamp"):
        return pa.timestamp("us", "UTC" if "with time zone" in name or name == "timestamptz" else None)
    if name.startswith("time") and "with time zone" not in name and name != "timetz":
        return pa.time64("us")
    if name == "interval":
        return pa.duration("us")
    match = re.fullmatch(r"(?:numeric|decimal)\((\d+),\s*(\d+)\)", name)
    if match:
        precision, scale = map(int, match.groups())
        return pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)
    return pa.null()


@timed("databases.native_schema")
def native_schema(description, metadata, postgres):
    columns = metadata.get("columns", {})
    fields = []
    for column in description:
        name = column[0]
        declared = column.type_display if postgres else columns.get(name, {}).get("type", "")
        fields.append((name, datatype(declared, postgres)))
    return pa.schema(fields)


def records(cursor, sql, columns, metadata, rewind):
    started = time.perf_counter()
    count = 0
    try:
        cursor.execute(sql)
        names = [column[0] for column in cursor.description]
        if len(set(names)) != len(names):
            raise ValueError("a database query returns duplicate column names")
        size = max(1, min(65536, capacity() // max(1024, len(names) * 1024)))
        schema = native_schema(cursor.description, metadata, rewind)
        while rows := cursor.fetchmany(size):
            batch = arrow(rows, names)
            schema = pa.unify_schemas([schema, batch.schema], promote_options="permissive")
            checked(schema.serialize().size, "database schema")
            count += batch.num_rows
        checked(bytesize(metadata), "database key metadata")
        schema = schema.with_metadata({b"graphspec.database": json.dumps(metadata, sort_keys=True).encode()})
        if count == 0:
            yield from yielded(schema.empty_table(), columns)
            return
        if rewind:
            cursor.scroll(0, mode="absolute")
        else:
            cursor.execute(sql)
        while rows := cursor.fetchmany(size):
            yield from yielded(arrow(rows, names, schema), columns)
    finally:
        LOG.debug("database records exit rows=%d seconds=%.4f", count, time.perf_counter() - started)


@contextmanager
def sqlite_connection(place, member=None):
    started = time.perf_counter()
    packaged = member is not None or Path(place).suffix.lower() in {*COMPRESSED, ".zip", ".tar"}
    connection = (sqlite3.connect(":memory:") if packaged else
                  sqlite3.connect(Path(place).resolve().as_uri() + "?mode=ro", uri=True))
    try:
        if packaged:
            with opened(place, member) as stream:
                connection.deserialize(loaded(stream, "compressed SQLite database"))
            connection.execute("PRAGMA cache_size = -%d" % max(1, capacity() // 4096))
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()
        LOG.debug("sqlite connection exit seconds=%.4f", time.perf_counter() - started)


@timed("databases.sqlite_structure")
def sqlite_structure(connection, table=None):
    entries = connection.execute(
        "SELECT name, type FROM sqlite_schema WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%'" + ("" if table is None else " AND name = ?") + " ORDER BY name",
        () if table is None else (table,),
    ).fetchall()
    result = {}
    for name, kind in entries:
        columns = connection.execute("PRAGMA table_xinfo(%s)" % quoted(name)).fetchall()
        keys = sorted((column[5], column[1]) for column in columns if column[5])
        foreign = {}
        for key, position, target, source, reference, update, delete, match in connection.execute(
            "PRAGMA foreign_key_list(%s)" % quoted(name)
        ):
            held = foreign.setdefault(key, dict(table=target, columns=[], references=[]))
            held["columns"].append((position, source))
            held["references"].append((position, reference))
        result[name] = dict(
            kind=kind,
            columns={column[1]: dict(type=column[2], nullable=not bool(column[3]), primary=bool(column[5]))
                     for column in columns if column[6] != 1},
            primary_key=[key for _, key in keys],
            foreign_keys=[dict(table=held["table"], columns=[value for _, value in sorted(held["columns"])],
                               references=[value for _, value in sorted(held["references"])])
                          for _, held in sorted(foreign.items())],
        )
    for entry in result.values():
        for foreign in entry["foreign_keys"]:
            if all(name is None for name in foreign["references"]):
                target = connection.execute("PRAGMA table_xinfo(%s)" % quoted(foreign["table"])).fetchall()
                foreign["references"] = [name for _, name in sorted((column[5], column[1]) for column in target if column[5])]
    return result


@timed("databases.sqlite_catalogue")
def sqlite_catalogue(place, member=None):
    with sqlite_connection(place, member) as connection:
        return sqlite_structure(connection)


def sqlite(place, columns, parameters):
    started = time.perf_counter()
    try:
        sql = statement(columns, {key: value for key, value in parameters.items() if key != "member"}, False)
        with sqlite_connection(place, parameters.get("member")) as connection, closing(connection.cursor()) as cursor:
            metadata = (sqlite_structure(connection, parameters["table"])[parameters["table"]]
                        if "table" in parameters else dict(kind="query"))
            yield from records(cursor, sql, columns, metadata, False)
    finally:
        LOG.debug("sqlite reader exit seconds=%.4f", time.perf_counter() - started)


@contextmanager
def postgres_connection(place):
    import psycopg

    started = time.perf_counter()
    try:
        with psycopg.connect(str(place), connect_timeout=10) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield connection
    finally:
        LOG.debug("postgres connection exit seconds=%.4f", time.perf_counter() - started)


@timed("databases.postgres_structure")
def postgres_structure(connection, table=None):
    result = {}
    rows = connection.execute(
        "SELECT n.nspname, c.relname, c.relkind, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), "
        "NOT a.attnotnull FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid "
        "WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f') AND a.attnum > 0 AND NOT a.attisdropped "
        "AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND left(n.nspname, 8) <> 'pg_toast' "
        + ("" if table is None else "AND c.oid = %s::regclass ") + "ORDER BY n.nspname, c.relname, a.attnum",
        () if table is None else (table,),
    )
    for schema, name, kind, column, datatype, nullable in rows:
        entry = result.setdefault(schema + "." + name, dict(
            kind="view" if kind in ("v", "m") else "table", columns={}, primary_key=[], foreign_keys=[]
        ))
        entry["columns"][column] = dict(type=datatype, nullable=nullable, primary=False)
    keys = connection.execute(
        "SELECT n.nspname, c.relname, k.contype, k.conname, rn.nspname, rc.relname, "
        "array_agg(a.attname ORDER BY x.position), array_agg(ra.attname ORDER BY x.position) "
        "FROM pg_catalog.pg_constraint k JOIN pg_catalog.pg_class c ON c.oid = k.conrelid "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "CROSS JOIN LATERAL unnest(k.conkey) WITH ORDINALITY AS x(number, position) "
        "JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = x.number "
        "LEFT JOIN pg_catalog.pg_class rc ON rc.oid = k.confrelid "
        "LEFT JOIN pg_catalog.pg_namespace rn ON rn.oid = rc.relnamespace "
        "LEFT JOIN pg_catalog.pg_attribute ra ON ra.attrelid = rc.oid AND ra.attnum = k.confkey[x.position] "
        "WHERE k.contype IN ('p', 'f') AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
        + ("" if table is None else "AND c.oid = %s::regclass ") +
        "GROUP BY n.nspname, c.relname, k.contype, k.conname, rn.nspname, rc.relname "
        "ORDER BY n.nspname, c.relname, k.conname",
        () if table is None else (table,),
    )
    for schema, name, kind, constraint, target_schema, target, columns, references in keys:
        entry = result[schema + "." + name]
        if kind == "p":
            entry["primary_key"] = columns
            for column in columns:
                entry["columns"][column]["primary"] = True
        else:
            entry["foreign_keys"].append(dict(table=target_schema + "." + target,
                                              columns=columns, references=references))
    return result


@timed("databases.postgres_catalogue")
def postgres_catalogue(place):
    with postgres_connection(place) as connection:
        return postgres_structure(connection)


def postgres(place, columns, parameters):
    started = time.perf_counter()
    try:
        sql = statement(columns, parameters, True)
        with postgres_connection(place) as connection:
            metadata = dict(kind="query")
            if "table" in parameters:
                qualified = ".".join(quoted(part) for part in parameters["table"].split("."))
                name = connection.execute(
                    "SELECT n.nspname || '.' || c.relname FROM pg_catalog.pg_class c "
                    "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE c.oid = %s::regclass",
                    (qualified,),
                ).fetchone()[0]
                metadata = postgres_structure(connection, qualified)[name]
            with connection.cursor(name="graphspec_source", scrollable=True) as cursor:
                yield from records(cursor, sql, columns, metadata, True)
    finally:
        LOG.debug("postgres reader exit seconds=%.4f", time.perf_counter() - started)


sqlite.catalogue = sqlite_catalogue
postgres.catalogue = postgres_catalogue
postgres.file = False
READERS.register("sqlite", sqlite)
READERS.register("postgres", postgres)
