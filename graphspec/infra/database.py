import logging
import time

from .timing import timed

LOG = logging.getLogger(__name__)
CHUNK = 1 << 20


LOAD_POOL = 16 << 30


@timed("database.literal")
def literal(value):
    return "'" + value.replace("'", "''") + "'"


@timed("database.connect")
def connect(path, read_only, buffer_pool_size=0):
    import ladybug

    database = ladybug.Database(str(path), read_only=read_only, buffer_pool_size=buffer_pool_size)
    return database, ladybug.Connection(database)


@timed("database.query")
def query(connection, statement, parameters=None):
    return connection.execute(statement, parameters).get_as_arrow(CHUNK)


@timed("database.rows")
def rows(connection, statement, parameters=None):
    return query(connection, statement, parameters).to_pylist()


@timed("database.tables")
def tables(connection):
    return sorted(rows(connection, "CALL show_tables() RETURN id, name, type"), key=lambda row: row["id"])


@timed("database.columns")
def columns(connection, table):
    return {
        row["name"]: row["type"] for row in rows(connection, "CALL table_info(%s) RETURN *" % literal(table))
    }


@timed("database.connections")
def connections(connection, table):
    return tuple(
        dict.fromkeys(
            (row["source table name"], row["destination table name"])
            for row in rows(connection, "CALL show_connection(%s) RETURN *" % literal(table))
        )
    )


@timed("database.width")
def width(kind):
    if not (kind.startswith("FLOAT[") and kind.endswith("]")):
        raise ValueError("expected a FLOAT[n] column, found " + kind)
    return int(kind[len("FLOAT[") : -1])


def copy(connection, stage):
    started = time.perf_counter()
    for (table, source, target), paths in sorted(
        stage.files.items(), key=lambda item: item[0][1] is not None
    ):
        clock = time.perf_counter()
        options = "" if source is None else " (from=%s, to=%s)" % (literal(source), literal(target))
        connection.execute(
            "COPY `%s` FROM [%s]%s"
            % (table.replace("`", "``"), ", ".join(literal(path.as_posix()) for path in paths), options)
        )
        LOG.info(
            "copy exit table=%s from=%s to=%s rows=%d seconds=%.3f",
            table,
            source,
            target,
            stage.rows[(table, source, target)],
            time.perf_counter() - clock,
        )
    LOG.info("copy all exit units=%d seconds=%.3f", len(stage.files), time.perf_counter() - started)
