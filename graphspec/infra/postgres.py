import logging
import time

import psycopg

LOG = logging.getLogger(__name__)


def connect(host, port, database, user, timeout_ms=600000):
    started = time.perf_counter()
    connection = psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        autocommit=True,
        options="-c statement_timeout=%d" % timeout_ms,
    )
    LOG.info(
        "postgres connect exit seconds=%.3f host=%s port=%d database=%s",
        time.perf_counter() - started,
        host,
        port,
        database,
    )
    return connection


def rows(connection, statement, parameters=None):
    started = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(statement, parameters)
        result = cursor.fetchall()
    LOG.debug("postgres rows exit seconds=%.4f rows=%d", time.perf_counter() - started, len(result))
    return result


def records(connection, statement, parameters=None):
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(statement, parameters)
        return cursor.fetchall()
