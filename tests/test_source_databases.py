import gzip
import json
import os
import sqlite3
import uuid
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphspec import BuildPlan, source
from graphspec.build.readers import READERS, batches
from graphspec.sources.databases import postgres_catalogue, sqlite_catalogue
from graphspec.sources.layout import catalogue, mapping

pytestmark = pytest.mark.extended
PLAN = BuildPlan(1 << 20, 1)
EXPECTED = [dict(id=1, label="alpha", amount=1.5), dict(id=2, label=None, amount=None),
            dict(id=3, label="gamma", amount=3.5)]


def read(place, reader, columns=None, plan=PLAN, **parameters):
    declaration = source(place, reader=reader, columns=columns or {"*": "number"}, **parameters)
    return list(batches(declaration, ".", plan))


@pytest.fixture
def sqlite_file(tmp_path):
    place = tmp_path / "source.sqlite"
    with sqlite3.connect(place) as connection:
        connection.executescript(
            'CREATE TABLE parent (tenant INTEGER, id INTEGER, label TEXT, amount REAL, PRIMARY KEY(tenant, id));'
            'CREATE TABLE child (id INTEGER PRIMARY KEY, tenant INTEGER, parent INTEGER, '
            'FOREIGN KEY(tenant, parent) REFERENCES parent(tenant, id));'
            'CREATE TABLE implicit (id INTEGER, tenant INTEGER, parent INTEGER, '
            'FOREIGN KEY(tenant, parent) REFERENCES parent);'
            'CREATE VIEW selected AS SELECT id, label, amount FROM parent;'
            'CREATE TABLE empty (id INTEGER, label TEXT);'
        )
        connection.executemany("INSERT INTO parent VALUES (1, ?, ?, ?)",
                               [(row["id"], row["label"], row["amount"]) for row in EXPECTED])
        connection.execute("INSERT INTO child VALUES (1, 1, 1)")
    return place


@pytest.mark.parametrize("parameters", [dict(table="parent"), dict(table="selected"),
                                        dict(query="SELECT id, label, amount FROM parent ORDER BY id;")])
def test_sqlite_tables_views_and_queries_equal_the_same_records(sqlite_file, parameters):
    found = read(sqlite_file, "sqlite", dict(id="identifier", label="text", amount="number"), **parameters)
    assert pa.concat_tables(found).to_pylist() == EXPECTED


def test_sqlite_catalogue_keeps_ordered_composite_primary_and_foreign_keys(sqlite_file):
    found = sqlite_catalogue(sqlite_file)
    assert found["parent"]["primary_key"] == ["tenant", "id"]
    key = dict(table="parent", columns=["tenant", "parent"], references=["tenant", "id"])
    assert found["child"]["foreign_keys"] == found["implicit"]["foreign_keys"] == [key]
    assert found["selected"]["kind"] == "view"
    assert found["parent"]["columns"]["amount"]["type"] == "REAL"
    table = read(sqlite_file, "sqlite", table="child")[0]
    assert json.loads(table.schema.metadata[b"graphspec.database"]) == found["child"]
    compressed = sqlite_file.with_suffix(".sqlite.gz")
    compressed.write_bytes(gzip.compress(sqlite_file.read_bytes()))
    assert pa.concat_tables(read(compressed, "sqlite", table="selected")).to_pylist() == EXPECTED
    archive = sqlite_file.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as output:
        output.write(sqlite_file, "nested/source.sqlite")
    assert pa.concat_tables(read(archive, "sqlite", table="selected", member="nested/source.sqlite")).to_pylist() == EXPECTED
    assert not (sqlite_file.parent / "nested").exists()


def test_sqlite_schema_is_stable_when_the_first_batch_is_null(sqlite_file):
    result = read(sqlite_file, "sqlite", plan=BuildPlan(4096, 1),
                  query="SELECT NULL AS value UNION ALL SELECT NULL UNION ALL SELECT NULL "
                        "UNION ALL SELECT NULL UNION ALL SELECT 2 UNION ALL SELECT 3.5")
    assert len(result) == 2
    assert all(table.schema.equals(result[0].schema) for table in result)
    assert pa.concat_tables(result).column("value").to_pylist() == [None] * 4 + [2.0, 3.5]
    empty = read(sqlite_file, "sqlite", table="empty")
    assert len(empty) == 1 and empty[0].num_rows == 0 and empty[0].column_names == ["id", "label"]
    assert empty[0].schema.field("id").type == pa.int64() and empty[0].schema.field("label").type == pa.string()


def test_sqlite_rejects_missing_files_invalid_selection_and_over_budget_rows(sqlite_file, tmp_path):
    with pytest.raises(ValueError, match="unable to open database"):
        read(tmp_path / "absent.sqlite", "sqlite", table="parent")
    assert not (tmp_path / "absent.sqlite").exists()
    with pytest.raises(ValueError, match="exactly one"):
        read(sqlite_file, "sqlite", table="parent", query="SELECT 1")
    with pytest.raises(ValueError, match="budget"):
        read(sqlite_file, "sqlite", plan=BuildPlan(4096, 1), query="SELECT zeroblob(4096) AS bytes")
    with pytest.raises(ValueError):
        read(sqlite_file, "sqlite", query="DELETE FROM parent RETURNING id")
    assert len(read(sqlite_file, "sqlite", table="parent")[0]) == 3


def test_fixed_layout_mapping_reads_partitioned_and_compressed_tables_without_task_json(tmp_path):
    (tmp_path / "nodes" / "person").mkdir(parents=True)
    (tmp_path / "edges").mkdir()
    rows = pa.Table.from_pylist(EXPECTED)
    pq.write_table(rows.slice(0, 1), tmp_path / "nodes" / "person" / "a.parquet")
    pq.write_table(rows.slice(1), tmp_path / "nodes" / "person" / "b.parquet")
    pq.write_table(rows, tmp_path / "edges" / "follows.parquet")
    buffer = pa.BufferOutputStream()
    pq.write_table(rows, buffer)
    (tmp_path / "tasks.parquet.gz").write_bytes(gzip.compress(buffer.getvalue().to_pybytes()))
    described = catalogue(tmp_path)
    assert list(described) == ["nodes/person", "edges/follows", "tasks"]
    assert {name: entry["word"] for name, entry in mapping(tmp_path).items()} == {
        "nodes/person": "nodes", "edges/follows": "edges", "tasks": "tasks",
    }
    for table in described:
        assert pa.concat_tables(read(tmp_path, "layout", table=table)).to_pylist() == EXPECTED
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(ValueError, match="no table named"):
        read(tmp_path, "layout", table="nodes/absent")
    pq.write_table(rows, tmp_path / "nodes" / "person.parquet")
    with pytest.raises(ValueError, match="twice"):
        catalogue(tmp_path)


def test_registered_database_readers_expose_their_catalogues():
    assert READERS.named("sqlite").catalogue is sqlite_catalogue
    assert READERS.named("postgres").catalogue is postgres_catalogue
    assert READERS.named("layout").catalogue is catalogue
    assert READERS.named("postgres").file is False


def test_postgres_tables_views_queries_keys_and_native_arrays():
    import psycopg

    place = os.environ.get("GRAPHSPEC_TEST_POSTGRES")
    if place is None:
        pytest.skip("GRAPHSPEC_TEST_POSTGRES names the PostgreSQL verification database")
    name = "graphspec_source_" + uuid.uuid4().hex
    with psycopg.connect(place, autocommit=True, connect_timeout=3) as connection:
        connection.execute('CREATE SCHEMA "%s"' % name)
        try:
            connection.execute('CREATE TABLE "%s".parent (tenant integer, id integer, label text, amount double precision, '
                               'PRIMARY KEY(tenant, id))' % name)
            connection.execute('CREATE TABLE "%s".child (id integer PRIMARY KEY, tenant integer, parent integer, '
                               'FOREIGN KEY(tenant, parent) REFERENCES "%s".parent(tenant, id))' % (name, name))
            connection.execute('CREATE VIEW "%s".selected AS SELECT id, label, amount FROM "%s".parent' % (name, name))
            with connection.cursor() as cursor:
                cursor.executemany('INSERT INTO "%s".parent VALUES (1, %%s, %%s, %%s)' % name,
                                   [(row["id"], row["label"], row["amount"]) for row in EXPECTED])
            for selection in (dict(table=name + ".parent"), dict(table=name + ".selected"),
                              dict(query='SELECT id, label, amount FROM "%s".parent ORDER BY id' % name)):
                tables = read(place, "postgres", dict(id="identifier", label="text", amount="number"), **selection)
                assert pa.concat_tables(tables).to_pylist() == EXPECTED
            found = postgres_catalogue(place)
            assert found[name + ".parent"]["primary_key"] == ["tenant", "id"]
            assert found[name + ".child"]["foreign_keys"] == [dict(
                table=name + ".parent", columns=["tenant", "parent"], references=["tenant", "id"],
            )]
            assert found[name + ".selected"]["kind"] == "view"
            native = read(place, "postgres", query="SELECT ARRAY[1,2] AS values, '{\"nested\":3}'::jsonb AS record")
            assert native[0].to_pylist() == [dict(values=[1, 2], record=dict(nested=3))]
            nulls = read(place, "postgres", plan=BuildPlan(4096, 1),
                         query="SELECT value FROM (VALUES(NULL::double precision), (NULL), (NULL), (NULL), "
                               "(2.0), (3.5)) AS t(value)")
            assert len(nulls) == 2 and pa.concat_tables(nulls).column("value").to_pylist() == [None] * 4 + [2.0, 3.5]
            empty = read(place, "postgres", query="SELECT 1::integer AS id, ''::text AS label WHERE FALSE")
            assert empty[0].num_rows == 0 and empty[0].schema.field("id").type == pa.int64()
        finally:
            connection.execute('DROP SCHEMA "%s" CASCADE' % name)
