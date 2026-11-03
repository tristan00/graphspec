import bz2
import gzip
import io
import json
import lzma
import pickle
import tarfile
import zipfile
from dataclasses import FrozenInstanceError

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from compression import zstd

from graphspec import READERS, Builder, BuildPlan, source
from graphspec.build.command import script
from graphspec.build.readers import batches
from graphspec.build.streams import capacity

pytestmark = pytest.mark.extended
PLAN = BuildPlan(8 << 20, 1)
ROWS = [{"key": 4, "value": 1.5, "label": "first"}, {"key": 7, "value": 2.0, "label": None}]
COMPRESSORS = {"": bytes, ".gz": gzip.compress, ".zst": zstd.compress, ".bz2": bz2.compress, ".xz": lzma.compress}


def content(reader):
    if reader == "parquet":
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(ROWS), buffer)
        return buffer.getvalue(), {}
    if reader == "csv":
        return b"key,value,label\n4,1.5,first\n7,2.0,\n", {}
    if reader == "delimited":
        return b"4|1.5|first\n7|2.0|\n", dict(delimiter="|", names=["key", "value", "label"])
    return json.dumps(ROWS).encode(), {}


def read(declaration, root, plan=PLAN):
    return list(batches(declaration, root, plan, "records"))


def expected(reader):
    return [{name: str(value) if value is not None else None for name, value in row.items()} for row in ROWS] \
        if reader in ("csv", "delimited") else ROWS


@pytest.mark.parametrize("reader", ["parquet", "csv", "delimited", "json"])
@pytest.mark.parametrize("compression", COMPRESSORS)
def test_file_readers_keep_records_and_project_columns(tmp_path, reader, compression):
    data, parameters = content(reader)
    place = tmp_path / ("records." + reader + compression)
    place.write_bytes(COMPRESSORS[compression](data))
    declaration = source(place.name, reader, dict(key="identifier", label="category"), **parameters)
    batches = read(declaration, tmp_path)
    assert pa.concat_tables(batches).to_pylist() == [{"key": row["key"], "label": row["label"]} for row in expected(reader)]
    assert all(batch.schema == batches[0].schema for batch in batches)


@pytest.mark.parametrize("reader", ["parquet", "csv", "delimited", "json"])
@pytest.mark.parametrize("archive", ["zip", "tar"])
def test_archive_member_without_extraction(tmp_path, reader, archive):
    data, parameters = content(reader)
    member = "members/records." + reader + ".gz"
    compressed = gzip.compress(data)
    place = tmp_path / ("bundle." + archive)
    if archive == "zip":
        with zipfile.ZipFile(place, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr("unrelated", b"not the source")
            output.writestr(member, compressed)
    else:
        with tarfile.open(place, "w") as output:
            info = tarfile.TarInfo(member)
            info.size = len(compressed)
            output.addfile(info, io.BytesIO(compressed))
    before = set(tmp_path.rglob("*"))
    declaration = source(place.name, reader, {"*": "number"}, member=member, **parameters)
    assert pa.concat_tables(read(declaration, tmp_path)).to_pylist() == expected(reader)
    assert set(tmp_path.rglob("*")) == before


def test_json_nested_records_missing_fields_and_record_path(tmp_path):
    rows = [{"key": "a", "record": {"child": [1, 2]}}, {"key": "b", "record": {"other": True}}]
    place = tmp_path / "nested.json"
    place.write_text(json.dumps({"payload": {"rows": rows}}), encoding="utf-8")
    held = read(source(place.name, "json", dict(key="identifier", record="nested"), records="payload.rows"), tmp_path)
    assert pa.concat_tables(held).to_pylist() == [
        {"key": "a", "record": {"child": [1, 2], "other": None}},
        {"key": "b", "record": {"child": None, "other": True}},
    ]
    place.write_text('\n'.join(json.dumps(row) for row in rows), encoding="utf-8")
    assert pa.concat_tables(read(source(place.name, "json", dict(key="identifier")), tmp_path)).to_pylist() == [
        {"key": "a"}, {"key": "b"}
    ]


def test_parquet_folder_and_empty_schema(tmp_path):
    place = tmp_path / "parts"
    place.mkdir()
    pq.write_table(pa.Table.from_pylist(ROWS[:1]), place / "a.parquet")
    pq.write_table(pa.Table.from_pylist(ROWS[1:], schema=pa.Table.from_pylist(ROWS).schema), place / "b.parquet")
    assert pa.concat_tables(read(source("parts", "parquet", {"*": "number"}), tmp_path)).to_pylist() == ROWS
    empty = pa.table(dict(key=pa.array([], pa.int64())))
    pq.write_table(empty, tmp_path / "empty.parquet")
    assert read(source("empty.parquet", "parquet", dict(key="identifier")), tmp_path)[0].schema == empty.schema


def test_parquet_folder_unifies_null_only_partitions(tmp_path):
    pq.write_table(pa.table(dict(key=[1], value=[None])), tmp_path / "a.parquet")
    pq.write_table(pa.table(dict(key=[2], value=["known"])), tmp_path / "b.parquet")
    assert pa.concat_tables(read(source(tmp_path, "parquet", {"*": "text"}), tmp_path)).to_pylist() == [
        dict(key=1, value=None), dict(key=2, value="known")
    ]


def test_parquet_dictionary_expansion_is_refused_before_decoding(tmp_path, monkeypatch):
    pq.write_table(pa.table(dict(value=["x" * 4096] * 256)), tmp_path / "repeated.parquet")
    decoded = []

    def decode(*args, **kwargs):
        decoded.append(True)
        raise AssertionError("the decoder must not run")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", decode)
    with pytest.raises(ValueError, match="Parquet row group.*budget"):
        read(source("repeated.parquet", "parquet", dict(value="text")), tmp_path, BuildPlan(65536, 1))
    assert not decoded


def test_csv_schema_is_stable_across_empty_and_later_populated_batches(tmp_path):
    place = tmp_path / "late.csv"
    place.write_text("key,value\n" + "1,\n" * 800 + "2,known\n", encoding="utf-8")
    held = read(source(place.name, "csv", {"*": "text"}), tmp_path, BuildPlan(8192, 1))
    assert len(held) > 1
    assert all(batch.schema == pa.schema([("key", pa.string()), ("value", pa.string())]) for batch in held)
    assert pa.concat_tables(held).to_pylist()[-1] == dict(key="2", value="known")
    place.write_text("key,value\n", encoding="utf-8")
    held = read(source(place.name, "csv", {"*": "text"}), tmp_path)
    assert len(held) == 1 and held[0].num_rows == 0
    assert held[0].schema == pa.schema([("key", pa.string()), ("value", pa.string())])


def test_source_declarations_are_immutable_and_pickleable():
    columns = dict(key=dict(type="identifier", nulls=["unknown"]))
    arrays = dict(key="ids")
    declaration = source("input", "numpy", columns, arrays=arrays)
    columns["key"]["nulls"].append("changed")
    arrays["key"] = "changed"
    declaration.columns["key"]["nulls"].append("changed")
    declaration.parameters["arrays"]["key"] = "changed"
    assert declaration.columns == dict(key=dict(type="identifier", nulls=["unknown"]))
    assert declaration.parameters == dict(arrays=dict(key="ids"))
    with pytest.raises(FrozenInstanceError):
        declaration.reader = "changed"
    restored = pickle.loads(pickle.dumps(declaration))
    assert restored.place == declaration.place and restored.columns == declaration.columns
    assert restored.parameters == declaration.parameters


def test_interleaved_sources_keep_their_own_budget_and_close_context(tmp_path, monkeypatch):
    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    observed = []

    def custom(place, columns, parameters):
        try:
            for _ in range(2):
                yield pa.table(dict(budget=[capacity()]))
        finally:
            observed.append(capacity())

    READERS.register("budget_reader", custom)
    builder = Builder(dict(records=source("input", "budget_reader", dict(budget="number"))), tmp_path)
    first = builder.read("records", BuildPlan(4096, 1))
    second = builder.read("records", BuildPlan(8192, 1))
    assert next(first).to_pylist() == [dict(budget=4096)]
    assert next(second).to_pylist() == [dict(budget=8192)]
    assert next(first).to_pylist() == [dict(budget=4096)]
    first.close()
    with pytest.raises(ValueError, match="explicit BuildPlan"):
        capacity()
    assert next(second).to_pylist() == [dict(budget=8192)]
    second.close()
    assert observed == [4096, 8192]
    with pytest.raises(ValueError, match="explicit BuildPlan"):
        capacity()


def test_registered_reader_used_by_script(tmp_path, monkeypatch):
    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    called = []

    def custom(place, columns, parameters):
        called.append((place, columns, parameters))
        yield pa.table(dict(key=[parameters["start"]], ignored=[9]))

    READERS.register("test_reader", custom)
    path = tmp_path / "declarations.py"
    path.write_text(
        "from graphspec import source\n"
        "records = source('input.test', reader='test_reader', columns=dict(key='identifier'), start=4)\n",
        encoding="utf-8",
    )
    builder = script(path, tmp_path)
    assert pa.concat_tables(builder.read("records", PLAN)).to_pylist() == [dict(key="4")]
    assert called == [(tmp_path / "input.test", ("key",), dict(start=4))]
    direct = Builder(path, tmp_path)
    assert not direct.graphs
    assert pa.concat_tables(direct.read("records", PLAN)).to_pylist() == [dict(key="4")]


@pytest.mark.parametrize("kind", ["schema", "missing", "empty", "not_table"])
def test_source_contract_refuses_bad_reader_batches(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})

    def custom(place, columns, parameters):
        if kind == "empty":
            return
        yield pa.table(dict(key=[1])) if kind != "not_table" else {"key": [1]}
        if kind == "schema":
            yield pa.table(dict(key=["changed"]))
        elif kind == "missing":
            yield pa.table(dict(other=[2]))

    READERS.register("bad_reader", custom)
    message = {"schema": "one schema", "missing": "key", "empty": "schema", "not_table": "pyarrow.Table"}[kind]
    with pytest.raises(ValueError, match="source records:.*" + message):
        read(source("input", "bad_reader", dict(key="identifier")), tmp_path)


def test_budget_and_reader_parameters_are_explicit(tmp_path):
    place = tmp_path / "large.json"
    place.write_text(json.dumps([dict(value="x" * 2048)]), encoding="utf-8")
    with pytest.raises(ValueError, match="source records:.*budget"):
        read(source(place.name, "json", dict(value="text")), tmp_path, BuildPlan(4096, 1))
    with pytest.raises(ValueError, match="delimiter and names"):
        read(source("missing.tsv", "delimited", dict(key="identifier"), delimiter="\t"), tmp_path)
    with pytest.raises(ValueError, match="does not take"):
        read(source("missing.csv", "csv", dict(key="identifier"), guessed=True), tmp_path)
    with pytest.raises(ValueError, match="lacks.*absent"):
        read(source(place.name, "json", dict(absent="text")), tmp_path)
    pq.write_table(pa.table(dict(value=["x" * 8192])), tmp_path / "large.parquet")
    with pytest.raises(ValueError, match="Parquet row group.*budget"):
        read(source("large.parquet", "parquet", dict(value="text")), tmp_path, BuildPlan(4096, 1))
    (tmp_path / "wide.csv").write_text("x" * 8192 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV column names.*budget"):
        read(source("wide.csv", "csv", {"*": "text"}), tmp_path, BuildPlan(4096, 1))
