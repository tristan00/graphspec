import bz2
import gzip
import io
import lzma
import math
import pickle
import tarfile
import zipfile

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from compression import zstd

from graphspec import Builder, BuildPlan, source
from graphspec.build.readers import batches
from graphspec.infra.timing import timed
from graphspec.sources import arrays

pytestmark = pytest.mark.builder
PLAN = BuildPlan(16 << 20, 1)


@timed("array_test.read")
def read(path, reader, columns, plan=PLAN, **parameters):
    statement = source(path.name, reader=reader, columns={name: "native" for name in columns}, **parameters)
    return pa.concat_tables(list(batches(statement, path.parent, plan, "records")))


@timed("array_test.archived")
def archived(path, content, form):
    if form in ("gz", "bz2", "xz", "zst"):
        codecs = {"gz": gzip, "bz2": bz2, "xz": lzma, "zst": zstd}
        target = path.with_name(path.name + "." + form)
        target.write_bytes(codecs[form].compress(content))
        return target, {}
    target = path.with_name(path.name + "." + form)
    member = "nested/" + path.name + ".gz"
    packed = gzip.compress(content)
    if form == "zip":
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(member, packed)
            archive.writestr("unused", b"other")
    else:
        with tarfile.open(target, "w") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(packed)
            archive.addfile(info, io.BytesIO(packed))
    return target, {"member": member}


def test_numpy_and_pickle_keep_native_nested_arrays_nulls_and_numeric_missing_values(tmp_path):
    records = np.array([(3, [1.0, 2.0]), (4, [3.0, 4.0])], dtype=[("count", "i4"), ("point", "f8", (2,))])
    content = dict(
        ids=np.array([10, 20], dtype=">i4"),
        vector=np.array([[2.0, 3.0], [4.0, 5.0]]),
        nested=records,
        ragged=np.array([[1, None], None], dtype=object),
        amount=np.array([1.0, np.nan]),
        unused=np.array([1j]),
    )
    requested = {"key": "ids", "vector": "vector", "nested": "nested", "ragged": "ragged", "amount": "amount"}
    for reader in ("numpy", "pickle"):
        path = tmp_path / ("native.npz" if reader == "numpy" else "native.pkl")
        if reader == "numpy":
            np.savez(path, **content)
        else:
            path.write_bytes(pickle.dumps(content))
        result = read(path, reader, tuple(requested), arrays={**requested, "ignored": "unused"})
        assert result.column_names == list(requested)
        assert result["key"].to_pylist() == [10, 20]
        assert result["vector"].to_pylist() == [[2.0, 3.0], [4.0, 5.0]]
        assert result["nested"].to_pylist() == [{"count": 3, "point": [1.0, 2.0]}, {"count": 4, "point": [3.0, 4.0]}]
        assert result["ragged"].to_pylist() == [[1, None], None]
        assert result["amount"].null_count == 0 and math.isnan(result["amount"][1].as_py())
    masked = tmp_path / "masked.pkl"
    masked.write_bytes(pickle.dumps({"a": np.ma.array([1, 2], mask=[False, True])}))
    assert read(masked, "pickle", ("a",), arrays={"a": "a"}).to_pylist() == [{"a": 1}, {"a": None}]


def test_pickled_native_containers_and_structured_array_select_named_columns(tmp_path):
    path = tmp_path / "frame.pkl"
    path.write_bytes(pickle.dumps({"id": (1, 2), "nested": [{"x": [1, 2]}, None]}))
    assert read(path, "pickle", ("key", "record"), arrays={"key": "id", "record": "nested"}).to_pylist() == [
        {"key": 1, "record": {"x": [1, 2]}}, {"key": 2, "record": None}
    ]
    path.write_bytes(pickle.dumps(np.array([(1, 2), (3, 4)], dtype=[("a", "i4"), ("b", "i4")])))
    assert read(path, "pickle", ("a",), arrays={"a": "a"}).to_pylist() == [{"a": 1}, {"a": 3}]


class Reconstruction:
    @timed("array_test.Reconstruction.init")
    def __init__(self, function, arguments, state=None):
        self.function, self.arguments, self.state = function, arguments, state

    @timed("array_test.Reconstruction.reduce")
    def __reduce__(self):
        return self.function, self.arguments, self.state


def test_pickle_and_object_archives_refuse_arbitrary_reconstruction_before_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = Reconstruction(eval, ("__import__('pathlib').Path(%r).write_text('called')" % str(marker),))
    path = tmp_path / "unsupported.pkl"
    path.write_bytes(pickle.dumps({"a": [payload]}))
    with pytest.raises(ValueError, match="unsupported pickled object builtins.eval"):
        read(path, "pickle", ("a",), arrays={"a": "a"})
    archive = tmp_path / "unsupported.npz"
    np.savez(archive, a=np.array([payload], dtype=object))
    with pytest.raises(ValueError, match="unsupported pickled object builtins.eval"):
        read(archive, "numpy", ("a",), arrays={"a": "a"})
    assert not marker.exists()
    path.write_bytes(pickle.dumps(pd.DataFrame({"a": [1]})))
    with pytest.raises(ValueError, match="unsupported pickled object pandas"):
        read(path, "pickle", ("a",), arrays={"a": "a"})


def test_pickle_checks_native_array_reconstruction_and_alias_expansion_before_allocation(tmp_path):
    path = tmp_path / "oversized.pkl"
    native = np.array([1], dtype=object).__reduce__()
    payload = Reconstruction(native[0], native[1], (1, (1 << 35,), np.dtype("i8"), False, b""))
    path.write_bytes(pickle.dumps({"a": payload}))
    with pytest.raises(ValueError, match="pickled array reconstruction.*worker budget"):
        read(path, "pickle", ("a",), plan=BuildPlan(8192, 1), arrays={"a": "a"})
    payload = Reconstruction(np.ndarray, ((1 << 35,), np.dtype("i8")))
    path.write_bytes(pickle.dumps({"a": payload}))
    with pytest.raises(ValueError, match="direct pickled ndarray construction"):
        read(path, "pickle", ("a",), arrays={"a": "a"})
    values = [1]
    for _ in range(20):
        values = [values, values]
    path.write_bytes(pickle.dumps({"a": values}))
    with pytest.raises(ValueError, match="expanded array values.*worker budget"):
        read(path, "pickle", ("a",), plan=BuildPlan(16384, 1), arrays={"a": "a"})
    cycle = []
    cycle.append(cycle)
    path.write_bytes(pickle.dumps({"a": cycle}))
    with pytest.raises(ValueError, match="cyclic native containers"):
        read(path, "pickle", ("a",), arrays={"a": "a"})


def test_hdf5_native_tables_labelled_matrices_and_scalar_columns(tmp_path):
    path = tmp_path / "native.h5"
    with h5py.File(path, "w") as file:
        file["table"] = np.array([(7, [1.0, 2.0]), (8, [3.0, np.nan])], dtype=[("id", "i4"), ("point", "f8", (2,))])
        matrix = file.create_dataset("matrix", data=np.array([[1.0, np.nan], [3.0, 4.0]]))
        rows = file.create_dataset("rows", data=[10, 20])
        headers = file.create_dataset("columns", data=np.array([b"left", b"right"]))
        rows.make_scale("row labels")
        headers.make_scale("column labels")
        matrix.dims[0].attach_scale(rows)
        matrix.dims[1].attach_scale(headers)
        file["scalar"] = np.array([1, 2])
        file.create_dataset("words", data=["first", "second"], dtype=h5py.string_dtype())
    result = read(path, "hdf5", ("id", "point"), dataset="table")
    assert result["id"].to_pylist() == [7, 8]
    assert result["point"][0].as_py() == [1.0, 2.0] and math.isnan(result["point"][1].as_py()[1])
    result = read(path, "hdf5", ("minute", "*"), dataset="matrix", axis="minute")
    assert result.column_names == ["minute", "left", "right"]
    assert result["minute"].to_pylist() == [10, 20] and math.isnan(result["right"][0].as_py())
    assert read(path, "hdf5", ("scalar",), dataset="scalar").to_pylist() == [{"scalar": 1}, {"scalar": 2}]
    assert read(path, "hdf5", ("words",), dataset="words").to_pylist() == [{"words": "first"}, {"words": "second"}]


def test_hdf5_native_dataframe_fixed_and_table_formats_keep_index_labels(tmp_path):
    frame = pd.DataFrame({"one": [1.0, np.nan], "two": [3.0, 4.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    for form in ("fixed", "table"):
        path = tmp_path / (form + ".h5")
        frame.to_hdf(path, key="reading", format=form)
        result = read(path, "hdf5", ("moment", "two", "one"), dataset="reading", axis="moment")
        assert result.column_names == ["moment", "two", "one"]
        assert result["moment"].to_pylist() == list(frame.index)
        assert result["two"].to_pylist() == [3.0, 4.0] and math.isnan(result["one"][1].as_py())
        assert result["one"].null_count == 0


def test_every_array_reader_handles_compression_and_nested_archive_members_without_extraction(tmp_path):
    numpy_bytes = io.BytesIO()
    np.savez(numpy_bytes, id=np.array([1, 2]))
    native_hdf = io.BytesIO()
    with h5py.File(native_hdf, "w") as file:
        file["id"] = [1, 2]
    for reader, suffix, content, settings in (
        ("numpy", "npz", numpy_bytes.getvalue(), {"arrays": {"id": "id"}}),
        ("pickle", "pkl", pickle.dumps({"id": np.array([1, 2])}), {"arrays": {"id": "id"}}),
        ("hdf5", "h5", native_hdf.getvalue(), {"dataset": "id"}),
    ):
        for form in ("gz", "bz2", "xz", "zst", "zip", "tar"):
            path, selection = archived(tmp_path / (reader + "." + suffix), content, form)
            before = set(tmp_path.iterdir())
            assert read(path, reader, ("id",), **settings, **selection).to_pylist() == [{"id": 1}, {"id": 2}]
            assert set(tmp_path.iterdir()) == before
    frame_path = tmp_path / "frame.h5"
    pd.DataFrame({"id": [1, 2]}, index=[10, 20]).to_hdf(frame_path, key="frame", format="table")
    path, selection = archived(frame_path, frame_path.read_bytes(), "zip")
    before = set(tmp_path.iterdir())
    assert read(path, "hdf5", ("row", "id"), dataset="frame", axis="row", **selection).to_pylist() == [
        {"row": 10, "id": 1}, {"row": 20, "id": 2}
    ]
    assert set(tmp_path.iterdir()) == before


def test_array_readers_report_missing_arrays_columns_lengths_labels_and_capacity(tmp_path):
    path = tmp_path / "arrays.npz"
    np.savez(path, a=np.array([1, 2]), b=np.array([1]))
    for columns, settings, error in (
        (("a", "b"), {"arrays": {"a": "a", "b": "b"}}, "equal lengths"),
        (("a",), {"arrays": {"a": "absent"}}, "does not hold arrays"),
        (("missing",), {"arrays": {"a": "a"}}, "does not hold columns"),
        (("a",), {}, "parameters missing"),
        (("a",), {"arrays": {"a": "a"}, "unknown": True}, "unknown"),
    ):
        with pytest.raises(ValueError, match=error):
            read(path, "numpy", columns, **settings)
    np.savez(path, a=np.zeros(1024))
    with pytest.raises(ValueError, match="worker budget"):
        read(path, "numpy", ("a",), plan=BuildPlan(4096, 1), arrays={"a": "a"})
    path = tmp_path / "missing.h5"
    with h5py.File(path, "w") as file:
        file["matrix"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="axis"):
        read(path, "hdf5", ("a",), dataset="matrix")
    with pytest.raises(ValueError, match="dimension scale"):
        read(path, "hdf5", ("a",), dataset="matrix", axis="row")
    with pytest.raises(ValueError, match="does not hold dataset"):
        read(path, "hdf5", ("a",), dataset="absent")


def test_empty_arrays_keep_schema_and_large_arrays_yield_batches(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, id=np.array([], dtype=np.int64), vector=np.empty((0, 2)))
    result = read(path, "numpy", ("id", "vector"), arrays={"id": "id", "vector": "vector"})
    assert result.num_rows == 0 and result.schema.field("id").type == pa.int64()
    assert result.schema.field("vector").type == pa.list_(pa.float64(), 2)
    np.savez(path, id=np.arange(65540))
    statement = source(path.name, reader="numpy", columns={"id": "number"}, arrays={"id": "id"})
    result = list(Builder({"records": statement}, tmp_path).read("records", PLAN))
    assert [table.num_rows for table in result] == [65536, 4]
    assert arrays.READERS.named("numpy") is arrays.numpy
