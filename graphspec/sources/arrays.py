import ast
import logging
import struct
import sys
import time
from collections.abc import Mapping
from math import prod

import numpy as np
import pyarrow as pa

from graphspec.build.readers import READERS
from graphspec.build.streams import capacity, checked, loaded, opened, yielded
from graphspec.infra.timing import timed

from .pickles import decoded

LOG = logging.getLogger(__name__)


@timed("arrays.parameters")
def accepted(stated, required, allowed):
    missing = set(required).difference(stated)
    unknown = set(stated).difference(allowed)
    if missing or unknown:
        raise ValueError("reader parameters missing=%s unknown=%s" % (sorted(missing), sorted(unknown)))


@timed("arrays.names")
def names(available, columns):
    absent = set(columns).difference(available).difference({"*"})
    if absent:
        raise ValueError("source does not hold columns %s" % sorted(absent))
    return tuple(available) if "*" in columns else tuple(columns)


@timed("arrays.mapping")
def mapping(stated, columns):
    if not isinstance(stated, Mapping) or not stated:
        raise ValueError("arrays must map column names to array names")
    if any(type(name) is not str or not name or name == "*" for name in stated):
        raise ValueError("arrays must state nonempty column names other than '*'")
    if any(type(name) is not str or not name for name in stated.values()):
        raise ValueError("arrays must select each array by its nonempty name")
    return {name: stated[name] for name in names(stated, columns)}


@timed("arrays.footprint")
def footprint(value, seen=None):
    seen = {} if seen is None else seen
    cached = isinstance(value, (np.ndarray, Mapping, list, tuple))
    if cached and id(value) in seen:
        if seen[id(value)] is None:
            raise ValueError("array columns cannot contain cyclic native containers")
        return seen[id(value)]
    if cached:
        seen[id(value)] = None
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        size = max(size, value.nbytes)
        if value.dtype.hasobject:
            for item in value.flat:
                size += footprint(item, seen)
                checked(4 * size, "expanded array values")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            size += footprint(key, seen) + footprint(item, seen)
            checked(4 * size, "expanded array values")
    elif isinstance(value, (list, tuple)):
        for item in value:
            size += footprint(item, seen)
            checked(4 * size, "expanded array values")
    elif isinstance(value, np.void) and value.dtype.names:
        for name in value.dtype.names:
            size += footprint(value[name], seen)
            checked(4 * size, "expanded structured array values")
    if cached:
        seen[id(value)] = size
    return size


@timed("arrays.native")
def native(values, dtype=None):
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            raise ValueError("an array column needs a row dimension")
        if np.ma.isMaskedArray(values):
            if dtype is None and not values.dtype.hasobject:
                dtype = pa.from_numpy_dtype(values.dtype)
                for width in reversed(values.shape[1:]):
                    dtype = pa.list_(dtype, int(width))
            return pa.array(values.tolist(), type=dtype, from_pandas=False)
        if values.ndim > 1:
            leaf_type = dtype
            for width in values.shape[1:]:
                if leaf_type is not None and pa.types.is_fixed_size_list(leaf_type):
                    if leaf_type.list_size != width:
                        raise ValueError("an array column shape disagrees with its native type")
                    leaf_type = leaf_type.value_type
            result = native(values.reshape(-1), leaf_type)
            if 0 in values.shape[1:]:
                dtype = result.type
                for width in reversed(values.shape[1:]):
                    dtype = pa.list_(dtype, int(width))
                return pa.array(values.tolist(), type=dtype, from_pandas=False)
            for width in reversed(values.shape[1:]):
                result = pa.FixedSizeListArray.from_arrays(result, int(width))
            return result
        if values.dtype.names:
            return pa.StructArray.from_arrays(
                [native(values[name], None if dtype is None else dtype.field(name).type) for name in values.dtype.names],
                names=list(values.dtype.names),
            )
        if not values.dtype.isnative:
            values = values.astype(values.dtype.newbyteorder("="))
    return pa.array(values, type=dtype, from_pandas=False)


@timed("arrays.table")
def table(arrays):
    sizes = {}
    for name, values in arrays.items():
        try:
            sizes[name] = len(values)
        except TypeError as error:
            raise ValueError("array %s needs a row dimension" % name) from error
    if len(set(sizes.values())) > 1:
        raise ValueError("array columns must have equal lengths, found %s" % sizes)
    checked(4 * footprint(arrays), "array columns and Arrow conversion")
    return pa.table({name: native(values) for name, values in arrays.items()})


@timed("arrays.array_file")
def array_file(stream):
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("unsupported NumPy array version %s" % (version,))
    width = 2 if version == (1, 0) else 4
    length = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
    checked(4 * length, "NumPy array header")
    header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
    if type(header) is not dict or set(header) != {"descr", "shape", "fortran_order"}:
        raise ValueError("a NumPy array header states its dtype, shape and order")
    shape = header["shape"]
    if type(shape) is not tuple or len(shape) > 64 or any(type(width) is not int or width < 0 for width in shape):
        raise ValueError("a NumPy array states nonnegative dimensions")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("a NumPy array states its order as a boolean")
    dtype = np.lib.format.descr_to_dtype(header["descr"])
    checked(4 * prod(shape) * dtype.itemsize, "NumPy array allocation")
    if dtype.hasobject:
        found = decoded(loaded(stream, "NumPy object array"))
        if not isinstance(found, np.ndarray) or found.shape != shape or found.dtype != dtype:
            raise ValueError("NumPy object array disagrees with its header")
        return found
    stream.seek(0)
    return np.lib.format.read_array(stream, allow_pickle=False)


def numpy(place, columns, parameters):
    started = time.perf_counter()
    try:
        accepted(parameters, ("arrays",), ("arrays", "member"))
        wanted = mapping(parameters["arrays"], columns)
        with opened(place, parameters.get("member")) as stream:
            with np.lib.npyio.NpzFile(stream, allow_pickle=False) as archive:
                absent = set(wanted.values()).difference(archive.files)
                if absent:
                    raise ValueError("NumPy archive does not hold arrays %s" % sorted(absent))
                size = sum(archive.zip.getinfo(name + ".npy").file_size for name in set(wanted.values()))
                checked(4 * size, "NumPy archive arrays")
                arrays = {}
                for name in set(wanted.values()):
                    with archive.zip.open(name + ".npy") as member:
                        arrays[name] = array_file(member)
                result = table({name: arrays[array] for name, array in wanted.items()})
                yield from yielded(result, columns)
    finally:
        LOG.debug("numpy exit seconds=%.4f", time.perf_counter() - started)


def pickled(place, columns, parameters):
    started = time.perf_counter()
    try:
        accepted(parameters, ("arrays",), ("arrays", "member"))
        wanted = mapping(parameters["arrays"], columns)
        with opened(place, parameters.get("member")) as stream:
            arrays = decoded(loaded(stream, "pickled arrays"))
        found = {}
        for name, array in wanted.items():
            try:
                found[name] = arrays[array]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError("pickled arrays do not hold array %r" % array) from error
        yield from yielded(table(found), columns)
    finally:
        LOG.debug("pickle exit seconds=%.4f", time.perf_counter() - started)


@timed("arrays.labels")
def labels(values):
    found = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    if any(not value or value == "*" for value in found) or len(found) != len(set(found)):
        raise ValueError("HDF5 column labels must give distinct nonempty column names other than '*'")
    return found


@timed("arrays.axis")
def axis(stated):
    if type(stated) is not str or not stated or stated == "*":
        raise ValueError("HDF5 axis must name the column holding row labels")
    return stated


@timed("arrays.hdf_type")
def hdf_type(dtype):
    import h5py

    if dtype.names:
        return pa.struct([pa.field(name, hdf_type(dtype.fields[name][0])) for name in dtype.names])
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        result = hdf_type(base)
        for width in reversed(shape):
            result = pa.list_(result, int(width))
        return result
    if h5py.check_string_dtype(dtype) is not None:
        return pa.string()
    variable = h5py.check_vlen_dtype(dtype)
    return pa.list_(hdf_type(variable)) if variable is not None else pa.from_numpy_dtype(dtype)


def dataset(found, columns, stated_axis):
    started = time.perf_counter()
    try:
        if found.ndim == 0:
            raise ValueError("HDF5 dataset %s needs a row dimension" % found.name)
        if found.dtype.names:
            wanted = names(found.dtype.names, columns)
            row_bytes = sum(found.dtype.fields[name][0].itemsize for name in wanted) * prod(found.shape[1:])
            types = {name: hdf_type(found.dtype.fields[name][0]) for name in wanted}
        elif found.ndim >= 2:
            stated_axis = axis(stated_axis)
            if len(found.dims[0]) != 1 or len(found.dims[1]) != 1:
                raise ValueError("HDF5 matrix %s needs one dimension scale for row and column labels" % found.name)
            rows, headers = found.dims[0][0], found.dims[1][0]
            if rows.shape != (len(found),) or headers.shape != (found.shape[1],):
                raise ValueError("HDF5 matrix %s dimension labels do not match its shape" % found.name)
            checked(4 * headers.size * max(64, headers.dtype.itemsize), "HDF5 column labels")
            fields = labels(headers[...])
            if stated_axis in fields:
                raise ValueError("HDF5 axis %s is also a column label" % stated_axis)
            wanted = names((stated_axis, *fields), columns)
            row_bytes = (rows.dtype.itemsize if stated_axis in wanted else 0) + (
                found.dtype.itemsize * prod(found.shape[2:]) * len(set(wanted).difference({stated_axis}))
            )
        else:
            name = found.name.rsplit("/", 1)[-1]
            wanted = names((name,), columns)
            row_bytes = found.dtype.itemsize
        checked(4 * max(1, row_bytes), "one HDF5 row")
        chunk = max(1, min(65536, capacity() // (8 * max(1, row_bytes))))
        for start in range(0, max(1, len(found)), chunk):
            stop = min(start + chunk, len(found))
            if found.dtype.names:
                values = found.fields(list(wanted))[start:stop]
                checked(4 * footprint(values), "HDF5 structured rows")
                arrays = {name: native(values[name], types[name]) for name in wanted}
            elif found.ndim >= 2:
                arrays = {}
                for name in wanted:
                    values = rows[start:stop] if name == stated_axis else found[start:stop, fields.index(name)]
                    checked(4 * footprint(values), "HDF5 array column")
                    arrays[name] = native(values, hdf_type(rows.dtype if name == stated_axis else found.dtype))
            else:
                arrays = {name: native(found[start:stop], hdf_type(found.dtype)) for name in wanted}
            yield from yielded(pa.table(arrays), columns)
    finally:
        LOG.debug("hdf5 dataset exit seconds=%.4f", time.perf_counter() - started)


def dataframe(stream, key, columns, stated_axis):
    started = time.perf_counter()
    try:
        import pandas as pd

        stated_axis = axis(stated_axis)
        stream.seek(0)
        image = loaded(stream, "HDF5 dataframe image")
        with pd.HDFStore(
            "graphspec-memory.h5", mode="r", driver="H5FD_CORE", driver_core_backing_store=0,
            driver_core_image=image,
        ) as store:
            frame = store.get(key)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("HDF5 dataset %s is not a dataframe" % key)
        checked(4 * int(frame.memory_usage(index=True, deep=True).sum()), "HDF5 dataframe")
        fields = labels(frame.columns)
        if stated_axis in fields:
            raise ValueError("HDF5 axis %s is also a dataframe column" % stated_axis)
        wanted = names((stated_axis, *fields), columns)
        arrays = {
            name: frame.index if name == stated_axis else frame.iloc[:, fields.index(name)]
            for name in wanted
        }
        yield from yielded(table(arrays), columns)
    finally:
        LOG.debug("hdf5 dataframe exit seconds=%.4f", time.perf_counter() - started)


def hdf5(place, columns, parameters):
    started = time.perf_counter()
    try:
        import h5py

        accepted(parameters, ("dataset",), ("dataset", "axis", "member"))
        key = parameters["dataset"]
        if type(key) is not str or not key:
            raise ValueError("HDF5 dataset must be a nonempty path")
        with opened(place, parameters.get("member")) as stream:
            with h5py.File(stream, "r") as file:
                if key not in file:
                    raise ValueError("HDF5 source does not hold dataset %s" % key)
                found = file[key]
                if isinstance(found, h5py.Dataset):
                    yield from dataset(found, columns, parameters.get("axis"))
                    return
                if "pandas_type" not in found.attrs:
                    raise ValueError("HDF5 path %s is not a dataset or dataframe table" % key)
                pending, size = [found], 0
                while pending:
                    group = pending.pop()
                    for entry in group.values():
                        if isinstance(entry, h5py.Group):
                            pending.append(entry)
                        else:
                            size += entry.size * entry.dtype.itemsize
                            checked(4 * size, "HDF5 dataframe arrays")
            yield from dataframe(stream, key, columns, parameters.get("axis"))
    finally:
        LOG.debug("hdf5 exit seconds=%.4f", time.perf_counter() - started)


READERS.register("numpy", numpy)
READERS.register("pickle", pickled)
READERS.register("hdf5", hdf5)
