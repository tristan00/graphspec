import logging
import math
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from .runtime import output_path

LOG = logging.getLogger(__name__)
ACTIVE = ContextVar("array_workspace", default=None)
BLOCK_BYTES = 16 << 20
MEMORY_BYTES = 64 << 20


class Workspace:
    def __init__(self, directory, memory_bytes=MEMORY_BYTES):
        self.directory = output_path(directory)
        self.memory_bytes = memory_bytes
        self.arrays = []

    def __enter__(self):
        self.started = time.perf_counter()
        self.temporary = tempfile.TemporaryDirectory(prefix="arrays-", dir=self.directory)
        self.token = ACTIVE.set(self)
        return self

    def __exit__(self, *exception):
        ACTIVE.reset(self.token)
        size = sum(array.nbytes for array in self.arrays)
        try:
            for array in self.arrays:
                array._mmap.close()
        finally:
            self.arrays.clear()
            self.temporary.cleanup()
        LOG.info("workspace exit bytes=%d seconds=%.3f", size, time.perf_counter() - self.started)


def allocate(shape, dtype):
    shape = tuple(shape)
    dtype = np.dtype(dtype)
    workspace = ACTIVE.get()
    if workspace is None or math.prod(shape) * dtype.itemsize <= workspace.memory_bytes:
        return np.empty(shape, dtype)
    path = Path(workspace.temporary.name) / ("array_%06d.npy" % len(workspace.arrays))
    result = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    workspace.arrays.append(result)
    return result


def block_rows(values):
    return max(1, BLOCK_BYTES // max(1, math.prod(values.shape[1:]) * values.dtype.itemsize))


def concatenate(values):
    values = tuple(values)
    if not values:
        raise ValueError("concatenation requires at least one array")
    if any(value.shape[1:] != values[0].shape[1:] for value in values):
        raise ValueError("concatenated arrays must have matching trailing dimensions")
    if ACTIVE.get() is None:
        return np.concatenate(values)
    shape = (sum(len(value) for value in values),) + values[0].shape[1:]
    result = allocate(shape, np.result_type(*(value.dtype for value in values)))
    offset = 0
    for value in values:
        for start in range(0, len(value), block_rows(value)):
            part = value[start : start + block_rows(value)]
            result[offset : offset + len(part)] = part
            offset += len(part)
    return result


def take(values, indices):
    indices = np.asarray(indices)
    if ACTIVE.get() is None or indices.ndim != 1:
        return values[indices]
    if indices.dtype == np.bool_ and len(indices) != len(values):
        raise IndexError("boolean row selector must match the array length")
    count = int(np.count_nonzero(indices)) if indices.dtype == np.bool_ else len(indices)
    result = allocate((count,) + values.shape[1:], values.dtype)
    offset = 0
    for start in range(0, len(indices), block_rows(values)):
        chosen = indices[start : start + block_rows(values)]
        part = values[start : start + len(chosen)][chosen] if indices.dtype == np.bool_ else values[chosen]
        result[offset : offset + len(part)] = part
        offset += len(part)
    return result


def finite(values):
    return all(
        np.isfinite(values[start : start + block_rows(values)]).all()
        for start in range(0, len(values), block_rows(values))
    )


def scatter(values, indices, source, order=None):
    width = block_rows(values)
    offset = 0
    for start in range(0, len(indices), width):
        chosen = indices[start : start + width]
        count = int(np.count_nonzero(chosen)) if indices.dtype == np.bool_ else len(chosen)
        part = source[offset : offset + count] if order is None else source[order[offset : offset + count]]
        if indices.dtype == np.bool_:
            values[start : start + len(chosen)][chosen] = part
        else:
            values[chosen] = part
        offset += count


def changed(values, left, right):
    result = allocate((len(left),), np.bool_)
    for start in range(0, len(left), block_rows(values)):
        stop = start + block_rows(values)
        result[start:stop] = np.any(values[left[start:stop]] != values[right[start:stop]], axis=1)
    return result
