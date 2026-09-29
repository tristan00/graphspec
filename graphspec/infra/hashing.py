import hashlib
from functools import lru_cache

import numpy as np
from numba import njit, prange

from .timing import timed


@lru_cache(maxsize=200000)
def token(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


@njit(cache=True, inline="always")
def mixed(value):
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@njit(cache=True, parallel=True)
def units(start, columns):
    count = columns.shape[1]
    result = np.empty(count, np.float64)
    for row in prange(count):
        value = mixed(start)
        for column in range(columns.shape[0]):
            value = mixed(value ^ columns[column, row])
        result[row] = np.float64(value >> np.uint64(11)) / 9007199254740992.0
    return result


@timed("hashing.unit")
def unit(seed, stream, *columns):
    count = len(columns[0])
    stacked = np.empty((len(columns), count), np.uint64)
    for position, column in enumerate(columns):
        stacked[position] = np.asarray(column, np.int64).astype(np.uint64)
    return units(np.uint64(seed ^ stream), stacked)
