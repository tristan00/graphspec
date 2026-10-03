import hashlib
from functools import lru_cache

import numpy as np
from numba import njit, prange

from .timing import timed

SCALE = 9007199254740992.0


@lru_cache(maxsize=200000)
@timed("hashing.token")
def token(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


@njit(cache=True, inline="always")
def mixed(value):
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@njit(cache=True, inline="always")
def uniform(value):
    return np.float64(value >> np.uint64(11)) / SCALE


@njit(cache=True, parallel=True)
def folded(start, columns):
    count = columns.shape[1]
    result = np.empty(count, np.uint64)
    for row in prange(count):
        value = mixed(start)
        for column in range(columns.shape[0]):
            value = mixed(value ^ columns[column, row])
        result[row] = value
    return result


def stacked(columns):
    count = len(columns[0])
    result = np.empty((len(columns), count), np.uint64)
    for position, column in enumerate(columns):
        column = np.asarray(column)
        result[position] = column if column.dtype == np.uint64 else column.astype(np.int64).astype(np.uint64)
    return result


@timed("hashing.salted")
def salted(seed, salt):
    if type(seed) is not int or type(salt) is not str:
        raise TypeError(
            "a draw is keyed by a stated seed, a whole number, and a salt naming it, found %r and %r" % (seed, salt)
        )
    return np.uint64((seed ^ token(salt)) & 0xFFFFFFFFFFFFFFFF)


@timed("hashing.hashed")
def hashed(*columns):
    return folded(np.uint64(0), stacked(columns))


@timed("hashing.unit")
def unit(seed, salt, *columns):
    return (folded(salted(seed, salt), stacked(columns)) >> np.uint64(11)).astype(np.float64) / SCALE
