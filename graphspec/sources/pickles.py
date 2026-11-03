import io
import pickle
import pickletools
from contextvars import ContextVar
from math import prod

import numpy as np

from graphspec.build.streams import checked
from graphspec.infra.timing import timed

CURRENT = ContextVar("array_pickle_allocation", default=None)


@timed("pickles.reserved")
def reserved(size):
    budget = CURRENT.get()
    if budget is None:
        raise ValueError("array reconstruction needs a source memory budget")
    budget[0] += size
    checked(budget[0], "pickled array reconstruction")


@timed("pickles.shape")
def shaped(shape, dtype):
    if type(shape) is not tuple or len(shape) > 64 or any(type(width) is not int or width < 0 for width in shape):
        raise ValueError("a pickled array states a tuple of nonnegative dimensions")
    if not isinstance(dtype, np.dtype):
        raise ValueError("a pickled array states a NumPy dtype")
    size = prod(shape) * dtype.itemsize
    reserved(4 * size)
    return size


@timed("pickles.state")
def state(stated):
    if type(stated) is not tuple or len(stated) != 5 or stated[0] != 1 or type(stated[3]) is not bool:
        raise ValueError("unsupported pickled array state")
    _, shape, dtype, _, data = stated
    size = shaped(shape, dtype)
    if dtype.hasobject:
        if type(data) is not list or len(data) != prod(shape):
            raise ValueError("pickled object array values do not match its shape")
    elif type(data) not in (bytes, bytearray) or len(data) != size:
        raise ValueError("pickled array bytes do not match its shape and dtype")


class Array(np.ndarray):
    @timed("pickles.Array.new")
    def __new__(cls, *args, **kwargs):
        raise ValueError("direct pickled ndarray construction is unsupported")

    @timed("pickles.Array.state")
    def __setstate__(self, stated):
        state(stated)
        np.ndarray.__setstate__(self, stated)


class Masked(np.ma.MaskedArray):
    @timed("pickles.Masked.new")
    def __new__(cls, *args, **kwargs):
        raise ValueError("direct pickled masked-array construction is unsupported")

    @timed("pickles.Masked.state")
    def __setstate__(self, stated):
        if type(stated) is not tuple or len(stated) != 7:
            raise ValueError("unsupported pickled masked-array state")
        state(stated[:5])
        state((1, stated[1], np.ma.make_mask_descr(stated[2]), stated[3], stated[5]))
        np.ma.MaskedArray.__setstate__(self, stated)


@timed("pickles.reconstruct")
def reconstruct(subtype, shape, dtype):
    if subtype is not Array or shape != (0,) or dtype not in ("b", b"b"):
        raise ValueError("unsupported pickled ndarray reconstruction")
    return np.ndarray.__new__(Array, (0,), np.int8)


@timed("pickles.masked")
def masked(subtype, baseclass, shape, dtype):
    if subtype is not Masked or baseclass is not Array or shape != (0,) or dtype not in ("b", b"b"):
        raise ValueError("unsupported pickled masked-array reconstruction")
    return np.ma.MaskedArray.__new__(Masked, np.empty(0, dtype=np.int8), mask=np.empty(0, dtype=bool))


@timed("pickles.buffered")
def buffered(data, dtype, shape, order, axis_order=None):
    size = shaped(shape, dtype)
    if dtype.hasobject or type(data) not in (bytes, bytearray) or len(data) != size:
        raise ValueError("pickled array buffer does not match its shape and dtype")
    if order not in ("C", "F", "K"):
        raise ValueError("pickled array order must be C, F or K")
    found = np.frombuffer(data, dtype=dtype)
    if order == "K":
        if type(axis_order) is not tuple or set(axis_order) != set(range(len(shape))):
            raise ValueError("pickled array axis order must be a permutation")
        return found.reshape(shape, order="C").transpose(axis_order)
    return found.reshape(shape, order=order)


@timed("pickles.scalar")
def scalar(dtype, data):
    return buffered(data, dtype, (), "C")[()]


class Unpickler(pickle.Unpickler):
    @timed("pickles.Unpickler.class")
    def find_class(self, module, name):
        if module == "numpy" and name in ("dtype", "ndarray"):
            return np.dtype if name == "dtype" else Array
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "_reconstruct":
                return reconstruct
            if name == "scalar":
                return scalar
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return buffered
        if module in ("numpy.ma", "numpy.ma.core"):
            if name == "MaskedArray":
                return Masked
            if name == "_mareconstruct":
                return masked
        raise ValueError("unsupported pickled object %s.%s; store native containers and NumPy arrays" % (module, name))


@timed("pickles.decoded")
def decoded(data):
    estimated = 4 * len(data)
    for opcode, _, _ in pickletools.genops(data):
        if opcode.name in ("EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "NEXT_BUFFER", "READONLY_BUFFER"):
            raise ValueError("unsupported pickled array opcode %s" % opcode.name)
        estimated += 64
        checked(estimated, "pickled container reconstruction")
    token = CURRENT.set([estimated])
    try:
        return Unpickler(io.BytesIO(data)).load()
    finally:
        CURRENT.reset(token)
