import logging
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

from .workspace import MEMORY_BYTES

LOG = logging.getLogger(__name__)
RESIDENT = 192
STAGED = 256 << 20
STREAMS, STAGING = {}, {}
TRANSFER = threading.Lock()


def transfer(host, device):
    nbytes = host.numel() * host.element_size()
    width = max(1, host[:1].numel() * host.element_size())
    rows = max(1, STAGED // width) if nbytes > STAGED else max(len(host), 1)
    with TRANSFER:
        if device not in STREAMS:
            STREAMS[device] = torch.cuda.Stream(device)
        needed = min(nbytes, rows * width)
        if device not in STAGING or STAGING[device].numel() < needed:
            STAGING[device] = torch.empty(needed + needed // 4, dtype=torch.uint8).pin_memory()
        with torch.cuda.stream(STREAMS[device]):
            tensor = torch.empty_like(host, device=device)
        for start in range(0, len(host), rows):
            part = host[start : start + rows]
            count = part.numel() * part.element_size()
            staged = STAGING[device][:count].view(host.dtype).reshape(part.shape)
            staged.copy_(part)
            with torch.cuda.stream(STREAMS[device]):
                tensor[start : start + rows].copy_(staged, non_blocking=True)
            STREAMS[device].synchronize()
        tensor.record_stream(torch.cuda.current_stream(device))
    return tensor


def between(values, low, high):
    return tuple(int(index) for index in np.searchsorted(values, np.asarray((low, high), values.dtype)))


class MappedRows:
    is_cuda = False

    def __init__(self, values, device, dtype=None):
        self.values, self.device, self.dtype = values, device, dtype
        self.shape = values.shape

    def __len__(self):
        return len(self.values)

    def index_select(self, dimension, indices):
        if dimension != 0:
            raise ValueError("mapped arrays select rows along dimension zero")
        started = time.perf_counter()
        rows = indices.detach().cpu().numpy()
        if rows.size and (rows.min() < 0 or rows.max() >= len(self)):
            raise IndexError("mapped array row outside its bounds")
        selected = self.values[rows]
        host = torch.as_tensor(selected, dtype=self.dtype)
        result = host if torch.device(self.device).type == "cpu" else transfer(host, self.device)
        LOG.debug(
            "mapped rows exit rows=%d bytes=%d seconds=%.4f",
            len(indices),
            selected.nbytes,
            time.perf_counter() - started,
        )
        return result


class JoinedRows:
    def __init__(self, parts, device):
        self.parts, self.device = parts, device
        self.shape = (sum(len(part) for part in parts),) + tuple(parts[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def index_select(self, dimension, indices):
        if dimension != 0:
            raise ValueError("joined arrays select rows along dimension zero")
        host = indices.detach().cpu().numpy()
        if host.size and (host.min() < 0 or host.max() >= len(self)):
            raise IndexError("joined array row outside its bounds")
        offset, result = 0, None
        for part in self.parts:
            positions = np.flatnonzero((host >= offset) & (host < offset + len(part)))
            selected = part.index_select(0, torch.as_tensor(host[positions] - offset, device=self.device))
            if result is None:
                result = selected.new_empty((len(host),) + self.shape[1:])
            result.index_copy_(0, torch.as_tensor(positions, device=self.device), selected)
            offset += len(part)
        return result


def join_tensors(parts, device):
    if len(parts) == 1:
        return parts[0]
    return (
        torch.cat(parts)
        if all(isinstance(part, torch.Tensor) for part in parts)
        else JoinedRows(parts, device)
    )


class Resident:
    def __init__(self, arrays, device, dtypes, limit=MEMORY_BYTES):
        self.arrays, self.device, self.dtypes = arrays, device, dtypes
        self.limit = limit if torch.device(device).type == "cuda" else MEMORY_BYTES
        self.tensors, self.loaded, self.columns, self.orders, self.lock = {}, {}, {}, {}, threading.Lock()

    def tensor(self, key):
        with self.lock:
            if key not in self.tensors:
                started = time.perf_counter()
                values = self.arrays[key]
                if values.nbytes > self.limit:
                    self.tensors[key] = MappedRows(values, self.device, self.dtypes.get(key))
                else:
                    host = torch.as_tensor(self.arrays.read(key), dtype=self.dtypes.get(key))
                    self.tensors[key] = (
                        host if torch.device(self.device).type == "cpu" else transfer(host, self.device)
                    )
                LOG.debug(
                    "resident tensor exit key=%s device=%s seconds=%.4f",
                    key,
                    self.device,
                    time.perf_counter() - started,
                )
            return self.tensors[key]

    def numpy(self, key):
        if key not in self.loaded:
            started = time.perf_counter()
            values = self.arrays[key]
            self.loaded[key] = values if values.nbytes > MEMORY_BYTES else self.arrays.read(key)
            LOG.debug(
                "resident numpy exit key=%s bytes=%d seconds=%.4f",
                key,
                self.loaded[key].nbytes,
                time.perf_counter() - started,
            )
        return self.loaded[key]

    def column(self, key, index):
        if (key, index) not in self.columns:
            started = time.perf_counter()
            values = self.numpy(key)[:, index]
            self.columns[(key, index)] = (
                values if values.nbytes > MEMORY_BYTES else np.ascontiguousarray(values)
            )
            LOG.debug(
                "resident column exit key=%s index=%d rows=%d seconds=%.4f",
                key,
                index,
                len(self.columns[(key, index)]),
                time.perf_counter() - started,
            )
        return self.columns[(key, index)]

    def order(self, key, index):
        if (key, index) not in self.orders:
            started = time.perf_counter()
            values = self.column(key, index)
            order = np.argsort(values, kind="stable")
            self.orders[(key, index)] = (order, values[order])
            LOG.debug(
                "resident order exit key=%s index=%d rows=%d seconds=%.4f",
                key,
                index,
                len(order),
                time.perf_counter() - started,
            )
        return self.orders[(key, index)]

    def nbytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensors.values()
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda
        )

    def rows_by(self, key, index, low, high):
        order, sorted_values = self.order(key, index)
        lo, hi = between(sorted_values, low, high)
        return np.sort(order[lo:hi])


class Residents:
    def __init__(self, dtypes, limit=RESIDENT):
        self.dtypes, self.limit, self.held, self.lock = dtypes, limit, OrderedDict(), threading.Lock()

    def get(self, key, device, open_arrays, limit=MEMORY_BYTES, budget=None):
        slot = (key, str(device))
        with self.lock:
            if budget is not None:
                held = [name for name in self.held if name[1] == slot[1] and name != slot]
                total = sum(self.held[name].nbytes() for name in held) + (
                    self.held[slot].nbytes() if slot in self.held else 0
                )
                for name in held:
                    if total <= budget:
                        break
                    total -= self.held.pop(name).nbytes()
            if slot not in self.held:
                started = time.perf_counter()
                self.held[slot] = Resident(open_arrays(), device, self.dtypes, limit)
                LOG.debug(
                    "residents open exit key=%s device=%s held=%d seconds=%.4f",
                    key,
                    device,
                    len(self.held),
                    time.perf_counter() - started,
                )
                while len(self.held) > self.limit:
                    self.held.popitem(last=False)
            return self.held[slot]


NUMPY = {
    torch.long: np.int64,
    torch.int32: np.int32,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.bool: np.bool_,
}
TORCH = {np.dtype(value): key for key, value in NUMPY.items()}


class Pending:
    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


def views(buffer, spans):
    result = []
    for start, nbytes, dtype, shape, target in spans:
        tensor = buffer[start : start + nbytes].view(dtype).reshape(shape)
        result.append(tensor if target == dtype else tensor.to(target))
    return result


def compact(array, dtype):
    if dtype == torch.long and array.dtype == np.int32:
        return array, torch.int32
    if dtype == torch.long and array.size and array.min() >= -(2**31) and array.max() < 2**31:
        return array.astype(np.int32), torch.int32
    return array, dtype


def shared(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8).share_memory_()


def slot_bytes(nbytes):
    return -(-(nbytes + nbytes // 4) // (1 << 20)) * (1 << 20)


def cuda_check(error):
    if error != torch.cuda.cudart().cudaError.success:
        raise RuntimeError("CUDA host registration failed: %s" % error)


class Slot:
    def __init__(self, tensor, device):
        started = time.perf_counter()
        self.tensor, self.cuda = tensor, torch.device(device).type == "cuda"
        if self.cuda:
            cuda_check(torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel(), 0))
        LOG.debug(
            "slot enter bytes=%d registered=%s seconds=%.4f",
            tensor.numel(),
            self.cuda,
            time.perf_counter() - started,
        )

    def transfer(self, nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device).copy_(self.tensor[:nbytes])

    def release(self):
        if self.cuda:
            cuda_check(torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr()))


class Bundle:
    def __init__(self, device):
        self.device, self.items = device, []

    def hold(self, value, dtype=None):
        array = np.asarray(value, order="C")
        if dtype is not None and array.dtype != NUMPY[dtype]:
            array = array.astype(NUMPY[dtype])
        self.items.append(
            (array, torch.long if dtype is None and array.dtype == np.int32 else TORCH[array.dtype])
        )
        return Pending(len(self.items) - 1)

    def layout(self):
        arrays, spans, offset = [], [], 0
        for array, target in self.items:
            array, dtype = compact(array, target)
            offset += -offset % 8
            arrays.append((offset, array))
            spans.append((offset, array.nbytes, dtype, array.shape, target))
            offset += array.nbytes
        self.items = []
        return arrays, spans, offset + 8

    def write(self, buffer, arrays):
        for offset, array in arrays:
            buffer[offset : offset + array.nbytes] = array.reshape(-1).view(np.uint8)

    def pack(self):
        started = time.perf_counter()
        arrays, spans, total = self.layout()
        buffer = np.zeros(total, np.uint8)
        self.write(buffer, arrays)
        LOG.debug(
            "bundle pack exit arrays=%d bytes=%d seconds=%.4f",
            len(spans),
            total,
            time.perf_counter() - started,
        )
        return buffer, spans

    def flush(self, tree):
        buffer, spans = self.pack()
        return self.replace(tree, views(torch.as_tensor(buffer, device=self.device), spans))

    def replace(self, value, tensors):
        if isinstance(value, Pending):
            return tensors[value.index]
        if isinstance(value, dict):
            return {key: self.replace(item, tensors) for key, item in value.items()}
        if isinstance(value, list):
            return [self.replace(item, tensors) for item in value]
        if isinstance(value, tuple):
            return tuple(self.replace(item, tensors) for item in value)
        return value
