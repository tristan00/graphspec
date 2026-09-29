import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
from numba import njit, prange

LOG = logging.getLogger(__name__)
GPU_FROM = 1024
FRACTION = 0.9
THREADS = 2


def require(device):
    started = time.perf_counter()
    kind = torch.device(device).type
    if kind not in ("cpu", "cuda"):
        raise ValueError("device %s is neither cpu nor cuda" % device)
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device %s was requested and no GPU is present" % device)
    LOG.debug("require exit device=%s seconds=%.4f", device, time.perf_counter() - started)
    return kind


def device_for(size, device):
    if size < GPU_FROM or torch.device(device).type == "cpu":
        return "cpu"
    require(device)
    return device


def matrix(values, device):
    return torch.as_tensor(np.asarray(values), dtype=torch.float64, device=device_for(len(values), device))


@contextmanager
def single():
    started = time.perf_counter()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        LOG.debug("single exit threads=%d seconds=%.4f", threads, time.perf_counter() - started)


def inverse(values, device):
    started = time.perf_counter()
    with single():
        result = torch.linalg.inv(matrix(values, device)).cpu().numpy()
    LOG.debug("inverse exit size=%d seconds=%.4f", len(values), time.perf_counter() - started)
    return result


def eigh(values, device):
    started = time.perf_counter()
    with single():
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix(values, device))
    LOG.debug("eigh exit size=%d seconds=%.4f", len(values), time.perf_counter() - started)
    return eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()


def compose(vectors, weights, device):
    started = time.perf_counter()
    with single():
        basis = matrix(vectors, device)
        scaled = basis * torch.as_tensor(weights, dtype=torch.float64, device=basis.device)
        result = (scaled @ basis.T).cpu().numpy()
    LOG.debug("compose exit size=%d seconds=%.4f", len(vectors), time.perf_counter() - started)
    return result


@njit(cache=True)
def ranked_before(value, index, other, other_index):
    if np.isnan(value):
        return not np.isnan(other) or index < other_index
    return not np.isnan(other) and (value > other or (value == other and index < other_index))


@njit(cache=True, parallel=True)
def top_rows(scores, width):
    indices = np.empty((len(scores), width), np.int64)
    values = np.empty((len(scores), width), np.float64)
    if width == 0:
        return indices, values
    for row in prange(len(scores)):
        heap_indices = np.empty(width, np.int64)
        heap_values = np.empty(width, np.float64)
        used = 0
        for column in range(scores.shape[1]):
            value = scores[row, column]
            if used < width:
                position = used
                used += 1
                while position:
                    parent = (position - 1) // 2
                    if not ranked_before(heap_values[parent], heap_indices[parent], value, column):
                        break
                    heap_values[position], heap_indices[position] = heap_values[parent], heap_indices[parent]
                    position = parent
                heap_values[position], heap_indices[position] = value, column
            elif ranked_before(value, column, heap_values[0], heap_indices[0]):
                position = 0
                while position * 2 + 1 < used:
                    child = position * 2 + 1
                    if child + 1 < used and ranked_before(
                        heap_values[child],
                        heap_indices[child],
                        heap_values[child + 1],
                        heap_indices[child + 1],
                    ):
                        child += 1
                    if not ranked_before(value, column, heap_values[child], heap_indices[child]):
                        break
                    heap_values[position], heap_indices[position] = heap_values[child], heap_indices[child]
                    position = child
                heap_values[position], heap_indices[position] = value, column
        while used:
            used -= 1
            values[row, used], indices[row, used] = heap_values[0], heap_indices[0]
            value, column = heap_values[used], heap_indices[used]
            position = 0
            while position * 2 + 1 < used:
                child = position * 2 + 1
                if child + 1 < used and ranked_before(
                    heap_values[child], heap_indices[child], heap_values[child + 1], heap_indices[child + 1]
                ):
                    child += 1
                if not ranked_before(value, column, heap_values[child], heap_indices[child]):
                    break
                heap_values[position], heap_indices[position] = heap_values[child], heap_indices[child]
                position = child
            heap_values[position], heap_indices[position] = value, column
    return indices, values


def top(scores, width, device):
    started = time.perf_counter()
    scores = np.asarray(scores, dtype=np.float64)
    if device_for(len(scores), device) != "cpu":
        values, indices = torch.sort(matrix(scores, device), dim=1, descending=True, stable=True)
        result = indices[:, :width].cpu().numpy(), values[:, :width].cpu().numpy()
    else:
        result = top_rows(scores, min(width, scores.shape[1]))
    LOG.debug("top exit rows=%d width=%d seconds=%.4f", len(scores), width, time.perf_counter() - started)
    return result


def reserve(device):
    started = time.perf_counter()
    torch.set_num_threads(THREADS)
    if require(device) == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.set_per_process_memory_fraction(FRACTION, device)
    LOG.info("reserve exit device=%s seconds=%.3f", device, time.perf_counter() - started)
