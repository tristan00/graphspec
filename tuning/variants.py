import os
import time
from pathlib import Path

import numba
import torch

from graphspec.infra import files
from graphspec.infra.log import emit
from graphspec.infra.processes import parallel
from graphspec.infra.timing import Timeline, measure

from .data import Bank

WORKER = {}


def attach(bank_directory, threads, timing_directory=None):
    numba.set_num_threads(threads)
    torch.set_num_threads(threads)
    WORKER["bank"] = Bank(bank_directory)
    WORKER["timeline"] = Timeline(
        None if timing_directory is None else Path(timing_directory) / ("graph-%s.json" % os.getpid())
    )


def materialize(job):
    index, cfg = job
    started = time.perf_counter()
    with WORKER["timeline"], measure("graph_job"):
        WORKER["bank"].arrays(index, cfg)
    return index, time.perf_counter() - started


def prepare(bank_directory, cfg, workers, timing_directory=None):
    started = time.perf_counter()
    if workers < 1:
        raise ValueError("variant preparation requires at least one worker")
    bank = Bank(bank_directory)
    root = bank.stamp(cfg)
    missing = [index for index in bank.graphs(cfg) if not (root / ("graph_%04d" % index)).exists()]
    seconds = 0.0
    if missing and bank.joining(cfg) and len(missing) < workers:
        clock = time.perf_counter()
        with measure("graph_job"):
            bank.segmented(missing, cfg, workers)
        seconds = time.perf_counter() - clock
    elif missing:
        threads = max(1, numba.config.NUMBA_NUM_THREADS // workers)
        seconds = sum(
            elapsed
            for _, elapsed in parallel(
                materialize,
                [(index, cfg) for index in missing],
                workers,
                attach,
                (str(bank_directory), threads, timing_directory),
            )
        )
    os.utime(root / "variant.json")
    emit(
        "variant_prepare",
        digest=root.name[:10],
        graphs=len(bank.graphs(cfg)),
        built=len(missing),
        workers=workers,
        build_seconds=round(seconds, 3),
        seconds=round(time.perf_counter() - started, 3),
    )
    return root


def prune(bank_directory, limit, keep):
    started = time.perf_counter()
    total, removed = files.prune(Bank(bank_directory).cache.parent, limit, keep, "variant.json", "*/variants/*")
    emit(
        "variant_prune",
        removed=removed,
        bytes=total,
        limit=limit,
        seconds=round(time.perf_counter() - started, 3),
    )
    return total
