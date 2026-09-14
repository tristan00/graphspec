import os
from pathlib import Path
import time

import numba
import torch

from graph_tuning_infra import files
from graph_tuning_infra.log import emit
from graph_tuning_infra.processes import parallel

from .data import Bank


WORKER = {}


def attach(bank_directory, threads):
    numba.set_num_threads(threads)
    torch.set_num_threads(threads)
    WORKER['bank'] = Bank(bank_directory)


def materialize(job):
    index, cfg = job
    started = time.perf_counter()
    WORKER['bank'].arrays(index, cfg)
    return index, time.perf_counter() - started


def prepare(bank_directory, cfg, workers):
    started = time.perf_counter()
    bank = Bank(bank_directory)
    root = bank.stamp(cfg)
    missing = [index for index in bank.graphs(cfg) if not (root / ('graph_%04d' % index)).exists()]
    seconds = 0.0
    if missing:
        threads = max(1, os.cpu_count() // workers)
        seconds = sum(elapsed for _, elapsed in parallel(materialize, [(index, cfg) for index in missing], workers, attach, (str(bank_directory), threads)))
    os.utime(root / 'variant.json')
    emit('variant_prepare', digest=root.name[:10], graphs=len(bank.graphs(cfg)), built=len(missing), workers=workers,
         build_seconds=round(seconds, 3), seconds=round(time.perf_counter() - started, 3))
    return root


def prune(bank_directory, limit, keep):
    started = time.perf_counter()
    total, removed = files.prune(Path(bank_directory) / 'variants', limit, keep, 'variant.json')
    emit('variant_prune', removed=removed, bytes=total, limit=limit, seconds=round(time.perf_counter() - started, 3))
    return total
