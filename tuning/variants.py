import os
from pathlib import Path
import time

from graph_tuning_infra import files
from graph_tuning_infra.log import emit
from graph_tuning_infra.processes import parallel

from .data import Bank


WORKER = {}


def attach(bank_directory):
    WORKER['bank'] = Bank(bank_directory)


def materialize(job):
    index, cfg = job
    started = time.perf_counter()
    WORKER['bank'].arrays(index, cfg)
    return index, time.perf_counter() - started


def prepare(bank_directory, cfg, workers):
    started = time.perf_counter()
    bank = Bank(bank_directory)
    root = bank.variant(cfg)[0]
    missing = [graph['index'] for graph in bank.manifest['graphs'] if not (root / ('graph_%04d' % graph['index'])).exists()]
    seconds = 0.0
    if missing:
        seconds = sum(elapsed for _, elapsed in parallel(materialize, [(index, cfg) for index in missing], workers, attach, (str(bank_directory),)))
    os.utime(root / 'variant.json')
    emit('variant_prepare', digest=root.name[:10], graphs=len(bank.manifest['graphs']), built=len(missing), workers=workers,
         build_seconds=round(seconds, 3), seconds=round(time.perf_counter() - started, 3))
    return root


def prune(bank_directory, limit, keep):
    started = time.perf_counter()
    total, removed = files.prune(Path(bank_directory) / 'variants', limit, keep, 'variant.json')
    emit('variant_prune', removed=removed, bytes=total, limit=limit, seconds=round(time.perf_counter() - started, 3))
    return total
