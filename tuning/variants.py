from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import shutil
import time

from graph_tuning_infra.log import emit

from .data import Bank


LOG = logging.getLogger(__name__)
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
        with ProcessPoolExecutor(max_workers=min(workers, len(missing)), initializer=attach, initargs=(str(bank_directory),)) as pool:
            for _, elapsed in pool.map(materialize, [(index, cfg) for index in missing], chunksize=4):
                seconds += elapsed
    os.utime(root / 'variant.json')
    emit('variant_prepare', digest=root.name[:10], graphs=len(bank.manifest['graphs']), built=len(missing), workers=workers,
         build_seconds=round(seconds, 3), seconds=round(time.perf_counter() - started, 3))
    return root


def size(folder):
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


def prune(bank_directory, limit, keep):
    started = time.perf_counter()
    root = Path(bank_directory) / 'variants'
    folders = [folder for folder in root.iterdir() if folder.is_dir() and folder != keep] if root.exists() else []
    sizes = {folder: size(folder) for folder in folders}
    total = sum(sizes.values()) + (size(keep) if keep.exists() else 0)
    removed = 0
    for folder in sorted(folders, key=lambda folder: (folder / 'variant.json').stat().st_mtime if (folder / 'variant.json').exists() else 0):
        if total <= limit:
            break
        shutil.rmtree(folder)
        total -= sizes[folder]
        removed += 1
    emit('variant_prune', variants=len(folders) + 1, removed=removed, bytes=total, limit=limit, seconds=round(time.perf_counter() - started, 3))
    return total
