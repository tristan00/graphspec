import logging
import os
import threading
import time
import traceback

import numba
import torch
import torch.multiprocessing as multiprocessing

from graph_tuning_infra.arrays import RESIDENT, Bundle, views
from graph_tuning_infra.log import configure

from .batching import recipe_batch
from .data import Bank, keys, prepare, resolve


LOG = logging.getLogger(__name__)


def serve(bank_directory, cfg, threads, resident, jobs, results):
    configure(json_lines=True)
    numba.set_num_threads(threads)
    torch.set_num_threads(1)
    bank = Bank(bank_directory, resident)
    while True:
        job = jobs.get()
        if job is None:
            break
        index, blocks = job
        started = time.perf_counter()
        try:
            bundle = Bundle('cpu')
            tree = prepare(recipe_batch(bank, blocks, cfg), bundle)
            buffer, spans = bundle.pack()
            results.put((index, torch.from_numpy(buffer), spans, tree, None))
        except Exception:
            results.put((index, None, None, None, traceback.format_exc()))
        LOG.debug('serve batch exit index=%d blocks=%d seconds=%.4f', index, len(blocks), time.perf_counter() - started)


class Packer:

    def __init__(self, bank, cfg, device, workers):
        started = time.perf_counter()
        self.bank, self.cfg, self.device = bank, cfg, device
        self.position, self.closed = 0, False
        self.pinned = torch.empty(0, dtype=torch.uint8, pin_memory=True)
        context = multiprocessing.get_context('spawn')
        self.jobs, self.results = context.Queue(), context.Queue()
        threads = max(1, os.cpu_count() // workers)
        resident = max(4, RESIDENT // workers)
        self.processes = [context.Process(target=serve, args=(str(bank.directory), cfg, threads, resident, self.jobs, self.results), daemon=True)
                          for _ in range(workers)]
        for process in self.processes:
            process.start()
        LOG.info('packer enter workers=%d threads=%d resident=%d seconds=%.3f', workers, threads, resident, time.perf_counter() - started)

    def prefetch(self, groups, wanted):
        started, loaded = time.perf_counter(), 0
        members = [set(block[0] for block in group) for group in groups]
        limit = self.bank.residents.limit - 32
        for index, needed in enumerate(members):
            while not self.closed and index > self.position and len(set().union(*members[self.position:index + 1])) > limit:
                time.sleep(0.005)
            if self.closed:
                break
            if index <= self.position:
                continue
            for graph in sorted(needed):
                for key in wanted:
                    self.bank.resident(graph, self.cfg, self.device).tensor(key)
                loaded += 1
        LOG.info('prefetch exit groups=%d graphs=%d keys=%s seconds=%.3f', len(groups), loaded, ','.join(sorted(wanted)), time.perf_counter() - started)

    def stream(self, groups):
        ahead = 2 * len(self.processes)
        submitted, held, loader = 0, {}, None
        while submitted < min(ahead, len(groups)):
            self.jobs.put((submitted, groups[submitted]))
            submitted += 1
        for index in range(len(groups)):
            self.position = index
            while index not in held:
                received = self.results.get()
                held[received[0]] = received
            _, buffer, spans, tree, failure = held.pop(index)
            if failure is not None:
                raise RuntimeError('packing failed in a worker process:\n' + failure)
            if submitted < len(groups):
                self.jobs.put((submitted, groups[submitted]))
                submitted += 1
            if loader is None:
                loader = threading.Thread(target=self.prefetch, args=(groups, keys(tree, set())), daemon=True)
                loader.start()
            started = time.perf_counter()
            if len(self.pinned) < len(buffer):
                self.pinned = torch.empty(max(len(buffer), 2 * len(self.pinned)), dtype=torch.uint8, pin_memory=True)
            staged = self.pinned[:len(buffer)]
            staged.copy_(buffer)
            copied = time.perf_counter()
            tensors = views(staged.to(self.device), spans)
            moved = time.perf_counter()
            batch = resolve(tree, self.bank, self.cfg, tensors, self.device)
            LOG.debug('packer batch exit index=%d bytes=%d copy=%.4f transfer=%.4f resolve=%.4f', index, len(buffer), copied - started, moved - copied, time.perf_counter() - moved)
            yield batch

    def close(self):
        started = time.perf_counter()
        self.closed = True
        for _ in self.processes:
            self.jobs.put(None)
        for process in self.processes:
            process.join(timeout=30)
        for process in self.processes:
            if process.is_alive():
                process.kill()
        LOG.info('packer exit workers=%d seconds=%.3f', len(self.processes), time.perf_counter() - started)
