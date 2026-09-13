import logging
import os
import time
import traceback

import numba
import torch
import torch.multiprocessing as multiprocessing

from graph_tuning_infra.arrays import Bundle, views
from graph_tuning_infra.log import configure

from .batching import recipe_batch
from .data import Bank, prepare, resolve


LOG = logging.getLogger(__name__)


def serve(bank_directory, cfg, threads, jobs, results):
    configure(json_lines=True)
    numba.set_num_threads(threads)
    torch.set_num_threads(1)
    bank = Bank(bank_directory)
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
        context = multiprocessing.get_context('spawn')
        self.jobs, self.results = context.Queue(), context.Queue()
        threads = max(1, os.cpu_count() // workers)
        self.processes = [context.Process(target=serve, args=(str(bank.directory), cfg, threads, self.jobs, self.results), daemon=True)
                          for _ in range(workers)]
        for process in self.processes:
            process.start()
        LOG.info('packer enter workers=%d threads=%d seconds=%.3f', workers, threads, time.perf_counter() - started)

    def stream(self, groups):
        ahead = 2 * len(self.processes)
        submitted, held = 0, {}
        while submitted < min(ahead, len(groups)):
            self.jobs.put((submitted, groups[submitted]))
            submitted += 1
        for index in range(len(groups)):
            while index not in held:
                received = self.results.get()
                held[received[0]] = received
            _, buffer, spans, tree, failure = held.pop(index)
            if failure is not None:
                raise RuntimeError('packing failed in a worker process:\n' + failure)
            if submitted < len(groups):
                self.jobs.put((submitted, groups[submitted]))
                submitted += 1
            started = time.perf_counter()
            batch = resolve(tree, self.bank, self.cfg, views(buffer.to(self.device), spans), self.device)
            LOG.debug('packer batch exit index=%d bytes=%d seconds=%.4f', index, len(buffer), time.perf_counter() - started)
            yield batch

    def close(self):
        started = time.perf_counter()
        for _ in self.processes:
            self.jobs.put(None)
        for process in self.processes:
            process.join(timeout=30)
        for process in self.processes:
            if process.is_alive():
                process.kill()
        LOG.info('packer exit workers=%d seconds=%.3f', len(self.processes), time.perf_counter() - started)
