from collections import deque
import logging
import os
import queue
import threading
import time
import traceback

import numba
import torch
import torch.multiprocessing as multiprocessing

from graph_tuning_infra.arrays import RESIDENT, Bundle, Slot, shared, slot_bytes, views
from graph_tuning_infra.log import configure

from .batching import recipe_batch
from .data import Bank, keys, prepare, resolve


LOG = logging.getLogger(__name__)


def serve(bank_directory, cfg, threads, resident, jobs, results):
    configure(json_lines=True)
    numba.set_num_threads(threads)
    torch.set_num_threads(1)
    bank = Bank(bank_directory, resident)
    slots = {}
    while True:
        job = jobs.get()
        if job is None:
            break
        index, blocks, number, slot = job
        started = time.perf_counter()
        try:
            bundle = Bundle('cpu')
            tree = prepare(recipe_batch(bank, blocks, cfg), bundle)
            arrays, spans, nbytes = bundle.layout()
            fresh = None
            if slot is None or slot.numel() < nbytes:
                fresh = slot = shared(slot_bytes(nbytes))
            slots[number] = slot
            bundle.write(slot.numpy(), arrays)
            results.put((index, number, fresh, nbytes, spans, tree, None))
        except Exception:
            results.put((index, number, None, 0, None, None, traceback.format_exc()))
        LOG.debug('serve batch exit index=%d blocks=%d slot=%d seconds=%.4f', index, len(blocks), number, time.perf_counter() - started)


class Packer:

    def __init__(self, bank, cfg, device, workers):
        started = time.perf_counter()
        self.bank, self.cfg, self.device = bank, cfg, device
        self.position, self.closed = 0, False
        self.slots, self.free = {}, deque(range(2 * workers))
        context = multiprocessing.get_context('spawn')
        self.jobs, self.results = context.Queue(), context.Queue()
        threads = max(1, os.cpu_count() // workers)
        resident = max(4, RESIDENT // workers)
        self.processes = [context.Process(target=serve, args=(str(bank.directory), cfg, threads, resident, self.jobs, self.results), daemon=True)
                          for _ in range(workers)]
        for process in self.processes:
            process.start()
        LOG.info('packer enter workers=%d threads=%d resident=%d slots=%d seconds=%.3f', workers, threads, resident, len(self.free), time.perf_counter() - started)

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

    def submit(self, index, groups):
        number = self.free.popleft()
        slot = self.slots.get(number)
        self.jobs.put((index, groups[index], number, None if slot is None else slot.tensor))

    def stream(self, groups):
        submitted, held, loader = 0, {}, None
        while submitted < len(groups) and self.free:
            self.submit(submitted, groups)
            submitted += 1
        for index in range(len(groups)):
            self.position = index
            while index not in held:
                try:
                    received = self.results.get(timeout=5)
                except queue.Empty:
                    if not all(process.is_alive() for process in self.processes):
                        raise RuntimeError('a packing process died') from None
                    continue
                held[received[0]] = received
            _, number, fresh, nbytes, spans, tree, failure = held.pop(index)
            if failure is not None:
                raise RuntimeError('packing failed in a worker process:\n' + failure)
            if fresh is not None:
                if number in self.slots:
                    self.slots[number].release()
                self.slots[number] = Slot(fresh, self.device)
            started = time.perf_counter()
            buffer = self.slots[number].transfer(nbytes, self.device)
            moved = time.perf_counter()
            self.free.append(number)
            if submitted < len(groups):
                self.submit(submitted, groups)
                submitted += 1
            if loader is None:
                loader = threading.Thread(target=self.prefetch, args=(groups, keys(tree, set())), daemon=True)
                loader.start()
            batch = resolve(tree, self.bank, self.cfg, views(buffer, spans), self.device)
            LOG.debug('packer batch exit index=%d bytes=%d slot=%d transfer=%.4f resolve=%.4f', index, nbytes, number, moved - started, time.perf_counter() - moved)
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
        for slot in self.slots.values():
            slot.release()
        LOG.info('packer exit workers=%d slots=%d bytes=%d seconds=%.3f', len(self.processes), len(self.slots), sum(slot.tensor.numel() for slot in self.slots.values()), time.perf_counter() - started)
