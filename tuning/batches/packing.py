import logging
import os
import sys
import threading
import time
import traceback
from collections import Counter, deque
from multiprocessing.connection import wait
from pathlib import Path

import numba
import torch
import torch.multiprocessing as multiprocessing

from graphspec.infra.arrays import RESIDENT, Bundle, Slot, shared, slot_bytes, views
from graphspec.infra.log import configure
from graphspec.infra.processes.processes import worker_threads
from graphspec.infra.timing import Timeline, measure, timed
from graphspec.registries import carried, restored
from tuning.bank.bank import Deriver

from .batching import recipe_batch
from .tensors import keys, prepare, resolve

LOG = logging.getLogger(__name__)


@timed("packing.serve")
def serve(bank_directory, cfg, threads, resident, device, jobs, results, entries, timing_directory=None):
    path = None if timing_directory is None else Path(timing_directory) / ("packing-%s.json" % os.getpid())
    with Timeline(path), measure("packing_worker"):
        serve_batches(bank_directory, cfg, threads, resident, device, jobs, results, entries)
    results.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def serve_batches(bank_directory, cfg, threads, resident, device, jobs, results, entries):
    configure(json_lines=True)
    numba.set_num_threads(threads)
    torch.set_num_threads(1)
    failure = None
    try:
        restored(entries)
        bank = Deriver(bank_directory, device, resident)
    except Exception:
        failure = traceback.format_exc()
    slots = {}
    while True:
        with measure("queue_wait"):
            job = jobs.get()
        if job is None:
            break
        index, blocks, number, slot = job
        if failure is not None:
            results.send((index, number, None, 0, None, None, failure))
            continue
        started = time.perf_counter()
        try:
            bundle = Bundle("cpu")
            with measure("batch_recipe"):
                recipe = recipe_batch(bank, blocks, cfg)
            with measure("tensor_bundle"):
                tree = prepare(recipe, bundle)
            arrays, spans, nbytes = bundle.layout()
            fresh = None
            if slot is None or slot.numel() < nbytes:
                fresh = slot = shared(slot_bytes(nbytes))
            slots[number] = slot
            with measure("shared_memory_write"):
                bundle.write(slot.numpy(), arrays)
            with measure("result_send"):
                results.send((index, number, fresh, nbytes, spans, tree, None))
        except Exception:
            results.send((index, number, None, 0, None, None, traceback.format_exc()))
        LOG.debug(
            "serve batch exit index=%d blocks=%d slot=%d seconds=%.4f",
            index,
            len(blocks),
            number,
            time.perf_counter() - started,
        )


class Packer:
    @timed("packing.Packer.__enter__")
    def __enter__(self):
        return self

    @timed("packing.Packer.__exit__")
    def __exit__(self, *exception):
        self.close()

    def __init__(self, bank, cfg, device, workers, timing_directory=None):
        started = time.perf_counter()
        if workers < 1:
            raise ValueError("packing requires at least one worker")
        self.bank, self.cfg, self.device = bank, cfg, device
        self.position, self.closed, self.pending = 0, False, 0
        self.condition = threading.Condition()
        self.streaming = False
        self.loader, self.loading_done = None, None
        self.slots, self.free = {}, deque(range(2 * workers))
        context = multiprocessing.get_context("spawn")
        self.jobs = context.Queue()
        self.results, self.processes = [], []
        threads = max(1, numba.config.NUMBA_NUM_THREADS // workers)
        resident = max(4, RESIDENT // workers)
        entries = carried()
        try:
            for _ in range(workers):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(
                    target=serve,
                    args=(
                        str(bank.directory),
                        cfg,
                        threads,
                        resident,
                        bank.device,
                        self.jobs,
                        writer,
                        entries,
                        timing_directory,
                    ),
                    daemon=True,
                )
                self.results.append(reader)
                try:
                    with worker_threads(threads):
                        process.start()
                finally:
                    writer.close()
                self.processes.append(process)
        except BaseException:
            self.pending = 1
            self.close()
            raise
        LOG.info(
            "packer enter workers=%d threads=%d resident=%d slots=%d seconds=%.3f",
            workers,
            threads,
            resident,
            len(self.free),
            time.perf_counter() - started,
        )

    @timed("resident_prefetch")
    def prefetch(self, groups, wanted, done, failures):
        started, loaded = time.perf_counter(), 0
        members = [set(block[0] for block in group) for group in groups]
        copies = 1 + int(torch.device(self.device).type != "cpu")
        limit = self.bank.residents.limit // copies
        active, first = Counter(), 0

        @timed("packing.Packer.prefetch.ready")
        def ready(index):
            nonlocal first
            while first < min(self.position, index + 1):
                for graph in members[first]:
                    active[graph] -= 1
                    if not active[graph]:
                        del active[graph]
                first += 1
            return done.is_set() or index <= self.position or len(active) <= limit

        try:
            for index, needed in enumerate(members):
                active.update(needed)
                with self.condition:
                    self.condition.wait_for(lambda index=index: ready(index))
                    if done.is_set():
                        break
                    if index <= self.position:
                        continue
                for graph in sorted(needed) if wanted else ():
                    held = self.bank.resident(graph, self.cfg, self.device)
                    for key in wanted:
                        held.tensor(key)
                    loaded += 1
        except Exception as error:
            failures.append(error)
        LOG.info(
            "prefetch exit groups=%d graphs=%d keys=%s seconds=%.3f",
            len(groups),
            loaded,
            ",".join(sorted(wanted)),
            time.perf_counter() - started,
        )

    @timed("packing.Packer.submit")
    def submit(self, index, groups):
        number = self.free.popleft()
        slot = self.slots.get(number)
        self.jobs.put((index, groups[index], number, None if slot is None else slot.tensor))
        self.pending += 1

    @timed("worker_wait")
    def receive(self):
        ready = wait(self.results + [process.sentinel for process in self.processes])
        for connection in self.results:
            if connection in ready:
                try:
                    result = connection.recv()
                except EOFError as error:
                    raise RuntimeError("packing worker closed its result channel") from error
                self.pending -= 1
                return result
        exited = [(process.pid, process.exitcode) for process in self.processes if process.sentinel in ready]
        raise RuntimeError("packing workers exited before returning their batches: %s" % exited)

    def stream(self, groups):
        if self.closed or self.streaming:
            raise RuntimeError("packer must be open and idle before starting a stream")
        self.streaming = True
        submitted, held, loader = 0, {}, None
        done, failures = threading.Event(), []
        self.loading_done = done
        completed = False
        try:
            while submitted < len(groups) and self.free:
                self.submit(submitted, groups)
                submitted += 1
            for index in range(len(groups)):
                if self.closed:
                    raise RuntimeError("packer closed while streaming")
                with self.condition:
                    self.position = index
                    self.condition.notify_all()
                if failures:
                    raise RuntimeError("resident prefetch failed") from failures[0]
                while index not in held:
                    received = self.receive()
                    held[received[0]] = received
                _, number, fresh, nbytes, spans, tree, failure = held.pop(index)
                if failure is not None:
                    raise RuntimeError("packing failed in a worker process:\n" + failure)
                if fresh is not None:
                    if number in self.slots:
                        self.slots[number].release()
                    self.slots[number] = Slot(fresh, self.device)
                started = time.perf_counter()
                with measure("batch_transfer"):
                    buffer = self.slots[number].transfer(nbytes, self.device)
                moved = time.perf_counter()
                self.free.append(number)
                if submitted < len(groups):
                    self.submit(submitted, groups)
                    submitted += 1
                if loader is None:
                    loader = threading.Thread(
                        target=self.prefetch, args=(groups, keys(tree, set()), done, failures), daemon=True
                    )
                    self.loader = loader
                    loader.start()
                batch = resolve(tree, self.bank, self.cfg, views(buffer, spans), self.device)
                LOG.debug(
                    "packer batch exit index=%d bytes=%d slot=%d transfer=%.4f resolve=%.4f",
                    index,
                    nbytes,
                    number,
                    moved - started,
                    time.perf_counter() - moved,
                )
                yield batch
            completed = True
        finally:
            self.stop_prefetch()
            self.streaming = False
            if not completed:
                self.close()
        if failures:
            raise RuntimeError("resident prefetch failed") from failures[0]

    @timed("prefetch_join")
    def stop_prefetch(self):
        if self.loading_done is not None:
            self.loading_done.set()
            with self.condition:
                self.condition.notify_all()
        if self.loader is not None:
            self.loader.join()
        self.loader, self.loading_done = None, None

    def close(self):
        started = time.perf_counter()
        if self.closed:
            LOG.debug("packer exit closed before seconds=%.4f", time.perf_counter() - started)
            return
        self.closed = True
        self.stop_prefetch()
        if self.pending:
            for process in self.processes:
                if process.is_alive():
                    process.kill()
            self.jobs.cancel_join_thread()
        else:
            for _ in self.processes:
                self.jobs.put(None)
        for process in self.processes:
            process.join()
            process.close()
        self.jobs.close()
        if not self.pending:
            self.jobs.join_thread()
        for connection in self.results:
            connection.close()
        for slot in self.slots.values():
            slot.release()
        LOG.info(
            "packer exit workers=%d slots=%d bytes=%d seconds=%.3f",
            len(self.processes),
            len(self.slots),
            sum(slot.tensor.numel() for slot in self.slots.values()),
            time.perf_counter() - started,
        )
