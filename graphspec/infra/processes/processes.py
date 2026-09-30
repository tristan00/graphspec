import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import psutil

from graphspec.infra.files import write_json
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
FAILED = []


def command(pid):
    try:
        return psutil.Process(pid).cmdline()
    except psutil.NoSuchProcess:
        return []


class WorkerProcess:
    def __init__(self, module, arguments, stream):
        if os.name == "nt":
            from .windows_job import ProcessGroup
        elif sys.platform == "linux":
            from .linux_group import ProcessGroup
        else:
            raise RuntimeError("isolated tuning workers require Windows or Linux")

        self.group = ProcessGroup()
        self.process = None
        self.closed = False
        self.started = False
        self.invocation = [sys.executable, "-B", "-u", "-m", "graphspec.infra.processes.worker", module, *arguments]
        self.stream = stream

    @property
    def pid(self):
        return self.process.pid

    @timed("worker_start")
    def __enter__(self):
        try:
            self.process = subprocess.Popen(
                self.invocation,
                stdin=subprocess.PIPE,
                stdout=self.stream,
                stderr=subprocess.STDOUT,
                **self.group.popen_options,
            )
            self.group.attach(self.process)
            self.started = True
            self.process.stdin.write(b"1")
            self.process.stdin.flush()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exception):
        self.close()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    @timed("worker_cleanup")
    def close(self):
        if self.closed:
            return
        started = time.perf_counter()
        try:
            if self.process is not None:
                if not self.started:
                    self.process.stdin.close()
                    self.process.wait()
                self.group.stop()
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
        finally:
            try:
                self.group.close()
            finally:
                self.closed = True
                if self.process is not None:
                    self.process.stdin.close()
            LOG.info("worker close seconds=%.3f", time.perf_counter() - started)


class Heartbeat:
    def __init__(self, path, deadline):
        self.path, self.deadline = Path(path), deadline

    def pulse(self, **fields):
        write_json(self.path, dict(fields, deadline=self.deadline, updated=time.perf_counter()))


def wait(process, deadline):
    return process.wait(timeout=max(0.0, deadline - time.perf_counter()))


@contextmanager
def worker_threads(threads):
    values = dict(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)
    before = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def attached(entries, restore, initializer, initargs):
    started = time.perf_counter()
    try:
        restore(entries)
        if initializer is not None:
            initializer(*initargs)
    except BaseException as error:
        FAILED.append(error)
    LOG.debug("attached exit failed=%d seconds=%.4f", len(FAILED), time.perf_counter() - started)


def guarded(function, *arguments):
    if FAILED:
        raise FAILED[0]
    return function(*arguments)


def spawned(initializer, initargs, carry, restore):
    return dict(initializer=attached, initargs=(carry(), restore, initializer, initargs))


@contextmanager
def pooled(workers, jobs, initializer, initargs, carry, restore):
    started = time.perf_counter()
    count = min(workers, jobs)
    if count <= 1:
        initializer(*initargs)
        yield lambda function, items: [function(item) for item in items]
    else:
        with (
            worker_threads(max(1, (os.cpu_count() or 1) // workers)),
            ProcessPoolExecutor(max_workers=count, **spawned(initializer, initargs, carry, restore)) as pool,
        ):
            yield lambda function, items: list(pool.map(partial(guarded, function), items))
    LOG.debug("pooled exit jobs=%d workers=%d seconds=%.3f", jobs, count, time.perf_counter() - started)


def parallel(function, jobs, workers, initializer, initargs, carry, restore):
    with pooled(workers, len(jobs), initializer, initargs, carry, restore) as mapped:
        return mapped(function, jobs)
