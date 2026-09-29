import logging
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from graphspec.infra.files import JsonCache
from graphspec.infra.processes import Heartbeat
from graphspec.infra.timing import Timeline

LOG = logging.getLogger(__name__)
TIMEOUT_EXIT_CODE = 124
REASON = "graph preparation exceeded its %d-second limit"


def policy(preparation_seconds):
    started = time.perf_counter()
    result = dict(
        version=4,
        graph_preparation_seconds=preparation_seconds,
        at_deadline="The trial is recorded as timed out. Its owned worker group is terminated and joined before the next trial starts.",
        scope="Selection, layout, source snapshot, variant preparation, packing, fitting and scoring count. Cleanup and cache pruning follow the trial budget.",
    )
    LOG.debug("policy exit seconds=%.4f", time.perf_counter() - started)
    return result


class FitBudget:
    def __init__(self, directory, deadline, preparation_seconds):
        self.timeline = Timeline()
        self.deadline = deadline
        self.preparation_seconds = preparation_seconds
        self.graph_preparation = None
        self.timeout_reason = None
        self.heartbeat = Heartbeat(Path(directory) / "fit_progress.json", deadline)
        self.epoch_seconds = []
        self.timings = {}
        self.active = {}
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None
        self.error = None
        self.phase_seconds = {}
        self.phase = "preparation"
        self.updated = time.perf_counter()
        self.pulse(self.phase)

    def __enter__(self):
        self.timeline.__enter__()
        self.root = self.timeline.measure("fit")
        self.root.__enter__()
        self.thread = threading.Thread(target=self.record, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        self.root.__exit__(*exception)
        self.pulse()
        self.timeline.__exit__(*exception)
        if exception[0] is None and self.error is not None:
            raise self.error

    def record(self):
        try:
            while not self.stopped.wait(1.0):
                self.pulse()
        except Exception as error:
            self.error = error
            self.stopped.set()

    @contextmanager
    def measure(self, name):
        with self.lock:
            self.active[name] = time.perf_counter()
            if name == "graph_prepare":
                self.graph_preparation = dict(
                    started=self.active[name],
                    deadline=self.active[name] + self.preparation_seconds,
                    finished=None,
                )
                self.pulse()
        try:
            with self.timeline.measure(name):
                yield
                if name == "graph_prepare" and time.perf_counter() >= self.graph_preparation["deadline"]:
                    self.timeout_reason = REASON % self.preparation_seconds
                    raise TimeoutError(self.timeout_reason)
        finally:
            with self.lock:
                self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - self.active.pop(name)
                if name == "graph_prepare":
                    self.graph_preparation["finished"] = time.perf_counter()
                    self.pulse()

    def batches(self, stream, name):
        iterator = iter(stream)
        while True:
            with self.measure(name):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("trial deadline reached")

    def pulse(self, phase=None):
        with self.lock:
            now = time.perf_counter()
            self.phase_seconds[self.phase] = self.phase_seconds.get(self.phase, 0.0) + now - self.updated
            self.updated = now
            if phase is not None:
                self.phase = phase
            timings = dict(self.timings)
            for name, started in self.active.items():
                timings[name] = timings.get(name, 0.0) + now - started
            self.heartbeat.pulse(
                timeline=self.timeline.snapshot(),
                phase=self.phase,
                phase_seconds=self.phase_seconds,
                timings=timings,
                epoch_seconds=list(self.epoch_seconds),
                recorded_at=time.time(),
                graph_preparation=self.graph_preparation,
                timeout_reason=self.timeout_reason,
            )


class PreparationTimeout(subprocess.TimeoutExpired):
    def __str__(self):
        return REASON % self.timeout


def wait_worker(process, deadline, directory, preparation_seconds):
    cache = JsonCache()
    while True:
        now = time.perf_counter()
        if now >= deadline:
            raise subprocess.TimeoutExpired("trial worker", 0)
        progress = cache.read(Path(directory) / "fit_progress.json") or {}
        preparation = progress.get("graph_preparation")
        remaining = deadline - now
        if preparation is not None:
            finished = preparation["finished"]
            if (now if finished is None else finished) >= preparation["deadline"]:
                raise PreparationTimeout("graph preparation", preparation_seconds)
            if finished is None:
                remaining = min(remaining, preparation["deadline"] - now)
        try:
            return process.wait(timeout=min(1.0, remaining))
        except subprocess.TimeoutExpired:
            pass
