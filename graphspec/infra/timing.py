import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

LOG = logging.getLogger(__name__)
CURRENT = None
UNMEASURED = nullcontext()


class Timeline:
    def __init__(self, path=None):
        started = time.perf_counter()
        self.path = None if path is None else Path(path)
        self.lock = threading.RLock()
        self.nodes = {}
        self.stacks = {}
        self.stopped = threading.Event()
        self.thread = None
        self.error = None
        LOG.debug("timeline exit recorded=%s seconds=%.4f", self.path is not None, time.perf_counter() - started)

    def __enter__(self):
        global CURRENT
        started = time.perf_counter()
        self.stopped.clear()
        self.error = None
        self.previous = CURRENT
        CURRENT = self
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self.record, daemon=True)
            self.thread.start()
        LOG.debug("timeline enter exit seconds=%.4f", time.perf_counter() - started)
        return self

    def __exit__(self, *exception):
        global CURRENT
        started = time.perf_counter()
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.flush()
        finally:
            CURRENT = self.previous
        LOG.debug("timeline leave exit seconds=%.4f", time.perf_counter() - started)
        if exception[0] is None and self.error is not None:
            raise self.error

    def record(self):
        started = time.perf_counter()
        try:
            while not self.stopped.wait(1):
                self.flush()
        except Exception as error:
            self.error = error
        LOG.debug("timeline record exit failed=%s seconds=%.4f", self.error is not None, time.perf_counter() - started)

    @contextmanager
    def measure(self, name):
        lane = "main" if threading.current_thread() is threading.main_thread() else "background"
        identity = threading.get_ident()
        with self.lock:
            stack = self.stacks.setdefault(identity, [])
            path = (*stack[-1][0], name) if stack else (lane, name)
            frame = [path, time.perf_counter(), 0.0]
            stack.append(frame)
        failure = None
        try:
            yield
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            with self.lock:
                elapsed = time.perf_counter() - frame[1]
                stack.pop()
                node = self.nodes.setdefault(path, dict(seconds=0.0, self_seconds=0.0, calls=0, failures=0))
                node["seconds"] += elapsed
                node["self_seconds"] += elapsed - frame[2]
                node["calls"] += 1
                if failure is not None:
                    node["failures"] += 1
                    node["last_error"] = failure
                if stack:
                    stack[-1][2] += elapsed

    def snapshot(self):
        with self.lock:
            now = time.perf_counter()
            nodes = {path: dict(value, active=0) for path, value in self.nodes.items()}
            for stack in self.stacks.values():
                for index, (path, started, children) in enumerate(stack):
                    elapsed = now - started
                    active_child = now - stack[index + 1][1] if index + 1 < len(stack) else 0
                    node = nodes.setdefault(
                        path, dict(seconds=0.0, self_seconds=0.0, calls=0, failures=0, active=0)
                    )
                    node["seconds"] += elapsed
                    node["self_seconds"] += elapsed - children - active_child
                    node["active"] += 1
            result = dict(
                version=1,
                pid=os.getpid(),
                recorded_at=time.time(),
                nodes=[dict(path=list(path), **value) for path, value in sorted(nodes.items())],
            )
        LOG.debug("timeline snapshot exit nodes=%d seconds=%.4f", len(nodes), time.perf_counter() - now)
        return result

    def flush(self):
        from .files import write_json

        started = time.perf_counter()
        if self.path is not None:
            with self.lock:
                write_json(self.path, self.snapshot())
        LOG.debug("timeline flush exit seconds=%.4f", time.perf_counter() - started)


def measure(name):
    recorder = CURRENT
    return UNMEASURED if recorder is None else recorder.measure(name)


def timed(name):
    def decorate(function):
        log = logging.getLogger(function.__module__)

        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                with measure(name):
                    return function(*args, **kwargs)
            finally:
                log.debug("%s exit seconds=%.4f", name, time.perf_counter() - started)

        return wrapped

    return decorate
