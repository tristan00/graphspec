import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from .files import write_json

CURRENT = None


class Timeline:
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.lock = threading.RLock()
        self.nodes = {}
        self.stacks = {}
        self.stopped = threading.Event()
        self.thread = None
        self.error = None

    def __enter__(self):
        global CURRENT
        self.stopped.clear()
        self.error = None
        self.previous = CURRENT
        CURRENT = self
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self.record, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exception):
        global CURRENT
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.flush()
        finally:
            CURRENT = self.previous
        if exception[0] is None and self.error is not None:
            raise self.error

    def record(self):
        try:
            while not self.stopped.wait(1):
                self.flush()
        except Exception as error:
            self.error = error

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
            return dict(
                version=1,
                pid=os.getpid(),
                recorded_at=time.time(),
                nodes=[dict(path=list(path), **value) for path, value in sorted(nodes.items())],
            )

    def flush(self):
        if self.path is not None:
            with self.lock:
                write_json(self.path, self.snapshot())


@contextmanager
def measure(name):
    recorder = CURRENT
    if recorder is None:
        yield
    else:
        with recorder.measure(name):
            yield


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate
