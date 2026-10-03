import errno
import logging
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from ..timing import timed

if os.name == "nt":
    import msvcrt
else:
    import fcntl

LOG = logging.getLogger(__name__)
ACTIVE = threading.local()
SERIAL = threading.RLock()


class LockBusy(OSError):
    pass


class Lock:
    @timed("locks.Lock.__init__")
    def __init__(self, path):
        self.path = Path(path)

    @timed("locks.Lock.__enter__")
    def __enter__(self):
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise LockBusy(str(self.path)) from error
            raise
        return self

    @timed("locks.Lock.__exit__")
    def __exit__(self, *exception):
        self.stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


@contextmanager
def singleton(name, message):
    started = time.perf_counter()
    if os.name == "nt":
        from .windows_mutex import Mutex

        lock = Mutex(name)
    else:
        directory = Path.home() / ".graphspec" / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        lock = Lock(directory / (name + ".lock"))
    try:
        lock.__enter__()
    except LockBusy as error:
        raise RuntimeError(message) from error
    try:
        yield
    finally:
        lock.__exit__(None, None, None)
        LOG.debug("singleton exit name=%s seconds=%.4f", name, time.perf_counter() - started)


@contextmanager
def heavy_job():
    with SERIAL:
        if getattr(ACTIVE, "held", False):
            yield
            return
        started = time.perf_counter()
        with singleton(
            "heavy", "another graph build or training job is running; finish it before starting this job"
        ):
            ACTIVE.held = True
            LOG.info("heavy job enter pid=%d seconds=%.3f", os.getpid(), time.perf_counter() - started)
            try:
                yield
            finally:
                ACTIVE.held = False
                LOG.info("heavy job exit pid=%d seconds=%.3f", os.getpid(), time.perf_counter() - started)


@timed("locks.exclusive")
def exclusive(function):
    @wraps(function)
    @timed("locks.exclusive.run")
    def run(*args, **kwargs):
        with heavy_job():
            return function(*args, **kwargs)

    return run
