import logging
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from ..timing import timed

RECORD = 32
if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
    for name in ("LockFile", "UnlockFile"):
        getattr(KERNEL, name).argtypes = (wintypes.HANDLE,) + (wintypes.DWORD,) * 4
        getattr(KERNEL, name).restype = wintypes.BOOL
    VIOLATION = 33
    PLACE = (2 * RECORD, 0, 1, 0)
else:
    import fcntl

LOG = logging.getLogger(__name__)
ACTIVE = threading.local()
SERIAL = threading.RLock()


class LockBusy(OSError):
    pass


if os.name == "nt":

    @timed("locks.claimed")
    def claimed(stream, path):
        if KERNEL.LockFile(msvcrt.get_osfhandle(stream.fileno()), *PLACE):
            return True
        code = ctypes.get_last_error()
        if code != VIOLATION:
            raise OSError(0, ctypes.FormatError(code), str(path), code)
        return False

    @timed("locks.released")
    def released(stream, path):
        if not KERNEL.UnlockFile(msvcrt.get_osfhandle(stream.fileno()), *PLACE):
            code = ctypes.get_last_error()
            raise OSError(0, ctypes.FormatError(code), str(path), code)

else:

    @timed("locks.claimed")
    def claimed(stream, path):
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as error:
            raise OSError(error.errno, error.strerror, str(path)) from error
        return True

    @timed("locks.released")
    def released(stream, path):
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@timed("locks.holder")
def holder(path):
    with open(path, "rb", buffering=0) as stream:
        text = stream.read(RECORD)
    return int(text) if text.isdigit() else None


class Lock:
    @timed("locks.Lock.__init__")
    def __init__(self, path):
        self.path = Path(path)

    @timed("locks.Lock.__enter__")
    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if not claimed(self.stream, self.path):
                raise LockBusy(str(self.path))
            self.stream.truncate(0)
            self.stream.write(b"%d" % os.getpid())
            self.stream.flush()
        except BaseException:
            self.stream.close()
            raise
        return self

    @timed("locks.Lock.__exit__")
    def __exit__(self, *exception):
        try:
            self.stream.truncate(0)
            released(self.stream, self.path)
        finally:
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
