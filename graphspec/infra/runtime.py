import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

LOG = logging.getLogger(__name__)
ACTIVE = threading.local()
SERIAL = threading.RLock()


def configured_home():
    value = os.environ.get("GRAPHSPEC_HOME")
    if not value or not Path(value).is_absolute():
        raise ValueError("GRAPHSPEC_HOME must name an absolute directory outside the code checkout")
    return Path(value).resolve()


def output_path(path):
    path = Path(path).expanduser().resolve()
    code = Path(__file__).resolve().parents[2]
    if path.is_relative_to(code) or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("runtime output must be outside the code directory: %s" % path)
    if os.name == "nt" and path.drive.casefold() != configured_home().drive.casefold():
        raise ValueError("runtime output must use the GRAPHSPEC_HOME volume: %s" % path)
    return path


def home():
    return output_path(configured_home())


def initialize_storage():
    started = time.perf_counter()
    root = home()
    temporary = output_path(root / "tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    directories = {
        "TMP": temporary,
        "TEMP": temporary,
        "TMPDIR": temporary,
        "PYTEST_DEBUG_TEMPROOT": temporary,
        "XDG_CACHE_HOME": root / "cache",
        "NUMBA_CACHE_DIR": root / "cache" / "numba",
        "TORCH_HOME": root / "cache" / "torch",
        "TORCH_EXTENSIONS_DIR": root / "cache" / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": root / "cache" / "torchinductor",
        "TRITON_CACHE_DIR": root / "cache" / "triton",
        "CUDA_CACHE_PATH": root / "cache" / "cuda",
        "PYTORCH_KERNEL_CACHE_PATH": root / "cache" / "torch_kernels",
        "RUFF_CACHE_DIR": root / "cache" / "ruff",
        "PIP_CACHE_DIR": root / "cache" / "pip",
        "npm_config_cache": root / "cache" / "npm",
    }
    os.environ.update({key: str(output_path(path)) for key, path in directories.items()})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(temporary)
    if "numba" in sys.modules:
        sys.modules["numba"].config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]
    LOG.info("storage ready root=%s seconds=%.3f", root, time.perf_counter() - started)


def cache_path(namespace, identity):
    return home() / "cache" / namespace / identity


@contextmanager
def singleton(name, message):
    from .processes import Lock, LockBusy

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


def exclusive(function):
    @wraps(function)
    def run(*args, **kwargs):
        with heavy_job():
            return function(*args, **kwargs)

    return run
