import itertools
import logging
import os
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from .processes.locks import Lock, LockBusy
from .timing import timed

LOG = logging.getLogger(__name__)


@timed("runtime.configured_home")
def configured_home():
    value = os.environ.get("GRAPHSPEC_HOME")
    if not value or not Path(value).is_absolute():
        raise ValueError("GRAPHSPEC_HOME must name an absolute directory outside the code checkout")
    return Path(os.path.abspath(value))


@timed("runtime.output_path")
def output_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    code = Path(os.path.abspath(__file__)).parents[2]
    if path.is_relative_to(code) or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("runtime output must be outside the code directory: %s" % path)
    if os.name == "nt" and path.drive.casefold() != configured_home().drive.casefold():
        raise ValueError("runtime output must use the GRAPHSPEC_HOME volume: %s" % path)
    return path


@timed("runtime.home")
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(temporary)
    if "numba" in sys.modules:
        sys.modules["numba"].config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]
    LOG.info("storage ready root=%s seconds=%.3f", root, time.perf_counter() - started)


def identities():
    from numba.core.bytecode import FunctionIdentity

    started = time.perf_counter()
    start = 1 + secrets.randbits(40)
    FunctionIdentity._unique_ids = itertools.count(start)
    LOG.info("identities of compiled functions start=%d seconds=%.3f", start, time.perf_counter() - started)
    return start


@contextmanager
def held(path):
    started = time.perf_counter()
    while True:
        try:
            lock = Lock(path).__enter__()
            break
        except LockBusy:
            time.sleep(0.001)
    try:
        yield lock
    finally:
        lock.__exit__(None, None, None)
        LOG.debug("held exit lock=%s seconds=%.4f", path.name, time.perf_counter() - started)


def reached(function):
    from numba.core.dispatcher import Dispatcher

    started = time.perf_counter()
    pending, seen = [function], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        codes, used = [current.__code__], set()
        while codes:
            code = codes.pop()
            used |= set(code.co_names)
            codes += [constant for constant in code.co_consts if isinstance(constant, type(code))]
        values = [current.__globals__.get(name) for name in used]
        values += [getattr(value, name, None) for value in values if isinstance(value, ModuleType) for name in used]
        pending += [value.py_func for value in values if isinstance(value, Dispatcher)]
    files = sorted({entry.__code__.co_filename for entry in seen})
    result = tuple((name, os.stat(name).st_mtime, os.stat(name).st_size) for name in files)
    LOG.debug(
        "reached exit function=%s files=%d seconds=%.4f", function.__name__, len(result), time.perf_counter() - started
    )
    return result


def compiled():
    from numba.core import caching

    started = time.perf_counter()
    keyed, loaded = caching.Cache._index_key, caching.IndexDataCacheFile.load

    def key(cache, signature, codegen):
        clock = time.perf_counter()
        result = keyed(cache, signature, codegen) + (reached(cache._py_func),)
        LOG.debug("compiled key exit seconds=%.4f", time.perf_counter() - clock)
        return result

    def load(index, found):
        clock = time.perf_counter()
        with held(Path(index._index_path + ".lock")):
            result = loaded(index, found)
        LOG.debug("compiled load exit hit=%s seconds=%.4f", result is not None, time.perf_counter() - clock)
        return result

    def save(index, found, data):
        clock = time.perf_counter()
        with held(Path(index._index_path + ".lock")):
            overloads = index._load_index()
            if found not in overloads:
                taken = set(overloads.values())
                name = next(name for name in map(index._data_name, itertools.count(1)) if name not in taken)
                index._save_data(name, data)
                index._save_index({**overloads, found: name})
        LOG.debug("compiled save exit seconds=%.4f", time.perf_counter() - clock)

    caching.Cache._index_key = key
    caching.IndexDataCacheFile.load = load
    caching.IndexDataCacheFile.save = save
    LOG.info("compiled code shared between processes seconds=%.3f", time.perf_counter() - started)


@timed("runtime.cache_path")
def cache_path(namespace, identity):
    return home() / "cache" / namespace / identity
