import itertools
import logging
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def configured_home():
    value = os.environ.get("GRAPHSPEC_HOME")
    if not value or not Path(value).is_absolute():
        raise ValueError("GRAPHSPEC_HOME must name an absolute directory outside the code checkout")
    return Path(os.path.abspath(value))


def output_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    code = Path(os.path.abspath(__file__)).parents[2]
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


def cache_path(namespace, identity):
    return home() / "cache" / namespace / identity
