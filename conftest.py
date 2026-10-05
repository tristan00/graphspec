import hashlib
import logging
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

LOG = logging.getLogger(__name__)
ALIVE = []
RUNS = "runs.lock"
CORES = 2
THREADS = dict(NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")


def pinned(cores):
    started = time.perf_counter()
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        raise RuntimeError(
            "the suite holds itself to %d cores by processor affinity, which %s does not offer" % (cores, sys.platform)
        )
    process.cpu_affinity(process.cpu_affinity()[:cores])
    result = process.cpu_affinity()
    LOG.debug("pinned exit cores=%s seconds=%.4f", result, time.perf_counter() - started)
    return result


def lasting(suite, checkout):
    started = time.perf_counter()
    checkout = Path(os.path.abspath(checkout))
    named = "%s-%s" % (checkout.name, hashlib.sha256(str(checkout).casefold().encode()).hexdigest()[:12])
    result = Path(suite) / "workers" / named
    LOG.debug("lasting exit checkout=%s seconds=%.4f", checkout.name, time.perf_counter() - started)
    return result


def home(suite, run, worker):
    started = time.perf_counter()
    result = Path(suite) / "workers" / run / worker
    LOG.debug("home exit worker=%s seconds=%.4f", worker, time.perf_counter() - started)
    return result


def linked(place, compiled):
    started = time.perf_counter()
    compiled.mkdir(parents=True, exist_ok=True)
    link = place / "cache" / "numba"
    link.parent.mkdir(parents=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(compiled), str(link))
    else:
        link.symlink_to(compiled, target_is_directory=True)
    LOG.debug("linked exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return link


def bytecode(place):
    started = time.perf_counter()
    os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix = str(place)
    LOG.debug("bytecode exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return place


def temporary(place):
    started = time.perf_counter()
    result = place / "tmp"
    result.mkdir(parents=True, exist_ok=True)
    LOG.debug("temporary exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return result


def gone(place):
    started = time.perf_counter()
    if place.is_dir():
        shutil.rmtree(place)
    else:
        place.unlink()
    LOG.debug("gone exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return place


def cleared(run):
    started = time.perf_counter()
    places = [place for worker in sorted(run.iterdir()) if worker.is_dir() for place in sorted(worker.iterdir())]
    with ThreadPoolExecutor(8) as pool:
        removed = list(pool.map(gone, places))
    shutil.rmtree(run)
    LOG.debug("cleared exit removed=%d seconds=%.4f", len(removed), time.perf_counter() - started)
    return removed


def dead(run):
    from graphspec.infra.processes.locks import Lock, LockBusy

    started = time.perf_counter()
    held, busy = [], False
    for place in sorted(run.glob("*.alive")):
        try:
            held.append(Lock(place).__enter__())
        except LockBusy:
            busy = True
            break
    for lock in held:
        lock.__exit__(None, None, None)
    result = bool(held) and not busy
    LOG.debug("dead exit run=%s dead=%s seconds=%.4f", run.name, result, time.perf_counter() - started)
    return result


def ended(suite, run):
    started = time.perf_counter()
    shared = Path(suite) / "tmp" / "suite" / run.name
    if shared.exists():
        shutil.rmtree(shared)
    removed = cleared(run)
    mirror = Path(sys.pycache_prefix) / run.relative_to(run.anchor)
    if mirror.exists():
        shutil.rmtree(mirror)
    LOG.info("ended exit run=%s removed=%d seconds=%.4f", run.name, len(removed), time.perf_counter() - started)
    return run


def begun(suite, run, worker, controller):
    from graphspec.infra.processes.locks import Lock
    from graphspec.infra.runtime import held

    started = time.perf_counter()
    workers, removed = Path(suite) / "workers", []
    (workers / run).mkdir(parents=True, exist_ok=True)
    with held(workers / RUNS):
        alive = Lock(workers / run / (worker + ".alive")).__enter__()
        if controller:
            others = [place for place in sorted(workers.iterdir()) if place.is_dir() and place.name != run]
            removed = [ended(suite, place) for place in others if dead(place)]
    LOG.info("begun exit run=%s removed=%d seconds=%.4f", run, len(removed), time.perf_counter() - started)
    return alive, removed


def finished(suite, run, alive):
    from graphspec.infra.runtime import held

    started = time.perf_counter()
    workers = Path(suite) / "workers"
    with held(workers / RUNS):
        alive.__exit__(None, None, None)
        for place in sorted((workers / run).glob("*.alive")):
            with held(place):
                pass
        result = ended(suite, workers / run)
    LOG.info("finished exit run=%s seconds=%.4f", run, time.perf_counter() - started)
    return result


pinned(CORES)
os.environ.update(THREADS)
if "GRAPHSPEC_SUITE_CONTROLLER" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_CONTROLLER"] = str(os.getpid())
if "GRAPHSPEC_SUITE_HOME" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_HOME"] = os.environ["GRAPHSPEC_HOME"]
if "GRAPHSPEC_SUITE_RUN" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_RUN"] = uuid.uuid4().hex[:12]
os.environ["GRAPHSPEC_SUITE_CHECKOUT"] = str(lasting(os.environ["GRAPHSPEC_SUITE_HOME"], Path(__file__).parent))
os.environ["GRAPHSPEC_HOME"] = str(
    home(
        os.environ["GRAPHSPEC_SUITE_HOME"],
        os.environ["GRAPHSPEC_SUITE_RUN"],
        os.environ.get("PYTEST_XDIST_WORKER", "serial"),
    )
)


@pytest.hookimpl(wrapper=True)
def pytest_sessionstart(session):
    linked(Path(os.environ["GRAPHSPEC_HOME"]), Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) / "numba")
    bytecode(Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) / "bytecode")
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(temporary(Path(os.environ["GRAPHSPEC_HOME"])))
    result = yield
    started = time.perf_counter()
    ALIVE.append(
        begun(
            os.environ["GRAPHSPEC_SUITE_HOME"],
            os.environ["GRAPHSPEC_SUITE_RUN"],
            os.environ.get("PYTEST_XDIST_WORKER", "serial"),
            "PYTEST_XDIST_WORKER" not in os.environ,
        )[0]
    )
    LOG.debug("session start exit seconds=%.4f", time.perf_counter() - started)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_unconfigure(config):
    result = yield
    while ALIVE and "PYTEST_XDIST_WORKER" not in os.environ:
        finished(os.environ["GRAPHSPEC_SUITE_HOME"], os.environ["GRAPHSPEC_SUITE_RUN"], ALIVE.pop())
    return result
