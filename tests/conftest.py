import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib import import_module

import pytest

from graphspec.infra import runtime
from graphspec.infra.processes import LockBusy

WORKER = "PYTEST_XDIST_WORKER" in os.environ
LOG = logging.getLogger(__name__)
SUITE = {"heavy": "heavy-tests"}


def suite_singleton(name, message, original=runtime.singleton):
    started = time.perf_counter()
    lock = original(SUITE.get(name, name), message)
    LOG.debug("suite_singleton exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return lock


runtime.singleton = suite_singleton


@contextmanager
def shared_heavy_job(original=runtime.heavy_job):
    while True:
        job = original()
        try:
            job.__enter__()
            break
        except RuntimeError as error:
            if not isinstance(error.__cause__, LockBusy):
                raise
            time.sleep(0.05)
    try:
        yield
    finally:
        job.__exit__(None, None, None)


if WORKER:
    runtime.heavy_job = shared_heavy_job


@pytest.fixture(scope="session")
def bank(tmp_path_factory):
    from graphspec.synthetic import generate
    from tuning.build import build

    root = tmp_path_factory.mktemp("bank")
    task = generate(
        root,
        "tiny",
        4,
        "query",
        1,
        entities=40,
        steps=12,
        features=8,
        signals=2,
        degree=2.0,
        max_degree=6,
        alive=0.8,
        lifetime=4.0,
        period=3,
    )
    build(task, root / "bank", 1)
    return root / "bank"


@pytest.fixture(scope="session", autouse=True)
def warmup(request):
    if WORKER:
        return
    for name in ("packers", "candidate_pool"):
        if any(name in item.fixturenames for item in request.session.items):
            request.getfixturevalue(name)


@pytest.fixture(scope="session")
def packers(bank):
    from tuning.data import Bank
    from tuning.model_spec import configuration
    from tuning.packing import Packer
    from tuning.variants import prepare

    held = {}
    for view in ("snapshot", "event_graph"):
        cfg = configuration(Bank(bank).registry, dict(view=view), {"time/history": 4}, 5)
        prepare(bank, cfg, 1)
        held[view] = cfg, Packer(Bank(bank), cfg, "cuda", 1)
    yield held
    for _, packer in held.values():
        packer.close()


@pytest.fixture(scope="session")
def candidate_pool():
    from tuning import space

    executor = ProcessPoolExecutor(2)
    for _ in range(2):
        executor.submit(import_module, "tuning.space")
    space.POOL["executor"] = executor
    yield executor
    executor.shutdown()
    del space.POOL["executor"]
