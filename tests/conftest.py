import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib import import_module

import pytest

from graphspec.infra import compute, runtime
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
compute.state("cpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: the test is about the GPU and is skipped where none is present")


def pytest_collection_modifyitems(items):
    import torch

    started = time.perf_counter()
    absent = pytest.mark.skip(reason="no GPU is present")
    marked = [item for item in items if "gpu" in item.keywords]
    if not torch.cuda.is_available():
        for item in marked:
            item.add_marker(absent)
    LOG.debug("gpu marks exit marked=%d seconds=%.4f", len(marked), time.perf_counter() - started)


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
    from tuning.batching import batches
    from tuning.data import Bank
    from tuning.model_spec import configuration
    from tuning.packing import Packer
    from tuning.turns import blocks as turn_blocks
    from tuning.variants import prepare

    held = {}
    for view in ("snapshot", "event_graph"):
        cfg = configuration(Bank(bank).registry, dict(view=view), {"time/history": 4}, 5)
        prepare(bank, cfg, 1)
        held[view] = cfg, Packer(Bank(bank), cfg, "cpu", 1)
    started = time.perf_counter()
    for cfg, packer in held.values():
        opened = packer.bank
        blocks = turn_blocks(opened, opened.blocks(0, cfg["graph"]["time/history"], cfg), cfg)
        list(packer.stream(batches(blocks, 2, opened.sizes(cfg), 5, frame_budget=1024)[:1]))
    LOG.debug("packers warm exit seconds=%.4f", time.perf_counter() - started)
    yield held
    for _, packer in held.values():
        packer.close()


@pytest.fixture
def packed(packers, monkeypatch):
    import torch

    started = time.perf_counter()
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])

    class Shared:
        def __init__(self, bank, cfg, device, workers, timing_directory=None):
            held = [packer for known, packer in packers.values() if known["model"] == cfg["model"]]
            if not held or device != "cpu" or held[0].cfg["graph"] != cfg["graph"]:
                raise ValueError("the suite holds no packer for this configuration")
            self.stream = held[0].stream

        def close(self):
            pass

    monkeypatch.setattr("tuning.packing.Packer", Shared)
    LOG.debug("packed exit seconds=%.4f", time.perf_counter() - started)
    return {view: dict(cfg, epochs=1) for view, (cfg, _) in packers.items()}


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
