import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest

from graphspec.infra import locks
from graphspec.infra.locks import Lock, LockBusy

WORKER = "PYTEST_XDIST_WORKER" in os.environ
LOG = logging.getLogger(__name__)
SUITE = {"heavy": "heavy-tests-" + os.environ.get("PYTEST_XDIST_WORKER", "serial")}
SETTINGS = dict(entities=40, steps=12, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
SPLITS = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))


def suite_singleton(name, message, original=locks.singleton):
    started = time.perf_counter()
    lock = original(SUITE.get(name, name), message)
    LOG.debug("suite_singleton exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return lock


locks.singleton = suite_singleton


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
def shared_heavy_job(original=locks.heavy_job):
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


locks.heavy_job = shared_heavy_job


def suite():
    return Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "tmp" / "suite" / os.environ["GRAPHSPEC_SUITE_RUN"]


def shared(name, make):
    started = time.perf_counter()
    root = suite() / name
    root.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            with Lock(root.parent / (name + ".lock")):
                if not (root / "made").exists():
                    root.mkdir()
                    make(root)
                    (root / "made").write_text("")
            break
        except LockBusy:
            time.sleep(0.01)
    LOG.debug("shared exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return root


def cleared(checkout):
    started = time.perf_counter()
    removed = []
    for worker in sorted(checkout.iterdir()):
        places = [place for place in sorted(worker.iterdir()) if place.name != "cache"]
        if (worker / "cache").is_dir():
            places += [place for place in sorted((worker / "cache").iterdir()) if place.name != "numba"]
        for place in places:
            if place.is_dir():
                shutil.rmtree(place)
            else:
                place.unlink()
            removed.append(place)
    LOG.debug("cleared exit removed=%d seconds=%.4f", len(removed), time.perf_counter() - started)
    return removed


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    started = time.perf_counter()
    if not WORKER:
        if suite().exists():
            shutil.rmtree(suite())
        cleared(Path(os.environ["GRAPHSPEC_HOME"]).parent)
    LOG.debug("suite removed seconds=%.4f", time.perf_counter() - started)


def synthetic_bank(root, level):
    from graphspec.synthetic import generate, uniform
    from tuning.build import build

    started = time.perf_counter()
    task = generate(root, "tiny", 4, 1, uniform(4, 2, level), **SETTINGS)
    build(task, root / "bank", 1, ("train",))
    LOG.debug("synthetic bank exit level=%s seconds=%.4f", level, time.perf_counter() - started)


@pytest.fixture(scope="session")
def bank():
    return shared("bank", lambda root: synthetic_bank(root, "task")) / "bank"


@pytest.fixture(scope="session")
def path_bank():
    return shared("path", lambda root: synthetic_bank(root, "path")) / "bank"


@pytest.fixture(scope="session")
def varied(tmp_path_factory):
    from graphspec import synthetic
    from tuning.build import build

    root = tmp_path_factory.mktemp("varied")
    task = synthetic.generate(root, "varied", 4, 1, synthetic.varied("task"), **SETTINGS)
    build(task, root / "bank", 1, ("train",))
    return root / "bank"


@pytest.fixture(scope="session")
def ground():
    place = Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "ground" / "fixture"
    if not (place / "fixture.json").exists():
        raise FileNotFoundError("the fixture of the ground is absent: %s" % place)
    return place


def stated(directory, view):
    from support import several

    from tuning.data import Bank
    from tuning.model_spec import configuration

    started = time.perf_counter()
    registry = Bank(directory).registry
    cfg = configuration(registry, dict(view=view), {"time/history": 4, **several(registry)}, 5, SPLITS)
    LOG.debug("stated exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg


def running(directory, view):
    from graphspec.infra.arrays import RESIDENT
    from tuning.data import Deriver
    from tuning.packing import Packer
    from tuning.variants import prepare

    started = time.perf_counter()
    cfg = stated(directory, view)
    prepare(directory, cfg, 1, "cpu")
    packer = Packer(Deriver(directory, "cpu", RESIDENT), cfg, "cpu", 1)
    LOG.debug("running exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg, packer


def warmed(held):
    from tuning.batching import batches
    from tuning.periods import blocks as period_blocks

    started = time.perf_counter()
    for cfg, packer in held:
        opened = packer.bank
        blocks = period_blocks(opened, opened.blocks(0, cfg["graph"]["time/history"], cfg), cfg)
        list(packer.stream(batches(blocks, 2, opened.sizes(cfg), 5, frame_budget=1024)[:1]))
    LOG.debug("packers warm exit seconds=%.4f", time.perf_counter() - started)


def closed(held):
    started = time.perf_counter()
    for cfg, packer in held:
        packer.close()
    LOG.debug("closed exit packers=%d seconds=%.4f", len(held), time.perf_counter() - started)


@pytest.fixture(scope="session", autouse=True)
def warmup(request):
    if WORKER:
        return
    for name in ("packers", "candidate_pool"):
        if any(name in item.fixturenames for item in request.session.items):
            request.getfixturevalue(name)


@pytest.fixture(scope="session")
def packers(bank):
    held = {view: running(bank, view) for view in ("snapshot", "event_graph")}
    warmed(held.values())
    yield held
    closed(held.values())


@pytest.fixture
def statement():
    from tuning.statement import VERSION

    return dict(
        version=VERSION,
        device="cpu",
        seed=5,
        seconds=30,
        preparation_seconds=300,
        workers=1,
        variant_cache_bytes=1 << 30,
        batch_sequences=16,
        **SPLITS,
    )


class Inline:
    def __init__(self, bank, cfg, device, workers, timing_directory=None):
        if device != "cpu" or workers != 1:
            raise ValueError("the suite packs in its own process, on cpu, as one worker")
        self.bank, self.cfg = bank, cfg

    def stream(self, groups):
        from tuning.batching import recipe_batch
        from tuning.tensors import materialise

        started = time.perf_counter()
        for group in groups:
            yield materialise(self.bank, self.cfg, recipe_batch(self.bank, group, self.cfg), "cpu")
        LOG.debug("inline stream exit batches=%d seconds=%.4f", len(groups), time.perf_counter() - started)

    def close(self):
        pass


def inline(directory, views, monkeypatch):
    import torch

    started = time.perf_counter()
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr("tuning.packing.Packer", Inline)
    result = {view: dict(stated(directory, view), epochs=1) for view in views}
    LOG.debug("inline exit views=%d seconds=%.4f", len(views), time.perf_counter() - started)
    return result


@pytest.fixture
def packed(bank, monkeypatch):
    return inline(bank, ("snapshot", "event_graph"), monkeypatch)


@pytest.fixture
def varied_packed(varied, monkeypatch):
    return inline(varied, ("snapshot",), monkeypatch)


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
