import logging
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest

from graphspec.infra.files import read_json, write_json
from graphspec.infra.processes import locks
from graphspec.infra.processes.locks import Lock, LockBusy

WORKER = "PYTEST_XDIST_WORKER" in os.environ
LOG = logging.getLogger(__name__)
SUITE = {"heavy": "heavy-tests-" + os.environ.get("PYTEST_XDIST_WORKER", "serial")}
SETTINGS = dict(entities=40, steps=12, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
SPLITS = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
RECORD = Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) / "durations.json"
SPENT = {}
EARLY = []


def suite_singleton(name, message, original=locks.singleton):
    started = time.perf_counter()
    lock = original(SUITE.get(name, name), message)
    LOG.debug("suite_singleton exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return lock


locks.singleton = suite_singleton


def pytest_configure(config):
    started = time.perf_counter()
    config.addinivalue_line("markers", "gpu: the test is about the GPU and is skipped where none is present")
    config.option.loadscopereorder = False
    LOG.debug("configure exit seconds=%.4f", time.perf_counter() - started)


def unit(nodeid):
    started = time.perf_counter()
    found = nodeid.rsplit("@", 1)[1] if nodeid.rfind("@") > nodeid.rfind("]") else nodeid
    LOG.debug("unit exit unit=%s seconds=%.6f", found, time.perf_counter() - started)
    return found


def ordered(items):
    started = time.perf_counter()
    spent = read_json(RECORD) if RECORD.exists() else {}
    units = [unit(item.nodeid) for item in items]
    counts, first = Counter(units), {}
    for position, name in enumerate(units):
        first.setdefault(name, position)
    rank = {name: (-spent.get(name, 0.0), -counts[name], first[name]) for name in counts}
    items[:] = [item for _, item in sorted(zip(units, items), key=lambda pair: rank[pair[0]])]
    LOG.debug(
        "ordered exit units=%d recorded=%d seconds=%.4f", len(counts), len(spent), time.perf_counter() - started
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    import torch

    started = time.perf_counter()
    absent = pytest.mark.skip(reason="no GPU is present")
    marked = [item for item in items if "gpu" in item.keywords]
    if not torch.cuda.is_available():
        for item in marked:
            item.add_marker(absent)
    if WORKER:
        ordered(items)
    LOG.debug("gpu marks exit marked=%d seconds=%.4f", len(marked), time.perf_counter() - started)


def pytest_runtest_logreport(report):
    started = time.perf_counter()
    name = unit(report.nodeid)
    SPENT[name] = SPENT.get(name, 0.0) + report.duration
    LOG.debug("spent exit unit=%s seconds=%.6f", name, time.perf_counter() - started)


def made_early():
    started = time.perf_counter()
    shared("bank", lambda root: synthetic_bank(root, "task"))
    shared("path", lambda root: synthetic_bank(root, "path"))
    ground = Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "ground" / "fixture"
    shared("ground", lambda made: ground_banks(made, ground))
    shared("varied", varied_bank)
    LOG.debug("made early exit seconds=%.4f", time.perf_counter() - started)


def pytest_sessionstart(session):
    started = time.perf_counter()
    if not WORKER and session.config.pluginmanager.has_plugin("dsession"):
        EARLY.append(threading.Thread(target=made_early, daemon=True))
        EARLY[0].start()
    LOG.debug("session start exit early=%d seconds=%.4f", len(EARLY), time.perf_counter() - started)


def pytest_sessionfinish(session):
    started = time.perf_counter()
    if not WORKER and session.config.pluginmanager.has_plugin("dsession"):
        held = read_json(RECORD) if RECORD.exists() else {}
        write_json(RECORD, {**held, **{name: round(seconds, 4) for name, seconds in SPENT.items()}})
    LOG.debug("durations recorded exit units=%d seconds=%.4f", len(SPENT), time.perf_counter() - started)


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


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    started = time.perf_counter()
    if WORKER:
        LOG.debug("worker ends without finalizing seconds=%.4f", time.perf_counter() - started)
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt":
            import _winapi

            _winapi.TerminateProcess(_winapi.GetCurrentProcess(), 0)
        os._exit(0)
    for thread in EARLY:
        thread.join()
    LOG.debug("early threads joined seconds=%.4f", time.perf_counter() - started)


def synthetic_bank(root, level):
    from graphspec.sources.synthetic import generate, uniform
    from tuning.bank.build import build

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


def varied_bank(root):
    from graphspec.sources import synthetic
    from tuning.bank.build import build

    started = time.perf_counter()
    task = synthetic.generate(root, "varied", 4, 1, synthetic.varied("task"), **SETTINGS)
    build(task, root / "bank", 1, ("train",))
    LOG.debug("varied bank exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def varied():
    return shared("varied", varied_bank) / "bank"


@pytest.fixture(scope="session")
def ground():
    place = Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "ground" / "fixture"
    if not (place / "fixture.json").exists():
        raise FileNotFoundError("the fixture of the ground is absent: %s" % place)
    return place


def ground_banks(root, place):
    import fixture

    from graphspec.sources import synthetic
    from tuning.bank.build import build

    started = time.perf_counter()
    for entry in fixture.configurations(place)[0]:
        source = entry["source"]
        held = root / source["name"]
        if not held.exists():
            held.mkdir()
            shape, settings = fixture.shaped(source)
            task = synthetic.generate(held, "source", source["graphs"], source["seed"], shape, **settings)
            build(task, held / "bank", 1, ("train",))
    LOG.debug("ground banks exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def grounded(ground):
    import fixture

    root = shared("ground", lambda made: ground_banks(made, ground))
    return {entry["source"]["name"]: root / entry["source"]["name"] for entry in fixture.configurations(ground)[0]}


def stated(directory, view):
    from support import configuration, several

    from tuning.bank.bank import Bank

    started = time.perf_counter()
    registry = Bank(directory).registry
    cfg = configuration(registry, dict(view=view), {"time/history": 4, **several(registry)}, 5, SPLITS)
    LOG.debug("stated exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg


def running(directory, view):
    from graphspec.infra.arrays import RESIDENT
    from tuning.bank.bank import Deriver
    from tuning.bank.variants import prepare
    from tuning.batches.packing import Packer

    started = time.perf_counter()
    cfg = stated(directory, view)
    packer = Packer(Deriver(directory, "cpu", RESIDENT), cfg, "cpu", 1)
    prepare(directory, cfg, 1, "cpu")
    LOG.debug("running exit view=%s seconds=%.4f", view, time.perf_counter() - started)
    return cfg, packer


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


@pytest.fixture(autouse=True)
def processes_early(request):
    started = time.perf_counter()
    grouped = request.node.get_closest_marker("xdist_group")
    early = dict(workers="packers", candidates="candidate_pool")
    if WORKER and grouped is not None and grouped.args[0] in early:
        request.getfixturevalue(early[grouped.args[0]])
    LOG.debug("processes early exit seconds=%.4f", time.perf_counter() - started)


@pytest.fixture(scope="session")
def packers(bank):
    held = {view: running(bank, view) for view in ("snapshot", "event_graph")}
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
        loss=dict(number="squared"),
        measures=["mse", "r2"],
    )


class Inline:
    def __init__(self, bank, cfg, device, workers, timing_directory=None):
        if device != "cpu" or workers != 1:
            raise ValueError("the suite packs in its own process, on cpu, as one worker")
        self.bank, self.cfg = bank, cfg

    def stream(self, groups):
        from tuning.batches.batching import recipe_batch
        from tuning.batches.tensors import materialise

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
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
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
    from tuning.searches import space

    executor = ProcessPoolExecutor(2)
    for _ in range(2):
        executor.submit(import_module, "tuning.searches.space")
    space.POOL["executor"] = executor
    yield executor
    executor.shutdown()
    del space.POOL["executor"]
