import importlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from graphspec.build import output
from graphspec.infra import code, files, runtime
from tuning import api
from tuning.bank import build

LOG = logging.getLogger(__name__)
HELPER = "from numba import njit\n\n\n@njit\ndef helper(x):\n    return x + 1\n"
KERNEL = (
    "from numba import njit\n\nimport {helper}\nfrom {helper} import helper\n\n\n"
    "@njit\ndef direct(x):\n    return helper(x) * 10\n\n\n"
    "@njit\ndef through(x):\n    return {helper}.helper(x) * 10\n\n\n"
    "@njit\ndef alone(x):\n    return x * 10\n"
)


def rooted(request):
    started = time.perf_counter()
    root = Path(__file__).parent.parent
    named = [
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) and Path(plugin.__file__) == root / "conftest.py"
    ]
    LOG.debug("rooted exit seconds=%.4f", time.perf_counter() - started)
    return named[0]


def lengthened(root, length):
    started = time.perf_counter()
    remaining, parts = length - len(str(root)) - 1, []
    while remaining > 0:
        size = min(40, remaining)
        if remaining - size == 1:
            size -= 1
        parts.append("x" * size)
        remaining -= size + 1
    result = root.joinpath(*parts)
    LOG.debug("lengthened exit length=%d seconds=%.4f", len(str(result)), time.perf_counter() - started)
    return result


@pytest.mark.skipif(os.name != "nt", reason="the limit is of the temporary folder Windows hands to libraries")
def test_a_home_whose_temporary_folder_is_longer_than_windows_hands_to_libraries_stops_the_import_naming_it(
    tmp_path,
):
    import subprocess
    import sys

    started = time.perf_counter()
    longest = lengthened(tmp_path, runtime.TEMPORARY - len("/tmp"))
    assert len(str(runtime.temporary_folder(longest))) == runtime.TEMPORARY
    longer = lengthened(tmp_path, runtime.TEMPORARY - len("/tmp") + 1)
    named = "GRAPHSPEC_HOME %s holds %d characters" % (longer, runtime.TEMPORARY - len("/tmp") + 1)
    with pytest.raises(ValueError) as refused:
        runtime.temporary_folder(longer)
    assert str(refused.value).startswith(named)
    longer.mkdir(parents=True)
    (tmp_path / "importing.py").write_text("import graphspec\n")
    root = Path(__file__).parent.parent
    done = subprocess.run(
        [sys.executable, str(tmp_path / "importing.py")],
        env=dict(os.environ, GRAPHSPEC_HOME=str(longer), PYTHONPATH=str(root)),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 1 and "ValueError: " + named in done.stderr
    LOG.info("long home test exit seconds=%.4f", time.perf_counter() - started)


def test_the_identities_of_compiled_functions_start_far_from_one_and_differ_between_seedings():
    from numba.core.bytecode import FunctionIdentity

    assert next(FunctionIdentity._unique_ids) > 1 << 20
    first, second = runtime.identities(), runtime.identities()
    assert first != second and next(FunctionIdentity._unique_ids) == second


def test_a_home_of_the_suite_is_named_by_its_run_and_worker_and_shares_the_compiled_code_of_the_checkout(request):
    import sys

    import numba

    started = time.perf_counter()
    root = Path(__file__).parent.parent
    held = rooted(request)
    suite, run = Path(os.environ["GRAPHSPEC_SUITE_HOME"]), os.environ["GRAPHSPEC_SUITE_RUN"]
    assert held.lasting(suite, root) == held.lasting(suite, root / "tests" / "..")
    assert held.lasting(suite, root) != held.lasting(suite, root.parent / "elsewhere" / root.name)
    assert held.lasting(suite, root).parent == suite / "workers"
    assert held.home(suite, run, "gw0").parent == held.home(suite, run, "serial").parent == suite / "workers" / run
    worker = os.environ.get("PYTEST_XDIST_WORKER", "serial")
    assert Path(os.environ["GRAPHSPEC_HOME"]) == held.home(suite, run, worker)
    assert Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) == held.lasting(suite, root)
    assert os.path.samefile(numba.config.CACHE_DIR, held.lasting(suite, root) / "numba")
    assert sys.pycache_prefix == os.environ["PYTHONPYCACHEPREFIX"] == str(held.lasting(suite, root) / "bytecode")
    assert "PYTHONDONTWRITEBYTECODE" not in os.environ
    LOG.info("home exit seconds=%.4f", time.perf_counter() - started)


def allowed(processes):
    import psutil

    started = time.perf_counter()
    found = {}
    for process in processes:
        try:
            found[process.pid] = len(process.cpu_affinity())
        except psutil.NoSuchProcess:
            continue
    LOG.debug("allowed exit processes=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def test_every_process_of_the_run_and_every_child_of_a_test_is_allowed_at_most_the_cores_of_the_suite(request):
    import subprocess
    import sys

    import psutil

    started = time.perf_counter()
    held = rooted(request)
    controller = psutil.Process(int(os.environ["GRAPHSPEC_SUITE_CONTROLLER"]))
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        found = allowed([controller, *controller.children(recursive=True)])
    finally:
        child.communicate(b"", timeout=30)
    assert held.CORES == 2 and {os.getpid(), child.pid} <= set(found)
    assert {pid: cores for pid, cores in found.items() if not 1 <= cores <= held.CORES} == {}
    LOG.info("cores test exit processes=%d seconds=%.4f", len(found), time.perf_counter() - started)


def test_a_run_of_the_suite_removes_its_own_homes_and_leaves_other_runs_and_the_compiled_code(request, tmp_path):
    started = time.perf_counter()
    compiled = tmp_path / "checkout" / "numba"
    for run in ("this", "other"):
        for worker in ("gw0", "serial"):
            rooted(request).linked(tmp_path / run / worker, compiled)
            for place in ("tmp/pytest", "cache/banks/one", "cache/partitions/two"):
                (tmp_path / run / worker / place).mkdir(parents=True)
            (tmp_path / run / worker / "stray.txt").write_text("")
            (tmp_path / run / (worker + ".alive")).write_text("")
    (compiled / "graphspec_1").mkdir()
    (compiled / "graphspec_1" / "kernel.nbi").write_text("")
    removed = rooted(request).cleared(tmp_path / "this")
    assert len(removed) == 6
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkout", "other"]
    for worker in ("gw0", "serial"):
        assert sorted(path.name for path in (tmp_path / "other" / worker).iterdir()) == ["cache", "stray.txt", "tmp"]
        assert os.path.samefile(tmp_path / "other" / worker / "cache" / "numba", compiled)
    assert (compiled / "graphspec_1" / "kernel.nbi").exists()
    with pytest.raises(FileExistsError):
        rooted(request).linked(tmp_path / "other" / "gw0", compiled)
    LOG.info("cleared test exit seconds=%.4f", time.perf_counter() - started)


def test_a_run_removes_at_its_start_the_homes_of_runs_whose_processes_all_ended_and_never_those_of_a_run_going(
    request, tmp_path
):
    import sys

    from graphspec.infra.processes.locks import Lock, LockBusy

    started = time.perf_counter()
    held, workers = rooted(request), tmp_path / "workers"
    mirror = Path(sys.pycache_prefix) / workers.relative_to(workers.anchor)
    for run in ("killed", "going", "orphaned", "this"):
        (mirror / run / "serial" / "tmp").mkdir(parents=True)
        (mirror / run / "serial" / "tmp" / "probe.cpython-314.pyc").write_bytes(b"")
    for run in ("killed", "going", "orphaned"):
        for worker in ("serial", "gw0"):
            (workers / run / worker / "tmp").mkdir(parents=True)
            (workers / run / (worker + ".alive")).write_text("")
        (tmp_path / "tmp" / "suite" / run / "bank").mkdir(parents=True)
    for place in ("gw3/tmp", "starting/serial/tmp", "graphspec-0123456789ab/numba"):
        (workers / place).mkdir(parents=True)
    going = [Lock(workers / "going" / "serial.alive").__enter__(), Lock(workers / "orphaned" / "gw0.alive").__enter__()]
    try:
        alive, removed = held.begun(tmp_path, "this", "serial", True)
        assert removed == [workers / "killed"]
        assert sorted(path.name for path in (tmp_path / "tmp" / "suite").iterdir()) == ["going", "orphaned"]
        kept = ["going", "graphspec-0123456789ab", "gw3", "orphaned", "runs.lock", "starting", "this"]
        assert sorted(path.name for path in workers.iterdir()) == kept
        assert sorted(path.name for path in mirror.iterdir()) == ["going", "orphaned", "this"]
        with pytest.raises(LockBusy):
            Lock(workers / "this" / "serial.alive").__enter__()
        worker, none = held.begun(tmp_path, "this", "gw0", False)
        assert none == [] and sorted(path.name for path in (workers / "this").iterdir()) == ["gw0.alive", "serial.alive"]
        worker.__exit__(None, None, None)
        assert held.finished(tmp_path, "this", alive) == workers / "this"
        assert sorted(path.name for path in workers.iterdir()) == [name for name in kept if name != "this"]
        assert sorted(path.name for path in mirror.iterdir()) == ["going", "orphaned"]
    finally:
        for lock in going:
            lock.__exit__(None, None, None)
    LOG.info("begun test exit seconds=%.4f", time.perf_counter() - started)


def keyed(module):
    from numba.core import caching, types

    started = time.perf_counter()
    codegen = module.direct.targetctx.codegen()
    result = {
        name: caching.Cache._index_key(SimpleNamespace(_py_func=getattr(module, name).py_func), (types.int64,), codegen)
        for name in ("direct", "through", "alone")
    }
    LOG.debug("keyed exit seconds=%.4f", time.perf_counter() - started)
    return result


def written(index, writer):
    started = time.perf_counter()
    for entry in range(3):
        index.save(("signature", writer, entry), ("code", writer, entry))
    LOG.debug("written exit writer=%d seconds=%.4f", writer, time.perf_counter() - started)
    return writer


def test_compiled_code_is_keyed_by_the_files_of_every_compiled_function_it_reaches(tmp_path, monkeypatch):
    import sys

    started = time.perf_counter()
    helper, kernel = "helper_%s" % tmp_path.name, "kernel_%s" % tmp_path.name
    (tmp_path / (helper + ".py")).write_text(HELPER)
    (tmp_path / (kernel + ".py")).write_text(KERNEL.format(helper=helper))
    monkeypatch.syspath_prepend(str(tmp_path))
    held = importlib.import_module(kernel)
    before = keyed(held)
    files = {name: [Path(entry[0]).name for entry in key[-1]] for name, key in before.items()}
    assert files == {
        "direct": sorted([helper + ".py", kernel + ".py"]),
        "through": sorted([helper + ".py", kernel + ".py"]),
        "alone": [kernel + ".py"],
    }
    (tmp_path / (helper + ".py")).write_text(HELPER + "\nVALUE = 1\n")
    after = keyed(held)
    assert after["direct"] != before["direct"] and after["through"] != before["through"]
    assert after["alone"] == before["alone"]
    for name in (helper, kernel):
        del sys.modules[name]
    LOG.info("keyed test exit seconds=%.4f", time.perf_counter() - started)


def test_compiled_code_saved_at_once_keeps_every_signature_with_its_own_code(tmp_path):
    from numba.core import caching

    started = time.perf_counter()
    writers = [caching.IndexDataCacheFile(str(tmp_path), "kernel", (1.0, 10)) for _ in range(4)]
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(written, writers, range(4)))
    entries = [(writer, entry) for writer in range(4) for entry in range(3)]
    found = {(writer, entry): writers[0].load(("signature", writer, entry)) for writer, entry in entries}
    assert found == {(writer, entry): ("code", writer, entry) for writer, entry in entries}
    LOG.info("saved test exit seconds=%.4f", time.perf_counter() - started)


def test_compiled_code_is_saved_only_once_the_lock_of_its_index_is_free(tmp_path):
    from numba.core import caching

    from graphspec.infra.processes.locks import Lock

    started = time.perf_counter()
    index = caching.IndexDataCacheFile(str(tmp_path), "kernel", (1.0, 10))
    lock = Lock(tmp_path / "kernel.nbi.lock").__enter__()
    with ThreadPoolExecutor(1) as pool:
        saving = pool.submit(index.save, "signature", "code")
        time.sleep(0.05)
        waited = not saving.done() and not (tmp_path / "kernel.nbi").exists()
        lock.__exit__(None, None, None)
        saving.result()
    assert waited and index.load("signature") == "code"
    LOG.info("lock test exit seconds=%.4f", time.perf_counter() - started)


def test_a_lock_that_cannot_be_taken_for_a_reason_other_than_a_holder_stops_at_once_naming_the_file_and_the_error(
    tmp_path, monkeypatch
):
    import ctypes
    import errno

    from graphspec.infra.processes import locks

    started = time.perf_counter()
    path, folder = tmp_path / "denied.lock", tmp_path / "folder.lock"

    def denied(*arguments):
        clock = time.perf_counter()
        if os.name == "nt":
            ctypes.set_last_error(5)
            LOG.debug("denied exit seconds=%.6f", time.perf_counter() - clock)
            return 0
        LOG.debug("denied exit seconds=%.6f", time.perf_counter() - clock)
        raise PermissionError(errno.EACCES, "Permission denied")

    if os.name == "nt":
        monkeypatch.setattr(locks.KERNEL, "LockFile", denied)
    else:
        monkeypatch.setattr(locks.fcntl, "flock", denied)
    with pytest.raises(PermissionError) as refused:
        locks.Lock(path).__enter__()
    assert refused.value.filename == str(path)
    with pytest.raises(PermissionError) as refused:
        with runtime.held(path):
            pass
    assert refused.value.filename == str(path)
    folder.mkdir()
    with pytest.raises(OSError) as refused:
        with runtime.held(folder):
            pass
    assert refused.value.filename == str(folder) and not isinstance(refused.value, locks.LockBusy)
    LOG.info("denied lock test exit seconds=%.4f", time.perf_counter() - started)


def test_a_lock_names_its_holder_and_a_wait_stops_naming_the_lock_the_holder_and_the_seconds_once_the_holder_ended(
    tmp_path,
):
    import re
    import subprocess
    import sys

    from graphspec.infra.processes.locks import Lock, holder

    started = time.perf_counter()
    ended = subprocess.Popen([sys.executable, "-V"], stdout=subprocess.DEVNULL)
    ended.wait()
    path = tmp_path / "kept.lock"
    lock = Lock(path).__enter__()
    try:
        assert holder(path) == os.getpid()
        path.write_bytes(b"%d" % ended.pid)
        named = "the lock %s is held by the process %d, which has ended; waited " % (path, ended.pid)
        with pytest.raises(RuntimeError, match=re.escape(named) + r"0\.\d{3} seconds$"):
            with runtime.held(path):
                pass
    finally:
        lock.__exit__(None, None, None)
    assert path.read_bytes() == b""
    with runtime.held(path):
        assert holder(path) == os.getpid()
    LOG.info("ended holder test exit seconds=%.4f", time.perf_counter() - started)


def test_json_write_failure_preserves_prior_record_and_removes_temporary(tmp_path):
    path = tmp_path / "record.json"
    files.write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        files.write_json(path, {"value": float("nan")})
    assert files.read_json(path) == {"value": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_json_writers_publish_complete_records(tmp_path):
    path = tmp_path / "record.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: files.write_json(path, {"value": value}), range(8)))
    assert files.read_json(path)["value"] in range(8)
    assert list(tmp_path.iterdir()) == [path]


def test_json_reader_allows_replacement_while_holding_prior_record(tmp_path):
    path = tmp_path / "record.json"
    files.write_json(path, {"value": 1})
    with files.open_read(path) as stream:
        files.write_json(path, {"value": 2})
        assert '"value": 1' in stream.read()
    assert files.read_json(path) == {"value": 2}


def test_array_cache_failure_does_not_leave_partial_arrays(tmp_path):
    folder = tmp_path / "variant"
    cache = files.ArrayStore()
    with pytest.raises(ValueError):
        cache.open(folder, lambda: {"old": np.ones(1), "invalid": np.array([object()])})
    assert not list(tmp_path.iterdir())
    arrays = cache.open(folder, lambda: {"new": np.ones(2)})
    assert set(arrays) == {"new"}


def test_array_cache_publish_race_preserves_completed_winner(tmp_path):
    folder = tmp_path / "variant"

    def build_loser():
        files.save_arrays(folder, {"winner": np.array([7])})
        return {"loser": np.array([3])}

    arrays = files.ArrayStore().open(folder, build_loser)
    assert set(arrays) == {"winner"}
    assert arrays["winner"].tolist() == [7]
    assert list(tmp_path.iterdir()) == [folder]


def test_array_cache_hit_reuses_validated_location(tmp_path, monkeypatch):
    folder = tmp_path / "variant"
    store = files.ArrayStore()
    first = store.open(folder, lambda: {"values": np.arange(3)})

    def unexpected(*args):
        raise AssertionError("cached arrays repeated path validation or construction")

    monkeypatch.setattr(files, "output_path", unexpected)
    assert store.open(folder, unexpected) is first
    assert first["values"].tolist() == [0, 1, 2]


def test_explicit_runtime_home_does_not_read_invalid_settings(tmp_path, monkeypatch):
    settings = tmp_path / ".graphspec"
    settings.mkdir()
    (settings / "settings.json").write_text("invalid", encoding="utf-8")
    destination = tmp_path / "runtime"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("GRAPHSPEC_HOME", str(destination))
    assert runtime.home() == destination


def test_stage_rejects_schema_changes_even_when_table_is_empty(tmp_path):
    key = ("graph", None, None)
    with output.Stage(tmp_path / "stage") as stage:
        stage.write({key: pa.table({"id": pa.array([], pa.int32())})})
        with pytest.raises(ValueError, match="inconsistent table schema"):
            stage.write({key: pa.table({"id": pa.array([], pa.float32())})})


def test_stage_closes_every_writer_after_close_failure(tmp_path, monkeypatch):
    closed = []

    class Writer:
        def __init__(self, path, schema, **kwargs):
            self.path, self.schema = path, schema

        def close(self):
            closed.append(self.path.name)
            if self.path.name == "00001.parquet":
                raise OSError("close failed")

    monkeypatch.setattr(output.pq, "ParquetWriter", Writer)
    stage = output.Stage(tmp_path / "stage")
    table = pa.table({"id": pa.array([], pa.int32())})
    stage.write({("first", None, None): table, ("second", None, None): table})
    with pytest.raises(OSError, match="close failed"):
        stage.close()
    assert set(closed) == {"00000.parquet", "00001.parquet"}
    assert not (stage.directory / "stage.json").exists()


def test_source_snapshot_rejects_package_path_collisions(tmp_path):
    packages = [tmp_path / side / "same" for side in ("left", "right")]
    for package in packages:
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping snapshot paths"):
        code.snapshot(packages, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_bank_build_closes_source_when_validation_fails(tmp_path, monkeypatch):
    closed = []

    def invalid(*args):
        raise ValueError("invalid schema")

    source = SimpleNamespace(close=lambda: closed.append(True), graphs=invalid)
    task = tmp_path / "task.json"
    files.write_json(task, {})
    monkeypatch.setattr(build, "open_source", lambda *args: source)
    with pytest.raises(ValueError, match="invalid schema"):
        build.build.__wrapped__(task, tmp_path / "bank", 1, ("train",))
    assert closed == [True]


@pytest.mark.parametrize(
    "changes,options,message",
    [
        ({"workers": 0}, {}, "the statement's workers is 0"),
        ({"seconds": -1}, {}, "the statement's seconds is -1"),
        ({"seconds": float("inf")}, {}, "the statement's seconds is inf"),
        ({"batch_sequences": 0}, {}, "the statement's batch_sequences is 0"),
    ],
)
def test_fit_rejects_invalid_resource_inputs_before_opening_bank(
    tmp_path, statement, changes, options, message
):
    with pytest.raises(ValueError, match=message):
        api.fit.__wrapped__(
            tmp_path / "missing-bank", {}, tmp_path / "fit", dict(statement, **changes), **options
        )


def test_dense_algebra_gives_the_same_bits_at_any_thread_count():
    import numpy as np
    import torch

    from graphspec.infra import compute

    random = np.random.default_rng(0)
    links = random.random((300, 300)) < 0.05
    links = links | links.T
    np.fill_diagonal(links, False)
    scale = 1 / np.sqrt(links.sum(1) + 1.0)
    transition = scale[:, None] * links * scale[None, :]
    found = []
    before = torch.get_num_threads()
    for threads in (1, 6, 24):
        torch.set_num_threads(threads)
        values, vectors = compute.eigh(transition, "cpu")
        found.append(
            (
                compute.inverse(np.eye(300) - 0.6 * transition, "cpu"),
                values,
                vectors,
                compute.compose(vectors, np.exp(-values), "cpu"),
            )
        )
        assert torch.get_num_threads() == threads
    torch.set_num_threads(before)
    for other in found[1:]:
        assert all(np.array_equal(first, second) for first, second in zip(found[0], other))
