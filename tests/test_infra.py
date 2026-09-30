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


def test_the_identities_of_compiled_functions_start_far_from_one_and_differ_between_seedings():
    from numba.core.bytecode import FunctionIdentity

    assert next(FunctionIdentity._unique_ids) > 1 << 20
    first, second = runtime.identities(), runtime.identities()
    assert first != second and next(FunctionIdentity._unique_ids) == second


def test_the_home_of_the_suite_is_named_by_the_checkout_and_the_worker(request):
    import os

    root = Path(__file__).parent.parent
    named = [
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) and Path(plugin.__file__) == root / "conftest.py"
    ]
    home = named[0].home
    suite = Path(os.environ["GRAPHSPEC_SUITE_HOME"])
    assert home(suite, root, "gw0") == home(suite, root / "tests" / "..", "gw0")
    assert home(suite, root, "gw0") != home(suite, root.parent / "elsewhere" / root.name, "gw0")
    assert home(suite, root, "gw0").parent == home(suite, root, "serial").parent
    assert home(suite, root, "gw0").parent.parent == suite / "workers"
    worker = os.environ.get("PYTEST_XDIST_WORKER", "serial")
    assert Path(os.environ["GRAPHSPEC_HOME"]) == home(suite, root, worker)


def test_the_suite_clears_what_it_made_under_its_homes_except_compiled_code(tmp_path):
    from conftest import cleared

    for worker in ("gw0", "serial"):
        for place in ("tmp/pytest", "cache/banks/one", "cache/numba/graphspec_1", "cache/partitions/two"):
            (tmp_path / worker / place).mkdir(parents=True)
        (tmp_path / worker / "cache" / "numba" / "graphspec_1" / "kernel.nbi").write_text("")
        (tmp_path / worker / "stray.txt").write_text("")
    removed = cleared(tmp_path)
    assert len(removed) == 8
    found = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert found == [
        "gw0",
        "gw0/cache",
        "gw0/cache/numba",
        "gw0/cache/numba/graphspec_1",
        "gw0/cache/numba/graphspec_1/kernel.nbi",
        "serial",
        "serial/cache",
        "serial/cache/numba",
        "serial/cache/numba/graphspec_1",
        "serial/cache/numba/graphspec_1/kernel.nbi",
    ]


def test_the_caches_of_a_bank_are_keyed_by_the_code_that_made_them(bank, monkeypatch):
    from graphspec.build import staged
    from tuning.bank import bank as data

    assert data.CODE == code.digest(data.PACKAGES)
    assert staged.CODE == code.digest((Path(staged.__file__).parents[1],))
    before = data.Bank(bank).cache
    monkeypatch.setattr(data, "CODE", "0" * 64)
    assert data.Bank(bank).cache != before and data.Bank(bank).cache.parent == before.parent


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
