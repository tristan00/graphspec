import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphspec.infra import runtime


@pytest.mark.parametrize("value", [None, "", "relative/runtime"])
def test_invalid_runtime_home_fails_before_profile_access_or_writes(monkeypatch, value):
    def unexpected_access(*args, **kwargs):
        pytest.fail("invalid runtime configuration accessed the filesystem")

    with monkeypatch.context() as patch:
        if value is None:
            patch.delenv("GRAPHSPEC_HOME", raising=False)
        else:
            patch.setenv("GRAPHSPEC_HOME", value)
        patch.setattr(Path, "home", unexpected_access)
        patch.setattr(Path, "expanduser", unexpected_access)
        patch.setattr(Path, "mkdir", unexpected_access)
        patch.setattr(Path, "open", unexpected_access)
        with pytest.raises(ValueError, match="GRAPHSPEC_HOME"):
            runtime.initialize_storage()


def test_runtime_output_rejects_the_source_checkout(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("GRAPHSPEC_HOME", str(tmp_path))
        checkout = Path(runtime.__file__).resolve().parents[2]
        with pytest.raises(ValueError, match="outside the code directory"):
            runtime.output_path(checkout / "artifacts" / "trial")


@pytest.mark.skipif(os.name != "nt", reason="Windows volume boundary")
def test_runtime_output_rejects_another_windows_volume(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("GRAPHSPEC_HOME", str(tmp_path))
        other = "C:/" if tmp_path.drive.casefold() != "c:" else "D:/"
        with pytest.raises(ValueError, match="GRAPHSPEC_HOME volume"):
            runtime.output_path(Path(other) / "graphspec-storage-boundary-probe")


@pytest.mark.skipif(os.name != "nt", reason="the final path name is a Windows call")
def test_runtime_output_survives_a_path_that_vanishes_while_it_is_resolved(tmp_path, monkeypatch):
    import ntpath

    calls, final = [], ntpath._getfinalpathname

    def vanishing(path):
        calls.append(path)
        if len(calls) % 2 == 0:
            raise OSError(2, "gone", path, 2)
        return final(path)

    checkout = Path(runtime.__file__).parents[2]
    with monkeypatch.context() as patch:
        patch.setenv("GRAPHSPEC_HOME", str(tmp_path))
        patch.setattr(ntpath, "_getfinalpathname", vanishing)
        for index in range(4):
            found = runtime.output_path(tmp_path / ("run-%d" % index))
            assert found == tmp_path / ("run-%d" % index)
            assert not str(found).startswith("\\\\?\\") and found.drive == tmp_path.drive
            assert not str(runtime.home()).startswith("\\\\?\\")
            with pytest.raises(ValueError, match="outside the code directory"):
                runtime.output_path(checkout / "artifacts" / ("trial-%d" % index))


@pytest.mark.parametrize("prefixed", [False, True])
def test_initialization_redirects_cached_temporary_and_dependency_storage(tmp_path, monkeypatch, prefixed):
    root = tmp_path / "runtime"
    stale = tmp_path / "stale"
    numba = SimpleNamespace(config=SimpleNamespace(CACHE_DIR=str(stale)))
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        patch.setenv("GRAPHSPEC_HOME", str(root))
        patch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
        for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "NUMBA_CACHE_DIR", "PIP_CACHE_DIR"):
            patch.setenv(key, str(stale))
        patch.setattr(tempfile, "tempdir", str(stale))
        patch.setitem(sys.modules, "numba", numba)
        patch.setattr(sys, "pycache_prefix", str(tmp_path / "bytecode") if prefixed else None)
        patch.setattr(sys, "dont_write_bytecode", False)

        runtime.initialize_storage()

        temporary = root / "tmp"
        assert temporary.is_dir()
        assert Path(tempfile.gettempdir()) == temporary
        assert all(Path(os.environ[key]) == temporary for key in ("TEMP", "TMP", "TMPDIR"))
        for key in ("TORCH_HOME", "NUMBA_CACHE_DIR", "PIP_CACHE_DIR", "CUDA_CACHE_PATH"):
            assert Path(os.environ[key]).is_relative_to(root / "cache")
        assert numba.config.CACHE_DIR == os.environ["NUMBA_CACHE_DIR"]
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == (None if prefixed else "1")
        assert sys.dont_write_bytecode == (not prefixed)
        assert not stale.exists()


def test_initialization_refuses_a_prefix_of_bytecode_inside_the_checkout(tmp_path, monkeypatch):
    checkout = Path(runtime.__file__).resolve().parents[2]
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        patch.setenv("GRAPHSPEC_HOME", str(tmp_path / "runtime"))
        patch.setattr(tempfile, "tempdir", tempfile.tempdir)
        patch.setitem(sys.modules, "numba", SimpleNamespace(config=SimpleNamespace(CACHE_DIR=None)))
        patch.setattr(sys, "dont_write_bytecode", True)
        patch.setattr(sys, "pycache_prefix", str(checkout / "bytecode"))
        with pytest.raises(ValueError, match="outside the code directory"):
            runtime.initialize_storage()


def test_child_initializes_storage_before_creating_temporary_files(tmp_path):
    root = tmp_path / "child-runtime"
    script = """
import json
import os
import sys
import tempfile

tempfile.tempdir = os.environ['TEMP']
checked = []
class ImportAudit:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'torch', 'numba', 'numpy', 'pyarrow'}:
            if tempfile.tempdir != os.path.join(os.environ['GRAPHSPEC_HOME'], 'tmp'):
                raise RuntimeError('dependency imported before storage initialization')
            checked.append(fullname)
sys.meta_path.insert(0, ImportAudit())
import graphspec
with tempfile.NamedTemporaryFile() as stream:
    print(json.dumps({
        'temporary': stream.name,
        'torch': os.environ['TORCH_HOME'],
        'numba': os.environ['NUMBA_CACHE_DIR'],
        'bytecode_disabled': sys.dont_write_bytecode,
        'told': os.environ.get('PYTHONDONTWRITEBYTECODE'),
        'checked_imports': checked,
    }))
"""
    environment = dict(os.environ, GRAPHSPEC_HOME=str(root), PYTHONPYCACHEPREFIX=sys.pycache_prefix)
    environment.update({key: str(tmp_path / "stale") for key in ("TEMP", "TMP", "TMPDIR")})
    child = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    result = json.loads(child.stdout)
    assert Path(result["temporary"]).parent == root / "tmp"
    assert not Path(result["temporary"]).exists()
    assert Path(result["torch"]).is_relative_to(root / "cache")
    assert Path(result["numba"]).is_relative_to(root / "cache")
    assert not result["bytecode_disabled"] and result["told"] is None
    assert result["checked_imports"]
