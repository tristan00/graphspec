import os
import subprocess
import sys
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows kernel mutexes")


def child(code, name, directory):
    return subprocess.run(
        [sys.executable, "-B", "-c", code, name],
        env=dict(os.environ, GRAPHSPEC_HOME=str(directory), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=True,
    )


def test_singleton_rejects_local_reentry_without_profile_access(monkeypatch):
    from graphspec.infra.runtime import Path, singleton

    def forbidden():
        raise AssertionError("Windows singleton must not access the user profile")

    monkeypatch.setattr(Path, "home", forbidden)
    name = uuid.uuid4().hex
    with singleton(name, "already held"):
        with pytest.raises(RuntimeError, match="already held"):
            with singleton(name, "already held"):
                pytest.fail("singleton allowed reentry")
    with singleton(name, "already held"):
        pass


def test_singleton_excludes_other_home_and_releases(tmp_path, monkeypatch):
    from graphspec.infra.runtime import singleton

    name = uuid.uuid4().hex
    monkeypatch.setenv("GRAPHSPEC_HOME", str(tmp_path / "first"))
    code = """
import sys
from graphspec.infra.runtime import singleton
try:
    with singleton(sys.argv[1], "busy"):
        print("acquired")
except RuntimeError:
    print("busy")
"""
    with singleton(name, "busy"):
        assert child(code, name, tmp_path / "second").stdout.strip() == "busy"
    assert child(code, name, tmp_path / "second").stdout.strip() == "acquired"


def test_abandoned_mutex_is_owned_and_released(tmp_path, monkeypatch):
    from graphspec.infra import windows_mutex as win

    name = uuid.uuid4().hex
    mutex = win.Mutex(name)
    keeper = win.checked(win.create_mutex(None, mutex.name, 0, win.MUTEX_ACCESS))
    waited, released = [], []
    original_wait, original_release = win.wait_mutex, win.release_mutex

    def wait(handle, milliseconds):
        result = original_wait(handle, milliseconds)
        waited.append(result)
        return result

    def release(handle):
        released.append(handle)
        return original_release(handle)

    code = """
import os
import sys
from graphspec.infra.windows_mutex import Mutex
owner = Mutex(sys.argv[1])
owner.__enter__()
os._exit(0)
"""
    try:
        child(code, name, tmp_path)
        monkeypatch.setattr(win, "wait_mutex", wait)
        monkeypatch.setattr(win, "release_mutex", release)
        with mutex:
            assert mutex.owned
        assert waited == [win.WAIT_ABANDONED]
        assert len(released) == 1
        assert mutex.handle is None
        assert mutex.name not in win.LOCAL_OWNERS
    finally:
        win.checked(win.close_handle(keeper))


@pytest.mark.parametrize("phase", ["create", "busy", "wait"])
def test_failed_acquisition_closes_handle_and_clears_local_claim(monkeypatch, phase):
    from graphspec.infra import windows_mutex as win

    mutex = win.Mutex(uuid.uuid4().hex)
    closed, released = [], []
    monkeypatch.setattr(win, "create_mutex", lambda *args: 0 if phase == "create" else 123)
    monkeypatch.setattr(
        win, "wait_mutex", lambda *args: win.WAIT_TIMEOUT if phase == "busy" else win.WAIT_FAILED
    )
    monkeypatch.setattr(win, "close_handle", lambda handle: closed.append(handle) or True)
    monkeypatch.setattr(win, "release_mutex", lambda handle: released.append(handle) or True)
    with pytest.raises(OSError):
        mutex.__enter__()
    assert closed == ([] if phase == "create" else [123])
    assert not released
    assert mutex.handle is None
    assert mutex.name not in win.LOCAL_OWNERS
