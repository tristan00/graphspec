import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from graphspec.infra.processes.processes import WorkerProcess

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process ownership integration")

PROBE = """import json, os, socket, subprocess, sys
port = int(sys.argv[1])
if len(sys.argv) == 2:
    subprocess.Popen([sys.executable, '-B', __file__, str(port), 'child'])
with socket.create_connection(('127.0.0.1', port)) as connection:
    connection.sendall((json.dumps(os.getpid()) + '\\n').encode())
    connection.recv(1)
"""


@pytest.fixture(autouse=True)
def probe_imports(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), str(Path(__file__).resolve().parents[1])))
    )


def accept_processes(listener):
    connections, processes = [], []
    try:
        for _ in range(2):
            connection, _ = listener.accept()
            connections.append(connection)
            connection.settimeout(5)
            with connection.makefile("r") as stream:
                processes.append(psutil.Process(json.loads(stream.readline())))
        return connections, processes
    except BaseException:
        for connection in connections:
            connection.close()
        raise


def test_windows_timeout_joins_worker_and_descendants(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    connections = []
    with socket.socket() as listener, (tmp_path / "worker.log").open("w") as stream:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        try:
            with WorkerProcess("probe", [str(listener.getsockname()[1])], stream) as worker:
                connections, processes = accept_processes(listener)
                descendants = psutil.Process(worker.pid).children(recursive=True)
                assert all(process.pid in worker.group.members() for process in processes)
                with pytest.raises(subprocess.TimeoutExpired):
                    worker.wait(timeout=0)
            for process in [*processes, *descendants]:
                process.wait(timeout=0)
        finally:
            for connection in connections:
                connection.close()


def test_windows_failed_job_attachment_exits_before_target_starts(tmp_path, monkeypatch):
    script = tmp_path / "probe.py"
    marker = tmp_path / "started"
    script.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n", encoding="utf-8")
    with (tmp_path / "worker.log").open("w") as stream:
        worker = WorkerProcess("probe", [], stream)

        def reject(process):
            raise OSError("attachment rejected")

        monkeypatch.setattr(worker.group, "attach", reject)
        with pytest.raises(OSError, match="attachment rejected"):
            with worker:
                pytest.fail("worker must not start")
        assert worker.process.poll() is not None
        assert worker.closed
        assert not marker.exists()


def test_windows_owner_death_terminates_its_job(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    owner_script = tmp_path / "owner.py"
    owner_script.write_text(
        "import os, socket, sys\n"
        "from graphspec.infra.processes.processes import WorkerProcess\n"
        "with WorkerProcess('probe', sys.argv[1:], sys.stdout):\n"
        "    with socket.socket() as blocker:\n"
        "        blocker.bind(('127.0.0.1', 0))\n"
        "        blocker.listen()\n"
        "        blocker.accept()\n",
        encoding="utf-8",
    )
    connections = []
    processes = []
    with socket.socket() as listener, (tmp_path / "owner.log").open("w") as stream:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        owner = subprocess.Popen(
            [sys.executable, "-B", str(owner_script), str(listener.getsockname()[1])],
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=0x08000000,
        )
        try:
            connections, processes = accept_processes(listener)
            worker_parent = processes[0].parent()
            while str(owner_script) not in worker_parent.cmdline():
                worker_parent = worker_parent.parent()
            worker_parent.kill()
            worker_parent.wait(timeout=5)
            owner.wait(timeout=5)
            for process in processes:
                process.wait(timeout=5)
            for process in processes:
                process.wait(timeout=0)
        finally:
            for connection in connections:
                connection.close()
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)
            for process in processes:
                if process.is_running():
                    process.kill()
                    process.wait(timeout=5)
