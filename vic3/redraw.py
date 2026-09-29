import json
import logging
import os
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)
SCRIPT = """
import json, sys
import psycopg
from advisors import RandomAdvisor
from loop import run
wanted = json.load(sys.stdin)
options = "-c default_transaction_read_only=on -c statement_timeout=%d" % wanted["timeout"]
with psycopg.connect(wanted["address"], options=options) as connection:
    found = [
        run.redraw(connection, wanted["run"], seq, wanted["count"], seed)[1]
        for seq, seed in wanted["moments"]
    ]
json.dump(dict(advisor=RandomAdvisor.name, blocks=found), sys.stdout)
"""


def interpreter(project):
    folder = Path(project) / ".venv"
    return folder / "Scripts" / "python.exe" if os.name == "nt" else folder / "bin" / "python"


def drawn(project, address, run, moments, count, timeout):
    started = time.perf_counter()
    done = subprocess.run(
        [str(interpreter(project)), "-B", "-c", SCRIPT],
        input=json.dumps(dict(address=address, run=run, moments=moments, count=count, timeout=timeout)),
        capture_output=True,
        text=True,
        cwd=project,
        env=dict(os.environ, PYTHONPATH=str(project), PYTHONDONTWRITEBYTECODE="1"),
    )
    if done.returncode:
        raise RuntimeError("blocks could not be redrawn for run %d: %s" % (run, done.stderr[-2000:]))
    found = json.loads(done.stdout)
    LOG.info(
        "drawn exit run=%d seconds=%.2f moments=%d blocks=%d",
        run,
        time.perf_counter() - started,
        len(moments),
        count,
    )
    return found["advisor"], [
        [[(name, params) for name, params in block] for block in blocks] for blocks in found["blocks"]
    ]
