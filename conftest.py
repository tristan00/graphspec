import os
from pathlib import Path

worker = os.environ.get("PYTEST_XDIST_WORKER")
if worker:
    os.environ["GRAPHSPEC_HOME"] = str(Path(os.environ["GRAPHSPEC_HOME"]) / "workers" / worker)
