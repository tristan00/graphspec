import os
from pathlib import Path

if "GRAPHSPEC_SUITE_HOME" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_HOME"] = os.environ["GRAPHSPEC_HOME"]
os.environ["GRAPHSPEC_HOME"] = str(
    Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "workers" / os.environ.get("PYTEST_XDIST_WORKER", "serial")
)
