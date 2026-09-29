import os
import uuid
from pathlib import Path

if "GRAPHSPEC_SUITE_HOME" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_HOME"] = os.environ["GRAPHSPEC_HOME"]
if "GRAPHSPEC_SUITE_RUN" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_RUN"] = uuid.uuid4().hex
os.environ["GRAPHSPEC_HOME"] = str(
    Path(os.environ["GRAPHSPEC_SUITE_HOME"]) / "workers" / os.environ.get("PYTEST_XDIST_WORKER", "serial")
)
