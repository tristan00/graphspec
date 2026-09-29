import hashlib
import logging
import os
import time
import uuid
from pathlib import Path

LOG = logging.getLogger(__name__)


def home(suite, checkout, worker):
    started = time.perf_counter()
    checkout = Path(os.path.abspath(checkout))
    named = "%s-%s" % (checkout.name, hashlib.sha256(str(checkout).casefold().encode()).hexdigest()[:12])
    result = Path(suite) / "workers" / named / worker
    LOG.debug("home exit worker=%s seconds=%.4f", worker, time.perf_counter() - started)
    return result


if "GRAPHSPEC_SUITE_HOME" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_HOME"] = os.environ["GRAPHSPEC_HOME"]
if "GRAPHSPEC_SUITE_RUN" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_RUN"] = uuid.uuid4().hex
os.environ["GRAPHSPEC_HOME"] = str(
    home(
        os.environ["GRAPHSPEC_SUITE_HOME"],
        Path(__file__).parent,
        os.environ.get("PYTEST_XDIST_WORKER", "serial"),
    )
)
