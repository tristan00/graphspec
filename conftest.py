import hashlib
import logging
import os
import time
import uuid
from pathlib import Path

LOG = logging.getLogger(__name__)


def lasting(suite, checkout):
    started = time.perf_counter()
    checkout = Path(os.path.abspath(checkout))
    named = "%s-%s" % (checkout.name, hashlib.sha256(str(checkout).casefold().encode()).hexdigest()[:12])
    result = Path(suite) / "workers" / named
    LOG.debug("lasting exit checkout=%s seconds=%.4f", checkout.name, time.perf_counter() - started)
    return result


def home(suite, run, worker):
    started = time.perf_counter()
    result = Path(suite) / "workers" / run / worker
    LOG.debug("home exit worker=%s seconds=%.4f", worker, time.perf_counter() - started)
    return result


def linked(place, compiled):
    started = time.perf_counter()
    compiled.mkdir(parents=True, exist_ok=True)
    link = place / "cache" / "numba"
    link.parent.mkdir(parents=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(compiled), str(link))
    else:
        link.symlink_to(compiled, target_is_directory=True)
    LOG.debug("linked exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return link


if "GRAPHSPEC_SUITE_HOME" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_HOME"] = os.environ["GRAPHSPEC_HOME"]
if "GRAPHSPEC_SUITE_RUN" not in os.environ:
    os.environ["GRAPHSPEC_SUITE_RUN"] = uuid.uuid4().hex[:12]
os.environ["GRAPHSPEC_SUITE_CHECKOUT"] = str(lasting(os.environ["GRAPHSPEC_SUITE_HOME"], Path(__file__).parent))
os.environ["GRAPHSPEC_HOME"] = str(
    home(
        os.environ["GRAPHSPEC_SUITE_HOME"],
        os.environ["GRAPHSPEC_SUITE_RUN"],
        os.environ.get("PYTEST_XDIST_WORKER", "serial"),
    )
)
linked(Path(os.environ["GRAPHSPEC_HOME"]), Path(os.environ["GRAPHSPEC_SUITE_CHECKOUT"]) / "numba")
