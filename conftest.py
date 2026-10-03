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


def linked(place):
    started = time.perf_counter()
    compiled = place.parent / "numba"
    compiled.mkdir(parents=True, exist_ok=True)
    link = place / "cache" / "numba"
    if not os.path.lexists(link):
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(compiled), str(link))
        else:
            link.symlink_to(compiled, target_is_directory=True)
    if not os.path.samefile(link, compiled):
        raise FileExistsError("compiled code of %s is held apart from its checkout's: %s" % (place, link))
    LOG.debug("linked exit place=%s seconds=%.4f", place, time.perf_counter() - started)
    return link


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
linked(Path(os.environ["GRAPHSPEC_HOME"]))
