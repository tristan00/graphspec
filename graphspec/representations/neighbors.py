import logging
import time

from graphspec.infra.timing import timed

from .groups import neighbors
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


@timed("neighbors.carries")
def carries(registry, spec):
    return {}


@timed("neighbors.horizon")
def horizon(spec):
    return 0, 0


@timed("neighbors.piece")
def piece(shown, state):
    return {}


def adjacency(arrays, registry, spec, limits):
    started = time.perf_counter()
    result = neighbors(arrays, spec)
    LOG.debug("adjacency exit ids=%d seconds=%.4f", len(result["neighbor_ids"]), time.perf_counter() - started)
    return result


@timed("neighbors.check")
def check(arrays, name, registry, spec):
    return None


NEIGHBORS = REPRESENTATIONS.register(
    "neighbors", Representation("neighbors", "sample", carries, horizon, piece, None, adjacency, check, 1)
)
