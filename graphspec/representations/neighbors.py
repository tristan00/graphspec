import logging
import time

from .groups import neighbors
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


def carries(registry, spec):
    return {}


def horizon(spec):
    return 0, 0


def piece(shown, state):
    return {}


def adjacency(arrays, registry, spec, limits):
    started = time.perf_counter()
    result = neighbors(arrays, spec)
    LOG.debug("adjacency exit ids=%d seconds=%.4f", len(result["neighbor_ids"]), time.perf_counter() - started)
    return result


def check(arrays, name, registry, spec):
    return None


NEIGHBORS = REPRESENTATIONS.register(
    "neighbors", Representation("neighbors", carries, horizon, piece, None, adjacency, check)
)
