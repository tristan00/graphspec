import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation, Phase
from .parameters import enabled, flag
from .rows import FACTS, grouped, select

LOG = logging.getLogger(__name__)


def joined(registry):
    started = time.perf_counter()
    result = {frozenset(pair) for group in registry.groups for pair in registry.named[group].pairs}
    LOG.debug("joined exit pairs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def relating(registry):
    stated = bool(registry.groups)
    return lambda spec: stated


def stated(registry):
    started = time.perf_counter()
    result = tuple(flag("relations/%s/keep" % group, True) for group in registry.groups)
    LOG.debug("relations stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.relations")
def relations(graph, registry, spec, seed, limits, carry, held):
    dropped = [group for group in registry.groups if not enabled(spec, "relations/%s/keep" % group)]
    if not dropped:
        return graph
    return select(graph, FACTS, ~np.isin(graph.fact_code, grouped(registry, dropped)))


RELATIONS = OPERATIONS.register(
    "graphspec.relations",
    Operation(
        namespace="graphspec",
        name="relations",
        phases=(Phase("relations", relations, None, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        adds=None,
        version=4,
        graphs=1,
    ),
)
