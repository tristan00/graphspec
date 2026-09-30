import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation
from .parameters import enabled, flag
from .rows import FACTS, grouped, select

LOG = logging.getLogger(__name__)


def linking(registry):
    started = time.perf_counter()
    names = tuple("relations/%s/keep" % group for group in registry.groups)
    LOG.debug("linking exit relations=%d seconds=%.4f", len(names), time.perf_counter() - started)
    return lambda spec: any(enabled(spec, name) for name in names)


def joining(registry):
    started = time.perf_counter()
    joined = {}
    for group in registry.groups:
        for pair in registry.named[group].pairs:
            joined.setdefault(frozenset(pair), []).append("relations/%s/keep" % group)
    LOG.debug("joining exit pairs=%d seconds=%.4f", len(joined), time.perf_counter() - started)
    return lambda spec, first, second: any(
        enabled(spec, name) for name in joined.get(frozenset((first, second)), ())
    )


def stated(registry):
    started = time.perf_counter()
    result = tuple(
        flag(
            "relations/%s/keep" % group,
            True,
            lambda spec, edge=registry.named[group]: any(
                spec["nodes/%s/keep" % source] and spec["nodes/%s/keep" % target]
                for source, target in edge.pairs
            ),
        )
        for group in registry.groups
    )
    LOG.debug("relations stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.relations")
def relations(graph, registry, spec, seed, limits, carry):
    dropped = [group for group in registry.groups if not enabled(spec, "relations/%s/keep" % group)]
    if not dropped:
        return graph
    return select(graph, FACTS, ~np.isin(graph.fact_code, grouped(registry, dropped)))


RELATIONS = OPERATIONS.register(
    "graphspec.relations",
    Operation(
        namespace="graphspec",
        name="relations",
        function=relations,
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        version=1,
    ),
)
