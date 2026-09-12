import logging
from pathlib import Path
import time

import graph_tuning_infra
from graph_tuning_infra.sources import digest

from . import operators, representations
from .truth import verify


LOG = logging.getLogger(__name__)
SOURCE = digest((Path(__file__).parent, Path(graph_tuning_infra.__file__).parent))
SEED = 0x6772617068
STAGES = (('periods', operators.periods), ('time window', operators.window), ('nodes', operators.nodes),
          ('derived relations', operators.derive), ('relations', operators.relations), ('edge filters', operators.edges),
          ('relation merging', operators.merging))


def derive(graph, registry, spec):
    started = time.perf_counter()
    for name, stage in STAGES:
        clock = time.perf_counter()
        graph = stage(graph, registry, spec, SEED)
        verify(graph, registry)
        LOG.info('stage exit stage=%s graph=%s seconds=%.3f facts=%d observations=%d', name, graph.name, time.perf_counter() - clock,
                 len(graph.fact_step), len(graph.obs_step))
    LOG.info('derive exit graph=%s seconds=%.3f', graph.name, time.perf_counter() - started)
    return graph


def materialize(graph, registry, spec, groups, categories):
    started = time.perf_counter()
    derived = derive(graph, registry, spec)
    arrays = representations.build(derived, registry, spec, groups, categories, SEED)
    representations.validate(arrays, derived, registry, spec, groups)
    LOG.info('materialize exit graph=%s seconds=%.3f arrays=%d', graph.name, time.perf_counter() - started, len(arrays))
    return arrays
