import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.sources import digest

from . import operators, representations
from .infra.timing import measure, timed
from .truth import conforms, referring, unobserved, unresolved, verify_dense

LOG = logging.getLogger(__name__)
SOURCE = digest((Path(__file__).parent,))
SEED = 0x6772617068
STAGES = (
    ("periods", operators.periods),
    ("time window", operators.window),
    ("nodes", operators.nodes),
    ("derived relations", operators.derive),
    ("relations", operators.relations),
    ("edge filters", operators.edges),
    ("relation merging", operators.merging),
)
CHANGES = {
    operators.periods: ("time",),
    operators.window: ("observations", "facts"),
    operators.nodes: ("observations", "facts", "references"),
    operators.derive: ("facts",),
    operators.relations: ("facts",),
    operators.edges: ("facts",),
    operators.merging: ("facts",),
}
REMOVES = "references"


def resolving(graph, registry):
    started = time.perf_counter()
    member, entity, moments, _ = referring(graph, registry)
    result = ~unobserved(len(graph.kinds), graph.obs_entity, graph.obs_step, entity, moments)
    LOG.debug(
        "resolving exit graph=%s ends=%d seconds=%.4f", graph.name, len(member), time.perf_counter() - started
    )
    return member, result


def kept(name, stage, before, graph, registry):
    started = time.perf_counter()
    if REMOVES not in CHANGES[stage] and "observations" in CHANGES[stage]:
        _, held = resolving(before, registry)
        member, found = resolving(graph, registry)
        lost = held & ~found
        if lost.any():
            raise ValueError(
                "the stage %s removes the node of %s and does not state that it may remove a referenced node"
                % (name, unresolved(graph, registry, member[np.flatnonzero(lost)[0]]))
            )
    LOG.debug("kept exit stage=%s graph=%s seconds=%.4f", name, graph.name, time.perf_counter() - started)


@timed("graph_transform")
def derive(graph, registry, spec, limits, carry=None, known=None, stages=STAGES, check=True):
    started, given = time.perf_counter(), graph
    for name, stage in stages:
        clock = time.perf_counter()
        options = dict(limits=limits) if stage in (operators.window, operators.derive) else {}
        with measure(name):
            before, graph = graph, stage(graph, registry, spec, SEED, carry=carry, **options)
            if graph is not before:
                kept(name, stage, before, graph, registry)
            if check and graph is not before:
                conforms(graph, registry, tuple(part for part in CHANGES[stage] if part != REMOVES))
        LOG.info(
            "stage exit stage=%s graph=%s seconds=%.3f facts=%d observations=%d",
            name,
            graph.name,
            time.perf_counter() - clock,
            len(graph.fact_step),
            len(graph.obs_step),
        )
    if check and graph is not given:
        with measure("verify"):
            verify_dense(graph, registry, known)
    LOG.info("derive exit graph=%s seconds=%.3f", graph.name, time.perf_counter() - started)
    return graph


@timed("materialize")
def materialize(graph, registry, spec, groups, limits):
    started = time.perf_counter()
    derived = derive(graph, registry, spec, limits)
    arrays = representations.build(derived, registry, spec, groups, SEED, limits)
    representations.validate(arrays, derived, registry, spec, groups)
    LOG.info(
        "materialize exit graph=%s seconds=%.3f arrays=%d",
        graph.name,
        time.perf_counter() - started,
        len(arrays),
    )
    return arrays
