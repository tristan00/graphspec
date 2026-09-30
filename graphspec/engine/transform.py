import logging
import time

import numpy as np

from graphspec.infra.timing import measure, timed
from graphspec.operations import sets
from graphspec.operations.operation import OPERATIONS, PIPELINE
from graphspec.representations import groups as representations
from graphspec.views import view as views
from graphspec.views.graph import conforms, referring, unobserved, unresolved, verify_dense
from graphspec.views.visible import unstated

LOG = logging.getLogger(__name__)
SEED = 0x6772617068
LABELS = (
    "periods",
    "time window",
    "nodes",
    "derived relations",
    "relations",
    "edge filters",
    "relation merging",
)
STAGES = tuple(zip(LABELS, (OPERATIONS.named(name) for name in PIPELINE), strict=True))
REMOVES = "references"


def resolving(graph, registry):
    started = time.perf_counter()
    member, entity, moments, _ = referring(graph, registry)
    result = ~unobserved(len(graph.kinds), graph.obs_entity, graph.obs_step, entity, moments)
    LOG.debug(
        "resolving exit graph=%s ends=%d seconds=%.4f", graph.name, len(member), time.perf_counter() - started
    )
    return member, result


def kept(operation, before, graph, registry):
    started = time.perf_counter()
    if REMOVES not in operation.changes and "observations" in operation.changes:
        _, held = resolving(before, registry)
        member, found = resolving(graph, registry)
        lost = held & ~found
        if lost.any():
            raise ValueError(
                "the operation %s removes the node of %s and does not state that it may remove a referenced "
                "node" % (operation.name, unresolved(graph, registry, member[np.flatnonzero(lost)[0]]))
            )
    LOG.debug(
        "kept exit operation=%s graph=%s seconds=%.4f",
        operation.name,
        graph.name,
        time.perf_counter() - started,
    )


@timed("graph_transform")
def derive(graph, registry, spec, limits, carry=None, known=None, stages=STAGES, check=True):
    started, given = time.perf_counter(), graph
    for name, operation in stages:
        clock = time.perf_counter()
        with measure(name):
            before, graph = graph, operation(graph, registry, spec, SEED, limits, carry)
            if graph is not before:
                kept(operation, before, graph, registry)
            if check and graph is not before:
                conforms(graph, registry, tuple(part for part in operation.changes if part != REMOVES))
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


VALIDATED = "first"


def validate(facts, registry, splits):
    started = time.perf_counter()
    facts = sets.stored(facts, registry, {kind: sets.NODE for kind in registry.schema.sets})
    several = VALIDATED if registry.extras["several"] else None
    steps = views.TASKS if len(facts.task_key) else views.CHANGES
    result = [
        views.view(facts, registry, held, several)
        for held in views.sequences(facts, registry, steps, unstated(splits))
    ]
    LOG.debug(
        "validate exit graph=%s steps=%s views=%d seconds=%.4f",
        facts.name,
        steps,
        len(result),
        time.perf_counter() - started,
    )
    return result
