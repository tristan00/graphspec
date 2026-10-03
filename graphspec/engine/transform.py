import logging
import time

import numpy as np

from graphspec.infra.timing import measure
from graphspec.operations import sets
from graphspec.views import view as views
from graphspec.views.graph import conforms, referring, unobserved, unresolved, verify_dense
from graphspec.views.visible import unstated

LOG = logging.getLogger(__name__)
VERSION = 53
REMOVES = "references"


def resolving(graph, registry):
    started = time.perf_counter()
    member, entity, moments, named = referring(graph, registry, registry.referenced)
    result = named & ~unobserved(len(graph.kinds), graph.obs_entity, graph.obs_step, entity, moments)
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


def checked(step, before, graph, registry):
    if graph is not before:
        kept(step.operation, before, graph, registry)
        conforms(graph, registry, tuple(part for part in step.operation.changes if part != REMOVES))
    return graph


def initial(registry, steps):
    started = time.perf_counter()
    result = {step.name: step.initial(registry) for step in steps}
    LOG.debug("initial exit steps=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def derive(graph, registry, steps, seed, limits, partitions, carry, states, edited):
    started, found, given = time.perf_counter(), {}, graph
    for step in steps:
        clock = time.perf_counter()
        with measure(step.name):
            before = graph
            graph, found[step.name] = step(graph, registry, seed, limits, carry, states[step.name], partitions)
            checked(step, before, graph, registry)
        LOG.info(
            "stage exit stage=%s graph=%s seconds=%.3f facts=%d observations=%d",
            step.name,
            graph.name,
            time.perf_counter() - clock,
            len(graph.fact_step),
            len(graph.obs_step),
        )
    if edited or graph is not given:
        with measure("verify"):
            verify_dense(graph, registry, carry.known)
    LOG.info("derive exit graph=%s seconds=%.3f", graph.name, time.perf_counter() - started)
    return graph, found


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
