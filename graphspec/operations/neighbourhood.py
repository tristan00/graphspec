import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.references import ends

from . import kernels
from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import choice, number
from .rows import OBSERVATIONS, prune, pruned, select
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
REFERENCES = "references"


def stated(registry):
    started = time.perf_counter()
    result = (
        number("neighbourhood/hops", 0, 3, 1),
        choice("neighbourhood/around", (REFERENCES,) + tuple(registry.nodes), REFERENCES),
    )
    LOG.debug("neighbourhood stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def carried(registry, spec):
    return {"neighbourhood/kept": Table(EMPTY, value=EMPTY), "prune/neighbourhood": pruned()}


@timed("operators.within")
def within(seeds, source, target, hops):
    reached = np.unique(np.asarray(seeds, np.int64))
    source, target = np.asarray(source, np.int64), np.asarray(target, np.int64)
    for _ in range(hops):
        grown = np.union1d(reached, np.r_[target[np.isin(source, reached)], source[np.isin(target, reached)]])
        if len(grown) == len(reached):
            break
        reached = grown
    return reached


@timed("operators.neighbourhood")
def neighbourhood(graph, registry, spec, seed, limits, carry, held):
    count, steps = len(graph.kinds), graph.steps
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    member, named = ends(graph.member_source, graph.member_target)
    moments = np.asarray(graph.task_step, np.int64)[np.asarray(graph.member_task, np.int64)[member]]
    seeds = moments * count + named
    if spec["neighbourhood/around"] != REFERENCES:
        chosen = np.asarray(graph.kinds)[entity] == registry.kind(spec["neighbourhood/around"])
        seeds = np.r_[seeds, step[chosen] * count + entity[chosen]]
    key = step * count + entity
    present = np.asarray(graph.fact_present, bool)
    moment = np.asarray(graph.fact_step, np.int64)
    source = moment * count + np.asarray(graph.fact_source, np.int64)
    target = moment * count + np.asarray(graph.fact_target, np.int64)
    reached = within(seeds[np.isin(seeds, key)], source[present], target[present], spec["neighbourhood/hops"])
    alive = np.isin(key, reached)
    first, _ = kernels.first_observed(entity, step, alive, count, steps)
    table = carry.state["neighbourhood/kept"]
    first[table.lookup("value", carry.glob, 0) > 0] = -1
    lasting = np.flatnonzero((first < steps) & np.asarray(graph.entity_connected, bool))
    carry.state["neighbourhood/kept"] = table.replaced(
        carry.glob[lasting], carry.glob[lasting], value=np.ones(len(lasting), np.int64)
    )
    absent = (first[graph.fact_source] <= moment) & (first[graph.fact_target] <= moment)
    kept = np.where(present, np.isin(source, reached) & np.isin(target, reached), absent)
    return prune(select(graph, OBSERVATIONS, alive), kept, carry, "neighbourhood")


NEIGHBOURHOOD = OPERATIONS.register(
    "graphspec.neighbourhood",
    Operation(
        namespace="graphspec",
        name="neighbourhood",
        phases=(Phase("neighbourhood", neighbourhood, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(
            Count(
                "neighbourhood/hops",
                "paths",
                lambda registry, spec: spec["neighbourhood/hops"] > 0,
                "stored",
                "stored",
            ),
        ),
        changes=("observations", "facts"),
        version=1,
        graphs=1,
    ),
)
