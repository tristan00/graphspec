import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.references import ends

from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import choice, number
from .rows import OBSERVATIONS, first_kept, kept, prune, pruned, select

LOG = logging.getLogger(__name__)
REFERENCES = "references"


def stated(registry):
    started = time.perf_counter()
    result = (
        number("neighbourhood/hops", 0, 3, 1),
        choice("neighbourhood/around", (REFERENCES,) + tuple(registry.declared), REFERENCES),
    )
    LOG.debug("neighbourhood stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def named(registry, settings):
    started = time.perf_counter()
    around = settings["neighbourhood/around"]
    result = {} if around == REFERENCES else {around: "names"}
    LOG.debug("neighbourhood named exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def carried(registry, spec):
    return {"neighbourhood/kept": kept(), "prune/neighbourhood": pruned()}


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
    count = len(graph.kinds)
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    member, named = ends(graph.member_source, graph.member_target)
    moments = np.asarray(graph.task_step, np.int64)[np.asarray(graph.member_task, np.int64)[member]]
    linked = np.asarray(graph.member_name)[member] == len(registry.references)
    seeds, protected = moments[~linked] * count + named[~linked], moments[linked] * count + named[linked]
    if spec["neighbourhood/around"] != REFERENCES:
        chosen = np.asarray(graph.kinds)[entity] == registry.kind(spec["neighbourhood/around"])
        seeds = np.r_[seeds, step[chosen] * count + entity[chosen]]
    key = step * count + entity
    present = np.asarray(graph.fact_present, bool)
    moment = np.asarray(graph.fact_step, np.int64)
    source = moment * count + np.asarray(graph.fact_source, np.int64)
    target = moment * count + np.asarray(graph.fact_target, np.int64)
    reached = np.union1d(
        within(seeds[np.isin(seeds, key)], source[present], target[present], spec["neighbourhood/hops"]), protected
    )
    alive = np.isin(key, reached)
    first = first_kept(carry, "neighbourhood/kept", graph, entity, step, alive)
    absent = (first[graph.fact_source] <= moment) & (first[graph.fact_target] <= moment)
    remains = np.where(present, np.isin(source, reached) & np.isin(target, reached), absent)
    return prune(select(graph, OBSERVATIONS, alive), remains, carry, "neighbourhood")


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
        adds=None,
        kinds=named,
        version=4,
        graphs=1,
    ),
)
