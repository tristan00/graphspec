import logging
import time

import numpy as np

from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.schema.references import ends

from . import kernels
from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import among, choice, equals, flag, holds, number, real
from .rows import OBSERVATIONS, SAMPLE, prune, pruned, select
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
FILTERS = ("none", "degree", "total_degree", "core", "activity", "sample")
REFERRED = ("keep",)
REMOVED = ("remove",)


def stated(registry):
    started = time.perf_counter()
    referred = REFERRED if registry.stops.any() else REFERRED + REMOVED
    result = [flag("nodes/%s/keep" % kind, True) for kind in registry.nodes]
    result.append(choice("nodes/referenced", referred, "keep"))
    for kind in registry.nodes:
        prefix = "nodes/%s/" % kind
        result += [
            choice(prefix + "filter", FILTERS, "none", lambda spec, kind=kind: spec["nodes/%s/keep" % kind]),
            number(prefix + "min_degree", 1, 16, 1, among(prefix + "filter", ("degree", "total_degree"))),
            number(prefix + "core", 2, 16, 2, equals(prefix + "filter", "core")),
            number(prefix + "min_activity", 2, 16, 2, equals(prefix + "filter", "activity")),
            real(prefix + "sample", 0.1, 0.9, 0.5, equals(prefix + "filter", "sample")),
        ]
    LOG.debug("nodes stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


@timed("transform.filtered")
def filtered(registry, spec, mode):
    return any(holds(spec, "nodes/%s/filter" % kind, mode) for kind in registry.nodes)


def counters(carry, name, graph):
    return np.asarray(carry.state[name].lookup("value", carry.glob, 0), np.int64).copy()


def remember(carry, name, graph, values):
    chosen = np.flatnonzero(graph.entity_connected)
    keys = carry.glob[chosen]
    carry.state[name] = carry.state[name].replaced(keys, keys, value=values[chosen])


def carried(registry, spec):
    return {
        "nodes/total_degree": Table(EMPTY, value=EMPTY),
        "nodes/activity": Table(EMPTY, value=EMPTY),
        "nodes/structural": Table(EMPTY, value=EMPTY),
        "prune/nodes": pruned(),
    }


@timed("operators.nodes")
def nodes(graph, registry, spec, seed, limits, carry, held):
    steps, entities = graph.steps, len(graph.kinds)
    retained = np.asarray([spec["nodes/%s/keep" % kind] for kind in registry.nodes])
    filters = {
        kind: spec["nodes/%s/filter" % kind]
        for kind in registry.nodes
        if "nodes/%s/filter" % kind in spec and spec["nodes/%s/filter" % kind] != "none"
    }
    if retained.all() and not filters and not graph.obs_anchor.any():
        return graph
    allowed = retained[graph.kinds]
    obs_entity, obs_step = graph.obs_entity, graph.obs_step
    use = graph.fact_present & allowed[graph.fact_source] & allowed[graph.fact_target]
    kinds = graph.kinds[obs_entity]
    passed, minimum, counts = np.ones(len(obs_step), bool), np.zeros(len(obs_step), np.int64), {}
    for kind, mode in filters.items():
        prefix, members = "nodes/%s/" % kind, kinds == registry.kind(kind)
        if mode in ("degree", "total_degree"):
            if mode not in counts:
                total = mode == "total_degree"
                count = counters(carry, "nodes/total_degree", graph) if total else np.zeros(entities, np.int64)
                counts[mode] = kernels.degrees(
                    obs_entity,
                    obs_step,
                    graph.fact_source,
                    graph.fact_target,
                    graph.fact_step,
                    use,
                    entities,
                    total,
                    count,
                )
                if total:
                    remember(carry, "nodes/total_degree", graph, count)
            passed[members] = counts[mode][members] >= spec[prefix + "min_degree"]
        elif mode == "activity":
            if mode not in counts:
                count = counters(carry, "nodes/activity", graph)
                counts[mode] = kernels.activity(obs_entity, count)
                remember(carry, "nodes/activity", graph, count)
            passed[members] = counts[mode][members] >= spec[prefix + "min_activity"]
        elif mode == "sample":
            passed[members] = unit(seed, SAMPLE, carry.glob[obs_entity[members]]) < spec[prefix + "sample"]
        else:
            minimum[members] = spec[prefix + "core"]
    structural = allowed[obs_entity] & passed
    if minimum.any():
        structural = kernels.peel(
            obs_entity,
            obs_step,
            structural,
            minimum,
            graph.fact_source,
            graph.fact_target,
            graph.fact_step,
            use,
            entities,
        )
    needed = np.zeros(len(obs_step), bool)
    if spec["nodes/referenced"] == "keep":
        member, entity = ends(graph.member_source, graph.member_target)
        moments = graph.task_step[graph.member_task[member]].astype(np.int64)
        needed = np.isin(obs_step.astype(np.int64) * entities + obs_entity, moments * entities + entity)
    first, _ = kernels.first_observed(obs_entity, obs_step, structural, entities, steps)
    first[counters(carry, "nodes/structural", graph) > 0] = -1
    remember(carry, "nodes/structural", graph, (first < steps).astype(np.int64))
    present = kernels.observed_at(
        obs_entity, obs_step, structural, graph.fact_source, graph.fact_step, entities
    ) & kernels.observed_at(obs_entity, obs_step, structural, graph.fact_target, graph.fact_step, entities)
    absent = (first[graph.fact_source] <= graph.fact_step) & (first[graph.fact_target] <= graph.fact_step)
    alive = structural | needed
    graph = select(graph, OBSERVATIONS, alive)
    anchor = ~structural[alive]
    if not np.array_equal(anchor, graph.obs_anchor):
        graph = graph.replace(obs_anchor=anchor)
    return prune(graph, np.where(graph.fact_present, present, absent), carry, "nodes")


NODES = OPERATIONS.register(
    "graphspec.nodes",
    Operation(
        namespace="graphspec",
        name="nodes",
        phases=(Phase("nodes", nodes, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=tuple(
            Count(name, counts, lambda registry, spec, mode=mode: filtered(registry, spec, mode), "stored", "stored")
            for name, counts, mode in (
                ("nodes/degree", "degree", "degree"),
                ("nodes/total_degree", "degree", "total_degree"),
                ("nodes/core", "core numbers", "core"),
            )
        ),
        changes=("observations", "facts", "references"),
        version=2,
    ),
)
