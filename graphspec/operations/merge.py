import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice
from .rows import FACTS, OBSERVATIONS, fact_keys, keyed_relations, prune, pruned, second_of, shown

LOG = logging.getLogger(__name__)
PREFERRED = ("first", "second")


def stated(registry):
    started = time.perf_counter()
    result = (choice("merge/prefer", PREFERRED, "first"),)
    LOG.debug("merge stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def carried(registry, spec):
    return {"prune/merge": pruned()}


@timed("operators.taken")
def taken(prefer, mine, theirs):
    if prefer == "first":
        return np.ones(len(mine), bool), ~np.isin(theirs, mine)
    return ~np.isin(mine, theirs), np.ones(len(theirs), bool)


@timed("operators.merged_rows")
def merged_rows(graph, names, added, kept, moments):
    order = np.argsort(np.concatenate((moments[0][kept[0]], moments[1][kept[1]])), kind="stable")
    return {
        name: np.concatenate((np.asarray(graph.arrays[name])[kept[0]], np.asarray(added[name])[kept[1]])).astype(
            graph.arrays[name].dtype
        )[order]
        for name in names
    }


@timed("operators.merge")
def merge(graph, registry, spec, seed, limits, carry, held):
    second, prefer = second_of(carry, "graphspec.merge"), spec["merge/prefer"]
    count, kinds = len(graph.kinds), np.asarray(graph.kinds)
    entity = carry.local(second.obs_entity)
    observed = taken(
        prefer,
        graph.obs_step.astype(np.int64) * count + graph.obs_entity,
        second.obs_step.astype(np.int64) * count + entity,
    )
    source, target = carry.local(second.fact_source), carry.local(second.fact_target)
    code = second.fact_code.astype(np.int64)
    swap = keyed_relations(registry)[1][code] & (kinds[source] == kinds[target]) & (source > target)
    source, target = np.where(swap, target, source), np.where(swap, source, target)
    chosen = taken(
        prefer,
        *fact_keys(
            registry,
            (graph.fact_step, graph.fact_code, graph.fact_source, graph.fact_target),
            (second.fact_step, code, source, target),
        ),
    )
    if all(mask.all() for mask in (observed[0], chosen[0])) and not (observed[1].any() or chosen[1].any()):
        return prune(graph, np.ones(len(graph.fact_step), bool), carry, "merge")
    x0, v0 = len(graph.x_table), len(graph.value_table)
    observations = merged_rows(
        graph,
        OBSERVATIONS,
        dict(
            obs_entity=entity,
            obs_step=second.obs_step,
            obs_ref=x0 + second.obs_ref.astype(np.int64),
            obs_anchor=np.zeros(len(entity), bool),
        ),
        observed,
        (graph.obs_step.astype(np.int64), second.obs_step.astype(np.int64)),
    )
    facts = merged_rows(
        graph,
        FACTS,
        dict(
            fact_source=source,
            fact_target=target,
            fact_code=code,
            fact_step=second.fact_step,
            fact_present=second.fact_present,
            fact_ref=np.where(second.fact_ref >= 0, v0 + second.fact_ref.astype(np.int64), -1),
        ),
        chosen,
        tuple(
            np.asarray(step, np.int64) * 2 + ~np.asarray(present, bool)
            for step, present in ((graph.fact_step, graph.fact_present), (second.fact_step, second.fact_present))
        ),
    )
    merged = graph.replace(
        x_table=np.concatenate((graph.x_table, second.x_table.astype(graph.x_table.dtype))),
        value_table=np.concatenate((graph.value_table, second.value_table.astype(graph.value_table.dtype))),
        **observations,
        **facts,
    )
    carry.x_global = np.concatenate((carry.x_global, second.x_global))
    carry.value_global = np.concatenate((carry.value_global, second.value_global))
    lit = shown(merged, merged.fact_step, merged.fact_source, merged.fact_target)
    return prune(merged, ~merged.fact_present | lit, carry, "merge")


MERGE = OPERATIONS.register(
    "graphspec.merge",
    Operation(
        namespace="graphspec",
        name="merge",
        phases=(Phase("merge", merge, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("observations", "facts"),
        adds=None,
        version=5,
        graphs=2,
    ),
)
