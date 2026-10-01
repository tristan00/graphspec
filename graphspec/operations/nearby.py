import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, equals, number, real
from .rows import extend, fact_bytes

LOG = logging.getLogger(__name__)
RULES = ("threshold", "nearest")
BLOCK = 1 << 21
NUMBER = "number"


@timed("operators.measured")
def measured(registry, first, second):
    columns, kinds = registry.schema.columns, registry.schema.entities()
    return tuple(
        name
        for name in kinds[first]
        if name in kinds[second] and columns[name]["type"] == NUMBER
    )


@timed("operators.fillable")
def fillable(registry):
    return tuple(
        edge.name
        for edge in registry.edges
        if edge.origin[0] == "base"
        and not edge.attributes
        and any(measured(registry, first, second) for first, second in edge.pairs)
    )


def stated(registry):
    started = time.perf_counter()
    names = fillable(registry)
    result = ()
    if names:
        result = (
            choice("nearby/relation", names, names[0]),
            choice("nearby/rule", RULES, "nearest"),
            real("nearby/threshold", 0.01, 100.0, 1.0, equals("nearby/rule", "threshold"), log=True),
            number("nearby/k", 1, 16, 4, equals("nearby/rule", "nearest")),
        )
    LOG.debug("nearby stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.placed")
def placed(registry, columns, relation, first, second):
    encoding, values, flags = registry.encoding, [], []
    start, flagged = 0, encoding.values
    for name in encoding.table:
        width, flag = encoding.sizes[name]
        if name in columns:
            values += range(start, start + width)
            flags += range(flagged, flagged + flag)
        start, flagged = start + width, flagged + flag
    if not values:
        raise ValueError(
            "graphspec.nearby fills %s from the columns of numbers its ends %s and %s share, %s, and the encoders "
            "fitted for this statement give none of them a width" % (relation, first, second, ", ".join(columns))
        )
    return np.asarray(values, np.int64), np.asarray(flags, np.int64)


@timed("operators.near_pairs")
def near_pairs(spec, left, right, ranks, same, budget):
    found = []
    rows = max(1, BLOCK // max(len(right[0]), 1))
    for start in range(0, len(left[0]), rows):
        stop = min(start + rows, len(left[0]))
        budget.workspace((stop - start) * len(right[0]) * 24, "nearby distances")
        distance = np.sqrt(
            ((left[1][start:stop, None, :] - right[1][None, :, :]) ** 2).sum(axis=2)
        )
        if same:
            distance[left[0][start:stop, None] == right[0][None, :]] = np.inf
        if spec["nearby/rule"] == "threshold":
            source, target = np.nonzero(distance <= spec["nearby/threshold"])
        else:
            order = np.argsort(distance[:, ranks], axis=1, kind="stable")[:, : spec["nearby/k"]]
            source = np.repeat(np.arange(stop - start), order.shape[1])
            target = ranks[order.reshape(-1)]
            kept = np.isfinite(distance[source, target])
            source, target = source[kept], target[kept]
        found.append((left[0][start + source], right[0][target]))
    return (
        np.concatenate([pair[0] for pair in found] + [np.zeros(0, np.int64)]),
        np.concatenate([pair[1] for pair in found] + [np.zeros(0, np.int64)]),
    )


@timed("operators.nearby")
def nearby(graph, registry, spec, seed, limits, carry, held):
    names = fillable(registry)
    if not names:
        raise ValueError(
            "graphspec.nearby fills a relation stated without columns between kinds of node that share a column "
            "of numbers, and the bank %s states none" % registry.name
        )
    edge = registry.by_name[spec["nearby/relation"]]
    budget, kinds = GraphBudget(limits, fact_bytes(graph)), np.asarray(graph.kinds)
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    structural, table, reference = ~np.asarray(graph.obs_anchor, bool), np.asarray(graph.x_table), graph.obs_ref
    bounds = np.searchsorted(step, np.arange(graph.steps + 1))
    sources, targets, moments = [], [], []
    for first, second in edge.pairs:
        columns = measured(registry, first, second)
        if not columns:
            continue
        values, flags = placed(registry, columns, edge.name, first, second)
        same = first == second
        for moment in range(graph.steps):
            rows = np.arange(bounds[moment], bounds[moment + 1])
            ends = []
            for kind in (first, second):
                chosen = rows[structural[rows] & (kinds[entity[rows]] == registry.kind(kind))]
                found = table[np.asarray(reference)[chosen]]
                full = np.all(found[:, flags] > 0, axis=1)
                ends.append((entity[chosen][full], found[full][:, values].astype(np.float64)))
            ranks = np.argsort(carry.tokens(ends[1][0]), kind="stable")
            source, target = near_pairs(spec, ends[0], ends[1], ranks, same, budget)
            if edge.symmetric and same:
                low, high = np.minimum(source, target), np.maximum(source, target)
                pairs = np.unique(np.stack((low, high), 1).reshape(-1, 2), axis=0)
                source, target = pairs[:, 0], pairs[:, 1]
            budget.retain(len(source), "nearby")
            sources.append(source)
            targets.append(target)
            moments.append(np.full(len(source), moment, np.int64))
    source, target, moment = (np.concatenate(held + [np.zeros(0, np.int64)]) for held in (sources, targets, moments))
    LOG.debug("nearby added relation=%s facts=%d", edge.name, len(source))
    return extend(
        graph,
        source,
        target,
        np.full(len(source), edge.code, np.int64),
        moment,
        np.ones(len(source), bool),
        np.full(len(source), -1, np.int64),
    )


NEARBY = OPERATIONS.register(
    "graphspec.nearby",
    Operation(
        namespace="graphspec",
        name="nearby",
        phases=(Phase("nearby", nearby, None, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        adds=None,
        version=2,
        graphs=1,
    ),
)
