import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.schema.schema import ENCODERS

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, equals, number, real
from .rows import extend, fact_bytes, made_values

LOG = logging.getLogger(__name__)
RULES = ("threshold", "nearest")
BLOCK = 1 << 21
NUMBER = "number"
LACKING = (
    "graphspec.nearby joins nodes of two kinds by the distance between the values of the columns of numbers both "
    "kinds state, and the bank %s states no two such kinds"
)


@timed("operators.measured")
def measured(registry, first, second):
    columns, kinds = registry.schema.columns, registry.schema.declared()
    return tuple(name for name in kinds[first] if name in kinds[second] and columns[name]["type"] == NUMBER)


@timed("operators.pairable")
def pairable(registry):
    return tuple(
        "%s:%s" % (first, second)
        for first in registry.declared
        for second in registry.declared
        if measured(registry, first, second)
    )


@timed("operators.reading")
def reading():
    return tuple(name for name in ENCODERS.names() if NUMBER in ENCODERS.named(name).types)


def stated(registry):
    started = time.perf_counter()
    names = pairable(registry)
    result = ()
    if names:
        result = (
            choice("nearby/kinds", names, names[0]),
            choice("nearby/rule", RULES, "nearest"),
            real("nearby/threshold", 0.01, 100.0, 1.0, equals("nearby/rule", "threshold"), log=True),
            number("nearby/k", 1, 16, 4, equals("nearby/rule", "nearest")),
            choice("nearby/encoder", reading(), None),
        )
    LOG.debug("nearby stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def named(registry, settings):
    started = time.perf_counter()
    result = dict.fromkeys(settings["nearby/kinds"].split(":") if pairable(registry) else (), "names")
    LOG.debug("nearby named exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.nearby_adds")
def adds(registry, settings, entry):
    if not pairable(registry):
        raise ValueError(LACKING % registry.name)
    first, second = settings["nearby/kinds"].split(":")
    distance = entry + ".distance"
    return Added(
        {distance: dict(type=NUMBER, encoder=settings["nearby/encoder"])},
        {},
        {},
        {entry: dict(pairs=((first, second),), directed=settings["nearby/rule"] == "nearest", columns=(distance,))},
        {},
        {},
    )


@timed("operators.placed")
def placed(registry, columns):
    slots = [registry.encoding.slots(name) for name in columns]
    values = np.concatenate([held[0] for held in slots] + [np.zeros(0, np.int64)])
    flags = np.concatenate([held[1] for held in slots] + [np.zeros(0, np.int64)])
    return values, flags


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
        found.append((left[0][start + source], right[0][target], distance[source, target]))
    return tuple(
        np.concatenate([pair[place] for pair in found] + [np.zeros(0, kind)])
        for place, kind in enumerate((np.int64, np.int64, np.float64))
    )


@timed("operators.nearby")
def nearby(graph, registry, spec, seed, limits, carry, held):
    if not pairable(registry):
        raise ValueError(LACKING % registry.name)
    first, second = spec["nearby/kinds"].split(":")
    edge, columns = registry.by_name[carry.entry], measured(registry, first, second)
    values, flags = placed(registry, columns)
    if not len(values):
        raise ValueError(
            "graphspec.nearby joins %s to %s by the columns of numbers they share, %s, and the encoders fitted for "
            "this statement give none of them a width" % (first, second, ", ".join(columns))
        )
    budget, kinds = GraphBudget(limits, fact_bytes(graph)), np.asarray(graph.kinds)
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    structural, table, reference = ~np.asarray(graph.obs_anchor, bool), np.asarray(graph.x_table), graph.obs_ref
    bounds = np.searchsorted(step, np.arange(graph.steps + 1))
    same, found = first == second, []
    for moment in range(graph.steps):
        rows = np.arange(bounds[moment], bounds[moment + 1])
        ends = []
        for kind in (first, second):
            chosen = rows[structural[rows] & (kinds[entity[rows]] == registry.kind(kind))]
            shown = table[np.asarray(reference)[chosen]]
            full = np.all(shown[:, flags] > 0, axis=1)
            ends.append((entity[chosen][full], shown[full][:, values].astype(np.float64)))
        ranks = np.argsort(carry.tokens(ends[1][0]), kind="stable")
        source, target, distance = near_pairs(spec, ends[0], ends[1], ranks, same, budget)
        if edge.symmetric and same:
            low, high = np.minimum(source, target), np.maximum(source, target)
            _, kept = np.unique(np.stack((low, high), 1).reshape(-1, 2), axis=0, return_index=True)
            source, target, distance = low[kept], high[kept], distance[kept]
        budget.retain(len(source), "nearby")
        found.append((source, target, distance, np.full(len(source), moment, np.int64)))
    source, target, distance, moment = (
        np.concatenate([part[place] for part in found] + [np.zeros(0, kind)])
        for place, kind in enumerate((np.int64, np.int64, np.float64, np.int64))
    )
    rows = np.full((len(source), graph.value_table.shape[1]), np.nan, np.float64)
    rows[:, 0] = distance
    graph, references = made_values(graph, carry, rows)
    LOG.debug("nearby added relation=%s facts=%d", edge.name, len(source))
    return extend(
        graph,
        source,
        target,
        np.full(len(source), edge.code, np.int64),
        moment,
        np.ones(len(source), bool),
        references,
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
        adds=adds,
        kinds=named,
        version=5,
        graphs=1,
    ),
)
