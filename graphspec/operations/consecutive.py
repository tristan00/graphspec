import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed
from graphspec.views.spans import encode

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, real
from .rows import extend, fact_bytes
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
SIDES = ("sources", "targets")
WHOLE = -1


@timed("operators.orderable")
def orderable(registry):
    found = []
    for edge in registry.edges:
        selves = {first for first, second in edge.pairs if first == second}
        if edge.origin[0] != "base" or edge.attributes or len(selves) != 1:
            continue
        (kind,) = selves
        found.append(edge.name)
        for other in registry.edges:
            if other.origin[0] != "base":
                continue
            for place, side in enumerate(SIDES):
                if any(pair[place] == kind for pair in other.pairs):
                    found.append("%s:%s:%s" % (edge.name, other.name, side))
    return tuple(dict.fromkeys(found))


def stated(registry):
    started = time.perf_counter()
    options = orderable(registry)
    result = ()
    if options:
        result = (
            choice("consecutive/pairing", options, options[0]),
            real("consecutive/gap", 1.0, 1e9, 1e9, log=True),
        )
    LOG.debug("consecutive stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def carried(registry, spec):
    return {
        "consecutive/joined": Table(EMPTY, value=EMPTY),
        "consecutive/last": Table(EMPTY, entity=EMPTY, time=EMPTY, graph=np.zeros(0, np.uint64)),
    }


@timed("operators.members")
def members(graph, registry, pairing, kind, moment, carry):
    if len(pairing) == 1:
        rows = np.flatnonzero(
            (graph.obs_step == moment)
            & ~np.asarray(graph.obs_anchor, bool)
            & (np.asarray(graph.kinds)[graph.obs_entity] == kind)
        )
        held = np.unique(np.asarray(graph.obs_entity, np.int64)[rows])
        return held, np.full(len(held), WHOLE, np.int64), carry.glob[held]
    _, grouping, side = pairing
    edge = registry.by_name[grouping]
    rows = np.flatnonzero(
        (graph.fact_step == moment) & np.asarray(graph.fact_present, bool) & np.isin(graph.fact_code, edge.code)
    )
    ends = [np.asarray(graph.fact_source, np.int64)[rows], np.asarray(graph.fact_target, np.int64)[rows]]
    place = SIDES.index(side)
    held, owners = [ends[place]], [ends[1 - place]]
    if edge.symmetric:
        held.append(ends[1 - place])
        owners.append(ends[place])
    held, owners = np.concatenate(held), np.concatenate(owners)
    chosen = np.asarray(graph.kinds)[held] == kind
    pairs = np.unique(np.stack((held[chosen], carry.glob[owners[chosen]]), 1).reshape(-1, 2), axis=0)
    return pairs[:, 0], pairs[:, 1], encode(np.zeros(len(pairs)), pairs[:, 1], carry.glob[pairs[:, 0]], carry.base)


@timed("operators.consecutive")
def consecutive(graph, registry, spec, seed, limits, carry, held):
    if not orderable(registry):
        raise ValueError(
            "graphspec.consecutive fills a relation stated without columns from a kind of node to itself, and the "
            "bank %s states none" % registry.name
        )
    pairing = spec["consecutive/pairing"].split(":")
    edge, gap = registry.by_name[pairing[0]], spec["consecutive/gap"]
    (kind,) = {registry.kind(first) for first, second in edge.pairs if first == second}
    state, budget = carry.state, GraphBudget(limits, fact_bytes(graph))
    joined, last = state["consecutive/joined"], state["consecutive/last"]
    tokens, found = carry.tokens(np.arange(len(graph.kinds))), []
    entity, structural = np.asarray(graph.obs_entity, np.int64), ~np.asarray(graph.obs_anchor, bool)
    for moment in range(graph.steps):
        rows, owners, keys = members(graph, registry, pairing, kind, moment, carry)
        fresh = joined.find(keys) < 0
        rows, owners, keys = rows[fresh], owners[fresh], keys[fresh]
        if not len(rows):
            continue
        joined = joined.replaced(EMPTY, keys, value=np.ones(len(keys), np.int64))
        order = np.lexsort((tokens[rows], owners))
        rows, owners = rows[order], owners[order]
        head = np.r_[True, owners[1:] != owners[:-1]]
        tail = np.r_[owners[1:] != owners[:-1], True]
        source, target = rows[:-1][~head[1:]], rows[1:][~head[1:]]
        now, mark = int(graph.step_time[moment]), np.uint64(graph.step_token[moment])
        before = last.find(owners[head])
        kept = np.flatnonzero(before >= 0)
        earlier = last.columns["entity"][before[kept]]
        within = (
            (last.columns["graph"][before[kept]] == mark)
            & (now - last.columns["time"][before[kept]] <= gap)
            & np.isin(earlier, carry.glob[entity[(graph.obs_step == moment) & structural]])
        )
        source = np.r_[source, carry.local(earlier[within])]
        target = np.r_[target, rows[head][kept[within]]]
        last = last.replaced(
            owners[tail],
            owners[tail],
            entity=carry.glob[rows[tail]],
            time=np.full(int(tail.sum()), now, np.int64),
            graph=np.full(int(tail.sum()), mark, np.uint64),
        )
        if edge.symmetric:
            source, target = np.minimum(source, target), np.maximum(source, target)
        pairs = np.unique(np.stack((source, target), 1).reshape(-1, 2).astype(np.int64), axis=0)
        found.append((pairs[:, 0], pairs[:, 1], np.full(len(pairs), moment, np.int64)))
    rows_of = joined.keys % carry.base
    place = np.minimum(np.searchsorted(carry.glob, rows_of), len(carry.glob) - 1)
    lasting = (carry.glob[place] != rows_of) | np.asarray(graph.entity_connected, bool)[place]
    state["consecutive/joined"] = Table(joined.keys[lasting], value=joined.columns["value"][lasting])
    state["consecutive/last"] = last
    source, target, moment = (np.concatenate([part[index] for part in found] + [EMPTY]) for index in range(3))
    budget.retain(len(source), "consecutive")
    return extend(
        graph,
        source,
        target,
        np.full(len(source), edge.code, np.int64),
        moment,
        np.ones(len(source), bool),
        np.full(len(source), -1, np.int64),
    )


CONSECUTIVE = OPERATIONS.register(
    "graphspec.consecutive",
    Operation(
        namespace="graphspec",
        name="consecutive",
        phases=(Phase("consecutive", consecutive, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        version=1,
        graphs=1,
    ),
)
