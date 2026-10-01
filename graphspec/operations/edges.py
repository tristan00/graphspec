import logging
import time

import numpy as np

from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed

from . import kernels
from .derive import anchoring
from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import DIRECTIONS, choice, differs, enabled, holds, number, real
from .relations import relating
from .rows import DROP, RANDOM, keyed_relations, open_runs, prune, pruned, run_state, runs

LOG = logging.getLogger(__name__)
EMITTED = ("project", "complete", "hop", "diffusion", "rewire")
DIRECTION = "edges/direction"


def filters(name, condition, attributes):
    prefix = "edges/%s/" % name
    orders = ("recent", "random") + tuple("largest:" + attribute for attribute in attributes)
    return [
        real(prefix + "drop", 0.0, 0.9, 0.0, condition),
        number(prefix + "persistence", 1, 8, 1, condition),
        number(prefix + "cap", 0, 64, 0, condition),
        choice(
            prefix + "cap_order",
            orders,
            "recent",
            lambda spec: prefix + "cap" in spec and spec[prefix + "cap"] > 0,
        ),
    ]


def stated(registry):
    started = time.perf_counter()
    nodes, joins = registry.declared, anchoring(registry)

    def anchored(spec, first, second):
        anchors = [kind for kind in nodes if enabled(spec, "derive/complete_within/" + kind)]
        return not anchors or any(first in joins[kind] and second in joins[kind] for kind in anchors)

    emits = dict(
        project=lambda spec, edge: enabled(spec, "derive/" + edge.name),
        complete=lambda spec, edge: (
            enabled(spec, "derive/complete/" + edge.origin[1])
            and enabled(spec, "derive/complete/" + edge.origin[2])
            and anchored(spec, edge.origin[1], edge.origin[2])
        ),
        hop=lambda spec, edge: "derive/hops" in spec and spec["derive/hops"] >= int(edge.origin[1]),
        diffusion=lambda spec, edge: differs("derive/diffusion", "none")(spec),
        rewire=lambda spec, edge: differs("derive/rewiring", "none")(spec),
    )
    result = [choice(DIRECTION, DIRECTIONS, "inverse", relating(registry))]
    for group in registry.groups:
        result += filters(group, None, registry.ordered(group))
    for edge in registry.edges:
        if edge.origin[0] in EMITTED:
            result += filters(edge.name, lambda spec, edge=edge: bool(emits[edge.origin[0]](spec, edge)), ())
    LOG.debug("edges stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


@timed("operators.filterable")
def filterable(registry):
    return [
        name
        for name in registry.coded
        if name in registry.groups or registry.named[name].origin[0] in EMITTED
    ]


def symmetric(registry, spec):
    return np.asarray(
        [edge.symmetric for edge in registry.edges] + [False] * (registry.relations - len(registry.edges))
    ) | holds(spec, DIRECTION, "symmetric")


def filtering(spec, name):
    prefix = "edges/%s/" % name
    return prefix + "drop" in spec and (
        spec[prefix + "drop"] > 0 or spec[prefix + "persistence"] > 1 or spec[prefix + "cap"] > 0
    )


@timed("operators.filtered_names")
def filtered_names(registry, spec):
    return [name for name in filterable(registry) if filtering(spec, name)]


@timed("transform.capped")
def capping(registry, spec):
    return any(
        "edges/%s/cap" % name in spec and spec["edges/%s/cap" % name] > 0 for name in filterable(registry)
    )


def parts(registry, spec):
    return tuple(filtered_names(registry, spec))


def filtering_runs(registry, spec):
    return {"edges/persistence": runs(), "edges/cap": runs()}


def pruning(registry, spec):
    return {"prune/edges": pruned()}


@timed("operators.edges_filter")
def filter_part(graph, registry, spec, seed, limits, carry, held):
    rows = np.flatnonzero(np.isin(graph.fact_code, registry.coded[carry.part]))
    pieces = {"edges/persistence": [], "edges/cap": []}
    kept = filtered(graph, registry, spec, seed, carry.part, rows, carry, pieces)
    for name, found in pieces.items():
        open_runs(carry, name, found)
    return rows, kept


@timed("operators.edges")
def edges(graph, registry, spec, seed, limits, carry, held):
    if not held:
        return graph
    kept = np.ones(len(graph.fact_step), bool)
    for rows, values in held.values():
        kept[rows] = values
    return prune(graph, kept, carry, "edges")


def endpoints(graph, registry, spec, rows):
    code, source, target = graph.fact_code[rows], graph.fact_source[rows], graph.fact_target[rows]
    both = symmetric(registry, spec)[code]
    return (
        code,
        both,
        np.where(both, np.minimum(source, target), source),
        np.where(both, np.maximum(source, target), target),
    )


@timed("operators.keyed_ends")
def keyed_ends(graph, registry, rows, both, carry):
    source, target = carry.tokens(graph.fact_source[rows]), carry.tokens(graph.fact_target[rows])
    return (
        np.where(both, np.minimum(source, target), source),
        np.where(both, np.maximum(source, target), target),
        keyed_relations(registry)[0][graph.fact_code[rows]],
        graph.step_token[graph.fact_step[rows]],
    )


def filtered(graph, registry, spec, seed, name, rows, carry, pieces):
    prefix = "edges/%s/" % name
    step, present = graph.fact_step[rows], graph.fact_present[rows]
    code, both, first, second = endpoints(graph, registry, spec, rows)
    kept = np.ones(len(rows), bool)
    if spec[prefix + "drop"] > 0:
        kept = unit(seed, DROP, *keyed_ends(graph, registry, rows, both, carry)) >= spec[prefix + "drop"]
    stated = np.flatnonzero(kept & present)
    ids, count = kernels.keyed(code[stated], first[stated], second[stated])
    keys = carry.keys(code[stated], first[stated], second[stated])
    (position, _), piece = run_state(carry, "edges/persistence", graph, ids, count, step[stated], keys)
    pieces["edges/persistence"].append(piece)
    lasting = position >= spec[prefix + "persistence"] - 1
    kept[stated[~lasting]] = False
    if spec[prefix + "cap"] == 0:
        return kept
    stated = stated[lasting]
    low, high = first[stated], second[stated]
    ids, count = kernels.keyed(code[stated], low, high)
    keys = carry.keys(code[stated], low, high)
    (_, begun), piece = run_state(carry, "edges/cap", graph, ids, count, step[stated], keys)
    pieces["edges/cap"].append(piece)
    order = spec[prefix + "cap_order"]
    ends = keyed_ends(graph, registry, rows[stated], both[stated], carry)
    if order == "recent":
        primary, secondary = -begun, np.zeros(len(stated))
    elif order == "random":
        primary, secondary = unit(seed, RANDOM, *ends), np.zeros(len(stated))
    else:
        reference = graph.fact_ref[rows[stated]]
        valued = reference >= 0
        value = np.full(len(stated), np.nan, graph.value_table.dtype)
        value[valued] = graph.value_table[
            reference[valued], registry.attributes(name).index(order.split(":", 1)[1])
        ]
        primary, secondary = np.where(np.isnan(value), np.inf, -value), -begun
    pair, count = kernels.keyed(step[stated].astype(np.int64) * registry.relations + code[stated], low, high)
    best = kernels.smallest(pair, primary.astype(np.float64), count)
    earliest = kernels.smallest(pair, secondary.astype(np.float64), count)
    pair_step, pair_code, pair_low, pair_high, mode = (np.zeros(count, np.int64) for _ in range(5))
    pair_step[pair], pair_code[pair], pair_low[pair], pair_high[pair] = step[stated], code[stated], low, high
    mode[pair] = np.where(both[stated], 2, 1 if holds(spec, DIRECTION, "inverse") else 0)
    pair_first, pair_second, pair_relation = (np.zeros(count, np.uint64) for _ in range(3))
    pair_first[pair], pair_second[pair], pair_relation[pair] = ends[0], ends[1], ends[2]
    extra = np.flatnonzero((mode == 1) | ((mode == 2) & (pair_low != pair_high)))
    owners = np.r_[np.arange(count), extra]
    flag = np.r_[np.zeros(count, np.int64), mode[extra] == 1]
    owner = np.r_[pair_high, pair_low[extra]]
    group, groups = kernels.keyed(
        (pair_step[owners] * registry.relations + registry.grouped[pair_code[owners]]) * 2 + flag,
        owner,
        np.zeros(len(owners), np.int64),
    )
    within = kernels.within_cap(
        group,
        groups,
        best[owners],
        earliest[owners],
        pair_first[owners],
        pair_second[owners],
        pair_relation[owners],
        spec[prefix + "cap"],
    )
    capped = np.ones(count, bool)
    capped[owners[~within]] = False
    kept[stated] = capped[pair]
    return kept


EDGES = OPERATIONS.register(
    "graphspec.edges",
    Operation(
        namespace="graphspec",
        name="edges",
        phases=(
            Phase("filter", filter_part, filtering_runs, parts),
            Phase("prune", edges, pruning, None),
        ),
        parameters=stated,
        reads=("graphspec.derive",),
        splits=(),
        requires=(Count("edges/cap", "degree", capping, "stored", "stored"),),
        changes=("facts",),
        adds=None,
        kinds=None,
        version=9,
        graphs=1,
    ),
)
