import functools
import logging
import time

import numpy as np

from graphspec.facts.stored import HALF
from graphspec.infra.hashing import hashed, token
from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.views.graph import unresolved

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice
from .rows import (
    OBSERVATIONS,
    PARALLEL,
    collapsed,
    lived,
    made_rows,
    observed,
    prune,
    pruned,
    reduced,
    select,
)
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
REDUCTIONS = ("mean", "sum", "min", "max")
JOINED = ("base", "member", "pairs", "added")
LACKING = (
    "graphspec.contract groups the nodes of a kind by one of the columns the bank states for it, and the bank %s "
    "states no kind of node with a column"
)


@timed("operators.groupable")
def groupable(registry):
    return tuple("%s:%s" % (kind, column) for kind, columns in registry.schema.nodes.items() for column in columns)


def stated(registry):
    started = time.perf_counter()
    options = groupable(registry)
    result = ()
    if options:
        result = (
            choice("contract/grouping", options, options[0]),
            choice("contract/aggregate", REDUCTIONS, "mean"),
            choice("contract/parallel", PARALLEL, "keep"),
        )
    LOG.debug("contract stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.blocks")
def blocks(registry, column, rows):
    values, flags = registry.encoding.slots(column)
    rows = np.asarray(rows)
    return np.all(rows[:, flags] > 0, axis=1), rows[:, np.r_[values, flags]]


@timed("operators.group_tokens")
def group_tokens(entry, held):
    held = np.ascontiguousarray(np.asarray(held, np.float16))
    flat = held.view(np.uint8).reshape(len(held), held.dtype.itemsize * held.shape[1])
    words = np.zeros((len(held), -(-flat.shape[1] // 8) * 8), np.uint8)
    words[:, : flat.shape[1]] = flat
    return hashed(np.full(len(held), token(entry), np.uint64), *words.view(np.uint64).T)


@timed("operators.grouping_tokens")
def grouping_tokens(entry, kind, column, kinds, rows, registry):
    known, held = blocks(registry, column, np.asarray(rows)[np.asarray(kinds) == registry.kind(kind)])
    return np.unique(group_tokens(entry, held[known]))


@timed("operators.contract_adds")
def adds(registry, settings, entry):
    if not groupable(registry):
        raise ValueError(LACKING % registry.name)
    kind, column = settings["contract/grouping"].split(":")
    joins = {}
    for edge in registry.edges:
        pairs = tuple(
            dict.fromkeys(
                (entry if first == kind else first, entry if second == kind else second)
                for first, second in edge.pairs
                if kind in (first, second)
            )
        )
        if edge.origin[0] in JOINED and pairs:
            joins[edge.name] = pairs
    columns = tuple(registry.schema.entities()[kind])
    tokens = functools.partial(grouping_tokens, entry, kind, column)
    return Added({}, {entry: dict(columns=columns, tokens=tokens)}, {}, {}, joins)


def carried(registry, spec):
    return {"contract/live": lived(), "contract/groups": Table(EMPTY, value=EMPTY), "prune/contract": pruned()}


@timed("operators.grouped_nodes")
def grouped_nodes(graph, registry, carry, entry, kind, column):
    kinds = np.asarray(graph.kinds)
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    members = np.flatnonzero(kinds[entity] == registry.kind(kind))
    known, held = blocks(registry, column, np.asarray(graph.x_table)[np.asarray(graph.obs_ref, np.int64)[members]])
    nodes = np.flatnonzero(kinds == registry.kind(entry))
    tokens = carry.tokens(nodes)
    order = np.argsort(tokens, kind="stable")
    wanted = group_tokens(entry, held)
    place = np.searchsorted(tokens[order], wanted)
    inside = np.flatnonzero(place < len(nodes))
    found = np.zeros(len(wanted), bool)
    found[inside] = tokens[order][place[inside]] == wanted[inside]
    lost = np.flatnonzero(known & ~found)
    if len(lost):
        raise ValueError(
            "graphspec.contract groups the node of token %d at the time %d by its value of %s into a group that no "
            "graph given to the derivation holds; the nodes of a contracted kind are found from those graphs "
            "before the derivation begins, and an edit or an earlier operation brought this value"
            % (carry.tokens(entity[members[lost[:1]]])[0], graph.step_time[step[members[lost[0]]]], column)
        )
    group = np.full(len(members), -1, np.int64)
    group[known] = nodes[order][place[known]]
    return members, group, entity[members] * graph.steps + step[members]


@timed("operators.groups_at")
def groups_at(graph, carry, group, keys, ends, moments, present, holding):
    ends, steps = np.asarray(ends, np.int64), graph.steps
    result, chosen = ends.copy(), np.flatnonzero(holding[ends])
    wanted = ends[chosen] * steps + np.asarray(moments, np.int64)[chosen]
    found = np.full(len(chosen), -1, np.int64)
    order = np.argsort(keys, kind="stable")
    place = np.minimum(np.searchsorted(keys[order], wanted), max(len(keys) - 1, 0))
    stated = np.flatnonzero(np.asarray(present, bool)[chosen])
    if len(keys):
        exact = keys[order][place[stated]] == wanted[stated]
        found[stated[exact]] = group[order][place[stated[exact]]]
    ended = np.flatnonzero(~np.asarray(present, bool)[chosen])
    known = keys[group >= 0]
    order = np.argsort(known, kind="stable")
    place = np.searchsorted(known[order], wanted[ended], "right") - 1
    within = np.zeros(len(ended), bool)
    inside = np.flatnonzero(place >= 0)
    within[inside] = known[order][place[inside]] // steps == ends[chosen][ended][inside]
    found[ended[within]] = group[group >= 0][order][place[within]]
    earlier = carry.state["contract/groups"].lookup("value", carry.glob[ends[chosen][ended[~within]]], -1)
    position = np.minimum(np.searchsorted(carry.glob, np.maximum(earlier, 0)), len(carry.glob) - 1)
    found[ended[~within]] = np.where((earlier >= 0) & (carry.glob[position] == earlier), position, -1)
    result[chosen] = found
    return result


@timed("operators.followed")
def followed(graph, registry, group, keys, holding, kind, column):
    moments = np.asarray(graph.task_step, np.int64)[np.asarray(graph.member_task, np.int64)]
    order = np.argsort(keys, kind="stable")
    result = {}
    for name in ("member_source", "member_target"):
        ends = np.asarray(graph.arrays[name], np.int64)
        named = np.flatnonzero((ends >= 0) & holding[np.maximum(ends, 0)])
        wanted = ends[named] * graph.steps + moments[named]
        place = np.minimum(np.searchsorted(keys[order], wanted), max(len(keys) - 1, 0))
        seen = keys[order][place] == wanted if len(keys) else np.zeros(len(named), bool)
        lost = np.flatnonzero(seen & (group[order][place] < 0))
        if len(lost):
            raise ValueError(
                "%s names a node of the kind %s that holds no value of %s at the time of the task, so "
                "graphspec.contract puts it in no group and the reference cannot follow it"
                % (unresolved(graph, registry, int(named[lost[0]])), kind, column)
            )
        ends = ends.copy()
        ends[named[seen]] = group[order][place[seen]]
        result[name] = ends.astype(graph.arrays[name].dtype)
    return result


@timed("operators.reduced_rows")
def reduced_rows(registry, rows, owner, count, columns, aggregate):
    result = np.zeros((count, registry.features))
    for column in columns:
        values, flags = registry.encoding.slots(column)
        known = np.all(rows[:, flags] > 0, axis=1)
        place, held = owner[known], rows[known][:, values]
        seen = np.bincount(place, minlength=count)
        if aggregate in ("sum", "mean"):
            found = np.zeros((count, len(values)))
            np.add.at(found, place, held)
            if aggregate == "mean":
                found = found / np.maximum(seen, 1)[:, None]
        else:
            found = np.full((count, len(values)), np.inf if aggregate == "min" else -np.inf)
            (np.minimum if aggregate == "min" else np.maximum).at(found, place, held)
            found[seen == 0] = 0.0
        result[:, values] = found
        result[:, flags] = (seen > 0)[:, None]
    return result


@timed("operators.remembered")
def remembered(graph, carry, members, group):
    members, group = members[group >= 0], group[group >= 0]
    entity, step = np.asarray(graph.obs_entity, np.int64)[members], np.asarray(graph.obs_step, np.int64)[members]
    order = np.lexsort((step, entity))
    last = order[np.r_[entity[order][1:] != entity[order][:-1], True]] if len(order) else order
    last = last[np.asarray(graph.entity_connected, bool)[entity[last]]]
    who = carry.glob[entity[last]]
    carry.state["contract/groups"] = carry.state["contract/groups"].replaced(who, who, value=carry.glob[group[last]])


@timed("operators.contract")
def contract(graph, registry, spec, seed, limits, carry, held):
    entry = carry.entry
    kind, column = spec["contract/grouping"].split(":")
    kinds, entity = np.asarray(graph.kinds), np.asarray(graph.obs_entity, np.int64)
    holding = kinds == registry.kind(kind)
    members, group, keys = grouped_nodes(graph, registry, carry, entry, kind, column)
    references = followed(graph, registry, group, keys, holding, kind, column)
    present = np.asarray(graph.fact_present, bool)
    source = groups_at(graph, carry, group, keys, graph.fact_source, graph.fact_step, present, holding)
    target = groups_at(graph, carry, group, keys, graph.fact_target, graph.fact_step, present, holding)
    remembered(graph, carry, members, group)
    touched = holding[graph.fact_source] | holding[graph.fact_target]
    dropped = touched & ((source < 0) | (target < 0))
    base = touched & ~dropped
    pair = carry.keys(graph.fact_code, np.maximum(source, 0), np.maximum(target, 0))
    remains = collapsed(graph, carry, "contract/live", base, pair) & ~dropped
    graph = graph.replace(
        fact_source=np.where(base, source, graph.fact_source).astype(graph.fact_source.dtype),
        fact_target=np.where(base, target, graph.fact_target).astype(graph.fact_target.dtype),
        **references,
    )
    graph = prune(graph, remains, carry, "contract")
    grouped = members[group >= 0]
    pairs, owner = np.unique(
        np.stack((group[group >= 0], np.asarray(graph.obs_step, np.int64)[grouped]), 1).reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    owner = owner.reshape(-1)
    shown = np.asarray(graph.x_table, np.float64)[np.asarray(graph.obs_ref, np.int64)[grouped]]
    columns = registry.schema.entities()[entry]
    rows = reduced_rows(registry, shown, owner, len(pairs), columns, spec["contract/aggregate"])
    with np.errstate(over="ignore"):
        made = rows.astype(graph.x_table.dtype)
    lost = np.flatnonzero(~np.isfinite(made).all(1))
    if len(lost):
        raise ValueError(
            "graphspec.contract reduces the nodes of a group at the time %d to values past the magnitudes up to %d "
            "that model inputs stored at 16 bits hold" % (graph.step_time[pairs[lost[0], 1]], HALF)
        )
    anchored = np.ones(len(pairs), bool)
    np.logical_and.at(anchored, owner, np.asarray(graph.obs_anchor, bool)[grouped])
    graph = select(graph, OBSERVATIONS, ~holding[entity])
    graph, places = made_rows(graph, carry, made)
    graph = observed(graph, pairs[:, 0], pairs[:, 1], places, anchored)
    if spec["contract/parallel"] != "keep":
        joined = registry.kind(entry)
        chosen = (kinds[graph.fact_source] == joined) | (kinds[graph.fact_target] == joined)
        graph = reduced(graph, carry, spec["contract/parallel"], chosen)
    return graph


CONTRACT = OPERATIONS.register(
    "graphspec.contract",
    Operation(
        namespace="graphspec",
        name="contract",
        phases=(Phase("contract", contract, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("observations", "facts"),
        adds=adds,
        version=1,
        graphs=1,
    ),
)
