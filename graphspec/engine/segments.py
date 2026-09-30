import functools
import heapq
import logging
import pickle
import struct
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
from numba import njit

from graphspec import registries
from graphspec.budgets import GraphBudget
from graphspec.infra.files import Arrays
from graphspec.infra.processes.processes import guarded, spawned, worker_threads
from graphspec.infra.timing import measure, timed
from graphspec.operations.derive import (
    DERIVE,
    derivation,
    derive_handoff,
    derive_prepare,
    derive_steps,
    derived_rows,
    rewiring,
)
from graphspec.operations.edges import EDGES, edges_mask, edges_survive, filtered_names
from graphspec.operations.parameters import enabled
from graphspec.operations.rows import FACTS, fact_bytes, prune_mask, select
from graphspec.operations.state import Segment, Table
from graphspec.representations import groups as representations
from graphspec.representations import kernels
from graphspec.representations.groups import (
    EMPTY,
    PERIOD_LOOKBACK,
    base,
    coupling_edges,
    coupling_setting,
    event_boundaries,
    event_edges,
    event_members,
    mixed_features,
    mixed_sizes,
    neighbor_lists,
    neighbor_members,
    neighbors,
    observation_boundaries,
    observed_features,
    periods,
    walk_arrays,
)
from graphspec.views.graph import DENSE, Dense, Graph, verify_dense
from graphspec.views.spans import encode

from . import join, transform

LOG = logging.getLogger(__name__)
ENTITIES = ("obs_entity", "fact_source", "fact_target", "member_source", "member_target")
MOMENTS = ("obs_step", "fact_step", "task_step")
TASKED = ("member_task", "label_task")
GROUP = ("kinds", "entity_token", "entity_connected", "x_table", "value_table")


def table_sizes(graph):
    if "x_table" in graph.arrays:
        return len(graph.x_table), len(graph.value_table)
    return len(graph.obs_x), len(graph.fact_values) + len(graph.removal_values)


def period_groups(period):
    return 1 + int(np.count_nonzero(np.diff(period))) if len(period) else 0


class Plan:
    def __init__(self, graphs, registry, spec):
        started = time.perf_counter()
        self.maps, self.kinds, self.tokens, self.connected = join.unify(graphs, registry, spec)
        steps = np.asarray([graph.steps for graph in graphs], np.int64)
        self.step_offsets = np.r_[0, np.cumsum(steps)]
        self.task_offsets = np.r_[0, np.cumsum([len(graph.task_key) for graph in graphs])]
        self.truth_periods = np.r_[0, np.cumsum([int(graph.period.max(initial=-1)) + 1 for graph in graphs])]
        if spec["time/periods"] == "every":
            self.period_offsets = np.r_[0, np.cumsum(-(-steps // spec["time/period_length"]))]
            self.group_offsets = self.period_offsets
        else:
            self.period_offsets = self.truth_periods
            self.group_offsets = np.r_[0, np.cumsum([period_groups(graph.period) for graph in graphs])]
        sizes = np.asarray([table_sizes(graph) for graph in graphs], np.int64).reshape(-1, 2)
        self.x_offsets = np.r_[0, np.cumsum(sizes[:, 0])]
        self.value_offsets = np.r_[0, np.cumsum(sizes[:, 1])]
        self.seen = np.zeros(len(self.kinds), bool)
        LOG.info(
            "plan exit seconds=%.3f graphs=%d entities=%d steps=%d",
            time.perf_counter() - started,
            len(graphs),
            len(self.kinds),
            int(self.step_offsets[-1]),
        )


def carried_entities(state):
    parts = [np.zeros(0, np.int64)]
    window = state.get("window")
    if window is not None:
        parts += [window["obs"]["entity"], window["facts"]["source"], window["facts"]["target"]]
    live = state.get("derive/live")
    if live is not None:
        parts += [live["source"], live["target"]]
    return np.concatenate([np.asarray(part, np.int64) for part in parts])


def segment(plan, index, graph, state):
    glob = np.union1d(plan.maps[index], carried_entities(state))
    local = np.searchsorted(glob, plan.maps[index])
    arrays = dict(graph.arrays)
    for name in ENTITIES:
        values = np.asarray(arrays[name])
        arrays[name] = np.where(values >= 0, local[np.maximum(values, 0)], -1).astype(values.dtype)
    arrays.update(
        kinds=plan.kinds[glob].astype(np.int32),
        entity_token=plan.tokens[glob],
        entity_connected=plan.connected[glob],
        period=(np.asarray(graph.period) + plan.truth_periods[index]).astype(graph.period.dtype),
        step_graph=np.zeros(graph.steps, np.int32),
    )
    carry = Segment(
        glob,
        len(plan.kinds),
        int(plan.step_offsets[index]),
        int(plan.group_offsets[index]),
        int(plan.period_offsets[index]),
        state,
        plan.x_offsets[index] + np.arange(len(graph.x_table), dtype=np.int64),
        plan.value_offsets[index] + np.arange(len(graph.value_table), dtype=np.int64),
    )
    return carry, Dense(graph.name, [], graph.steps, **arrays)


def derive_segment(plan, index, graph, state, registry, spec, limits):
    started = time.perf_counter()
    with measure("segment"):
        carry, dense = segment(plan, index, graph, state)
    known = plan.seen[carry.glob]
    plan.seen[carry.glob[dense.obs_entity]] = True
    result = transform.derive(dense, registry, spec, limits, carry, known)
    plan.seen[carry.glob[result.obs_entity]] = True
    LOG.info(
        "segment exit index=%d seconds=%.3f entities=%d facts=%d",
        index,
        time.perf_counter() - started,
        len(carry.glob),
        len(result.fact_step),
    )
    return carry, result


def derived(plan, graphs, registry, spec, limits):
    started = time.perf_counter()
    state = {}
    outputs = [
        derive_segment(plan, index, graph, state, registry, spec, limits)
        for index, graph in enumerate(graphs)
    ]
    LOG.info("derived exit seconds=%.3f segments=%d", time.perf_counter() - started, len(outputs))
    return outputs


def globalized(plan, carry, index, name, values):
    values = np.asarray(values)
    if name in ENTITIES:
        return np.where(values >= 0, carry.glob[np.maximum(values, 0)], -1).astype(values.dtype)
    if name in MOMENTS:
        return (values + carry.step0).astype(values.dtype)
    if name in TASKED:
        return (values + plan.task_offsets[index]).astype(values.dtype)
    if name == "obs_ref":
        return carry.x_global[values].astype(values.dtype)
    if name == "fact_ref":
        table = np.r_[carry.value_global, -1]
        return np.where(values >= 0, table[np.maximum(values, 0)], -1).astype(values.dtype)
    if name == "step_graph":
        return np.full(len(values), index, values.dtype)
    return values


def stitch(plan, graphs, outputs):
    started = time.perf_counter()
    columns = {
        name: np.concatenate(
            [
                globalized(plan, carry, index, name, dense.arrays[name])
                for index, (carry, dense) in enumerate(outputs)
            ]
        )
        for name in DENSE
        if name not in GROUP
    }
    result = Dense(
        "joined",
        [],
        int(plan.step_offsets[-1]),
        kinds=plan.kinds.astype(np.int32),
        entity_token=plan.tokens,
        entity_connected=plan.connected,
        x_table=np.concatenate([np.asarray(graph.x_table) for graph in graphs]),
        value_table=np.concatenate([np.asarray(graph.value_table) for graph in graphs]),
        **columns,
    )
    LOG.info("stitch exit seconds=%.3f segments=%d", time.perf_counter() - started, len(outputs))
    return result


def mapped(glob, values):
    values = np.asarray(values)
    return np.where(values >= 0, glob[np.maximum(values, 0)], -1).astype(np.int32)


def localized(*columns):
    values, inverse = np.unique(
        np.concatenate([np.asarray(column, np.int64) for column in columns] + [EMPTY]), return_inverse=True
    )
    parts, offset = [], 0
    for column in columns:
        parts.append(inverse[offset : offset + len(column)].astype(np.int64))
        offset += len(column)
    return values, parts


def resolved(values, count, halo, base):
    values = np.asarray(values, np.int64)
    if not count:
        return base + values - count
    return np.where(values < count, halo[np.clip(values, 0, count - 1)], base + values - count)


class Link:
    def __init__(self, carry, state, obs0, attributes):
        self.glob, self.entities, self.state, self.step0 = carry.glob, carry.entities, state, carry.step0
        self.obs0, self.attributes = obs0, attributes
        self.before = self.table("last")

    def table(self, name):
        return self.state.get(name, Table())

    def replay_keys(self, graph):
        table = self.table("replay")
        if not len(table):
            return (EMPTY,) * 4
        found = table.find(
            encode(graph.fact_code, self.glob[graph.fact_source], self.glob[graph.fact_target], self.entities)
        )
        chosen = np.unique(found[found >= 0])
        return (
            np.searchsorted(self.glob, table.columns["source"][chosen]),
            np.searchsorted(self.glob, table.columns["target"][chosen]),
            table.columns["code"][chosen],
            table.columns["state"][chosen],
        )

    def remember(self, graph, entity, reference, table):
        last = np.full(len(graph.kinds), -1, np.int64)
        np.maximum.at(last, entity, np.arange(len(entity)))
        chosen = np.flatnonzero((last >= 0) & graph.entity_connected)
        keys = self.glob[chosen]
        self.state["last"] = self.table("last").replaced(
            keys, keys, row=self.obs0 + last[chosen], x=np.asarray(graph.x_table)[reference[last[chosen]]]
        )
        source, target, code, state = (np.asarray(values, np.int64) for values in table)
        both = graph.entity_connected[source] & graph.entity_connected[target]
        left, right = self.glob[source[both]], self.glob[target[both]]
        keys = encode(code[both], left, right, self.entities)
        self.state["replay"] = self.table("replay").replaced(
            keys, keys, source=left, target=right, code=code[both], state=state[both]
        )


class Context:
    def __init__(self, plan, spec, groups):
        self.steps = int(plan.step_offsets[-1])
        self.obs0 = self.events0 = self.nodes0 = self.holders0 = self.periods = 0
        self.tasks0 = self.rows0 = 0
        self.attributes = {}
        self.period_starts, self.period_ends = [EMPTY], [EMPTY]
        self.history = []
        horizon, lookback = 0, 0
        if "coupling" in groups:
            lookback = max(lookback, spec["coupling/periods"])
        if "events" in groups:
            horizon = max(horizon, spec["events/window"] // 2 + 2)
            lookback = max(lookback, spec["events/periods"])
        if "walks" in groups:
            horizon, lookback = max(horizon, spec["walks/window"] + 2), max(lookback, PERIOD_LOOKBACK)
        self.horizon, self.period_horizon = horizon, lookback + 1

    def remember(self, piece, next_step):
        self.history.append(piece)
        self.history = [
            item
            for item in self.history
            if item["last_step"] >= next_step - self.horizon
            or item["last_period"] >= self.periods - self.period_horizon
        ]

    def recent(self, name, empty):
        parts = [item[name] for item in self.history]
        if not parts:
            return empty
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def tail_length(spec):
    chosen, mode, width, cap, limit = coupling_setting(spec, 0, 0)
    if mode == 0:
        return None if spec["coupling/scheme"] == "all" else cap
    return int(chosen.max())


def coupled_segment(own, spec, context, state, budget):
    entity, step = own["entity"].astype(np.int64), own["step"].astype(np.int64)
    held = state.get("coupling/tail", dict(entity=EMPTY, row=EMPTY, step=EMPTY))
    halo = np.flatnonzero(np.isin(held["entity"], entity))
    count = len(halo)
    values, (ext_entity,) = localized(np.r_[held["entity"][halo], entity])
    source, target = coupling_edges(
        ext_entity,
        np.r_[held["step"][halo], step],
        spec,
        context.steps,
        np.int64(1) << 60,
        len(values),
        budget,
    )
    kept = target >= count
    source = resolved(source[kept], count, held["row"][halo], context.obs0)
    target = context.obs0 + target[kept].astype(np.int64) - count
    budget.retain(len(source), "temporal coupling")
    connected = own["connected"]
    merged_entity = np.r_[held["entity"], entity[connected]]
    merged_row = np.r_[held["row"], context.obs0 + np.flatnonzero(connected)]
    merged_step = np.r_[held["step"], step[connected]]
    order = np.lexsort((merged_row, merged_entity))
    merged_entity, merged_row, merged_step = merged_entity[order], merged_row[order], merged_step[order]
    limit = tail_length(spec)
    if limit is not None and len(order):
        ends = np.r_[np.flatnonzero(merged_entity[1:] != merged_entity[:-1]), len(order) - 1]
        from_end = np.repeat(ends, np.diff(np.r_[-1, ends])) - np.arange(len(order))
        keep = from_end < limit
        merged_entity, merged_row, merged_step = merged_entity[keep], merged_row[keep], merged_step[keep]
    state["coupling/tail"] = dict(entity=merged_entity, row=merged_row, step=merged_step)
    return np.stack((source, target), 1).astype(np.int32).reshape(-1, 2)


EMPTY_OBS = dict(entity=EMPTY, step=EMPTY, group=EMPTY, period=EMPTY, row=EMPTY)
EMPTY_EVENTS = dict(
    events=np.zeros((0, 9), np.int32), id=EMPTY, group=EMPTY, period=EMPTY, member=np.zeros(0, bool)
)


def boundary_segment(own, lookback, window, registry, context, budget):
    recent = context.recent("obs", EMPTY_OBS)
    shift = max(context.periods - lookback, 0)
    halo = np.flatnonzero((recent["group"] >= shift) & np.isin(recent["entity"], own["entity"]))
    count = len(halo)
    values, (ext_entity,) = localized(np.r_[recent["entity"][halo], own["entity"]])
    result = observation_boundaries(
        ext_entity,
        np.r_[recent["step"][halo], own["step"]].astype(np.int64),
        np.r_[recent["group"][halo], own["group"]].astype(np.int64) - shift,
        np.r_[recent["period"][halo], own["obs_period"]],
        lookback,
        window,
        len(values),
        context.periods + own["periods"] - shift,
        registry,
        budget,
    ).astype(np.int64)
    result = result[result[:, 1] >= count]
    result[:, 0] = resolved(result[:, 0], count, recent["row"][halo], context.obs0)
    result[:, 1] = context.obs0 + result[:, 1] - count
    return result.astype(np.int32)


def extended(recent, halo, events):
    ext = np.r_[recent["events"][halo], events].astype(np.int32)
    values, (left, right) = localized(ext[:, 0], ext[:, 1])
    ext[:, 0], ext[:, 1] = left, right
    return values, ext


def event_segment(own, spec, registry, context, budget):
    events, member = own["events"], own["member"]
    recent = context.recent("events", EMPTY_EVENTS)
    halo = np.flatnonzero(
        recent["member"] & (recent["events"][:, 2] >= 2 * own["first_step"] - spec["events/window"])
    )
    count = len(halo)
    values, ext = extended(recent, halo, events)
    before, after = event_edges(ext, np.r_[recent["member"][halo], member], spec, len(values), budget)
    kept = after >= count
    before = resolved(before[kept], count, recent["id"][halo], context.events0)
    after = context.events0 + after[kept].astype(np.int64) - count
    budget.retain(len(before), "event graph")
    result = dict(
        event_dag=np.stack((before, after), 1).astype(np.int32).reshape(-1, 2),
        event_member=member,
        event_boundaries=np.zeros((0, 4), np.int32),
    )
    lookback = spec["events/periods"]
    if lookback:
        shift = max(context.periods - lookback, 0)
        halo = np.flatnonzero(recent["group"] >= shift)
        count = len(halo)
        values, ext = extended(recent, halo, events)
        edges = event_boundaries(
            ext,
            np.r_[recent["member"][halo], member],
            np.r_[recent["group"][halo], own["event_group"]].astype(np.int64) - shift,
            np.r_[recent["period"][halo], own["event_period"]],
            lookback,
            spec["events/period_window"],
            len(values),
            registry,
            budget,
        ).astype(np.int64)
        edges = edges[edges[:, 1] >= count]
        edges[:, 0] = resolved(edges[:, 0], count, recent["id"][halo], context.events0)
        edges[:, 1] = context.events0 + edges[:, 1] - count
        result["event_boundaries"] = edges.astype(np.int32)
    return result


def walk_segment(own, spec, seed, context, budget):
    local = own["member_row"].astype(np.int64) - context.rows0
    moments = own["row_step"].astype(np.int64)[local]
    roots = own["member_entity"].astype(np.int64)
    first = np.concatenate(context.period_starts + [own["period_starts"]])
    last = np.concatenate(context.period_ends + [own["period_ends"]])
    prior = own["row_group"][local][:, None].astype(np.int64) - np.arange(1, PERIOD_LOOKBACK + 1)
    floor = np.inf
    if len(moments):
        floor = min(floor, int(np.maximum(0, (moments - spec["walks/window"] + 1) * 2).min()))
    if (prior >= 0).any():
        floor = min(floor, int(2 * first[prior[prior >= 0]].min()))
    recent = context.recent("events", EMPTY_EVENTS)
    halo = np.flatnonzero(neighbor_members(recent["events"], spec) & (recent["events"][:, 2] >= floor))
    count = len(halo)
    ext = np.r_[recent["events"][halo], own["events"]].astype(np.int32)
    values, (left, right, local_roots) = localized(ext[:, 0], ext[:, 1], roots.reshape(-1))
    ext[:, 0], ext[:, 1] = left, right
    ids, offsets = neighbor_lists(ext, neighbor_members(ext, spec), len(values))
    result = walk_arrays(
        ids,
        offsets,
        ext,
        local_roots.reshape(roots.shape),
        roots,
        moments,
        prior,
        first[prior.clip(0)].astype(np.int64),
        last[prior.clip(0)].astype(np.int64),
        spec,
        seed,
        budget,
    )
    for held in ("walks", "period_walks"):
        walked = result[held].astype(np.int64)
        result[held] = np.where(
            walked < 0, -1, resolved(walked, count, recent["id"][halo], context.events0)
        ).astype(np.int32)
    return result


class Shifted:
    def __init__(self, table, base):
        self.table, self.base = table, base

    def __getitem__(self, rows):
        return self.table[np.asarray(rows, np.int64) - self.base]


def mixed_segment(graph, arrays, link, spec, context, state, budget, values):
    entity, offsets = arrays["entity"].astype(np.int64), arrays["snapshot_offsets"]
    x, glob = observed_features(arrays), link.glob
    changes, trace, bundles = spec["mixed/changes"], spec["mixed/trace"], spec["mixed/bundles"]
    rows, moments, entities = len(entity), len(offsets) - 1, len(graph.kinds)
    placed = np.searchsorted(np.asarray(arrays["event_step"], np.int64), np.arange(moments + 1))
    order = np.argsort(entity, kind="stable")
    prior = np.full(rows, -1, np.int64)
    heads = order
    if rows:
        same = entity[order][1:] == entity[order][:-1]
        prior[order[1:][same]] = context.obs0 + order[:-1][same]
        heads = order[np.r_[True, ~same]]
    found = link.before.find(glob[entity[heads]])
    known = found >= 0
    extra_rows, extra_x = [EMPTY], [np.zeros((0, x.shape[1]), x.dtype)]
    if known.any():
        prior[heads[known]] = link.before.columns["row"][found[known]]
        extra_rows.append(link.before.columns["row"][found[known]])
        extra_x.append(link.before.columns["x"][found[known]])
    span = trace if trace > 0 else 1
    last_node = np.full(entities, -1, np.int64)
    trace_node = np.zeros((entities, span), np.int64)
    trace_row = np.zeros((entities, span), np.int64)
    trace_count = np.zeros(entities, np.int64)
    nodes = state.get("mixed/nodes", Table())
    found = nodes.find(glob)
    use = np.flatnonzero(found >= 0)
    if len(use):
        where = found[use]
        last_node[use] = nodes.columns["last"][where]
        trace_count[use] = nodes.columns["count"][where]
        trace_node[use] = nodes.columns["node"][where]
        trace_row[use] = nodes.columns["row"][where]
        filled = np.arange(span)[None, :] < trace_count[use][:, None]
        extra_rows.append(trace_row[use][filled])
        extra_x.append(nodes.columns["x"][where][filled])
    carried_rows, carried_x = np.concatenate(extra_rows), np.concatenate(extra_x)
    arrangement = np.argsort(carried_rows, kind="stable")
    carried_rows, carried_x = carried_rows[arrangement], carried_x[arrangement]

    def lookup(ids):
        ids = np.asarray(ids, np.int64)
        local = ids - context.obs0
        inside = (local >= 0) & (local < rows)
        result = np.empty((len(ids), x.shape[1]), x.dtype)
        result[inside] = x[local[inside]]
        outside = ids[~inside]
        if len(outside):
            position = np.minimum(np.searchsorted(carried_rows, outside), max(len(carried_rows) - 1, 0))
            if not len(carried_rows) or np.any(carried_rows[position] != outside):
                raise ValueError("a mixed representation row is missing from its carry")
            result[~inside] = carried_x[position]
        return result

    altered = np.zeros(rows, bool)
    if changes:
        seen = np.flatnonzero(prior >= 0)
        altered[seen] = np.any(x[seen] != lookup(prior[seen]), axis=1)
    limit, wires, holders = mixed_sizes(
        rows, moments, trace, changes, bundles, int(placed[-1]), int(trace_count.sum())
    )
    budget.pairs(wires, "mixed representation edges")
    budget.workspace(limit * (32 + x.shape[1] * 2) + wires * 12 + holders * 4, "mixed representation")
    kind, tick, owner, mode, fill, member_offset, member_id, source, target, flag, action = kernels.bundled(
        np.ascontiguousarray(entity, np.int64),
        np.ascontiguousarray(offsets, np.int64),
        prior,
        altered,
        np.asarray(arrays["event_first"], np.int64),
        np.asarray(arrays["event_task"], np.int64) + context.tasks0,
        placed,
        trace,
        bundles,
        limit,
        wires,
        holders,
        np.int64(context.nodes0),
        np.int64(context.obs0),
        last_node,
        trace_node,
        trace_row,
        trace_count,
        np.int64(state.get("mixed/action", -1)),
    )
    features = mixed_features(
        kind,
        mode,
        fill,
        member_offset,
        member_id,
        lookup,
        x.shape[1],
        values,
        Shifted(arrays["task"], context.tasks0),
        offsets,
        spec,
        context.obs0,
    )
    state["mixed/action"] = int(action)
    chosen = np.flatnonzero(graph.entity_connected & ((last_node >= 0) | (trace_count > 0)))
    filled = np.arange(span)[None, :] < trace_count[chosen][:, None]
    trace_x = np.zeros((len(chosen), span, x.shape[1]), x.dtype)
    if filled.any():
        trace_x[filled] = lookup(trace_row[chosen][filled])
    keys = glob[chosen]
    state["mixed/nodes"] = nodes.replaced(
        keys,
        keys,
        last=last_node[chosen],
        count=trace_count[chosen],
        node=trace_node[chosen],
        row=trace_row[chosen],
        x=trace_x,
    )
    return dict(
        mixed_x=features.reshape(-1, x.shape[1]),
        mixed_kind=kind,
        mixed_tick=(tick.astype(np.int64) + 2 * link.step0).astype(np.int32),
        mixed_entity=mapped(glob, owner),
        mixed_edges=np.stack((source, target, flag), 1).astype(np.int32).reshape(-1, 3),
        mixed_member_ids=member_id,
        mixed_member_offsets=member_offset[:-1] + context.holders0,
    )


@timed("segments.shifted")
def shifted(values, base):
    values = np.asarray(values).astype(np.int64)
    return np.where(values >= 0, values + base, -1).astype(np.int32)


def own_arrays(arrays, carry, context, graph, spec):
    glob, step0, obs0 = carry.glob, carry.step0, context.obs0
    events = np.asarray(arrays["events"]).astype(np.int64)
    moments = events[:, 2] // 2
    events[:, 0], events[:, 1] = glob[events[:, 0]], glob[events[:, 1]]
    events[:, 2] += 2 * step0
    events[events[:, 3] == 2, 7] += context.tasks0
    spatial = np.asarray(arrays["spatial"]).astype(np.int64)
    spatial[:, :2] += obs0
    period = np.asarray(arrays["period"])
    first = np.r_[0, np.flatnonzero(np.diff(period)) + 1]
    last = np.r_[first[1:] - 1, len(period) - 1]
    membership = np.repeat(np.arange(len(first)), last - first + 1) + context.periods
    step = np.asarray(arrays["step"])
    own = dict(
        x_ref=carry.x_global[arrays["x_ref"]].astype(np.int32),
        entity=glob[arrays["entity"]].astype(np.int32),
        step=(step + step0).astype(np.int32),
        obs_period=arrays["obs_period"],
        spatial=spatial.astype(np.int32),
        spatial_attribute_row=arrays["spatial_attribute_row"],
        spatial_step=(arrays["spatial_step"] + step0).astype(np.int32),
        task=arrays["task"],
        event_first=mapped(glob, arrays["event_first"].astype(np.int64)),
        event_task=(arrays["event_task"] + context.tasks0).astype(np.int32),
        event_step=(arrays["event_step"] + step0).astype(np.int32),
        row_step=(arrays["row_step"] + step0).astype(np.int32),
        row_task=(arrays["row_task"] + context.tasks0).astype(np.int32),
        row_kind=arrays["row_kind"],
        events=events.astype(np.int32),
        signals=arrays["signals"],
        row_index=arrays["row_index"],
        key=arrays["key"],
        period=arrays["period"],
        member_row=(arrays["member_row"] + context.rows0).astype(np.int32),
        member_name=arrays["member_name"],
        member_entity=mapped(glob, arrays["member_entity"]),
        member_observed=shifted(arrays["member_observed"], obs0),
        member_place=arrays["member_place"],
        whole_row=(arrays["whole_row"] + context.rows0).astype(np.int32),
        whole_name=arrays["whole_name"],
        **{name: value for name, value in arrays.items() if name.startswith("attributes_")},
    )
    extra = dict(
        connected=graph.entity_connected[arrays["entity"]],
        group=membership[step],
        event_group=membership[moments],
        event_period=period[moments],
        periods=len(first),
        period_starts=(first + step0).astype(np.int64),
        period_ends=(last + step0).astype(np.int64),
        row_group=membership[np.asarray(arrays["row_step"])],
        first_step=step0,
        member=np.zeros(len(events), bool),
    )
    return own, extra


def represent_segment(carry, graph, registry, spec, groups, seed, limits, state, context):
    started = time.perf_counter()
    budget = GraphBudget(limits)
    link = Link(carry, state, context.obs0, dict(context.attributes))
    arrays = base(graph, registry, spec, link)
    own, extra = own_arrays(arrays, carry, context, graph, spec)
    work = dict(own, **extra)
    piece = dict(own)
    if "coupling" in groups:
        piece["coupling"] = coupled_segment(work, spec, context, state, budget)
        piece["coupling_boundaries"] = (
            boundary_segment(
                work, spec["coupling/periods"], spec["coupling/period_window"], registry, context, budget
            )
            if spec["coupling/periods"]
            else np.zeros((0, 4), np.int32)
        )
    if "events" in groups:
        work["member"] = event_members(own["events"], spec)
        piece.update(event_segment(work, spec, registry, context, budget))
    if "walks" in groups:
        piece.update(walk_segment(work, spec, seed, context, budget))
    if "mixed" in groups:
        piece.update(mixed_segment(graph, arrays, link, spec, context, state, budget, registry.values))
    rows, count = len(own["entity"]), len(own["events"])
    context.remember(
        dict(
            obs=dict(
                entity=own["entity"].astype(np.int64),
                step=own["step"].astype(np.int64),
                group=work["group"].astype(np.int64),
                period=own["obs_period"],
                row=context.obs0 + np.arange(rows, dtype=np.int64),
            ),
            events=dict(
                events=own["events"],
                id=context.events0 + np.arange(count, dtype=np.int64),
                group=work["event_group"].astype(np.int64),
                period=work["event_period"],
                member=work["member"] if "events" in groups else np.zeros(count, bool),
            ),
            last_step=carry.step0 + graph.steps - 1,
            last_period=context.periods + work["periods"] - 1,
        ),
        carry.step0 + graph.steps,
    )
    context.period_starts.append(work["period_starts"])
    context.period_ends.append(work["period_ends"])
    for group in registry.valued:
        context.attributes[group] = context.attributes.get(group, 0) + len(own["attributes_" + group])
    context.obs0 += rows
    context.events0 += count
    context.periods += work["periods"]
    context.tasks0 += len(own["task"])
    context.rows0 += len(own["row_step"])
    if "mixed" in groups:
        context.nodes0 += len(piece["mixed_kind"])
        context.holders0 += len(piece["mixed_member_ids"])
    LOG.info(
        "segment representation exit seconds=%.3f rows=%d events=%d",
        time.perf_counter() - started,
        rows,
        count,
    )
    return piece


def represented(plan, outputs, registry, spec, groups, seed, limits):
    started = time.perf_counter()
    state, context = {}, Context(plan, spec, groups)
    pieces = [
        represent_segment(carry, graph, registry, spec, groups, seed, limits, state, context)
        for carry, graph in outputs
    ]
    tables = [
        np.asarray(graph.x_table)[: plan.x_offsets[index + 1] - plan.x_offsets[index]]
        for index, (_, graph) in enumerate(outputs)
    ]
    names = [name for name in pieces[0] if name != "mixed_member_offsets"]
    arrays = {name: np.concatenate([piece[name] for piece in pieces]) for name in names}
    if "mixed" in groups:
        arrays["mixed_member_offsets"] = np.concatenate(
            [piece["mixed_member_offsets"] for piece in pieces] + [np.asarray([context.holders0], np.int64)]
        )
    result = stitched_arrays(plan, arrays, tables, registry, spec, groups, limits, context)
    LOG.info("represented exit seconds=%.3f segments=%d", time.perf_counter() - started, len(pieces))
    return result


HEADER = 128


def header(dtype, shape):
    text = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (
        np.lib.format.dtype_to_descr(dtype),
        shape,
    )
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", HEADER - 10)
        + (text.ljust(HEADER - 11) + "\n").encode("latin1")
    )


class Appender:
    def __init__(self, directory):
        self.directory, self.streams, self.kinds, self.rows = Path(directory), {}, {}, {}

    def add(self, piece):
        for name, values in piece.items():
            if name not in self.streams:
                values = np.asarray(values)
                self.kinds[name], self.rows[name] = (values.dtype, values.shape[1:]), 0
                self.streams[name] = open(self.directory / (name + ".npy"), "wb")
                self.streams[name].write(bytes(HEADER))
            dtype, trailing = self.kinds[name]
            if np.asarray(values).dtype != dtype or np.shape(values)[1:] != trailing:
                raise ValueError("segment array %s changed dtype or shape" % name)
            values = np.ascontiguousarray(values)
            self.rows[name] += len(values)
            if values.size:
                self.streams[name].write(values.data)

    def close(self):
        for name, stream in self.streams.items():
            dtype, trailing = self.kinds[name]
            stream.seek(0)
            stream.write(header(dtype, (self.rows[name],) + trailing))
            stream.close()
        self.streams = {}


def stitched_arrays(plan, arrays, tables, registry, spec, groups, limits, context, destination=None):
    started = time.perf_counter()
    arrays = dict(arrays)
    table = np.concatenate([np.asarray(table).astype(np.float16, copy=False) for table in tables])
    written = set(arrays)
    kinds = plan.kinds.astype(np.int32)
    entity, step = arrays["entity"], arrays["step"]
    observation_ids, observation_offsets = grouped(np.asarray(entity), len(kinds))
    arrays.update(
        x_table=table,
        entity_token=(plan.tokens & np.uint64((1 << 62) - 1)).astype(np.int64),
        entity_kind=(np.zeros_like(kinds) if enabled(spec, "nodes/untyped") else kinds).astype(np.int32),
        snapshot_offsets=np.searchsorted(step, np.arange(context.steps + 1)).astype(np.int64),
        observation_ids=observation_ids,
        observation_offsets=observation_offsets,
    )
    budget = GraphBudget(limits)
    arrays.update(periods(arrays, len(registry.events), budget))
    if "neighbors" in groups or "walks" in groups:
        arrays.update(neighbors(arrays, spec))
    if destination is not None:
        for name in sorted(set(arrays) - written):
            np.save(Path(destination) / (name + ".npy"), arrays[name], allow_pickle=False)
    LOG.info("stitched exit seconds=%.3f arrays=%d", time.perf_counter() - started, len(arrays))
    return arrays


def finished(appender, context, groups):
    if "mixed" in groups:
        appender.add(dict(mixed_member_offsets=np.asarray([context.holders0], np.int64)))
    appender.close()


def build(members, registry, spec, groups, seed, spill, limits, destination=None):
    started = time.perf_counter()
    plan = Plan(members, registry, spec)
    transform_state, representation_state, context = {}, {}, Context(plan, spec, groups)
    folder = Path(spill if destination is None else destination)
    appender = Appender(folder)
    for index, member in enumerate(members):
        clock = time.perf_counter()
        with measure("dense"):
            dense = member.dense()
        carry, graph = derive_segment(plan, index, dense, transform_state, registry, spec, limits)
        with measure("represent"):
            piece = represent_segment(
                carry, graph, registry, spec, groups, seed, limits, representation_state, context
            )
        with measure("spill"):
            appender.add(piece)
        LOG.info("segment build exit index=%d seconds=%.3f", index, time.perf_counter() - clock)
    finished(appender, context, groups)
    with measure("stitch"):
        arrays = stitched_arrays(
            plan,
            Arrays(folder),
            [member.obs_x for member in members],
            registry,
            spec,
            groups,
            limits,
            context,
            destination,
        )
    representations.validate(arrays, SimpleNamespace(name="joined"), registry, spec, groups)
    LOG.info(
        "segments build exit seconds=%.3f segments=%d observations=%d events=%d",
        time.perf_counter() - started,
        len(members),
        context.obs0,
        context.events0,
    )
    return None if destination is not None else {name: np.array(values) for name, values in arrays.items()}


PIPE = {}
CHAIN = ("window", "nodes", "finish", "merging", "represent")
SHARDS = 3
RANK = dict(stitch=0, assemble=1, steps=2, expand=3)


def attach(jobs, registry, spec, groups, seed, limits, threads):
    numba.set_num_threads(threads)
    PIPE.clear()
    deriving, filtering = DERIVE.settings(registry, spec), EDGES.settings(registry, spec)
    PIPE.update(
        jobs=[
            dict(
                plan=plan,
                paths=paths,
                spill=Path(spill),
                destination=destination,
                states={},
                context=Context(plan, spec, groups),
            )
            for plan, paths, spill, destination in jobs
        ],
        registry=registry,
        spec=spec,
        groups=groups,
        seed=seed,
        limits=limits,
        deriving=deriving,
        edges=filtering,
        filtering=bool(filtered_names(registry, filtering)),
    )


def staged(function):
    @functools.wraps(function)
    def run(job, *arguments):
        started = time.perf_counter()
        result = function(job, *arguments) or {}
        seconds = time.perf_counter() - started
        LOG.info("%s exit job=%d index=%s seconds=%.3f", function.__name__, job, arguments[0], seconds)
        return dict(result, seconds=seconds)

    return run


def state_of(job, stage, index):
    version, state = PIPE["jobs"][job]["states"].get(stage, (0, {}))
    if version != index:
        version, state = peek(job, 0, "state-" + stage)
    if version != index:
        raise RuntimeError("segment stage %s received state %d for segment %d" % (stage, version, index))
    return state


def keep(job, stage, index, state, floating=True):
    PIPE["jobs"][job]["states"][stage] = (index + 1, state)
    if floating:
        park(job, 0, "state-" + stage, (index + 1, state))


def folder(job, index, stage):
    return PIPE["jobs"][job]["spill"] / ("%s_%05d.pickle" % (stage, index))


def park(job, index, stage, value):
    with open(folder(job, index, stage), "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def peek(job, index, stage):
    with open(folder(job, index, stage), "rb") as stream:
        return pickle.load(stream)


def taken(job, index, stage):
    value = peek(job, index, stage)
    folder(job, index, stage).unlink()
    return value


def packed(carry, graph, known, **extra):
    return dict(
        graph=(graph.name, graph.steps, graph.arrays),
        carry=(carry.glob, carry.step0, carry.group0, carry.period0, carry.x_global, carry.value_global),
        entities=carry.entities,
        known=known,
        **extra,
    )


def carried(value, state):
    glob, step0, group0, period0, x_global, value_global = value["carry"]
    return Segment(glob, value["entities"], step0, group0, period0, state, x_global, value_global)


def unpacked(value, state):
    name, steps, arrays = value["graph"]
    return Dense(name, [], steps, **arrays), carried(value, state), value["known"]


def relabelled(plan, graph, glob, position):
    if len(glob) == len(position):
        return graph
    arrays = {
        name: np.where(values >= 0, position[np.maximum(values, 0)], -1).astype(values.dtype)
        for name, values in ((name, np.asarray(graph.arrays[name])) for name in ENTITIES)
    }
    arrays.update(
        kinds=plan.kinds[glob].astype(np.int32),
        entity_token=plan.tokens[glob],
        entity_connected=plan.connected[glob],
    )
    return graph.replace(**arrays)


@staged
def expand_task(job, index):
    member = Graph.load(PIPE["jobs"][job]["paths"][index], mapped=True).dense()
    park(job, index, "expanded", (member.name, member.steps, member.arrays))


@staged
def window_task(job, index):
    registry, spec, limits, plan = PIPE["registry"], PIPE["spec"], PIPE["limits"], PIPE["jobs"][job]["plan"]
    name, steps, arrays = taken(job, index, "expanded")
    state = state_of(job, "window", index)
    carry, dense = segment(plan, index, Dense(name, [], steps, **arrays), state)
    seen = state.get("pipe/seen", EMPTY)
    known = np.isin(carry.glob, seen)
    graph = transform.derive(dense, registry, spec, limits, carry, known, transform.STAGES[:2], False)
    observed = np.zeros(len(carry.glob), bool)
    observed[dense.obs_entity] = True
    observed[graph.obs_entity] = True
    state["pipe/seen"] = np.union1d(seen, carry.glob[observed & graph.entity_connected])
    park(job, index, "window", packed(carry, graph, known))
    keep(job, "window", index, state)


@staged
def nodes_task(job, index):
    registry, spec, limits = PIPE["registry"], PIPE["spec"], PIPE["limits"]
    state = state_of(job, "nodes", index)
    graph, carry, known = unpacked(taken(job, index, "window"), state)
    graph = transform.derive(graph, registry, spec, limits, carry, known, transform.STAGES[2:3], False)
    deriving = PIPE["deriving"]
    recency = None
    if derivation(registry, deriving) is not None:
        recency = derive_prepare(graph, registry, deriving, carry)
    value = packed(carry, graph, known, recency=recency)
    park(job, index, "nodes", value)
    park(job, index, "meta", dict(value, graph=None, recency=None, steps=graph.steps))
    keep(job, "nodes", index, state)


@staged
def steps_task(job, index):
    value = peek(job, index, "nodes")
    graph, carry, _ = unpacked(value, {})
    found, removed, budget = derive_steps(
        graph,
        PIPE["registry"],
        PIPE["deriving"],
        transform.SEED,
        value["recency"],
        carry.step0,
        PIPE["limits"],
    )
    park(job, index, "steps", (found, removed, budget.facts))


@staged
def finish_task(job, index):
    registry, deriving, plan = PIPE["registry"], PIPE["deriving"], PIPE["jobs"][job]["plan"]
    state = state_of(job, "finish", index)
    meta = taken(job, index, "meta")
    carry = carried(meta, state)
    live = state.get("derive/live")
    glob = np.union1d(carry.glob, EMPTY if live is None else np.r_[live["source"], live["target"]])
    position = np.searchsorted(glob, carry.glob)
    known = np.ones(len(glob), bool)
    known[position] = meta["known"]
    carry.glob = glob
    result = dict(glob=glob, position=position, known=known, deriving=False, kept=None, found=None)
    if derivation(registry, deriving) is not None:
        (source, target, code, step), removed, facts = taken(job, index, "steps")
        source, target = position[source].astype(source.dtype), position[target].astype(target.dtype)
        connected = plan.connected[glob]
        result.update(deriving=True)
        if rewiring(deriving):
            graph = relabelled(plan, unpacked(peek(job, index, "nodes"), {})[0], glob, position)
            result.update(
                kept=prune_mask(
                    graph.fact_source,
                    graph.fact_target,
                    graph.fact_code,
                    graph.fact_present,
                    ~removed,
                    connected,
                    carry,
                    "derive",
                )
            )
        if not facts and (live is None or not len(live["source"])):
            state["derive/live"] = None
        else:
            found = (source, target, code, step)
            result.update(
                found=found, stop=derive_handoff(carry, found, meta["steps"], connected), facts=facts
            )
    park(job, index, "finish", result)
    keep(job, "finish", index, state)


@staged
def assemble_task(job, index):
    registry, spec, limits, work = PIPE["registry"], PIPE["spec"], PIPE["limits"], PIPE["jobs"][job]
    graph, carry, _ = unpacked(taken(job, index, "nodes"), {})
    finish = taken(job, index, "finish")
    graph = relabelled(work["plan"], graph, finish["glob"], finish["position"])
    carry.glob, known = finish["glob"], finish["known"]
    if finish["deriving"]:
        with measure("derived relations"):
            if finish["kept"] is not None:
                graph = select(graph, FACTS, finish["kept"])
            if finish["found"] is not None:
                budget = GraphBudget(limits, fact_bytes(graph))
                budget.facts = finish["facts"]
                graph = derived_rows(graph, finish["found"], finish["stop"], budget)
    graph = transform.derive(graph, registry, spec, limits, carry, known, transform.STAGES[4:5], False)
    if not PIPE["filtering"]:
        park(job, index, "assembled", packed(carry, graph, known))
        return {}
    survives = edges_survive(graph, registry, PIPE["edges"], transform.SEED, carry)
    park(job, index, "assembled", packed(carry, graph, known, survives=survives))
    rows = np.bincount(graph.fact_code, minlength=registry.relations)
    return dict(
        rows={name: int(rows[registry.coded[name]].sum()) for name in filtered_names(registry, PIPE["edges"])}
    )


@staged
def edges_task(job, index, shard, names):
    value = peek(job, index, "assembled")
    state = state_of(job, "edges%d" % shard, index)
    graph, carry, _ = unpacked(value, state)
    kept = edges_mask(graph, PIPE["registry"], PIPE["edges"], transform.SEED, carry, names, value["survives"])
    park(job, index, "edges%d" % shard, kept)
    keep(job, "edges%d" % shard, index, state)


@staged
def merging_task(job, index, shards):
    registry, spec, limits = PIPE["registry"], PIPE["spec"], PIPE["limits"]
    state = state_of(job, "merging", index)
    graph, carry, known = unpacked(taken(job, index, "assembled"), state)
    if shards:
        kept = np.ones(len(graph.fact_step), bool)
        for shard in range(shards):
            kept &= taken(job, index, "edges%d" % shard)
        graph = select(graph, FACTS, kept)
    graph = transform.derive(graph, registry, spec, limits, carry, known, transform.STAGES[6:], False)
    with measure("verify"):
        verify_dense(graph, registry, known)
    park(job, index, "merged", packed(carry, graph, known))
    keep(job, "merging", index, state)


@staged
def represent_task(job, index):
    work = PIPE["jobs"][job]
    graph, carry, _ = unpacked(taken(job, index, "merged"), {})
    context, state = work["context"], state_of(job, "represent", index)
    piece = represent_segment(
        carry,
        graph,
        PIPE["registry"],
        PIPE["spec"],
        PIPE["groups"],
        PIPE["seed"],
        PIPE["limits"],
        state,
        context,
    )
    keep(job, "represent", index, state, False)
    if "appender" not in work:
        work["appender"] = Appender(work["destination"])
    work["appender"].add(piece)
    if index == len(work["paths"]) - 1:
        finished(work.pop("appender"), context, PIPE["groups"])


@staged
def stitch_task(job, index):
    work = PIPE["jobs"][job]
    plan, registry, spec, groups = work["plan"], PIPE["registry"], PIPE["spec"], PIPE["groups"]
    context = SimpleNamespace(steps=int(plan.step_offsets[-1]))
    tables = [Graph.load(path, mapped=True).obs_x for path in work["paths"]]
    arrays = stitched_arrays(
        plan,
        Arrays(work["destination"]),
        tables,
        registry,
        spec,
        groups,
        PIPE["limits"],
        context,
        work["destination"],
    )
    representations.validate(arrays, SimpleNamespace(name="joined"), registry, spec, groups)


TASKS = dict(
    expand=expand_task,
    window=window_task,
    nodes=nodes_task,
    steps=steps_task,
    finish=finish_task,
    assemble=assemble_task,
    edges=edges_task,
    merging=merging_task,
    represent=represent_task,
    stitch=stitch_task,
)


def balanced(rows, count):
    shards, load = [[] for _ in range(count)], [0] * count
    for name in sorted(rows, key=lambda name: (-rows[name], name)):
        target = load.index(min(load))
        shards[target].append(name)
        load[target] += rows[name]
    return [tuple(shard) for shard in shards]


def build_parallel(jobs, registry, spec, groups, seed, workers, limits):
    started = time.perf_counter()
    settings = {operation: operation.settings(registry, spec) for operation in (DERIVE, EDGES)}
    for operation, held in settings.items():
        operation.check(registry, held)
    count = min(SHARDS, len(filtered_names(registry, settings[EDGES])))
    deriving = derivation(registry, settings[DERIVE]) is not None
    stages = CHAIN[:3] + tuple("edges%d" % shard for shard in range(count)) + CHAIN[3:]
    plans = [Plan(job["members"], registry, spec) for job in jobs]
    sizes = [len(job["members"]) for job in jobs]
    initargs = (
        [
            (plan, [str(path) for path in job["paths"]], str(job["spill"]), str(job["destination"]))
            for plan, job in zip(plans, jobs)
        ],
        registry,
        spec,
        groups,
        seed,
        limits,
        max(1, numba.config.NUMBA_NUM_THREADS // workers),
    )
    hosts, dedicated = max(1, workers // 8), workers > 2
    after = dict(window="expand", nodes="window", finish="nodes", merging="filtered", represent="merging")
    after.update({stage: "assemble" for stage in stages if stage.startswith("edges")})
    ordered = {(job, stage): 0 for job in range(len(jobs)) for stage in stages}
    last = {}
    done = {(job, stage): set() for job in range(len(jobs)) for stage in tuple(RANK) + stages + ("filtered",)}
    expanded, shards, active = [0] * len(jobs), [None] * len(jobs), set()
    waiting, running, pending, busy, stitched = [], {}, {}, {}, 0
    lookahead = 2 * workers

    def ready(job, stage):
        index = ordered[job, stage]
        return (
            (job, stage) not in active
            and index in done[job, after[stage]]
            and (stage != "finish" or not deriving or index in done[job, "steps"])
            and (not stage.startswith("edges") or shards[job] is not None)
        )

    def chosen(worker):
        reserved = dedicated and worker < hosts
        found = [
            (ordered[key], -stages.index(key[1]), last.get(key) != worker, key)
            for key in ordered
            if ready(*key) and (key[0] % hosts == worker if key[1] == "represent" else not reserved)
        ]
        if found:
            job, stage = min(found)[3]
            index = ordered[job, stage]
            ordered[job, stage] += 1
            active.add((job, stage))
            last[job, stage] = worker
            if stage.startswith("edges"):
                return job, stage, (index, int(stage[5:]), shards[job][int(stage[5:])])
            return job, stage, (index, count) if stage == "merging" else (index,)
        if waiting and not reserved:
            _, _, job, stage, arguments = heapq.heappop(waiting)
            return job, stage, arguments
        return None

    executors = [
        ProcessPoolExecutor(1, **spawned(attach, initargs, registries.carried, registries.restored)) for _ in range(workers)
    ]
    try:
        with worker_threads(initargs[-1]):
            while stitched < len(jobs):
                for job in range(len(jobs)):
                    while (
                        expanded[job] < sizes[job] and expanded[job] - len(done[job, "represent"]) < lookahead
                    ):
                        heapq.heappush(
                            waiting, (expanded[job], RANK["expand"], job, "expand", (expanded[job],))
                        )
                        expanded[job] += 1
                for worker in range(workers):
                    if worker not in pending:
                        task = chosen(worker)
                        if task is not None:
                            job, stage, arguments = task
                            function = TASKS["edges" if stage.startswith("edges") else stage]
                            running[executors[worker].submit(guarded, function, job, *arguments)] = worker
                            pending[worker] = task
                if not running:
                    raise RuntimeError("parallel segment build stalled")
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    job, stage, arguments = pending.pop(running.pop(future))
                    result = future.result()
                    busy[job, stage] = busy.get((job, stage), 0.0) + result["seconds"]
                    active.discard((job, stage))
                    if stage == "stitch":
                        stitched += 1
                        continue
                    index = arguments[0]
                    done[job, stage].add(index)
                    if stage == "nodes" and deriving:
                        heapq.heappush(waiting, (index, RANK["steps"], job, "steps", (index,)))
                    if stage == "finish":
                        heapq.heappush(waiting, (index, RANK["assemble"], job, "assemble", (index,)))
                    if stage == "assemble" and not count:
                        done[job, "filtered"].add(index)
                    if stage == "assemble" and count and shards[job] is None:
                        shards[job] = balanced(result["rows"], count)
                    if stage.startswith("edges") and all(
                        index in done[job, "edges%d" % shard] for shard in range(count)
                    ):
                        done[job, "filtered"].add(index)
                    if stage == "represent" and index == sizes[job] - 1:
                        LOG.info(
                            "segments represented job=%d seconds=%.3f", job, time.perf_counter() - started
                        )
                        heapq.heappush(waiting, (-1, RANK["stitch"], job, "stitch", (index,)))
    finally:
        for executor in executors:
            executor.shutdown(cancel_futures=True)
    totals = {}
    for (_, stage), seconds in busy.items():
        totals[stage] = totals.get(stage, 0.0) + seconds
    LOG.info(
        "segments parallel build exit seconds=%.3f groups=%d segments=%d workers=%d shards=%d busy=%s",
        time.perf_counter() - started,
        len(jobs),
        sum(sizes),
        workers,
        count,
        ",".join("%s:%.1f" % item for item in sorted(totals.items())),
    )


@njit(cache=True)
def grouped(ids, groups):
    offsets = np.zeros(groups + 1, np.int64)
    for row in range(len(ids)):
        offsets[ids[row] + 1] += 1
    for group in range(groups):
        offsets[group + 1] += offsets[group]
    cursor = offsets[:-1].copy()
    order = np.empty(len(ids), np.int32)
    for row in range(len(ids)):
        order[cursor[ids[row]]] = row
        cursor[ids[row]] += 1
    return order, offsets
