import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.operations.state import EMPTY, Table

from . import kernels
from .base import mapped
from .groups import mixed_features, mixed_sizes, observed_features
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


@timed("mixed.span")
def span(spec):
    return spec["mixed/trace"] if spec["mixed/trace"] > 0 else 1


@timed("mixed.carries")
def carries(registry, spec):
    width = span(spec)
    return dict(
        nodes=Table(
            EMPTY,
            last=EMPTY,
            count=EMPTY,
            node=np.zeros((0, width), np.int64),
            row=np.zeros((0, width), np.int64),
            x=np.zeros((0, width, registry.features), np.float16),
        ),
        action=-1,
        action_member=-1,
        count=0,
        holders=0,
    )


@timed("mixed.horizon")
def horizon(spec):
    return 0, 0


class Shifted:
    @timed("mixed.Shifted.__init__")
    def __init__(self, table, base):
        self.table, self.base = table, base

    @timed("mixed.Shifted.__getitem__")
    def __getitem__(self, rows):
        return self.table[np.asarray(rows, np.int64) - self.base]


def piece(shown, state):
    started = time.perf_counter()
    graph, arrays, spec, budget, before = shown.graph, shown.local, shown.spec, shown.budget, shown.before
    obs0, tasks0, glob = before["obs"], before["tasks"], shown.carry.glob
    entity, offsets = arrays["entity"].astype(np.int64), arrays["snapshot_offsets"]
    x = observed_features(arrays)
    changes, trace, bundles = spec["mixed/changes"], spec["mixed/trace"], spec["mixed/bundles"]
    rows, moments, entities = len(entity), len(offsets) - 1, len(graph.kinds)
    placed = np.searchsorted(np.asarray(arrays["event_step"], np.int64), np.arange(moments + 1))
    order = np.argsort(entity, kind="stable")
    prior = np.full(rows, -1, np.int64)
    heads = order
    if rows:
        same = entity[order][1:] == entity[order][:-1]
        prior[order[1:][same]] = obs0 + order[:-1][same]
        heads = order[np.r_[True, ~same]]
    last_seen = before["last"]
    found = last_seen.find(glob[entity[heads]])
    known = found >= 0
    extra_rows, extra_x = [EMPTY], [np.zeros((0, x.shape[1]), x.dtype)]
    if known.any():
        prior[heads[known]] = last_seen.columns["row"][found[known]]
        extra_rows.append(last_seen.columns["row"][found[known]])
        extra_x.append(last_seen.columns["x"][found[known]])
    width = span(spec)
    last_node = np.full(entities, -1, np.int64)
    trace_node = np.zeros((entities, width), np.int64)
    trace_row = np.zeros((entities, width), np.int64)
    trace_count = np.zeros(entities, np.int64)
    nodes = state["nodes"]
    found = nodes.find(glob)
    use = np.flatnonzero(found >= 0)
    if len(use):
        where = found[use]
        last_node[use] = nodes.columns["last"][where]
        trace_count[use] = nodes.columns["count"][where]
        trace_node[use] = nodes.columns["node"][where]
        trace_row[use] = nodes.columns["row"][where]
        filled = np.arange(width)[None, :] < trace_count[use][:, None]
        extra_rows.append(trace_row[use][filled])
        extra_x.append(nodes.columns["x"][where][filled])
    carried = Table(np.concatenate(extra_rows), x=np.concatenate(extra_x))

    @timed("mixed.piece.lookup")
    def lookup(ids):
        ids = np.asarray(ids, np.int64)
        local = ids - obs0
        inside = (local >= 0) & (local < rows)
        result = np.empty((len(ids), x.shape[1]), x.dtype)
        result[inside] = x[local[inside]]
        outside = ids[~inside]
        if len(outside):
            position = carried.find(outside)
            if np.any(position < 0):
                raise ValueError("a mixed representation row is missing from its carry")
            result[~inside] = carried.columns["x"][position]
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
    bundled = kernels.bundled(
        np.ascontiguousarray(entity, np.int64),
        np.ascontiguousarray(offsets, np.int64),
        prior,
        altered,
        np.asarray(arrays["event_first"], np.int64),
        np.asarray(arrays["event_task"], np.int64) + tasks0,
        placed,
        trace,
        bundles,
        limit,
        wires,
        holders,
        np.int64(state["count"]),
        np.int64(obs0),
        last_node,
        trace_node,
        trace_row,
        trace_count,
        np.int64(state["action"]),
        np.asarray(graph.step_graph, np.int64),
        np.int64(state["action_member"]),
    )
    kind, tick, owner, mode, fill, member_offset, member_id, source, target, flag, action, member = bundled
    features = mixed_features(
        kind,
        mode,
        fill,
        member_offset,
        member_id,
        lookup,
        x.shape[1],
        shown.registry.values,
        Shifted(arrays["task"], tasks0),
        offsets,
        spec,
        obs0,
    )
    chosen = np.flatnonzero(graph.entity_connected & ((last_node >= 0) | (trace_count > 0)))
    filled = np.arange(width)[None, :] < trace_count[chosen][:, None]
    trace_x = np.zeros((len(chosen), width, x.shape[1]), x.dtype)
    if filled.any():
        trace_x[filled] = lookup(trace_row[chosen][filled])
    keys = glob[chosen]
    result = dict(
        mixed_x=features.reshape(-1, x.shape[1]),
        mixed_kind=kind,
        mixed_tick=(tick.astype(np.int64) + 2 * shown.carry.step0).astype(np.int32),
        mixed_entity=mapped(glob, owner),
        mixed_edges=np.stack((source, target, flag), 1).astype(np.int32).reshape(-1, 3),
        mixed_member_ids=member_id,
        mixed_member_offsets=member_offset[:-1] + state["holders"],
    )
    state.update(
        nodes=nodes.replaced(
            keys,
            keys,
            last=last_node[chosen],
            count=trace_count[chosen],
            node=trace_node[chosen],
            row=trace_row[chosen],
            x=trace_x,
        ),
        action=int(action),
        action_member=int(member),
        count=state["count"] + len(kind),
        holders=state["holders"] + len(member_id),
    )
    LOG.debug("mixed piece exit nodes=%d seconds=%.4f", len(kind), time.perf_counter() - started)
    return result


@timed("mixed.ended")
def ended(state):
    return dict(mixed_member_offsets=np.asarray([state["holders"]], np.int64))


@timed("mixed.check")
def check(arrays, name, registry, spec):
    return None


MIXED = REPRESENTATIONS.register(
    "mixed", Representation("mixed", "structure", carries, horizon, piece, ended, None, check, 1)
)
