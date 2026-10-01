import logging
import time
from pathlib import Path

import numpy as np
from numba import njit

from graphspec.budgets import GraphBudget
from graphspec.infra.files import write_pack
from graphspec.infra.hashing import hashed
from graphspec.infra.timing import timed
from graphspec.infra.workspace import finite, gathered
from graphspec.operations.parameters import enabled
from graphspec.operations.rows import keyed_relations
from graphspec.operations.state import EMPTY, Table
from graphspec.views.spans import encode

from . import kernels
from .groups import base, periods

LOG = logging.getLogger(__name__)
INFINITE = np.int64(1) << np.int64(60)
STITCHED = "stitched.pack"


def carried(registry):
    started = time.perf_counter()
    result = dict(
        last=Table(EMPTY, row=EMPTY, x=np.zeros((0, registry.features), np.float16)),
        replay=Table(EMPTY, source=EMPTY, target=EMPTY, code=EMPTY, state=EMPTY),
        obs=0,
        events=0,
        tasks=0,
        rows=0,
        steps=0,
        attributes={group: 0 for group in registry.valued},
        groups=0,
        period=None,
        starts=EMPTY,
        ends=EMPTY,
        history=(),
    )
    LOG.debug("base carried exit seconds=%.4f", time.perf_counter() - started)
    return result


def initial(registry, spec, chosen):
    started = time.perf_counter()
    result = dict(base=carried(registry), **{entry.name: entry.carries(registry, spec) for entry in chosen})
    LOG.debug("representations initial exit chosen=%d seconds=%.4f", len(chosen), time.perf_counter() - started)
    return result


def mapped(glob, values):
    values = np.asarray(values)
    return gathered(glob, values, -1).astype(np.int32)


@timed("segments.shifted")
def shifted(values, base):
    values = np.asarray(values).astype(np.int64)
    return np.where(values >= 0, values + base, -1).astype(np.int32)


def localized(*columns):
    held = np.concatenate([np.asarray(column, np.int64) for column in columns] + [EMPTY])
    values = np.unique(held)
    inverse = np.searchsorted(values, held)
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


def recent(history, name):
    parts = [item[name] for item in history]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


EMPTY_OBS = dict(entity=EMPTY, step=EMPTY, group=EMPTY, period=EMPTY, row=EMPTY)
EMPTY_EVENTS = dict(
    events=np.zeros((0, 9), np.int32), id=EMPTY, group=EMPTY, period=EMPTY, identity=np.zeros(0, np.uint64)
)


@timed("representations.identities")
def identities(events, carry, graph, registry, tasks0):
    events = np.asarray(events, np.int64)
    kind, local = events[:, 3], events[:, 2] // 2 - carry.step0
    relation, task = kind == 1, kind == 2
    first, second = carry.entities.tokens[events[:, 0]], carry.entities.tokens[events[:, 1]]
    code = np.where(relation, events[:, 6], 0)
    tokens, symmetric = keyed_relations(registry)
    symmetric = relation & symmetric[code]
    keys = np.zeros(len(events), np.int64)
    keys[task] = np.asarray(graph.task_key, np.int64)[events[task, 7] - tasks0]
    return hashed(
        kind,
        np.where(symmetric, np.minimum(first, second), first),
        np.where(symmetric, np.maximum(first, second), second),
        np.where(relation, tokens[code], np.where(task, events[:, 6], 0).astype(np.uint64)),
        np.where(relation, events[:, 7] * 2 + events[:, 8], 0),
        keys,
        np.asarray(graph.step_time)[local],
        np.asarray(graph.step_token)[local],
        events[:, 2] % 2,
    )


class Link:
    def __init__(self, carry, state):
        self.glob, self.base, self.state, self.step0 = carry.glob, carry.base, state, carry.step0
        self.obs0, self.attributes = state["obs"], state["attributes"]

    def table(self, name):
        return self.state[name]

    def replay_keys(self, graph):
        table = self.table("replay")
        if not len(table):
            return (EMPTY,) * 4
        found = table.find(
            encode(graph.fact_code, self.glob[graph.fact_source], self.glob[graph.fact_target], self.base)
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
        keys = encode(code[both], left, right, self.base)
        self.state["replay"] = self.table("replay").replaced(
            keys, keys, source=left, target=right, code=code[both], state=state[both]
        )


class Shown:
    def __init__(self, carry, graph, local, own, work, before, registry, spec, seed, budget):
        self.carry, self.graph, self.local, self.own, self.work = carry, graph, local, own, work
        self.before, self.registry, self.spec, self.seed, self.budget = before, registry, spec, seed, budget

    def recent(self, name):
        if not self.before["history"]:
            return dict(obs=EMPTY_OBS, events=EMPTY_EVENTS)[name]
        return recent(self.before["history"], name)


def own_arrays(arrays, carry, before, graph):
    glob, step0, obs0, tasks0, rows0 = carry.glob, carry.step0, before["obs"], before["tasks"], before["rows"]
    events = np.asarray(arrays["events"]).astype(np.int64)
    moments = events[:, 2] // 2
    events[:, 0], events[:, 1] = glob[events[:, 0]], glob[events[:, 1]]
    events[:, 2] += 2 * step0
    events[events[:, 3] == 2, 7] += tasks0
    spatial = np.asarray(arrays["spatial"]).astype(np.int64)
    spatial[:, :2] += obs0
    period = np.asarray(arrays["period"])
    first = np.r_[0, np.flatnonzero(np.diff(period)) + 1]
    last = np.r_[first[1:] - 1, len(period) - 1]
    continued = before["period"] is not None and period[0] == before["period"]
    group0 = before["groups"] - int(continued)
    membership = np.repeat(np.arange(len(first)), last - first + 1) + group0
    starts, ends = (first + step0).astype(np.int64), (last + step0).astype(np.int64)
    if continued:
        starts[0] = before["starts"][-1]
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
        event_task=(arrays["event_task"] + tasks0).astype(np.int32),
        event_step=(arrays["event_step"] + step0).astype(np.int32),
        row_step=(arrays["row_step"] + step0).astype(np.int32),
        row_task=(arrays["row_task"] + tasks0).astype(np.int32),
        row_kind=arrays["row_kind"],
        events=events.astype(np.int32),
        signals=arrays["signals"],
        row_index=arrays["row_index"],
        key=arrays["key"],
        period=arrays["period"],
        member_row=(arrays["member_row"] + rows0).astype(np.int32),
        member_name=arrays["member_name"],
        member_entity=mapped(glob, arrays["member_entity"]),
        member_observed=shifted(arrays["member_observed"], obs0),
        member_place=arrays["member_place"],
        whole_row=(arrays["whole_row"] + rows0).astype(np.int32),
        whole_name=arrays["whole_name"],
        **{name: value for name, value in arrays.items() if name.startswith("attributes_")},
    )
    work = dict(
        connected=graph.entity_connected[arrays["entity"]],
        group=membership[step],
        event_group=membership[moments],
        event_period=period[moments],
        group0=group0,
        groups=group0 + len(first),
        starts=np.r_[before["starts"][:group0], starts],
        ends=np.r_[before["ends"][:group0], ends],
        row_group=membership[np.asarray(arrays["row_step"])],
        first_step=step0,
    )
    return own, work


def horizons(spec, chosen):
    found = [entry.horizon(spec) for entry in chosen] + [(0, 0)]
    return max(steps for steps, _ in found), max(groups for _, groups in found)


def remembered(history, piece, next_step, groups, spec, chosen):
    steps, lookback = horizons(spec, chosen)
    return tuple(
        item
        for item in history + (piece,)
        if item["last_step"] >= next_step - steps or item["last_group"] >= groups - lookback - 1
    )


def represented(carry, graph, registry, spec, chosen, seed, limits, states):
    started = time.perf_counter()
    budget = GraphBudget(limits)
    before = states["base"]
    state = dict(before, attributes=dict(before["attributes"]))
    link = Link(carry, state)
    arrays = base(graph, registry, spec, link)
    own, work = own_arrays(arrays, carry, before, graph)
    work["identity"] = identities(own["events"], carry, graph, registry, before["tasks"])
    own.update(
        event_identity=work["identity"],
        step_time=np.asarray(graph.step_time, np.int64),
        step_token=np.asarray(graph.step_token, np.uint64),
    )
    if registry.candidates:
        drawn = carry.drawn()
        own.update(
            candidate_code=drawn["code"].astype(np.int32),
            candidate_step=(drawn["step"] + carry.step0).astype(np.int32),
            candidate_source=drawn["source"].astype(np.int32),
            candidate_target=drawn["target"].astype(np.int32),
        )
    shown = Shown(carry, graph, arrays, own, work, before, registry, spec, seed, budget)
    piece, found = dict(own), dict(base=state)
    for entry in chosen:
        held = dict(states[entry.name])
        piece.update(entry.piece(shown, held))
        found[entry.name] = held
    rows, count = len(own["entity"]), len(own["events"])
    next_step = carry.step0 + graph.steps
    item = dict(
        obs=dict(
            entity=own["entity"].astype(np.int64),
            step=own["step"].astype(np.int64),
            group=work["group"].astype(np.int64),
            period=own["obs_period"],
            row=before["obs"] + np.arange(rows, dtype=np.int64),
        ),
        events=dict(
            events=own["events"],
            id=before["events"] + np.arange(count, dtype=np.int64),
            group=work["event_group"].astype(np.int64),
            period=work["event_period"],
            identity=work["identity"],
        ),
        last_step=next_step - 1,
        last_group=work["groups"] - 1,
    )
    for group in registry.valued:
        state["attributes"][group] += len(own["attributes_" + group])
    state.update(
        history=remembered(before["history"], item, next_step, work["groups"], spec, chosen),
        obs=before["obs"] + rows,
        events=before["events"] + count,
        tasks=before["tasks"] + len(own["task"]),
        rows=before["rows"] + len(own["row_step"]),
        steps=next_step,
        groups=work["groups"],
        period=int(own["period"][-1]),
        starts=work["starts"],
        ends=work["ends"],
    )
    LOG.info(
        "segment representation exit seconds=%.3f rows=%d events=%d",
        time.perf_counter() - started,
        rows,
        count,
    )
    return piece, found


def ended(chosen, states):
    started = time.perf_counter()
    result = {}
    for entry in chosen:
        if entry.ended is not None:
            result.update(entry.ended(states[entry.name]))
    LOG.debug("ended exit rows=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def stitched(arrays, table, kinds, tokens, registry, spec, chosen, limits, steps, destination):
    started = time.perf_counter()
    arrays = dict(arrays)
    written = set(arrays)
    kinds = np.asarray(kinds).astype(np.int32)
    entity, step = arrays["entity"], arrays["step"]
    observation_ids, observation_offsets = grouped(np.asarray(entity), len(kinds))
    arrays.update(
        x_table=table,
        entity_token=(np.asarray(tokens) & np.uint64((1 << 62) - 1)).astype(np.int64),
        entity_kind=(np.zeros_like(kinds) if enabled(spec, "nodes/untyped") else kinds).astype(np.int32),
        snapshot_offsets=np.searchsorted(step, np.arange(steps + 1)).astype(np.int64),
        observation_ids=observation_ids,
        observation_offsets=observation_offsets,
    )
    budget = GraphBudget(limits)
    arrays.update(periods(arrays, len(registry.events), budget))
    for function in dict.fromkeys(entry.stitched for entry in chosen if entry.stitched is not None):
        arrays.update(function(arrays, registry, spec, limits))
    if destination is not None:
        write_pack(Path(destination) / STITCHED, {name: arrays[name] for name in sorted(set(arrays) - written)})
    LOG.info("stitched exit seconds=%.3f arrays=%d", time.perf_counter() - started, len(arrays))
    return arrays


@timed("validate")
def validate(arrays, name, registry, spec, chosen):
    x, entity, reference = arrays["x_table"], arrays["entity"], arrays["x_ref"]
    if len(reference) != len(entity) or (len(reference) and (reference.min() < 0 or reference.max() >= len(x))):
        raise ValueError("observation feature references out of range in graph %s" % name)
    if not finite(x):
        raise ValueError("nonfinite observation features in graph %s" % name)
    sizes = np.asarray([len(arrays["attributes_" + group]) for group in registry.valued] + [-1], np.int64)
    limits = sizes[registry.value_group]
    fault = kernels.misplaced_attributes(
        np.asarray(arrays["spatial"]), np.asarray(arrays["spatial_attribute_row"]), limits
    )
    if fault >= 0:
        raise ValueError("attribute rows of relation code %d are missing, unexpected or out of range" % fault)
    for entry in chosen:
        entry.check(arrays, name, registry, spec)


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
