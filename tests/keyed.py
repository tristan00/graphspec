import hashlib
import logging
import time

import numpy as np

from graphspec.schema.references import MEMBERS
from graphspec.views.graph import LABELS, PER_STEP, TASKS

LOG = logging.getLogger(__name__)
DENSE = dict(
    entities=("kinds", "entity_token", "entity_connected"),
    steps=PER_STEP,
    observations=("obs_entity", "obs_step", "obs_ref", "obs_anchor"),
    facts=("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref"),
    x_table=("x_table",),
    value_table=("value_table",),
    tasks=TASKS,
    members=MEMBERS,
    labels=LABELS,
)
VARIANT = dict(
    entity_kind=("entity_kind",),
    entity_token=("entity_token",),
    x_table=("x_table",),
    observations=("entity", "step", "x_ref", "obs_period"),
    snapshot_offsets=("snapshot_offsets",),
    observation_ids=("observation_ids", "observation_offsets"),
    period_observation_keys=("period_observation_keys",),
    spatial=("spatial", "spatial_step", "spatial_attribute_row"),
    task=("task",),
    event_tasks=("event_first", "event_task", "event_step"),
    rows=("row_step", "row_task", "row_kind", "row_index", "key", "signals"),
    members=("member_row", "member_name", "member_entity", "member_observed", "member_place"),
    whole=("whole_row", "whole_name"),
    events=("events",),
    event_identity=("event_identity",),
    step_keys=("step_time", "step_token"),
    period=("period",),
    period_index=("period_starts", "period_ends", "period_numbers"),
    period_membership=("period_membership",),
    affinity=("affinity_keys", "affinity_counts"),
    affinity_steps=("affinity_steps",),
    walks=("walks",),
    period_walks=("period_walks",),
    coupling=("coupling",),
    coupling_boundaries=("coupling_boundaries",),
    event_dag=("event_dag",),
    event_boundaries=("event_boundaries",),
    event_member=("event_member",),
    neighbors=("neighbor_ids", "neighbor_offsets"),
)


def integral(values):
    values = np.ascontiguousarray(values)
    shape = (len(values), int(np.prod(values.shape[1:], dtype=np.int64)))
    if values.dtype == np.bool_:
        return values.reshape(shape).astype(np.int64)
    return values.view(np.dtype("u%d" % values.dtype.itemsize)).reshape(shape).astype(np.int64)


def table(*columns):
    return np.concatenate([integral(column) for column in columns], 1)


def labelled(*columns):
    found = table(*columns)
    if not len(found):
        return np.zeros(0, np.int64)
    return np.unique(found, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)


def through(labels, index):
    index = np.asarray(index, np.int64)
    return np.where(index >= 0, np.asarray(labels)[np.maximum(index, 0)], -1)


def rows(*columns):
    found = table(*columns)
    return found[np.lexsort(found.T[::-1])]


class Keys:
    def __init__(self, dense, registry):
        started = time.perf_counter()
        self.registry = registry
        self.symmetric = np.zeros(registry.relations, bool)
        for edge in registry.edges:
            self.symmetric[edge.code] = edge.symmetric
        self.steps = labelled(np.asarray(dense.step_token), np.asarray(dense.step_time))
        step, entity = np.asarray(dense.obs_step, np.int64), np.asarray(dense.obs_entity, np.int64)
        first = np.full(len(dense.kinds), -1, np.int64)
        order = np.argsort(step, kind="stable")[::-1]
        first[entity[order]] = step[order]
        home = np.where(first >= 0, np.asarray(dense.step_token)[np.maximum(first, 0)], np.uint64(0))
        self.entities = labelled(np.asarray(dense.entity_token), home)
        self.tasks = labelled(through(self.steps, dense.task_step), np.asarray(dense.task_key))
        values, place = np.unique(np.asarray(dense.period, np.int64), return_index=True)
        self.periods = dict(zip(values.tolist(), self.steps[place].tolist()))
        LOG.debug("keys exit entities=%d seconds=%.4f", len(self.entities), time.perf_counter() - started)

    def period(self, values):
        return np.asarray([self.periods[value] for value in np.asarray(values, np.int64).tolist()], np.int64)

    def ends(self, both, source, target):
        source, target = through(self.entities, source), through(self.entities, target)
        swap = both & (source > target)
        return np.where(swap, target, source), np.where(swap, source, target), swap


def dense_keyed(dense, keys):
    started = time.perf_counter()
    x, values = labelled(dense.x_table), labelled(dense.value_table)
    both = keys.symmetric[np.asarray(dense.fact_code, np.int64)]
    source, target, _ = keys.ends(both, dense.fact_source, dense.fact_target)
    result = dict(
        entities=rows(keys.entities, dense.kinds, dense.entity_token, dense.entity_connected),
        steps=rows(keys.steps, keys.period(dense.period), dense.step_time, dense.step_token),
        observations=rows(
            through(keys.steps, dense.obs_step),
            through(keys.entities, dense.obs_entity),
            through(x, dense.obs_ref),
            dense.obs_anchor,
        ),
        facts=rows(
            through(keys.steps, dense.fact_step),
            dense.fact_present,
            dense.fact_code,
            source,
            target,
            through(values, dense.fact_ref),
        ),
        x_table=rows(dense.x_table),
        value_table=rows(dense.value_table),
        tasks=rows(keys.tasks, dense.task_kind, dense.task_x, dense.task_signals),
        members=rows(
            through(keys.tasks, dense.member_task),
            dense.member_name,
            dense.member_place,
            through(keys.entities, dense.member_source),
            through(keys.entities, dense.member_target),
            dense.member_code,
        ),
        labels=rows(through(keys.tasks, dense.label_task), dense.label_row),
    )
    LOG.debug("dense keyed exit groups=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def dense_part(dense, keys, names):
    found = dense_keyed(dense, keys)
    return {group: found[group] for group, held in DENSE.items() if set(held) & set(names)}


def events_keyed(events, keys, observations):
    events = np.asarray(events, np.int64)
    kind, code = events[:, 3], events[:, 6]
    relation, task = kind == 1, kind == 2
    both = relation & keys.symmetric[np.where(relation, code, 0)]
    source, target, swap = keys.ends(both, events[:, 0], events[:, 1])
    first, second = through(observations, events[:, 4]), through(observations, events[:, 5])
    return table(
        kind,
        source,
        target,
        through(keys.steps, events[:, 2] // 2),
        events[:, 2] % 2,
        np.where(swap, second, first),
        np.where(swap, first, second),
        np.where(kind == 0, through(observations, code), code),
        np.where(task, through(keys.tasks, events[:, 7]), events[:, 7]),
        events[:, 8],
    )


class Spaces:
    def __init__(self, arrays, keys):
        started = time.perf_counter()
        self.arrays, self.keys = arrays, keys
        self.observations = labelled(
            through(keys.steps, arrays["step"]), through(keys.entities, arrays["entity"])
        )
        self.x = labelled(arrays["x_table"])
        self.events = events_keyed(arrays["events"], keys, self.observations)
        self.event = labelled(self.events)
        self.rows = through(keys.tasks, arrays["row_task"])
        self.members = table(
            through(self.rows, arrays["member_row"]),
            arrays["member_name"],
            through(keys.entities, arrays["member_entity"]),
            through(self.observations, arrays["member_observed"]),
            arrays["member_place"],
        )
        self.member = labelled(self.members)
        self.groups = through(keys.steps, arrays["period_starts"])
        self.attributes, offset = [], 0
        for group in keys.registry.valued:
            held = labelled(arrays["attributes_" + group])
            self.attributes.append(held + offset)
            offset += len(held)
        LOG.debug("spaces exit events=%d seconds=%.4f", len(self.event), time.perf_counter() - started)

    def keyed(self, group):
        arrays, keys, observations, event = self.arrays, self.keys, self.observations, self.event
        if group == "entity_kind":
            return rows(keys.entities, arrays["entity_kind"])
        if group == "entity_token":
            return rows(keys.entities, arrays["entity_token"])
        if group == "x_table":
            return rows(arrays["x_table"])
        if group == "observations":
            return rows(observations, through(self.x, arrays["x_ref"]), keys.period(arrays["obs_period"]))
        if group == "snapshot_offsets":
            return rows(keys.steps, np.diff(arrays["snapshot_offsets"]))
        if group == "observation_ids":
            offsets = arrays["observation_offsets"]
            owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
            return rows(through(keys.entities, owner), through(observations, arrays["observation_ids"]))
        if group == "period_observation_keys":
            held, width = arrays["period_observation_keys"].astype(np.int64), len(arrays["period_ends"])
            return rows(through(keys.entities, held // width), through(self.groups, held % width))
        if group == "spatial":
            spatial = arrays["spatial"].astype(np.int64)
            valued, place = self.keys.registry.value_group[spatial[:, 2]], arrays["spatial_attribute_row"]
            found = np.full(len(spatial), -1, np.int64)
            for index, labels in enumerate(self.attributes):
                chosen = (valued == index) & (place >= 0)
                found[chosen] = labels[place[chosen]]
            return rows(
                through(observations, spatial[:, 0]),
                through(observations, spatial[:, 1]),
                spatial[:, 2],
                through(keys.steps, arrays["spatial_step"]),
                found,
            )
        if group == "task":
            return rows(keys.tasks, arrays["task"])
        if group == "event_tasks":
            return rows(
                through(keys.entities, arrays["event_first"]),
                through(keys.tasks, arrays["event_task"]),
                through(keys.steps, arrays["event_step"]),
            )
        if group == "rows":
            return rows(
                self.rows,
                through(keys.steps, arrays["row_step"]),
                arrays["row_kind"],
                arrays["row_index"],
                arrays["key"],
                arrays["signals"],
            )
        if group == "members":
            return rows(self.members)
        if group == "whole":
            return rows(through(self.rows, arrays["whole_row"]), arrays["whole_name"])
        if group == "events":
            return rows(self.events)
        if group == "event_identity":
            return rows(event, arrays["event_identity"])
        if group == "step_keys":
            return rows(keys.steps, arrays["step_time"], arrays["step_token"])
        if group == "period":
            return rows(keys.steps, keys.period(arrays["period"]))
        if group == "period_index":
            ends = through(keys.steps, arrays["period_ends"])
            return rows(self.groups, ends, keys.period(arrays["period_numbers"]))
        if group == "period_membership":
            return rows(keys.steps, through(self.groups, arrays["period_membership"]))
        if group == "affinity":
            held, span = arrays["affinity_keys"].astype(np.int64), 2 * len(arrays["period"]) + 2
            return rows(
                through(keys.entities, held // span),
                through(keys.steps, held % span // 2),
                held % 2,
                arrays["affinity_counts"],
            )
        if group == "affinity_steps":
            return rows(keys.steps, np.diff(arrays["affinity_steps"], axis=0))
        if group in ("walks", "period_walks"):
            walks = arrays[group]
            index = np.indices(walks.shape[:-1]).reshape(walks.ndim - 1, -1)
            return rows(self.member[index[0]], *index[1:], through(event, walks.reshape(-1, walks.shape[-1])))
        if group in ("coupling", "coupling_boundaries"):
            held = arrays[group].astype(np.int64)
            return rows(through(observations, held[:, 0]), through(observations, held[:, 1]), held[:, 2:])
        if group in ("event_dag", "event_boundaries"):
            held = arrays[group].astype(np.int64)
            return rows(through(event, held[:, 0]), through(event, held[:, 1]), held[:, 2:])
        if group == "event_member":
            return rows(event, arrays["event_member"])
        if group == "neighbors":
            offsets = arrays["neighbor_offsets"]
            owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
            return rows(through(keys.entities, owner), through(event, arrays["neighbor_ids"]))
        if group.startswith("attributes_"):
            return rows(arrays[group])
        raise KeyError("no key is stated for the arrays %s" % group)


def variant_part(arrays, keys, names):
    started = time.perf_counter()
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    stated = dict(VARIANT, **{"attributes_" + group: ("attributes_" + group,) for group in keys.registry.valued})
    unknown = sorted(set(names) - {name for held in stated.values() for name in held})
    if unknown:
        raise KeyError("no key is stated for the arrays %s" % ", ".join(unknown))
    spaces = Spaces(arrays, keys)
    found = {group: spaces.keyed(group) for group, held in stated.items() if set(held) & set(names)}
    LOG.debug("variant part exit groups=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def equal(left, right):
    return sorted(left) == sorted(right) and all(
        left[name].shape == right[name].shape and np.array_equal(left[name], right[name]) for name in left
    )


def differing(left, right):
    return sorted(
        set(left) ^ set(right)
        | {name for name in set(left) & set(right) if not equal({name: left[name]}, {name: right[name]})}
    )


def digest(found):
    value = hashlib.sha256()
    for name in sorted(found):
        value.update(name.encode())
        value.update(repr(found[name].shape).encode())
        value.update(np.ascontiguousarray(found[name]).tobytes())
    return value.hexdigest()


def pinned(graph, registry, names, arrays=None):
    started = time.perf_counter()
    keys = Keys(graph, registry)
    found = dense_part(graph, keys, names) if arrays is None else variant_part(arrays, keys, names)
    result = digest(found)
    LOG.debug("pinned exit graph=%s arrays=%d seconds=%.4f", graph.name, len(names), time.perf_counter() - started)
    return result


def combined(digests):
    return hashlib.sha256("".join(digests).encode()).hexdigest()[:16]
