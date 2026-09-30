import logging
import time

import numpy as np
from numba import njit

from graphspec.facts.stored import Arrays, tokens
from graphspec.infra.timing import timed
from graphspec.infra.workspace import finite
from graphspec.schema.references import MEMBERS, ends

from . import spans

LOG = logging.getLogger(__name__)
TASKS = ("task_step", "task_key", "task_kind", "task_x", "task_signals")
LABELS = ("label_task", "label_row")
OBSERVATIONS = ("obs_entity", "obs_start", "obs_stop", "obs_x")
FACTS = ("fact_source", "fact_target", "fact_code", "fact_start", "fact_stop", "fact_rank", "fact_values")
ORDER = ("order_step", "order_position", "order_span")
REMOVALS = ("removal_source", "removal_target", "removal_code", "removal_step", "removal_values")
STATED = ("obs_values", "task_values", "graph_values")
ARRAYS = (
    ("kinds", "entity_token", "entity_held", "period")
    + OBSERVATIONS
    + FACTS
    + ORDER
    + REMOVALS
    + TASKS
    + MEMBERS
    + LABELS
    + STATED
)
DENSE = (
    ("kinds", "entity_token", "entity_connected", "period", "step_graph")
    + ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
    + ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    + ("x_table", "value_table")
    + TASKS
    + MEMBERS
    + LABELS
)
CHANGES = ("time", "observations", "facts")
ENCODED = ("obs_x", "task_x", "entity_held")
SHOWN = tuple(name for name in ARRAYS if name not in ENCODED)


class Shown(Arrays):
    NAMES = SHOWN

    @timed("truth.shown.tables")
    def tables(self):
        return self.obs_values, self.task_values

    def dense(self):
        started = time.perf_counter()
        entity, step, reference = spans.observation_rows(
            np.asarray(self.obs_entity), np.asarray(self.obs_start), np.asarray(self.obs_stop)
        )
        facts = spans.fact_rows(
            {
                name: np.asarray(self.arrays[name])
                for name in FACTS + ORDER + REMOVALS
                if "values" not in name
            },
            self.steps,
        )
        facts["fact_code"] = facts["fact_code"].astype(np.int64)
        result = Dense(
            self.name,
            self.keys,
            self.steps,
            kinds=np.asarray(self.kinds),
            entity_token=np.asarray(self.entity_token),
            entity_connected=np.zeros(len(self.kinds), bool),
            period=np.asarray(self.period),
            step_graph=np.zeros(self.steps, np.int32),
            obs_entity=entity,
            obs_step=step,
            obs_ref=reference,
            obs_anchor=np.zeros(len(step), bool),
            x_table=self.tables()[0],
            value_table=np.concatenate((self.fact_values, self.removal_values)),
            **facts,
            task_x=np.asarray(self.tables()[1]),
            **{name: np.asarray(self.arrays[name]) for name in TASKS + MEMBERS + LABELS if name != "task_x"},
        )
        LOG.debug(
            "dense exit graph=%s observations=%d facts=%d seconds=%.4f",
            self.name,
            len(step),
            len(facts["fact_step"]),
            time.perf_counter() - started,
        )
        return result


class Graph(Shown):
    NAMES = ARRAYS

    @timed("truth.graph.tables")
    def tables(self):
        return self.obs_x, self.task_x


class Dense(Arrays):
    NAMES = DENSE


def conforms(graph, registry, changed):
    started = time.perf_counter()
    unknown = sorted(set(changed) - set(CHANGES))
    if unknown:
        raise ValueError("an operation states changes the schema does not know: " + ", ".join(unknown))
    name, kinds = graph.name, np.asarray(graph.kinds)
    if "time" in changed and len(graph.period) != graph.steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if "observations" in changed:
        if graph.x_table.shape[1] != registry.features:
            raise ValueError("graph %s holds node values of a width its schema does not state" % name)
        if np.any(kinds < 0) or np.any(kinds >= len(registry.nodes)):
            raise ValueError("graph %s names kinds of node its schema does not state" % name)
    if "facts" in changed:
        found = np.asarray(graph.fact_code)
        if np.any(found < 0) or np.any(found >= registry.relations) or not registry.canonical[found].all():
            raise ValueError("graph %s holds facts of relations its schema does not state" % name)
        pairs = kinds[graph.fact_source].astype(np.int64) * len(registry.nodes) + kinds[graph.fact_target]
        for code in np.unique(found).tolist():
            edge = registry.by_code[code]
            allowed = {(registry.kind(source), registry.kind(target)) for source, target in edge.pairs}
            if edge.symmetric:
                allowed |= {(target, source) for source, target in allowed}
            allowed = np.asarray([source * len(registry.nodes) + target for source, target in allowed])
            if not np.isin(pairs[found == code], allowed).all():
                raise ValueError(
                    "graph %s holds a fact of %s between kinds of node the relation does not join"
                    % (name, edge.name)
                )
    LOG.debug(
        "conforms exit graph=%s changed=%s seconds=%.4f",
        name,
        "+".join(changed),
        time.perf_counter() - started,
    )


@timed("truth.unobserved")
def unobserved(entities, obs_entity, obs_step, who, moments):
    held = np.asarray(obs_step, np.int64) * entities + np.asarray(obs_entity, np.int64)
    return ~np.isin(np.asarray(moments, np.int64) * entities + np.asarray(who, np.int64), held)


def referring(graph, registry):
    started = time.perf_counter()
    member, entity = ends(graph.member_source, graph.member_target)
    task = np.asarray(graph.member_task, np.int64)[member]
    stops = registry.stops[np.asarray(graph.task_kind, np.int64)[task]]
    result = (
        member,
        entity,
        np.asarray(graph.task_step, np.int64)[task],
        stops,
    )
    LOG.debug(
        "referring exit graph=%s ends=%d seconds=%.4f", graph.name, len(member), time.perf_counter() - started
    )
    return result


@timed("truth.unresolved")
def unresolved(graph, registry, member):
    task = int(graph.member_task[member])
    return "the reference %s of the task %d of the kind %s of graph %s" % (
        list(registry.schema.tasks["references"])[int(graph.member_name[member])],
        int(graph.task_key[task]),
        list(registry.schema.tasks["kinds"])[int(graph.task_kind[task])],
        graph.name,
    )


def placed(graph, registry, steps, entities):
    started = time.perf_counter()
    name, tasks = graph.name, len(graph.task_key)
    if (
        np.any(graph.task_step < 0)
        or np.any(graph.task_step >= steps)
        or np.any(np.diff(graph.task_step) < 0)
    ):
        raise ValueError("tasks of graph %s sit at steps outside 0..%d or out of order" % (name, steps - 1))
    if np.any(graph.task_kind < 0) or np.any(graph.task_kind >= len(registry.schema.tasks["kinds"])):
        raise ValueError("a task of graph %s is of a kind the schema does not state" % name)
    for label, values in (("reference", graph.member_task), ("label", graph.label_task)):
        if np.any(values < 0) or np.any(values >= tasks) or np.any(np.diff(values) < 0):
            raise ValueError(
                "a %s of graph %s names a task outside the graph or out of order" % (label, name)
            )
    if (
        np.any(graph.member_source < 0)
        or np.any(graph.member_source >= entities)
        or np.any(graph.member_target < -1)
        or np.any(graph.member_target >= entities)
    ):
        raise ValueError("a reference of graph %s names an entity outside 0..%d" % (name, entities - 1))
    if not registry.aimed[np.asarray(graph.task_kind, np.int64)[graph.label_task]].all():
        raise ValueError("a label of graph %s sits on a task whose kind states no target" % name)
    LOG.debug("placed exit graph=%s tasks=%d seconds=%.4f", name, tasks, time.perf_counter() - started)


def verify_dense(graph, registry, known=None):
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_step),
        fact=len(graph.fact_step),
        task=len(graph.task_key),
        member=len(graph.member_task),
        label=len(graph.label_task),
    )
    dimensions = dict(x_table=2, value_table=2, task_x=2, task_signals=3)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if (
            key not in dimensions
            and key not in ("obs_anchor", "fact_present", "entity_connected")
            and values.dtype.kind not in "iu"
        ):
            raise ValueError("graph %s requires integer indices in %s" % (name, key))
    if graph.obs_anchor.dtype != np.bool_ or graph.fact_present.dtype != np.bool_:
        raise ValueError("graph %s requires boolean anchor and presence arrays" % name)
    if (
        np.any(graph.obs_ref < 0)
        or np.any(graph.obs_ref >= len(graph.x_table))
        or np.any(graph.fact_ref < -1)
        or np.any(graph.fact_ref >= len(graph.value_table))
    ):
        raise ValueError("graph %s references rows outside its feature or value tables" % name)
    if len(graph.entity_token) != entities or len(graph.entity_connected) != entities or steps < 1:
        raise ValueError("graph %s requires one token and flag per entity and at least one step" % name)
    if len(graph.step_graph) != steps or np.any(np.diff(graph.step_graph) < 0):
        raise ValueError("graph %s needs one joined graph per step, never decreasing" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.x_table.shape[1] != registry.features or graph.task_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if not finite(graph.x_table) or not finite(graph.task_x):
        raise ValueError("graph %s observation and task features must be finite" % name)
    placed(graph, registry, steps, entities)
    flags = audit(
        graph.obs_entity,
        graph.obs_step,
        graph.obs_anchor,
        graph.fact_source,
        graph.fact_target,
        graph.fact_step,
        graph.fact_present,
        graph.fact_code,
        registry.canonical,
        steps,
        entities,
        np.zeros(entities, bool) if known is None else known,
    )
    for index, label in enumerate(("observation", "fact")):
        if flags[index]:
            raise ValueError("a %s row of graph %s has a step outside 0..%d" % (label, name, steps - 1))
    for index, label in enumerate(("observation", "fact source", "fact target"), 2):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if flags[5]:
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    for index, message in (
        (6, "observations of graph %s are not ordered by step"),
        (7, "facts of graph %s are not ordered by step with present facts first"),
        (8, "an entity of graph %s has two observations at one step"),
    ):
        if flags[index]:
            raise ValueError(message % name)
    for index, label, when in (
        (9, "present fact", "outside the anchors of references at its step"),
        (10, "absent fact", "at or before its step"),
    ):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity with no observation %s" % (label, name, when))
    member, entity, moments, stops = referring(graph, registry)
    lost = stops & unobserved(entities, graph.obs_entity, graph.obs_step, entity, moments)
    if lost.any():
        raise ValueError(
            "%s names a node not observed at the time of the task, and its kind states that this stops the work"
            % unresolved(graph, registry, member[np.flatnonzero(lost)[0]])
        )


def ordered(*keys):
    if len(keys[0]) < 2:
        return True
    order = np.lexsort(keys[::-1])
    return np.array_equal(order, np.arange(len(order)))


def verify(graph, registry):
    started = time.perf_counter()
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_start),
        fact=len(graph.fact_start),
        removal=len(graph.removal_step),
        task=len(graph.task_key),
        member=len(graph.member_task),
        label=len(graph.label_task),
    )
    dimensions = dict(
        obs_x=2,
        obs_values=2,
        fact_values=2,
        removal_values=2,
        task_x=2,
        task_values=2,
        task_signals=3,
        graph_values=1,
        entity_held=2,
    )
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if key not in dimensions and values.dtype.kind not in "iu":
            raise ValueError("graph %s requires integer indices in %s" % (name, key))
    if len(graph.keys) != entities or len(graph.entity_token) != entities or steps < 1:
        raise ValueError("graph %s requires one key and token per entity and at least one step" % name)
    if not np.array_equal(graph.entity_token, tokens(registry, graph.kinds, graph.keys)):
        raise ValueError("entity tokens of graph %s do not match its keys" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    layout = len(registry.encoding.layout)
    if graph.obs_values.shape[1] != layout or graph.task_values.shape[1] != layout:
        raise ValueError("graph %s holds values of a width its schema does not state" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if graph.fact_values.shape[1] != graph.removal_values.shape[1]:
        raise ValueError("graph %s fact and removal attribute widths differ" % name)
    if graph.graph_values.shape != (len(registry.schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    for label, start, stop in (
        ("observation", graph.obs_start, graph.obs_stop),
        ("fact", graph.fact_start, graph.fact_stop),
    ):
        if np.any(start < 0) or np.any(stop <= start) or np.any(stop > steps):
            raise ValueError("a %s span of graph %s lies outside 0..%d or is empty" % (label, name, steps))
    if np.any(graph.removal_step < 0) or np.any(graph.removal_step >= steps):
        raise ValueError("a removal row of graph %s has a step outside 0..%d" % (name, steps - 1))
    placed(graph, registry, steps, entities)
    for label, values in (
        ("observation", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
        ("removal source", graph.removal_source),
        ("removal target", graph.removal_target),
    ):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    for label, codes in (("facts", graph.fact_code), ("removals", graph.removal_code)):
        if (
            np.any(codes < 0)
            or np.any(codes >= len(registry.canonical))
            or not registry.canonical[codes].all()
        ):
            raise ValueError(
                "%s of graph %s carry codes that are not canonical relation codes" % (label, name)
            )
    if not ordered(graph.obs_start, graph.obs_entity):
        raise ValueError("observation spans of graph %s are not ordered by start and entity" % name)
    if not np.array_equal(np.sort(graph.fact_rank), np.arange(len(graph.fact_rank))):
        raise ValueError("fact ranks of graph %s are not a permutation" % name)
    if not ordered(graph.fact_start, graph.fact_rank):
        raise ValueError("fact spans of graph %s are not ordered by start and rank" % name)
    if np.any(np.diff(graph.removal_step) < 0):
        raise ValueError("removals of graph %s are not ordered by step" % name)
    moments, positions, moved = (np.asarray(graph.arrays[key]) for key in ORDER)
    change = np.zeros(steps + 1, np.int64)
    np.add.at(change, np.asarray(graph.fact_start), 1)
    np.add.at(change, np.asarray(graph.fact_stop), -1)
    active = np.cumsum(change)[:steps]
    if len(moments) and (
        np.any(np.diff(moments) < 0)
        or np.any(moments < 0)
        or np.any(moments >= steps)
        or np.any(moved < 0)
        or np.any(moved >= len(graph.fact_start))
        or np.any(positions < 0)
        or np.any(positions >= active[moments])
        or np.any(graph.fact_start[moved] > moments)
        or np.any(graph.fact_stop[moved] <= moments)
        or len(np.unique(moments.astype(np.int64) * (1 << 32) + positions)) != len(moments)
        or len(np.unique(moments.astype(np.int64) * (1 << 32) + moved)) != len(moments)
    ):
        raise ValueError("fact order patches of graph %s are malformed" % name)
    order = np.lexsort((graph.obs_start, graph.obs_entity))
    entity, start, stop = graph.obs_entity[order], graph.obs_start[order], graph.obs_stop[order]
    same = entity[1:] == entity[:-1]
    if np.any(same & (start[1:] < stop[:-1])):
        raise ValueError("an entity of graph %s has two observations at one step" % name)
    payload = spans.flat(np.asarray(graph.obs_values)[order])
    if np.any(same & (start[1:] == stop[:-1]) & np.all(payload[1:] == payload[:-1], axis=1)):
        raise ValueError("observation spans of graph %s are not maximal" % name)
    offsets, lows, highs = spans.coverage(
        np.asarray(graph.obs_entity), np.asarray(graph.obs_start), np.asarray(graph.obs_stop), entities
    )
    for who in (graph.fact_source, graph.fact_target):
        row = spans.all_covered(offsets, lows, highs, np.asarray(who), graph.fact_start, graph.fact_stop)
        if row >= 0:
            end = int(who[row])
            raise ValueError(
                "the edge of %s from %s to %s of graph %s holds over the steps %d to %d, and at one of them "
                "its end %s, a node of the kind %s, has no observation"
                % (
                    registry.by_code[int(graph.fact_code[row])].name,
                    graph.keys[int(graph.fact_source[row])],
                    graph.keys[int(graph.fact_target[row])],
                    name,
                    int(graph.fact_start[row]),
                    int(graph.fact_stop[row]) - 1,
                    graph.keys[end],
                    registry.nodes[int(graph.kinds[end])],
                )
            )
    first = np.full(entities, steps, np.int64)
    np.minimum.at(first, graph.obs_entity, graph.obs_start)
    if np.any(first[graph.removal_source] > graph.removal_step) or np.any(
        first[graph.removal_target] > graph.removal_step
    ):
        raise ValueError(
            "an absent fact of graph %s names an entity with no observation at or before its step" % name
        )
    member, entity, moments, stops = referring(graph, registry)
    chosen = np.flatnonzero(stops)
    lost = spans.all_covered(
        offsets,
        lows,
        highs,
        np.ascontiguousarray(entity[chosen]),
        np.ascontiguousarray(moments[chosen]),
        np.ascontiguousarray(moments[chosen] + 1),
    )
    if lost >= 0:
        raise ValueError(
            "%s names a node not observed at the time of the task, and its kind states that this stops the work"
            % unresolved(graph, registry, member[chosen[lost]])
        )
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    LOG.debug("verify exit graph=%s seconds=%.4f", name, time.perf_counter() - started)


@njit(cache=True)
def audit(
    obs_entity,
    obs_step,
    obs_anchor,
    fact_source,
    fact_target,
    fact_step,
    fact_present,
    fact_code,
    canonical,
    steps,
    entities,
    known,
):
    flags = np.zeros(11, np.bool_)
    observations, facts = len(obs_step), len(fact_step)
    for row in range(observations):
        if obs_step[row] < 0 or obs_step[row] >= steps:
            flags[0] = True
        if obs_entity[row] < 0 or obs_entity[row] >= entities:
            flags[2] = True
        if row > 0 and obs_step[row] < obs_step[row - 1]:
            flags[6] = True
    for row in range(facts):
        if fact_step[row] < 0 or fact_step[row] >= steps:
            flags[1] = True
        if fact_source[row] < 0 or fact_source[row] >= entities:
            flags[3] = True
        if fact_target[row] < 0 or fact_target[row] >= entities:
            flags[4] = True
        if fact_code[row] < 0 or fact_code[row] >= len(canonical) or not canonical[fact_code[row]]:
            flags[5] = True
        if row > 0 and np.int64(fact_step[row]) * 2 + (0 if fact_present[row] else 1) < np.int64(
            fact_step[row - 1]
        ) * 2 + (0 if fact_present[row - 1] else 1):
            flags[7] = True
    if flags.any():
        return flags
    stamp = np.full(entities, -1, np.int64)
    structure = np.full(entities, -1, np.int64)
    first = np.full(entities, steps, np.int64)
    for who in range(entities):
        if known[who]:
            first[who] = -1
    cursor = 0
    for row in range(facts):
        moment = np.int64(fact_step[row])
        while cursor < observations and obs_step[cursor] <= moment:
            who, at = obs_entity[cursor], np.int64(obs_step[cursor])
            if stamp[who] == at:
                flags[8] = True
            stamp[who] = at
            if not obs_anchor[cursor]:
                structure[who] = at
            if first[who] == steps:
                first[who] = at
            cursor += 1
        if fact_present[row]:
            if structure[fact_source[row]] != moment or structure[fact_target[row]] != moment:
                flags[9] = True
        elif first[fact_source[row]] > moment or first[fact_target[row]] > moment:
            flags[10] = True
    while cursor < observations:
        who, at = obs_entity[cursor], np.int64(obs_step[cursor])
        if stamp[who] == at:
            flags[8] = True
        stamp[who] = at
        cursor += 1
    return flags
