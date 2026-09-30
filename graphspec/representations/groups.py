import logging
import math
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed
from graphspec.infra.workspace import allocate, block_rows, changed, finite
from graphspec.operations.kernels import first_rows
from graphspec.operations.stages import enabled
from graphspec.schema.declaration import BOUNDARIES
from graphspec.schema.references import SORTS, ends

from . import kernels

LOG = logging.getLogger(__name__)
EVENT_KINDS = ("observation", "relation", "task")
WALK_SAMPLES, WALK_HOPS, PERIOD_LOOKBACK = 32, 4, 3
WALK_STREAM, PERIOD_STREAM = 16, 32
EMPTY = np.zeros(0, np.int64)


def lags(maximum):
    values = tuple(2**i for i in range(int(math.log2(maximum)) + 1))
    return values if values[-1] == maximum else (*values, maximum)


def replay(
    entity,
    step,
    altered,
    present,
    present_step,
    absent,
    absent_step,
    steps,
    seen,
    name,
    last,
    base=0,
    initial=(EMPTY, EMPTY, EMPTY, EMPTY),
):
    events, status, *table = kernels.replayed(
        entity.astype(np.int64),
        step.astype(np.int64),
        altered,
        present[:, 0].astype(np.int64),
        present[:, 1].astype(np.int64),
        present[:, 2].astype(np.int64),
        present_step.astype(np.int64),
        absent[:, 0].astype(np.int64),
        absent[:, 1].astype(np.int64),
        absent[:, 2].astype(np.int64),
        absent_step.astype(np.int64),
        np.asarray(seen["first"], np.int64),
        np.asarray(seen["second"], np.int64),
        np.ascontiguousarray(seen["starts"], np.int64),
        np.asarray(seen["kind"], np.int64),
        np.asarray(seen["task"], np.int64),
        np.asarray(seen["offsets"], np.int64),
        steps,
        last,
        np.int64(base),
        *(np.asarray(values, np.int64) for values in initial),
    )
    if status < 0:
        raise ValueError("an absent fact of graph %s names an entity before its first observation" % name)
    return events, table


def observed_features(arrays):
    return np.asarray(arrays["x_table"])[arrays["x_ref"]]


def referenced_change(table, left, right):
    result = left != right
    differ = np.flatnonzero(result)
    result[differ] = changed(table, left[differ], right[differ])
    return result


def named(graph, registry, tasks, name):
    started = time.perf_counter()
    chosen = np.flatnonzero(np.asarray(graph.member_name) == registry.references.index(name))
    found = np.full(len(graph.task_key), -1, np.int64)
    found[np.asarray(graph.member_task)[chosen]] = np.asarray(graph.member_source)[chosen]
    LOG.debug(
        "named exit reference=%s tasks=%d seconds=%.4f", name, len(tasks), time.perf_counter() - started
    )
    return found[tasks]


def seen_tasks(graph, registry, observed):
    started = time.perf_counter()
    kinds = registry.seen[np.asarray(graph.task_kind, np.int64)]
    tasks = np.flatnonzero(kinds >= 0)
    first, second = np.full(len(tasks), -1, np.int64), np.full(len(tasks), -1, np.int64)
    for code, kind in enumerate(registry.events):
        members = kinds[tasks] == code
        left, right = registry.schema.tasks["kinds"][kind]["event"]
        first[members] = named(graph, registry, tasks[members], left)
        second[members] = named(graph, registry, tasks[members], right)
    moments = np.asarray(graph.task_step, np.int64)[tasks]
    result = dict(
        first=first,
        second=second,
        starts=np.stack((observed(first, moments), observed(second, moments)), 1).reshape(-1, 2),
        kind=kinds[tasks],
        task=tasks,
        step=moments,
        offsets=np.searchsorted(moments, np.arange(graph.steps + 1)),
    )
    LOG.debug(
        "seen tasks exit graph=%s events=%d seconds=%.4f",
        graph.name,
        len(tasks),
        time.perf_counter() - started,
    )
    return result


def referred(graph, registry, rows, found):
    started = time.perf_counter()
    place = np.full(len(graph.task_key), -1, np.int64)
    place[rows] = np.arange(len(rows))
    owner = place[np.asarray(graph.member_task, np.int64)]
    names = np.asarray(graph.member_name, np.int64)
    moments = np.asarray(graph.task_step, np.int64)[np.asarray(graph.member_task, np.int64)]
    source, target = np.asarray(graph.member_source, np.int64), np.asarray(graph.member_target, np.int64)
    first, second = np.full(len(owner), -1, np.int64), np.full(len(owner), -1, np.int64)
    listed = list(registry.schema.tasks["kinds"].values())
    kinds = np.asarray(graph.task_kind)[rows].tolist()
    pairs, ordered = [], np.zeros(len(registry.references), bool)
    for code, (name, entry) in enumerate(registry.schema.tasks["references"].items()):
        sort = SORTS.named(entry["sort"])
        ordered[code] = sort.ordered
        if sort.whole:
            pairs += [(row, code) for row, kind in enumerate(kinds) if name in listed[kind]["references"]]
            continue
        chosen = np.flatnonzero((names == code) & (owner >= 0))
        first[chosen], second[chosen] = sort.resolved(source[chosen], target[chosen], moments[chosen], found)
    member, entity = ends(source, target)
    kept = owner[member] >= 0
    member, entity = member[kept], entity[kept]
    side = np.r_[False, member[1:] == member[:-1]][: len(member)]
    place = np.asarray(graph.member_place, np.int64)[member]
    order = np.lexsort((side, place, names[member], owner[member]))
    member, entity, side = member[order], entity[order], side[order]
    place = 2 * place[order] + side
    result = dict(
        member_row=owner[member].astype(np.int32),
        member_name=names[member].astype(np.int32),
        member_entity=entity.astype(np.int32),
        member_observed=np.where(side, second[member], first[member]).astype(np.int32),
        member_place=np.where(ordered[names[member]], place, 0).astype(np.int32),
        whole_row=np.asarray([row for row, _ in pairs], np.int32),
        whole_name=np.asarray([name for _, name in pairs], np.int32),
    )
    LOG.debug(
        "referred exit graph=%s members=%d seconds=%.4f",
        graph.name,
        len(member),
        time.perf_counter() - started,
    )
    return result


def carried_rows(link, entity, x_table):
    last = link.table("last")
    found = last.find(link.glob[entity])
    rows = np.full(len(entity), -1, np.int64)
    known = found >= 0
    values = np.zeros((len(entity), x_table.shape[1]), x_table.dtype)
    if known.any():
        rows[known] = last.columns["row"][found[known]]
        values[known] = last.columns["x"][found[known]]
    return rows, values


@timed("base")
def base(graph, registry, spec, link=None):
    steps, kinds = graph.steps, graph.kinds
    step, entity, reference = graph.obs_step, graph.obs_entity, graph.obs_ref
    lookup = step.astype(np.int64) * len(kinds) + entity
    by_key = np.argsort(lookup, kind="stable")
    sorted_keys = lookup[by_key]

    def found(owners, moments):
        probe = np.asarray(moments, np.int64) * len(kinds) + owners
        positions = np.minimum(np.searchsorted(sorted_keys, probe), max(len(by_key) - 1, 0))
        held = (sorted_keys[positions] == probe) if len(by_key) else np.zeros(len(probe), bool)
        return np.where(held & (np.asarray(owners) >= 0), by_key[positions] if len(by_key) else -1, -1)

    def observed(owners, moments):
        rows = found(owners, moments)
        if np.any(rows < 0):
            raise ValueError("a row of graph %s names an entity with no observation at its step" % graph.name)
        return rows.astype(np.int32)

    present = graph.fact_present
    codes, fact_step, values = (
        graph.fact_code[present],
        graph.fact_step[present],
        graph.fact_ref[present],
    )
    source, target = (
        observed(graph.fact_source[present], fact_step),
        observed(graph.fact_target[present], fact_step),
    )
    symmetric = np.asarray(
        [edge.symmetric for edge in registry.edges] + [False] * (registry.relations - len(registry.edges))
    )[codes]
    direction = spec.get("relations/direction")
    doubled = symmetric | (direction != "directed")
    reverse = np.where(symmetric | (direction == "symmetric"), codes, registry.flipped[codes])
    index = np.repeat(np.arange(len(codes)), np.where(doubled, 2, 1))
    flip = np.r_[False, index[1:] == index[:-1]] if len(index) else np.zeros(0, bool)
    spatial = np.stack(
        (
            np.where(flip, target[index], source[index]),
            np.where(flip, source[index], target[index]),
            np.where(flip, reverse[index], codes[index]),
        ),
        1,
    ).astype(np.int32)
    attribute_row = np.full(len(codes), -1, np.int32)
    arrays = {}
    valued = registry.value_group[codes]
    for place, group in enumerate(registry.valued):
        use = valued == place
        attribute_row[use] = np.arange(int(use.sum())) + (0 if link is None else link.attributes.get(group, 0))
        chosen, width = values[use], len(registry.attributes(group))
        rows = np.full((len(chosen), width), np.nan, np.float32)
        rows[chosen >= 0] = graph.value_table[chosen[chosen >= 0], :width]
        arrays["attributes_" + group] = rows
    seen = seen_tasks(graph, registry, observed)
    rows = np.flatnonzero(registry.aimed[np.asarray(graph.task_kind, np.int64)])
    row_step = np.asarray(graph.task_step)[rows]
    place = np.full(len(graph.task_key), -1, np.int64)
    place[rows] = np.arange(len(rows))
    row_index = np.full(len(rows), -1, np.int64)
    row_index[place[graph.label_task]] = graph.label_row
    absent = ~graph.fact_present
    order = np.argsort(entity, kind="stable")
    altered = np.ones(len(entity), bool)
    if len(order) > 1:
        owners = entity[order]
        altered[order[1:]] = (owners[1:] != owners[:-1]) | referenced_change(
            graph.x_table, reference[order[1:]], reference[order[:-1]]
        )
    last = np.full(len(kinds), -1, np.int64)
    initial = (EMPTY, EMPTY, EMPTY, EMPTY)
    if link is not None:
        heads = order[np.r_[True, entity[order][1:] != entity[order][:-1]]] if len(order) else order
        prior_rows, prior_x = carried_rows(link, entity[heads], graph.x_table)
        carried = prior_rows >= 0
        altered[heads[carried]] = np.any(
            np.asarray(graph.x_table)[reference[heads[carried]]] != prior_x[carried], axis=1
        )
        last[:] = carried_rows(link, np.arange(len(kinds)), graph.x_table)[0]
        initial = link.replay_keys(graph)
    events, table = replay(
        entity,
        step,
        altered,
        np.stack((source, target, codes), 1),
        fact_step,
        np.stack((graph.fact_source[absent], graph.fact_target[absent], graph.fact_code[absent]), 1),
        graph.fact_step[absent],
        steps,
        seen,
        graph.name,
        last,
        0 if link is None else link.obs0,
        initial,
    )
    arrays.update(
        x_ref=reference.astype(np.int32),
        x_table=np.asarray(graph.x_table).astype(np.float16, copy=False),
        entity=entity.astype(np.int32),
        entity_token=(np.asarray(graph.entity_token) & np.uint64((1 << 62) - 1)).astype(np.int64),
        step=step.astype(np.int32),
        obs_period=graph.period[step].astype(np.int32),
        snapshot_offsets=np.searchsorted(step, np.arange(steps + 1)).astype(np.int64),
        entity_kind=(np.zeros_like(kinds) if enabled(spec, "nodes/untyped") else kinds).astype(np.int32),
        spatial=spatial,
        spatial_attribute_row=attribute_row[index],
        spatial_step=fact_step[index].astype(np.int32),
        observation_ids=np.argsort(entity, kind="stable").astype(np.int32),
        observation_offsets=np.r_[0, np.cumsum(np.bincount(entity, minlength=len(kinds)))].astype(np.int64),
        task=np.asarray(graph.task_x).astype(np.float16, copy=False),
        event_first=seen["first"].astype(np.int32),
        event_task=seen["task"].astype(np.int32),
        event_step=seen["step"].astype(np.int32),
        row_step=np.asarray(row_step, np.int32),
        row_task=rows.astype(np.int32),
        row_kind=np.asarray(graph.task_kind, np.int32)[rows],
        events=events,
        signals=np.asarray(graph.task_signals, np.float32)[rows],
        row_index=row_index,
        key=np.asarray(graph.task_key, np.int64)[rows],
        period=graph.period.astype(np.int32),
        **referred(graph, registry, rows, found),
    )
    if link is not None:
        link.remember(graph, entity, reference, table)
    return arrays


@timed("periods")
def periods(arrays, kinds, budget):
    held = np.asarray(arrays["period"])
    first = np.r_[0, np.flatnonzero(np.diff(held)) + 1].astype(np.int32)
    last = np.r_[first[1:] - 1, len(held) - 1].astype(np.int32)
    membership = np.repeat(np.arange(len(first)), last - first + 1).astype(np.int32)
    order = arrays["observation_ids"]
    observed_keys = allocate((len(order),), np.int64)
    width = block_rows(observed_keys)
    for start in range(0, len(order), width):
        ids = order[start : start + width]
        observed_keys[start : start + width] = (
            arrays["entity"][ids].astype(np.int64) * len(last) + membership[arrays["step"][ids]]
        )
    return dict(
        period_starts=first,
        period_ends=last,
        period_membership=membership,
        period_numbers=held[first],
        period_observation_keys=observed_keys,
        **affinity_index(np.asarray(arrays["events"]), len(held), kinds),
    )


@timed("representations.tick_span")
def tick_span(steps):
    return 2 * np.int64(steps) + 2


def affinity_index(events, steps, kinds):
    started = time.perf_counter()
    seen = events[events[:, 3] == 2]
    entity, tick, kind = (seen[:, column].astype(np.int64) for column in (0, 2, 6))
    order = np.lexsort((tick, entity))
    counts = np.zeros((len(order), kinds), np.float32)
    counts[np.arange(len(order)), kind[order]] = 1
    counts = np.cumsum(counts, 0)
    owner = entity[order]
    head = np.maximum.accumulate(
        np.where(np.r_[True, owner[1:] != owner[:-1]][: len(owner)], np.arange(len(owner)), 0)
    )
    counts -= np.where((head > 0)[:, None], counts[np.maximum(head - 1, 0)], 0)
    totals = np.zeros((steps + 1, kinds), np.float32)
    np.add.at(totals, (tick // 2 + 1, kind), 1)
    result = dict(
        affinity_keys=owner * tick_span(steps) + tick[order],
        affinity_counts=counts,
        affinity_steps=np.cumsum(totals, 0),
    )
    LOG.debug(
        "affinity index exit events=%d kinds=%d seconds=%.4f",
        len(order),
        kinds,
        time.perf_counter() - started,
    )
    return result


def marked_edges(pieces, period, codes):
    if not pieces:
        return np.empty((0, 4), np.int32)
    values = np.concatenate(pieces).astype(np.int64)
    if not len(values):
        return np.empty((0, 4), np.int32)
    ids, count = kernels.keyed2(values[:, 0], values[:, 1])
    flags = kernels.combined(ids, values[:, 2], count)
    kept = first_rows(ids, count)
    source, destination = values[kept, 0], values[kept, 1]
    return ordered(
        source,
        destination,
        codes[flags[ids[kept]]],
        period[destination] - period[source],
        int(destination.max(initial=-1)) + 1,
    )


def ordered(source, destination, relation, gap, size):
    order = kernels.by_target(np.asarray(source, np.int64), np.asarray(destination, np.int64), size)
    output = np.empty((len(order), 4), np.int32)
    for column, values in enumerate((source, destination, relation, gap)):
        output[:, column] = np.asarray(values)[order]
    return output


def boundary_codes(registry):
    return np.asarray([-1] + [registry.code(name) for name in BOUNDARIES], np.int32)


def coupling_setting(spec, steps, rows):
    scheme = spec["coupling/scheme"]
    chosen, width, cap, limit = np.zeros(0, np.int64), 0, 0, 0
    if scheme in ("window", "all"):
        mode = 0
        width, cap = (spec["coupling/window"], spec["coupling/cap"]) if scheme == "window" else (steps, rows)
    else:
        chosen = np.asarray((1,) if scheme == "ordinal" else lags(spec["coupling/max_lag"]), np.int64)
        if spec["coupling/unit"] == "observation":
            mode = 1
            limit = spec["coupling/lag_window"] if scheme == "lags" else steps
        else:
            mode = 2
    return chosen, mode, width, cap, limit


def coupling_edges(entity, step, spec, steps, rows, entities, budget):
    chosen, mode, width, cap, limit = coupling_setting(spec, steps, rows)
    return kernels.coupled(
        np.ascontiguousarray(entity, np.int64),
        np.ascontiguousarray(step, np.int64),
        chosen,
        mode,
        width,
        cap,
        limit,
        entities,
        budget.pair_limit,
    )


def grouped(period):
    return np.cumsum(np.r_[0, np.diff(period) != 0]).astype(np.int32)


@timed("coupling")
def coupling(arrays, registry, spec, budget):
    step, entity = arrays["step"], arrays["entity"]
    source, target = coupling_edges(
        entity, step, spec, len(arrays["period"]), len(entity), len(arrays["entity_kind"]), budget
    )
    budget.retain(len(source), "temporal coupling")
    lookback = spec["coupling/periods"]
    held = arrays["period"]
    return dict(
        coupling=np.stack((source, target), 1).astype(np.int32).reshape(-1, 2),
        coupling_boundaries=observation_boundaries(
            entity,
            step,
            grouped(held)[step],
            held[step],
            lookback,
            spec["coupling/period_window"],
            len(arrays["entity_kind"]),
            int(grouped(held)[-1]) + 1,
            registry,
            budget,
        )
        if lookback
        else np.zeros((0, 4), np.int32),
    )


def observation_boundaries(entity, step, groups, period, lookback, window, entities, count, registry, budget):
    shape = (count, entities)
    budget.workspace(shape[0] * shape[1] * 8, "observation boundary tables")
    budget.pairs(len(entity) * lookback * 2, "observation boundary edges")
    first = np.full(shape, len(entity), np.int32)
    last = np.full(shape, -1, np.int32)
    ids = np.arange(len(entity), dtype=np.int32)
    np.minimum.at(first, (groups, entity), ids)
    np.maximum.at(last, (groups, entity), ids)
    pieces = []
    for lag in range(1, lookback + 1):
        prior = groups - lag
        for endpoints, flag in ((first, 1), (last, 2)):
            source = endpoints[prior.clip(0), entity]
            valid = (prior >= 0) & (source >= 0) & (source < len(entity))
            selected = ids[valid]
            source = source[valid]
            keep = step[selected] - step[source] < window
            pieces.append(np.column_stack((source[keep], selected[keep], np.full(keep.sum(), flag))))
    result = marked_edges(pieces, period, boundary_codes(registry))
    budget.retain(len(result), "observation boundary edges")
    return result


def event_setting(spec):
    link = spec["events/link"]
    chosen = np.asarray(
        (1,) if link == "next" else lags(spec["events/max_lag"]) if link == "lags" else (), np.int64
    )
    return chosen, link == "window", spec["events/window"], spec["events/cap"]


def event_members(events, spec):
    return np.isin(events[:, 3], [EVENT_KINDS.index(kind) for kind in spec["events/kinds"].split("+")])


def event_edges(events, member, spec, entities, budget):
    chosen, windowed, window, cap = event_setting(spec)
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    return kernels.event_dag(
        source, target, ticks, member, chosen, windowed, window, cap, entities, budget.pair_limit
    )


@timed("event_graph")
def event_graph(arrays, registry, spec, budget):
    events = np.asarray(arrays["events"])
    member = event_members(events, spec)
    before, after = event_edges(events, member, spec, len(arrays["entity_kind"]), budget)
    budget.retain(len(before), "event graph")
    lookback = spec["events/periods"]
    held = arrays["period"]
    moments = events[:, 2] // 2
    return dict(
        event_dag=np.stack((before, after), 1).astype(np.int32).reshape(-1, 2),
        event_member=member,
        event_boundaries=event_boundaries(
            events,
            member,
            grouped(held)[moments].astype(np.int64),
            held[moments],
            lookback,
            spec["events/period_window"],
            len(arrays["entity_kind"]),
            registry,
            budget,
        )
        if lookback
        else np.zeros((0, 4), np.int32),
    )


def event_boundaries(events, member, groups, period, lookback, window, entities, registry, budget):
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    before, after, flags = kernels.boundaries(
        source,
        target,
        ticks,
        member,
        np.ascontiguousarray(groups, np.int64),
        lookback,
        window,
        entities,
        budget.pair_limit,
    )
    budget.retain(len(before), "event boundary edges")
    return ordered(
        before, after, boundary_codes(registry)[flags], period[after] - period[before], len(events)
    )


def neighbor_members(events, spec):
    kinds = spec["neighbors/kinds"].split("+")
    use = (events[:, 3] == 2) & ("task" in kinds)
    if "relation" in kinds:
        use |= (events[:, 3] == 1) & (spec["neighbors/removals"] | (events[:, 8] == 1))
    return use


def neighbor_lists(events, use, entities):
    ids = np.flatnonzero(use)
    source, target = events[ids, 0], events[ids, 1]
    distinct = target != source
    owner, event = np.r_[source, target[distinct]], np.r_[ids, ids[distinct]]
    order = np.lexsort((event, owner))
    return (
        event[order].astype(np.int32),
        np.r_[0, np.cumsum(np.bincount(owner, minlength=entities))].astype(np.int64),
    )


@timed("neighbors")
def neighbors(arrays, spec):
    events = arrays["events"]
    ids, offsets = neighbor_lists(events, neighbor_members(events, spec), len(arrays["entity_kind"]))
    return dict(neighbor_ids=ids, neighbor_offsets=offsets)


def walk_arrays(ids, offsets, events, roots, identities, moments, prior, first, last, spec, seed, budget):
    bias, window, fanout = spec["walks/bias"], spec["walks/window"], spec["walks/fanout"]
    count = len(moments)
    budget.workspace(
        count * WALK_SAMPLES * (1 + PERIOD_LOOKBACK) * (64 + WALK_HOPS * 4), "sampled walk representations"
    )
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    member, draw = (
        axis.reshape(-1) for axis in np.meshgrid(np.arange(count), np.arange(WALK_SAMPLES), indexing="ij")
    )
    identity = np.stack((identities[member], moments[member], draw), 1)
    result = kernels.walked(
        ids,
        offsets,
        source,
        target,
        ticks,
        roots[member],
        np.maximum(0, (moments[member] - window + 1) * 2),
        moments[member] * 2 + 1,
        np.ones(len(member), np.bool_),
        identity,
        fanout,
        bias,
        seed,
        WALK_STREAM,
        WALK_HOPS,
    )
    member, back, draw = (
        axis.reshape(-1)
        for axis in np.meshgrid(
            np.arange(count), np.arange(PERIOD_LOOKBACK), np.arange(WALK_SAMPLES), indexing="ij"
        )
    )
    identity = np.stack((identities[member], moments[member], back, draw), 1)
    period_walks = kernels.walked(
        ids,
        offsets,
        source,
        target,
        ticks,
        roots[member],
        2 * first[member, back],
        2 * last[member, back] + 2,
        np.ascontiguousarray(prior[member, back] >= 0),
        identity,
        fanout,
        bias,
        seed,
        PERIOD_STREAM,
        WALK_HOPS,
    )
    return dict(
        walks=result.reshape((count, WALK_SAMPLES, WALK_HOPS)),
        period_walks=period_walks.reshape((count, PERIOD_LOOKBACK, WALK_SAMPLES, WALK_HOPS)),
    )


def prior_periods(membership, moments):
    return membership[moments][:, None] - np.arange(1, PERIOD_LOOKBACK + 1)


@timed("walks")
def walks(arrays, spec, seed, budget):
    roots = np.asarray(arrays["member_entity"], np.int64)
    moments = np.asarray(arrays["row_step"], np.int64)[np.asarray(arrays["member_row"], np.int64)]
    prior = prior_periods(arrays["period_membership"], moments)
    return walk_arrays(
        np.asarray(arrays["neighbor_ids"]),
        np.asarray(arrays["neighbor_offsets"]),
        np.asarray(arrays["events"]),
        roots,
        roots,
        moments,
        prior,
        arrays["period_starts"][prior.clip(0)].astype(np.int64),
        arrays["period_ends"][prior.clip(0)].astype(np.int64),
        spec,
        seed,
        budget,
    )


def mixed_sizes(rows, moments, trace, changes, bundles, seen, carried=0):
    groups = (rows + carried) // trace if trace else 0
    limit = rows + (rows if changes else 0) + groups + (moments if bundles else 0) + seen
    wires = rows + (2 * rows if changes else 0) + groups * (trace + 1) + (rows if bundles else 0) + 2 * seen
    holders = rows + (2 * rows if changes else 0) + groups * trace + (rows if bundles else 0)
    return limit, wires, holders


def mixed_features(
    kind, mode, fill, member_offset, member_id, rows, columns, width, task, offsets, spec, base=0
):
    changes, trace, bundles = spec["mixed/changes"], spec["mixed/trace"], spec["mixed/bundles"]
    features = allocate((len(kind), columns), np.float16)
    chunk = block_rows(features)
    for start in range(0, len(features), chunk):
        stop = start + chunk
        part = features[start:stop]
        part[:] = 0
        observed = mode[start:stop] == 0
        part[observed] = rows(fill[start:stop][observed].astype(np.int64) + base)
        if changes:
            picked = np.flatnonzero(mode[start:stop] == 1) + start
            if len(picked):
                current = rows(member_id[member_offset[picked] + 1])
                previous = rows(member_id[member_offset[picked]])
                part[picked - start] = np.concatenate(
                    (current[:, :width].astype(np.float32) - previous[:, :width], current[:, width:]), 1
                ).astype(np.float16)
        picked = mode[start:stop] == 3
        part[picked] = task[fill[start:stop][picked]]
    if trace:
        picked = np.flatnonzero(mode == 2)
        chunk = max(1, chunk // trace)
        for start in range(0, len(picked), chunk):
            nodes = picked[start : start + chunk]
            members = member_id[member_offset[nodes][:, None] + np.arange(trace)]
            features[nodes] = rows(members.reshape(-1)).reshape(len(nodes), trace, -1).mean(1)
    if bundles:
        for node in np.flatnonzero(mode == 5):
            features[node] = rows(np.arange(offsets[fill[node]], offsets[fill[node] + 1]) + base).mean(0)
    return features


@timed("mixed")
def mixed(arrays, spec, budget, values):
    x, entity, offsets = observed_features(arrays), arrays["entity"], arrays["snapshot_offsets"]
    changes, trace, bundles = spec["mixed/changes"], spec["mixed/trace"], spec["mixed/bundles"]
    rows, moments = len(entity), len(offsets) - 1
    placed = np.searchsorted(np.asarray(arrays["event_step"], np.int64), np.arange(moments + 1))
    entities = len(arrays["entity_kind"])
    order = np.argsort(entity, kind="stable")
    prior = np.full(rows, -1, np.int64)
    if rows > 1:
        same = entity[order][1:] == entity[order][:-1]
        prior[order[1:][same]] = order[:-1][same]
    altered = np.zeros(rows, bool)
    if changes:
        seen = np.flatnonzero(prior >= 0)
        altered[seen] = changed(x, seen, prior[seen])
    limit, wires, holders = mixed_sizes(rows, moments, trace, changes, bundles, int(placed[-1]))
    budget.pairs(wires, "mixed representation edges")
    budget.workspace(limit * (32 + x.shape[1] * 2) + wires * 12 + holders * 4, "mixed representation")
    span = trace if trace > 0 else 1
    kind, tick, owner, mode, fill, member_offset, member_id, source, target, flag, _ = kernels.bundled(
        np.ascontiguousarray(entity, np.int64),
        np.ascontiguousarray(offsets, np.int64),
        prior,
        altered,
        np.asarray(arrays["event_first"], np.int64),
        np.asarray(arrays["event_task"], np.int64),
        placed,
        trace,
        bundles,
        limit,
        wires,
        holders,
        np.int64(0),
        np.int64(0),
        np.full(entities, -1, np.int64),
        np.empty((entities, span), np.int64),
        np.empty((entities, span), np.int64),
        np.zeros(entities, np.int64),
        np.int64(-1),
    )
    features = mixed_features(
        kind,
        mode,
        fill,
        member_offset,
        member_id,
        lambda ids: x[ids],
        x.shape[1],
        values,
        arrays["task"],
        offsets,
        spec,
    )
    return dict(
        mixed_x=features.reshape(-1, x.shape[1]),
        mixed_kind=kind,
        mixed_tick=tick,
        mixed_entity=owner,
        mixed_edges=np.stack((source, target, flag), 1).astype(np.int32).reshape(-1, 3),
        mixed_member_ids=member_id,
        mixed_member_offsets=member_offset,
    )


@timed("build")
def build(graph, registry, spec, groups, seed, limits):
    started = time.perf_counter()
    budget = GraphBudget(limits)
    arrays = base(graph, registry, spec)
    parts = (
        ("periods", True, lambda: periods(arrays, len(registry.events), budget)),
        ("coupling", "coupling" in groups, lambda: coupling(arrays, registry, spec, budget)),
        ("events", "events" in groups, lambda: event_graph(arrays, registry, spec, budget)),
        ("neighbors", "neighbors" in groups or "walks" in groups, lambda: neighbors(arrays, spec)),
        ("walks", "walks" in groups, lambda: walks(arrays, spec, seed, budget)),
        ("mixed", "mixed" in groups, lambda: mixed(arrays, spec, budget, registry.values)),
    )
    for name, wanted, make in parts:
        if wanted:
            clock = time.perf_counter()
            arrays.update(make())
            LOG.info(
                "representation group exit graph=%s group=%s seconds=%.3f",
                graph.name,
                name,
                time.perf_counter() - clock,
            )
    LOG.info(
        "representation exit graph=%s seconds=%.3f events=%d groups=%s",
        graph.name,
        time.perf_counter() - started,
        len(arrays["events"]),
        "+".join(groups),
    )
    return arrays


@timed("validate")
def validate(arrays, graph, registry, spec, groups):
    x, step, events, entity = arrays["x_table"], arrays["step"], arrays["events"], arrays["entity"]
    reference = arrays["x_ref"]
    if len(reference) != len(entity) or (
        len(reference) and (reference.min() < 0 or reference.max() >= len(x))
    ):
        raise ValueError("observation feature references out of range in graph %s" % graph.name)
    if not finite(x):
        raise ValueError("nonfinite observation features in graph %s" % graph.name)
    sizes = np.asarray([len(arrays["attributes_" + group]) for group in registry.valued] + [-1], np.int64)
    limits = sizes[registry.value_group]
    fault = kernels.misplaced_attributes(
        np.asarray(arrays["spatial"]), np.asarray(arrays["spatial_attribute_row"]), limits
    )
    if fault >= 0:
        raise ValueError("attribute rows of relation code %d are missing, unexpected or out of range" % fault)
    period = arrays["period"]
    if "coupling" in groups:
        source, destination = arrays["coupling"].T
        if np.any(step[source] >= step[destination]) or np.any(entity[source] != entity[destination]):
            raise ValueError(
                "coupling edges of graph %s must link an entity to its own earlier observation" % graph.name
            )
        source, destination, relation, gaps = arrays["coupling_boundaries"].T
        if (
            np.any(entity[source] != entity[destination])
            or np.any(period[step[source]] >= period[step[destination]])
            or not np.isin(relation, boundary_codes(registry)[1:]).all()
            or not np.array_equal(gaps, period[step[destination]] - period[step[source]])
            or np.any(
                step[destination] - step[source] >= spec["coupling/period_window"]
                if spec["coupling/periods"]
                else False
            )
        ):
            raise ValueError("coupling boundary edges of graph %s are invalid" % graph.name)
    if "events" in groups:
        member = arrays["event_member"]
        source, destination = arrays["event_dag"].T
        if np.any(events[source, 2] >= events[destination, 2]) or not (
            member[source].all() and member[destination].all()
        ):
            raise ValueError(
                "event graph of graph %s links a later, simultaneous or excluded event" % graph.name
            )
        if kernels.check_boundaries(
            np.asarray(arrays["event_boundaries"]),
            np.asarray(events),
            np.asarray(member),
            np.asarray(period),
            boundary_codes(registry)[1:],
        ):
            raise ValueError("event boundary edges of graph %s are invalid" % graph.name)
    if "walks" in groups:
        moments = np.asarray(arrays["row_step"], np.int64)[np.asarray(arrays["member_row"], np.int64)]
        for name, limit in (
            ("walks", moments[:, None] * 2),
            ("period_walks", moments[:, None, None] * 2),
        ):
            walks = arrays[name]
            times = np.where(walks >= 0, events[np.maximum(walks, 0), 2], -1)
            if np.any(times[..., 0] > limit) or np.any(
                (np.diff(times, axis=-1) >= 0) & (times[..., 1:] >= 0)
            ):
                raise ValueError("%s of graph %s are not strictly causal" % (name, graph.name))
