import logging
import math
import time

import numpy as np

from graphspec.infra.hashing import hashed, salted
from graphspec.infra.timing import timed
from graphspec.infra.workspace import allocate, block_rows, changed, gathered
from graphspec.operations.kernels import first_rows
from graphspec.operations.parameters import enabled
from graphspec.operations.state import Table
from graphspec.schema.declaration import BOUNDARIES
from graphspec.schema.references import SORTS, ends

from . import kernels

LOG = logging.getLogger(__name__)
VERSION = 13
EVENT_KINDS = ("observation", "relation", "task")
WALK_SAMPLES, WALK_HOPS, PERIOD_LOOKBACK = 32, 4, 3
WALKS, PERIOD_WALKS = "graphspec.walks/walks", "graphspec.walks/period_walks"
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
def base(graph, registry, spec, link):
    steps, kinds = graph.steps, graph.kinds
    step, entity, reference = graph.obs_step, graph.obs_entity, graph.obs_ref
    lookup = step.astype(np.int64) * len(kinds) + entity
    held = Table(lookup, row=np.arange(len(lookup), dtype=np.int64))

    def found(owners, moments):
        probe = np.asarray(moments, np.int64) * len(kinds) + owners
        return np.where(np.asarray(owners) >= 0, held.lookup("row", probe, -1), -1)

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
    doubled, reverse = symmetric, codes
    if registry.groups:
        direction = spec["relations/direction"]
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
        attribute_row[use] = np.arange(int(use.sum())) + link.attributes[group]
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
    heads = order[np.r_[True, entity[order][1:] != entity[order][:-1]]] if len(order) else order
    prior_rows, prior_x = carried_rows(link, entity[heads], graph.x_table)
    carried = prior_rows >= 0
    altered[heads[carried]] = np.any(
        np.asarray(graph.x_table)[reference[heads[carried]]] != prior_x[carried], axis=1
    )
    last = carried_rows(link, np.arange(len(kinds)), graph.x_table)[0]
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
        link.obs0,
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
    counts -= gathered(counts, head - 1, 0)
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


def event_edges(events, identity, member, spec, entities, budget):
    chosen, windowed, window, cap = event_setting(spec)
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    return kernels.event_dag(
        source,
        target,
        ticks,
        np.ascontiguousarray(identity, np.uint64),
        member,
        chosen,
        windowed,
        window,
        cap,
        entities,
        budget.pair_limit,
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


def neighbor_lists(events, use, entities, identity):
    ids = np.flatnonzero(use)
    source, target = events[ids, 0], events[ids, 1]
    distinct = target != source
    owner, event = np.r_[source, target[distinct]], np.r_[ids, ids[distinct]]
    order = np.lexsort((event, identity[event], events[event, 2], owner))
    return (
        event[order].astype(np.int32),
        np.r_[0, np.cumsum(np.bincount(owner, minlength=entities))].astype(np.int64),
    )


@timed("neighbors")
def neighbors(arrays, spec):
    events = arrays["events"]
    ids, offsets = neighbor_lists(
        events, neighbor_members(events, spec), len(arrays["entity_kind"]), np.asarray(arrays["event_identity"])
    )
    return dict(neighbor_ids=ids, neighbor_offsets=offsets)


def walk_arrays(ids, offsets, events, identity, roots, keys, moments, prior, first, last, spec, seed, budget):
    bias, window, fanout = spec["walks/bias"], spec["walks/window"], spec["walks/fanout"]
    count = len(moments)
    budget.workspace(
        count * WALK_SAMPLES * (1 + PERIOD_LOOKBACK) * (64 + WALK_HOPS * 4), "sampled walk representations"
    )
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    identity = np.ascontiguousarray(identity, np.uint64)
    member, draw = (
        axis.reshape(-1) for axis in np.meshgrid(np.arange(count), np.arange(WALK_SAMPLES), indexing="ij")
    )
    result = kernels.walked(
        ids,
        offsets,
        source,
        target,
        ticks,
        identity,
        roots[member],
        np.maximum(0, (moments[member] - window + 1) * 2),
        moments[member] * 2 + 1,
        np.ones(len(member), np.bool_),
        hashed(keys[member], draw),
        fanout,
        bias,
        salted(seed, WALKS),
        WALK_HOPS,
    )
    member, back, draw = (
        axis.reshape(-1)
        for axis in np.meshgrid(
            np.arange(count), np.arange(PERIOD_LOOKBACK), np.arange(WALK_SAMPLES), indexing="ij"
        )
    )
    period_walks = kernels.walked(
        ids,
        offsets,
        source,
        target,
        ticks,
        identity,
        roots[member],
        2 * first[member, back],
        2 * last[member, back] + 2,
        np.ascontiguousarray(prior[member, back] >= 0),
        hashed(keys[member], back, draw),
        fanout,
        bias,
        salted(seed, PERIOD_WALKS),
        WALK_HOPS,
    )
    return dict(
        walks=result.reshape((count, WALK_SAMPLES, WALK_HOPS)),
        period_walks=period_walks.reshape((count, PERIOD_LOOKBACK, WALK_SAMPLES, WALK_HOPS)),
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


