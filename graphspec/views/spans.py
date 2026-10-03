import logging
import time

import numpy as np
from numba import njit

from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)


@timed("spans.counted")
def counted(counts):
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


@timed("spans.flat")
def flat(column):
    column = np.ascontiguousarray(column)
    return column.reshape(column.shape[0], int(np.prod(column.shape[1:], dtype=np.int64))).view(np.uint8)


@timed("spans.words")
def words(*columns):
    parts = [flat(column) for column in columns]
    raw = np.concatenate(parts, axis=1) if parts else np.zeros((0, 0), np.uint8)
    padding = -raw.shape[1] % 8
    if padding:
        raw = np.concatenate((raw, np.zeros((len(raw), padding), np.uint8)), axis=1)
    return np.ascontiguousarray(raw).view(np.int64)


@njit(cache=True)
def mix(value):
    value ^= value >> np.uint64(33)
    value *= np.uint64(0xFF51AFD7ED558CCD)
    value ^= value >> np.uint64(33)
    value *= np.uint64(0xC4CEB9FE1A85EC53)
    value ^= value >> np.uint64(33)
    return value


@njit(cache=True)
def identity(rows):
    count, width = rows.shape
    capacity = 16
    while capacity < 2 * count:
        capacity *= 2
    mask = np.uint64(capacity - 1)
    slots = np.full(capacity, -1, np.int64)
    representative = np.empty(count, np.int64)
    ids = np.empty(count, np.int64)
    groups = 0
    for row in range(count):
        hashed = np.uint64(width)
        for column in range(width):
            hashed = mix(hashed ^ np.uint64(rows[row, column]))
        slot = np.int64(hashed & mask)
        while True:
            group = slots[slot]
            if group < 0:
                slots[slot] = groups
                representative[groups] = row
                ids[row] = groups
                groups += 1
                break
            other = representative[group]
            same = True
            for column in range(width):
                if rows[other, column] != rows[row, column]:
                    same = False
                    break
            if same:
                ids[row] = group
                break
            slot = (slot + 1) & (capacity - 1)
    return ids, groups


@njit(cache=True)
def topological(sources, targets, nodes, priority):
    degree = np.zeros(nodes, np.int64)
    for edge in range(len(sources)):
        degree[targets[edge]] += 1
    offsets = np.zeros(nodes + 1, np.int64)
    for edge in range(len(sources)):
        offsets[sources[edge] + 1] += 1
    for node in range(nodes):
        offsets[node + 1] += offsets[node]
    fill = offsets[:-1].copy()
    followers = np.empty(len(sources), np.int64)
    for edge in range(len(sources)):
        followers[fill[sources[edge]]] = targets[edge]
        fill[sources[edge]] += 1
    queue = np.empty(nodes, np.int64)
    head, tail = 0, 0
    for node in range(nodes):
        if degree[node] == 0:
            queue[tail] = node
            tail += 1
    rank = np.full(nodes, -1, np.int64)
    queued = degree == 0
    cursor = 0
    while head < nodes:
        if head == tail:
            while queued[priority[cursor]]:
                cursor += 1
            queue[tail] = priority[cursor]
            queued[priority[cursor]] = True
            tail += 1
        node = queue[head]
        rank[node] = head
        head += 1
        for position in range(offsets[node], offsets[node + 1]):
            follower = followers[position]
            degree[follower] -= 1
            if degree[follower] == 0 and not queued[follower]:
                queued[follower] = True
                queue[tail] = follower
                tail += 1
    return rank


@timed("spans.runs")
def runs(group, step):
    order = np.lexsort((step, group))
    group, step = group[order], step[order]
    starts = np.ones(len(order), bool)
    starts[1:] = (group[1:] != group[:-1]) | (step[1:] != step[:-1] + 1)
    run = np.cumsum(starts) - 1
    result = np.empty(len(order), np.int64)
    result[order] = run
    return result, order[starts], starts.sum()


def observation_spans(entity, step, x):
    started = time.perf_counter()
    order = np.lexsort((step, entity))
    entity, step, x = entity[order], step[order], x[order]
    payload = flat(x)
    starts = np.r_[
        True,
        (entity[1:] != entity[:-1])
        | (step[1:] != step[:-1] + 1)
        | np.any(payload[1:] != payload[:-1], axis=1),
    ][: len(entity)]
    first = np.flatnonzero(starts)
    stop = step[np.r_[first[1:], len(step)][: len(first)] - 1].astype(np.int64) + 1
    arrangement = np.lexsort((entity[first], step[first]))
    first, stop = first[arrangement], stop[arrangement]
    result = dict(
        obs_entity=entity[first].astype(np.int32),
        obs_start=step[first].astype(np.int32),
        obs_stop=stop.astype(np.int32),
        obs_x=x[first],
    )
    LOG.debug(
        "observation spans exit rows=%d spans=%d seconds=%.4f",
        len(order),
        len(first),
        time.perf_counter() - started,
    )
    return result


@timed("spans.observation_rows")
def observation_rows(entity, start, stop):
    lengths = (stop - start).astype(np.int64)
    span = np.repeat(np.arange(len(start)), lengths)
    step = np.repeat(start.astype(np.int64), lengths) + counted(lengths)
    order = np.lexsort((entity[span], step))
    span, step = span[order], step[order]
    return entity[span], step.astype(np.int32), span.astype(np.int32)


@njit(cache=True)
def displaced(ranks):
    count = len(ranks)
    tails = np.empty(count, np.int64)
    previous = np.full(count, -1, np.int64)
    length = 0
    for index in range(count):
        lo, hi = 0, length
        while lo < hi:
            middle = (lo + hi) // 2
            if ranks[tails[middle]] < ranks[index]:
                lo = middle + 1
            else:
                hi = middle
        if lo > 0:
            previous[index] = tails[lo - 1]
        tails[lo] = index
        if lo == length:
            length += 1
    moved = np.ones(count, np.bool_)
    index = tails[length - 1] if length else -1
    while index >= 0:
        moved[index] = False
        index = previous[index]
    return moved


@timed("spans.patches")
def patches(at, span, rank):
    same = np.flatnonzero(at[1:] == at[:-1])
    broken = np.unique(at[same[rank[same] > rank[same + 1]]])
    bounds = np.searchsorted(at, np.r_[broken, broken + 1])
    steps, positions, spans = [], [], []
    for lo, hi, moment in zip(bounds[: len(broken)], bounds[len(broken) :], broken.tolist()):
        moved = np.flatnonzero(displaced(rank[lo:hi]))
        steps.append(np.full(len(moved), moment, np.int32))
        positions.append(moved.astype(np.int32))
        spans.append(span[lo:hi][moved].astype(np.int32))
    empty = [np.zeros(0, np.int32)]
    return np.concatenate(steps + empty), np.concatenate(positions + empty), np.concatenate(spans + empty)


def fact_spans(step, source, target, code, present, values):
    started = time.perf_counter()
    if len(code) and (code.min() < 0 or code.max() > np.iinfo(np.int16).max):
        raise ValueError("fact codes must fit 0..%d" % np.iinfo(np.int16).max)
    if len(step) and step.max() >= np.iinfo(np.int32).max:
        raise ValueError("fact steps must stay below %d" % np.iinfo(np.int32).max)
    kept = np.flatnonzero(present)
    ident, _ = identity(words(code[kept], source[kept], target[kept], values[kept]))
    at = step[kept].astype(np.int64)
    order = np.lexsort((kept, at, ident))
    grouped = np.ones(len(order), bool)
    grouped[1:] = (ident[order][1:] != ident[order][:-1]) | (at[order][1:] != at[order][:-1])
    first = np.maximum.accumulate(np.where(grouped, np.arange(len(order)), 0))
    occurrence = np.empty(len(order), np.int64)
    occurrence[order] = np.arange(len(order)) - first
    member, _ = identity(np.stack((ident, occurrence), 1))
    span, heads, count = runs(member, at)
    lengths = np.bincount(span, minlength=count)
    tail = np.zeros(count, np.int64)
    np.maximum.at(tail, span, at)
    same_step = np.flatnonzero(at[1:] == at[:-1])
    rank = topological(span[same_step], span[same_step + 1], count, np.argsort(heads, kind="stable"))
    arrangement = np.lexsort((rank, at[heads]))
    renumber = np.empty(count, np.int64)
    renumber[arrangement] = np.arange(count)
    heads, tail, rank = heads[arrangement], tail[arrangement], rank[arrangement]
    order_rank = np.empty(count, np.int64)
    order_rank[np.argsort(rank, kind="stable")] = np.arange(count)
    row_span = renumber[span]
    moved_step, moved_position, moved_span = patches(at, row_span, order_rank[row_span])
    rows = kept[heads]
    absent = np.flatnonzero(~present)
    result = dict(
        fact_source=source[rows].astype(np.int32),
        fact_target=target[rows].astype(np.int32),
        fact_code=code[rows].astype(np.int16),
        fact_start=step[rows].astype(np.int32),
        fact_stop=(tail + 1).astype(np.int32),
        fact_rank=order_rank.astype(np.int32),
        fact_values=values[rows],
        order_step=moved_step,
        order_position=moved_position,
        order_span=moved_span,
        removal_source=source[absent].astype(np.int32),
        removal_target=target[absent].astype(np.int32),
        removal_code=code[absent].astype(np.int16),
        removal_step=step[absent].astype(np.int32),
        removal_values=values[absent],
    )
    if not np.array_equal(lengths[arrangement], result["fact_stop"] - result["fact_start"]):
        raise ValueError("fact spans are not contiguous")
    LOG.debug(
        "fact spans exit rows=%d spans=%d removals=%d moved=%d seconds=%.4f",
        len(step),
        count,
        len(absent),
        len(moved_span),
        time.perf_counter() - started,
    )
    return result


@timed("spans.ordered_facts")
def ordered_facts(spans, steps):
    start, stop, rank = spans["fact_start"], spans["fact_stop"], spans["fact_rank"]
    lengths = (stop - start).astype(np.int64)
    span = np.repeat(np.arange(len(start)), lengths)
    at = np.repeat(start.astype(np.int64), lengths) + counted(lengths)
    order = np.lexsort((rank[span], at))
    span = span[order]
    bounds = np.searchsorted(at[order], np.arange(steps + 1))
    moments, positions, moved = spans["order_step"], spans["order_position"], spans["order_span"]
    edges = np.flatnonzero(np.r_[True, moments[1:] != moments[:-1]][: len(moments)])
    for first, last in zip(edges.tolist(), np.r_[edges[1:], len(moments)].tolist()):
        moment = int(moments[first])
        lo, hi = bounds[moment], bounds[moment + 1]
        chosen, where = moved[first:last], positions[first:last]
        current = span[lo:hi]
        staying = current[~np.isin(current, chosen)]
        if len(staying) + len(chosen) != hi - lo or np.any(where >= hi - lo):
            raise ValueError("fact order patch at step %d does not match the active spans" % moment)
        placed = np.zeros(hi - lo, bool)
        placed[where] = True
        current[placed] = chosen
        current[~placed] = staying
    return bounds, span


@timed("spans.fact_rows")
def fact_rows(spans, steps):
    bounds, span = ordered_facts(spans, steps)
    at = np.repeat(np.arange(steps, dtype=np.int64), np.diff(bounds))
    removed = len(spans["removal_step"])
    step = np.r_[at, spans["removal_step"].astype(np.int64)]
    absent = np.r_[np.zeros(len(at), bool), np.ones(removed, bool)]
    sequence = np.r_[np.arange(len(at)), np.arange(removed)]
    order = np.lexsort((sequence, absent, step))
    rows = np.r_[span, -1 - np.arange(removed)][order]
    present = rows >= 0
    chosen, gone = rows[present], -1 - rows[~present]

    @timed("spans.fact_rows.pick")
    def pick(name, removal):
        values = spans[name]
        result = np.empty(len(rows), values.dtype)
        result[present] = values[chosen]
        result[~present] = spans[removal][gone]
        return result

    reference = np.empty(len(rows), np.int32)
    reference[present] = chosen
    reference[~present] = len(spans["fact_start"]) + gone
    return dict(
        fact_source=pick("fact_source", "removal_source"),
        fact_target=pick("fact_target", "removal_target"),
        fact_code=pick("fact_code", "removal_code"),
        fact_step=step[order].astype(np.int32),
        fact_present=present,
        fact_ref=reference,
    )


@njit(cache=True)
def coverage(entity, start, stop, entities):
    order = np.argsort(entity * np.int64(1 << 32) + start)
    offsets = np.zeros(entities + 1, np.int64)
    lows = np.empty(len(order), np.int64)
    highs = np.empty(len(order), np.int64)
    runs_count = 0
    previous = -1
    for position in range(len(order)):
        span = order[position]
        current = entity[span]
        if current == previous and start[span] <= highs[runs_count - 1]:
            if stop[span] > highs[runs_count - 1]:
                highs[runs_count - 1] = stop[span]
            continue
        lows[runs_count] = start[span]
        highs[runs_count] = stop[span]
        offsets[current + 1] += 1
        runs_count += 1
        previous = current
    for index in range(entities):
        offsets[index + 1] += offsets[index]
    return offsets, lows[:runs_count], highs[:runs_count]


@njit(cache=True)
def covered(offsets, lows, highs, entity, low, high):
    lo, hi = offsets[entity], offsets[entity + 1]
    while lo < hi:
        middle = (lo + hi) // 2
        if lows[middle] <= low:
            lo = middle + 1
        else:
            hi = middle
    return lo > offsets[entity] and highs[lo - 1] >= high


@njit(cache=True)
def all_covered(offsets, lows, highs, entity, low, high):
    for row in range(len(entity)):
        if entity[row] < 0:
            continue
        if not covered(offsets, lows, highs, entity[row], low[row], high[row]):
            return row
    return -1


@timed("spans.active")
def active(start, stop, steps, key=None):
    lengths = (stop - start).astype(np.int64)
    span = np.repeat(np.arange(len(start)), lengths)
    step = np.repeat(start.astype(np.int64), lengths) + counted(lengths)
    order = np.lexsort((span if key is None else key[span], step))
    return np.searchsorted(step[order], np.arange(steps + 1)), span[order]


def encode(code, source, target, entities):
    return (np.asarray(code, np.int64) * entities + np.asarray(source, np.int64)) * entities + np.asarray(
        target, np.int64
    )
