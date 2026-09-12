import numpy as np
from numba import njit, prange, types
from numba.typed import Dict

from graphspec.kernels import mix


@njit(cache=True)
def uniform(seed, stream, first, second, third):
    value = mix(np.uint64(seed ^ stream))
    value = mix(value ^ np.uint64(np.int64(first)))
    value = mix(value ^ np.uint64(np.int64(second)))
    value = mix(value ^ np.uint64(np.int64(third)))
    return np.float64(value >> np.uint64(11)) / np.float64(1 << 53)


@njit(cache=True)
def latest(observation_ids, observation_offsets, step, entities, cutoffs):
    result = np.full(len(entities), -1, np.int32)
    for row in range(len(entities)):
        lo, hi = observation_offsets[entities[row]], observation_offsets[entities[row] + 1]
        bound = cutoffs[row]
        while lo < hi:
            middle = (lo + hi) // 2
            if step[observation_ids[middle]] * 2 < bound:
                lo = middle + 1
            else:
                hi = middle
        if lo > observation_offsets[entities[row]]:
            result[row] = observation_ids[lo - 1]
    return result


@njit(cache=True)
def bound(neighbor_ids, ticks, lo, hi, value):
    while lo < hi:
        middle = (lo + hi) // 2
        if ticks[neighbor_ids[middle]] < value:
            lo = middle + 1
        else:
            hi = middle
    return lo


@njit(cache=True)
def neighbors(neighbor_ids, neighbor_offsets, ticks, entity, cutoff, floor, count, mode, decay, seed, stream, out):
    lo = bound(neighbor_ids, ticks, neighbor_offsets[entity], neighbor_offsets[entity + 1], floor)
    hi = bound(neighbor_ids, ticks, lo, neighbor_offsets[entity + 1], cutoff)
    if mode == 0 or hi - lo <= count:
        start = max(lo, hi - count)
        for slot in range(hi - start):
            out[slot] = neighbor_ids[start + slot]
        return hi - start
    keys = np.empty(hi - lo)
    for slot in range(hi - lo):
        candidate = neighbor_ids[lo + slot]
        draw = uniform(seed, stream, entity, cutoff, candidate)
        draw = min(draw + 2.0 ** -54, 1 - 2.0 ** -53)
        logit = 0.0 if mode == 1 else -decay * (cutoff - ticks[candidate]) / 2.0
        keys[slot] = logit - np.log(-np.log(draw))
    order = np.argsort(-keys)
    chosen = np.sort(order[:count])
    for slot in range(count):
        out[slot] = neighbor_ids[lo + chosen[slot]]
    return count


@njit(cache=True, parallel=True)
def neighbor_table(neighbor_ids, neighbor_offsets, ticks, entities, cutoffs, floor, count, mode, decay, seed, stream):
    ids = np.full((len(entities), count), -1, np.int32)
    sizes = np.zeros(len(entities), np.int64)
    for row in prange(len(entities)):
        sizes[row] = neighbors(neighbor_ids, neighbor_offsets, ticks, entities[row], cutoffs[row], floor, count, mode, decay, seed, stream, ids[row])
    return ids, sizes


@njit(cache=True)
def tree_level(events, ids, sizes, entities, cutoffs, count, non_relation):
    rows = len(entities)
    children = np.repeat(entities, count)
    times = np.repeat(cutoffs, count)
    mask = np.zeros((rows, count), np.bool_)
    gaps = np.zeros((rows, count), np.float32)
    relation = np.zeros((rows, count), np.int64)
    for row in range(rows):
        for slot in range(sizes[row]):
            event = ids[row, slot]
            children[row * count + slot] = events[event, 1] if events[event, 0] == entities[row] else events[event, 0]
            times[row * count + slot] = events[event, 2]
            mask[row, slot] = True
            gaps[row, slot] = cutoffs[row] - events[event, 2]
            relation[row, slot] = events[event, 6] + 1 if events[event, 3] == 1 else non_relation
    return children, times, mask, gaps, relation


@njit(cache=True)
def partners(memo, entity, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch):
    if entity in memo:
        return memo[entity]
    size = neighbors(neighbor_ids, neighbor_offsets, ticks, entity, cutoff, floor, count, mode, decay, seed, stream, scratch)
    result = np.empty(size, np.int64)
    for slot in range(size):
        event = scratch[slot]
        result[slot] = events[event, 1] if events[event, 0] == entity else events[event, 0]
    memo[entity] = result
    return result


@njit(cache=True)
def overlap(first, second):
    if len(first) == 0 or len(second) == 0:
        return 0.0
    hits = 0
    for x in first:
        for y in second:
            hits += x == y
    return hits / (len(first) * len(second))


@njit(cache=True)
def two_hop(memo, entity, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch):
    first = partners(memo, entity, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch)
    take = max(1, int(count ** 0.5))
    start = max(0, len(first) - take)
    total = 0
    for other in first[start:]:
        total += len(partners(memo, other, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch))
    collected = np.empty(total, np.int64)
    fill = 0
    for other in first[start:]:
        for value in partners(memo, other, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch):
            collected[fill] = value
            fill += 1
    return np.unique(collected)


@njit(cache=True)
def occurrences(values, target):
    hits = 0
    for value in values:
        hits += value == target
    return hits


@njit(cache=True)
def history(events, neighbor_ids, neighbor_offsets, ticks, actor, subject, query_step, start, count, mode, decay, seed, stream, structure, selected):
    rows = len(actor)
    observed = np.full((rows, 2, count), -1, np.int64)
    interaction = np.zeros((rows, 2, count), np.bool_)
    query_tick = np.zeros((rows, 2, count), np.int64)
    mask = np.zeros((rows, 2, count), np.bool_)
    gap = np.zeros((rows, 2, count), np.float32)
    ids = np.full((rows, 2, count), -1, np.int32)
    structural = np.zeros((rows, 2, count, 12), np.float32)
    common_ids = np.full((rows, count), -1, np.int32)
    common_frequency = np.zeros((rows, count), np.float32)
    common_stats = np.zeros((rows, 4), np.float32)
    pair_gaps = np.zeros((rows, count), np.float32)
    pair_mask = np.zeros((rows, count), np.bool_)
    scratch = np.empty(count, np.int32)
    floor = start * 2
    for i in range(rows):
        if not selected[i]:
            continue
        memo = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        roots = (np.int64(actor[i]), np.int64(subject[i]))
        cutoff = np.int64(query_step[i]) * 2 + 1
        first_events = np.empty(0, np.int32)
        for side in range(2):
            root = roots[side]
            size = neighbors(neighbor_ids, neighbor_offsets, ticks, root, cutoff, floor, count, mode, decay, seed, stream, scratch)
            chosen = scratch[:size].copy()
            if side == 0:
                first_events = chosen
            for slot in range(size):
                event = chosen[slot]
                forward = events[event, 0] == root
                ids[i, side, slot] = events[event, 1] if forward else events[event, 0]
                observed[i, side, slot] = events[event, 5] if forward else events[event, 4]
                interaction[i, side, slot] = events[event, 3] == 2
                query_tick[i, side, slot] = events[event, 2] // 2
                mask[i, side, slot] = True
                gap[i, side, slot] = cutoff - events[event, 2]
                if structure:
                    column = 0
                    for end in range(2):
                        endpoint = np.int64(events[event, end])
                        for which in range(2):
                            first = partners(memo, endpoint, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch)
                            second = partners(memo, roots[which], events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch)
                            structural[i, side, slot, column] = occurrences(second, endpoint) / max(len(second), 1)
                            structural[i, side, slot, column + 1] = overlap(first, second)
                            structural[i, side, slot, column + 2] = overlap(
                                two_hop(memo, endpoint, events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch),
                                two_hop(memo, roots[which], events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch))
                            column += 3
        first = partners(memo, roots[0], events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch)
        second = partners(memo, roots[1], events, neighbor_ids, neighbor_offsets, ticks, cutoff, floor, count, mode, decay, seed, stream, scratch)
        shared = 0
        for value in np.unique(first):
            if occurrences(second, value) > 0:
                common_ids[i, shared] = value
                common_frequency[i, shared] = occurrences(first, value) * occurrences(second, value)
                shared += 1
        common_stats[i, 0] = shared
        common_stats[i, 1] = occurrences(first, roots[1])
        common_stats[i, 2] = occurrences(second, roots[0])
        common_stats[i, 3] = len(np.unique(np.concatenate((first, second))))
        filled = 0
        previous = np.int64(0)
        for slot in range(len(first_events)):
            event = first_events[slot]
            other = events[event, 1] if events[event, 0] == roots[0] else events[event, 0]
            if other == roots[1]:
                if filled > 0:
                    pair_gaps[i, filled - 1] = events[event, 2] - previous
                previous = np.int64(events[event, 2])
                filled += 1
        if filled > 0:
            pair_gaps[i, filled - 1] = cutoff - previous
            for slot in range(filled):
                pair_mask[i, slot] = True
    return observed, interaction, query_tick, mask, gap, ids, structural, common_ids, common_frequency, common_stats, pair_gaps, pair_mask


@njit(cache=True)
def reachable(owners, positions, times, roots, cutoffs):
    span = np.int64(len(positions))
    order = np.argsort(owners.astype(np.int64) * span + positions)
    sorted_owners, sorted_positions = owners[order], positions[order]
    counts = np.zeros(len(roots), np.int64)
    found = []
    for row in range(len(roots)):
        pieces = np.empty(0, np.int64)
        for root in np.unique(roots[row]):
            lo = np.searchsorted(sorted_owners, root, side='left')
            hi = np.searchsorted(sorted_owners, root, side='right')
            members = sorted_positions[lo:hi]
            stop = np.searchsorted(times[members], cutoffs[row], side='right')
            pieces = np.concatenate((pieces, members[:stop]))
        unique = np.unique(pieces)
        found.append(unique)
        counts[row] = len(unique)
    offsets = np.zeros(len(roots) + 1, np.int64)
    for row in range(len(roots)):
        offsets[row + 1] = offsets[row] + counts[row]
    flat = np.empty(offsets[-1], np.int64)
    for row in range(len(roots)):
        flat[offsets[row]:offsets[row + 1]] = found[row]
    return flat, offsets


@njit(cache=True, parallel=True)
def walk_identities(nodes, known):
    rows, sides, walks, hops = nodes.shape
    out = np.zeros((rows, sides, walks, hops, 2 * hops), np.float32)
    for q in prange(rows):
        for s in range(sides):
            for w in range(walks):
                for h in range(hops):
                    node = nodes[q, s, w, h]
                    for side in range(sides):
                        for hop in range(hops):
                            hits = 0
                            for other in range(walks):
                                hits += (nodes[q, side, other, hop] == node) & known[q, side, other, hop]
                            out[q, s, w, h, side * hops + hop] = hits
    return out
