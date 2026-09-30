import numpy as np
from numba import njit, prange, types
from numba.typed import Dict

from graphspec.infra.hashing import mixed, uniform


@njit(cache=True)
def rooted(drawn, keyed, entity, cutoff):
    tokens, step_token, step_time = keyed[0], keyed[1], keyed[2]
    moment = cutoff // 2
    value = mixed(mixed(drawn) ^ np.uint64(tokens[entity]))
    value = mixed(value ^ step_token[moment])
    value = mixed(value ^ np.uint64(step_time[moment]))
    return mixed(value ^ np.uint64(cutoff % 2))


@njit(cache=True, parallel=True)
def gaussian(drawn, tokens, homes, width):
    result = np.empty((len(tokens), width), np.float32)
    for row in prange(len(tokens)):
        value = mixed(mixed(mixed(drawn) ^ np.uint64(tokens[row])) ^ homes[row])
        for column in range(width):
            first = max(uniform(mixed(value ^ np.uint64(2 * column))), 2.0**-53)
            second = uniform(mixed(value ^ np.uint64(2 * column + 1)))
            result[row, column] = np.sqrt(-2.0 * np.log(first)) * np.cos(2.0 * np.pi * second)
    return result


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
def neighbors(
    neighbor_ids, neighbor_offsets, ticks, entity, cutoff, floor, count, mode, decay, drawn, keyed, out
):
    lo = bound(neighbor_ids, ticks, neighbor_offsets[entity], neighbor_offsets[entity + 1], floor)
    hi = bound(neighbor_ids, ticks, lo, neighbor_offsets[entity + 1], cutoff)
    if mode == 0 or hi - lo <= count:
        first = max(lo, hi - count)
        for slot in range(hi - first):
            out[slot] = neighbor_ids[first + slot]
        return hi - first
    identity, root = keyed[3], rooted(drawn, keyed, entity, cutoff)
    keys = np.empty(hi - lo)
    for slot in range(hi - lo):
        candidate = neighbor_ids[lo + slot]
        draw = min(uniform(mixed(root ^ identity[candidate])) + 2.0**-54, 1 - 2.0**-53)
        logit = 0.0 if mode == 1 else -decay * (cutoff - ticks[candidate]) / 2.0
        keys[slot] = logit - np.log(-np.log(draw))
    order = np.argsort(-keys, kind="mergesort")
    chosen = np.sort(order[:count])
    for slot in range(count):
        out[slot] = neighbor_ids[lo + chosen[slot]]
    return count


@njit(cache=True, parallel=True)
def neighbor_table(
    neighbor_ids, neighbor_offsets, ticks, entities, cutoffs, floor, count, mode, decay, drawn, keyed
):
    ids = np.full((len(entities), count), -1, np.int32)
    sizes = np.zeros(len(entities), np.int64)
    for row in prange(len(entities)):
        sizes[row] = neighbors(
            neighbor_ids,
            neighbor_offsets,
            ticks,
            entities[row],
            cutoffs[row],
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            ids[row],
        )
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
            children[row * count + slot] = (
                events[event, 1] if events[event, 0] == entities[row] else events[event, 0]
            )
            times[row * count + slot] = events[event, 2]
            mask[row, slot] = True
            gaps[row, slot] = cutoffs[row] - events[event, 2]
            relation[row, slot] = events[event, 6] + 1 if events[event, 3] == 1 else non_relation
    return children, times, mask, gaps, relation


@njit(cache=True)
def partners(
    memo,
    entity,
    events,
    neighbor_ids,
    neighbor_offsets,
    ticks,
    cutoff,
    floor,
    count,
    mode,
    decay,
    drawn,
    keyed,
    scratch,
):
    if entity in memo:
        return memo[entity]
    size = neighbors(
        neighbor_ids,
        neighbor_offsets,
        ticks,
        entity,
        cutoff,
        floor,
        count,
        mode,
        decay,
        drawn,
        keyed,
        scratch,
    )
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
def two_hop(
    memo,
    entity,
    events,
    neighbor_ids,
    neighbor_offsets,
    ticks,
    cutoff,
    floor,
    count,
    mode,
    decay,
    drawn,
    keyed,
    scratch,
):
    first = partners(
        memo,
        entity,
        events,
        neighbor_ids,
        neighbor_offsets,
        ticks,
        cutoff,
        floor,
        count,
        mode,
        decay,
        drawn,
        keyed,
        scratch,
    )
    take = max(1, int(count**0.5))
    start = max(0, len(first) - take)
    total = 0
    for other in first[start:]:
        total += len(
            partners(
                memo,
                other,
                events,
                neighbor_ids,
                neighbor_offsets,
                ticks,
                cutoff,
                floor,
                count,
                mode,
                decay,
                drawn,
                keyed,
                scratch,
            )
        )
    collected = np.empty(total, np.int64)
    fill = 0
    for other in first[start:]:
        for value in partners(
            memo,
            other,
            events,
            neighbor_ids,
            neighbor_offsets,
            ticks,
            cutoff,
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            scratch,
        ):
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
def history(
    events,
    neighbor_ids,
    neighbor_offsets,
    ticks,
    roots,
    others,
    moments,
    start,
    count,
    mode,
    decay,
    drawn,
    keyed,
    structure,
    symmetric,
    selected,
):
    tokens = keyed[0]
    members = len(roots)
    observed = np.full((members, count), -1, np.int64)
    interaction = np.zeros((members, count), np.bool_)
    task_row = np.zeros((members, count), np.int64)
    mask = np.zeros((members, count), np.bool_)
    gap = np.zeros((members, count), np.float32)
    ids = np.full((members, count), -1, np.int32)
    relation = np.full((members, count), -1, np.int64)
    structural = np.zeros((members, count, 12), np.float32)
    scratch = np.empty(count, np.int32)
    floor = start * 2
    for i in range(members):
        if not selected[i]:
            continue
        memo = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        root = np.int64(roots[i])
        anchors = (root, np.int64(others[i]))
        cutoff = np.int64(moments[i]) * 2 + 1
        size = neighbors(
            neighbor_ids,
            neighbor_offsets,
            ticks,
            root,
            cutoff,
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            scratch,
        )
        chosen = scratch[:size].copy()
        for slot in range(size):
            event = chosen[slot]
            forward = events[event, 0] == root
            ids[i, slot] = events[event, 1] if forward else events[event, 0]
            observed[i, slot] = events[event, 5] if forward else events[event, 4]
            interaction[i, slot] = events[event, 3] == 2
            if events[event, 3] == 1:
                relation[i, slot] = events[event, 6]
            task_row[i, slot] = events[event, 7]
            mask[i, slot] = True
            gap[i, slot] = cutoff - events[event, 2]
            if structure:
                column = 0
                left, right = np.int64(events[event, 0]), np.int64(events[event, 1])
                if events[event, 3] == 1 and symmetric[events[event, 6]] and tokens[right] < tokens[left]:
                    left, right = right, left
                for endpoint in (left, right):
                    for which in range(2):
                        anchor = anchors[which]
                        if anchor >= 0:
                            first = partners(
                                memo,
                                endpoint,
                                events,
                                neighbor_ids,
                                neighbor_offsets,
                                ticks,
                                cutoff,
                                floor,
                                count,
                                mode,
                                decay,
                                drawn,
                                keyed,
                                scratch,
                            )
                            second = partners(
                                memo,
                                anchor,
                                events,
                                neighbor_ids,
                                neighbor_offsets,
                                ticks,
                                cutoff,
                                floor,
                                count,
                                mode,
                                decay,
                                drawn,
                                keyed,
                                scratch,
                            )
                            structural[i, slot, column] = occurrences(second, endpoint) / max(len(second), 1)
                            structural[i, slot, column + 1] = overlap(first, second)
                            structural[i, slot, column + 2] = overlap(
                                two_hop(
                                    memo,
                                    endpoint,
                                    events,
                                    neighbor_ids,
                                    neighbor_offsets,
                                    ticks,
                                    cutoff,
                                    floor,
                                    count,
                                    mode,
                                    decay,
                                    drawn,
                                    keyed,
                                    scratch,
                                ),
                                two_hop(
                                    memo,
                                    anchor,
                                    events,
                                    neighbor_ids,
                                    neighbor_offsets,
                                    ticks,
                                    cutoff,
                                    floor,
                                    count,
                                    mode,
                                    decay,
                                    drawn,
                                    keyed,
                                    scratch,
                                ),
                            )
                        column += 3
    return observed, interaction, task_row, mask, gap, ids, relation, structural


@njit(cache=True)
def paired(
    events,
    neighbor_ids,
    neighbor_offsets,
    ticks,
    lefts,
    rights,
    moments,
    start,
    count,
    mode,
    decay,
    drawn,
    keyed,
    selected,
):
    rows = len(lefts)
    common_ids = np.full((rows, count), -1, np.int32)
    common_frequency = np.zeros((rows, count), np.float32)
    common_stats = np.zeros((rows, 4), np.float32)
    pair_gaps = np.zeros((rows, count), np.float32)
    pair_mask = np.zeros((rows, count), np.bool_)
    scratch = np.empty(count, np.int32)
    floor = start * 2
    for i in range(rows):
        if not selected[i] or lefts[i] < 0:
            continue
        memo = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        left, right = np.int64(lefts[i]), np.int64(rights[i])
        cutoff = np.int64(moments[i]) * 2 + 1
        size = neighbors(
            neighbor_ids,
            neighbor_offsets,
            ticks,
            left,
            cutoff,
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            scratch,
        )
        left_events = scratch[:size].copy()
        first = partners(
            memo,
            left,
            events,
            neighbor_ids,
            neighbor_offsets,
            ticks,
            cutoff,
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            scratch,
        )
        second = partners(
            memo,
            right,
            events,
            neighbor_ids,
            neighbor_offsets,
            ticks,
            cutoff,
            floor,
            count,
            mode,
            decay,
            drawn,
            keyed,
            scratch,
        )
        shared = 0
        distinct = np.unique(first)
        for value in distinct[np.argsort(keyed[0][distinct], kind="mergesort")]:
            if occurrences(second, value) > 0:
                common_ids[i, shared] = value
                common_frequency[i, shared] = occurrences(first, value) * occurrences(second, value)
                shared += 1
        common_stats[i, 0] = shared
        common_stats[i, 1] = occurrences(first, right)
        common_stats[i, 2] = occurrences(second, left)
        common_stats[i, 3] = len(np.unique(np.concatenate((first, second))))
        filled = 0
        previous = np.int64(0)
        for slot in range(len(left_events)):
            event = left_events[slot]
            other = events[event, 1] if events[event, 0] == left else events[event, 0]
            if other == right:
                if filled > 0:
                    pair_gaps[i, filled - 1] = events[event, 2] - previous
                previous = np.int64(events[event, 2])
                filled += 1
        if filled > 0:
            pair_gaps[i, filled - 1] = cutoff - previous
            for slot in range(filled):
                pair_mask[i, slot] = True
    return common_ids, common_frequency, common_stats, pair_gaps, pair_mask


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
            lo = np.searchsorted(sorted_owners, root, side="left")
            hi = np.searchsorted(sorted_owners, root, side="right")
            members = sorted_positions[lo:hi]
            stop = np.searchsorted(times[members], cutoffs[row], side="right")
            pieces = np.concatenate((pieces, members[:stop]))
        unique = np.unique(pieces)
        found.append(unique)
        counts[row] = len(unique)
    offsets = np.zeros(len(roots) + 1, np.int64)
    for row in range(len(roots)):
        offsets[row + 1] = offsets[row] + counts[row]
    flat = np.empty(offsets[-1], np.int64)
    for row in range(len(roots)):
        flat[offsets[row] : offsets[row + 1]] = found[row]
    return flat, offsets


@njit(cache=True, parallel=True)
def walk_identities(nodes, known, others):
    members, walks, hops = nodes.shape
    out = np.zeros((members, walks, hops, 2 * hops), np.float32)
    for q in prange(members):
        for w in range(walks):
            for h in range(hops):
                node = nodes[q, w, h]
                for side in range(2):
                    source = np.int64(q) if side == 0 else np.int64(others[q])
                    if source < 0:
                        continue
                    for hop in range(hops):
                        hits = 0
                        for other in range(walks):
                            hits += (nodes[source, other, hop] == node) & known[source, other, hop]
                        out[q, w, h, side * hops + hop] = hits
    return out
