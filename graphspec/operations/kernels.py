import numpy as np
from numba import njit, prange

from graphspec.infra.hashing import mixed


@njit(cache=True)
def keyed(first, second, third):
    rows = len(first)
    capacity = 16
    while capacity < 2 * rows:
        capacity *= 2
    mask = np.uint64(capacity - 1)
    slots = np.full(capacity, -1, np.int64)
    representative = np.empty(rows, np.int64)
    ids = np.empty(rows, np.int64)
    count = 0
    for row in range(rows):
        a, b, c = np.int64(first[row]), np.int64(second[row]), np.int64(third[row])
        slot = np.int64(mixed(mixed(mixed(np.uint64(a)) ^ np.uint64(b)) ^ np.uint64(c)) & mask)
        while True:
            group = slots[slot]
            if group < 0:
                slots[slot] = count
                representative[count] = row
                ids[row] = count
                count += 1
                break
            other = representative[group]
            if np.int64(first[other]) == a and np.int64(second[other]) == b and np.int64(third[other]) == c:
                ids[row] = group
                break
            slot = (slot + 1) & (capacity - 1)
    return ids, count


@njit(cache=True)
def rehash(slots, keys_low, keys_high, count):
    capacity = len(slots)
    mask = np.uint64(capacity - 1)
    for index in range(count):
        slot = np.int64(mixed(mixed(np.uint64(keys_low[index])) ^ np.uint64(keys_high[index])) & mask)
        while slots[slot] != 0:
            slot = (slot + 1) & (capacity - 1)
        slots[slot] = index + 1


@njit(cache=True)
def novel(slots, keys_low, keys_high, count, low, high):
    capacity = len(slots)
    mask = np.uint64(capacity - 1)
    fresh = np.zeros(len(low), np.bool_)
    for row in range(len(low)):
        a, b = np.int64(low[row]), np.int64(high[row])
        slot = np.int64(mixed(mixed(np.uint64(a)) ^ np.uint64(b)) & mask)
        while True:
            stored = slots[slot]
            if stored == 0:
                slots[slot] = count + 1
                keys_low[count] = a
                keys_high[count] = b
                count += 1
                fresh[row] = True
                break
            if keys_low[stored - 1] == a and keys_high[stored - 1] == b:
                break
            slot = (slot + 1) & (capacity - 1)
    return count, fresh


@njit(cache=True)
def runs(ids, step, last, current, begun):
    position = np.empty(len(ids), np.int64)
    started = np.empty(len(ids), np.int64)
    for row in range(len(ids)):
        group, moment = ids[row], np.int64(step[row])
        if moment > last[group] + 1:
            current[group] = 0
            begun[group] = moment
        elif moment == last[group] + 1:
            current[group] += 1
        last[group] = moment
        position[row] = current[group]
        started[row] = begun[group]
    return position, started


@njit(cache=True)
def orphans(ids, present, kept, groups):
    stated = np.zeros(groups, np.int64)
    surviving = np.zeros(groups, np.int64)
    result = kept.copy()
    for row in range(len(ids)):
        group = ids[row]
        if present[row]:
            stated[group] += 1
            if kept[row]:
                surviving[group] += 1
        elif stated[group] > 0 and surviving[group] == 0:
            result[row] = False
    return result


@njit(cache=True)
def carried(ids, step, present, groups, steps):
    removal = np.full(groups, steps, np.int64)
    latest = np.full(groups, steps, np.int64)
    after = np.full(groups, steps, np.int64)
    limit = np.full(len(ids), steps, np.int64)
    for row in range(len(ids) - 1, -1, -1):
        group, moment = ids[row], np.int64(step[row])
        if not present[row]:
            removal[group] = moment
            continue
        if latest[group] != moment:
            after[group] = latest[group]
            latest[group] = moment
        limit[row] = min(removal[group], after[group])
    return limit


@njit(cache=True)
def first_observed(entity, step, include, entities, steps):
    first = np.full(entities, steps, np.int64)
    stamp = np.full(entities, -1, np.int64)
    duplicate = False
    for row in range(len(entity)):
        who, moment = entity[row], np.int64(step[row])
        if stamp[who] == moment:
            duplicate = True
        stamp[who] = moment
        if include[row] and first[who] == steps:
            first[who] = moment
    return first, duplicate


@njit(cache=True)
def observed_at(obs_entity, obs_step, include, entity, step, entities):
    stamp = np.full(entities, -1, np.int64)
    found = np.zeros(len(entity), np.bool_)
    cursor = 0
    for row in range(len(entity)):
        moment = np.int64(step[row])
        while cursor < len(obs_step) and obs_step[cursor] <= moment:
            if include[cursor]:
                stamp[obs_entity[cursor]] = obs_step[cursor]
            cursor += 1
        found[row] = stamp[entity[row]] == moment
    return found


@njit(cache=True)
def degrees(obs_entity, obs_step, fact_source, fact_target, fact_step, use, entities, cumulative, count):
    stamp = np.full(entities, -1, np.int64)
    result = np.zeros(len(obs_entity), np.int64)
    cursor = 0
    for row in range(len(obs_entity)):
        moment = np.int64(obs_step[row])
        while cursor < len(fact_step) and fact_step[cursor] <= moment:
            if use[cursor]:
                at = np.int64(fact_step[cursor])
                source, target = fact_source[cursor], fact_target[cursor]
                if not cumulative and stamp[source] != at:
                    stamp[source] = at
                    count[source] = 0
                count[source] += 1
                if target != source:
                    if not cumulative and stamp[target] != at:
                        stamp[target] = at
                        count[target] = 0
                    count[target] += 1
            cursor += 1
        who = obs_entity[row]
        if cumulative or stamp[who] == moment:
            result[row] = count[who]
    return result


@njit(cache=True)
def activity(obs_entity, count):
    result = np.empty(len(obs_entity), np.int64)
    for row in range(len(obs_entity)):
        count[obs_entity[row]] += 1
        result[row] = count[obs_entity[row]]
    return result


@njit(cache=True)
def peel(obs_entity, obs_step, alive, minimum, fact_source, fact_target, fact_step, use, entities):
    alive = alive.copy()
    where = np.full(entities, -1, np.int64)
    stamp = np.full(entities, -1, np.int64)
    degree = np.zeros(len(obs_entity), np.int64)
    observations, facts = len(obs_entity), len(fact_step)
    low, cursor = 0, 0
    while low < observations:
        moment = np.int64(obs_step[low])
        high = low
        while high < observations and obs_step[high] == moment:
            where[obs_entity[high]] = high
            stamp[obs_entity[high]] = moment
            high += 1
        while cursor < facts and fact_step[cursor] < moment:
            cursor += 1
        end = cursor
        while end < facts and fact_step[end] == moment:
            end += 1
        width, span = high - low, end - cursor
        offsets = np.zeros(width + 1, np.int64)
        left = np.full(span, -1, np.int64)
        right = np.full(span, -1, np.int64)
        for local in range(span):
            row = cursor + local
            source, target = fact_source[row], fact_target[row]
            if not use[row] or stamp[source] != moment or stamp[target] != moment:
                continue
            a, b = where[source], where[target]
            if not (alive[a] and alive[b]):
                continue
            left[local], right[local] = a, b
            degree[a] += 1
            offsets[a - low + 1] += 1
            if b != a:
                degree[b] += 1
                offsets[b - low + 1] += 1
        for index in range(width):
            offsets[index + 1] += offsets[index]
        incident = np.empty(offsets[width], np.int64)
        fill = offsets[:width].copy()
        for local in range(span):
            a, b = left[local], right[local]
            if a < 0:
                continue
            incident[fill[a - low]] = local
            fill[a - low] += 1
            if b != a:
                incident[fill[b - low]] = local
                fill[b - low] += 1
        queue = np.empty(width, np.int64)
        queued = np.zeros(width, np.bool_)
        head, tail = 0, 0
        for index in range(low, high):
            if alive[index] and minimum[index] > 0 and degree[index] < minimum[index]:
                queue[tail] = index
                tail += 1
                queued[index - low] = True
        while head < tail:
            index = queue[head]
            head += 1
            alive[index] = False
            for slot in range(offsets[index - low], offsets[index - low + 1]):
                local = incident[slot]
                if left[local] < 0:
                    continue
                other = right[local] if left[local] == index else left[local]
                left[local] = -1
                if other != index:
                    degree[other] -= 1
                    if (
                        alive[other]
                        and minimum[other] > 0
                        and degree[other] < minimum[other]
                        and not queued[other - low]
                    ):
                        queue[tail] = other
                        tail += 1
                        queued[other - low] = True
        low = high
    return alive


@njit(cache=True)
def untyped(typed, pairs, present, base, typed_groups, pair_groups):
    state = np.zeros(typed_groups, np.bool_)
    live = np.zeros(pair_groups, np.int64)
    kept = np.ones(len(typed), np.bool_)
    for row in range(len(typed)):
        if not base[row]:
            continue
        group, pair = typed[row], pairs[row]
        if present[row]:
            if not state[group]:
                state[group] = True
                live[pair] += 1
        else:
            if state[group]:
                state[group] = False
                live[pair] -= 1
            kept[row] = live[pair] == 0
    return kept


@njit(cache=True)
def continuing(ids, step, groups):
    following = np.full(groups, -2, np.int64)
    result = np.zeros(len(ids), np.bool_)
    for row in range(len(ids) - 1, -1, -1):
        result[row] = following[ids[row]] == step[row] + 1
        following[ids[row]] = step[row]
    return result


@njit(cache=True)
def last_rows(ids, groups):
    seen = np.zeros(groups, np.bool_)
    kept = np.zeros(len(ids), np.bool_)
    for row in range(len(ids) - 1, -1, -1):
        if not seen[ids[row]]:
            seen[ids[row]] = True
            kept[row] = True
    return kept


@njit(cache=True)
def first_rows(ids, groups):
    seen = np.zeros(groups, np.bool_)
    kept = np.zeros(len(ids), np.bool_)
    for row in range(len(ids)):
        if not seen[ids[row]]:
            seen[ids[row]] = True
            kept[row] = True
    return kept


@njit(cache=True)
def project_pairs(members, owners, kinds, left_kind, right_kind, cap, max_pairs):
    rows = len(owners)
    stamp = np.zeros(len(kinds), np.int64)
    picked = np.empty(rows, np.int64)
    bounds = np.empty(rows + 1, np.int64)
    bounds[0] = 0
    groups, fill, start = 0, 0, 0
    while start < rows:
        stop = start
        while stop < rows and owners[stop] == owners[start]:
            stop += 1
        groups += 1
        taken = 0
        for index in range(stop - 1, start - 1, -1):
            member = members[index]
            if stamp[member] == groups:
                continue
            stamp[member] = groups
            picked[fill] = member
            fill += 1
            taken += 1
            if cap > 0 and taken == cap:
                break
        bounds[groups] = fill
        start = stop
    total = 0
    for group in range(groups):
        lo, hi = bounds[group], bounds[group + 1]
        picked[lo:hi] = np.sort(picked[lo:hi])
        left, right = 0, 0
        for slot in range(lo, hi):
            if kinds[picked[slot]] == left_kind:
                left += 1
            if kinds[picked[slot]] == right_kind:
                right += 1
        total += left * (left - 1) // 2 if left_kind == right_kind else left * right
        if total > max_pairs:
            raise ValueError("projection candidate edges exceed the graph resource budget")
    source = np.empty(total, np.int32)
    target = np.empty(total, np.int32)
    at = 0
    for group in range(groups):
        lo, hi = bounds[group], bounds[group + 1]
        for i in range(lo, hi):
            if kinds[picked[i]] != left_kind:
                continue
            for j in range(i + 1 if left_kind == right_kind else lo, hi):
                if kinds[picked[j]] != right_kind:
                    continue
                source[at] = picked[i]
                target[at] = picked[j]
                at += 1
    return source, target


@njit(cache=True)
def within_pairs(members, offsets, kinds, nodes, max_pairs):
    chunks = len(offsets) - 1
    total = 0
    for chunk in range(chunks):
        size = offsets[chunk + 1] - offsets[chunk]
        total += size * (size - 1) // 2
        if total > max_pairs:
            raise ValueError("completion candidate edges exceed the graph resource budget")
    picked = np.empty(len(members), np.int64)
    bounds = np.empty(chunks + 1, np.int64)
    bounds[0] = 0
    fill = 0
    for chunk in range(chunks):
        for index in range(offsets[chunk], offsets[chunk + 1]):
            picked[fill] = members[index]
            fill += 1
        section = np.sort(picked[bounds[chunk] : fill])
        counts = np.zeros(nodes + 1, np.int64)
        for value in section:
            counts[kinds[value] + 1] += 1
        for kind in range(nodes):
            counts[kind + 1] += counts[kind]
        ordered = np.empty(len(section), np.int64)
        for value in section:
            ordered[counts[kinds[value]]] = value
            counts[kinds[value]] += 1
        picked[bounds[chunk] : fill] = ordered
        bounds[chunk + 1] = fill
    source = np.empty(total, np.int32)
    target = np.empty(total, np.int32)
    at = 0
    for chunk in range(chunks):
        lo, hi = bounds[chunk], bounds[chunk + 1]
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                source[at] = picked[i]
                target[at] = picked[j]
                at += 1
    return source, target


@njit(cache=True)
def smallest(ids, values, groups):
    result = np.full(groups, np.inf)
    for row in range(len(ids)):
        if values[row] < result[ids[row]]:
            result[ids[row]] = values[row]
    return result


@njit(cache=True)
def precedes(a, b, best, earliest, first, second, relation):
    if best[a] != best[b]:
        return best[a] < best[b]
    if earliest[a] != earliest[b]:
        return earliest[a] < earliest[b]
    if first[a] != first[b]:
        return first[a] < first[b]
    if second[a] != second[b]:
        return second[a] < second[b]
    return relation[a] < relation[b]


@njit(cache=True)
def within_cap(groups, count, best, earliest, first, second, relation, cap):
    entries = len(groups)
    sizes = np.zeros(count + 1, np.int64)
    for entry in range(entries):
        sizes[groups[entry] + 1] += 1
    for group in range(count):
        sizes[group + 1] += sizes[group]
    order = np.empty(entries, np.int64)
    fill = sizes[:count].copy()
    for entry in range(entries):
        order[fill[groups[entry]]] = entry
        fill[groups[entry]] += 1
    result = np.zeros(entries, np.bool_)
    heap = np.empty(cap, np.int64)
    for group in range(count):
        lo, hi = sizes[group], sizes[group + 1]
        if hi - lo <= cap:
            for slot in range(lo, hi):
                result[order[slot]] = True
            continue
        size = 0
        for slot in range(lo, hi):
            item = order[slot]
            if size < cap:
                child = size
                heap[child] = item
                size += 1
                while child > 0:
                    parent = (child - 1) // 2
                    if not precedes(heap[parent], heap[child], best, earliest, first, second, relation):
                        break
                    heap[parent], heap[child] = heap[child], heap[parent]
                    child = parent
            elif precedes(item, heap[0], best, earliest, first, second, relation):
                heap[0] = item
                parent = 0
                while True:
                    left, right, worst = 2 * parent + 1, 2 * parent + 2, parent
                    if left < size and precedes(heap[worst], heap[left], best, earliest, first, second, relation):
                        worst = left
                    if right < size and precedes(heap[worst], heap[right], best, earliest, first, second, relation):
                        worst = right
                    if worst == parent:
                        break
                    heap[parent], heap[worst] = heap[worst], heap[parent]
                    parent = worst
        for slot in range(size):
            result[heap[slot]] = True
    return result


@njit(cache=True)
def components(source, target, count):
    parent = np.arange(count)
    for row in range(len(source)):
        a, b = source[row], target[row]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a < b:
            parent[b] = a
        elif b < a:
            parent[a] = b
    labels = np.empty(count, np.int64)
    for node in range(count):
        root = node
        while parent[root] != root:
            root = parent[root]
        labels[node] = root
    return labels


@njit(cache=True)
def linked(adjacency, a, b, k, partner):
    return adjacency[a, b] or (a == k and b == partner) or (a == partner and b == k)


@njit(cache=True)
def curvature_with(adjacency, offsets, columns, i, j, k, partner):
    di = offsets[i + 1] - offsets[i] + (1 if i == k or i == partner else 0)
    dj = offsets[j + 1] - offsets[j] + (1 if j == k or j == partner else 0)
    if min(di, dj) == 1:
        return 0.0
    triangles = 0
    for position in range(offsets[i], offsets[i + 1] + (1 if i == k or i == partner else 0)):
        x = columns[position] if position < offsets[i + 1] else (partner if i == k else k)
        triangles += linked(adjacency, j, x, k, partner)
    largest, smallest = max(di, dj), min(di, dj)
    value = 2 / di + 2 / dj - 2 + 2 * triangles / largest + triangles / smallest
    squares, gamma = 0, 0
    for side in range(2):
        near, far, other = (i, j, j) if side == 0 else (j, i, i)
        extra_near = 1 if near == k or near == partner else 0
        for position in range(offsets[near], offsets[near + 1] + extra_near):
            kk = columns[position] if position < offsets[near + 1] else (partner if near == k else k)
            if linked(adjacency, far, kk, k, partner) or kk == other:
                continue
            count = -1
            extra = 1 if kk == k or kk == partner else 0
            for slot in range(offsets[kk], offsets[kk + 1] + extra):
                x = columns[slot] if slot < offsets[kk + 1] else (partner if kk == k else k)
                count += linked(adjacency, far, x, k, partner) & (not linked(adjacency, near, x, k, partner))
            if count > 0:
                squares += 1
            if count > gamma:
                gamma = count
    if gamma > 0:
        value += squares / (gamma * largest)
    return value


@njit(cache=True, parallel=True)
def forman(adjacency, offsets, columns, sources, targets):
    values = np.zeros(len(sources))
    for edge in prange(len(sources)):
        values[edge] = curvature_with(adjacency, offsets, columns, sources[edge], targets[edge], -1, -1)
    return values


@njit(cache=True, parallel=True)
def forman_with(adjacency, offsets, columns, i, j, ks, ls):
    values = np.zeros(len(ks))
    for candidate in prange(len(ks)):
        values[candidate] = curvature_with(adjacency, offsets, columns, i, j, ks[candidate], ls[candidate])
    return values
