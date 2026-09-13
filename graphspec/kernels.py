import numpy as np
from numba import njit, prange


@njit(cache=True)
def lower(ids, offsets, ticks, entity, bound):
    lo, hi = offsets[entity], offsets[entity + 1]
    while lo < hi:
        middle = (lo + hi) // 2
        if ticks[ids[middle]] < bound:
            lo = middle + 1
        else:
            hi = middle
    return lo


@njit(cache=True, parallel=True)
def walked(ids, offsets, source, target, ticks, roots, floors, cutoffs, active, identity, fanout, bias, seed, stream, hops):
    result = np.full((len(roots), hops), -1, np.int32)
    for row in prange(len(roots)):
        if not active[row]:
            continue
        current, cutoff, floor = roots[row], cutoffs[row], floors[row]
        weights = np.empty(fanout, np.float64)
        for hop in range(hops):
            lo = lower(ids, offsets, ticks, current, floor)
            hi = lower(ids, offsets, ticks, current, cutoff)
            if hi - fanout > lo:
                lo = hi - fanout
            if hi <= lo:
                break
            value = mix(np.uint64(seed ^ (stream + hop)))
            for column in range(identity.shape[1]):
                value = mix(value ^ np.uint64(identity[row, column]))
            draw = np.float64(value >> np.uint64(11)) / np.float64(1 << 53)
            count = hi - lo
            best = -np.inf
            for slot in range(count):
                logit = -bias * (cutoff - ticks[ids[lo + slot]]) / 2.0
                weights[slot] = logit
                if logit > best:
                    best = logit
            total = 0.0
            for slot in range(count):
                weights[slot] = np.exp(weights[slot] - best)
                total += weights[slot]
            wanted = draw * total
            cumulative, index = 0.0, 0
            for slot in range(count):
                cumulative += weights[slot]
                if cumulative <= wanted:
                    index += 1
            if index > count - 1:
                index = count - 1
            event = ids[lo + index]
            result[row, hop] = event
            current = target[event] if source[event] == current else source[event]
            cutoff = ticks[event]
    return result


@njit(cache=True)
def mix(value):
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


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
        slot = np.int64(mix(mix(mix(np.uint64(a)) ^ np.uint64(b)) ^ np.uint64(c)) & mask)
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
def keyed2(first, second):
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
        a, b = np.int64(first[row]), np.int64(second[row])
        slot = np.int64(mix(mix(np.uint64(a)) ^ np.uint64(b)) & mask)
        while True:
            group = slots[slot]
            if group < 0:
                slots[slot] = count
                representative[count] = row
                ids[row] = count
                count += 1
                break
            other = representative[group]
            if np.int64(first[other]) == a and np.int64(second[other]) == b:
                ids[row] = group
                break
            slot = (slot + 1) & (capacity - 1)
    return ids, count


@njit(cache=True)
def rehash(slots, keys_low, keys_high, count):
    capacity = len(slots)
    mask = np.uint64(capacity - 1)
    for index in range(count):
        slot = np.int64(mix(mix(np.uint64(keys_low[index])) ^ np.uint64(keys_high[index])) & mask)
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
        slot = np.int64(mix(mix(np.uint64(a)) ^ np.uint64(b)) & mask)
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
def runs(ids, step, groups):
    last = np.full(groups, -2, np.int64)
    current = np.zeros(groups, np.int64)
    begun = np.zeros(groups, np.int64)
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
def degrees(obs_entity, obs_step, fact_source, fact_target, fact_step, use, entities, cumulative):
    count = np.zeros(entities, np.int64)
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
def activity(obs_entity, entities):
    count = np.zeros(entities, np.int64)
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
                    if alive[other] and minimum[other] > 0 and degree[other] < minimum[other] and not queued[other - low]:
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
def last_rows(ids, groups):
    seen = np.zeros(groups, np.bool_)
    kept = np.zeros(len(ids), np.bool_)
    for row in range(len(ids) - 1, -1, -1):
        if not seen[ids[row]]:
            seen[ids[row]] = True
            kept[row] = True
    return kept


@njit(cache=True)
def combined(ids, values, groups):
    result = np.zeros(groups, np.int64)
    for row in range(len(ids)):
        result[ids[row]] |= values[row]
    return result


@njit(cache=True)
def turn_index(source, target, ticks, member, turn_of, entities):
    events = len(ticks)
    stamp = np.full(entities, -1, np.int64)
    turn_count = np.zeros(entities, np.int64)
    entry_count = np.zeros(entities, np.int64)
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != turn_of[row]:
                stamp[who] = turn_of[row]
                turn_count[who] += 1
            entry_count[who] += 1
    turn_offset = np.zeros(entities + 1, np.int64)
    entry_offset = np.zeros(entities + 1, np.int64)
    for who in range(entities):
        turn_offset[who + 1] = turn_offset[who] + turn_count[who]
        entry_offset[who + 1] = entry_offset[who] + entry_count[who]
    turn_value = np.empty(turn_offset[entities], np.int64)
    turn_lo = np.empty(turn_offset[entities], np.int64)
    turn_hi = np.empty(turn_offset[entities], np.int64)
    entries = np.empty(entry_offset[entities], np.int64)
    stamp[:] = -1
    turn_fill = turn_offset[:entities].copy()
    entry_fill = entry_offset[:entities].copy()
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != turn_of[row]:
                stamp[who] = turn_of[row]
                turn_value[turn_fill[who]] = turn_of[row]
                turn_lo[turn_fill[who]] = entry_fill[who]
                turn_fill[who] += 1
            entries[entry_fill[who]] = row
            entry_fill[who] += 1
            turn_hi[turn_fill[who] - 1] = entry_fill[who]
    return turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset


@njit(cache=True)
def turn_of_entry(turn_offset, turn_hi, entries, entry_offset, who, row):
    lo, hi = entry_offset[who], entry_offset[who + 1]
    while lo < hi:
        middle = (lo + hi) // 2
        if entries[middle] < row:
            lo = middle + 1
        else:
            hi = middle
    base, count = turn_offset[who], turn_offset[who + 1] - turn_offset[who]
    low, high = 0, count
    while low < high:
        middle = (low + high) // 2
        if turn_hi[base + middle] <= lo:
            low = middle + 1
        else:
            high = middle
    return base + low


@njit(cache=True)
def boundary_flag(turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset, ticks, who, anchor, destination, lookback, window):
    if ticks[destination] // 2 - ticks[anchor] // 2 >= window:
        return 0
    prior = turn_of_entry(turn_offset, turn_hi, entries, entry_offset, who, anchor)
    current = turn_of_entry(turn_offset, turn_hi, entries, entry_offset, who, destination)
    lag = turn_value[current] - turn_value[prior]
    if lag < 1 or lag > lookback:
        return 0
    flag = 0
    if ticks[anchor] == ticks[entries[turn_lo[prior]]]:
        flag |= 1
    if ticks[anchor] == ticks[entries[turn_hi[prior] - 1]]:
        flag |= 2
    return flag


@njit(cache=True)
def boundary_rows(turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset, source, target, ticks, who, lookback, window, out_source, out_target, out_flag, start, write):
    base, count = turn_offset[who], turn_offset[who + 1] - turn_offset[who]
    total = 0
    for index in range(count):
        destination_lo, destination_hi = turn_lo[base + index], turn_hi[base + index]
        for lag in range(1, lookback + 1):
            wanted = turn_value[base + index] - lag
            low, high = 0, index
            while low < high:
                middle = (low + high) // 2
                if turn_value[base + middle] < wanted:
                    low = middle + 1
                else:
                    high = middle
            if low >= index or turn_value[base + low] != wanted:
                continue
            prior_lo, prior_hi = turn_lo[base + low], turn_hi[base + low]
            first, last = ticks[entries[prior_lo]], ticks[entries[prior_hi - 1]]
            for anchor in range(prior_lo, prior_hi):
                a = entries[anchor]
                flag = (1 if ticks[a] == first else 0) | (2 if ticks[a] == last else 0)
                if flag == 0:
                    continue
                other_a = target[a] if source[a] == who else source[a]
                for destination in range(destination_lo, destination_hi):
                    b = entries[destination]
                    if ticks[b] // 2 - ticks[a] // 2 >= window:
                        continue
                    other_b = target[b] if source[b] == who else source[b]
                    merged = flag
                    if other_a == other_b and other_a != who:
                        shared = boundary_flag(turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset, ticks, other_a, a, b, lookback, window)
                        if other_a < who and shared != 0:
                            continue
                        merged |= shared
                    if write:
                        out_source[start + total] = a
                        out_target[start + total] = b
                        out_flag[start + total] = merged
                    total += 1
    return total


@njit(cache=True, parallel=True)
def boundaries(source, target, ticks, member, turn_of, lookback, window, entities):
    turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset = turn_index(source, target, ticks, member, turn_of, entities)
    counts = np.zeros(entities, np.int64)
    empty = np.empty(0, np.int32)
    for who in prange(entities):
        counts[who] = boundary_rows(turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset, source, target, ticks, who, lookback, window,
                                    empty, empty, empty, 0, False)
    offsets = np.zeros(entities + 1, np.int64)
    for who in range(entities):
        offsets[who + 1] = offsets[who] + counts[who]
    out_source = np.empty(offsets[entities], np.int32)
    out_target = np.empty(offsets[entities], np.int32)
    out_flag = np.empty(offsets[entities], np.int32)
    for who in prange(entities):
        boundary_rows(turn_offset, turn_value, turn_lo, turn_hi, entries, entry_offset, source, target, ticks, who, lookback, window,
                      out_source, out_target, out_flag, offsets[who], True)
    return out_source, out_target, out_flag


@njit(cache=True)
def coupled(entity, step, chosen, mode, width, cap, limit, entities):
    rows = len(entity)
    offsets = np.zeros(entities + 1, np.int64)
    for ix in range(rows):
        offsets[entity[ix] + 1] += 1
    for who in range(entities):
        offsets[who + 1] += offsets[who]
    csr = np.empty(rows, np.int64)
    position = np.empty(rows, np.int64)
    cursor = offsets[:entities].copy()
    for ix in range(rows):
        who = entity[ix]
        position[ix] = cursor[who] - offsets[who]
        csr[cursor[who]] = ix
        cursor[who] += 1
    out_source = np.empty(0, np.int32)
    out_target = np.empty(0, np.int32)
    total = 0
    for phase in range(2):
        if phase == 1:
            out_source = np.empty(total, np.int32)
            out_target = np.empty(total, np.int32)
            total = 0
        for ix in range(rows):
            base, place = offsets[entity[ix]], position[ix]
            if mode == 0:
                floor = place - cap
                if floor < 0:
                    floor = 0
                for slot in range(place - 1, floor - 1, -1):
                    previous = csr[base + slot]
                    if step[ix] - step[previous] >= width:
                        break
                    if phase == 1:
                        out_source[total] = previous
                        out_target[total] = ix
                    total += 1
            elif mode == 1:
                for index in range(len(chosen)):
                    lag = chosen[index]
                    if place >= lag:
                        previous = csr[base + place - lag]
                        if step[ix] - step[previous] <= limit:
                            if phase == 1:
                                out_source[total] = previous
                                out_target[total] = ix
                            total += 1
            else:
                for index in range(len(chosen)):
                    wanted = step[ix] - chosen[index]
                    low, high = 0, place
                    while low < high:
                        middle = (low + high) // 2
                        if step[csr[base + middle]] < wanted:
                            low = middle + 1
                        else:
                            high = middle
                    if low < place and step[csr[base + low]] == wanted:
                        if phase == 1:
                            out_source[total] = csr[base + low]
                            out_target[total] = ix
                        total += 1
    return out_source, out_target


@njit(cache=True)
def bundled(entity, offsets, prior, altered, actor, trace, bundles, decided, entities, limit, wires, holders):
    kind = np.empty(limit, np.int8)
    tick = np.empty(limit, np.int32)
    owner = np.empty(limit, np.int32)
    mode = np.empty(limit, np.int8)
    fill = np.empty(limit, np.int32)
    member_offset = np.empty(limit + 1, np.int64)
    member_id = np.empty(holders, np.int32)
    out_source = np.empty(wires, np.int32)
    out_target = np.empty(wires, np.int32)
    out_flag = np.empty(wires, np.int32)
    span = trace if trace > 0 else 1
    last_node = np.full(entities, -1, np.int64)
    trace_node = np.empty((entities, span), np.int64)
    trace_row = np.empty((entities, span), np.int64)
    trace_count = np.zeros(entities, np.int64)
    node_of_row = np.empty(len(entity), np.int64)
    total, held, used = 0, 0, 0
    member_offset[0] = 0
    previous_action, bundle = -1, -1
    for moment in range(len(offsets) - 1):
        start, stop = offsets[moment], offsets[moment + 1]
        for row in range(start, stop):
            who = entity[row]
            node = total
            kind[node] = 0
            tick[node] = 2 * moment
            owner[node] = who
            mode[node] = 0
            fill[node] = row
            member_id[used] = row
            used += 1
            member_offset[node + 1] = used
            node_of_row[row] = node
            total += 1
            if last_node[who] >= 0:
                out_source[held] = last_node[who]
                out_target[held] = node
                out_flag[held] = 4
                held += 1
                if altered[row]:
                    change = total
                    kind[change] = 1
                    tick[change] = 2 * moment
                    owner[change] = who
                    mode[change] = 1
                    fill[change] = row
                    member_id[used] = prior[row]
                    member_id[used + 1] = row
                    used += 2
                    member_offset[change + 1] = used
                    total += 1
                    out_source[held] = last_node[who]
                    out_target[held] = change
                    out_flag[held] = 32
                    held += 1
                    out_source[held] = change
                    out_target[held] = node
                    out_flag[held] = 32
                    held += 1
            last_node[who] = node
            if trace > 0:
                trace_node[who, trace_count[who]] = node
                trace_row[who, trace_count[who]] = row
                trace_count[who] += 1
                if trace_count[who] == trace:
                    group = total
                    kind[group] = 3
                    tick[group] = 2 * moment
                    owner[group] = who
                    mode[group] = 2
                    fill[group] = row
                    for slot in range(trace):
                        member_id[used] = trace_row[who, slot]
                        used += 1
                    member_offset[group + 1] = used
                    total += 1
                    for slot in range(trace):
                        out_source[held] = trace_node[who, slot]
                        out_target[held] = group
                        out_flag[held] = 512
                        held += 1
                    out_source[held] = group
                    out_target[held] = node
                    out_flag[held] = 1024
                    held += 1
                    trace_count[who] = 0
        if bundles:
            bundle = total
            kind[bundle] = 2
            tick[bundle] = 2 * moment
            owner[bundle] = -1
            mode[bundle] = 5 if stop > start else 4
            fill[bundle] = moment
            for row in range(start, stop):
                member_id[used] = row
                used += 1
            member_offset[bundle + 1] = used
            total += 1
            for row in range(start, stop):
                out_source[held] = node_of_row[row]
                out_target[held] = bundle
                out_flag[held] = 512
                held += 1
        if decided:
            action = total
            kind[action] = 4
            tick[action] = 2 * moment + 1
            owner[action] = actor[moment]
            mode[action] = 3
            fill[action] = moment
            member_offset[action + 1] = used
            total += 1
            if bundles:
                out_source[held] = bundle
                out_target[held] = action
                out_flag[held] = 8
                held += 1
            if moment > 0:
                out_source[held] = previous_action
                out_target[held] = action
                out_flag[held] = 1
                held += 1
            previous_action = action
    return (kind[:total], tick[:total], owner[:total], mode[:total], fill[:total], member_offset[:total + 1], member_id[:used],
            out_source[:held], out_target[:held], out_flag[:held])


@njit(cache=True)
def replayed(entity, step, altered, present_source, present_target, present_code, present_step, absent_source, absent_target, absent_code, absent_step,
             actor, subject, context, category, steps, entities):
    observations, statements, removals = len(step), len(present_step), len(absent_step)
    decided = len(actor) > 0
    total = observations + statements + removals + (steps if decided else 0)
    events = np.empty((total, 9), np.int32)
    last = np.full(entities, -1, np.int64)
    capacity = 16
    while capacity < 2 * (statements + removals + 1):
        capacity *= 2
    slots = np.zeros(capacity, np.int64)
    keys_source = np.empty(capacity, np.int64)
    keys_target = np.empty(capacity, np.int64)
    keys_code = np.empty(capacity, np.int64)
    states = np.empty(capacity, np.int64)
    stored = 0
    filled, observed, asserted, retracted = 0, 0, 0, 0
    for moment in range(steps):
        while observed < observations and step[observed] == moment:
            owner = entity[observed]
            previous = last[owner]
            if altered[observed]:
                events[filled, 0] = owner
                events[filled, 1] = owner
                events[filled, 2] = 2 * moment
                events[filled, 3] = 0
                events[filled, 4] = observed
                events[filled, 5] = observed
                events[filled, 6] = previous
                events[filled, 7] = -1
                events[filled, 8] = -1
                filled += 1
            last[owner] = observed
            observed += 1
        for phase in range(2):
            while True:
                if phase == 0:
                    if asserted >= statements or present_step[asserted] != moment:
                        break
                    row, live = asserted, 1
                    left, right = entity[present_source[row]], entity[present_target[row]]
                    code = present_code[row]
                    origin, destination = present_source[row], present_target[row]
                else:
                    if retracted >= removals or absent_step[retracted] != moment:
                        break
                    row, live = retracted, 0
                    left, right = absent_source[row], absent_target[row]
                    code = absent_code[row]
                    if last[left] < 0 or last[right] < 0:
                        return events[:0], np.int64(-1)
                    origin, destination = last[left], last[right]
                slot = np.int64(mix(mix(mix(np.uint64(left)) ^ np.uint64(right)) ^ np.uint64(code)) & np.uint64(capacity - 1))
                while True:
                    held = slots[slot]
                    if held == 0:
                        slots[slot] = stored + 1
                        keys_source[stored] = left
                        keys_target[stored] = right
                        keys_code[stored] = code
                        states[stored] = -1
                        held = stored + 1
                        stored += 1
                        break
                    if keys_source[held - 1] == left and keys_target[held - 1] == right and keys_code[held - 1] == code:
                        break
                    slot = (slot + 1) & (capacity - 1)
                before = states[held - 1]
                if before != live:
                    events[filled, 0] = left
                    events[filled, 1] = right
                    events[filled, 2] = 2 * moment
                    events[filled, 3] = 1
                    events[filled, 4] = origin
                    events[filled, 5] = destination
                    events[filled, 6] = code
                    events[filled, 7] = before
                    events[filled, 8] = live
                    filled += 1
                    states[held - 1] = live
                if phase == 0:
                    asserted += 1
                else:
                    retracted += 1
        if decided:
            events[filled, 0] = actor[moment]
            events[filled, 1] = subject[moment]
            events[filled, 2] = 2 * moment + 1
            events[filled, 3] = 2
            events[filled, 4] = context[moment, 0]
            events[filled, 5] = context[moment, 1]
            events[filled, 6] = category[moment]
            events[filled, 7] = -1
            events[filled, 8] = -1
            filled += 1
    return events[:filled], np.int64(0)


@njit(cache=True)
def group_index(source, target, ticks, member, entities):
    events = len(ticks)
    belongs = np.empty(events, np.int64)
    groups = 0
    for row in range(events):
        if row > 0 and ticks[row] != ticks[row - 1]:
            groups += 1
        belongs[row] = groups
    stamp = np.full(entities, -1, np.int64)
    group_count = np.zeros(entities, np.int64)
    entry_count = np.zeros(entities, np.int64)
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != belongs[row]:
                stamp[who] = belongs[row]
                group_count[who] += 1
            entry_count[who] += 1
    group_offset = np.zeros(entities + 1, np.int64)
    entry_offset = np.zeros(entities + 1, np.int64)
    for who in range(entities):
        group_offset[who + 1] = group_offset[who] + group_count[who]
        entry_offset[who + 1] = entry_offset[who] + entry_count[who]
    group_lo = np.empty(group_offset[entities], np.int64)
    group_hi = np.empty(group_offset[entities], np.int64)
    group_tick = np.empty(group_offset[entities], np.int64)
    entries = np.empty(entry_offset[entities], np.int64)
    stamp[:] = -1
    group_fill = group_offset[:entities].copy()
    entry_fill = entry_offset[:entities].copy()
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != belongs[row]:
                stamp[who] = belongs[row]
                group_lo[group_fill[who]] = entry_fill[who]
                group_tick[group_fill[who]] = belongs[row]
                group_fill[who] += 1
            entries[entry_fill[who]] = row
            entry_fill[who] += 1
            group_hi[group_fill[who] - 1] = entry_fill[who]
    return belongs, group_offset, group_lo, group_hi, group_tick, entries


@njit(cache=True)
def dag_row(row, source, target, ticks, belongs, group_offset, group_lo, group_hi, group_tick, entries, lags, windowed, window, cap, buffer):
    fill = 0
    for side in range(2):
        who = source[row] if side == 0 else target[row]
        if side == 1 and target[row] == source[row]:
            continue
        base, count = group_offset[who], group_offset[who + 1] - group_offset[who]
        low, high = 0, count
        while low < high:
            middle = (low + high) // 2
            if group_tick[base + middle] < belongs[row]:
                low = middle + 1
            else:
                high = middle
        available = low
        step = 0
        while True:
            if windowed:
                lag = step + 1
            else:
                if step >= len(lags):
                    break
                lag = lags[step]
            if available < lag:
                break
            slot = base + available - lag
            lo, hi = group_lo[slot], group_hi[slot]
            if windowed and ticks[row] - ticks[entries[hi - 1]] > window:
                break
            start = lo if cap == 0 else max(lo, hi - cap)
            for index in range(start, hi):
                previous = entries[index]
                if ticks[row] - ticks[previous] > window:
                    continue
                if fill == len(buffer):
                    grown = np.empty(2 * len(buffer), np.int64)
                    grown[:fill] = buffer[:fill]
                    buffer = grown
                buffer[fill] = previous
                fill += 1
            step += 1
    if fill == 0:
        return buffer, 0
    ordered = np.unique(buffer[:fill])
    buffer[:len(ordered)] = ordered
    return buffer, len(ordered)


@njit(cache=True, parallel=True)
def event_dag(source, target, ticks, member, lags, windowed, window, cap, entities):
    events = len(ticks)
    belongs, group_offset, group_lo, group_hi, group_tick, entries = group_index(source, target, ticks, member, entities)
    counts = np.zeros(events, np.int64)
    for row in prange(events):
        if member[row]:
            buffer = np.empty(64, np.int64)
            _, counts[row] = dag_row(row, source, target, ticks, belongs, group_offset, group_lo, group_hi, group_tick, entries, lags, windowed, window, cap, buffer)
    offsets = np.zeros(events + 1, np.int64)
    for row in range(events):
        offsets[row + 1] = offsets[row] + counts[row]
    out_source = np.empty(offsets[events], np.int32)
    out_target = np.empty(offsets[events], np.int32)
    for row in prange(events):
        if member[row] and counts[row] > 0:
            buffer = np.empty(max(64, counts[row]), np.int64)
            buffer, fill = dag_row(row, source, target, ticks, belongs, group_offset, group_lo, group_hi, group_tick, entries, lags, windowed, window, cap, buffer)
            for index in range(fill):
                out_source[offsets[row] + index] = buffer[index]
                out_target[offsets[row] + index] = row
    return out_source, out_target


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
def project_pairs(members, owners, kinds, left_kind, right_kind, cap):
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
def within_pairs(members, offsets, kinds, nodes):
    chunks = len(offsets) - 1
    picked = np.empty(len(members), np.int64)
    bounds = np.empty(chunks + 1, np.int64)
    bounds[0] = 0
    fill = 0
    for chunk in range(chunks):
        for index in range(offsets[chunk], offsets[chunk + 1]):
            picked[fill] = members[index]
            fill += 1
        section = np.sort(picked[bounds[chunk]:fill])
        counts = np.zeros(nodes + 1, np.int64)
        for value in section:
            counts[kinds[value] + 1] += 1
        for kind in range(nodes):
            counts[kind + 1] += counts[kind]
        ordered = np.empty(len(section), np.int64)
        for value in section:
            ordered[counts[kinds[value]]] = value
            counts[kinds[value]] += 1
        picked[bounds[chunk]:fill] = ordered
        bounds[chunk + 1] = fill
    total = 0
    for chunk in range(chunks):
        size = bounds[chunk + 1] - bounds[chunk]
        total += size * (size - 1) // 2
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
def precedes(a, b, best, earliest, part, low, high):
    if best[a] != best[b]:
        return best[a] < best[b]
    if earliest[a] != earliest[b]:
        return earliest[a] < earliest[b]
    if part[a] != part[b]:
        return part[a] < part[b]
    if low[a] != low[b]:
        return low[a] < low[b]
    return high[a] < high[b]


@njit(cache=True)
def within_cap(groups, count, best, earliest, part, low, high, cap):
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
                    if not precedes(heap[parent], heap[child], best, earliest, part, low, high):
                        break
                    heap[parent], heap[child] = heap[child], heap[parent]
                    child = parent
            elif precedes(item, heap[0], best, earliest, part, low, high):
                heap[0] = item
                parent = 0
                while True:
                    left, right, worst = 2 * parent + 1, 2 * parent + 2, parent
                    if left < size and precedes(heap[worst], heap[left], best, earliest, part, low, high):
                        worst = left
                    if right < size and precedes(heap[worst], heap[right], best, earliest, part, low, high):
                        worst = right
                    if worst == parent:
                        break
                    heap[parent], heap[worst] = heap[worst], heap[parent]
                    parent = worst
        for slot in range(size):
            result[heap[slot]] = True
    return result


@njit(cache=True)
def audit(obs_entity, obs_step, obs_anchor, fact_source, fact_target, fact_step, fact_present, fact_code, canonical,
          actor, subject, context, label_entity, label_step, labelled, steps, entities):
    flags = np.zeros(19, np.bool_)
    observations, facts, labels = len(obs_step), len(fact_step), len(label_step)
    for row in range(observations):
        if obs_step[row] < 0 or obs_step[row] >= steps:
            flags[0] = True
        if obs_entity[row] < 0 or obs_entity[row] >= entities:
            flags[3] = True
        if row > 0 and obs_step[row] < obs_step[row - 1]:
            flags[11] = True
    for row in range(facts):
        if fact_step[row] < 0 or fact_step[row] >= steps:
            flags[1] = True
        if fact_source[row] < 0 or fact_source[row] >= entities:
            flags[4] = True
        if fact_target[row] < 0 or fact_target[row] >= entities:
            flags[5] = True
        if fact_code[row] < 0 or fact_code[row] >= len(canonical) or not canonical[fact_code[row]]:
            flags[10] = True
        if row > 0 and np.int64(fact_step[row]) * 2 + (0 if fact_present[row] else 1) < np.int64(fact_step[row - 1]) * 2 + (0 if fact_present[row - 1] else 1):
            flags[12] = True
    for row in range(labels):
        if label_step[row] < 0 or label_step[row] >= steps:
            flags[2] = True
        if labelled and (label_entity[row] < 0 or label_entity[row] >= entities):
            flags[9] = True
        if row > 0 and label_step[row] < label_step[row - 1]:
            flags[13] = True
    for row in range(len(actor)):
        if actor[row] < 0 or actor[row] >= entities:
            flags[6] = True
    for row in range(len(subject)):
        if subject[row] < 0 or subject[row] >= entities:
            flags[7] = True
    for row in range(len(context)):
        if context[row] < 0 or context[row] >= entities:
            flags[8] = True
    if flags.any():
        return flags
    stamp = np.full(entities, -1, np.int64)
    structure = np.full(entities, -1, np.int64)
    first = np.full(entities, steps, np.int64)
    cursor = 0
    for row in range(facts):
        moment = np.int64(fact_step[row])
        while cursor < observations and obs_step[cursor] <= moment:
            who, at = obs_entity[cursor], np.int64(obs_step[cursor])
            if stamp[who] == at:
                flags[14] = True
            stamp[who] = at
            if not obs_anchor[cursor]:
                structure[who] = at
            if first[who] == steps:
                first[who] = at
            cursor += 1
        if fact_present[row]:
            if structure[fact_source[row]] != moment or structure[fact_target[row]] != moment:
                flags[15] = True
        elif first[fact_source[row]] > moment or first[fact_target[row]] > moment:
            flags[16] = True
    while cursor < observations:
        who, at = obs_entity[cursor], np.int64(obs_step[cursor])
        if stamp[who] == at:
            flags[14] = True
        stamp[who] = at
        cursor += 1
    stamp[:] = -1
    cursor = 0
    for moment in range(len(actor)):
        while cursor < observations and obs_step[cursor] <= moment:
            stamp[obs_entity[cursor]] = obs_step[cursor]
            cursor += 1
        if stamp[actor[moment]] != moment or stamp[subject[moment]] != moment or stamp[context[moment]] != moment:
            flags[17] = True
    if labelled:
        stamp[:] = -1
        cursor = 0
        for row in range(labels):
            moment = np.int64(label_step[row])
            while cursor < observations and obs_step[cursor] <= moment:
                stamp[obs_entity[cursor]] = obs_step[cursor]
                cursor += 1
            if stamp[label_entity[row]] != moment:
                flags[18] = True
    return flags


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


@njit(cache=True, parallel=True)
def forman(adjacency, sources, targets):
    size = adjacency.shape[0]
    values = np.zeros(len(sources))
    for edge in prange(len(sources)):
        i, j = sources[edge], targets[edge]
        di, dj, triangles = 0, 0, 0
        for k in range(size):
            di += adjacency[i, k]
            dj += adjacency[j, k]
            triangles += adjacency[i, k] & adjacency[j, k]
        if min(di, dj) == 1:
            continue
        largest, smallest = max(di, dj), min(di, dj)
        value = 2 / di + 2 / dj - 2 + 2 * triangles / largest + triangles / smallest
        squares, gamma = 0, 0
        for side in range(2):
            near, far, other = (i, j, j) if side == 0 else (j, i, i)
            for k in range(size):
                if not adjacency[near, k] or adjacency[far, k] or k == other:
                    continue
                count = -1
                for x in range(size):
                    count += adjacency[k, x] & adjacency[far, x] & (not adjacency[near, x])
                if count > 0:
                    squares += 1
                if count > gamma:
                    gamma = count
        if gamma > 0:
            value += squares / (gamma * largest)
        values[edge] = value
    return values
