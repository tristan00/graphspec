import numpy as np
from numba import njit, prange

from graphspec.infra.hashing import mixed, uniform

TINY = 2.0**-54


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
def walked(
    ids,
    offsets,
    source,
    target,
    ticks,
    identity,
    roots,
    floors,
    cutoffs,
    active,
    keys,
    fanout,
    bias,
    start,
    hops,
):
    result = np.full((len(roots), hops), -1, np.int32)
    for row in prange(len(roots)):
        if not active[row]:
            continue
        current, cutoff, floor = roots[row], cutoffs[row], floors[row]
        walk = mixed(mixed(start) ^ keys[row])
        for hop in range(hops):
            lo = lower(ids, offsets, ticks, current, floor)
            hi = lower(ids, offsets, ticks, current, cutoff)
            if hi - fanout > lo:
                lo = hi - fanout
            if hi <= lo:
                break
            value = mixed(walk ^ np.uint64(hop))
            best, chosen = -np.inf, lo
            for slot in range(lo, hi):
                event = ids[slot]
                draw = max(uniform(mixed(value ^ identity[event])), TINY)
                score = -bias * (cutoff - ticks[event]) / 2.0 - np.log(-np.log(draw))
                if score > best:
                    best, chosen = score, slot
            event = ids[chosen]
            result[row, hop] = event
            current = target[event] if source[event] == current else source[event]
            cutoff = ticks[event]
    return result


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
        slot = np.int64(mixed(mixed(np.uint64(a)) ^ np.uint64(b)) & mask)
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
def misplaced_attributes(table, rows, limits):
    for row in range(len(table)):
        limit = limits[table[row, 2]]
        if (limit < 0 and rows[row] != -1) or (limit >= 0 and (rows[row] < 0 or rows[row] >= limit)):
            return table[row, 2]
    return -1


@njit(cache=True)
def combined(ids, values, groups):
    result = np.zeros(groups, np.int64)
    for row in range(len(ids)):
        result[ids[row]] |= values[row]
    return result


@njit(cache=True)
def period_index(source, target, ticks, member, period_of, entities):
    events = len(ticks)
    stamp = np.full(entities, -1, np.int64)
    period_count = np.zeros(entities, np.int64)
    entry_count = np.zeros(entities, np.int64)
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != period_of[row]:
                stamp[who] = period_of[row]
                period_count[who] += 1
            entry_count[who] += 1
    period_offset = np.zeros(entities + 1, np.int64)
    entry_offset = np.zeros(entities + 1, np.int64)
    for who in range(entities):
        period_offset[who + 1] = period_offset[who] + period_count[who]
        entry_offset[who + 1] = entry_offset[who] + entry_count[who]
    period_value = np.empty(period_offset[entities], np.int64)
    period_lo = np.empty(period_offset[entities], np.int64)
    period_hi = np.empty(period_offset[entities], np.int64)
    entries = np.empty(entry_offset[entities], np.int64)
    stamp[:] = -1
    period_fill = period_offset[:entities].copy()
    entry_fill = entry_offset[:entities].copy()
    for row in range(events):
        if not member[row]:
            continue
        for side in range(2):
            who = source[row] if side == 0 else target[row]
            if side == 1 and target[row] == source[row]:
                continue
            if stamp[who] != period_of[row]:
                stamp[who] = period_of[row]
                period_value[period_fill[who]] = period_of[row]
                period_lo[period_fill[who]] = entry_fill[who]
                period_fill[who] += 1
            entries[entry_fill[who]] = row
            entry_fill[who] += 1
            period_hi[period_fill[who] - 1] = entry_fill[who]
    return period_offset, period_value, period_lo, period_hi, entries, entry_offset


@njit(cache=True)
def period_of_entry(period_offset, period_hi, entries, entry_offset, who, row):
    lo, hi = entry_offset[who], entry_offset[who + 1]
    while lo < hi:
        middle = (lo + hi) // 2
        if entries[middle] < row:
            lo = middle + 1
        else:
            hi = middle
    base, count = period_offset[who], period_offset[who + 1] - period_offset[who]
    low, high = 0, count
    while low < high:
        middle = (low + high) // 2
        if period_hi[base + middle] <= lo:
            low = middle + 1
        else:
            high = middle
    return base + low


@njit(cache=True)
def boundary_flag(
    period_offset,
    period_value,
    period_lo,
    period_hi,
    entries,
    entry_offset,
    ticks,
    who,
    anchor,
    destination,
    lookback,
    window,
):
    if ticks[destination] // 2 - ticks[anchor] // 2 >= window:
        return 0
    prior = period_of_entry(period_offset, period_hi, entries, entry_offset, who, anchor)
    current = period_of_entry(period_offset, period_hi, entries, entry_offset, who, destination)
    lag = period_value[current] - period_value[prior]
    if lag < 1 or lag > lookback:
        return 0
    flag = 0
    if ticks[anchor] == ticks[entries[period_lo[prior]]]:
        flag |= 1
    if ticks[anchor] == ticks[entries[period_hi[prior] - 1]]:
        flag |= 2
    return flag


@njit(cache=True)
def boundary_rows(
    period_offset,
    period_value,
    period_lo,
    period_hi,
    entries,
    entry_offset,
    source,
    target,
    ticks,
    who,
    lookback,
    window,
    out_source,
    out_target,
    out_flag,
    start,
    write,
):
    base, count = period_offset[who], period_offset[who + 1] - period_offset[who]
    total = 0
    for index in range(count):
        destination_lo, destination_hi = period_lo[base + index], period_hi[base + index]
        for lag in range(1, lookback + 1):
            wanted = period_value[base + index] - lag
            low, high = 0, index
            while low < high:
                middle = (low + high) // 2
                if period_value[base + middle] < wanted:
                    low = middle + 1
                else:
                    high = middle
            if low >= index or period_value[base + low] != wanted:
                continue
            prior_lo, prior_hi = period_lo[base + low], period_hi[base + low]
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
                        shared = boundary_flag(
                            period_offset,
                            period_value,
                            period_lo,
                            period_hi,
                            entries,
                            entry_offset,
                            ticks,
                            other_a,
                            a,
                            b,
                            lookback,
                            window,
                        )
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
def boundaries(source, target, ticks, member, period_of, lookback, window, entities, max_pairs):
    period_offset, period_value, period_lo, period_hi, entries, entry_offset = period_index(
        source, target, ticks, member, period_of, entities
    )
    counts = np.zeros(entities, np.int64)
    empty = np.empty(0, np.int32)
    for who in prange(entities):
        counts[who] = boundary_rows(
            period_offset,
            period_value,
            period_lo,
            period_hi,
            entries,
            entry_offset,
            source,
            target,
            ticks,
            who,
            lookback,
            window,
            empty,
            empty,
            empty,
            0,
            False,
        )
    offsets = np.zeros(entities + 1, np.int64)
    for who in range(entities):
        offsets[who + 1] = offsets[who] + counts[who]
        if offsets[who + 1] > max_pairs:
            raise ValueError("event boundary edges exceed the graph resource budget")
    out_source = np.empty(offsets[entities], np.int32)
    out_target = np.empty(offsets[entities], np.int32)
    out_flag = np.empty(offsets[entities], np.int32)
    for who in prange(entities):
        boundary_rows(
            period_offset,
            period_value,
            period_lo,
            period_hi,
            entries,
            entry_offset,
            source,
            target,
            ticks,
            who,
            lookback,
            window,
            out_source,
            out_target,
            out_flag,
            offsets[who],
            True,
        )
    return out_source, out_target, out_flag


@njit(cache=True)
def coupled(entity, step, chosen, mode, width, cap, limit, entities, max_pairs):
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
            if total > max_pairs:
                raise ValueError("temporal coupling edges exceed the graph resource budget")
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
            if phase == 0 and total > max_pairs:
                raise ValueError("temporal coupling edges exceed the graph resource budget")
    return out_source, out_target


@njit(cache=True)
def bundled(
    entity,
    offsets,
    prior,
    altered,
    first,
    task,
    seen,
    trace,
    bundles,
    limit,
    wires,
    holders,
    node_base,
    row_base,
    last_node,
    trace_node,
    trace_row,
    trace_count,
    previous_action,
):
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
    node_of_row = np.empty(len(entity), np.int64)
    total, held, used = 0, 0, 0
    member_offset[0] = 0
    bundle = -1
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
            member_id[used] = row + row_base
            used += 1
            member_offset[node + 1] = used
            node_of_row[row] = node + node_base
            total += 1
            if last_node[who] >= 0:
                out_source[held] = last_node[who]
                out_target[held] = node + node_base
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
                    member_id[used + 1] = row + row_base
                    used += 2
                    member_offset[change + 1] = used
                    total += 1
                    out_source[held] = last_node[who]
                    out_target[held] = change + node_base
                    out_flag[held] = 32
                    held += 1
                    out_source[held] = change + node_base
                    out_target[held] = node + node_base
                    out_flag[held] = 32
                    held += 1
            last_node[who] = node + node_base
            if trace > 0:
                trace_node[who, trace_count[who]] = node + node_base
                trace_row[who, trace_count[who]] = row + row_base
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
                        out_target[held] = group + node_base
                        out_flag[held] = 512
                        held += 1
                    out_source[held] = group + node_base
                    out_target[held] = node + node_base
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
                member_id[used] = row + row_base
                used += 1
            member_offset[bundle + 1] = used
            total += 1
            for row in range(start, stop):
                out_source[held] = node_of_row[row]
                out_target[held] = bundle + node_base
                out_flag[held] = 512
                held += 1
        for event in range(seen[moment], seen[moment + 1]):
            action = total
            kind[action] = 4
            tick[action] = 2 * moment + 1
            owner[action] = first[event]
            mode[action] = 3
            fill[action] = task[event]
            member_offset[action + 1] = used
            total += 1
            if bundles:
                out_source[held] = bundle + node_base
                out_target[held] = action + node_base
                out_flag[held] = 8
                held += 1
            if previous_action >= 0:
                out_source[held] = previous_action
                out_target[held] = action + node_base
                out_flag[held] = 1
                held += 1
            previous_action = action + node_base
    return (
        kind[:total],
        tick[:total],
        owner[:total],
        mode[:total],
        fill[:total],
        member_offset[: total + 1],
        member_id[:used],
        out_source[:held],
        out_target[:held],
        out_flag[:held],
        previous_action,
    )


@njit(cache=True)
def replayed(
    entity,
    step,
    altered,
    present_source,
    present_target,
    present_code,
    present_step,
    absent_source,
    absent_target,
    absent_code,
    absent_step,
    first,
    second,
    starts,
    kind,
    task,
    seen,
    steps,
    last,
    base,
    initial_source,
    initial_target,
    initial_code,
    initial_state,
):
    observations, statements, removals = len(step), len(present_step), len(absent_step)
    total = observations + statements + removals + len(first)
    events = np.empty((total, 9), np.int32)
    capacity = 16
    while capacity < 2 * (statements + removals + len(initial_state) + 1):
        capacity *= 2
    slots = np.zeros(capacity, np.int64)
    keys_source = np.empty(capacity, np.int64)
    keys_target = np.empty(capacity, np.int64)
    keys_code = np.empty(capacity, np.int64)
    states = np.empty(capacity, np.int64)
    stored = 0
    for index in range(len(initial_state)):
        left, right, code = initial_source[index], initial_target[index], initial_code[index]
        slot = np.int64(
            mixed(mixed(mixed(np.uint64(left)) ^ np.uint64(right)) ^ np.uint64(code)) & np.uint64(capacity - 1)
        )
        while slots[slot] != 0:
            slot = (slot + 1) & (capacity - 1)
        slots[slot] = stored + 1
        keys_source[stored], keys_target[stored], keys_code[stored] = left, right, code
        states[stored] = initial_state[index]
        stored += 1
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
                events[filled, 4] = observed + base
                events[filled, 5] = observed + base
                events[filled, 6] = previous
                events[filled, 7] = -1
                events[filled, 8] = -1
                filled += 1
            last[owner] = observed + base
            observed += 1
        for phase in range(2):
            while True:
                if phase == 0:
                    if asserted >= statements or present_step[asserted] != moment:
                        break
                    row, live = asserted, 1
                    left, right = entity[present_source[row]], entity[present_target[row]]
                    code = present_code[row]
                    origin, destination = present_source[row] + base, present_target[row] + base
                else:
                    if retracted >= removals or absent_step[retracted] != moment:
                        break
                    row, live = retracted, 0
                    left, right = absent_source[row], absent_target[row]
                    code = absent_code[row]
                    if last[left] < 0 or last[right] < 0:
                        return (
                            events[:0],
                            np.int64(-1),
                            keys_source[:0],
                            keys_target[:0],
                            keys_code[:0],
                            states[:0],
                        )
                    origin, destination = last[left], last[right]
                slot = np.int64(
                    mixed(mixed(mixed(np.uint64(left)) ^ np.uint64(right)) ^ np.uint64(code))
                    & np.uint64(capacity - 1)
                )
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
                    if (
                        keys_source[held - 1] == left
                        and keys_target[held - 1] == right
                        and keys_code[held - 1] == code
                    ):
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
        for event in range(seen[moment], seen[moment + 1]):
            events[filled, 0] = first[event]
            events[filled, 1] = second[event]
            events[filled, 2] = 2 * moment + 1
            events[filled, 3] = 2
            events[filled, 4] = starts[event, 0] + base
            events[filled, 5] = starts[event, 1] + base
            events[filled, 6] = kind[event]
            events[filled, 7] = task[event]
            events[filled, 8] = -1
            filled += 1
    return (
        events[:filled].copy(),
        np.int64(0),
        keys_source[:stored],
        keys_target[:stored],
        keys_code[:stored],
        states[:stored],
    )


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
def dag_row(
    row,
    source,
    target,
    ticks,
    belongs,
    group_offset,
    group_lo,
    group_hi,
    group_tick,
    entries,
    lags,
    windowed,
    window,
    cap,
    buffer,
    max_pairs,
):
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
                if fill >= max_pairs:
                    raise ValueError("event neighbors exceed the graph resource budget")
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
    buffer[: len(ordered)] = ordered
    return buffer, len(ordered)


@njit(cache=True, parallel=True)
def event_dag(source, target, ticks, member, lags, windowed, window, cap, entities, max_pairs):
    events = len(ticks)
    belongs, group_offset, group_lo, group_hi, group_tick, entries = group_index(
        source, target, ticks, member, entities
    )
    counts = np.zeros(events, np.int64)
    for row in prange(events):
        if member[row]:
            buffer = np.empty(64, np.int64)
            _, counts[row] = dag_row(
                row,
                source,
                target,
                ticks,
                belongs,
                group_offset,
                group_lo,
                group_hi,
                group_tick,
                entries,
                lags,
                windowed,
                window,
                cap,
                buffer,
                max_pairs,
            )
    offsets = np.zeros(events + 1, np.int64)
    for row in range(events):
        offsets[row + 1] = offsets[row] + counts[row]
        if offsets[row + 1] > max_pairs:
            raise ValueError("event graph edges exceed the graph resource budget")
    out_source = np.empty(offsets[events], np.int32)
    out_target = np.empty(offsets[events], np.int32)
    for row in prange(events):
        if member[row] and counts[row] > 0:
            buffer = np.empty(max(64, counts[row]), np.int64)
            buffer, fill = dag_row(
                row,
                source,
                target,
                ticks,
                belongs,
                group_offset,
                group_lo,
                group_hi,
                group_tick,
                entries,
                lags,
                windowed,
                window,
                cap,
                buffer,
                max_pairs,
            )
            for index in range(fill):
                out_source[offsets[row] + index] = buffer[index]
                out_target[offsets[row] + index] = row
    return out_source, out_target


@njit(cache=True, parallel=True)
def by_target(before, after, size):
    rows = len(after)
    offsets = np.zeros(size + 1, np.int64)
    for row in range(rows):
        offsets[after[row] + 1] += 1
    for slot in range(size):
        offsets[slot + 1] += offsets[slot]
    order = np.empty(rows, np.int64)
    fill = offsets[:size].copy()
    for row in range(rows):
        order[fill[after[row]]] = row
        fill[after[row]] += 1
    for slot in prange(size):
        lo, hi = offsets[slot], offsets[slot + 1]
        if hi - lo > 1:
            segment = order[lo:hi]
            order[lo:hi] = segment[np.argsort(before[segment], kind="mergesort")]
    return order


@njit(cache=True, parallel=True)
def check_boundaries(rows, events, member, period, codes):
    problems = 0
    for index in prange(len(rows)):
        source, destination, relation, gap = rows[index, 0], rows[index, 1], rows[index, 2], rows[index, 3]
        shared = False
        for i in range(2):
            for j in range(2):
                shared |= events[source, i] == events[destination, j]
        before, after = period[events[source, 2] // 2], period[events[destination, 2] // 2]
        known = False
        for code in codes:
            known |= relation == code
        if (
            not shared
            or before >= after
            or not member[source]
            or not member[destination]
            or gap != after - before
            or not known
        ):
            problems += 1
    return problems
