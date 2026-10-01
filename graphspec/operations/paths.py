import logging
import time

import numpy as np

from graphspec.budgets import GraphCapacityError
from graphspec.infra.timing import timed

from . import kernels
from .rows import counted, extend, member_end, open_runs, run_state
from .state import EMPTY

LOG = logging.getLogger(__name__)
BLOCK = 1 << 22
UNION = "*"
DIRECTIONS = ("forward", "backward", "both")
FLIPPED = dict(forward="backward", backward="forward", both="both")
AGGREGATES = ("product", "shortest", "widest")
ALONG = dict(paths=np.multiply, product=np.multiply, shortest=np.add, widest=np.minimum)
ACROSS = dict(paths=np.add, product=np.add, shortest=np.minimum, widest=np.maximum)
EMPTIED = dict(paths=0.0, product=0.0, shortest=np.inf, widest=-np.inf)
SERIES = ("geometric", "poisson")


@timed("paths.parsed")
def parsed(path):
    steps = []
    for text in path.split(","):
        parts = text.split(":")
        if len(parts) not in (2, 3) or not parts[0] or parts[1] not in DIRECTIONS or (len(parts) == 3 and not parts[2]):
            raise ValueError(
                "a path is its steps joined by commas, each a relation or %s, a direction among %s and, where "
                "the path holds values, a column; found %r" % (UNION, ", ".join(DIRECTIONS), path)
            )
        steps.append((parts[0], parts[1], parts[2] if len(parts) == 3 else None))
    return tuple(steps)


def written(steps):
    return ",".join(":".join(part for part in step if part is not None) for step in steps)


@timed("paths.reversed")
def reversed_path(steps):
    return tuple((table, FLIPPED[direction], column) for table, direction, column in steps[::-1])


@timed("paths.symmetric")
def symmetric(steps):
    return reversed_path(steps) == tuple(steps)


@timed("paths.valued")
def valued(steps):
    return all(column is not None for _, _, column in steps)


@timed("paths.kinds")
def kinds(registry, table, direction):
    if table == UNION:
        pairs = {pair for group in registry.groups for pair in registry.named[group].pairs}
        return pairs | {(second, first) for first, second in pairs}
    edge = registry.named[table]
    pairs, flipped = set(edge.pairs), {(second, first) for first, second in edge.pairs}
    if edge.symmetric or direction == "both":
        return pairs | flipped
    return pairs if direction == "forward" else flipped


@timed("paths.composed")
def composed(left, right):
    return {(first, last) for first, middle in left for joined, last in right if middle == joined}


@timed("paths.joined")
def joined(registry, steps, length):
    found = kinds(registry, *steps[0][:2])
    for table, direction, _ in steps[1:]:
        found = composed(found, kinds(registry, table, direction))
    power, every = found, set(found)
    for _ in range(length - 1):
        power = composed(power, found)
        every |= power
    return tuple(sorted(every))


@timed("paths.chains")
def chains(registry, steps):
    found = set(joined(registry, steps, 1))
    return bool(composed(found, found))


def offered(registry):
    started = time.perf_counter()
    if UNION in registry.groups:
        raise ValueError(
            "the relation %s of the bank %s has the name a path gives the union of every relation"
            % (UNION, registry.name)
        )
    if not registry.groups:
        LOG.debug("paths offered exit paths=0 seconds=%.4f", time.perf_counter() - started)
        return ()
    plain, priced = [], []
    for group in registry.groups:
        edge = registry.named[group]
        for direction in ("both",) if edge.symmetric else DIRECTIONS:
            plain.append((group, direction, None))
            priced += [(group, direction, column) for column in registry.ordered(group)]
    result = [((UNION, "both", None),)] + [(step,) for step in plain + priced]
    for held in (plain, priced):
        for first in held:
            for second in held:
                if composed(kinds(registry, *first[:2]), kinds(registry, *second[:2])):
                    result.append((first, second))
    result = tuple(written(steps) for steps in result)
    LOG.debug("paths offered exit paths=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("paths.recency")
def recency(graph, carry, name, stated):
    result = np.zeros(len(graph.fact_step), np.int64)
    rows = np.flatnonzero(stated)
    code, source, target = graph.fact_code[rows], graph.fact_source[rows], graph.fact_target[rows]
    ids, count = kernels.keyed(code, source, target)
    (_, result[rows]), piece = run_state(
        carry, name, graph, ids, count, graph.fact_step[rows], carry.keys(code, source, target)
    )
    open_runs(carry, name, [piece])
    return result


@timed("paths.oriented")
def oriented(source, target, direction):
    rows = np.arange(len(source))
    if direction == "forward":
        return source, target, rows
    if direction == "backward":
        return target, source, rows
    return np.r_[source, target], np.r_[target, source], np.r_[rows, rows]


@timed("paths.latest")
def latest(middle, other, age, tokens, cap):
    if cap == 0 or not len(middle):
        return np.ones(len(middle), bool)
    order = np.lexsort((tokens[other], age, middle))[::-1]
    ids, count = kernels.keyed(middle[order], other[order], np.zeros(len(order), np.int64))
    first = np.flatnonzero(kernels.first_rows(ids, count))
    owners = middle[order][first]
    starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
    rank = np.arange(len(first)) - np.repeat(starts, np.diff(np.r_[starts, len(first)]))
    kept = np.zeros(count, bool)
    kept[ids[first[rank < cap]]] = True
    result = np.empty(len(middle), bool)
    result[order] = kept[ids]
    return result


@timed("paths.walks")
def walks(first, second, entities, along, budget, operation):
    origin, middle, held = first
    joined_at, reached, given = second
    order = np.argsort(joined_at, kind="stable")
    joined_at, reached = joined_at[order], reached[order]
    starts = np.searchsorted(joined_at, np.arange(entities + 1))
    counts = starts[np.asarray(middle, np.int64) + 1] - starts[np.asarray(middle, np.int64)]
    total = int(counts.sum())
    budget.pairs(total, operation)
    budget.workspace(total * 24, operation)
    taken = np.repeat(starts[middle], counts) + counted(counts)
    values = None
    if held is not None:
        values = ALONG[along](np.repeat(held, counts), np.asarray(given)[order][taken])
    return np.repeat(origin, counts), reached[taken], values


@timed("paths.reduced")
def reduced(source, target, values, across, length=None):
    ids, count = kernels.keyed(source, target, np.zeros(len(source), np.int64))
    first = np.flatnonzero(kernels.first_rows(ids, count))
    total = None
    if values is not None:
        total = np.full(count, EMPTIED[across])
        ACROSS[across].at(total, ids, values)
        total = total[ids[first]]
    shortest = None
    if length is not None:
        shortest = np.full(count, np.iinfo(np.int64).max)
        np.minimum.at(shortest, ids, length)
        shortest = shortest[ids[first]]
    return source[first], target[first], total, shortest


@timed("paths.once")
def once(source, target, values, aggregate):
    if aggregate == "paths":
        source, target, _, _ = reduced(source, target, None, aggregate)
        return source, target, np.ones(len(source))
    return reduced(source, target, values, aggregate)[:3]


@timed("paths.product")
def product(steps, entities, aggregate, budget, operation):
    source, target, values = once(*steps[0], aggregate)
    for step in steps[1:]:
        walked = walks((source, target, values), once(*step, aggregate), entities, aggregate, budget, operation)
        source, target, values, _ = reduced(*walked, aggregate)
    kept = source != target
    return source[kept], target[kept], None if values is None else values[kept]


@timed("paths.powers")
def powers(source, target, values, limit, entities, aggregate, budget, operation):
    base = (source, target, values)
    every = [(source, target, values, np.ones(len(source), np.int64))]
    power = base
    for distance in range(2, limit + 1):
        walked = walks(power, base, entities, aggregate, budget, operation)
        power = reduced(*walked, aggregate)[:3]
        every.append(power + (np.full(len(power[0]), distance, np.int64),))
    source, target, values, length = (np.concatenate([held[place] for held in every]) for place in range(4))
    source, target, values, length = reduced(source, target, values, aggregate, length)
    kept = source != target
    budget.retain(int(kept.sum()), operation)
    return source[kept], target[kept], values[kept], length[kept]


def widen(slots, keys_low, keys_high, count, extra, budget, operation):
    budget.workspace(128 * (count + extra), operation + " index")
    if count + extra > len(keys_low):
        size = max(2 * len(keys_low), count + extra, 16)
        grown_low, grown_high = np.empty(size, np.int64), np.empty(size, np.int64)
        grown_low[:count], grown_high[:count] = keys_low[:count], keys_high[:count]
        keys_low, keys_high = grown_low, grown_high
    capacity = max(len(slots), 16)
    while capacity < 2 * (count + extra):
        capacity *= 2
    if capacity != len(slots):
        slots = np.zeros(capacity, np.int64)
        kernels.rehash(slots, keys_low, keys_high, count)
    return slots, keys_low, keys_high


def closure(u, v, limit, entities, budget, both, operation):
    started = time.perf_counter()
    if both:
        source, target = np.concatenate((u, v)).astype(np.int64), np.concatenate((v, u)).astype(np.int64)
    else:
        source, target = np.asarray(u, np.int64), np.asarray(v, np.int64)
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    starts = np.searchsorted(source, np.arange(entities + 1))
    slots, keys_low, keys_high = widen(
        np.zeros(16, np.int64), np.zeros(16, np.int64), np.zeros(16, np.int64), 0, len(source), budget, operation
    )
    first, second = (np.minimum(source, target), np.maximum(source, target)) if both else (source, target)
    count, _ = kernels.novel(slots, keys_low, keys_high, 0, first, second)
    left, middle = source, target
    result = []
    for distance in range(2, limit + 1):
        counts = starts[middle + 1] - starts[middle]
        edges = np.cumsum(counts)
        budget.workspace(int(edges[-1]) * 32 if len(edges) else 0, operation + " candidates")
        pieces, lo = [], 0
        while lo < len(counts):
            hi = max(int(np.searchsorted(edges, (edges[lo - 1] if lo else 0) + BLOCK, "right")), lo + 1)
            block = counts[lo:hi]
            reached = target[np.repeat(starts[middle[lo:hi]], block) + counted(block)]
            origin = np.repeat(left[lo:hi], block)
            use = origin != reached
            if both:
                low, high = np.minimum(origin[use], reached[use]), np.maximum(origin[use], reached[use])
            else:
                low, high = origin[use], reached[use]
            budget.pairs(len(low), operation + " candidates")
            slots, keys_low, keys_high = widen(slots, keys_low, keys_high, count, len(low), budget, operation)
            count, fresh = kernels.novel(slots, keys_low, keys_high, count, low, high)
            budget.retain(int(fresh.sum()), operation)
            pieces.append((low[fresh], high[fresh]))
            lo = hi
        a = np.concatenate([piece[0] for piece in pieces] + [np.zeros(0, np.int64)])
        b = np.concatenate([piece[1] for piece in pieces] + [np.zeros(0, np.int64)])
        result.append((a.astype(np.int32), b.astype(np.int32)))
        left, middle = (np.concatenate((a, b)), np.concatenate((b, a))) if both else (a, b)
    LOG.debug(
        "closure exit limit=%d pairs=%d seconds=%.4f",
        limit,
        sum(len(pair[0]) for pair in result),
        time.perf_counter() - started,
    )
    return result


@timed("paths.summed")
def summed(adjacency, series, parameter, device):
    from graphspec.infra import compute

    count = len(adjacency)
    scale = 1 / np.sqrt(adjacency.sum(1))
    transition = scale[:, None] * adjacency * scale[None, :]
    if series == "geometric":
        return parameter * compute.inverse(np.eye(count) - (1 - parameter) * transition, device)
    values, vectors = compute.eigh(transition, device)
    return compute.compose(vectors, np.exp(-parameter * (1 - values)), device)


def unlived():
    return dict(key=EMPTY, source=EMPTY, target=EMPTY, code=EMPTY)


@timed("paths.widened")
def widened(graph, carry, name, found):
    live = carry.state[name]
    graph, position = carry.widened(graph, np.r_[live["source"], live["target"]])
    source, target = found[:2]
    return graph, (position[source].astype(source.dtype), position[target].astype(target.dtype)) + found[2:]


@timed("paths.handoff")
def handoff(carry, name, found, steps, connected):
    source, target, code, step = found[:4]
    live = carry.state[name]
    stop = (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.int64),)
    if len(live["source"]):
        begun = carry.keys(code[step == 0], source[step == 0], target[step == 0])
        ended = ~np.isin(live["key"], begun)
        stop = (
            carry.local(live["source"][ended]).astype(np.int32),
            carry.local(live["target"][ended]).astype(np.int32),
            live["code"][ended],
        )
    final = np.flatnonzero((step == steps - 1) & connected[source] & connected[target])
    final = final[np.lexsort((target[final], source[final], code[final]))]
    carry.state[name] = dict(
        key=carry.keys(code[final], source[final], target[final]),
        source=carry.glob[source[final]],
        target=carry.glob[target[final]],
        code=code[final],
    )
    return stop


@timed("paths.assembled")
def assembled(graph, found, stop, budget):
    source, target, code, step, reference = found
    entities = len(graph.kinds)
    if (int(code.max(initial=0)) + 1) * entities * entities >= 1 << 63:
        raise GraphCapacityError("derived fact keys exceed 63 bits for %d entities" % entities)
    if np.any(np.diff(step) < 0):
        by_step = np.argsort(step, kind="stable")
        source, target, code, step, reference = (held[by_step] for held in (source, target, code, step, reference))
    pair = (code.astype(np.int64) * entities + source) * entities + target
    bounds = np.flatnonzero(np.r_[True, np.diff(step) != 0, True])
    order = np.empty(len(step), np.int64)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        order[lo:hi] = lo + np.argsort(pair[lo:hi])
    source, target, code, step, reference = (held[order] for held in (source, target, code, step, reference))
    ids, count = kernels.keyed(code, source, target)
    continues = kernels.continuing(ids, step.astype(np.int64), count)
    limit = np.where(
        graph.entity_connected[source] & graph.entity_connected[target],
        graph.steps,
        member_end(graph)[step],
    )
    ends = np.flatnonzero(~continues & (step + 1 < limit))
    total = len(step) + len(ends) + len(stop[0])
    budget.workspace(total * 32 + (len(ends) + len(stop[0])) * budget.fact_bytes * 2, "derived fact assembly")
    return extend(
        graph,
        np.concatenate((source, source[ends], stop[0])),
        np.concatenate((target, target[ends], stop[1])),
        np.concatenate((code, code[ends], stop[2])),
        np.concatenate((step, step[ends] + 1, np.zeros(len(stop[0]), np.int32))),
        np.concatenate((np.ones(len(step), bool), np.zeros(len(ends) + len(stop[0]), bool))),
        np.concatenate((reference, np.full(len(ends) + len(stop[0]), -1, np.int64))),
    )
