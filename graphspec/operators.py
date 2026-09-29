import numpy as np

from graphspec.infra import compute
from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.infra.workspace import allocate, concatenate, scatter, take

from . import kernels
from .carry import FOREVER, Table
from .resources import GraphBudget, GraphCapacityError
from .vocabulary import UNTYPED, completion, hop

FACTS = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
OBSERVATIONS = ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
SAMPLE, DROP, RANDOM, REWIRE = 1, 2, 3, 4
BLOCK = 1 << 22


def counted(counts):
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


def enabled(spec, name):
    return name in spec and spec[name]


def codes(registry, names):
    return [registry.code(name) for name in names]


def fact_bytes(graph):
    return sum(graph.arrays[name].dtype.itemsize for name in FACTS)


def select(graph, names, mask):
    return graph.replace(**{name: take(graph.arrays[name], mask) for name in names})


def extend(graph, source, target, code, step, present, references):
    added = dict(
        fact_source=source,
        fact_target=target,
        fact_code=code,
        fact_step=step,
        fact_present=present,
        fact_ref=references,
    )
    added = {name: np.asarray(value, graph.arrays[name].dtype) for name, value in added.items()}
    order = np.argsort(added["fact_step"].astype(np.int64) * 2 + ~added["fact_present"], kind="stable")
    key = graph.fact_step.astype(np.int64) * 2 + ~graph.fact_present
    position = np.searchsorted(
        key, added["fact_step"][order].astype(np.int64) * 2 + ~added["fact_present"][order], "right"
    ) + np.arange(len(order))
    total = len(key) + len(order)
    existing = np.ones(total, bool)
    existing[position] = False
    merged = {}
    for name in FACTS:
        column = allocate((total,) + graph.arrays[name].shape[1:], graph.arrays[name].dtype)
        scatter(column, existing, graph.arrays[name])
        scatter(column, position, added[name], order)
        merged[name] = column
    return graph.replace(**merged)


def identities(carry, entity):
    return entity if carry is None else carry.glob[entity]


@timed("operators.prune")
def prune(graph, kept, carry=None, name=None):
    if carry is None:
        if kept.all():
            return graph
        ids, count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
        return select(graph, FACTS, kernels.orphans(ids, graph.fact_present, kept, count))
    return select(
        graph,
        FACTS,
        prune_mask(
            graph.fact_source,
            graph.fact_target,
            graph.fact_code,
            graph.fact_present,
            kept,
            graph.entity_connected,
            carry,
            name,
        ),
    )


def prune_mask(fact_source, fact_target, fact_code, fact_present, kept, connected, carry, name):
    table = carry.table("prune/" + name)
    keys = carry.keys(fact_code, fact_source, fact_target)
    state = table.lookup("state", keys, 0)
    carried = np.flatnonzero(state > 0)
    first = carried[np.unique(keys[carried], return_index=True)[1]]
    source, target, code = (np.r_[column[first], column] for column in (fact_source, fact_target, fact_code))
    present = np.r_[np.ones(len(first), bool), fact_present]
    everything = np.r_[state[first] == 2, kept]
    ids, count = kernels.keyed(source, target, code)
    result = kernels.orphans(ids, present, everything, count)[len(first) :]
    stated = np.bincount(ids[present], minlength=count) > 0
    surviving = np.bincount(ids[present & everything], minlength=count) > 0
    group_key = np.zeros(count, np.int64)
    group_key[ids] = np.r_[keys[first], keys]
    lasting = np.zeros(count, bool)
    lasting[ids[connected[source] & connected[target]]] = True
    chosen = lasting & stated
    carry.state["prune/" + name] = table.replaced(
        group_key[chosen], group_key[chosen], state=np.where(surviving[chosen], 2, 1).astype(np.int8)
    )
    return result


def run_state(carry, name, graph, ids, count, step, keys):
    last, current, begun = np.full(count, -2, np.int64), np.zeros(count, np.int64), np.zeros(count, np.int64)
    if carry is None:
        return kernels.runs(ids, step, last, current, begun), None
    group_key = np.zeros(count, np.int64)
    group_key[ids] = keys
    table = carry.table(name)
    found = table.find(group_key)
    use = found >= 0
    if use.any():
        last[use] = table.columns["last"][found[use]] - carry.step0
        current[use] = table.columns["current"][found[use]]
        begun[use] = table.columns["begun"][found[use]] - carry.step0
    result = kernels.runs(ids, step, last, current, begun)
    active = last == graph.steps - 1
    return result, (
        group_key[active],
        last[active] + carry.step0,
        current[active],
        begun[active] + carry.step0,
    )


def open_runs(carry, name, pieces):
    pieces = [piece for piece in pieces if piece is not None]
    if carry is None:
        return
    carry.state[name] = Table(
        np.concatenate([piece[0] for piece in pieces] + [np.zeros(0, np.int64)]),
        **{
            column: np.concatenate([piece[index] for piece in pieces] + [np.zeros(0, np.int64)])
            for index, column in enumerate(("last", "current", "begun"), 1)
        },
    )


def member_end(graph):
    member = graph.step_graph
    first = np.r_[0, np.flatnonzero(np.diff(member)) + 1]
    return np.r_[first[1:], graph.steps][np.cumsum(np.r_[0, np.diff(member) != 0])]


def reach(graph, spec, start, lasting):
    if spec["time/scheme"] == "state":
        reached = np.full(len(start), graph.steps - 1, np.int64)
    elif spec.get("time/unit") == "period":
        group = np.cumsum(np.r_[0, np.diff(graph.period) != 0])
        reached = np.searchsorted(group, group[start] + spec["time/past_periods"], "right") - 1
    else:
        reached = start + spec["time/window"] - 1
    return np.where(lasting, reached, np.minimum(reached, member_end(graph)[start] - 1))


def groups_of(graph):
    return np.cumsum(np.r_[0, np.diff(graph.period) != 0])


def horizon(graph, spec, carry, start):
    if spec["time/scheme"] == "state":
        return np.full(len(start), FOREVER, np.int64)
    if spec.get("time/unit") == "period":
        return carry.group0 + groups_of(graph)[start] + spec["time/past_periods"]
    return carry.step0 + np.asarray(start, np.int64) + spec["time/window"] - 1


def beyond(graph, spec, carry, end):
    if spec["time/scheme"] == "state":
        return np.ones(len(end), bool)
    if spec.get("time/unit") == "period":
        return end > carry.group0 + groups_of(graph)[-1]
    return end >= carry.step0 + graph.steps


def carried_reach(graph, spec, carry, end):
    if spec["time/scheme"] == "state":
        return np.full(len(end), graph.steps - 1, np.int64)
    if spec.get("time/unit") == "period":
        return np.searchsorted(groups_of(graph), end - carry.group0, "right") - 1
    return end - carry.step0


def held(graph, carry):
    items = carry.state.get("window")
    if items is None:
        return graph, None, None
    obs, facts = items["obs"], items["facts"]
    added_obs, added_facts = len(obs["entity"]), len(facts["source"])
    tables = len(graph.x_table), len(graph.value_table)
    carry.x_global = np.r_[carry.x_global, obs["ref"]]
    carry.value_global = np.r_[carry.value_global, facts["ref"]]

    def prefixed(name, values):
        return np.r_[np.asarray(values), graph.arrays[name]].astype(graph.arrays[name].dtype)

    graph = graph.replace(
        x_table=np.concatenate((graph.x_table, obs["x"].astype(graph.x_table.dtype))),
        value_table=np.concatenate((graph.value_table, facts["values"].astype(graph.value_table.dtype))),
        obs_entity=prefixed("obs_entity", carry.local(obs["entity"])),
        obs_step=prefixed("obs_step", np.full(added_obs, -1)),
        obs_ref=prefixed("obs_ref", tables[0] + np.arange(added_obs)),
        obs_anchor=prefixed("obs_anchor", obs["anchor"]),
        fact_source=prefixed("fact_source", carry.local(facts["source"])),
        fact_target=prefixed("fact_target", carry.local(facts["target"])),
        fact_code=prefixed("fact_code", facts["code"]),
        fact_step=prefixed("fact_step", np.full(added_facts, -1)),
        fact_present=prefixed("fact_present", np.ones(added_facts, bool)),
        fact_ref=prefixed("fact_ref", np.where(facts["valued"], tables[1] + np.arange(added_facts), -1)),
    )
    return graph, obs, facts


@timed("operators.window")
def window(graph, registry, spec, seed, limits, carry=None):
    if (
        spec["time/scheme"] == "window"
        and spec.get("time/unit", "step") == "step"
        and spec["time/window"] == 1
    ):
        return graph
    steps = graph.steps
    carried_obs = carried_facts = None
    if carry is not None:
        graph, carried_obs, carried_facts = held(graph, carry)
    pseudo_obs = 0 if carried_obs is None else len(carried_obs["entity"])
    pseudo_facts = 0 if carried_facts is None else len(carried_facts["source"])
    order = np.lexsort((graph.obs_step, graph.obs_entity))
    entity, step = graph.obs_entity[order], graph.obs_step[order].astype(np.int64)
    later = np.r_[entity[1:] == entity[:-1], False]
    limit = np.where(later, np.r_[step[1:], 0], steps)
    reached = reach(graph, spec, np.maximum(step, 0), graph.entity_connected[entity])
    pseudo = order < pseudo_obs
    if pseudo.any():
        reached[pseudo] = carried_reach(graph, spec, carry, carried_obs["end"][order[pseudo]])
    counts = np.maximum(np.minimum(reached, limit - 1) - step, 0)
    if carry is not None:
        rows = order[~later & graph.entity_connected[entity]]
        end = horizon(graph, spec, carry, np.maximum(graph.obs_step[rows], 0))
        inherited = rows < pseudo_obs
        if inherited.any():
            end[inherited] = carried_obs["end"][rows[inherited]]
        rows, end = rows[beyond(graph, spec, carry, end)], end[beyond(graph, spec, carry, end)]
        kept_obs = dict(
            entity=carry.glob[graph.obs_entity[rows]],
            end=end,
            x=np.asarray(graph.x_table)[graph.obs_ref[rows]],
            ref=carry.x_global[graph.obs_ref[rows]],
            anchor=graph.obs_anchor[rows],
        )
    budget = GraphBudget(limits, fact_bytes(graph))
    budget.pairs(int(counts.sum()), "time window observations")
    budget.workspace(
        int(counts.sum()) * sum(graph.arrays[name].dtype.itemsize + 8 for name in OBSERVATIONS),
        "time window observations",
    )
    origin = np.repeat(order, counts)
    copies = {
        name: concatenate((graph.arrays[name], take(graph.arrays[name], origin))) for name in OBSERVATIONS
    }
    copies["obs_step"][len(graph.obs_step) :] += (counted(counts) + 1).astype(np.int32)
    arrangement = np.lexsort((np.arange(len(copies["obs_step"])), copies["obs_step"]))
    graph = graph.replace(**{name: take(value, arrangement)[pseudo_obs:] for name, value in copies.items()})
    ids, count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
    limit = kernels.carried(ids, graph.fact_step, graph.fact_present, count, steps)
    present = np.flatnonzero(graph.fact_present)
    start = graph.fact_step[present].astype(np.int64)
    lasting = (
        graph.entity_connected[graph.fact_source[present]]
        & graph.entity_connected[graph.fact_target[present]]
    )
    reached = reach(graph, spec, np.maximum(start, 0), lasting)
    if pseudo_facts:
        reached[:pseudo_facts] = carried_reach(graph, spec, carry, carried_facts["end"])
    counts = np.maximum(np.minimum(reached, limit[present] - 1) - start, 0)
    if carry is not None:
        final = lasting & (limit[present] == steps)
        rows = present[final]
        end = horizon(graph, spec, carry, np.maximum(start[final], 0))
        stamp, rank = carry.step0 + start[final], rows.astype(np.int64)
        inherited = rows < pseudo_facts
        if inherited.any():
            for values, source in ((end, "end"), (stamp, "stamp"), (rank, "rank")):
                values[inherited] = carried_facts[source][rows[inherited]]
        wanted = beyond(graph, spec, carry, end)
        rows, end, stamp, rank = rows[wanted], end[wanted], stamp[wanted], rank[wanted]
        arranged = np.lexsort((rank, stamp))
        rows, end, stamp, rank = rows[arranged], end[arranged], stamp[arranged], rank[arranged]
        reference = graph.fact_ref[rows]
        values = np.full((len(rows), graph.value_table.shape[1]), np.nan, graph.value_table.dtype)
        values[reference >= 0] = graph.value_table[reference[reference >= 0]]
        carry.state["window"] = dict(
            obs=kept_obs,
            facts=dict(
                source=carry.glob[graph.fact_source[rows]],
                target=carry.glob[graph.fact_target[rows]],
                code=graph.fact_code[rows],
                end=end,
                stamp=stamp,
                rank=rank,
                values=values,
                valued=reference >= 0,
                ref=np.where(reference >= 0, carry.value_global[np.maximum(reference, 0)], -1),
            ),
        )
    budget.pairs(int(counts.sum()), "time window facts")
    origin = np.repeat(present, counts)
    graph = extend(
        graph,
        graph.fact_source[origin],
        graph.fact_target[origin],
        graph.fact_code[origin],
        (graph.fact_step[origin].astype(np.int64) + counted(counts) + 1).astype(np.int32),
        np.ones(len(origin), bool),
        graph.fact_ref[origin],
    )
    if pseudo_facts:
        graph = graph.replace(**{name: graph.arrays[name][pseudo_facts:] for name in FACTS})
    return graph


@timed("operators.periods")
def periods(graph, registry, spec, seed, carry=None):
    if spec["time/periods"] != "every":
        return graph
    length = spec["time/period_length"]
    joined = graph.step_graph
    first = np.r_[0, np.flatnonzero(np.diff(joined)) + 1]
    sizes = np.diff(np.r_[first, graph.steps])
    offsets = np.r_[0, np.cumsum(-(-sizes // length))][:-1]
    member = np.repeat(np.arange(len(first)), sizes)
    position = np.arange(graph.steps) - first[member]
    base = 0 if carry is None else carry.period0
    return graph.replace(period=(base + offsets[member] + position // length).astype(np.int32))


def counters(carry, name, graph):
    count = np.zeros(len(graph.kinds), np.int64)
    if carry is not None:
        count[:] = carry.table(name).lookup("value", carry.glob, 0)
    return count


def remember(carry, name, graph, values):
    if carry is not None:
        chosen = np.flatnonzero(graph.entity_connected)
        keys = carry.glob[chosen]
        carry.state[name] = carry.table(name).replaced(keys, keys, value=values[chosen])


@timed("operators.nodes")
def nodes(graph, registry, spec, seed, carry=None):
    steps, entities = graph.steps, len(graph.kinds)
    protected = registry.extras["protected"]
    retained = np.asarray([kind in protected or spec["nodes/%s/keep" % kind] for kind in registry.nodes])
    filters = {
        kind: spec["nodes/%s/filter" % kind]
        for kind in registry.nodes
        if spec.get("nodes/%s/filter" % kind, "none") != "none"
    }
    if retained.all() and not filters and not graph.obs_anchor.any():
        return graph
    allowed = retained[graph.kinds]
    obs_entity, obs_step = graph.obs_entity, graph.obs_step
    use = graph.fact_present & allowed[graph.fact_source] & allowed[graph.fact_target]
    kinds = graph.kinds[obs_entity]
    passed, minimum, counts = np.ones(len(obs_step), bool), np.zeros(len(obs_step), np.int64), {}
    for kind, mode in filters.items():
        prefix, members = "nodes/%s/" % kind, kinds == registry.kind(kind)
        if mode in ("degree", "total_degree"):
            if mode not in counts:
                total = mode == "total_degree"
                count = counters(carry if total else None, "nodes/total_degree", graph)
                counts[mode] = kernels.degrees(
                    obs_entity,
                    obs_step,
                    graph.fact_source,
                    graph.fact_target,
                    graph.fact_step,
                    use,
                    entities,
                    total,
                    count,
                )
                if total:
                    remember(carry, "nodes/total_degree", graph, count)
            passed[members] = counts[mode][members] >= spec[prefix + "min_degree"]
        elif mode == "activity":
            if mode not in counts:
                count = counters(carry, "nodes/activity", graph)
                counts[mode] = kernels.activity(obs_entity, count)
                remember(carry, "nodes/activity", graph, count)
            passed[members] = counts[mode][members] >= spec[prefix + "min_activity"]
        elif mode == "sample":
            passed[members] = (
                unit(seed, SAMPLE, identities(carry, obs_entity[members])) < spec[prefix + "sample"]
            )
        else:
            minimum[members] = spec[prefix + "core"]
    structural = allowed[obs_entity] & passed
    if minimum.any():
        structural = kernels.peel(
            obs_entity,
            obs_step,
            structural,
            minimum,
            graph.fact_source,
            graph.fact_target,
            graph.fact_step,
            use,
            entities,
        )
    needed = np.zeros(len(obs_step), bool)
    if len(graph.query_key):
        needed = (graph.query_references[obs_step] == obs_entity[:, None]).any(1)
    first, _ = kernels.first_observed(obs_entity, obs_step, structural, entities, steps)
    if carry is not None:
        first[counters(carry, "nodes/structural", graph) > 0] = -1
        remember(carry, "nodes/structural", graph, (first < steps).astype(np.int64))
    present = kernels.observed_at(
        obs_entity, obs_step, structural, graph.fact_source, graph.fact_step, entities
    ) & kernels.observed_at(obs_entity, obs_step, structural, graph.fact_target, graph.fact_step, entities)
    absent = (first[graph.fact_source] <= graph.fact_step) & (first[graph.fact_target] <= graph.fact_step)
    alive = structural | needed
    graph = select(graph, OBSERVATIONS, alive).replace(obs_anchor=~structural[alive])
    return prune(graph, np.where(graph.fact_present, present, absent), carry, "nodes")


def undirected(source, target, entities):
    use = source != target
    key = np.unique(
        np.minimum(source[use], target[use]).astype(np.int64) * entities
        + np.maximum(source[use], target[use])
    )
    return (key // entities).astype(np.int32), (key % entities).astype(np.int32)


def unique_pairs(chunks):
    source = np.concatenate([chunk[0] for chunk in chunks] + [np.zeros(0, np.int32)])
    target = np.concatenate([chunk[1] for chunk in chunks] + [np.zeros(0, np.int32)])
    ids, count = kernels.keyed(source, target, np.zeros(len(source), np.int64))
    fresh = kernels.first_rows(ids, count)
    return source[fresh], target[fresh]


def recalled(memo, name, inputs, budget, compute):
    held = memo.get(name)
    if held is not None and all(np.array_equal(old, new) for old, new in zip(held[0], inputs)):
        budget.retain(held[2], name)
        return held[1]
    before = budget.facts
    result = compute()
    memo[name] = (inputs, result, budget.facts - before)
    return result


@timed("operators.project")
def project(graph, registry, edge, source, target, code, recency, cap, budget, memo):
    relation, side, first, second = edge.origin[1:]
    use = np.flatnonzero(code == registry.code(relation))
    members, owners = (source[use], target[use]) if side == "sources" else (target[use], source[use])
    ages = recency[use]

    def compute():
        order = np.lexsort((np.arange(len(use)), ages, owners))
        pairs = kernels.project_pairs(
            members[order],
            owners[order],
            graph.kinds,
            registry.kind(first),
            registry.kind(second),
            cap,
            budget.pair_limit,
        )
        return unique_pairs([pairs])

    return recalled(memo, edge.name, (members, owners, ages), budget, compute)


@timed("operators.complete")
def complete(graph, registry, selected, within, present, u, v, budget):
    members = present[np.isin(graph.kinds[present], selected)]
    if not len(within):
        flat, offsets = members, np.asarray([0, len(members)], np.int64)
    else:
        anchors, linked = np.r_[u, v], np.r_[v, u]
        use = np.isin(graph.kinds[anchors], within) & np.isin(linked, members)
        anchors, linked = anchors[use], linked[use]
        order = np.argsort(anchors, kind="stable")
        anchors, flat = anchors[order], linked[order]
        offsets = (
            np.r_[0, np.flatnonzero(np.diff(anchors)) + 1, len(anchors)]
            if len(order)
            else np.zeros(1, np.int64)
        )
    source, target = unique_pairs(
        [kernels.within_pairs(flat, offsets, graph.kinds, len(registry.nodes), budget.pair_limit)]
    )
    table = np.full((len(registry.nodes), len(registry.nodes)), -1, np.int64)
    for i, left in enumerate(registry.nodes):
        for right in registry.nodes[i:]:
            table[i, registry.kind(right)] = registry.code(completion(left, right))
    return source, target, table[graph.kinds[source], graph.kinds[target]]


def widen(slots, keys_low, keys_high, count, extra, budget):
    budget.workspace(128 * (count + extra), "hop closure index")
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


@timed("operators.closure")
def closure(registry, u, v, limit, entities, budget):
    source, target = np.r_[u, v].astype(np.int64), np.r_[v, u].astype(np.int64)
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    starts = np.searchsorted(source, np.arange(entities + 1))
    slots, keys_low, keys_high = widen(
        np.zeros(16, np.int64), np.zeros(16, np.int64), np.zeros(16, np.int64), 0, len(source), budget
    )
    count, _ = kernels.novel(
        slots, keys_low, keys_high, 0, np.minimum(source, target), np.maximum(source, target)
    )
    left, middle = source, target
    result = []
    for distance in range(2, limit + 1):
        counts = starts[middle + 1] - starts[middle]
        edges = np.cumsum(counts)
        budget.workspace(int(edges[-1]) * 32 if len(edges) else 0, "hop closure candidates")
        pieces, lo = [], 0
        while lo < len(counts):
            hi = max(int(np.searchsorted(edges, (edges[lo - 1] if lo else 0) + BLOCK, "right")), lo + 1)
            block = counts[lo:hi]
            reached = target[np.repeat(starts[middle[lo:hi]], block) + counted(block)]
            origin = np.repeat(left[lo:hi], block)
            use = origin != reached
            low, high = np.minimum(origin[use], reached[use]), np.maximum(origin[use], reached[use])
            budget.pairs(len(low), "hop closure candidates")
            slots, keys_low, keys_high = widen(slots, keys_low, keys_high, count, len(low), budget)
            count, fresh = kernels.novel(slots, keys_low, keys_high, count, low, high)
            budget.retain(int(fresh.sum()), "hop closure")
            pieces.append((low[fresh], high[fresh]))
            lo = hi
        a = np.concatenate([piece[0] for piece in pieces] + [np.zeros(0, np.int64)])
        b = np.concatenate([piece[1] for piece in pieces] + [np.zeros(0, np.int64)])
        result.append(
            (a.astype(np.int32), b.astype(np.int32), np.full(len(a), registry.code(hop(distance)), np.int64))
        )
        left, middle = np.r_[a, b], np.r_[b, a]
    return result


def dense(members, u, v):
    index = np.full(int(max(members.max(initial=-1), u.max(initial=-1), v.max(initial=-1))) + 1, -1, np.int64)
    index[members] = np.arange(len(members))
    use = (index[u] >= 0) & (index[v] >= 0)
    adjacency = np.zeros((len(members), len(members)), bool)
    adjacency[index[u[use]], index[v[use]]] = True
    adjacency[index[v[use]], index[u[use]]] = True
    return adjacency


def blocks(labels):
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(labels[order]) != 0, True])
    return [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:]) if hi - lo > 1]


def diffusion_scores(adjacency, spec, device):
    count = len(adjacency)
    scale = 1 / np.sqrt(adjacency.sum(1))
    transition = scale[:, None] * adjacency * scale[None, :]
    if spec["derive/diffusion"] == "ppr":
        alpha = spec["derive/ppr_alpha"]
        scores = alpha * compute.inverse(np.eye(count) - (1 - alpha) * transition, device)
    else:
        values, vectors = compute.eigh(transition, device)
        scores = compute.compose(vectors, np.exp(-spec["derive/heat_time"] * (1 - values)), device)
    scores[adjacency] = -np.inf
    np.fill_diagonal(scores, -np.inf)
    return scores


@timed("operators.diffuse")
def diffuse(registry, spec, present, u, v, entities, budget):
    count = len(present)
    if count < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    sources, targets, retained = [], [], 0
    for members, source, target in component_edges(present, u, v):
        budget.dense(len(members), "diffusion")
        budget.pairs(
            retained + len(members) * min(spec["derive/diffusion_top_k"], len(members) - 1),
            "diffusion",
        )
        adjacency = np.zeros((len(members), len(members)), bool)
        adjacency[source, target] = adjacency[target, source] = True
        scores = diffusion_scores(adjacency, spec, budget.limits.device)
        width = min(spec["derive/diffusion_top_k"], len(members) - 1)
        chosen, ranked = compute.top(scores, width, budget.limits.device)
        valid = np.isfinite(ranked)
        rows = np.broadcast_to(np.arange(len(members))[:, None], chosen.shape)
        sources.append(present[members[rows[valid]]])
        targets.append(present[members[chosen[valid]]])
        retained += int(valid.sum())
    source = np.concatenate(sources + [np.zeros(0, np.int32)]).astype(np.int32)
    target = np.concatenate(targets + [np.zeros(0, np.int32)]).astype(np.int32)
    return source, target, np.full(len(source), registry.code("diffusion"), np.int64)


def neighbours(adjacency):
    return np.r_[0, np.cumsum(adjacency.sum(1))].astype(np.int64), np.nonzero(adjacency)[1].astype(np.int64)


def forman(adjacency, sources, targets):
    offsets, columns = neighbours(adjacency)
    return kernels.forman(adjacency, offsets, columns, sources.astype(np.int64), targets.astype(np.int64))


def curvature(adjacency, i, j):
    return float(forman(adjacency, np.asarray([i], np.int64), np.asarray([j], np.int64))[0])


def spectral(adjacency, members, device):
    local = adjacency[np.ix_(members, members)]
    if local.sum() == len(members) * (len(members) - 1):
        return None
    scale = 1 / np.sqrt(local.sum(1) + 1.0)
    values, vectors = compute.eigh(scale[:, None] * (local + np.eye(len(members))) * scale[None, :], device)
    fiedler = vectors[:, -2] * scale
    scores = np.outer(fiedler, fiedler)
    scores[local] = np.inf
    np.fill_diagonal(scores, np.inf)
    i, j = divmod(int(np.argmin(scores)), len(members))
    return float(values[-1] - values[-2]), members[i], members[j]


def fosr(adjacencies, found, device):
    best = None
    for label, adjacency in enumerate(adjacencies):
        if label not in found:
            found[label] = spectral(adjacency, np.arange(len(adjacency)), device)
        candidate = found[label]
        if candidate is not None and (best is None or candidate[0] < best[1][0] - 1e-9):
            best = label, candidate
    return best


def component_edges(present, u, v):
    local_u, local_v = np.searchsorted(present, u), np.searchsorted(present, v)
    labels = kernels.components(local_u.astype(np.int64), local_v.astype(np.int64), len(present))
    groups = blocks(labels)
    edge_labels = labels[local_u]
    order = np.argsort(edge_labels, kind="stable")
    edge_labels = edge_labels[order]
    for members in groups:
        lo = np.searchsorted(edge_labels, labels[members[0]], side="left")
        hi = np.searchsorted(edge_labels, labels[members[0]], side="right")
        rows = order[lo:hi]
        source, target = np.searchsorted(members, local_u[rows]), np.searchsorted(members, local_v[rows])
        yield members, source, target


def component_adjacencies(present, u, v, budget, operation):
    edges = list(component_edges(present, u, v))
    cells = sum(len(members) ** 2 for members, _, _ in edges)
    budget.workspace(
        2 * cells + max((len(members) ** 2 * 78 for members, _, _ in edges), default=0), operation
    )
    groups, adjacencies = [], []
    for members, source, target in edges:
        adjacency = np.zeros((len(members), len(members)), bool)
        adjacency[source, target] = adjacency[target, source] = True
        groups.append(members)
        adjacencies.append(adjacency)
    return groups, adjacencies


def curvature_extreme(groups, adjacencies, found, maximum=False):
    best = None
    for label, adjacency in enumerate(adjacencies):
        if label not in found:
            edges = np.argwhere(np.triu(adjacency, 1))
            found[label] = edges, forman(adjacency, edges[:, 0], edges[:, 1])
        edges, values = found[label]
        if not len(edges):
            continue
        row = int(np.argmax(values) if maximum else np.argmin(values))
        i, j = edges[row]
        key = (-values[row] if maximum else values[row], groups[label][i], groups[label][j])
        if best is None or key < best[0]:
            best = key, label, i, j, values[row]
    return best


@timed("operators.rewire")
def rewire(spec, present, u, v, entities, rng, budget):
    budget.pairs(spec["derive/rewiring_budget"], "rewiring")
    groups, adjacencies = component_adjacencies(present, u, v, budget, spec["derive/rewiring"] + " rewiring")
    originals = [adjacency.copy() for adjacency in adjacencies]
    found = {}
    for _ in range(spec["derive/rewiring_budget"]):
        if spec["derive/rewiring"] == "fosr":
            best = fosr(adjacencies, found, budget.limits.device)
            if best is None:
                break
            label, (_, i, j) = best
            adjacency = adjacencies[label]
            adjacency[i, j] = adjacency[j, i] = True
            del found[label]
            continue
        best = curvature_extreme(groups, adjacencies, found)
        if best is None:
            break
        _, label, i, j, value = best
        adjacency = adjacencies[label]
        left = np.concatenate(([i], np.flatnonzero(adjacency[i]))).astype(np.int64)
        right = np.concatenate(([j], np.flatnonzero(adjacency[j]))).astype(np.int64)
        k, partner = np.repeat(left, len(right)), np.tile(right, len(left))
        open_pair = (k != partner) & ~adjacency[k, partner]
        chosen = np.stack((k[open_pair], partner[open_pair]), 1)
        if len(chosen):
            offsets, columns = neighbours(adjacency)
            gains = (
                kernels.forman_with(adjacency, offsets, columns, int(i), int(j), chosen[:, 0], chosen[:, 1])
                - value
            )
            scaled = np.asarray(gains) * spec["derive/sdrf_temperature"]
            probability = np.exp(scaled - scaled.max())
            k, partner = chosen[int(rng.choice(len(chosen), p=probability / probability.sum()))]
            adjacency[k, partner] = adjacency[partner, k] = True
            del found[label]
        best = curvature_extreme(groups, adjacencies, found, maximum=True)
        if best is not None and best[4] > spec["derive/sdrf_threshold"]:
            _, label, p, q, _ = best
            adjacencies[label][p, q] = adjacencies[label][q, p] = False
            del found[label]
    additions, removals = [], []
    for members, adjacency, original in zip(groups, adjacencies, originals):
        for changed, result in ((adjacency & ~original, additions), (original & ~adjacency, removals)):
            rows = np.argwhere(np.triu(changed, 1))
            result.append(
                present[members[rows[:, 0]]].astype(np.int64) * entities + present[members[rows[:, 1]]]
            )
    added = np.sort(np.concatenate(additions + [np.zeros(0, np.int64)]))
    cut = np.sort(np.concatenate(removals + [np.zeros(0, np.int64)]))
    return ((added // entities).astype(present.dtype), (added % entities).astype(present.dtype)), cut


def derivation(registry, spec):
    projected = [
        edge
        for edge in registry.edges
        if edge.origin[0] == "project" and enabled(spec, "derive/" + edge.name)
    ]
    selected = np.asarray(
        [registry.kind(kind) for kind in registry.nodes if enabled(spec, "derive/complete/" + kind)], np.int64
    )
    within = np.asarray(
        [registry.kind(kind) for kind in registry.nodes if enabled(spec, "derive/complete_within/" + kind)],
        np.int64,
    )
    wanted = (
        projected
        or len(selected)
        or spec.get("derive/hops", 1) != 1
        or spec.get("derive/diffusion", "none") != "none"
        or spec.get("derive/rewiring", "none") != "none"
    )
    return (projected, selected, within) if wanted else None


def derivable(graph, registry, spec):
    names = registry.extras["base"]
    stated = graph.fact_present & np.isin(graph.fact_code, codes(registry, names))
    kept = graph.fact_present & np.isin(
        graph.fact_code,
        codes(registry, [name for name in names if enabled(spec, "relations/%s/keep" % name)]),
    )
    return stated, kept


@timed("operators.derive_prepare")
def derive_prepare(graph, registry, spec, carry=None):
    stated, _ = derivable(graph, registry, spec)
    recency = np.zeros(len(graph.fact_step), np.int64)
    statements = np.flatnonzero(stated)
    ids, count = kernels.keyed(
        graph.fact_code[statements], graph.fact_source[statements], graph.fact_target[statements]
    )
    keys = (
        None
        if carry is None
        else carry.keys(
            graph.fact_code[statements], graph.fact_source[statements], graph.fact_target[statements]
        )
    )
    (_, recency[statements]), piece = run_state(
        carry, "derive/recency", graph, ids, count, graph.fact_step[statements], keys
    )
    open_runs(carry, "derive/recency", [piece])
    return recency


@timed("operators.derive_steps")
def derive_steps(graph, registry, spec, seed, recency, step0, limits):
    projected, selected, within = derivation(registry, spec)
    steps, entities = graph.steps, len(graph.kinds)
    stated, kept = derivable(graph, registry, spec)
    fact_bounds = np.searchsorted(graph.fact_step, np.arange(steps + 1))
    observed_bounds = np.searchsorted(graph.obs_step, np.arange(steps + 1))
    budget = GraphBudget(limits, fact_bytes(graph))
    chunks, removed, memo = [], np.zeros(len(graph.fact_step), bool), {}
    for step in range(steps):
        span = np.arange(fact_bounds[step], fact_bounds[step + 1])
        every, rows = span[stated[span]], span[kept[span]]
        source, target = graph.fact_source[rows], graph.fact_target[rows]
        lo, hi = observed_bounds[step], observed_bounds[step + 1]
        present = np.sort(graph.obs_entity[lo:hi][~graph.obs_anchor[lo:hi]])
        u, v = undirected(source, target, entities)
        found = []
        for edge in projected:
            a, b = project(
                graph,
                registry,
                edge,
                graph.fact_source[every],
                graph.fact_target[every],
                graph.fact_code[every],
                recency[every],
                spec["derive/project_cap"],
                budget,
                memo,
            )
            budget.retain(len(a), edge.name)
            found.append((a, b, np.full(len(a), edge.code, np.int64)))
        if len(selected):
            completed = recalled(
                memo,
                "completion",
                (present, u, v),
                budget,
                lambda: complete(graph, registry, selected, within, present, u, v, budget),
            )
            budget.retain(len(completed[0]), "completion")
            found.append(completed)
        if spec.get("derive/hops", 1) > 1:
            found.extend(
                recalled(
                    memo,
                    "hop closure",
                    (u, v),
                    budget,
                    lambda: closure(registry, u, v, spec["derive/hops"], entities, budget),
                )
            )
        if spec.get("derive/diffusion", "none") != "none":
            diffused = recalled(
                memo,
                "diffusion",
                (present, u, v),
                budget,
                lambda: diffuse(registry, spec, present, u, v, entities, budget),
            )
            budget.retain(len(diffused[0]), "diffusion")
            found.append(diffused)
        if spec.get("derive/rewiring", "none") != "none":
            (a, b), cut = rewire(
                spec,
                present,
                u,
                v,
                entities,
                np.random.default_rng([seed, REWIRE, step + step0]),
                budget,
            )
            budget.retain(len(a), "rewiring")
            found.append((a, b, np.full(len(a), registry.code("rewire"), np.int64)))
            keys = np.minimum(source, target).astype(np.int64) * entities + np.maximum(source, target)
            removed[rows[np.isin(keys, cut)]] = True
        chunks.extend((a, b, c, np.full(len(a), step, np.int32)) for a, b, c in found)
    source, target, code, step = (
        np.concatenate([chunk[i] for chunk in chunks] + [np.zeros(0, dtype)])
        for i, dtype in enumerate((np.int32, np.int32, np.int64, np.int32))
    )
    return (source, target, code, step), removed, budget


def derive_handoff(carry, found, steps, connected):
    source, target, code, step = found
    live = carry.state.get("derive/live")
    stop = (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.int64),)
    if live is not None and len(live["source"]):
        begun = carry.keys(code[step == 0], source[step == 0], target[step == 0])
        ended = ~np.isin(live["key"], begun)
        stop = (
            carry.local(live["source"][ended]).astype(np.int32),
            carry.local(live["target"][ended]).astype(np.int32),
            live["code"][ended],
        )
    final = np.flatnonzero((step == steps - 1) & connected[source] & connected[target])
    final = final[np.lexsort((target[final], source[final], code[final]))]
    carry.state["derive/live"] = dict(
        key=carry.keys(code[final], source[final], target[final]),
        source=carry.glob[source[final]],
        target=carry.glob[target[final]],
        code=code[final],
    )
    return stop


def derived_rows(graph, found, stop, budget):
    source, target, code, step = found
    entities = len(graph.kinds)
    if (int(code.max(initial=0)) + 1) * entities * entities >= 1 << 63:
        raise GraphCapacityError("derived fact keys exceed 63 bits for %d entities" % entities)
    if np.any(np.diff(step) < 0):
        by_step = np.argsort(step, kind="stable")
        source, target, code, step = source[by_step], target[by_step], code[by_step], step[by_step]
    pair = (code.astype(np.int64) * entities + source) * entities + target
    bounds = np.flatnonzero(np.r_[True, np.diff(step) != 0, True])
    order = np.empty(len(step), np.int64)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        order[lo:hi] = lo + np.argsort(pair[lo:hi])
    source, target, code, step = source[order], target[order], code[order], step[order]
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
        np.r_[source, source[ends], stop[0]],
        np.r_[target, target[ends], stop[1]],
        np.r_[code, code[ends], stop[2]],
        np.r_[step, step[ends] + 1, np.zeros(len(stop[0]), np.int32)],
        np.r_[np.ones(len(step), bool), np.zeros(len(ends) + len(stop[0]), bool)],
        np.full(total, -1, np.int32),
    )


@timed("operators.derive_finish")
def derive_finish(graph, found, removed, budget, carry=None):
    graph = prune(graph, ~removed, carry, "derive")
    live = None if carry is None else carry.state.get("derive/live")
    if not budget.facts and (live is None or not len(live["source"])):
        if carry is not None:
            carry.state["derive/live"] = None
        return graph
    stop = (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.int64),)
    if carry is not None:
        stop = derive_handoff(carry, found, graph.steps, graph.entity_connected)
    return derived_rows(graph, found, stop, budget)


@timed("operators.derive")
def derive(graph, registry, spec, seed, limits, carry=None):
    if derivation(registry, spec) is None:
        return graph
    recency = derive_prepare(graph, registry, spec, carry)
    found, removed, budget = derive_steps(
        graph, registry, spec, seed, recency, 0 if carry is None else carry.step0, limits
    )
    return derive_finish(graph, found, removed, budget, carry)


@timed("operators.untype")
def untype(graph, registry, carry=None):
    base = np.isin(graph.fact_code, codes(registry, registry.extras["base"]))
    source, target, code, present = graph.fact_source, graph.fact_target, graph.fact_code, graph.fact_present
    added = 0
    if carry is not None:
        live = carry.table("merging/live")
        pair_keys = carry.keys(np.zeros(len(source), np.int64), source, target)
        column = {
            name: live.columns.get(name, np.zeros(0, np.int64))
            for name in ("pair", "source", "target", "code")
        }
        chosen = np.flatnonzero(np.isin(column["pair"], pair_keys))
        added = len(chosen)
        source = np.r_[carry.local(column["source"][chosen]), source].astype(source.dtype)
        target = np.r_[carry.local(column["target"][chosen]), target].astype(target.dtype)
        code = np.r_[column["code"][chosen], code].astype(code.dtype)
        present = np.r_[np.ones(added, bool), present]
        base = np.r_[np.ones(added, bool), base]
    typed, typed_count = kernels.keyed(source, target, code)
    pairs, pair_count = kernels.keyed(source, target, np.zeros(len(base), np.int64))
    kept = kernels.untyped(typed, pairs, present, base, typed_count, pair_count)[added:]
    if carry is not None:
        rows = np.flatnonzero(base & graph.entity_connected[source] & graph.entity_connected[target])
        last = np.full(typed_count, -1, np.int64)
        np.maximum.at(last, typed[rows], rows)
        final = last[last >= 0]
        keys = carry.keys(code[final], source[final], target[final])
        state = present[final]
        carry.state["merging/live"] = live.replaced(
            keys,
            keys[state],
            pair=carry.keys(np.zeros(int(state.sum()), np.int64), source[final][state], target[final][state]),
            source=carry.glob[source[final][state]],
            target=carry.glob[target[final][state]],
            code=code[final][state],
        )
        base = base[added:]
    graph = graph.replace(
        fact_code=np.where(base, registry.code(UNTYPED), graph.fact_code),
        fact_ref=np.where(base, -1, graph.fact_ref).astype(graph.fact_ref.dtype),
    )
    return select(graph, FACTS, kept)


@timed("operators.relations")
def relations(graph, registry, spec, seed, carry=None):
    dropped = [name for name in registry.extras["base"] if not enabled(spec, "relations/%s/keep" % name)]
    if not dropped:
        return graph
    return select(graph, FACTS, ~np.isin(graph.fact_code, codes(registry, dropped)))


@timed("operators.merging")
def merging(graph, registry, spec, seed, carry=None):
    if spec.get("relations/types") == "untyped":
        graph = untype(graph, registry, carry)
    if spec.get("relations/parallel") != "merge":
        return graph
    ids, count = kernels.keyed(
        graph.fact_step.astype(np.int64) * 2 + graph.fact_present,
        graph.fact_code * len(graph.kinds) + graph.fact_source,
        graph.fact_target,
    )
    return select(graph, FACTS, kernels.last_rows(ids, count))


def symmetric(registry, spec):
    return np.asarray(
        [edge.symmetric for edge in registry.edges] + [False] * (registry.relations - len(registry.edges))
    ) | (spec.get("relations/direction") == "symmetric")


def filtering(spec, name):
    prefix = "edges/%s/" % name
    return prefix + "drop" in spec and (
        spec[prefix + "drop"] > 0 or spec[prefix + "persistence"] > 1 or spec[prefix + "cap"] > 0
    )


@timed("operators.edges")
def edges(graph, registry, spec, seed, carry=None):
    named = [edge for edge in registry.edges if filtering(spec, edge.name)]
    if not named:
        return graph
    kept = np.ones(len(graph.fact_step), bool)
    pieces = {"edges/persistence": [], "edges/cap": []}
    for edge in named:
        rows = np.flatnonzero(graph.fact_code == edge.code)
        kept[rows] = filtered(graph, registry, spec, seed, edge.name, rows, carry, pieces)
    for name, found in pieces.items():
        open_runs(carry, name, found)
    return prune(graph, kept, carry, "edges")


def edges_survive(graph, registry, spec, seed, carry):
    survives = np.ones(len(graph.fact_step), bool)
    for edge in registry.edges:
        if filtering(spec, edge.name):
            rows = np.flatnonzero(graph.fact_code == edge.code)
            survives[rows] = undropped(graph, registry, spec, seed, edge.name, rows, carry)
    return survives


def edges_mask(graph, registry, spec, seed, carry, names, survives):
    named = [edge for edge in registry.edges if filtering(spec, edge.name) and edge.name in names]
    kept = np.ones(len(graph.fact_step), bool)
    if not named:
        return kept
    pieces = {"edges/persistence": [], "edges/cap": []}
    for edge in named:
        rows = np.flatnonzero(graph.fact_code == edge.code)
        kept[rows] = filtered(graph, registry, spec, seed, edge.name, rows, carry, pieces, survives[rows])
    for name, found in pieces.items():
        open_runs(carry, name, found)
    rows = np.flatnonzero(np.isin(graph.fact_code, [edge.code for edge in named]))
    kept[rows] = prune_mask(
        graph.fact_source[rows],
        graph.fact_target[rows],
        graph.fact_code[rows],
        graph.fact_present[rows],
        kept[rows],
        graph.entity_connected,
        carry,
        "edges",
    )
    return kept


def endpoints(graph, registry, spec, rows):
    code, source, target = graph.fact_code[rows], graph.fact_source[rows], graph.fact_target[rows]
    both = symmetric(registry, spec)[code]
    return (
        code,
        both,
        np.where(both, np.minimum(source, target), source),
        np.where(both, np.maximum(source, target), target),
    )


def undropped(graph, registry, spec, seed, name, rows, carry=None):
    code, _, first, second = endpoints(graph, registry, spec, rows)
    return (
        unit(seed, DROP, identities(carry, first), identities(carry, second), code)
        >= spec["edges/%s/drop" % name]
    )


def filtered(graph, registry, spec, seed, name, rows, carry=None, pieces=None, survives=None):
    prefix = "edges/%s/" % name
    step, present = graph.fact_step[rows], graph.fact_present[rows]
    code, both, first, second = endpoints(graph, registry, spec, rows)
    kept = undropped(graph, registry, spec, seed, name, rows, carry) if survives is None else survives.copy()
    stated = np.flatnonzero(kept & present)
    ids, count = kernels.keyed(code[stated], first[stated], second[stated])
    keys = None if carry is None else carry.keys(code[stated], first[stated], second[stated])
    (position, _), piece = run_state(carry, "edges/persistence", graph, ids, count, step[stated], keys)
    if pieces is not None:
        pieces["edges/persistence"].append(piece)
    lasting = position >= spec[prefix + "persistence"] - 1
    kept[stated[~lasting]] = False
    if spec[prefix + "cap"] == 0:
        return kept
    stated = stated[lasting]
    low, high = first[stated], second[stated]
    ids, count = kernels.keyed(code[stated], low, high)
    keys = None if carry is None else carry.keys(code[stated], low, high)
    (_, begun), piece = run_state(carry, "edges/cap", graph, ids, count, step[stated], keys)
    if pieces is not None:
        pieces["edges/cap"].append(piece)
    order = spec[prefix + "cap_order"]
    if order == "recent":
        primary, secondary = -begun, rows[stated]
    elif order == "random":
        primary, secondary = (
            unit(seed, RANDOM, identities(carry, low), identities(carry, high), code[stated]),
            rows[stated],
        )
    else:
        reference = graph.fact_ref[rows[stated]]
        value = np.where(
            reference >= 0,
            graph.value_table[
                np.maximum(reference, 0), registry.attributes(name).index(order.split(":", 1)[1])
            ],
            np.nan,
        )
        primary, secondary = np.where(np.isnan(value), np.inf, -value), -begun
    pair, count = kernels.keyed(step[stated].astype(np.int64) * registry.relations + code[stated], low, high)
    best = kernels.smallest(pair, primary.astype(np.float64), count)
    earliest = kernels.smallest(pair, secondary.astype(np.float64), count)
    pair_step, pair_code, pair_low, pair_high, mode = (np.zeros(count, np.int64) for _ in range(5))
    pair_step[pair], pair_code[pair], pair_low[pair], pair_high[pair] = step[stated], code[stated], low, high
    mode[pair] = np.where(both[stated], 2, 1 if spec.get("relations/direction") == "inverse" else 0)
    extra = np.flatnonzero((mode == 1) | ((mode == 2) & (pair_low != pair_high)))
    owners = np.r_[np.arange(count), extra]
    flag = np.r_[np.zeros(count, np.int64), mode[extra] == 1]
    owner = np.r_[pair_high, pair_low[extra]]
    part = np.r_[np.zeros(count, np.int64), np.ones(len(extra), np.int64)]
    group, groups = kernels.keyed(
        (pair_step[owners] * registry.relations + pair_code[owners]) * 2 + flag,
        owner,
        np.zeros(len(owners), np.int64),
    )
    within = kernels.within_cap(
        group,
        groups,
        best[owners],
        earliest[owners],
        part,
        pair_low[owners],
        pair_high[owners],
        spec[prefix + "cap"],
    )
    capped = np.ones(count, bool)
    capped[owners[~within]] = False
    kept[stated] = capped[pair]
    return kept
