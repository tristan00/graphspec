import numpy as np

from graph_tuning_infra import compute
from graph_tuning_infra.hashing import unit

from . import kernels
from .vocabulary import UNTYPED, completion, hop


FACTS = ('fact_source', 'fact_target', 'fact_code', 'fact_step', 'fact_present', 'fact_values')
OBSERVATIONS = ('obs_entity', 'obs_step', 'obs_x', 'obs_anchor')
SAMPLE, DROP, RANDOM, REWIRE = 1, 2, 3, 4
BLOCK = 1 << 22


def counted(counts):
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


def enabled(spec, name):
    return name in spec and spec[name]


def codes(registry, names):
    return [registry.code(name) for name in names]


def select(graph, names, mask):
    return graph.replace(**{name: graph.arrays[name][mask] for name in names})


def extend(graph, source, target, code, step, present, values):
    added = dict(fact_source=source, fact_target=target, fact_code=code, fact_step=step, fact_present=present, fact_values=values)
    added = {name: np.asarray(value, graph.arrays[name].dtype) for name, value in added.items()}
    order = np.lexsort((np.arange(len(added['fact_step'])), ~added['fact_present'], added['fact_step']))
    key = graph.fact_step.astype(np.int64) * 2 + ~graph.fact_present
    position = np.searchsorted(key, added['fact_step'][order].astype(np.int64) * 2 + ~added['fact_present'][order], 'right') + np.arange(len(order))
    total = len(key) + len(order)
    existing = np.ones(total, bool)
    existing[position] = False
    merged = {}
    for name in FACTS:
        column = np.empty((total,) + graph.arrays[name].shape[1:], graph.arrays[name].dtype)
        column[existing] = graph.arrays[name]
        column[position] = added[name][order]
        merged[name] = column
    return graph.replace(**merged)


def prune(graph, kept):
    if kept.all():
        return graph
    ids, count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
    return select(graph, FACTS, kernels.orphans(ids, graph.fact_present, kept, count))


def reach(graph, spec, start):
    if spec['time/scheme'] == 'state':
        return np.full(len(start), graph.steps - 1, np.int64)
    if spec.get('time/unit') == 'period':
        group = np.cumsum(np.r_[0, np.diff(graph.period) != 0])
        return np.searchsorted(group, group[start] + spec['time/past_periods'], 'right') - 1
    return start + spec['time/window'] - 1


def window(graph, registry, spec, seed):
    if spec['time/scheme'] == 'window' and spec.get('time/unit', 'step') == 'step' and spec['time/window'] == 1:
        return graph
    steps = graph.steps
    order = np.lexsort((graph.obs_step, graph.obs_entity))
    entity, step = graph.obs_entity[order], graph.obs_step[order].astype(np.int64)
    later = np.r_[entity[1:] == entity[:-1], False]
    limit = np.where(later, np.r_[step[1:], 0], steps)
    counts = np.maximum(np.minimum(reach(graph, spec, step), limit - 1) - step, 0)
    if counts.sum() > compute.FACTS:
        raise ValueError('the time window of graph %s carries %d observation copies, over the budget of %d' % (graph.name, counts.sum(), compute.FACTS))
    origin = np.repeat(order, counts)
    copies = {name: np.concatenate((graph.arrays[name], graph.arrays[name][origin])) for name in OBSERVATIONS}
    copies['obs_step'][len(graph.obs_step):] += (counted(counts) + 1).astype(np.int32)
    arrangement = np.lexsort((np.arange(len(copies['obs_step'])), copies['obs_step']))
    graph = graph.replace(**{name: value[arrangement] for name, value in copies.items()})
    ids, count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
    limit = kernels.carried(ids, graph.fact_step, graph.fact_present, count, steps)
    present = np.flatnonzero(graph.fact_present)
    start = graph.fact_step[present].astype(np.int64)
    counts = np.maximum(np.minimum(reach(graph, spec, start), limit[present] - 1) - start, 0)
    if counts.sum() > compute.FACTS:
        raise ValueError('the time window of graph %s carries %d fact copies, over the budget of %d' % (graph.name, counts.sum(), compute.FACTS))
    origin = np.repeat(present, counts)
    return extend(graph, graph.fact_source[origin], graph.fact_target[origin], graph.fact_code[origin],
                  (graph.fact_step[origin].astype(np.int64) + counted(counts) + 1).astype(np.int32), np.ones(len(origin), bool), graph.fact_values[origin])


def periods(graph, registry, spec, seed):
    if spec['time/periods'] == 'truth':
        return graph
    return graph.replace(period=(np.arange(graph.steps) // spec['time/period_length']).astype(np.int32))


def nodes(graph, registry, spec, seed):
    steps, entities = graph.steps, len(graph.kinds)
    protected = registry.extras['protected']
    retained = np.asarray([kind in protected or spec['nodes/%s/keep' % kind] for kind in registry.nodes])
    filters = {kind: spec['nodes/%s/filter' % kind] for kind in registry.nodes if spec.get('nodes/%s/filter' % kind, 'none') != 'none'}
    if retained.all() and not filters and not graph.obs_anchor.any():
        return graph
    allowed = retained[graph.kinds]
    obs_entity, obs_step = graph.obs_entity, graph.obs_step
    use = graph.fact_present & allowed[graph.fact_source] & allowed[graph.fact_target]
    kinds = graph.kinds[obs_entity]
    passed, minimum, counts = np.ones(len(obs_step), bool), np.zeros(len(obs_step), np.int64), {}
    for kind, mode in filters.items():
        prefix, members = 'nodes/%s/' % kind, kinds == registry.kind(kind)
        if mode in ('degree', 'total_degree'):
            if mode not in counts:
                counts[mode] = kernels.degrees(obs_entity, obs_step, graph.fact_source, graph.fact_target, graph.fact_step, use, entities, mode == 'total_degree')
            passed[members] = counts[mode][members] >= spec[prefix + 'min_degree']
        elif mode == 'activity':
            if mode not in counts:
                counts[mode] = kernels.activity(obs_entity, entities)
            passed[members] = counts[mode][members] >= spec[prefix + 'min_activity']
        elif mode == 'sample':
            passed[members] = unit(seed, SAMPLE, obs_entity[members]) < spec[prefix + 'sample']
        else:
            minimum[members] = spec[prefix + 'core']
    structural = allowed[obs_entity] & passed
    if minimum.any():
        structural = kernels.peel(obs_entity, obs_step, structural, minimum, graph.fact_source, graph.fact_target, graph.fact_step, use, entities)
    needed = np.zeros(len(obs_step), bool)
    if len(graph.query_key):
        needed = (graph.query_actor[obs_step] == obs_entity) | (graph.query_subject[obs_step] == obs_entity) | (graph.query_context[obs_step] == obs_entity)
    first, _ = kernels.first_observed(obs_entity, obs_step, structural, entities, steps)
    present = (kernels.observed_at(obs_entity, obs_step, structural, graph.fact_source, graph.fact_step, entities)
               & kernels.observed_at(obs_entity, obs_step, structural, graph.fact_target, graph.fact_step, entities))
    absent = (first[graph.fact_source] <= graph.fact_step) & (first[graph.fact_target] <= graph.fact_step)
    alive = structural | needed
    graph = select(graph, OBSERVATIONS, alive).replace(obs_anchor=~structural[alive])
    return prune(graph, np.where(graph.fact_present, present, absent))


def undirected(source, target, entities):
    use = source != target
    key = np.unique(np.minimum(source[use], target[use]).astype(np.int64) * entities + np.maximum(source[use], target[use]))
    return (key // entities).astype(np.int32), (key % entities).astype(np.int32)


def unique_pairs(chunks):
    source = np.concatenate([chunk[0] for chunk in chunks] + [np.zeros(0, np.int32)])
    target = np.concatenate([chunk[1] for chunk in chunks] + [np.zeros(0, np.int32)])
    ids, count = kernels.keyed(source, target, np.zeros(len(source), np.int64))
    fresh = kernels.first_rows(ids, count)
    return source[fresh], target[fresh]


def project(graph, registry, edge, source, target, code, recency, cap):
    relation, side, first, second = edge.origin[1:]
    use = np.flatnonzero(code == registry.code(relation))
    members, owners = (source[use], target[use]) if side == 'sources' else (target[use], source[use])
    order = np.lexsort((np.arange(len(use)), recency[use], owners))
    pairs = kernels.project_pairs(members[order], owners[order], graph.kinds, registry.kind(first), registry.kind(second), cap)
    return unique_pairs([pairs])


def complete(graph, registry, selected, within, present, u, v):
    members = present[np.isin(graph.kinds[present], selected)]
    if not len(within):
        flat, offsets = members, np.asarray([0, len(members)], np.int64)
    else:
        anchors, linked = np.r_[u, v], np.r_[v, u]
        use = np.isin(graph.kinds[anchors], within) & np.isin(linked, members)
        anchors, linked = anchors[use], linked[use]
        order = np.argsort(anchors, kind='stable')
        anchors, flat = anchors[order], linked[order]
        offsets = np.r_[0, np.flatnonzero(np.diff(anchors)) + 1, len(anchors)] if len(order) else np.zeros(1, np.int64)
    source, target = unique_pairs([kernels.within_pairs(flat, offsets, graph.kinds, len(registry.nodes))])
    table = np.full((len(registry.nodes), len(registry.nodes)), -1, np.int64)
    for i, left in enumerate(registry.nodes):
        for right in registry.nodes[i:]:
            table[i, registry.kind(right)] = registry.code(completion(left, right))
    return source, target, table[graph.kinds[source], graph.kinds[target]]


def widen(slots, keys_low, keys_high, count, extra):
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


def closure(registry, u, v, limit, entities):
    source, target = np.r_[u, v].astype(np.int64), np.r_[v, u].astype(np.int64)
    order = np.argsort(source, kind='stable')
    source, target = source[order], target[order]
    starts = np.searchsorted(source, np.arange(entities + 1))
    slots, keys_low, keys_high = widen(np.zeros(16, np.int64), np.zeros(16, np.int64), np.zeros(16, np.int64), 0, len(source))
    count, _ = kernels.novel(slots, keys_low, keys_high, 0, np.minimum(source, target), np.maximum(source, target))
    left, middle = source, target
    result = []
    for distance in range(2, limit + 1):
        counts = starts[middle + 1] - starts[middle]
        edges = np.cumsum(counts)
        pieces, lo = [], 0
        while lo < len(counts):
            hi = max(int(np.searchsorted(edges, (edges[lo - 1] if lo else 0) + BLOCK, 'right')), lo + 1)
            block = counts[lo:hi]
            reached = target[np.repeat(starts[middle[lo:hi]], block) + counted(block)]
            origin = np.repeat(left[lo:hi], block)
            use = origin != reached
            low, high = np.minimum(origin[use], reached[use]), np.maximum(origin[use], reached[use])
            slots, keys_low, keys_high = widen(slots, keys_low, keys_high, count, len(low))
            count, fresh = kernels.novel(slots, keys_low, keys_high, count, low, high)
            pieces.append((low[fresh], high[fresh]))
            lo = hi
        a = np.concatenate([piece[0] for piece in pieces] + [np.zeros(0, np.int64)])
        b = np.concatenate([piece[1] for piece in pieces] + [np.zeros(0, np.int64)])
        result.append((a.astype(np.int32), b.astype(np.int32), np.full(len(a), registry.code(hop(distance)), np.int64)))
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
    order = np.argsort(labels, kind='stable')
    bounds = np.flatnonzero(np.r_[True, np.diff(labels[order]) != 0, True])
    return [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:]) if hi - lo > 1]


def diffusion_scores(adjacency, spec):
    count = len(adjacency)
    scale = 1 / np.sqrt(adjacency.sum(1))
    transition = scale[:, None] * adjacency * scale[None, :]
    if spec['derive/diffusion'] == 'ppr':
        alpha = spec['derive/ppr_alpha']
        scores = alpha * compute.inverse(np.eye(count) - (1 - alpha) * transition)
    else:
        values, vectors = compute.eigh(transition)
        scores = compute.compose(vectors, np.exp(-spec['derive/heat_time'] * (1 - values)))
    scores[adjacency] = -np.inf
    np.fill_diagonal(scores, -np.inf)
    return scores


def diffuse(registry, spec, present, u, v, entities):
    count = len(present)
    if count < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    local_u, local_v = np.searchsorted(present, u), np.searchsorted(present, v)
    labels = kernels.components(local_u.astype(np.int64), local_v.astype(np.int64), count)
    sources, targets = [], []
    for members in blocks(labels):
        scores = diffusion_scores(dense(members, local_u, local_v), spec)
        width = min(spec['derive/diffusion_top_k'], len(members) - 1)
        chosen, ranked = compute.top(scores, width)
        valid = np.isfinite(ranked)
        rows = np.broadcast_to(np.arange(len(members))[:, None], chosen.shape)
        sources.append(present[members[rows[valid]]])
        targets.append(present[members[chosen[valid]]])
    source = np.concatenate(sources + [np.zeros(0, np.int32)]).astype(np.int32)
    target = np.concatenate(targets + [np.zeros(0, np.int32)]).astype(np.int32)
    return source, target, np.full(len(source), registry.code('diffusion'), np.int64)


def curvature(adjacency, i, j):
    return float(kernels.forman(adjacency, np.asarray([i], np.int64), np.asarray([j], np.int64))[0])


def spectral(adjacency, members):
    local = adjacency[np.ix_(members, members)]
    if local.sum() == len(members) * (len(members) - 1):
        return None
    scale = 1 / np.sqrt(local.sum(1) + 1.0)
    values, vectors = compute.eigh(scale[:, None] * (local + np.eye(len(members))) * scale[None, :])
    fiedler = vectors[:, -2] * scale
    scores = np.outer(fiedler, fiedler)
    scores[local] = np.inf
    np.fill_diagonal(scores, np.inf)
    i, j = divmod(int(np.argmin(scores)), len(members))
    return float(values[-1] - values[-2]), members[i], members[j]


def fosr(adjacency, groups, found):
    best = None
    for label, members in enumerate(groups):
        if label not in found:
            found[label] = spectral(adjacency, members)
        candidate = found[label]
        if candidate is not None and (best is None or candidate[0] < best[1][0] - 1e-9):
            best = label, candidate
    return best


def rewire(spec, present, u, v, entities, rng):
    original = dense(present, u, v)
    adjacency = original.copy()
    groups = blocks(kernels.components(np.searchsorted(present, u).astype(np.int64), np.searchsorted(present, v).astype(np.int64), len(present)))
    found = {}
    for _ in range(spec['derive/rewiring_budget']):
        if spec['derive/rewiring'] == 'fosr':
            best = fosr(adjacency, groups, found)
            if best is None:
                break
            label, (_, i, j) = best
            adjacency[i, j] = adjacency[j, i] = True
            del found[label]
            continue
        edges = np.argwhere(np.triu(adjacency, 1))
        if not len(edges):
            break
        values = kernels.forman(adjacency, edges[:, 0], edges[:, 1])
        i, j = edges[int(np.argmin(values))]
        candidates = [(k, l) for k in np.r_[i, np.flatnonzero(adjacency[i])] for l in np.r_[j, np.flatnonzero(adjacency[j])]
                      if k != l and not adjacency[k, l]]
        if candidates:
            gains = []
            for k, l in candidates:
                adjacency[k, l] = adjacency[l, k] = True
                gains.append(curvature(adjacency, i, j) - values.min())
                adjacency[k, l] = adjacency[l, k] = False
            scaled = np.asarray(gains) * spec['derive/sdrf_temperature']
            probability = np.exp(scaled - scaled.max())
            k, l = candidates[int(rng.choice(len(candidates), p=probability / probability.sum()))]
            adjacency[k, l] = adjacency[l, k] = True
        edges = np.argwhere(np.triu(adjacency, 1))
        values = kernels.forman(adjacency, edges[:, 0], edges[:, 1])
        if values.max() > spec['derive/sdrf_threshold']:
            p, q = edges[int(np.argmax(values))]
            adjacency[p, q] = adjacency[q, p] = False
    added = np.argwhere(np.triu(adjacency & ~original, 1))
    cut = np.argwhere(np.triu(original & ~adjacency, 1))
    low, high = present[cut[:, 0]].astype(np.int64), present[cut[:, 1]].astype(np.int64)
    return (present[added[:, 0]], present[added[:, 1]]), np.minimum(low, high) * entities + np.maximum(low, high)


def derive(graph, registry, spec, seed):
    steps, entities = graph.steps, len(graph.kinds)
    names = registry.extras['base']
    projected = [edge for edge in registry.edges if edge.origin[0] == 'project' and enabled(spec, 'derive/' + edge.name)]
    selected = np.asarray([registry.kind(kind) for kind in registry.nodes if enabled(spec, 'derive/complete/' + kind)], np.int64)
    within = np.asarray([registry.kind(kind) for kind in registry.nodes if enabled(spec, 'derive/complete_within/' + kind)], np.int64)
    if (not projected and not len(selected) and spec.get('derive/hops', 1) == 1 and spec.get('derive/diffusion', 'none') == 'none'
            and spec.get('derive/rewiring', 'none') == 'none'):
        return graph
    stated = graph.fact_present & np.isin(graph.fact_code, codes(registry, names))
    kept = graph.fact_present & np.isin(graph.fact_code, codes(registry, [name for name in names if enabled(spec, 'relations/%s/keep' % name)]))
    fact_bounds = np.searchsorted(graph.fact_step, np.arange(steps + 1))
    observed_bounds = np.searchsorted(graph.obs_step, np.arange(steps + 1))
    recency = np.zeros(len(graph.fact_step), np.int64)
    statements = np.flatnonzero(stated)
    ids, count = kernels.keyed(graph.fact_code[statements], graph.fact_source[statements], graph.fact_target[statements])
    recency[statements] = kernels.runs(ids, graph.fact_step[statements], count)[1]
    chunks, removed, derived = [], np.zeros(len(graph.fact_step), bool), 0
    for step in range(steps):
        span = np.arange(fact_bounds[step], fact_bounds[step + 1])
        every, rows = span[stated[span]], span[kept[span]]
        source, target = graph.fact_source[rows], graph.fact_target[rows]
        lo, hi = observed_bounds[step], observed_bounds[step + 1]
        present = np.sort(graph.obs_entity[lo:hi][~graph.obs_anchor[lo:hi]])
        u, v = undirected(source, target, entities)
        found = []
        for edge in projected:
            a, b = project(graph, registry, edge, graph.fact_source[every], graph.fact_target[every], graph.fact_code[every], recency[every],
                           spec['derive/project_cap'])
            found.append((a, b, np.full(len(a), edge.code, np.int64)))
        if len(selected):
            found.append(complete(graph, registry, selected, within, present, u, v))
        if spec.get('derive/hops', 1) > 1:
            found.extend(closure(registry, u, v, spec['derive/hops'], entities))
        if spec.get('derive/diffusion', 'none') != 'none':
            found.append(diffuse(registry, spec, present, u, v, entities))
        if spec.get('derive/rewiring', 'none') != 'none':
            (a, b), cut = rewire(spec, present, u, v, entities, np.random.default_rng([seed, REWIRE, step]))
            found.append((a, b, np.full(len(a), registry.code('rewire'), np.int64)))
            keys = np.minimum(source, target).astype(np.int64) * entities + np.maximum(source, target)
            removed[rows[np.isin(keys, cut)]] = True
        chunks.extend((a, b, c, np.full(len(a), step, np.int32)) for a, b, c in found)
        derived += sum(len(a) for a, _, _ in found)
        if derived > compute.FACTS:
            raise ValueError('derived relations of graph %s exceed %d facts by step %d' % (graph.name, compute.FACTS, step))
    graph = prune(graph, ~removed)
    source, target, code, step = (np.concatenate([chunk[i] for chunk in chunks] + [np.zeros(0, dtype)])
                                  for i, dtype in enumerate((np.int32, np.int32, np.int64, np.int32)))
    order = np.lexsort((step, target, source, code))
    source, target, code, step = source[order], target[order], code[order], step[order]
    continues = np.r_[(code[1:] == code[:-1]) & (source[1:] == source[:-1]) & (target[1:] == target[:-1]) & (step[1:] == step[:-1] + 1), False]
    ends = np.flatnonzero(~continues & (step + 1 < steps))
    width = graph.fact_values.shape[1]
    total = len(step) + len(ends)
    return extend(graph, np.r_[source, source[ends]], np.r_[target, target[ends]], np.r_[code, code[ends]], np.r_[step, step[ends] + 1],
                  np.r_[np.ones(len(step), bool), np.zeros(len(ends), bool)], np.full((total, width), np.nan, np.float32))


def untype(graph, registry):
    base = np.isin(graph.fact_code, codes(registry, registry.extras['base']))
    typed, typed_count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
    pairs, pair_count = kernels.keyed(graph.fact_source, graph.fact_target, np.zeros(len(base), np.int64))
    kept = kernels.untyped(typed, pairs, graph.fact_present, base, typed_count, pair_count)
    values = graph.fact_values.copy()
    values[base] = np.nan
    graph = graph.replace(fact_code=np.where(base, registry.code(UNTYPED), graph.fact_code), fact_values=values)
    return select(graph, FACTS, kept)


def relations(graph, registry, spec, seed):
    dropped = [name for name in registry.extras['base'] if not enabled(spec, 'relations/%s/keep' % name)]
    if not dropped:
        return graph
    return select(graph, FACTS, ~np.isin(graph.fact_code, codes(registry, dropped)))


def merging(graph, registry, spec, seed):
    if spec.get('relations/types') == 'untyped':
        graph = untype(graph, registry)
    if spec.get('relations/parallel') != 'merge':
        return graph
    ids, count = kernels.keyed(graph.fact_step.astype(np.int64) * 2 + graph.fact_present, graph.fact_code * len(graph.kinds) + graph.fact_source,
                               graph.fact_target)
    return select(graph, FACTS, kernels.last_rows(ids, count))


def symmetric(registry, spec):
    return np.asarray([edge.symmetric for edge in registry.edges] + [False] * (registry.relations - len(registry.edges))) | (spec.get('relations/direction') == 'symmetric')


def filtering(spec, name):
    prefix = 'edges/%s/' % name
    return prefix + 'drop' in spec and (spec[prefix + 'drop'] > 0 or spec[prefix + 'persistence'] > 1 or spec[prefix + 'cap'] > 0)


def edges(graph, registry, spec, seed):
    named = [edge for edge in registry.edges if filtering(spec, edge.name)]
    if not named:
        return graph
    kept = np.ones(len(graph.fact_step), bool)
    for edge in named:
        rows = np.flatnonzero(graph.fact_code == edge.code)
        kept[rows] = filtered(graph, registry, spec, seed, edge.name, rows)
    return prune(graph, kept)


def filtered(graph, registry, spec, seed, name, rows):
    prefix = 'edges/%s/' % name
    code, step, present = graph.fact_code[rows], graph.fact_step[rows], graph.fact_present[rows]
    source, target = graph.fact_source[rows], graph.fact_target[rows]
    both = symmetric(registry, spec)[code]
    first, second = np.where(both, np.minimum(source, target), source), np.where(both, np.maximum(source, target), target)
    kept = unit(seed, DROP, first, second, code) >= spec[prefix + 'drop']
    stated = np.flatnonzero(kept & present)
    ids, count = kernels.keyed(code[stated], first[stated], second[stated])
    position, _ = kernels.runs(ids, step[stated], count)
    lasting = position >= spec[prefix + 'persistence'] - 1
    kept[stated[~lasting]] = False
    if spec[prefix + 'cap'] == 0:
        return kept
    stated = stated[lasting]
    low, high = first[stated], second[stated]
    ids, count = kernels.keyed(code[stated], low, high)
    _, begun = kernels.runs(ids, step[stated], count)
    order = spec[prefix + 'cap_order']
    if order == 'recent':
        primary, secondary = -begun, rows[stated]
    elif order == 'random':
        primary, secondary = unit(seed, RANDOM, low, high, code[stated]), rows[stated]
    else:
        value = graph.fact_values[rows[stated], registry.attributes(name).index(order.split(':', 1)[1])]
        primary, secondary = np.where(np.isnan(value), np.inf, -value), -begun
    pair, count = kernels.keyed(step[stated].astype(np.int64) * registry.relations + code[stated], low, high)
    best = kernels.smallest(pair, primary.astype(np.float64), count)
    earliest = kernels.smallest(pair, secondary.astype(np.float64), count)
    pair_step, pair_code, pair_low, pair_high, mode = (np.zeros(count, np.int64) for _ in range(5))
    pair_step[pair], pair_code[pair], pair_low[pair], pair_high[pair] = step[stated], code[stated], low, high
    mode[pair] = np.where(both[stated], 2, 1 if spec.get('relations/direction') == 'inverse' else 0)
    extra = np.flatnonzero((mode == 1) | ((mode == 2) & (pair_low != pair_high)))
    owners = np.r_[np.arange(count), extra]
    flag = np.r_[np.zeros(count, np.int64), mode[extra] == 1]
    owner = np.r_[pair_high, pair_low[extra]]
    part = np.r_[np.zeros(count, np.int64), np.ones(len(extra), np.int64)]
    group, groups = kernels.keyed((pair_step[owners] * registry.relations + pair_code[owners]) * 2 + flag, owner, np.zeros(len(owners), np.int64))
    within = kernels.within_cap(group, groups, best[owners], earliest[owners], part, pair_low[owners], pair_high[owners], spec[prefix + 'cap'])
    capped = np.ones(count, bool)
    capped[owners[~within]] = False
    kept[stated] = capped[pair]
    return kept
