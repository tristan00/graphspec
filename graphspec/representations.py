import logging
import math
import time

import numpy as np

from graph_tuning_infra.features import token

from . import kernels
from .operators import enabled
from .vocabulary import BOUNDARIES


LOG = logging.getLogger(__name__)
EVENT_KINDS = ('observation', 'relation', 'query')
WALK_SAMPLES, WALK_HOPS, TURN_LOOKBACK = 32, 4, 3
WALK_STREAM, TURN_STREAM = 16, 32


def lags(maximum):
    return tuple(2 ** i for i in range(int(math.log2(maximum)) + 1))


def replay(entity, step, altered, present, present_step, absent, absent_step, steps, decisions, name, entities):
    empty = np.zeros(0, np.int64)
    actor, subject, category = (decisions[0], decisions[1], decisions[3]) if decisions else (empty, empty, empty)
    context = decisions[2] if decisions else np.zeros((0, 3), np.int64)
    events, status = kernels.replayed(entity.astype(np.int64), step.astype(np.int64), altered, present[:, 0].astype(np.int64), present[:, 1].astype(np.int64),
                                      present[:, 2].astype(np.int64), present_step.astype(np.int64), absent[:, 0].astype(np.int64),
                                      absent[:, 1].astype(np.int64), absent[:, 2].astype(np.int64), absent_step.astype(np.int64),
                                      np.asarray(actor, np.int64), np.asarray(subject, np.int64), np.ascontiguousarray(context, np.int64),
                                      np.asarray(category, np.int64), steps, entities)
    if status < 0:
        raise ValueError('an absent fact of graph %s names an entity before its first observation' % name)
    return events


def base(graph, registry, spec, categories):
    steps, kinds = graph.steps, graph.kinds
    step, entity, x = graph.obs_step, graph.obs_entity, graph.obs_x
    lookup = step.astype(np.int64) * len(kinds) + entity
    by_key = np.argsort(lookup, kind='stable')
    sorted_keys = lookup[by_key]

    def observed(owners, moments):
        probe = np.asarray(moments, np.int64) * len(kinds) + owners
        found = by_key[np.minimum(np.searchsorted(sorted_keys, probe), len(by_key) - 1)]
        if not np.array_equal(lookup[found], probe):
            raise ValueError('a row of graph %s names an entity with no observation at its step' % graph.name)
        return found.astype(np.int32)

    present = graph.fact_present
    codes, fact_step, values = graph.fact_code[present], graph.fact_step[present], graph.fact_values[present]
    source, target = observed(graph.fact_source[present], fact_step), observed(graph.fact_target[present], fact_step)
    symmetric = np.asarray([edge.symmetric for edge in registry.edges] + [False] * (registry.relations - len(registry.edges)))[codes]
    direction = spec.get('relations/direction')
    doubled = symmetric | (direction != 'directed')
    reverse = np.where(symmetric | (direction == 'symmetric'), codes, registry.flipped[codes])
    index = np.repeat(np.arange(len(codes)), np.where(doubled, 2, 1))
    flip = np.r_[False, index[1:] == index[:-1]] if len(index) else np.zeros(0, bool)
    spatial = np.stack((np.where(flip, target[index], source[index]), np.where(flip, source[index], target[index]),
                        np.where(flip, reverse[index], codes[index])), 1).astype(np.int32)
    attribute_row = np.full(len(codes), -1, np.int32)
    by_code = np.argsort(spatial[:, 2], kind='stable')
    arrays = dict(spatial_order=by_code.astype(np.int32),
                  spatial_code_offsets=np.searchsorted(spatial[by_code, 2], np.arange(registry.relations + 1)).astype(np.int64))
    for name in registry.attributed:
        edge = registry.by_name[name]
        use = codes == edge.code
        attribute_row[use] = np.arange(int(use.sum()))
        arrays['attributes_' + name] = values[use, :len(edge.attributes)].astype(np.float32)
    decided, width = registry.extras['queries'], len(categories)
    moments = np.arange(steps)
    if decided:
        decision_actor, decision_subject, category = graph.query_actor, graph.query_subject, graph.query_category
        decision_context = np.stack((observed(decision_actor, moments), observed(decision_subject, moments), observed(graph.query_context, moments)), 1)
    if registry.extras['level'] == 'node':
        actor = subject = graph.label_entity
        row_step = graph.label_step
        context = observed(actor, row_step)
        query_context = np.stack((context, context, context), 1)
        row_index, decision, signals = graph.label_row, graph.label_id, np.zeros((len(actor), 0, 2), np.float32)
    else:
        actor, subject, query_context, row_step = decision_actor, decision_subject, decision_context, moments
        row_index = np.full(steps, -1, np.int64)
        row_index[graph.label_step] = graph.label_row
        decision, signals = graph.query_key, graph.query_signals
    counts, totals, affinity, global_affinity = {}, np.zeros(width, np.float32), [], []
    bounds = np.searchsorted(row_step, np.arange(steps + 1))
    for s in range(steps):
        for r in range(bounds[s], bounds[s + 1]):
            affinity.append(counts.setdefault(int(actor[r]), np.zeros(width, np.float32)).copy())
            global_affinity.append(totals.copy())
        if decided:
            counts.setdefault(int(decision_actor[s]), np.zeros(width, np.float32))[category[s]] += 1
            totals[category[s]] += 1
    absent = ~graph.fact_present
    order = np.argsort(entity, kind='stable')
    altered = np.ones(len(entity), bool)
    if len(order) > 1:
        owners, vectors = entity[order], x[order]
        altered[order[1:]] = (owners[1:] != owners[:-1]) | np.any(vectors[1:] != vectors[:-1], 1)
    events = replay(entity, step, altered, np.stack((source, target, codes), 1), fact_step,
                    np.stack((graph.fact_source[absent], graph.fact_target[absent], graph.fact_code[absent]), 1), graph.fact_step[absent], steps,
                    (decision_actor, decision_subject, decision_context, category) if decided else None, graph.name, len(kinds))
    arrays.update(x=x.astype(np.float16, copy=False), entity=entity.astype(np.int32),
                  entity_token=np.asarray([token(registry.nodes[kind] + ':' + key) & ((1 << 62) - 1) for kind, key in zip(kinds.tolist(), graph.keys)], np.int64), step=step.astype(np.int32), turn=graph.period[step].astype(np.int32),
                  snapshot_offsets=np.searchsorted(step, np.arange(steps + 1)).astype(np.int64),
                  entity_kind=(np.zeros_like(kinds) if enabled(spec, 'nodes/untyped') else kinds).astype(np.int8),
                  spatial=spatial, spatial_attribute_row=attribute_row[index], spatial_step=fact_step[index].astype(np.int32),
                  observation_ids=np.argsort(entity, kind='stable').astype(np.int32),
                  observation_offsets=np.r_[0, np.cumsum(np.bincount(entity, minlength=len(kinds)))].astype(np.int64),
                  query=(graph.query_x if decided else np.zeros((0, registry.features))).astype(np.float16, copy=False),
                  decision_actor=(decision_actor if decided else np.zeros(0)).astype(np.int32),
                  actor=np.asarray(actor, np.int32), subject=np.asarray(subject, np.int32), query_context=np.asarray(query_context, np.int32).reshape(-1, 3),
                  query_step=np.asarray(row_step, np.int32), events=events,
                  affinity=np.asarray(affinity, np.float32).reshape(len(row_step), width),
                  global_affinity=np.asarray(global_affinity, np.float32).reshape(len(row_step), width),
                  signals=np.asarray(signals, np.float32), row_index=np.asarray(row_index, np.int64), decision=np.asarray(decision, np.int64),
                  period=graph.period.astype(np.int32))
    return arrays


def turns(arrays):
    periods = np.asarray(arrays['period'])
    starts = np.r_[0, np.flatnonzero(np.diff(periods)) + 1].astype(np.int32)
    ends = np.r_[starts[1:] - 1, len(periods) - 1].astype(np.int32)
    membership = np.repeat(np.arange(len(starts)), ends - starts + 1).astype(np.int32)
    size = len(arrays['entity_kind'])
    observed = np.full((len(ends), size), -1, np.int32)
    state = np.full(size, -1, np.int32)
    previous = 0
    for turn, end in enumerate(ends):
        stop = int(arrays['snapshot_offsets'][end + 1])
        ids = np.arange(previous, stop)
        np.maximum.at(state, arrays['entity'][ids], ids)
        observed[turn] = state
        previous = stop
    choices = np.asarray(arrays['events'])
    choices = choices[choices[:, 3] == 2]
    moments = choices[:, 2] // 2
    categories = np.zeros(len(periods), np.int32)
    categories[moments] = choices[:, 6]
    actor_decision = np.full((len(ends), size), -1, np.int32)
    last_actor = np.full(size, -1, np.int32)
    for turn, (start, end) in enumerate(zip(starts, ends)):
        lo, hi = np.searchsorted(moments, (start, end + 1))
        np.maximum.at(last_actor, choices[lo:hi, 0], moments[lo:hi])
        actor_decision[turn] = last_actor
    return dict(turn_starts=starts, turn_ends=ends, turn_membership=membership, turn_actor_decision=actor_decision,
                turn_action_category=categories, turn_numbers=periods[starts], turn_observations=observed)


def marked_edges(pieces, turn, codes):
    if not pieces:
        return np.empty((0, 4), np.int32)
    values = np.concatenate(pieces).astype(np.int64)
    if not len(values):
        return np.empty((0, 4), np.int32)
    ids, count = kernels.keyed2(values[:, 0], values[:, 1])
    flags = kernels.combined(ids, values[:, 2], count)
    kept = kernels.first_rows(ids, count)
    source, destination = values[kept, 0], values[kept, 1]
    return ordered(source, destination, codes[flags[ids[kept]]], turn[destination] - turn[source], int(destination.max(initial=-1)) + 1)


def ordered(source, destination, relation, gap, size):
    order = kernels.by_target(np.asarray(source, np.int64), np.asarray(destination, np.int64), size)
    output = np.empty((len(order), 4), np.int32)
    for column, values in enumerate((source, destination, relation, gap)):
        output[:, column] = np.asarray(values)[order]
    return output


def boundary_codes(registry):
    return np.asarray([-1] + [registry.code(name) for name in BOUNDARIES], np.int32)


def coupling(arrays, registry, spec):
    step, entity = arrays['step'], arrays['entity']
    scheme = spec['coupling/scheme']
    chosen, width, cap, limit = np.zeros(0, np.int64), 0, 0, 0
    if scheme in ('window', 'all'):
        mode = 0
        width, cap = (spec['coupling/window'], spec['coupling/cap']) if scheme == 'window' else (len(arrays['period']), len(entity))
    else:
        chosen = np.asarray((1,) if scheme == 'ordinal' else lags(spec['coupling/max_lag']), np.int64)
        if spec['coupling/unit'] == 'observation':
            mode = 1
            limit = spec['coupling/lag_window'] if scheme == 'lags' else len(arrays['period'])
        else:
            mode = 2
    source, target = kernels.coupled(np.ascontiguousarray(entity, np.int64), np.ascontiguousarray(step, np.int64), chosen, mode,
                                     width, cap, limit, len(arrays['entity_kind']))
    periods = spec['coupling/periods']
    return dict(coupling=np.stack((source, target), 1).astype(np.int32).reshape(-1, 2),
                coupling_boundaries=observation_boundaries(arrays, registry, periods, spec['coupling/period_window']) if periods else np.zeros((0, 4), np.int32))


def observation_boundaries(arrays, registry, lookback, window):
    step, entity = arrays['step'], arrays['entity']
    turns = arrays['period']
    membership = np.cumsum(np.r_[0, np.diff(turns) != 0]).astype(np.int32)
    groups = membership[step]
    shape = (int(membership[-1]) + 1, len(arrays['entity_kind']))
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
    return marked_edges(pieces, turns[step], boundary_codes(registry))


def event_graph(arrays, registry, spec):
    events = np.asarray(arrays['events'])
    member = np.isin(events[:, 3], [EVENT_KINDS.index(kind) for kind in spec['events/kinds'].split('+')])
    link, window, cap = spec['events/link'], spec['events/window'], spec['events/cap']
    chosen = np.asarray((1,) if link == 'next' else lags(spec['events/max_lag']) if link == 'lags' else (), np.int64)
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    before, after = kernels.event_dag(source, target, ticks, member, chosen, link == 'window', window, cap, len(arrays['entity_kind']))
    periods = spec['events/periods']
    return dict(event_dag=np.stack((before, after), 1).astype(np.int32).reshape(-1, 2), event_member=member,
                event_boundaries=event_boundaries(arrays, registry, member, periods, spec['events/period_window']) if periods else np.zeros((0, 4), np.int32))


def event_boundaries(arrays, registry, member, lookback, window):
    events = np.asarray(arrays['events'])
    turns = arrays['period']
    membership = np.cumsum(np.r_[0, np.diff(turns) != 0]).astype(np.int32)
    moments = events[:, 2] // 2
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    before, after, flags = kernels.boundaries(source, target, ticks, member, membership[moments].astype(np.int64), lookback, window,
                                              len(arrays['entity_kind']))
    turn = turns[moments]
    return ordered(before, after, boundary_codes(registry)[flags], turn[after] - turn[before], len(events))


def neighbors(arrays, spec):
    events, kinds = arrays['events'], spec['neighbors/kinds'].split('+')
    use = (events[:, 3] == 2) & ('query' in kinds)
    if 'relation' in kinds:
        use |= (events[:, 3] == 1) & (spec['neighbors/removals'] | (events[:, 8] == 1))
    ids = np.flatnonzero(use)
    source, target = events[ids, 0], events[ids, 1]
    distinct = target != source
    owner, event = np.r_[source, target[distinct]], np.r_[ids, ids[distinct]]
    order = np.lexsort((event, owner))
    return dict(neighbor_ids=event[order].astype(np.int32),
                neighbor_offsets=np.r_[0, np.cumsum(np.bincount(owner, minlength=len(arrays['entity_kind'])))].astype(np.int64))


def walks(arrays, spec, seed):
    bias, window, fanout = spec['walks/bias'], spec['walks/window'], spec['walks/fanout']
    rows = len(arrays['query_step'])
    roots = np.stack((arrays['actor'], arrays['subject']), 1).astype(np.int64)
    query = np.asarray(arrays['query_step'], np.int64)
    events = np.asarray(arrays['events'])
    source, target, ticks = (np.ascontiguousarray(events[:, column].astype(np.int64)) for column in range(3))
    ids, offsets = np.asarray(arrays['neighbor_ids']), np.asarray(arrays['neighbor_offsets'])
    row, side, draw = (axis.reshape(-1) for axis in np.meshgrid(np.arange(rows), np.arange(2), np.arange(WALK_SAMPLES), indexing='ij'))
    identity = np.stack((roots[row, side], query[row], side, draw), 1)
    result = kernels.walked(ids, offsets, source, target, ticks, roots[row, side], np.maximum(0, (query[row] - window + 1) * 2), query[row] * 2 + 1,
                            np.ones(len(row), np.bool_), identity, fanout, bias, seed, WALK_STREAM, WALK_HOPS)
    prior = arrays['turn_membership'][query][:, None] - np.arange(1, TURN_LOOKBACK + 1)
    ends, starts = arrays['turn_ends'][prior.clip(0)].astype(np.int64), arrays['turn_starts'][prior.clip(0)].astype(np.int64)
    row, side, back, draw = (axis.reshape(-1) for axis in np.meshgrid(np.arange(rows), np.arange(2), np.arange(TURN_LOOKBACK), np.arange(WALK_SAMPLES), indexing='ij'))
    identity = np.stack((roots[row, side], query[row], side, back, draw), 1)
    turn_walks = kernels.walked(ids, offsets, source, target, ticks, roots[row, side], 2 * starts[row, back], 2 * ends[row, back] + 2,
                                np.ascontiguousarray(prior[row, back] >= 0), identity, fanout, bias, seed, TURN_STREAM, WALK_HOPS)
    return dict(walks=result.reshape((rows, 2, WALK_SAMPLES, WALK_HOPS)), turn_walks=turn_walks.reshape((rows, 2, TURN_LOOKBACK, WALK_SAMPLES, WALK_HOPS)))


def mixed(arrays, spec):
    x, entity, offsets = arrays['x'], arrays['entity'], arrays['snapshot_offsets']
    changes, trace, bundles = spec['mixed/changes'], spec['mixed/trace'], spec['mixed/bundles']
    width = x.shape[1] // 2
    decided = len(arrays['query']) > 0
    rows, moments = len(entity), len(offsets) - 1
    order = np.argsort(entity, kind='stable')
    prior = np.full(rows, -1, np.int64)
    if rows > 1:
        same = entity[order][1:] == entity[order][:-1]
        prior[order[1:][same]] = order[:-1][same]
    altered = np.zeros(rows, bool)
    if changes:
        seen = prior >= 0
        altered[seen] = np.any(x[seen] != x[prior[seen]], 1)
    groups = rows // trace if trace else 0
    limit = rows + (rows if changes else 0) + groups + (moments if bundles else 0) + (moments if decided else 0)
    wires = rows + (2 * rows if changes else 0) + groups * (trace + 1) + (rows if bundles else 0) + 2 * moments
    holders = rows + (2 * rows if changes else 0) + groups * trace + (rows if bundles else 0)
    actor = np.asarray(arrays['decision_actor'], np.int32) if decided else np.zeros(0, np.int32)
    kind, tick, owner, mode, fill, member_offset, member_id, source, target, flag = kernels.bundled(
        np.ascontiguousarray(entity, np.int64), np.ascontiguousarray(offsets, np.int64), prior, altered, actor,
        trace, bundles, decided, len(arrays['entity_kind']), limit, wires, holders)
    features = np.zeros((len(kind), x.shape[1]), np.float16)
    observed = mode == 0
    features[observed] = x[fill[observed]]
    if changes:
        picked = mode == 1
        current = fill[picked]
        features[picked] = np.concatenate((x[current, :width].astype(np.float32) - x[prior[current], :width], x[current, width:]), 1).astype(np.float16)
    if trace:
        picked = np.flatnonzero(mode == 2)
        if len(picked):
            features[picked] = x[member_id[member_offset[picked][:, None] + np.arange(trace)]].mean(1)
    if bundles:
        for node in np.flatnonzero(mode == 5):
            features[node] = x[offsets[fill[node]]:offsets[fill[node] + 1]].mean(0)
    if decided:
        picked = mode == 3
        features[picked] = arrays['query'][fill[picked]]
    return dict(mixed_x=features.reshape(-1, x.shape[1]), mixed_kind=kind, mixed_tick=tick, mixed_entity=owner,
                mixed_edges=np.stack((source, target, flag), 1).astype(np.int32).reshape(-1, 3),
                mixed_member_ids=member_id, mixed_member_offsets=member_offset)


def build(graph, registry, spec, groups, categories, seed):
    started = time.perf_counter()
    arrays = base(graph, registry, spec, categories)
    parts = (('turns', True, lambda: turns(arrays)), ('coupling', 'coupling' in groups, lambda: coupling(arrays, registry, spec)),
             ('events', 'events' in groups, lambda: event_graph(arrays, registry, spec)),
             ('neighbors', 'neighbors' in groups or 'walks' in groups, lambda: neighbors(arrays, spec)),
             ('walks', 'walks' in groups, lambda: walks(arrays, spec, seed)), ('mixed', 'mixed' in groups, lambda: mixed(arrays, spec)))
    for name, wanted, make in parts:
        if wanted:
            clock = time.perf_counter()
            arrays.update(make())
            LOG.info('representation group exit graph=%s group=%s seconds=%.3f', graph.name, name, time.perf_counter() - clock)
    LOG.info('representation exit graph=%s seconds=%.3f events=%d groups=%s', graph.name, time.perf_counter() - started, len(arrays['events']), '+'.join(groups))
    return arrays


def validate(arrays, graph, registry, spec, groups):
    x, step, events, entity = arrays['x'], arrays['step'], arrays['events'], arrays['entity']
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observation features in graph %s' % graph.name)
    codes, rows = arrays['spatial'][:, 2], arrays['spatial_attribute_row']
    by_code = arrays['spatial_order']
    sorted_codes, sorted_rows = codes[by_code], rows[by_code]
    for edge in registry.edges:
        use = np.concatenate([sorted_rows[lo:hi] for lo, hi in np.searchsorted(sorted_codes, np.asarray([[code, code + 1] for code in (edge.code, edge.reverse)], sorted_codes.dtype))])
        if not edge.attributes and np.any(use != -1):
            raise ValueError('relation %s carries attribute rows without attributes' % edge.name)
        if edge.attributes and (np.any(use < 0) or np.any(use >= len(arrays['attributes_' + edge.name]))):
            raise ValueError('attribute row out of range for relation %s' % edge.name)
    turn = arrays['period']
    if 'coupling' in groups:
        source, destination = arrays['coupling'].T
        if np.any(step[source] >= step[destination]) or np.any(entity[source] != entity[destination]):
            raise ValueError('coupling edges of graph %s must link an entity to its own earlier observation' % graph.name)
        source, destination, relation, gaps = arrays['coupling_boundaries'].T
        if (np.any(entity[source] != entity[destination]) or np.any(turn[step[source]] >= turn[step[destination]])
                or not np.isin(relation, boundary_codes(registry)[1:]).all() or not np.array_equal(gaps, turn[step[destination]] - turn[step[source]])
                or np.any(step[destination] - step[source] >= spec['coupling/period_window'] if spec['coupling/periods'] else False)):
            raise ValueError('coupling boundary edges of graph %s are invalid' % graph.name)
    if 'events' in groups:
        member = arrays['event_member']
        source, destination = arrays['event_dag'].T
        if np.any(events[source, 2] >= events[destination, 2]) or not (member[source].all() and member[destination].all()):
            raise ValueError('event graph of graph %s links a later, simultaneous or excluded event' % graph.name)
        if kernels.check_boundaries(np.asarray(arrays['event_boundaries']), np.asarray(events), np.asarray(member), np.asarray(turn), boundary_codes(registry)[1:]):
            raise ValueError('event boundary edges of graph %s are invalid' % graph.name)
    if 'walks' in groups:
        query = np.asarray(arrays['query_step'], np.int64)
        for name, limit in (('walks', query[:, None, None] * 2), ('turn_walks', query[:, None, None, None] * 2)):
            walks = arrays[name]
            times = np.where(walks >= 0, events[np.maximum(walks, 0), 2], -1)
            if np.any(times[..., 0] > limit) or np.any((np.diff(times, axis=-1) >= 0) & (times[..., 1:] >= 0)):
                raise ValueError('%s of graph %s are not strictly causal' % (name, graph.name))
