import numpy as np

from graph_tuning_infra.hashing import unit

NEIGHBOURS = 8


def latest(arrays, entities, cutoff):
    result = np.full(len(entities), -1, np.int32)
    for i, eid in enumerate(entities):
        ids = arrays['observation_ids'][arrays['observation_offsets'][eid]:arrays['observation_offsets'][eid + 1]]
        end = np.searchsorted(arrays['step'][ids] * 2, cutoff, side='left')
        if end:
            result[i] = ids[end - 1]
    return result


def neighbors(arrays, entity, cutoff, floor, count, spec, stream):
    ids = arrays['neighbor_ids'][arrays['neighbor_offsets'][entity]:arrays['neighbor_offsets'][entity + 1]]
    times = arrays['events'][ids, 2]
    lo, hi = np.searchsorted(times, (floor, cutoff))
    order = spec['neighbors/order']
    if order == 'recent' or hi - lo <= count:
        return np.asarray(ids[max(lo, hi - count):hi], np.int32)
    candidates = np.asarray(ids[lo:hi], np.int64)
    draws = np.minimum(unit(stream, NEIGHBOURS, np.full(len(candidates), entity), np.full(len(candidates), cutoff), candidates) + 2.0 ** -54, 1 - 2.0 ** -53)
    logits = np.zeros(hi - lo) if order == 'uniform' else -spec['neighbors/decay'] * (cutoff - times[lo:hi]) / 2.0
    keys = logits - np.log(-np.log(draws))
    return np.asarray(ids[lo + np.sort(np.argpartition(-keys, count - 1)[:count])], np.int32)


def other_endpoint(events, ids, entity):
    return np.where(events[ids, 0] == entity, events[ids, 1], events[ids, 0])


def tree(arrays, entities, cutoffs, floor, count, depth, registry, spec, stream):
    observations = np.concatenate([latest(arrays, [eid], cutoff) for eid, cutoff in zip(entities, cutoffs)])
    raw = np.zeros((len(entities), registry.features), np.float32)
    known = observations >= 0
    raw[known] = arrays['x'][observations[known]]
    result = dict(x=raw, kind=np.asarray(arrays['entity_kind'][entities], np.int64))
    if depth:
        children = np.repeat(entities, count)
        times = np.repeat(cutoffs, count)
        mask = np.zeros((len(entities), count), bool)
        gaps = np.zeros((len(entities), count), np.float32)
        relation = np.zeros((len(entities), count), np.int64)
        for i, (eid, cutoff) in enumerate(zip(entities, cutoffs)):
            ids = neighbors(arrays, int(eid), int(cutoff), floor, count, spec, stream)
            size = len(ids)
            if size:
                children[i * count:i * count + size] = other_endpoint(arrays['events'], ids, eid)
                times[i * count:i * count + size] = arrays['events'][ids, 2]
                mask[i, :size] = True
                gaps[i, :size] = cutoff - arrays['events'][ids, 2]
                relation[i, :size] = np.where(arrays['events'][ids, 3] == 1, arrays['events'][ids, 6] + 1, registry.marker('non_relation'))
        result.update(child=tree(arrays, children, times, floor, count, depth - 1, registry, spec, stream), mask=mask, gap=gaps, relation=relation)
    return result
