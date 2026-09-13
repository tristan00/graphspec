import numpy as np
import torch

from graphspec.transform import SEED

from .data import Features, MixedEdges, array, features, gathered, joined, length, tree
from .turns import lookback


def add(b, graph, held, first, last, start, cfg):
    queries = held.numpy('query_step')[first:last].astype(np.int64)
    membership = held.numpy('turn_membership')[queries]
    previous = membership[:, None] - np.arange(1, lookback(cfg) + 1)
    valid = previous >= 0
    prior = previous.clip(0)
    ends = held.numpy('turn_ends')[prior]
    valid &= ends >= start
    entity, step = held.numpy('entity'), held.numpy('step')
    roots = np.stack((held.numpy('actor')[first:last], held.numpy('subject')[first:last], entity[held.numpy('query_context')[first:last, 2]]), 1)
    observed = held.numpy('turn_observations')[prior[:, :, None], roots[:, None, :]]
    mask = (observed >= 0) & valid[:, :, None]
    age = np.maximum(ends[:, :, None] - step[observed.clip(0)], 0)
    age[~mask] = 0
    delta = queries[:, None] - ends
    numbers = held.numpy('turn_numbers')
    turn_gap = numbers[membership, None] - numbers[prior]
    gaps = np.stack((delta, turn_gap, age.max(-1)), -1).clip(0).astype(np.float32)
    token_gaps = np.stack((queries[:, None, None] - step[observed.clip(0)],
        np.broadcast_to(turn_gap[:, :, None], age.shape), age), -1).clip(0).astype(np.float32)
    local = observed - int(held.numpy('snapshot_offsets')[start])
    index = np.where(mask, observed, -1).transpose(0, 2, 1)
    b.update(end_steps=array(np.where(valid, ends - start, -1), torch.long),
        end_mask=array(valid), end_gaps=gaps,
        end_raw=features(graph, index.shape, ('x', index, 1.0)), end_raw_mask=array(mask.transpose(0, 2, 1)),
        end_token_gaps=array(token_gaps.transpose(0, 2, 1, 3)),
        end_observation_local=array(np.where(mask & (local >= 0), local, -1), torch.long))
    if cfg['model']['view'] == 'affinity':
        decision = held.numpy('turn_actor_decision')[prior, roots[:, 0, None]]
        category = held.numpy('turn_action_category')
        affinity = np.array(held.numpy('affinity')[decision.clip(0)], np.float32)
        affinity += np.eye(affinity.shape[-1], dtype=np.float32)[category[decision.clip(0)]]
        affinity[(decision < 0) | ~valid] = 0
        global_affinity = np.array(held.numpy('global_affinity')[ends], np.float32)
        global_affinity += np.eye(affinity.shape[-1], dtype=np.float32)[category[ends]]
        global_affinity[~valid] = 0
        b.update(end_affinity=affinity, end_global_affinity=global_affinity)


def temporal_edges(b, graph, held, lo, hi, registry):
    temporal, context = registry.code('temporal'), registry.marker('turn_context')
    nodes = np.arange(lo, hi)
    step, entity = held.numpy('step'), held.numpy('entity')
    query = step[nodes]
    membership = held.numpy('turn_membership')[query]
    prior = membership[:, None] - np.arange(1, b['end_steps'].shape[1] + 1)
    valid = prior >= 0
    observed = held.numpy('turn_observations')[prior.clip(0), entity[nodes, None]]
    valid &= observed >= 0
    valid &= held.numpy('turn_ends')[prior.clip(0)] >= b['start']
    source = observed[valid]
    destination = np.broadcast_to(nodes[:, None], observed.shape)[valid]
    ghosts = np.unique(source[source < lo])
    if len(ghosts):
        count = b['observations']
        b['x'] = joined([b['x'], features(graph, (len(ghosts),), ('x', ghosts, None))])
        b['kind'] = joined([b['kind'], gathered(graph, 'entity_kind', entity[ghosts])])
        b['entity'] = np.concatenate((b['entity'], entity[ghosts].astype(np.int64)))
        b['step'] = np.concatenate((b['step'], (step[ghosts] - b['start']).astype(np.int64)))
        b['align'] = np.where(b['align'] == count, count + len(ghosts), b['align'])
        b['observations'] = count + len(ghosts)
    source_local = np.where(source >= lo, source - lo, hi - lo + np.searchsorted(ghosts, source))
    extra = np.stack((source_local, destination - lo))
    pairs = {tuple(edge): int(kind) for edge, kind in zip(b['temporal_edges'].T.tolist(), b['temporal_relation'])}
    for edge in extra.T.tolist():
        pairs.setdefault(tuple(edge), context)
        if pairs[tuple(edge)] == temporal:
            pairs[tuple(edge)] = context
    edges = np.asarray(list(pairs), np.int64).reshape(-1, 2)
    original_ids = np.r_[np.arange(lo, hi), ghosts]
    steps = step[original_ids]
    turns = held.numpy('turn')[original_ids]
    b.update(temporal_edges=array(edges.T, torch.long),
        temporal_relation=array(np.asarray(list(pairs.values())), torch.long),
        temporal_gap=array(steps[edges[:, 1]] - steps[edges[:, 0]], torch.float32),
        temporal_turn_gap=array(turns[edges[:, 1]] - turns[edges[:, 0]], torch.float32))


def mixed_edges(b, graph):
    mixed = b['mixed']
    mask = b['end_raw_mask'].reshape(-1)
    chosen = b['end_raw'].segments[0][2][0][1].reshape(-1)[mask]
    source = np.arange(len(chosen)) + length(mixed['x'])
    destination = np.broadcast_to(mixed['query'][:, None, None], b['end_raw_mask'].shape).reshape(-1)[mask]
    gaps = b['end_token_gaps'].reshape(-1, 3)[mask]
    feature = np.zeros((len(chosen), 16), np.float32)
    feature[:, 12:14] = 1
    feature[:, 14:16] = np.log1p(gaps[:, :2])
    mixed['x'] = joined([mixed['x'], features(graph, (len(chosen),), ('x', chosen, 1.0))])
    mixed['kind'] = np.concatenate((mixed['kind'], np.zeros(len(chosen), np.int64)))
    mixed['edge_index'] = np.concatenate((mixed['edge_index'], np.stack((source, destination))), 1)
    mixed['edge'] = MixedEdges(mixed['edge'].segments + [('raw', feature)])


def history_tokens(b, graph, held, first, last, start, cfg, registry):
    n, depth = last - first, lookback(cfg)
    if cfg['model']['view'] == 'tree':
        roots = np.stack((held.numpy('actor')[first:last], held.numpy('subject')[first:last]), 1)
        roots = np.broadcast_to(roots[:, :, None], (n, 2, depth)).reshape(-1).copy()
        ends = b['end_steps'] + start
        cutoffs = np.broadcast_to((ends * 2 + 1)[:, None, :], (n, 2, depth)).reshape(-1).copy()
        b['end_tree'] = tree(graph, held, roots, cutoffs, start * 2, cfg['model']['tree/neighbors'], cfg['model']['tree/depth'], registry, cfg['graph'], SEED)
        return
    original = b['history_mask'].shape[-1]
    valid = b['end_raw_mask'][:, :2]
    gaps = b['end_token_gaps'][:, :2]
    raw_index = b['end_raw'].segments[0][2][0][1][:, :2]
    (x_key, x_index, x_weight), (q_key, q_index, q_weight) = b['history_x'].segments[0][2]
    pieces = [(x_key, np.concatenate((x_index, raw_index), 2), np.concatenate((x_weight, np.ones(raw_index.shape, np.float32)), 2)),
              (q_key, np.concatenate((q_index, np.full(raw_index.shape, -1, np.int64)), 2), q_weight)]
    b['history_mask'] = np.concatenate((b['history_mask'], valid), 2)
    b['history_gap'] = np.concatenate((b['history_gap'], gaps[..., 0] * 2 + 1), 2)
    end_time = np.zeros((n, 2, original + depth, 3), np.float32)
    end_time[:, :, original:] = gaps
    b['history_end_time'] = end_time
    b['history_turn_kind'] = np.concatenate((np.zeros((n, 2, original), np.int64), np.ones((n, 2, depth), np.int64)), 2)
    for key, width in (('occurrence', 2), ('structure', 12)):
        b[key] = np.concatenate((b[key], np.zeros((n, 2, depth, width), np.float32)), 2)
    order = np.argsort(-np.where(b['history_mask'], b['history_gap'], np.inf), axis=-1, kind='stable')
    for key in ('history_mask', 'history_gap', 'history_end_time', 'history_turn_kind', 'occurrence', 'structure'):
        value = b[key]
        b[key] = np.take_along_axis(value, order if value.ndim == 3 else order[..., None], 2)
    pieces = [(key, np.take_along_axis(index, order, 2), np.take_along_axis(weight, order, 2) if isinstance(weight, np.ndarray) else weight)
              for key, index, weight in pieces]
    b['history_x'] = Features([(graph, (n, 2, original + depth), pieces)])


def capture_schedule(b):
    ends = b['end_steps']
    result = []
    for time_step in range(b['time_steps']):
        query, slot = np.nonzero((ends >= 0) & (ends % b['time_steps'] == time_step))
        result.append(((query * ends.shape[1] + slot).astype(np.int64), np.stack((b['actor'][query], b['subject'][query]), -1)))
    b['end_capture'] = result
