import numpy as np
import torch

from .data import pack


def batches(blocks, size, seed=None, frame_budget=None):
    buckets = {}
    for block in blocks:
        buckets.setdefault(block[2] - block[1], []).append(block)
    result = []
    for length, values in buckets.items():
        count = min(size, max(1, frame_budget // length)) if frame_budget else size
        result.extend(values[i:i + count] for i in range(0, len(values), count))
    if seed is not None:
        np.random.default_rng(seed).shuffle(result)
    return result


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def combine_tree(trees):
    return {key: combine_tree([tree[key] for tree in trees]) if key == 'child' else torch.cat([tree[key] for tree in trees]) for key in trees[0]}


def pack_batch(bank, blocks, cfg, device='cuda'):
    parts = [pack(bank, block, cfg, 'cpu') for block in blocks]
    t, size = parts[0]['frames'], len(parts)
    if any(p['frames'] != t for p in parts):
        raise ValueError('a batch must contain equal-length independent sequences')
    entity_offsets = np.r_[0, np.cumsum([p['entities'] for p in parts])]
    nodes = np.r_[0, np.cumsum([len(p.get('x', [])) for p in parts])]
    total_entities = int(entity_offsets[-1])
    b = dict(frames=t * size, time_steps=t, batch_size=size, entities=total_entities, start=0, stop=t,
        campaign=parts[0]['campaign'], campaigns=[p['campaign'] for p in parts],
        entity_sizes=[p['entities'] for p in parts],
        entity_batch=torch.repeat_interleave(torch.arange(size), torch.tensor([p['entities'] for p in parts])),
        query_steps=torch.arange(t).repeat(size), frame_queries=torch.arange(t * size).reshape(size, t).T,
        rows=np.concatenate([p['rows'] for p in parts]))
    direct = ('selected', 'target', 'query', 'context', 'entity_kind', 'signal', 'affinity', 'global_affinity',
        'x', 'kind', 'relation', 'attributes', 'temporal_gap', 'initial', 'history_x', 'history_mask',
        'history_gap', 'history_events', 'occurrence', 'structure', 'common_x', 'common_stats',
        'common_frequency', 'pair_gap', 'pair_mask', 'walk_x', 'walk_mask', 'walk_gap', 'walk_identity', 'dag_gap',
        'end_gaps', 'end_mask', 'end_raw', 'end_raw_mask', 'end_token_gaps', 'temporal_relation', 'temporal_turn_gap', 'dag_relation',
        'history_turn_kind', 'history_end_time', 'end_affinity', 'end_global_affinity', 'walk_turn_kind', 'walk_end_time')
    for key in direct:
        if key in parts[0]:
            b[key] = torch.cat([p[key] for p in parts])
    if 'end_steps' in parts[0]:
        b['end_steps'] = torch.cat([torch.where(p['end_steps'] >= 0, p['end_steps'] + i * t, -1) for i, p in enumerate(parts)])
        b['end_observation_local'] = torch.cat([torch.where(p['end_observation_local'] >= 0,
            p['end_observation_local'] + int(offset), -1) for p, offset in zip(parts, nodes)])
    for key in ('actor', 'subject', 'entity', 'history_ids', 'common_ids'):
        if key in parts[0]:
            b[key] = torch.cat([torch.where(p[key] >= 0, p[key] + int(offset), p[key]) for p, offset in zip(parts, entity_offsets)])
    if 'x' in b:
        b['context_index'] = torch.cat([p['context_index'] + int(offset) for p, offset in zip(parts, nodes)])
        b['step'] = torch.cat([p['step'] + i * t for i, p in enumerate(parts)])
        b['align'] = torch.cat([torch.where(p['align'] < len(p['x']), p['align'] + int(offset), int(nodes[-1])) for p, offset in zip(parts, nodes)], 1)
        b['present'] = torch.cat([p['present'] for p in parts], 1)
        for key in ('edges', 'temporal_edges'):
            if key in parts[0]:
                b[key] = torch.cat([p[key] + int(offset) for p, offset in zip(parts, nodes)], 1)
        for key in ('frame_edges', 'frame_directed'):
            b[key] = [torch.cat([p[key][step] + int(offset) for p, offset in zip(parts, entity_offsets)], 1) for step in range(t)]
    if 'tree' in parts[0]:
        b['tree'] = combine_tree([p['tree'] for p in parts])
    if 'end_tree' in parts[0]:
        b['end_tree'] = combine_tree([p['end_tree'] for p in parts])
    if 'mixed' in parts[0]:
        offsets = np.r_[0, np.cumsum([len(p['mixed']['x']) for p in parts])]
        b['mixed'] = {key: torch.cat([p['mixed'][key] for p in parts]) for key in ('x', 'kind', 'edge')}
        b['mixed']['edge_index'] = torch.cat([p['mixed']['edge_index'] + int(offset) for p, offset in zip(parts, offsets)], 1)
        b['mixed']['query'] = torch.cat([p['mixed']['query'] + int(offset) for p, offset in zip(parts, offsets)])
    if 'events' in parts[0]:
        events = []
        for p, offset in zip(parts, entity_offsets):
            event = p['events'].clone()
            event[:, :2] += int(offset)
            event[:, 2] -= p['start'] * 2
            events.append(event)
        events = torch.cat(events)
        order = torch.argsort(events[:, 2], stable=True)
        b['events'], b['event_x'] = events[order], torch.cat([p['event_x'] for p in parts])[order]
        times = b['events'][:, 2].numpy()
        boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(times)]
        b['event_groups'] = [(int(times[left]), int(left), int(right)) for left, right in zip(boundaries[:-1], boundaries[1:])]
        if 'dag_edges' in parts[0]:
            offsets = np.r_[0, np.cumsum([len(p['events']) for p in parts])]
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(len(order))
            b['dag_edges'] = inverse[torch.cat([p['dag_edges'] + int(offset) for p, offset in zip(parts, offsets)], 1)]
            b['dag_query_edges'] = torch.cat([torch.stack((inverse[p['dag_query_edges'][0] + int(offset)], p['dag_query_edges'][1] + i * t)) for i, (p, offset) in enumerate(zip(parts, offsets))], 1)
    from .turn_inputs import capture_schedule

    capture_schedule(b)
    return move(b, device)
