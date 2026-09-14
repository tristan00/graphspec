import logging
import time

import numpy as np

from .data import ROWS, Frames, Host, joined, length, materialise, recipe
from .turns import enabled


LOG = logging.getLogger(__name__)
ENTITY_FRAMES = 1 << 20
FACTS = 4 << 20
CHUNK = 64


def batches(blocks, size, sizes, seed=None, frame_budget=None, chunk=CHUNK):
    started = time.perf_counter()
    rng = np.random.default_rng(seed) if seed is not None else None
    graphs = list(dict.fromkeys(block[0] for block in blocks))
    if rng is not None:
        rng.shuffle(graphs)
    position = {graph: index // chunk for index, graph in enumerate(graphs)}
    chunks = {}
    for block in blocks:
        chunks.setdefault(position[block[0]], {}).setdefault(block[2] - block[1], []).append(block)
    result = []
    for index in sorted(chunks):
        groups = []
        for length, values in chunks[index].items():
            limit = min(size, max(1, frame_budget // length)) if frame_budget else size
            group, rows, cells, facts = [], 0, 0, 0.0
            for block in values:
                entities, rate = sizes[block[0]]
                width, span = block[6] - block[5], length * entities
                if group and (len(group) == limit or rows + width > ROWS or cells + span > ENTITY_FRAMES or facts + length * rate > FACTS):
                    groups.append(group)
                    group, rows, cells, facts = [], 0, 0, 0.0
                group.append(block)
                rows += width
                cells += span
                facts += length * rate
            groups.append(group)
        if rng is not None:
            rng.shuffle(groups)
        result.extend(groups)
    LOG.info('batches exit seconds=%.3f blocks=%d batches=%d chunks=%d', time.perf_counter() - started, len(blocks), len(result), len(chunks))
    return result


def combine_tree(trees):
    return {key: combine_tree([tree[key] for tree in trees]) if key == 'child' else joined([tree[key] for tree in trees]) for key in trees[0]}


def recipe_batch(bank, blocks, cfg):
    started = time.perf_counter()
    parts = [recipe(bank, block, cfg) for block in blocks]
    t, size = parts[0]['frames'], len(parts)
    if any(p['frames'] != t for p in parts):
        raise ValueError('a batch must contain equal-length independent sequences')
    entity_offsets = np.r_[0, np.cumsum([p['entities'] for p in parts])]
    nodes = np.r_[0, np.cumsum([p['observations'] for p in parts])]
    row_offsets = np.r_[0, np.cumsum([p['predictions'] for p in parts])]
    total_entities = int(entity_offsets[-1])
    b = dict(frames=t * size, time_steps=t, batch_size=size, predictions=int(row_offsets[-1]), entities=total_entities, start=0, stop=t,
        graph=parts[0]['graph'], graphs=[p['graph'] for p in parts],
        entity_sizes=[p['entities'] for p in parts],
        entity_batch=np.repeat(np.arange(size), [p['entities'] for p in parts]),
        query_steps=np.concatenate([p['query_steps'] for p in parts]),
        query_frames=np.concatenate([p['query_steps'] + i * t for i, p in enumerate(parts)]),
        frame_queries=[np.concatenate([p['frame_queries'][time] + int(offset) for p, offset in zip(parts, row_offsets)]) for time in range(t)],
        rows=Host(np.concatenate([p['rows'].value for p in parts])))
    direct = ('selected', 'target', 'query', 'context', 'entity_kind', 'signal', 'affinity', 'global_affinity',
        'x', 'kind', 'relation', 'temporal_gap', 'initial', 'history_x', 'history_mask',
        'history_gap', 'occurrence', 'structure', 'common_stats',
        'common_frequency', 'pair_gap', 'pair_mask', 'walk_x', 'walk_mask', 'walk_gap', 'walk_identity', 'dag_gap',
        'end_gaps', 'end_mask', 'end_raw', 'end_raw_mask', 'end_token_gaps', 'temporal_relation', 'temporal_turn_gap', 'dag_relation',
        'history_turn_kind', 'history_end_time', 'end_affinity', 'end_global_affinity', 'walk_turn_kind', 'walk_end_time', 'decision_query')
    for key in direct:
        if key in parts[0]:
            b[key] = joined([p[key] for p in parts])
    if 'end_steps' in parts[0]:
        b['end_steps'] = np.concatenate([np.where(p['end_steps'] >= 0, p['end_steps'] + i * t, -1) for i, p in enumerate(parts)])
        b['end_observation_local'] = np.concatenate([np.where(p['end_observation_local'] >= 0,
            p['end_observation_local'] + int(offset), -1) for p, offset in zip(parts, nodes)])
    for key in ('actor', 'subject', 'entity', 'history_ids', 'common_ids'):
        if key in parts[0]:
            b[key] = np.concatenate([np.where(p[key] >= 0, p[key] + int(offset), p[key]) for p, offset in zip(parts, entity_offsets)])
    if 'attributes' in parts[0]:
        edge_offsets = np.r_[0, np.cumsum([len(p['relation']) for p in parts])]
        names = list(parts[0]['attributes'])
        b['attribute_values'] = np.concatenate([p['attributes'][name].reshape(-1) for name in names for p in parts])
        b['attribute_edges'] = np.concatenate([p['attribute_edges'][name] + int(offset) for name in names for p, offset in zip(parts, edge_offsets)])
        b['attribute_counts'] = Host([sum(len(p['attributes'][name]) for p in parts) for name in names])
    if 'x' in b:
        b['context_index'] = np.concatenate([p['context_index'] + int(offset) for p, offset in zip(parts, nodes)])
        b['decision_context'] = np.concatenate([np.where(p['decision_context'] >= 0, p['decision_context'] + int(offset), -1) for p, offset in zip(parts, nodes)])
        b['step'] = np.concatenate([p['step'] + i * t for i, p in enumerate(parts)])
        b['align'] = np.concatenate([np.where(p['align'] < p['observations'], p['align'] + int(offset), int(nodes[-1])) for p, offset in zip(parts, nodes)], 1)
        b['present'] = np.concatenate([p['present'] for p in parts], 1)
        for key in ('edges', 'temporal_edges'):
            if key in parts[0]:
                b[key] = np.concatenate([p[key] + int(offset) for p, offset in zip(parts, nodes)], 1)
        edge_offsets = np.r_[0, np.cumsum([len(p['relation']) for p in parts])]
        canonical_offsets = np.r_[0, np.cumsum([int(p['frame_edges'].directed.sum()) for p in parts])]
        directed = np.concatenate([p['frame_edges'].directed for p in parts])
        for key, offsets, canonical in (('frame_edges', edge_offsets, False), ('frame_directed', canonical_offsets, True)):
            b[key] = Frames(directed, [[(lo + int(offset), hi + int(offset)) for p, offset in zip(parts, offsets) for lo, hi in p[key].ranges[step]] for step in range(t)], canonical)
    if 'tree' in parts[0]:
        b['tree'] = combine_tree([p['tree'] for p in parts])
    if 'end_tree' in parts[0]:
        b['end_tree'] = combine_tree([p['end_tree'] for p in parts])
    if 'mixed' in parts[0]:
        offsets = np.r_[0, np.cumsum([length(p['mixed']['x']) for p in parts])]
        b['mixed'] = {key: joined([p['mixed'][key] for p in parts]) for key in ('x', 'kind', 'edge_code', 'edge_gap')}
        b['mixed']['edge_index'] = np.concatenate([p['mixed']['edge_index'] + int(offset) for p, offset in zip(parts, offsets)], 1)
        b['mixed']['query'] = np.concatenate([p['mixed']['query'] + int(offset) for p, offset in zip(parts, offsets)])
    if 'events' in parts[0]:
        events = []
        for p, offset in zip(parts, entity_offsets):
            event = p['events'].copy()
            event[:, :2] += int(offset)
            event[:, 2] -= p['start'] * 2
            events.append(event)
        events = np.concatenate(events)
        order = np.argsort(events[:, 2], kind='stable')
        b['events'], b['event_x'] = events[order], joined([p['event_x'] for p in parts], order=order)
        times = b['events'][:, 2]
        starts = np.flatnonzero(np.diff(times, prepend=-1))
        b['event_groups'] = [(int(times[left]), int(left), int(right)) for left, right in zip(starts, np.append(starts[1:], len(times)))]
        if 'dag_edges' in parts[0]:
            offsets = np.r_[0, np.cumsum([len(p['events']) for p in parts])]
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            b['dag_edges'] = inverse[np.concatenate([p['dag_edges'] + int(offset) for p, offset in zip(parts, offsets)], 1)]
            b['dag_query_edges'] = np.concatenate([np.stack((inverse[p['dag_query_edges'][0] + int(offset)], p['dag_query_edges'][1] + int(row_offset)))
                for p, offset, row_offset in zip(parts, offsets, row_offsets)], 1)
    if enabled(cfg):
        from .turn_inputs import capture_schedule

        capture_schedule(b)
    LOG.debug('recipe batch exit blocks=%d predictions=%d seconds=%.4f', size, b['predictions'], time.perf_counter() - started)
    return b


def pack_batch(bank, blocks, cfg, device):
    return materialise(bank, cfg, recipe_batch(bank, blocks, cfg), device)
