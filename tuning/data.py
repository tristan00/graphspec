import logging
from pathlib import Path
import time

import numpy as np
import torch

from graphspec import Registry
from graphspec import spec as specs
from graphspec.transform import SEED, SOURCE, materialize
from graphspec.truth import VERSION, Graph
from graph_tuning_infra.files import ArrayStore, load_arrays, read_json, write_json
from graph_tuning_infra.hashing import unit


LOG = logging.getLogger(__name__)
ROWS = 1024
LOADED = 8
NEIGHBOURS = 8


def layout(index, rows, steps, partitions, partition, history, level):
    chosen = np.flatnonzero(rows >= 0)
    chosen = chosen[partitions[rows[chosen]] == partition]
    result = []
    for moment in np.unique(steps[chosen]).tolist():
        start, stop = max(0, moment - history + 1), moment + 1
        if level == 'node':
            members = chosen[steps[chosen] == moment]
            for piece in range(0, len(members), ROWS):
                part = members[piece:piece + ROWS]
                result.append((index, start, stop, moment, partition, int(part[0]), int(part[-1]) + 1))
        else:
            first, last = np.searchsorted(steps, (start, stop))
            result.append((index, start, stop, moment, partition, int(first), int(last)))
    return result


class Bank:

    def __init__(self, directory):
        started = time.perf_counter()
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        if self.manifest['version'] != VERSION:
            raise ValueError('bank %s has version %s; this code reads version %d' % (self.directory, self.manifest['version'], VERSION))
        self.registry = Registry.from_declaration(self.manifest['spec'])
        if self.registry.fingerprint() != self.manifest['spec_sha256']:
            raise ValueError('bank declaration does not reproduce its pinned fingerprint')
        self.level = self.registry.extras['level']
        self.categories = self.manifest['categories']
        supervision = read_json(self.directory / 'supervision.json')
        self.rows, self.split = supervision['rows'], supervision['split']
        self.targets = np.asarray([row['y'] for row in self.rows], np.float32)
        self.partitions = np.zeros(len(self.rows), np.int8)
        self.partitions[self.split['validation_indices']] = 1
        self.row_index, self.row_steps, self.entities = {}, {}, {}
        for graph in self.manifest['graphs']:
            index = graph['index']
            labels = load_arrays(self.truth(index), ('label_row', 'label_step'))
            if self.level == 'node':
                self.row_index[index], self.row_steps[index] = labels['label_row'], labels['label_step']
            else:
                self.row_index[index] = np.full(graph['steps'], -1, np.int64)
                self.row_index[index][labels['label_step']] = labels['label_row']
                self.row_steps[index] = np.arange(graph['steps'], dtype=np.int32)
            self.entities[index] = graph['entities']
        self.variants, self.store = {}, ArrayStore(LOADED)
        LOG.info('bank exit seconds=%.3f graphs=%d rows=%d level=%s', time.perf_counter() - started, len(self.manifest['graphs']), len(self.rows), self.level)

    def truth(self, index):
        return self.directory / 'truth' / ('graph_%04d' % index)

    def variant(self, cfg):
        from .roster import groups

        key = (cfg['model'], specs.digest(cfg['graph']))
        if key not in self.variants:
            chosen = groups(cfg['model'])
            spec = specs.canonical(specs.space(self.registry, chosen), cfg['graph'])
            self.variants[key] = (self.directory / 'variants' / specs.digest(dict(spec=specs.materialized(spec), groups=list(chosen), source=SOURCE)), spec, chosen)
        return self.variants[key]

    def arrays(self, index, cfg):
        root, spec, chosen = self.variant(cfg)

        def build():
            started = time.perf_counter()
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / 'variant.json', dict(spec=specs.materialized(spec), groups=list(chosen), source=SOURCE))
            arrays = materialize(Graph.load(self.truth(index)), self.registry, spec, chosen, self.categories)
            LOG.info('variant exit graph=%d seconds=%.3f arrays=%d', index, time.perf_counter() - started, len(arrays))
            return arrays

        return self.store.open(root / ('graph_%04d' % index), build)

    def blocks(self, partition, history):
        started = time.perf_counter()
        result = []
        for graph in self.manifest['graphs']:
            index = graph['index']
            result.extend(layout(index, self.row_index[index], self.row_steps[index], self.partitions, partition, history, self.level))
        LOG.info('bank blocks exit seconds=%.3f partition=%d history=%d blocks=%d', time.perf_counter() - started, partition, history, len(result))
        return result


def tensor(value, device, dtype=None):
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)


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


def reachable(owners, positions, times, roots, cutoffs):
    order = np.lexsort((positions, owners))
    owners, positions = owners[order], positions[order]
    found = []
    for group, cutoff in zip(roots, cutoffs):
        pieces = [np.zeros(0, np.int64)]
        for root in np.unique(group):
            lo, hi = np.searchsorted(owners, (root, root + 1))
            members = positions[lo:hi]
            pieces.append(members[:np.searchsorted(times[members], cutoff, side='right')])
        found.append(np.unique(np.concatenate(pieces)))
    return found


def tree(arrays, entities, cutoffs, floor, count, depth, device, registry, spec, stream):
    observations = np.concatenate([latest(arrays, [eid], cutoff) for eid, cutoff in zip(entities, cutoffs)])
    raw = np.zeros((len(entities), registry.features), np.float32)
    known = observations >= 0
    raw[known] = arrays['x'][observations[known]]
    result = dict(x=tensor(raw, device), kind=tensor(arrays['entity_kind'][entities], device, torch.long))
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
        result.update(child=tree(arrays, children, times, floor, count, depth - 1, device, registry, spec, stream),
            mask=tensor(mask, device), gap=tensor(gaps, device), relation=tensor(relation, device))
    return result


def history_inputs(arrays, first, last, start, count, device, features, spec, stream, include_structure=False, selected=None):
    n = last - first
    raw = np.zeros((n, 2, count, features), np.float32)
    mask, gap, ids = np.zeros((n, 2, count), bool), np.zeros((n, 2, count), np.float32), np.full((n, 2, count), -1, np.int32)
    event_ids = np.full((n, 2, count), -1, np.int32)
    pair_gaps, pair_mask = np.zeros((n, count), np.float32), np.zeros((n, count), bool)
    structural = np.zeros((n, 2, count, 12), np.float32)
    common = np.zeros((n, features), np.float32)
    common_ids = np.full((n, count), -1, np.int32)
    common_frequency = np.zeros((n, count), np.float32)
    common_stats = np.zeros((n, 4), np.float32)
    events = arrays['events']
    for i in range(n) if selected is None else np.flatnonzero(selected):
        query = first + i
        roots = int(arrays['actor'][query]), int(arrays['subject'][query])
        cutoff, floor = int(arrays['query_step'][query]) * 2 + 1, start * 2
        lists = []
        memo = {}

        def partners(entity):
            if entity not in memo:
                ix = neighbors(arrays, entity, cutoff, floor, count, spec, stream)
                memo[entity] = other_endpoint(events, ix, entity).tolist()
            return memo[entity]

        def overlap(a, b):
            if not a or not b:
                return 0.0
            return sum(x == y for x in a for y in b) / (len(a) * len(b))

        def two_hop(entity):
            first_hop = partners(entity)[-max(1, int(count**0.5)):]
            return list({p for other in first_hop for p in partners(other)})

        for side, root in enumerate(roots):
            ix = neighbors(arrays, root, cutoff, floor, count, spec, stream)
            lists.append(ix)
            length = len(ix)
            if length:
                others = other_endpoint(events, ix, root)
                observed = np.where(events[ix, 0] == root, events[ix, 5], events[ix, 4])
                raw[i, side, :length] = arrays['x'][observed]
                interaction = events[ix, 3] == 2
                raw[i, side, :length][interaction] = (raw[i, side, :length][interaction] + arrays['query'][events[ix[interaction], 2] // 2]) / 2
                mask[i, side, :length] = True
                gap[i, side, :length] = cutoff - events[ix, 2]
                ids[i, side, :length] = others
                event_ids[i, side, :length] = ix
                if include_structure:
                    for j, event in enumerate(ix):
                        values = []
                        for endpoint in events[event, :2]:
                            for query_endpoint in roots:
                                a, b = partners(int(endpoint)), partners(query_endpoint)
                                values.extend((b.count(int(endpoint)) / max(len(b), 1), overlap(a, b), overlap(two_hop(int(endpoint)), two_hop(query_endpoint))))
                        structural[i, side, j] = values
        a, b = partners(roots[0]), partners(roots[1])
        shared = sorted(set(a) & set(b))
        if shared:
            common_ids[i, :len(shared)] = shared
            common_frequency[i, :len(shared)] = [a.count(e) * b.count(e) for e in shared]
            obs = latest(arrays, shared, cutoff)
            obs = obs[obs >= 0]
            if len(obs):
                common[i] = arrays['x'][obs].mean(0)
        common_stats[i] = [len(shared), a.count(roots[1]), b.count(roots[0]), len(set(a) | set(b))]
        pair = lists[0][other_endpoint(events, lists[0], roots[0]) == roots[1]]
        if len(pair):
            times = np.r_[events[pair, 2], cutoff]
            diffs = np.diff(times)[-count:]
            pair_gaps[i, :len(diffs)] = diffs
            pair_mask[i, :len(diffs)] = True
    occurrence = np.zeros((n, 2, count, 2), np.float32)
    for side in range(2):
        for other in range(2):
            occurrence[:, side, :, other] = ((ids[:, side, :, None] == ids[:, other, None, :]) & mask[:, side, :, None] & mask[:, other, None, :]).sum(-1)
    return {name: tensor(value, device) for name, value in dict(history_x=raw, history_mask=mask, history_gap=gap,
        history_ids=ids, history_events=event_ids, occurrence=occurrence, structure=structural,
        common_x=common, common_stats=common_stats, common_ids=common_ids, common_frequency=common_frequency,
        pair_gap=pair_gaps, pair_mask=pair_mask).items()}


def pack(bank, block, cfg, device='cuda'):
    from .roster import BLOCKS
    from .turns import enabled, lookback

    started = time.perf_counter()
    graph, start, stop, target_start, partition, row_lo, row_hi = block
    a = bank.arrays(graph, cfg)
    registry, spec, stream = bank.registry, cfg['graph'], SEED
    node = registry.extras['level'] == 'node'
    blocks = BLOCKS[cfg['model']]
    lo, hi = a['snapshot_offsets'][[start, stop]]
    n, count = stop - start, cfg['neighbors']
    entity = np.array(a['entity'][lo:hi])
    step = np.array(a['step'][lo:hi]) - start
    size = len(a['entity_kind'])
    align = np.full((n, size), hi - lo, np.int64)
    align[step, entity] = np.arange(hi - lo)
    first, last = (row_lo, row_hi) if node else tuple(int(value) for value in np.searchsorted(a['query_step'], (start, stop)))
    m = last - first
    moment = np.array(a['query_step'][first:last]).astype(np.int64)
    rows = np.array(a['row_index'][first:last])
    ids = np.arange(first, last)
    selected = (rows >= 0) & (moment >= target_start) & (ids >= row_lo) & (ids < row_hi)
    selected[selected] = bank.partitions[rows[selected]] == partition
    local = moment - start
    bounds = np.searchsorted(local, np.arange(n + 1))
    query_context = np.array(a['query_context'][first:last])
    features = np.zeros((m, registry.features), np.float32) if node else np.array(a['query'][first:last], np.float32)
    actors, subjects = np.array(a['actor'][first:last]), np.array(a['subject'][first:last])
    b = dict(graph=graph, start=start, stop=stop, frames=n, time_steps=n, batch_size=1, predictions=m, observations=int(hi - lo),
        graphs=[graph], entity_sizes=[size], entity_batch=torch.zeros(size, dtype=torch.long, device=device),
        query_steps=tensor(local, device, torch.long), query_frames=tensor(local, device, torch.long),
        frame_queries=[torch.arange(int(bounds[t]), int(bounds[t + 1]), device=device) for t in range(n)],
        entities=size, selected=tensor(selected, device),
        rows=rows[selected], target=tensor(bank.targets[rows[selected]], device),
        query=tensor(features, device), context=tensor(a['x'][query_context], device, torch.float32),
        actor=tensor(actors, device, torch.long), subject=tensor(subjects, device, torch.long),
        context_index=tensor(query_context - lo, device, torch.long),
        entity_kind=tensor(a['entity_kind'], device, torch.long),
        signal=tensor(a['signals'][first:last], device), affinity=tensor(a['affinity'][first:last], device),
        global_affinity=tensor(a['global_affinity'][first:last], device))
    if enabled(cfg):
        from .turn_inputs import add

        add(b, a, first, last, start, cfg, device)
    if 'snapshot' in blocks:
        b.update(x=tensor(a['x'][lo:hi], device, torch.float32), kind=tensor(a['entity_kind'][entity], device, torch.long),
                 entity=tensor(entity, device, torch.long), step=tensor(step, device, torch.long), align=tensor(align, device, torch.long),
                 present=tensor(align < hi - lo, device))
        if registry.extras['queries']:
            left, right = np.searchsorted(a['events'][:, 2], (start * 2, stop * 2))
            choices = np.array(a['events'][left:right])
            choices = choices[choices[:, 3] == 2]
            b.update(decision_query=tensor(a['query'][start:stop], device, torch.float32),
                     decision_context=tensor(choices[:, 4:6] - lo, device, torch.long))
        else:
            b.update(decision_query=torch.zeros(n, registry.features, device=device),
                     decision_context=torch.full((n, 2), -1, dtype=torch.long, device=device))
        first_edge, last_edge = np.searchsorted(a['spatial_step'], (start, stop))
        spatial = np.array(a['spatial'][first_edge:last_edge])
        attribute_rows = np.array(a['spatial_attribute_row'][first_edge:last_edge])
        edges = spatial[:, :2].T - lo
        b.update(edges=tensor(edges, device, torch.long), relation=tensor(spatial[:, 2], device, torch.long))
        b['attributes'], b['attribute_edges'] = {}, {}
        for name in registry.attributed:
            declared = registry.by_name[name]
            chosen = np.flatnonzero(np.isin(spatial[:, 2], (declared.code, declared.reverse)))
            values = np.asarray(a['attributes_' + name])[attribute_rows[chosen]]
            b['attributes'][name] = tensor(values.reshape(-1, len(declared.attributes)), device, torch.float32)
            b['attribute_edges'][name] = tensor(chosen, device, torch.long)
        b['frame_edges'], b['frame_directed'] = [], []
        directed = registry.canonical[spatial[:, 2]]
        for t in range(n):
            use = step[edges[1]] == t
            b['frame_edges'].append(tensor(entity[edges[:, use]], device, torch.long))
            b['frame_directed'].append(tensor(entity[edges[:, use & directed]], device, torch.long))
        if 'lifted' in blocks:
            coupled = a['coupling']
            left, right = np.searchsorted(coupled[:, 1], (lo, hi))
            coupled = np.array(coupled[left:right])
            coupled = coupled[coupled[:, 0] >= lo] - lo
            extra = a['coupling_boundaries']
            low, high = np.searchsorted(extra[:, 1], (lo, hi))
            extra = np.array(extra[low:high])
            extra = extra[extra[:, 0] >= lo]
            boundary = extra[:, :2] - lo
            b.update(temporal_edges=tensor(np.concatenate((coupled, boundary)).T, device, torch.long),
                temporal_relation=tensor(np.r_[np.full(len(coupled), registry.code('temporal')), extra[:, 2]], device, torch.long),
                temporal_gap=tensor(np.r_[step[coupled[:, 1]] - step[coupled[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]], device, torch.float32),
                temporal_turn_gap=tensor(np.r_[a['turn'][lo + coupled[:, 1]] - a['turn'][lo + coupled[:, 0]], extra[:, 3]], device, torch.float32))
            if enabled(cfg):
                from .turn_inputs import temporal_edges

                temporal_edges(b, a, lo, hi, device, registry)
        if 'mixed' in blocks:
            left, right = np.searchsorted(a['mixed_tick'], (start * 2, stop * 2))
            edges = np.array(a['mixed_edges'])
            use = (edges[:, 0] >= left) & (edges[:, 1] >= left) & (edges[:, 0] < right) & (edges[:, 1] < right)
            edges = edges[use]
            ticks = np.array(a['mixed_tick'][left:right])
            mx = np.concatenate((a['mixed_x'][left:right], features)).astype(np.float32)
            mk = np.r_[a['mixed_kind'][left:right], np.full(m, 5)].astype(np.int64)
            qi = np.arange(right - left, right - left + m)
            roots = np.stack((actors, subjects, np.full(m, -1)), 1)
            sources = reachable(np.array(a['mixed_entity'][left:right], np.int64), np.arange(right - left), ticks, roots, moment * 2)
            extra = np.column_stack((np.concatenate(sources + [np.zeros(0, np.int64)]), np.repeat(qi, [len(s) for s in sources]),
                                     np.full(sum(len(s) for s in sources), 2048))).astype(np.int64)
            edge = np.concatenate((np.column_stack((edges[:, :2] - left, edges[:, 2])), extra))
            all_ticks = np.r_[ticks, moment * 2]
            feature = np.zeros((len(edge), 16), np.float32)
            feature[:, :13] = (edge[:, 2, None] & (1 << np.arange(13))) != 0
            feature[:, 13] = 1
            feature[:, 14] = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2)
            b['mixed'] = dict(x=tensor(mx, device), kind=tensor(mk, device), edge_index=tensor(edge[:, :2].T, device, torch.long),
                              edge=tensor(feature, device), query=tensor(qi, device, torch.long))
            if enabled(cfg):
                from .turn_inputs import mixed_edges

                mixed_edges(b, a, start, stop, device, registry.features)
    if 'events' in blocks:
        left, right = np.searchsorted(a['events'][:, 2], (start * 2, stop * 2))
        events = np.array(a['events'][left:right])
        if 'dag' in blocks:
            members = np.flatnonzero(np.asarray(a['event_member'][left:right]))
            position = np.full(right - left, -1, np.int64)
            position[members] = np.arange(len(members))
            events = events[members]
        ex = np.array((a['x'][events[:, 4]].astype(np.float32) + a['x'][events[:, 5]]) / 2)
        interaction = events[:, 3] == 2
        ex[interaction] = a['query'][events[interaction, 2] // 2]
        initial = latest(a, np.arange(size), start * 2)
        initial_x = np.zeros((size, registry.features), np.float32)
        initial_x[initial >= 0] = a['x'][initial[initial >= 0]]
        starts = np.flatnonzero(np.diff(events[:, 2], prepend=-1))
        b.update(events=tensor(events, device, torch.long), event_x=tensor(ex, device), initial=tensor(initial_x, device),
            event_groups=[(int(events[s, 2]), int(s), int(e)) for s, e in zip(starts, np.append(starts[1:], len(events)))])
        if 'dag' in blocks:
            linked = a['event_dag']
            use = (linked[:, 0] >= left) & (linked[:, 1] >= left) & (linked[:, 0] < right) & (linked[:, 1] < right)
            linked = position[np.array(linked[use]) - left]
            extra = a['event_boundaries']
            low, high = np.searchsorted(extra[:, 1], (left, right))
            extra = np.array(extra[low:high])
            extra = extra[extra[:, 0] >= left]
            boundary = position[extra[:, :2] - left]
            b.update(dag_edges=tensor(np.concatenate((linked, boundary)).T, device, torch.long),
                dag_relation=tensor(np.r_[events[linked[:, 1], 3], extra[:, 2]], device, torch.long),
                dag_gap=tensor(np.r_[events[linked[:, 1], 2] - events[linked[:, 0], 2], events[boundary[:, 1], 2] - events[boundary[:, 0], 2]], device, torch.float32))
            positions = np.r_[np.arange(len(events)), np.arange(len(events))]
            reached = reachable(np.r_[events[:, 0], events[:, 1]].astype(np.int64), positions, events[:, 2], np.stack((actors, subjects), 1), moment * 2)
            links = np.stack((np.concatenate(reached + [np.zeros(0, np.int64)]), np.repeat(np.arange(m), [len(r) for r in reached])))
            b['dag_query_edges'] = tensor(links, device, torch.long)
    if 'tree' in blocks:
        roots = np.stack((actors, subjects), 1).reshape(-1)
        cutoffs = np.repeat(moment * 2 + 1, 2)
        b['tree'] = tree(a, roots, cutoffs, start * 2, count, cfg['layers'], device, registry, spec, stream)
    if 'history' in blocks:
        b.update(history_inputs(a, first, last, start, count, device, registry.features, spec, stream, 'structure' in blocks,
            selected if 'history_targets' in blocks else None))
    if enabled(cfg) and 'turn_history' in blocks:
        from .turn_inputs import history_tokens

        history_tokens(b, a, first, last, start, cfg, device, registry)
    if 'walks' in blocks:
        if cfg['walks'] > a['walks'].shape[2] or cfg['walk_hops'] > a['walks'].shape[3]:
            raise ValueError('requested walks exceed the walks stored for this graph variant')
        walks = np.array(a['walks'][first:last, :, :cfg['walks'], :cfg['walk_hops']])
        if enabled(cfg):
            extra = a['turn_walks'][first:last, :, :lookback(cfg), :cfg['walks'], :cfg['walk_hops']]
            extra = np.array(extra).reshape(m, 2, lookback(cfg) * cfg['walks'], cfg['walk_hops'])
            walks = np.concatenate((walks, extra), 2)
            kinds = np.zeros(walks.shape, np.int64)
            kinds[:, :, cfg['walks']:] = 1
            gaps = np.zeros(walks.shape + (3,), np.float32)
            past = b['end_token_gaps'][:, :2].cpu().numpy()
            gaps[:, :, cfg['walks']:] = np.repeat(past, cfg['walks'], axis=2)[:, :, :, None, :]
            b.update(walk_turn_kind=tensor(kinds, device, torch.long), walk_end_time=tensor(gaps, device))
        known = walks >= 0
        known &= a['events'][np.maximum(walks, 0), 2] >= start * 2
        events = a['events'][np.maximum(walks, 0)]
        nodes = np.empty(walks.shape, np.int32)
        observations = np.empty_like(nodes)
        current = np.broadcast_to(np.stack((actors, subjects), 1)[:, :, None], walks.shape[:3]).copy()
        for hop in range(cfg['walk_hops']):
            forward = events[:, :, :, hop, 0] == current
            nodes[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 1], events[:, :, :, hop, 0])
            observations[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 5], events[:, :, :, hop, 4])
            current = nodes[:, :, :, hop]
        walk_features = np.array(a['x'][observations], np.float32)
        walk_features[~known] = 0
        gap = moment[:, None, None, None] * 2 + 1 - events[..., 2]
        identities = np.zeros(walks.shape + (2 * cfg['walk_hops'],), np.float32)
        for q in range(m):
            for side in range(2):
                for hop in range(cfg['walk_hops']):
                    reference = nodes[q, side, :, hop]
                    identities[q, ..., side * cfg['walk_hops'] + hop] = ((nodes[q, ..., None] == reference) & known[q, side, :, hop]).sum(-1)
        b.update(walk_x=tensor(walk_features, device), walk_mask=tensor(known, device), walk_gap=tensor(gap, device, torch.float32), walk_identity=tensor(identities, device))
    LOG.debug('pack exit graph=%d start=%d stop=%d predictions=%d seconds=%.4f', graph, start, stop, m, time.perf_counter() - started)
    return b
