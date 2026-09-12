from collections import OrderedDict
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
from . import gather


LOG = logging.getLogger(__name__)
ROWS = 1024
LOADED = 8
NEIGHBOURS = 8
ORDERS = dict(recent=0, uniform=1, decay=2)
DTYPES = dict(entity_kind=torch.long)


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


class Resident:

    def __init__(self, arrays, device):
        self.arrays, self.device, self.tensors = arrays, device, {}
        self.ticks = np.ascontiguousarray(arrays['events'][:, 2])

    def tensor(self, key):
        if key not in self.tensors:
            started = time.perf_counter()
            self.tensors[key] = torch.as_tensor(np.array(self.arrays[key]), dtype=DTYPES.get(key), device=self.device)
            LOG.debug('resident exit key=%s device=%s seconds=%.4f', key, self.device, time.perf_counter() - started)
        return self.tensors[key]


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
        self.variants, self.store, self.residents = {}, ArrayStore(LOADED), OrderedDict()
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

    def resident(self, index, cfg, device):
        key = (self.variant(cfg)[0] / ('graph_%04d' % index), str(device))
        if key not in self.residents:
            self.residents[key] = Resident(self.arrays(index, cfg), device)
            while len(self.residents) > LOADED:
                self.residents.popitem(last=False)
        self.residents.move_to_end(key)
        return self.residents[key]

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


def sampling(spec):
    order = spec['neighbors/order']
    return ORDERS[order], spec['neighbors/decay'] if order == 'decay' else 0.0


def features_at(held, observations, device):
    index = tensor(observations, device, torch.long)
    known = index >= 0
    return torch.where(known[..., None], held.tensor('x')[index.clamp_min(0)].float(), 0.0)


def tree(a, held, entities, cutoffs, floor, count, depth, device, registry, spec, stream):
    mode, decay = sampling(spec)
    events, ticks = a['events'], held.ticks
    ids, offsets = a['neighbor_ids'], a['neighbor_offsets']
    observation_ids, observation_offsets, step = a['observation_ids'], a['observation_offsets'], a['step']
    levels = []
    for _ in range(depth):
        table, sizes = gather.neighbor_table(ids, offsets, ticks, entities, cutoffs, floor, count, mode, decay, stream, NEIGHBOURS)
        children, times, mask, gaps, relation = gather.tree_level(events, table, sizes, entities, cutoffs, count, registry.marker('non_relation'))
        levels.append((entities, cutoffs, mask, gaps, relation))
        entities, cutoffs = children, times

    def node(members, moments):
        observations = gather.latest(observation_ids, observation_offsets, step, members, moments)
        return dict(x=features_at(held, observations, device), kind=held.tensor('entity_kind')[tensor(members, device, torch.long)])

    result = node(entities, cutoffs)
    for entities, cutoffs, mask, gaps, relation in reversed(levels):
        result = dict(node(entities, cutoffs), child=result, mask=tensor(mask, device), gap=tensor(gaps, device), relation=tensor(relation, device))
    return result


def history_inputs(a, held, first, last, start, count, device, spec, stream, include_structure, selected):
    mode, decay = sampling(spec)
    chosen = np.ones(last - first, bool) if selected is None else np.asarray(selected, bool)
    (observed, interaction, query_tick, mask, gap, ids, structural, common_ids, common_frequency, common_stats,
     pair_gaps, pair_mask) = gather.history(a['events'], a['neighbor_ids'], a['neighbor_offsets'], held.ticks, np.asarray(a['actor'][first:last], np.int64),
                                            np.asarray(a['subject'][first:last], np.int64), np.asarray(a['query_step'][first:last], np.int64), start, count,
                                            mode, decay, stream, NEIGHBOURS, include_structure, chosen)
    raw = features_at(held, observed, device)
    picked = tensor(interaction, device)
    raw[picked] = (raw[picked] + held.tensor('query')[tensor(query_tick, device, torch.long)[picked]].float()) / 2
    occurrence = np.zeros((last - first, 2, count, 2), np.float32)
    for side in range(2):
        for other in range(2):
            occurrence[:, side, :, other] = ((ids[:, side, :, None] == ids[:, other, None, :]) & mask[:, side, :, None] & mask[:, other, None, :]).sum(-1)
    result = dict(history_x=raw)
    result.update({name: tensor(value, device) for name, value in dict(history_mask=mask, history_gap=gap, history_ids=ids, occurrence=occurrence,
        structure=structural, common_stats=common_stats, common_ids=common_ids, common_frequency=common_frequency,
        pair_gap=pair_gaps, pair_mask=pair_mask).items()})
    return result


def pack(bank, block, cfg, device='cuda'):
    from .roster import BLOCKS
    from .turns import enabled, lookback

    started = time.perf_counter()
    graph, start, stop, target_start, partition, row_lo, row_hi = block
    a = bank.arrays(graph, cfg)
    held = bank.resident(graph, cfg, device)
    registry, spec, stream = bank.registry, cfg['graph'], SEED
    node = registry.extras['level'] == 'node'
    blocks = BLOCKS[cfg['model']]
    lo, hi = (int(value) for value in a['snapshot_offsets'][[start, stop]])
    n, count = stop - start, cfg['neighbors']
    entity = np.array(a['entity'][lo:hi])
    step = np.array(a['step'][lo:hi]) - start
    size = len(a['entity_kind'])
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
    features = torch.zeros(m, registry.features, device=device) if node else held.tensor('query')[first:last].float()
    actors, subjects = np.array(a['actor'][first:last]), np.array(a['subject'][first:last])
    b = dict(graph=graph, start=start, stop=stop, frames=n, time_steps=n, batch_size=1, predictions=m, observations=hi - lo,
        graphs=[graph], entity_sizes=[size], entity_batch=torch.zeros(size, dtype=torch.long, device=device),
        query_steps=tensor(local, device, torch.long), query_frames=tensor(local, device, torch.long),
        frame_queries=[torch.arange(int(bounds[t]), int(bounds[t + 1]), device=device) for t in range(n)],
        entities=size, selected=tensor(selected, device),
        rows=rows[selected], target=tensor(bank.targets[rows[selected]], device),
        query=features, context=held.tensor('x')[tensor(query_context, device, torch.long)].float(),
        actor=tensor(actors, device, torch.long), subject=tensor(subjects, device, torch.long),
        context_index=tensor(query_context - lo, device, torch.long),
        entity_kind=held.tensor('entity_kind'),
        signal=tensor(a['signals'][first:last], device), affinity=tensor(a['affinity'][first:last], device),
        global_affinity=tensor(a['global_affinity'][first:last], device))
    if enabled(cfg):
        from .turn_inputs import add

        add(b, a, first, last, start, cfg, device)
    if 'snapshot' in blocks:
        entity_index, step_index = tensor(entity, device, torch.long), tensor(step, device, torch.long)
        align = torch.full((n, size), hi - lo, dtype=torch.long, device=device)
        align[step_index, entity_index] = torch.arange(hi - lo, device=device)
        b.update(x=held.tensor('x')[lo:hi].float(), kind=held.tensor('entity_kind')[entity_index], entity=entity_index, step=step_index, align=align,
                 present=align < hi - lo)
        if registry.extras['queries']:
            left, right = np.searchsorted(held.ticks, (start * 2, stop * 2))
            choices = np.array(a['events'][left:right])
            choices = choices[choices[:, 3] == 2]
            b.update(decision_query=held.tensor('query')[start:stop].float(), decision_context=tensor(choices[:, 4:6] - lo, device, torch.long))
        else:
            b.update(decision_query=torch.zeros(n, registry.features, device=device), decision_context=torch.full((n, 2), -1, dtype=torch.long, device=device))
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
            mixed_edges = np.array(a['mixed_edges'])
            use = (mixed_edges[:, 0] >= left) & (mixed_edges[:, 1] >= left) & (mixed_edges[:, 0] < right) & (mixed_edges[:, 1] < right)
            mixed_edges = mixed_edges[use]
            ticks = np.array(a['mixed_tick'][left:right])
            mk = np.r_[a['mixed_kind'][left:right], np.full(m, 5)].astype(np.int64)
            qi = np.arange(right - left, right - left + m)
            roots = np.stack((actors, subjects, np.full(m, -1)), 1).astype(np.int64)
            sources, offsets = gather.reachable(np.array(a['mixed_entity'][left:right], np.int64), np.arange(right - left), ticks, roots, moment * 2)
            extra = np.column_stack((sources, np.repeat(qi, np.diff(offsets)), np.full(len(sources), 2048))).astype(np.int64)
            edge = np.concatenate((np.column_stack((mixed_edges[:, :2] - left, mixed_edges[:, 2])), extra))
            all_ticks = np.r_[ticks, moment * 2]
            feature = np.zeros((len(edge), 16), np.float32)
            feature[:, :13] = (edge[:, 2, None] & (1 << np.arange(13))) != 0
            feature[:, 13] = 1
            feature[:, 14] = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2)
            b['mixed'] = dict(x=torch.cat((held.tensor('mixed_x')[left:right].float(), features)), kind=tensor(mk, device),
                              edge_index=tensor(edge[:, :2].T, device, torch.long), edge=tensor(feature, device), query=tensor(qi, device, torch.long))
            if enabled(cfg):
                from .turn_inputs import mixed_edges as extend_mixed

                extend_mixed(b, a, start, stop, device, registry.features)
    if 'events' in blocks:
        left, right = np.searchsorted(held.ticks, (start * 2, stop * 2))
        events = np.array(a['events'][left:right])
        if 'dag' in blocks:
            members = np.flatnonzero(np.asarray(a['event_member'][left:right]))
            position = np.full(right - left, -1, np.int64)
            position[members] = np.arange(len(members))
            events = events[members]
        endpoints = tensor(events[:, 4:6], device, torch.long)
        ex = (held.tensor('x')[endpoints[:, 0]].float() + held.tensor('x')[endpoints[:, 1]].float()) / 2
        interaction = events[:, 3] == 2
        ex[tensor(interaction, device)] = held.tensor('query')[tensor(events[interaction, 2] // 2, device, torch.long)].float()
        initial = gather.latest(a['observation_ids'], a['observation_offsets'], a['step'], np.arange(size), np.full(size, start * 2, np.int64))
        starts = np.flatnonzero(np.diff(events[:, 2], prepend=-1))
        b.update(events=tensor(events, device, torch.long), event_x=ex, initial=features_at(held, initial, device),
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
            reached, offsets = gather.reachable(np.r_[events[:, 0], events[:, 1]].astype(np.int64), positions, events[:, 2].astype(np.int64),
                                                np.stack((actors, subjects), 1).astype(np.int64), moment * 2)
            b['dag_query_edges'] = tensor(np.stack((reached, np.repeat(np.arange(m), np.diff(offsets)))), device, torch.long)
    if 'tree' in blocks:
        roots = np.stack((actors, subjects), 1).reshape(-1).astype(np.int64)
        cutoffs = np.repeat(moment * 2 + 1, 2)
        b['tree'] = tree(a, held, roots, cutoffs, start * 2, count, cfg['layers'], device, registry, spec, stream)
    if 'history' in blocks:
        b.update(history_inputs(a, held, first, last, start, count, device, spec, stream, 'structure' in blocks, selected if 'history_targets' in blocks else None))
    if enabled(cfg) and 'turn_history' in blocks:
        from .turn_inputs import history_tokens

        history_tokens(b, a, held, first, last, start, cfg, device, registry)
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
        known &= held.ticks[np.maximum(walks, 0)] >= start * 2
        events = a['events'][np.maximum(walks, 0)]
        nodes = np.empty(walks.shape, np.int32)
        observations = np.empty_like(nodes)
        current = np.broadcast_to(np.stack((actors, subjects), 1)[:, :, None], walks.shape[:3]).copy()
        for hop in range(cfg['walk_hops']):
            forward = events[:, :, :, hop, 0] == current
            nodes[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 1], events[:, :, :, hop, 0])
            observations[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 5], events[:, :, :, hop, 4])
            current = nodes[:, :, :, hop]
        known_index = tensor(known, device)
        walk_features = torch.where(known_index[..., None], held.tensor('x')[tensor(observations, device, torch.long)].float(), 0.0)
        gap = moment[:, None, None, None] * 2 + 1 - events[..., 2]
        b.update(walk_x=walk_features, walk_mask=known_index, walk_gap=tensor(gap, device, torch.float32),
                 walk_identity=tensor(gather.walk_identities(nodes, known), device))
    LOG.debug('pack exit graph=%d start=%d stop=%d predictions=%d seconds=%.4f', graph, start, stop, m, time.perf_counter() - started)
    return b
