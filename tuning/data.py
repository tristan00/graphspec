import logging
from pathlib import Path
import time

import numpy as np
import torch

from graphspec import Registry
from graphspec import spec as specs
from graphspec.transform import SEED, SOURCE, materialize
from graphspec import join
from graphspec.truth import VERSION, Graph
from graph_tuning_infra.arrays import Bundle, Residents, between
from graph_tuning_infra.files import ArrayStore, load_arrays, read_json, write_json
from . import gather


LOG = logging.getLogger(__name__)
ROWS = 1024
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
            first, last = between(steps, start, stop)
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
        self.variants, self.recent, self.store, self.residents = {}, {}, ArrayStore(), Residents(DTYPES)
        LOG.info('bank exit seconds=%.3f graphs=%d rows=%d level=%s', time.perf_counter() - started, len(self.manifest['graphs']), len(self.rows), self.level)

    def truth(self, index):
        return self.directory / 'truth' / ('graph_%04d' % index)

    def joining(self, cfg):
        return self.variant(cfg)[1]['graphs/join'] == 'single'

    def graphs(self, cfg):
        return [0] if self.joining(cfg) else [graph['index'] for graph in self.manifest['graphs']]

    def joined(self, cfg):
        started = time.perf_counter()
        spec = self.variant(cfg)[1]
        graphs = [Graph.load(self.truth(graph['index'])) for graph in self.manifest['graphs']]
        result = join.joined(graphs, self.registry, spec)
        LOG.info('bank join exit seconds=%.3f graphs=%d', time.perf_counter() - started, len(graphs))
        return result

    def sizes(self, cfg):
        if not self.joining(cfg):
            return self.entities
        return {0: len(self.arrays(0, cfg)['entity_kind'])}

    def variant(self, cfg):
        from .model_spec import groups

        held = self.recent.get(id(cfg))
        if held is not None and held[0] is cfg:
            return held[1]
        chosen = groups(cfg['model'])
        key = (chosen, specs.digest(cfg['graph']))
        if key not in self.variants:
            spec = specs.canonical(specs.space(self.registry, chosen), cfg['graph'])
            self.variants[key] = (self.directory / 'variants' / specs.digest(dict(spec=specs.materialized(spec), groups=list(chosen), source=SOURCE)), spec, chosen)
        self.recent[id(cfg)] = (cfg, self.variants[key])
        return self.variants[key]

    def stamp(self, cfg):
        root, spec, chosen = self.variant(cfg)
        root.mkdir(parents=True, exist_ok=True)
        if not (root / 'variant.json').exists():
            write_json(root / 'variant.json', dict(spec=specs.materialized(spec), groups=list(chosen), source=SOURCE))
        return root

    def arrays(self, index, cfg):
        root, spec, chosen = self.variant(cfg)

        def build():
            started = time.perf_counter()
            self.stamp(cfg)
            truth = self.joined(cfg) if self.joining(cfg) else Graph.load(self.truth(index))
            arrays = materialize(truth, self.registry, spec, chosen, self.categories)
            LOG.info('variant exit graph=%d seconds=%.3f arrays=%d', index, time.perf_counter() - started, len(arrays))
            return arrays

        return self.store.open(root / ('graph_%04d' % index), build)

    def resident(self, index, cfg, device):
        return self.residents.get(self.variant(cfg)[0] / ('graph_%04d' % index), device, lambda: self.arrays(index, cfg))

    def blocks(self, partition, history, cfg):
        started = time.perf_counter()
        result = []
        if self.joining(cfg):
            rows, steps = join.layout([(graph['steps'], self.row_index[graph['index']], self.row_steps[graph['index']]) for graph in self.manifest['graphs']], self.level)
            result = layout(0, rows, steps, self.partitions, partition, history, self.level)
        for graph in ([] if self.joining(cfg) else self.manifest['graphs']):
            index = graph['index']
            result.extend(layout(index, self.row_index[index], self.row_steps[index], self.partitions, partition, history, self.level))
        LOG.info('bank blocks exit seconds=%.3f partition=%d history=%d blocks=%d', time.perf_counter() - started, partition, history, len(result))
        return result


def tensor(value, device, dtype=None):
    return torch.as_tensor(np.require(value, requirements=('C', 'W')), dtype=dtype, device=device)


def sampling(spec):
    order = spec['neighbors/order']
    return ORDERS[order], spec['neighbors/decay'] if order == 'decay' else 0.0


def features_at(held, observations, device):
    index = tensor(observations, device, torch.long)
    known = index >= 0
    return torch.where(known[..., None], held.tensor('x')[index.clamp_min(0)].float(), 0.0)


def tree(held, entities, cutoffs, floor, count, depth, device, registry, spec, stream, hold):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    events, ticks = held.numpy('events'), held.column('events', 2)
    ids, offsets = held.numpy('neighbor_ids'), held.numpy('neighbor_offsets')
    observation_ids, observation_offsets, step = held.numpy('observation_ids'), held.numpy('observation_offsets'), held.numpy('step')
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
        result = dict(node(entities, cutoffs), child=result, mask=hold(mask), gap=hold(gaps), relation=hold(relation))
    LOG.debug('tree exit roots=%d depth=%d count=%d seconds=%.4f', len(levels[0][0]) if levels else len(entities), depth, count, time.perf_counter() - started)
    return result


def history_inputs(held, first, last, start, count, device, spec, stream, include_structure, selected, hold):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    chosen = np.ones(last - first, bool) if selected is None else np.asarray(selected, bool)
    (observed, interaction, query_tick, mask, gap, ids, structural, common_ids, common_frequency, common_stats,
     pair_gaps, pair_mask) = gather.history(held.numpy('events'), held.numpy('neighbor_ids'), held.numpy('neighbor_offsets'), held.column('events', 2),
                                            held.numpy('actor')[first:last].astype(np.int64), held.numpy('subject')[first:last].astype(np.int64),
                                            held.numpy('query_step')[first:last].astype(np.int64), start, count, mode, decay, stream, NEIGHBOURS,
                                            include_structure, chosen)
    raw = features_at(held, observed, device)
    picked = tensor(interaction, device)
    raw[picked] = (raw[picked] + held.tensor('query')[tensor(query_tick, device, torch.long)[picked]].float()) / 2
    occurrence = np.zeros((last - first, 2, count, 2), np.float32)
    for side in range(2):
        for other in range(2):
            occurrence[:, side, :, other] = ((ids[:, side, :, None] == ids[:, other, None, :]) & mask[:, side, :, None] & mask[:, other, None, :]).sum(-1)
    result = dict(history_x=raw)
    result.update({name: hold(value) for name, value in dict(history_mask=mask, history_gap=gap, history_ids=ids, occurrence=occurrence,
        structure=structural, common_stats=common_stats, common_ids=common_ids, common_frequency=common_frequency,
        pair_gap=pair_gaps, pair_mask=pair_mask).items()})
    LOG.debug('history exit rows=%d count=%d structure=%s seconds=%.4f', last - first, count, include_structure, time.perf_counter() - started)
    return result


def pack(bank, block, cfg, device):
    from .model_spec import blocks as model_blocks, neighbors
    from .turns import enabled, lookback

    started = time.perf_counter()
    graph, start, stop, target_start, partition, row_lo, row_hi = block
    a = bank.arrays(graph, cfg)
    held = bank.resident(graph, cfg, device)
    bundle = Bundle(device)
    hold = bundle.hold
    registry, spec, stream = bank.registry, cfg['graph'], SEED
    node = registry.extras['level'] == 'node'
    blocks = model_blocks(cfg['model'])
    lo, hi = (int(value) for value in held.numpy('snapshot_offsets')[[start, stop]])
    n = stop - start
    entity = held.numpy('entity')[lo:hi]
    step = held.numpy('step')[lo:hi] - start
    size = len(a['entity_kind'])
    query_steps = held.numpy('query_step')
    first, last = (row_lo, row_hi) if node else between(query_steps, start, stop)
    m = last - first
    moment = query_steps[first:last].astype(np.int64)
    rows = held.numpy('row_index')[first:last]
    ids = np.arange(first, last)
    selected = (rows >= 0) & (moment >= target_start) & (ids >= row_lo) & (ids < row_hi)
    selected[selected] = bank.partitions[rows[selected]] == partition
    local = moment - start
    bounds = np.searchsorted(local, np.arange(n + 1))
    query_context = held.numpy('query_context')[first:last]
    features = torch.zeros(m, registry.features, device=device) if node else held.tensor('query')[first:last].float()
    actors, subjects = held.numpy('actor')[first:last], held.numpy('subject')[first:last]
    b = dict(graph=graph, start=start, stop=stop, frames=n, time_steps=n, batch_size=1, predictions=m, observations=hi - lo,
        graphs=[graph], entity_sizes=[size], entity_batch=hold(np.zeros(size, np.int64)),
        query_steps=hold(local, torch.long), query_frames=hold(local, torch.long),
        frame_queries=[hold(np.arange(bounds[t], bounds[t + 1], dtype=np.int64)) for t in range(n)],
        entities=size, selected=hold(selected),
        rows=rows[selected], target=hold(bank.targets[rows[selected]]),
        query=features, context=held.tensor('x')[tensor(query_context, device, torch.long)].float(),
        actor=hold(actors, torch.long), subject=hold(subjects, torch.long),
        context_index=hold(query_context - lo, torch.long),
        entity_kind=held.tensor('entity_kind'),
        signal=hold(held.numpy('signals')[first:last]), affinity=hold(held.numpy('affinity')[first:last]),
        global_affinity=hold(held.numpy('global_affinity')[first:last]))
    if enabled(cfg):
        from .turn_inputs import add

        add(b, a, first, last, start, cfg, hold)
        b = bundle.flush(b)
    if 'snapshot' in blocks:
        entity_index, step_index = tensor(entity, device, torch.long), tensor(step, device, torch.long)
        align = torch.full((n, size), hi - lo, dtype=torch.long, device=device)
        align[step_index, entity_index] = torch.arange(hi - lo, device=device)
        b.update(x=held.tensor('x')[lo:hi].float(), kind=held.tensor('entity_kind')[entity_index], entity=entity_index, step=step_index, align=align,
                 present=align < hi - lo)
        if registry.extras['queries']:
            left, right = between(held.column('events', 2), start * 2, stop * 2)
            choices = held.numpy('events')[left:right]
            choices = choices[choices[:, 3] == 2]
            b.update(decision_query=held.tensor('query')[start:stop].float(), decision_context=hold(choices[:, 4:6] - lo, torch.long))
        else:
            b.update(decision_query=torch.zeros(n, registry.features, device=device), decision_context=hold(np.full((n, 2), -1, np.int64)))
        spatial_steps = held.numpy('spatial_step')
        first_edge, last_edge = between(spatial_steps, start, stop)
        spatial = held.numpy('spatial')[first_edge:last_edge]
        edge_bounds = np.searchsorted(spatial_steps[first_edge:last_edge], np.arange(start, stop + 1, dtype=spatial_steps.dtype))
        attribute_rows = held.numpy('spatial_attribute_row')[first_edge:last_edge]
        edges = spatial[:, :2].T - lo
        b.update(edges=hold(edges, torch.long), relation=hold(spatial[:, 2], torch.long))
        b['attributes'], b['attribute_edges'] = {}, {}
        by_code = np.argsort(spatial[:, 2], kind='stable')
        sorted_codes = spatial[by_code, 2]
        codes = [list(dict.fromkeys((registry.by_name[name].code, registry.by_name[name].reverse))) for name in registry.attributed]
        bounds = np.searchsorted(sorted_codes, np.asarray([[code, code + 1] for group in codes for code in group], sorted_codes.dtype).reshape(-1)).reshape(-1, 2)
        offset = 0
        for name, group in zip(registry.attributed, codes):
            declared = registry.by_name[name]
            pieces = [by_code[lo:hi] for lo, hi in bounds[offset:offset + len(group)]]
            offset += len(group)
            chosen = pieces[0] if len(pieces) == 1 else np.sort(np.concatenate(pieces))
            values = held.numpy('attributes_' + name)[attribute_rows[chosen]]
            b['attributes'][name] = hold(values.reshape(-1, len(declared.attributes)), torch.float32)
            b['attribute_edges'][name] = hold(chosen, torch.long)
        directed = registry.canonical[spatial[:, 2]]
        endpoints = entity[edges]
        canonical = entity[edges[:, directed]]
        canonical_bounds = np.r_[0, np.cumsum(directed)][edge_bounds]
        b['frame_edges'] = [hold(endpoints[:, edge_bounds[t]:edge_bounds[t + 1]], torch.long) for t in range(n)]
        b['frame_directed'] = [hold(canonical[:, canonical_bounds[t]:canonical_bounds[t + 1]], torch.long) for t in range(n)]
        if 'lifted' in blocks:
            left, right = between(held.column('coupling', 1), lo, hi)
            coupled = held.numpy('coupling')[left:right]
            coupled = coupled[coupled[:, 0] >= lo] - lo
            low, high = between(held.column('coupling_boundaries', 1), lo, hi)
            extra = held.numpy('coupling_boundaries')[low:high]
            extra = extra[extra[:, 0] >= lo]
            boundary = extra[:, :2] - lo
            turn = held.numpy('turn')
            b.update(temporal_edges=hold(np.concatenate((coupled, boundary)).T, torch.long),
                temporal_relation=hold(np.r_[np.full(len(coupled), registry.code('temporal')), extra[:, 2]], torch.long),
                temporal_gap=hold(np.r_[step[coupled[:, 1]] - step[coupled[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]], torch.float32),
                temporal_turn_gap=hold(np.r_[turn[lo + coupled[:, 1]] - turn[lo + coupled[:, 0]], extra[:, 3]], torch.float32))
            if enabled(cfg):
                from .turn_inputs import temporal_edges

                b = bundle.flush(b)
                temporal_edges(b, a, lo, hi, device, registry, hold)
    if 'mixed' in blocks:
        mixed_ticks = held.numpy('mixed_tick')
        left, right = between(mixed_ticks, start * 2, stop * 2)
        mixed_edges = held.numpy('mixed_edges')[held.rows_by('mixed_edges', 1, left, right)]
        mixed_edges = mixed_edges[(mixed_edges[:, 0] >= left) & (mixed_edges[:, 0] < right)]
        ticks = mixed_ticks[left:right]
        mk = np.r_[held.numpy('mixed_kind')[left:right], np.full(m, 5)].astype(np.int64)
        qi = np.arange(right - left, right - left + m)
        roots = np.stack((actors, subjects, np.full(m, -1)), 1).astype(np.int64)
        sources, offsets = gather.reachable(held.numpy('mixed_entity')[left:right].astype(np.int64), np.arange(right - left), ticks, roots, moment * 2)
        extra = np.column_stack((sources, np.repeat(qi, np.diff(offsets)), np.full(len(sources), 2048))).astype(np.int64)
        edge = np.concatenate((np.column_stack((mixed_edges[:, :2] - left, mixed_edges[:, 2])), extra))
        all_ticks = np.r_[ticks, moment * 2]
        edge_index = hold(edge[:, :2].T, torch.long)
        feature = torch.zeros((len(edge), 16), device=device)
        feature[:, :13] = ((tensor(edge[:, 2], device, torch.long)[:, None] & (1 << torch.arange(13, device=device))) != 0).float()
        feature[:, 13] = 1
        feature[:, 14] = tensor(np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2), device, torch.float32)
        b['mixed'] = dict(x=torch.cat((held.tensor('mixed_x')[left:right].float(), features)), kind=hold(mk),
                          edge_index=edge_index, edge=feature, query=hold(qi, torch.long))
        if enabled(cfg):
            from .turn_inputs import mixed_edges as extend_mixed

            b = bundle.flush(b)
            extend_mixed(b, a, start, stop, device, registry.features)
    if 'events' in blocks:
        left, right = between(held.column('events', 2), start * 2, stop * 2)
        events = held.numpy('events')[left:right]
        if 'dag' in blocks:
            members = np.flatnonzero(held.numpy('event_member')[left:right])
            position = np.full(right - left, -1, np.int64)
            position[members] = np.arange(len(members))
            events = events[members]
        endpoints = tensor(events[:, 4:6], device, torch.long)
        ex = (held.tensor('x')[endpoints[:, 0]].float() + held.tensor('x')[endpoints[:, 1]].float()) / 2
        interaction = events[:, 3] == 2
        ex[tensor(interaction, device)] = held.tensor('query')[tensor(events[interaction, 2] // 2, device, torch.long)].float()
        initial = gather.latest(held.numpy('observation_ids'), held.numpy('observation_offsets'), held.numpy('step'), np.arange(size), np.full(size, start * 2, np.int64))
        starts = np.flatnonzero(np.diff(events[:, 2], prepend=-1))
        b.update(events=tensor(events, device, torch.long), event_x=ex, initial=features_at(held, initial, device),
            event_groups=[(int(events[s, 2]), int(s), int(e)) for s, e in zip(starts, np.append(starts[1:], len(events)))])
        if 'dag' in blocks:
            low, high = between(held.column('event_dag', 1), left, right)
            linked = np.array(a['event_dag'][low:high])
            linked = position[linked[linked[:, 0] >= left] - left]
            low, high = between(held.column('event_boundaries', 1), left, right)
            extra = held.numpy('event_boundaries')[low:high]
            extra = extra[extra[:, 0] >= left]
            boundary = position[extra[:, :2] - left]
            dag = tensor(np.concatenate((linked, boundary)).T, device, torch.long)
            b.update(dag_edges=dag, dag_relation=torch.cat((b['events'][dag[1, :len(linked)], 3], tensor(extra[:, 2], device, torch.long))),
                dag_gap=(b['events'][dag[1], 2] - b['events'][dag[0], 2]).float())
            positions = np.r_[np.arange(len(events)), np.arange(len(events))]
            reached, offsets = gather.reachable(np.r_[events[:, 0], events[:, 1]].astype(np.int64), positions, events[:, 2].astype(np.int64),
                                                np.stack((actors, subjects), 1).astype(np.int64), moment * 2)
            b['dag_query_edges'] = hold(np.stack((reached, np.repeat(np.arange(m), np.diff(offsets)))), torch.long)
    if 'tree' in blocks:
        roots = np.stack((actors, subjects), 1).reshape(-1).astype(np.int64)
        cutoffs = np.repeat(moment * 2 + 1, 2)
        b['tree'] = tree(held, roots, cutoffs, start * 2, neighbors(cfg['model']), cfg['model']['tree/depth'], device, registry, spec, stream, hold)
    if 'history' in blocks:
        b.update(history_inputs(held, first, last, start, neighbors(cfg['model']), device, spec, stream, 'structure' in blocks, selected if 'history_targets' in blocks else None, hold))
    if enabled(cfg) and 'turn_history' in blocks:
        from .turn_inputs import history_tokens

        b = bundle.flush(b)
        history_tokens(b, a, held, first, last, start, cfg, device, registry, hold)
    if 'walks' in blocks:
        walk_count, hops = cfg['model']['walks/count'], cfg['model']['walks/hops']
        if walk_count > a['walks'].shape[2] or hops > a['walks'].shape[3]:
            raise ValueError('requested walks exceed the walks stored for this graph variant')
        walks = held.numpy('walks')[first:last, :, :walk_count, :hops]
        if enabled(cfg):
            extra = held.numpy('turn_walks')[first:last, :, :lookback(cfg), :walk_count, :hops]
            extra = extra.reshape(m, 2, lookback(cfg) * walk_count, hops)
            walks = np.concatenate((walks, extra), 2)
            kinds = np.zeros(walks.shape, np.int64)
            kinds[:, :, walk_count:] = 1
            gaps = np.zeros(walks.shape + (3,), np.float32)
            past = b['end_token_gaps'][:, :2].cpu().numpy()
            gaps[:, :, walk_count:] = np.repeat(past, walk_count, axis=2)[:, :, :, None, :]
            b.update(walk_turn_kind=hold(kinds, torch.long), walk_end_time=hold(gaps))
        known = walks >= 0
        known &= held.column('events', 2)[np.maximum(walks, 0)] >= start * 2
        events = held.numpy('events')[np.maximum(walks, 0)]
        nodes = np.empty(walks.shape, np.int32)
        observations = np.empty_like(nodes)
        current = np.broadcast_to(np.stack((actors, subjects), 1)[:, :, None], walks.shape[:3]).copy()
        for hop in range(hops):
            forward = events[:, :, :, hop, 0] == current
            nodes[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 1], events[:, :, :, hop, 0])
            observations[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 5], events[:, :, :, hop, 4])
            current = nodes[:, :, :, hop]
        known_index = tensor(known, device)
        walk_features = torch.where(known_index[..., None], held.tensor('x')[tensor(observations, device, torch.long)].float(), 0.0)
        gap = moment[:, None, None, None] * 2 + 1 - events[..., 2]
        b.update(walk_x=walk_features, walk_mask=known_index, walk_gap=hold(gap, torch.float32),
                 walk_identity=hold(gather.walk_identities(nodes, known)))
    b = bundle.flush(b)
    LOG.debug('pack exit graph=%d start=%d stop=%d predictions=%d seconds=%.4f', graph, start, stop, m, time.perf_counter() - started)
    return b
