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
from graph_tuning_infra.arrays import NUMPY, RESIDENT, Bundle, Pending, Residents, between, views
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

    def __init__(self, directory, resident=RESIDENT):
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
        self.variants, self.recent, self.store, self.residents = {}, {}, ArrayStore(), Residents(DTYPES, resident)
        LOG.info('bank exit seconds=%.3f graphs=%d rows=%d level=%s', time.perf_counter() - started, len(self.manifest['graphs']), len(self.rows), self.level)

    def truth(self, index):
        return self.directory / 'truth' / ('graph_%04d' % index)

    def shared_kinds(self):
        path = self.directory / 'shared_kinds.json'
        if not path.exists():
            started = time.perf_counter()
            seen = {}
            for graph in self.manifest['graphs']:
                folder = self.truth(graph['index'])
                for kind, key in zip(load_arrays(folder, ('kinds',))['kinds'].tolist(), read_json(folder / 'graph.json')['keys']):
                    seen.setdefault((kind, key), set()).add(graph['index'])
            counts = {kind: [0, 0] for kind in range(len(self.registry.nodes))}
            for (kind, _), graphs in seen.items():
                counts[kind][0] += 1
                counts[kind][1] += len(graphs) > 1
            shared = [len(self.manifest['graphs']) > 1 and counts[kind][0] > 0 and counts[kind][1] * 2 >= counts[kind][0] for kind in range(len(self.registry.nodes))]
            write_json(path, shared)
            LOG.info('shared kinds exit seconds=%.3f shared=%s', time.perf_counter() - started, ','.join(self.registry.nodes[kind] for kind in range(len(shared)) if shared[kind]))
        return np.asarray(read_json(path), bool)

    def tokens(self, held):
        return np.where(self.shared_kinds()[held.numpy('entity_kind')], held.numpy('entity_token'), -1)

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
        graphs = self.manifest['graphs']
        if not self.joining(cfg):
            return {graph['index']: (graph['entities'], graph['facts'] / graph['steps']) for graph in graphs}
        return {0: (len(self.arrays(0, cfg)['entity_kind']), sum(graph['facts'] for graph in graphs) / sum(graph['steps'] for graph in graphs))}

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
            linked = []
            for name, source in (() if self.joining(cfg) else (('x', 'obs_x'), ('query', 'query_x'))):
                if identical(arrays[name], truth.arrays[source]):
                    arrays[name] = self.truth(index) / (source + '.npy')
                    linked.append(name)
            LOG.info('variant exit graph=%d seconds=%.3f arrays=%d linked=%s', index, time.perf_counter() - started, len(arrays), ','.join(linked))
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


def array(value, dtype=None):
    result = np.ascontiguousarray(value)
    if dtype is not None and result.dtype != NUMPY[dtype] and not (dtype == torch.long and result.dtype == np.int32):
        result = result.astype(NUMPY[dtype])
    return result


class Host:

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value


class Gather:

    __slots__ = ('segments',)

    def __init__(self, segments):
        self.segments = segments


class Features:

    __slots__ = ('segments', 'order')

    def __init__(self, segments, order=None):
        self.segments, self.order = segments, order


class Frames:

    __slots__ = ('directed', 'ranges', 'canonical')

    def __init__(self, directed, ranges, canonical):
        self.directed, self.ranges, self.canonical = directed, ranges, canonical


def gathered(graph, key, index=None):
    return Gather([(graph, key, None if index is None else array(index, torch.long))])


def features(graph, shape, *pieces):
    return Features([(graph, tuple(shape), [(key, index if isinstance(index, tuple) else array(index, torch.long), weight) for key, index, weight in pieces])])


def features_at(graph, observations):
    index = array(observations, torch.long)
    return features(graph, index.shape, ('x', index, 1.0))


def length(value):
    return sum(segment[1][0] for segment in value.segments)


def joined(items, axis=0, order=None):
    first = items[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis)
    if isinstance(first, Gather):
        return Gather([segment for item in items for segment in item.segments])
    return Features([segment for item in items for segment in item.segments], order)


def sampling(spec):
    order = spec['neighbors/order']
    return ORDERS[order], spec['neighbors/decay'] if order == 'decay' else 0.0


def tree(graph, held, entities, cutoffs, floor, count, depth, registry, spec, stream, tokens=None):
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
        result = dict(x=features_at(graph, observations), kind=gathered(graph, 'entity_kind', members))
        if tokens is not None:
            result['token'] = array(tokens[members])
        return result

    result = node(entities, cutoffs)
    for entities, cutoffs, mask, gaps, relation in reversed(levels):
        result = dict(node(entities, cutoffs), child=result, mask=array(mask), gap=array(gaps), relation=array(relation))
    LOG.debug('tree exit roots=%d depth=%d count=%d seconds=%.4f', len(levels[0][0]) if levels else len(entities), depth, count, time.perf_counter() - started)
    return result


def history_inputs(graph, held, first, last, start, count, spec, stream, include_structure, selected):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    chosen = np.ones(last - first, bool) if selected is None else np.asarray(selected, bool)
    (observed, interaction, query_tick, mask, gap, ids, structural, common_ids, common_frequency, common_stats,
     pair_gaps, pair_mask) = gather.history(held.numpy('events'), held.numpy('neighbor_ids'), held.numpy('neighbor_offsets'), held.column('events', 2),
                                            held.numpy('actor')[first:last].astype(np.int64), held.numpy('subject')[first:last].astype(np.int64),
                                            held.numpy('query_step')[first:last].astype(np.int64), start, count, mode, decay, stream, NEIGHBOURS,
                                            include_structure, chosen)
    occurrence = np.zeros((last - first, 2, count, 2), np.float32)
    for side in range(2):
        for other in range(2):
            occurrence[:, side, :, other] = ((ids[:, side, :, None] == ids[:, other, None, :]) & mask[:, side, :, None] & mask[:, other, None, :]).sum(-1)
    weight = np.where(interaction, np.float32(0.5), np.float32(1.0))
    result = dict(history_x=features(graph, observed.shape, ('x', observed, weight), ('query', np.where(interaction, query_tick, -1), 0.5)))
    result.update({name: array(value) for name, value in dict(history_mask=mask, history_gap=gap, history_ids=ids, occurrence=occurrence,
        structure=structural, common_stats=common_stats, common_ids=common_ids, common_frequency=common_frequency,
        pair_gap=pair_gaps, pair_mask=pair_mask).items()})
    LOG.debug('history exit rows=%d count=%d structure=%s seconds=%.4f', last - first, count, include_structure, time.perf_counter() - started)
    return result


def recipe(bank, block, cfg):
    from .model_spec import blocks as model_blocks, neighbors
    from .turns import enabled, lookback

    started = time.perf_counter()
    graph, start, stop, target_start, partition, row_lo, row_hi = block
    held = bank.resident(graph, cfg, 'cpu')
    registry, spec, stream = bank.registry, cfg['graph'], SEED
    node = registry.extras['level'] == 'node'
    blocks = model_blocks(cfg['model'])
    tokens = bank.tokens(held) if cfg['model'].get('keys/embedding') else None
    lo, hi = (int(value) for value in held.numpy('snapshot_offsets')[[start, stop]])
    n = stop - start
    entity = held.numpy('entity')[lo:hi]
    step = held.numpy('step')[lo:hi] - start
    size = len(held.numpy('entity_kind'))
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
    actors, subjects = held.numpy('actor')[first:last], held.numpy('subject')[first:last]
    b = dict(graph=graph, start=start, stop=stop, frames=n, time_steps=n, batch_size=1, predictions=m, observations=hi - lo,
        graphs=[graph], entity_sizes=[size], entity_batch=np.zeros(size, np.int64),
        query_steps=array(local, torch.long), query_frames=array(local, torch.long),
        frame_queries=[np.arange(bounds[t], bounds[t + 1], dtype=np.int64) for t in range(n)],
        entities=size, selected=array(selected),
        rows=Host(rows[selected]), target=array(bank.targets[rows[selected]]),
        query=features(graph, (m,)) if node else features(graph, (m,), ('query', (first, last), None)),
        context=features(graph, query_context.shape, ('x', query_context, None)),
        actor=array(actors, torch.long), subject=array(subjects, torch.long),
        context_index=array(query_context - lo, torch.long),
        entity_kind=gathered(graph, 'entity_kind'),
        signal=array(held.numpy('signals')[first:last]), affinity=array(held.numpy('affinity')[first:last]),
        global_affinity=array(held.numpy('global_affinity')[first:last]))
    if tokens is not None:
        b['entity_token'] = array(tokens)
        b['root_token'] = array(tokens[np.stack((actors, subjects), 1)])
    if enabled(cfg):
        from .turn_inputs import add

        add(b, graph, held, first, last, start, cfg)
    if 'snapshot' in blocks:
        align = np.full((n, size), hi - lo, np.int64)
        align[step, entity] = np.arange(hi - lo)
        b.update(x=features(graph, (hi - lo,), ('x', (lo, hi), None)), kind=gathered(graph, 'entity_kind', entity), entity=array(entity, torch.long),
                 step=array(step, torch.long), align=align, present=align < hi - lo)
        if tokens is not None:
            b['token'] = array(tokens[entity])
        if registry.extras['queries']:
            left, right = between(held.column('events', 2), start * 2, stop * 2)
            choices = held.numpy('events')[left:right]
            choices = choices[choices[:, 3] == 2]
            b.update(decision_query=features(graph, (n,), ('query', (start, stop), None)), decision_context=array(choices[:, 4:6] - lo, torch.long))
        else:
            b.update(decision_query=features(graph, (n,)), decision_context=np.full((n, 2), -1, np.int64))
        spatial_steps = held.numpy('spatial_step')
        first_edge, last_edge = between(spatial_steps, start, stop)
        spatial = held.numpy('spatial')[first_edge:last_edge]
        edge_bounds = np.searchsorted(spatial_steps[first_edge:last_edge], np.arange(start, stop + 1, dtype=spatial_steps.dtype))
        attribute_rows = held.numpy('spatial_attribute_row')
        edges = spatial[:, :2].T - lo
        b.update(edges=array(edges, torch.long), relation=array(spatial[:, 2], torch.long))
        b['attributes'], b['attribute_edges'] = {}, {}
        by_code, code_offsets = held.numpy('spatial_order'), held.numpy('spatial_code_offsets')
        window = np.asarray((first_edge, last_edge), by_code.dtype)
        for name in registry.attributed:
            declared = registry.by_name[name]
            pieces = []
            for code in dict.fromkeys((declared.code, declared.reverse)):
                lo_code, hi_code = code_offsets[code], code_offsets[code + 1]
                low, high = by_code[lo_code:hi_code].searchsorted(window)
                pieces.append(by_code[lo_code + low:lo_code + high])
            chosen = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
            values = held.numpy('attributes_' + name)[attribute_rows[chosen]]
            b['attributes'][name] = array(values.reshape(-1, len(declared.attributes)), torch.float32)
            b['attribute_edges'][name] = array(chosen - first_edge, torch.long)
        directed = registry.canonical[spatial[:, 2]]
        canonical_bounds = np.r_[0, np.cumsum(directed)][edge_bounds]
        b['frame_edges'] = Frames(directed, [[(int(edge_bounds[t]), int(edge_bounds[t + 1]))] for t in range(n)], False)
        b['frame_directed'] = Frames(directed, [[(int(canonical_bounds[t]), int(canonical_bounds[t + 1]))] for t in range(n)], True)
        if 'lifted' in blocks:
            left, right = between(held.column('coupling', 1), lo, hi)
            coupled = held.numpy('coupling')[left:right]
            coupled = coupled[coupled[:, 0] >= lo] - lo
            low, high = between(held.column('coupling_boundaries', 1), lo, hi)
            extra = held.numpy('coupling_boundaries')[low:high]
            extra = extra[extra[:, 0] >= lo]
            boundary = extra[:, :2] - lo
            turn = held.numpy('turn')
            b.update(temporal_edges=array(np.concatenate((coupled, boundary)).T, torch.long),
                temporal_relation=array(np.r_[np.full(len(coupled), registry.code('temporal')), extra[:, 2]], torch.long),
                temporal_gap=array(np.r_[step[coupled[:, 1]] - step[coupled[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]], torch.float32),
                temporal_turn_gap=array(np.r_[turn[lo + coupled[:, 1]] - turn[lo + coupled[:, 0]], extra[:, 3]], torch.float32))
            if enabled(cfg):
                from .turn_inputs import temporal_edges

                temporal_edges(b, graph, held, lo, hi, registry)
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
        gaps = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2).astype(np.float32)
        b['mixed'] = dict(x=joined([features(graph, (right - left,), ('mixed_x', (left, right), None)), b['query']]), kind=mk,
                          edge_index=array(edge[:, :2].T, torch.long), edge_code=array(edge[:, 2] | (1 << 13), torch.long),
                          edge_gap=array(np.stack((gaps, np.zeros_like(gaps)), 1)), query=array(qi, torch.long))
        if tokens is not None:
            b['mixed']['token'] = np.r_[tokens[held.numpy('mixed_entity')[left:right]], np.full(m, -1, np.int64)]
        if enabled(cfg):
            from .turn_inputs import mixed_edges as extend_mixed

            extend_mixed(b, graph)
    if 'events' in blocks:
        left, right = between(held.column('events', 2), start * 2, stop * 2)
        events = held.numpy('events')[left:right]
        if 'dag' in blocks:
            members = np.flatnonzero(held.numpy('event_member')[left:right])
            position = np.full(right - left, -1, np.int64)
            position[members] = np.arange(len(members))
            events = events[members]
        interaction = events[:, 3] == 2
        endpoints = np.where(interaction[:, None], -1, events[:, 4:6])
        initial = gather.latest(held.numpy('observation_ids'), held.numpy('observation_offsets'), held.numpy('step'), np.arange(size), np.full(size, start * 2, np.int64))
        starts = np.flatnonzero(np.diff(events[:, 2], prepend=-1))
        b.update(events=array(events, torch.long), initial=features_at(graph, initial),
            event_x=features(graph, (len(events),), ('x', endpoints[:, 0], 0.5), ('x', endpoints[:, 1], 0.5), ('query', np.where(interaction, events[:, 2] // 2, -1), 1.0)),
            event_groups=[(int(events[s, 2]), int(s), int(e)) for s, e in zip(starts, np.append(starts[1:], len(events)))])
        if 'dag' in blocks:
            low, high = between(held.column('event_dag', 1), left, right)
            linked = held.numpy('event_dag')[low:high]
            linked = position[linked[linked[:, 0] >= left] - left]
            low, high = between(held.column('event_boundaries', 1), left, right)
            extra = held.numpy('event_boundaries')[low:high]
            extra = extra[extra[:, 0] >= left]
            boundary = position[extra[:, :2] - left]
            dag = np.concatenate((linked, boundary)).T.astype(np.int64)
            b.update(dag_edges=array(dag), dag_relation=array(np.concatenate((events[dag[1, :len(linked)], 3], extra[:, 2])), torch.long),
                dag_gap=array(events[dag[1], 2] - events[dag[0], 2], torch.float32))
            positions = np.r_[np.arange(len(events)), np.arange(len(events))]
            reached, offsets = gather.reachable(np.r_[events[:, 0], events[:, 1]].astype(np.int64), positions, events[:, 2].astype(np.int64),
                                                np.stack((actors, subjects), 1).astype(np.int64), moment * 2)
            b['dag_query_edges'] = array(np.stack((reached, np.repeat(np.arange(m), np.diff(offsets)))), torch.long)
    if 'tree' in blocks:
        roots = np.stack((actors, subjects), 1).reshape(-1).astype(np.int64)
        cutoffs = np.repeat(moment * 2 + 1, 2)
        b['tree'] = tree(graph, held, roots, cutoffs, start * 2, neighbors(cfg['model']), cfg['model']['tree/depth'], registry, spec, stream, tokens)
    if 'history' in blocks:
        b.update(history_inputs(graph, held, first, last, start, neighbors(cfg['model']), spec, stream, 'structure' in blocks, selected if 'history_targets' in blocks else None))
        if tokens is not None:
            b['history_token'] = np.where(b['history_ids'] >= 0, tokens[np.maximum(b['history_ids'], 0)], -1)
    if enabled(cfg) and 'turn_history' in blocks:
        from .turn_inputs import history_tokens

        history_tokens(b, graph, held, first, last, start, cfg, registry)
    if 'walks' in blocks:
        walk_count, hops = cfg['model']['walks/count'], cfg['model']['walks/hops']
        stored = held.numpy('walks')
        if walk_count > stored.shape[2] or hops > stored.shape[3]:
            raise ValueError('requested walks exceed the walks stored for this graph variant')
        walks = stored[first:last, :, :walk_count, :hops]
        if enabled(cfg):
            extra = held.numpy('turn_walks')[first:last, :, :lookback(cfg), :walk_count, :hops]
            extra = extra.reshape(m, 2, lookback(cfg) * walk_count, hops)
            walks = np.concatenate((walks, extra), 2)
            kinds = np.zeros(walks.shape, np.int64)
            kinds[:, :, walk_count:] = 1
            gaps = np.zeros(walks.shape + (3,), np.float32)
            past = b['end_token_gaps'][:, :2]
            gaps[:, :, walk_count:] = np.repeat(past, walk_count, axis=2)[:, :, :, None, :]
            b.update(walk_turn_kind=kinds, walk_end_time=gaps)
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
        gap = moment[:, None, None, None] * 2 + 1 - events[..., 2]
        b.update(walk_x=features(graph, known.shape, ('x', np.where(known, observations, -1), 1.0)), walk_mask=array(known), walk_gap=array(gap, torch.float32),
                 walk_identity=array(gather.walk_identities(nodes, known)))
        if tokens is not None:
            b['walk_token'] = np.where(known, tokens[nodes], -1)
    LOG.debug('recipe exit graph=%d start=%d stop=%d predictions=%d seconds=%.4f', graph, start, stop, m, time.perf_counter() - started)
    return b


def prepare(value, bundle):
    if isinstance(value, np.ndarray):
        return bundle.hold(value)
    if isinstance(value, Host):
        return value
    if isinstance(value, Gather):
        return Gather([(graph, key, None if index is None else bundle.hold(index, torch.int32)) for graph, key, index in value.segments])
    if isinstance(value, Features):
        segments = [(graph, shape, [(key, index if isinstance(index, tuple) else bundle.hold(index, torch.int32), bundle.hold(weight) if isinstance(weight, np.ndarray) else weight)
                                    for key, index, weight in pieces]) for graph, shape, pieces in value.segments]
        return Features(segments, None if value.order is None else bundle.hold(value.order))
    if isinstance(value, Frames):
        return Frames(bundle.hold(value.directed), value.ranges, value.canonical)
    if isinstance(value, dict):
        return {key: prepare(item, bundle) for key, item in value.items()}
    if isinstance(value, list):
        return [prepare(item, bundle) for item in value]
    if isinstance(value, tuple):
        return tuple(prepare(item, bundle) for item in value)
    return value


def identical(left, right):
    return left is right or (left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left.view(np.uint8), right.view(np.uint8)))


class Resolver:

    def __init__(self, bank, cfg, tensors, device, graphs):
        self.bank, self.cfg, self.tensors, self.device, self.graphs, self.tables = bank, cfg, tensors, device, graphs, {}

    def table(self, key):
        if key not in self.tables:
            parts, offsets, lengths, total = [], {}, {}, 0
            for graph in self.graphs:
                tensor = self.bank.resident(graph, self.cfg, self.device).tensor(key)
                if tensor.is_cuda:
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
                offsets[graph], lengths[graph], total = total, len(tensor), total + len(tensor)
                parts.append(tensor)
            self.tables[key] = (parts[0] if len(parts) == 1 else torch.cat(parts), offsets, lengths)
        return self.tables[key]

    def numbers(self, values):
        return torch.as_tensor(np.asarray(values, np.int64), device=self.device)

    def spread(self, values, lengths, total):
        return torch.repeat_interleave(self.numbers(values), self.numbers(lengths), output_size=total)

    def ranges(self, offsets, spans):
        starts = np.asarray([offsets[graph] + lo for graph, lo, hi in spans], np.int64)
        lengths = np.asarray([hi - lo for graph, lo, hi in spans], np.int64)
        total = int(lengths.sum())
        return self.spread(starts - np.r_[0, np.cumsum(lengths[:-1])], lengths, total) + torch.arange(total, device=self.device)

    def grouped(self, items, resolve):
        groups = {}
        for position, (signature, _) in enumerate(items):
            groups.setdefault(signature, []).append(position)
        parts = [resolve(signature, positions) for signature, positions in groups.items()]
        if len(parts) == 1:
            return parts[0]
        starts, offset = {}, 0
        for positions in groups.values():
            for position in positions:
                starts[position], offset = offset, offset + items[position][1]
        order = np.concatenate([np.arange(starts[position], starts[position] + rows) for position, (_, rows) in enumerate(items)])
        return torch.cat(parts)[torch.as_tensor(order, device=self.device)]

    def local(self, indices):
        return torch.cat([index.reshape(-1) for index in indices]) if len(indices) > 1 else indices[0].reshape(-1)

    def features(self, value):
        segments = value.segments
        items = [((tuple(shape[1:]), tuple((key, isinstance(index, tuple), weight is None, isinstance(weight, Pending)) for key, index, weight in pieces)), int(shape[0]))
                 for _, shape, pieces in segments]

        def resolve(signature, positions):
            trailing, layout = signature
            chosen = [segments[position] for position in positions]
            rows, out = sum(shape[0] for _, shape, _ in chosen), None
            for p, (key, is_range, unweighted, per_row) in enumerate(layout):
                source, offsets, _ = self.table(key)
                if is_range:
                    values = source.index_select(0, self.ranges(offsets, [(graph, pieces[p][1][0], pieces[p][1][1]) for graph, _, pieces in chosen])).float()
                elif not len(source):
                    continue
                else:
                    lengths = [int(np.prod(shape)) for _, shape, _ in chosen]
                    local = self.local([self.tensors[pieces[p][1].index] for _, _, pieces in chosen])
                    shift = self.spread([offsets[graph] for graph, _, _ in chosen], lengths, sum(lengths))
                    if unweighted:
                        values = source.index_select(0, local + shift).float()
                    else:
                        mask = local >= 0
                        weight = self.local([self.tensors[pieces[p][2].index] for _, _, pieces in chosen]) if per_row else chosen[0][2][p][2]
                        values = source.index_select(0, torch.where(mask, local + shift, shift)).float() * (mask.float() * weight)[:, None]
                values = values.reshape((rows,) + trailing + (source.shape[1],))
                out = values if out is None else out + values
            return torch.zeros((rows,) + trailing + (self.bank.registry.features,), device=self.device) if out is None else out

        result = self.grouped(items, resolve)
        return result if value.order is None else result[self.tensors[value.order.index]]

    def gather(self, value):
        segments = value.segments
        items = [((key, index is None), self.table(key)[2][graph] if index is None else int(self.tensors[index.index].shape[0])) for graph, key, index in segments]

        def resolve(signature, positions):
            key, whole = signature
            source, offsets, lengths = self.table(key)
            chosen = [segments[position] for position in positions]
            if whole:
                return source.index_select(0, self.ranges(offsets, [(graph, 0, lengths[graph]) for graph, _, _ in chosen]))
            local = [self.tensors[index.index] for _, _, index in chosen]
            shape = (sum(index.shape[0] for index in local),) + tuple(local[0].shape[1:])
            shift = self.spread([offsets[graph] for graph, _, _ in chosen], [index.numel() for index in local], sum(index.numel() for index in local))
            return source.index_select(0, self.local(local) + shift).reshape(shape + tuple(source.shape[1:]))

        return self.grouped(items, resolve)

    def run(self, value):
        if isinstance(value, Pending):
            return self.tensors[value.index]
        if isinstance(value, Host):
            return value.value
        if isinstance(value, Gather):
            return self.gather(value)
        if isinstance(value, Features):
            return self.features(value)
        if isinstance(value, dict):
            result = {key: self.run(item) for key, item in value.items() if not isinstance(item, Frames)}
            for key, item in value.items():
                if isinstance(item, Frames):
                    endpoints = result['entity'][result['edges']]
                    if item.canonical:
                        endpoints = endpoints[:, self.tensors[item.directed.index]]
                    result[key] = [torch.cat([endpoints[:, lo:hi] for lo, hi in step], 1) if len(step) > 1 else endpoints[:, step[0][0]:step[0][1]] for step in item.ranges]
            return result
        if isinstance(value, list):
            return [self.run(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.run(item) for item in value)
        return value


def keys(value, found):
    if isinstance(value, Gather):
        found.update(key for _, key, _ in value.segments)
    elif isinstance(value, Features):
        found.update(key for _, _, pieces in value.segments for key, _, _ in pieces)
    elif isinstance(value, dict):
        for item in value.values():
            keys(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys(item, found)
    return found


def resolve(value, bank, cfg, tensors, device):
    return Resolver(bank, cfg, tensors, device, list(dict.fromkeys(value['graphs']))).run(value)


def materialise(bank, cfg, tree, device):
    started = time.perf_counter()
    bundle = Bundle(device)
    prepared = prepare(tree, bundle)
    buffer, spans = bundle.pack()
    result = resolve(prepared, bank, cfg, views(torch.as_tensor(buffer, device=device), spans), device)
    LOG.debug('materialise exit arrays=%d bytes=%d device=%s seconds=%.4f', len(spans), len(buffer), device, time.perf_counter() - started)
    return result

