from collections import OrderedDict
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import torch

from . import spec
from .schema import KINDS


class CampaignArrays(Mapping):

    def __init__(self, sources):
        self.sources = sources
        self.loaded = {}

    def __getitem__(self, key):
        if key not in self.loaded:
            path, member = self.sources[key]
            if member is None:
                self.loaded[key] = np.load(path, mmap_mode='r', allow_pickle=False)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    self.loaded[key] = archive[member]
        return self.loaded[key]

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def __contains__(self, key):
        return key in self.sources


class Bank:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['spec_sha256'] != spec.fingerprint():
            raise ValueError('cache was built under a different graph spec')
        supervision = json.loads((self.directory / 'supervision.json').read_text())
        self.rows, self.split = supervision['rows'], supervision['split']
        self.targets = np.asarray([row['y'] for row in self.rows], np.float32)
        self.partitions = np.zeros(len(self.rows), np.int8)
        self.partitions[self.split['validation_indices']] = 1
        self.campaign_partition = {row['campaign']: int(self.partitions[i]) for i, row in enumerate(self.rows)}
        self.loaded = OrderedDict()
        from .turns import cache_path as turn_cache_path

        self.turn_directory = turn_cache_path(self.directory)
        if (self.turn_directory / 'manifest.json').exists():
            turn_manifest = json.loads((self.turn_directory / 'manifest.json').read_text())
            if turn_manifest['version'] != 2 or turn_manifest['supervision_sha256'] != self.manifest['supervision_sha256']:
                raise ValueError('turn cache belongs to different supervision')
        else:
            self.turn_directory = None
        from .walk_cache import cache_path

        self.walk_directory = cache_path(self.directory)
        if (self.walk_directory / 'manifest.json').exists():
            walk_manifest = json.loads((self.walk_directory / 'manifest.json').read_text())
            if walk_manifest['supervision_sha256'] != self.manifest['supervision_sha256']:
                raise ValueError('walk cache belongs to different supervision')
        else:
            self.walk_directory = None
        from .graph_variants import VERSION, cache_path as variant_cache_path

        self.variant_directory = variant_cache_path(self.directory)
        if (self.variant_directory / 'manifest.json').exists():
            variant_manifest = json.loads((self.variant_directory / 'manifest.json').read_text())
            if variant_manifest['version'] != VERSION or variant_manifest['supervision_sha256'] != self.manifest['supervision_sha256']:
                raise ValueError('graph variants belong to different supervision')
        else:
            self.variant_directory = None

    def campaign(self, index):
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        directory = self.directory / ('campaign_%04d' % index)
        sources = {path.stem: (path, None) for path in directory.glob('*.npy')}
        if self.turn_directory is not None:
            path = self.turn_directory / ('campaign_%04d.npz' % index)
            with np.load(path, allow_pickle=False) as turns:
                sources.update({key: (path, key) for key in turns.files})
        if self.walk_directory is not None:
            sources['walks'] = (self.walk_directory / ('campaign_%04d.npy' % index), None)
        if self.variant_directory is not None:
            path = self.variant_directory / ('campaign_%04d.npz' % index)
            with np.load(path, allow_pickle=False) as variants:
                sources.update({key: (path, key) for key in variants.files})
        result = CampaignArrays(sources)
        self.loaded[index] = result
        while len(self.loaded) > 8:
            self.loaded.popitem(last=False)
        return result

    def blocks(self, partition, history):
        result = []
        for campaign in self.manifest['campaigns']:
            if self.campaign_partition[campaign['name']] == partition:
                result.extend((campaign['index'], start, min(start + history, campaign['decisions'])) for start in range(0, campaign['decisions'], history))
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


def neighbors(arrays, entity, cutoff, floor, count):
    ids = arrays['neighbor_ids'][arrays['neighbor_offsets'][entity]:arrays['neighbor_offsets'][entity + 1]]
    times = arrays['events'][ids, 2]
    lo, hi = np.searchsorted(times, (floor, cutoff))
    return np.asarray(ids[max(lo, hi - count):hi], np.int32)


def other_endpoint(events, ids, entity):
    return np.where(events[ids, 0] == entity, events[ids, 1], events[ids, 0])


def tree(arrays, entities, cutoffs, floor, count, depth, device):
    observations = np.concatenate([latest(arrays, [eid], cutoff) for eid, cutoff in zip(entities, cutoffs)])
    raw = np.zeros((len(entities), spec.FEATURES), np.float32)
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
            ids = neighbors(arrays, int(eid), int(cutoff), floor, count)
            size = len(ids)
            if size:
                children[i * count:i * count + size] = other_endpoint(arrays['events'], ids, eid)
                times[i * count:i * count + size] = arrays['events'][ids, 2]
                mask[i, :size] = True
                gaps[i, :size] = cutoff - arrays['events'][ids, 2]
                relation[i, :size] = np.where(arrays['events'][ids, 3] == 1, arrays['events'][ids, 6] + 1, spec.NON_RELATION)
        result.update(child=tree(arrays, children, times, floor, count, depth - 1, device),
            mask=tensor(mask, device), gap=tensor(gaps, device), relation=tensor(relation, device))
    return result


def history_inputs(arrays, start, stop, count, device, include_structure=False, selected=None):
    n = stop - start
    raw = np.zeros((n, 2, count, spec.FEATURES), np.float32)
    mask, gap, ids = np.zeros((n, 2, count), bool), np.zeros((n, 2, count), np.float32), np.full((n, 2, count), -1, np.int32)
    event_ids = np.full((n, 2, count), -1, np.int32)
    pair_gaps, pair_mask = np.zeros((n, count), np.float32), np.zeros((n, count), bool)
    structural = np.zeros((n, 2, count, 12), np.float32)
    common = np.zeros((n, spec.FEATURES), np.float32)
    common_ids = np.full((n, count), -1, np.int32)
    common_frequency = np.zeros((n, count), np.float32)
    common_stats = np.zeros((n, 4), np.float32)
    events = arrays['events']
    for i in range(n) if selected is None else np.flatnonzero(selected):
        query = start + i
        roots = int(arrays['actor'][query]), int(arrays['subject'][query])
        cutoff, floor = query * 2 + 1, start * 2
        lists = []
        memo = {}

        def partners(entity):
            if entity not in memo:
                ix = neighbors(arrays, entity, cutoff, floor, count)
                memo[entity] = other_endpoint(events, ix, entity).tolist()
            return memo[entity]

        def overlap(a, b):
            if not a or not b:
                return 0.0
            return sum(x == y for x in a for y in b) / (len(a) * len(b))

        def two_hop(entity):
            first = partners(entity)[-max(1, int(count**0.5)):]
            return list({p for other in first for p in partners(other)})

        for side, root in enumerate(roots):
            ix = neighbors(arrays, root, cutoff, floor, count)
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
    from .turns import enabled

    campaign, start, stop = block[:3]
    target_start = block[3] if len(block) == 4 else start
    a = bank.campaign(campaign)
    family = int(cfg['model'][1:])
    recipe = spec.resolve(cfg['design'])
    from .graph_variants import DESIGNS as VARIANTS

    if cfg['design'] in VARIANTS and 'boundary_observations' not in a:
        raise ValueError('graph variant requires the prepared companion cache')
    lo, hi = a['snapshot_offsets'][[start, stop]]
    n, count = stop - start, cfg['neighbors']
    entity = np.array(a['entity'][lo:hi])
    step = np.array(a['step'][lo:hi]) - start
    size = len(a['entity_kind'])
    align = np.full((n, size), hi - lo, np.int64)
    align[step, entity] = np.arange(hi - lo)
    rows = np.array(a['row_index'][start:stop])
    selected = (rows >= 0) & (np.arange(start, stop) >= target_start)
    query_context = np.array(a['query_context'][start:stop]) - lo
    b = dict(campaign=campaign, start=start, stop=stop, frames=n, time_steps=n, batch_size=1,
        campaigns=[campaign], entity_sizes=[size], entity_batch=torch.zeros(size, dtype=torch.long, device=device),
        query_steps=torch.arange(n, device=device), frame_queries=torch.arange(n, device=device)[:, None],
        entities=size, selected=tensor(selected, device),
        rows=rows[selected], target=tensor(bank.targets[rows[selected]], device),
        query=tensor(a['query'][start:stop], device, torch.float32),
        context=tensor(a['x'][a['query_context'][start:stop]], device, torch.float32),
        actor=tensor(a['actor'][start:stop], device, torch.long), subject=tensor(a['subject'][start:stop], device, torch.long),
        context_index=tensor(query_context, device, torch.long),
        entity_kind=tensor(a['entity_kind'], device, torch.long),
        signal=tensor(a['signals'][start:stop], device), affinity=tensor(a['affinity'][start:stop], device),
        global_affinity=tensor(a['global_affinity'][start:stop], device))
    if enabled(cfg):
        from .turn_inputs import add

        add(b, a, start, stop, cfg, device)
    if family <= 16 or family in (20, 29):
        b.update(x=tensor(a['x'][lo:hi], device, torch.float32), kind=tensor(a['entity_kind'][entity], device, torch.long),
                 entity=tensor(entity, device, torch.long), step=tensor(step, device, torch.long), align=tensor(align, device, torch.long),
                 present=tensor(align < hi - lo, device))
        first, last = np.searchsorted(a['spatial_step'], (start, stop))
        spatial = np.array(a['spatial'][first:last])
        attribute_rows = np.array(a['spatial_attribute_row'][first:last])
        if spec.COLOCATED in recipe.families:
            left, right = np.searchsorted(a['colocated_steps'], (start, stop))
            extra = np.array(a['colocated_edges'][left:right])
            spatial = np.concatenate((spatial, extra))
            attribute_rows = np.concatenate((attribute_rows, np.full(len(extra), -1, attribute_rows.dtype)))
        edges = spatial[:, :2].T - lo
        allowed = recipe.kind_mask()
        kinds = np.asarray(a['entity_kind'])[entity]
        keep = np.isin(spatial[:, 2], recipe.codes()) & allowed[kinds[edges[0]]] & allowed[kinds[edges[1]]]
        if not allowed.all():
            b['present'] &= tensor(allowed, device)[b['entity_kind']][None, :]
        b.update(edges=tensor(edges[:, keep], device, torch.long), relation=tensor(spatial[keep, 2], device, torch.long))
        position = np.full(len(spatial), -1, np.int64)
        position[keep] = np.arange(int(keep.sum()))
        b['attributes'], b['attribute_edges'] = {}, {}
        for name in recipe.attributed():
            declared = spec.BY_NAME[name]
            selected = keep & np.isin(spatial[:, 2], (declared.code, declared.reverse))
            block = np.asarray(a['attributes_' + name])[attribute_rows[selected]]
            b['attributes'][name] = tensor(block.reshape(-1, len(declared.attributes)), device, torch.float32)
            b['attribute_edges'][name] = tensor(position[selected], device, torch.long)
        b['frame_edges'], b['frame_directed'] = [], []
        directed = spec.CANONICAL[spatial[:, 2]]
        for t in range(n):
            use = keep & (step[edges[1]] == t)
            b['frame_edges'].append(tensor(entity[edges[:, use]], device, torch.long))
            b['frame_directed'].append(tensor(entity[edges[:, use & directed]], device, torch.long))
        if family in (13, 14):
            edges = a[recipe.lifted]
            left, right = np.searchsorted(edges[:, 1], (lo, hi))
            edges = np.array(edges[left:right])
            edges = edges[edges[:, 0] >= lo] - lo
            b.update(temporal_edges=tensor(edges.T, device, torch.long),
                temporal_gap=tensor(step[edges[:, 1]] - step[edges[:, 0]], device, torch.float32))
            if spec.BOUNDARY in recipe.families:
                extra = a['boundary_observations']
                left, right = np.searchsorted(extra[:, 1], (lo, hi))
                extra = np.array(extra[left:right])
                extra = extra[extra[:, 0] >= lo]
                boundary = extra[:, :2] - lo
                b.update(temporal_edges=tensor(np.concatenate((edges, boundary)).T, device, torch.long),
                    temporal_relation=tensor(np.r_[np.full(len(edges), spec.code('temporal')), extra[:, 2]], device, torch.long),
                    temporal_gap=tensor(np.r_[step[edges[:, 1]] - step[edges[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]], device, torch.float32),
                    temporal_turn_gap=tensor(np.r_[a['turn'][lo + edges[:, 1]] - a['turn'][lo + edges[:, 0]], extra[:, 3]], device, torch.float32))
            if enabled(cfg):
                from .turn_inputs import temporal_edges

                temporal_edges(b, a, lo, hi, device)
        if family == 12:
            left, right = np.searchsorted(a['mixed_tick'], (start * 2, stop * 2))
            edges = np.array(a['mixed_edges'])
            use = (edges[:, 0] >= left) & (edges[:, 1] >= left) & (edges[:, 0] < right) & (edges[:, 1] < right)
            edges = edges[use]
            ticks = np.array(a['mixed_tick'][left:right])
            mx = np.concatenate((a['mixed_x'][left:right], a['query'][start:stop])).astype(np.float32)
            mk = np.r_[a['mixed_kind'][left:right], np.full(n, 5)].astype(np.int64)
            qi = np.arange(right - left, right - left + n)
            extra = []
            for i in range(n):
                roots = {int(a['actor'][start + i]), int(a['subject'][start + i])}
                source = np.flatnonzero((ticks <= (start + i) * 2) & ((np.isin(a['mixed_entity'][left:right], list(roots))) | (a['mixed_kind'][left:right] == 2)))
                extra.extend((s, qi[i], 2048) for s in source)
            edge = np.concatenate((np.column_stack((edges[:, :2] - left, edges[:, 2])), np.asarray(extra, np.int64).reshape(-1, 3)))
            all_ticks = np.r_[ticks, np.arange(start, stop) * 2]
            feature = np.zeros((len(edge), 16), np.float32)
            feature[:, :13] = (edge[:, 2, None] & (1 << np.arange(13))) != 0
            feature[:, 13] = 1
            feature[:, 14] = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2)
            b['mixed'] = dict(x=tensor(mx, device), kind=tensor(mk, device), edge_index=tensor(edge[:, :2].T, device, torch.long),
                              edge=tensor(feature, device), query=tensor(qi, device, torch.long))
            if enabled(cfg):
                from .turn_inputs import mixed_edges

                mixed_edges(b, a, start, stop, device)
    if family in (15, 16, 18, 19, 20, 24, 26, 28, 29):
        left, right = np.searchsorted(a['events'][:, 2], (start * 2, stop * 2))
        events = np.array(a['events'][left:right])
        ex = np.array((a['x'][events[:, 4]].astype(np.float32) + a['x'][events[:, 5]]) / 2)
        interaction = events[:, 3] == 2
        ex[interaction] = a['query'][events[interaction, 2] // 2]
        initial = latest(a, np.arange(size), start * 2)
        initial_x = np.zeros((size, spec.FEATURES), np.float32)
        initial_x[initial >= 0] = a['x'][initial[initial >= 0]]
        b.update(events=tensor(events, device, torch.long), event_x=tensor(ex, device), initial=tensor(initial_x, device),
            event_groups=[(int(t), int(s), int(e)) for t, s, e in zip(*(
                lambda boundaries: (events[boundaries[:-1], 2], boundaries[:-1], boundaries[1:]))(
                    np.r_[0, np.flatnonzero(np.diff(events[:, 2])) + 1, len(events)]))])
        if family in (15, 16):
            edges = a['event_dag']
            use = (edges[:, 0] >= left) & (edges[:, 1] >= left) & (edges[:, 0] < right) & (edges[:, 1] < right)
            edges = np.array(edges[use]) - left
            b.update(dag_edges=tensor(edges.T, device, torch.long), dag_gap=tensor(events[edges[:, 1], 2] - events[edges[:, 0], 2], device, torch.float32))
            if spec.BOUNDARY in recipe.families:
                extra = a['boundary_events']
                low, high = np.searchsorted(extra[:, 1], (left, right))
                extra = np.array(extra[low:high])
                extra = extra[extra[:, 0] >= left]
                boundary = extra[:, :2] - left
                b.update(dag_edges=tensor(np.concatenate((edges, boundary)).T, device, torch.long),
                    dag_relation=tensor(np.r_[events[edges[:, 1], 3], extra[:, 2]], device, torch.long),
                    dag_gap=tensor(np.r_[events[edges[:, 1], 2] - events[edges[:, 0], 2], events[boundary[:, 1], 2] - events[boundary[:, 0], 2]], device, torch.float32))
            links = []
            for i in range(n):
                roots = [int(a['actor'][start + i]), int(a['subject'][start + i])]
                indices = np.flatnonzero((events[:, 2] <= (start + i) * 2) & np.isin(events[:, :2], roots).any(1))
                links.extend((int(e), i) for e in indices)
            b['dag_query_edges'] = tensor(np.asarray(links, np.int64).reshape(-1, 2).T, device, torch.long)
    if family == 17:
        roots = np.stack((a['actor'][start:stop], a['subject'][start:stop]), 1).reshape(-1)
        cutoffs = np.repeat(np.arange(start, stop) * 2 + 1, 2)
        b['tree'] = tree(a, roots, cutoffs, start * 2, count, cfg['layers'], device)
    if family in (18, 20, 21, 22, 23, 24, 26, 27):
        b.update(history_inputs(a, start, stop, count, device, family == 23,
            selected if family in (21, 22, 23, 27) else None))
    if enabled(cfg) and family in (17, 21, 22, 23, 27):
        from .turn_inputs import history_tokens

        history_tokens(b, a, start, stop, cfg, device)
    if family == 25:
        if cfg['walks'] > a['walks'].shape[2] or cfg['walk_hops'] > a['walks'].shape[3]:
            raise ValueError('requested CAWN walks exceed the prepared cache; run research.graph_temporal.walk_cache')
        walks = np.array(a['walks'][start:stop, :, :cfg['walks'], :cfg['walk_hops']])
        if enabled(cfg):
            extra = a['turn_walks'][start:stop, :, :cfg['turn_lookback'], :cfg['walks'], :cfg['walk_hops']]
            extra = np.array(extra).reshape(n, 2, cfg['turn_lookback'] * cfg['walks'], cfg['walk_hops'])
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
        current = np.broadcast_to(np.stack((a['actor'][start:stop], a['subject'][start:stop]), 1)[:, :, None], walks.shape[:3]).copy()
        for hop in range(cfg['walk_hops']):
            forward = events[:, :, :, hop, 0] == current
            nodes[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 1], events[:, :, :, hop, 0])
            observations[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 5], events[:, :, :, hop, 4])
            current = nodes[:, :, :, hop]
        features = np.array(a['x'][observations], np.float32)
        features[~known] = 0
        gap = np.arange(start, stop)[:, None, None, None] * 2 + 1 - events[..., 2]
        identities = np.zeros(walks.shape + (2 * cfg['walk_hops'],), np.float32)
        for q in range(n):
            for side in range(2):
                for hop in range(cfg['walk_hops']):
                    reference = nodes[q, side, :, hop]
                    identities[q, ..., side * cfg['walk_hops'] + hop] = ((nodes[q, ..., None] == reference) & known[q, side, :, hop]).sum(-1)
        b.update(walk_x=tensor(features, device), walk_mask=tensor(known, device), walk_gap=tensor(gap, device, torch.float32), walk_identity=tensor(identities, device))
    if enabled(cfg):
        from .turn_inputs import capture_schedule

        capture_schedule(b)
    return b
