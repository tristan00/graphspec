import numpy as np
import torch

from .data import tensor, tree


def add(b, a, start, stop, cfg, device):
    if 'turn_membership' not in a:
        raise ValueError('turn-end variant requires the prepared turn cache')
    queries = np.arange(start, stop)
    membership = a['turn_membership'][queries]
    previous = membership[:, None] - np.arange(1, cfg['turn_lookback'] + 1)
    valid = previous >= 0
    prior = previous.clip(0)
    ends = a['turn_ends'][prior]
    valid &= ends >= start
    roots = np.stack((a['actor'][queries], a['subject'][queries], a['entity'][a['query_context'][queries, 2]]), 1)
    observed = a['turn_observations'][prior[:, :, None], roots[:, None, :]]
    mask = (observed >= 0) & valid[:, :, None]
    raw = np.array(a['x'][observed.clip(0)], np.float32)
    raw[~mask] = 0
    age = np.maximum(ends[:, :, None] - a['step'][observed.clip(0)], 0)
    age[~mask] = 0
    delta = queries[:, None] - ends
    turn_gap = a['turn_numbers'][membership, None] - a['turn_numbers'][prior]
    gaps = np.stack((delta, turn_gap, age.max(-1)), -1).clip(0).astype(np.float32)
    token_gaps = np.stack((queries[:, None, None] - a['step'][observed.clip(0)],
        np.broadcast_to(turn_gap[:, :, None], age.shape), age), -1).clip(0).astype(np.float32)
    local = observed - int(a['snapshot_offsets'][start])
    b.update(end_steps=tensor(np.where(valid, ends - start, -1), device, torch.long),
        end_mask=tensor(valid, device), end_gaps=tensor(gaps, device),
        end_raw=tensor(raw.transpose(0, 2, 1, 3), device), end_raw_mask=tensor(mask.transpose(0, 2, 1), device),
        end_token_gaps=tensor(token_gaps.transpose(0, 2, 1, 3), device),
        end_observation_local=tensor(np.where(mask & (local >= 0), local, -1), device, torch.long))
    if cfg['model'] == 'M28':
        last = a['turn_actor_decision'][prior, roots[:, 0, None]]
        affinity = np.array(a['affinity'][last.clip(0)], np.float32)
        categories = a['turn_action_category'][last.clip(0)]
        affinity += np.eye(affinity.shape[-1], dtype=np.float32)[categories]
        affinity[(last < 0) | ~valid] = 0
        global_affinity = np.array(a['global_affinity'][ends], np.float32)
        global_affinity += np.eye(affinity.shape[-1], dtype=np.float32)[a['turn_action_category'][ends]]
        global_affinity[~valid] = 0
        b.update(end_affinity=tensor(affinity, device), end_global_affinity=tensor(global_affinity, device))


def temporal_edges(b, a, lo, hi, device):
    nodes = np.arange(lo, hi)
    query = a['step'][nodes]
    membership = a['turn_membership'][query]
    prior = membership[:, None] - np.arange(1, b['end_steps'].shape[1] + 1)
    valid = prior >= 0
    observed = a['turn_observations'][prior.clip(0), a['entity'][nodes, None]]
    valid &= observed >= 0
    valid &= a['turn_ends'][prior.clip(0)] >= b['start']
    source = observed[valid]
    destination = np.broadcast_to(nodes[:, None], observed.shape)[valid]
    ghosts = np.unique(source[source < lo])
    if len(ghosts):
        count = len(b['x'])
        b['x'] = torch.cat((b['x'], tensor(a['x'][ghosts], device, torch.float32)))
        b['kind'] = torch.cat((b['kind'], tensor(a['entity_kind'][a['entity'][ghosts]], device, torch.long)))
        b['entity'] = torch.cat((b['entity'], tensor(a['entity'][ghosts], device, torch.long)))
        b['step'] = torch.cat((b['step'], tensor(a['step'][ghosts] - b['start'], device, torch.long)))
        b['align'] = torch.where(b['align'] == count, len(b['x']), b['align'])
    source_local = np.where(source >= lo, source - lo, hi - lo + np.searchsorted(ghosts, source))
    extra = np.stack((source_local, destination - lo))
    original = b['temporal_edges'].cpu().numpy()
    relation = b['temporal_relation'].cpu().numpy() if 'temporal_relation' in b else np.full(original.shape[1], 41)
    pairs = {tuple(edge): int(kind) for edge, kind in zip(original.T.tolist(), relation)}
    for edge in extra.T.tolist():
        pairs.setdefault(tuple(edge), 42)
        if pairs[tuple(edge)] == 41:
            pairs[tuple(edge)] = 42
    edges = np.asarray(list(pairs), np.int64).reshape(-1, 2)
    original_ids = np.r_[np.arange(lo, hi), ghosts]
    steps = a['step'][original_ids]
    turns = a['turn'][original_ids]
    b.update(temporal_edges=tensor(edges.T, device, torch.long),
        temporal_relation=tensor(np.asarray(list(pairs.values())), device, torch.long),
        temporal_gap=tensor(steps[edges[:, 1]] - steps[edges[:, 0]], device, torch.float32),
        temporal_turn_gap=tensor(turns[edges[:, 1]] - turns[edges[:, 0]], device, torch.float32))


def mixed_edges(b, a, start, stop, device):
    mixed = b['mixed']
    mask = b['end_raw_mask'].reshape(-1)
    raw = b['end_raw'].reshape(-1, 128)[mask]
    source = torch.arange(len(raw), device=device) + len(mixed['x'])
    destination = mixed['query'][:, None, None].expand(-1, 3, b['end_steps'].shape[1]).reshape(-1)[mask]
    gaps = b['end_token_gaps'].reshape(-1, 3)[mask]
    features = raw.new_zeros(len(raw), 16)
    features[:, 12:14] = 1
    features[:, 14:16] = gaps[:, :2].log1p()
    mixed['x'] = torch.cat((mixed['x'], raw))
    mixed['kind'] = torch.cat((mixed['kind'], torch.zeros(len(raw), dtype=torch.long, device=device)))
    mixed['edge_index'] = torch.cat((mixed['edge_index'], torch.stack((source, destination))), 1)
    mixed['edge'] = torch.cat((mixed['edge'], features))


def history_tokens(b, a, start, stop, cfg, device):
    n, lookback = stop - start, cfg['turn_lookback']
    if cfg['model'] == 'M17':
        roots = np.stack((a['actor'][start:stop], a['subject'][start:stop]), 1)
        roots = np.broadcast_to(roots[:, :, None], (n, 2, lookback)).reshape(-1)
        ends = b['end_steps'].cpu().numpy() + start
        cutoffs = np.broadcast_to((ends * 2 + 1)[:, None, :], (n, 2, lookback)).reshape(-1)
        b['end_tree'] = tree(a, roots, cutoffs, start * 2, cfg['neighbors'], cfg['layers'], device)
        return
    original = b['history_mask'].shape[-1]
    valid = b['end_raw_mask'][:, :2]
    gaps = b['end_token_gaps'][:, :2]
    b['history_x'] = torch.cat((b['history_x'], b['end_raw'][:, :2]), 2)
    b['history_mask'] = torch.cat((b['history_mask'], valid), 2)
    b['history_gap'] = torch.cat((b['history_gap'], gaps[..., 0] * 2 + 1), 2)
    end_time = b['history_x'].new_zeros(n, 2, original + lookback, 3)
    end_time[:, :, original:] = gaps
    b['history_end_time'] = end_time
    b['history_turn_kind'] = torch.cat((torch.zeros(n, 2, original, dtype=torch.long, device=device),
        torch.ones(n, 2, lookback, dtype=torch.long, device=device)), 2)
    for key, width in (('occurrence', 2), ('structure', 12)):
        b[key] = torch.cat((b[key], b[key].new_zeros(n, 2, lookback, width)), 2)
    order = b['history_gap'].masked_fill(~b['history_mask'], torch.inf).argsort(dim=-1, descending=True, stable=True)
    for key in ('history_x', 'history_mask', 'history_gap', 'history_end_time', 'history_turn_kind', 'occurrence', 'structure'):
        value = b[key]
        indices = order if value.ndim == 3 else order[..., None].expand_as(value)
        b[key] = value.gather(2, indices)


def capture_schedule(b):
    if 'end_steps' not in b:
        return
    ends = b['end_steps'].cpu().numpy()
    result = []
    for time_step in range(b['time_steps']):
        query, slot = np.nonzero((ends >= 0) & (ends % b['time_steps'] == time_step))
        flat = torch.as_tensor(query * ends.shape[1] + slot, device=b['query'].device, dtype=torch.long)
        roots = torch.stack((b['actor'][query], b['subject'][query]), -1)
        result.append((flat, roots))
    b['end_capture'] = result
