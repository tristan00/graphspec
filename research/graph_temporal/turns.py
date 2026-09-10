import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .cache import write_json


def enabled(cfg):
    return cfg.get('turn_context', 'none') == 'end'


def graph_name(cfg):
    return cfg['design'] + ('+turn-end' if enabled(cfg) else '')


def cache_path(directory):
    return directory.with_name(directory.name + '_turns')


def index(arrays):
    turns = np.asarray(arrays['decision_turn'])
    if np.any(np.diff(turns) < 0):
        raise ValueError('campaign turn numbers are not monotonic')
    starts = np.r_[0, np.flatnonzero(np.diff(turns)) + 1].astype(np.int32)
    ends = np.r_[starts[1:] - 1, len(turns) - 1].astype(np.int32)
    membership = np.repeat(np.arange(len(starts)), ends - starts + 1).astype(np.int32)
    observed = np.full((len(ends), len(arrays['entity_kind'])), -1, np.int32)
    state = np.full(len(arrays['entity_kind']), -1, np.int32)
    previous = 0
    for turn, end in enumerate(ends):
        stop = int(arrays['snapshot_offsets'][end + 1])
        ids = np.arange(previous, stop)
        np.maximum.at(state, arrays['entity'][ids], ids)
        observed[turn] = state
        previous = stop
    actor_decision = np.full((len(ends), len(arrays['entity_kind'])), -1, np.int32)
    last_actor = np.full(len(arrays['entity_kind']), -1, np.int32)
    categories = np.zeros(len(turns), np.int32)
    choices = arrays['events'][arrays['events'][:, 3] == 2]
    categories[choices[:, 2] // 2] = choices[:, 6]
    for turn, (start, end) in enumerate(zip(starts, ends)):
        decisions = np.arange(start, end + 1)
        np.maximum.at(last_actor, arrays['actor'][decisions], decisions)
        actor_decision[turn] = last_actor
    return dict(turn_starts=starts, turn_ends=ends, turn_membership=membership,
        turn_actor_decision=actor_decision, turn_action_category=categories,
        turn_numbers=turns[starts], turn_observations=observed)


def walks(arrays, turns, count=32, hops=4, lookback=3):
    from .walk_cache import search

    shape = (len(arrays['actor']), 2, lookback, count)
    current = np.broadcast_to(np.stack((arrays['actor'], arrays['subject']), 1)[:, :, None, None], shape).copy().reshape(-1)
    prior = turns['turn_membership'][:, None] - np.arange(1, lookback + 1)
    ends = turns['turn_ends'][prior.clip(0)]
    cutoff = np.broadcast_to((2 * ends + 2)[:, None, :, None], shape).copy().reshape(-1)
    floor = np.broadcast_to((2 * turns['turn_starts'][prior.clip(0)])[:, None, :, None], shape).reshape(-1)
    valid = np.broadcast_to((prior >= 0)[:, None, :, None], shape).copy().reshape(-1)
    result = np.full((len(current), hops), -1, np.int32)
    for hop in range(hops):
        lo, hi = search(arrays, current, floor), search(arrays, current, cutoff)
        lo = np.maximum(lo, hi - 64)
        valid &= hi > lo
        selected = np.flatnonzero(valid)
        rng = np.random.default_rng(np.random.SeedSequence([909, hop]))
        chosen = lo[selected] + (rng.random(len(selected)) * (hi[selected] - lo[selected])).astype(np.int64)
        ids = arrays['neighbor_ids'][chosen]
        event = arrays['events'][ids]
        result[selected, hop] = ids
        current[selected] = np.where(event[:, 0] == current[selected], event[:, 1], event[:, 0])
        cutoff[selected] = event[:, 2]
    return result.reshape(shape + (hops,))


def verify(directory):
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    target = cache_path(directory)
    cached = json.loads((target / 'manifest.json').read_text())
    if cached['version'] != 2 or cached['supervision_sha256'] != manifest['supervision_sha256']:
        raise ValueError('turn cache version or supervision does not match')
    expected = {'campaign_%04d.npz' % c['index'] for c in manifest['campaigns']}
    if {entry['file'] for entry in cached['files']} != expected:
        raise ValueError('turn cache does not cover every campaign')
    for entry in cached['files']:
        if hashlib.sha256((target / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('turn cache checksum mismatch: ' + entry['file'])
    print(json.dumps(dict(event='turn_cache_verified', campaigns=len(expected), seconds=time.perf_counter() - started)), flush=True)
    return cached


def prepare(directory):
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    target = cache_path(directory)
    target.mkdir(exist_ok=True)
    files = []
    for campaign in manifest['campaigns']:
        name = 'campaign_%04d' % campaign['index']
        arrays = {key: np.load(directory / name / (key + '.npy'), mmap_mode='r')
            for key in ('decision_turn', 'entity_kind', 'entity', 'snapshot_offsets', 'actor', 'subject', 'events', 'neighbor_ids', 'neighbor_offsets')}
        result = index(arrays)
        result['turn_walks'] = walks(arrays, result)
        path = target / (name + '.npz')
        np.savez_compressed(path, **result)
        files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), turns=len(result['turn_ends'])))
        if len(files) % 250 == 0:
            print(json.dumps(dict(event='turn_cache_progress', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)
    write_json(target / 'manifest.json', dict(supervision_sha256=manifest['supervision_sha256'],
        version=2, lookback=3, walks=32, hops=4,
        definition='Last observed entity state and causal actor/subject walks at completed past-turn cutoffs; no turn-summary nodes.',
        files=files, seconds=time.perf_counter() - started))
    verify(directory)
    print(json.dumps(dict(event='turn_cache_complete', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)


def blocks(bank, original, cfg):
    if not enabled(cfg):
        return original
    output = []
    for campaign, start, stop in original:
        arrays = bank.campaign(campaign)
        turn = arrays['turn_membership'][start]
        begin = int(arrays['turn_starts'][max(0, turn - cfg['turn_lookback'])]) if turn else start
        output.append((campaign, begin, stop, start))
    return output


class EndContext(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.value = nn.Linear(width, width, bias=False)
        self.time = nn.Linear(3, width)
        self.gate = nn.Linear(width * 2, width)

    def forward(self, current, previous, gaps, mask):
        weights = mask.to(current.dtype)
        value = self.value(previous) + self.time(gaps.log1p())
        context = (value * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]
        update = self.gate(torch.cat((current, context), -1)).sigmoid() * context
        return current + update * mask.any(1)[:, None]


def gather_end(sequence, entities, b):
    times = b['end_steps'].clamp_min(0) % b['time_steps']
    return sequence[times, entities[:, None]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    prepare(parser.parse_args().directory)
