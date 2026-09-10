import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import spec
from .cache import write_json


VERSION = 2
DESIGNS = tuple(sorted({base + '+' + modifier.name for modifier in spec.MODIFIERS for base in modifier.bases}))
LOOKBACK = spec.CONSTRUCTION['boundary']['lookback']
WINDOW = spec.CONSTRUCTION['boundary']['window']
COLOCATED = spec.code('colocated')
BOUNDARY_BY_FLAG = np.asarray([-1, spec.code('boundary_first'), spec.code('boundary_last'), spec.code('boundary_both')], np.int32)


def cache_path(directory):
    directory = Path(directory)
    return directory.with_name(directory.name + '_variants')


def marked_edges(pieces, turns):
    if not pieces:
        return np.empty((0, 4), np.int32)
    values = np.concatenate(pieces).astype(np.int32)
    pairs, inverse = np.unique(values[:, :2], axis=0, return_inverse=True)
    flags = np.zeros(len(pairs), np.int32)
    np.bitwise_or.at(flags, inverse, values[:, 2])
    output = np.column_stack((pairs, BOUNDARY_BY_FLAG[flags], turns[pairs[:, 1]] - turns[pairs[:, 0]])).astype(np.int32)
    return output[np.lexsort((output[:, 0], output[:, 1]))]


def colocated(arrays):
    spatial = arrays['spatial']
    kind = arrays['entity_kind'][arrays['entity']]
    relation = spec.code('located')
    allowed = np.isin(kind[spatial[:, 0]], [spec.kind('character'), spec.kind('settlement')])
    located = spatial[(spatial[:, 2] == relation) & allowed]
    groups = defaultdict(set)
    for source, location, _ in located:
        groups[int(location)].add(int(source))
    edges = [(a, b, COLOCATED) for members in groups.values() for a in sorted(members) for b in sorted(members) if a != b]
    edges = np.asarray(edges, np.int32).reshape(-1, 3)
    steps = arrays['step'][edges[:, 1]]
    order = np.lexsort((edges[:, 1], edges[:, 0], steps))
    return dict(colocated_edges=edges[order], colocated_steps=steps[order])


def observation_boundaries(arrays):
    step, entity = arrays['step'], arrays['entity']
    turns = arrays['decision_turn']
    membership = np.cumsum(np.r_[0, np.diff(turns) != 0]).astype(np.int32)
    groups = membership[step]
    shape = (int(membership[-1]) + 1, len(arrays['entity_kind']))
    first = np.full(shape, len(entity), np.int32)
    last = np.full(shape, -1, np.int32)
    ids = np.arange(len(entity), dtype=np.int32)
    np.minimum.at(first, (groups, entity), ids)
    np.maximum.at(last, (groups, entity), ids)
    pieces = []
    for lag in range(1, LOOKBACK + 1):
        prior = groups - lag
        for endpoints, flag in ((first, 1), (last, 2)):
            source = endpoints[prior.clip(0), entity]
            valid = (prior >= 0) & (source >= 0) & (source < len(entity))
            selected = ids[valid]
            source = source[valid]
            keep = step[selected] - step[source] < WINDOW
            pieces.append(np.column_stack((source[keep], selected[keep], np.full(keep.sum(), flag))))
    return marked_edges(pieces, turns[step])


def event_boundaries(arrays):
    events = arrays['events']
    turns = arrays['decision_turn']
    membership = np.cumsum(np.r_[0, np.diff(turns) != 0]).astype(np.int32)
    groups = membership[events[:, 2] // 2]
    participants = defaultdict(lambda: defaultdict(list))
    for i, (left, right) in enumerate(events[:, :2]):
        participants[int(left)][int(groups[i])].append(i)
        if right != left:
            participants[int(right)][int(groups[i])].append(i)
    pieces = []
    for history in participants.values():
        for turn, destinations in history.items():
            destination = np.asarray(destinations, np.int32)
            for lag in range(1, LOOKBACK + 1):
                if turn - lag not in history:
                    continue
                prior = np.asarray(history[turn - lag], np.int32)
                ticks = events[prior, 2]
                for tick, flag in ((ticks[0], 1), (ticks[-1], 2)):
                    anchors = prior[ticks == tick]
                    sources = np.repeat(anchors, len(destination))
                    targets = np.tile(destination, len(anchors))
                    keep = events[targets, 2] // 2 - events[sources, 2] // 2 < WINDOW
                    pieces.append(np.column_stack((sources[keep], targets[keep], np.full(keep.sum(), flag))))
    return marked_edges(pieces, turns[events[:, 2] // 2])


def build(arrays):
    if np.any(np.diff(arrays['decision_turn']) < 0):
        raise ValueError('graph variants require monotonic campaign turns')
    return dict(colocated(arrays), boundary_observations=observation_boundaries(arrays), boundary_events=event_boundaries(arrays))


def validate(arrays, extra):
    edges = extra['colocated_edges']
    if not np.array_equal(arrays['step'][edges[:, 0]], extra['colocated_steps']) or not np.array_equal(arrays['step'][edges[:, 1]], extra['colocated_steps']):
        raise ValueError('colocation edge crosses decisions')
    if np.any(edges[:, 0] == edges[:, 1]) or np.any(edges[:, 2] != COLOCATED):
        raise ValueError('invalid colocation relation')
    for name, step, participants in (
        ('boundary_observations', arrays['step'], arrays['entity'][:, None]),
        ('boundary_events', arrays['events'][:, 2] // 2, arrays['events'][:, :2])):
        source, target, relation, gaps = extra[name].T
        turns = arrays['decision_turn'][step]
        shared = (participants[source, :, None] == participants[target, None, :]).any((1, 2))
        if not shared.all() or not (turns[source] < turns[target]).all():
            raise ValueError('boundary edge lacks a shared participant or reads an incomplete turn')
        if not np.isin(relation, spec.BOUNDARY_CODES).all() or not np.array_equal(gaps, turns[target] - turns[source]):
            raise ValueError('invalid turn-boundary relation or gap')
        if not ((step[target] - step[source]) < WINDOW).all():
            raise ValueError('turn-boundary edge exceeds its cached window')


def verify(directory):
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    target = cache_path(directory)
    cached = json.loads((target / 'manifest.json').read_text())
    if cached['version'] != VERSION or cached['supervision_sha256'] != manifest['supervision_sha256']:
        raise ValueError('graph variant version or supervision does not match')
    expected = {'campaign_%04d.npz' % c['index'] for c in manifest['campaigns']}
    if len(cached['files']) != len(expected) or {entry['file'] for entry in cached['files']} != expected:
        raise ValueError('graph variants do not cover every campaign')
    for entry in cached['files']:
        if hashlib.sha256((target / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('graph variant checksum mismatch: ' + entry['file'])
    print(json.dumps(dict(event='graph_variants_verified', campaigns=len(expected), seconds=time.perf_counter() - started)), flush=True)
    return cached


def prepare(directory):
    started = time.perf_counter()
    directory = Path(directory)
    target = cache_path(directory)
    if (target / 'manifest.json').exists():
        return verify(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    target.mkdir(exist_ok=True)
    files = []
    for campaign in manifest['campaigns']:
        name = 'campaign_%04d' % campaign['index']
        arrays = {key: np.load(directory / name / (key + '.npy'), mmap_mode='r', allow_pickle=False)
            for key in ('spatial', 'entity', 'entity_kind', 'step', 'decision_turn', 'events')}
        extra = build(arrays)
        validate(arrays, extra)
        path = target / (name + '.npz')
        np.savez_compressed(path, **extra)
        files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
            edges={key:len(value) for key,value in extra.items() if key != 'colocated_steps'}))
        if len(files) % 100 == 0:
            print(json.dumps(dict(event='graph_variants_progress', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)
    write_json(target / 'manifest.json', dict(version=VERSION, supervision_sha256=manifest['supervision_sha256'],
        spec_sha256=spec.fingerprint(),
        designs=DESIGNS, previous_completed_turns=LOOKBACK, decision_window=WINDOW, files=files,
        definition='Same-location spatial shortcuts and typed first/last completed-turn participant links; original graphs retained.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seconds=time.perf_counter() - started))
    verified = verify(directory)
    print(json.dumps(dict(event='graph_variants_complete', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    prepare(parser.parse_args().directory)
