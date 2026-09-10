import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cache import write_json
from . import spec


def search(arrays, entities, cutoffs):
    lo = np.array(arrays['neighbor_offsets'][entities])
    hi = np.array(arrays['neighbor_offsets'][entities + 1])
    while np.any(lo < hi):
        active = lo < hi
        middle = (lo + hi) // 2
        ids = arrays['neighbor_ids'][middle[active]]
        before = arrays['events'][ids, 2] < cutoffs[active]
        lo[active] = np.where(before, middle[active] + 1, lo[active])
        hi[active] = np.where(before, hi[active], middle[active])
    return lo


def extend(arrays, count=spec.CONSTRUCTION['expanded_walk']['samples']):
    original = arrays['walks']
    if count <= original.shape[2]:
        return np.array(original[:, :, :count])
    shape = original.shape[:2] + (count - original.shape[2],)
    result = np.full(original.shape[:2] + (count, original.shape[3]), -1, np.int32)
    result[:, :, :original.shape[2]] = original
    current = np.broadcast_to(np.stack((arrays['actor'], arrays['subject']), 1)[:, :, None], shape).copy().reshape(-1)
    query = np.broadcast_to(np.arange(len(original))[:, None, None], shape).reshape(-1)
    cutoff = query * 2 + 1
    floor = np.maximum(0, (query - spec.CONSTRUCTION['walk']['window'] + 1) * 2)
    valid = np.ones(len(current), bool)
    for hop in range(original.shape[3]):
        lo, hi = search(arrays, current, floor), search(arrays, current, cutoff)
        lo = np.maximum(lo, hi - spec.CONSTRUCTION['walk']['fanout'])
        valid &= hi > lo
        for extra in range(shape[2]):
            rng = np.random.default_rng(np.random.SeedSequence([spec.CONSTRUCTION['walk']['seed'], extra + original.shape[2], hop]))
            selected = np.arange(extra, len(current), shape[2])
            selected = selected[valid[selected]]
            chosen = lo[selected] + (rng.random(len(selected)) * (hi[selected] - lo[selected])).astype(np.int64)
            event_ids = arrays['neighbor_ids'][chosen]
            event = arrays['events'][event_ids]
            result[:, :, extra + original.shape[2], hop].reshape(-1)[valid.reshape(shape)[:, :, extra].reshape(-1)] = event_ids
            current[selected] = np.where(event[:, 0] == current[selected], event[:, 1], event[:, 0])
            cutoff[selected] = event[:, 2]
    return result


def cache_path(directory):
    return directory.with_name(directory.name + '_walks%d' % spec.CONSTRUCTION['expanded_walk']['samples'])


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
            for key in ('actor', 'subject', 'events', 'neighbor_ids', 'neighbor_offsets', 'walks')}
        expanded = extend(arrays)
        if not np.array_equal(expanded[:, :, :8], arrays['walks']):
            raise ValueError('original walks changed')
        times = np.where(expanded >= 0, arrays['events'][np.maximum(expanded, 0), 2], -1)
        if np.any(times[..., 0] >= np.arange(len(expanded))[:, None, None] * 2 + 1):
            raise ValueError('walk starts in the future')
        if np.any((times[..., 1:] >= times[..., :-1]) & (expanded[..., 1:] >= 0)):
            raise ValueError('walk does not move strictly into the past')
        path = target / (name + '.npy')
        np.save(path, expanded, allow_pickle=False)
        files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(files) % 250 == 0:
            print(json.dumps(dict(event='walk_cache_progress', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)
    write_json(target / 'manifest.json', dict(supervision_sha256=manifest['supervision_sha256'],
        walks=spec.CONSTRUCTION['expanded_walk']['samples'], hops=spec.CONSTRUCTION['walk']['hops'],
        files=files, seconds=time.perf_counter() - started))
    print(json.dumps(dict(event='walk_cache_complete', campaigns=len(files), seconds=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    prepare(parser.parse_args().directory)
