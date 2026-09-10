import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import spec
from .cache import validate, write_json
from .schema import DESIGNS


VIEW_ARRAYS = {
    'S-core': ('x', 'entity', 'entity_kind', 'snapshot_offsets', 'spatial', 'spatial_step', 'query_context'),
    'S-full': ('x', 'entity', 'entity_kind', 'snapshot_offsets', 'spatial', 'spatial_attribute_row', 'query_context'),
    'X-chain': ('x', 'step', 'turn', 'spatial', 'chain'),
    'X-skip': ('x', 'step', 'turn', 'spatial', 'skip'),
    'X-allpast': ('x', 'step', 'turn', 'spatial', 'allpast'),
    'O-mixed': ('mixed_x', 'mixed_kind', 'mixed_tick', 'mixed_entity', 'mixed_edges', 'mixed_member_ids', 'mixed_member_offsets'),
    'D-event': ('events', 'event_dag', 'x'),
    'E-entity': ('events', 'x', 'entity_kind', 'observation_ids', 'observation_offsets'),
    'B-role': ('events', 'role_left', 'role_right', 'x'),
    'H-neighbor': ('events', 'neighbor_ids', 'neighbor_offsets', 'observation_ids', 'observation_offsets'),
    'W-temporal': ('walks', 'propagation_ids', 'events', 'neighbor_ids', 'neighbor_offsets'),
    'N-affinity': ('actor', 'affinity', 'global_affinity'),
    'V-signals': ('signals', 'query'),
}


def audit(directory):
    from advisor.base_model import stable_split

    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    supervision_bytes = (directory / 'supervision.json').read_bytes()
    assert hashlib.sha256(supervision_bytes).hexdigest() == manifest['supervision_sha256']
    supervision = json.loads(supervision_bytes)
    rows, split = supervision['rows'], supervision['split']
    validation, train = stable_split(len(rows), [r['campaign'] for r in rows])
    assert train == split['train_indices'] and validation == split['validation_indices']
    assert set(rows[i]['campaign'] for i in train).isdisjoint(rows[i]['campaign'] for i in validation)
    assert set(manifest['designs']) == set(DESIGNS) == set(VIEW_ARRAYS)
    assert manifest['spec_sha256'] == spec.fingerprint()
    seen, counts, unresolved = [], Counter(), Counter()
    for campaign in manifest['campaigns']:
        path = directory / ('campaign_%04d' % campaign['index'])
        arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in path.glob('*.npy')}
        assert all(set(names) <= arrays.keys() for names in VIEW_ARRAYS.values())
        validate(arrays)
        selected = arrays['row_index'] >= 0
        for index, decision in zip(arrays['row_index'][selected], arrays['decision'][selected]):
            assert rows[index]['campaign'] == campaign['name'] and rows[index]['decision'] == decision
        seen.extend(arrays['row_index'][selected].tolist())
        for name, values in arrays.items():
            assert len(values) == campaign['counts'][name]
        counts.update(campaign['counts'])
        unresolved.update(campaign['unresolved_subjects'])
    assert sorted(seen) == list(range(len(rows)))
    result = dict(ok=True, seconds=time.perf_counter() - started, campaigns=len(manifest['campaigns']),
        rows=len(rows), train=len(train), holdout=len(validation), view_arrays=VIEW_ARRAYS,
        counts=dict(counts), unresolved_subjects=dict(unresolved), supervision_sha256=manifest['supervision_sha256'])
    write_json(directory / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('counts', 'view_arrays')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    audit(parser.parse_args().directory)


if __name__ == '__main__':
    main()
