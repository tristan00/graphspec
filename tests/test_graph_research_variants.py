from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.graph_temporal.data import pack
from research.graph_temporal import spec
from research.graph_temporal.graph_variants import DESIGNS, build, prepare, validate, verify
from research.graph_temporal.registry import DESIGNS as MODELS, create, sample
from research.graph_temporal.turns import index, walks
from tests.test_graph_research_cache import make, records


def source(changed=False):
    entries = records(12)
    for step, (row, record) in enumerate(entries):
        record['world']['armies'][0]['region'] = 'r1' if step < 6 else 'r2'
        if changed:
            row['y'] = 99999
            record['campaign'].update(validated=True, execution_succeeded=True, raw_reward=99999)
            if step >= 8:
                record['campaign']['treasury'] = 99999
                record['world']['armies'][0].update(region='r1', units=999)
    _, arrays = make(entries)
    arrays.update(row_index=np.arange(12), decision_turn=np.arange(12) // 2 + 1)
    arrays.update(index(arrays))
    arrays['turn_walks'] = walks(arrays, arrays)
    arrays.update(build(arrays))
    return arrays


def test_colocation_is_observed_same_decision_and_stops_when_entity_moves():
    arrays = source()
    validate(arrays, arrays)
    for step in range(12):
        ids = range(*arrays['snapshot_offsets'][step:step + 2])
        actor = next(i for i in ids if arrays['entity'][i] == arrays['actor'][step])
        subject = next(i for i in ids if arrays['entity'][i] == arrays['subject'][step])
        pairs = set(map(tuple, arrays['colocated_edges'][arrays['colocated_steps'] == step, :2]))
        assert ((actor, subject) in pairs) == (step < 6)
        assert ((subject, actor) in pairs) == (step < 6)


def test_observation_links_mark_first_last_and_single_observation_turns():
    arrays = source()
    actor = arrays['actor'][0]
    observed = np.flatnonzero(arrays['entity'] == actor)
    target = observed[4]
    edges = arrays['boundary_observations']
    incoming = {(int(s), int(r), int(g)) for s, d, r, g in edges if d == target}
    assert (int(observed[2]), spec.code('boundary_first'), 1) in incoming
    assert (int(observed[3]), spec.code('boundary_last'), 1) in incoming
    assert all(s < target for s, _, _ in incoming)
    single = deepcopy(arrays)
    single['decision_turn'] = np.arange(12)
    extra = build(single)
    assert set(extra['boundary_observations'][:, 2]) == {spec.code('boundary_both')}


def test_event_boundaries_preserve_all_events_tied_at_endpoint_ticks():
    arrays = source()
    events = arrays['events']
    turns = arrays['decision_turn'][events[:, 2] // 2]
    edges = arrays['boundary_events']
    incoming = {(int(s), int(d)):int(r) for s, d, r, _ in edges}
    for participant in np.unique(events[:, :2]):
        ids = np.flatnonzero((events[:, :2] == participant).any(1))
        prior = ids[turns[ids] == 1]
        targets = ids[turns[ids] == 2]
        if not len(prior):
            continue
        for flag, tick in ((1, events[prior[0], 2]), (2, events[prior[-1], 2])):
            for s in prior[events[prior, 2] == tick]:
                for d in targets:
                    assert (incoming[int(s), int(d)] - 42) & flag


def test_graph_variants_ignore_future_and_execution_outcomes():
    before, after = source(), source(True)
    for name, steps in (('boundary_observations', before['step']), ('boundary_events', before['events'][:, 2] // 2)):
        left = before[name][steps[before[name][:, 1]] < 8]
        other_steps = after['step'] if name == 'boundary_observations' else after['events'][:, 2] // 2
        right = after[name][other_steps[after[name][:, 1]] < 8]
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(before['colocated_edges'][before['colocated_steps'] < 8], after['colocated_edges'][after['colocated_steps'] < 8])


@pytest.mark.parametrize('model,design', [(m,d) for m,ds in MODELS.items() for d in ds if d in DESIGNS])
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_new_graph_predictions_are_causal(model, design, turn_context):
    torch.set_num_threads(2)
    cfg = sample(model, design, np.random.default_rng(7), 7)
    cfg.update(hidden=16, layers=2, dropout=0, neighbors=4, turn_context=turn_context, turn_lookback=3)
    torch.manual_seed(7)
    net = create(cfg, 1).eval()
    outputs = []
    for changed in (False, True):
        arrays = source(changed)
        bank = SimpleNamespace(campaign=lambda _: arrays, targets=np.arange(12, dtype=np.float32))
        with torch.no_grad():
            outputs.append(net(pack(bank, (0, 0, 12), cfg, 'cpu'))[:8])
    torch.testing.assert_close(*outputs, atol=2e-5, rtol=2e-5)


def test_variant_cache_verifies_inventory_and_checksums(tmp_path):
    directory = tmp_path / 'bank'
    campaign = directory / 'campaign_0000'
    campaign.mkdir(parents=True)
    for key, value in source().items():
        np.save(campaign / (key + '.npy'), value)
    (directory / 'manifest.json').write_text(json.dumps(dict(supervision_sha256='frozen', spec_sha256=spec.fingerprint(), campaigns=[dict(index=0)])))
    manifest = prepare(directory)
    assert manifest == verify(directory)
    assert manifest['files'][0]['edges']['boundary_observations'] > 0
    path = tmp_path / 'bank_variants' / 'campaign_0000.npz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify(directory)


def test_new_graph_requires_prepared_arrays():
    arrays = source()
    del arrays['boundary_observations']
    bank = SimpleNamespace(campaign=lambda _: arrays, targets=np.arange(12, dtype=np.float32))
    cfg = sample('M07', 'S-full+colocated', np.random.default_rng(7), 7)
    with pytest.raises(ValueError, match='prepared companion cache'):
        pack(bank, (0, 0, 4), cfg, 'cpu')


@pytest.mark.parametrize('model,design', [('M06', 'S-full+colocated'), ('M13', 'X-chain+turn-boundary'), ('M15', 'D-event+turn-boundary')])
def test_new_relation_parameters_receive_learning_signal(model, design):
    torch.set_num_threads(2)
    arrays = source()
    bank = SimpleNamespace(campaign=lambda _: arrays, targets=np.arange(12, dtype=np.float32))
    cfg = sample(model, design, np.random.default_rng(7), 7)
    cfg.update(hidden=16, layers=2, dropout=0.0, turn_context='none')
    net = create(cfg, 1)
    net(pack(bank, (0, 0, 12), cfg, 'cpu')).square().mean().backward()
    if model == 'M06':
        gradients = [layer.weights.grad[38] for layer in net.spatial]
    else:
        layers = net.temporal if model == 'M13' else net.layers
        gradients = [layer.relation.weight.grad[list(spec.BOUNDARY_CODES)] for layer in layers]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
