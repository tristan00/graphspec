import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.graph_temporal.batching import pack_batch
from research.graph_temporal.data import Bank, CampaignArrays
from research.graph_temporal.registry import DESIGNS, sample
from research.graph_temporal.turns import index, walks
from research.graph_temporal.walk_cache import extend
from tests.test_graph_research_cache import make, records
from research.graph_temporal.graph_variants import build as graph_variants


@pytest.fixture(scope='module')
def saved_bank(tmp_path_factory):
    torch.set_num_threads(2)
    directory = tmp_path_factory.mktemp('lazy_bank')
    campaign = directory / 'campaign_0000'
    campaign.mkdir()
    _, arrays = make(records(8))
    arrays.update(row_index=np.arange(8), decision_turn=np.arange(8) // 2 + 1)
    for key, value in arrays.items():
        np.save(campaign / (key + '.npy'), value)
    turn_arrays = index(arrays)
    turn_arrays['turn_walks'] = walks(arrays, turn_arrays)
    turn_directory = directory.with_name(directory.name + '_turns')
    turn_directory.mkdir()
    np.savez_compressed(turn_directory / 'campaign_0000.npz', **turn_arrays)
    (turn_directory / 'manifest.json').write_text(json.dumps(dict(version=2, supervision_sha256='frozen')))
    expanded = extend(arrays)
    walk_directory = directory.with_name(directory.name + '_walks32')
    walk_directory.mkdir()
    np.save(walk_directory / 'campaign_0000.npy', expanded)
    (walk_directory / 'manifest.json').write_text(json.dumps(dict(supervision_sha256='frozen')))
    variants = graph_variants(arrays)
    variant_directory = directory.with_name(directory.name + '_variants')
    variant_directory.mkdir()
    np.savez_compressed(variant_directory / 'campaign_0000.npz', **variants)
    (variant_directory / 'manifest.json').write_text(json.dumps(dict(version=1, supervision_sha256='frozen')))
    (directory / 'manifest.json').write_text(json.dumps(dict(supervision_sha256='frozen')))
    rows = [dict(campaign='example', y=float(i)) for i in range(8)]
    (directory / 'supervision.json').write_text(json.dumps(dict(rows=rows,
        split=dict(train_indices=list(range(8)), validation_indices=[]))))
    expected = dict(arrays, **turn_arrays, **variants, walks=expanded)
    return directory, expected


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_lazy_loading_preserves_every_model_graph_input(saved_bank, model, design, turn_context):
    directory, expected = saved_bank
    bank = Bank(directory)
    eager = SimpleNamespace(campaign=lambda _: expected, targets=bank.targets)
    cfg = sample(model, design, np.random.default_rng(7), 9)
    cfg.update(hidden=16, layers=1, neighbors=4, turn_context=turn_context, turn_lookback=2, walks=32, walk_hops=4)
    blocks = [(0, 0, 8, 6), (0, 0, 8, 5)]
    same(pack_batch(bank, blocks, cfg, 'cpu'), pack_batch(eager, blocks, cfg, 'cpu'))


def test_campaign_inventory_does_not_open_unused_array_payloads(saved_bank, monkeypatch):
    directory, expected = saved_bank
    bank = Bank(directory)
    original = np.load
    loaded = []

    def track(path, *args, **kwargs):
        loaded.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', track)
    arrays = bank.campaign(0)
    assert isinstance(arrays, CampaignArrays)
    assert set(arrays) == set(expected) and len(arrays) == len(expected)
    assert 'walks' in arrays and 'turn_walks' in arrays
    assert loaded == ['campaign_0000.npz', 'campaign_0000.npz']
    np.testing.assert_array_equal(arrays['query'], expected['query'])
    assert loaded == ['campaign_0000.npz', 'campaign_0000.npz', 'query.npy']
    assert arrays['query'] is arrays['query']
    assert bank.campaign(0) is arrays
    assert loaded == ['campaign_0000.npz', 'campaign_0000.npz', 'query.npy']
    with pytest.raises(KeyError):
        arrays['missing']


def test_lazy_mapping_retains_all_original_and_companion_arrays(saved_bank):
    directory, expected = saved_bank
    actual = dict(Bank(directory).campaign(0))
    same(actual, expected)
    assert actual['walks'].shape[2] == 32
    assert not actual['x'].flags.writeable
