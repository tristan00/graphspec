from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.graph_temporal.batching import batches, pack_batch
from research.graph_temporal.data import pack
from research.graph_temporal.registry import DESIGNS, create, sample
from research.graph_temporal.turns import blocks, index, prepare, verify, walks
from tests.test_graph_research_cache import make, records
from research.graph_temporal.graph_variants import build as graph_variants


def source(count=8):
    _, arrays = make(records(count))
    arrays['decision_turn'] = np.arange(count) // 2 + 1
    arrays['row_index'] = np.arange(count)
    arrays.update(index(arrays))
    arrays['turn_walks'] = walks(arrays, arrays)
    arrays.update(graph_variants(arrays))
    return arrays


def test_endpoints_are_past_completed_turns_with_matching_entity_identity():
    arrays = source()
    bank = SimpleNamespace(campaign=lambda _: arrays, targets=np.arange(8, dtype=np.float32))
    cfg = sample('M13', 'X-chain', np.random.default_rng(7), 7)
    cfg.update(turn_context='end', turn_lookback=3, history=1)
    original = [(0, i, i + 1) for i in range(8)]
    expanded = blocks(bank, original, cfg)
    rows = []
    for block in expanded:
        batch = pack(bank, block, cfg, 'cpu')
        rows.extend(batch['rows'].tolist())
        query = block[3]
        anchors = batch['end_steps'][-1].numpy()
        anchors = anchors[anchors >= 0] + block[1]
        assert np.all(arrays['decision_turn'][anchors] < arrays['decision_turn'][query])
        assert set(anchors) <= set(arrays['turn_ends'])
        edge = batch['temporal_edges'][:, batch['temporal_relation'] == 42]
        torch.testing.assert_close(batch['entity'][edge[0]], batch['entity'][edge[1]])
        assert (batch['step'][edge[0]] < batch['step'][edge[1]]).all()
    assert rows == list(range(8))
    assert expanded[6][1] == 0
    grouped = batches(expanded, 99, frame_budget=12)
    assert sorted(block for group in grouped for block in group) == sorted(expanded)
    assert all(sum(b[2] - b[1] for b in group) <= 12 for group in grouped)


def test_turn_walks_follow_current_query_entities_at_prior_turn_cutoffs():
    arrays = source(12)
    for query in range(12):
        for side, root in enumerate((arrays['actor'][query], arrays['subject'][query])):
            for lag in range(3):
                turn = arrays['turn_membership'][query] - lag - 1
                paths = arrays['turn_walks'][query, side, lag]
                if turn < 0:
                    assert (paths == -1).all()
                    continue
                for path in paths:
                    node, cutoff = root, 2 * arrays['turn_ends'][turn] + 2
                    for event_id in path:
                        if event_id < 0:
                            break
                        event = arrays['events'][event_id]
                        assert node in event[:2]
                        assert 2 * arrays['turn_starts'][turn] <= event[2] < cutoff < query * 2 + 1
                        node, cutoff = event[1] if event[0] == node else event[0], event[2]


def test_cache_is_complete_and_tampering_prevents_launch(tmp_path):
    from research.graph_temporal.cache import write_json

    directory = tmp_path / 'bank'
    directory.mkdir()
    arrays = source()
    campaign = directory / 'campaign_0000'
    campaign.mkdir()
    for key, value in arrays.items():
        np.save(campaign / (key + '.npy'), value, allow_pickle=False)
    write_json(directory / 'manifest.json', dict(supervision_sha256='frozen', campaigns=[dict(index=0)]))
    prepare(directory)
    assert len(verify(directory)['files']) == 1
    path = tmp_path / 'bank_turns' / 'campaign_0000.npz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify(directory)


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
def test_turn_readout_learns_with_one_target_decision_and_batched_replay(model, design):
    torch.set_num_threads(2)
    arrays = source()
    bank = SimpleNamespace(campaign=lambda _: arrays, targets=np.arange(8, dtype=np.float32))
    cfg = sample(model, design, np.random.default_rng(7), 7)
    cfg.update(hidden=16, layers=2, history=1, neighbors=4, dropout=0.0, turn_context='end', turn_lookback=2)
    net = create(cfg, 1).eval()
    expanded = blocks(bank, [(0, 6, 7), (1, 6, 7)], cfg)
    batch = pack_batch(bank, expanded, cfg, 'cpu')
    actual = net(batch)
    independent = torch.cat([net(pack(bank, block, cfg, 'cpu')) for block in expanded])
    torch.testing.assert_close(actual, independent, atol=2e-5, rtol=2e-5)
    assert batch['rows'].tolist() == [6, 6]
    (actual - batch['target']).square().mean().backward()
    if model != 'M12':
        gradients = [p.grad for name, p in net.named_parameters() if 'turn' in name and p.grad is not None]
        assert gradients and any(g.abs().sum() > 0 for g in gradients)
        assert all(torch.isfinite(g).all() for g in gradients)
    else:
        stripped = deepcopy(batch)
        keep = stripped['mixed']['edge'][:, 15] == 0
        stripped['mixed']['edge'] = stripped['mixed']['edge'][keep]
        stripped['mixed']['edge_index'] = stripped['mixed']['edge_index'][:, keep]
        assert not torch.allclose(net(stripped), actual)


def test_relational_memory_optimization_preserves_outputs_and_gradients():
    from research.graph_temporal.layers import Spatial, mean

    torch.manual_seed(4)
    layer = Spatial(16, 'relational', dropout=0.0)
    h = torch.randn(41, 16, requires_grad=True)
    edges = torch.randint(41, (2, 200))
    relations = torch.randint(38, (200,))
    attrs = torch.randn(200, 4)
    actual = layer(h, edges, relations, attrs)
    values = torch.bmm(h[edges[0], None], layer.weights[relations]).squeeze(1)
    values = values + layer.attributes(attrs.sign() * attrs.abs().log1p())
    expected = layer.norm(h + torch.nn.functional.silu(layer.root(h) + mean(values, edges[1], len(h))))
    torch.testing.assert_close(actual, expected)
    params = (h, *layer.parameters())
    old = torch.autograd.grad(expected.square().sum(), params, retain_graph=True)
    new = torch.autograd.grad(actual.square().sum(), params)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
