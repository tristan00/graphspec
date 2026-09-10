import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.graph_temporal.data import pack
from research.graph_temporal.batching import pack_batch, batches
from research.graph_temporal.registry import DESIGNS, create, sample
from tests.test_graph_research_cache import make, records
from research.graph_temporal.turns import index as turn_index, walks as turn_walks
from research.graph_temporal.graph_variants import build as graph_variants


@pytest.fixture(scope='module')
def bank():
    torch.set_num_threads(2)
    _, arrays = make(records(8))
    arrays['row_index'] = np.arange(8)
    arrays['decision'] = np.arange(10, 18)
    arrays['decision_turn'] = np.arange(8) // 2 + 1
    arrays.update(turn_index(arrays))
    arrays['turn_walks'] = turn_walks(arrays, arrays)
    arrays.update(graph_variants(arrays))
    return SimpleNamespace(campaign=lambda index: arrays, targets=np.arange(8, dtype=np.float32))


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_every_model_forward_backward_reload(bank, model, design, turn_context):
    cfg = sample(model, design, np.random.default_rng(71), 77)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4)
    cfg.update(turn_context=turn_context, turn_lookback=3)
    b = pack(bank, (0, 0, 8), cfg, 'cpu')
    net = create(cfg, 1)
    predictions = net(b)
    assert predictions.shape == (8,) and torch.isfinite(predictions).all()
    (predictions - b['target']).square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    clone = create(cfg, 1)
    clone.load_state_dict(net.state_dict(), strict=True)
    clone.eval(); net.eval()
    torch.testing.assert_close(clone(b), net(b))


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_batch_preserves_independent_predictions_and_gradients(bank, model, design, turn_context):
    cfg = sample(model, design, np.random.default_rng(71), 77)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4)
    cfg.update(turn_context=turn_context, turn_lookback=2)
    net = create(cfg, 1).eval()
    blocks = [(0, 0, 4), (1, 4, 8)]
    separate = torch.cat([net(pack(bank, block, cfg, 'cpu')) for block in blocks])
    separate.square().mean().backward()
    gradients = {key: p.grad.clone() for key, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    combined = net(pack_batch(bank, blocks, cfg, 'cpu'))
    torch.testing.assert_close(combined, separate, atol=2e-5, rtol=2e-5)
    combined.square().mean().backward()
    for key, p in net.named_parameters():
        if key in gradients:
            torch.testing.assert_close(p.grad, gradients[key], atol=5e-5, rtol=5e-4)


def test_batches_keep_every_block_including_short_tails():
    blocks = [(i, j, min(j + 4, i + 1)) for i in range(9) for j in range(0, i + 1, 4)]
    grouped = batches(blocks, 3, 77)
    assert sorted(block for batch in grouped for block in batch) == sorted(blocks)
    assert all(len({block[2] - block[1] for block in batch}) == 1 for batch in grouped)


@pytest.mark.parametrize('model', ['M21', 'M22', 'M23', 'M27'])
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_history_target_selection_preserves_context_predictions_and_gradients(bank, model, turn_context):
    cfg = sample(model, 'H-neighbor', np.random.default_rng(71), 77)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4, turn_context=turn_context, turn_lookback=2)
    full = pack_batch(bank, [(0, 0, 8), (1, 0, 8)], cfg, 'cpu')
    targets = pack_batch(bank, [(0, 0, 8, 6), (1, 0, 8, 5)], cfg, 'cpu')
    selected = targets['selected']
    np.testing.assert_array_equal(targets['rows'], full['rows'][selected.numpy()])
    torch.testing.assert_close(targets['target'], full['target'][selected])
    keys = ('context', 'history_x', 'history_mask', 'history_gap', 'occurrence', 'structure', 'pair_gap', 'pair_mask')
    if turn_context == 'end':
        keys += ('history_end_time', 'history_turn_kind')
    for key in keys:
        torch.testing.assert_close(targets[key][selected], full[key][selected])
    net = create(cfg, 1).eval()
    expected = net(full)[selected]
    expected.square().mean().backward()
    gradients = {key: p.grad.clone() for key, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    actual = net(targets)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.square().mean().backward()
    for key, p in net.named_parameters():
        if key in gradients:
            torch.testing.assert_close(p.grad, gradients[key], atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize('model', ['M21', 'M22', 'M23', 'M27'])
def test_history_context_without_targets_returns_empty_predictions(bank, model):
    cfg = sample(model, 'H-neighbor', np.random.default_rng(71), 77)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4, turn_context='end', turn_lookback=2)
    batch = pack(bank, (0, 0, 8, 8), cfg, 'cpu')
    assert create(cfg, 1)(batch).shape == (0,)


@pytest.mark.parametrize('model', list(DESIGNS))
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_future_observation_and_action_do_not_change_earlier_predictions(model, turn_context):
    original = records(8)
    changed = copy.deepcopy(original)
    changed[-1][1]['campaign']['treasury'] = 999999
    changed[-1][1]['entities'][0]['state']['hp'] = 999
    changed[-1][0]['action']['key'] = 'cqi:777'
    cfg = sample(model, DESIGNS[model][0], np.random.default_rng(7), 9)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4)
    cfg.update(turn_context=turn_context, turn_lookback=3)
    net = create(cfg, 1).eval()
    outputs = []
    for source in (original, changed):
        _, arrays = make(source)
        arrays['row_index'] = np.arange(8)
        arrays['decision_turn'] = np.arange(8) // 2 + 1
        arrays.update(turn_index(arrays))
        arrays['turn_walks'] = turn_walks(arrays, arrays)
        bank = SimpleNamespace(campaign=lambda index: arrays, targets=np.arange(8, dtype=np.float32))
        with torch.no_grad():
            outputs.append(net(pack(bank, (0, 0, 8), cfg, 'cpu')))
    torch.testing.assert_close(outputs[0][:-1], outputs[1][:-1], atol=2e-5, rtol=2e-5)


def test_antisymmetric_damping_and_learned_directed_graph():
    cfg = sample('M29', 'E-entity', np.random.default_rng(7), 9)
    net = create(cfg, 1)
    matrix = net.antisymmetric()
    torch.testing.assert_close(matrix + matrix.T, -2 * cfg['damping'] * torch.eye(cfg['hidden']))
    cfg = sample('M30', 'V-signals', np.random.default_rng(7), 9)
    net = create(cfg, 1)
    adjacency = net.adjacency()
    assert not torch.allclose(adjacency, adjacency.T)
    torch.testing.assert_close(adjacency.sum(1), torch.ones(20))


@pytest.mark.parametrize('model,parameter', [
    ('M08', 'cells.0.gates.weight'), ('M09', 'cells.0.candidate.weight'), ('M10', 'evolving.0.initial'),
    ('M11', 'temporal.qkv.weight'), ('M18', 'update.weight_hh'), ('M19', 'left_cell.weight_hh'),
    ('M20', 'intensity.0.weight'), ('M22', 'structure.0.weight'), ('M23', 'structure.0.weight'),
    ('M25', 'anonymous.0.weight'), ('M27', 'node_layers.0.log_a'), ('M27', 'time_layers.0.log_a'),
    ('M28', 'gs.weight'), ('M29', 'weight'), ('M30', 'left')])
def test_defining_components_receive_learning_signal(bank, model, parameter):
    cfg = sample(model, DESIGNS[model][0], np.random.default_rng(7), 9)
    cfg.update(hidden=16, layers=2, dropout=0.0, neighbors=4)
    net = create(cfg, 1)
    b = pack(bank, (0, 0, 8), cfg, 'cpu')
    (net(b) - b['target']).square().mean().backward()
    gradient = dict(net.named_parameters())[parameter].grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.fixture(scope='module')
def medium_bank():
    from research.graph_temporal.walk_cache import extend

    torch.set_num_threads(2)
    _, arrays = make(records(32))
    arrays['walks'] = extend(arrays)
    arrays['row_index'] = np.arange(32)
    arrays['decision_turn'] = np.arange(32) // 2 + 1
    arrays.update(turn_index(arrays))
    arrays['turn_walks'] = turn_walks(arrays, arrays)
    arrays.update(graph_variants(arrays))
    return SimpleNamespace(campaign=lambda index: arrays, targets=np.arange(32, dtype=np.float32))


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
def test_medium_model_limits_with_longer_history(medium_bank, model, design):
    from research.graph_temporal.selector import space

    cfg = sample(model, design, np.random.default_rng(71), 77)
    for key, distribution in space(model).items():
        cfg[key] = distribution.choices[-1] if hasattr(distribution, 'choices') else distribution.high
    net = create(cfg, 1)
    batch = pack_batch(medium_bank, [(0, 0, 32), (1, 0, 32)], cfg, 'cpu')
    predicted = net(batch)
    assert predicted.shape == (64,) and torch.isfinite(predicted).all()
    (predicted - batch['target']).square().mean().backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize('model,design', [(model, design) for model, designs in DESIGNS.items() for design in designs])
def test_single_decision_history_and_nondivisible_patches(medium_bank, model, design):
    cfg = sample(model, design, np.random.default_rng(71), 77)
    cfg.update(hidden=16, layers=1, history=1, dropout=0.37, clip=0.83, epochs=12, neighbors=10, patch=16)
    net = create(cfg, 1)
    batch = pack_batch(medium_bank, [(0, 0, 1), (1, 5, 6)], cfg, 'cpu')
    output = net(batch)
    assert output.shape == (2,) and torch.isfinite(output).all()
    (output - batch['target']).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
