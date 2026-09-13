import pytest
import torch

from graphspec import spec as specs
from graphspec.synthetic import generate
from tuning import model_spec
from tuning.build import build as build_bank
from tuning.batching import pack_batch
from tuning.data import Bank
from tuning.models import build
from tuning.turns import blocks as turn_blocks

from .support import config


CONFIGURATIONS = (
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'graph_sequence', 'snapshot/sequence': 'gru'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'graph_sequence', 'snapshot/sequence': 'lstm'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'graph_sequence', 'snapshot/sequence': 'attention'},
    {'view': 'snapshot', 'snapshot/spatial': 'normalised', 'snapshot/temporal': 'entity_gru'},
    {'view': 'snapshot', 'snapshot/spatial': 'mean', 'snapshot/temporal': 'entity_gru'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'entity_gru'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'entity_tcn', 'snapshot/temporal_layers': 3},
    {'view': 'snapshot', 'snapshot/spatial': 'none', 'snapshot/temporal': 'graph_cell', 'snapshot/diffusion_hops': 0},
    {'view': 'snapshot', 'snapshot/spatial': 'none', 'snapshot/temporal': 'graph_cell', 'snapshot/diffusion_hops': 1},
    {'view': 'snapshot', 'snapshot/spatial': 'none', 'snapshot/temporal': 'evolving'},
    {'view': 'snapshot', 'snapshot/spatial': 'attention', 'snapshot/temporal': 'entity_attention'},
    {'view': 'mixed', 'mixed/aggregation': 'mean', 'mixed/update': 'gru'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'coupling_edges', 'snapshot/message': 'gated'},
    {'view': 'snapshot', 'snapshot/spatial': 'relational', 'snapshot/temporal': 'coupling_edges', 'snapshot/message': 'attention'},
    {'view': 'event_graph', 'event_graph/message': 'gated'},
    {'view': 'event_graph', 'event_graph/message': 'attention'},
    {'view': 'tree', 'tree/depth': 2},
    {'view': 'memory', 'memory/update': 'mailbox_gru', 'memory/embedding': 'attention', 'memory/pair_terms': False},
    {'view': 'memory', 'memory/update': 'mutual_rnn'},
    {'view': 'memory', 'memory/update': 'intensity', 'memory/embedding': 'attention'},
    {'view': 'sequence', 'sequence/operator': 'mixer', 'sequence/structure': 'none'},
    {'view': 'sequence', 'sequence/operator': 'transformer', 'sequence/structure': 'occurrence', 'sequence/patch': 2},
    {'view': 'sequence', 'sequence/operator': 'transformer', 'sequence/structure': 'overlap', 'sequence/patch': 1},
    {'view': 'projection', 'projection/hops': 2, 'projection/width': 16},
    {'view': 'walks', 'walks/count': 4, 'walks/hops': 3},
    {'view': 'memory', 'memory/update': 'mailbox_gru', 'memory/embedding': 'attention', 'memory/pair_terms': True},
    {'view': 'sequence', 'sequence/operator': 'ssm', 'sequence/structure': 'none', 'sequence/state': 4},
    {'view': 'affinity', 'affinity/buffer': 4},
    {'view': 'memory', 'memory/update': 'antisymmetric', 'memory/embedding': 'state', 'memory/integration_steps': 2},
    {'view': 'signal', 'signal/topk': 3, 'signal/hops': 1},
    {'view': 'memory', 'memory/update': 'mailbox_gru', 'memory/embedding': 'state', 'memory/pair_terms': True},
    {'view': 'mixed', 'mixed/aggregation': 'attention', 'mixed/update': 'residual'},
    {'view': 'sequence', 'sequence/operator': 'mixer', 'sequence/structure': 'overlap'},
)
CONTEXT = ({'view': 'snapshot', 'snapshot/temporal': 'graph_sequence'}, {'view': 'snapshot', 'snapshot/temporal': 'entity_gru'},
           {'view': 'snapshot', 'snapshot/temporal': 'coupling_edges'}, {'view': 'mixed'}, {'view': 'event_graph'}, {'view': 'tree'},
           {'view': 'event_graph', 'event_graph/message': 'attention'}, {'view': 'memory', 'memory/update': 'mailbox_gru'},
           {'view': 'memory', 'memory/update': 'mutual_rnn'}, {'view': 'memory', 'memory/update': 'intensity'},
           {'view': 'memory', 'memory/update': 'antisymmetric', 'memory/embedding': 'state'}, {'view': 'projection'},
           {'view': 'sequence', 'sequence/operator': 'mixer'}, {'view': 'sequence', 'sequence/operator': 'transformer'},
           {'view': 'sequence', 'sequence/operator': 'ssm'}, {'view': 'walks'}, {'view': 'affinity'}, {'view': 'signal'})


def offered(registry, values):
    parameters = {parameter.name: parameter for parameter in model_spec.space(registry)}
    return values['view'] in parameters['view'].options and all(value in parameters[name].options for name, value in values.items()
                                                                if parameters[name].kind in ('choice', 'bool'))


def exercise(bank, values, history, context, device):
    cfg = config(bank.registry, dict(values, hidden=16), {'time/history': history, 'time/context': context})
    b = pack_batch(bank, turn_blocks(bank, [bank.blocks(0, history, cfg)[-1]], cfg), cfg, device)
    net = build(cfg, bank.registry, len(bank.categories)).to(device)
    values = net(b)
    target = (b['target'] - bank.split['reward_mean']) / bank.split['reward_sd']
    assert values.shape == target.shape and torch.isfinite(values).all()
    (values - target).square().mean().backward()
    dead = [name for name, parameter in net.named_parameters() if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
    assert not dead, dead
    net.eval()
    with torch.no_grad():
        clone = build(cfg, bank.registry, len(bank.categories)).to(device).eval()
        clone.load_state_dict(net.state_dict(), strict=True)
        torch.testing.assert_close(net(b), clone(b))


@pytest.fixture(scope='module')
def banks(book, tmp_path_factory):
    root = tmp_path_factory.mktemp('models')
    task = generate(root / 'query', 'query', 4, 'query', 5, entities=40, steps=10, features=8, signals=2, degree=2.0, max_degree=4, alive=0.8, lifetime=3.0, period=3)
    build_bank(task, root / 'query' / 'bank', 1)
    return Bank(book), Bank(root / 'query' / 'bank')


def test_the_space_is_conditional_and_describes_itself(banks):
    registry = banks[1].registry
    parameters = model_spec.space(registry)
    for values in CONFIGURATIONS:
        spec = model_spec.configured(registry, values)
        assert specs.canonical(parameters, spec) == spec and all(spec[name] == value for name, value in values.items())
        assert model_spec.describe(spec).startswith(values['view'])
    assert model_spec.blocks(model_spec.configured(registry, {'view': 'memory', 'memory/embedding': 'state', 'memory/pair_terms': True})) == {'events', 'history'}
    assert model_spec.blocks(model_spec.configured(registry, {'view': 'snapshot', 'snapshot/temporal': 'coupling_edges'})) == {'snapshot', 'lifted'}
    assert model_spec.groups(model_spec.configured(registry, {'view': 'event_graph'})) == ('events',)
    assert 'dropout' not in model_spec.configured(registry, {'view': 'snapshot', 'snapshot/spatial': 'none'})
    assert 'memory/neighbors' not in model_spec.configured(registry, {'view': 'memory', 'memory/embedding': 'state'})
    assert 'memory/neighbors' in model_spec.configured(registry, {'view': 'memory', 'memory/embedding': 'state', 'memory/pair_terms': True})
    with pytest.raises(ValueError):
        model_spec.configured(registry, {'view': 'memory', 'memory/neighbours': 8})
    assert 'view' not in {parameter.name for parameter in model_spec.TRAINING}
    assert 'affinity' not in model_spec.views(banks[0].registry) and 'signal' not in model_spec.views(banks[0].registry)
    assert len(model_spec.samples(registry)) > len(CONFIGURATIONS)


def test_every_configuration_fits_a_window_and_reloads(banks):
    node, query = banks
    for values in CONFIGURATIONS:
        bank = query if values['view'] in ('affinity', 'signal') or values.get('memory/update') == 'mutual_rnn' else node
        assert offered(bank.registry, values), values
        exercise(bank, values, 3, 0, 'cuda')
    for values in CONTEXT:
        bank = query if values['view'] in ('affinity', 'signal') or values.get('memory/update') == 'mutual_rnn' else node
        exercise(bank, values, 2, 1, 'cuda')


def test_bounds_narrow_every_space(banks):
    import numpy as np
    from graphspec import spec as specs
    from tuning.selector import check_bounds, random_suggest

    bank = banks[1]
    bounds = {'view': ['tree', 'walks'], 'hidden': [16, 24], 'epochs': {'low': 1, 'high': 2}, 'time/history': {'low': 2, 'high': 4},
              'graphs/join': 'separate', 'derive/rewiring': 'none', 'lr': 0.002}
    check_bounds(bank.registry, bounds)
    for seed in range(12):
        config, selection = random_suggest(bank.registry, seed, 7, bounds)
        assert config['model']['view'] in ('tree', 'walks') and config['model']['hidden'] in (16, 24) and 1 <= config['epochs'] <= 2
        assert 2 <= config['graph']['time/history'] <= 4 and config['graph']['graphs/join'] == 'separate' and config['lr'] == 0.002
        assert config['graph'].get('derive/rewiring', 'none') == 'none' and selection['phase'] == 'random'
    with pytest.raises(ValueError):
        check_bounds(bank.registry, {'no/such': 1})
    with pytest.raises(ValueError):
        check_bounds(bank.registry, {'time/history': {'low': 0, 'high': 4}})
    with pytest.raises(ValueError):
        specs.narrow(specs.space(bank.registry, ()), {'time/history': {'low': 5, 'high': 4}})
