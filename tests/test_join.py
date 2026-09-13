import numpy as np
import torch

from graphspec import join
from graphspec.truth import verify
from tuning.data import Bank, pack
from tuning.models import build
from tuning.turns import blocks as turn_blocks

from .graphs import random_graph
from .support import config, configured


def spec_for(registry, values):
    return configured(registry, dict({'graphs/join': 'single', 'graphs/connect_fraction': 1.0}, **values))


def test_join_unifies_shared_keys_and_offsets_steps():
    registry, first = random_graph(3, 'query')
    _, second = random_graph(4, 'query')
    first = first.replace(**first.arrays)
    first.keys[:] = ['shared_%d' % i if i < 4 else 'a_%d' % i for i in range(len(first.keys))]
    second.keys[:] = ['shared_%d' % i if i < 4 else 'b_%d' % i for i in range(len(second.keys))]
    everything = spec_for(registry, {})
    merged = join.joined([first, second], registry, everything)
    verify(merged, registry)
    assert merged.steps == first.steps + second.steps and len(merged.kinds) == len(first.kinds) + len(second.kinds) - 4
    assert merged.keys[:len(first.keys)] == first.keys and np.all(np.diff(merged.period) >= 0)
    assert merged.period[first.steps] > merged.period[first.steps - 1]
    assert np.array_equal(merged.label_step[len(first.label_step):], second.label_step + first.steps)
    assert np.array_equal(merged.label_row, np.r_[first.label_row, second.label_row])
    assert len(merged.fact_step) == len(first.fact_step) + len(second.fact_step) and len(merged.query_key) == merged.steps
    separate = join.joined([first, second], registry, spec_for(registry, {'graphs/connect_fraction': 0.01}))
    assert len(separate.kinds) == len(first.kinds) + len(second.kinds)
    partial = join.joined([first, second], registry, spec_for(registry, {'graphs/connect/' + registry.nodes[0]: False}))
    assert len(first.kinds) + len(second.kinds) - 4 < len(partial.kinds) <= len(first.kinds) + len(second.kinds)


def test_a_joined_variant_packs_and_fits(book):
    bank = Bank(book)
    cfg = config(bank.registry, {'view': 'snapshot', 'hidden': 16}, {'time/history': 3, 'time/context': 1, 'graphs/join': 'single'})
    assert bank.graphs(cfg) == [0] and bank.joining(cfg)
    blocks = bank.blocks(0, 3, cfg)
    assert all(block[0] == 0 for block in blocks) and len(blocks) == len(bank.blocks(0, 3, config(bank.registry, {'view': 'snapshot'}, {'time/history': 3})))
    arrays = bank.arrays(0, cfg)
    assert bank.sizes(cfg) == {0: len(arrays['entity_kind'])} and len(arrays['period']) == sum(graph['steps'] for graph in bank.manifest['graphs'])
    block = turn_blocks(bank, [blocks[-1]], cfg)[0]
    b = pack(bank, block, cfg, 'cuda')
    net = build(cfg, bank.registry, len(bank.categories)).to('cuda')
    output = net(b)
    assert output.shape == b['target'].shape and torch.isfinite(output).all()
