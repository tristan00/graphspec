import numpy as np

from graphspec.transform import SEED
from tuning import gather

from . import reference_pack as reference
from .graphs import random_graph
from .support import built


def arrays_for(seed, level, order):
    registry, graph = random_graph(seed, level)
    arrays = built(graph, registry, {'neighbors/order': order, 'neighbors/decay': 0.3}, ('neighbors',))
    return registry, arrays, dict(spec={'neighbors/order': order, 'neighbors/decay': 0.3})


def test_latest_matches_reference():
    registry, arrays, _ = arrays_for(3, 'query', 'recent')
    size = len(arrays['entity_kind'])
    rng = np.random.default_rng(0)
    entities = rng.integers(0, size, 200)
    cutoffs = rng.integers(0, 2 * len(arrays['period']) + 2, 200)
    expected = np.concatenate([reference.latest(arrays, [e], c) for e, c in zip(entities, cutoffs)])
    found = gather.latest(np.asarray(arrays['observation_ids']), np.asarray(arrays['observation_offsets']), np.asarray(arrays['step']), entities, cutoffs)
    assert np.array_equal(expected, found)


def test_neighbors_match_reference_for_every_order():
    for order, mode in (('recent', 0), ('uniform', 1), ('decay', 2)):
        registry, arrays, settings = arrays_for(5, 'query', order)
        events = np.asarray(arrays['events'])
        ids, offsets = np.asarray(arrays['neighbor_ids']), np.asarray(arrays['neighbor_offsets'])
        size = len(arrays['entity_kind'])
        rng = np.random.default_rng(1)
        entities = rng.integers(0, size, 300)
        cutoffs = rng.integers(0, 2 * len(arrays['period']) + 2, 300)
        for count in (1, 2, 4, 8):
            table, sizes = gather.neighbor_table(ids, offsets, np.ascontiguousarray(events[:, 2]), entities, cutoffs, 0, count, mode, 0.3, SEED, reference.NEIGHBOURS)
            for row, (entity, cutoff) in enumerate(zip(entities, cutoffs)):
                expected = reference.neighbors(arrays, int(entity), int(cutoff), 0, count, settings['spec'], SEED)
                assert sizes[row] == len(expected) and np.array_equal(table[row, :sizes[row]], expected), (order, count, row)
                assert (table[row, sizes[row]:] == -1).all()


def test_tree_levels_match_reference():
    registry, arrays, settings = arrays_for(9, 'node', 'decay')
    events = np.asarray(arrays['events'])
    ids, offsets = np.asarray(arrays['neighbor_ids']), np.asarray(arrays['neighbor_offsets'])
    rng = np.random.default_rng(2)
    roots = rng.integers(0, len(arrays['entity_kind']), 40)
    cutoffs = rng.integers(1, 2 * len(arrays['period']) + 1, 40)
    expected = reference.tree(arrays, roots, cutoffs, 2, 3, 2, registry, settings['spec'], SEED)
    entities, moments = roots, cutoffs
    for level in range(2):
        table, sizes = gather.neighbor_table(ids, offsets, np.ascontiguousarray(events[:, 2]), entities, moments, 2, 3, 2, 0.3, SEED, reference.NEIGHBOURS)
        children, times, mask, gaps, relation = gather.tree_level(events, table, sizes, entities, moments, 3, registry.marker('non_relation'))
        assert np.array_equal(mask, expected['mask']) and np.array_equal(gaps, expected['gap']) and np.array_equal(relation, expected['relation'])
        assert np.array_equal(np.asarray(arrays['entity_kind'][children], np.int64), expected['child']['kind'])
        expected, entities, moments = expected['child'], children, times
    observations = gather.latest(np.asarray(arrays['observation_ids']), np.asarray(arrays['observation_offsets']), np.asarray(arrays['step']), entities, moments)
    raw = np.zeros((len(entities), registry.features), np.float32)
    raw[observations >= 0] = arrays['x'][observations[observations >= 0]]
    assert np.array_equal(raw, expected['x']) and np.array_equal(np.asarray(arrays['entity_kind'][entities], np.int64), expected['kind'])
