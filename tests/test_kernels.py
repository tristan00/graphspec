import numpy as np

from .graphs import links, mesh, relation
from .support import derived, rows


def test_kernels():
    from graphspec import kernels

    rng = np.random.default_rng(41)
    columns = [rng.integers(0, 4, 5000), rng.integers(0, 4, 5000), rng.integers(0, 64, 5000)]
    ids, count = kernels.keyed(*columns)
    _, inverse = np.unique(np.stack(columns, 1), axis=0, return_inverse=True)
    matched = np.unique(np.stack((ids, inverse.reshape(-1)), 1), axis=0)
    assert count == int(inverse.max()) + 1 == len(matched) == len(np.unique(matched[:, 0])) == len(np.unique(matched[:, 1]))
    ids, count = kernels.keyed(np.zeros(4096, np.int64), np.zeros(4096, np.int64), np.arange(4096))
    assert count == 4096 and np.array_equal(ids, np.arange(4096))
    position, begun = kernels.runs(np.zeros(6, np.int64), np.array([0, 1, 1, 3, 4, 6]), 1)
    assert position.tolist() == [0, 1, 1, 0, 1, 0] and begun.tolist() == [0, 0, 0, 3, 3, 6]
    slots, low, high = np.zeros(16384, np.int64), np.zeros(4100, np.int64), np.zeros(4100, np.int64)
    count, fresh = kernels.novel(slots, low, high, 0, np.zeros(4096, np.int64), np.arange(4096))
    assert count == 4096 and fresh.all()
    count, again = kernels.novel(slots, low, high, count, np.array([0, 1, 0]), np.array([7, 5, 9]))
    assert count == 4097 and again.tolist() == [False, True, False]
    registry, graph = links([(0, 3, 'link', 0, True, ()), (1, 3, 'link', 0, True, ()), (0, 4, 'link', 0, True, ()), (1, 4, 'link', 0, True, ())], 5, 1)
    projection = 'project:link:sources:thing:thing'
    assert rows(derived(graph, registry, {'derive/' + projection: True}), registry, projection) == [(0, 1, 0)]
    registry, graph = mesh(('anchor', 'leaf'), [relation('tie', ('anchor',), ('leaf',))], [0, 0, 1, 1], 1,
                           [(0, 2, 'tie', 0, True, ()), (0, 3, 'tie', 0, True, ()), (1, 2, 'tie', 0, True, ()), (1, 3, 'tie', 0, True, ())])
    shared = derived(graph, registry, {'derive/complete/leaf': True, 'derive/complete_within/anchor': True})
    assert rows(shared, registry, 'complete:leaf:leaf') == [(2, 3, 0)]
    registry, graph = links([(0, 1, 'link', 0, True, ()), (1, 2, 'link', 0, True, ()), (0, 2, 'link', 0, True, ()), (2, 3, 'link', 0, True, ()),
                             (1, 3, 'link', 0, True, ()), (3, 4, 'link', 0, True, ())], 5, 1)
    reached = derived(graph, registry, {'derive/hops': 3})
    assert rows(reached, registry, 'hop:2') == [(0, 3, 0), (1, 4, 0), (2, 4, 0)] and rows(reached, registry, 'hop:3') == [(0, 4, 0)]

