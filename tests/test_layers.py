import torch

from tuning.layers import Groups, Spatial, segment_matmul


def edge_loop(h, source, destination, relation, weights):
    expected = torch.zeros_like(h)
    for s, d, r in zip(source.tolist(), destination.tolist(), relation.tolist()):
        expected[d] += h[s] @ weights[r]
    return expected


def test_segment_matmul_without_edges():
    h, weights = torch.randn(5, 4), torch.randn(3, 4, 4)
    empty = torch.zeros(0, dtype=torch.long)
    groups = Groups((empty, empty), empty, 5)
    update = h.new_zeros((6, 4)).index_add_(
        0, groups.relational.targets, segment_matmul(h, weights, groups.relational)
    )
    assert torch.equal(update, torch.zeros(6, 4)) and torch.equal(groups.degree, torch.ones(5))


def test_segment_matmul_matches_edge_loop():
    torch.manual_seed(0)
    h, weights = torch.randn(60, 4), torch.randn(3, 4, 4)
    source, destination = (
        torch.randint(0, 60, (200,)),
        torch.randint(0, 60, (200,)),
    )
    relation = torch.randint(0, 3, (200,))
    groups = Groups((source, destination), relation, 60)
    update = h.new_zeros((61, 4)).index_add_(
        0, groups.relational.targets, segment_matmul(h, weights, groups.relational)
    )[:60]
    torch.testing.assert_close(update, edge_loop(h, source, destination, relation, weights))
    counts = torch.zeros(60).index_add_(0, destination, torch.ones(200))
    assert (counts == 0).any() and torch.equal(groups.degree, counts.clamp_min(1))


def test_relational_layer_matches_per_name_linears():
    torch.manual_seed(1)
    layer = Spatial(4, "relational", 3, dict(first=2, second=5))
    h = torch.randn(40, 4)
    source, destination = (
        torch.randint(0, 40, (90,)),
        torch.randint(0, 40, (90,)),
    )
    relation = torch.randint(0, 3, (90,))
    first, second = torch.randn(30, 2), torch.randn(20, 5)
    edges = torch.cat((torch.arange(30), torch.arange(30, 50)))
    groups = Groups(
        (source, destination),
        relation,
        40,
        torch.cat((first.reshape(-1), second.reshape(-1))),
        edges,
        [30, 20],
        [2, 5],
    )
    update = edge_loop(h, source, destination, relation, layer.weights)
    for values, rows, slot, width in ((first, edges[:30], 0, 2), (second, edges[30:], 1, 5)):
        update.index_add_(
            0,
            destination[rows],
            (values.sign() * values.abs().log1p()) @ layer.attribute_weights[slot, :width],
        )
    count = torch.zeros(40).index_add_(0, destination, torch.ones(90)).clamp_min(1)
    expected = layer.norm(h + torch.nn.functional.silu(layer.root(h) + update / count[:, None]))
    torch.testing.assert_close(layer(h, (source, destination), groups), expected)
