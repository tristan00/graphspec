import torch

from tuning.layers import grouping, relational_messages


def test_relational_messages_without_edges():
    h, weights = torch.randn(5, 4), torch.randn(3, 4, 4)
    empty = torch.zeros(0, dtype=torch.long)
    assert torch.equal(relational_messages(h, grouping(empty, empty, empty), weights), torch.zeros(5, 4))


def test_relational_messages_match_edge_loop():
    torch.manual_seed(0)
    h, weights = torch.randn(6, 4), torch.randn(3, 4, 4)
    source, destination, relation = torch.tensor([0, 1, 2, 3, 4, 5, 0]), torch.tensor([1, 1, 3, 0, 5, 5, 2]), torch.tensor([2, 0, 1, 1, 0, 2, 1])
    expected = torch.zeros(6, 4)
    for s, d, r in zip(source.tolist(), destination.tolist(), relation.tolist()):
        expected[d] += h[s] @ weights[r]
    torch.testing.assert_close(relational_messages(h, grouping(source, destination, relation), weights), expected)
