import torch

from tuning.families.layers import KeyEmbedding, Spatial, TimeEncoding, chebyshev
from tuning.families.model_spec import space


def test_chebyshev_matches_dense_polynomial():
    torch.manual_seed(2)
    n, hidden = 30, 4
    source, destination = torch.randint(0, n, (120,)), torch.randint(0, n, (120,))
    h, weights = torch.randn(n, hidden), torch.randn(4, hidden, hidden)
    adjacency = torch.zeros(n, n).index_put_((destination, source), torch.ones(120), accumulate=True)
    degree = adjacency.sum(1).clamp_min(1)
    scaled = -degree.rsqrt()[:, None] * adjacency * degree.rsqrt()[None, :]
    terms = [h, scaled @ h]
    for _ in range(2):
        terms.append(2 * scaled @ terms[-1] - terms[-2])
    expected = sum(term @ weight for term, weight in zip(terms, weights))
    torch.testing.assert_close(chebyshev(h, (source, destination), weights), expected, atol=1e-5, rtol=1e-5)


def test_relation_aware_attention_runs_and_differs_by_relation():
    torch.manual_seed(3)
    layer = Spatial(8, "relational_attention", 3, {}, slots=5)
    h = torch.randn(12, 8)
    source, destination = torch.randint(0, 12, (40,)), torch.randint(0, 12, (40,))
    first = layer(h, (source, destination), torch.zeros(40, dtype=torch.long))
    second = layer(h, (source, destination), torch.ones(40, dtype=torch.long))
    assert first.shape == (12, 8) and not torch.allclose(first, second)


def test_key_embedding_masks_negative_tokens():
    keys = KeyEmbedding(6, 16)
    token = torch.tensor([[3, -1], [19, 3]])
    out = keys(token)
    assert (
        out.shape == (2, 2, 6)
        and torch.equal(out[0, 1], torch.zeros(6))
        and torch.equal(out[0, 0], out[1, 1])
        and torch.equal(out[1, 0], out[0, 0])
    )


def test_time_encoding_modes():
    gap = torch.tensor([0.0, 1.0, 100.0])
    for mode in ("log_cosine", "cosine", "log_linear"):
        out = TimeEncoding(5, mode)(gap)
        assert out.shape == (3, 5) and torch.isfinite(out).all()
    assert torch.equal(TimeEncoding(5, "log_cosine")(gap)[0], torch.ones(5))


def test_model_space_offers_the_additions(bank):
    from tuning.bank.bank import Bank
    from tuning.families.model_spec import configured

    registry = Bank(bank).registry
    names = {parameter.name for parameter in space(registry)}
    assert {"snapshot/chebyshev_order", "time/encoding", "keys/embedding", "keys/table"} <= names
    spec = configured(registry, {"view": "snapshot", "snapshot/spatial": "chebyshev", "keys/embedding": True})
    assert (
        spec["snapshot/chebyshev_order"] == 2 and spec["keys/table"] == 16384 and "time/encoding" not in spec
    )
    assert configured(registry, {"view": "tree"})["time/encoding"] == "log_cosine"
