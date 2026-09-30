from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tuning.families.layers import CausalAttention, normalization
from tuning.families.walks import WalkNet


def assert_finite_gradients(module, values):
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("heads", (1, 2, 3, 4))
def test_attention_accepts_every_hidden_width_with_finite_gradients(heads):
    torch.manual_seed(17)
    mask = torch.tensor([[True, True, False]])
    for hidden in range(1, 129):
        layer = CausalAttention(hidden, heads)
        values = torch.randn(1, 3, hidden, requires_grad=True)
        result = layer(values, mask)
        assert result.shape == values.shape
        assert torch.isfinite(result).all()
        assert not result[:, 2].count_nonzero()
        (result * torch.randn_like(result)).sum().backward()
        assert_finite_gradients(layer, values)


@pytest.mark.parametrize("hidden", (1, 3, 127))
def test_projected_attention_preserves_causality_and_input_gradient(hidden):
    torch.manual_seed(29)
    layer = CausalAttention(hidden, heads=4).eval()
    values = torch.randn(1, 3, hidden, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    original = layer(values, mask)
    changed = values.detach().clone()
    changed[:, -1] += 100
    torch.testing.assert_close(layer(changed, mask)[:, :-1], original[:, :-1])
    (original * torch.randn_like(original)).sum().backward()
    assert values.grad.abs().sum() > 0


def test_walks_accept_every_hidden_width_with_finite_gradients():
    torch.manual_seed(31)
    registry = SimpleNamespace(
        features=3,
        relation_slots=2,
        extras={"target": "number", "series": ["out"]},
        references=("unit",),
        tasks=("ask",),
        schema=SimpleNamespace(tasks=dict(references=dict(unit=dict(sort="node")))),
    )
    for hidden in range(1, 129):
        cfg = {
            "graph": {"time/context": 0},
            "model": {"hidden": hidden, "walks/hops": 3, "time/encoding": "log_cosine"},
        }
        net = WalkNet(cfg, registry)
        values = torch.randn(2, 2, 3, 3, requires_grad=True)
        batch = {
            "walk_x": values,
            "walk_identity": torch.ones(2, 2, 3, 6),
            "walk_gap": torch.arange(3).float().expand(2, 2, 3),
            "walk_relation": torch.tensor([[[1, -1, 0], [-1, -1, -1]]] * 2),
            "walk_mask": torch.tensor([[[True, True, True], [False, False, False]]] * 2),
            "member_x": torch.randn(2, 3),
            "member_row": torch.tensor([0, 0]),
            "member_name": torch.tensor([0, 0]),
            "member_place": torch.tensor([0, 0]),
            "whole_row": torch.zeros(0, dtype=torch.long),
            "whole_name": torch.zeros(0, dtype=torch.long),
            "whole_x": torch.zeros(0, 3),
            "predictions": 1,
            "task": torch.randn(1, 3),
            "row_kind": torch.tensor([0]),
            "series": torch.tensor([0]),
            "selected": torch.tensor([0]),
        }
        prediction = net(batch)
        assert prediction.shape == (1,)
        assert torch.isfinite(prediction).all()
        prediction.sum().backward()
        assert_finite_gradients(net, values)
        assert values.grad.abs().sum() > 0
        if hidden % 2:
            assert isinstance(net.walk_output, nn.Linear)
        else:
            assert isinstance(net.walk_output, nn.Identity)
            assert not any(name.startswith("walk_output.") for name in net.state_dict())


def test_normalization_preserves_scalar_input_and_existing_wider_parameters():
    values = torch.tensor([[2.0], [-3.0]], requires_grad=True)
    scalar = normalization(1)
    torch.testing.assert_close(scalar(values), values)
    scalar(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    wider = normalization(7)
    assert isinstance(wider, nn.LayerNorm)
    assert wider.weight.shape == wider.bias.shape == (7,)
    layer = CausalAttention(12, heads=4)
    assert layer.qkv.weight.shape == (36, 12)
    assert layer.output.weight.shape == (12, 12)


@pytest.mark.parametrize("width", (0, -1, True, 1.5))
def test_layer_widths_reject_nonpositive_or_noninteger_values(width):
    with pytest.raises(ValueError, match="positive integer"):
        normalization(width)
    with pytest.raises(ValueError, match="positive integer"):
        CausalAttention(width)
    with pytest.raises(ValueError, match="positive integer"):
        WalkNet({"model": {"hidden": width}}, None)


@pytest.mark.parametrize("heads", (0, -1, True, 1.5))
def test_attention_rejects_nonpositive_or_noninteger_heads(heads):
    with pytest.raises(ValueError, match="positive integer"):
        CausalAttention(3, heads=heads)
