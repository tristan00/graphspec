import torch
from torch import nn
from torch.nn import functional as F

from graphspec.infra.timing import timed

from .layers import Asked, Head, References, normalization
from .periods import EndContext, enabled


class SignalNet(nn.Module):
    @timed("signal.signalnet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        hidden, channels = spec["hidden"], registry.extras["signals"]
        self.channels = channels
        self.has_periods = enabled(cfg)
        if self.has_periods:
            self.periods = EndContext(hidden)
        self.topk, self.mix_hops = spec["signal/topk"], spec["signal/hops"]
        self.left, self.right = (
            nn.Parameter(torch.randn(channels, hidden)),
            nn.Parameter(torch.randn(channels, hidden)),
        )
        self.left_map, self.right_map = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.input = nn.Linear(2, hidden)
        self.convs = nn.ModuleList(
            nn.Conv2d(hidden, hidden * 2, (1, 3), dilation=(1, 2**i)) for i in range(spec["signal/layers"])
        )
        self.mix = nn.ModuleList(nn.Linear(hidden * (1 + 2 * self.mix_hops), hidden) for _ in self.convs)
        self.norms = nn.ModuleList(normalization(hidden) for _ in self.convs)
        self.task = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.member = nn.Linear(registry.features, hidden)
        self.references = References(registry, hidden, hidden)
        self.head = Head(hidden * 3, spec, registry)

    def adjacency(self):
        left, right = self.left_map(self.left).tanh(), self.right_map(self.right).tanh()
        scores = (3 * (left @ right.T - right @ left.T)).tanh().relu()
        indices = scores.topk(min(self.topk, len(scores)), dim=1).indices
        scores = scores * torch.zeros_like(scores).scatter_(1, indices, 1)
        scores = scores + torch.eye(len(scores), device=scores.device)
        return scores / scores.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, b):
        raw = b["signal"].clone()
        raw[:, :, 0] = raw[:, :, 0].sign() * raw[:, :, 0].abs().log1p()
        h = self.input(raw).reshape(b["batch_size"], b["time_steps"], self.channels, -1)
        adjacency = self.adjacency()
        for i, (conv, mix, norm) in enumerate(zip(self.convs, self.mix, self.norms)):
            channels = h.permute(0, 3, 2, 1)
            gate, value = conv(F.pad(channels, (2 * 2**i, 0, 0, 0))).chunk(2, 1)
            temporal = (gate.sigmoid() * value.tanh()).permute(0, 3, 2, 1)
            states = [temporal]
            for support in (adjacency, adjacency.T):
                current = temporal
                for _ in range(self.mix_hops):
                    current = 0.05 * temporal + 0.95 * torch.einsum("ij,btjh->btih", support, current)
                    states.append(current)
            h = norm(h + mix(torch.cat(states, -1)))
        frames = h.mean(2).reshape(b["frames"], -1)
        values = frames[b["row_frames"]]
        if self.has_periods:
            values = self.periods(values, frames[b["end_steps"].clamp_min(0)], b["end_gaps"], b["end_mask"])
        features = torch.cat(
            (
                values,
                self.references(b, self.member(b["member_x"]), values[b["whole_row"]]),
                self.asked(self.task(b["task"]), b),
            ),
            -1,
        )
        return self.head(features[b["selected"]], b)
