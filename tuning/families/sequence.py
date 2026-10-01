import torch
from torch import nn
from torch.nn import functional as F

from graphspec.infra.timing import timed

from .layers import (
    Asked,
    CausalAttention,
    Head,
    References,
    Relations,
    SelectiveSSM,
    TimeEncoding,
    keys_of,
    masked_mean,
    masked_weights,
    normalization,
    padded,
)
from .periods import enabled, lookback


class MixerLayer(nn.Module):
    @timed("neighbors.mixerlayer")
    def __init__(self, hidden, tokens):
        super().__init__()
        self.norm1, self.norm2 = normalization(hidden), normalization(hidden)
        self.token = nn.Sequential(nn.Linear(tokens, tokens * 2), nn.GELU(), nn.Linear(tokens * 2, tokens))
        self.channel = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x, mask):
        x = (x + self.token((self.norm1(x) * mask[..., None]).transpose(-1, -2)).transpose(-1, -2)) * mask[
            ..., None
        ]
        return (x + self.channel(self.norm2(x))) * mask[..., None]


class SequenceNet(nn.Module):
    @timed("neighbors.sequencenet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        self.cfg = cfg
        self.operator, self.structure_mode = spec["sequence/operator"], spec["sequence/structure"]
        hidden, depth = spec["hidden"], spec["sequence/layers"]
        if enabled(cfg):
            self.period_time = nn.Linear(3, hidden)
            self.period_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.keys = keys_of(spec, hidden)
        self.time = TimeEncoding(hidden, spec["time/encoding"])
        self.relation = Relations(registry, hidden)
        self.norm = normalization(hidden)
        if self.structure_mode != "none":
            self.structure = nn.Sequential(
                nn.Linear(2 if self.structure_mode == "occurrence" else 12, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
        if self.operator == "mixer":
            tokens = spec["sequence/neighbors"] + (lookback(cfg) if enabled(cfg) else 0)
            self.layers = nn.ModuleList(MixerLayer(hidden, tokens) for _ in range(depth))
            self.node = nn.Linear(hidden * 2, hidden)
        elif self.operator == "transformer":
            self.patch_size = spec["sequence/patch"]
            self.patch = nn.Linear(hidden * self.patch_size, hidden) if self.patch_size > 1 else nn.Identity()
            self.layers = nn.ModuleList(
                CausalAttention(hidden, spec["sequence/heads"], spec["dropout"]) for _ in range(depth)
            )
        else:
            self.node_layers = nn.ModuleList(
                SelectiveSSM(hidden, spec["sequence/state"]) for _ in range(depth)
            )
            self.time_layers = nn.ModuleList(
                SelectiveSSM(hidden, spec["sequence/state"]) for _ in range(depth)
            )
            self.opposite = nn.Linear(hidden, hidden)
            self.temporal_select = nn.Linear(hidden, hidden)
        self.asked = Asked(registry, hidden)
        self.whole = nn.Linear(registry.features, hidden)
        self.references = References(registry, hidden, hidden * 2)
        self.head = Head(hidden * 3, spec, registry, hidden)

    def mixer(self, x, raw, mask, roots, b):
        for layer in self.layers:
            x = layer(x, mask)
        temporal = masked_mean(x, mask, 1)
        spatial = masked_mean(raw, mask, 1)
        return self.node(torch.cat((temporal, spatial + roots), -1))

    def transformer(self, x, mask, roots):
        n, tokens, hidden = x.shape
        patch = self.patch_size
        padding = (-tokens) % patch
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding))
            tokens += padding
        if patch > 1:
            x = self.patch(x.reshape(n, tokens // patch, hidden * patch))
            mask = mask.reshape(n, tokens // patch, patch).any(-1)
        x = torch.cat((x, roots[:, None]), 1)
        mask = torch.cat((mask, torch.ones(n, 1, device=x.device, dtype=torch.bool)), 1)
        for layer in self.layers:
            x = layer(x, mask, causal=False)
        return x[:, -1]

    def ssm(self, x, mask, roots, b):
        n = len(x)
        x = torch.cat((x, roots[:, None]), 1)
        mask = torch.cat((mask, torch.ones(n, 1, device=x.device, dtype=torch.bool)), 1)
        for layer in self.node_layers:
            x = layer(x, mask)
        times = self.time(b["pair_gap"])
        for layer in self.time_layers:
            times = layer(times, b["pair_mask"])
        pattern = self.temporal_select(masked_mean(times, b["pair_mask"], 1))
        own = masked_mean(x, mask, 1)
        partner = b["partner"]
        other = torch.where((partner >= 0)[:, None], padded(own, partner), own)
        key = self.opposite(other) * pattern[b["member_row"]]
        weights = masked_weights((x * key[:, None]).sum(-1), mask)
        return (x * weights[..., None]).sum(1)

    def forward(self, b):
        roots = self.input(b["member_x"])
        mask = b["history_mask"]
        raw = self.input(b["history_x"]) + self.relation(b["history_relation"])
        if self.keys is not None:
            roots = roots + self.keys(b["member_token"])
            raw = raw + self.keys(b["history_token"])
        x = self.norm(raw + self.time(b["history_gap"])) * mask[..., None]
        if enabled(self.cfg):
            x = (
                x
                + self.period_time(b["history_end_time"].log1p())
                + self.period_kind(b["history_period_kind"])
            ) * mask[..., None]
        if self.structure_mode != "none":
            structure = b["occurrence"].log1p() if self.structure_mode == "occurrence" else b["structure"]
            x = (x + self.structure(structure)) * mask[..., None]
        if self.operator == "mixer":
            summary = self.mixer(x, raw, mask, roots, b)
        elif self.operator == "transformer":
            summary = self.transformer(x, mask, roots)
        else:
            summary = self.ssm(x, mask, roots, b)
        features = torch.cat(
            (self.references(b, summary, self.whole(b["whole_x"])), self.asked(self.input(b["task"]), b)), -1
        )
        return self.head(features[b["selected"]], b, summary)
