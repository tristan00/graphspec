import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Asked, Head, References, Relations, TimeEncoding, keys_of, masked_weights
from .periods import enabled


class WalkNet(nn.Module):
    @timed("neighbors.walknet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        hidden = spec["hidden"]
        if type(hidden) is not int or hidden < 1:
            raise ValueError("walk hidden width must be a positive integer")
        self.has_periods = enabled(cfg)
        if self.has_periods:
            self.period_time = nn.Linear(3, hidden)
            self.period_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.anonymous = nn.Sequential(
            nn.Linear(2 * spec["walks/hops"], hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.keys = keys_of(spec, hidden)
        self.time = TimeEncoding(hidden, spec["time/encoding"])
        self.relation = Relations(registry, hidden)
        direction_width = (hidden + 1) // 2
        self.walk = nn.LSTM(hidden, direction_width, batch_first=True, bidirectional=True)
        self.walk_output = (
            nn.Identity() if direction_width * 2 == hidden else nn.Linear(direction_width * 2, hidden)
        )
        self.score = nn.Linear(hidden, 1)
        self.asked = Asked(registry, hidden)
        self.whole = nn.Linear(registry.features, hidden)
        self.references = References(registry, hidden, hidden * 2)
        self.head = Head(hidden * 3, spec, registry)

    def walked(self, raw, mask):
        members, walks, hops, hidden = raw.shape
        if not members:
            return raw.new_zeros(0, hidden)
        x = (raw * mask[..., None]).reshape(-1, hops, hidden)
        lengths = mask.sum(-1).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (state, _) = self.walk(packed)
        encoded = self.walk_output(state.transpose(0, 1).reshape(members, walks, -1))
        weight = masked_weights(self.score(encoded).squeeze(-1), lengths.reshape(members, walks) > 0)
        return (encoded * weight[..., None]).sum(1)

    def forward(self, b):
        raw = self.input(b["walk_x"]) + self.anonymous(b["walk_identity"].log1p()) + self.time(b["walk_gap"])
        raw = raw + self.relation(b["walk_relation"])
        if self.keys is not None:
            raw = raw + self.keys(b["walk_token"])
        if self.has_periods:
            raw = raw + self.period_time(b["walk_end_time"].log1p()) + self.period_kind(b["walk_period_kind"])
        summary = self.walked(raw, b["walk_mask"]) + self.input(b["member_x"])
        if self.keys is not None:
            summary = summary + self.keys(b["member_token"])
        features = torch.cat(
            (self.references(b, summary, self.whole(b["whole_x"])), self.asked(self.input(b["task"]), b)), -1
        )
        return self.head(features[b["selected"]], b)
