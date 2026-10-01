import math

import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Asked, Head, References, TimeEncoding, keys_of, masked_weights, normalization
from .periods import EndContext, enabled, lookback


class TreeNet(nn.Module):
    @timed("neighbors.treenet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        hidden = spec["hidden"]
        self.cfg = cfg
        self.has_periods = enabled(cfg)
        if self.has_periods:
            self.periods = EndContext(hidden)
        self.depth = spec["tree/depth"]
        self.input = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        self.keys = keys_of(spec, hidden)
        self.time = TimeEncoding(hidden, spec["time/encoding"])
        self.relation = nn.Embedding(registry.relation_slots, hidden)
        self.q = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.k = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.v = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.merge = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), normalization(hidden))
            for _ in range(self.depth)
        )
        self.whole = nn.Linear(registry.features, hidden)
        self.references = References(registry, hidden, hidden * 2)
        self.head = Head(hidden * 3, spec, registry, hidden)

    def encode(self, tree, depth):
        root = self.input(tree["x"]) + self.kind(tree["kind"])
        if self.keys is not None:
            root = root + self.keys(tree["token"])
        if not depth:
            return root
        children = self.encode(tree["child"], depth - 1).reshape(
            len(root), tree["mask"].shape[1], root.shape[-1]
        )
        tokens = children + self.time(tree["gap"]) + self.relation(tree["relation"])
        score = (self.q[depth - 1](root)[:, None] * self.k[depth - 1](tokens)).sum(-1) / math.sqrt(
            root.shape[-1]
        )
        weights = masked_weights(score, tree["mask"])
        neighbor = (self.v[depth - 1](tokens) * weights[:, :, None]).sum(1)
        return self.merge[depth - 1](torch.cat((root, neighbor), -1))

    def forward(self, b):
        members = self.encode(b["tree"], self.depth)
        if self.has_periods:
            owner = b["member_row"]
            previous = self.encode(b["end_tree"], self.depth).reshape(len(owner), lookback(self.cfg), -1)
            previous = previous * b["member_end_mask"][..., None]
            members = self.periods(members, previous, b["end_gaps"][owner], b["end_mask"][owner])
        features = torch.cat(
            (self.references(b, members, self.whole(b["whole_x"])), self.asked(self.input(b["task"]), b)), -1
        )
        return self.head(features[b["selected"]], b, members)
