import torch
from torch import nn
from torch.nn import functional as F

from .layers import Head, keys_of, normalization, segment_softmax

CODE_BITS = 14


class MessageLayer(nn.Module):
    def __init__(self, hidden, aggregation, update, dropout):
        super().__init__()
        self.aggregation, self.update = aggregation, update
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Linear(CODE_BITS + 2, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.attention = nn.Linear(hidden * 3, 1) if aggregation == "attention" else None
        self.cell = nn.GRUCell(hidden, hidden) if update == "gru" else nn.Linear(hidden, hidden)
        self.norm = normalization(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge):
        source, destination = edge_index
        features = F.silu(self.edge(edge))
        messages = self.source(h)[source] * torch.sigmoid(self.gate(features)) + features
        if self.aggregation == "attention":
            hidden = h.shape[1]
            weight = self.attention.weight[0]
            score = (
                (h @ weight[:hidden])[source]
                + (h @ weight[hidden : 2 * hidden])[destination]
                + (features * weight[2 * hidden :]).sum(-1)
                + self.attention.bias
            )
            messages = messages * segment_softmax(score, destination, len(h))[:, None]
        aggregate = h.new_zeros(h.shape).index_add_(0, destination, messages)
        if self.aggregation == "mean":
            count = h.new_zeros(len(h)).index_add_(0, destination, h.new_ones(len(destination)))
            aggregate = aggregate / count.clamp_min(1)[:, None]
        update = self.cell(aggregate, h) if self.update == "gru" else h + F.silu(self.cell(aggregate))
        return self.norm(h + self.dropout(update - h))


class MixedNet(nn.Module):
    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg["model"]
        hidden = spec["hidden"]
        self.input = nn.Sequential(nn.Linear(registry.features, hidden), normalization(hidden), nn.SiLU())
        self.kind = nn.Embedding(7, hidden)
        self.keys = keys_of(spec, hidden)
        self.layers = nn.ModuleList(
            MessageLayer(hidden, spec["mixed/aggregation"], spec["mixed/update"], spec["dropout"])
            for _ in range(spec["mixed/layers"])
        )
        self.reward = Head(hidden, spec, registry)
        self.register_buffer("shifts", torch.arange(CODE_BITS), persistent=False)

    def forward(self, b):
        mixed = b["mixed"]
        h = self.input(mixed["x"]) + self.kind(mixed["kind"])
        if self.keys is not None:
            h = h + self.keys(mixed["token"])
        edge = torch.cat((((mixed["edge_code"][:, None] >> self.shifts) & 1).float(), mixed["edge_gap"]), 1)
        for layer in self.layers:
            h = layer(h, mixed["edge_index"], edge)
        return self.reward(h[mixed["query"]][b["selected"]], b)
