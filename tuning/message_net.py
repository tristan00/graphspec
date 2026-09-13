import torch
from torch import nn
from torch.nn import functional as F

from .layers import segment_softmax


class MessageLayer(nn.Module):

    def __init__(self, hidden, aggregation, update, dropout):
        super().__init__()
        self.aggregation, self.update = aggregation, update
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Sequential(nn.Linear(16, hidden), nn.SiLU())
        self.gate = nn.Linear(hidden, hidden)
        self.attention = nn.Linear(hidden * 3, 1) if aggregation == 'attention' else None
        self.cell = nn.GRUCell(hidden, hidden) if update == 'gru' else nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_features):
        source, destination = edge_index
        features = self.edge(edge_features)
        messages = self.source(h[source]) * torch.sigmoid(self.gate(features)) + features
        if self.aggregation == 'attention':
            score = self.attention(torch.cat((h[source], h[destination], features), dim=-1)).squeeze(-1)
            messages = messages * segment_softmax(score, destination, len(h))[:, None]
        aggregate = h.new_zeros(h.shape).index_add_(0, destination, messages)
        if self.aggregation == 'mean':
            count = h.new_zeros(len(h)).index_add_(0, destination, h.new_ones(len(destination)))
            aggregate = aggregate / count.clamp_min(1)[:, None]
        update = self.cell(aggregate, h) if self.update == 'gru' else h + F.silu(self.cell(aggregate))
        return self.norm(h + self.dropout(update - h))


class MixedNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        hidden = spec['hidden']
        self.input = nn.Sequential(nn.Linear(registry.features, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.kind = nn.Embedding(7, hidden)
        self.layers = nn.ModuleList(MessageLayer(hidden, spec['mixed/aggregation'], spec['mixed/update'], spec['dropout'])
                                    for _ in range(spec['mixed/layers']))
        self.reward = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        mixed = b['mixed']
        h = self.input(mixed['x']) + self.kind(mixed['kind'])
        for layer in self.layers:
            h = layer(h, mixed['edge_index'], mixed['edge'])
        return self.reward(h[mixed['query']]).squeeze(-1)[b['selected']]
