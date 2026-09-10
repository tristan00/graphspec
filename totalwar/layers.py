import math

import torch
from torch import nn
from torch.nn import functional as F

from . import spec


def mean(values, index, size):
    result = values.new_zeros((size, values.shape[-1])).index_add_(0, index, values)
    count = values.new_zeros(size).index_add_(0, index, values.new_ones(len(index)))
    return result / count.clamp_min(1)[:, None]


def segment_softmax(scores, index, size):
    shape = (size,) + scores.shape[1:]
    expanded = index.reshape((-1,) + (1,) * (scores.ndim - 1)).expand_as(scores)
    maxima = scores.new_full(shape, -torch.inf).scatter_reduce_(0, expanded, scores.detach(), reduce='amax', include_self=True)
    weights = (scores - maxima[index]).exp()
    total = weights.new_zeros(shape).index_add_(0, index, weights)
    return weights / total[index].clamp_min(1e-8)


def aggregate(h, edges, mode='mean'):
    source, destination = edges
    if mode == 'gcn':
        degree = h.new_ones(len(h)).index_add_(0, destination, h.new_ones(len(destination)))
        scale = degree.rsqrt()
        result = h * scale[:, None].square()
        return result.index_add(0, destination, h[source] * (scale[source] * scale[destination])[:, None])
    return mean(h[source], destination, len(h))


class TimeEncoding(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.frequency = nn.Parameter(torch.logspace(0, -3, width))
        self.phase = nn.Parameter(torch.zeros(width))

    def forward(self, gap):
        return torch.cos(torch.log1p(gap.clamp_min(0))[..., None] * self.frequency + self.phase)


class Spatial(nn.Module):

    def __init__(self, hidden, mode, relations=spec.RELATIONS, attributed=(), dropout=0.0):
        super().__init__()
        self.mode, self.dropout = mode, dropout
        self.root = nn.Linear(hidden, hidden)
        if mode != 'relational':
            self.message = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        if mode == 'relational':
            self.weights = nn.Parameter(torch.empty(relations, hidden, hidden))
            nn.init.xavier_uniform_(self.weights)
            self.attributes = nn.ModuleDict({name: nn.Linear(len(spec.attributes(name)), hidden, bias=False) for name in attributed})
        elif mode == 'attention':
            self.query, self.key = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)

    def forward(self, h, edges, relation=None, attributes=None, attribute_edges=None):
        source, destination = edges
        if self.mode == 'relational':
            messages = h.new_zeros(len(source), h.shape[-1])
            for kind in relation.unique().tolist():
                indices = torch.nonzero(relation == kind, as_tuple=True)[0]
                messages = messages.index_copy(0, indices, h[source[indices]] @ self.weights[kind])
            if attributes:
                for name, values in attributes.items():
                    if len(values):
                        messages = messages.index_add(0, attribute_edges[name], self.attributes[name](values.sign() * values.abs().log1p()))
            update = mean(messages, destination, len(h))
        elif self.mode == 'attention':
            score = (self.query(h)[destination] * self.key(h)[source]).sum(-1) / math.sqrt(h.shape[-1])
            weight = segment_softmax(score, destination, len(h))
            update = h.new_zeros(h.shape).index_add_(0, destination, self.message(h)[source] * weight[:, None])
        else:
            update = self.message(aggregate(h, edges, 'gcn' if self.mode == 'gcn' else 'mean'))
        return self.norm(h + F.dropout(F.silu(self.root(h) + update), self.dropout, self.training))


class CausalAttention(nn.Module):

    def __init__(self, hidden, heads=2, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.feed = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x, mask, causal=True):
        n, t, h = x.shape
        positions = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, h, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / h))
        position = torch.stack(((positions * frequency).sin(), (positions * frequency).cos()), -1).reshape(t, -1)[:, :h]
        z = (x + position) * mask[:, :, None]
        q, k, v = self.qkv(z).reshape(n, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        allowed = mask[:, None, :].expand(-1, t, -1)
        if causal:
            allowed = allowed & torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        allowed = allowed | (~mask[:, :, None] & torch.eye(t, device=x.device, dtype=torch.bool))
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], dropout_p=self.dropout if self.training else 0.0)
        z = z.transpose(1, 2).reshape(n, t, h)
        z = self.norm1(x + self.output(z))
        return self.norm2(z + self.feed(z)) * mask[:, :, None]


class GatedTCN(nn.Module):

    def __init__(self, hidden, depth):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden * 2, 3, dilation=2**i) for i in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(depth))

    def forward(self, x, mask):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            gate, value = conv(F.pad(x.transpose(1, 2), (2 * 2**i, 0))).chunk(2, 1)
            x = norm(x + (gate.sigmoid() * value.tanh()).transpose(1, 2)) * mask[:, :, None]
        return x


class GraphCell(nn.Module):

    def __init__(self, hidden, diffusion=False, hops=2):
        super().__init__()
        self.diffusion, self.hops = diffusion, hops
        factor = 1 + 2 * hops if diffusion else 1
        self.gates = nn.Linear(2 * hidden * factor, 2 * hidden)
        self.candidate = nn.Linear(2 * hidden * factor, hidden)
        nn.init.ones_(self.gates.bias)

    def support(self, x, edges, directed):
        if not self.diffusion:
            return aggregate(x, edges, 'gcn')
        values = [x]
        for support in (directed, directed.flip(0)):
            current = x
            for _ in range(self.hops):
                current = aggregate(current, support)
                values.append(current)
        return torch.cat(values, -1)

    def forward(self, x, previous, edges, directed):
        reset, keep = self.gates(self.support(torch.cat((x, previous), -1), edges, directed)).sigmoid().chunk(2, -1)
        candidate = self.candidate(self.support(torch.cat((x, reset * previous), -1), edges, directed)).tanh()
        return keep * previous + (1 - keep) * candidate


class MatrixGate(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.input, self.recurrent = nn.Parameter(torch.empty(hidden, hidden)), nn.Parameter(torch.empty(hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(hidden, hidden))
        nn.init.xavier_uniform_(self.input)
        nn.init.orthogonal_(self.recurrent)

    def forward(self, x, state):
        return self.input @ x + self.recurrent @ state + self.bias


class EvolvingLayer(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.initial = nn.Parameter(torch.empty(hidden, hidden))
        self.scorer = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        nn.init.xavier_uniform_(self.initial)
        self.reset, self.keep, self.candidate = (MatrixGate(hidden) for _ in range(3))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, state, edges, present, entity_batch, entity_sizes):
        scores = x @ F.normalize(self.scorer, dim=0)
        padded_x = nn.utils.rnn.pad_sequence(x.split(entity_sizes), batch_first=True)
        padded_scores = nn.utils.rnn.pad_sequence(scores.masked_fill(~present, -torch.inf).split(entity_sizes), batch_first=True, padding_value=-torch.inf)
        values, indices = padded_scores.topk(min(padded_scores.shape[1], x.shape[-1]), dim=1)
        visible = padded_x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))
        summary = F.pad((visible * torch.where(values.isfinite(), values.tanh(), 0)[:, :, None]).transpose(1, 2), (0, x.shape[-1] - indices.shape[1]))
        reset, keep = self.reset(summary, state).sigmoid(), self.keep(summary, state).sigmoid()
        state = keep * state + (1 - keep) * self.candidate(summary, reset * state).tanh()
        update = torch.bmm(aggregate(x, edges, 'gcn')[:, None], state[entity_batch]).squeeze(1)
        return self.norm(x + F.silu(update)) * present[:, None], state


class TemporalEdges(nn.Module):

    def __init__(self, hidden, attention, dropout=0.0):
        super().__init__()
        self.attention, self.dropout = attention, dropout
        self.time = TimeEncoding(hidden)
        self.relation = nn.Embedding(spec.RELATION_SLOTS, hidden)
        self.value = nn.Linear(hidden, hidden)
        if attention:
            self.score = nn.Linear(hidden * 3, 1)
        else:
            self.gate = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edges, relation, gap):
        source, destination = edges
        temporal = self.time(gap) + self.relation(relation)
        message = self.value(x)[source] + temporal
        if self.attention:
            score = self.score(torch.cat((x[source], x[destination], temporal), -1)).squeeze(-1)
            weight = segment_softmax(score, destination, len(x))
            update = x.new_zeros(x.shape).index_add_(0, destination, message * weight[:, None])
        else:
            update = mean(message * self.gate(temporal).sigmoid(), destination, len(x))
        return self.norm(x + F.dropout(F.silu(update), self.dropout, self.training))


class SelectiveSSM(nn.Module):

    def __init__(self, hidden, state_size=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.in_proj = nn.Linear(hidden, hidden * 2)
        self.conv = nn.Conv1d(hidden, hidden, 3, groups=hidden)
        self.delta = nn.Linear(hidden, hidden)
        self.b, self.c = nn.Linear(hidden, state_size), nn.Linear(hidden, state_size)
        self.log_a = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(hidden, 1))
        self.skip = nn.Parameter(torch.ones(hidden))
        self.output = nn.Linear(hidden, hidden)

    def forward(self, sequence, mask):
        x, gate = self.in_proj(self.norm(sequence)).chunk(2, -1)
        x = F.silu(self.conv(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2))
        delta = F.softplus(self.delta(x)).clamp_max(10)
        b, c = self.b(x), self.c(x)
        a = -self.log_a.exp()
        state = x.new_zeros((len(x), x.shape[-1], a.shape[-1]))
        output = []
        for t in range(x.shape[1]):
            exponent = delta[:, t, :, None] * a
            transition = exponent.exp()
            injection = (transition - 1) / a * b[:, t, None, :]
            updated = transition * state + injection * x[:, t, :, None]
            state = torch.where(mask[:, t, None, None], updated, state)
            output.append((state * c[:, t, None, :]).sum(-1) + self.skip * x[:, t])
        y = torch.stack(output, 1) * F.silu(gate)
        return (sequence + self.output(y)) * mask[:, :, None]
