import math

import torch
from torch import nn

from . import spec
from .layers import CausalAttention, SelectiveSSM, TimeEncoding
from .turns import EndContext, enabled


def masked_mean(x, mask, dim):
    return (x * mask[..., None]).sum(dim) / mask.sum(dim).clamp_min(1)[..., None]


def masked_weights(score, mask):
    score = score.masked_fill(~mask, -1e4)
    weight = score.softmax(-1) * mask
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)


class TemporalAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg['hidden']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turns = EndContext(hidden * 2)
        self.depth = cfg['layers']
        self.input = nn.Linear(spec.FEATURES, hidden)
        self.kind = nn.Embedding(len(spec.NODES), hidden)
        self.time = TimeEncoding(hidden)
        self.relation = nn.Embedding(spec.RELATION_SLOTS, hidden)
        self.q = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.k = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.v = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.depth))
        self.merge = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)) for _ in range(self.depth))
        self.reward = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def encode(self, tree, depth):
        root = self.input(tree['x']) + self.kind(tree['kind'])
        if not depth:
            return root
        children = self.encode(tree['child'], depth - 1).reshape(len(root), tree['mask'].shape[1], -1)
        tokens = children + self.time(tree['gap']) + self.relation(tree['relation'])
        score = (self.q[depth - 1](root)[:, None] * self.k[depth - 1](tokens)).sum(-1) / math.sqrt(root.shape[-1])
        weights = masked_weights(score, tree['mask'])
        neighbor = (self.v[depth - 1](tokens) * weights[:, :, None]).sum(1)
        return self.merge[depth - 1](torch.cat((root, neighbor), -1))

    def forward(self, b):
        roots = self.encode(b['tree'], self.depth).reshape(b['frames'], -1)
        if self.has_turns:
            previous = self.encode(b['end_tree'], self.depth).reshape(b['frames'], 2, -1, roots.shape[-1] // 2)
            previous = previous * b['end_raw_mask'][:, :2, :, None]
            previous = previous.transpose(1, 2).reshape(b['frames'], -1, roots.shape[-1])
            roots = self.turns(roots, previous, b['end_gaps'], b['end_mask'])
        return self.reward(torch.cat((roots, self.input(b['query'])), -1)).squeeze(-1)[b['selected']]


class MixerLayer(nn.Module):

    def __init__(self, hidden, tokens):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.token = nn.Sequential(nn.Linear(tokens, tokens * 2), nn.GELU(), nn.Linear(tokens * 2, tokens))
        self.channel = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x, mask):
        x = (x + self.token((self.norm1(x) * mask[..., None]).transpose(-1, -2)).transpose(-1, -2)) * mask[..., None]
        return (x + self.channel(self.norm2(x))) * mask[..., None]


class HistoryModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.family, self.cfg = int(cfg['model'][1:]), cfg
        hidden, depth = cfg['hidden'], cfg['layers']
        if enabled(cfg):
            self.turn_time = nn.Linear(3, hidden)
            self.turn_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(spec.FEATURES, hidden)
        self.time = TimeEncoding(hidden)
        self.norm = nn.LayerNorm(hidden)
        if self.family == 21:
            tokens = cfg['neighbors'] + (cfg['turn_lookback'] if enabled(cfg) else 0)
            self.layers = nn.ModuleList(MixerLayer(hidden, tokens) for _ in range(depth))
            self.node = nn.Linear(hidden * 2, hidden)
        elif self.family in (22, 23):
            self.structure = nn.Sequential(nn.Linear(2 if self.family == 22 else 12, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.patch = nn.Linear(hidden * cfg['patch'], hidden) if self.family == 22 else nn.Identity()
            self.layers = nn.ModuleList(CausalAttention(hidden, cfg['heads'], cfg['dropout']) for _ in range(depth))
        elif self.family == 27:
            self.node_layers = nn.ModuleList(SelectiveSSM(hidden, cfg['state_size']) for _ in range(depth))
            self.time_layers = nn.ModuleList(SelectiveSSM(hidden, cfg['state_size']) for _ in range(depth))
            self.opposite = nn.Linear(hidden, hidden)
            self.temporal_select = nn.Linear(hidden, hidden)
        self.reward = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        keys = ('context', 'query', 'history_x', 'history_mask', 'history_gap', 'occurrence', 'structure', 'pair_gap', 'pair_mask')
        if enabled(self.cfg):
            keys += ('history_end_time', 'history_turn_kind')
        b = {key: b[key][b['selected']] for key in keys}
        roots = self.input(b['context'][:, :2])
        mask = b['history_mask']
        raw = self.input(b['history_x'])
        x = self.norm(raw + self.time(b['history_gap'])) * mask[..., None]
        if enabled(self.cfg):
            x = (x + self.turn_time(b['history_end_time'].log1p()) + self.turn_kind(b['history_turn_kind'])) * mask[..., None]
        n, sides, tokens, hidden = x.shape
        if self.family == 21:
            for layer in self.layers:
                x = layer(x, mask)
            temporal = masked_mean(x, mask, 2)
            spatial = masked_mean(raw, mask, 2)
            summary = self.node(torch.cat((temporal, spatial + roots), -1)).flatten(1)
        elif self.family in (22, 23):
            structure = b['occurrence'].log1p() if self.family == 22 else b['structure']
            x = (x + self.structure(structure)) * mask[..., None]
            if self.family == 22:
                patch = self.cfg['patch']
                padding = (-tokens) % patch
                if padding:
                    x = torch.nn.functional.pad(x, (0, 0, 0, padding))
                    mask = torch.nn.functional.pad(mask, (0, padding))
                    tokens += padding
                x = self.patch(x.reshape(n, sides, tokens // patch, hidden * patch))
                mask = mask.reshape(n, sides, tokens // patch, patch).any(-1)
            x, mask = x.flatten(1, 2), mask.flatten(1)
            x = torch.cat((x, roots), 1)
            mask = torch.cat((mask, torch.ones(n, 2, device=x.device, dtype=torch.bool)), 1)
            for layer in self.layers:
                x = layer(x, mask, causal=False)
            summary = x[:, -2:].flatten(1)
        elif self.family == 27:
            x = torch.cat((x, roots[:, :, None]), 2).reshape(n * 2, tokens + 1, hidden)
            mask = torch.cat((mask, torch.ones(n, 2, 1, device=x.device, dtype=torch.bool)), 2).reshape(n * 2, tokens + 1)
            for layer in self.node_layers:
                x = layer(x, mask)
            times = self.time(b['pair_gap'])
            for layer in self.time_layers:
                times = layer(times, b['pair_mask'])
            pattern = self.temporal_select(masked_mean(times, b['pair_mask'], 1))
            sequence = x.reshape(n, 2, tokens + 1, hidden)
            mask = mask.reshape(n, 2, tokens + 1)
            other = self.opposite(masked_mean(sequence, mask, 2).flip(1))
            key = other * pattern[:, None]
            weights = masked_weights((sequence * key[:, :, None]).sum(-1), mask)
            summary = (sequence * weights[..., None]).sum(2).flatten(1)
        return self.reward(torch.cat((summary, self.input(b['query'])), -1)).squeeze(-1)


class AnonymousWalk(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg['hidden']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turn_time = nn.Linear(3, hidden)
            self.turn_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(spec.FEATURES, hidden)
        self.anonymous = nn.Sequential(nn.Linear(2 * cfg['walk_hops'], hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.time = TimeEncoding(hidden)
        self.walk = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(hidden, 1)
        self.reward = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        raw = self.input(b['walk_x']) + self.anonymous(b['walk_identity'].log1p()) + self.time(b['walk_gap'])
        if self.has_turns:
            raw = raw + self.turn_time(b['walk_end_time'].log1p()) + self.turn_kind(b['walk_turn_kind'])
        mask = b['walk_mask']
        n, sides, walks, hops, hidden = raw.shape
        x = (raw * mask[..., None]).reshape(-1, hops, hidden)
        lengths = mask.sum(-1).reshape(-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (state, _) = self.walk(packed)
        encoded = state.transpose(0, 1).reshape(n, sides, walks, hidden)
        valid = lengths.reshape(n, sides, walks) > 0
        weight = masked_weights(self.score(encoded).squeeze(-1), valid)
        summary = (encoded * weight[..., None]).sum(2) + self.input(b['context'][:, :2])
        return self.reward(torch.cat((summary.reshape(n, -1), self.input(b['query'])), -1)).squeeze(-1)[b['selected']]
