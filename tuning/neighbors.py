import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import CausalAttention, SelectiveSSM, TimeEncoding, masked_mean, masked_weights
from .turns import EndContext, enabled, lookback


class TreeNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        hidden = spec['hidden']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turns = EndContext(hidden * 2)
        self.depth = spec['tree/depth']
        self.input = nn.Linear(registry.features, hidden)
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        self.time = TimeEncoding(hidden)
        self.relation = nn.Embedding(registry.relation_slots, hidden)
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
        roots = self.encode(b['tree'], self.depth).reshape(b['predictions'], -1)
        if self.has_turns:
            previous = self.encode(b['end_tree'], self.depth).reshape(b['predictions'], 2, -1, roots.shape[-1] // 2)
            previous = previous * b['end_raw_mask'][:, :2, :, None]
            previous = previous.transpose(1, 2).reshape(b['predictions'], -1, roots.shape[-1])
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


class SequenceNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        self.cfg = cfg
        self.operator, self.structure_mode = spec['sequence/operator'], spec['sequence/structure']
        hidden, depth = spec['hidden'], spec['sequence/layers']
        if enabled(cfg):
            self.turn_time = nn.Linear(3, hidden)
            self.turn_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.time = TimeEncoding(hidden)
        self.norm = nn.LayerNorm(hidden)
        if self.structure_mode != 'none':
            self.structure = nn.Sequential(nn.Linear(2 if self.structure_mode == 'occurrence' else 12, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        if self.operator == 'mixer':
            tokens = spec['sequence/neighbors'] + (lookback(cfg) if enabled(cfg) else 0)
            self.layers = nn.ModuleList(MixerLayer(hidden, tokens) for _ in range(depth))
            self.node = nn.Linear(hidden * 2, hidden)
        elif self.operator == 'transformer':
            self.patch_size = spec['sequence/patch']
            self.patch = nn.Linear(hidden * self.patch_size, hidden) if self.patch_size > 1 else nn.Identity()
            self.layers = nn.ModuleList(CausalAttention(hidden, spec['sequence/heads'], spec['dropout']) for _ in range(depth))
        else:
            self.node_layers = nn.ModuleList(SelectiveSSM(hidden, spec['sequence/state']) for _ in range(depth))
            self.time_layers = nn.ModuleList(SelectiveSSM(hidden, spec['sequence/state']) for _ in range(depth))
            self.opposite = nn.Linear(hidden, hidden)
            self.temporal_select = nn.Linear(hidden, hidden)
        self.reward = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def mixer(self, x, raw, mask, roots, b):
        for layer in self.layers:
            x = layer(x, mask)
        temporal = masked_mean(x, mask, 2)
        spatial = masked_mean(raw, mask, 2)
        return self.node(torch.cat((temporal, spatial + roots), -1)).flatten(1)

    def transformer(self, x, mask, roots):
        n, sides, tokens, hidden = x.shape
        patch = self.patch_size
        padding = (-tokens) % patch
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding))
            tokens += padding
        if patch > 1:
            x = self.patch(x.reshape(n, sides, tokens // patch, hidden * patch))
            mask = mask.reshape(n, sides, tokens // patch, patch).any(-1)
        x, mask = x.flatten(1, 2), mask.flatten(1)
        x = torch.cat((x, roots), 1)
        mask = torch.cat((mask, torch.ones(n, 2, device=x.device, dtype=torch.bool)), 1)
        for layer in self.layers:
            x = layer(x, mask, causal=False)
        return x[:, -2:].flatten(1)

    def ssm(self, x, mask, roots, b):
        n, sides, tokens, hidden = x.shape
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
        return (sequence * weights[..., None]).sum(2).flatten(1)

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
        if self.structure_mode != 'none':
            structure = b['occurrence'].log1p() if self.structure_mode == 'occurrence' else b['structure']
            x = (x + self.structure(structure)) * mask[..., None]
        if self.operator == 'mixer':
            summary = self.mixer(x, raw, mask, roots, b)
        elif self.operator == 'transformer':
            summary = self.transformer(x, mask, roots)
        else:
            summary = self.ssm(x, mask, roots, b)
        return self.reward(torch.cat((summary, self.input(b['query'])), -1)).squeeze(-1)


class WalkNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        hidden = spec['hidden']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turn_time = nn.Linear(3, hidden)
            self.turn_kind = nn.Embedding(2, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.anonymous = nn.Sequential(nn.Linear(2 * spec['walks/hops'], hidden), nn.ReLU(), nn.Linear(hidden, hidden))
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
