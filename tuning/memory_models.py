import math

import torch
from torch import nn
from torch.nn import functional as F

from .history_models import masked_mean, masked_weights
from .layers import Spatial, TimeEncoding, aggregate, mean
from .turns import EndContext, enabled, lookback


class MemoryModel(nn.Module):

    def __init__(self, cfg, registry):
        super().__init__()
        self.family, self.cfg = int(cfg['model'][1:]), cfg
        self.width, self.non_relation = registry.width, registry.marker('non_relation')
        hidden = cfg['hidden']
        if enabled(cfg):
            self.turns = EndContext(hidden * 2)
        self.input = nn.Linear(registry.features, hidden)
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        if self.family != 29:
            self.time = TimeEncoding(hidden)
        if self.family in (18, 26):
            self.event_kind = nn.Embedding(3, hidden)
            self.relation = nn.Embedding(registry.relation_slots, hidden)
            self.status = nn.Embedding(3, hidden)
            self.message = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.update = nn.GRUCell(hidden, hidden)
        if self.family in (18, 20, 26):
            self.query_key, self.query_value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        if self.family == 19:
            self.left_cell, self.right_cell = nn.RNNCell(hidden * 3, hidden), nn.RNNCell(hidden * 3, hidden)
            self.projection = nn.Linear(1, hidden)
        elif self.family == 20:
            self.processes = nn.ModuleList(nn.Linear(hidden * 4, hidden) for _ in range(2))
            self.intensity = nn.ModuleList(nn.Linear(hidden * 2, 1) for _ in range(2))
            self.structural = Spatial(hidden, 'attention', registry.relations, {})
            self.state_map = nn.Linear(hidden, hidden)
            self.change_status = nn.Embedding(3, hidden)
        elif self.family == 29:
            self.weight = nn.Parameter(torch.empty(hidden, hidden))
            nn.init.xavier_uniform_(self.weight)
            self.neighbor = nn.Linear(hidden, hidden, bias=False)
            self.drive = nn.Linear(hidden, hidden)
        head_width = hidden * (5 if self.family == 26 else 3)
        self.reward = nn.Sequential(nn.Linear(head_width, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def antisymmetric(self):
        return self.weight - self.weight.T - self.cfg['damping'] * torch.eye(len(self.weight), device=self.weight.device)

    def embedding(self, state, b, step):
        roots = torch.stack((b['actor'][step], b['subject'][step]), -1)
        values = state[roots] + self.input(b['context'][step, :2])
        ids, mask = b['history_ids'][step].long(), b['history_mask'][step]
        neighbors = state[ids.clamp_min(0)] + self.time(b['history_gap'][step])
        scores = (self.query_key(values)[:, :, None] * neighbors).sum(-1) / math.sqrt(state.shape[-1])
        weights = masked_weights(scores, mask)
        return values + (self.query_value(neighbors) * weights[..., None]).sum(2)

    def forward(self, b):
        state = self.input(b['initial']) + self.kind(b['entity_kind'])
        left, right = state, state.clone()
        observed_features = self.input(b['initial'])
        initialized = b['initial'][:, self.width:].any(-1)
        last = state.new_full((b['entities'],), float(b['start'] * 2 - 1))
        groups = {tick: (begin, end) for tick, begin, end in b['event_groups']}
        encoded = self.input(b['event_x'])
        predictions, order = [], []
        if enabled(self.cfg):
            end_values = state.new_zeros(b['predictions'] * lookback(self.cfg), state.shape[-1] * 2)
        previous_tick = b['start'] * 2 - 1
        for time_step in range(b['time_steps']):
            step = b['frame_queries'][time_step]
            tick = (b['start'] + time_step) * 2
            for phase in (0, 1):
                if phase:
                    q = self.input(b['query'][step])
                    if self.family == 19:
                        root = b['actor'][step]
                        projected = left[root] * (1 + self.projection(torch.log1p((tick - last[root]).clamp_min(0))[:, None]))
                        features = torch.cat((projected + observed_features[root], right[b['subject'][step]] + observed_features[b['subject'][step]], q), -1)
                    elif self.family == 26:
                        endpoints = self.embedding(state, b, step)
                        if enabled(self.cfg):
                            endpoints = self.turns(endpoints.flatten(1),
                                end_values.reshape(b['predictions'], lookback(self.cfg), -1)[step],
                                b['end_gaps'][step], b['end_mask'][step]).reshape(len(step), 2, state.shape[-1])
                        ids = b['common_ids'][step].long()
                        weight = b['common_frequency'][step] * (ids >= 0)
                        common = (state[ids.clamp_min(0)] * weight[:, :, None]).sum(1)
                        counts = b['common_stats'][step]
                        features = torch.cat((endpoints[:, 0] * endpoints[:, 1], common, endpoints[:, 0] * counts[:, 1, None].log1p(), endpoints[:, 1] * counts[:, 2, None].log1p(), q), -1)
                    elif self.family in (18, 20):
                        features = torch.cat((self.embedding(state, b, step).flatten(1), q), -1)
                    else:
                        features = torch.cat((state[b['actor'][step]], state[b['subject'][step]], q), -1)
                    if enabled(self.cfg) and self.family != 26:
                        first = self.turns(features[:, :state.shape[-1] * 2],
                            end_values.reshape(b['predictions'], lookback(self.cfg), -1)[step],
                            b['end_gaps'][step], b['end_mask'][step])
                        features = torch.cat((first, q), -1)
                    predictions.append(self.reward(features).squeeze(-1))
                    order.append(step)
                group = groups.get(tick + phase)
                if group is None:
                    continue
                begin, end = group
                events, raw = b['events'][begin:end], encoded[begin:end]
                source, destination, event_tick, kind = events[:, :4].T
                receivers = torch.cat((source, destination))
                partners = torch.cat((destination, source))
                touched = torch.zeros(b['entities'], dtype=torch.bool, device=state.device)
                touched[receivers] = True
                if self.family == 19:
                    observed = kind == 0
                    obs = source[observed]
                    new = ~initialized[obs]
                    left = left.index_copy(0, obs[new], raw[observed][new])
                    right = right.index_copy(0, obs[new], raw[observed][new])
                    observed_features = observed_features.index_copy(0, obs, raw[observed])
                    initialized = initialized.clone()
                    initialized[obs] = True
                    choice = kind == 2
                    s, d = source[choice], destination[choice]
                    old_left, old_right = left[s], right[d]
                    elapsed = self.time((tick + phase - last[s]).clamp_min(0))
                    new_left = self.left_cell(torch.cat((old_right, raw[choice] + observed_features[s], elapsed), -1), old_left)
                    new_right = self.right_cell(torch.cat((old_left, raw[choice] + observed_features[d], elapsed), -1), old_right)
                    left, right = left.index_copy(0, s, new_left), right.index_copy(0, d, new_right)
                    touched = torch.zeros_like(touched)
                    touched[torch.cat((obs[new], s, d))] = True
                elif self.family == 20:
                    neighborhood = self.structural(state, b['frame_edges'][time_step])
                    messages = torch.zeros_like(state)
                    for process in range(2):
                        use = kind == process + 1
                        ids = torch.cat((source[use], destination[use]))
                        others = torch.cat((destination[use], source[use]))
                        pair = torch.cat((state[ids], state[others]), -1)
                        intensity = F.softplus(self.intensity[process](pair))
                        time = self.time((tick + phase - last[ids]).clamp_min(0))
                        typed = raw[use] + self.change_status(events[use, 8] + 1)
                        content = torch.cat((state[others], neighborhood[ids], typed.repeat(2, 1), time), -1)
                        messages = messages + mean(self.processes[process](content).tanh() * intensity, ids, len(state))
                    observed = kind == 0
                    messages = messages + mean(raw[observed], source[observed], len(state))
                    updated = (self.state_map(state) + messages).tanh()
                    state = torch.where(touched[:, None], updated, state)
                elif self.family == 29:
                    forcing = mean(raw.repeat(2, 1), receivers, len(state))
                    matrix = self.antisymmetric()
                    interval = 0.5 * (tick + phase - previous_tick) / self.cfg['integration_steps']
                    for _ in range(self.cfg['integration_steps']):
                        derivative = (state @ matrix + self.neighbor(aggregate(state, b['frame_edges'][time_step])) + self.drive(forcing)).tanh()
                        state = state + interval * derivative
                else:
                    rel = torch.where(kind == 1, events[:, 6] + 1, torch.where(kind == 2, torch.full_like(kind, self.non_relation), torch.zeros_like(kind)))
                    edge = self.event_kind(kind).repeat(2, 1) + self.relation(rel).repeat(2, 1)
                    edge = edge + self.status(events[:, 8] + 1).repeat(2, 1)
                    temporal = self.time((tick + phase - last[receivers]).clamp_min(0))
                    content = torch.cat((state[receivers], state[partners], raw.repeat(2, 1), temporal, edge), -1)
                    mailbox = mean(self.message(content), receivers, len(state))
                    updated = self.update(mailbox, state)
                    state = torch.where(touched[:, None], updated, state)
                last = torch.where(touched, torch.full_like(last, tick + phase), last)
                previous_tick = tick + phase
            if enabled(self.cfg):
                ids, roots = b['end_capture'][time_step]
                if len(ids):
                    prior = torch.cat((left[roots[:, 0]] + observed_features[roots[:, 0]],
                        right[roots[:, 1]] + observed_features[roots[:, 1]]), -1) if self.family == 19 else state[roots].reshape(len(ids), -1)
                    known = b['end_raw_mask'][:, :2].transpose(1, 2).reshape(-1, 2)[ids]
                    prior = prior * known.repeat_interleave(state.shape[-1], -1)
                    end_values = end_values.index_copy(0, ids, prior)
        return torch.cat(predictions)[torch.argsort(torch.cat(order))][b['selected']]


class WalkProjection(nn.Module):

    def __init__(self, cfg, registry):
        super().__init__()
        self.cfg = cfg
        hidden, hops = cfg['hidden'], cfg['projection_hops']
        if enabled(cfg):
            self.turns = EndContext(hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.time = TimeEncoding(hidden)
        self.projection = nn.Sequential(nn.Linear((2 * (hops + 1))**2, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.reward = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def structural(self, b):
        hops, width, decay = self.cfg['projection_hops'], self.cfg['projection_width'], self.cfg['decay']
        basis = []
        for graph, size in zip(b['graphs'], b['entity_sizes']):
            generator = torch.Generator(device=b['query'].device).manual_seed(self.cfg['seed'] + graph)
            basis.append(torch.randn(size, width, device=b['query'].device, generator=generator) / math.sqrt(width))
        basis = torch.cat(basis)
        state = basis.new_zeros(b['entities'], hops + 1, width)
        state[:, 0] = basis
        groups = {tick: (start, end) for tick, start, end in b['event_groups']}
        orders = torch.arange(hops + 1, device=basis.device)[None, :, None]
        previous, output, order = b['start'] * 2, [], []
        if enabled(self.cfg):
            end_values = basis.new_zeros(b['predictions'] * lookback(self.cfg), (2 * (hops + 1))**2)
        for time_step in range(b['time_steps']):
            step = b['frame_queries'][time_step]
            for phase in (0, 1):
                tick = (b['start'] + time_step) * 2 + phase
                state = state * torch.exp(-decay * orders * (tick - previous))
                previous = tick
                if phase:
                    ids = torch.stack((b['actor'][step], b['subject'][step]), -1)
                    values = state[ids].reshape(len(step), 2 * (hops + 1), width)
                    gram = values @ values.transpose(1, 2)
                    output.append((gram.sign() * gram.abs().log1p()).flatten(1))
                    order.append(step)
                group = groups.get(tick)
                if group:
                    start, end = group
                    events = b['events'][start:end]
                    events = events[events[:, 3] != 0]
                    source, destination = events[:, :2].T
                    ids, others = torch.cat((source, destination)), torch.cat((destination, source))
                    increment = torch.zeros_like(state)
                    increment[:, 1:] = increment[:, 1:].index_add(0, ids, state[others, :-1])
                    state = state + increment
            if enabled(self.cfg):
                ids, roots = b['end_capture'][time_step]
                if len(ids):
                    values = state[roots].reshape(len(ids), 2 * (hops + 1), width)
                    gram = values @ values.transpose(1, 2)
                    end_values = end_values.index_copy(0, ids, (gram.sign() * gram.abs().log1p()).reshape(len(ids), -1))
        current = torch.cat(output)[torch.argsort(torch.cat(order))]
        return (current, end_values.reshape(b['predictions'], lookback(self.cfg), -1)) if enabled(self.cfg) else current

    def forward(self, b):
        values = self.structural(b)
        if enabled(self.cfg):
            structural = self.turns(self.projection(values[0]), self.projection(values[1]), b['end_gaps'], b['end_mask'])
        else:
            structural = self.projection(values)
        history = self.input(b['history_x']) + self.time(b['history_gap'])
        roots = self.input(b['context'][:, :2]) + masked_mean(history, b['history_mask'], 2)
        return self.reward(torch.cat((roots.reshape(b['predictions'], -1), structural, self.input(b['query'])), -1)).squeeze(-1)[b['selected']]


class VirtualState(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        self.buffer_size = cfg['affinity_buffer']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turns = EndContext(categories * 2)
        self.xh, self.hh, self.xs, self.hs, self.gs = (nn.Linear(categories, categories, bias=False) for _ in range(5))
        self.bh, self.bs = nn.Parameter(torch.zeros(categories)), nn.Parameter(torch.zeros(categories))
        self.query = nn.Linear(registry.features, cfg['hidden'])
        self.reward = nn.Sequential(nn.Linear(categories * 2 + cfg['hidden'], cfg['hidden']), nn.SiLU(), nn.Linear(cfg['hidden'], 1))

    def forward(self, b):
        state = b['affinity'].new_zeros(b['entities'], b['affinity'].shape[-1])
        buffer, output, order = [], [], []
        for time_step in range(b['time_steps']):
            step = b['frame_queries'][time_step]
            actor = b['actor'][step]
            x = b['affinity'][step].log1p()
            global_state = torch.stack(buffer[-self.buffer_size:]).mean(0) if buffer else torch.zeros_like(x)
            keep = (self.xh(x) + self.hh(state[actor]) + self.bh).sigmoid()
            hidden = keep * state[actor] + (1 - keep) * x
            select = (self.xs(x) + self.hs(hidden) + self.gs(global_state) + self.bs).sigmoid()
            affinity = select * hidden + (1 - select) * x
            pair = torch.cat((affinity, global_state), -1)
            if self.has_turns:
                previous = torch.cat((b['end_affinity'][step].log1p(), b['end_global_affinity'][step].log1p()), -1)
                pair = self.turns(pair, previous, b['end_gaps'][step], b['end_mask'][step])
            output.append(self.reward(torch.cat((pair, self.query(b['query'][step])), -1)).squeeze(-1))
            order.append(step)
            state = state.index_copy(0, actor, hidden)
            events = b['events']
            choice = events[(events[:, 2] == (b['start'] + time_step) * 2 + 1) & (events[:, 3] == 2)]
            category = choice[:, 6]
            profile = b['affinity'][step].clone()
            profile[torch.arange(len(step), device=state.device), category] += 1
            buffer.append(profile.log1p())
        return torch.cat(output)[torch.argsort(torch.cat(order))][b['selected']]
