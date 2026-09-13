import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Spatial, TemporalEdges, TimeEncoding, aggregate, masked_mean, masked_weights, mean
from .turns import EndContext, enabled, lookback


class EventGraphNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        hidden = spec['hidden']
        self.features, self.turn_context = registry.features, registry.marker('turn_context')
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turn_time = nn.Linear(3, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.kind = nn.Embedding(5 if self.has_turns else 4, hidden)
        self.layers = nn.ModuleList(TemporalEdges(hidden, spec['event_graph/message'] == 'attention', registry.relation_slots, spec['dropout'])
                                    for _ in range(spec['event_graph/layers']))
        self.reward = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        event = self.input(b['event_x']) + self.kind(b['events'][:, 3])
        initial = self.input(b['initial']) + self.kind.weight[3]
        count = len(event)
        x = torch.cat((event, initial))
        roots = b['events'][:, :2].T.reshape(-1) + count
        destinations = torch.arange(count, device=x.device).repeat(2)
        edges = torch.cat((b['dag_edges'], torch.stack((roots, destinations))), 1)
        relation = torch.cat((b['dag_relation'], torch.full_like(destinations, 3)))
        gap = torch.cat((b['dag_gap'], (b['events'][:, 2] - b['start'] * 2 + 1).float().repeat(2)))
        if self.has_turns:
            mask = b['end_raw_mask'].reshape(-1)
            anchors = self.input(b['end_raw'].reshape(-1, self.features)[mask]) + self.kind.weight[3]
            anchors = anchors + self.turn_time(b['end_token_gaps'].reshape(-1, 3)[mask].log1p())
            anchor_ids = torch.arange(len(anchors), device=x.device) + len(x)
            queries = torch.arange(b['predictions'], device=x.device) + len(x) + len(anchors)
            source, target = b['dag_query_edges']
            anchor_target = queries[:, None, None].expand(-1, 3, b['end_steps'].shape[1]).reshape(-1)[mask]
            edges = torch.cat((edges, torch.stack((source, queries[target])), torch.stack((anchor_ids, anchor_target))), 1)
            relation = torch.cat((relation, b['events'][source, 3], torch.full_like(anchor_ids, self.turn_context)))
            gap = torch.cat((gap, ((b['query_steps'][target] + b['start']) * 2 - b['events'][source, 2]).float(),
                b['end_token_gaps'].reshape(-1, 3)[mask, 0]))
            x = torch.cat((x, anchors, self.input(b['query']) + self.kind.weight[4]))
        for layer in self.layers:
            x = layer(x, edges, relation, gap)
        source, destination = b['dag_query_edges']
        q = self.input(b['query'])
        pooled = x[queries] if self.has_turns else mean(x[source], destination, b['predictions'])
        features = torch.cat((pooled, self.input(b['context'][:, 0]), self.input(b['context'][:, 1]), q), -1)
        return self.reward(features).squeeze(-1)[b['selected']]


class Rule(nn.Module):

    def initial(self, state, observed, initialized):
        return state

    def state(self, memory):
        return memory

    def read(self, memory, roots, last, frame_tick=None):
        return memory[roots]


class MailboxGRU(Rule):

    def __init__(self, hidden, registry):
        super().__init__()
        self.non_relation = registry.marker('non_relation')
        self.event_kind = nn.Embedding(3, hidden)
        self.relation = nn.Embedding(registry.relation_slots, hidden)
        self.status = nn.Embedding(3, hidden)
        self.message = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.GRUCell(hidden, hidden)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous):
        source, destination, kind = events[:, 0], events[:, 1], events[:, 3]
        receivers, partners = torch.cat((source, destination)), torch.cat((destination, source))
        rel = torch.where(kind == 1, events[:, 6] + 1, torch.where(kind == 2, torch.full_like(kind, self.non_relation), torch.zeros_like(kind)))
        edge = (self.event_kind(kind) + self.relation(rel) + self.status(events[:, 8] + 1)).repeat(2, 1)
        content = torch.cat((memory[receivers], memory[partners], raw.repeat(2, 1), clock(receivers), edge), -1)
        mailbox = mean(self.message(content), receivers, len(memory))
        return torch.where(touched[:, None], self.update(mailbox, memory), memory), touched


class MutualRNN(Rule):

    def __init__(self, hidden, registry):
        super().__init__()
        self.left_cell, self.right_cell = nn.RNNCell(hidden * 3, hidden), nn.RNNCell(hidden * 3, hidden)
        self.projection = nn.Linear(1, hidden)

    def initial(self, state, observed, initialized):
        return state, state.clone(), observed, initialized

    def read(self, memory, roots, last, frame_tick=None):
        left, right, observed, _ = memory
        actor, subject = roots[:, 0], roots[:, 1]
        head = left[actor]
        if frame_tick is not None:
            head = head * (1 + self.projection(torch.log1p((frame_tick - last[actor]).clamp_min(0))[:, None]))
        return torch.stack((head + observed[actor], right[subject] + observed[subject]), 1)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous):
        left, right, observed, initialized = memory
        source, destination, kind = events[:, 0], events[:, 1], events[:, 3]
        seen = kind == 0
        obs = source[seen]
        new = ~initialized[obs]
        left = left.index_copy(0, obs[new], raw[seen][new])
        right = right.index_copy(0, obs[new], raw[seen][new])
        observed = observed.index_copy(0, obs, raw[seen])
        initialized = initialized.clone()
        initialized[obs] = True
        chosen = kind == 2
        s, d = source[chosen], destination[chosen]
        old_left, old_right = left[s], right[d]
        elapsed = clock(s)
        new_left = self.left_cell(torch.cat((old_right, raw[chosen] + observed[s], elapsed), -1), old_left)
        new_right = self.right_cell(torch.cat((old_left, raw[chosen] + observed[d], elapsed), -1), old_right)
        left, right = left.index_copy(0, s, new_left), right.index_copy(0, d, new_right)
        touched = torch.zeros_like(touched)
        touched[torch.cat((obs[new], s, d))] = True
        return (left, right, observed, initialized), touched


class Intensity(Rule):

    def __init__(self, hidden, registry):
        super().__init__()
        self.processes = nn.ModuleList(nn.Linear(hidden * 4, hidden) for _ in range(2))
        self.intensity = nn.ModuleList(nn.Linear(hidden * 2, 1) for _ in range(2))
        self.structural = Spatial(hidden, 'attention', registry.relations, {})
        self.state_map = nn.Linear(hidden, hidden)
        self.change_status = nn.Embedding(3, hidden)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous):
        source, destination, kind = events[:, 0], events[:, 1], events[:, 3]
        neighborhood = self.structural(memory, b['frame_edges'][time_step])
        messages = torch.zeros_like(memory)
        for process in range(2):
            use = kind == process + 1
            ids = torch.cat((source[use], destination[use]))
            others = torch.cat((destination[use], source[use]))
            intensity = F.softplus(self.intensity[process](torch.cat((memory[ids], memory[others]), -1)))
            typed = raw[use] + self.change_status(events[use, 8] + 1)
            content = torch.cat((memory[others], neighborhood[ids], typed.repeat(2, 1), clock(ids)), -1)
            messages = messages + mean(self.processes[process](content).tanh() * intensity, ids, len(memory))
        seen = kind == 0
        messages = messages + mean(raw[seen], source[seen], len(memory))
        updated = (self.state_map(memory) + messages).tanh()
        return torch.where(touched[:, None], updated, memory), touched


class Antisymmetric(Rule):

    def __init__(self, hidden, registry, steps, damping):
        super().__init__()
        self.steps, self.damping = steps, damping
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.drive = nn.Linear(hidden, hidden)

    def matrix(self):
        return self.weight - self.weight.T - self.damping * torch.eye(len(self.weight), device=self.weight.device)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous):
        receivers = torch.cat((events[:, 0], events[:, 1]))
        forcing = mean(raw.repeat(2, 1), receivers, len(memory))
        matrix = self.matrix()
        interval = 0.5 * (now - previous) / self.steps
        for _ in range(self.steps):
            derivative = (memory @ matrix + self.neighbor(aggregate(memory, b['frame_edges'][time_step])) + self.drive(forcing)).tanh()
            memory = memory + interval * derivative
        return memory, touched


class MemoryNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        self.cfg, self.width = cfg, registry.width
        hidden, update = spec['hidden'], spec['memory/update']
        mutual = update == 'mutual_rnn'
        self.attention = not mutual and spec['memory/embedding'] == 'attention'
        self.pair_terms = not mutual and spec['memory/pair_terms']
        if enabled(cfg):
            self.turns = EndContext(hidden * 2)
        self.input = nn.Linear(registry.features, hidden)
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        if update != 'antisymmetric' or self.attention:
            self.time = TimeEncoding(hidden)
        if update == 'mailbox_gru':
            self.rule = MailboxGRU(hidden, registry)
        elif update == 'mutual_rnn':
            self.rule = MutualRNN(hidden, registry)
        elif update == 'intensity':
            self.rule = Intensity(hidden, registry)
        else:
            self.rule = Antisymmetric(hidden, registry, spec['memory/integration_steps'], spec['memory/damping'])
        if self.attention:
            self.query_key, self.query_value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.reward = nn.Sequential(nn.Linear(hidden * (5 if self.pair_terms else 3), hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def embedding(self, memory, roots, step, b, last, frame_tick):
        endpoints = self.rule.read(memory, roots, last, frame_tick)
        if not self.attention:
            return endpoints
        state = self.rule.state(memory)
        values = endpoints + self.input(b['context'][step, :2])
        ids, mask = b['history_ids'][step].long(), b['history_mask'][step]
        neighbors = state[ids.clamp_min(0)] + self.time(b['history_gap'][step])
        scores = (self.query_key(values)[:, :, None] * neighbors).sum(-1) / math.sqrt(state.shape[-1])
        weights = masked_weights(scores, mask)
        return values + (self.query_value(neighbors) * weights[..., None]).sum(2)

    def pairs(self, endpoints, memory, step, b):
        state = self.rule.state(memory)
        ids = b['common_ids'][step].long()
        weight = b['common_frequency'][step] * (ids >= 0)
        common = (state[ids.clamp_min(0)] * weight[:, :, None]).sum(1)
        counts = b['common_stats'][step]
        return torch.cat((endpoints[:, 0] * endpoints[:, 1], common, endpoints[:, 0] * counts[:, 1, None].log1p(), endpoints[:, 1] * counts[:, 2, None].log1p()), -1)

    def forward(self, b):
        state = self.input(b['initial']) + self.kind(b['entity_kind'])
        hidden = state.shape[-1]
        memory = self.rule.initial(state, self.input(b['initial']), b['initial'][:, self.width:].any(-1))
        last = state.new_full((b['entities'],), float(b['start'] * 2 - 1))
        groups = {tick: (begin, end) for tick, begin, end in b['event_groups']}
        encoded = self.input(b['event_x'])
        predictions, order = [], []
        context = enabled(self.cfg)
        if context:
            end_values = state.new_zeros(b['predictions'] * lookback(self.cfg), hidden * 2)
        previous = b['start'] * 2 - 1
        for time_step in range(b['time_steps']):
            step = b['frame_queries'][time_step]
            tick = (b['start'] + time_step) * 2
            for phase in (0, 1):
                now = tick + phase
                if phase:
                    q = self.input(b['query'][step])
                    roots = torch.stack((b['actor'][step], b['subject'][step]), -1)
                    endpoints = self.embedding(memory, roots, step, b, last, tick).flatten(1)
                    if context:
                        endpoints = self.turns(endpoints, end_values.reshape(b['predictions'], lookback(self.cfg), -1)[step],
                                               b['end_gaps'][step], b['end_mask'][step])
                    if self.pair_terms:
                        endpoints = self.pairs(endpoints.reshape(len(step), 2, hidden), memory, step, b)
                    predictions.append(self.reward(torch.cat((endpoints, q), -1)).squeeze(-1))
                    order.append(step)
                group = groups.get(now)
                if group is None:
                    continue
                begin, end = group
                events, raw = b['events'][begin:end], encoded[begin:end]
                touched = torch.zeros(b['entities'], dtype=torch.bool, device=state.device)
                touched[torch.cat((events[:, 0], events[:, 1]))] = True

                def clock(ids):
                    return self.time((now - last[ids]).clamp_min(0))

                memory, touched = self.rule.step(memory, events, raw, clock, touched, b, time_step, now, previous)
                last = torch.where(touched, torch.full_like(last, now), last)
                previous = now
            if context:
                ids, roots = b['end_capture'][time_step]
                if len(ids):
                    prior = self.rule.read(memory, roots, last).reshape(len(ids), -1)
                    known = b['end_raw_mask'][:, :2].transpose(1, 2).reshape(-1, 2)[ids]
                    end_values = end_values.index_copy(0, ids, prior * known.repeat_interleave(hidden, -1))
        return torch.cat(predictions)[torch.argsort(torch.cat(order))][b['selected']]


class ProjectionNet(nn.Module):

    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg['model']
        self.cfg = cfg
        hidden, hops = spec['hidden'], spec['projection/hops']
        if enabled(cfg):
            self.turns = EndContext(hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.time = TimeEncoding(hidden)
        self.projection = nn.Sequential(nn.Linear((2 * (hops + 1))**2, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.reward = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def structural(self, b):
        spec = self.cfg['model']
        hops, width, decay = spec['projection/hops'], spec['projection/width'], spec['projection/decay']
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
        context = enabled(self.cfg)
        if context:
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
            if context:
                ids, roots = b['end_capture'][time_step]
                if len(ids):
                    values = state[roots].reshape(len(ids), 2 * (hops + 1), width)
                    gram = values @ values.transpose(1, 2)
                    end_values = end_values.index_copy(0, ids, (gram.sign() * gram.abs().log1p()).reshape(len(ids), -1))
        current = torch.cat(output)[torch.argsort(torch.cat(order))]
        return (current, end_values.reshape(b['predictions'], lookback(self.cfg), -1)) if context else current

    def forward(self, b):
        values = self.structural(b)
        if enabled(self.cfg):
            structural = self.turns(self.projection(values[0]), self.projection(values[1]), b['end_gaps'], b['end_mask'])
        else:
            structural = self.projection(values)
        history = self.input(b['history_x']) + self.time(b['history_gap'])
        roots = self.input(b['context'][:, :2]) + masked_mean(history, b['history_mask'], 2)
        return self.reward(torch.cat((roots.reshape(b['predictions'], -1), structural, self.input(b['query'])), -1)).squeeze(-1)[b['selected']]
