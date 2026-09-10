import torch
from torch import nn
from torch.nn import functional as F

from . import spec
from .message_net import TemporalRewardNet
from .layers import CausalAttention, EvolvingLayer, GatedTCN, GraphCell, Spatial, TemporalEdges, mean
from .turns import EndContext, enabled, gather_end


class SnapshotModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.family, self.cfg = int(cfg['model'][1:]), cfg
        hidden, depth = cfg['hidden'], cfg['layers']
        self.recipe = spec.resolve(cfg['design'])
        self.has_turn_edges = enabled(cfg) or spec.BOUNDARY in self.recipe.families
        if self.has_turn_edges and self.family in (13, 14):
            self.turn_edge_time = nn.Linear(1, hidden, bias=False)
        elif enabled(cfg):
            self.turns = EndContext(hidden * 3)
        self.input = nn.Sequential(nn.Linear(spec.FEATURES, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.kind = nn.Embedding(len(spec.NODES), hidden)
        self.query = nn.Linear(spec.FEATURES, hidden)
        mode = {4: 'gcn', 5: 'sage', 11: 'attention'}.get(self.family, 'relational')
        self.spatial = nn.ModuleList(Spatial(hidden, mode, spec.RELATIONS, self.recipe.attributed(),
            cfg['dropout']) for _ in range(0 if self.family in (8, 9, 10) else depth))
        if self.family in (1, 2, 3):
            self.action_round = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.feedback = nn.Linear(hidden, hidden)
            self.temporal = nn.GRU(hidden, hidden, batch_first=True) if self.family == 1 else (
                nn.LSTM(hidden, hidden, batch_first=True) if self.family == 2 else CausalAttention(hidden, cfg['heads'], cfg['dropout']))
        elif self.family in (4, 5, 6):
            self.temporal = nn.GRUCell(hidden, hidden)
        elif self.family == 7:
            self.temporal = GatedTCN(hidden, depth + 1)
        elif self.family in (8, 9):
            self.cells = nn.ModuleList(GraphCell(hidden, self.family == 9, cfg['diffusion_hops']) for _ in range(depth))
        elif self.family == 10:
            self.evolving = nn.ModuleList(EvolvingLayer(hidden) for _ in range(depth))
        elif self.family == 11:
            self.temporal = CausalAttention(hidden, cfg['heads'], cfg['dropout'])
        elif self.family in (13, 14):
            self.temporal = nn.ModuleList(TemporalEdges(hidden, self.family == 14, cfg['dropout']) for _ in range(depth))
            self.final_spatial = Spatial(hidden, 'sage', dropout=cfg['dropout'])
        self.reward = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer('kinds', torch.as_tensor(self.recipe.kind_mask(), dtype=torch.float32), persistent=False)

    def forward(self, b):
        h = self.input(b['x']) + self.kind(b['kind'])
        h = h * self.kinds[b['kind']][:, None]
        if self.family not in (8, 9, 10):
            for layer in self.spatial:
                h = layer(h, b['edges'], b['relation'], b['attributes'], b['attribute_edges'])
        q = self.query(b['query'])
        if self.family in (1, 2, 3):
            actor, subject = h[b['context_index'][:, 0]], h[b['context_index'][:, 1]]
            pooled = mean(h, b['step'], b['frames'])
            action = self.action_round(torch.cat((q, actor, subject, pooled), -1))
            feedback = h.new_zeros(h.shape).index_add_(0, b['context_index'][:, 0], self.feedback(action))
            h = h + feedback
            action = self.action_round(torch.cat((action, h[b['context_index'][:, 0]], h[b['context_index'][:, 1]], mean(h, b['step'], b['frames'])), -1))
            if self.family == 3:
                sequence = self.temporal(action.reshape(b['batch_size'], b['time_steps'], -1), torch.ones(b['batch_size'], b['time_steps'], dtype=torch.bool, device=h.device)).reshape(b['frames'], -1)
            else:
                sequence = self.temporal(action.reshape(b['batch_size'], b['time_steps'], -1))[0].reshape(b['frames'], -1)
            features = torch.cat((sequence, actor, subject, q), -1)
        elif self.family in (13, 14):
            edges = b['temporal_edges']
            relation = b['temporal_relation'] if self.has_turn_edges else torch.full_like(edges[0], spec.code('temporal'))
            if self.has_turn_edges:
                h = h + mean(self.turn_edge_time(b['temporal_turn_gap'].log1p()[:, None]), edges[1], len(h))
            for layer in self.temporal:
                h = layer(h, edges, relation, b['temporal_gap'])
            h = self.final_spatial(h, b['edges'])
            features = torch.cat((h[b['context_index'][:, 0]], h[b['context_index'][:, 1]], h[b['context_index'][:, 2]], q), -1)
        else:
            sequence = torch.cat((h, h.new_zeros(1, h.shape[-1])))[b['align']]
            present = b['present']
            if self.family in (4, 5, 6):
                state, output = h.new_zeros(b['entities'], h.shape[-1]), []
                for t in range(b['time_steps']):
                    state = torch.where(present[t, :, None], self.temporal(sequence[t], state), state)
                    output.append(state)
                sequence = torch.stack(output)
            elif self.family in (7, 11):
                sequence = self.temporal(sequence.transpose(0, 1), present.T).transpose(0, 1)
            elif self.family in (8, 9):
                states = [h.new_zeros(b['entities'], h.shape[-1]) for _ in self.cells]
                output = []
                for t in range(b['time_steps']):
                    x = sequence[t]
                    for i, cell in enumerate(self.cells):
                        update = cell(x, states[i], b['frame_edges'][t], b['frame_directed'][t])
                        states[i] = torch.where(present[t, :, None], update, states[i])
                        x = states[i] * present[t, :, None]
                    output.append(states[-1])
                sequence = torch.stack(output)
            elif self.family == 10:
                weights, output = [layer.initial.expand(b['batch_size'], -1, -1) for layer in self.evolving], []
                for t in range(b['time_steps']):
                    x = sequence[t]
                    for i, layer in enumerate(self.evolving):
                        x, weights[i] = layer(x, weights[i], b['frame_edges'][t], present[t], b['entity_batch'], b['entity_sizes'])
                    output.append(x)
                sequence = torch.stack(output)
            index = b['query_steps']
            camp = b['entity'][b['context_index'][:, 2]]
            features = torch.cat((sequence[index, b['actor']], sequence[index, b['subject']], sequence[index, camp], q), -1)
        if enabled(self.cfg) and self.family not in (13, 14):
            if self.family in (1, 2, 3):
                historical = sequence[b['end_steps'].clamp_min(0)]
                known = b['end_observation_local'][:, :, :2]
                padded = torch.cat((h, h.new_zeros(1, h.shape[-1])))
                past = padded[torch.where(known >= 0, known, len(h))]
                raw = self.input(b['end_raw'][:, :2]).permute(0, 2, 1, 3)
                past = torch.where((known >= 0)[..., None], past, raw)
                past = past * b['end_raw_mask'][:, :2].transpose(1, 2)[..., None]
                previous = torch.cat((historical, past.reshape(b['frames'], -1, h.shape[-1] * 2)), -1)
            else:
                previous = torch.cat([gather_end(sequence, ids, b) for ids in (b['actor'], b['subject'], camp)], -1)
                raw = self.input(b['end_raw']).permute(0, 2, 1, 3).reshape_as(previous)
                known = (b['end_observation_local'] >= 0).repeat_interleave(h.shape[-1], -1)
                previous = torch.where(known, previous, raw)
                previous = previous * b['end_raw_mask'].transpose(1, 2).repeat_interleave(h.shape[-1], -1)
            first = self.turns(features[:, :3 * h.shape[-1]], previous, b['end_gaps'], b['end_mask'])
            features = torch.cat((first, q), -1)
        return self.reward(features).squeeze(-1)[b['selected']]


class MixedModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.net = TemporalRewardNet(spec.FEATURES, dict(cfg, aggregation='mean', update='gru'))

    def forward(self, b):
        return self.net(b['mixed'])[b['selected']]


class EventDAG(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg['hidden']
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turn_time = nn.Linear(3, hidden)
        self.input = nn.Linear(spec.FEATURES, hidden)
        self.kind = nn.Embedding(5 if self.has_turns else 4, hidden)
        self.layers = nn.ModuleList(TemporalEdges(hidden, cfg['model'] == 'M16', cfg['dropout']) for _ in range(cfg['layers']))
        self.reward = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        event = self.input(b['event_x']) + self.kind(b['events'][:, 3])
        initial = self.input(b['initial']) + self.kind.weight[3]
        count = len(event)
        x = torch.cat((event, initial))
        roots = b['events'][:, :2].T.reshape(-1) + count
        destinations = torch.arange(count, device=x.device).repeat(2)
        edges = torch.cat((b['dag_edges'], torch.stack((roots, destinations))), 1)
        relation = torch.cat((b['dag_relation'] if 'dag_relation' in b else b['events'][b['dag_edges'][1], 3], torch.full_like(destinations, 3)))
        gap = torch.cat((b['dag_gap'], (b['events'][:, 2] - b['start'] * 2 + 1).float().repeat(2)))
        if self.has_turns:
            mask = b['end_raw_mask'].reshape(-1)
            anchors = self.input(b['end_raw'].reshape(-1, spec.FEATURES)[mask]) + self.kind.weight[3]
            anchors = anchors + self.turn_time(b['end_token_gaps'].reshape(-1, 3)[mask].log1p())
            anchor_ids = torch.arange(len(anchors), device=x.device) + len(x)
            queries = torch.arange(b['frames'], device=x.device) + len(x) + len(anchors)
            source, target = b['dag_query_edges']
            anchor_target = queries[:, None, None].expand(-1, 3, b['end_steps'].shape[1]).reshape(-1)[mask]
            edges = torch.cat((edges, torch.stack((source, queries[target])), torch.stack((anchor_ids, anchor_target))), 1)
            relation = torch.cat((relation, b['events'][source, 3], torch.full_like(anchor_ids, spec.TURN_CONTEXT)))
            gap = torch.cat((gap, ((b['query_steps'][target] + b['start']) * 2 - b['events'][source, 2]).float(),
                b['end_token_gaps'].reshape(-1, 3)[mask, 0]))
            x = torch.cat((x, anchors, self.input(b['query']) + self.kind.weight[4]))
        for layer in self.layers:
            x = layer(x, edges, relation, gap)
        source, destination = b['dag_query_edges']
        q = self.input(b['query'])
        pooled = mean(x[source], destination, b['frames'])
        if self.has_turns:
            pooled = x[queries]
        features = torch.cat((pooled, self.input(b['context'][:, 0]), self.input(b['context'][:, 1]), q), -1)
        return self.reward(features).squeeze(-1)[b['selected']]


class SignalGraph(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        hidden, channels = cfg['hidden'], 20
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turns = EndContext(hidden)
        self.topk, self.mix_hops = cfg['signal_topk'], cfg['diffusion_hops']
        self.left, self.right = nn.Parameter(torch.randn(channels, hidden)), nn.Parameter(torch.randn(channels, hidden))
        self.left_map, self.right_map = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.input = nn.Linear(2, hidden)
        self.convs = nn.ModuleList(nn.Conv2d(hidden, hidden * 2, (1, 3), dilation=(1, 2**i)) for i in range(cfg['layers']))
        self.mix = nn.ModuleList(nn.Linear(hidden * (1 + 2 * self.mix_hops), hidden) for _ in self.convs)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.convs)
        self.query = nn.Linear(spec.FEATURES, hidden)
        self.reward = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def adjacency(self):
        left, right = self.left_map(self.left).tanh(), self.right_map(self.right).tanh()
        scores = (3 * (left @ right.T - right @ left.T)).tanh().relu()
        indices = scores.topk(min(self.topk, len(scores)), dim=1).indices
        scores = scores * torch.zeros_like(scores).scatter_(1, indices, 1)
        scores = scores + torch.eye(len(scores), device=scores.device)
        return scores / scores.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, b):
        raw = b['signal'].clone()
        raw[:, :, 0] = raw[:, :, 0].sign() * raw[:, :, 0].abs().log1p()
        h = self.input(raw).reshape(b['batch_size'], b['time_steps'], 20, -1)
        adjacency = self.adjacency()
        for i, (conv, mix, norm) in enumerate(zip(self.convs, self.mix, self.norms)):
            channels = h.permute(0, 3, 2, 1)
            gate, value = conv(F.pad(channels, (2 * 2**i, 0, 0, 0))).chunk(2, 1)
            temporal = (gate.sigmoid() * value.tanh()).permute(0, 3, 2, 1)
            states = [temporal]
            for support in (adjacency, adjacency.T):
                current = temporal
                for _ in range(self.mix_hops):
                    current = 0.05 * temporal + 0.95 * torch.einsum('ij,btjh->btih', support, current)
                    states.append(current)
            h = norm(h + mix(torch.cat(states, -1)))
        values = h.mean(2).reshape(b['frames'], -1)
        if self.has_turns:
            values = self.turns(values, values[b['end_steps'].clamp_min(0)], b['end_gaps'], b['end_mask'])
        return self.reward(torch.cat((values, self.query(b['query'])), -1)).squeeze(-1)[b['selected']]
