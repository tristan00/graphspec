import math

import torch
from torch import nn
from torch.nn import functional as F

from graphspec.infra.timing import timed

from .layers import (
    Asked,
    Head,
    References,
    Relations,
    Spatial,
    TemporalEdges,
    TimeEncoding,
    aggregate,
    keys_of,
    masked_mean,
    masked_weights,
    mean,
    normalization,
    padded,
    related,
)
from .periods import EndContext, enabled, lookback

KINDS = 4


class EventGraphNet(nn.Module):
    @timed("events.eventgraphnet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        hidden = spec["hidden"]
        self.features, self.period_context = registry.features, registry.marker("period_context")
        self.slots = registry.relation_slots
        self.has_periods = enabled(cfg)
        if self.has_periods:
            self.period_time = nn.Linear(3, hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.kind = nn.Embedding(KINDS + 1 if self.has_periods else KINDS, hidden)
        self.relation = Relations(registry, hidden)
        self.keys = keys_of(spec, hidden)
        self.layers = nn.ModuleList(
            TemporalEdges(
                hidden,
                spec["event_graph/message"] == "attention",
                self.slots + KINDS,
                spec["dropout"],
                spec["time/encoding"],
            )
            for _ in range(spec["event_graph/layers"])
        )
        self.whole = nn.Linear(registry.features, hidden * 2)
        self.references = References(registry, hidden * 2, hidden * 3)
        self.head = Head(hidden * 4, spec, registry)

    def forward(self, b):
        event = self.input(b["event_x"]) + self.kind(b["events"][:, 3]) + self.relation(related(b["events"]))
        initial = self.input(b["initial"]) + self.kind.weight[3]
        if self.keys is not None:
            token = b["entity_token"]
            event = event + 0.5 * (self.keys(token[b["events"][:, 0]]) + self.keys(token[b["events"][:, 1]]))
            initial = initial + self.keys(token)
        count = len(event)
        x = torch.cat((event, initial))
        roots = b["events"][:, :2].T.reshape(-1) + count
        destinations = torch.arange(count, device=x.device).repeat(2)
        edges = torch.cat((b["dag_edges"], torch.stack((roots, destinations))), 1)
        relation = torch.cat((b["dag_relation"], torch.full_like(destinations, self.slots + 3)))
        gap = torch.cat((b["dag_gap"], (b["events"][:, 2] - b["start"] * 2 + 1).float().repeat(2)))
        source, member = b["dag_member_edges"]
        members = len(b["member_row"])
        if self.has_periods:
            held = b["member_end_mask"]
            mask = held.reshape(-1)
            anchors = self.input(b["member_end_raw"].reshape(-1, self.features)[mask]) + self.kind.weight[3]
            anchors = anchors + self.period_time(b["member_end_gaps"].reshape(-1, 3)[mask].log1p())
            anchor_ids = torch.arange(len(anchors), device=x.device) + len(x)
            nodes = torch.arange(members, device=x.device) + len(x) + len(anchors)
            anchor_target = nodes[:, None].expand(-1, held.shape[1]).reshape(-1)[mask]
            edges = torch.cat(
                (edges, torch.stack((source, nodes[member])), torch.stack((anchor_ids, anchor_target))), 1
            )
            relation = torch.cat(
                (relation, b["events"][source, 3] + self.slots, torch.full_like(anchor_ids, self.period_context))
            )
            moment = b["row_steps"][b["member_row"][member]] + b["start"]
            gap = torch.cat(
                (
                    gap,
                    (moment * 2 - b["events"][source, 2]).float(),
                    b["member_end_gaps"].reshape(-1, 3)[mask, 0],
                )
            )
            x = torch.cat((x, anchors, self.input(b["member_x"]) + self.kind.weight[4]))
        for layer in self.layers:
            x = layer(x, edges, relation, gap)
        pooled = x[nodes] if self.has_periods else mean(x[source], member, members)
        values = torch.cat((pooled, self.input(b["member_x"])), -1)
        features = torch.cat(
            (
                self.references(b, values, self.whole(b["whole_x"])),
                self.asked(self.input(b["task"]), b),
            ),
            -1,
        )
        return self.head(features[b["selected"]], b)


class Rule(nn.Module):
    def initial(self, state, observed, initialized):
        return state

    def prepare(self, events, raw):
        return None

    def state(self, memory):
        return memory

    def read(self, memory, roots, last, frame_tick=None):
        return memory[roots]


class MailboxGRU(Rule):
    @timed("events.mailboxgru")
    def __init__(self, hidden, registry):
        super().__init__()
        self.event_kind = nn.Embedding(3, hidden)
        self.status = nn.Embedding(3, hidden)
        self.message = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.GRUCell(hidden, hidden)

    def prepare(self, events, raw):
        return self.event_kind(events[:, 3]) + self.status(events[:, 8] + 1)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous, edge, count):
        source, destination = events[:, 0], events[:, 1]
        receivers, partners = torch.cat((source, destination)), torch.cat((destination, source))
        content = torch.cat(
            (memory[receivers], memory[partners], raw.repeat(2, 1), clock(receivers), edge.repeat(2, 1)), -1
        )
        mailbox = (
            memory.new_zeros(memory.shape).index_add_(0, receivers, self.message(content)) / count[:, None]
        )
        return torch.where(touched[:, None], self.update(mailbox, memory), memory), touched


class MutualRNN(Rule):
    @timed("events.mutualrnn")
    def __init__(self, hidden, registry):
        super().__init__()
        self.left_cell, self.right_cell = nn.RNNCell(hidden * 3, hidden), nn.RNNCell(hidden * 3, hidden)
        self.projection = nn.Linear(1, hidden)

    def initial(self, state, observed, initialized):
        return state, state.clone(), observed, initialized

    def read(self, memory, roots, last, frame_tick=None):
        left, right, observed, _ = memory
        head = left[roots]
        if frame_tick is not None:
            head = head * (1 + self.projection(torch.log1p((frame_tick - last[roots]).clamp_min(0))[:, None]))
        return head + right[roots] + observed[roots]

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous, edge, count):
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
    @timed("events.intensity")
    def __init__(self, hidden, registry):
        super().__init__()
        self.processes = nn.ModuleList(nn.Linear(hidden * 4, hidden) for _ in range(2))
        self.intensity = nn.ModuleList(nn.Linear(hidden * 2, 1) for _ in range(2))
        self.structural = Spatial(hidden, "attention", registry.group_codes, {})
        self.state_map = nn.Linear(hidden, hidden)
        self.change_status = nn.Embedding(3, hidden)

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous, edge, count):
        source, destination, kind = events[:, 0], events[:, 1], events[:, 3]
        neighborhood = self.structural(memory, b["frame_edges"][time_step])
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
    @timed("events.antisymmetric")
    def __init__(self, hidden, registry, steps, damping):
        super().__init__()
        self.steps, self.damping = steps, damping
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.drive = nn.Linear(hidden, hidden)

    def matrix(self):
        return (
            self.weight
            - self.weight.T
            - self.damping * torch.eye(len(self.weight), device=self.weight.device)
        )

    def step(self, memory, events, raw, clock, touched, b, time_step, now, previous, edge, count):
        receivers = torch.cat((events[:, 0], events[:, 1]))
        forcing = mean(raw.repeat(2, 1), receivers, len(memory))
        matrix = self.matrix()
        interval = 0.5 * (now - previous) / self.steps
        for _ in range(self.steps):
            derivative = (
                memory @ matrix
                + self.neighbor(aggregate(memory, b["frame_edges"][time_step]))
                + self.drive(forcing)
            ).tanh()
            memory = memory + interval * derivative
        return memory, touched


class MemoryNet(nn.Module):
    @timed("events.memorynet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        self.cfg, self.values = cfg, registry.values
        hidden, update = spec["hidden"], spec["memory/update"]
        mutual = update == "mutual_rnn"
        self.attention = not mutual and spec["memory/embedding"] == "attention"
        self.pair_terms = not mutual and spec["memory/pair_terms"]
        if enabled(cfg):
            self.periods = EndContext(hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        self.relation = Relations(registry, hidden)
        self.keys = keys_of(spec, hidden)
        if update != "antisymmetric" or self.attention:
            self.time = TimeEncoding(hidden, spec["time/encoding"])
        if update == "mailbox_gru":
            self.rule = MailboxGRU(hidden, registry)
        elif update == "mutual_rnn":
            self.rule = MutualRNN(hidden, registry)
        elif update == "intensity":
            self.rule = Intensity(hidden, registry)
        else:
            self.rule = Antisymmetric(
                hidden, registry, spec["memory/integration_steps"], spec["memory/damping"]
            )
        if self.attention:
            self.query_key, self.query_value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.whole = nn.Linear(registry.features, hidden)
        self.references = References(registry, hidden, hidden * 2)
        self.head = Head(hidden * (7 if self.pair_terms else 3), spec, registry)

    def readout(self, values, neighbors, end_values, b):
        if self.attention:
            values = values + self.input(b["member_x"])
            neighbors = neighbors + self.time(b["history_gap"]) + self.relation(b["history_relation"])
            scores = (self.query_key(values)[:, None] * neighbors).sum(-1) / math.sqrt(values.shape[-1])
            weights = masked_weights(scores, b["history_mask"])
            values = values + (self.query_value(neighbors) * weights[..., None]).sum(1)
        if end_values is not None:
            owner = b["member_row"]
            values = self.periods(
                values,
                end_values.reshape(len(owner), lookback(self.cfg), -1),
                b["end_gaps"][owner],
                b["end_mask"][owner],
            )
        return values

    def paired(self, values, common, b):
        ends = padded(values, b["pair"])
        ids = b["common_ids"]
        weight = b["common_frequency"] * (ids >= 0)
        counts = b["common_stats"]
        return torch.cat(
            (
                ends[:, 0] * ends[:, 1],
                (common * weight[:, :, None]).sum(1),
                ends[:, 0] * counts[:, 1, None].log1p(),
                ends[:, 1] * counts[:, 2, None].log1p(),
            ),
            -1,
        )

    def forward(self, b):
        state = self.input(b["initial"]) + self.kind(b["entity_kind"])
        if self.keys is not None:
            state = state + self.keys(b["entity_token"])
        hidden = state.shape[-1]
        memory = self.rule.initial(state, self.input(b["initial"]), b["initial"][:, self.values :].any(-1))
        last = state.new_full((b["entities"],), float(b["start"] * 2 - 1))
        groups = {tick: (index, begin, end) for index, (tick, begin, end) in enumerate(b["event_groups"])}
        encoded = self.input(b["event_x"]) + self.relation(related(b["events"]))
        static = self.rule.prepare(b["events"], encoded)
        sizes = torch.as_tensor([end - begin for _, begin, end in b["event_groups"]], device=state.device)
        member = torch.repeat_interleave(
            torch.arange(len(sizes), device=state.device), sizes, output_size=len(b["events"])
        ).repeat(2)
        receivers = torch.cat((b["events"][:, 0], b["events"][:, 1]))
        touched_all = torch.zeros(len(sizes), b["entities"], dtype=torch.bool, device=state.device)
        touched_all[member, receivers] = True
        counts = (
            torch.zeros(len(sizes), b["entities"], device=state.device)
            .index_put_((member, receivers), state.new_ones(len(receivers)), accumulate=True)
            .clamp_min_(1)
        )
        values, neighbors, common, order, rows = [], [], [], [], []
        context = enabled(self.cfg)
        if context:
            end_values = state.new_zeros(len(b["member_row"]) * lookback(self.cfg), hidden)
        previous = b["start"] * 2 - 1
        for time_step in range(b["time_steps"]):
            step, chosen = b["frame_rows"][time_step], b["frame_members"][time_step]
            tick = (b["start"] + time_step) * 2
            for phase in (0, 1):
                now = tick + phase
                if phase:
                    values.append(self.rule.read(memory, b["member_entity"][chosen], last, tick))
                    order.append(chosen)
                    if self.attention:
                        neighbors.append(
                            self.rule.state(memory)[b["history_ids"][chosen].long().clamp_min(0)]
                        )
                    if self.pair_terms:
                        common.append(self.rule.state(memory)[b["common_ids"][step].long().clamp_min(0)])
                        rows.append(step)
                group = groups.get(now)
                if group is None:
                    continue
                index, begin, end = group
                events, raw = b["events"][begin:end], encoded[begin:end]

                def clock(ids, now=now, last=last):
                    return self.time((now - last[ids]).clamp_min(0))

                memory, touched = self.rule.step(
                    memory,
                    events,
                    raw,
                    clock,
                    touched_all[index],
                    b,
                    time_step,
                    now,
                    previous,
                    None if static is None else static[begin:end],
                    counts[index],
                )
                last = last.masked_fill(touched, float(now))
                previous = now
            if context:
                ids, roots = b["end_capture"][time_step]
                if len(ids):
                    known = b["member_end_mask"].reshape(-1)[ids]
                    end_values = end_values.index_copy(
                        0, ids, self.rule.read(memory, roots, last) * known[:, None]
                    )
        inverse = torch.argsort(torch.cat(order))
        values = self.readout(
            torch.cat(values)[inverse],
            torch.cat(neighbors)[inverse] if self.attention else None,
            end_values if context else None,
            b,
        )
        parts = [self.references(b, values, self.whole(b["whole_x"]))]
        if self.pair_terms:
            parts.append(self.paired(values, torch.cat(common)[torch.argsort(torch.cat(rows))], b))
        features = torch.cat(parts + [self.asked(self.input(b["task"]), b)], -1)
        return self.head(features[b["selected"]], b)


class ProjectionNet(nn.Module):
    @timed("events.projectionnet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        self.cfg = cfg
        hidden, hops = spec["hidden"], spec["projection/hops"]
        if enabled(cfg):
            self.periods = EndContext(hidden)
        self.input = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.keys = keys_of(spec, hidden)
        self.time = TimeEncoding(hidden, spec["time/encoding"])
        self.relation = Relations(registry, hidden)
        self.projection = nn.Sequential(nn.Linear((hops + 1) ** 2, hidden), nn.SiLU(), normalization(hidden))
        self.crossing = nn.Sequential(nn.Linear((hops + 1) ** 2, hidden), nn.SiLU(), normalization(hidden))
        self.whole = nn.Linear(registry.features, hidden * 2)
        self.references = References(registry, hidden * 2, hidden * 3)
        self.head = Head(hidden * 5, spec, registry)

    def structural(self, b):
        spec = self.cfg["model"]
        hops, width, decay = spec["projection/hops"], spec["projection/width"], spec["projection/decay"]
        basis = []
        for graph, size in zip(b["graphs"], b["entity_sizes"]):
            generator = torch.Generator(device=b["task"].device).manual_seed(self.cfg["seed"] + graph)
            basis.append(
                torch.randn(size, width, device=b["task"].device, generator=generator) / math.sqrt(width)
            )
        basis = torch.cat(basis)
        state = basis.new_zeros(b["entities"], hops + 1, width)
        state[:, 0] = basis
        groups = {tick: (start, end) for tick, start, end in b["event_groups"]}
        orders = torch.arange(hops + 1, device=basis.device)[None, :, None]
        previous, output, order, crossed, rows = b["start"] * 2, [], [], [], []
        pair = b["pair"]
        ends = padded(b["member_entity"], pair)
        context = enabled(self.cfg)
        if context:
            end_values = basis.new_zeros(len(b["member_row"]) * lookback(self.cfg), (hops + 1) ** 2)
        for time_step in range(b["time_steps"]):
            step, chosen = b["frame_rows"][time_step], b["frame_members"][time_step]
            for phase in (0, 1):
                tick = (b["start"] + time_step) * 2 + phase
                state = state * torch.exp(-decay * orders * (tick - previous))
                previous = tick
                if phase:
                    values = state[b["member_entity"][chosen]]
                    gram = values @ values.transpose(1, 2)
                    output.append((gram.sign() * gram.abs().log1p()).flatten(1))
                    order.append(chosen)
                    held = state[ends[step]]
                    gram = (held[:, 0] @ held[:, 1].transpose(1, 2)) * (pair[step, :1] >= 0)[..., None]
                    crossed.append((gram.sign() * gram.abs().log1p()).flatten(1))
                    rows.append(step)
                group = groups.get(tick)
                if group:
                    start, end = group
                    events = b["events"][start:end]
                    events = events[events[:, 3] != 0]
                    source, destination = events[:, :2].T
                    ids, others = torch.cat((source, destination)), torch.cat((destination, source))
                    increment = torch.zeros_like(state)
                    increment[:, 1:] = increment[:, 1:].index_add(0, ids, state[others, :-1])
                    state = state + increment
            if context:
                ids, roots = b["end_capture"][time_step]
                if len(ids):
                    values = state[roots]
                    gram = values @ values.transpose(1, 2)
                    end_values = end_values.index_copy(
                        0, ids, (gram.sign() * gram.abs().log1p()).reshape(len(ids), -1)
                    )
        current = torch.cat(output)[torch.argsort(torch.cat(order))]
        cross = torch.cat(crossed)[torch.argsort(torch.cat(rows))]
        if not context:
            return current, None, cross
        return current, end_values.reshape(len(b["member_row"]), lookback(self.cfg), -1), cross

    def forward(self, b):
        current, ended, cross = self.structural(b)
        structural = self.projection(current)
        if ended is not None:
            owner = b["member_row"]
            structural = self.periods(
                structural, self.projection(ended), b["end_gaps"][owner], b["end_mask"][owner]
            )
        history = self.input(b["history_x"]) + self.time(b["history_gap"]) + self.relation(b["history_relation"])
        roots = self.input(b["member_x"]) + masked_mean(history, b["history_mask"], 1)
        if self.keys is not None:
            roots = roots + self.keys(b["entity_token"][b["member_entity"]])
        members = torch.cat((roots, structural), -1)
        features = torch.cat(
            (
                self.references(b, members, self.whole(b["whole_x"])),
                self.crossing(cross) * (b["pair"][:, :1] >= 0),
                self.asked(self.input(b["task"]), b),
            ),
            -1,
        )
        return self.head(features[b["selected"]], b)
