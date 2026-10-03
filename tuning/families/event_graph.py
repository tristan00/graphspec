import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Asked, Head, References, Relations, TemporalEdges, keys_of, mean, related
from .periods import enabled

KINDS = 4


class Received(nn.Module):
    @timed("events.received")
    def __init__(self, spec, hidden, labels):
        super().__init__()
        self.layers = nn.ModuleList(
            TemporalEdges(hidden, spec["event_graph/message"] == "attention", labels, spec["dropout"], None)
            for _ in range(spec["event_graph/layers"])
        )

    def forward(self, sinks, sources, edges, label):
        count = len(sources)
        joined = torch.stack((edges[0], edges[1] + count))
        for layer in self.layers:
            sinks = layer(torch.cat((sources, sinks)), joined, label, None)[count:]
        return sinks


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
        self.head = Head(hidden * 4, spec, registry, hidden * 2)

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
        return self.head(features[b["selected"]], b, values)
