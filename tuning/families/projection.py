import math

import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import (
    Asked,
    Head,
    References,
    Relations,
    TimeEncoding,
    keys_of,
    masked_mean,
    normalization,
    padded,
)
from .periods import EndContext, enabled, lookback


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
        basis = b["projection_basis"] / math.sqrt(width)
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
