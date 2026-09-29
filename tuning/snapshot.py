import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import (
    Asked,
    CausalAttention,
    EvolvingLayer,
    GatedTCN,
    GraphCell,
    Groups,
    Head,
    References,
    Spatial,
    TemporalEdges,
    keys_of,
    mean,
    normalization,
)
from .periods import EndContext, enabled, gather_end


class SnapshotNet(nn.Module):
    @timed("snapshot.snapshotnet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        self.cfg, self.temporal_mode = cfg, spec["snapshot/temporal"]
        hidden = spec["hidden"]
        coupling = self.temporal_mode == "coupling_edges"
        self.context = enabled(cfg) and not coupling
        if coupling:
            self.period_edge_time = nn.Linear(1, hidden, bias=False)
        elif self.context:
            self.periods = EndContext(hidden)
            if self.temporal_mode == "graph_sequence":
                self.frames = EndContext(hidden)
        self.input = nn.Sequential(nn.Linear(registry.features, hidden), normalization(hidden), nn.SiLU())
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        self.task = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.keys = keys_of(spec, hidden)
        attributes = {group: registry.attribute_width(group) for group in registry.valued}
        self.widths = list(attributes.values())
        self.register_buffer("grouped", torch.as_tensor(registry.grouped), persistent=False)
        self.spatial = nn.ModuleList()
        if spec["snapshot/spatial"] != "none":
            self.spatial.extend(
                Spatial(
                    hidden,
                    spec["snapshot/spatial"],
                    registry.group_codes,
                    attributes,
                    spec["dropout"],
                    spec.get("snapshot/chebyshev_order", 0),
                    registry.relation_slots,
                )
                for _ in range(spec["snapshot/layers"])
            )
        if self.temporal_mode == "graph_sequence":
            self.sequence_mode = spec["snapshot/sequence"]
            self.action_round = nn.Sequential(
                nn.Linear(hidden * 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.feedback = nn.Linear(hidden, hidden)
            if self.sequence_mode == "gru":
                self.temporal = nn.GRU(hidden, hidden, batch_first=True)
            elif self.sequence_mode == "lstm":
                self.temporal = nn.LSTM(hidden, hidden, batch_first=True)
            else:
                self.temporal = CausalAttention(hidden, spec["snapshot/heads"], spec["dropout"])
        elif self.temporal_mode == "entity_gru":
            self.temporal = nn.GRUCell(hidden, hidden)
        elif self.temporal_mode == "entity_tcn":
            self.temporal = GatedTCN(hidden, spec["snapshot/temporal_layers"])
        elif self.temporal_mode == "entity_attention":
            self.temporal = CausalAttention(hidden, spec["snapshot/heads"], spec["dropout"])
        elif self.temporal_mode == "graph_cell":
            self.cells = nn.ModuleList(
                GraphCell(hidden, spec["snapshot/diffusion_hops"])
                for _ in range(spec["snapshot/temporal_layers"])
            )
        elif self.temporal_mode == "evolving":
            self.evolving = nn.ModuleList(
                EvolvingLayer(hidden) for _ in range(spec["snapshot/temporal_layers"])
            )
        else:
            self.temporal = nn.ModuleList(
                TemporalEdges(
                    hidden,
                    spec["snapshot/message"] == "attention",
                    registry.relation_slots,
                    spec["dropout"],
                    spec["time/encoding"],
                )
                for _ in range(spec["snapshot/temporal_layers"])
            )
            self.final_spatial = Spatial(hidden, "mean", registry.group_codes, {}, spec["dropout"])
        self.references = References(registry, hidden, hidden * 3)
        self.head = Head(hidden * 4, spec, registry)

    def values(self, h, sequence, b):
        rows, whole = b["row_frames"], b["whole_row"]
        if sequence is None:
            return h[b["member_index"]], mean(h, b["step"], b["frames"])[rows[whole]]
        shown = b["present"].to(sequence.dtype)
        size = (sequence.shape[0], b["batch_size"])
        total = sequence.new_zeros(size + sequence.shape[-1:]).index_add_(
            1, b["entity_batch"], sequence * shown[..., None]
        )
        count = sequence.new_zeros(size).index_add_(1, b["entity_batch"], shown)
        pooled = total / count.clamp_min(1)[..., None]
        return (
            sequence[b["row_steps"][b["member_row"]], b["member_entity"]],
            pooled[b["row_steps"][whole], rows[whole] // b["time_steps"]],
        )

    def previous(self, h, sequence, b):
        known = b["member_end_local"]
        if sequence is None:
            past = torch.cat((h, h.new_zeros(1, h.shape[-1])))[torch.where(known >= 0, known, len(h))]
        else:
            owner = b["member_row"]
            past = gather_end(sequence, b["member_entity"], b["end_steps"][owner], b["time_steps"])
        past = torch.where((known >= 0)[..., None], past, self.input(b["member_end_raw"]))
        return past * b["member_end_mask"][..., None]

    def referred(self, h, sequence, b, later):
        members, whole = self.values(h, sequence, b)
        if self.context:
            owner = b["member_row"]
            members = self.periods(
                members, self.previous(later, sequence, b), b["end_gaps"][owner], b["end_mask"][owner]
            )
        return self.references(b, members, whole)

    def graph_sequence(self, h, b):
        seen = b["frame_starts"]
        known = seen >= 0
        index = torch.where(known, seen, len(h))
        padded = torch.cat((h, h.new_zeros(1, h.shape[-1])))
        pooled = mean(h, b["step"], b["frames"])
        action = self.action_round(
            torch.cat((self.task(b["frame_task"]), padded[index[:, 0]], padded[index[:, 1]], pooled), -1)
        )
        feedback = h.new_zeros(h.shape).index_add_(
            0, seen[known[:, 0], 0], self.feedback(action[known[:, 0]])
        )
        h = h + feedback
        padded = torch.cat((h, h.new_zeros(1, h.shape[-1])))
        action = self.action_round(
            torch.cat((action, padded[index[:, 0]], padded[index[:, 1]], mean(h, b["step"], b["frames"])), -1)
        )
        frames = action.reshape(b["batch_size"], b["time_steps"], -1)
        if self.sequence_mode == "attention":
            sequence = self.temporal(
                frames, torch.ones(b["batch_size"], b["time_steps"], dtype=torch.bool, device=h.device)
            )
        else:
            sequence = self.temporal(frames)[0]
        return h, sequence.reshape(b["frames"], -1)

    def coupling(self, h, b):
        edges, relation = b["temporal_edges"], b["temporal_relation"]
        h = h + mean(self.period_edge_time(b["temporal_period_gap"].log1p()[:, None]), edges[1], len(h))
        for layer in self.temporal:
            h = layer(h, edges, relation, b["temporal_gap"])
        return self.final_spatial(h, b["edges"])

    def entity_sequence(self, h, b):
        sequence = (
            torch.cat((h, h.new_zeros(1, h.shape[-1])))
            .index_select(0, b["align"].reshape(-1))
            .reshape(b["align"].shape + (h.shape[-1],))
        )
        present = b["present"]
        if self.temporal_mode == "entity_gru":
            state, output = h.new_zeros(b["entities"], h.shape[-1]), []
            for t in range(b["time_steps"]):
                state = torch.where(present[t, :, None], self.temporal(sequence[t], state), state)
                output.append(state)
            return torch.stack(output)
        if self.temporal_mode in ("entity_tcn", "entity_attention"):
            return self.temporal(sequence.transpose(0, 1), present.T).transpose(0, 1)
        if self.temporal_mode == "graph_cell":
            states = [h.new_zeros(b["entities"], h.shape[-1]) for _ in self.cells]
            output = []
            for t in range(b["time_steps"]):
                x = sequence[t]
                for i, cell in enumerate(self.cells):
                    update = cell(x, states[i], b["frame_edges"][t], b["frame_directed"][t])
                    states[i] = torch.where(present[t, :, None], update, states[i])
                    x = states[i] * present[t, :, None]
                output.append(states[-1])
            return torch.stack(output)
        weights, output = [layer.initial.expand(b["batch_size"], -1, -1) for layer in self.evolving], []
        for t in range(b["time_steps"]):
            x = sequence[t]
            for i, layer in enumerate(self.evolving):
                x, weights[i] = layer(
                    x, weights[i], b["frame_edges"][t], present[t], b["entity_batch"], b["entity_sizes"]
                )
            output.append(x)
        return torch.stack(output)

    def observed(self, b):
        h = self.input(b["x"]) + self.kind(b["kind"])
        if self.keys is not None:
            h = h + self.keys(b["token"])
        groups = None
        if self.spatial and self.spatial[0].mode == "relational":
            groups = Groups(
                b["edges"],
                b["relation"],
                self.grouped,
                len(h),
                b["attribute_values"],
                b["attribute_edges"],
                b["attribute_counts"],
                self.widths,
            )
        elif self.spatial and self.spatial[0].mode == "relational_attention":
            groups = b["relation"]
        for layer in self.spatial:
            h = layer(h, b["edges"], groups)
        return h

    def forward(self, b):
        h = self.observed(b["unique"])[b["expand"]] if "unique" in b else self.observed(b)
        q = self.asked(self.task(b["task"]), b)
        width = h.shape[-1]
        if self.temporal_mode == "graph_sequence":
            before = h
            h, sequence = self.graph_sequence(h, b)
            frame = sequence[b["row_frames"]]
            if self.context:
                frame = self.frames(
                    frame, sequence[b["end_steps"].clamp_min(0)], b["end_gaps"], b["end_mask"]
                )
            held = self.referred(before, None, b, h)
            features = torch.cat((held[:, :width] + frame, held[:, width:], q), -1)
        elif self.temporal_mode == "coupling_edges":
            h = self.coupling(h, b)
            features = torch.cat((self.referred(h, None, b, h), q), -1)
        else:
            features = torch.cat((self.referred(h, self.entity_sequence(h, b), b, h), q), -1)
        return self.head(features[b["selected"]], b)
