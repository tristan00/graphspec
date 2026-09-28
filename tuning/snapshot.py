import torch
from torch import nn

from .layers import (
    CausalAttention,
    EvolvingLayer,
    GatedTCN,
    GraphCell,
    Groups,
    Head,
    Spatial,
    TemporalEdges,
    keys_of,
    mean,
    normalization,
)
from .turns import EndContext, enabled, gather_end


class SnapshotNet(nn.Module):
    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg["model"]
        self.cfg, self.temporal_mode = cfg, spec["snapshot/temporal"]
        hidden = spec["hidden"]
        coupling = self.temporal_mode == "coupling_edges"
        if coupling:
            self.turn_edge_time = nn.Linear(1, hidden, bias=False)
        elif enabled(cfg):
            self.turns = EndContext(hidden * 3)
        self.input = nn.Sequential(nn.Linear(registry.features, hidden), normalization(hidden), nn.SiLU())
        self.kind = nn.Embedding(len(registry.nodes), hidden)
        self.query = nn.Linear(registry.features, hidden)
        self.keys = keys_of(spec, hidden)
        attributes = {name: len(registry.attributes(name)) for name in registry.attributed}
        self.widths = list(attributes.values())
        self.spatial = nn.ModuleList()
        if spec["snapshot/spatial"] != "none":
            self.spatial.extend(
                Spatial(
                    hidden,
                    spec["snapshot/spatial"],
                    registry.relations,
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
            self.final_spatial = Spatial(hidden, "mean", registry.relations, {}, spec["dropout"])
        self.reward = Head(hidden * 4, spec, registry)

    def graph_sequence(self, h, b):
        decision = b["decision_context"]
        known = decision >= 0
        index = torch.where(known, decision, len(h))
        padded = torch.cat((h, h.new_zeros(1, h.shape[-1])))
        pooled = mean(h, b["step"], b["frames"])
        action = self.action_round(
            torch.cat((self.query(b["decision_query"]), padded[index[:, 0]], padded[index[:, 1]], pooled), -1)
        )
        feedback = h.new_zeros(h.shape).index_add_(
            0, decision[known[:, 0], 0], self.feedback(action[known[:, 0]])
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
        h = h + mean(self.turn_edge_time(b["temporal_turn_gap"].log1p()[:, None]), edges[1], len(h))
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
        q = self.query(b["query"])
        width = h.shape[-1]
        if self.temporal_mode == "graph_sequence":
            actor, subject = h[b["context_index"][:, 0]], h[b["context_index"][:, 1]]
            h, sequence = self.graph_sequence(h, b)
            features = torch.cat((sequence[b["query_frames"]], actor, subject, q), -1)
        elif self.temporal_mode == "coupling_edges":
            h = self.coupling(h, b)
            features = torch.cat(
                (h[b["context_index"][:, 0]], h[b["context_index"][:, 1]], h[b["context_index"][:, 2]], q), -1
            )
            return self.reward(features[b["selected"]], b)
        else:
            sequence = self.entity_sequence(h, b)
            index = b["query_steps"]
            context = b["entity"][b["context_index"][:, 2]]
            features = torch.cat(
                (sequence[index, b["actor"]], sequence[index, b["subject"]], sequence[index, context], q), -1
            )
        if enabled(self.cfg):
            if self.temporal_mode == "graph_sequence":
                historical = sequence[b["end_steps"].clamp_min(0)]
                known = b["end_observation_local"][:, :, :2]
                padded = torch.cat((h, h.new_zeros(1, width)))
                past = padded[torch.where(known >= 0, known, len(h))]
                raw = self.input(b["end_raw"][:, :2]).permute(0, 2, 1, 3)
                past = torch.where((known >= 0)[..., None], past, raw)
                past = past * b["end_raw_mask"][:, :2].transpose(1, 2)[..., None]
                previous = torch.cat((historical, past.reshape(b["predictions"], -1, width * 2)), -1)
            else:
                previous = torch.cat(
                    [gather_end(sequence, ids, b) for ids in (b["actor"], b["subject"], context)], -1
                )
                raw = self.input(b["end_raw"]).permute(0, 2, 1, 3).reshape_as(previous)
                known = (b["end_observation_local"] >= 0).repeat_interleave(width, -1)
                previous = torch.where(known, previous, raw)
                previous = previous * b["end_raw_mask"].transpose(1, 2).repeat_interleave(width, -1)
            first = self.turns(features[:, : 3 * width], previous, b["end_gaps"], b["end_mask"])
            features = torch.cat((first, q), -1)
        return self.reward(features[b["selected"]], b)
