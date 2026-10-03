import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graphspec.infra.timing import timed
from graphspec.schema.references import SORTS
from graphspec.schema.targets import TARGETS

from .heads import FIELD, HEADS

CHUNK = 128


@timed("layers.normalization")
def normalization(width):
    if type(width) is not int or width < 1:
        raise ValueError("normalization width must be a positive integer")
    return nn.Identity() if width == 1 else nn.LayerNorm(width)


def mean(values, index, size):
    result = values.new_zeros((size, values.shape[-1])).index_add_(0, index, values)
    count = values.new_zeros(size).index_add_(0, index, values.new_ones(len(index)))
    return result / count.clamp_min(1)[:, None]


def segment_softmax(scores, index, size):
    shape = (size,) + scores.shape[1:]
    expanded = index.reshape((-1,) + (1,) * (scores.ndim - 1)).expand_as(scores)
    maxima = scores.new_full(shape, -torch.inf).scatter_reduce_(
        0, expanded, scores.detach(), reduce="amax", include_self=True
    )
    weights = (scores - maxima[index]).exp()
    total = weights.new_zeros(shape).index_add_(0, index, weights)
    return weights / total[index].clamp_min(1e-8)


def padded(values, index):
    return torch.cat((values, values.new_zeros((1,) + values.shape[1:])))[
        torch.where(index >= 0, index, len(values))
    ]


def masked_mean(x, mask, dim):
    return (x * mask[..., None]).sum(dim) / mask.sum(dim).clamp_min(1)[..., None]


def masked_weights(score, mask):
    score = score.masked_fill(~mask, -1e4)
    weight = score.softmax(-1) * mask
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)


class Segments:
    __slots__ = ("sources", "targets", "kinds", "length")

    def __init__(self, sources, targets, kinds, length):
        self.sources, self.targets, self.kinds, self.length = sources, targets, kinds, length


def segments(order, counts, source, destination, size, labels, length=CHUNK):
    counts = np.asarray(counts, np.int64)
    starts = np.cumsum(counts) - counts
    pieces = -(-counts // length)
    total = int(pieces.sum())
    kind = np.repeat(np.arange(len(counts)), pieces)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    start = np.repeat(starts, pieces) + (np.arange(total) - first) * length
    stop = np.repeat(starts + counts, pieces)
    device = order.device
    position = torch.as_tensor(start, device=device)[:, None] + torch.arange(length, device=device)
    valid = position < torch.as_tensor(stop, device=device)[:, None]
    row = order[position.clamp_max(max(len(order) - 1, 0))]
    sources = torch.where(valid, source[row], 0).reshape(-1)
    targets = torch.where(valid, destination[row], size).reshape(-1)
    return Segments(sources, targets, labels[torch.as_tensor(kind, device=device)], length)


def segment_matmul(x, weights, seg):
    gathered = x.index_select(0, seg.sources).view(-1, seg.length, x.shape[1])
    return torch.bmm(gathered, weights.index_select(0, seg.kinds)).reshape(-1, weights.shape[2])


def attribute_inputs(values, counts, widths, device):
    rows, kinds = int(sum(counts)), len(counts)
    counts, widths = np.asarray(counts, np.int64), np.asarray(widths, np.int64)
    kind = torch.repeat_interleave(
        torch.arange(kinds, device=device), torch.as_tensor(counts, device=device), output_size=rows
    )
    width = torch.as_tensor(widths, device=device)[kind]
    within = (
        torch.arange(rows, device=device) - torch.as_tensor(np.cumsum(counts) - counts, device=device)[kind]
    )
    base = torch.as_tensor(np.cumsum(counts * widths) - counts * widths, device=device)[kind] + within * width
    column = torch.arange(int(widths.max()), device=device)
    padded = torch.where(
        column[None] < width[:, None],
        values[(base[:, None] + column).clamp_max(max(len(values) - 1, 0))],
        0.0,
    )
    return padded


class Groups:
    def __init__(
        self, edges, relation, grouped, size, values=None, attribute_edges=None, counts=None, widths=None
    ):
        source, destination = edges
        self.relation, group = relation, grouped[relation]
        order = torch.argsort(group)
        kinds, held = torch.unique_consecutive(group[order], return_counts=True)
        self.relational = segments(order, held.tolist(), source, destination, size, kinds)
        self.degree = (
            torch.zeros(size, device=source.device)
            .index_add_(0, destination, torch.ones(len(destination), device=source.device))
            .clamp_min_(1)
        )
        self.attributes = self.inputs = None
        if counts and sum(counts):
            rows = torch.arange(len(attribute_edges), device=source.device)
            self.attributes = segments(
                rows,
                counts,
                rows,
                destination[attribute_edges],
                size,
                torch.arange(len(counts), device=source.device),
            )
            self.inputs = attribute_inputs(values, counts, widths, source.device)


def aggregate(h, edges, mode="mean"):
    if mode not in ("mean", "normalised"):
        raise ValueError("unknown aggregation mode: " + str(mode))
    source, destination = edges
    if mode == "normalised":
        degree = h.new_ones(len(h)).index_add_(0, destination, h.new_ones(len(destination)))
        scale = degree.rsqrt()
        result = h * scale[:, None].square()
        return result.index_add(0, destination, h[source] * (scale[source] * scale[destination])[:, None])
    return mean(h[source], destination, len(h))


class TimeEncoding(nn.Module):
    @timed("layers.timeencoding")
    def __init__(self, width, mode):
        super().__init__()
        if mode not in ("log_cosine", "cosine", "log_linear"):
            raise ValueError("unknown time encoding: " + str(mode))
        self.mode = mode
        if mode == "log_linear":
            self.linear = nn.Linear(1, width)
        else:
            self.frequency = nn.Parameter(torch.logspace(0, -3 if mode == "log_cosine" else -4, width))
            self.phase = nn.Parameter(torch.zeros(width))

    def forward(self, gap):
        gap = gap.clamp_min(0)
        if self.mode == "log_linear":
            return self.linear(torch.log1p(gap)[..., None])
        return torch.cos(
            (torch.log1p(gap) if self.mode == "log_cosine" else gap)[..., None] * self.frequency + self.phase
        )


class References(nn.Module):
    @timed("layers.references")
    def __init__(self, registry, width, output):
        super().__init__()
        self.names = len(registry.references)
        stated = registry.schema.tasks["references"].values()
        ordered = [SORTS.named(entry["sort"]).ordered for entry in stated]
        self.register_buffer("ordered", torch.as_tensor(ordered, dtype=torch.float32), persistent=False)
        self.place = nn.Linear(1, width, bias=False)
        bound = 1 / math.sqrt(width)
        self.project = nn.Parameter(torch.empty(self.names, width, output).uniform_(-bound, bound))

    def forward(self, b, members, whole):
        name, place, row = b["member_name"], b["member_place"], b["member_row"]
        referred = name < self.names
        members, name, place, row = members[referred], name[referred], place[referred], row[referred]
        place = self.place(torch.log1p(place.float())[:, None])
        members = members + place * self.ordered[name][:, None]
        keys = torch.cat((row * self.names + name, b["whole_row"] * self.names + b["whole_name"]))
        pooled = mean(torch.cat((members, whole)), keys, b["predictions"] * self.names)
        pooled = pooled.reshape(b["predictions"], self.names, pooled.shape[-1])
        return torch.einsum("rnw,nwo->ro", pooled, self.project)


class Relations(nn.Module):
    @timed("layers.relations")
    def __init__(self, registry, hidden):
        super().__init__()
        self.embedding = nn.Embedding(registry.relation_slots, hidden)

    def forward(self, code):
        return self.embedding(code.clamp_min(0)) * (code >= 0)[..., None]


def related(events):
    return torch.where(events[:, 3] == 1, events[:, 6], -1)


class Asked(nn.Module):
    @timed("layers.asked")
    def __init__(self, registry, hidden):
        super().__init__()
        self.kind = nn.Embedding(len(registry.tasks), hidden)

    def forward(self, values, b):
        return values + self.kind(b["row_kind"])


class NumberHead(nn.Module):
    @timed("layers.numberhead")
    def __init__(self, hidden):
        super().__init__()
        self.output = nn.Linear(hidden * 2, 1)

    def forward(self, h, series, b, points):
        return self.output(torch.cat((h, series), -1)).squeeze(-1)


class ScoreHead(nn.Module):
    @timed("layers.scorehead")
    def __init__(self, hidden):
        super().__init__()
        self.task = nn.Linear(hidden, hidden)
        self.bias = nn.Linear(hidden, 1)

    def forward(self, h, series, b, points):
        return (self.task(h) * series).sum(-1) + self.bias(series).squeeze(-1)


class NodeHead(nn.Module):
    @timed("layers.nodehead")
    def __init__(self, hidden):
        super().__init__()
        self.query = nn.Linear(hidden * 2, hidden)
        self.bias = nn.Linear(hidden, 1)

    def forward(self, h, series, b, points):
        task, node = h.chunk(2, -1)
        return (self.query(torch.cat((task, series), -1)) * node).sum(-1) + self.bias(node).squeeze(-1)


class SequenceHead(nn.Module):
    @timed("layers.sequencehead")
    def __init__(self, hidden, longest):
        super().__init__()
        self.place = nn.Embedding(longest, hidden)
        self.task = nn.Linear(hidden * 2, hidden)
        self.bias = nn.Linear(hidden * 2, 1)

    def forward(self, h, series, b, points):
        place = self.place(b[FIELD + "place"][points].long())
        return (self.task(torch.cat((h, place), -1)) * series).sum(-1) + self.bias(
            torch.cat((series, place), -1)
        ).squeeze(-1)


class PathHead(nn.Module):
    @timed("layers.pathhead")
    def __init__(self, hidden, encoding):
        super().__init__()
        self.time = TimeEncoding(hidden, encoding)
        self.output = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h, series, b, points):
        return self.output(torch.cat((h, self.time(b[FIELD + "tau"][points]), series), -1)).squeeze(-1)


class Head(nn.Module):
    @timed("layers.head")
    def __init__(self, width, spec, registry, nodes):
        super().__init__()
        hidden = spec["hidden"]
        self.targets = registry.schema.targets()
        self.task = nn.Sequential(nn.Linear(width, hidden), nn.SiLU())
        self.series = nn.Embedding(len(registry.schema.series()), hidden)
        from graphspec.schema.decisions import Decision

        from .options import Options, realised

        self.kinds = nn.ModuleList(HEADS.named(kind).module(hidden, spec, registry) for kind in self.targets)
        self.named = [TARGETS.named(kind).names is not None for kind in self.targets]
        self.node = nn.Sequential(nn.Linear(nodes, hidden), nn.SiLU()) if any(self.named) else None
        name = realised(spec)
        self.options = None if name is None else Options(name, registry, hidden, nodes, spec)
        self.decides = [self.options is not None and isinstance(TARGETS.named(kind), Decision) for kind in self.targets]

    def forward(self, features, b, members):
        h, series = self.task(features)[b["point_row"]], self.series(b["series"])
        node = None if self.node is None else padded(self.node(members), b["point_node"])
        content = None if self.options is None else self.options(b, members)
        if len(self.kinds) == 1:
            held = [part for part in (node, content) if part is not None]
            given = torch.cat([h] + held, -1) if held else h
            return self.kinds[0](given, series, b, torch.arange(len(h), device=h.device))
        outputs, places = [], []
        for code, kind in enumerate(self.kinds):
            points = (b["aim"] == code).nonzero().flatten()
            given = torch.cat((h[points], node[points]), -1) if self.named[code] else h[points]
            if self.decides[code]:
                given = torch.cat((given, content[points]), -1)
            outputs.append(kind(given, series[points], b, points))
            places.append(points)
        values = torch.cat(outputs)
        return values.new_zeros(len(h)).scatter(0, torch.cat(places), values)


class KeyEmbedding(nn.Module):
    @timed("layers.keyembedding")
    def __init__(self, hidden, table):
        super().__init__()
        self.table = table
        self.embedding = nn.Embedding(table, hidden)
        nn.init.normal_(self.embedding.weight, std=hidden**-0.5)

    def forward(self, token):
        return self.embedding(token.clamp_min(0) % self.table) * (token >= 0)[..., None]


@timed("layers.keys_of")
def keys_of(spec, hidden):
    return KeyEmbedding(hidden, spec["keys/table"]) if spec.get("keys/embedding") else None


def chebyshev(h, edges, weights):
    source, destination = edges
    scale = h.new_zeros(len(h)).index_add_(0, destination, h.new_ones(len(destination))).clamp_min_(1).rsqrt()
    factor = (scale[source] * scale[destination])[:, None]

    def laplacian(x):
        return -x.new_zeros(x.shape).index_add_(0, destination, x[source] * factor)

    previous, current = h, laplacian(h)
    result = h @ weights[0] + current @ weights[1]
    for order in range(2, len(weights)):
        previous, current = current, 2 * laplacian(current) - previous
        result = result + current @ weights[order]
    return result


class Spatial(nn.Module):
    @timed("layers.spatial")
    def __init__(self, hidden, mode, relations, attributes, dropout=0.0, order=2, slots=0):
        super().__init__()
        if mode not in ("mean", "normalised", "relational", "chebyshev", "attention", "relational_attention"):
            raise ValueError("unknown spatial mode: " + str(mode))
        self.mode, self.dropout = mode, dropout
        self.root = nn.Linear(hidden, hidden)
        if mode not in ("relational", "chebyshev"):
            self.message = nn.Linear(hidden, hidden, bias=False)
        self.norm = normalization(hidden)
        if mode == "relational":
            self.weights = nn.Parameter(torch.empty(relations, hidden, hidden))
            nn.init.xavier_uniform_(self.weights)
            if attributes:
                widths = list(attributes.values())
                self.attribute_weights = nn.Parameter(torch.zeros(len(widths), max(widths), hidden))
                for slot, width in enumerate(widths):
                    if width:
                        self.attribute_weights.data[slot, :width] = nn.init.kaiming_uniform_(
                            torch.empty(hidden, width), a=math.sqrt(5)
                        ).T
        elif mode == "chebyshev":
            self.weights = nn.Parameter(torch.empty(order + 1, hidden, hidden))
            nn.init.xavier_uniform_(self.weights)
        elif mode in ("attention", "relational_attention"):
            self.query, self.key = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        if mode in ("relational", "relational_attention"):
            self.relation = nn.Embedding(slots, hidden)

    def forward(self, h, edges, groups=None):
        source, destination = edges
        if self.mode == "relational":
            update = h.new_zeros((len(h) + 1, h.shape[1]))
            update.index_add_(
                0, groups.relational.targets, segment_matmul(h, self.weights, groups.relational)
            )
            update.index_add_(0, destination, self.relation(groups.relation))
            if groups.attributes is not None:
                update.index_add_(
                    0,
                    groups.attributes.targets,
                    segment_matmul(groups.inputs, self.attribute_weights, groups.attributes),
                )
            update = update[: len(h)] / groups.degree[:, None]
        elif self.mode == "chebyshev":
            update = chebyshev(h, edges, self.weights)
        elif self.mode == "attention":
            score = (self.query(h)[destination] * self.key(h)[source]).sum(-1) / math.sqrt(h.shape[-1])
            weight = segment_softmax(score, destination, len(h))
            update = h.new_zeros(h.shape).index_add_(
                0, destination, self.message(h)[source] * weight[:, None]
            )
        elif self.mode == "relational_attention":
            relation = self.relation(groups)
            score = (self.query(h)[destination] * (self.key(h)[source] + relation)).sum(-1) / math.sqrt(
                h.shape[-1]
            )
            weight = segment_softmax(score, destination, len(h))
            update = h.new_zeros(h.shape).index_add_(
                0, destination, (self.message(h)[source] + relation) * weight[:, None]
            )
        else:
            update = self.message(aggregate(h, edges, self.mode))
        return self.norm(h + F.dropout(F.silu(self.root(h) + update), self.dropout, self.training))


class CausalAttention(nn.Module):
    @timed("layers.causalattention")
    def __init__(self, hidden, heads=2, dropout=0.0):
        super().__init__()
        if type(hidden) is not int or hidden < 1 or type(heads) is not int or heads < 1:
            raise ValueError("attention hidden width and head count must be positive integers")
        self.heads, self.dropout = heads, dropout
        self.head_width = (hidden + heads - 1) // heads
        self.attention_width = heads * self.head_width
        self.qkv = nn.Linear(hidden, self.attention_width * 3)
        self.output = nn.Linear(self.attention_width, hidden)
        self.norm1, self.norm2 = normalization(hidden), normalization(hidden)
        self.feed = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x, mask, causal=True):
        n, t, h = x.shape
        positions = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, h, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / h))
        position = torch.stack(((positions * frequency).sin(), (positions * frequency).cos()), -1).reshape(
            t, -1
        )[:, :h]
        z = (x + position) * mask[:, :, None]
        q, k, v = self.qkv(z).reshape(n, t, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        allowed = mask[:, None, :].expand(-1, t, -1)
        if causal:
            allowed = allowed & torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        allowed = allowed | (~mask[:, :, None] & torch.eye(t, device=x.device, dtype=torch.bool))
        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None], dropout_p=self.dropout if self.training else 0.0
        )
        z = z.transpose(1, 2).reshape(n, t, self.attention_width)
        z = self.norm1(x + self.output(z))
        return self.norm2(z + self.feed(z)) * mask[:, :, None]


class GatedTCN(nn.Module):
    @timed("layers.gatedtcn")
    def __init__(self, hidden, depth):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden * 2, 3, dilation=2**i) for i in range(depth))
        self.norms = nn.ModuleList(normalization(hidden) for _ in range(depth))

    def forward(self, x, mask):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            gate, value = conv(F.pad(x.transpose(1, 2), (2 * 2**i, 0))).chunk(2, 1)
            x = norm(x + (gate.sigmoid() * value.tanh()).transpose(1, 2)) * mask[:, :, None]
        return x


class GraphCell(nn.Module):
    @timed("layers.graphcell")
    def __init__(self, hidden, hops):
        super().__init__()
        self.hops = hops
        factor = 1 + 2 * hops
        self.gates = nn.Linear(2 * hidden * factor, 2 * hidden)
        self.candidate = nn.Linear(2 * hidden * factor, hidden)
        nn.init.ones_(self.gates.bias)

    def support(self, x, edges, directed):
        if not self.hops:
            return aggregate(x, edges, "normalised")
        values = [x]
        for support in (directed, directed.flip(0)):
            current = x
            for _ in range(self.hops):
                current = aggregate(current, support)
                values.append(current)
        return torch.cat(values, -1)

    def forward(self, x, previous, edges, directed):
        reset, keep = (
            self.gates(self.support(torch.cat((x, previous), -1), edges, directed)).sigmoid().chunk(2, -1)
        )
        candidate = self.candidate(self.support(torch.cat((x, reset * previous), -1), edges, directed)).tanh()
        return keep * previous + (1 - keep) * candidate


class MatrixGate(nn.Module):
    @timed("layers.matrixgate")
    def __init__(self, hidden):
        super().__init__()
        self.input, self.recurrent = (
            nn.Parameter(torch.empty(hidden, hidden)),
            nn.Parameter(torch.empty(hidden, hidden)),
        )
        self.bias = nn.Parameter(torch.zeros(hidden, hidden))
        nn.init.xavier_uniform_(self.input)
        nn.init.orthogonal_(self.recurrent)

    def forward(self, x, state):
        return self.input @ x + self.recurrent @ state + self.bias


class EvolvingLayer(nn.Module):
    @timed("layers.evolvinglayer")
    def __init__(self, hidden):
        super().__init__()
        self.initial = nn.Parameter(torch.empty(hidden, hidden))
        self.scorer = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        nn.init.xavier_uniform_(self.initial)
        self.reset, self.keep, self.candidate = (MatrixGate(hidden) for _ in range(3))
        self.norm = normalization(hidden)

    def forward(self, x, state, edges, present, entity_batch, entity_sizes):
        scores = x @ F.normalize(self.scorer, dim=0)
        padded_x = nn.utils.rnn.pad_sequence(x.split(entity_sizes), batch_first=True)
        padded_scores = nn.utils.rnn.pad_sequence(
            scores.masked_fill(~present, -torch.inf).split(entity_sizes),
            batch_first=True,
            padding_value=-torch.inf,
        )
        values, indices = padded_scores.topk(min(padded_scores.shape[1], x.shape[-1]), dim=1)
        visible = padded_x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))
        summary = F.pad(
            (visible * torch.where(values.isfinite(), values.tanh(), 0)[:, :, None]).transpose(1, 2),
            (0, x.shape[-1] - indices.shape[1]),
        )
        reset, keep = self.reset(summary, state).sigmoid(), self.keep(summary, state).sigmoid()
        state = keep * state + (1 - keep) * self.candidate(summary, reset * state).tanh()
        update = torch.bmm(aggregate(x, edges, "normalised")[:, None], state[entity_batch]).squeeze(1)
        return self.norm(x + F.silu(update)) * present[:, None], state


class TemporalEdges(nn.Module):
    @timed("layers.temporaledges")
    def __init__(self, hidden, attention, relation_slots, dropout, encoding):
        super().__init__()
        self.attention, self.dropout = attention, dropout
        self.time = TimeEncoding(hidden, encoding)
        self.relation = nn.Embedding(relation_slots, hidden)
        self.value = nn.Linear(hidden, hidden)
        if attention:
            self.score = nn.Linear(hidden * 3, 1)
        else:
            self.gate = nn.Linear(hidden, hidden)
        self.norm = normalization(hidden)

    def forward(self, x, edges, relation, gap):
        source, destination = edges
        temporal = self.time(gap) + self.relation(relation)
        message = self.value(x)[source] + temporal
        if self.attention:
            score = self.score(torch.cat((x[source], x[destination], temporal), -1)).squeeze(-1)
            weight = segment_softmax(score, destination, len(x))
            update = x.new_zeros(x.shape).index_add_(0, destination, message * weight[:, None])
        else:
            update = mean(message * self.gate(temporal).sigmoid(), destination, len(x))
        return self.norm(x + F.dropout(F.silu(update), self.dropout, self.training))


class SelectiveSSM(nn.Module):
    @timed("layers.selectivessm")
    def __init__(self, hidden, state_size=4):
        super().__init__()
        self.norm = normalization(hidden)
        self.in_proj = nn.Linear(hidden, hidden * 2)
        self.conv = nn.Conv1d(hidden, hidden, 3, groups=hidden)
        self.delta = nn.Linear(hidden, hidden)
        self.b, self.c = nn.Linear(hidden, state_size), nn.Linear(hidden, state_size)
        self.log_a = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(hidden, 1))
        self.skip = nn.Parameter(torch.ones(hidden))
        self.output = nn.Linear(hidden, hidden)

    def forward(self, sequence, mask):
        x, gate = self.in_proj(self.norm(sequence)).chunk(2, -1)
        x = F.silu(self.conv(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2))
        delta = F.softplus(self.delta(x)).clamp_max(10)
        b, c = self.b(x), self.c(x)
        a = -self.log_a.exp()
        state = x.new_zeros((len(x), x.shape[-1], a.shape[-1]))
        output = []
        for t in range(x.shape[1]):
            exponent = delta[:, t, :, None] * a
            transition = exponent.exp()
            injection = (transition - 1) / a * b[:, t, None, :]
            updated = transition * state + injection * x[:, t, :, None]
            state = torch.where(mask[:, t, None, None], updated, state)
            output.append((state * c[:, t, None, :]).sum(-1) + self.skip * x[:, t])
        y = torch.stack(output, 1) * F.silu(gate)
        return (sequence + self.output(y)) * mask[:, :, None]
