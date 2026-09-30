import logging
import time

import numpy as np
import torch

from graphspec.infra.arrays import NUMPY, Bundle, Pending, join_tensors, views
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)


def array(value, dtype=None):
    result = np.ascontiguousarray(value)
    if (
        dtype is not None
        and result.dtype != NUMPY[dtype]
        and not (dtype == torch.long and result.dtype == np.int32)
    ):
        result = result.astype(NUMPY[dtype])
    return result


class Host:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class Gather:
    __slots__ = ("segments",)

    def __init__(self, segments):
        self.segments = segments


class Features:
    __slots__ = ("segments", "order")

    def __init__(self, segments, order=None):
        self.segments, self.order = segments, order


class Frames:
    __slots__ = ("directed", "ranges", "canonical")

    def __init__(self, directed, ranges, canonical):
        self.directed, self.ranges, self.canonical = directed, ranges, canonical


def gathered(graph, key, index=None):
    return Gather([(graph, key, None if index is None else array(index, torch.long))])


def features(graph, shape, *pieces):
    return Features(
        [
            (
                graph,
                tuple(shape),
                [
                    (key, index if isinstance(index, tuple) else array(index, torch.long), weight)
                    for key, index, weight in pieces
                ],
            )
        ]
    )


def features_at(graph, observations):
    index = array(observations, torch.long)
    return features(graph, index.shape, ("x", index, 1.0))


def length(value):
    return sum(segment[1][0] for segment in value.segments)


def joined(items, axis=0, order=None):
    first = items[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis)
    if isinstance(first, Gather):
        return Gather([segment for item in items for segment in item.segments])
    return Features([segment for item in items for segment in item.segments], order)


def prepare(value, bundle):
    if isinstance(value, np.ndarray):
        return bundle.hold(value)
    if isinstance(value, Host):
        return value
    if isinstance(value, Gather):
        return Gather(
            [
                (graph, key, None if index is None else bundle.hold(index, torch.int32))
                for graph, key, index in value.segments
            ]
        )
    if isinstance(value, Features):
        segments = [
            (
                graph,
                shape,
                [
                    (
                        key,
                        index if isinstance(index, tuple) else bundle.hold(index, torch.int32),
                        bundle.hold(weight) if isinstance(weight, np.ndarray) else weight,
                    )
                    for key, index, weight in pieces
                ],
            )
            for graph, shape, pieces in value.segments
        ]
        return Features(segments, None if value.order is None else bundle.hold(value.order))
    if isinstance(value, Frames):
        return Frames(bundle.hold(value.directed), value.ranges, value.canonical)
    if isinstance(value, dict):
        return {key: prepare(item, bundle) for key, item in value.items()}
    if isinstance(value, list):
        return [prepare(item, bundle) for item in value]
    if isinstance(value, tuple):
        return tuple(prepare(item, bundle) for item in value)
    return value


class Resolver:
    def __init__(self, bank, cfg, tensors, device, graphs):
        self.bank, self.cfg, self.tensors, self.device, self.graphs, self.tables = (
            bank,
            cfg,
            tensors,
            device,
            graphs,
            {},
        )
        self.residents = {}

    def table(self, key):
        if key not in self.tables:
            parts, offsets, lengths, total = [], {}, {}, 0
            for graph in self.graphs:
                if graph not in self.residents:
                    self.residents[graph] = self.bank.resident(graph, self.cfg, self.device)
                tensor = self.residents[graph].tensor(key)
                if tensor.is_cuda:
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
                offsets[graph], lengths[graph], total = total, len(tensor), total + len(tensor)
                parts.append(tensor)
            self.tables[key] = (join_tensors(parts, self.device), offsets, lengths)
        return self.tables[key]

    def rows(self, key, index):
        if key != "x":
            return self.table(key)[0].index_select(0, index)
        references, offsets, _ = self.table("x_ref")
        table, shifts, _ = self.table("x_table")
        if "x_moves" not in self.tables:
            self.tables["x_moves"] = (
                self.numbers([offsets[graph] for graph in self.graphs]),
                self.numbers([shifts[graph] for graph in self.graphs]),
            )
        starts, moves = self.tables["x_moves"]
        owner = torch.bucketize(index, starts, right=True) - 1
        return table.index_select(0, references.index_select(0, index).long() + moves.index_select(0, owner))

    def numbers(self, values):
        return torch.as_tensor(np.asarray(values, np.int64), device=self.device)

    def spread(self, values, lengths, total):
        return torch.repeat_interleave(self.numbers(values), self.numbers(lengths), output_size=total)

    def ranges(self, offsets, spans):
        starts = np.asarray([offsets[graph] + lo for graph, lo, hi in spans], np.int64)
        lengths = np.asarray([hi - lo for graph, lo, hi in spans], np.int64)
        total = int(lengths.sum())
        return self.spread(starts - np.r_[0, np.cumsum(lengths[:-1])], lengths, total) + torch.arange(
            total, device=self.device
        )

    def grouped(self, items, resolve):
        groups = {}
        for position, (signature, _) in enumerate(items):
            groups.setdefault(signature, []).append(position)
        parts = [resolve(signature, positions) for signature, positions in groups.items()]
        if len(parts) == 1:
            return parts[0]
        starts, offset = {}, 0
        for positions in groups.values():
            for position in positions:
                starts[position], offset = offset, offset + items[position][1]
        order = np.concatenate(
            [np.arange(starts[position], starts[position] + rows) for position, (_, rows) in enumerate(items)]
        )
        return torch.cat(parts)[torch.as_tensor(order, device=self.device)]

    def local(self, indices):
        return (
            torch.cat([index.reshape(-1) for index in indices])
            if len(indices) > 1
            else indices[0].reshape(-1)
        )

    def features(self, value):
        segments = value.segments
        items = [
            (
                (
                    tuple(shape[1:]),
                    tuple(
                        (
                            key,
                            isinstance(index, tuple),
                            weight is None,
                            isinstance(weight, Pending),
                            weight if isinstance(weight, (int, float)) else None,
                        )
                        for key, index, weight in pieces
                    ),
                ),
                int(shape[0]),
            )
            for _, shape, pieces in segments
        ]

        def resolve(signature, positions):
            trailing, layout = signature
            chosen = [segments[position] for position in positions]
            rows, out = sum(shape[0] for _, shape, _ in chosen), None
            for p, (key, is_range, unweighted, per_row, _) in enumerate(layout):
                source, offsets, _ = self.table("x_ref" if key == "x" else key)
                if is_range:
                    values = self.rows(
                        key,
                        self.ranges(
                            offsets,
                            [(graph, pieces[p][1][0], pieces[p][1][1]) for graph, _, pieces in chosen],
                        ),
                    ).float()
                elif not len(source):
                    continue
                else:
                    lengths = [int(np.prod(shape)) for _, shape, _ in chosen]
                    local = self.local([self.tensors[pieces[p][1].index] for _, _, pieces in chosen])
                    shift = self.spread([offsets[graph] for graph, _, _ in chosen], lengths, sum(lengths))
                    if unweighted:
                        values = self.rows(key, local + shift).float()
                    else:
                        mask = local >= 0
                        weight = (
                            self.local([self.tensors[pieces[p][2].index] for _, _, pieces in chosen])
                            if per_row
                            else chosen[0][2][p][2]
                        )
                        values = (
                            self.rows(key, torch.where(mask, local + shift, 0)).float()
                            * (mask.float() * weight)[:, None]
                        )
                values = values.reshape((rows,) + trailing + (values.shape[-1],))
                out = values if out is None else out + values
            return (
                torch.zeros((rows,) + trailing + (self.bank.fitted(self.cfg).features,), device=self.device)
                if out is None
                else out
            )

        result = self.grouped(items, resolve)
        return result if value.order is None else result[self.tensors[value.order.index]]

    def gather(self, value):
        segments = value.segments
        items = [
            (
                (key, index is None),
                self.table(key)[2][graph] if index is None else int(self.tensors[index.index].shape[0]),
            )
            for graph, key, index in segments
        ]

        def resolve(signature, positions):
            key, whole = signature
            source, offsets, lengths = self.table(key)
            chosen = [segments[position] for position in positions]
            if whole:
                return source.index_select(
                    0, self.ranges(offsets, [(graph, 0, lengths[graph]) for graph, _, _ in chosen])
                )
            local = [self.tensors[index.index] for _, _, index in chosen]
            shape = (sum(index.shape[0] for index in local),) + tuple(local[0].shape[1:])
            shift = self.spread(
                [offsets[graph] for graph, _, _ in chosen],
                [index.numel() for index in local],
                sum(index.numel() for index in local),
            )
            return source.index_select(0, self.local(local) + shift).reshape(shape + tuple(source.shape[1:]))

        return self.grouped(items, resolve)

    def run(self, value):
        if isinstance(value, Pending):
            return self.tensors[value.index]
        if isinstance(value, Host):
            return value.value
        if isinstance(value, Gather):
            return self.gather(value)
        if isinstance(value, Features):
            return self.features(value)
        if isinstance(value, dict):
            result = {key: self.run(item) for key, item in value.items() if not isinstance(item, Frames)}
            for key, item in value.items():
                if isinstance(item, Frames):
                    endpoints = result["entity"][result["edges"]]
                    if item.canonical:
                        endpoints = endpoints[:, self.tensors[item.directed.index]]
                    result[key] = [
                        torch.cat([endpoints[:, lo:hi] for lo, hi in step], 1)
                        if len(step) > 1
                        else endpoints[:, step[0][0] : step[0][1]]
                        for step in item.ranges
                    ]
            return result
        if isinstance(value, list):
            return [self.run(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.run(item) for item in value)
        return value


def keys(value, found):
    if isinstance(value, Gather):
        found.update(key for _, key, _ in value.segments)
    elif isinstance(value, Features):
        found.update(
            name
            for _, _, pieces in value.segments
            for key, _, _ in pieces
            for name in (("x_ref", "x_table") if key == "x" else (key,))
        )
    elif isinstance(value, dict):
        for item in value.values():
            keys(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys(item, found)
    return found


@timed("resolve")
def resolve(value, bank, cfg, tensors, device):
    return Resolver(bank, cfg, tensors, device, list(dict.fromkeys(value["graphs"]))).run(value)


def materialise(bank, cfg, tree, device):
    started = time.perf_counter()
    bundle = Bundle(device)
    prepared = prepare(tree, bundle)
    buffer, spans = bundle.pack()
    result = resolve(prepared, bank, cfg, views(torch.as_tensor(buffer, device=device), spans), device)
    LOG.debug(
        "materialise exit arrays=%d bytes=%d device=%s seconds=%.4f",
        len(spans),
        len(buffer),
        device,
        time.perf_counter() - started,
    )
    return result
