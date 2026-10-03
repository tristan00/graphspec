import functools
import logging
import time

import numpy as np

from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.infra.workspace import allocate, block_rows, gathered
from graphspec.views import spans
from graphspec.views.graph import DENSE, Dense, verify_dense

LOG = logging.getLogger(__name__)
CONNECT = "graphs/connect"


@timed("join.parameters")
def parameters(spec):
    return {name: value for name, value in spec.items() if name.startswith("graphs/")}


def connecting(kinds, tokens, registry, spec, seed):
    started = time.perf_counter()
    connect = np.asarray([spec["graphs/connect/" + kind] for kind in registry.declared], bool)
    tokens = np.asarray(tokens).astype(np.uint64)
    result = connect[np.asarray(kinds, np.int64)] & (unit(seed, CONNECT, tokens) < spec["graphs/connect_fraction"])
    LOG.debug("connecting exit entities=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def apart(kinds, tokens):
    started = time.perf_counter()
    result = np.zeros(len(kinds), bool)
    LOG.debug("apart exit entities=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def unify(graphs, registry, spec, seed):
    started = time.perf_counter()
    kinds = np.concatenate([np.asarray(graph.kinds) for graph in graphs]).astype(np.int64)
    tokens = np.concatenate([np.asarray(graph.entity_token) for graph in graphs]).astype(np.uint64)
    connected = connecting(kinds, tokens, registry, spec, seed)
    chosen = np.flatnonzero(connected)
    group, count = spans.identity(np.stack((kinds[chosen], tokens[chosen].view(np.int64)), 1))
    first = np.full(count, len(kinds), np.int64)
    np.minimum.at(first, group, chosen)
    creator = ~connected
    creator[first] = True
    rank = np.cumsum(creator) - 1
    ids = rank.copy()
    ids[chosen] = rank[first[group]]
    offsets = np.r_[0, np.cumsum([len(graph.kinds) for graph in graphs])]
    LOG.info(
        "unify exit seconds=%.3f graphs=%d entities=%d merged=%d connected=%d",
        time.perf_counter() - started,
        len(graphs),
        len(kinds),
        int(creator.sum()),
        count,
    )
    return (
        [ids[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:])],
        kinds[creator],
        tokens[creator],
        connected[creator],
        functools.partial(connecting, registry=registry, spec=spec, seed=seed),
    )


def joined(graphs, registry, spec, seed):
    started = time.perf_counter()
    graphs = tuple(graphs)
    if not graphs:
        raise ValueError("joining requires at least one graph")
    maps, kinds, tokens, connected, _ = unify(graphs, registry, spec, seed)
    step_offsets = np.r_[0, np.cumsum([graph.steps for graph in graphs])]
    task_offsets = np.r_[0, np.cumsum([len(graph.task_key) for graph in graphs])]
    period_offsets = np.r_[0, np.cumsum([int(graph.period.max(initial=-1)) + 1 for graph in graphs])]
    entities = {"obs_entity", "fact_source", "fact_target", "member_source", "member_target"}
    moments = {"obs_step", "fact_step", "task_step"}
    tasks = {"member_task", "label_task"}
    references = dict(obs_ref="x_table", fact_ref="value_table")
    tables = {
        table: np.r_[0, np.cumsum([len(graph.arrays[table]) for graph in graphs])]
        for table in references.values()
    }
    arrays = {}
    for name in DENSE:
        if name in ("kinds", "entity_token", "entity_connected", "step_graph"):
            continue
        example = graphs[0].arrays[name]
        column = allocate(
            (sum(len(graph.arrays[name]) for graph in graphs),) + example.shape[1:], example.dtype
        )
        offset = 0
        for member, (graph, table, step_offset, period_offset) in enumerate(
            zip(graphs, maps, step_offsets, period_offsets)
        ):
            values = graph.arrays[name]
            width = block_rows(values)
            for start in range(0, len(values), width):
                part = values[start : start + width]
                if name in entities:
                    part = gathered(table, part, -1)
                elif name in moments:
                    part = part + step_offset
                elif name in tasks:
                    part = part + task_offsets[member]
                elif name == "period":
                    part = part + period_offset
                elif name in references:
                    part = np.where(part >= 0, part + tables[references[name]][member], -1)
                column[offset : offset + len(part)] = part
                offset += len(part)
        arrays[name] = column
    arrays["kinds"] = kinds.astype(np.int32)
    arrays["entity_token"] = tokens
    arrays["entity_connected"] = connected
    arrays["step_graph"] = np.repeat(
        np.arange(len(graphs), dtype=np.int32), [graph.steps for graph in graphs]
    )
    result = Dense("joined", [], int(step_offsets[-1]), **arrays)
    verify_dense(result, registry)
    LOG.info(
        "join exit seconds=%.3f graphs=%d entities=%d shared=%d steps=%d facts=%d",
        time.perf_counter() - started,
        len(graphs),
        len(kinds),
        int(connected.sum()),
        result.steps,
        len(result.fact_step),
    )
    return result


def layout(graphs):
    started = time.perf_counter()
    offsets = np.r_[0, np.cumsum([steps for steps, _, _ in graphs])]
    rows = np.concatenate([row_index for _, row_index, _ in graphs])
    steps = np.concatenate(
        [(row_steps + offset).astype(np.int32) for (_, _, row_steps), offset in zip(graphs, offsets)]
    )
    LOG.debug(
        "layout exit graphs=%d rows=%d seconds=%.4f", len(graphs), len(rows), time.perf_counter() - started
    )
    return rows, steps
