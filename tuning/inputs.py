import logging
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from graphspec import Registry
from graphspec.infra.arrays import Resident, between
from graphspec.transform import SEED

from . import gather
from .tensors import Frames, Host, array, features, features_at, gathered, joined

LOG = logging.getLogger(__name__)
NEIGHBOURS = 8
ORDERS = dict(recent=0, uniform=1, decay=2)


def sampling(spec):
    order = spec["neighbors/order"]
    return ORDERS[order], spec["neighbors/decay"] if order == "decay" else 0.0


def tree(graph, held, entities, cutoffs, floor, count, depth, registry, spec, stream, tokens=None):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    events, ticks = held.numpy("events"), held.column("events", 2)
    ids, offsets = held.numpy("neighbor_ids"), held.numpy("neighbor_offsets")
    observation_ids, observation_offsets, step = (
        held.numpy("observation_ids"),
        held.numpy("observation_offsets"),
        held.numpy("step"),
    )
    levels = []
    for _ in range(depth):
        table, sizes = gather.neighbor_table(
            ids, offsets, ticks, entities, cutoffs, floor, count, mode, decay, stream, NEIGHBOURS
        )
        children, times, mask, gaps, relation = gather.tree_level(
            events, table, sizes, entities, cutoffs, count, registry.marker("non_relation")
        )
        levels.append((entities, cutoffs, mask, gaps, relation))
        entities, cutoffs = children, times

    def node(members, moments):
        observations = gather.latest(observation_ids, observation_offsets, step, members, moments)
        result = dict(x=features_at(graph, observations), kind=gathered(graph, "entity_kind", members))
        if tokens is not None:
            result["token"] = array(tokens[members])
        return result

    result = node(entities, cutoffs)
    for entities, cutoffs, mask, gaps, relation in reversed(levels):
        result = dict(
            node(entities, cutoffs), child=result, mask=array(mask), gap=array(gaps), relation=array(relation)
        )
    LOG.debug(
        "tree exit roots=%d depth=%d count=%d seconds=%.4f",
        len(levels[0][0]) if levels else len(entities),
        depth,
        count,
        time.perf_counter() - started,
    )
    return result


def history_inputs(graph, held, first, last, start, count, spec, stream, include_structure, selected):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    chosen = np.ones(last - first, bool) if selected is None else np.asarray(selected, bool)
    (
        observed,
        interaction,
        query_tick,
        mask,
        gap,
        ids,
        structural,
        common_ids,
        common_frequency,
        common_stats,
        pair_gaps,
        pair_mask,
    ) = gather.history(
        held.numpy("events"),
        held.numpy("neighbor_ids"),
        held.numpy("neighbor_offsets"),
        held.column("events", 2),
        held.numpy("actor")[first:last].astype(np.int64),
        held.numpy("subject")[first:last].astype(np.int64),
        held.numpy("query_step")[first:last].astype(np.int64),
        start,
        count,
        mode,
        decay,
        stream,
        NEIGHBOURS,
        include_structure,
        chosen,
    )
    occurrence = np.zeros((last - first, 2, count, 2), np.float32)
    for side in range(2):
        for other in range(2):
            occurrence[:, side, :, other] = (
                (ids[:, side, :, None] == ids[:, other, None, :])
                & mask[:, side, :, None]
                & mask[:, other, None, :]
            ).sum(-1)
    weight = np.where(interaction, np.float32(0.5), np.float32(1.0))
    result = dict(
        history_x=features(
            graph,
            observed.shape,
            ("x", observed, weight),
            ("query", np.where(interaction, query_tick, -1), 0.5),
        )
    )
    result.update(
        {
            name: array(value)
            for name, value in dict(
                history_mask=mask,
                history_gap=gap,
                history_ids=ids,
                occurrence=occurrence,
                structure=structural,
                common_stats=common_stats,
                common_ids=common_ids,
                common_frequency=common_frequency,
                pair_gap=pair_gaps,
                pair_mask=pair_mask,
            ).items()
        }
    )
    LOG.debug(
        "history exit rows=%d count=%d structure=%s seconds=%.4f",
        last - first,
        count,
        include_structure,
        time.perf_counter() - started,
    )
    return result


def chosen_points(bank, block, cfg):
    graph, _, stop, target_start, partition, _, _ = block
    rows, steps = bank.points(graph, cfg)
    picked = (steps >= target_start) & (steps < stop)
    picked[picked] = bank.partitions[rows[picked]] == partition
    return rows[picked], steps[picked].astype(np.int64)


def chosen_rows(bank, block, held, first, last, cfg):
    _, _, _, target_start, partition, row_lo, row_hi = block
    moment = held.numpy("query_step")[first:last].astype(np.int64)
    if bank.level == "path":
        return np.isin(moment, chosen_points(bank, block, cfg)[1])
    rows = held.numpy("row_index")[first:last]
    ids = np.arange(first, last)
    selected = (rows >= 0) & (moment >= target_start) & (ids >= row_lo) & (ids < row_hi)
    selected[selected] = bank.partitions[rows[selected]] == partition
    return selected


@dataclass(frozen=True)
class Window:
    graph: int
    start: int
    stop: int
    first: int
    last: int
    lo: int
    hi: int
    held: Resident
    registry: Registry
    cfg: dict
    tokens: np.ndarray | None
    compact: bool

    @classmethod
    def open(cls, bank, block, cfg):
        from .model_spec import row_local

        graph, start, stop, _, _, row_lo, row_hi = block
        held = bank.resident(graph, cfg, "cpu")
        lo, hi = (int(value) for value in held.numpy("snapshot_offsets")[[start, stop]])
        if bank.registry.extras["level"] == "node":
            first, last = row_lo, row_hi
        else:
            first, last = between(held.numpy("query_step"), start, stop)
        if row_local(cfg["model"]):
            chosen = np.flatnonzero(chosen_rows(bank, block, held, first, last, cfg))
            if len(chosen):
                first, last = first + int(chosen[0]), first + int(chosen[-1]) + 1
        tokens = bank.tokens(held) if cfg["model"].get("keys/embedding") else None
        return cls(
            graph, start, stop, first, last, lo, hi, held, bank.registry, cfg, tokens, bank.joining(cfg)
        )


def observed_inputs(window, lo, hi):
    entity = window.held.numpy("entity")[lo:hi]
    result = dict(
        x=features(window.graph, (hi - lo,), ("x", (lo, hi), None)),
        kind=gathered(window.graph, "entity_kind", entity),
    )
    if window.tokens is not None:
        result["token"] = array(window.tokens[entity])
    return result


def frame_inputs(window, start, stop):
    lo, hi = (int(value) for value in window.held.numpy("snapshot_offsets")[[start, stop]])
    result = dict(observed_inputs(window, lo, hi), lo=lo, observations=hi - lo)
    spatial_inputs(result, replace(window, start=start, stop=stop, lo=lo, hi=hi))
    return result


def snapshot_inputs(b, window):
    from .model_spec import shared_frames

    graph, start, stop = window.graph, window.start, window.stop
    lo, hi = window.lo, window.hi
    held, registry = window.held, window.registry
    n = stop - start
    entity = held.numpy("entity")[lo:hi]
    step = held.numpy("step")[lo:hi] - start
    b.update(entity=array(entity, torch.long), step=array(step, torch.long))
    if shared_frames(window.cfg["model"]):
        b["window"] = window
    else:
        b.update(observed_inputs(window, lo, hi))
    if registry.extras["queries"]:
        left, right = between(held.column("events", 2), start * 2, stop * 2)
        choices = held.numpy("events")[left:right]
        choices = choices[choices[:, 3] == 2]
        b.update(
            decision_query=features(graph, (n,), ("query", (start, stop), None)),
            decision_context=array(choices[:, 4:6] - lo, torch.long),
        )
    else:
        b.update(decision_query=features(graph, (n,)), decision_context=np.full((n, 2), -1, np.int64))
    if "window" not in b:
        spatial_inputs(b, window)
    elif window.cfg["model"]["snapshot/temporal"] in ("graph_cell", "evolving"):
        spatial_inputs(b, window)
        for key in ("relation", "attributes", "attribute_edges"):
            del b[key]


def spatial_inputs(b, window):
    held, registry = window.held, window.registry
    start, stop, lo = window.start, window.stop, window.lo
    n = stop - start
    spatial_steps = held.numpy("spatial_step")
    first_edge, last_edge = between(spatial_steps, start, stop)
    spatial = held.numpy("spatial")[first_edge:last_edge]
    edge_bounds = np.searchsorted(
        spatial_steps[first_edge:last_edge], np.arange(start, stop + 1, dtype=spatial_steps.dtype)
    )
    attribute_rows = held.numpy("spatial_attribute_row")
    edges = spatial[:, :2].T - lo
    b.update(edges=array(edges, torch.long), relation=array(spatial[:, 2], torch.long))
    b["attributes"], b["attribute_edges"] = {}, {}
    by_code, code_offsets = held.numpy("spatial_order"), held.numpy("spatial_code_offsets")
    edge_window = np.asarray((first_edge, last_edge), by_code.dtype)
    for name in registry.attributed:
        declared = registry.by_name[name]
        pieces = []
        for code in dict.fromkeys((declared.code, declared.reverse)):
            lo_code, hi_code = code_offsets[code], code_offsets[code + 1]
            low, high = by_code[lo_code:hi_code].searchsorted(edge_window)
            pieces.append(by_code[lo_code + low : lo_code + high])
        chosen = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        values = held.numpy("attributes_" + name)[attribute_rows[chosen]]
        b["attributes"][name] = array(registry.encoding.edges(registry.tables[name], values), torch.float32)
        b["attribute_edges"][name] = array(chosen - first_edge, torch.long)
    directed = registry.canonical[spatial[:, 2]]
    canonical_bounds = np.r_[0, np.cumsum(directed)][edge_bounds]
    b["frame_edges"] = Frames(
        directed, [[(int(edge_bounds[t]), int(edge_bounds[t + 1]))] for t in range(n)], False
    )
    b["frame_directed"] = Frames(
        directed, [[(int(canonical_bounds[t]), int(canonical_bounds[t + 1]))] for t in range(n)], True
    )


def lifted_inputs(b, window):
    held, registry = window.held, window.registry
    lo, hi = window.lo, window.hi
    step = held.numpy("step")[lo:hi] - window.start
    left, right = between(held.column("coupling", 1), lo, hi)
    coupled = held.numpy("coupling")[left:right]
    coupled = coupled[coupled[:, 0] >= lo] - lo
    low, high = between(held.column("coupling_boundaries", 1), lo, hi)
    extra = held.numpy("coupling_boundaries")[low:high]
    extra = extra[extra[:, 0] >= lo]
    boundary = extra[:, :2] - lo
    turn = held.numpy("turn")
    b.update(
        temporal_edges=array(np.concatenate((coupled, boundary)).T, torch.long),
        temporal_relation=array(
            np.r_[np.full(len(coupled), registry.code("temporal")), extra[:, 2]], torch.long
        ),
        temporal_gap=array(
            np.r_[step[coupled[:, 1]] - step[coupled[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]],
            torch.float32,
        ),
        temporal_turn_gap=array(
            np.r_[turn[lo + coupled[:, 1]] - turn[lo + coupled[:, 0]], extra[:, 3]], torch.float32
        ),
    )


def mixed_inputs(b, window):
    graph, start, stop = window.graph, window.start, window.stop
    first, last = window.first, window.last
    held, tokens = window.held, window.tokens
    m = last - first
    actors, subjects = b["actor"], b["subject"]
    moment = b["query_steps"] + start
    mixed_ticks = held.numpy("mixed_tick")
    left, right = between(mixed_ticks, start * 2, stop * 2)
    mixed_edges = held.numpy("mixed_edges")[held.rows_by("mixed_edges", 1, left, right)]
    mixed_edges = mixed_edges[(mixed_edges[:, 0] >= left) & (mixed_edges[:, 0] < right)]
    ticks = mixed_ticks[left:right]
    mk = np.r_[held.numpy("mixed_kind")[left:right], np.full(m, 5)].astype(np.int64)
    qi = np.arange(right - left, right - left + m)
    roots = np.stack((actors, subjects, np.full(m, -1)), 1).astype(np.int64)
    sources, offsets = gather.reachable(
        held.numpy("mixed_entity")[left:right].astype(np.int64),
        np.arange(right - left),
        ticks,
        roots,
        moment * 2,
    )
    extra = np.column_stack((sources, np.repeat(qi, np.diff(offsets)), np.full(len(sources), 2048))).astype(
        np.int64
    )
    edge = np.concatenate((np.column_stack((mixed_edges[:, :2] - left, mixed_edges[:, 2])), extra))
    all_ticks = np.r_[ticks, moment * 2]
    gaps = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2).astype(np.float32)
    b["mixed"] = dict(
        x=joined([features(graph, (right - left,), ("mixed_x", (left, right), None)), b["query"]]),
        kind=mk,
        edge_index=array(edge[:, :2].T, torch.long),
        edge_code=array(edge[:, 2] | (1 << 13), torch.long),
        edge_gap=array(np.stack((gaps, np.zeros_like(gaps)), 1)),
        query=array(qi, torch.long),
    )
    if tokens is not None:
        owner = held.numpy("mixed_entity")[left:right]
        b["mixed"]["token"] = np.r_[
            np.where(owner >= 0, tokens[np.maximum(owner, 0)], -1), np.full(m, -1, np.int64)
        ]


def events_inputs(b, window, blocks):
    graph, start, stop = window.graph, window.start, window.stop
    held = window.held
    left, right = between(held.column("events", 2), start * 2, stop * 2)
    events = held.numpy("events")[left:right]
    if "dag" in blocks:
        members = np.flatnonzero(held.numpy("event_member")[left:right])
        position = np.full(right - left, -1, np.int64)
        position[members] = np.arange(len(members))
        events = events[members]
    interaction = events[:, 3] == 2
    endpoints = np.where(interaction[:, None], -1, events[:, 4:6])
    starts = np.flatnonzero(np.diff(events[:, 2], prepend=-1))
    b.update(
        events=array(events, torch.long),
        event_x=features(
            graph,
            (len(events),),
            ("x", endpoints[:, 0], 0.5),
            ("x", endpoints[:, 1], 0.5),
            ("query", np.where(interaction, events[:, 2] // 2, -1), 1.0),
        ),
        event_groups=[
            (int(events[s, 2]), int(s), int(e)) for s, e in zip(starts, np.append(starts[1:], len(events)))
        ],
    )
    if "dag" in blocks:
        dag_inputs(b, window, left, right, position)


def dag_inputs(b, window, left, right, position):
    held, events = window.held, b["events"]
    actors, subjects = b["actor"], b["subject"]
    moment = b["query_steps"] + window.start
    m = window.last - window.first
    low, high = between(held.column("event_dag", 1), left, right)
    linked = held.numpy("event_dag")[low:high]
    linked = position[linked[linked[:, 0] >= left] - left]
    low, high = between(held.column("event_boundaries", 1), left, right)
    extra = held.numpy("event_boundaries")[low:high]
    extra = extra[extra[:, 0] >= left]
    boundary = position[extra[:, :2] - left]
    dag = np.concatenate((linked, boundary)).T.astype(np.int64)
    b.update(
        dag_edges=array(dag),
        dag_relation=array(np.concatenate((events[dag[1, : len(linked)], 3], extra[:, 2])), torch.long),
        dag_gap=array(events[dag[1], 2] - events[dag[0], 2], torch.float32),
    )
    positions = np.r_[np.arange(len(events)), np.arange(len(events))]
    reached, offsets = gather.reachable(
        np.r_[events[:, 0], events[:, 1]].astype(np.int64),
        positions,
        events[:, 2].astype(np.int64),
        np.stack((actors, subjects), 1).astype(np.int64),
        moment * 2,
    )
    b["dag_query_edges"] = array(np.stack((reached, np.repeat(np.arange(m), np.diff(offsets)))), torch.long)


def walks_inputs(b, window, context):
    from .turns import enabled, lookback

    graph, start = window.graph, window.start
    first, last = window.first, window.last
    held, cfg, tokens = window.held, window.cfg, window.tokens
    m = last - first
    actors, subjects = b["actor"], b["subject"]
    moment = b["query_steps"] + start
    walk_count, hops = cfg["model"]["walks/count"], cfg["model"]["walks/hops"]
    stored = held.numpy("walks")
    if walk_count > stored.shape[2] or hops > stored.shape[3]:
        raise ValueError("requested walks exceed the walks stored for this graph variant")
    walks = stored[first:last, :, :walk_count, :hops]
    if enabled(cfg):
        extra = held.numpy("turn_walks")[first:last, :, : lookback(cfg), :walk_count, :hops]
        extra = extra.reshape(m, 2, lookback(cfg) * walk_count, hops)
        walks = np.concatenate((walks, extra), 2)
        kinds = np.zeros(walks.shape, np.int64)
        kinds[:, :, walk_count:] = 1
        gaps = np.zeros(walks.shape + (3,), np.float32)
        past = context["end_token_gaps"][:, :2]
        gaps[:, :, walk_count:] = np.repeat(past, walk_count, axis=2)[:, :, :, None, :]
        b.update(walk_turn_kind=kinds, walk_end_time=gaps)
    known = walks >= 0
    known &= held.column("events", 2)[np.maximum(walks, 0)] >= start * 2
    events = held.numpy("events")[np.maximum(walks, 0)]
    nodes = np.empty(walks.shape, np.int32)
    observations = np.empty_like(nodes)
    current = np.broadcast_to(np.stack((actors, subjects), 1)[:, :, None], walks.shape[:3]).copy()
    for hop in range(hops):
        forward = events[:, :, :, hop, 0] == current
        nodes[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 1], events[:, :, :, hop, 0])
        observations[:, :, :, hop] = np.where(forward, events[:, :, :, hop, 5], events[:, :, :, hop, 4])
        current = nodes[:, :, :, hop]
    gap = moment[:, None, None, None] * 2 + 1 - events[..., 2]
    b.update(
        walk_x=features(graph, known.shape, ("x", np.where(known, observations, -1), 1.0)),
        walk_mask=array(known),
        walk_gap=array(gap, torch.float32),
        walk_identity=array(gather.walk_identities(nodes, known)),
    )
    if tokens is not None:
        b["walk_token"] = np.where(known, tokens[nodes], -1)


def base_inputs(bank, block, window):
    _, _, _, target_start, partition, row_lo, row_hi = block
    graph, start, stop = window.graph, window.start, window.stop
    first, last, lo, hi = window.first, window.last, window.lo, window.hi
    held, tokens = window.held, window.tokens
    node = window.registry.extras["level"] == "node"
    n = stop - start
    query_steps = held.numpy("query_step")
    m = last - first
    moment = query_steps[first:last].astype(np.int64)
    selected = chosen_rows(bank, block, held, first, last, window.cfg)
    if bank.level == "path":
        rows, steps = chosen_points(bank, block, window.cfg)
        if not np.isin(steps, moment).all():
            raise ValueError("a point of graph %d lies outside its window" % graph)
        point_query = np.searchsorted(moment[selected], steps)
    else:
        rows = held.numpy("row_index")[first:last][selected]
    local = moment - start
    bounds = np.searchsorted(local, np.arange(n + 1))
    query_context = held.numpy("query_context")[first:last]
    actors, subjects = held.numpy("actor")[first:last], held.numpy("subject")[first:last]
    b = dict(
        graph=graph,
        start=start,
        stop=stop,
        frames=n,
        time_steps=n,
        batch_size=1,
        predictions=m,
        observations=hi - lo,
        graphs=[graph],
        query_steps=array(local, torch.long),
        query_frames=array(local, torch.long),
        frame_queries=[np.arange(bounds[t], bounds[t + 1], dtype=np.int64) for t in range(n)],
        selected=array(selected),
        rows=Host(rows),
        target=array(bank.targets[rows]),
        query=features(graph, (m,)) if node else features(graph, (m,), ("query", (first, last), None)),
        context=features(graph, query_context.shape, ("x", query_context, None)),
        actor=array(actors, torch.long),
        subject=array(subjects, torch.long),
        context_index=array(query_context - lo, torch.long),
        signal=array(held.numpy("signals")[first:last]),
        affinity=array(held.numpy("affinity")[first:last]),
        global_affinity=array(held.numpy("global_affinity")[first:last]),
    )
    if tokens is not None:
        b["root_token"] = array(tokens[np.stack((actors, subjects), 1)])
    if bank.level == "path":
        b.update(
            tau=array(bank.taus[rows]),
            series=array(bank.series[rows], torch.long),
            point_query=array(point_query, torch.long),
            queries=int(selected.sum()),
        )
    return b


def tree_inputs(b, window):
    from .model_spec import neighbors

    graph, start = window.graph, window.start
    held, registry, cfg, tokens = window.held, window.registry, window.cfg, window.tokens
    spec, stream = cfg["graph"], SEED
    actors, subjects = b["actor"], b["subject"]
    moment = b["query_steps"] + start
    roots = np.stack((actors, subjects), 1).reshape(-1).astype(np.int64)
    cutoffs = np.repeat(moment * 2 + 1, 2)
    b["tree"] = tree(
        graph,
        held,
        roots,
        cutoffs,
        start * 2,
        neighbors(cfg["model"]),
        cfg["model"]["tree/depth"],
        registry,
        spec,
        stream,
        tokens,
    )


def neighbor_history_inputs(b, window, blocks):
    from .model_spec import neighbors

    graph, start = window.graph, window.start
    held, cfg, tokens = window.held, window.cfg, window.tokens
    spec, stream = cfg["graph"], SEED
    first, last = window.first, window.last
    selected = b["selected"]
    b.update(
        history_inputs(
            graph,
            held,
            first,
            last,
            start,
            neighbors(cfg["model"]),
            spec,
            stream,
            "structure" in blocks,
            selected if "history_targets" in blocks else None,
        )
    )
    if tokens is not None:
        b["history_token"] = np.where(b["history_ids"] >= 0, tokens[np.maximum(b["history_ids"], 0)], -1)


def prepare_context(window):
    from .turn_inputs import add
    from .turns import enabled

    result = {}
    if enabled(window.cfg):
        add(result, window.graph, window.held, window.first, window.last, window.start, window.cfg)
    return result


def context_inputs(b, window, blocks, context):
    from .turn_inputs import history_tokens, mixed_edges, temporal_edges

    if not context:
        return
    b.update(context)
    if "lifted" in blocks:
        temporal_edges(b, window.graph, window.held, window.lo, window.hi, window.registry, window.tokens)
    if "mixed" in blocks:
        mixed_edges(b, window.graph)
    if "turn_history" in blocks:
        history_tokens(
            b,
            window.graph,
            window.held,
            window.first,
            window.last,
            window.start,
            window.cfg,
            window.registry,
            window.tokens,
        )


def entity_space(b, window, blocks):
    graph, held, start = window.graph, window.held, window.start
    fields = [key for key in ("actor", "subject", "entity", "history_ids", "common_ids") if key in b]
    space = None
    if window.compact:
        ids = [np.asarray(b[key]).reshape(-1) for key in fields]
        if "events" in b:
            ids += [b["events"][:, 0], b["events"][:, 1]]
        space = np.unique(np.concatenate(ids + [np.zeros(0, np.int64)]))
        space = space[space >= 0]
        for key in fields:
            values = np.asarray(b[key])
            b[key] = np.where(values >= 0, np.searchsorted(space, values), values).astype(values.dtype)
        if "events" in b:
            events = b["events"].copy()
            events[:, :2] = np.searchsorted(space, events[:, :2])
            b["events"] = events
    size = len(held.numpy("entity_kind")) if space is None else len(space)
    b.update(
        entities=size,
        entity_sizes=[size],
        entity_batch=np.zeros(size, np.int64),
        entity_kind=gathered(graph, "entity_kind")
        if space is None
        else gathered(graph, "entity_kind", space),
    )
    if window.tokens is not None:
        b["entity_token"] = array(window.tokens if space is None else window.tokens[space])
    if "snapshot" in blocks:
        count = window.hi - window.lo
        align = np.full((b["frames"], size), b["observations"], np.int64)
        align[b["step"][:count], b["entity"][:count]] = np.arange(count)
        b.update(align=align, present=align < count)
    if "events" in blocks:
        chosen = np.arange(size) if space is None else space
        initial = gather.latest(
            held.numpy("observation_ids"),
            held.numpy("observation_offsets"),
            held.numpy("step"),
            chosen,
            np.full(size, start * 2, np.int64),
        )
        b["initial"] = features_at(graph, initial)


def recipe(bank, block, cfg):
    from .model_spec import blocks as model_blocks

    started = time.perf_counter()
    window = Window.open(bank, block, cfg)
    blocks = model_blocks(cfg["model"])
    b = base_inputs(bank, block, window)
    context = prepare_context(window)
    if "snapshot" in blocks:
        snapshot_inputs(b, window)
    if "lifted" in blocks:
        lifted_inputs(b, window)
    if "mixed" in blocks:
        mixed_inputs(b, window)
    if "events" in blocks:
        events_inputs(b, window, blocks)
    if "tree" in blocks:
        tree_inputs(b, window)
    if "history" in blocks:
        neighbor_history_inputs(b, window, blocks)
    if "walks" in blocks:
        walks_inputs(b, window, context)
    context_inputs(b, window, blocks, context)
    entity_space(b, window, blocks)
    LOG.debug(
        "recipe exit graph=%d start=%d stop=%d predictions=%d seconds=%.4f",
        window.graph,
        window.start,
        window.stop,
        b["predictions"],
        time.perf_counter() - started,
    )
    return b
