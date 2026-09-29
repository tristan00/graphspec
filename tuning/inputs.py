import logging
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from graphspec import Registry
from graphspec.infra.arrays import Resident, between
from graphspec.references import SORTS
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


def history_inputs(window, b, count, include_structure, selected):
    started = time.perf_counter()
    graph, held, start = window.graph, window.held, window.start
    mode, decay = sampling(window.cfg["graph"])
    rows, partner = b["member_row"], b["partner"]
    chosen = np.ones(len(b["row_kind"]), bool) if selected is None else np.asarray(selected, bool)
    entity = b["member_entity"].astype(np.int64)
    moments = (b["row_steps"] + start).astype(np.int64)
    ends = np.append(entity, -1)[b["pair"]]
    shared = (
        held.numpy("events"),
        held.numpy("neighbor_ids"),
        held.numpy("neighbor_offsets"),
        held.column("events", 2),
    )
    sampled = (start, count, mode, decay, SEED, NEIGHBOURS)
    observed, interaction, task_row, mask, gap, ids, relation, structural = gather.history(
        *shared,
        entity,
        np.append(entity, -1)[partner],
        moments[rows],
        *sampled,
        include_structure,
        chosen[rows],
    )
    common_ids, common_frequency, common_stats, pair_gaps, pair_mask = gather.paired(
        *shared, ends[:, 0], ends[:, 1], moments, *sampled, chosen
    )
    others = np.maximum(partner, 0)
    occurrence = np.stack(
        (
            ((ids[:, :, None] == ids[:, None, :]) & mask[:, :, None] & mask[:, None, :]).sum(-1),
            (
                (ids[:, :, None] == ids[others][:, None, :])
                & mask[:, :, None]
                & (mask[others] & (partner >= 0)[:, None])[:, None, :]
            ).sum(-1),
        ),
        -1,
    ).astype(np.float32)
    weight = np.where(interaction, np.float32(0.5), np.float32(1.0))
    result = dict(
        history_x=features(
            graph,
            observed.shape,
            ("x", observed, weight),
            ("task", np.where(interaction, task_row, -1), 0.5),
        )
    )
    result.update(
        {
            name: array(value)
            for name, value in dict(
                history_mask=mask,
                history_gap=gap,
                history_ids=ids,
                history_relation=relation,
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
        "history exit members=%d count=%d structure=%s seconds=%.4f",
        len(rows),
        count,
        include_structure,
        time.perf_counter() - started,
    )
    return result


def chosen_points(bank, block, cfg):
    started = time.perf_counter()
    graph, _, stop, target_start, partition, _, _ = block
    rows, places = bank.labels(graph, cfg)
    steps = bank.points(graph, cfg)[1][places]
    picked = (steps >= target_start) & (steps < stop)
    picked[picked] = bank.partitions(cfg)[rows[picked]] == partition
    LOG.debug(
        "chosen points exit graph=%d points=%d seconds=%.4f",
        graph,
        int(picked.sum()),
        time.perf_counter() - started,
    )
    return rows[picked], places[picked].astype(np.int64)


def chosen_rows(bank, block, held, first, last, cfg):
    _, _, _, target_start, partition, row_lo, row_hi = block
    moment = held.numpy("row_step")[first:last].astype(np.int64)
    ids = np.arange(first, last)
    if bank.path:
        return np.isin(ids, chosen_points(bank, block, cfg)[1])
    rows = held.numpy("row_index")[first:last]
    selected = (rows >= 0) & (moment >= target_start) & (ids >= row_lo) & (ids < row_hi)
    selected[selected] = bank.partitions(cfg)[rows[selected]] == partition
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
        from .model_spec import row_local, windowed

        graph, start, stop, _, _, row_lo, row_hi = block
        held = bank.resident(graph, cfg, "cpu")
        lo, hi = (int(value) for value in held.numpy("snapshot_offsets")[[start, stop]])
        if windowed(cfg["model"]) or bank.path:
            first, last = between(held.numpy("row_step"), start, stop)
        else:
            first, last = row_lo, row_hi
        if row_local(cfg["model"]):
            chosen = np.flatnonzero(chosen_rows(bank, block, held, first, last, cfg))
            if len(chosen):
                first, last = first + int(chosen[0]), first + int(chosen[-1]) + 1
        tokens = bank.tokens(held) if cfg["model"].get("keys/embedding") else None
        return cls(
            graph, start, stop, first, last, lo, hi, held, bank.fitted(cfg), cfg, tokens, bank.joining(cfg)
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

    start, lo, hi, held = window.start, window.lo, window.hi, window.held
    entity = held.numpy("entity")[lo:hi]
    step = held.numpy("step")[lo:hi] - start
    b.update(entity=array(entity, torch.long), step=array(step, torch.long))
    if shared_frames(window.cfg["model"]):
        b["window"] = window
    else:
        b.update(observed_inputs(window, lo, hi))
    if window.cfg["model"]["snapshot/temporal"] == "graph_sequence":
        b.update(frame_inputs_of(window))
    if "window" not in b:
        spatial_inputs(b, window)
    elif window.cfg["model"]["snapshot/temporal"] in ("graph_cell", "evolving"):
        spatial_inputs(b, window)
        for key in ("relation", "attributes", "attribute_edges"):
            del b[key]


def frame_inputs_of(window):
    started = time.perf_counter()
    graph, start, stop, lo, held = window.graph, window.start, window.stop, window.lo, window.held
    left, right = between(held.column("events", 2), start * 2, stop * 2)
    seen = held.numpy("events")[left:right]
    seen = seen[seen[:, 3] == 2]
    result = dict(
        frame_event=array(seen[:, 2] // 2 - start, torch.long),
        frame_task=features(graph, (len(seen),), ("task", seen[:, 7], 1.0)),
        frame_starts=array(seen[:, 4:6] - lo, torch.long),
    )
    LOG.debug(
        "frame inputs exit graph=%d frames=%d events=%d seconds=%.4f",
        graph,
        stop - start,
        len(seen),
        time.perf_counter() - started,
    )
    return result


def pooled(held, steps):
    started = time.perf_counter()
    offsets, table, references = held.numpy("snapshot_offsets"), held.numpy("x_table"), held.numpy("x_ref")
    result = np.zeros((len(steps), table.shape[1]), np.float32)
    for step in np.unique(steps).tolist():
        lo, hi = int(offsets[step]), int(offsets[step + 1])
        if hi > lo:
            result[steps == step] = np.asarray(table[references[lo:hi]], np.float32).mean(0)
    LOG.debug("pooled exit rows=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def member_inputs(window, first, last):
    started = time.perf_counter()
    held, lo = window.held, window.lo
    rows = held.numpy("member_row")
    left, right = between(rows, first, last)
    observed = held.numpy("member_observed")[left:right]
    stored = left + np.flatnonzero(observed >= 0)
    observed = observed[observed >= 0]
    wholes = held.numpy("whole_row")
    low, high = between(wholes, first, last)
    whole_row = wholes[low:high] - first
    result = dict(
        member_row=array(rows[stored] - first, torch.long),
        member_name=array(held.numpy("member_name")[stored], torch.long),
        member_entity=array(held.numpy("member_entity")[stored], torch.long),
        member_index=array(observed - lo, torch.long),
        member_place=array(held.numpy("member_place")[stored], torch.long),
        member_stored=stored,
        member_x=features(window.graph, (len(stored),), ("x", observed, 1.0)),
        whole_row=array(whole_row, torch.long),
        whole_name=array(held.numpy("whole_name")[low:high], torch.long),
        whole_x=pooled(held, held.numpy("row_step")[first + whole_row]),
    )
    LOG.debug(
        "member inputs exit graph=%d members=%d wholes=%d seconds=%.4f",
        window.graph,
        len(stored),
        len(whole_row),
        time.perf_counter() - started,
    )
    return result


def couples(registry):
    started = time.perf_counter()
    names, result = list(registry.references), {}
    for code, entry in enumerate(registry.schema.tasks["kinds"].values()):
        if entry["event"] is not None:
            result[code] = tuple(names.index(name) for name in entry["event"])
        elif entry["target"] is not None and entry["target"]["about"] is not None:
            about = entry["target"]["about"]
            sort = SORTS.named(registry.schema.tasks["references"][about]["sort"])
            if sort.edges and not sort.nodes and sort.most == 1:
                result[code] = (names.index(about),) * 2
    LOG.debug("couples exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def paired(registry, kinds, rows, names):
    started = time.perf_counter()
    count = len(registry.references)
    keys = np.asarray(rows, np.int64) * count + np.asarray(names, np.int64)
    order = np.lexsort((np.arange(len(keys)), keys))
    ranked = keys[order]
    result = np.full((len(kinds), 2), -1, np.int64)
    for code, (left, right) in couples(registry).items():
        chosen = np.flatnonzero(np.asarray(kinds) == code)
        if not len(chosen) or not len(ranked):
            continue
        wanted = np.stack((chosen * count + left, chosen * count + right), 1)
        at = np.searchsorted(ranked, wanted)
        at[:, 1] += left == right
        inside = at < len(ranked)
        at = np.minimum(at, len(ranked) - 1)
        found = (inside & (ranked[at] == wanted)).all(1)
        result[chosen[found]] = order[at[found]]
    LOG.debug("paired exit rows=%d seconds=%.4f", len(kinds), time.perf_counter() - started)
    return result


def partnered(pair, members):
    started = time.perf_counter()
    result = np.full(members, -1, np.int64)
    held = pair[:, 0] >= 0
    result[pair[held, 0]], result[pair[held, 1]] = pair[held, 1], pair[held, 0]
    LOG.debug("partnered exit members=%d seconds=%.4f", members, time.perf_counter() - started)
    return result


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
    valued = registry.value_group[spatial[:, 2]]
    for place, group in enumerate(registry.valued):
        chosen = np.flatnonzero(valued == place)
        values = held.numpy("attributes_" + group)[attribute_rows[first_edge + chosen]]
        b["attributes"][group] = array(
            registry.encoding.vectors(registry.attributes(group), values), torch.float32
        )
        b["attribute_edges"][group] = array(chosen, torch.long)
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
    period = held.numpy("obs_period")
    b.update(
        temporal_edges=array(np.concatenate((coupled, boundary)).T, torch.long),
        temporal_relation=array(
            np.r_[np.full(len(coupled), registry.code("temporal")), extra[:, 2]], torch.long
        ),
        temporal_gap=array(
            np.r_[step[coupled[:, 1]] - step[coupled[:, 0]], step[boundary[:, 1]] - step[boundary[:, 0]]],
            torch.float32,
        ),
        temporal_period_gap=array(
            np.r_[period[lo + coupled[:, 1]] - period[lo + coupled[:, 0]], extra[:, 3]], torch.float32
        ),
    )


def mixed_inputs(b, window):
    graph, start, stop = window.graph, window.start, window.stop
    held, tokens = window.held, window.tokens
    m = len(b["member_row"])
    moment = b["row_steps"][b["member_row"]] + start
    mixed_ticks = held.numpy("mixed_tick")
    left, right = between(mixed_ticks, start * 2, stop * 2)
    mixed_edges = held.numpy("mixed_edges")[held.rows_by("mixed_edges", 1, left, right)]
    mixed_edges = mixed_edges[(mixed_edges[:, 0] >= left) & (mixed_edges[:, 0] < right)]
    ticks = mixed_ticks[left:right]
    mk = np.r_[held.numpy("mixed_kind")[left:right], np.full(m, 5)].astype(np.int64)
    qi = np.arange(right - left, right - left + m)
    sources, offsets = gather.reachable(
        held.numpy("mixed_entity")[left:right].astype(np.int64),
        np.arange(right - left),
        ticks,
        b["member_entity"][:, None].astype(np.int64),
        moment * 2,
    )
    extra = np.column_stack((sources, np.repeat(qi, np.diff(offsets)), np.full(len(sources), 2048))).astype(
        np.int64
    )
    edge = np.concatenate((np.column_stack((mixed_edges[:, :2] - left, mixed_edges[:, 2])), extra))
    all_ticks = np.r_[ticks, moment * 2]
    gaps = np.log1p((all_ticks[edge[:, 1]] - all_ticks[edge[:, 0]]).clip(0) / 2).astype(np.float32)
    b["mixed"] = dict(
        x=joined([features(graph, (right - left,), ("mixed_x", (left, right), None)), b["member_x"]]),
        kind=mk,
        edge_index=array(edge[:, :2].T, torch.long),
        edge_code=array(edge[:, 2] | (1 << 13), torch.long),
        edge_gap=array(np.stack((gaps, np.zeros_like(gaps)), 1)),
        row=array(qi, torch.long),
    )
    if tokens is not None:
        owner = held.numpy("mixed_entity")[left:right]
        b["mixed"]["token"] = np.r_[np.where(owner >= 0, tokens[np.maximum(owner, 0)], -1), b["member_token"]]


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
            ("task", np.where(interaction, events[:, 7], -1), 1.0),
        ),
        event_groups=[
            (int(events[s, 2]), int(s), int(e)) for s, e in zip(starts, np.append(starts[1:], len(events)))
        ],
    )
    if "dag" in blocks:
        dag_inputs(b, window, left, right, position)


def dag_inputs(b, window, left, right, position):
    held, events = window.held, b["events"]
    moment = b["row_steps"][b["member_row"]] + window.start
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
        b["member_entity"][:, None].astype(np.int64),
        moment * 2,
    )
    b["dag_member_edges"] = array(
        np.stack((reached, np.repeat(np.arange(len(moment)), np.diff(offsets)))), torch.long
    )


def walks_inputs(b, window, context):
    from .periods import enabled, lookback

    graph, start = window.graph, window.start
    held, cfg, tokens = window.held, window.cfg, window.tokens
    chosen = b["member_stored"]
    moment = b["row_steps"][b["member_row"]] + start
    walk_count, hops = cfg["model"]["walks/count"], cfg["model"]["walks/hops"]
    stored = held.numpy("walks")
    if walk_count > stored.shape[1] or hops > stored.shape[2]:
        raise ValueError("requested walks exceed the walks stored for this graph variant")
    walks = stored[chosen, :walk_count, :hops]
    if enabled(cfg):
        extra = held.numpy("period_walks")[chosen, : lookback(cfg), :walk_count, :hops]
        walks = np.concatenate((walks, extra.reshape(len(chosen), lookback(cfg) * walk_count, hops)), 1)
        kinds = np.zeros(walks.shape, np.int64)
        kinds[:, walk_count:] = 1
        gaps = np.zeros(walks.shape + (3,), np.float32)
        gaps[:, walk_count:] = np.repeat(context["member_end_gaps"], walk_count, axis=1)[:, :, None, :]
        b.update(walk_period_kind=kinds, walk_end_time=gaps)
    known = walks >= 0
    known &= held.column("events", 2)[np.maximum(walks, 0)] >= start * 2
    events = held.numpy("events")[np.maximum(walks, 0)]
    nodes = np.empty(walks.shape, np.int32)
    observations = np.empty_like(nodes)
    current = np.broadcast_to(b["member_entity"][:, None], walks.shape[:2]).copy()
    for hop in range(hops):
        forward = events[:, :, hop, 0] == current
        nodes[:, :, hop] = np.where(forward, events[:, :, hop, 1], events[:, :, hop, 0])
        observations[:, :, hop] = np.where(forward, events[:, :, hop, 5], events[:, :, hop, 4])
        current = nodes[:, :, hop]
    gap = moment[:, None, None] * 2 + 1 - events[..., 2]
    b.update(
        walk_x=features(graph, known.shape, ("x", np.where(known, observations, -1), 1.0)),
        walk_mask=array(known),
        walk_gap=array(gap, torch.float32),
        walk_relation=array(np.where(known & (events[..., 3] == 1), events[..., 6], -1), torch.long),
        walk_identity=array(gather.walk_identities(nodes, known, b["partner"])),
    )
    if tokens is not None:
        b["walk_token"] = np.where(known, tokens[nodes], -1)


def base_inputs(bank, block, window):
    _, _, _, target_start, partition, row_lo, row_hi = block
    graph, start, stop = window.graph, window.start, window.stop
    first, last, lo, hi = window.first, window.last, window.lo, window.hi
    held, tokens = window.held, window.tokens
    n = stop - start
    row_steps = held.numpy("row_step")
    m = last - first
    moment = row_steps[first:last].astype(np.int64)
    selected = chosen_rows(bank, block, held, first, last, window.cfg)
    if bank.path:
        rows, places = chosen_points(bank, block, window.cfg)
        if np.any(places < first) or np.any(places >= last):
            raise ValueError("a point of graph %d lies outside its window" % graph)
        point_row = np.searchsorted(np.flatnonzero(selected) + first, places)
    else:
        rows = held.numpy("row_index")[first:last][selected]
    local = moment - start
    bounds = np.searchsorted(local, np.arange(n + 1))
    kinds = held.numpy("row_kind")[first:last]
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
        row_steps=array(local, torch.long),
        row_frames=array(local, torch.long),
        frame_rows=[np.arange(bounds[t], bounds[t + 1], dtype=np.int64) for t in range(n)],
        selected=array(selected),
        rows=Host(rows),
        target=array(bank.targets[rows]),
        weight=array(bank.weights[rows]),
        series=array(bank.series[rows], torch.long),
        task=features(graph, (m,), ("task", held.numpy("row_task")[first:last], 1.0)),
        row_kind=array(kinds, torch.long),
        **member_inputs(window, first, last),
    )
    b["pair"] = paired(window.registry, kinds, b["member_row"], b["member_name"])
    b["partner"] = partnered(b["pair"], len(b["member_row"]))
    if tokens is not None:
        b["member_token"] = array(tokens[b["member_entity"]])
    if bank.path:
        b.update(
            tau=array(bank.taus[rows]),
            point_row=array(point_row, torch.long),
            tasks=int(selected.sum()),
        )
    return b


def tree_inputs(b, window):
    from .model_spec import neighbors

    graph, start = window.graph, window.start
    held, registry, cfg, tokens = window.held, window.registry, window.cfg, window.tokens
    spec, stream = cfg["graph"], SEED
    moment = b["row_steps"][b["member_row"]] + start
    b["tree"] = tree(
        graph,
        held,
        b["member_entity"].astype(np.int64),
        (moment * 2 + 1).astype(np.int64),
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

    tokens = window.tokens
    b.update(
        history_inputs(
            window,
            b,
            neighbors(window.cfg["model"]),
            "structure" in blocks,
            b["selected"] if "history_targets" in blocks else None,
        )
    )
    if tokens is not None:
        b["history_token"] = np.where(b["history_ids"] >= 0, tokens[np.maximum(b["history_ids"], 0)], -1)


def pooled_frames(window, values):
    started = time.perf_counter()
    steps = window.held.numpy("row_step")
    first, last = between(steps, window.start, window.stop)
    frames, local = window.stop - window.start, steps[first:last].astype(np.int64) - window.start
    total = np.zeros((frames,) + values.shape[1:], np.float32)
    np.add.at(total, local, values)
    count = np.maximum(np.bincount(local, minlength=frames), 1).reshape((-1,) + (1,) * (values.ndim - 1))
    LOG.debug(
        "pooled frames exit graph=%d rows=%d frames=%d seconds=%.4f",
        window.graph,
        len(local),
        frames,
        time.perf_counter() - started,
    )
    return (total / count).astype(np.float32)


def signal_inputs(b, window):
    started = time.perf_counter()
    first, last = between(window.held.numpy("row_step"), window.start, window.stop)
    b["signal"] = pooled_frames(window, np.asarray(window.held.numpy("signals")[first:last], np.float32))
    LOG.debug("signal inputs exit graph=%d seconds=%.4f", window.graph, time.perf_counter() - started)


def affinity_frames(window):
    from .period_inputs import profiles

    started = time.perf_counter()
    held = window.held
    steps = held.numpy("row_step").astype(np.int64)
    first, last = between(steps, window.start, window.stop)
    left, right = between(held.numpy("member_row"), first, last)
    shown = held.numpy("member_observed")[left:right] >= 0
    owner = held.numpy("member_row")[left:right][shown].astype(np.int64) - first
    profile = profiles(held, held.numpy("member_entity")[left:right][shown], 2 * steps[first:last][owner])
    rows = np.zeros((last - first, profile.shape[1]), np.float32)
    np.add.at(rows, owner, profile)
    rows /= np.maximum(np.bincount(owner, minlength=last - first), 1)[:, None]
    code = window.registry.seen[held.numpy("row_kind")[first:last].astype(np.int64)]
    rows[np.flatnonzero(code >= 0), code[code >= 0]] += 1
    result = pooled_frames(window, rows)
    LOG.debug(
        "affinity frames exit graph=%d rows=%d seconds=%.4f", window.graph, last - first, time.perf_counter() - started
    )
    return result


def affinity_inputs(b, window):
    from .period_inputs import profiles

    started = time.perf_counter()
    held, start = window.held, window.start
    moment = b["row_steps"] + start
    b.update(
        member_affinity=profiles(held, b["member_entity"], 2 * moment[b["member_row"]]),
        global_affinity=np.asarray(held.numpy("affinity_steps")[moment], np.float32),
        frame_affinity=affinity_frames(window),
    )
    LOG.debug(
        "affinity inputs exit graph=%d members=%d seconds=%.4f",
        window.graph,
        len(b["member_row"]),
        time.perf_counter() - started,
    )


def prepare_context(window, b):
    from .period_inputs import add
    from .periods import enabled

    result = {}
    if enabled(window.cfg):
        add(result, window, b)
    return result


def context_inputs(b, window, blocks, context):
    from .period_inputs import history_tokens, mixed_edges, temporal_edges

    if not context:
        return
    b.update(context)
    if "lifted" in blocks:
        temporal_edges(b, window.graph, window.held, window.lo, window.hi, window.registry, window.tokens)
    if "mixed" in blocks:
        mixed_edges(b, window.graph)
    if "period_history" in blocks:
        history_tokens(b, window)


def entity_space(b, window, blocks):
    graph, held, start = window.graph, window.held, window.start
    fields = [key for key in ("entity", "history_ids", "common_ids", "member_entity") if key in b]
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
    context = prepare_context(window, b)
    if "snapshot" in blocks:
        snapshot_inputs(b, window)
    if "lifted" in blocks:
        lifted_inputs(b, window)
    if "mixed" in blocks:
        mixed_inputs(b, window)
    if "events" in blocks:
        events_inputs(b, window, blocks)
    if "affinity" in blocks:
        affinity_inputs(b, window)
    if "signal" in blocks:
        signal_inputs(b, window)
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
