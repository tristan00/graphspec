import numpy as np
import torch

from graphspec.engine.transform import SEED
from graphspec.infra.timing import timed
from graphspec.representations.groups import tick_span
from tuning.families.periods import lookback

from .inputs import tree
from .tensors import Features, array, features, gathered, joined, length


def prior_rows(held, periods, entities):
    periods, entities = np.broadcast_arrays(periods, entities)
    count = len(held.numpy("period_ends"))
    keys, rows = held.numpy("period_observation_keys"), held.numpy("observation_ids")
    if not len(keys):
        return np.full(periods.shape, -1, np.int32)
    positions = np.searchsorted(keys, entities.astype(np.int64) * count + periods, side="right") - 1
    safe = positions.clip(0)
    valid = (positions >= 0) & (keys[safe] // count == entities)
    return np.where(valid, rows[safe], -1).astype(np.int32)


@timed("period_inputs.profiles")
def profiles(held, entities, moments):
    keys, counts = held.numpy("affinity_keys"), held.numpy("affinity_counts")
    entities, moments = np.broadcast_arrays(np.asarray(entities, np.int64), np.asarray(moments, np.int64))
    if not len(keys):
        return np.zeros(entities.shape + counts.shape[1:], np.float32)
    span = tick_span(len(held.numpy("period")))
    at = np.searchsorted(keys, entities * span + moments, side="left") - 1
    safe = at.clip(0)
    valid = (at >= 0) & (keys[safe] // span == entities)
    return np.where(valid[..., None], counts[safe], 0).astype(np.float32)


@timed("period_inputs.add")
def add(b, window, members):
    graph, held, first, last, start, cfg = (
        window.graph,
        window.held,
        window.first,
        window.last,
        window.start,
        window.cfg,
    )
    owner, entities = members["member_row"], members["member_entity"]
    tasks = held.numpy("row_step")[first:last].astype(np.int64)
    membership = held.numpy("period_membership")[tasks]
    previous = membership[:, None] - np.arange(1, lookback(cfg) + 1)
    valid = previous >= 0
    prior = previous.clip(0)
    ends = held.numpy("period_ends")[prior]
    valid &= ends >= start
    step = held.numpy("step")
    observed = prior_rows(held, prior[owner], entities[:, None])
    mask = (observed >= 0) & valid[owner]
    age = np.maximum(ends[owner] - step[observed.clip(0)], 0)
    age[~mask] = 0
    oldest = np.zeros(valid.shape, np.int64)
    referred = members["member_name"] < len(window.registry.references)
    np.maximum.at(oldest, owner[referred], age[referred])
    numbers = held.numpy("period_numbers")
    period_gap = numbers[membership, None] - numbers[prior]
    gaps = np.stack((tasks[:, None] - ends, period_gap, oldest), -1).clip(0).astype(np.float32)
    member_gaps = (
        np.stack((tasks[owner][:, None] - step[observed.clip(0)], period_gap[owner], age), -1)
        .clip(0)
        .astype(np.float32)
    )
    local = observed - int(held.numpy("snapshot_offsets")[start])
    b.update(
        end_steps=array(np.where(valid, ends - start, -1), torch.long),
        end_mask=array(valid),
        end_gaps=gaps,
        member_end_raw=features(graph, observed.shape, ("x", np.where(mask, observed, -1), 1.0)),
        member_end_mask=array(mask),
        member_end_gaps=member_gaps,
        member_end_local=array(np.where(mask & (local >= 0), local, -1), torch.long),
    )
    if cfg["model"]["view"] == "affinity":
        b.update(
            member_end_affinity=profiles(held, entities[:, None], 2 * ends[owner] + 2)
            * valid[owner][..., None],
            end_global_affinity=np.asarray(held.numpy("affinity_steps")[ends + 1], np.float32)
            * valid[..., None],
        )


@timed("period_inputs.temporal_edges")
def temporal_edges(b, graph, held, lo, hi, registry, tokens):
    temporal, context = registry.code("temporal"), registry.marker("period_context")
    nodes = np.arange(lo, hi)
    step, entity = held.numpy("step"), held.numpy("entity")
    task = step[nodes]
    membership = held.numpy("period_membership")[task]
    prior = membership[:, None] - np.arange(1, b["end_steps"].shape[1] + 1)
    valid = prior >= 0
    observed = prior_rows(held, prior.clip(0), entity[nodes, None])
    valid &= observed >= 0
    valid &= held.numpy("period_ends")[prior.clip(0)] >= b["start"]
    source = observed[valid]
    destination = np.broadcast_to(nodes[:, None], observed.shape)[valid]
    ghosts = np.unique(source[source < lo])
    if len(ghosts):
        count = b["observations"]
        b["x"] = joined([b["x"], features(graph, (len(ghosts),), ("x", ghosts, None))])
        b["kind"] = joined([b["kind"], gathered(graph, "entity_kind", entity[ghosts])])
        b["entity"] = np.concatenate((b["entity"], entity[ghosts].astype(np.int64)))
        if "token" in b:
            b["token"] = np.concatenate((b["token"], tokens[entity[ghosts]]))
        b["step"] = np.concatenate((b["step"], (step[ghosts] - b["start"]).astype(np.int64)))
        b["observations"] = count + len(ghosts)
    source_local = np.where(source >= lo, source - lo, hi - lo + np.searchsorted(ghosts, source))
    extra = np.stack((source_local, destination - lo))
    pairs = {
        tuple(edge): int(kind) for edge, kind in zip(b["temporal_edges"].T.tolist(), b["temporal_relation"])
    }
    for edge in extra.T.tolist():
        pairs.setdefault(tuple(edge), context)
        if pairs[tuple(edge)] == temporal:
            pairs[tuple(edge)] = context
    edges = np.asarray(list(pairs), np.int64).reshape(-1, 2)
    original_ids = np.r_[np.arange(lo, hi), ghosts]
    steps = step[original_ids]
    periods = held.numpy("obs_period")[original_ids]
    b.update(
        temporal_edges=array(edges.T, torch.long),
        temporal_relation=array(np.asarray(list(pairs.values())), torch.long),
        temporal_gap=array(steps[edges[:, 1]] - steps[edges[:, 0]], torch.float32),
        temporal_period_gap=array(periods[edges[:, 1]] - periods[edges[:, 0]], torch.float32),
    )


@timed("period_inputs.mixed_edges")
def mixed_edges(b, graph):
    mixed = b["mixed"]
    held = b["member_end_mask"]
    mask = held.reshape(-1)
    chosen = b["member_end_raw"].segments[0][2][0][1].reshape(-1)[mask]
    source = np.arange(len(chosen)) + length(mixed["x"])
    destination = np.broadcast_to(mixed["row"][:, None], held.shape).reshape(-1)[mask]
    gaps = b["member_end_gaps"].reshape(-1, 3)[mask]
    mixed["x"] = joined([mixed["x"], features(graph, (len(chosen),), ("x", chosen, 1.0))])
    mixed["kind"] = np.concatenate((mixed["kind"], np.zeros(len(chosen), np.int64)))
    mixed["edge_index"] = np.concatenate((mixed["edge_index"], np.stack((source, destination))), 1)
    mixed["edge_code"] = np.concatenate(
        (mixed["edge_code"], np.full(len(chosen), (1 << 12) | (1 << 13), np.int64))
    )
    mixed["edge_gap"] = np.concatenate((mixed["edge_gap"], np.log1p(gaps[:, :2]).astype(np.float32)))
    if "token" in mixed:
        mixed["token"] = np.concatenate(
            (mixed["token"], np.broadcast_to(b["member_token"][:, None], held.shape).reshape(-1)[mask])
        )


@timed("period_inputs.history_tokens")
def history_tokens(b, window):
    graph, held, start, cfg = window.graph, window.held, window.start, window.cfg
    count, depth = len(b["member_row"]), lookback(cfg)
    if cfg["model"]["view"] == "tree":
        roots = np.repeat(b["member_entity"].astype(np.int64), depth)
        cutoffs = ((b["end_steps"][b["member_row"]] + start) * 2 + 1).reshape(-1).astype(np.int64)
        b["end_tree"] = tree(
            graph,
            held,
            roots,
            cutoffs,
            start * 2,
            cfg["model"]["tree/neighbors"],
            cfg["model"]["tree/depth"],
            window.registry,
            cfg["graph"],
            SEED,
            window.tokens,
        )
        return
    original = b["history_mask"].shape[-1]
    valid, gaps = b["member_end_mask"], b["member_end_gaps"]
    raw_index = b["member_end_raw"].segments[0][2][0][1]
    (x_key, x_index, x_weight), (q_key, q_index, q_weight) = b["history_x"].segments[0][2]
    pieces = [
        (
            x_key,
            np.concatenate((x_index, raw_index), 1),
            np.concatenate((x_weight, np.ones(raw_index.shape, np.float32)), 1),
        ),
        (q_key, np.concatenate((q_index, np.full(raw_index.shape, -1, np.int64)), 1), q_weight),
    ]
    b["history_mask"] = np.concatenate((b["history_mask"], valid), 1)
    b["history_gap"] = np.concatenate((b["history_gap"], gaps[..., 0] * 2 + 1), 1)
    b["history_relation"] = np.concatenate((b["history_relation"], np.full((count, depth), -1, np.int64)), 1)
    end_time = np.zeros((count, original + depth, 3), np.float32)
    end_time[:, original:] = gaps
    b["history_end_time"] = end_time
    b["history_period_kind"] = np.concatenate(
        (np.zeros((count, original), np.int64), np.ones((count, depth), np.int64)), 1
    )
    for key, width in (("occurrence", 2), ("structure", 12)):
        b[key] = np.concatenate((b[key], np.zeros((count, depth, width), np.float32)), 1)
    if "history_token" in b:
        b["history_token"] = np.concatenate(
            (b["history_token"], np.repeat(b["member_token"][:, None], depth, 1)), 1
        )
    order = np.argsort(-np.where(b["history_mask"], b["history_gap"], np.inf), axis=-1, kind="stable")
    for key in (
        "history_mask",
        "history_gap",
        "history_relation",
        "history_end_time",
        "history_period_kind",
        "occurrence",
        "structure",
    ) + (("history_token",) if "history_token" in b else ()):
        value = b[key]
        b[key] = np.take_along_axis(value, order if value.ndim == 2 else order[..., None], 1)
    pieces = [
        (
            key,
            np.take_along_axis(index, order, 1),
            np.take_along_axis(weight, order, 1) if isinstance(weight, np.ndarray) else weight,
        )
        for key, index, weight in pieces
    ]
    b["history_x"] = Features([(graph, (count, original + depth), pieces)])


@timed("period_inputs.capture_schedule")
def capture_schedule(b):
    ends = b["end_steps"][b["member_row"]]
    result = []
    for time_step in range(b["time_steps"]):
        member, slot = np.nonzero((ends >= 0) & (ends % b["time_steps"] == time_step))
        result.append(((member * ends.shape[1] + slot).astype(np.int64), b["member_entity"][member]))
    b["end_capture"] = result
