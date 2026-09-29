import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.references import SORTS

from .data import ROWS
from .inputs import frame_inputs, recipe
from .periods import enabled
from .tensors import Frames, Host, joined, length, materialise

LOG = logging.getLogger(__name__)
ENTITY_FRAMES = 1 << 20
FACTS = 4 << 20
CHUNK = 64


def batches(blocks, size, sizes, seed=None, frame_budget=None, chunk=CHUNK):
    started = time.perf_counter()
    if size < 1 or chunk < 1 or (frame_budget is not None and frame_budget < 1):
        raise ValueError("batch size, chunk size and frame budget must be positive")
    rng = np.random.default_rng(seed) if seed is not None else None
    blocks = [tuple(block) for block in blocks]
    graphs = list(dict.fromkeys(block[0] for block in blocks))
    if rng is not None:
        rng.shuffle(graphs)
    members = max((sizes[graph][0].members for graph in graphs if callable(sizes[graph][0])), default=1)
    position = {graph: index // max(1, chunk // members) for index, graph in enumerate(graphs)}
    counted, grouped = {}, {}
    for block in blocks:
        if callable(sizes[block[0]][0]):
            grouped.setdefault(block[0], []).append(block)
    for graph, chosen in grouped.items():
        counts = sizes[graph][0]([block[1] for block in chosen], [block[2] for block in chosen])
        counted.update(zip(chosen, counts.tolist()))
    chunks = {}
    for block in blocks:
        chunks.setdefault(position[block[0]], {}).setdefault(block[2] - block[1], []).append(block)
    result = []
    for index in sorted(chunks):
        groups = []
        for frames, values in chunks[index].items():
            limit = min(size, max(1, frame_budget // frames)) if frame_budget is not None else size
            group, rows, cells, facts = [], 0, 0, 0.0
            for block in values:
                entities, rate = sizes[block[0]]
                if callable(entities):
                    entities = counted[block]
                width, span = block[6] - block[5], frames * entities
                if group and (
                    len(group) == limit
                    or rows + width > ROWS
                    or cells + span > ENTITY_FRAMES
                    or facts + frames * rate > FACTS
                ):
                    groups.append(group)
                    group, rows, cells, facts = [], 0, 0, 0.0
                group.append(block)
                rows += width
                cells += span
                facts += frames * rate
            groups.append(group)
        if rng is not None:
            rng.shuffle(groups)
        result.extend(groups)
    LOG.info(
        "batches exit seconds=%.3f blocks=%d batches=%d chunks=%d",
        time.perf_counter() - started,
        len(blocks),
        len(result),
        len(chunks),
    )
    return result


def combine_tree(trees):
    return {
        key: combine_tree([tree[key] for tree in trees])
        if key == "child"
        else joined([tree[key] for tree in trees])
        for key in trees[0]
    }


def attributed(parts):
    edge_offsets = np.r_[0, np.cumsum([len(p["relation"]) for p in parts])]
    names = list(parts[0]["attributes"])
    return dict(
        attribute_values=(
            np.concatenate([p["attributes"][name].reshape(-1) for name in names for p in parts])
            if names
            else np.empty(0, np.float32)
        ),
        attribute_edges=(
            np.concatenate(
                [
                    p["attribute_edges"][name] + int(offset)
                    for name in names
                    for p, offset in zip(parts, edge_offsets)
                ]
            )
            if names
            else np.empty(0, np.int64)
        ),
        attribute_counts=Host([sum(len(p["attributes"][name]) for p in parts) for name in names]),
    )


def shared(parts):
    spans = {}
    for p in parts:
        window = p["window"]
        spans.setdefault(window.graph, []).append((window.start, window.stop))
    frames, place, total = [], {}, 0
    for graph, found in spans.items():
        window = next(p["window"] for p in parts if p["window"].graph == graph)
        merged = []
        for start, stop in sorted(found):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        for start, stop in merged:
            frame = frame_inputs(window, start, stop)
            frames.append(frame)
            place[graph, start] = (stop, frame["lo"], total)
            total += frame["observations"]
    nodes = np.r_[0, np.cumsum([frame["observations"] for frame in frames])]
    expand = []
    for p in parts:
        window = p["window"]
        start = max(key[1] for key in place if key[0] == window.graph and key[1] <= window.start)
        _, lo, offset = place[window.graph, start]
        expand.append(offset + window.lo - lo + np.arange(window.hi - window.lo))
    unique = dict(
        x=joined([frame["x"] for frame in frames]),
        kind=joined([frame["kind"] for frame in frames]),
        edges=np.concatenate([frame["edges"] + int(offset) for frame, offset in zip(frames, nodes)], 1),
        relation=joined([frame["relation"] for frame in frames]),
        **attributed(frames),
    )
    if "token" in frames[0]:
        unique["token"] = joined([frame["token"] for frame in frames])
    return dict(unique=unique, expand=np.concatenate(expand))


def referred(registry, b):
    started = time.perf_counter()
    result = {}
    for code, (name, entry) in enumerate(registry.schema.tasks["references"].items()):
        if SORTS.named(entry["sort"]).whole:
            result[name] = dict(row=b["whole_row"][b["whole_name"] == code])
        else:
            chosen = b["member_name"] == code
            result[name] = dict(
                row=b["member_row"][chosen],
                entity=b["member_entity"][chosen],
                index=b["member_index"][chosen],
            )
    for key in ("member_row", "member_name", "member_entity", "member_index", "whole_row", "whole_name"):
        del b[key]
    LOG.debug("referred exit references=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("recipe_batch")
def recipe_batch(bank, blocks, cfg):
    started = time.perf_counter()
    if not blocks:
        raise ValueError("a batch requires at least one window")
    parts = [recipe(bank, block, cfg) for block in blocks]
    t, size = parts[0]["frames"], len(parts)
    if any(p["frames"] != t for p in parts):
        raise ValueError("a batch must contain equal-length independent sequences")
    entity_offsets = np.r_[0, np.cumsum([p["entities"] for p in parts])]
    nodes = np.r_[0, np.cumsum([p["observations"] for p in parts])]
    row_offsets = np.r_[0, np.cumsum([p["predictions"] for p in parts])]
    total_entities = int(entity_offsets[-1])
    b = dict(
        frames=t * size,
        time_steps=t,
        batch_size=size,
        predictions=int(row_offsets[-1]),
        entities=total_entities,
        start=0,
        stop=t,
        graph=parts[0]["graph"],
        graphs=[p["graph"] for p in parts],
        entity_sizes=[p["entities"] for p in parts],
        entity_batch=np.repeat(np.arange(size), [p["entities"] for p in parts]),
        row_steps=np.concatenate([p["row_steps"] for p in parts]),
        row_frames=np.concatenate([p["row_steps"] + i * t for i, p in enumerate(parts)]),
        frame_rows=[
            np.concatenate([p["frame_rows"][time] + int(offset) for p, offset in zip(parts, row_offsets)])
            for time in range(t)
        ],
        rows=Host(np.concatenate([p["rows"].value for p in parts])),
    )
    direct = (
        "selected",
        "target",
        "weight",
        "tau",
        "series",
        "member_name",
        "whole_name",
        "frame_task",
        "task",
        "start_x",
        "entity_kind",
        "signal",
        "affinity",
        "global_affinity",
        "x",
        "kind",
        "relation",
        "temporal_gap",
        "initial",
        "history_x",
        "history_mask",
        "history_gap",
        "occurrence",
        "structure",
        "common_stats",
        "common_frequency",
        "pair_gap",
        "pair_mask",
        "walk_x",
        "walk_mask",
        "walk_gap",
        "walk_identity",
        "dag_gap",
        "end_gaps",
        "end_mask",
        "end_raw",
        "end_raw_mask",
        "end_token_gaps",
        "temporal_relation",
        "temporal_period_gap",
        "dag_relation",
        "history_period_kind",
        "history_end_time",
        "end_affinity",
        "end_global_affinity",
        "walk_period_kind",
        "walk_end_time",
        "entity_token",
        "root_token",
        "token",
        "history_token",
        "walk_token",
    )
    for key in direct:
        if key in parts[0]:
            b[key] = joined([p[key] for p in parts])
    if "point_row" in parts[0]:
        offsets = np.r_[0, np.cumsum([p["tasks"] for p in parts])]
        b["point_row"] = np.concatenate([p["point_row"] + int(offset) for p, offset in zip(parts, offsets)])
        b["tasks"] = int(offsets[-1])
    if "end_steps" in parts[0]:
        b["end_steps"] = np.concatenate(
            [np.where(p["end_steps"] >= 0, p["end_steps"] + i * t, -1) for i, p in enumerate(parts)]
        )
        b["end_observation_local"] = np.concatenate(
            [
                np.where(p["end_observation_local"] >= 0, p["end_observation_local"] + int(offset), -1)
                for p, offset in zip(parts, nodes)
            ]
        )
    for key in ("member_row", "whole_row"):
        b[key] = np.concatenate([p[key] + int(offset) for p, offset in zip(parts, row_offsets)])
    b["member_index"] = np.concatenate(
        [
            np.where(p["member_index"] >= 0, p["member_index"] + int(offset), -1)
            for p, offset in zip(parts, nodes)
        ]
    )
    for key in ("first", "second", "entity", "history_ids", "common_ids", "member_entity"):
        if key in parts[0]:
            b[key] = np.concatenate(
                [
                    np.where(p[key] >= 0, p[key] + int(offset), p[key])
                    for p, offset in zip(parts, entity_offsets)
                ]
            )
    if "attributes" in parts[0]:
        b.update(attributed(parts))
    if "window" in parts[0]:
        b.update(shared(parts))
    if "window" in parts[0] or "x" in b:
        b["start_index"] = np.concatenate(
            [
                np.where(p["start_index"] >= 0, p["start_index"] + int(offset), -1)
                for p, offset in zip(parts, nodes)
            ]
        )
        if "frame_starts" in parts[0]:
            b["frame_starts"] = np.concatenate(
                [
                    np.where(p["frame_starts"] >= 0, p["frame_starts"] + int(offset), -1)
                    for p, offset in zip(parts, nodes)
                ]
            )
        b["step"] = np.concatenate([p["step"] + i * t for i, p in enumerate(parts)])
        b["align"] = np.concatenate(
            [
                np.where(p["align"] < p["observations"], p["align"] + int(offset), int(nodes[-1]))
                for p, offset in zip(parts, nodes)
            ],
            1,
        )
        b["present"] = np.concatenate([p["present"] for p in parts], 1)
        for key in ("edges", "temporal_edges"):
            if key in parts[0]:
                b[key] = np.concatenate([p[key] + int(offset) for p, offset in zip(parts, nodes)], 1)
    if "frame_edges" in parts[0]:
        edge_offsets = np.r_[0, np.cumsum([p["edges"].shape[1] for p in parts])]
        canonical_offsets = np.r_[0, np.cumsum([int(p["frame_edges"].directed.sum()) for p in parts])]
        directed = np.concatenate([p["frame_edges"].directed for p in parts])
        for key, offsets, canonical in (
            ("frame_edges", edge_offsets, False),
            ("frame_directed", canonical_offsets, True),
        ):
            b[key] = Frames(
                directed,
                [
                    [
                        (lo + int(offset), hi + int(offset))
                        for p, offset in zip(parts, offsets)
                        for lo, hi in p[key].ranges[step]
                    ]
                    for step in range(t)
                ],
                canonical,
            )
    if "tree" in parts[0]:
        b["tree"] = combine_tree([p["tree"] for p in parts])
    if "end_tree" in parts[0]:
        b["end_tree"] = combine_tree([p["end_tree"] for p in parts])
    if "mixed" in parts[0]:
        offsets = np.r_[0, np.cumsum([length(p["mixed"]["x"]) for p in parts])]
        b["mixed"] = {
            key: joined([p["mixed"][key] for p in parts])
            for key in ("x", "kind", "edge_code", "edge_gap")
            + (("token",) if "token" in parts[0]["mixed"] else ())
        }
        b["mixed"]["edge_index"] = np.concatenate(
            [p["mixed"]["edge_index"] + int(offset) for p, offset in zip(parts, offsets)], 1
        )
        b["mixed"]["row"] = np.concatenate(
            [p["mixed"]["row"] + int(offset) for p, offset in zip(parts, offsets)]
        )
    if "events" in parts[0]:
        events = []
        for p, offset in zip(parts, entity_offsets):
            event = p["events"].copy()
            event[:, :2] += int(offset)
            event[:, 2] -= p["start"] * 2
            events.append(event)
        events = np.concatenate(events)
        order = np.argsort(events[:, 2], kind="stable")
        b["events"], b["event_x"] = events[order], joined([p["event_x"] for p in parts], order=order)
        times = b["events"][:, 2]
        starts = np.flatnonzero(np.diff(times, prepend=-1))
        b["event_groups"] = [
            (int(times[left]), int(left), int(right))
            for left, right in zip(starts, np.append(starts[1:], len(times)))
        ]
        if "dag_edges" in parts[0]:
            offsets = np.r_[0, np.cumsum([len(p["events"]) for p in parts])]
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            b["dag_edges"] = inverse[
                np.concatenate([p["dag_edges"] + int(offset) for p, offset in zip(parts, offsets)], 1)
            ]
            b["dag_row_edges"] = np.concatenate(
                [
                    np.stack(
                        (
                            inverse[p["dag_row_edges"][0] + int(offset)],
                            p["dag_row_edges"][1] + int(row_offset),
                        )
                    )
                    for p, offset, row_offset in zip(parts, offsets, row_offsets)
                ],
                1,
            )
    if enabled(cfg):
        from .period_inputs import capture_schedule

        capture_schedule(b)
    b["references"] = referred(bank.registry, b)
    LOG.debug(
        "recipe batch exit blocks=%d predictions=%d seconds=%.4f",
        size,
        b["predictions"],
        time.perf_counter() - started,
    )
    return b


def pack_batch(bank, blocks, cfg, device):
    return materialise(bank, cfg, recipe_batch(bank, blocks, cfg), device)
