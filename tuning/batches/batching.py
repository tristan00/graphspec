import logging
import time
from dataclasses import dataclass

import numpy as np

from graphspec.infra.hashing import hashed, token, unit
from graphspec.infra.timing import timed
from tuning.bank.bank import ROWS
from tuning.families.heads import FIELD
from tuning.families.periods import enabled

from .inputs import frame_inputs, recipe
from .tensors import Frames, Host, joined, length, materialise

LOG = logging.getLogger(__name__)
ENTITY_FRAMES = 1 << 20
FACTS = 4 << 20
CHUNK = 64
GRAPHS, GROUPS = "tuning.batches/graphs", "tuning.batches/groups"
POINTS = ("weight", "aim", "series")


@dataclass(frozen=True)
class Shuffle:
    seed: int
    epoch: int
    keys: dict


def identities(bank, cfg):
    started = time.perf_counter()
    names = {graph["index"]: graph["name"] for graph in bank.viewed(cfg).graphs}
    graphs = bank.graphs(cfg)
    members = bank.groups(cfg) if bank.joining(cfg) else [[graph] for graph in graphs]
    result = {
        graph: int(hashed(*(np.asarray([token(names[member])], np.uint64) for member in held))[0])
        for graph, held in zip(graphs, members, strict=True)
    }
    LOG.debug("identities exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def batches(blocks, size, sizes, shuffle, frame_budget=None, chunk=CHUNK):
    started = time.perf_counter()
    if size < 1 or chunk < 1 or (frame_budget is not None and frame_budget < 1):
        raise ValueError("batch size, chunk size and frame budget must be positive")
    blocks = [tuple(block) for block in blocks]
    graphs = list(dict.fromkeys(block[0] for block in blocks))
    if shuffle is not None:
        tokens = np.asarray([shuffle.keys[graph] for graph in graphs], np.uint64)
        drawn = unit(shuffle.seed, GRAPHS, tokens, np.full(len(graphs), shuffle.epoch))
        graphs = [graphs[index] for index in np.lexsort((tokens, drawn))]
        place = {graph: index for index, graph in enumerate(graphs)}
        blocks.sort(key=lambda block: place[block[0]])
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
        if shuffle is not None:
            first = np.asarray([group[0] for group in groups], np.int64)
            keyed = np.asarray([shuffle.keys[graph] for graph in first[:, 0]], np.uint64)
            drawn = unit(shuffle.seed, GROUPS, keyed, *first[:, 1:].T, np.full(len(groups), shuffle.epoch))
            groups = [groups[at] for at in np.argsort(drawn, kind="stable")]
        result.extend(groups)
    LOG.info(
        "batches exit seconds=%.3f blocks=%d batches=%d chunks=%d",
        time.perf_counter() - started,
        len(blocks),
        len(result),
        len(chunks),
    )
    return result


@timed("batching.combine_tree")
def combine_tree(trees):
    return {
        key: combine_tree([tree[key] for tree in trees])
        if key == "child"
        else joined([tree[key] for tree in trees])
        for key in trees[0]
    }


@timed("batching.attributed")
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


@timed("batching.shared")
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


def framed(b):
    started = time.perf_counter()
    steps = b["row_steps"][b["member_row"]]
    order = np.argsort(steps, kind="stable")
    bounds = np.searchsorted(steps[order], np.arange(b["time_steps"] + 1))
    result = [order[bounds[t] : bounds[t + 1]].astype(np.int64) for t in range(b["time_steps"])]
    LOG.debug("framed exit members=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def moved(parts, key, offsets):
    return np.concatenate(
        [np.where(p[key] >= 0, p[key] + int(shift), p[key]) for p, shift in zip(parts, offsets)]
    )


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
        left=Host(np.concatenate([p["left"].value for p in parts])),
    )
    direct = (
        "selected",
        "row_kind",
        "member_name",
        "member_place",
        "member_x",
        "member_token",
        "whole_name",
        "whole_x",
        "frame_task",
        "task",
        "entity_kind",
        "signal",
        "member_affinity",
        "global_affinity",
        "frame_affinity",
        "x",
        "kind",
        "relation",
        "temporal_gap",
        "initial",
        "history_x",
        "history_mask",
        "history_gap",
        "history_relation",
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
        "walk_relation",
        "dag_gap",
        "end_gaps",
        "end_mask",
        "member_end_raw",
        "member_end_mask",
        "member_end_gaps",
        "temporal_relation",
        "temporal_period_gap",
        "dag_relation",
        "history_period_kind",
        "history_end_time",
        "member_end_affinity",
        "end_global_affinity",
        "walk_period_kind",
        "walk_end_time",
        "entity_token",
        "projection_basis",
        "token",
        "history_token",
        "walk_token",
    )
    for key in direct:
        if key in parts[0]:
            b[key] = joined([p[key] for p in parts])
    offsets = np.r_[0, np.cumsum([p["tasks"] for p in parts])]
    member_offsets = np.r_[0, np.cumsum([len(p["member_row"]) for p in parts])]
    real, drawn = (np.r_[0, np.cumsum([len(p[key].value) for p in parts])] for key in ("rows", "drawn"))
    for key in POINTS + tuple(key for key in parts[0] if key.startswith(FIELD)):
        b[key] = np.concatenate([p[key] for p in parts])
    b["point_row"] = np.concatenate([p["point_row"] + int(offset) for p, offset in zip(parts, offsets)])
    b["point_node"] = np.concatenate(
        [np.where(p["point_node"] >= 0, p["point_node"] + int(shift), -1) for p, shift in zip(parts, member_offsets)]
    )
    b["drawn"] = Host(np.concatenate([p["drawn"].value for p in parts]))
    b["placed"] = Host(
        np.concatenate(
            [
                np.where(
                    p["placed"].value < len(p["rows"].value),
                    p["placed"].value + real[at],
                    p["placed"].value - len(p["rows"].value) + real[-1] + drawn[at],
                )
                for at, p in enumerate(parts)
            ]
        )
    )
    b["tasks"] = int(offsets[-1])
    if "end_steps" in parts[0]:
        b["end_steps"] = moved(parts, "end_steps", [i * t for i in range(size)])
        b["member_end_local"] = moved(parts, "member_end_local", nodes)
    for key in ("member_row", "whole_row"):
        b[key] = np.concatenate([p[key] + int(shift) for p, shift in zip(parts, row_offsets)])
    b["member_index"] = moved(parts, "member_index", nodes)
    for key in ("pair", "partner"):
        b[key] = moved(parts, key, member_offsets)
    for key in ("entity", "history_ids", "common_ids", "member_entity"):
        if key in parts[0]:
            b[key] = moved(parts, key, entity_offsets)
    if "attributes" in parts[0]:
        b.update(attributed(parts))
    if "window" in parts[0]:
        b.update(shared(parts))
    if "window" in parts[0] or "x" in b:
        if "frame_starts" in parts[0]:
            b["frame_starts"] = np.concatenate([p["frame_starts"] + int(offset) for p, offset in zip(parts, nodes)])
            b["frame_event"] = np.concatenate([p["frame_event"] + i * t for i, p in enumerate(parts)])
            b["frame_fed"] = moved(parts, "frame_fed", nodes)
        b["step"] = moved(parts, "step", [i * t for i in range(size)])
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
            b["dag_member_edges"] = np.concatenate(
                [
                    np.stack(
                        (
                            inverse[p["dag_member_edges"][0] + int(offset)],
                            p["dag_member_edges"][1] + int(member_offset),
                        )
                    )
                    for p, offset, member_offset in zip(parts, offsets, member_offsets)
                ],
                1,
            )
    b["frame_members"] = framed(b)
    if enabled(cfg):
        from .period_inputs import capture_schedule

        capture_schedule(b)
    LOG.debug(
        "recipe batch exit blocks=%d predictions=%d seconds=%.4f",
        size,
        b["predictions"],
        time.perf_counter() - started,
    )
    return b


@timed("batching.pack_batch")
def pack_batch(bank, blocks, cfg, device):
    return materialise(bank, cfg, recipe_batch(bank, blocks, cfg), device)
