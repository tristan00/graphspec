import logging
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from graphspec import Registry
from graphspec.budgets import GraphBudget
from graphspec.infra import workspace
from graphspec.infra.arrays import Resident, between
from graphspec.infra.hashing import hashed, salted
from graphspec.infra.timing import timed
from graphspec.operations.negatives import needed
from graphspec.operations.rows import counted, keyed_relations
from graphspec.representations.groups import PERIOD_LOOKBACK, walk_arrays
from graphspec.schema.decisions import ROLES
from graphspec.schema.references import SORTS
from tuning.bank.options import offered
from tuning.families.heads import FIELD
from tuning.families.options import realised

from . import gather
from .tensors import Frames, Host, array, features, features_at, gathered, joined

LOG = logging.getLogger(__name__)
NEIGHBOURS = "tuning.batches/neighbors"
BASIS = "tuning.families/projection"
ORDERS = dict(recent=0, uniform=1, decay=2)
CANDIDATES = ("code", "step", "source", "target")


@timed("inputs.sampling")
def sampling(spec):
    order = spec["neighbors/order"]
    return ORDERS[order], spec["neighbors/decay"] if order == "decay" else 0.0


def keyed(held):
    started = time.perf_counter()
    result = tuple(held.numpy(name) for name in ("entity_token", "step_token", "step_time", "event_identity"))
    LOG.debug("keyed exit events=%d seconds=%.4f", len(result[3]), time.perf_counter() - started)
    return result


def tree(graph, held, entities, cutoffs, floor, count, depth, registry, spec, seed, tokens=None):
    started = time.perf_counter()
    mode, decay = sampling(spec)
    events, ticks = held.numpy("events"), held.column("events", 2)
    ids, offsets = held.numpy("neighbor_ids"), held.numpy("neighbor_offsets")
    observation_ids, observation_offsets, step = (
        held.numpy("observation_ids"),
        held.numpy("observation_offsets"),
        held.numpy("step"),
    )
    drawn, identified = salted(seed, NEIGHBOURS), keyed(held)
    levels = []
    for _ in range(depth):
        table, sizes = gather.neighbor_table(
            ids, offsets, ticks, entities, cutoffs, floor, count, mode, decay, drawn, identified
        )
        children, times, mask, gaps, relation = gather.tree_level(
            events, table, sizes, entities, cutoffs, count, registry.marker("non_relation")
        )
        levels.append((entities, cutoffs, mask, gaps, relation))
        entities, cutoffs = children, times

    @timed("inputs.tree.node")
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
    sampled = (start, count, mode, decay, salted(window.seed, NEIGHBOURS), keyed(held))
    observed, interaction, task_row, mask, gap, ids, relation, structural = gather.history(
        *shared,
        entity,
        np.append(entity, -1)[partner],
        moments[rows],
        *sampled,
        include_structure,
        keyed_relations(window.registry)[1],
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
    graph, _, stop, target_start, partition, row_lo, row_hi = block
    rows, places = bank.labels(graph, cfg)
    steps = bank.points(graph, cfg)[1][places]
    picked = (steps >= target_start) & (steps < stop) & (places >= row_lo) & (places < row_hi)
    picked[picked] = bank.partitions(cfg)[rows[picked]] == partition
    LOG.debug(
        "chosen points exit graph=%d points=%d seconds=%.4f",
        graph,
        int(picked.sum()),
        time.perf_counter() - started,
    )
    return rows[picked], places[picked].astype(np.int64)


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
    seed: int

    @classmethod
    @timed("inputs.Window.open")
    def open(cls, bank, block, cfg):
        from tuning.families.model_spec import row_local, windowed

        graph, start, stop, _, _, row_lo, row_hi = block
        held = bank.resident(graph, cfg, "cpu")
        lo, hi = (int(value) for value in held.numpy("snapshot_offsets")[[start, stop]])
        if windowed(cfg["model"]):
            first, last = between(held.numpy("row_step"), start, stop)
        else:
            first, last = row_lo, row_hi
        if row_local(cfg["model"]):
            places = chosen_points(bank, block, cfg)[1]
            places = places[(places >= first) & (places < last)]
            if len(places):
                first, last = int(places.min()), int(places.max()) + 1
        tokens = bank.tokens(held) if cfg["model"].get("keys/embedding") else None
        return cls(
            graph, start, stop, first, last, lo, hi, held, bank.fitted(cfg), cfg, tokens, bank.joining(cfg), bank.seed
        )


@timed("inputs.observed_inputs")
def observed_inputs(window, lo, hi):
    entity = window.held.numpy("entity")[lo:hi]
    result = dict(
        x=features(window.graph, (hi - lo,), ("x", (lo, hi), None)),
        kind=gathered(window.graph, "entity_kind", entity),
    )
    if window.tokens is not None:
        result["token"] = array(window.tokens[entity])
    return result


@timed("inputs.frame_inputs")
def frame_inputs(window, start, stop):
    lo, hi = (int(value) for value in window.held.numpy("snapshot_offsets")[[start, stop]])
    result = dict(observed_inputs(window, lo, hi), lo=lo, observations=hi - lo)
    spatial_inputs(result, replace(window, start=start, stop=stop, lo=lo, hi=hi))
    return result


@timed("inputs.snapshot_inputs")
def snapshot_inputs(b, window):
    from tuning.families.model_spec import shared_frames, temporal

    spec = window.cfg["model"]
    start, lo, hi, held = window.start, window.lo, window.hi, window.held
    entity = held.numpy("entity")[lo:hi]
    step = held.numpy("step")[lo:hi] - start
    b.update(entity=array(entity, torch.long), step=array(step, torch.long))
    if shared_frames(spec):
        b["window"] = window
    else:
        b.update(observed_inputs(window, lo, hi))
    if temporal(spec, ("graph_sequence",)):
        b.update(frame_inputs_of(window))
    if "window" not in b:
        spatial_inputs(b, window)
    elif temporal(spec, ("graph_cell", "evolving")):
        spatial_inputs(b, window)
        for key in ("relation", "attributes", "attribute_edges"):
            del b[key]


def frame_inputs_of(window):
    started = time.perf_counter()
    graph, start, stop, lo, held = window.graph, window.start, window.stop, window.lo, window.held
    left, right = between(held.column("events", 2), start * 2, stop * 2 - 1)
    seen = held.numpy("events")[left:right]
    seen = seen[seen[:, 3] == 2]
    after = seen[:, 2].astype(np.int64) // 2 + 1
    fed = observing(held, after, seen[:, 0].astype(np.int64), np.arange(len(held.numpy("entity_kind"))))
    result = dict(
        frame_event=array(after - start, torch.long),
        frame_task=features(graph, (len(seen),), ("task", seen[:, 7], 1.0)),
        frame_starts=array(seen[:, 4:6] - lo, torch.long),
        frame_fed=array(np.where(fed >= 0, fed - lo, -1), torch.long),
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


def observing(held, steps, wanted, keys):
    started = time.perf_counter()
    offsets, entity = held.numpy("snapshot_offsets"), held.numpy("entity")
    result = np.full(len(steps), -1, np.int64)
    for step in np.unique(steps).tolist():
        lo, hi = int(offsets[step]), int(offsets[step + 1])
        if hi == lo:
            continue
        found = keys[entity[lo:hi]]
        order = np.argsort(found, kind="stable")
        chosen = np.flatnonzero(steps == step)
        at = np.minimum(np.searchsorted(found[order], wanted[chosen]), hi - lo - 1)
        known = found[order][at] == wanted[chosen]
        result[chosen[known]] = lo + order[at[known]]
    LOG.debug("observing exit points=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def drawing(schema, pipeline):
    started = time.perf_counter()
    kinds, result = list(schema.tasks["kinds"]), {}
    for kind, (operation, values) in needed(schema).items():
        found = [
            step.get("label", step["operation"].split(".", 1)[1])
            for step in pipeline
            if step["operation"] == operation
            and all(step["parameters"][name] == value for name, value in values.items())
        ]
        if len(found) != 1:
            raise ValueError(
                "the kind of task %s states a link among sampled negatives, and the pipeline draws them in %d entries "
                "of %s with %s; it draws them in one"
                % (kind, len(found), operation, ", ".join("%s %s" % pair for pair in values.items()))
            )
        result[kinds.index(kind)] = found[0]
    LOG.debug("drawing exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def negatives(window, b, owner):
    started = time.perf_counter()
    held, registry, start = window.held, window.registry, window.start
    names, entities = list(registry.references), len(held.numpy("entity_kind"))
    stated = list(registry.schema.tasks["kinds"].values())
    kinds, link, drawn = b["row_kind"][owner].astype(np.int64), [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    entries, candidates = drawing(registry.schema, window.cfg["pipeline"]), {}
    if entries:
        left, right = between(held.numpy("candidate_step"), start, window.stop)
        candidates = {name: held.numpy("candidate_" + name)[left:right].astype(np.int64) for name in CANDIDATES}
    for code, entry in entries.items():
        chosen = np.flatnonzero(kinds == code)
        referred = np.flatnonzero(b["member_name"] == names.index(stated[code]["target"]["about"]))
        anchor = np.full(len(b["row_kind"]), -1, np.int64)
        anchor[b["member_row"][referred]] = b["member_entity"][referred]
        drawing_entry = candidates["code"] == registry.by_candidate[entry].code
        keys = candidates["step"][drawing_entry] * entities + candidates["source"][drawing_entry]
        order = np.argsort(keys, kind="stable")
        known = anchor[owner[chosen]] >= 0
        wanted = (b["row_steps"][owner[chosen]] + start) * entities + anchor[owner[chosen]]
        low = np.where(known, np.searchsorted(keys[order], wanted, "left"), 0)
        high = np.where(known, np.searchsorted(keys[order], wanted, "right"), 0)
        link.append(np.repeat(chosen, high - low))
        drawn.append(candidates["target"][drawing_entry][order[np.repeat(low, high - low) + counted(high - low)]])
    result = np.concatenate(link), np.concatenate(drawn)
    LOG.debug(
        "negatives exit graph=%d drawn=%d seconds=%.4f", window.graph, len(result[0]), time.perf_counter() - started
    )
    return result


def pointed(bank, window, b, rows, owner):
    started = time.perf_counter()
    held, registry, start = window.held, window.registry, window.start
    tokens, kinds = bank.nodes[rows], b["row_kind"][owner].astype(np.int64)
    left = registry.linked[kinds] & held.numpy("row_left")[window.first + owner]
    named = np.flatnonzero((tokens >= 0) & ~left)
    row = owner[named]
    observed = observing(held, b["row_steps"][row] + start, tokens[named], held.numpy("entity_token"))
    lost = np.flatnonzero((observed < 0) & registry.stops[kinds[named]])
    if len(lost):
        found = bank.rows[rows[named[lost[0]]]]
        raise ValueError(
            "the node %s that a point of the task %s of the kind %s of graph %s names is not observed at the time "
            "of its task, and the kind states absent as stop"
            % (found["node"], found[registry.schema.tasks["key"]], found["kind"], found["graph"])
        )
    point, drawn = negatives(window, b, owner)
    point, drawn = point[~left[point]], drawn[~left[point]]
    moments = b["row_steps"][owner[point]] + start
    placed = observing(held, moments, drawn, np.arange(len(held.numpy("entity_kind"))))
    if np.any(placed < 0):
        raise ValueError("a negative drawn for graph %d is not observed at the step of its task" % window.graph)
    noted = np.full(len(rows), -1, np.int64)
    noted[named] = observed
    noted, owners = np.r_[noted, placed], np.r_[owner, owner[point]]
    for key in tuple(key for key in b if key.startswith(FIELD)):
        b[key] = np.r_[b[key], np.full(len(point), np.nan, b[key].dtype)]
    for key in ("weight", "series", "aim", "point_row"):
        b[key] = np.r_[b[key], b[key][point]]
    linking = registry.linked[b["row_kind"][owners].astype(np.int64)]
    marks = held.numpy("entity_token")
    keyed, linking = np.zeros(len(noted), marks.dtype), linking & (noted >= 0)
    keyed[linking] = marks[held.numpy("entity")[noted[linking]]]
    order = np.lexsort((np.arange(len(noted)), keyed, b["point_row"]))
    for key in ("weight", "series", "aim", "point_row") + tuple(key for key in b if key.startswith(FIELD)):
        b[key] = b[key][order]
    having = np.flatnonzero(noted[order] >= 0)
    point_node = np.full(len(noted), -1, np.int64)
    point_node[having] = len(b["member_row"]) + np.arange(len(having))
    observed, row = noted[order][having], owners[order][having]
    entity = held.numpy("entity")[observed]
    b["drawn"], b["placed"], b["left"] = Host(rows[point]), Host(order), Host(left)
    b.update(
        member_row=np.concatenate((b["member_row"], row)),
        member_name=np.concatenate((b["member_name"], np.full(len(row), len(registry.references), np.int64))),
        member_entity=np.concatenate((b["member_entity"], entity.astype(np.int64))),
        member_index=np.concatenate((b["member_index"], observed - window.lo)),
        member_place=np.concatenate((b["member_place"], np.zeros(len(row), np.int64))),
        member_stored=np.concatenate((b["member_stored"], np.full(len(row), -1, np.int64))),
        member_x=joined([b["member_x"], features(window.graph, (len(row),), ("x", observed, 1.0))]),
        partner=np.concatenate((b["partner"], np.full(len(row), -1, np.int64))),
        point_node=point_node,
    )
    LOG.debug(
        "pointed exit graph=%d points=%d nodes=%d seconds=%.4f",
        window.graph,
        len(rows),
        len(row),
        time.perf_counter() - started,
    )


def edged(window, moments, ends, relation):
    started = time.perf_counter()
    held = window.held
    low, high = between(held.numpy("spatial_step"), window.start, window.stop)
    spatial, entity = held.numpy("spatial")[low:high].astype(np.int64), held.numpy("entity").astype(np.int64)
    shown = hashed(
        held.numpy("spatial_step")[low:high].astype(np.int64), entity[spatial[:, 0]], entity[spatial[:, 1]], spatial[:, 2]
    )
    result = np.isin(hashed(moments, entity[ends[:, 0]], entity[ends[:, 1]], relation), shown)
    LOG.debug("edged exit graph=%d edges=%d seconds=%.4f", window.graph, len(result), time.perf_counter() - started)
    return result


def optioned(bank, window, b, rows, owner, place):
    started = time.perf_counter()
    held, registry, found = window.held, window.registry, offered(bank)
    decision = found.decision[rows]
    chosen = np.flatnonzero(decision >= 0)
    ids, first = np.unique(decision[chosen], return_index=True)
    task, row = owner[chosen[first]], place[chosen[first]]
    option = np.repeat(found.first[ids], found.count[ids]) + counted(found.count[ids])
    decided = np.repeat(np.arange(len(ids)), found.count[ids])
    action = np.repeat(found.actions[0][option], found.actions[1][option]) + counted(found.actions[1][option])
    acting = np.repeat(np.arange(len(option)), found.actions[1][option])
    role = np.repeat(found.roles[0][action], found.roles[1][action]) + counted(found.roles[1][action])
    roling = np.repeat(np.arange(len(action)), found.roles[1][action])
    sort = found.sorts[found.role["code"][role]]
    moments = b["row_steps"][task[decided[acting[roling]]]].astype(np.int64) + window.start
    ends = np.stack((found.role["source"][role], found.role["target"][role]), 1)
    observed = np.full(ends.shape, -1, np.int64)
    for column, sorts in enumerate(((ROLES.index("node"), ROLES.index("edge")), (ROLES.index("edge"),))):
        wanted = np.flatnonzero(np.isin(sort, sorts))
        observed[wanted, column] = observing(held, moments[wanted], ends[wanted, column], held.numpy("entity_token"))
    edge = np.flatnonzero(sort == ROLES.index("edge"))
    unheld = (observed[:, 0] < 0) & (sort != ROLES.index("new"))
    whole = edge[(observed[edge] >= 0).all(1)]
    unheld[edge] |= observed[edge, 1] < 0
    unheld[whole] |= ~edged(window, moments[whole], observed[whole], found.role["relation"][role[whole]])
    left = np.zeros(len(option), bool)
    np.logical_or.at(left, acting[roling], unheld)
    kept = np.flatnonzero(~left)
    acts = np.flatnonzero(~left[acting])
    roles = np.flatnonzero(~left[acting[roling]])
    local = np.full(len(option), -1, np.int64)
    local[kept] = np.arange(len(kept))
    taken = np.full(len(action), -1, np.int64)
    taken[acts] = np.arange(len(acts))
    ends = np.where(sort[roles, None] == ROLES.index("new"), -1, observed[roles])
    ends[:, 1] = np.where(sort[roles] == ROLES.index("edge"), ends[:, 1], -1)
    shown = ends >= 0
    members = ends[shown]
    member = np.full(ends.shape, -1, np.int64)
    member[shown] = len(b["member_row"]) + np.arange(len(members))
    owners = np.broadcast_to(task[decided[acting[roling[roles]]]][:, None], ends.shape)[shown]
    b.update(
        member_row=np.concatenate((b["member_row"], owners)),
        member_name=np.concatenate((b["member_name"], np.full(len(members), len(registry.references) + 1, np.int64))),
        member_entity=np.concatenate((b["member_entity"], held.numpy("entity")[members].astype(np.int64))),
        member_index=np.concatenate((b["member_index"], members - window.lo)),
        member_place=np.concatenate((b["member_place"], np.zeros(len(members), np.int64))),
        member_stored=np.concatenate((b["member_stored"], np.full(len(members), -1, np.int64))),
        member_x=joined([b["member_x"], features(window.graph, (len(members),), ("x", members, 1.0))]),
        partner=np.concatenate((b["partner"], np.full(len(members), -1, np.int64))),
    )
    order = np.asarray(b["placed"].value, np.int64)
    source = np.where(order < len(rows), rows[np.minimum(order, len(rows) - 1)], -1)
    pointing = np.where(source >= 0, found.row_option[np.maximum(source, 0)], -1)
    acted = np.where(source >= 0, found.row_action[np.maximum(source, 0)], -1)
    globally, places = np.argsort(option, kind="stable"), np.argsort(action, kind="stable")
    point_option = local[globally[np.minimum(np.searchsorted(option[globally], pointing), len(option) - 1)]]
    point_action = taken[places[np.minimum(np.searchsorted(action[places], acted), len(action) - 1)]]
    lost = np.isin(found.row_option[rows], option[left])
    b["left"] = Host(np.asarray(b["left"].value, bool) | lost)
    columns = registry.schema.parameters()
    b["options"] = dict(
        row=row[decided[kept]],
        decision=decided[kept],
        sequence=found.option["sequence"][option[kept]],
        action_option=local[acting[acts]],
        action_kind=found.action["kind"][action[acts]],
        action_place=found.action["place"][action[acts]],
        action_values=registry.encoding.vectors(columns, found.action["values"][action[acts]]),
        role_action=taken[roling[roles]],
        role_code=found.role["code"][role[roles]],
        role_member=member,
        role_relation=np.where(sort[roles] == ROLES.index("edge"), found.role["relation"][role[roles]], -1),
        point_option=np.where(pointing >= 0, point_option, -1),
        point_action=np.where((acted >= 0) & (pointing >= 0) & (point_option >= 0), point_action, -1),
        content=Host(found.option["content"][option[kept]]),
        kind=Host(found.option["kind"][option[kept]]),
        left=Host(found.option["kind"][option[left]]),
    )
    LOG.debug(
        "optioned exit graph=%d decisions=%d options=%d left=%d seconds=%.4f",
        window.graph,
        len(ids),
        len(kept),
        int(left.sum()),
        time.perf_counter() - started,
    )


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


@timed("inputs.spatial_inputs")
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


@timed("inputs.lifted_inputs")
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


@timed("inputs.mixed_inputs")
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
        b["mixed"]["token"] = np.r_[workspace.gathered(tokens, owner, -1), b["member_token"]]


@timed("inputs.events_inputs")
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


@timed("inputs.dag_inputs")
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
    kinds = events[dag[1, : len(linked)], 3] + window.registry.relation_slots
    b.update(
        dag_edges=array(dag),
        dag_relation=array(np.concatenate((kinds, extra[:, 2])), torch.long),
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


def walked(bank, window, roots, moments):
    started = time.perf_counter()
    held, roots, moments = window.held, np.asarray(roots, np.int64), np.asarray(moments, np.int64)
    prior = held.numpy("period_membership")[moments][:, None].astype(np.int64) - np.arange(1, PERIOD_LOOKBACK + 1)
    starts, ends = held.numpy("period_starts"), held.numpy("period_ends")
    result = walk_arrays(
        held.numpy("neighbor_ids"),
        held.numpy("neighbor_offsets"),
        held.numpy("events"),
        held.numpy("event_identity"),
        roots,
        hashed(held.numpy("entity_token")[roots], held.numpy("step_time")[moments], held.numpy("step_token")[moments]),
        moments,
        prior,
        starts[prior.clip(0)].astype(np.int64),
        ends[prior.clip(0)].astype(np.int64),
        window.cfg["graph"],
        window.seed,
        GraphBudget(bank.limits()),
    )
    LOG.debug("walked exit roots=%d seconds=%.4f", len(roots), time.perf_counter() - started)
    return result


@timed("inputs.walks_inputs")
def walks_inputs(b, window, context, bank):
    from tuning.families.periods import enabled, lookback

    graph, start = window.graph, window.start
    held, cfg, tokens = window.held, window.cfg, window.tokens
    chosen = b["member_stored"]
    moment = b["row_steps"][b["member_row"]] + start
    walk_count, hops = cfg["model"]["walks/count"], cfg["model"]["walks/hops"]
    stored = held.numpy("walks")
    if walk_count > stored.shape[1] or hops > stored.shape[2]:
        raise ValueError("requested walks exceed the walks stored for this graph variant")
    drawn = walked(bank, window, b["member_entity"][chosen < 0], moment[chosen < 0]) if np.any(chosen < 0) else None
    found = {name: held.numpy(name)[np.maximum(chosen, 0)] for name in ("walks", "period_walks")[: 1 + enabled(cfg)]}
    for name, values in found.items():
        if drawn is not None:
            values[chosen < 0] = drawn[name]
    walks = found["walks"][:, :walk_count, :hops]
    if enabled(cfg):
        extra = found["period_walks"][:, : lookback(cfg), :walk_count, :hops]
        walks = np.concatenate((walks, extra.reshape(len(chosen), lookback(cfg) * walk_count, hops)), 1)
        kinds = np.zeros(walks.shape, np.int64)
        kinds[:, walk_count:] = 1
        gaps = np.zeros(walks.shape + (3,), np.float32)
        gaps[:, walk_count:] = np.repeat(context["member_end_gaps"], walk_count, axis=1)[:, :, None, :]
        b.update(walk_period_kind=kinds, walk_end_time=gaps)
    events = workspace.gathered(held.numpy("events"), walks, -1)
    known = (walks >= 0) & (events[..., 2] >= start * 2)
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
        b["walk_token"] = workspace.gathered(tokens, np.where(known, nodes, -1), -1)


@timed("inputs.base_inputs")
def base_inputs(bank, block, window):
    graph, start, stop = window.graph, window.start, window.stop
    first, last, lo, hi = window.first, window.last, window.lo, window.hi
    held, tokens = window.held, window.tokens
    n = stop - start
    row_steps = held.numpy("row_step")
    m = last - first
    moment = row_steps[first:last].astype(np.int64)
    rows, places = chosen_points(bank, block, window.cfg)
    if np.any(places < first) or np.any(places >= last):
        raise ValueError("a point of graph %d lies outside its window" % graph)
    selected = np.isin(np.arange(first, last), places)
    point_row = np.searchsorted(np.flatnonzero(selected) + first, places)
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
        drawn=Host(np.zeros(0, np.int64)),
        weight=array(bank.weights[rows]),
        series=array(bank.series[rows], torch.long),
        aim=array(bank.aims[rows], torch.long),
        point_row=array(point_row, torch.long),
        tasks=int(selected.sum()),
        task=features(graph, (m,), ("task", held.numpy("row_task")[first:last], 1.0)),
        row_kind=array(kinds, torch.long),
        **member_inputs(window, first, last),
        **{FIELD + name: array(values[rows]) for name, values in bank.inputs.items()},
    )
    b["pair"] = paired(window.registry, kinds, b["member_row"], b["member_name"])
    b["partner"] = partnered(b["pair"], len(b["member_row"]))
    pointed(bank, window, b, rows, np.flatnonzero(selected)[point_row])
    if realised(window.cfg["model"]) is not None:
        optioned(bank, window, b, rows, np.flatnonzero(selected)[point_row], point_row)
    if tokens is not None:
        b["member_token"] = array(tokens[b["member_entity"]])
    return b


@timed("inputs.tree_inputs")
def tree_inputs(b, window):
    from tuning.families.model_spec import neighbors

    graph, start = window.graph, window.start
    held, registry, cfg, tokens = window.held, window.registry, window.cfg, window.tokens
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
        cfg["graph"],
        window.seed,
        tokens,
    )


@timed("inputs.neighbor_history_inputs")
def neighbor_history_inputs(b, window, blocks):
    from tuning.families.model_spec import neighbors

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
        b["history_token"] = workspace.gathered(tokens, b["history_ids"], -1)


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


@timed("inputs.prepare_context")
def prepare_context(window, b):
    from tuning.families.periods import enabled

    from .period_inputs import add

    result = {}
    if enabled(window.cfg):
        add(result, window, b)
    return result


@timed("inputs.context_inputs")
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


def basis(window, entities):
    started = time.perf_counter()
    held, cfg = window.held, window.cfg
    offsets, order, step = held.numpy("observation_offsets"), held.numpy("observation_ids"), held.numpy("step")
    observed = offsets[entities + 1] > offsets[entities]
    homes = np.zeros(len(entities), np.uint64)
    homes[observed] = held.numpy("step_token")[step[order[offsets[entities[observed]]]]]
    result = gather.gaussian(
        salted(cfg["seed"], BASIS), held.numpy("entity_token")[entities], homes, cfg["model"]["projection/width"]
    )
    LOG.debug("basis exit entities=%d seconds=%.4f", len(entities), time.perf_counter() - started)
    return result


@timed("inputs.entity_space")
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
    if window.cfg["model"]["view"] == "projection":
        b["projection_basis"] = basis(window, np.arange(size) if space is None else space.astype(np.int64))
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
    from tuning.families.model_spec import blocks as model_blocks

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
        walks_inputs(b, window, context, bank)
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
