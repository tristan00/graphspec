import logging
import time

import numpy as np

from graphspec.facts.stored import GROUPS, encoded, ranks, unheld
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema import time as clock
from graphspec.schema.decisions import ROLES, roled
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import ABOUT, ALWAYS, NEVER, SEVERAL

from . import spans, visible
from .graph import Graph, Shown, verify
from .visible import permitted

LOG = logging.getLogger(__name__)
VERSION = 79
STEPS = Names("choice of steps", True)


@timed("view.sequence")
def sequence(observer, moments, looking, tasks, task_step):
    return dict(
        observer=int(observer),
        moments=np.asarray(moments, np.int64),
        looking=np.asarray(looking, np.int64),
        tasks=np.asarray(tasks, np.int64),
        task_step=np.asarray(task_step, np.int32),
    )


@timed("view.at")
def at(graph, moments, times):
    times = np.asarray(times)
    found = np.searchsorted(moments, times, "right") - 1
    inside = found >= 0
    absent = ~inside
    absent[inside] = moments[found[inside]] != times[inside]
    if absent.any():
        raise ValueError(
            "a task of graph %s sits at %s, which no step of the view shows"
            % (graph, clock.shown(np.asarray(times)[absent][0]))
        )
    return found


class Tasks:
    version = 1

    @timed("view.tasks.serves")
    def serves(self, registry):
        return True

    @timed("view.tasks.refusal")
    def refusal(self, registry):
        untasked = registry.extras["untasked"]
        if not untasked:
            return None
        return "shows a graph at the times of its tasks, and these graphs of the bank hold no task: " + ", ".join(
            untasked
        )

    def __call__(self, facts, registry, tasks):
        started = time.perf_counter()
        looking = np.asarray(facts.task_observer)[tasks]
        if not len(looking):
            raise ValueError(
                "graph %s holds no task, and the choice of steps %s shows the times of the tasks of a graph; "
                "a graph without tasks is shown by another choice, such as %s" % (facts.name, TASKS, CHANGES)
            )
        times = np.asarray(facts.task_time)
        result = []
        for observer in np.unique(looking).tolist():
            chosen = tasks[looking == observer]
            moments = np.unique(times[chosen])
            result.append(
                sequence(observer, moments, moments, chosen, at(facts.name, moments, times[chosen]))
            )
        LOG.debug(
            "steps of tasks exit graph=%s sequences=%d seconds=%.4f",
            facts.name,
            len(result),
            time.perf_counter() - started,
        )
        return result


class Changes:
    version = 1

    @timed("view.changes.serves")
    def serves(self, registry):
        return True

    @timed("view.changes.refusal")
    def refusal(self, registry):
        return None

    def __call__(self, facts, registry, tasks):
        started = time.perf_counter()
        times = []
        for group in GROUPS:
            times += [
                np.asarray(facts.arrays[group + "_" + name], np.int64)
                for name in ("since", "until", "known_time", "replaced_time")
            ]
        looking, held = np.asarray(facts.task_observer)[tasks], np.asarray(facts.task_time, np.int64)
        result = []
        for observer in np.unique(looking).tolist() if len(looking) else [-1]:
            chosen = tasks[looking == observer]
            found = np.concatenate(times + [held[chosen]])
            moments = np.unique(found[(found > ALWAYS) & (found < NEVER)])
            if len(moments):
                result.append(
                    sequence(observer, moments, moments, chosen, at(facts.name, moments, held[chosen]))
                )
        LOG.debug(
            "steps of changes exit graph=%s sequences=%d seconds=%.4f",
            facts.name,
            len(result),
            time.perf_counter() - started,
        )
        return result


TASKS = "tasks"
CHANGES = "changes"
STEPS.register(TASKS, Tasks())
STEPS.register(CHANGES, Changes())


@timed("view.offered")
def offered(registry):
    return tuple(name for name in STEPS.names() if STEPS.named(name).serves(registry))


def stepped(facts, registry, steps, tasks):
    started = time.perf_counter()
    found = STEPS.named(steps)(facts, registry, tasks)
    owner = np.asarray(facts.target_task)
    placed = np.concatenate([held["tasks"] for held in found] + [np.zeros(0, np.int64)])
    if len(np.unique(placed)) != len(placed) or not np.isin(owner[np.isin(owner, tasks)], placed).all():
        raise ValueError(
            "the choice of steps %s does not give every task of graph %s that holds a target one step"
            % (steps, facts.name)
        )
    LOG.debug("stepped exit graph=%s tasks=%d seconds=%.4f", facts.name, len(tasks), time.perf_counter() - started)
    return {held["observer"]: held for held in found}


def sequences(facts, registry, steps, looking):
    started = time.perf_counter()
    if not STEPS.named(steps).serves(registry):
        raise ValueError("the choice of steps %s does not serve the graphs of this bank" % steps)
    split, owner = np.asarray(facts.task_split), np.asarray(facts.target_task)
    found = stepped(facts, registry, steps, np.arange(len(split)))
    ruled = [code for code, rules in looking.rules.items() if rules]
    unruled = tuple(code for code in range(len(looking.splits)) if code not in ruled)
    shared = looking.trained if set(unruled) & set(looking.trained) else unruled + looking.trained
    common = (split < 0) | np.isin(split, looking.trained)
    own = {
        code: stepped(facts, registry, steps, np.flatnonzero(common | (split == code)))
        for code in ruled
        if np.any(split == code)
    }
    result = []
    for observer, held in found.items():
        position = np.full(len(split), -1, np.int64)
        position[held["tasks"]] = np.arange(len(held["tasks"]))
        rest = (position[owner] >= 0) & ~np.isin(split[owner], ruled)
        if rest.any() or not np.isin(split[held["tasks"]], ruled).any():
            kept = ~np.isin(split[held["tasks"]], ruled) | np.isin(split[held["tasks"]], looking.trained)
            result.append(
                dict(
                    held,
                    tasks=held["tasks"][kept],
                    task_step=held["task_step"][kept],
                    split=-1,
                    rules=(),
                    trained=looking.trained,
                    scored=looking.scored,
                    labels=np.flatnonzero(rest),
                    matures=shared,
                )
            )
        for code, observed in own.items():
            if observer not in observed or not np.any(split[observed[observer]["tasks"]] == code):
                continue
            part = observed[observer]
            position = np.full(len(split), -1, np.int64)
            position[part["tasks"]] = np.arange(len(part["tasks"]))
            result.append(
                dict(
                    part,
                    split=code,
                    rules=looking.rules[code],
                    trained=looking.trained,
                    scored=looking.scored,
                    labels=np.flatnonzero((position[owner] >= 0) & (split[owner] == code)),
                    matures=looking.trained if code in looking.trained else (code,) + looking.trained,
                )
            )
    LOG.debug(
        "sequences exit graph=%s steps=%s sequences=%d seconds=%.4f",
        facts.name,
        steps,
        len(result),
        time.perf_counter() - started,
    )
    return result


def shown(facts, group, held, observer, kept):
    started = time.perf_counter()
    stamps = facts.stamps(group)
    allowed = visible.allowed(
        len(stamps["since"]),
        facts.arrays[group + "_seen_row"],
        facts.arrays[group + "_seen_observer"],
        observer,
    )
    begun, ended = visible.visible(
        stamps["since"],
        stamps["until"],
        stamps["known"],
        stamps["replaced"],
        allowed,
        held["moments"],
        held["looking"],
    )
    counts = ended - begun
    row = np.repeat(np.arange(len(counts)), counts)
    step = np.repeat(begun, counts) + spans.counted(counts)
    seen = ~visible.withheld(row, step, kept, held["looking"], len(held["moments"]))
    row, step = row[seen], step[seen]
    LOG.debug(
        "shown exit graph=%s group=%s facts=%d rows=%d seconds=%.4f",
        facts.name,
        group,
        len(counts),
        len(row),
        time.perf_counter() - started,
    )
    return row, step, stamps["since"][row]


def matured(facts, registry, held, entity, step, values):
    started = time.perf_counter()
    schema, steps = registry.schema, len(held["moments"])
    owner = np.asarray(facts.target_task, np.int64)
    kinds, names = np.asarray(facts.task_kind)[owner], list(schema.tasks["references"])
    looking = np.asarray(facts.task_observer, np.int64)[owner]
    seen = ((looking == held["observer"]) | (looking < 0)) & np.isin(
        np.asarray(facts.task_split)[owner], held["matures"]
    )
    layout, encoding = registry.encoding.layout, registry.encoding
    named = np.asarray(schema.series(), object)[np.asarray(facts.target_series, np.int64)]
    parts = []
    for code, entry in enumerate(schema.tasks["kinds"].values()):
        ripe = entry["matures"]
        chosen = np.flatnonzero((kinds == code) & seen)
        if ripe is None or not len(chosen):
            continue
        members = np.flatnonzero(np.asarray(facts.member_name) == names.index(ripe["reference"]))
        place = np.full(len(facts.task_key), -1, np.int64)
        place[np.asarray(facts.member_task)[members]] = np.asarray(facts.member_source)[members]
        stated = TARGETS.named(entry["target"]["kind"]).ripened(
            named[chosen], np.asarray(facts.target_y, np.float64)[chosen]
        )
        columns = np.asarray([ripe["columns"][name] for name in named[chosen]], object)
        written = np.asarray([value is not None for value in stated], bool)
        for column in dict.fromkeys(ripe["columns"].values()):
            rows = np.flatnonzero((columns == column) & written)
            kept = chosen[rows]
            parts.append(
                (
                    place[owner[kept]],
                    np.full(len(kept), layout.index(column), np.int64),
                    np.asarray(facts.task_time, np.int64)[owner[kept]],
                    np.asarray(facts.target_known, np.int64)[kept],
                    np.asarray(
                        schema.kind(column).codes(
                            [stated[at] for at in rows.tolist()],
                            encoding.dictionaries[column] if schema.kind(column).symbols else None,
                        ),
                        np.float64,
                    ),
                )
            )
    count = 0
    if parts:
        who, column, since, known, y = (np.concatenate(found) for found in zip(*parts))
        order = np.lexsort((since, known, column, who))
        who, column, since, known, y = (found[order] for found in (who, column, since, known, y))
        later = np.r_[(who[1:] == who[:-1]) & (column[1:] == column[:-1]), False]
        begun, ended = visible.visible(
            since,
            np.full(len(since), NEVER, np.int64),
            known,
            np.where(later, np.r_[known[1:], NEVER], NEVER),
            np.ones(len(since), bool),
            held["moments"],
            held["looking"],
        )
        counts = ended - begun
        row = np.repeat(np.arange(len(counts)), counts)
        moment = np.repeat(begun, counts) + spans.counted(counts)
        keys = entity.astype(np.int64) * steps + step
        arranged = np.argsort(keys, kind="stable")
        ordered, wanted = keys[arranged], who[row] * steps + moment
        position = np.searchsorted(ordered, wanted)
        found = position < len(keys)
        found[found] = ordered[position[found]] == wanted[found]
        values[arranged[position[found]], column[row[found]]] = y[row[found]]
        count = int(found.sum())
    LOG.debug(
        "matured exit graph=%s values=%d seconds=%.4f", facts.name, count, time.perf_counter() - started
    )
    return values


def grouped(facts, held, steps):
    started = time.perf_counter()
    grouping = np.asarray(facts.task_grouping)[held["tasks"]]
    first = np.searchsorted(held["task_step"], np.arange(steps))
    if np.any(first >= len(grouping)) or np.any(
        held["task_step"][np.minimum(first, len(grouping) - 1)] != np.arange(steps)
    ):
        raise ValueError(
            "a step of the view of graph %s holds no task to state the group the step belongs to" % facts.name
        )
    result = grouping[first].astype(np.int32)
    if np.any(grouping != result[held["task_step"]]):
        raise ValueError("tasks of graph %s at one time state different groups" % facts.name)
    LOG.debug("grouped exit graph=%s steps=%d seconds=%.4f", facts.name, steps, time.perf_counter() - started)
    return result


def about(facts, registry, lit, rows):
    started = time.perf_counter()
    kept = np.ones(len(rows["step"]), bool)
    kinds, count = np.asarray(facts.kinds, np.int64), len(facts.kinds)
    joined = np.asarray([kind in registry.schema.sets for kind in registry.nodes], bool)
    for code, rule in registry.about.items():
        chosen = np.flatnonzero(rows["present"] & (rows["code"] == code))
        moments = rows["step"][chosen].astype(np.int64)
        lacking = np.zeros(len(chosen), bool)
        for end in (rows["source"][chosen], rows["target"][chosen]):
            lacking |= joined[kinds[end]] & ~np.isin(moments * count + end, lit)

        @timed("view.about.named")
        def named(position):
            row = chosen[position]
            return "the edge of %s from %s to %s of graph %s at %s" % (
                registry.by_code[code].name,
                facts.keys[rows["source"][row]],
                facts.keys[rows["target"][row]],
                facts.name,
                clock.shown(rows["moments"][rows["step"][row]]),
            )

        kept[chosen] = ABOUT.named(rule).kept(lacking, named)
    LOG.debug(
        "about exit graph=%s dropped=%d seconds=%.4f",
        facts.name,
        int((~kept).sum()),
        time.perf_counter() - started,
    )
    return kept


def referred(facts, registry, held):
    started = time.perf_counter()
    place = np.full(len(facts.task_key), -1, np.int64)
    place[held["tasks"]] = np.arange(len(held["tasks"]))
    labels, owner = held["labels"], np.asarray(facts.target_task, np.int64)
    linking = labels[registry.linked[np.asarray(facts.task_kind, np.int64)[owner[labels]]]]
    stated = place[np.asarray(facts.member_task)]
    kept = np.flatnonzero(stated >= 0)
    task = np.r_[stated[kept], place[owner[linking]]]
    order = np.argsort(task, kind="stable")
    added = dict(
        member_name=np.full(len(linking), len(registry.references)),
        member_place=np.zeros(len(linking)),
        member_source=np.asarray(facts.target_node)[linking],
        member_target=np.full(len(linking), -1),
        member_code=np.full(len(linking), -1),
    )
    result = {
        name: np.r_[np.asarray(facts.arrays[name])[kept], added[name]].astype(facts.arrays[name].dtype)[order]
        for name in added
    }
    result["member_task"] = task[order].astype(np.int32)
    result["label_task"] = place[owner[labels]].astype(np.int32)
    result["label_row"] = np.asarray(facts.target_row)[labels]
    LOG.debug(
        "referred exit graph=%s members=%d linked=%d labels=%d seconds=%.4f",
        facts.name,
        len(task),
        len(linking),
        len(labels),
        time.perf_counter() - started,
    )
    return result


def sighted(facts, group, rows, observer):
    started = time.perf_counter()
    stamps = facts.stamps(group)
    allowed = visible.allowed(
        len(stamps["since"]), facts.arrays[group + "_seen_row"], facts.arrays[group + "_seen_observer"], observer
    )
    rows = rows[allowed[rows]]
    result = int(np.maximum(stamps["since"], stamps["known"])[rows].min()) if len(rows) else None
    LOG.debug("sighted exit graph=%s group=%s rows=%d seconds=%.4f", facts.name, group, len(rows), time.perf_counter() - started)
    return result


def untargeted(facts, registry, held, row):
    started = time.perf_counter()
    action = int(facts.role_action[row])
    option = int(facts.action_option[action])
    task = int(facts.option_task[option])
    kinds = registry.schema.tasks["kinds"]
    kind = list(kinds)[int(facts.task_kind[task])]
    name = list(kinds[kind]["target"]["actions"])[int(facts.action_kind[action])]
    declared = kinds[kind]["target"]["actions"][name]
    source, other, code = (int(facts.arrays["role_" + part][row]) for part in ("source", "target", "code"))
    if other < 0:
        named = "the node %s" % facts.keys[source]
        rows = np.flatnonzero(np.asarray(facts.obs_entity) == source)
    else:
        named = "the edge of %s from %s to %s" % (registry.by_code[code].name, facts.keys[source], facts.keys[other])
        first, second = np.asarray(facts.fact_source), np.asarray(facts.fact_target)
        either = registry.flipped[code] == code
        rows = np.flatnonzero(
            np.asarray(facts.fact_present)
            & (np.asarray(facts.fact_code) == code)
            & (((first == source) & (second == other)) | (either & (first == other) & (second == source)))
        )
    seen = sighted(facts, "obs" if other < 0 else "fact", rows, held["observer"])
    result = (
        "the target %s of the action %d, of the kind %s, of the option %s of the decision %d of the kind %s of graph %s "
        "names %s, which is not observed at the step of its decision, at %s; %s"
        % (
            list(declared["roles"])[int(facts.role_name[row])],
            action - int(np.searchsorted(facts.action_option, option)),
            name,
            facts.options[option],
            int(facts.task_key[task]),
            kind,
            facts.name,
            named,
            clock.shown(facts.task_time[task]),
            "it is never observed" if seen is None else "it is first observed at " + clock.shown(seen),
        )
    )
    LOG.debug("untargeted exit graph=%s row=%d seconds=%.4f", facts.name, row, time.perf_counter() - started)
    return result


def targeted(facts, registry, held, shown):
    started = time.perf_counter()
    action = np.asarray(facts.role_action, np.int64)
    option = np.asarray(facts.action_option, np.int64)[action]
    task = np.asarray(facts.option_task, np.int64)[option]
    placed = np.full(len(facts.task_key), -1, np.int64)
    placed[held["tasks"]] = held["task_step"]
    moment, entities = placed[task], len(facts.kinds)
    sort = roled(registry.schema)[
        np.asarray(facts.task_kind, np.int64)[task],
        np.asarray(facts.action_kind, np.int64)[action],
        np.asarray(facts.role_name, np.int64),
    ]
    source, other, code = (np.asarray(facts.arrays["role_" + part], np.int64) for part in ("source", "target", "code"))
    lost = []
    nodes = np.flatnonzero((moment >= 0) & (sort == ROLES.index("node")))
    if len(nodes):
        offsets, lows, highs = spans.coverage(
            np.asarray(shown.obs_entity), np.asarray(shown.obs_start), np.asarray(shown.obs_stop), entities
        )
        found = spans.all_covered(offsets, lows, highs, source[nodes], moment[nodes], moment[nodes] + 1)
        if found >= 0:
            lost.append(nodes[found])
    edges = np.flatnonzero((moment >= 0) & (sort == ROLES.index("edge")))
    if len(edges):
        relation, first, second = (
            np.asarray(shown.arrays["fact_" + part], np.int64) for part in ("code", "source", "target")
        )
        either = registry.flipped[relation] == relation
        known, ids = np.unique(
            np.r_[
                spans.encode(relation, first, second, entities),
                spans.encode(relation[either], second[either], first[either], entities),
            ],
            return_inverse=True,
        )
        begun, ended = (np.r_[shown.arrays[name], shown.arrays[name][either]] for name in ("fact_start", "fact_stop"))
        offsets, lows, highs = spans.coverage(ids.astype(np.int32), begun, ended, len(known))
        wanted = spans.encode(code[edges], source[edges], other[edges], entities)
        place = np.searchsorted(known, wanted)
        named = place < len(known)
        named[named] = known[place[named]] == wanted[named]
        if not named.all():
            lost.append(edges[np.flatnonzero(~named)[0]])
        found = spans.all_covered(offsets, lows, highs, place[named], moment[edges][named], moment[edges][named] + 1)
        if found >= 0:
            lost.append(edges[np.flatnonzero(named)[found]])
    if lost:
        raise ValueError(untargeted(facts, registry, held, int(min(lost))))
    LOG.debug(
        "targeted exit graph=%s targets=%d seconds=%.4f", facts.name, len(nodes) + len(edges), time.perf_counter() - started
    )


@timed("view.view")
def view(facts, registry, held, several):
    steps = len(held["moments"])
    if steps < 1:
        raise ValueError("the view of graph %s shows no step" % facts.name)
    if registry.schema.tasks["grouping"] is not None:
        return showing(facts, registry, held, several, grouped(facts, held, steps))
    return showing(facts, registry, held, several, np.arange(steps, dtype=np.int32))


def beside(facts, registry, held, several):
    started = time.perf_counter()
    if held["rules"]:
        raise ValueError(
            "a second graph is shown to a split without rules of what it may see; graph %s would be shown beside "
            "a split whose rules are %s" % (facts.name, ", ".join(held["rules"]))
        )
    none = np.zeros(0, np.int64)
    stripped = dict(held, tasks=none, task_step=none.astype(np.int32), labels=none, rules=(), matures=())
    result = showing(facts, registry, stripped, several, np.arange(len(held["moments"]), dtype=np.int32))
    LOG.info("beside exit graph=%s steps=%d seconds=%.3f", facts.name, result.steps, time.perf_counter() - started)
    return result


def showing(facts, registry, held, several, period):
    started = time.perf_counter()
    schema, observer, steps = registry.schema, held["observer"], len(held["moments"])
    if steps < 1:
        raise ValueError("the view of graph %s shows no step" % facts.name)
    if np.any(np.diff(held["task_step"]) < 0):
        raise ValueError("the tasks of the view of graph %s are not in the order of its steps" % facts.name)
    if several is None and registry.extras["several"]:
        raise ValueError(
            "the bank of graph %s holds facts about one entity that a view can show at one step, and no "
            "time/several is stated; it is one of %s" % (facts.name, ", ".join(SEVERAL.names()))
        )
    place = ranks(facts.keys)
    seen = permitted(facts, registry, held)
    row, step, since = shown(facts, "obs", held, observer, seen["obs"])
    entity = np.asarray(facts.obs_entity, np.int64)[row]
    keys = clock.keys(
        schema.ties,
        dict(
            seq=np.asarray(facts.obs_seq)[row],
            code=np.asarray(facts.kinds)[entity],
            source=place[entity],
            target=place[entity],
        ),
    )
    order = np.lexsort(tuple(keys[::-1]) + (since, entity, step))
    first = np.r_[True, (step[order][1:] != step[order][:-1]) | (entity[order][1:] != entity[order][:-1])]
    starts = np.flatnonzero(first[: len(order)])
    stops = np.r_[starts[1:], len(order)][: len(starts)]
    if several is None:
        crowded = np.flatnonzero(stops - starts > 1)
        if len(crowded):
            named = order[starts[crowded[0]]]
            raise ValueError(
                "the view of graph %s shows at %s several facts about the node %s, and no time/several is stated"
                % (facts.name, clock.shown(held["moments"][step[named]]), facts.keys[int(entity[named])])
            )
        chosen = order[starts]
    else:
        chosen = order[SEVERAL.named(several)(starts, stops)]
    observed = spans.observation_spans(
        entity[chosen].astype(np.int32),
        step[chosen].astype(np.int32),
        matured(
            facts,
            registry,
            held,
            entity[chosen],
            step[chosen],
            np.array(np.asarray(facts.obs_values)[row[chosen]]),
        ),
    )
    observed["obs_values"] = observed.pop("obs_x")
    lit = np.unique(step[chosen].astype(np.int64) * len(facts.kinds) + entity[chosen])
    row, step, since = shown(facts, "fact", held, observer, seen["fact"])
    source, target = np.asarray(facts.fact_source)[row], np.asarray(facts.fact_target)[row]
    code, present = np.asarray(facts.fact_code, np.int64)[row], np.asarray(facts.fact_present)[row]
    kept = about(
        facts,
        registry,
        lit,
        dict(step=step, source=source, target=target, code=code, present=present, moments=held["moments"]),
    )
    row, step, since, source, target, code, present = (
        values[kept] for values in (row, step, since, source, target, code, present)
    )
    keys = clock.keys(
        schema.ties,
        dict(seq=np.asarray(facts.fact_seq)[row], code=code, source=place[source], target=place[target]),
    )
    order = np.lexsort((since,) + tuple(keys[::-1]) + (~present, step))
    tasks = held["tasks"]
    name = facts.name
    if observer >= 0:
        name = "%s seen by the observer %s" % (facts.name, facts.observers[observer])
    graph = Shown(
        name if held["split"] < 0 else "%s for split %d" % (name, held["split"]),
        facts.keys,
        steps,
        kinds=np.asarray(facts.kinds),
        entity_token=np.asarray(facts.entity_token),
        period=period,
        step_time=np.asarray(held["moments"], np.int64),
        step_token=np.full(steps, token(facts.name), np.uint64),
        **observed,
        **spans.fact_spans(
            step[order].astype(np.int32),
            source[order],
            target[order],
            code[order],
            present[order],
            np.asarray(facts.fact_values)[row[order]],
        ),
        task_step=held["task_step"].astype(np.int32),
        task_key=np.asarray(facts.task_key)[tasks],
        task_kind=np.asarray(facts.task_kind)[tasks],
        task_values=np.asarray(facts.task_values)[tasks],
        task_signals=np.asarray(facts.task_signals)[tasks],
        graph_values=np.asarray(facts.graph_values),
        **referred(facts, registry, held),
    )
    verify(graph, registry)
    LOG.info(
        "view exit graph=%s observer=%d split=%d steps=%d observations=%d facts=%d seconds=%.3f",
        facts.name,
        observer,
        held["split"],
        steps,
        len(chosen),
        len(order),
        time.perf_counter() - started,
    )
    return graph


def finished(shown, registry, inner):
    started = time.perf_counter()
    entity, kinds = np.asarray(shown.obs_entity, np.int64), np.asarray(shown.kinds)
    tasks = list(registry.schema.tasks["kinds"])
    held = np.zeros_like(inner)
    held[entity] = inner[entity]
    graph = Graph(
        shown.name,
        shown.keys,
        shown.steps,
        **shown.arrays,
        entity_held=held,
        obs_x=encoded(
            registry,
            shown.obs_values,
            inner[entity],
            None,
            lambda row: (
                "the node %s of the kind %s in the view of graph %s"
                % (shown.keys[entity[row]], registry.nodes[kinds[entity[row]]], shown.name)
            ),
        ),
        task_x=encoded(
            registry,
            shown.task_values,
            unheld(registry, len(shown.task_key)),
            np.broadcast_to(
                np.asarray(shown.graph_values), (len(shown.task_key), len(registry.schema.graph))
            ),
            lambda row: (
                "the task %d of the kind %s in the view of graph %s"
                % (shown.task_key[row], tasks[shown.task_kind[row]], shown.name)
            ),
        ),
    )
    LOG.debug(
        "finished exit graph=%s observations=%d seconds=%.4f",
        shown.name,
        len(entity),
        time.perf_counter() - started,
    )
    return graph


def sampled(facts, registry, held, several):
    started = time.perf_counter()
    schema, layout = registry.schema, registry.encoding.layout
    result = {name: [] for name in schema.columns}
    split = np.asarray(facts.task_split)
    trained = np.isin(split[np.asarray(facts.target_task, np.int64)[held["labels"]]], held["trained"])
    if trained.any():
        shown = view(facts, registry, held, several)
        steps = np.asarray(shown.task_step, np.int64)
        bound = int(steps[np.asarray(shown.label_task, np.int64)[trained]].max())
        tasks = (steps <= bound) & (
            np.isin(split[held["tasks"]], held["trained"]) | (split[held["tasks"]] < 0)
        )
        observed = np.maximum(
            np.minimum(np.asarray(shown.obs_stop, np.int64), bound + 1) - shown.obs_start, 0
        )
        stated = np.maximum(
            np.minimum(np.asarray(shown.fact_stop, np.int64), bound + 1) - shown.fact_start, 0
        )
        removed = np.flatnonzero(np.asarray(shown.removal_step) <= bound)
        owners = np.unique(np.asarray(shown.obs_entity, np.int64)[observed > 0])
        inner = np.isin(np.asarray(facts.inner_owner, np.int64), owners)
        codes = np.r_[
            np.repeat(np.asarray(shown.fact_code, np.int64), stated),
            np.asarray(shown.removal_code, np.int64)[removed],
        ]
        values = np.concatenate(
            (np.repeat(np.asarray(shown.fact_values), stated, 0), np.asarray(shown.removal_values)[removed])
        )
        parts = [
            (layout, np.repeat(np.asarray(shown.obs_values), observed, 0)),
            (layout, np.asarray(shown.task_values)[tasks]),
            (schema.graph, np.asarray(facts.graph_values)[None, :]),
            (schema.inner_columns(), np.asarray(facts.inner_values)[inner]),
        ]
        valued = registry.value_group[codes]
        parts += [
            (registry.attributes(group), values[valued == place]) for place, group in enumerate(registry.valued)
        ]
        for columns, found in parts:
            for position, name in enumerate(columns):
                column = found[:, position]
                result[name].append(column[~np.isnan(column)])
    LOG.debug(
        "sampled exit graph=%s trained=%s seconds=%.4f",
        facts.name,
        bool(trained.any()),
        time.perf_counter() - started,
    )
    return result
