import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from . import clock, spans
from .carry import encode
from .clock import ABOUT, ALWAYS, NEVER, SEVERAL
from .names import Names
from .references import MEMBERS, ends
from .sets import NODE, stored
from .truth import GROUPS, Graph, encoded, ranks, verify

LOG = logging.getLogger(__name__)
STEPS = Names("choice of steps")
SEES = Names("rule of what a split may see")
STATEMENT = ("trained", "scored", "sees")


class Looking:
    def __init__(self, splits, trained, scored, sees):
        started = time.perf_counter()
        self.splits = tuple(splits)
        for label, names in (("trained on", trained), ("scored on", scored)):
            if not isinstance(names, (list, tuple)) or len(set(names)) != len(names):
                raise ValueError("the splits %s are a list of distinct names, found %r" % (label, names))
            strangers = [name for name in names if name not in self.splits]
            if strangers:
                raise ValueError(
                    "the splits %s name %s, and the bank holds the splits %s"
                    % (label, ", ".join(map(str, strangers)), ", ".join(self.splits))
                )
        both = sorted(set(trained) & set(scored))
        if both:
            raise ValueError("the splits %s are both trained on and scored on" % ", ".join(both))
        named = list(trained) + list(scored)
        if not isinstance(sees, dict) or sorted(sees) != sorted(named):
            raise ValueError(
                "what a split may see is stated for every split trained on or scored on, %s, found %r"
                % (", ".join(named) or "none", sees)
            )
        self.trained = tuple(self.splits.index(name) for name in trained)
        self.scored = tuple(self.splits.index(name) for name in scored)
        self.rules = {}
        for name, rules in sees.items():
            if not isinstance(rules, (list, tuple)) or len(set(rules)) != len(rules):
                raise ValueError(
                    "what the split %s may see is a list of distinct rules, found %r" % (name, rules)
                )
            for rule in rules:
                SEES.named(rule)
            self.rules[self.splits.index(name)] = tuple(rules)
        LOG.debug(
            "looking exit trained=%d scored=%d seconds=%.4f",
            len(self.trained),
            len(self.scored),
            time.perf_counter() - started,
        )

    @timed("view.looking.mapping")
    def mapping(self):
        return dict(
            trained=[self.splits[code] for code in self.trained],
            scored=[self.splits[code] for code in self.scored],
            sees={self.splits[code]: list(rules) for code, rules in self.rules.items()},
        )


@timed("view.unstated")
def unstated(splits):
    return Looking(splits, (), (), {})


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
    found = np.searchsorted(moments, times, "right") - 1
    absent = (found < 0) | (moments[np.maximum(found, 0)] != times)
    if absent.any():
        raise ValueError(
            "a task of graph %s sits at %s, which no step of the view shows"
            % (graph, clock.shown(np.asarray(times)[absent][0]))
        )
    return found


class Tasks:
    @timed("view.tasks.serves")
    def serves(self, registry):
        return True

    def __call__(self, facts, registry):
        started = time.perf_counter()
        looking = np.asarray(facts.task_observer)
        if not len(looking):
            raise ValueError("graph %s holds no task, and the steps of the view are its tasks" % facts.name)
        times = np.asarray(facts.task_time)
        result = []
        for observer in np.unique(looking).tolist():
            chosen = np.flatnonzero(looking == observer)
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
    @timed("view.changes.serves")
    def serves(self, registry):
        return True

    def __call__(self, facts, registry):
        started = time.perf_counter()
        times = []
        for group in GROUPS:
            times += [
                np.asarray(facts.arrays[group + "_" + name], np.int64)
                for name in ("since", "until", "known_time", "replaced_time")
            ]
        looking, held = np.asarray(facts.task_observer), np.asarray(facts.task_time, np.int64)
        result = []
        for observer in np.unique(looking).tolist():
            chosen = np.flatnonzero(looking == observer)
            found = np.concatenate(times + [held[chosen]])
            moments = np.unique(found[(found > ALWAYS) & (found < NEVER)])
            result.append(sequence(observer, moments, moments, chosen, at(facts.name, moments, held[chosen])))
        LOG.debug(
            "steps of changes exit graph=%s sequences=%d seconds=%.4f",
            facts.name,
            len(result),
            time.perf_counter() - started,
        )
        return result


STEPS.register("tasks", Tasks())
STEPS.register("changes", Changes())


def without_targets(facts, registry, held):
    started = time.perf_counter()
    aimed = np.zeros(len(facts.task_key), bool)
    aimed[np.asarray(facts.target_task)] = True
    aimed &= np.asarray(facts.task_split) == held["split"]
    chosen = aimed[np.asarray(facts.member_task)] & (np.asarray(facts.member_code) >= 0)
    entities = len(facts.kinds)
    hidden = encode(
        np.asarray(facts.member_code)[chosen],
        np.asarray(facts.member_source)[chosen],
        np.asarray(facts.member_target)[chosen],
        entities,
    )
    stated = encode(facts.fact_code, facts.fact_source, facts.fact_target, entities)
    result = np.ones(len(facts.obs_entity), bool), ~np.isin(stated, hidden)
    LOG.debug(
        "without targets exit graph=%s hidden=%d seconds=%.4f",
        facts.name,
        int((~result[1]).sum()),
        time.perf_counter() - started,
    )
    return result


def without_scored(facts, registry, held):
    started = time.perf_counter()
    if held["split"] in held["scored"]:
        raise ValueError(
            "the entities of the splits scored on are hidden from the splits trained on, "
            "and the rule is stated for a split scored on"
        )
    member, entity = ends(facts.member_source, facts.member_target)
    scored = np.isin(np.asarray(facts.task_split)[np.asarray(facts.member_task)[member]], held["scored"])
    hidden = np.setdiff1d(entity[scored], entity[~scored])
    result = (
        ~np.isin(facts.obs_entity, hidden),
        ~(np.isin(facts.fact_source, hidden) | np.isin(facts.fact_target, hidden)),
    )
    LOG.debug(
        "without scored exit graph=%s hidden=%d seconds=%.4f",
        facts.name,
        len(hidden),
        time.perf_counter() - started,
    )
    return result


SEES.register("without_targets", without_targets)
SEES.register("without_scored", without_scored)


@timed("view.offered")
def offered(registry):
    return tuple(name for name in STEPS.names() if STEPS.named(name).serves(registry))


def sequences(facts, registry, steps, looking):
    started = time.perf_counter()
    chooser = STEPS.named(steps)
    if not chooser.serves(registry):
        raise ValueError("the choice of steps %s does not serve the graphs of this bank" % steps)
    found = chooser(facts, registry)
    split, owner = np.asarray(facts.task_split), np.asarray(facts.target_task)
    placed = np.concatenate([held["tasks"] for held in found] + [np.zeros(0, np.int64)])
    if len(np.unique(placed)) != len(placed) or not np.isin(owner, placed).all():
        raise ValueError(
            "the choice of steps %s does not give every task of graph %s that holds a target one step"
            % (steps, facts.name)
        )
    ruled = [code for code, rules in looking.rules.items() if rules]
    result = []
    for held in found:
        own = split[held["tasks"]]
        position = np.full(len(split), -1, np.int64)
        position[held["tasks"]] = np.arange(len(held["tasks"]))
        labelled = position[owner] >= 0
        rest = labelled & ~np.isin(split[owner], ruled)
        if rest.any() or not np.isin(own, ruled).any():
            kept = ~np.isin(own, ruled) | np.isin(own, looking.trained)
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
                )
            )
        for code in ruled:
            if not np.any(own == code):
                continue
            kept = (own == code) | (own < 0) | np.isin(own, looking.trained)
            result.append(
                dict(
                    held,
                    tasks=held["tasks"][kept],
                    task_step=held["task_step"][kept],
                    split=code,
                    rules=looking.rules[code],
                    trained=looking.trained,
                    scored=looking.scored,
                    labels=np.flatnonzero(labelled & (split[owner] == code)),
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


def permitted(facts, registry, held):
    started = time.perf_counter()
    result = dict(
        obs=np.ones(len(facts.obs_entity), bool),
        fact=np.ones(len(facts.fact_source), bool),
    )
    for name in held["rules"]:
        observed, stated = SEES.named(name)(facts, registry, held)
        result["obs"] &= observed
        result["fact"] &= stated
    LOG.debug(
        "permitted exit graph=%s rules=%d seconds=%.4f",
        facts.name,
        len(held["rules"]),
        time.perf_counter() - started,
    )
    return result


def shown(facts, group, held, observer, seen):
    started = time.perf_counter()
    stamps = facts.stamps(group)
    allowed = seen & clock.allowed(
        len(stamps["since"]),
        facts.arrays[group + "_seen_row"],
        facts.arrays[group + "_seen_observer"],
        observer,
    )
    begun, ended = clock.visible(
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
    layout = registry.encoding.layout
    parts = []
    for code, entry in enumerate(schema.tasks["kinds"].values()):
        ripe = entry["matures"]
        chosen = np.flatnonzero(kinds == code)
        if ripe is None or not len(chosen):
            continue
        members = np.flatnonzero(np.asarray(facts.member_name) == names.index(ripe["reference"]))
        place = np.full(len(facts.task_key), -1, np.int64)
        place[np.asarray(facts.member_task)[members]] = np.asarray(facts.member_source)[members]
        parts.append(
            (
                place[owner[chosen]],
                np.full(len(chosen), layout.index(ripe["column"]), np.int64),
                np.asarray(facts.task_time, np.int64)[owner[chosen]],
                np.asarray(facts.target_known, np.int64)[chosen],
                np.asarray(facts.target_y, np.float64)[chosen],
            )
        )
    count = 0
    if parts:
        who, column, since, known, y = (np.concatenate(found) for found in zip(*parts))
        order = np.lexsort((since, known, column, who))
        who, column, since, known, y = (found[order] for found in (who, column, since, known, y))
        later = np.r_[(who[1:] == who[:-1]) & (column[1:] == column[:-1]), False]
        begun, ended = clock.visible(
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
        wanted = who[row] * steps + moment
        position = np.minimum(np.searchsorted(keys[arranged], wanted), max(len(keys) - 1, 0))
        found = (keys[arranged][position] == wanted) if len(keys) else np.zeros(len(wanted), bool)
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


def referred(facts, held):
    started = time.perf_counter()
    place = np.full(len(facts.task_key), -1, np.int64)
    place[held["tasks"]] = np.arange(len(held["tasks"]))
    owner = place[np.asarray(facts.member_task)]
    kept = owner >= 0
    result = {name: np.asarray(facts.arrays[name])[kept] for name in MEMBERS}
    result["member_task"] = owner[kept].astype(np.int32)
    labels = held["labels"]
    result["label_task"] = place[np.asarray(facts.target_task)[labels]].astype(np.int32)
    result["label_row"] = np.asarray(facts.target_row)[labels]
    LOG.debug(
        "referred exit graph=%s members=%d labels=%d seconds=%.4f",
        facts.name,
        int(kept.sum()),
        len(labels),
        time.perf_counter() - started,
    )
    return result


def view(facts, registry, held, several):
    started = time.perf_counter()
    schema, observer, steps = registry.schema, held["observer"], len(held["moments"])
    if steps < 1:
        raise ValueError("the view of graph %s shows no step" % facts.name)
    if np.any(np.diff(held["task_step"]) < 0):
        raise ValueError("the tasks of the view of graph %s are not in the order of its steps" % facts.name)
    choose, place = SEVERAL.named(several), ranks(facts.keys)
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
    chosen = order[choose(starts, np.r_[starts[1:], len(order)][: len(starts)])]
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
    name = facts.name if observer < 0 else "%s seen by observer %d" % (facts.name, observer)
    inner = np.asarray(facts.entity_held)
    graph = Graph(
        name if held["split"] < 0 else "%s for split %d" % (name, held["split"]),
        facts.keys,
        steps,
        kinds=np.asarray(facts.kinds),
        entity_token=np.asarray(facts.entity_token),
        entity_held=inner,
        period=grouped(facts, held, steps)
        if schema.tasks["grouping"] is not None
        else np.arange(steps, dtype=np.int32),
        obs_x=encoded(registry, observed["obs_values"], inner[observed["obs_entity"]]),
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
        task_x=np.asarray(facts.task_x)[tasks],
        task_signals=np.asarray(facts.task_signals)[tasks],
        graph_values=np.asarray(facts.graph_values),
        **referred(facts, held),
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


def validate(facts, registry, splits):
    started = time.perf_counter()
    names = offered(registry)
    if not names:
        raise ValueError("no choice of steps serves the graphs of this bank")
    facts = stored(facts, registry, {kind: NODE for kind in registry.schema.sets})
    result = [
        view(facts, registry, held, SEVERAL.names()[0])
        for held in sequences(facts, registry, names[0], unstated(splits))
    ]
    LOG.debug(
        "validate exit graph=%s views=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started
    )
    return result
