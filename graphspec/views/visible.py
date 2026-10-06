import logging
import time

import numpy as np

from graphspec.budgets import Refused
from graphspec.facts.stored import GROUPS
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.references import ends
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import ALWAYS, NEVER

from .spans import counted, encode

LOG = logging.getLogger(__name__)


def allowed(count, rows, observers, observer):
    started = time.perf_counter()
    rows, observers = np.asarray(rows, np.int64), np.asarray(observers, np.int64)
    result = np.ones(count, bool)
    if observer >= 0:
        result[rows] = False
        result[rows[observers == observer]] = True
    LOG.debug(
        "allowed exit facts=%d stated=%d observer=%d seconds=%.4f",
        count,
        len(rows),
        observer,
        time.perf_counter() - started,
    )
    return result


def visible(since, until, known, replaced, permitted, moments, looking):
    started = time.perf_counter()
    moments, looking = np.asarray(moments, np.int64), np.asarray(looking, np.int64)
    if moments.shape != looking.shape or moments.ndim != 1:
        raise ValueError("a view states one time of looking for every time it shows")
    if np.any(np.diff(moments) < 0) or np.any(np.diff(looking) < 0):
        raise ValueError("the times a view shows, and the times of looking, never decrease")
    if np.any(looking < moments):
        raise ValueError("a view shows no time later than the time of looking at it")
    since, until = np.asarray(since, np.int64), np.asarray(until, np.int64)
    first = np.searchsorted(moments, since, "left")
    begun = np.maximum(first, np.searchsorted(looking, known, "left"))
    held = np.searchsorted(moments, until, "left")
    held = np.where(until == since + 1, np.minimum(first + 1, len(moments)), held)
    ended = np.minimum(held, np.searchsorted(looking, replaced, "left"))
    ended = np.maximum(np.where(permitted, ended, begun), begun)
    LOG.debug(
        "visible exit facts=%d moments=%d seconds=%.4f",
        len(begun),
        len(moments),
        time.perf_counter() - started,
    )
    return begun.astype(np.int64), ended.astype(np.int64)


SEES = Names("rule of what a split may see", True)
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
                raise Refused(
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


@timed("view.during")
def during(keys, named, since, until):
    keys, named, since, until = (np.asarray(values, np.int64) for values in (keys, named, since, until))
    order = np.argsort(named, kind="stable")
    named, since, until = named[order], since[order], until[order]
    low = np.searchsorted(named, keys, "left")
    counts = np.searchsorted(named, keys, "right") - low
    place = np.repeat(low, counts) + counted(counts)
    return np.repeat(np.arange(len(keys), dtype=np.int64), counts), since[place], until[place]


@timed("view.joined")
def joined(*parts):
    return tuple(np.concatenate([part[at] for part in parts] + [np.zeros(0, np.int64)]) for at in range(3))


def linked(facts, registry, aimed, known):
    started = time.perf_counter()
    schema, names = registry.schema, list(registry.schema.tasks["references"])
    owner, node = np.asarray(facts.target_task, np.int64), np.asarray(facts.target_node, np.int64)
    kind, entities = np.asarray(facts.task_kind, np.int64), len(facts.kinds)
    keys, moments, ends = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for code, entry in enumerate(schema.tasks["kinds"].values()):
        aim = entry["target"]
        if aim is None or TARGETS.named(aim["kind"]).names != "linked":
            continue
        members = np.flatnonzero(np.asarray(facts.member_name) == names.index(aim["about"]))
        anchor = np.full(len(facts.task_key), -1, np.int64)
        anchor[np.asarray(facts.member_task, np.int64)[members]] = np.asarray(facts.member_source, np.int64)[members]
        chosen = np.flatnonzero(aimed[owner] & (kind[owner] == code) & (node >= 0))
        source, target = anchor[owner[chosen]], node[chosen]
        moment = np.asarray(facts.task_time, np.int64)[owner[chosen]]
        for relation in np.asarray(registry.coded[aim["relation"]], np.int64).tolist():
            pairs = ((source, target),)
            if registry.flipped[relation] == relation:
                pairs += ((target, source),)
            for first, second in pairs:
                keys.append(encode(np.full(len(chosen), relation, np.int64), first, second, entities))
                moments.append(moment)
                ends.append(known[owner[chosen]])
    result = np.concatenate(keys), np.concatenate(moments), np.concatenate(ends)
    LOG.debug("linked exit graph=%s edges=%d seconds=%.4f", facts.name, len(result[0]), time.perf_counter() - started)
    return result


def without_targets(facts, registry, held):
    started = time.perf_counter()
    schema, names = registry.schema, list(registry.schema.tasks["references"])
    about = np.asarray(
        [
            -1
            if entry["target"] is None or entry["target"]["about"] is None
            else names.index(entry["target"]["about"])
            for entry in schema.tasks["kinds"].values()
        ],
        np.int64,
    )
    aimed = np.zeros(len(facts.task_key), bool)
    aimed[np.asarray(facts.target_task)] = True
    aimed &= np.asarray(facts.task_split) == held["split"]
    known = np.full(len(facts.task_key), ALWAYS, np.int64)
    np.maximum.at(known, np.asarray(facts.target_task, np.int64), np.asarray(facts.target_known, np.int64))
    task = np.asarray(facts.member_task, np.int64)
    chosen = aimed[task] & (
        np.asarray(facts.member_name) == about[np.asarray(facts.task_kind, np.int64)[task]]
    )
    moment, until = np.asarray(facts.task_time, np.int64)[task[chosen]], known[task[chosen]]
    code, source, target = (
        np.asarray(facts.arrays[name], np.int64)[chosen]
        for name in ("member_code", "member_source", "member_target")
    )
    joins = np.asarray([kind in schema.sets for kind in registry.nodes], bool)
    edge = np.flatnonzero(code >= 0)
    whole = (code < 0) & joins[np.asarray(facts.kinds, np.int64)[source]]
    either = edge[registry.flipped[code[edge]] == code[edge]]
    entities = len(facts.kinds)
    named = np.r_[
        encode(code[edge], source[edge], target[edge], entities),
        encode(code[either], target[either], source[either], entities),
    ]
    linking, moments, ends = linked(facts, registry, aimed, known)
    stated = encode(facts.fact_code, facts.fact_source, facts.fact_target, entities)
    result = (
        during(facts.obs_entity, source[whole], moment[whole], until[whole]),
        joined(
            during(
                stated,
                np.r_[named, linking],
                np.r_[moment[edge], moment[either], moments],
                np.r_[until[edge], until[either], ends],
            ),
            during(facts.fact_source, source[whole], moment[whole], until[whole]),
            during(facts.fact_target, source[whole], moment[whole], until[whole]),
        ),
    )
    LOG.debug(
        "without targets exit graph=%s withheld=%d seconds=%.4f",
        facts.name,
        len(result[0][0]) + len(result[1][0]),
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
    result = tuple(
        (rows, np.full(len(rows), ALWAYS, np.int64), np.full(len(rows), NEVER, np.int64))
        for rows in (
            np.flatnonzero(np.isin(facts.obs_entity, hidden)),
            np.flatnonzero(np.isin(facts.fact_source, hidden) | np.isin(facts.fact_target, hidden)),
        )
    )
    LOG.debug(
        "without scored exit graph=%s hidden=%d seconds=%.4f",
        facts.name,
        len(hidden),
        time.perf_counter() - started,
    )
    return result


without_targets.version = 1
without_scored.version = 1
SEES.register("without_targets", without_targets)
SEES.register("without_scored", without_scored)


def permitted(facts, registry, held):
    started = time.perf_counter()
    counts = dict(obs=len(facts.obs_entity), fact=len(facts.fact_source))
    result = {group: [] for group in GROUPS}
    for name in held["rules"]:
        found = [[np.asarray(values) for values in intervals] for intervals in SEES.named(name)(facts, registry, held)]
        if len(found) != len(GROUPS) or any(
            len(intervals) != 3
            or any(values.ndim != 1 or values.dtype.kind not in "iu" for values in intervals)
            or len({len(values) for values in intervals}) != 1
            or np.any(intervals[0] < 0)
            or np.any(intervals[0] >= counts[group])
            for group, intervals in zip(GROUPS, found)
        ):
            raise ValueError(
                "the rule %s of what a split may see gives, for the facts about nodes and for the facts about "
                "edges of graph %s, the intervals over which a fact is kept from the split as three arrays of "
                "whole numbers of one length: the fact, the time from which and the time until which it is kept"
                % (name, facts.name)
            )
        for group, intervals in zip(GROUPS, found):
            result[group].append(tuple(values.astype(np.int64) for values in intervals))
    result = {group: joined(*parts) for group, parts in result.items()}
    LOG.debug(
        "permitted exit graph=%s rules=%d seconds=%.4f",
        facts.name,
        len(held["rules"]),
        time.perf_counter() - started,
    )
    return result


@timed("view.withheld")
def withheld(row, step, intervals, looking, steps):
    rows, since, until = intervals
    low, high = np.searchsorted(looking, since, "left"), np.searchsorted(looking, until, "left")
    counts = np.maximum(high - low, 0)
    hidden = np.repeat(rows, counts) * steps + np.repeat(low, counts) + counted(counts)
    return np.isin(np.asarray(row, np.int64) * steps + step, hidden)
