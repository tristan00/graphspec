import logging
import time

import numpy as np

from graphspec.facts.stored import GROUPS
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.references import ends
from graphspec.schema.time import ALWAYS, NEVER

from .spans import encode

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


@timed("view.withheld")
def withheld(keys, named, moments):
    keys, named, moments = (np.asarray(values, np.int64) for values in (keys, named, moments))
    result = np.full(len(keys), NEVER, np.int64)
    if len(named):
        order = np.lexsort((moments, named))
        named, moments = named[order], moments[order]
        first = np.r_[True, named[1:] != named[:-1]]
        named, moments = named[first], moments[first]
        position = np.searchsorted(named, keys)
        found = position < len(named)
        found[found] = named[position[found]] == keys[found]
        result[found] = moments[position[found]]
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
    task = np.asarray(facts.member_task, np.int64)
    chosen = aimed[task] & (
        np.asarray(facts.member_name) == about[np.asarray(facts.task_kind, np.int64)[task]]
    )
    moment = np.asarray(facts.task_time, np.int64)[task[chosen]]
    code, source, target = (
        np.asarray(facts.arrays[name], np.int64)[chosen]
        for name in ("member_code", "member_source", "member_target")
    )
    joined = np.asarray([kind in schema.sets for kind in registry.nodes], bool)
    edge = np.flatnonzero(code >= 0)
    whole = (code < 0) & joined[np.asarray(facts.kinds, np.int64)[source]]
    either = edge[registry.flipped[code[edge]] == code[edge]]
    entities = len(facts.kinds)
    named = np.r_[
        encode(code[edge], source[edge], target[edge], entities),
        encode(code[either], target[either], source[either], entities),
    ]
    stated = encode(facts.fact_code, facts.fact_source, facts.fact_target, entities)
    result = (
        withheld(facts.obs_entity, source[whole], moment[whole]),
        np.minimum.reduce(
            (
                withheld(stated, named, np.r_[moment[edge], moment[either]]),
                withheld(facts.fact_source, source[whole], moment[whole]),
                withheld(facts.fact_target, source[whole], moment[whole]),
            )
        ),
    )
    LOG.debug(
        "without targets exit graph=%s withheld=%d seconds=%.4f",
        facts.name,
        int((result[1] < NEVER).sum() + (result[0] < NEVER).sum()),
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
        np.where(np.isin(facts.obs_entity, hidden), ALWAYS, NEVER),
        np.where(np.isin(facts.fact_source, hidden) | np.isin(facts.fact_target, hidden), ALWAYS, NEVER),
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


def permitted(facts, registry, held):
    started = time.perf_counter()
    result = dict(
        obs=np.full(len(facts.obs_entity), NEVER, np.int64),
        fact=np.full(len(facts.fact_source), NEVER, np.int64),
    )
    for name in held["rules"]:
        for group, found in zip(GROUPS, SEES.named(name)(facts, registry, held)):
            found = np.asarray(found)
            if found.shape != result[group].shape or found.dtype.kind not in "iu":
                raise ValueError(
                    "the rule %s of what a split may see gives for every fact of graph %s the time from which it "
                    "is kept from the split, as whole numbers, and gave %s of the shape %s"
                    % (name, facts.name, found.dtype, found.shape)
                )
            result[group] = np.minimum(result[group], found)
    LOG.debug(
        "permitted exit graph=%s rules=%d seconds=%.4f",
        facts.name,
        len(held["rules"]),
        time.perf_counter() - started,
    )
    return result
