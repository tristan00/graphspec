import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .names import Names

LOG = logging.getLogger(__name__)
SORTS = Names("sort of reference")
ABSENT = ("stop", "keep")
ENDS = dict(relation="STRING", to_kind="STRING", to_key="STRING")
ORDER = dict(place="INT32")
MEMBERS = ("member_task", "member_name", "member_place", "member_source", "member_target", "member_code")


class Sort:
    def __init__(self, nodes, edges, least, most, ordered):
        started = time.perf_counter()
        if type(nodes) is not bool or type(edges) is not bool or type(ordered) is not bool:
            raise ValueError("a sort of reference states with true or false what it holds and its order")
        if (
            type(least) is not int
            or least < 0
            or (most is not None and (type(most) is not int or most < least))
        ):
            raise ValueError(
                "a sort of reference states the least and the most members it holds, found %r and %r"
                % (least, most)
            )
        if (nodes or edges) != (most is None or most > 0):
            raise ValueError("a sort of reference holds members exactly when it states what they are")
        self.nodes, self.edges, self.least, self.most, self.ordered = nodes, edges, least, most, ordered
        self.whole = not nodes and not edges
        LOG.debug("sort exit seconds=%.4f", time.perf_counter() - started)

    @timed("references.sort.columns")
    def columns(self):
        if self.whole:
            return None
        held = dict(ENDS) if self.edges else {}
        if self.ordered:
            held.update(ORDER)
        return held

    def check(self, where, counts, task, source, target, code, place):
        started = time.perf_counter()
        counts = np.asarray(counts, np.int64)
        if np.any(counts < self.least) or (self.most is not None and np.any(counts > self.most)):
            found = counts[
                (counts < self.least) | (counts > (counts.max() if self.most is None else self.most))
            ]
            raise ValueError(
                "%s names for every task members to the number of %s, and a task names %d"
                % (
                    where,
                    "%d or more" % self.least
                    if self.most is None
                    else "%d" % self.least
                    if self.most == self.least
                    else "%d to %d" % (self.least, self.most),
                    int(found[0]),
                )
            )
        edge = np.asarray(code) >= 0
        if (edge.any() and not self.edges) or ((~edge).any() and not self.nodes):
            raise ValueError(
                "%s names %s, and a task names %s"
                % (
                    where,
                    "nodes and edges" if self.nodes and self.edges else "nodes" if self.nodes else "edges",
                    "an edge" if edge.any() and not self.edges else "a node",
                )
            )
        if self.ordered and len(task):
            first = np.r_[True, task[1:] != task[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(task)), 0))
            if not np.array_equal(place, np.arange(len(task)) - start):
                raise ValueError("%s numbers the members of a task from zero in their order" % where)
            joined = ~first
            before = np.r_[0, np.arange(len(task) - 1)]
            shared = (
                (source == source[before])
                | (source == target[before])
                | (target == source[before])
                | (target == target[before])
            )
            if np.any(joined & ~shared):
                raise ValueError("%s holds edges in order, each sharing a node with the one before" % where)
        LOG.debug(
            "sort check exit tasks=%d members=%d seconds=%.4f",
            len(counts),
            len(task),
            time.perf_counter() - started,
        )

    def resolved(self, source, target, step, observed):
        started = time.perf_counter()
        source, target = np.asarray(source, np.int64), np.asarray(target, np.int64)
        first = observed(source, step)
        second = np.full(len(target), -1, np.int64)
        ended = target >= 0
        second[ended] = observed(target[ended], np.asarray(step)[ended])
        LOG.debug("sort resolved exit members=%d seconds=%.4f", len(source), time.perf_counter() - started)
        return first, second


SORTS.register("node", Sort(True, False, 1, 1, False))
SORTS.register("edge", Sort(False, True, 1, 1, False))
SORTS.register("nodes", Sort(True, False, 1, None, False))
SORTS.register("path", Sort(False, True, 1, None, True))
SORTS.register("subgraph", Sort(True, True, 1, None, False))
SORTS.register("graph", Sort(False, False, 0, 0, False))


def ends(source, target):
    started = time.perf_counter()
    source, target = np.asarray(source, np.int64), np.asarray(target, np.int64)
    ended = target >= 0
    member = np.r_[np.arange(len(source)), np.flatnonzero(ended)]
    entity = np.r_[source, target[ended]]
    side = np.r_[np.zeros(len(source), np.int64), np.ones(int(ended.sum()), np.int64)]
    order = np.lexsort((side, member))
    LOG.debug(
        "ends exit members=%d ends=%d seconds=%.4f", len(source), len(order), time.perf_counter() - started
    )
    return member[order], entity[order]


def starts(task, source, target, tasks, width):
    started = time.perf_counter()
    member, entity = ends(source, target)
    owner = np.asarray(task, np.int64)[member]
    counts = np.bincount(owner, minlength=tasks)
    offsets = np.r_[0, np.cumsum(counts)]
    place = np.minimum(np.arange(width)[None, :], np.maximum(counts - 1, 0)[:, None])
    result = np.full((tasks, width), -1, np.int64)
    held = counts > 0
    result[held] = entity[(offsets[:-1][:, None] + place)[held]]
    LOG.debug("starts exit tasks=%d seconds=%.4f", tasks, time.perf_counter() - started)
    return result
