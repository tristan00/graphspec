import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names

LOG = logging.getLogger(__name__)
ALWAYS = -(1 << 62)
NEVER = 1 << 62
MODELS = ("none", "order", "clock", "clocks")
STAMPS = dict(none=(), order=("since", "until"), clock=("since", "until"))
STAMPS["clocks"] = ("since", "until", "known", "replaced")
UNITED = ("clock", "clocks")
OPEN = dict(since=ALWAYS, until=NEVER, replaced=NEVER)
TIES = Names("rule for ties", True)
SEVERAL = Names("choice among several facts", True)
ABOUT = Names("rule for an edge about an edge", True)


class Sequence:
    version = 1
    columns = dict(seq="INT32")

    @timed("clock.sequence.keys")
    def keys(self, facts):
        return (facts["seq"],)


class Key:
    version = 1
    columns = {}

    @timed("clock.key.keys")
    def keys(self, facts):
        return (facts["code"], facts["source"], facts["target"])


@timed("clock.first")
def first(starts, stops):
    return starts


@timed("clock.last")
def last(starts, stops):
    return stops - 1


class Stop:
    version = 1

    def kept(self, lacking, named):
        started = time.perf_counter()
        lacking = np.asarray(lacking, bool)
        if lacking.any():
            raise ValueError(
                "%s holds at a time when the edge it is about does not, and the rule stop states that this "
                "stops the work" % named(int(np.flatnonzero(lacking)[0]))
            )
        LOG.debug("about stop exit rows=%d seconds=%.4f", len(lacking), time.perf_counter() - started)
        return ~lacking


class Lapse:
    version = 1

    @timed("clock.lapse.kept")
    def kept(self, lacking, named):
        return ~np.asarray(lacking, bool)


first.version = 1
last.version = 1
TIES.register("sequence", Sequence())
TIES.register("key", Key())
SEVERAL.register("first", first)
SEVERAL.register("last", last)
ABOUT.register("stop", Stop())
ABOUT.register("lapse", Lapse())


@timed("clock.shown")
def shown(moment):
    moment = int(moment)
    if moment <= ALWAYS:
        return "the time before every time"
    if moment >= NEVER:
        return "the time after every time"
    return "the time %d" % moment


@timed("clock.keys")
def keys(name, facts):
    return () if name is None else tuple(TIES.named(name).keys(facts))


def untied(graph, named, ties, owner, stamps, facts):
    started = time.perf_counter()
    held = keys(ties, facts)
    columns = [np.asarray(owner, np.int64)] + [np.asarray(key, np.int64) for key in held]
    columns.append(np.asarray(stamps["since"], np.int64))
    known, replaced = np.asarray(stamps["known"], np.int64), np.asarray(stamps["replaced"], np.int64)
    order = np.lexsort([known] + columns[::-1])
    same = known[order][1:] < replaced[order][:-1]
    for column in columns:
        same &= column[order][1:] == column[order][:-1]
    if same.any():
        row = int(order[1:][same][0])
        raise ValueError(
            "two facts about %s of graph %s hold from %s and %s"
            % (
                named(row),
                graph,
                shown(columns[-1][row]),
                "the schema states no rule for ties"
                if ties is None
                else "the rule %s for ties leaves them unordered" % ties,
            )
        )
    LOG.debug("untied exit graph=%s rows=%d seconds=%.4f", graph, len(order), time.perf_counter() - started)


@timed("clock.filled")
def filled(count, base, rows, times):
    result = np.array(np.broadcast_to(np.asarray(base, np.int64), (count,)))
    result[np.asarray(rows, np.int64)] = np.asarray(times, np.int64)
    return result


@timed("clock.differing")
def differing(values, base):
    values = np.asarray(values, np.int64)
    rows = np.flatnonzero(values != np.broadcast_to(np.asarray(base, np.int64), values.shape))
    return rows.astype(np.int64), values[rows]


def stamped(table, model, label, graph):
    started = time.perf_counter()
    count = table.num_rows

    @timed("time.stamped.read")
    def read(name, base):
        if name not in STAMPS[model]:
            return np.array(np.broadcast_to(np.asarray(base, np.int64), (count,)))
        column = table.column(name)
        values = np.asarray(column.fill_null(0).to_numpy(zero_copy_only=False), np.int64)
        if np.any((values <= ALWAYS) | (values >= NEVER)):
            raise ValueError("a %s of graph %s states a time outside the range of times" % (label, graph))
        missing = np.asarray(column.is_null().to_numpy(zero_copy_only=False), bool)
        return np.where(missing, base, values).astype(np.int64)

    since = read("since", ALWAYS)
    until = read("until", NEVER)
    known = read("known", since)
    replaced = read("replaced", NEVER)
    if np.any(until <= since):
        moment = since[np.flatnonzero(until <= since)[0]]
        raise ValueError(
            "a %s of graph %s holds from %s until no later than that" % (label, graph, shown(moment))
        )
    if np.any(replaced <= known):
        moment = known[np.flatnonzero(replaced <= known)[0]]
        raise ValueError(
            "a %s of graph %s is known from %s and replaced no later than that"
            % (label, graph, shown(moment))
        )
    LOG.debug(
        "stamped exit graph=%s label=%s rows=%d seconds=%.4f",
        graph,
        label,
        count,
        time.perf_counter() - started,
    )
    return dict(since=since, until=until, known=known, replaced=replaced)


def watched(table, observers, label, graph):
    started = time.perf_counter()
    rows, found = [], []
    if "observers" in table.column_names:
        position = {name: code for code, name in enumerate(observers)}
        for row, names in enumerate(table.column("observers").to_pylist()):
            if names is None:
                continue
            if not names or None in names or len(set(names)) != len(names):
                raise ValueError(
                    "a %s of graph %s names its observers once each, or states none" % (label, graph)
                )
            strangers = sorted(set(names) - set(position))
            if strangers:
                raise ValueError(
                    "a %s of graph %s names observers the bank does not hold: %s"
                    % (label, graph, ", ".join(strangers))
                )
            rows.extend([row] * len(names))
            found.extend(sorted(position[name] for name in names))
    LOG.debug(
        "watched exit graph=%s label=%s pairs=%d seconds=%.4f",
        graph,
        label,
        len(rows),
        time.perf_counter() - started,
    )
    return np.asarray(rows, np.int64), np.asarray(found, np.int32)
