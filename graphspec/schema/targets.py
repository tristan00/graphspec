import logging
import math
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .references import SORTS

LOG = logging.getLogger(__name__)
TARGETS = Names("kind of target")
UTILITIES = Names("utility")
BETTER = ("higher", "lower")
UTILITY = ("name", "better")
SUMMED = 1e-6
NEGATIVES = "graphspec.negatives"


@timed("targets.real")
def real(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@timed("targets.listed")
def listed(stated, where):
    if (
        not isinstance(stated, list)
        or not stated
        or len(set(stated)) != len(stated)
        or any(type(name) is not str or not name or ":" in name for name in stated)
    ):
        raise ValueError(
            "%s names its series as a list of distinct names without colons, found %r" % (where, stated)
        )
    return tuple(stated)


def classed(rows, series, where):
    started = time.perf_counter()
    owner = np.asarray(rows["task"], np.int64)
    names = list(rows["series"])
    strangers = sorted({repr(name) for name in names if name not in series})
    if strangers:
        raise ValueError(
            "a point of %s names the class %s, which its kind does not state" % (where, ", ".join(strangers))
        )
    code = np.asarray([series.index(name) for name in names], np.int64)
    order = np.lexsort((code, owner))
    _, first, place, counts = np.unique(owner[order], return_index=True, return_inverse=True, return_counts=True)
    wrong = counts != len(series)
    if not wrong.any():
        wrong = (code[order].reshape(-1, len(series)) != np.arange(len(series))).any(1)
    if wrong.any():
        raise ValueError(
            "a task of graph %s holds under %s one point for every class its kind states, %s, each once"
            % (rows["graph"][int(order[first[np.argmax(wrong)]])], where, ", ".join(series))
        )
    held = np.empty(len(owner), np.int64)
    held[order] = place.reshape(-1)
    LOG.debug("classed exit points=%d seconds=%.4f", len(owner), time.perf_counter() - started)
    return held, np.asarray(rows["y"], np.float64)


@timed("targets.one")
def one(stated, where):
    if type(stated) is not str or not stated or ":" in stated:
        raise ValueError("%s names its one series without colons, found %r" % (where, stated))
    return (stated,)


@timed("targets.unconstrained")
def unconstrained(stated, schema, references, where):
    return {}


@timed("targets.listing")
def listing(about, references, where):
    sort = None if about is None else SORTS.named(references[about]["sort"])
    if sort is None or not sort.nodes or sort.edges:
        raise ValueError(
            "the target of %s is about the reference that lists the nodes its points name, of a sort that holds "
            "nodes only, found %r" % (where, about)
        )
    return sort


def located(rows, where):
    started = time.perf_counter()
    owner = np.asarray(rows["task"], np.int64)
    nodes = list(rows["node"])
    if any(type(node) is not str for node in nodes):
        raise ValueError("every point of %s names by its key the node it scores" % where)
    names, code = np.unique(np.asarray(nodes, str), return_inverse=True)
    _, first, counts = np.unique(owner * max(len(names), 1) + code.reshape(-1), return_index=True, return_counts=True)
    if np.any(counts > 1):
        at = int(first[np.argmax(counts > 1)])
        raise ValueError("a task of graph %s names the node %s twice as %s" % (rows["graph"][at], nodes[at], where))
    LOG.debug("located exit points=%d seconds=%.4f", len(owner), time.perf_counter() - started)
    return owner, np.asarray(rows["y"], np.float64)


@timed("targets.logistic")
def logistic(values):
    return np.exp(-np.logaddexp(0, -np.asarray(values, np.float64)))


@timed("targets.normalised")
def normalised(values, task):
    owner = np.unique(np.asarray(task, np.int64), return_inverse=True)[1].reshape(-1)
    values = np.asarray(values, np.float64)
    top = np.full(owner.max() + 1, -np.inf)
    np.maximum.at(top, owner, values)
    raised = np.exp(values - top[owner])
    return raised / np.bincount(owner, raised)[owner]


@timed("targets.single")
def single(column, series, where):
    if type(column) is not str:
        raise ValueError("%s is one column, found %r" % (where, column))
    return dict.fromkeys(series, column)


@timed("targets.each")
def each(column, series, where):
    if (
        not isinstance(column, dict)
        or set(column) != set(series)
        or any(type(name) is not str for name in column.values())
    ):
        raise ValueError(
            "%s is a column for every class of %s, as a mapping from class to column, found %r"
            % (where, ", ".join(series), column)
        )
    return {name: column[name] for name in series}


class Number:
    scaled = True
    matures = "number"
    names = None
    parameters = ()
    columns = {}
    fields = ()
    checked = staticmethod(unconstrained)

    @timed("targets.number.ripe")
    def ripe(self, column, series, where):
        return single(column, series, where)

    @timed("targets.number.ripened")
    def ripened(self, series, y):
        return [float(value) for value in y]

    @timed("targets.number.series")
    def series(self, stated, where):
        return one(stated, where)

    @timed("targets.number.written")
    def written(self, series):
        return series[0]

    def points(self, rows, series, where):
        started = time.perf_counter()
        owner = np.asarray(rows["task"], np.int64)
        found, counts = np.unique(owner, return_counts=True)
        if np.any(counts > 1):
            twice = int(np.flatnonzero(owner == found[np.argmax(counts > 1)])[0])
            raise ValueError("a task of graph %s holds more than one number as %s" % (rows["graph"][twice], where))
        result = dict(series=[series[0]] * len(owner))
        LOG.debug("number points exit points=%d seconds=%.4f", len(owner), time.perf_counter() - started)
        return result

    @timed("targets.number.staged")
    def staged(self, stated, series, where):
        if not real(stated):
            raise ValueError("%s states its target as a finite number, found %r" % (where, stated))
        return [dict(y=float(stated))]

    @timed("targets.number.read")
    def read(self, rows):
        (row,) = rows
        return row["y"]

    @timed("targets.number.predicted")
    def predicted(self, values, task):
        return values


class Path:
    scaled = True
    matures = None
    names = None
    parameters = ()
    columns = dict(tau="FLOAT", series="STRING")
    fields = ("tau",)
    checked = staticmethod(unconstrained)

    @timed("targets.path.series")
    def series(self, stated, where):
        return listed(stated, where)

    @timed("targets.path.written")
    def written(self, series):
        return list(series)

    def points(self, rows, series, where):
        started = time.perf_counter()
        owner = np.asarray(rows["task"], np.int64)
        tau = np.asarray([np.nan if value is None else value for value in rows["tau"]], np.float64)
        if not np.isfinite(tau).all() or np.any(tau <= 0):
            raise ValueError("the points of %s state their tau finite and strictly positive" % where)
        names = list(rows["series"])
        strangers = sorted({repr(name) for name in names if name not in series})
        if strangers:
            raise ValueError(
                "a point of %s names the series %s, which its kind does not state" % (where, ", ".join(strangers))
            )
        code = np.asarray([series.index(name) for name in names], np.int64)
        order = np.lexsort((tau, code, owner))
        same = (np.diff(owner[order]) == 0) & (np.diff(code[order]) == 0) & (np.diff(tau[order]) == 0)
        if same.any():
            raise ValueError(
                "graph %s has two points with one task, one series and one tau"
                % rows["graph"][int(order[np.flatnonzero(same)[0]])]
            )
        LOG.debug("path points exit points=%d seconds=%.4f", len(owner), time.perf_counter() - started)
        return dict(series=names, tau=tau.tolist())

    @timed("targets.path.staged")
    def staged(self, stated, series, where):
        if (
            not isinstance(stated, list)
            or not stated
            or any(not isinstance(point, (list, tuple)) or len(point) != 3 for point in stated)
        ):
            raise ValueError(
                "%s states its target as a list of points, each its tau, its series and its number; found %r"
                % (where, stated)
            )
        return [dict(tau=tau, series=name, y=y) for tau, name, y in stated]

    @timed("targets.path.read")
    def read(self, rows):
        return [[row["tau"], row["series"], row["y"]] for row in rows]

    @timed("targets.path.predicted")
    def predicted(self, values, task):
        return values


class Over:
    scaled = False
    names = None
    parameters = ()
    columns = dict(series="STRING")
    fields = ()
    checked = staticmethod(unconstrained)

    @timed("targets.over.series")
    def series(self, stated, where):
        names = listed(stated, where)
        if len(names) < self.least:
            raise ValueError("%s names at least %d classes, found %r" % (where, self.least, stated))
        return names

    @timed("targets.over.written")
    def written(self, series):
        return list(series)


class Class(Over):
    least = 2
    matures = "category"

    @timed("targets.class.ripe")
    def ripe(self, column, series, where):
        return single(column, series, where)

    @timed("targets.class.ripened")
    def ripened(self, series, y):
        return [name if value == 1 else None for name, value in zip(series, y)]

    @timed("targets.class.points")
    def points(self, rows, series, where):
        place, y = classed(rows, series, where)
        if np.any((y != 0) & (y != 1)) or np.any(np.bincount(place, y) != 1):
            raise ValueError("the points of %s state 1 for the one class of a task and 0 for every other" % where)
        return dict(series=list(rows["series"]))

    @timed("targets.class.staged")
    def staged(self, stated, series, where):
        if type(stated) is not str or stated not in series:
            raise ValueError(
                "%s states its target as one of the classes %s, found %r" % (where, ", ".join(series), stated)
            )
        return [dict(series=name, y=float(name == stated)) for name in series]

    @timed("targets.class.read")
    def read(self, rows):
        (found,) = [row["series"] for row in rows if row["y"] == 1]
        return found

    @timed("targets.class.predicted")
    def predicted(self, values, task):
        return normalised(values, task)


class Classes(Over):
    least = 1
    matures = "flag"

    @timed("targets.classes.ripe")
    def ripe(self, column, series, where):
        return each(column, series, where)

    @timed("targets.classes.ripened")
    def ripened(self, series, y):
        return [bool(value == 1) for value in y]

    @timed("targets.classes.points")
    def points(self, rows, series, where):
        _, y = classed(rows, series, where)
        if np.any((y != 0) & (y != 1)):
            raise ValueError(
                "the points of %s state 1 for every class of the set of a task and 0 for every other" % where
            )
        return dict(series=list(rows["series"]))

    @timed("targets.classes.staged")
    def staged(self, stated, series, where):
        if (
            not isinstance(stated, list)
            or any(type(name) is not str or name not in series for name in stated)
            or len(set(stated)) != len(stated)
        ):
            raise ValueError(
                "%s states its target as a list of distinct classes of %s, possibly none, found %r"
                % (where, ", ".join(series), stated)
            )
        return [dict(series=name, y=float(name in stated)) for name in series]

    @timed("targets.classes.read")
    def read(self, rows):
        return [row["series"] for row in rows if row["y"] == 1]

    @timed("targets.classes.predicted")
    def predicted(self, values, task):
        return logistic(values)


class Proportions(Over):
    least = 2
    matures = "number"

    @timed("targets.proportions.ripe")
    def ripe(self, column, series, where):
        return each(column, series, where)

    @timed("targets.proportions.ripened")
    def ripened(self, series, y):
        return [float(value) for value in y]

    @timed("targets.proportions.points")
    def points(self, rows, series, where):
        place, y = classed(rows, series, where)
        if not np.isfinite(y).all() or np.any(y < 0) or np.any(np.abs(np.bincount(place, y) - 1) > SUMMED):
            raise ValueError(
                "the points of %s state shares of at least 0 that sum to 1 within %g over the classes of a task"
                % (where, SUMMED)
            )
        return dict(series=list(rows["series"]))

    @timed("targets.proportions.staged")
    def staged(self, stated, series, where):
        if (
            not isinstance(stated, dict)
            or set(stated) != set(series)
            or not all(real(share) and share >= 0 for share in stated.values())
            or abs(sum(stated.values()) - 1) > SUMMED
        ):
            raise ValueError(
                "%s states its target as a share of at least 0 for every class of %s, summing to 1 within %g, "
                "found %r" % (where, ", ".join(series), SUMMED, stated)
            )
        return [dict(series=name, y=float(stated[name])) for name in series]

    @timed("targets.proportions.read")
    def read(self, rows):
        return {row["series"]: row["y"] for row in rows}

    @timed("targets.proportions.predicted")
    def predicted(self, values, task):
        return normalised(values, task)


class Pointed:
    scaled = False
    matures = None
    columns = dict(node="STRING")
    fields = ()

    @timed("targets.pointed.series")
    def series(self, stated, where):
        return one(stated, where)

    @timed("targets.pointed.written")
    def written(self, series):
        return series[0]


class Members(Pointed):
    names = "members"
    parameters = ()

    @timed("targets.members.checked")
    def checked(self, stated, schema, references, where):
        listing(stated["about"], references, where)
        return {}


class Ranking(Members):
    @timed("targets.ranking.points")
    def points(self, rows, series, where):
        owner, y = located(rows, where)
        if not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError("the points of %s state the relevance of each item, a finite number of at least 0" % where)
        found = np.unique(owner)
        if np.any(np.bincount(np.searchsorted(found, owner), y > 0, len(found)) == 0):
            raise ValueError("the points of %s state for every task one item at least of a relevance above 0" % where)
        return dict(series=[series[0]] * len(owner), node=list(rows["node"]))

    @timed("targets.ranking.staged")
    def staged(self, stated, series, where):
        if (
            not isinstance(stated, dict)
            or any(type(key) is not str for key in stated)
            or not all(real(value) and value >= 0 for value in stated.values())
            or not any(value > 0 for value in stated.values())
        ):
            raise ValueError(
                "%s states its target as a mapping from the key of every item its task lists to its relevance, a "
                "finite number of at least 0, one at least above 0, found %r" % (where, stated)
            )
        return [dict(node=key, y=float(value)) for key, value in stated.items()]

    @timed("targets.ranking.read")
    def read(self, rows):
        return {row["node"]: row["y"] for row in rows}

    @timed("targets.ranking.predicted")
    def predicted(self, values, task):
        return normalised(values, task)


class Nodes(Members):
    @timed("targets.nodes.points")
    def points(self, rows, series, where):
        owner, y = located(rows, where)
        if np.any((y != 0) & (y != 1)):
            raise ValueError("the points of %s state 1 for a node that belongs and 0 for one that does not" % where)
        return dict(series=[series[0]] * len(owner), node=list(rows["node"]))

    @timed("targets.nodes.staged")
    def staged(self, stated, series, where):
        if (
            not isinstance(stated, dict)
            or not stated
            or any(type(key) is not str for key in stated)
            or any(type(value) is not bool for value in stated.values())
        ):
            raise ValueError(
                "%s states its target as a mapping from the key of every node its task lists to whether it "
                "belongs, true or false, found %r" % (where, stated)
            )
        return [dict(node=key, y=float(value)) for key, value in stated.items()]

    @timed("targets.nodes.read")
    def read(self, rows):
        return {row["node"]: row["y"] == 1 for row in rows}

    @timed("targets.nodes.predicted")
    def predicted(self, values, task):
        return logistic(values)


class Link(Pointed):
    names = "linked"
    parameters = ("relation",)

    @timed("targets.link.checked")
    def checked(self, stated, schema, references, where):
        about, relation = stated["about"], stated["relation"]
        sort = None if about is None else SORTS.named(references[about]["sort"])
        if sort is None or not sort.nodes or sort.edges or sort.most != 1:
            raise ValueError(
                "the target of %s is about the reference of the one node its link starts from, found %r"
                % (where, about)
            )
        edge = schema.edges.get(relation) if type(relation) is str else None
        joined = set() if edge is None else {pair[0] for pair in edge["endpoints"]}
        if edge is not None and not edge["directed"]:
            joined |= {pair[1] for pair in edge["endpoints"]}
        if edge is None or not set(references[about]["kinds"]) <= joined:
            raise ValueError(
                "the target of %s states as its relation an edge table that joins the kinds of %s to other nodes, "
                "found %r" % (where, about, relation)
            )
        return dict(relation=relation)

    @timed("targets.link.points")
    def points(self, rows, series, where):
        owner, y = located(rows, where)
        if len(np.unique(owner)) != len(owner) or np.any(y != 1):
            raise ValueError("the points of %s state one node, the one its task links to, with y 1" % where)
        return dict(series=[series[0]] * len(owner), node=list(rows["node"]))

    @timed("targets.link.staged")
    def staged(self, stated, series, where):
        if type(stated) is not str or not stated:
            raise ValueError(
                "%s states its target as the key of the node its task links to, found %r" % (where, stated)
            )
        return [dict(node=stated, y=1.0)]

    @timed("targets.link.read")
    def read(self, rows):
        (row,) = rows
        return row["node"]

    @timed("targets.link.predicted")
    def predicted(self, values, task):
        return normalised(values, task)


class Weighted:
    reads = ("number", "path", "class", "classes", "proportions", "ranking", "nodes", "link")
    parameters = ("weights",)

    @timed("targets.weighted.checked")
    def checked(self, stated, series, where):
        weights = stated["weights"]
        if (
            not isinstance(weights, dict)
            or sorted(weights) != sorted(series)
            or not all(real(value) for value in weights.values())
        ):
            raise ValueError(
                "%s weighs every series of its target, %s, by a finite number, found %r"
                % (where, ", ".join(series), weights)
            )
        return dict(weights={name: float(weights[name]) for name in series})

    def value(self, stated, points):
        started = time.perf_counter()
        weight = np.asarray([stated["weights"][name] for name in points["series"]], np.float64)
        tasks, owner = np.unique(np.asarray(points["task"], np.int64), return_inverse=True)
        result = np.bincount(owner.reshape(-1), weight * np.asarray(points["y"], np.float64), minlength=len(tasks))
        LOG.debug("weighted value exit tasks=%d seconds=%.4f", len(tasks), time.perf_counter() - started)
        return tasks, result


@timed("targets.valued")
def valued(target, points):
    stated = target["utility"]
    return UTILITIES.named(stated["name"]).value(stated, points)


def needed(schema):
    started = time.perf_counter()
    result = {}
    for kind, entry in schema.tasks["kinds"].items():
        target = entry["target"]
        if target is not None and TARGETS.named(target["kind"]).names == "linked":
            result[kind] = (NEGATIVES, {"negatives/of": "link:%s:%s" % (target["relation"], target["about"])})
    LOG.debug("needed exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


TARGETS.register("number", Number())
TARGETS.register("path", Path())
TARGETS.register("class", Class())
TARGETS.register("classes", Classes())
TARGETS.register("proportions", Proportions())
TARGETS.register("ranking", Ranking())
TARGETS.register("nodes", Nodes())
TARGETS.register("link", Link())
UTILITIES.register("weighted", Weighted())
