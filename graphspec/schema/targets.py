import logging
import math
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names

LOG = logging.getLogger(__name__)
TARGETS = Names("kind of target")
UTILITIES = Names("utility")
BETTER = ("higher", "lower")
UTILITY = ("name", "better")


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


class Number:
    scaled = True
    matures = True
    columns = {}
    fields = ()

    @timed("targets.number.series")
    def series(self, stated, where):
        if type(stated) is not str or not stated or ":" in stated:
            raise ValueError("%s names its one series without colons, found %r" % (where, stated))
        return (stated,)

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


class Path:
    scaled = True
    matures = False
    columns = dict(tau="FLOAT", series="STRING")
    fields = ("tau",)

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


class Weighted:
    reads = ("number", "path")
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


TARGETS.register("number", Number())
TARGETS.register("path", Path())
UTILITIES.register("weighted", Weighted())
