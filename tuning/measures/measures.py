import logging
import time
from dataclasses import dataclass

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship
from graphspec.schema.targets import TARGETS

LOG = logging.getLogger(__name__)
MEASURES = Names("measure")


@dataclass(frozen=True)
class Scored:
    target: np.ndarray
    prediction: np.ndarray
    task: np.ndarray
    series: np.ndarray
    names: tuple
    scale: np.ndarray


def scored(target, prediction, task, series, names, scale):
    started = time.perf_counter()
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    task, series = np.asarray(task, np.int64), np.asarray(series, np.int64)
    if target.ndim != 1 or not target.shape == prediction.shape == task.shape == series.shape:
        raise ValueError("every point scored needs a target, a prediction, a task and a series")
    if not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("the predictions scored must be nonempty and finite")
    lacking = [name for code, name in enumerate(names) if not (series == code).any()]
    if lacking:
        raise ValueError("the series %s have no point scored" % ", ".join(lacking))
    result = Scored(target, prediction, task, series, tuple(names), np.asarray(scale, np.float64))
    LOG.debug("scored exit points=%d seconds=%.4f", len(target), time.perf_counter() - started)
    return result


@timed("measures.determined")
def determined(target, prediction, where):
    variance = float(target.var())
    if variance == 0:
        raise ValueError("R² requires varying targets scored in %s" % where)
    return 1 - float(((prediction - target) ** 2).mean()) / variance


class Squared:
    reads = ("number", "path")
    better = "lower"
    label, short = "Validation MSE", "MSE"

    @timed("measures.squared")
    def __call__(self, held):
        error = (held.prediction - held.target) ** 2
        return dict(
            value=float(error.mean()),
            series={name: float(error[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


class Determination:
    reads = ("number",)
    better = "higher"
    label, short = "Validation R²", "R²"

    @timed("measures.determination")
    def __call__(self, held):
        return dict(
            value=determined(held.target, held.prediction, "every series"),
            series={
                name: determined(
                    held.target[held.series == code], held.prediction[held.series == code], "the series " + name
                )
                for code, name in enumerate(held.names)
            },
        )


class Balanced:
    reads = ("number", "path")
    better = "lower"
    label, short = "Validation loss", "Loss"

    @timed("measures.balanced")
    def __call__(self, held):
        series = {}
        for code, name in enumerate(held.names):
            member = held.series == code
            owner = np.unique(held.task[member], return_inverse=True)[1].reshape(-1)
            weight = 1.0 / np.bincount(owner)[owner]
            error = (held.prediction[member] - held.target[member]) ** 2
            series[name] = float((weight * error).sum() / weight.sum()) / float(held.scale[code]) ** 2
        return dict(value=float(np.mean(list(series.values()))), series=series)


@timed("measures.pertask")
def pertask(held, part):
    tasks = len(np.unique(held.task))
    return dict(
        value=float(part.sum() / tasks),
        series={name: float(part[held.series == code].sum() / tasks) for code, name in enumerate(held.names)},
    )


class Accuracy:
    reads = ("class",)
    better = "higher"
    label, short = "Validation accuracy", "Accuracy"

    @timed("measures.accuracy")
    def __call__(self, held):
        owner = np.unique(held.task, return_inverse=True)[1].reshape(-1)
        order = np.lexsort((-held.prediction, owner))
        first = np.unique(owner[order], return_index=True)[1]
        picked = np.zeros(len(owner), bool)
        picked[order[first]] = True
        agreed = picked == (held.target == 1)
        return dict(
            value=float((held.target[picked] == 1).mean()),
            series={name: float(agreed[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


class Entropy:
    reads = ("class", "proportions")
    better = "lower"
    label, short = "Validation log loss", "Log loss"

    @timed("measures.entropy")
    def __call__(self, held):
        part, stated = np.zeros(len(held.target)), held.target > 0
        part[stated] = -held.target[stated] * np.log(held.prediction[stated])
        return pertask(held, part)


class Binary:
    reads = ("classes", "nodes")
    better = "lower"
    label, short = "Validation binary log loss", "Binary log loss"

    @timed("measures.binary")
    def __call__(self, held):
        part, stated = np.zeros(len(held.target)), held.target == 1
        part[stated] = -np.log(held.prediction[stated])
        part[~stated] = -np.log1p(-held.prediction[~stated])
        return dict(
            value=float(part.mean()),
            series={name: float(part[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


class Divergence:
    reads = ("proportions",)
    better = "lower"
    label, short = "Validation divergence", "Divergence"

    @timed("measures.divergence")
    def __call__(self, held):
        part, stated = np.zeros(len(held.target)), held.target > 0
        part[stated] = held.target[stated] * np.log(held.target[stated] / held.prediction[stated])
        return pertask(held, part)


def ranked(held):
    started = time.perf_counter()
    owner = np.unique(held.task, return_inverse=True)[1].reshape(-1)
    order = np.lexsort((-held.prediction, owner))
    first = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
    position = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    tied = np.cumsum(np.r_[True, (np.diff(owner[order]) != 0) | (np.diff(held.prediction[order]) != 0)]) - 1
    counts = np.bincount(tied)
    rank, discount = np.empty(len(order)), np.empty(len(order))
    rank[order] = (np.bincount(tied, position) / counts)[tied] + 1
    discount[order] = (np.bincount(tied, 1 / np.log2(position + 2)) / counts)[tied]
    LOG.debug("ranked exit points=%d seconds=%.4f", len(order), time.perf_counter() - started)
    return owner, rank, discount, 1 / np.log2(position + 2)


@timed("measures.tasked")
def tasked(held, owner, values):
    series = np.zeros(owner.max() + 1, np.int64)
    series[owner] = held.series
    return dict(
        value=float(values.mean()),
        series={name: float(values[series == code].mean()) for code, name in enumerate(held.names)},
    )


@timed("measures.first")
def first(held):
    owner, rank, _, _ = ranked(held)
    best = np.full(owner.max() + 1, np.inf)
    np.minimum.at(best, owner[held.target > 0], rank[held.target > 0])
    return owner, best


class Gain:
    reads = ("ranking",)
    better = "higher"
    label, short = "Validation normalised discounted gain", "NDCG"

    @timed("measures.gain")
    def __call__(self, held):
        owner, _, discount, ideal = ranked(held)
        tasks = owner.max() + 1
        gained = np.bincount(owner, held.target * discount, tasks)
        best = np.lexsort((-held.target, owner))
        return tasked(held, owner, gained / np.bincount(owner[best], held.target[best] * ideal, tasks))


class Reciprocal:
    reads = ("ranking",)
    better = "higher"
    label, short = "Validation reciprocal rank", "MRR"

    @timed("measures.reciprocal")
    def __call__(self, held):
        owner, best = first(held)
        return tasked(held, owner, 1 / best)


@timed("measures.decided")
def decided(held):
    owner = np.unique(held.task, return_inverse=True)[1].reshape(-1)
    tasks, chosen, stated = owner.max() + 1, held.prediction >= 0.5, held.target == 1
    return owner, *(np.bincount(owner, values, tasks) for values in (chosen & stated, chosen, stated))


class Agreement:
    reads = ("nodes",)
    better = "higher"
    label, short = "Validation F1", "F1"

    @timed("measures.agreement")
    def __call__(self, held):
        owner, both, chosen, stated = decided(held)
        total = chosen + stated
        return tasked(held, owner, np.where(total > 0, 2 * both / np.maximum(total, 1), 1.0))


class Jaccard:
    reads = ("nodes",)
    better = "higher"
    label, short = "Validation Jaccard index", "Jaccard"

    @timed("measures.jaccard")
    def __call__(self, held):
        owner, both, chosen, stated = decided(held)
        union = chosen + stated - both
        return tasked(held, owner, np.where(union > 0, both / np.maximum(union, 1), 1.0))


def predicted(kinds, aims, task, values):
    started = time.perf_counter()
    result = np.asarray(values, np.float64).copy()
    for code, kind in enumerate(kinds):
        chosen = aims == code
        if chosen.any():
            result[chosen] = TARGETS.named(kind).predicted(result[chosen], task[chosen])
    LOG.debug("predicted exit points=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def restricted(held, wanted):
    started = time.perf_counter()
    chosen = np.isin(held.series, wanted)
    place = np.full(len(held.names), -1, np.int64)
    place[wanted] = np.arange(len(wanted))
    result = Scored(
        held.target[chosen],
        held.prediction[chosen],
        held.task[chosen],
        place[held.series[chosen]],
        tuple(held.names[code] for code in wanted),
        held.scale[wanted],
    )
    LOG.debug("restricted exit points=%d seconds=%.4f", int(chosen.sum()), time.perf_counter() - started)
    return result


@timed("measures.read")
def read(kinds, name):
    return tuple(series for series, kind in kinds.items() if kind in MEASURES.named(name).reads)


def measured(names, held, kinds):
    started = time.perf_counter()
    result = {}
    for name in names:
        wanted = [held.names.index(series) for series in read(kinds, name)]
        result[name] = MEASURES.named(name)(restricted(held, wanted))
    LOG.debug("measured exit measures=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


MEASURES.register("mse", Squared())
MEASURES.register("r2", Determination())
MEASURES.register("balanced", Balanced())
MEASURES.register("accuracy", Accuracy())
MEASURES.register("entropy", Entropy())
MEASURES.register("binary", Binary())
MEASURES.register("divergence", Divergence())
MEASURES.register("gain", Gain())
MEASURES.register("reciprocal", Reciprocal())
MEASURES.register("f1", Agreement())
MEASURES.register("jaccard", Jaccard())
ship([MEASURES])
