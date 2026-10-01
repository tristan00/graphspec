import logging
import time
from dataclasses import dataclass

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship

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
ship([MEASURES])
