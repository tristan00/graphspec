import logging
import time

import numpy as np

LOG = logging.getLogger(__name__)
BINS = 30
WINDOWS = (90.0, 180.0, 365.0, 730.0, np.inf)
DAMPINGS = (30.0, 90.0, 180.0, 365.0, 730.0, 1825.0, np.inf)


def histories(part):
    started = time.perf_counter()
    count = len(part["start"]) - 1
    date, level = np.zeros(count), np.zeros(count)
    order = np.lexsort((part["path_step"], part["path_series"], part["path_graph"]))
    for before, path in zip(order[:-1], order[1:]):
        if (part["path_graph"][before], part["path_series"][before]) != (
            part["path_graph"][path],
            part["path_series"][path],
        ):
            continue
        gap = part["path_turn"][path] - part["path_turn"][before]
        if gap == 0:
            date[path], level[path] = date[before], level[before]
            continue
        points = slice(part["start"][before], part["start"][before + 1])
        nearest = np.abs(part["tau"][points] - gap).argmin()
        elapsed = part["tau"][points][nearest]
        if abs(elapsed - gap) >= 1:
            raise ValueError("a decision is absent from the future of the decision before it")
        date[path], level[path] = date[before] + elapsed, level[before] + part["target"][points][nearest]
    LOG.info("histories exit seconds=%.3f paths=%d", time.perf_counter() - started, count)
    return order, date, level


def rates(part, window):
    started = time.perf_counter()
    order, date, level = histories(part)
    rate = np.zeros(len(date))
    first = 0
    for position, path in enumerate(order):
        origin = order[first]
        if (part["path_graph"][origin], part["path_series"][origin]) != (
            part["path_graph"][path],
            part["path_series"][path],
        ):
            first = position
        while date[order[first]] < date[path] - window:
            first += 1
        elapsed = date[path] - date[order[first]]
        if elapsed > 0:
            rate[path] = (level[path] - level[order[first]]) / elapsed
    LOG.info("rates exit seconds=%.3f window=%s", time.perf_counter() - started, window)
    return rate


def lengths(part):
    return np.diff(part["start"])


def extrapolate(part, rate, damping):
    tau, slope = part["tau"], np.repeat(rate, lengths(part))
    if np.isinf(damping):
        return slope * tau
    return slope * damping * (1 - np.exp(-tau / damping))


def absolute(part, values, series):
    member = np.repeat(part["path_series"], lengths(part)) == series
    weight = np.repeat(1.0 / lengths(part), lengths(part))[member]
    return float((weight * np.abs(values - part["target"])[member]).sum() / weight.sum())


def by_horizon(train, held, span, reduce):
    started = time.perf_counter()
    edges = np.linspace(0, span, BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    result = np.empty(len(held["tau"]))
    for series in np.unique(train["series"]):
        member = train["series"] == series
        place = np.clip(np.searchsorted(edges, train["tau"][member], side="right") - 1, 0, BINS - 1)
        level = np.asarray([reduce(train["target"][member][place == index]) for index in range(BINS)])
        if not np.isfinite(level).all():
            raise ValueError("a horizon bin has no training points")
        chosen = held["series"] == series
        result[chosen] = np.interp(held["tau"][chosen], centers, level)
    LOG.info("by_horizon exit seconds=%.3f", time.perf_counter() - started)
    return result


def damped(train, held):
    started = time.perf_counter()
    result, chosen = np.empty(len(held["tau"])), {}
    known = {window: (rates(train, window), rates(held, window)) for window in WINDOWS}
    for series in np.unique(train["series"]):
        best = min(
            ((window, damping) for window in WINDOWS for damping in DAMPINGS),
            key=lambda pair: absolute(train, extrapolate(train, known[pair[0]][0], pair[1]), series),
        )
        chosen[int(series)] = best
        member = held["series"] == series
        result[member] = extrapolate(held, known[best[0]][1], best[1])[member]
    LOG.info("damped exit seconds=%.3f chosen=%s", time.perf_counter() - started, chosen)
    return result, chosen


def build(sequences, span):
    started = time.perf_counter()
    train, held = sequences.train, sequences.held
    fitted, chosen = damped(train, held)
    result = {
        "reference: no change": np.zeros(len(held["tau"])),
        "reference: training mean by horizon": by_horizon(train, held, span, np.mean),
        "reference: training median by horizon": by_horizon(train, held, span, np.median),
        "reference: own drift, last 365 days": extrapolate(held, rates(held, 365.0), np.inf),
        "reference: own drift, since run start": extrapolate(held, rates(held, np.inf), np.inf),
        "reference: own damped drift, fitted on training runs": fitted,
    }
    LOG.info("references exit seconds=%.3f rows=%d", time.perf_counter() - started, len(result))
    return result, chosen
