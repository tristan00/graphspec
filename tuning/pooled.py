import logging
import math
import time

import numpy as np
from numba import njit
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from . import measures

LOG = logging.getLogger(__name__)
THRESHOLDS = 200
SHARE = 0.1
PERSISTENCE = 0.9
ENOUGH = 5


@njit(cache=True)
def violation(actual, predicted):
    size = len(actual)
    low, high = actual.min(), actual.max()
    total = 0.0
    right, pairs = 0, 0
    for i in range(size):
        worst = 0.0
        for j in range(size):
            if (predicted[i] < predicted[j]) != (actual[i] < actual[j]):
                worst = max(worst, abs(actual[i] - actual[j]) / (high - low))
            if j > i and actual[i] != actual[j]:
                pairs += 1
                if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                    right += 1
        total += worst
    return total / size, right / pairs


def moments(actual, predicted, weight):
    weight = weight / weight.sum()
    first, second = (weight * actual).sum(), (weight * predicted).sum()
    spread = math.sqrt((weight * (actual - first) ** 2).sum())
    width = math.sqrt((weight * (predicted - second) ** 2).sum())
    shared = (weight * (actual - first) * (predicted - second)).sum()
    return weight, first, second, spread, width, shared


def overlap(first, second, persistence):
    seen_first, seen_second, common, total = set(), set(), 0, 0.0
    for depth, (one, other) in enumerate(zip(first, second), 1):
        common += 1 if one == other else (one in seen_second) + (other in seen_first)
        seen_first.add(one)
        seen_second.add(other)
        total += persistence**depth * common / depth
    return common / len(first) * persistence ** len(first) + (1 - persistence) / persistence * total


def dependence(first, second):
    def centred(values):
        gaps = np.abs(values[:, None] - values[None, :])
        return gaps - gaps.mean(0) - gaps.mean(1)[:, None] + gaps.mean()

    one, other = centred(first), centred(second)
    scale = math.sqrt((one * one).mean() * (other * other).mean())
    return math.sqrt(max(0.0, (one * other).mean()) / scale) if scale > 0 else math.nan


def ranking(actual, predicted, groups):
    names = (
        "spearman",
        "kendall",
        "weighted kendall",
        "pairwise accuracy",
        "distance correlation",
        "ndcg",
        "top share overlap",
        "rank biased overlap",
        "rank violation",
        "regret",
        "captured gain",
        "within run spearman",
    )
    if len(actual) < 3 or np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        return dict.fromkeys(names, math.nan)
    kept = max(1, math.ceil(SHARE * len(actual)))
    by_guess, by_truth = np.argsort(-predicted, kind="stable"), np.argsort(-actual, kind="stable")
    gain = actual - actual.min()
    discount = 1 / np.log2(np.arange(kept) + 2)
    broken, right = violation(actual, predicted)
    regret, captured, inside = [], [], []
    for run in np.unique(groups):
        member = groups == run
        truth, guess = actual[member], predicted[member]
        if len(truth) >= 2 and np.ptp(truth) > 0:
            picked = truth[guess.argmax()]
            regret.append(truth.max() - picked)
            captured.append((picked - truth.mean()) / (truth.max() - truth.mean()))
        if len(truth) >= ENOUGH and np.ptp(truth) > 0 and np.ptp(guess) > 0:
            inside.append(measures.correlation(measures.ranks(truth), measures.ranks(guess)))
    return {
        "spearman": measures.correlation(measures.ranks(actual), measures.ranks(predicted)),
        "kendall": measures.concordance(actual, predicted),
        "weighted kendall": float(stats.weightedtau(actual, predicted).statistic),
        "pairwise accuracy": right,
        "distance correlation": dependence(actual, predicted),
        "ndcg": float((gain[by_guess[:kept]] * discount).sum() / (gain[by_truth[:kept]] * discount).sum()),
        "top share overlap": len(set(by_guess[:kept].tolist()) & set(by_truth[:kept].tolist())) / kept,
        "rank biased overlap": overlap(by_guess.tolist(), by_truth.tolist(), PERSISTENCE),
        "rank violation": broken,
        "regret": float(np.mean(regret)) if regret else math.nan,
        "captured gain": float(np.mean(captured)) if captured else math.nan,
        "within run spearman": float(np.mean(inside)) if inside else math.nan,
    }


def elementary(actual, predicted, weight, thresholds):
    low, high = np.minimum(actual, predicted), np.maximum(actual, predicted)
    mean, median = np.empty(len(thresholds)), np.empty(len(thresholds))
    for index, threshold in enumerate(thresholds):
        inside = (low <= threshold) & (threshold < high)
        mean[index] = (weight[inside] * np.abs(actual[inside] - threshold)).sum()
        median[index] = weight[inside].sum()
    return mean, median


def dominated(curves):
    result = np.zeros(len(curves))
    for index, curve in enumerate(curves):
        for other in curves:
            result[index] += bool((other <= curve).all() and (other < curve).any())
    return result


def one(actual, predicted, weight):
    weight, first, second, spread, width, shared = moments(actual, predicted, weight)
    error = predicted - actual
    square = (weight * error**2).sum()
    related = shared / (spread * width) if width > 0 else math.nan
    result = {
        "pooled/spread ratio": width / spread,
        "pooled/normalised bias": (second - first) / spread,
        "pooled/r squared": 1 - square / spread**2,
        "pooled/concordance correlation": 2 * shared / (spread**2 + width**2 + (second - first) ** 2),
        "pooled/willmott": willmott(actual, predicted, weight, first),
        "pooled/theil u1": math.sqrt(square)
        / (math.sqrt((weight * predicted**2).sum()) + math.sqrt((weight * actual**2).sum())),
        "pooled/theil u2": math.sqrt(square / (weight * actual**2).sum()),
        "pooled/bias share": (second - first) ** 2 / square,
        "pooled/variance share": (width - spread) ** 2 / square,
        "pooled/pearson": related,
        "pooled/sign test": directions(actual, predicted),
    }
    if width > 0:
        ranked = moments(measures.ranks(actual), measures.ranks(predicted), weight)
        slope = shared / width**2
        fitted = IsotonicRegression(out_of_bounds="clip").fit_transform(
            predicted, actual, sample_weight=weight
        )
        refined = (weight * (fitted - actual) ** 2).sum()
        result.update(
            {
                "pooled/spearman": ranked[5] / (ranked[3] * ranked[4]),
                "pooled/kendall": measures.concordance(actual, predicted),
                "pooled/kling gupta": 1
                - math.sqrt(
                    (related - 1) ** 2 + (width / spread - 1) ** 2 + ((second - first) / spread) ** 2
                ),
                "pooled/covariance share": 2 * (1 - related) * width * spread / square,
                "pooled/potential skill": related**2,
                "pooled/conditional bias": (related - width / spread) ** 2,
                "pooled/unconditional bias": ((second - first) / spread) ** 2,
                "pooled/mincer zarnowitz slope": slope,
                "pooled/mincer zarnowitz intercept": first - slope * second,
                "pooled/miscalibration": square - refined,
                "pooled/discrimination": spread**2 - refined,
                "pooled/uncertainty": spread**2,
            }
        )
    return result


def willmott(actual, predicted, weight, centre):
    missed = (weight * np.abs(predicted - actual)).sum()
    varied = 2 * (weight * np.abs(actual - centre)).sum()
    return 1 - missed / varied if missed <= varied else varied / missed - 1


def directions(actual, predicted):
    moved = (actual != 0) & (predicted != 0)
    if moved.sum() < 3:
        return math.nan
    up, called = (actual[moved] > 0).mean(), (predicted[moved] > 0).mean()
    size = moved.sum()
    hit = (np.sign(actual[moved]) == np.sign(predicted[moved])).mean()
    chance = up * called + (1 - up) * (1 - called)
    spread = chance * (1 - chance) / size
    other = (
        (2 * up - 1) ** 2 * called * (1 - called)
        + (2 * called - 1) ** 2 * up * (1 - up)
        + 4 * up * called * (1 - up) * (1 - called) / size
    ) / size
    return (hit - chance) / math.sqrt(spread - other) if spread > other else math.nan


def build(actual, values, weight):
    started = time.perf_counter()
    result = {}
    for row, predicted in enumerate(values):
        for name, value in one(actual, predicted, weight).items():
            result.setdefault(name, np.full(len(values), np.nan))[row] = value
    thresholds = np.quantile(np.r_[actual, values.reshape(-1)], np.linspace(0.005, 0.995, THRESHOLDS))
    share = weight / weight.sum()
    curves = [elementary(actual, predicted, share, thresholds) for predicted in values]
    result["pooled/murphy mean dominated"] = dominated([curve[0] for curve in curves])
    result["pooled/murphy median dominated"] = dominated([curve[1] for curve in curves])
    LOG.info(
        "pooled exit seconds=%.3f rows=%d points=%d", time.perf_counter() - started, len(values), len(actual)
    )
    return result
