import math

import numpy as np
from numba import njit

from . import pathwise

EDGES = (0.0, 0.05, 0.2, 0.4, 1.0)
MARKS = ((0.2, 0.025), (1.0, 0.066))
DEPTH = 4
JOINT_DEPTH = 3
REFINE = 2
GAMMA = 0.01
BLEND = 0.5
AMERCE = 0.1
STIFFNESS = 0.001
DELETE = 1.0
TOLERANCE = 0.25
MOVE = 0.5
POWER = 0.5
MOMENT = 50.0
ENTROPY = 0.1
WIDTH = 1.0
ROUNDS = 50
REACH = 2
NORM = 4
PATCH = 8
CROSSING = 0.1
WAVES = 16


def bins(span):
    return [
        (low * span, high * span, "%d-%d days" % (round(low * span), round(high * span)))
        for low, high in zip(EDGES[:-1], EDGES[1:])
    ]


def summaries(span):
    return ["terminal", "mean level"] + ["at %d days" % round(mark * span) for mark, _ in MARKS]


@njit(cache=True)
def concordance(a, b):
    same, opposite, only_a, only_b = 0, 0, 0, 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            product = (a[i] - a[j]) * (b[i] - b[j])
            if product > 0:
                same += 1
            elif product < 0:
                opposite += 1
            elif a[i] != a[j]:
                only_a += 1
            elif b[i] != b[j]:
                only_b += 1
    scale = math.sqrt((same + opposite + only_a) * (same + opposite + only_b))
    return (same - opposite) / scale if scale > 0 else np.nan


def ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    result[order] = np.arange(len(values))
    ordered = values[order]
    start = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    stop = np.r_[start[1:], len(values)]
    mean = (start + stop - 1) / 2
    result[order] = np.repeat(mean, stop - start)
    return result


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    scale = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / scale) if scale > 0 else math.nan


def turning(values):
    before, after = values[1:-1] - values[:-2], values[2:] - values[1:-1]
    return np.where(before * after < 0, np.sign(before), 0)


def turns(a, b):
    actual, predicted = turning(a), turning(b)
    if not actual.any() and not predicted.any():
        return math.nan
    hit = 0
    for index in np.flatnonzero(actual):
        near = predicted[max(0, index - 1) : index + 2]
        hit += bool((near == actual[index]).any())
    found = 0
    for index in np.flatnonzero(predicted):
        near = actual[max(0, index - 1) : index + 2]
        found += bool((near == predicted[index]).any())
    recall = hit / max(1, np.count_nonzero(actual))
    precision = found / max(1, np.count_nonzero(predicted))
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def between(u, error):
    first, second, width = np.abs(error[:-1]), np.abs(error[1:]), np.diff(u)
    crossing = error[:-1] * error[1:] < 0
    total = np.where(
        crossing, (first**2 + second**2) / (2 * np.where(crossing, first + second, 1)), (first + second) / 2
    )
    return float((total * width).sum() / (u[-1] - u[0]))


def area(u, values):
    return float(((values[:-1] + values[1:]) / 2 * np.diff(u)).sum() / (u[-1] - u[0]))


def functionals(tau, values, span):
    result = [values[-1], area(np.r_[0.0, tau], np.r_[0.0, values])]
    for mark, slack in MARKS:
        nearest = np.abs(tau - mark * span).argmin()
        result.append(values[nearest] if abs(tau[nearest] - mark * span) <= slack * span else math.nan)
    return result


def invariant(a, b):
    error = a - b
    size = len(error)
    lifted = np.exp(error - error.max())
    shift = size * np.abs(1 / size - lifted / lifted.sum()).sum()
    first, second = np.fft.rfft(a, norm="ortho"), np.fft.rfft(b, norm="ortho")
    strong = np.zeros(len(first), bool)
    strong[np.argsort(-np.abs(first))[: math.ceil(math.sqrt(size))]] = True
    phase = math.sqrt((np.abs(first - second)[strong] ** 2).sum()) + math.sqrt(
        (np.abs(second)[~strong] ** 2).sum()
    )
    own = np.correlate(a, a, "full")
    cross = np.correlate(a, b, "full")
    scale = math.sqrt((a * a).sum() * (b * b).sum())
    amplification = math.sqrt(((own / (a * a).sum() - cross / scale) ** 2).sum()) if scale > 0 else math.nan
    return shift, phase, amplification


def density(tau):
    size = len(tau)
    if size == 1:
        return np.ones(1)
    width = 1.06 * tau.std() * size ** (-0.2)
    gaps = (tau[:, None] - tau[None, :]) / width
    near = np.exp(-0.5 * gaps**2)
    np.fill_diagonal(near, 0.0)
    weight = 1 / near.sum(1)
    return weight / weight.sum()


def patches(a, b):
    size = len(a)
    strongest = 1 + int(np.abs(np.fft.rfft(a))[1 : size // 2 + 1].argmax())
    length = max(2, min(math.ceil(math.ceil(size / strongest) / 2), PATCH))
    stride = max(1, length // 2)
    together, spread, level = [], [], []
    for start in range(0, size - length + 1, stride):
        first, second = a[start : start + length], b[start : start + length]
        together.append(1 - correlation(first, second))
        own, other = np.exp(first - first.max()), np.exp(second - second.max())
        own, other = own / own.sum(), other / other.sum()
        spread.append((own * np.log(own / other)).sum())
        level.append(abs(first.mean() - second.mean()))
    known = np.isfinite(together)
    return (
        float(np.mean(np.asarray(together)[known])) if known.any() else math.nan,
        float(np.mean(spread)),
        float(np.mean(level)),
    )


def adjusted(a, b):
    from scipy.optimize import linear_sum_assignment

    place = np.arange(len(a))
    cost = np.abs(a[:, None] - b[None, :]) ** NORM
    cost[np.abs(place[:, None] - place[None, :]) > REACH] = 1e12
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].mean() ** (1 / NORM))


def passage(u, values, level):
    beyond = values >= level if level > 0 else values <= level
    if not beyond.any():
        return math.nan
    index = int(beyond.argmax())
    share = (level - values[index - 1]) / (values[index] - values[index - 1])
    return u[index - 1] + share * (u[index] - u[index - 1])


def crossings(u, a, b):
    same, apart = [], []
    for level in (CROSSING, -CROSSING):
        first, second = passage(u, a, level), passage(u, b, level)
        same.append(math.isnan(first) == math.isnan(second))
        if not math.isnan(first) and not math.isnan(second):
            apart.append(abs(first - second))
    return float(np.mean(same)), float(np.mean(apart)) if apart else math.nan


def periodogram(tau, values, span):
    from scipy.signal import lombscargle

    waves = 2 * math.pi * np.arange(1, WAVES + 1) / span
    return lombscargle(tau, values - values.mean(), waves)


class Truth:
    def __init__(self, tau, values, span):
        self.tau = tau
        self.u = np.r_[0.0, tau / span]
        self.values = np.r_[0.0, values]
        self.size = len(tau)
        self.plane = np.ascontiguousarray(np.stack((self.u, self.values), 1))
        self.signature = pathwise.signature(self.plane, DEPTH)
        self.kernel = pathwise.signature_kernel(self.plane, self.plane, REFINE)
        self.soft = pathwise.soft_warp(self.values, self.values, GAMMA)
        self.slope = np.gradient(self.values, self.u)
        self.even = pathwise.resample(self.u, self.values, pathwise.GRID + 1)
        self.steps = np.diff(self.values)
        self.functionals = functionals(tau, values, span)
        self.bins = bins(span)
        self.varying = self.size >= 3 and np.ptp(values) > 0
        self.ranks = ranks(values) if self.varying else None
        self.density = density(tau)
        self.staircase = pathwise.signature(pathwise.staircase(self.values), DEPTH)
        self.span = span
        self.heard = periodogram(tau, values, span) if self.size >= 8 else None


def compare(truth, predicted, ratio):
    u, a, b = truth.u, truth.values, np.r_[0.0, predicted]
    error = predicted - a[1:]
    size = truth.size
    absolute = np.abs(error)
    result = {
        "point/mae": absolute.mean(),
        "point/mse": (error**2).mean(),
        "point/path rmse": math.sqrt((error**2).mean()),
        "point/mdae": float(np.median(absolute)),
        "point/max": absolute.max(),
        "point/huber": np.where(absolute <= 1, 0.5 * error**2, absolute - 0.5).mean(),
        "point/logcosh": (absolute + np.log1p(np.exp(-2 * absolute)) - math.log(2)).mean(),
        "point/continuous squared": (truth.density * error**2).sum(),
        "point/continuous absolute": (truth.density * absolute).sum(),
        "bias/mean error": error.mean(),
        "stability/horizontal": np.abs(np.diff(b)).mean(),
    }
    total = np.abs(a[1:]) + np.abs(predicted)
    result["point/smape"] = (
        (2 * absolute[total > 0] / total[total > 0]).mean() if (total > 0).any() else math.nan
    )
    moved = a[1:] != 0
    result["point/maape"] = (
        np.arctan(absolute[moved] / np.abs(a[1:][moved])).mean() if moved.any() else math.nan
    )
    if ratio is not None:
        result["point/level ape"] = np.abs(np.expm1(error * ratio)).mean()
    called = moved & (predicted != 0)
    result["direction/sign"] = (
        (np.sign(predicted[called]) == np.sign(a[1:][called])).mean() if called.any() else math.nan
    )
    steps = np.diff(b)
    stepped = (truth.steps != 0) & (steps != 0)
    result["direction/step sign"] = (
        (np.sign(steps[stepped]) == np.sign(truth.steps[stepped])).mean() if stepped.any() else math.nan
    )
    result["dynamics/step mae"] = np.abs(steps - truth.steps).mean()
    result["dynamics/rate mae"] = (np.abs(steps - truth.steps) / np.diff(u)).mean()
    result["dynamics/terminal"] = absolute[-1]
    result["dynamics/area signed"] = abs(area(u, b - a))
    result["dynamics/area between"] = between(u, b - a)
    result["dynamics/peak time"] = abs(u[b.argmax()] - u[a.argmax()])
    result["dynamics/trough time"] = abs(u[b.argmin()] - u[a.argmin()])
    result["dynamics/peak value"] = abs(b.max() - a.max())
    result["dynamics/trough value"] = abs(b.min() - a.min())
    result["dynamics/range"] = abs(np.ptp(b) - np.ptp(a))
    varied = np.abs(truth.steps).sum()
    result["dynamics/variation ratio"] = np.abs(steps).sum() / varied if varied > 0 else math.nan
    result["dynamics/turning f1"] = turns(a, b)
    if truth.varying and np.ptp(predicted) > 0:
        result["path/pearson"] = correlation(a[1:], predicted)
        result["path/spearman"] = correlation(truth.ranks, ranks(predicted))
        result["path/kendall"] = concordance(a[1:], predicted)
    cost, cells, distortion = pathwise.warp(a, b, u, 0.0)
    result["elastic/dtw"] = cost / cells
    result["elastic/tdi"] = distortion / cells
    result["elastic/adtw"] = pathwise.warp(a, b, u, AMERCE)[0] / (size + 1)
    result["elastic/ddtw"] = pathwise.warp(truth.slope, np.gradient(b, u), u, 0.0)[0] / cells
    soft = pathwise.soft_warp(a, b, GAMMA)
    result["elastic/soft dtw divergence"] = (
        soft - 0.5 * truth.soft - 0.5 * pathwise.soft_warp(b, b, GAMMA)
    ) / (size + 1)
    bent = pathwise.soft_distortion(a, b, u, GAMMA)
    result["elastic/soft tdi"] = bent / (size + 1)
    result["elastic/dilate"] = (BLEND * soft + (1 - BLEND) * bent) / (size + 1)
    result["elastic/twed"] = pathwise.twed(a, b, u, STIFFNESS, DELETE) / (size + 1)
    result["elastic/erp"] = pathwise.erp(a, b, 0.0) / (size + 1)
    result["elastic/edr"], result["elastic/lcss"] = pathwise.edits(a, b, TOLERANCE)
    result["elastic/msm"] = pathwise.msm(a, b, MOVE) / (size + 1)
    result["geometry/frechet"], result["geometry/hausdorff"] = pathwise.curves(a, b, u)
    plane = np.ascontiguousarray(np.stack((u, b), 1))
    result["signature/truncated"] = math.sqrt(
        ((pathwise.signature(plane, DEPTH) - truth.signature) ** 2).sum()
    )
    result["signature/kernel"] = math.sqrt(
        max(
            0.0,
            truth.kernel
            - 2 * pathwise.signature_kernel(truth.plane, plane, REFINE)
            + pathwise.signature_kernel(plane, plane, REFINE),
        )
    )
    result["signature/lead lag"] = math.sqrt(
        ((pathwise.signature(pathwise.staircase(b), DEPTH) - truth.staircase) ** 2).sum()
    )
    result["dependence/variogram"] = pathwise.variogram(a, b, u, POWER)
    result["transport/order preserving"] = pathwise.preserving(a, b, u, MOMENT, ENTROPY, WIDTH, ROUNDS)
    result["point/adjusted"] = adjusted(a[1:], predicted)
    result["dynamics/crossing agreement"], result["dynamics/crossing time"] = crossings(u, a, b)
    if truth.heard is not None:
        heard = periodogram(truth.tau, predicted, truth.span)
        audible = (truth.heard > 0) & (heard > 0)
        if audible.any():
            result["spectral/lomb scargle"] = math.sqrt(
                ((10 * np.log10(truth.heard[audible] / heard[audible])) ** 2).mean()
            )
    even = pathwise.resample(u, b, pathwise.GRID + 1)
    result["shape/sbd"] = pathwise.shape_distance(truth.even, even)
    rough, smooth = math.sqrt((np.diff(truth.even) ** 2).sum()), math.sqrt((np.diff(even) ** 2).sum())
    straight = math.sqrt(((truth.even - even) ** 2).sum())
    result["shape/cid"] = (
        straight * max(rough, smooth) / min(rough, smooth) if min(rough, smooth) > 0 else math.nan
    )
    result["shape/srvf amplitude"], result["shape/srvf phase"] = pathwise.elastic(truth.even, even)
    (
        result["shape/tildeq amplitude shift"],
        result["shape/tildeq phase"],
        result["shape/tildeq amplification"],
    ) = invariant(truth.even, even)
    result["transport/warping"] = pathwise.transport(truth.even, even, 1.0) / len(even) ** 2
    result["shape/ps correlation"], result["shape/ps variance"], result["shape/ps mean"] = patches(
        truth.even, even
    )
    first, second = np.fft.rfft(truth.even), np.fft.rfft(even)
    result["spectral/fredf"] = np.abs(first - second).mean()
    power = (np.abs(first[1:]) ** 2, np.abs(second[1:]) ** 2)
    heard = (power[0] > 0) & (power[1] > 0)
    result["spectral/log spectral"] = (
        math.sqrt(((10 * np.log10(power[0][heard] / power[1][heard])) ** 2).mean())
        if heard.any()
        else math.nan
    )
    for low, high, name in truth.bins:
        inside = (truth.tau > low) & (truth.tau <= high)
        if inside.any():
            result["horizon/mae " + name] = absolute[inside].mean()
            result["horizon/mean error " + name] = error[inside].mean()
            known = inside & called
            if known.any():
                result["horizon/sign " + name] = (np.sign(predicted[known]) == np.sign(a[1:][known])).mean()
    return {name: float(value) for name, value in result.items()}, functionals(
        truth.tau, predicted, truth.span
    )


def joint(u, actual, predicted, whitener):
    error = predicted - actual
    lifted = np.zeros((len(u), actual.shape[1] + 1))
    lifted[:, 0] = u
    lifted[1:, 1:] = actual
    other = lifted.copy()
    other[1:, 1:] = predicted
    flat_a, flat_b = actual.reshape(-1), predicted.reshape(-1)
    place = np.repeat(u[1:], actual.shape[1]) + np.tile(np.arange(actual.shape[1]), len(actual))
    return {
        "joint/energy": math.sqrt((error**2).mean()),
        "joint/mahalanobis": math.sqrt((((error @ whitener) ** 2).sum(1)).mean()),
        "joint/signature": math.sqrt(
            ((pathwise.signature(lifted, JOINT_DEPTH) - pathwise.signature(other, JOINT_DEPTH)) ** 2).sum()
        ),
        "joint/signature kernel": math.sqrt(
            max(
                0.0,
                pathwise.signature_kernel(lifted, lifted, REFINE)
                - 2 * pathwise.signature_kernel(lifted, other, REFINE)
                + pathwise.signature_kernel(other, other, REFINE),
            )
        ),
        "joint/variogram": pathwise.variogram(flat_a, flat_b, place, POWER),
    }
