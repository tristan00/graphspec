import logging
import time
from dataclasses import dataclass

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship
from graphspec.schema.targets import TARGETS, accumulated

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
    fields: dict


def scored(target, prediction, task, series, names, scale, fields):
    started = time.perf_counter()
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    task, series = np.asarray(task, np.int64), np.asarray(series, np.int64)
    fields = {name: np.asarray(values) for name, values in fields.items()}
    if target.ndim != 1 or not target.shape == prediction.shape == task.shape == series.shape:
        raise ValueError("every point scored needs a target, a prediction, a task and a series")
    if any(values.shape != target.shape for values in fields.values()):
        raise ValueError("every field of the points scored holds one value per point")
    if not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("the predictions scored must be nonempty and finite")
    result = Scored(target, prediction, task, series, tuple(names), np.asarray(scale, np.float64), fields)
    LOG.debug("scored exit points=%d seconds=%.4f", len(target), time.perf_counter() - started)
    return result


@timed("measures.needless")
def needless(held):
    return None


@timed("measures.determined")
def determined(target, prediction):
    return 1 - float(((prediction - target) ** 2).mean()) / float(target.var())


class Squared:
    reads = ("number", "path")
    better = "lower"
    label, short = "Validation MSE", "MSE"
    refusal = staticmethod(needless)

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

    @timed("measures.determination.refusal")
    def refusal(self, held):
        flat = [name for code, name in enumerate(held.names) if held.target[held.series == code].var() == 0]
        if flat:
            return "R² requires targets that vary within each series it reads, and those of the series %s do not" % (
                ", ".join(flat)
            )
        return None

    @timed("measures.determination")
    def __call__(self, held):
        return dict(
            value=determined(held.target, held.prediction),
            series={
                name: determined(held.target[held.series == code], held.prediction[held.series == code])
                for code, name in enumerate(held.names)
            },
        )


class Balanced:
    reads = ("number", "path")
    better = "lower"
    label, short = "Validation loss", "Loss"
    refusal = staticmethod(needless)

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
    refusal = staticmethod(needless)

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
    reads = ("class", "proportions", "link")
    better = "lower"
    label, short = "Validation log loss", "Log loss"
    refusal = staticmethod(needless)

    @timed("measures.entropy")
    def __call__(self, held):
        part, stated = np.zeros(len(held.target)), held.target > 0
        part[stated] = -held.target[stated] * np.log(held.prediction[stated])
        return pertask(held, part)


class Binary:
    reads = ("classes", "nodes")
    better = "lower"
    label, short = "Validation binary log loss", "Binary log loss"
    refusal = staticmethod(needless)

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
    refusal = staticmethod(needless)

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
    refusal = staticmethod(needless)

    @timed("measures.gain")
    def __call__(self, held):
        owner, _, discount, ideal = ranked(held)
        tasks = owner.max() + 1
        gained = np.bincount(owner, held.target * discount, tasks)
        best = np.lexsort((-held.target, owner))
        return tasked(held, owner, gained / np.bincount(owner[best], held.target[best] * ideal, tasks))


class Reciprocal:
    reads = ("ranking", "link")
    better = "higher"
    label, short = "Validation reciprocal rank", "MRR"
    refusal = staticmethod(needless)

    @timed("measures.reciprocal")
    def __call__(self, held):
        owner, best = first(held)
        return tasked(held, owner, 1 / best)


class Hits:
    reads = ("ranking", "link")
    better = "higher"
    refusal = staticmethod(needless)

    @timed("measures.hits.init")
    def __init__(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("hits are counted within a whole number of ranks of at least 1, found %r" % count)
        self.count = count
        self.label, self.short = "Validation hits within %d" % count, "Hits@%d" % count

    @timed("measures.hits")
    def __call__(self, held):
        owner, best = first(held)
        return tasked(held, owner, (best <= self.count).astype(np.float64))


@timed("measures.decided")
def decided(held):
    owner = np.unique(held.task, return_inverse=True)[1].reshape(-1)
    tasks, chosen, stated = owner.max() + 1, held.prediction >= 0.5, held.target == 1
    return owner, *(np.bincount(owner, values, tasks) for values in (chosen & stated, chosen, stated))


class Agreement:
    reads = ("nodes",)
    better = "higher"
    label, short = "Validation F1", "F1"
    refusal = staticmethod(needless)

    @timed("measures.agreement")
    def __call__(self, held):
        owner, both, chosen, stated = decided(held)
        total = chosen + stated
        return tasked(held, owner, np.where(total > 0, 2 * both / np.maximum(total, 1), 1.0))


class Jaccard:
    reads = ("nodes",)
    better = "higher"
    label, short = "Validation Jaccard index", "Jaccard"
    refusal = staticmethod(needless)

    @timed("measures.jaccard")
    def __call__(self, held):
        owner, both, chosen, stated = decided(held)
        union = chosen + stated - both
        return tasked(held, owner, np.where(union > 0, both / np.maximum(union, 1), 1.0))


def positioned(held):
    started = time.perf_counter()
    keys = np.stack((held.task, held.fields["place"].astype(np.int64)), 1)
    found, group = np.unique(keys, axis=0, return_inverse=True)
    group, groups, stated = group.reshape(-1), len(found), held.target == 1
    chance = np.maximum(1 - np.bincount(group, held.prediction, groups), 0.0)
    order = np.lexsort((held.series, -held.prediction, group))
    top = order[np.unique(group[order], return_index=True)[1]]
    best = np.where(held.prediction[top] > chance, held.series[top], -1)
    truth = np.full(groups, -1, np.int64)
    truth[group[stated]] = held.series[stated]
    chance[group[stated]] = held.prediction[stated]
    owner = np.unique(found[:, 0], return_inverse=True)[1].reshape(-1)
    LOG.debug("positioned exit places=%d seconds=%.4f", groups, time.perf_counter() - started)
    return found[:, 1], owner, truth, best, chance


@timed("measures.agreeing")
def agreeing(held, best, truth, chosen):
    return {name: float(((best == code) == (truth == code))[chosen].mean()) for code, name in enumerate(held.names)}


class Tokens:
    reads = ("sequence",)
    better = "lower"
    label, short = "Validation loss per token", "Token loss"
    refusal = staticmethod(needless)

    @timed("measures.tokens")
    def __call__(self, held):
        _, _, truth, _, chance = positioned(held)
        part = -np.log(chance)
        return dict(
            value=float(part.mean()),
            series={name: float(part[truth == code].sum() / len(part)) for code, name in enumerate(held.names)},
        )


class Placed:
    reads = ("sequence",)
    better = "higher"
    label, short = "Validation token accuracy", "Token accuracy"
    refusal = staticmethod(needless)

    @timed("measures.placed")
    def __call__(self, held):
        place, owner, truth, best, _ = positioned(held)
        counted = place <= np.bincount(owner, truth >= 0)[owner]
        return dict(value=float((best == truth)[counted].mean()), series=agreeing(held, best, truth, counted))


class Exact:
    reads = ("sequence",)
    better = "higher"
    label, short = "Validation exact sequences", "Exact"
    refusal = staticmethod(needless)

    @timed("measures.exact")
    def __call__(self, held):
        _, owner, truth, best, _ = positioned(held)
        stops = np.cumsum(best < 0)
        start = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
        before = np.repeat(stops[start] - (best[start] < 0), np.diff(np.r_[start, len(owner)]))
        decoded = np.where(stops > before, -1, best)
        return dict(
            value=float((np.bincount(owner, decoded != truth) == 0).mean()),
            series={
                name: float((np.bincount(owner, (decoded == code) != (truth == code)) == 0).mean())
                for code, name in enumerate(held.names)
            },
        )


def cased(held):
    started = time.perf_counter()
    owner = np.unique(held.task, return_inverse=True)[1].reshape(-1)
    tasks = owner.max() + 1
    cases = dict(horizon=np.zeros(tasks, held.fields["tau"].dtype))
    np.maximum.at(cases["horizon"], owner, held.fields["tau"])
    for name, values in (("duration", held.fields["duration"]), ("happened", held.fields["happened"] == 1)):
        cases[name] = np.zeros(tasks, values.dtype)
        cases[name][owner] = values
    cases["series"] = np.zeros(tasks, np.int64)
    cases["series"][owner] = held.series
    cases["earlier"] = cases["happened"] & (cases["duration"] <= cases["horizon"])
    LOG.debug("cased exit tasks=%d seconds=%.4f", tasks, time.perf_counter() - started)
    return owner, cases


def survived(held):
    started = time.perf_counter()
    owner, cases = cased(held)
    lasting = 1 - accumulated(held.prediction, held.task, held.fields["tau"])
    cases["risk"] = np.bincount(owner, held.prediction, len(cases["series"]))
    LOG.debug("survived exit tasks=%d seconds=%.4f", len(cases["series"]), time.perf_counter() - started)
    return owner, lasting, cases


def concordant(duration, risk, earlier):
    started = time.perf_counter()
    codes = (np.unique(risk, return_inverse=True)[1].reshape(-1) + 1).tolist()
    tree, order = [0] * (max(codes, default=0) + 1), np.argsort(-duration, kind="stable").tolist()
    agreed, pairs, inserted, low = 0.0, 0, 0, 0
    while low < len(order):
        high = low
        while high < len(order) and duration[order[high]] == duration[order[low]]:
            high += 1
        for at in order[low:high]:
            if earlier[at]:
                counts = []
                for place in (codes[at] - 1, codes[at]):
                    count = 0
                    while place > 0:
                        count += tree[place]
                        place -= place & -place
                    counts.append(count)
                agreed += counts[0] + 0.5 * (counts[1] - counts[0])
                pairs += inserted
        for at in order[low:high]:
            place = codes[at]
            while place < len(tree):
                tree[place] += 1
                place += place & -place
        inserted, low = inserted + high - low, high
    LOG.debug("concordant exit tasks=%d pairs=%d seconds=%.4f", len(order), pairs, time.perf_counter() - started)
    return agreed, pairs


@timed("measures.censoring")
def censoring(duration, happened):
    times = np.unique(duration)
    risk = len(duration) - np.searchsorted(np.sort(duration), times, "left")
    censored = np.bincount(np.searchsorted(times, duration[~happened]), minlength=len(times))
    return times, np.r_[1.0, np.cumprod(1 - censored / risk)]


class Concordance:
    reads = ("survival",)
    better = "higher"
    label, short = "Validation concordance", "Concordance"

    @timed("measures.concordance.refusal")
    def refusal(self, held):
        _, cases = cased(held)
        lacking = []
        for code, name in enumerate(held.names):
            chosen = cases["series"] == code
            duration = cases["duration"][chosen]
            if not (cases["earlier"][chosen] & (duration < duration.max())).any():
                lacking.append(name)
        if lacking:
            return (
                "concordance requires, in the series %s, a task whose event happened within its intervals before "
                "another task of the series ended" % ", ".join(lacking)
            )
        return None

    @timed("measures.concordance")
    def __call__(self, held):
        _, _, cases = survived(held)
        series, agreed, pairs = {}, 0.0, 0
        for code, name in enumerate(held.names):
            chosen = cases["series"] == code
            found, count = concordant(cases["duration"][chosen], cases["risk"][chosen], cases["earlier"][chosen])
            series[name], agreed, pairs = found / count, agreed + found, pairs + count
        return dict(value=agreed / pairs, series=series)


class Brier:
    reads = ("survival",)
    better = "lower"
    label, short = "Validation integrated Brier score", "Brier"
    refusal = staticmethod(needless)

    @timed("measures.brier")
    def __call__(self, held):
        owner, lasting, cases = survived(held)
        tau, part = held.fields["tau"], np.zeros(len(owner))
        duration, happened = cases["duration"][owner], cases["happened"][owner]
        for code in range(len(held.names)):
            chosen = cases["series"] == code
            times, kept = censoring(cases["duration"][chosen], cases["happened"][chosen])
            died = (held.series == code) & happened & (duration <= tau)
            lived = (held.series == code) & (duration > tau)
            part[died] = lasting[died] ** 2 / kept[np.searchsorted(times, duration[died], "left")]
            part[lived] = (1 - lasting[lived]) ** 2 / kept[np.searchsorted(times, tau[lived], "right")]
        return dict(
            value=float(part.mean()),
            series={name: float(part[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


def predicted(kinds, aims, task, values, fields):
    started = time.perf_counter()
    result = np.asarray(values, np.float64).copy()
    for code, kind in enumerate(kinds):
        chosen = aims == code
        if chosen.any():
            held = {name: np.asarray(found)[chosen] for name, found in fields.items()}
            result[chosen] = TARGETS.named(kind).predicted(result[chosen], task[chosen], held)
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
        {name: values[chosen] for name, values in held.fields.items()},
    )
    LOG.debug("restricted exit points=%d seconds=%.4f", int(chosen.sum()), time.perf_counter() - started)
    return result


@timed("measures.read")
def read(kinds, name):
    return tuple(series for series, kind in kinds.items() if kind in MEASURES.named(name).reads)


def checked(name, held, kinds, where):
    started = time.perf_counter()
    part = restricted(held, [held.names.index(series) for series in read(kinds, name)])
    lacking = [series for code, series in enumerate(part.names) if not (part.series == code).any()]
    if lacking:
        raise ValueError(
            "%s hold no point of the series %s, which the measure %s reads" % (where, ", ".join(lacking), name)
        )
    refused = MEASURES.named(name).refusal(part)
    if refused is not None:
        raise ValueError("%s cannot be measured by %s: %s" % (where, name, refused))
    LOG.debug("measure checked exit measure=%s seconds=%.4f", name, time.perf_counter() - started)
    return part


def measurable(names, bank, scored):
    started = time.perf_counter()
    strangers = [name for name in scored if name not in bank.split["indices"]]
    if strangers:
        raise ValueError(
            "the splits scored on name %s, and the bank holds the splits %s"
            % (", ".join(strangers), ", ".join(sorted(bank.split["indices"])))
        )
    rows = np.sort(np.concatenate([np.asarray(bank.split["indices"][name], np.int64) for name in scored]))
    held = Scored(
        np.asarray(bank.targets[rows], np.float64),
        np.full(len(rows), np.nan),
        bank.task_of[rows],
        bank.series[rows],
        tuple(bank.series_names),
        np.asarray(bank.scale, np.float64),
        {name: values[rows] for name, values in bank.fields.items()},
    )
    for name in names:
        checked(name, held, bank.registry.schema.measured(), "the splits scored on, %s," % ", ".join(scored))
    LOG.info("measurable exit measures=%d points=%d seconds=%.4f", len(names), len(rows), time.perf_counter() - started)


def measured(names, held, kinds):
    started = time.perf_counter()
    result = {name: MEASURES.named(name)(checked(name, held, kinds, "the points scored")) for name in names}
    LOG.debug("measured exit measures=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


MEASURES.register("mse", Squared())
MEASURES.register("r2", Determination())
MEASURES.register("balanced", Balanced())
MEASURES.register("accuracy", Accuracy())
MEASURES.register("entropy", Entropy())
MEASURES.register("binary", Binary())
MEASURES.register("divergence", Divergence())
MEASURES.register("tokens", Tokens())
MEASURES.register("token_accuracy", Placed())
MEASURES.register("exact", Exact())
MEASURES.register("concordance", Concordance())
MEASURES.register("brier", Brier())
MEASURES.register("gain", Gain())
MEASURES.register("reciprocal", Reciprocal())
MEASURES.register("hits1", Hits(1))
MEASURES.register("hits10", Hits(10))
MEASURES.register("f1", Agreement())
MEASURES.register("jaccard", Jaccard())
ship([MEASURES])
