import argparse
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from graphspec.infra.files import write_json
from graphspec.infra.log import configure
from graphspec.infra.runtime import output_path

from . import catalog, measures, pathwise, pooled, references
from .sequences import Sequences, collect

LOG = logging.getLogger(__name__)
STATE = {}
LEVEL = 0.95
JOINT = "joint"
BLEND = "all series"


def rows(sequences, span):
    started = time.perf_counter()
    found, chosen = references.build(sequences, span)
    members = sequences.prediction
    names = [
        "T%s %s" % (model["name"].split("_")[1].lstrip("0") or "0", model["view"])
        for model in sequences.about["models"]
    ]
    kinds = ["model"] * len(names)
    values = [members[index] for index in range(len(names))]
    names += ["ensemble: mean of the models", "ensemble: median of the models"]
    kinds += ["ensemble"] * 2
    values += [members.mean(0), np.median(members, 0)]
    for name, value in found.items():
        names.append(name)
        kinds.append("reference")
        values.append(value)
    LOG.info("rows exit seconds=%.3f rows=%d", time.perf_counter() - started, len(names))
    return names, kinds, np.asarray(values), chosen


def whitener(sequences):
    started = time.perf_counter()
    train, count = sequences.train, len(sequences.series)
    key = np.stack((train["graph"], train["step"], np.round(train["tau"] * 4).astype(np.int64)), 1)
    moment, place = np.unique(key, axis=0, return_inverse=True)
    table = np.full((len(moment), count), np.nan)
    table[place.reshape(-1), train["series"]] = train["target"] / sequences.scale[train["series"]]
    table = table[np.isfinite(table).all(1)]
    result = np.linalg.cholesky(np.linalg.inv(np.cov(table.T)))
    LOG.info("whitener exit seconds=%.3f moments=%d", time.perf_counter() - started, len(table))
    return result


def prepare(directory, values, members, span, ratio, white):
    sequences = Sequences(directory)
    held = sequences.held
    scale = sequences.scale[held["series"]]
    STATE.update(
        held=held,
        target=held["target"] / scale,
        values=values / scale,
        members=members,
        span=span,
        scale=sequences.scale,
        ratio=ratio,
        white=white,
        count=len(sequences.series),
    )


def crowd(truth, members):
    size = len(members)
    actual = truth.values[1:]
    near = np.abs(members - actual).mean(0)
    apart = np.abs(members[:, None] - members[None]).sum((0, 1))
    away = np.sqrt(((members - actual) ** 2).mean(1))
    between = np.sqrt(((members[:, None] - members[None]) ** 2).mean(2))
    planes = [np.ascontiguousarray(np.stack((truth.u, np.r_[0.0, member]), 1)) for member in members]
    shared = sum(
        pathwise.signature_kernel(planes[i], planes[j], 0) for i in range(size) for j in range(i + 1, size)
    )
    met = sum(pathwise.signature_kernel(plane, truth.plane, 0) for plane in planes)
    varied, weight = 0.0, 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            share = 1.0 / (truth.u[j + 1] - truth.u[i + 1])
            expected = (np.abs(members[:, i] - members[:, j]) ** measures.POWER).mean()
            varied += share * (abs(actual[i] - actual[j]) ** measures.POWER - expected) ** 2
            weight += share
    below = (members < actual).sum(0)
    return {
        "ensemble/crps": float((near - apart / (2 * size * size)).mean()),
        "ensemble/fair crps": float((near - apart / (2 * size * (size - 1))).mean()),
        "ensemble/energy": float(away.mean() - between.sum() / (2 * size * size)),
        "ensemble/variogram": varied / weight if weight > 0 else math.nan,
        "ensemble/signature kernel": float(2 * shared / (size * (size - 1)) - 2 * met / size),
        "ensemble/spread": float(members.var(0, ddof=1).mean()),
        "ensemble/mean squared error": float(((members.mean(0) - actual) ** 2).mean()),
        "ensemble/envelope coverage": float(((below > 0) & (below < size)).mean()),
    }, np.bincount(below, minlength=size + 1) / len(actual)


def score(decisions):
    started = time.perf_counter()
    held, values, target = STATE["held"], STATE["values"], STATE["target"]
    wanted = np.flatnonzero(np.isin(held["decision"], decisions))
    single, summary, actual, together, flock, histogram = {}, {}, {}, {}, {}, {}
    for path in wanted:
        points = slice(held["start"][path], held["start"][path + 1])
        series = held["path_series"][path]
        truth = measures.Truth(held["tau"][points], target[points], STATE["span"])
        ratio = STATE["scale"][series] if series in STATE["ratio"] else None
        actual[int(path)] = truth.functionals
        for row in range(len(values)):
            single[int(path), row], summary[int(path), row] = measures.compare(
                truth, values[row, points], ratio
            )
        flock[int(path)], histogram[int(path)] = crowd(truth, values[STATE["members"]][:, points])
    for decision in decisions:
        paths = wanted[held["decision"][wanted] == decision]
        if len(paths) != STATE["count"]:
            continue
        spans = [slice(held["start"][path], held["start"][path + 1]) for path in paths]
        moments = [held["tau"][points] for points in spans]
        common = moments[0]
        for other in moments[1:]:
            common = np.intersect1d(common, other)
        if not len(common):
            continue
        chosen = [points.start + np.searchsorted(moment, common) for points, moment in zip(spans, moments)]
        u = np.r_[0.0, common / STATE["span"]]
        truth = np.stack([target[index] for index in chosen], 1)
        for row in range(len(values)):
            predicted = np.stack([values[row, index] for index in chosen], 1)
            together[int(decision), row] = measures.joint(u, truth, predicted, STATE["white"])
    LOG.info(
        "score exit seconds=%.3f decisions=%d paths=%d",
        time.perf_counter() - started,
        len(decisions),
        len(wanted),
    )
    return single, summary, actual, together, flock, histogram


def gather(directory, values, members, span, ratio, white, workers):
    started = time.perf_counter()
    sequences = Sequences(directory)
    held = sequences.held
    decisions = int(held["decision"].max()) + 1
    size = np.bincount(held["decision"], weights=np.diff(held["start"]), minlength=decisions)
    order = np.argsort(-size)
    parts = [order[index :: workers * 4].tolist() for index in range(workers * 4)]
    parts = [part for part in parts if part]
    settings = (directory, values, members, span, ratio, white)
    if workers == 1:
        prepare(*settings)
        results = [score(part) for part in parts]
    else:
        with ProcessPoolExecutor(workers, initializer=prepare, initargs=settings) as pool:
            results = list(pool.map(score, parts))
    paths = len(held["start"]) - 1
    single, together, flock = {}, {}, {}
    summary = np.full((len(values), paths, len(measures.summaries(span))), np.nan)
    actual = np.full((paths, summary.shape[2]), np.nan)
    histogram = np.zeros((paths, len(members) + 1))
    for one, short, truth, joint, crowd_scores, ranks in results:
        for (path, row), entry in one.items():
            for name, value in entry.items():
                single.setdefault(name, np.full((len(values), paths), np.nan))[row, path] = value
        for (path, row), entry in short.items():
            summary[row, path] = entry
        for path, entry in truth.items():
            actual[path] = entry
        for (decision, row), entry in joint.items():
            for name, value in entry.items():
                together.setdefault(name, np.full((len(values), decisions), np.nan))[row, decision] = value
        for path, entry in crowd_scores.items():
            for name, value in entry.items():
                flock.setdefault(name, np.full(paths, np.nan))[path] = value
        for path, entry in ranks.items():
            histogram[path] = entry
    LOG.info(
        "gather exit seconds=%.3f measures=%d", time.perf_counter() - started, len(single) + len(together)
    )
    return sequences, single, summary, actual, together, flock, histogram


def scaled(sequences, single, span):
    started = time.perf_counter()
    train, held = sequences.train, sequences.held
    unit = dict(train, target=train["target"] / sequences.scale[train["series"]])
    sizes = np.diff(held["start"])
    owner = np.repeat(np.arange(len(sizes)), sizes)
    linear = references.by_horizon(unit, held, span, lambda values: np.abs(values).mean())
    square = references.by_horizon(unit, held, span, lambda values: (values**2).mean())
    single["scaled/mase"] = single["point/mae"] / (np.bincount(owner, linear) / sizes)
    single["scaled/rmsse"] = np.sqrt(single["point/mse"] / (np.bincount(owner, square) / sizes))
    LOG.info("scaled exit seconds=%.3f", time.perf_counter() - started)


def revisions(sequences, values):
    started = time.perf_counter()
    held = sequences.held
    scale = sequences.scale[held["series"]]
    target, values = held["target"] / scale, values / scale
    order, date, _ = references.histories(held)
    pairs, owners, kinds, before, after = [], [], [], [], []
    last = {}
    for first, second in zip(order[:-1], order[1:]):
        same = (held["path_graph"][first], held["path_series"][first]) == (
            held["path_graph"][second],
            held["path_series"][second],
        )
        gap = date[second] - date[first]
        if not same or gap <= 0:
            continue
        old = slice(held["start"][first], held["start"][first + 1])
        new = slice(held["start"][second], held["start"][second + 1])
        moved = target[old][np.abs(held["tau"][old] - gap).argmin()]
        shared, at_old, at_new = np.intersect1d(
            np.round((held["tau"][old] - gap) * 4), np.round(held["tau"][new] * 4), return_indices=True
        )
        if not len(shared):
            continue
        change = moved + values[:, new][:, at_new] - values[:, old][:, at_old]
        pairs.append(np.abs(change).mean(1))
        owners.append(held["path_graph"][second])
        kinds.append(held["path_series"][second])
        key = (held["path_graph"][second], held["path_series"][second])
        absolute = np.round((date[second] + held["tau"][new][at_new]) * 4).astype(np.int64)
        if key in last and last[key][0] == first:
            known, earlier = last[key][1], last[key][2]
            both, here, there = np.intersect1d(absolute, known, return_indices=True)
            before.append((held["path_series"][second], earlier[:, there]))
            after.append(change[:, here])
        last[key] = (second, absolute, change)
    result = dict(
        values=np.stack(pairs, 1),
        groups=np.asarray(owners),
        series=np.asarray(kinds),
        memory={},
    )
    for series in range(len(sequences.series)):
        old = np.concatenate([entry[1] for entry, _ in zip(before, after) if entry[0] == series], 1)
        new = np.concatenate([change for entry, change in zip(before, after) if entry[0] == series], 1)
        result["memory"][series] = [measures.correlation(old[row], new[row]) for row in range(len(values))]
    LOG.info("revisions exit seconds=%.3f pairs=%d", time.perf_counter() - started, len(owners))
    return result


def oriented(values, better):
    if better == "higher":
        return -values
    if better == "zero":
        return np.abs(values)
    if better == "one":
        return np.abs(np.log(values))
    if better == "unit":
        return np.abs(values - 1)
    return values


def confidence(loss, sums, counts, draws, models):
    observed = loss[models]
    alive = list(range(len(models)))
    resampled = sums[models][:, draws].sum(2) / counts[models][:, draws].sum(2)
    result, reached = np.zeros(len(models)), 0.0
    while len(alive) > 1:
        gap = observed[alive] - observed[alive].mean()
        moved = resampled[alive] - resampled[alive].mean(0)
        spread = np.sqrt(((moved - gap[:, None]) ** 2).mean(1))
        if not (spread > 0).all():
            break
        statistic = (gap / spread).max()
        ceiling = ((moved - gap[:, None]) / spread[:, None]).max(0)
        reached = max(reached, float((ceiling >= statistic).mean()))
        worst = alive[int((gap / spread).argmax())]
        result[worst] = reached
        alive.remove(worst)
    for index in alive:
        result[index] = 1.0
    return result


def summarise(values, groups, better, draws, models):
    known = np.isfinite(values)
    names = np.unique(groups)
    member = groups[None, :] == names[:, None]
    sums = np.stack([np.where(known & chosen, values, 0).sum(1) for chosen in member], 1)
    counts = np.stack([(known & chosen).sum(1) for chosen in member], 1).astype(np.float64)
    total = counts.sum(1)
    mean = np.divide(sums.sum(1), total, out=np.full(len(values), np.nan), where=total > 0)
    present = counts > 0
    within = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=present)
    picks = draws[:, : len(names)] % len(names)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = sums[:, picks].sum(2) / counts[:, picks].sum(2)
    low, high = np.nanquantile(resampled, [(1 - LEVEL) / 2, (1 + LEVEL) / 2], axis=1)
    loss = oriented(mean, better)
    result = dict(
        decisions=mean,
        runs=np.nanmean(within, 1),
        median=np.nanmedian(np.where(known, values, np.nan), 1),
        low=low,
        high=high,
        count=total,
    )
    usable = np.isfinite(loss[models]).all() and better in ("lower", "higher")
    if usable:
        turned = oriented(resampled, better)
        best = turned[models].argmin(0)
        result["best"] = np.bincount(best, minlength=len(models)) / turned.shape[1]
        sign = -1.0 if better == "higher" else 1.0
        result["confidence"] = confidence(loss, sign * sums, counts, picks, models)
        full = present[models].all(0)
        if full.sum() >= 3:
            table = oriented(within[models][:, full], better)
            placed = stats.rankdata(table, axis=0)
            result["rank"] = placed.mean(1)
            result["friedman"] = float(stats.friedmanchisquare(*table).pvalue)
            wins = (table[:, None] < table[None]).sum(2) + 0.5 * (table[:, None] == table[None]).sum(2)
            result["wins"] = (wins.sum(1) - 0.5 * full.sum()) / (full.sum() * (len(models) - 1))
    return result


def relative(single, groups, series, kinds, names):
    started = time.perf_counter()
    result = {}
    anchors = [index for index, kind in enumerate(kinds) if kind == "reference"]
    for code in np.unique(series):
        chosen = series == code
        linear, square = single["point/mae"][:, chosen], single["point/mse"][:, chosen]
        runs = groups[chosen]
        for anchor in anchors:
            ratio = linear / linear[anchor]
            usable = np.isfinite(ratio) & (ratio > 0)
            logged = np.where(usable, np.log(np.where(usable, ratio, 1)), np.nan)
            by_run = np.stack(
                [
                    linear[:, runs == run].mean(1) / linear[anchor, runs == run].mean()
                    for run in np.unique(runs)
                ],
                1,
            )
            weight = np.asarray([(runs == run).sum() for run in np.unique(runs)], np.float64)
            entry = {
                "relative/mae ratio": linear.mean(1) / linear[anchor].mean(),
                "relative/geometric mean ratio": np.exp(np.nanmean(logged, 1)),
                "relative/median ratio": np.nanmedian(np.where(usable, ratio, np.nan), 1),
                "relative/share better": (linear < linear[anchor]).mean(1),
                "relative/squared error skill": 1 - square.mean(1) / square[anchor].mean(),
                "relative/average relative mae": np.exp((weight * np.log(by_run)).sum(1) / weight.sum()),
                "relative/clipped skill": 1 - np.exp(np.log(np.clip(by_run, 0.01, 100)).mean(1)),
            }
            for name, values in entry.items():
                result.setdefault(name, {}).setdefault(names[anchor], {})[int(code)] = values
    LOG.info("relative exit seconds=%.3f", time.perf_counter() - started)
    return result


def ordering(summary, actual, held, weights, span):
    started = time.perf_counter()
    count = len(weights)
    decisions = int(held["decision"].max()) + 1
    truth = np.full((count + 1, decisions, actual.shape[1]), np.nan)
    guess = np.full((summary.shape[0], count + 1, decisions, actual.shape[1]), np.nan)
    truth[held["path_series"], held["decision"]] = actual
    guess[:, held["path_series"], held["decision"]] = summary
    share = np.asarray(weights) / np.sum(weights)
    truth[count] = np.tensordot(share, truth[:count], 1)
    guess[:, count] = np.tensordot(share, guess[:, :count], (0, 1))
    run = np.zeros(decisions, np.int64)
    run[held["decision"]] = held["path_graph"]
    result = {}
    for series in range(count + 1):
        for index, name in enumerate(measures.summaries(span)):
            known = np.isfinite(truth[series, :, index])
            for row in range(summary.shape[0]):
                usable = known & np.isfinite(guess[row, series, :, index])
                entry = pooled.ranking(
                    truth[series, usable, index], guess[row, series, usable, index], run[usable]
                )
                for key, value in entry.items():
                    result.setdefault("ranking/%s, %s" % (key, name), {}).setdefault(
                        series, np.full(summary.shape[0], np.nan)
                    )[row] = value
    LOG.info("ordering exit seconds=%.3f measures=%d", time.perf_counter() - started, len(result))
    return result


def calibration(flock, histogram, held, series):
    chosen = held["path_series"] == series
    shares = histogram[chosen].mean(0)
    spread = flock["ensemble/spread"][chosen].mean()
    size = histogram.shape[1] - 1
    result = {name: float(np.nanmean(values[chosen])) for name, values in flock.items()}
    result["ensemble/spread skill ratio"] = float(
        math.sqrt((size + 1) / size * spread) / math.sqrt(result["ensemble/mean squared error"])
    )
    result["ensemble/rank histogram"] = shares.tolist()
    result["ensemble/reliability index"] = float(np.abs(shares - 1 / (size + 1)).sum())
    return result


def agreement(table):
    names = list(table)
    matrix = np.full((len(names), len(names)), np.nan)
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            both = np.isfinite(table[first]) & np.isfinite(table[second])
            if both.sum() >= 3:
                matrix[i, j] = measures.concordance(table[first][both], table[second][both])
    return names, matrix


def build(directory, workers, draws, ratio):
    started = time.perf_counter()
    sequences = Sequences(directory)
    span = float(sequences.train["tau"].max())
    names, kinds, values, chosen = rows(sequences, span)
    models = [index for index, kind in enumerate(kinds) if kind == "model"]
    ratio = [sequences.series.index(name) for name in ratio]
    sequences, single, summary, actual, together, flock, histogram = gather(
        directory, values, models, span, ratio, whitener(sequences), workers
    )
    held = sequences.held
    scaled(sequences, single, span)
    picks = np.random.default_rng(0).integers(0, 1 << 30, (draws, len(np.unique(held["path_graph"]))))
    weights = list((sequences.about["weights"] or dict.fromkeys(sequences.series, 1.0)).values())
    count = len(sequences.series)
    table = {}
    for name, matrix in single.items():
        better = catalog.entry(name)["better"]
        for series in range(count):
            member = held["path_series"] == series
            table.setdefault(name, {})[sequences.series[series]] = summarise(
                matrix[:, member], held["path_graph"][member], better, picks, models
            )
    run = np.zeros(together["joint/energy"].shape[1], np.int64)
    run[held["decision"]] = held["path_graph"]
    for name, matrix in together.items():
        table[name] = {JOINT: summarise(matrix, run, catalog.entry(name)["better"], picks, models)}
    revised = revisions(sequences, values)
    for series in range(count):
        member = revised["series"] == series
        label = sequences.series[series]
        table.setdefault("stability/revision", {})[label] = summarise(
            revised["values"][:, member], revised["groups"][member], "lower", picks, models
        )
        table.setdefault("stability/revision memory", {})[label] = dict(
            decisions=np.asarray(revised["memory"][series])
        )
    scale = sequences.scale[held["series"]]
    weight = np.repeat(1.0 / np.diff(held["start"]), np.diff(held["start"]))
    for series in range(count):
        member = held["series"] == series
        found = pooled.build(
            held["target"][member] / scale[member], values[:, member] / scale[member], weight[member]
        )
        for name, column in found.items():
            table.setdefault(name, {})[sequences.series[series]] = dict(decisions=column)
    for name, entry in ordering(summary, actual, held, weights, span).items():
        for series, column in entry.items():
            label = sequences.series[series] if series < count else BLEND
            table.setdefault(name, {})[label] = dict(decisions=column)
    compared = relative(single, held["path_graph"], held["path_series"], kinds, names)
    ranks, losses = {}, {}
    for name, entry in table.items():
        better = catalog.entry(name)["better"]
        if better == "none":
            continue
        columns = [oriented(column["decisions"], better) for column in entry.values()]
        if len(columns) == count and BLEND not in entry:
            entry[BLEND] = dict(decisions=np.mean([column["decisions"] for column in entry.values()], 0))
            columns.append(np.mean(columns, 0))
        for label, column in zip(entry, columns):
            placed = np.full(len(names), np.nan)
            known = np.isfinite(column)
            placed[known] = stats.rankdata(column[known], method="min")
            ranks.setdefault(label, {})[name] = placed
            contest = np.full(len(names), np.nan)
            usable = known[models]
            if usable.all():
                contest[models] = stats.rankdata(column[models], method="min")
            losses.setdefault(label, {})[name] = contest
    agreed = {}
    for label, entry in losses.items():
        complete = {
            name: column[models] for name, column in entry.items() if np.isfinite(column[models]).all()
        }
        order, matrix = agreement(complete)
        agreed[label] = dict(measures=order, kendall=matrix)
    result = dict(
        about=dict(
            sequences.about,
            span=span,
            draws=draws,
            level=LEVEL,
            runs=len(np.unique(held["path_graph"])),
            decisions=int(held["decision"].max()) + 1,
            points=int(len(held["tau"])),
            paths={
                name: int((held["path_series"] == index).sum()) for index, name in enumerate(sequences.series)
            },
            damped={
                sequences.series[series]: "window %s days, damping %s days" % pair
                for series, pair in chosen.items()
            },
            settings=catalog.settings(),
        ),
        rows=[dict(name=name, kind=kind) for name, kind in zip(names, kinds)],
        catalog=[
            dict(catalog.entry(name), key=name)
            for name in list(table) + list(compared) + list(flock) + list(catalog.CALIBRATION)
        ],
        families=catalog.FAMILIES,
        absent=catalog.ABSENT,
        measures=table,
        relative=compared,
        ensemble={
            name: calibration(flock, histogram, held, series) for series, name in enumerate(sequences.series)
        },
        ranks=ranks,
        contest=losses,
        agreement=agreed,
    )
    LOG.info(
        "build exit seconds=%.3f measures=%d rows=%d", time.perf_counter() - started, len(table), len(names)
    )
    return result


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(entry) for entry in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def panel(run, output, top, workers, draws, ratio):
    from .page import write

    started = time.perf_counter()
    output = collect(run, output, top)
    result = plain(build(output, workers, draws, ratio))
    write_json(output / "panel.json", result)
    write(output / "panel.html", result)
    LOG.info("panel exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--ratio", nargs="*", default=[])
    args = parser.parse_args()
    configure()
    panel(output_path(args.run), args.output, args.top, args.workers, args.draws, args.ratio)


if __name__ == "__main__":
    main()
