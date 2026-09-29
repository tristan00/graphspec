import argparse
import json
import logging
import shutil
import time
from contextlib import closing
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np

from graphspec import Registry, graphdb, staged, vocabulary
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.log import configure
from graphspec.infra.processes import WORKERS, parallel
from graphspec.infra.runtime import exclusive, output_path
from graphspec.truth import ROWS, VERSION, Rows, encode, parse, verify

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validation", "spread")
WORKER = {}


def open_source(task_path, task):
    root = Path(task_path).parent
    if task.get("staging"):
        return staged.Staged(root / task["staging"])
    if not task.get("database"):
        raise ValueError("task must name a staging directory or a database")
    return graphdb.Database((root / task["database"]).resolve())


def series_statistics(name, y, member, learned, held):
    train, validation = y[member & learned].astype(np.float32), y[member & held].astype(np.float32)
    if len(train) < 2 or len(validation) < 2 or np.ptp(train) == 0 or np.ptp(validation) == 0:
        raise ValueError("series %s needs at least two varying training and validation points" % name)
    return dict(
        name=name, mean=float(train.astype(np.float64).mean()), sd=float(train.astype(np.float64).std(ddof=1))
    )


def supervision(source, level, target, names):
    started = time.perf_counter()
    if level == "node":
        found = source.targets(level, target)
        owners, steps = (
            found.column("graph").to_numpy().astype(np.int64),
            found.column("step").to_numpy().astype(np.int64),
        )
        if np.any(steps < 0):
            raise ValueError("target steps must not be negative")
        if len(
            np.unique(
                np.stack((owners, found.column("entity").to_numpy().astype(np.int64), steps), 1), axis=0
            )
        ) != len(owners):
            raise ValueError("a node carries more than one target at one step")
        described = [
            dict(graph=names[owner], step=step, node=node)
            for owner, step, node in zip(owners.tolist(), steps.tolist(), found.column("node").to_pylist())
        ]
    else:
        queries = source.query_index()
        counts = dict(
            zip(
                *(
                    values.tolist()
                    for values in np.unique(queries.column("graph").to_numpy(), return_counts=True)
                )
            )
        )
        empty = sorted(set(names) - set(counts))
        if empty:
            raise ValueError("%d graphs have no queries, for example graph %d" % (len(empty), empty[0]))
        lookup = {
            (g, s): (p, k)
            for g, s, p, k in zip(
                *(queries.column(name).to_pylist() for name in ("graph", "step", "period", "key"))
            )
        }
        found = source.targets(level, target)
        owners = found.column("graph").to_numpy().astype(np.int64)
        if level == "graph":
            steps = np.asarray([counts[owner] - 1 for owner in owners.tolist()], np.int64)
        else:
            steps = found.column("step").to_numpy().astype(np.int64)
        described = [
            dict(
                graph=names[owner],
                decision=lookup[(owner, step)][1],
                step=step,
                turn=lookup[(owner, step)][0],
            )
            for owner, step in zip(owners.tolist(), steps.tolist())
        ]
        if level == "path":
            tau = found.column("tau").to_numpy().astype(np.float64)
            if not np.isfinite(tau).all() or np.any(tau <= 0):
                raise ValueError("path tau values must be finite and strictly positive")
            series = found.column("series").to_pylist()
            if None in series or "" in series:
                raise ValueError("every path point needs a series name")
            names_of, series_index = np.unique(np.asarray(series, object), return_inverse=True)
            order = np.lexsort((tau, series_index, steps, owners))
            keys = np.stack((owners, steps, series_index), 1)[order]
            same = (np.diff(keys, axis=0) == 0).all(1) & (np.diff(tau[order]) == 0)
            if same.any():
                raise ValueError(
                    "graph %s has two points with one query, one series and one tau"
                    % names[int(owners[order][np.flatnonzero(same)[0]])]
                )
            described = [
                dict(entry, tau=value, series=name)
                for entry, value, name in zip(described, tau.tolist(), series)
            ]
        else:
            order = np.argsort(owners, kind="stable")
            if len(np.unique(np.stack((owners, steps), 1), axis=0)) != len(owners):
                raise ValueError("a query or graph carries more than one target")
    split = found.column("split").to_pylist()
    unknown = sorted(set(split) - set(SPLITS))
    if unknown:
        raise ValueError(
            "target splits must be train, validation or spread, found " + ", ".join(map(str, unknown))
        )
    unlabelled = sorted(set(names) - set(owners.tolist()))
    if unlabelled:
        raise ValueError("%d graphs have no target, for example graph %d" % (len(unlabelled), unlabelled[0]))
    held = np.asarray([value == "validation" for value in split])
    learned = np.asarray([value == "train" for value in split])
    part = np.asarray([SPLITS.index(value) for value in split], np.int64)
    if level == "path":
        queries = np.unique(np.stack((owners, steps), 1), axis=0)
        if len(np.unique(np.stack((owners, steps, part), 1), axis=0)) != len(queries):
            raise ValueError("every point of one query must carry the same split")
    y = found.column("y").to_numpy().astype(np.float64)
    train, validation = np.flatnonzero(learned), np.flatnonzero(held)
    stored = y.astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError("targets must be finite at float32 precision")
    if len(train) < 2 or len(validation) < 2 or np.ptp(stored[train]) == 0 or np.ptp(stored[validation]) == 0:
        raise ValueError("need at least two varying training targets and two varying validation targets")
    rows = [dict(entry, y=value) for entry, value in zip(described, y.tolist())]
    if level == "path":
        standard = dict(
            series=[
                series_statistics(name, y, series_index == code, learned, held)
                for code, name in enumerate(names_of.tolist())
            ]
        )
    else:
        standard = dict(reward_mean=float(y[train].mean()), reward_sd=float(y[train].std(ddof=1)))
    result = dict(
        rows=rows,
        split=dict(
            train_indices=train.tolist(),
            validation_indices=validation.tolist(),
            spread_indices=np.flatnonzero(part == 2).tolist(),
            **standard,
        ),
    )
    ids = found.column("id").to_numpy().astype(np.int64)
    if level == "node":
        order = np.argsort(owners, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(owners[order])) + 1, len(order)]
    groups = [
        (int(owners[order[lo]]), steps[order[lo:hi]], ids[order[lo:hi]], order[lo:hi].astype(np.int64))
        for lo, hi in zip(boundaries[:-1], boundaries[1:])
    ]
    LOG.info(
        "supervision exit seconds=%.3f rows=%d train=%d validation=%d level=%s",
        time.perf_counter() - started,
        len(rows),
        len(train),
        len(validation),
        level,
    )
    return result, groups


def attach(task_path, declared, categories, target, output, level):
    configure(level)
    release()
    source = open_source(task_path, read_json(task_path))
    try:
        WORKER.update(
            reader=source.reader(target),
            registry=Registry.from_declaration(declared),
            categories=categories,
            output=Path(output),
            finalizer=Finalize(None, source.close, exitpriority=10),
        )
    except BaseException:
        source.close()
        raise


def release():
    if WORKER:
        try:
            WORKER["finalizer"]()
        finally:
            WORKER.clear()


def graph(job):
    index, name, steps, ids, rows = job
    started = time.perf_counter()
    registry = WORKER["registry"]
    tables = WORKER["reader"].graph(index)
    if registry.extras["level"] == "node":
        found = tables["targets"].column("id").to_numpy().astype(np.int64)
        position = np.searchsorted(ids, found)
        if len(found) != len(ids) or np.any(position >= len(ids)) or not np.array_equal(ids[position], found):
            raise ValueError("graph %s reads different targets than its supervision" % name)
        labels = rows[position]
    elif registry.extras["level"] == "path":
        labels = (steps, rows)
    else:
        labels = np.full(tables["queries"].num_rows, -1, np.int64)
        labels[steps] = rows
    truth = parse(registry, WORKER["categories"], tables, labels, name)
    truth.save(WORKER["output"] / "truth" / ("graph_%04d" % index))
    metrics = dict(index=index, name=name, **counts(truth), seconds=time.perf_counter() - started)
    LOG.info("bank graph exit graph=%d seconds=%.3f", index, metrics["seconds"])
    return metrics


def counts(truth):
    return dict(
        steps=truth.steps,
        entities=len(truth.kinds),
        observations=int((truth.obs_stop.astype(np.int64) - truth.obs_start).sum()),
        facts=int((truth.fact_stop.astype(np.int64) - truth.fact_start).sum()) + len(truth.removal_step),
        labels=len(truth.label_step),
        observation_spans=len(truth.obs_start),
        fact_spans=len(truth.fact_start),
    )


def catalogue(output, graphs):
    started = time.perf_counter()
    kinds, tokens, keys = [], [], []
    for graph in graphs:
        folder = output / "truth" / ("graph_%04d" % graph["index"])
        kinds.append(np.load(folder / "kinds.npy").astype(np.int64))
        tokens.append(np.load(folder / "entity_token.npy"))
        keys.extend(read_json(folder / "graph.json")["keys"])
    kinds, tokens = (
        np.concatenate(kinds + [np.zeros(0, np.int64)]),
        np.concatenate(tokens + [np.zeros(0, np.uint64)]),
    )
    order = np.lexsort((tokens, kinds))
    first = np.r_[True, (kinds[order][1:] != kinds[order][:-1]) | (tokens[order][1:] != tokens[order][:-1])]
    owner = order[np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))]
    if any(keys[a] != keys[b] for a, b in zip(order.tolist(), owner.tolist()) if a != b):
        raise ValueError("two entity keys of one node type share a token; entity identity would merge them")
    LOG.info(
        "catalogue exit seconds=%.3f entities=%d distinct=%d",
        time.perf_counter() - started,
        len(kinds),
        int(first.sum()),
    )
    return int(first.sum())


def attach_upgrade(declared, source, output, level):
    configure(level)
    WORKER.update(registry=Registry.from_declaration(declared), source=Path(source), output=Path(output))


def upgraded(job):
    index, name = job
    started = time.perf_counter()
    folder = "graph_%04d" % index
    registry = WORKER["registry"]
    rows = Rows.load(WORKER["source"] / "truth" / folder)
    truth = encode(rows, registry)
    verify(truth, registry)
    back = truth.rows()
    for key in ROWS:
        left, right = np.asarray(rows.arrays[key]), back.arrays[key]
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left.view(np.uint8), right.view(np.uint8))
        ):
            raise ValueError("graph %s does not reproduce %s from its spans" % (name, key))
    truth.save(WORKER["output"] / "truth" / folder)
    metrics = dict(index=index, name=name, **counts(truth), seconds=time.perf_counter() - started)
    LOG.info("bank upgrade exit graph=%d seconds=%.3f", index, metrics["seconds"])
    return metrics


@exclusive
def upgrade(source, output, workers=1):
    started = time.perf_counter()
    source, output = Path(source), output_path(output)
    manifest = read_json(source / "manifest.json")
    if manifest["version"] != VERSION - 1:
        raise ValueError(
            "bank %s has version %s; upgrade reads version %d" % (source, manifest["version"], VERSION - 1)
        )
    if output.exists():
        raise FileExistsError("bank exists; choose a new output")
    (output / "truth").mkdir(parents=True)
    shutil.copyfile(source / "supervision.json", output / "supervision.json")
    jobs = [(graph["index"], graph["name"]) for graph in manifest["graphs"]]
    metrics = parallel(
        upgraded,
        jobs,
        workers,
        attach_upgrade,
        (manifest["spec"], str(source), str(output), LOG.getEffectiveLevel()),
    )
    manifest = dict(
        manifest,
        version=VERSION,
        graphs=metrics,
        distinct_entities=catalogue(output, metrics),
        seconds=time.perf_counter() - started,
    )
    write_json(output / "manifest.json", manifest)
    LOG.info("bank upgrade exit seconds=%.3f graphs=%d", manifest["seconds"], len(metrics))
    return manifest


@exclusive
def build(task_path, output, workers=1):
    started = time.perf_counter()
    if type(workers) is not int or workers < 1:
        raise ValueError("build workers must be a positive integer")
    task, output = read_json(task_path), output_path(output)
    if output.exists():
        raise FileExistsError("bank exists; choose a new output")
    with closing(open_source(task_path, task)) as source:
        schema = source.schema
        level = vocabulary.target_level(schema, task)
        graphs = source.graphs()
        names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
        if len(set(names.values())) != len(names):
            raise ValueError("graph names must be unique")
        categories = source.categories() if schema.queries else []
        rows, groups = supervision(source, level, task["target"], names)
        series = [entry["name"] for entry in rows["split"]["series"]] if level == "path" else []
        registry = vocabulary.registry(schema, task, series)
        graphdb.check(registry)
    (output / "truth").mkdir(parents=True)
    write_json(output / "supervision.json", rows)
    jobs = [(index, names[index], steps, ids, members) for index, steps, ids, members in groups]
    try:
        metrics = parallel(
            graph,
            jobs,
            workers,
            attach,
            (
                str(task_path),
                registry.declaration(),
                categories,
                task["target"],
                str(output),
                LOG.getEffectiveLevel(),
            ),
        )
    finally:
        release()
    manifest = dict(
        version=VERSION,
        database=source.location,
        task=task,
        target=dict(table=task["target"], level=level),
        spec=registry.declaration(),
        spec_sha256=registry.fingerprint(),
        categories=categories,
        signals=registry.extras["signals"],
        graphs=metrics,
        distinct_entities=catalogue(output, metrics),
        rows=len(rows["rows"]),
        train_rows=len(rows["split"]["train_indices"]),
        validation_rows=len(rows["split"]["validation_indices"]),
        supervision_sha256=sha256_file(output / "supervision.json"),
        seconds=time.perf_counter() - started,
    )
    write_json(output / "manifest.json", manifest)
    LOG.info(
        "bank exit seconds=%.3f graphs=%d rows=%d level=%s",
        manifest["seconds"],
        len(metrics),
        manifest["rows"],
        level,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--upgrade", help="a version 6 bank to rewrite as spans")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if (args.task is None) == (args.upgrade is None):
        parser.error("give exactly one of --task and --upgrade")
    configure()
    if args.upgrade:
        manifest = upgrade(args.upgrade, args.output, args.workers)
    else:
        manifest = build(args.task, args.output, args.workers)
    print(
        json.dumps({key: value for key, value in manifest.items() if key not in ("graphs", "spec")}, indent=2)
    )


if __name__ == "__main__":
    main()
