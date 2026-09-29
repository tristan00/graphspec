import argparse
import json
import logging
import time
from contextlib import closing
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np

from graphspec import Registry, graphdb, staged, vocabulary
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.log import configure
from graphspec.infra.processes import parallel
from graphspec.infra.runtime import exclusive, output_path
from graphspec.schema import Schema, fitted
from graphspec.truth import VERSION, gather, parse
from graphspec.view import validate

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validation")
WORKER = {}


def open_source(task_path, task):
    started = time.perf_counter()
    root = Path(task_path).parent
    if "schema" not in task:
        raise ValueError("the task %s states no schema" % task_path)
    stated = Schema(task["schema"])
    if "staging" in task:
        result = staged.Staged(root / task["staging"], stated)
    elif "database" in task:
        result = graphdb.Database((root / task["database"]).resolve(), stated)
    else:
        raise ValueError("task must name a staging directory or a database")
    LOG.debug("open_source exit seconds=%.4f", time.perf_counter() - started)
    return result


def seen_until(rows, groups, trained, names):
    started = time.perf_counter()
    chosen = np.zeros(len(rows["rows"]), bool)
    for split in trained:
        chosen[rows["split"][split + "_indices"]] = True
    result = {index: None for index in names}
    for index, _, _, members, times in groups:
        if chosen[members].any():
            result[index] = int(np.asarray(times)[chosen[members]].max())
    LOG.info(
        "seen_until exit seconds=%.3f graphs=%d trained=%d",
        time.perf_counter() - started,
        len(result),
        sum(value is not None for value in result.values()),
    )
    return result


def observing(tables):
    started = time.perf_counter()
    found = set()
    for table in list(tables["observations"].values()) + list(tables["edges"].values()):
        if "observers" in table.column_names:
            for names in table.column("observers").to_pylist():
                found.update(names or ())
    if "queries" in tables and "observer" in tables["queries"].column_names:
        found.update(tables["queries"].column("observer").to_pylist())
    found.discard(None)
    LOG.debug("observing exit observers=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return sorted(found)


def scanned(job):
    index, until = job
    started = time.perf_counter()
    stated = WORKER["reader"].source.stated
    tables = WORKER["reader"].graph(index)
    result = (
        gather(stated, tables, until),
        {
            table: sorted(set(tables["edges"][table].column("relation").to_pylist()), key=str)
            for table, entry in stated.edges.items()
            if entry["column"] is not None
        },
        observing(tables),
    )
    LOG.info("bank scan exit graph=%d seconds=%.3f", index, time.perf_counter() - started)
    return result


def merged(stated, found):
    started = time.perf_counter()
    result = {
        name: {
            part: np.concatenate([entry[name][part] for entry in found]) for part in ("trained", "distinct")
        }
        for name in stated.columns
    }
    LOG.info("merged exit seconds=%.3f columns=%d", time.perf_counter() - started, len(result))
    return result


def related(stated, found):
    started = time.perf_counter()
    result = {}
    for table, entry in stated.edges.items():
        if entry["column"] is None:
            result[table] = [table]
            continue
        names = {name for _, relations, _ in found for name in relations[table]}
        if None in names:
            raise ValueError(
                "rows of the edge table %s name no relation in the column %s" % (table, entry["column"])
            )
        result[table] = sorted(names)
    LOG.info(
        "related exit seconds=%.3f relations=%d",
        time.perf_counter() - started,
        sum(len(names) for names in result.values()),
    )
    return result


def series_statistics(name, y, member, held):
    train, validation = y[member & ~held].astype(np.float32), y[member & held].astype(np.float32)
    if len(train) < 2 or len(validation) < 2 or np.ptp(train) == 0 or np.ptp(validation) == 0:
        raise ValueError("series %s needs at least two varying training and validation points" % name)
    return dict(
        name=name, mean=float(train.astype(np.float64).mean()), sd=float(train.astype(np.float64).std(ddof=1))
    )


def asked(source, names):
    started = time.perf_counter()
    queries = source.query_index()
    owners = queries.column("graph").to_numpy().astype(np.int64)
    timed = "time" in queries.column_names
    if timed and queries.column("time").null_count:
        raise ValueError("a task of the source states no time")
    moments = queries.column("time").to_numpy().astype(np.int64) if timed else np.zeros(len(owners), np.int64)
    ids = queries.column("id").to_numpy().astype(np.int64)
    order = np.lexsort((ids, moments, owners))
    first = np.searchsorted(owners[order], owners[order])
    place = np.empty(len(order), np.int64)
    place[order] = np.arange(len(order)) - first
    counts = dict(zip(*(values.tolist() for values in np.unique(owners, return_counts=True))))
    empty = sorted(set(names) - set(counts))
    if empty:
        raise ValueError("%d graphs have no queries, for example graph %d" % (len(empty), empty[0]))
    result = dict(
        counts=counts,
        row={number: row for row, number in enumerate(ids.tolist())},
        last={int(owners[row]): row for row in order.tolist()},
        place=place,
        time=moments,
        timed=timed,
        key=queries.column("key").to_numpy().astype(np.int64),
        period=queries.column("period").to_numpy().astype(np.int64)
        if "period" in queries.column_names
        else None,
    )
    LOG.info("asked exit seconds=%.3f tasks=%d", time.perf_counter() - started, len(ids))
    return result


def supervision(source, level, target, names):
    started = time.perf_counter()
    if level == "node":
        found = source.targets(level, target)
        owners = found.column("graph").to_numpy().astype(np.int64)
        timed = "time" in found.column_names
        if timed and found.column("time").null_count:
            raise ValueError("a target of the source states no time")
        times = found.column("time").to_numpy().astype(np.int64) if timed else np.zeros(len(owners), np.int64)
        places = times
        if len(
            np.unique(
                np.stack((owners, found.column("entity").to_numpy().astype(np.int64), times), 1), axis=0
            )
        ) != len(owners):
            raise ValueError("a node carries more than one target at one time")
        described = [
            dict(graph=names[owner], node=node, **(dict(time=moment) if timed else {}))
            for owner, moment, node in zip(owners.tolist(), times.tolist(), found.column("node").to_pylist())
        ]
    else:
        queries = asked(source, names)
        found = source.targets(level, target)
        owners = found.column("graph").to_numpy().astype(np.int64)
        if level == "graph":
            rows = np.asarray([queries["last"][owner] for owner in owners.tolist()], np.int64)
        else:
            rows = np.asarray(
                [queries["row"][number] for number in found.column("query").to_pylist()], np.int64
            )
        places, times = queries["place"][rows], queries["time"][rows]
        stated = source.stated.queries
        described = []
        for owner, row in zip(owners.tolist(), rows.tolist()):
            entry = {"graph": names[owner], stated["key"]: int(queries["key"][row])}
            if queries["timed"]:
                entry["time"] = int(queries["time"][row])
            if queries["period"] is not None:
                entry[stated["grouping"]] = int(queries["period"][row])
            described.append(entry)
        if level == "path":
            tau = found.column("tau").to_numpy().astype(np.float64)
            if not np.isfinite(tau).all() or np.any(tau <= 0):
                raise ValueError("path tau values must be finite and strictly positive")
            series = found.column("series").to_pylist()
            if None in series or "" in series:
                raise ValueError("every path point needs a series name")
            names_of, series_index = np.unique(np.asarray(series, object), return_inverse=True)
            order = np.lexsort((tau, series_index, places, owners))
            keys = np.stack((owners, places, series_index), 1)[order]
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
            if len(np.unique(np.stack((owners, places), 1), axis=0)) != len(owners):
                raise ValueError("a query or graph carries more than one target")
    split = found.column("split").to_pylist()
    unknown = sorted(set(split) - set(SPLITS))
    if unknown:
        raise ValueError("target splits must be train or validation, found " + ", ".join(map(str, unknown)))
    unlabelled = sorted(set(names) - set(owners.tolist()))
    if unlabelled:
        raise ValueError("%d graphs have no target, for example graph %d" % (len(unlabelled), unlabelled[0]))
    held = np.asarray([value == "validation" for value in split])
    if level == "path":
        queries = np.unique(np.stack((owners, places), 1), axis=0)
        if len(np.unique(np.stack((owners, places, held), 1), axis=0)) != len(queries):
            raise ValueError("every point of one query must carry the same split")
    y = found.column("y").to_numpy().astype(np.float64)
    train, validation = np.flatnonzero(~held), np.flatnonzero(held)
    stored = y.astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError("targets must be finite at float32 precision")
    if len(train) < 2 or len(validation) < 2 or np.ptp(stored[train]) == 0 or np.ptp(stored[validation]) == 0:
        raise ValueError("need at least two varying training targets and two varying validation targets")
    rows = [dict(entry, y=value) for entry, value in zip(described, y.tolist())]
    if level == "path":
        standard = dict(
            series=[
                series_statistics(name, y, series_index == code, held)
                for code, name in enumerate(names_of.tolist())
            ]
        )
    else:
        standard = dict(
            series=[dict(name=target, mean=float(y[train].mean()), sd=float(y[train].std(ddof=1)))]
        )
    result = dict(
        rows=rows,
        split=dict(train_indices=train.tolist(), validation_indices=validation.tolist(), **standard),
    )
    ids = found.column("id").to_numpy().astype(np.int64)
    if level == "node":
        order = np.argsort(owners, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(owners[order])) + 1, len(order)]
    groups = [
        (
            int(owners[order[lo]]),
            places[order[lo:hi]],
            ids[order[lo:hi]],
            order[lo:hi].astype(np.int64),
            times[order[lo:hi]],
        )
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


def attach(task_path, declared, categories, observers, target, output, level):
    configure(level)
    release()
    source = open_source(task_path, read_json(task_path))
    try:
        WORKER.update(
            reader=source.reader(target),
            registry=None if declared is None else Registry.from_declaration(declared),
            categories=categories,
            observers=observers,
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
    index, name, places, ids, rows = job
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
        labels = (places, rows)
    else:
        labels = np.full(tables["queries"].num_rows, -1, np.int64)
        labels[places] = rows
    truth = parse(registry, WORKER["categories"], WORKER["observers"], tables, labels, name)
    validate(truth, registry)
    truth.save(WORKER["output"] / "truth" / ("graph_%04d" % index))
    metrics = dict(index=index, name=name, **counts(truth), seconds=time.perf_counter() - started)
    LOG.info("bank graph exit graph=%d seconds=%.3f", index, metrics["seconds"])
    return metrics


def counts(truth):
    return dict(
        unit=truth.unit,
        entities=len(truth.kinds),
        observations=len(truth.obs_since),
        facts=len(truth.fact_since),
        tasks=len(truth.query_time),
        labels=len(truth.label_time),
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


@exclusive
def build(task_path, output, workers, trained):
    started = time.perf_counter()
    if type(workers) is not int or workers < 1:
        raise ValueError("build workers must be a positive integer")
    trained = tuple(trained)
    if not trained or not set(trained) <= set(SPLITS) or "validation" in trained:
        raise ValueError(
            "the splits trained on are stated among %s and leave validation out, found %s"
            % (", ".join(SPLITS), ", ".join(map(str, trained)))
        )
    task, output = read_json(task_path), output_path(output)
    if output.exists():
        raise FileExistsError("bank exists; choose a new output")
    with closing(open_source(task_path, task)) as source:
        level = vocabulary.target_level(source, task)
        graphs = source.graphs()
        names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
        if len(set(names.values())) != len(names):
            raise ValueError("graph names must be unique")
        categories = source.categories() if source.queries else []
        rows, groups = supervision(source, level, task["target"], names)
        series = [entry["name"] for entry in rows["split"]["series"]] if level == "path" else []
        location = source.location
    until = seen_until(rows, groups, trained, names)
    try:
        found = parallel(
            scanned,
            [(index, until[index]) for index, _, _, _, _ in groups],
            workers,
            attach,
            (str(task_path), None, categories, [], task["target"], str(output), LOG.getEffectiveLevel()),
        )
    finally:
        release()
    stated = Schema(task["schema"])
    encoding = fitted(stated, merged(stated, [entry[0] for entry in found]), related(stated, found))
    observers = sorted({name for entry in found for name in entry[2]})
    with closing(open_source(task_path, task)) as source:
        registry = vocabulary.registry(source, task, series, encoding)
    (output / "truth").mkdir(parents=True)
    write_json(output / "supervision.json", rows)
    jobs = [(index, names[index], places, ids, members) for index, places, ids, members, _ in groups]
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
                observers,
                task["target"],
                str(output),
                LOG.getEffectiveLevel(),
            ),
        )
    finally:
        release()
    manifest = dict(
        version=VERSION,
        database=location,
        task=task,
        target=dict(table=task["target"], level=level),
        trained=list(trained),
        spec=registry.declaration(),
        spec_sha256=registry.fingerprint(),
        categories=categories,
        observers=observers,
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
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trained", required=True, nargs="+", help="the splits the encoders are fitted on")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure()
    manifest = build(args.task, args.output, args.workers, args.trained)
    LOG.info("build command exit seconds=%.3f", time.perf_counter() - started)
    print(
        json.dumps({key: value for key, value in manifest.items() if key not in ("graphs", "spec")}, indent=2)
    )


if __name__ == "__main__":
    main()
