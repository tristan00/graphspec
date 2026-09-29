import argparse
import json
import logging
import math
import time
from contextlib import closing
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np

from graphspec import Registry, graphdb, staged, vocabulary
from graphspec.clock import ALWAYS, NEVER
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.log import configure
from graphspec.infra.processes import parallel
from graphspec.infra.runtime import exclusive, output_path
from graphspec.schema import Schema, fitted
from graphspec.truth import VERSION, gather, parse
from graphspec.view import validate

LOG = logging.getLogger(__name__)
WORKER = {}
FLOAT32 = float(np.finfo(np.float32).max)


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
        chosen[rows["split"]["indices"][split]] = True
    result = {index: None for index in names}
    for index, _, members, times in groups:
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
    if "observer" in tables["tasks"].column_names:
        found.update(tables["tasks"].column("observer").to_pylist())
    found.discard(None)
    LOG.debug("observing exit observers=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return sorted(found)


def several(stated, tables):
    started = time.perf_counter()
    parts = []
    for kind in stated.entities():
        table = tables["observations"][kind]
        stamps = [
            np.asarray(table.column(name).fill_null(base).to_numpy(zero_copy_only=False), np.int64)
            if name in table.column_names
            else np.full(table.num_rows, base, np.int64)
            for name, base in (("since", ALWAYS), ("until", NEVER))
        ]
        parts.append((np.asarray(table.column("entity").to_numpy(), np.int64), *stamps))
    empty = np.zeros(0, np.int64)
    entity, since, until = (np.concatenate([part[at] for part in parts] + [empty]) for at in range(3))
    order = np.lexsort((since, entity))
    entity, since, until = entity[order], since[order], until[order]
    result = bool(np.any((entity[1:] == entity[:-1]) & (since[1:] < until[:-1])))
    tasks = tables["tasks"]
    times = (
        np.asarray(tasks.column("time").to_numpy(), np.int64)
        if "time" in tasks.column_names
        else np.zeros(tasks.num_rows, np.int64)
    )
    looking = np.asarray(
        tasks.column("observer").to_pylist() if "observer" in tasks.column_names else [None] * tasks.num_rows,
        object,
    )
    for observer in set(looking.tolist()):
        moments = np.unique(times[looking == observer])
        step = np.searchsorted(moments, since, "left")
        held = until > moments[np.minimum(step, len(moments) - 1)]
        shown = np.stack((entity, step), 1)[(step < len(moments)) & ((until == since + 1) | held)]
        result |= len(np.unique(shown, axis=0)) < len(shown)
    LOG.debug(
        "several exit facts=%d several=%s seconds=%.4f", len(order), result, time.perf_counter() - started
    )
    return result


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
        several(stated, tables),
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
        names = {name for _, relations, _, _ in found for name in relations[table]}
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


def single(stated, found):
    started = time.perf_counter()
    owners = found.column("graph").to_numpy().astype(np.int64)
    timed = "time" in found.column_names
    moments = found.column("time").to_numpy().astype(np.int64) if timed else np.zeros(len(owners), np.int64)
    looking = (
        np.unique(np.asarray(found.column("observer").to_pylist(), object).astype(str), return_inverse=True)[
            1
        ]
        if "observer" in found.column_names
        else np.zeros(len(owners), np.int64)
    )
    places = np.stack((owners, looking.reshape(-1), moments), 1)
    aimed = set(stated.aimed_kinds())
    result = bool(
        len(places)
        and len(np.unique(places, axis=0)) == len(places)
        and all(kind in aimed for kind in found.column("kind").to_pylist())
    )
    LOG.debug(
        "single exit tasks=%d single=%s seconds=%.4f", len(owners), result, time.perf_counter() - started
    )
    return result


def supervision(source, trained, names):
    started = time.perf_counter()
    stated = source.stated
    tasks, found = source.asked(), source.aims()
    kinds = tasks.column("kind").to_pylist()
    strangers = sorted({str(kind) for kind in kinds if kind not in stated.tasks["kinds"]})
    if strangers:
        raise ValueError("tasks are of kinds the schema does not state: " + ", ".join(strangers))
    owner = found.column("task").to_numpy().astype(np.int64)
    graphs = tasks.column("graph").to_numpy().astype(np.int64)[owner]
    timed = "time" in tasks.column_names
    if timed and tasks.column("time").null_count:
        raise ValueError("a task of the source states no time")
    times = (
        tasks.column("time").to_numpy().astype(np.int64)[owner] if timed else np.zeros(len(owner), np.int64)
    )
    keys = tasks.column("key").to_numpy().astype(np.int64)
    grouping = tasks.column("period").to_numpy().astype(np.int64) if "period" in tasks.column_names else None
    splits, weights = tasks.column("split").to_pylist(), tasks.column("weight").to_pylist()
    known = found.column("known").to_pylist() if "known" in found.column_names else [None] * len(owner)
    path = stated.aim() == "path"
    y = found.column("y").to_numpy().astype(np.float64)
    if not np.isfinite(y.astype(np.float32)).all():
        raise ValueError("targets must be finite at float32 precision")
    rows, series = [], []
    for place, task in enumerate(owner.tolist()):
        kind = stated.tasks["kinds"][kinds[task]]
        if kind["target"] is None:
            raise ValueError(
                "a task of the kind %s holds a target, which its kind does not state" % kinds[task]
            )
        if splits[task] is None:
            raise ValueError("a task of graph %s holds a target and states no split" % names[graphs[place]])
        entry = {
            "graph": names[int(graphs[place])],
            "kind": kinds[task],
            stated.tasks["key"]: int(keys[task]),
        }
        if timed:
            entry["time"] = int(times[place])
        if grouping is not None:
            entry[stated.tasks["grouping"]] = int(grouping[task])
        if weights[task] is not None:
            weight = float(weights[task])
            if not (math.isfinite(weight) and 0 < weight <= FLOAT32):
                raise ValueError(
                    "the task %d of the kind %s of graph %s states the weight %r; a weight is above zero and "
                    "finite at float32 precision"
                    % (keys[task], kinds[task], names[int(graphs[place])], weight)
                )
            entry["weight"] = weight
        if known[place] is not None:
            entry["known"] = int(known[place])
        series.append(kind["target"]["series"][0])
        rows.append(entry)
    if path:
        tau = found.column("tau").to_numpy(zero_copy_only=False).astype(np.float64)
        if not np.isfinite(tau).all() or np.any(tau <= 0):
            raise ValueError("path tau values must be finite and strictly positive")
        series = found.column("series").to_pylist()
        for place, (task, name) in enumerate(zip(owner.tolist(), series)):
            if name not in stated.tasks["kinds"][kinds[task]]["target"]["series"]:
                raise ValueError(
                    "a point of a task of the kind %s names the series %r, which its kind does not state"
                    % (kinds[task], name)
                )
        listed = list(stated.series())
        order = np.lexsort((tau, [listed.index(name) for name in series], owner))
        same = (
            (np.diff(owner[order]) == 0)
            & (np.diff(tau[order]) == 0)
            & (np.asarray(series, object)[order][1:] == np.asarray(series, object)[order][:-1])
        )
        if same.any():
            raise ValueError(
                "graph %s has two points with one task, one series and one tau"
                % names[int(graphs[order][np.flatnonzero(same)[0]])]
            )
        rows = [dict(entry, tau=value, series=name) for entry, value, name in zip(rows, tau.tolist(), series)]
    elif len(np.unique(owner)) != len(owner):
        raise ValueError("a task carries more than one target")
    rows = [dict(entry, y=value) for entry, value in zip(rows, y.tolist())]
    held = np.asarray([splits[task] for task in owner.tolist()], object)
    found_splits = sorted({name for name in splits if name is not None})
    strangers = [name for name in trained if name not in found_splits]
    if strangers:
        raise ValueError(
            "the splits trained on name %s, and the source holds the splits %s"
            % (", ".join(map(str, strangers)), ", ".join(found_splits))
        )
    chosen = np.isin(held, list(trained))
    statistics = []
    for name in stated.series():
        values = y[chosen & (np.asarray(series, object) == name)]
        if len(values) < 2 or np.ptp(values.astype(np.float32)) == 0:
            raise ValueError(
                "the series %s needs at least two varying targets in the splits trained on" % name
            )
        statistics.append(dict(name=name, mean=float(values.mean()), sd=float(values.std(ddof=1))))
    result = dict(
        rows=rows,
        split=dict(
            indices={name: np.flatnonzero(held == name).tolist() for name in found_splits},
            series=statistics,
        ),
    )
    ids = found.column("id").to_numpy().astype(np.int64)
    order = np.argsort(graphs, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(graphs[order])) + 1, len(order)][: len(order) + 1]
    groups = {
        int(graphs[order[lo]]): (
            int(graphs[order[lo]]),
            ids[order[lo:hi]],
            order[lo:hi].astype(np.int64),
            times[order[lo:hi]],
        )
        for lo, hi in zip(boundaries[:-1], boundaries[1:])
        if hi > lo
    }
    empty = np.zeros(0, np.int64)
    groups = [groups.get(index, (index, empty, empty, empty)) for index in names]
    LOG.info(
        "supervision exit seconds=%.3f rows=%d splits=%s",
        time.perf_counter() - started,
        len(rows),
        ",".join(found_splits),
    )
    return result, groups, found_splits, single(stated, tasks)


def attach(task_path, declared, splits, observers, output, level):
    configure(level)
    release()
    source = open_source(task_path, read_json(task_path))
    try:
        WORKER.update(
            reader=source.reader(),
            registry=None if declared is None else Registry.from_declaration(declared),
            splits=splits,
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
    index, name, ids, rows = job
    started = time.perf_counter()
    registry = WORKER["registry"]
    tables = WORKER["reader"].graph(index)
    found = tables["targets"].column("id").to_numpy().astype(np.int64)
    position = np.searchsorted(ids, found)
    if len(found) != len(ids) or np.any(position >= len(ids)) or not np.array_equal(ids[position], found):
        raise ValueError("graph %s reads different targets than its supervision" % name)
    truth = parse(registry, WORKER["splits"], WORKER["observers"], tables, rows[position], name)
    shown = validate(truth, registry, WORKER["splits"])
    truth.save(WORKER["output"] / "truth" / ("graph_%04d" % index))
    metrics = dict(
        index=index,
        name=name,
        **counts(truth),
        seconds=time.perf_counter() - started,
    )
    LOG.info("bank graph exit graph=%d views=%d seconds=%.3f", index, len(shown), metrics["seconds"])
    return metrics


def counts(truth):
    return dict(
        unit=truth.unit,
        entities=len(truth.kinds),
        observations=len(truth.obs_since),
        facts=len(truth.fact_since),
        members=len(truth.set_edge),
        held=len(truth.inner_owner),
        tasks=len(truth.task_time),
        targets=len(truth.target_task),
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
    if not trained or len(set(trained)) != len(trained):
        raise ValueError(
            "the splits trained on are stated as distinct names, found %s" % ", ".join(map(str, trained))
        )
    task, output = read_json(task_path), output_path(output)
    if output.exists():
        raise FileExistsError("bank exists; choose a new output")
    with closing(open_source(task_path, task)) as source:
        graphs = source.graphs()
        names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
        if len(set(names.values())) != len(names):
            raise ValueError("graph names must be unique")
        rows, groups, splits, alone = supervision(source, trained, names)
        location = source.location
    until = seen_until(rows, groups, trained, names)
    try:
        found = parallel(
            scanned,
            [(index, until[index]) for index, _, _, _ in groups],
            workers,
            attach,
            (str(task_path), None, splits, [], str(output), LOG.getEffectiveLevel()),
        )
    finally:
        release()
    stated = Schema(task["schema"])
    encoding = fitted(stated, merged(stated, [entry[0] for entry in found]), related(stated, found))
    observers = sorted({name for entry in found for name in entry[2]})
    registry = vocabulary.registry(task, encoding, alone, any(entry[3] for entry in found))
    (output / "truth").mkdir(parents=True)
    write_json(output / "supervision.json", rows)
    jobs = [(index, names[index], ids, members) for index, ids, members, _ in groups]
    try:
        metrics = parallel(
            graph,
            jobs,
            workers,
            attach,
            (
                str(task_path),
                registry.declaration(),
                splits,
                observers,
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
        trained=list(trained),
        spec=registry.declaration(),
        spec_sha256=registry.fingerprint(),
        splits=splits,
        observers=observers,
        signals=registry.extras["signals"],
        graphs=metrics,
        distinct_entities=catalogue(output, metrics),
        rows=len(rows["rows"]),
        split_rows={name: len(members) for name, members in rows["split"]["indices"].items()},
        supervision_sha256=sha256_file(output / "supervision.json"),
        seconds=time.perf_counter() - started,
    )
    write_json(output / "manifest.json", manifest)
    LOG.info(
        "bank exit seconds=%.3f graphs=%d rows=%d target=%s",
        manifest["seconds"],
        len(metrics),
        manifest["rows"],
        registry.extras["target"],
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
