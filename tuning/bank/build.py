import argparse
import json
import logging
import math
import re
import time
from collections import Counter
from contextlib import closing
from functools import partial
from multiprocessing.util import Finalize
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec import Registry
from graphspec.budgets import GraphCapacityError
from graphspec.build import staged
from graphspec.build.generators import appended, looking, recorded, regenerated
from graphspec.build.parse import distinct, parse, parsed, scope
from graphspec.build.planning import stated as build_plan
from graphspec.engine.transform import shown as viewed
from graphspec.engine.transform import validate
from graphspec.facts.appending import lineage
from graphspec.facts.stored import VERSION, Facts
from graphspec.infra.files import read_json, save_array, sha256_file, sha256_json, write_json
from graphspec.infra.log import configure
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.processes.processes import pooled
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.registries import carried, restored
from graphspec.schema import declaration
from graphspec.schema.decisions import counted, generating
from graphspec.schema.references import SORTS
from graphspec.schema.schema import VALUE, Encoding, Schema
from graphspec.schema.targets import TARGETS, content
from graphspec.schema.targets import pointed as aimed
from graphspec.schema.time import ALWAYS, NEVER
from graphspec.sources import ladybug

LOG = logging.getLogger(__name__)
WORKER = {}
UNSTORED = ("database", "graphs")
FLOAT32 = float(np.finfo(np.float32).max)


@timed("build.truth_bytes")
def truth_bytes(directory, visited=None):
    directory = Path(directory).resolve()
    visited = set() if visited is None else visited
    if directory in visited:
        raise ValueError("a bank truth lineage repeats a layer")
    visited.add(directory)
    meta = read_json(directory / "graph.json")
    total = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    if "append" in meta:
        total += sum(truth_bytes(lineage(directory, meta["append"][name]), visited) for name in ("base", "delta"))
    return total


@timed("build.budget")
def budget(task_path, task, plan, bank=None):
    if plan is None:
        return None
    plan = build_plan(plan)
    if "staging" not in task:
        raise ValueError("a bounded bank build requires staged output")
    directory = (Path(task_path).parent / task["staging"]).resolve()
    entries = read_json(directory / "stage.json")
    groups, rows, seen = {}, 0, set()
    for entry in entries:
        rows += entry["rows"]
        for name in entry["files"]:
            if name in seen:
                continue
            seen.add(name)
            found = re.findall(r"graph_(\d+)", name)
            key = found[-1] if found else "all"
            with pq.ParquetFile(directory / name) as file:
                groups[key] = groups.get(key, 0) + sum(file.metadata.row_group(at).total_byte_size for at in range(file.num_row_groups))
    reserve = rows * 128 + max(groups.values(), default=0) * 8
    if bank is not None:
        root, manifest = bank
        root = Path(root).resolve()
        truth = (root / manifest["truth_root"]).resolve()
        supervision_path = (root / manifest["supervision"]).resolve()
        if not truth.is_relative_to(root) or not supervision_path.is_relative_to(root):
            raise ValueError("the bank truth or supervision path leaves its bank")
        reserve += max((truth_bytes(truth / ("graph_%04d" % entry["index"]))
                        for entry in manifest["graphs"]), default=0) * 8
        reserve += supervision_path.stat().st_size * 16
    if reserve > plan.worker_bytes:
        raise GraphCapacityError("bank staging indexes and graph materialization need at least %d conservative working bytes per worker; the build plan permits %d"
                                 % (reserve, plan.worker_bytes))
    return plan.worker_bytes


@timed("build.table_bytes")
def table_bytes(value):
    if isinstance(value, pa.Table):
        return value.nbytes
    return sum(table_bytes(part) for part in value.values())


@timed("build.within_budget")
def within_budget(tables, limit):
    if limit is not None and table_bytes(tables) * 8 > limit:
        raise GraphCapacityError("graph Arrow tables require %d conservative working bytes; the build plan permits %d"
                                 % (table_bytes(tables) * 8, limit))


def open_source(task_path, task):
    started = time.perf_counter()
    root = Path(task_path).parent
    if "schema" not in task:
        raise ValueError("the task %s states no schema" % task_path)
    stated = Schema(task["schema"])
    if "staging" in task:
        result = staged.Staged(root / task["staging"], stated)
    elif "database" in task:
        result = ladybug.Database((root / task["database"]).resolve(), stated)
    else:
        raise ValueError("task must name a staging directory or a database")
    LOG.debug("open_source exit seconds=%.4f", time.perf_counter() - started)
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


def crowded(entity, since, until, once, sittings):
    started = time.perf_counter()
    order = np.lexsort((since, entity))
    entity, since, until, once = entity[order], since[order], until[order], once[order]
    result = bool(np.any((entity[1:] == entity[:-1]) & (since[1:] < until[:-1])))
    for moments in sittings:
        step = np.searchsorted(moments, since, "left")
        held = until > moments[np.minimum(step, len(moments) - 1)]
        shown = np.stack((entity, step), 1)[(step < len(moments)) & (once | held)]
        result |= len(np.unique(shown, axis=0)) < len(shown)
    LOG.debug(
        "crowded exit facts=%d sittings=%d crowded=%s seconds=%.4f",
        len(order),
        len(sittings),
        result,
        time.perf_counter() - started,
    )
    return result


def several(stated, tables, trained):
    started = time.perf_counter()
    parts = []
    masks, partial = [], False
    layout = tuple(stated.columns)
    for kind, columns in stated.entities().items():
        table = tables["observations"][kind]
        mask = scope(table, columns, layout, "the observations of " + kind)
        full = np.zeros(len(layout), bool)
        full[[layout.index(name) for name in columns]] = True
        partial |= not np.array_equal(mask, np.broadcast_to(full, mask.shape))
        masks.append(mask)
        stamps = [
            np.asarray(table.column(name).fill_null(base).to_numpy(zero_copy_only=False), np.int64)
            if name in table.column_names
            else np.full(table.num_rows, base, np.int64)
            for name, base in (("since", ALWAYS), ("until", NEVER), ("known", ALWAYS))
        ]
        parts.append((np.asarray(table.column("entity").to_numpy(), np.int64), *stamps))
    empty = np.zeros(0, np.int64)
    entity, since, until, known = (np.concatenate([part[at] for part in parts] + [empty]) for at in range(4))
    if partial:
        mask = np.concatenate(masks)
        mask = np.column_stack((mask, ~mask.any(1)))
        rows, columns = np.nonzero(mask)
        entity = entity[rows] * mask.shape[1] + columns
        since, until, known = (values[rows] for values in (since, until, known))
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
    splits = np.asarray(tasks.column("split").to_pylist(), object)
    common = np.asarray([split is None or split in trained for split in splits.tolist()], bool)
    sittings = []
    for observer in set(looking.tolist()):
        mine = looking == observer
        own = [mine & (common | (splits == split)) for split in set(splits[mine].tolist()) - {None}]
        sittings += [np.unique(times[chosen]) for chosen in [mine] + own]
    learned = np.maximum(since, known)
    result = crowded(entity, since, until, until == since + 1, sittings) or crowded(
        entity, learned, np.maximum(until, learned + 1), (until <= learned + 1) | (known > since), sittings
    )
    LOG.debug(
        "several exit facts=%d several=%s seconds=%.4f", len(entity), result, time.perf_counter() - started
    )
    return result


@timed("build.nameable")
def nameable(stated):
    found = {kind for kinds in aimed(stated).values() for kind in kinds}
    for entry in stated.tasks["references"].values():
        found |= set(entry["kinds"])
        if SORTS.named(entry["sort"]).edges:
            found |= {kind for table in stated.edges.values() for pair in table["endpoints"] for kind in pair}
    return [kind for kind in stated.nodes if kind in found]


def unvalued(stated, tables):
    started = time.perf_counter()
    result = {}
    for kind in nameable(stated):
        table = tables["observations"][kind]
        entity = np.asarray(table.column("entity").to_numpy(), np.int64)
        for name in stated.nodes[kind]:
            lacking = np.asarray(table.column(VALUE + name).is_null().to_numpy(zero_copy_only=False), bool)
            count = len(np.unique(entity[lacking]))
            if count:
                result["%s:%s" % (kind, name)] = count
    LOG.debug("unvalued exit columns=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("build.largest")
def largest(owner):
    return int(np.unique(np.asarray(owner, np.int64), return_counts=True)[1].max()) if len(owner) else 0


def scanned(trained, index):
    started = time.perf_counter()
    stated = WORKER["reader"].source.stated
    tables = WORKER["reader"].graph(index)
    within_budget(tables, WORKER["budget"])
    result = (
        distinct(stated, tables),
        {
            table: sorted(set(tables["edges"][table].column("relation").to_pylist()), key=str)
            for table, entry in stated.edges.items()
            if entry["column"] is not None
        },
        observing(tables),
        several(stated, tables, trained),
        unvalued(stated, tables),
        largest(tables["decisions"]["action"].column("option").to_numpy()) if stated.decided() else 0,
    )
    LOG.info("bank scan exit graph=%d seconds=%.3f", index, time.perf_counter() - started)
    return result


def merged(stated, found):
    started = time.perf_counter()
    ripened = stated.ripened()
    result = {
        name: sorted(set().union(*(entry[name] for entry in found), ripened.get(name, ())))
        for name in stated.symbolic()
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
        names = {name for entry in found for name in entry[1][table]}
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


def untasked(tasks, names):
    started = time.perf_counter()
    asked = set(tasks.column("graph").to_numpy().astype(np.int64).tolist())
    result = [name for index, name in names.items() if index not in asked]
    LOG.debug("untasked exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def pointed(stated, found, owner, kinds, graphs, rows):
    started = time.perf_counter()
    stored, numbers = stated.target_columns(), found.column("y").to_numpy().astype(np.float64)
    columns = {name: found.column(name).to_pylist() for name in stored}
    graphs, result = np.asarray(graphs, object), [dict(row) for row in rows]
    for kind, entry in stated.tasks["kinds"].items():
        chosen = np.flatnonzero(kinds == kind)
        if entry["target"] is None or not len(chosen):
            continue
        target, where = TARGETS.named(entry["target"]["kind"]), "the target of the kind of task " + kind
        for name in stored:
            if name not in target.columns and any(columns[name][place] is not None for place in chosen.tolist()):
                raise ValueError(
                    "%s states the column %s of the table of targets, which its kind of target %s does not store"
                    % (where, name, entry["target"]["kind"])
                )
        given = {name: [columns[name][place] for place in chosen.tolist()] for name in target.columns}
        points = target.points(
            dict(task=owner[chosen], graph=graphs[chosen], y=numbers[chosen], **given), entry["target"], where
        )
        given = ("series",) + tuple(target.inputs) + tuple(target.outcomes) + content(target)
        if sorted(points) != sorted(given):
            raise ValueError(
                "the kind of target %s gives for its points %s, and states the inputs %s and the outcomes %s beside "
                "their series%s"
                % (
                    entry["target"]["kind"],
                    ", ".join(sorted(points)),
                    ", ".join(target.inputs) or "none",
                    ", ".join(target.outcomes) or "none",
                    "" if target.names is None else " and the node each names",
                )
            )
        for name, values in points.items():
            for place, value in zip(chosen.tolist(), values):
                result[place][name] = value
    LOG.debug("pointed exit points=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def supervision(source, trained, names, previous=None):
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
    y = found.column("y").to_numpy().astype(np.float64)
    if not np.isfinite(y.astype(np.float32)).all():
        raise ValueError("targets must be finite at float32 precision")
    rows = []
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
        rows.append(entry)
    rows = pointed(stated, found, owner, np.asarray(kinds, object)[owner], [row["graph"] for row in rows], rows)
    rows = [dict(entry, y=value) for entry, value in zip(rows, y.tolist())]
    held = np.asarray([splits[task] for task in owner.tolist()], object)
    ids = found.column("id").to_numpy().astype(np.int64)
    if previous is not None:
        prior, prior_ids = previous
        prior_splits = np.empty(len(prior["rows"]), object)
        for split, members in prior["split"]["indices"].items():
            prior_splits[members] = split
        named = {name: index for index, name in names.items()}
        rows = prior["rows"] + rows
        held = np.r_[prior_splits, held]
        ids = np.r_[prior_ids, ids]
        graphs = np.r_[[named[row["graph"]] for row in prior["rows"]], graphs].astype(np.int64)
        times = np.r_[[row["time"] if timed else 0 for row in prior["rows"]], times].astype(np.int64)
        y = np.asarray([row["y"] for row in rows], np.float64)
    series = [row["series"] for row in rows]
    found_splits = sorted({name for name in splits if name is not None} | {name for name in held.tolist() if name is not None})
    strangers = [name for name in trained if name not in found_splits]
    if strangers:
        raise ValueError(
            "the splits trained on name %s, and the source holds the splits %s"
            % (", ".join(map(str, strangers)), ", ".join(found_splits))
        )
    chosen = np.isin(held, list(trained)) & counted(stated, rows)
    statistics = []
    measured = stated.measured()
    for name in stated.series():
        if not TARGETS.named(measured[name]).scaled:
            continue
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
    return result, groups, found_splits, untasked(tasks, names)


@timed("build.attach")
def attach(task_path, output, level, limit=None):
    configure(level)
    release()
    source = open_source(task_path, read_json(task_path))
    try:
        WORKER.update(
            reader=source.reader(),
            output=Path(output),
            budget=limit,
            finalizer=Finalize(None, source.close, exitpriority=10),
        )
    except BaseException:
        source.close()
        raise


@timed("build.release")
def release():
    if WORKER:
        try:
            WORKER["finalizer"]()
        finally:
            WORKER.clear()


def graph(context, job):
    declared, splits, observers, trained = context
    index, name, ids, rows, series = job
    started = time.perf_counter()
    if WORKER.get("declared") != declared:
        WORKER.update(declared=declared, registry=Registry.from_declaration(declared))
    registry = WORKER["registry"]
    tables = WORKER["reader"].graph(index)
    within_budget(tables, WORKER["budget"])
    found = tables["targets"].column("id").to_numpy().astype(np.int64)
    position = np.searchsorted(ids, found)
    if len(found) != len(ids) or np.any(position >= len(ids)) or not np.array_equal(ids[position], found):
        raise ValueError("graph %s reads different targets than its supervision" % name)
    parts = parsed(registry, splits, observers, tables, rows[position], series[position], name)
    truth = parse(registry, tables, parts)
    if generating(registry.schema):
        drawn = regenerated(truth, registry, viewed(truth, registry, looking(splits, trained))[1])
        truth = parse(registry, dict(tables, decisions=appended(registry, tables, drawn)), parts)
    shown = validate(truth, registry, splits)
    truth.save(WORKER["output"] / "truth" / ("graph_%04d" % index))
    metrics = dict(
        index=index,
        name=name,
        **counts(truth),
        inspection=inspection(truth, registry, splits),
        acted=largest(truth.action_option),
        seconds=time.perf_counter() - started,
    )
    LOG.info("bank graph exit graph=%d views=%d seconds=%.3f", index, len(shown), metrics["seconds"])
    return metrics


@timed("build.counts")
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


@timed("build.inspection")
def inspection(truth, registry, splits):
    kinds, tasks, partitions = np.asarray(truth.kinds), np.asarray(truth.task_kind), np.asarray(truth.task_split)
    return dict(nodes={name: int(np.count_nonzero(kinds == code)) for code, name in enumerate(registry.nodes)},
                observations={name: int(np.count_nonzero(kinds[np.asarray(truth.obs_entity)] == code))
                              for code, name in enumerate(registry.nodes)},
                edges={name: int(np.count_nonzero(np.isin(truth.fact_code, registry.codes(registry.encoding.relations[name]))))
                       for name in registry.schema.edges},
                tasks={name: dict(splits={split: int(np.count_nonzero((tasks == code) & (partitions == position)))
                                         for position, split in enumerate(splits)},
                                  without_split=int(np.count_nonzero((tasks == code) & (partitions < 0))))
                       for code, name in enumerate(registry.schema.tasks["kinds"])})


def catalogue(output, graphs):
    started = time.perf_counter()
    kinds, tokens, keys = [], [], []
    for graph in graphs:
        folder = output / "truth" / ("graph_%04d" % graph["index"])
        arrays = Facts.read(folder, ("kinds", "entity_token"))
        kinds.append(arrays["kinds"].astype(np.int64))
        tokens.append(arrays["entity_token"])
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


def stored(output):
    started = time.perf_counter()
    output = Path(output)
    files = [output / "supervision.json", output / "source-target-ids.npy"] + [path for path in (output / "truth").rglob("*") if path.is_file()]
    result = sha256_json(sorted([path.relative_to(output).as_posix(), sha256_file(path)] for path in files))
    LOG.debug("stored exit files=%d seconds=%.4f", len(files), time.perf_counter() - started)
    return result


@exclusive
def build(task_path, output, workers, trained, plan=None):
    started = time.perf_counter()
    if type(workers) is not int or workers < 1:
        raise ValueError("build workers must be a positive integer")
    trained = tuple(trained)
    if not trained or len(set(trained)) != len(trained):
        raise ValueError(
            "the splits trained on are stated as distinct names, found %s" % ", ".join(map(str, trained))
        )
    task, output = read_json(task_path), output_path(output)
    source_manifest = None
    if "staging" in task:
        source_manifest_path = Path(task_path).parent / task["staging"] / "build.json"
        if source_manifest_path.is_file():
            source_manifest = read_json(source_manifest_path)
            if plan is None:
                plan = source_manifest["plan"]
    if plan is not None and build_plan(plan).workers != workers:
        raise ValueError("bank workers must equal the build plan workers")
    limit = budget(task_path, task, plan)
    if output.exists():
        raise FileExistsError("bank exists; choose a new output")
    with closing(open_source(task_path, task)) as source:
        graphs = source.graphs()
        names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
        if len(set(names.values())) != len(names):
            raise ValueError("graph names must be unique")
        rows, groups, splits, bare = supervision(source, trained, names)
        location = source.location
    attached = (str(task_path), str(output), LOG.getEffectiveLevel(), limit)
    try:
        with pooled(workers, len(groups), attach, attached, carried, restored) as mapped:
            found = mapped(partial(scanned, trained), [index for index, _, _, _ in groups])
            stated = Schema(task["schema"])
            encoding = Encoding(
                stated, merged(stated, [entry[0] for entry in found]), None, related(stated, found)
            )
            observers = sorted({name for entry in found for name in entry[2]})
            lacking = sum((Counter(entry[4]) for entry in found), Counter())
            given = (task, encoding, any(entry[3] for entry in found), bare, lacking)
            registry = declaration.registry(*given, max(entry[5] for entry in found))
            (output / "truth").mkdir(parents=True)
            write_json(output / "supervision.json", rows)
            code = {name: place for place, name in enumerate(stated.series())}
            series = np.asarray([code[row["series"]] for row in rows["rows"]], np.int32)
            metrics = mapped(
                partial(graph, (registry.declaration(), splits, observers, trained)),
                [(index, names[index], ids, members, series[members]) for index, ids, members, _ in groups],
            )
            registry = declaration.registry(*given, max(entry["acted"] for entry in metrics))
    finally:
        release()
    save_array(output / "source-target-ids.npy", target_ids(groups, len(rows["rows"])))
    manifest = dict(
        version=VERSION,
        truth_root="truth",
        source_target_ids="source-target-ids.npy",
        supervision="supervision.json",
        skipped_references=0,
        source_fingerprint=None if source_manifest is None else source_manifest["source_fingerprint"],
        database=location,
        task=task,
        trained=list(trained),
        spec=registry.declaration(),
        spec_sha256=registry.fingerprint(),
        splits=splits,
        observers=observers,
        signals=registry.extras["signals"],
        generators=recorded(registry.schema),
        graphs=metrics,
        distinct_entities=catalogue(output, metrics),
        rows=len(rows["rows"]),
        split_rows={name: len(members) for name, members in rows["split"]["indices"].items()},
        supervision_sha256=sha256_file(output / "supervision.json"),
    )
    manifest["stored_sha256"] = sha256_json(
        dict(
            files=stored(output),
            manifest={name: value for name, value in manifest.items() if name not in UNSTORED},
            graphs=[{name: value for name, value in graph.items() if name != "seconds"} for graph in metrics],
        )
    )
    manifest["seconds"] = time.perf_counter() - started
    write_json(output / "manifest.json", manifest)
    LOG.info(
        "bank exit seconds=%.3f graphs=%d rows=%d targets=%s",
        manifest["seconds"],
        len(metrics),
        manifest["rows"],
        ",".join(registry.schema.targets()),
    )
    return manifest


@timed("build.target_ids")
def target_ids(groups, count):
    result = np.empty(count, np.int64)
    for _, ids, members, _ in groups:
        result[members] = ids
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trained", required=True, nargs="+", help="the splits the scaling of the targets is fitted on")
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
