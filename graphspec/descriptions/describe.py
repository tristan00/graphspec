import logging
import time
from pathlib import Path

import numpy as np

from graphspec.facts.stored import Facts, admitted
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.declaration import Registry
from graphspec.schema.time import ALWAYS, NEVER

LOG = logging.getLogger(__name__)
VERSION = 9
MODELS = dict(none="no time", order="order without a clock", clock="one clock", clocks="two clocks")
READING = (
    "at a time: the nodes observed and their observations, and the edges a present fact holds; a run: a stretch of "
    "time over which one edge holds without a break, bounded where it has a start and an end, lasting in the unit "
    "of its graph"
)


@timed("describe.stem")
def stem(path, manifest):
    key = sha256_json(dict(stored=manifest["stored_sha256"], version=VERSION))
    return Path(path) / "descriptions" / "truth" / key


@timed("describe.moments")
def moments(*stamps):
    held = np.concatenate([np.asarray(values, np.int64) for values in stamps])
    return np.unique(held[(held > ALWAYS) & (held < NEVER)])


@timed("describe.counted")
def counted(since, until, times):
    return (
        np.searchsorted(np.sort(since), times, "right") - np.searchsorted(np.sort(until), times, "right")
    ).tolist()


def runs(groups, since, until):
    started = time.perf_counter()
    groups = np.unique(groups, return_inverse=True)[1].astype(np.int64).reshape(-1)
    order = np.lexsort((since, groups))
    groups, since, until = groups[order], since[order], until[order]
    bounds = np.unique(np.concatenate([since, until]))
    lifted = groups * (len(bounds) + 1)
    start = np.searchsorted(bounds, since) + lifted
    stop = np.maximum.accumulate(np.searchsorted(bounds, until) + lifted)
    first = np.r_[True, start[1:] > stop[:-1]][: len(groups)]
    last = np.r_[np.flatnonzero(first)[1:] - 1, len(groups) - 1][: int(first.sum())]
    result = since[first], bounds[stop[last] - lifted[last]]
    LOG.debug("describe runs exit facts=%d runs=%d seconds=%.4f", len(groups), len(last), time.perf_counter() - started)
    return result


@timed("describe.lasting")
def lasting(since, until):
    lengths = (until - since)[(since > ALWAYS) & (until < NEVER)]
    held = len(lengths) > 0
    return dict(
        runs=len(since),
        bounded=len(lengths),
        least=int(lengths.min()) if held else None,
        most=int(lengths.max()) if held else None,
        mean=float(lengths.mean()) if held else None,
    )


def kinds(registry, facts, times):
    started = time.perf_counter()
    entity = np.asarray(facts.obs_entity, np.int64)
    since, until = np.asarray(facts.obs_since, np.int64), np.asarray(facts.obs_until, np.int64)
    owner, result = np.asarray(facts.kinds, np.int64), {}
    for kind in registry.nodes:
        rows = owner[entity] == registry.kind(kind)
        result[kind] = dict(
            nodes=int((owner == registry.kind(kind)).sum()),
            observations=int(rows.sum()),
            nodes_at=counted(*runs(entity[rows], since[rows], until[rows]), times),
            observations_at=counted(since[rows], until[rows], times),
        )
    LOG.debug("describe kinds exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


def relations(registry, facts, times):
    started = time.perf_counter()
    owner = np.asarray(facts.kinds, np.int64)
    code, present = np.asarray(facts.fact_code, np.int64), np.asarray(facts.fact_present, bool)
    source, target = np.asarray(facts.fact_source, np.int64), np.asarray(facts.fact_target, np.int64)
    since, until = np.asarray(facts.fact_since, np.int64), np.asarray(facts.fact_until, np.int64)
    result = {}
    for table, entry in registry.schema.edges.items():
        for name in registry.encoding.relations[table]:
            rows = code == registry.code(name)
            held = rows & present
            pairs = source[held] * len(owner) + target[held]
            found = runs(pairs, since[held], until[held])
            ends = sorted(set(zip(owner[source[rows]].tolist(), owner[target[rows]].tolist())))
            result[name] = dict(
                table=table,
                ends=[[registry.nodes[first], registry.nodes[second]] for first, second in ends],
                columns=list(entry["columns"]),
                edges=len(np.unique(pairs)),
                facts=int(held.sum()),
                absent=int((rows & ~present).sum()),
                edges_at=counted(*found, times),
                **lasting(*found),
            )
    LOG.debug(
        "describe relations exit graph=%s relations=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started
    )
    return result


def tasks(registry, facts, splits):
    started = time.perf_counter()
    kind, split = np.asarray(facts.task_kind, np.int64), np.asarray(facts.task_split, np.int64)
    result = {}
    for code, name in enumerate(registry.schema.tasks["kinds"]):
        rows = kind == code
        result[name] = dict(
            tasks=int(rows.sum()),
            splits={held: int((rows & (split == at)).sum()) for at, held in enumerate(splits)},
            unsplit=int((rows & (split < 0)).sum()),
        )
    LOG.debug("describe tasks exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


def decisions(registry, facts):
    started = time.perf_counter()
    schema = registry.schema
    clocked, listed = schema.model != "none", list(schema.tasks["kinds"])
    drawn = [None] * len(facts.option_task)
    for option, generator in zip(np.asarray(facts.origin_option).tolist(), np.asarray(facts.origin_generator).tolist()):
        drawn[option] = facts.generators[generator]
    decided, first, count = (
        found.tolist() for found in np.unique(np.asarray(facts.option_task), return_index=True, return_counts=True)
    )
    kind, keys, times = (np.asarray(facts.arrays[name]).tolist() for name in ("task_kind", "task_key", "task_time"))
    result = {}
    for name in schema.decided():
        found = [
            (task, drawn[at : at + size])
            for task, at, size in zip(decided, first, count)
            if kind[task] == listed.index(name)
        ]
        stated = [held["name"] for held in schema.tasks["kinds"][name]["target"]["generators"]]
        named = list(dict.fromkeys(stated + sorted({held for _, options in found for held in options} - {None})))
        each = [
            dict(
                key=keys[task],
                time=times[task] if clocked else None,
                options=len(options),
                recorded=options.count(None),
                regenerated=len(options) - options.count(None),
                generators={held: options.count(held) for held in named},
            )
            for task, options in found
        ]
        result[name] = dict(
            decisions=len(each),
            **{part: sum(entry[part] for entry in each) for part in ("options", "recorded", "regenerated")},
            generators={held: sum(entry["generators"][held] for entry in each) for held in named},
            each=each,
        )
    LOG.debug("describe decisions exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


@timed("describe.clocked")
def clocked(registry, facts):
    stamps = [facts.arrays[name] for name in ("obs_since", "obs_until", "fact_since", "fact_until")]
    return moments(*stamps, facts.task_time if registry.schema.model != "none" else [])


@timed("describe.stored")
def stored(path, graph):
    return Facts.load(Path(path) / "truth" / ("graph_%04d" % graph["index"]))


def graphed(registry, facts, splits):
    started = time.perf_counter()
    times = clocked(registry, facts)
    result = dict(
        name=facts.name,
        unit=facts.unit,
        times=times.tolist(),
        kinds=kinds(registry, facts, times),
        relations=relations(registry, facts, times),
        tasks=tasks(registry, facts, splits),
        decisions=decisions(registry, facts),
    )
    LOG.info("describe graph exit graph=%s times=%d seconds=%.3f", facts.name, len(times), time.perf_counter() - started)
    return result


def recorded(path, manifest):
    started = time.perf_counter()
    registry = Registry.from_declaration(manifest["spec"])
    result = dict(
        version=VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=dict(model=registry.schema.model, units=list(registry.schema.units)),
        splits=list(manifest["splits"]),
        graphs=[graphed(registry, stored(path, graph), manifest["splits"]) for graph in manifest["graphs"]],
    )
    LOG.info("describe recorded exit graphs=%d seconds=%.3f", len(result["graphs"]), time.perf_counter() - started)
    return result


@timed("describe.table")
def table(header, rows):
    cells = [list(header)] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[at]) for row in cells) for at in range(len(header))]
    return ["    " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells]


@timed("describe.related")
def related(name, entry):
    return "  relation %s of the table %s: %s; %s; edges %d, facts %d, absent %d; runs %d, bounded %d%s" % (
        name,
        entry["table"],
        "ends " + ", ".join("%s to %s" % tuple(pair) for pair in entry["ends"]) if entry["ends"] else "no ends",
        "columns " + ", ".join(entry["columns"]) if entry["columns"] else "no columns",
        entry["edges"],
        entry["facts"],
        entry["absent"],
        entry["runs"],
        entry["bounded"],
        ", lasting %d to %d, mean %s" % (entry["least"], entry["most"], entry["mean"]) if entry["bounded"] else "",
    )


def decided(name, entry):
    started = time.perf_counter()
    lines = [
        "  decisions of %s: decisions %d, options %d, recorded %d, regenerated %d; %s"
        % (
            name,
            entry["decisions"],
            entry["options"],
            entry["recorded"],
            entry["regenerated"],
            ", ".join("%s %d" % pair for pair in entry["generators"].items()) or "no generator",
        )
    ]
    if entry["each"]:
        lines += table(
            ["task", "time", "options", "recorded", "regenerated"] + list(entry["generators"]),
            [
                [held["key"], "none" if held["time"] is None else held["time"]]
                + [held[part] for part in ("options", "recorded", "regenerated")]
                + list(held["generators"].values())
                for held in entry["each"]
            ],
        )
    LOG.debug("describe decided exit kind=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
    return lines


def graph_text(graph):
    started = time.perf_counter()
    times = graph["times"]
    lines = [
        "",
        "graph %s, %s, %s"
        % (
            graph["name"],
            "no unit" if graph["unit"] is None else "unit " + graph["unit"],
            "times %d from %d to %d" % (len(times), times[0], times[-1]) if times else "no time",
        ),
    ]
    lines += [
        "  kind %s: nodes %d, observations %d" % (name, entry["nodes"], entry["observations"])
        for name, entry in graph["kinds"].items()
    ]
    lines += [related(name, entry) for name, entry in graph["relations"].items()]
    if times:
        columns = [
            pair
            for name, entry in graph["kinds"].items()
            for pair in ((name + " nodes", entry["nodes_at"]), (name + " observations", entry["observations_at"]))
        ] + [(name + " edges", entry["edges_at"]) for name, entry in graph["relations"].items()]
        lines.append("  held at each time:")
        lines += table(
            ["time"] + [header for header, _ in columns],
            [[moment] + [values[at] for _, values in columns] for at, moment in enumerate(times)],
        )
    lines += [
        "  kind of task %s: tasks %d; %s"
        % (
            name,
            entry["tasks"],
            ", ".join(["%s %d" % pair for pair in entry["splits"].items()] + ["no split %d" % entry["unsplit"]]),
        )
        for name, entry in graph["tasks"].items()
    ]
    for name, entry in graph["decisions"].items():
        lines += decided(name, entry)
    LOG.debug("describe graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    held = record["time"]
    lines = [
        "truth graphs of the bank stored as %s, described at version %d" % (record["stored_sha256"], record["version"]),
        "time: %s, %s" % (MODELS[held["model"]], "units " + ", ".join(held["units"]) if held["units"] else "no unit"),
        "splits: %s" % (", ".join(record["splits"]) or "none"),
        READING,
    ]
    for graph in record["graphs"]:
        lines += graph_text(graph)
    LOG.debug("describe text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def manifested(path):
    started = time.perf_counter()
    found = Path(path) / "manifest.json"
    if not found.is_file():
        raise FileNotFoundError("%s holds no manifest.json; a description reads the folder of a bank" % path)
    result = admitted(path, read_json(found))
    LOG.debug("describe manifested exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return result


def kept(held, made, written):
    started = time.perf_counter()
    served = held.with_suffix(".json").exists()
    if served:
        record = read_json(held.with_suffix(".json"))
    else:
        record = made()
        output_path(held.parent).mkdir(parents=True, exist_ok=True)
        output_path(held.with_suffix(".txt")).write_text(written(record), encoding="utf-8")
        write_json(held.with_suffix(".json"), record)
    LOG.info("describe kept exit key=%s served=%s seconds=%.3f", held.name, served, time.perf_counter() - started)
    return record


def describe(path):
    started = time.perf_counter()
    manifest = manifested(path)
    record = kept(stem(path, manifest), lambda: recorded(path, manifest), text)
    LOG.info("describe exit path=%s seconds=%.3f", path, time.perf_counter() - started)
    return record
