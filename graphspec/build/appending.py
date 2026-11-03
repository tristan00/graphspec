import copy
import logging
import shutil
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.budgets import GraphCapacityError
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.timing import timed
from graphspec.schema.decisions import deciding
from graphspec.schema.schema import EDGE, FACT_COLUMNS, OBSERVED, VALUE, Schema
from graphspec.schema.time import ALWAYS

from .output import Stage
from .output import manifest as stage_manifest
from .planning import BuildPlan, stated
from .writing import (
    ENCODER,
    canonical_records,
    columned,
    fingerprint,
    preflight,
    recorded,
    row_at,
    size,
    verified,
    write_graph,
)

LOG = logging.getLogger(__name__)


@timed("appending.reserved")
def reserved(amount, limit, where):
    if amount > limit:
        raise GraphCapacityError("%s needs %d reserved working bytes, above its append budget %d" % (where, amount, limit))


@contextmanager
@timed("appending.unpublished")
def unpublished(directory, parent):
    try:
        yield
    except BaseException:
        discarded(directory, parent)
        raise


@timed("appending.discarded")
def discarded(directory, parent):
    resolved, root = parent.resolve(), directory.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("an unpublished append generation leaves its staging directory")
    if resolved.exists():
        shutil.rmtree(resolved)


@contextmanager
@timed("appending.publishing")
def publishing(directory, parent, entries, previous):
    try:
        yield
    except BaseException:
        if read_json(directory / "stage.json") != entries:
            write_json(directory / "stage.json", entries)
        if read_json(directory / "build.json") != previous:
            write_json(directory / "build.json", previous)
        discarded(directory, parent)
        raise


@timed("appending.start")
def start(statement, row):
    form = statement["form"]
    if form in (None, "state"):
        return ALWAYS if row.get("since") is None else row["since"]
    if form in ("change", "event"):
        return row["at"]
    if form == "signal":
        return statement["parameters"]["start"] + row["index"] * statement["parameters"]["stride"]
    raise ValueError("incremental output requires state, change, event or signal records")


@timed("appending.watermark")
def watermark(graph):
    found = [ALWAYS] + list(graph["tasks"].get("time", ()))
    found.extend(value for value in graph["tasks"]["known"] if value is not None)
    for field in ("values", "edges"):
        for statement in graph[field]:
            for at in range(len(next(iter(statement["rows"].values())))):
                row = row_at(statement["rows"], at)
                found.append(start(statement, row))
                if row.get("known") is not None:
                    found.append(row["known"])
    return max(found)


@timed("appending.header")
def header(statement):
    return ENCODER.encode(dict({name: value for name, value in statement.items() if name != "rows"},
                              columns=sorted(statement["rows"])))


@timed("appending.identity")
def identity(row):
    return ENCODER.encode({name: value for name, value in row.items() if name not in ("until", "replaced")})


@timed("appending.described")
def described(schema, statement, row):
    values = {name[len(VALUE):]: value for name, value in row.items() if name.startswith(VALUE)}
    result = dict(since=start(statement, row), seq=row.get("seq", 0), known=row.get("known"),
                  observers=row.get("observers"), values=values)
    if "kind" in statement:
        result.update(group="obs", kind=statement["kind"], key=row["key"],
                      scope=[name for name in schema.entities()[statement["kind"]] if name in values])
    else:
        result.update(group="fact", table=statement["table"], ends=list(statement["ends"]),
                      source=row["source"], target=row["target"], present=row["present"],
                      relation=row.get("relation", statement["table"]))
    return result


@timed("appending.statements")
def statements(schema, before, after, boundary):
    available, fresh, older = defaultdict(deque), [], {}
    for index, statement in enumerate(after):
        rows = [row_at(statement["rows"], at) for at in range(len(next(iter(statement["rows"].values()))))]
        fresh.append(rows)
        for at, row in enumerate(rows):
            available[header(statement), identity(row)].append((index, at))
    patches = []
    for statement in before:
        for at in range(len(next(iter(statement["rows"].values())))):
            old = row_at(statement["rows"], at)
            key = header(statement), identity(old)
            if not available[key]:
                raise ValueError("append removes or changes an existing fact")
            index, position = available[key].popleft()
            new = fresh[index][position]
            fresh[index][position] = None
            for name in ("until", "replaced"):
                if old.get(name) != new.get(name):
                    value = new.get(name)
                    if old.get(name) is not None or type(value) is not int or value <= boundary:
                        raise ValueError("append only closes an open %s after the previous watermark" % name)
                    patches.append(dict(described(schema, statement, old), column=name, value=value))
            if statement["form"] == "change":
                change = header(statement), ENCODER.encode([old[name] for name in statement["parameters"]["keys"]]), ENCODER.encode(old.get("observers"))
                if change not in older or old["at"] > older[change][1]["at"]:
                    older[change] = statement, old
    result, earliest = [], {}
    for statement, rows in zip(after, fresh):
        rows = [row for row in rows if row is not None]
        for row in rows:
            moment = start(statement, row)
            effective = max(moment, moment if row.get("known") is None else row["known"])
            if effective <= boundary:
                raise ValueError("new facts must be known after the previous watermark")
            if statement["form"] == "change":
                change = header(statement), ENCODER.encode([row[name] for name in statement["parameters"]["keys"]]), ENCODER.encode(row.get("observers"))
                if change in older:
                    if row["at"] <= older[change][1]["at"]:
                        raise ValueError("append cannot insert a change before an existing change")
                    earliest[change] = min(earliest.get(change, row["at"]), row["at"])
        if rows:
            result.append(dict(statement, rows=columned(rows)))
    for key, moment in earliest.items():
        statement, old = older[key]
        if moment <= boundary:
            raise ValueError("an appended change closes the prior interval after the previous watermark")
        patches.append(dict(described(schema, statement, old), column="until", value=moment))
    return result, patches


@timed("appending.target_delta")
def target_delta(schema, prior, row, boundary):
    target = schema.tasks["kinds"][row["kind"]]["target"]
    previous, current = prior["target"], row["target"]
    if deciding(target):
        if previous is None or current is None:
            raise ValueError("an appended decision must already state its immutable options and inputs")
        if any(previous[name] != current[name] for name in ("ended", "discount")):
            raise ValueError("append changes an existing decision input")
        if sorted(previous["concurrent"]) != sorted(current["concurrent"]):
            raise ValueError("append cannot change the simultaneous decisions of an existing task")
        if any(current["next"].get(actor) != key for actor, key in previous["next"].items()):
            raise ValueError("append cannot change an existing next-decision link")
        if len(previous["options"]) != len(current["options"]):
            raise ValueError("append changes an existing decision's options")
        result = copy.deepcopy(current)
        completed = False
        had_outcomes = any(option["outcome"] is not None for option in previous["options"])
        for old, new, emitted in zip(previous["options"], current["options"], result["options"]):
            if any(old[name] != new[name] for name in old if name != "outcome"):
                raise ValueError("append changes the inputs of an existing decision option")
            if old["outcome"] == new["outcome"]:
                emitted["outcome"] = None
            elif old["outcome"] is None and new["outcome"] is not None and not had_outcomes:
                completed = True
            else:
                raise ValueError("append only completes a decision whose outcomes were all unstated")
        if completed:
            if prior["known"] is not None or type(row["known"]) is not int or row["known"] <= boundary:
                raise ValueError("completed decision outcomes must become known after the previous watermark")
        elif prior["known"] != row["known"]:
            raise ValueError("append cannot change the knowledge time of an existing outcome")
        result["next"] = {actor: key for actor, key in current["next"].items() if actor not in previous["next"]}
        result["concurrent"] = []
        return result
    if previous is not None or prior["known"] is not None or current is None:
        raise ValueError("append only fills a previously unstated target and knowledge time")
    if type(row["known"]) is not int or row["known"] <= boundary:
        raise ValueError("a completed target must become known after the previous watermark")
    return current


@timed("appending.tasks")
def tasks(schema, before, after, boundary):
    old = {key: row_at(before, at) for at, key in enumerate(before["key"])}
    moments = dict(zip(after["key"], after.get("time", [ALWAYS] * len(after["key"]))))
    rows = []
    for at, key in enumerate(after["key"]):
        row = row_at(after, at)
        if key not in old:
            if "time" not in row or row["time"] <= boundary:
                raise ValueError("new tasks must occur after the previous watermark")
            rows.append(row)
            continue
        prior = old.pop(key)
        if prior == row:
            continue
        if any(prior[name] != row[name] for name in prior if name not in ("target", "known")):
            raise ValueError("append changes the inputs of an existing task")
        row["target"] = target_delta(schema, prior, row, boundary)
        if deciding(schema.tasks["kinds"][row["kind"]]["target"]):
            if any(key not in moments or moments[key] <= boundary for key in row["target"]["next"].values()):
                raise ValueError("an appended next-decision link must name a later task")
        rows.append(row)
    if old:
        raise ValueError("append removes an existing task")
    return columned(rows) if rows else {name: {key: [] for key in value} if isinstance(value, dict) else []
                                      for name, value in after.items()}


@timed("appending.difference")
def difference(schema, before, after):
    for name in ("name", "unit", "columns"):
        if before[name] != after[name]:
            raise ValueError("append changes the graph's " + name)
    for kind, keys in before["nodes"].items():
        if not set(keys).issubset(after["nodes"][kind]):
            raise ValueError("append removes an existing typed node key")
    boundary = watermark(before)
    delta, patches = dict(after), []
    for field in ("values", "edges"):
        delta[field], changed = statements(schema, before[field], after[field], boundary)
        patches.extend(changed)
    delta["tasks"] = tasks(schema, before["tasks"], after["tasks"], boundary)
    for field in ("sets", "held"):
        if field in after:
            if before[field] != after[field]:
                raise ValueError("append of %s contents requires a declared incremental handler" % field)
            delta[field] = {kind: {column: [] for column in rows} for kind, rows in after[field].items()}
    return delta, patches, boundary


@timed("appending.rows")
def file_rows(directory, entry, budget):
    for name in entry["files"]:
        with pq.ParquetFile(directory / name) as file:
            offset = 0
            for batch in file.iter_batches(batch_size=max(1, budget // 4096)):
                for at, row in enumerate(batch.to_pylist()):
                    yield name, offset + at, row
                offset += batch.num_rows


@timed("appending.matches")
def matches(row, values):
    return all(row.get(name) == value for name, value in values.items())


@timed("appending.stamps_match")
def stamps_match(row, patch):
    return all((ALWAYS if name == "since" and row[name] is None else row[name]) == patch[name]
               for name in ("since", "seq", "known", "observers") if name in row)


@timed("appending.located")
def located(directory, entries, schema, graph, patches, budget):
    if not patches:
        return []
    by_table = {(entry["table"], entry["source"], entry["target"]): entry for entry in entries}
    pending, edges = defaultdict(list), defaultdict(list)
    with sqlite3.connect("file:%s?mode=ro" % (directory / graph["key_file"]).as_posix(), uri=True) as keys:
        @timed("appending.node")
        def node(kind, key):
            found = keys.execute("SELECT id FROM nodes WHERE kind=? AND key=?", (kind, key)).fetchone()
            if found is None:
                raise ValueError("append patch names an unknown node")
            return found[0]

        for index, patch in enumerate(patches):
            values = {name: pa.scalar(value, schema.kind(name).written).as_py() for name, value in patch["values"].items()}
            if patch["group"] == "obs":
                values.update({FACT_COLUMNS: patch["scope"]})
                pending[patch["kind"], node(patch["kind"], patch["key"])].append((index, values))
            else:
                table = schema.edges[patch["table"]]
                values.update(present=patch["present"])
                if table["column"] is not None:
                    values[table["column"]] = patch["relation"]
                ends = tuple(patch["ends"])
                edges[(EDGE + patch["table"],) + ends].append((index, node(ends[0], patch["source"]), node(ends[1], patch["target"]), values))
    candidates = defaultdict(list)
    for kind in {kind for kind, _ in pending}:
        entry = by_table[OBSERVED + kind, kind, "observation"]
        for _, _, row in file_rows(directory, entry, budget):
            for index, values in pending.get((kind, row["from"]), ()):
                if matches(row, values):
                    candidates[row["to"]].append(index)
    found = defaultdict(list)
    observation = by_table["observation", None, None]
    for name, at, row in file_rows(directory, observation, budget):
        for index in candidates.get(row["id"], ()):
            patch = patches[index]
            if stamps_match(row, patch):
                found[index].append((observation, name, at))
    for key, requests in edges.items():
        entry = by_table[key]
        for name, at, row in file_rows(directory, entry, budget):
            for index, source, target, values in requests:
                patch = patches[index]
                if row["from"] == source and row["to"] == target and matches(row, values) and stamps_match(row, patch):
                    found[index].append((entry, name, at))
    result = []
    for index, patch in enumerate(patches):
        if len(found[index]) != 1:
            raise ValueError("append patch must identify one stored fact; found %d" % len(found[index]))
        entry, name, at = found[index][0]
        result.append(dict(table=entry["table"], source=entry["source"], target=entry["target"],
                           patch=dict(file=name, row=at, column=patch["column"], value=patch["value"])))
    return result


@exclusive
def append_stage(builder, task_path, plan):
    started = time.perf_counter()
    plan, task_path = stated(plan), Path(task_path).resolve()
    task = read_json(task_path)
    if "staging" not in task:
        raise ValueError("append requires an existing staged task")
    directory = (task_path.parent / task["staging"]).resolve()
    reserved((directory / "build.json").stat().st_size * 64, plan.memory_bytes // 2, "the append manifest")
    previous = read_json(directory / "build.json")
    verified(directory, previous["source_fingerprint"])
    if task["schema"] != previous["schema"]:
        raise ValueError("the staged task schema differs from its recorded declaration")
    case = canonical_records(builder.records(snapshot=False))
    required = size(case) * 4 + sum((directory / graph["records"]).stat().st_size * 64 for graph in previous["graphs"])
    reserved(required, plan.memory_bytes // 2, "cumulative append prefix verification")
    reserved(plan.workers * 4096, plan.memory_bytes // 2, "append writers")
    writing_plan = BuildPlan(plan.memory_bytes // 2, plan.workers, plan.shards)
    if case["schema"] != previous["declarations"]:
        raise ValueError("append keeps the declared schema unchanged")
    old_names = [graph["name"] for graph in previous["graphs"]]
    if [graph["name"] for graph in case["graphs"][:len(old_names)]] != old_names:
        raise ValueError("append preserves existing graphs in their original order")
    preflight(case, plan)
    source = fingerprint(case)
    if source == previous["source_fingerprint"]:
        raise ValueError("append requires new facts, tasks or target knowledge")
    schema, entries = Schema(case["schema"]), stage_manifest(directory)
    plans, changes = [], []
    generation = previous.get("generation", 0) + 1
    for index, graph in enumerate(case["graphs"]):
        prior = previous["graphs"][index] if index < len(old_names) else None
        if prior is None:
            delta, patches, boundary = graph, [], ALWAYS
        else:
            delta, patches, boundary = difference(schema, read_json(directory / prior["records"]), graph)
            changes.extend(located(directory, entries, schema, prior, patches, plan.worker_bytes))
        folder = directory / ("append_%08d" % generation) / ("graph_%08d" % index)
        plans.append(dict(index=index, name=graph["name"], previous_watermark=boundary, previous=prior,
                          delta=folder.relative_to(directory).as_posix(), patches=patches, records=delta))
    parent = directory / ("append_%08d" % generation)
    if parent.exists():
        raise FileExistsError("append generation already exists: " + str(parent))
    parent.mkdir()
    with unpublished(directory, parent):
        written = [write_graph(case["schema"], item["records"], item["index"], directory / item["delta"],
                               writing_plan.mapping(), None if item["previous"] is None else item["previous"]["counters"],
                               None if item["previous"] is None else directory / item["previous"]["key_file"])
                   for item in plans]
        graphs, shards = [], []
        for graph, item, entry in zip(case["graphs"], plans, written):
            prior = item["previous"]
            prefixes = [] if prior is None else prior.get("shards", [prior["directory"]])
            entry["shards"] = prefixes + [item["delta"]]
            shards.extend(directory / path for path in entry["shards"])
            cumulative = directory / item["delta"] / "cumulative.json"
            entry["source_fingerprint"] = recorded(cumulative, graph)
            entry["records"], entry["records_sha256"] = str(cumulative), sha256_file(cumulative)
            for field in ("directory", "stage_directory", "records", "key_file"):
                entry[field] = Path(entry[field]).relative_to(directory).as_posix()
            if prior is not None:
                for field in ("observations", "edges"):
                    for name, count in prior["counts"][field].items():
                        entry["counts"][field][name] = entry["counts"][field].get(name, 0) + count
            entry["counts"]["nodes"] = {kind: len(keys) for kind, keys in graph["nodes"].items()}
            entry["counts"]["tasks"] = {kind: dict(splits={}, without_split=0) for kind in schema.tasks["kinds"]}
            for kind, split in zip(graph["tasks"]["kind"], graph["tasks"]["split"]):
                counts = entry["counts"]["tasks"][kind]
                if split is None:
                    counts["without_split"] += 1
                else:
                    counts["splits"][split] = counts["splits"].get(split, 0) + 1
            graphs.append(entry)
            del item["records"]
        write_json(directory / ("build_%08d.json" % (generation - 1)), previous)
        write_json(directory / ("stage_%08d.json" % (generation - 1)), entries)
    with publishing(directory, parent, entries, previous):
        Stage.merge(directory, shards)
        merged = read_json(directory / "stage.json")
        by_table = {(entry["table"], entry["source"], entry["target"]): entry for entry in merged}
        for entry in entries:
            if entry.get("patches"):
                by_table[entry["table"], entry["source"], entry["target"]]["patches"] = copy.deepcopy(entry["patches"])
        for change in changes:
            target = by_table[change["table"], change["source"], change["target"]]
            patch = change["patch"]
            identity = patch["file"], patch["row"], patch["column"]
            target["patches"] = [old for old in target.get("patches", ())
                                 if (old["file"], old["row"], old["column"]) != identity] + [patch]
        write_json(directory / "stage.json", merged)
        vocabularies = defaultdict(set)
        for graph in graphs:
            for column, values in graph["vocabularies"].items():
                vocabularies[column].update(values)
        result = dict(previous, generation=generation, source_fingerprint=source, fingerprint=source,
                      stage_sha256=sha256_file(directory / "stage.json"),
                      sources=dict(records=source), graphs=graphs, plan=plan.mapping(),
                      vocabularies={name: sorted(values) for name, values in sorted(vocabularies.items())},
                      append=dict(previous_source=previous["source_fingerprint"], source=source, graphs=plans))
        write_json(directory / "build.json", result)
    LOG.info("append stage exit graphs=%d generation=%d seconds=%.4f", len(graphs), generation, time.perf_counter() - started)
    return task_path
