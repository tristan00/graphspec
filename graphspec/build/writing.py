import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.budgets import GraphCapacityError
from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.schema import EDGE, FACT_COLUMNS, OBSERVED, VALUE, WRITTEN, Schema
from graphspec.schema.time import UNITED

from . import records
from .output import Stage
from .planning import stated

LOG = logging.getLogger(__name__)
VERSION = 1
IDS = 32
ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@timed("writing.fingerprint")
def fingerprint(value):
    digest = hashlib.sha256()
    for part in ENCODER.iterencode(value):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


@timed("writing.recorded")
def recorded(path, value):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for part in ENCODER.iterencode(value):
            stream.write(part)
        stream.write("\n")
    return fingerprint(value)


@timed("writing.canonical_records")
def canonical_records(case):
    schema = dict(case["schema"])
    for name in ("nodes", "sets", "edges"):
        schema[name] = dict(sorted(schema[name].items()))
    schema["tasks"] = dict(schema["tasks"], kinds=dict(sorted(schema["tasks"]["kinds"].items())))
    schema = Schema(schema).mapping()
    graphs = []
    for graph in case["graphs"]:
        held = dict(graph)
        held["nodes"] = {kind: graph["nodes"][kind] for kind in Schema(schema).entities()}
        graphs.append(held)
    return dict(schema=schema, graphs=graphs)


@timed("writing.size")
def size(value):
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(size(key) + size(found) for key, found in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(size(found) for found in value)
    return sys.getsizeof(value)


@timed("writing.row_at")
def row_at(columns, at):
    return {name: {key: values[at] for key, values in value.items()} if isinstance(value, dict) else value[at]
            for name, value in columns.items()}


@timed("writing.preflight")
def preflight(case, plan):
    limit = plan.worker_bytes // 8
    for graph in case["graphs"]:
        capacity(dict(name=graph["name"], unit=graph["unit"], columns=graph["columns"]), limit, "graph metadata")
        for kind, keys in graph["nodes"].items():
            for key in keys:
                capacity(dict(key=key), limit, "nodes of " + kind)
        for field in ("values", "edges"):
            for statement in graph[field]:
                for at in range(len(next(iter(statement["rows"].values())))):
                    capacity(row_at(statement["rows"], at), limit, field + " of " + graph["name"])
        for rows in graph.get("sets", {}).values():
            for at in range(len(rows["set"])):
                capacity(row_at(rows, at), limit, "set members of " + graph["name"])
        for at in range(len(graph["tasks"]["key"])):
            capacity(row_at(graph["tasks"], at), limit, "tasks of " + graph["name"])


@timed("writing.capacity")
def capacity(row, budget, where):
    amount = size(row) * 4
    if amount > budget:
        raise GraphCapacityError("%s needs %d working bytes, above the build chunk budget %d" % (where, amount, budget))
    return amount


@timed("writing.batches")
def batches(rows, budget, where):
    batch, used = [], 0
    for number, row in enumerate(rows):
        amount = size(row) * 4
        if amount > budget:
            raise GraphCapacityError("%s row %d needs %d working bytes, above the build chunk budget %d" %
                                     (where, number, amount, budget))
        if batch and used + amount > budget:
            yield batch
            batch, used = [], 0
        batch.append(row)
        used += amount
    if batch:
        yield batch


@timed("writing.columned")
def columned(rows):
    if not rows:
        raise ValueError("a record batch holds at least one row")
    return {name: {key: [row[name][key] for row in rows] for key in rows[0][name]}
            if name == "references" and isinstance(rows[0][name], dict) else [row[name] for row in rows] for name in rows[0]}


class BlockCounter:
    @timed("writing.counter")
    def __init__(self, index, counters):
        self.index, self.next = index, {} if counters is None else dict(counters)

    @timed("writing.counter.take")
    def take(self, table, count):
        start = self.next.get(table, 0)
        if start + count >= 1 << IDS:
            raise GraphCapacityError("the graph has exhausted its row IDs in table " + table)
        self.next[table] = start + count
        return (self.index << IDS) + np.arange(start, start + count, dtype=np.int64)


class KindKeys:
    @timed("writing.kind_keys")
    def __init__(self, connection, kind):
        self.connection, self.kind = connection, kind

    @timed("writing.kind_keys.contains")
    def __contains__(self, key):
        return self.connection.execute("SELECT 1 FROM nodes WHERE kind=? AND key=?", (self.kind, key)).fetchone() is not None

    @timed("writing.kind_keys.get")
    def __getitem__(self, key):
        found = self.connection.execute("SELECT id FROM nodes WHERE kind=? AND key=?", (self.kind, key)).fetchone()
        if found is None:
            raise KeyError("the node %s:%s is not declared" % (self.kind, key))
        return found[0]


class TaskKeys:
    @timed("writing.task_keys")
    def __init__(self, connection):
        self.connection = connection

    @timed("writing.task_keys.contains")
    def __contains__(self, key):
        return self.connection.execute("SELECT 1 FROM tasks WHERE key=?", (key,)).fetchone() is not None

    @timed("writing.task_keys.get")
    def __getitem__(self, key):
        found = self.connection.execute("SELECT id FROM tasks WHERE key=?", (key,)).fetchone()
        if found is None:
            raise KeyError("the task %s is not declared" % key)
        return found[0]


class Keys:
    @timed("writing.keys")
    def __init__(self, path, schema, budget, previous):
        self.path, self.schema = Path(path), schema
        self.connection = sqlite3.connect(self.path)
        if previous is not None:
            with sqlite3.connect("file:%s?mode=ro" % Path(previous).as_posix(), uri=True) as old:
                old.backup(self.connection)
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA cache_size=-%d" % max(1, budget // 4096))
        self.connection.execute("CREATE TABLE IF NOT EXISTS nodes(kind TEXT,key TEXT,id INTEGER,entity INTEGER,PRIMARY KEY(kind,key))")
        self.connection.execute("CREATE TABLE IF NOT EXISTS tasks(key INTEGER PRIMARY KEY,id INTEGER)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS vocabulary(name TEXT,value TEXT,PRIMARY KEY(name,value))")
        self.tasks = TaskKeys(self.connection)

    @timed("writing.keys.kind")
    def __getitem__(self, kind):
        if kind not in self.schema.entities():
            raise KeyError("undeclared kind " + kind)
        return KindKeys(self.connection, kind)

    @timed("writing.keys.found")
    def found(self, kind, keys):
        return pa.array([self[kind][key] for key in keys], pa.int64())

    @timed("writing.keys.vocabulary")
    def vocabulary(self, columns):
        for name in self.schema.symbolic():
            if name in columns:
                self.connection.executemany("INSERT OR IGNORE INTO vocabulary VALUES(?,?)",
                                            ((name, value) for value in columns[name] if value is not None))

    @timed("writing.keys.labels")
    def labels(self, name, values):
        self.connection.executemany("INSERT OR IGNORE INTO vocabulary VALUES(?,?)",
                                    ((name, value) for value in values if value is not None))

    @timed("writing.keys.vocabularies")
    def vocabularies(self, budget):
        required = self.connection.execute(
            "SELECT COALESCE(SUM(length(CAST(name AS BLOB))+length(CAST(value AS BLOB))+256),0)*4 FROM vocabulary"
        ).fetchone()[0]
        if required > budget:
            raise GraphCapacityError("fitted vocabularies need %d reserved bytes, above the metadata budget %d" % (required, budget))
        result = {name: [] for name in self.schema.symbolic()}
        for name, value in self.connection.execute("SELECT name,value FROM vocabulary ORDER BY name,value"):
            result.setdefault(name, []).append(value)
        return result

    @timed("writing.keys.close")
    def close(self):
        self.connection.commit()
        self.connection.close()


@timed("writing.empty_tables")
def empty_tables(schema):
    result = {}
    for name, entry in schema.tables().items():
        required = {}
        for column, written in entry["required"].items():
            written = written if isinstance(written, str) else written[0]
            required[column] = pa.list_(pa.float32(), int(written[6:-1])) if written.startswith("FLOAT[") else WRITTEN[written]
        required.update({column: schema.kind(column).written for column in entry["columns"]})
        if name.startswith(OBSERVED):
            required[FACT_COLUMNS] = pa.list_(pa.string())
        pairs = entry["pairs"] if entry["kind"] == "REL" else ((None, None),)
        for source, target in pairs:
            fields = dict({"from": pa.int64(), "to": pa.int64()} if source is not None else {}, **required)
            result[name, source, target] = pa.table({column: pa.array([], kind) for column, kind in fields.items()})
    return result


@timed("writing.written")
def written(stage, tables, limit, keys):
    total = sum(table.nbytes for table in tables.values())
    if total > limit:
        raise GraphCapacityError("a converted batch needs %d bytes, above its build worker budget %d" % (total, limit))
    normalized = {}
    declared = keys.schema.tables()
    for identity, table in tables.items():
        expected = stage.writers[identity].schema
        if set(table.column_names) != set(expected.names):
            raise ValueError("the converted columns do not match the declared table %s" % (identity,))
        table = table.select(expected.names)
        keys.vocabulary({name: table.column(name).to_pylist() for name in declared[identity[0]]["columns"]})
        if identity[0].startswith(EDGE):
            name = identity[0][len(EDGE):]
            relation = keys.schema.edges[name]["column"]
            keys.labels("relations:" + name, [name] if relation is None else table.column(relation).to_pylist())
        if "role" in table.column_names:
            keys.labels("roles:" + identity[0], table.column("role").to_pylist())
        normalized[identity] = table
    stage.write(normalized)


@timed("writing.statement_rows")
def statement_rows(statement, keys):
    rows = statement["rows"]
    count = len(next(iter(rows.values())))
    if any(len(values) != count for values in rows.values()):
        raise ValueError("a fact statement has columns of different lengths")
    if statement["form"] != "change":
        for at in range(count):
            yield row_at(rows, at)
        return
    fields = statement["parameters"]["keys"]
    keys.connection.execute("DROP TABLE IF EXISTS changes")
    keys.connection.execute("CREATE TEMP TABLE changes(identity TEXT,moment INTEGER,position INTEGER)")
    keys.connection.executemany("INSERT INTO changes VALUES(?,?,?)", (
        (json.dumps([rows[name][at] for name in fields], separators=(",", ":")), rows["at"][at], at)
        for at in range(count)
    ))
    repeated = keys.connection.execute("SELECT moment FROM changes GROUP BY identity,moment HAVING COUNT(*)>1 LIMIT 1").fetchone()
    if repeated is not None:
        raise ValueError("the form change meets two changes of one thing at the time %d" % repeated[0])
    for at, until in keys.connection.execute(
        "SELECT position,LEAD(moment) OVER(PARTITION BY identity ORDER BY moment) FROM changes ORDER BY identity,moment"
    ):
        row = row_at(rows, at)
        row["since"], row["until"] = row.pop("at"), until
        yield row


@timed("writing.statement_batches")
def statement_batches(statement, keys, budget, where):
    form = "state" if statement["form"] == "change" else statement["form"]
    parameters = {} if statement["form"] == "change" else statement["parameters"]
    for batch in batches(statement_rows(statement, keys), budget, where):
        yield dict(statement, form=form, parameters=parameters, rows=columned(batch))


@timed("writing.write_graph")
def write_graph(schema_mapping, graph, index, directory, plan_mapping, counters=None, keys=None):
    started = time.perf_counter()
    plan, schema = stated(plan_mapping), Schema(schema_mapping)
    counter = BlockCounter(index, counters)
    directory = output_path(directory)
    counts = dict(nodes={}, observations={kind: 0 for kind in schema.entities()},
                  edges={name: 0 for name in schema.edges},
                  tasks={kind: dict(splits={}, without_split=0) for kind in schema.tasks["kinds"]}, skipped_references=0)
    directory.mkdir(parents=True)
    preparation = directory / "prepared"
    with Stage(preparation) as stage:
        lookup = Keys(directory / "keys.sqlite3", schema, plan.worker_bytes, keys)
        try:
            empty = empty_tables(schema)
            stage.write(empty)
            lookup.labels("kinds", schema.entities())
            for kind, entry in schema.tasks["kinds"].items():
                if entry["target"] is not None:
                    lookup.labels("targets:" + kind, entry["target"]["series"])
            if counters is None:
                described = dict(id=pa.array([index], pa.int32()), name=pa.array([graph["name"]], pa.string()))
                if schema.model in UNITED:
                    described["unit"] = pa.array([graph["unit"]], pa.string())
                described.update(records.feature_table(schema, {name: [value] for name, value in graph["columns"].items()}).to_pydict())
                written(stage, {("graph", None, None): records.table_of(schema, described)}, plan.worker_bytes // 4, lookup)
            found = lookup.connection.execute("SELECT MAX(entity) FROM nodes").fetchone()[0]
            entity = 0 if found is None else found + 1
            for kind in schema.entities():
                new = (dict(key=key) for key in graph["nodes"][kind] if key not in lookup[kind])
                for batch in batches(new, plan.worker_bytes // 8, "nodes of " + kind):
                    names = [row["key"] for row in batch]
                    ids = counter.take(kind, len(names))
                    positions = list(range(entity, entity + len(names)))
                    lookup.connection.executemany("INSERT INTO nodes VALUES(?,?,?,?)", zip([kind] * len(names), names, ids.tolist(), positions))
                    table = pa.table(dict(id=pa.array(ids), graph=pa.array([index] * len(names), pa.int32()),
                                          entity=pa.array(positions, pa.int32()), key=pa.array(names, pa.string())))
                    written(stage, {(kind, None, None): table}, plan.worker_bytes // 4, lookup)
                    entity += len(names)
                counts["nodes"][kind] = len(graph["nodes"][kind])
            stamps = list(schema.stamps())
            for statement in graph["values"]:
                kind, columns = statement["kind"], schema.entities()[statement["kind"]]
                for part in statement_batches(statement, lookup, plan.worker_bytes // 8, "observations of " + kind):
                    rows = records.observations(schema, [part], columns, "observations of " + kind)
                    ids = counter.take("observation", rows.num_rows)
                    tables = {
                        ("observation", None, None): pa.table(dict(id=pa.array(ids), **{name: rows.column(name) for name in stamps})),
                        (OBSERVED + kind, kind, "observation"): pa.table({"from": lookup.found(kind, rows.column("key").to_pylist()),
                            "to": pa.array(ids), **{name: rows.column(VALUE + name) for name in columns}, FACT_COLUMNS: rows.column(FACT_COLUMNS)}),
                    }
                    written(stage, tables, plan.worker_bytes // 4, lookup)
                    counts["observations"][kind] = counts["observations"].get(kind, 0) + rows.num_rows
            for statement in graph["edges"]:
                name, ends = statement["table"], tuple(statement["ends"])
                entry = schema.edges[name]
                given = ("source", "target", "present") + (() if entry["column"] is None else ("relation",))
                for part in statement_batches(statement, lookup, plan.worker_bytes // 8, "edges of " + name):
                    rows = records.facts(schema, part, given, entry["columns"], "edges of " + name)
                    values = {"from": lookup.found(ends[0], rows.column("source").to_pylist()),
                              "to": lookup.found(ends[1], rows.column("target").to_pylist()), "present": rows.column("present")}
                    values.update({field: rows.column(field) for field in stamps})
                    values.update({field: rows.column(VALUE + field) for field in entry["columns"]})
                    if entry["column"] is not None:
                        values[entry["column"]] = rows.column("relation")
                    written(stage, {(EDGE + name,) + ends: pa.table(values)}, plan.worker_bytes // 4, lookup)
                    counts["edges"][name] = counts["edges"].get(name, 0) + rows.num_rows
            for kind in schema.sets:
                member_rows = graph["sets"][kind]
                for batch in batches((row_at(member_rows, at) for at in range(len(member_rows["set"]))), plan.worker_bytes // 8, "members of " + kind):
                    written(stage, records.case_members(schema, kind, columned(batch), lookup, lookup.found, graph["name"]), plan.worker_bytes // 4, lookup)
            if schema.holds:
                raise ValueError("held graph output requires an owner-batched mapping")
            tasks = graph["tasks"]
            lookup.connection.execute("CREATE TEMP TABLE fresh_tasks(id INTEGER PRIMARY KEY)")
            for key in tasks["key"]:
                if key not in lookup.tasks:
                    value = int(counter.take("task", 1)[0])
                    lookup.connection.execute("INSERT INTO tasks VALUES(?,?)", (key, value))
                    lookup.connection.execute("INSERT INTO fresh_tasks VALUES(?)", (value,))
            for batch in batches((row_at(tasks, at) for at in range(len(tasks["key"]))), plan.worker_bytes // 8, "tasks of " + graph["name"]):
                rows = columned(batch)
                ids = np.asarray([lookup.tasks[key] for key in rows["key"]], np.int64)
                tables = records.case_tasks(schema, index, rows, counter, lookup, graph["name"], ids=ids, task_ids=lookup.tasks)
                tables = records.fresh_decisions(tables, [value for value in ids.tolist()
                    if lookup.connection.execute("SELECT 1 FROM fresh_tasks WHERE id=?", (value,)).fetchone() is not None])
                for identity, table in list(tables.items()):
                    if identity[0] == "task":
                        table = table.filter(pa.array([lookup.connection.execute("SELECT 1 FROM fresh_tasks WHERE id=?", (value,)).fetchone() is not None
                                                      for value in table.column("id").to_pylist()], pa.bool_()))
                    elif identity[1] == "task" and identity[2] in schema.entities():
                        table = table.filter(pa.array([lookup.connection.execute("SELECT 1 FROM fresh_tasks WHERE id=?", (value,)).fetchone() is not None
                                                      for value in table.column("from").to_pylist()], pa.bool_()))
                    tables[identity] = table
                written(stage, tables, plan.worker_bytes // 4, lookup)
                for kind, split in zip(rows["kind"], rows["split"]):
                    held = counts["tasks"][kind]
                    if split is None:
                        held["without_split"] += 1
                    else:
                        held["splits"][split] = held["splits"].get(split, 0) + 1
            vocabulary = lookup.vocabularies(plan.worker_bytes // 4)
            source = recorded(directory / "records.json", graph)
        finally:
            lookup.close()
    if plan.shards == 1:
        Stage.merge(directory, [preparation])
    else:
        segment(directory, preparation, plan)
    result = dict(index=index, name=graph["name"], unit=graph["unit"], directory=str(directory),
                  stage_directory=str(directory),
                  records=str(directory / "records.json"), key_file=str(directory / "keys.sqlite3"),
                  counters=counter.next, source_fingerprint=source, counts=counts, vocabularies=vocabulary,
                  records_sha256=sha256_file(directory / "records.json"), keys_sha256=sha256_file(directory / "keys.sqlite3"),
                  tables=read_json(directory / "stage.json"))
    LOG.info("write graph exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)
    return result


@timed("writing.verified")
def verified(directory, expected):
    directory = Path(directory)
    manifest = read_json(directory / "build.json")
    if manifest["version"] != VERSION or manifest["source_fingerprint"] != expected:
        raise ValueError("staged output does not match the requested source or build format")
    if sha256_file(directory / "stage.json") != manifest["stage_sha256"]:
        raise ValueError("staged table manifest changed")
    for graph in manifest["graphs"]:
        for field, digest in (("records", "records_sha256"), ("key_file", "keys_sha256")):
            if sha256_file(directory / graph[field]) != graph[digest]:
                raise ValueError("staged metadata changed: " + graph[field])
    for entry in read_json(directory / "stage.json"):
        for name, digest in zip(entry["files"], entry["sha256"]):
            if sha256_file(directory / name) != digest:
                raise ValueError("staged file changed: " + name)
    return manifest


@exclusive
@timed("writing.stage")
def stage(case, output, name, plan, presence, reuse=False):
    started = time.perf_counter()
    plan = stated(plan)
    if size(case["schema"]) * 4 > plan.worker_bytes // 4:
        raise GraphCapacityError("the resolved schema exceeds its build worker metadata budget")
    case = canonical_records(case)
    preflight(case, plan)
    output = output_path(output)
    directory, task = output / (name + ".staging"), output / (name + ".task.json")
    task_record = dict(name=name, staging=directory.name, schema=case["schema"])
    source = fingerprint(case)
    if directory.exists():
        if not reuse:
            raise FileExistsError("staging exists; request reuse explicitly: " + str(directory))
        verified(directory, source)
        if not task.exists():
            raise FileNotFoundError("the staged output has no task record: " + str(task))
        if read_json(task) != task_record:
            raise ValueError("staged task record changed: " + str(task))
        return task
    if task.exists():
        raise FileExistsError("task record exists: " + str(task))
    directory.mkdir(parents=True)
    graphs, metadata = [], 0
    for index, graph in enumerate(case["graphs"]):
        entry = write_graph(case["schema"], graph, index, directory / ("graph_%08d" % index), plan.mapping())
        metadata += size(entry) * 4
        if metadata > plan.memory_bytes // 4:
            raise GraphCapacityError("the graph manifests exceed the reserved build metadata budget")
        graphs.append(entry)
    Stage.merge(directory, [graph["directory"] for graph in graphs])
    for graph in graphs:
        for field in ("directory", "stage_directory", "records", "key_file"):
            graph[field] = Path(graph[field]).relative_to(directory).as_posix()
    if sum(size(graph["vocabularies"]) * 4 for graph in graphs) > plan.memory_bytes // 4:
        raise GraphCapacityError("the merged vocabularies exceed the reserved build metadata budget")
    vocabularies = {}
    for graph in graphs:
        for column, values in graph["vocabularies"].items():
            vocabularies.setdefault(column, set()).update(values)
    manifest = dict(version=VERSION, source_fingerprint=source, fingerprint=source, sources=dict(records=source),
                    stage_sha256=sha256_file(directory / "stage.json"),
                    declarations=case["schema"], schema=case["schema"], plan=plan.mapping(), graphs=graphs,
                    vocabularies={name: sorted(values) for name, values in sorted(vocabularies.items())},
                    skipped_references=0,
                    presence={label: {field: dict(sorted(rules.items())) for field, rules in sorted(fields.items())}
                              for label, fields in sorted(presence.items())})
    write_json(directory / "build.json", manifest)
    write_json(task, task_record)
    LOG.info("stage exit graphs=%d workers=%d seconds=%.4f", len(graphs), plan.workers, time.perf_counter() - started)
    return task


@timed("writing.segment")
def segment(directory, preparation, plan):
    source = Stage.open(preparation)
    if len(source.files) * plan.shards * 4096 > plan.memory_bytes // 4:
        raise GraphCapacityError("shard manifests exceed the reserved build metadata budget")
    jobs = ((source, directory / ("segment_%08d" % index), index, plan) for index in range(plan.shards))
    with ThreadPoolExecutor(max_workers=plan.workers) as pool:
        parts = list(pool.map(segment_job, jobs, buffersize=plan.workers))
    Stage.merge(directory, parts)
    resolved = preparation.resolve()
    if resolved.parent != directory.resolve() or resolved.name != "prepared":
        raise ValueError("the build preparation directory leaves its graph output")
    shutil.rmtree(resolved)


@timed("writing.segment_job")
def segment_job(job):
    source, directory, index, plan = job
    with Stage(directory) as sink:
        for key, paths in source.files.items():
            first = index * source.rows[key] // plan.shards
            last = (index + 1) * source.rows[key] // plan.shards
            position = 0
            for path in paths:
                with pq.ParquetFile(path) as file:
                    sink.write({key: file.schema_arrow.empty_table()})
                    for group in range(file.num_row_groups):
                        count = file.metadata.row_group(group).num_rows
                        start, stop = max(first, position), min(last, position + count)
                        if start < stop:
                            table = file.read_row_group(group)
                            if table.nbytes > plan.worker_bytes // 4:
                                raise GraphCapacityError("a prepared row group exceeds its build worker budget")
                            sink.write({key: table.slice(start - position, stop - start)})
                        position += count
    return directory
