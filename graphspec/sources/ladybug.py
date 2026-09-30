import argparse
import logging
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.build.output import Stage
from graphspec.infra.database import columns, connect, connections, copy, query, tables
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.references import SORTS
from graphspec.schema.schema import EDGE, LINK, OBSERVED, TARGET, TASK, VALUE
from graphspec.schema.time import UNITED

LOG = logging.getLogger(__name__)


def valued(holder, names):
    return "".join(", %s.`%s` AS `%s%s`" % (holder, name, VALUE, name) for name in names)


@timed("graphdb.stamped")
def stamped(holder, names):
    return "".join(", %s.`%s` AS `%s`" % (holder, name, name) for name in names)


@timed("graphdb.asked")
def asked(stated):
    held = dict(id="id", kind="kind", split="split", weight="weight")
    if stated.model != "none":
        held["time"] = "time"
    if stated.observers:
        held["observer"] = "observer"
    if stated.tasks["grouping"] is not None:
        held[stated.tasks["grouping"]] = "period"
    held[stated.tasks["key"]] = "key"
    return held


@timed("graphdb.aimed")
def aimed(stated):
    return (
        "t.id AS id, t.y AS y"
        + ("" if stated.model == "none" else ", t.known AS known")
        + (", t.tau AS tau, t.series AS series" if stated.aim() == "path" else "")
        + ", q.id AS owner"
    )


@timed("graphdb.placed")
def placed(found, ids, name, where):
    wanted = found.column(name).to_numpy().astype(np.int64)
    position = np.searchsorted(ids, wanted)
    if np.any(position >= len(ids)) or not np.array_equal(ids[np.minimum(position, len(ids) - 1)], wanted):
        raise ValueError("%s names tasks the source does not hold" % where)
    return found.drop_columns([name]).append_column("task", pa.array(position, pa.int64()))


class Database:
    def __init__(self, path, stated, buffer_pool_size=4 << 30):
        started = time.perf_counter()
        if stated.sets or stated.holds:
            raise ValueError(
                "a database source is read with no edges joining sets and no held graphs, and the schema states "
                "the kinds of set edge %s and the kinds that hold graphs %s; stage the source instead"
                % (", ".join(stated.sets) or "none", ", ".join(stated.holds) or "none")
            )
        self.location, self.stated = str(path), stated
        self.store, self.connection = connect(path, read_only=True, buffer_pool_size=buffer_pool_size)
        try:
            self.kinds = {row["name"]: row["type"] for row in tables(self.connection)}
            self.nodes = tuple(stated.nodes)
            stated.conform(self)
        except Exception:
            self.close()
            raise
        LOG.info(
            "database exit seconds=%.3f nodes=%d edges=%d",
            time.perf_counter() - started,
            len(self.nodes),
            len(stated.edges),
        )

    def pairs(self, table):
        return connections(self.connection, table)

    def columns(self, table):
        return columns(self.connection, table)

    def close(self):
        try:
            self.connection.close()
        finally:
            self.store.close()

    def reader(self):
        return Reader(self)

    def graphs(self):
        return query(self.connection, "MATCH (g:graph) RETURN g.id AS id, g.name AS name ORDER BY id")

    def asked(self):
        started = time.perf_counter()
        result = query(
            self.connection,
            "MATCH (q:task) RETURN q.graph AS graph%s ORDER BY id"
            % "".join(", q.`%s` AS `%s`" % pair for pair in asked(self.stated).items()),
        )
        LOG.debug("asked exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
        return result

    def aims(self):
        started = time.perf_counter()
        found = query(
            self.connection,
            "MATCH (t:`%s`)-[:`%s`]->(q:task) RETURN %s ORDER BY id" % (TARGET, LINK, aimed(self.stated)),
        )
        ids = self.asked().column("id").to_numpy().astype(np.int64)
        result = placed(found, ids, "owner", "a target")
        LOG.debug("aims exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
        return result


class Reader:
    def __init__(self, source):
        started = time.perf_counter()
        self.source, self.connection, stated = source, source.connection, source.stated
        self.statements = dict(
            graph="MATCH (g:graph) WHERE g.id = $graph RETURN g.name AS name"
            + (", g.unit AS unit" if stated.model in UNITED else "")
            + valued("g", stated.graph)
        )
        self.statements["tasks"] = "MATCH (q:task) WHERE q.graph = $graph RETURN %s%s%s ORDER BY id" % (
            ", ".join("q.`%s` AS `%s`" % pair for pair in asked(stated).items()),
            valued("q", stated.asked_columns()),
            ", q.signal AS signal, q.signal_observed AS signal_observed" if stated.tasks["signals"] else "",
        )
        for name, entry in stated.tasks["references"].items():
            stored = SORTS.named(entry["sort"]).columns()
            if stored is not None:
                self.statements["reference:" + name] = (
                    "MATCH (q:task)-[r:`%s%s`]->(n) WHERE q.graph = $graph "
                    "RETURN q.id AS owner, n.entity AS source%s"
                    % (TASK, name, "".join(", r.`%s` AS `%s`" % (held, held) for held in stored))
                )
        self.statements["targets"] = (
            "MATCH (t:`%s`)-[:`%s`]->(q:task) WHERE q.graph = $graph RETURN %s ORDER BY id"
            % (TARGET, LINK, aimed(stated))
        )
        for kind, held in stated.nodes.items():
            self.statements["node:" + kind] = (
                "MATCH (n:`%s`) WHERE n.graph = $graph RETURN n.entity AS entity, n.key AS key" % kind
            )
            self.statements["observed:" + kind] = (
                "MATCH (e:`%s`)-[r:`%s%s`]->(o:observation) WHERE e.graph = $graph "
                "RETURN e.entity AS entity%s%s"
                % (kind, OBSERVED, kind, stamped("o", stated.stamps()), valued("r", held))
            )
        for table, entry in stated.edges.items():
            self.statements["edge:" + table] = (
                "MATCH (a)-[r:`%s%s`]->(b) WHERE a.graph = $graph RETURN a.entity AS source, b.entity AS target, "
                "r.present AS present%s%s%s"
                % (
                    EDGE,
                    table,
                    stamped("r", stated.stamps()),
                    "" if entry["column"] is None else ", r.`%s` AS relation" % entry["column"],
                    valued("r", entry["columns"]),
                )
            )
        LOG.info(
            "reader exit seconds=%.3f edges=%d",
            time.perf_counter() - started,
            len(stated.edges),
        )

    def run(self, name, graph):
        return query(self.connection, self.statements[name], dict(graph=graph))

    def references(self, index, nodes, ids):
        started = time.perf_counter()
        found = {}
        for entity, key, kind in zip(*(nodes.column(name).to_pylist() for name in ("entity", "key", "kind"))):
            found[(kind, key)] = entity
        result = {}
        for name, entry in self.source.stated.tasks["references"].items():
            stored = SORTS.named(entry["sort"]).columns()
            if stored is None:
                continue
            links = placed(
                self.run("reference:" + name, index),
                ids,
                "owner",
                "the reference %s of graph %d" % (name, index),
            )
            columns = dict(task=links.column("task"), source=links.column("source").cast(pa.int32()))
            if "relation" in stored:
                ends = list(zip(links.column("to_kind").to_pylist(), links.column("to_key").to_pylist()))
                strangers = sorted({end for end in ends if end[0] is not None and end not in found})
                if strangers:
                    raise ValueError(
                        "the reference %s of graph %d names nodes the graph does not hold: %s"
                        % (name, index, ", ".join("%s %s" % end for end in strangers))
                    )
                columns["relation"] = links.column("relation")
                columns["target"] = pa.array(
                    [None if end[0] is None else found[end] for end in ends], pa.int32()
                )
            if "place" in stored:
                columns["place"] = links.column("place")
            result[name] = pa.table(columns)
        LOG.debug("read references exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
        return result

    def graph(self, index):
        started = time.perf_counter()
        stated = self.source.stated
        parts = []
        for kind in stated.nodes:
            part = self.run("node:" + kind, index)
            parts.append(part.append_column("kind", pa.array([kind] * part.num_rows, pa.string())))
        nodes, tasks = pa.concat_tables(parts), self.run("tasks", index)
        ids = tasks.column("id").to_numpy().astype(np.int64)
        tables = dict(
            graph=self.run("graph", index),
            nodes=nodes,
            observations={kind: self.run("observed:" + kind, index) for kind in stated.nodes},
            edges={table: self.run("edge:" + table, index) for table in stated.edges},
            sets={},
            held={},
            tasks=tasks,
            references=self.references(index, nodes, ids),
            targets=placed(self.run("targets", index), ids, "owner", "a target of graph %d" % index),
        )
        LOG.info(
            "read graph exit graph=%d seconds=%.3f observations=%d",
            index,
            time.perf_counter() - started,
            sum(table.num_rows for table in tables["observations"].values()),
        )
        return tables


TYPES = {
    pa.int64(): "INT64",
    pa.int32(): "INT32",
    pa.float32(): "FLOAT",
    pa.float64(): "DOUBLE",
    pa.bool_(): "BOOLEAN",
    pa.string(): "STRING",
}


def typename(dtype):
    if pa.types.is_fixed_size_list(dtype) and dtype.value_type == pa.float32():
        return "FLOAT[%d]" % dtype.list_size
    if pa.types.is_list(dtype) and dtype.value_type == pa.string():
        return "STRING[]"
    if dtype not in TYPES:
        raise ValueError("unsupported graph property type: " + str(dtype))
    return TYPES[dtype]


def identifier(name):
    return "`" + name.replace("`", "``") + "`"


def declarations(stage):
    tables = {}
    for (name, source, target), paths in stage.files.items():
        schema = pq.read_schema(paths[0])
        properties = [
            (field.name, typename(field.type)) for field in schema if field.name not in ("from", "to")
        ]
        entry = tables.setdefault(name, dict(node=source is None, properties=properties, pairs=[]))
        if entry["properties"] != properties or entry["node"] != (source is None):
            raise ValueError("inconsistent table schema for " + name)
        if source is not None:
            entry["pairs"].append((source, target))
    for name, entry in sorted(tables.items(), key=lambda item: not item[1]["node"]):
        fields = [
            identifier(key) + " " + dtype + (" PRIMARY KEY" if entry["node"] and key == "id" else "")
            for key, dtype in entry["properties"]
        ]
        if not entry["node"]:
            fields = [
                "FROM %s TO %s" % (identifier(source), identifier(target))
                for source, target in entry["pairs"]
            ] + fields
        yield "CREATE %s TABLE %s(%s)" % (
            "NODE" if entry["node"] else "REL",
            identifier(name),
            ", ".join(fields),
        )


@exclusive
def export(staging, output, buffer_pool_size=1 << 30):
    started = time.perf_counter()
    output = output_path(output)
    if output.exists():
        raise FileExistsError("database exists: " + str(output))
    stage = Stage.open(staging)
    statements = list(declarations(stage))
    output.parent.mkdir(parents=True, exist_ok=True)
    store, connection = connect(output, read_only=False, buffer_pool_size=buffer_pool_size)
    try:
        for statement in statements:
            connection.execute(statement)
        copy(connection, stage)
    finally:
        try:
            connection.close()
        finally:
            store.close()
    LOG.info("database export exit tables=%d seconds=%.3f", len(statements), time.perf_counter() - started)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.staging, args.output))


if __name__ == "__main__":
    main()
