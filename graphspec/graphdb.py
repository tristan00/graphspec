import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.database import columns, connect, connections, query, tables

from .schema import EDGE, OBSERVED, QUERY, VALUE, Tables

LOG = logging.getLogger(__name__)


def valued(holder, names):
    return "".join(", %s.`%s` AS `%s%s`" % (holder, name, VALUE, name) for name in names)


class Database(Tables):
    def __init__(self, path, stated, buffer_pool_size=4 << 30):
        started = time.perf_counter()
        self.location, self.stated = str(path), stated
        self.store, self.connection = connect(path, read_only=True, buffer_pool_size=buffer_pool_size)
        try:
            self.kinds = {row["name"]: row["type"] for row in tables(self.connection)}
            self.nodes = tuple(stated.nodes)
            self.queries = stated.queries is not None
            self.signals = stated.queries["signals"] if self.queries else 0
            stated.conform(self)
        except Exception:
            self.close()
            raise
        LOG.info(
            "database exit seconds=%.3f nodes=%d edges=%d signals=%d queries=%s",
            time.perf_counter() - started,
            len(self.nodes),
            len(stated.edges),
            self.signals,
            self.queries,
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

    def reader(self, target):
        return Reader(self, target)

    def graphs(self):
        return query(self.connection, "MATCH (g:graph) RETURN g.id AS id, g.name AS name ORDER BY id")

    def categories(self):
        return (
            query(self.connection, "MATCH (q:query) RETURN DISTINCT q.category AS category ORDER BY category")
            .column("category")
            .to_pylist()
        )

    def query_index(self):
        return query(
            self.connection,
            "MATCH (q:query) RETURN q.graph AS graph, q.step AS step, q.`%s` AS period, q.`%s` AS key"
            % (self.stated.queries["grouping"], self.stated.queries["key"]),
        )

    def targets(self, level, target):
        if level == "node":
            return query(
                self.connection,
                "MATCH (t:`%s`)-[:`%s_of`]->(n) RETURN t.id AS id, t.y AS y, t.split AS split, t.step AS step, "
                "n.graph AS graph, n.entity AS entity, n.key AS node ORDER BY id" % (target, target),
            )
        if level in ("query", "path"):
            return query(
                self.connection,
                "MATCH (t:`%s`)-[:`%s_of`]->(q:query) RETURN t.id AS id, t.y AS y, t.split AS split, "
                "q.graph AS graph, q.step AS step%s ORDER BY id"
                % (target, target, ", t.tau AS tau, t.series AS series" if level == "path" else ""),
            )
        return query(
            self.connection,
            "MATCH (t:`%s`)-[:`%s_of`]->(g:graph) RETURN t.id AS id, t.y AS y, t.split AS split, g.id AS graph ORDER BY id"
            % (target, target),
        )


class Reader:
    def __init__(self, source, target):
        started = time.perf_counter()
        self.source, self.connection, stated = source, source.connection, source.stated
        self.level = source.target(target)
        self.statements = dict(
            graph="MATCH (g:graph) WHERE g.id = $graph RETURN g.name AS name" + valued("g", stated.graph)
        )
        if source.queries:
            asked = stated.queries
            self.statements["queries"] = (
                "MATCH %s WHERE q.graph = $graph "
                "RETURN q.step AS step, q.`%s` AS period, q.`%s` AS key, q.category AS category%s%s%s "
                "ORDER BY step"
                % (
                    ", ".join(
                        "(%s)-[:`%s%s`]->(r%d)" % ("q:query" if place == 0 else "q", QUERY, name, place)
                        for place, name in enumerate(asked["references"])
                    ),
                    asked["grouping"],
                    asked["key"],
                    "".join(
                        ", r%d.entity AS r%d" % (place, place) for place in range(len(asked["references"]))
                    ),
                    valued("q", asked["columns"]),
                    ", q.signal AS signal, q.signal_observed AS signal_observed" if source.signals else "",
                )
            )
        if self.level in ("query", "path"):
            self.statements["targets"] = (
                "MATCH (t:`%s`)-[:`%s_of`]->(q:query) WHERE q.graph = $graph "
                "RETURN q.step AS step, t.id AS id, t.y AS y, t.split AS split%s"
                % (target, target, ", t.tau AS tau" if self.level == "path" else "")
            )
        elif self.level == "graph":
            self.statements["targets"] = (
                "MATCH (t:`%s`)-[:`%s_of`]->(g:graph) WHERE g.id = $graph "
                "RETURN t.id AS id, t.y AS y, t.split AS split" % (target, target)
            )
        else:
            self.statements["targets"] = (
                "MATCH (t:`%s`)-[:`%s_of`]->(n) WHERE n.graph = $graph "
                "RETURN t.id AS id, t.y AS y, t.split AS split, t.step AS step, n.entity AS entity"
                % (target, target)
            )
        for kind, held in stated.nodes.items():
            self.statements["node:" + kind] = (
                "MATCH (n:`%s`) WHERE n.graph = $graph RETURN n.entity AS entity, n.key AS key" % kind
            )
            self.statements["observed:" + kind] = (
                "MATCH (e:`%s`)-[r:`%s%s`]->(o:observation) WHERE e.graph = $graph "
                "RETURN e.entity AS entity, o.seq AS seq, o.step AS step%s"
                % (kind, OBSERVED, kind, valued("r", held))
            )
        for table, entry in stated.edges.items():
            self.statements["edge:" + table] = (
                "MATCH (a)-[r:`%s%s`]->(b) WHERE a.graph = $graph RETURN a.entity AS source, b.entity AS target, "
                "r.seq AS seq, r.step AS step, r.present AS present%s%s"
                % (
                    EDGE,
                    table,
                    "" if entry["column"] is None else ", r.`%s` AS relation" % entry["column"],
                    valued("r", entry["columns"]),
                )
            )
        LOG.info(
            "reader exit seconds=%.3f level=%s edges=%d",
            time.perf_counter() - started,
            self.level,
            len(stated.edges),
        )

    def run(self, name, graph):
        return query(self.connection, self.statements[name], dict(graph=graph))

    def queries(self, index):
        started = time.perf_counter()
        found = self.run("queries", index)
        count = len(self.source.stated.queries["references"])
        references = np.stack(
            [found.column("r%d" % place).to_numpy().astype(np.int32) for place in range(count)], 1
        )
        result = found.drop_columns(["r%d" % place for place in range(count)]).append_column(
            "references", pa.FixedSizeListArray.from_arrays(pa.array(references.reshape(-1)), count)
        )
        LOG.debug("read queries exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
        return result

    def graph(self, index):
        started = time.perf_counter()
        stated = self.source.stated
        parts = []
        for kind in stated.nodes:
            part = self.run("node:" + kind, index)
            parts.append(part.append_column("kind", pa.array([kind] * part.num_rows, pa.string())))
        tables = dict(
            graph=self.run("graph", index),
            nodes=pa.concat_tables(parts),
            observations={kind: self.run("observed:" + kind, index) for kind in stated.nodes},
            targets=self.run("targets", index),
            edges={table: self.run("edge:" + table, index) for table in stated.edges},
        )
        if self.source.queries:
            tables["queries"] = self.queries(index)
        LOG.info(
            "read graph exit graph=%d seconds=%.3f observations=%d",
            index,
            time.perf_counter() - started,
            sum(table.num_rows for table in tables["observations"].values()),
        )
        return tables
