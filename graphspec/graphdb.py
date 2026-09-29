import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.database import columns, connect, connections, query, tables
from graphspec.infra.timing import timed

from .clock import UNITED
from .schema import EDGE, OBSERVED, QUERY, VALUE, Tables

LOG = logging.getLogger(__name__)


def valued(holder, names):
    return "".join(", %s.`%s` AS `%s%s`" % (holder, name, VALUE, name) for name in names)


@timed("graphdb.stamped")
def stamped(holder, names):
    return "".join(", %s.`%s` AS `%s`" % (holder, name, name) for name in names)


@timed("graphdb.asked")
def asked(stated):
    held = dict(id="id")
    if stated.model != "none":
        held["time"] = "time"
    if stated.observers:
        held["observer"] = "observer"
    if stated.queries["grouping"] is not None:
        held[stated.queries["grouping"]] = "period"
    held[stated.queries["key"]] = "key"
    return held


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
        started = time.perf_counter()
        result = query(
            self.connection,
            "MATCH (q:query) RETURN q.graph AS graph%s ORDER BY id"
            % "".join(
                ", q.`%s` AS `%s`" % pair for pair in asked(self.stated).items() if pair[1] != "observer"
            ),
        )
        LOG.debug("query index exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
        return result

    def targets(self, level, target):
        if level == "node":
            return query(
                self.connection,
                "MATCH (t:`%s`)-[:`%s_of`]->(n) RETURN t.id AS id, t.y AS y, t.split AS split%s, "
                "n.graph AS graph, n.entity AS entity, n.key AS node ORDER BY id"
                % (target, target, "" if self.stated.model == "none" else ", t.time AS time"),
            )
        if level in ("query", "path"):
            return query(
                self.connection,
                "MATCH (t:`%s`)-[:`%s_of`]->(q:query) RETURN t.id AS id, t.y AS y, t.split AS split, "
                "q.graph AS graph, q.id AS query%s ORDER BY id"
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
            graph="MATCH (g:graph) WHERE g.id = $graph RETURN g.name AS name"
            + (", g.unit AS unit" if stated.model in UNITED else "")
            + valued("g", stated.graph)
        )
        if source.queries:
            names = asked(stated)
            self.statements["queries"] = (
                "MATCH %s WHERE q.graph = $graph "
                "RETURN q.category AS category%s%s%s%s "
                "ORDER BY %sid"
                % (
                    ", ".join(
                        "(%s)-[:`%s%s`]->(r%d)" % ("q:query" if place == 0 else "q", QUERY, name, place)
                        for place, name in enumerate(stated.queries["references"])
                    ),
                    "".join(", q.`%s` AS `%s`" % pair for pair in names.items()),
                    "".join(
                        ", r%d.entity AS r%d" % (place, place)
                        for place in range(len(stated.queries["references"]))
                    ),
                    valued("q", stated.queries["columns"]),
                    ", q.signal AS signal, q.signal_observed AS signal_observed" if source.signals else "",
                    "" if stated.model == "none" else "time, ",
                )
            )
        if self.level in ("query", "path"):
            self.statements["targets"] = (
                "MATCH (t:`%s`)-[:`%s_of`]->(q:query) WHERE q.graph = $graph "
                "RETURN t.id AS id, t.y AS y, t.split AS split%s"
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
                "RETURN t.id AS id, t.y AS y, t.split AS split%s, n.entity AS entity"
                % (target, target, "" if stated.model == "none" else ", t.time AS time")
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
