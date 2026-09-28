import logging
import time

import pyarrow as pa

from graphspec.infra.database import columns, connect, connections, query, tables

from .schema import EDGE, LEVELS, RESERVED, TableSchema

LOG = logging.getLogger(__name__)


def check(registry):
    clash = sorted(kind for kind in registry.nodes if kind in RESERVED or kind.startswith(EDGE))
    if clash:
        raise ValueError("node types collide with reserved table names: " + ", ".join(clash))
    if registry.extras["level"] not in LEVELS:
        raise ValueError("registry level must be one of " + ", ".join(LEVELS))


class Schema(TableSchema):
    def __init__(self, connection):
        started = time.perf_counter()
        self.connection = connection
        super().__init__({row["name"]: row["type"] for row in tables(connection)})
        LOG.info(
            "schema exit seconds=%.3f nodes=%d relations=%d features=%d signals=%d queries=%s",
            time.perf_counter() - started,
            len(self.nodes),
            len(self.relations),
            self.features,
            self.signals,
            self.queries,
        )

    def pairs(self, table):
        return connections(self.connection, table)

    def columns(self, table):
        return columns(self.connection, table)


class Database:
    def __init__(self, path, buffer_pool_size=4 << 30):
        self.location = str(path)
        self.store, self.connection = connect(path, read_only=True, buffer_pool_size=buffer_pool_size)
        try:
            self.schema = Schema(self.connection)
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            self.connection.close()
        finally:
            self.store.close()

    def reader(self, target):
        return Reader(self.store, self.connection, self.schema, target)

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
            "MATCH (q:query) RETURN q.graph AS graph, q.step AS step, q.period AS period, q.key AS key",
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
                % (target, target, ", t.tau AS tau" if level == "path" else ""),
            )
        return query(
            self.connection,
            "MATCH (t:`%s`)-[:`%s_of`]->(g:graph) RETURN t.id AS id, t.y AS y, t.split AS split, g.id AS graph ORDER BY id"
            % (target, target),
        )


class Reader:
    def __init__(self, database, connection, schema, target):
        started = time.perf_counter()
        self.database, self.connection, self.schema = database, connection, schema
        self.level = schema.target(target)
        signal = ", q.signal AS signal, q.signal_observed AS signal_observed" if schema.signals else ""
        self.statements = dict(
            observations="MATCH (e)-[:observed]->(o:observation) WHERE e.graph = $graph "
            "RETURN e.entity AS entity, o.seq AS seq, o.step AS step, o.x AS x"
        )
        if schema.queries:
            self.statements["queries"] = (
                "MATCH (a)<-[:query_actor]-(q:query)-[:query_subject]->(s), (q)-[:query_context]->(c) WHERE q.graph = $graph "
                "RETURN q.step AS step, q.period AS period, q.key AS key, q.category AS category, a.entity AS actor, "
                "s.entity AS subject, c.entity AS context, q.x AS query%s ORDER BY step" % signal
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
        for kind in schema.nodes:
            self.statements["node:" + kind] = (
                "MATCH (n:`%s`) WHERE n.graph = $graph RETURN n.entity AS entity, n.key AS key" % kind
            )
        for relation in schema.relations:
            self.statements["edge:" + relation["name"]] = (
                "MATCH (a)-[r:`%s%s`]->(b) WHERE a.graph = $graph RETURN a.entity AS source, b.entity AS target, "
                "r.seq AS seq, r.step AS step, r.present AS present%s"
                % (
                    EDGE,
                    relation["name"],
                    "".join(", r.`%s` AS `%s`" % (name, name) for name in relation["attributes"]),
                )
            )
        LOG.info(
            "reader exit seconds=%.3f level=%s relations=%d",
            time.perf_counter() - started,
            self.level,
            len(schema.relations),
        )

    def run(self, name, graph):
        return query(self.connection, self.statements[name], dict(graph=graph))

    def graph(self, index):
        started = time.perf_counter()
        parts = []
        for kind in self.schema.nodes:
            part = self.run("node:" + kind, index)
            parts.append(part.append_column("kind", pa.array([kind] * part.num_rows, pa.string())))
        tables = dict(
            nodes=pa.concat_tables(parts),
            observations=self.run("observations", index),
            targets=self.run("targets", index),
            edges={
                relation["name"]: self.run("edge:" + relation["name"], index)
                for relation in self.schema.relations
            },
        )
        if self.schema.queries:
            tables["queries"] = self.run("queries", index)
        LOG.info(
            "read graph exit graph=%d seconds=%.3f observations=%d",
            index,
            time.perf_counter() - started,
            tables["observations"].num_rows,
        )
        return tables
