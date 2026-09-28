import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import sha256_json
from graphspec.infra.runtime import cache_path
from graphspec.infra.staging import manifest

from .schema import EDGE, FACTS, ROLES, TableSchema

LOG = logging.getLogger(__name__)
IDS = 32
TYPES = {
    pa.int64(): "INT64",
    pa.int32(): "INT32",
    pa.float32(): "FLOAT",
    pa.float64(): "DOUBLE",
    pa.bool_(): "BOOL",
    pa.string(): "STRING",
}


def typename(kind):
    if pa.types.is_fixed_size_list(kind) and kind.value_type == pa.float32():
        return "FLOAT[%d]" % kind.list_size
    if kind not in TYPES:
        raise ValueError("unsupported graph property type: " + str(kind))
    return TYPES[kind]


def numbers(table, name):
    return table.column(name).to_numpy().astype(np.int64)


def unique_positions(positions, assigned, name):
    if np.any(positions < 0) or np.any(positions >= len(assigned)):
        raise ValueError("relationships leave their row partition in " + name)
    if np.any(assigned[positions] >= 0) or len(np.unique(positions)) != len(positions):
        raise ValueError("relationships assign more than one endpoint in " + name)


class Unit:
    def __init__(self, directory, entry):
        self.table, self.source, self.target, self.rows = (
            entry["table"],
            entry["source"],
            entry["target"],
            entry["rows"],
        )
        self.name = "%s|%s|%s" % (self.table, self.source, self.target)
        self.files = [pq.ParquetFile(directory / name) for name in entry["files"]]
        self.members = [
            (index, group) for index, file in enumerate(self.files) for group in range(file.num_row_groups)
        ]
        self.groups = np.r_[
            0,
            np.cumsum(
                [self.files[index].metadata.row_group(group).num_rows for index, group in self.members]
            ),
        ]
        self.schema = self.files[0].schema_arrow
        if any(file.schema_arrow != self.schema for file in self.files):
            raise ValueError("inconsistent Parquet schemas for " + self.name)
        if int(self.groups[-1]) != self.rows:
            raise ValueError("manifest row count does not match Parquet rows for " + self.name)
        self.bounds = None

    def types(self):
        return {field.name: typename(field.type) for field in self.schema}

    def read(self, columns=None):
        return pa.concat_tables([file.read(columns=columns) for file in self.files])

    def column(self, name):
        return self.read([name]).column(name)

    def rows_between(self, start, stop, columns=None):
        if start < 0 or stop < start or stop > self.rows:
            raise ValueError("row range outside " + self.name)
        if stop <= start:
            return self.schema.empty_table().select(self.schema.names if columns is None else columns)
        first = int(np.searchsorted(self.groups, start, "right") - 1)
        last = int(np.searchsorted(self.groups, stop, "left"))
        chosen = {}
        for index, group in self.members[first:last]:
            chosen.setdefault(index, []).append(group)
        table = pa.concat_tables(
            [self.files[index].read_row_groups(groups, columns=columns) for index, groups in chosen.items()]
        )
        return table.slice(start - int(self.groups[first]), stop - start)

    def graph_rows(self, graph, columns=None):
        return self.rows_between(int(self.bounds[graph]), int(self.bounds[graph + 1]), columns)


class Staged(TableSchema):
    def __init__(self, directory):
        started = time.perf_counter()
        self.directory = Path(directory)
        entries = manifest(self.directory)
        metadata = [
            (name, (self.directory / name).stat().st_size, (self.directory / name).stat().st_mtime_ns)
            for entry in entries
            for name in entry["files"]
        ]
        self.index_path = (
            cache_path("partitions", sha256_json(dict(version=2, entries=entries, files=metadata)))
            / "partition.npz"
        )
        self.units = [Unit(self.directory, entry) for entry in entries]
        self.by_table = {}
        for unit in self.units:
            self.by_table.setdefault(unit.table, []).append(unit)
        super().__init__(
            {table: "NODE" if units[0].source is None else "REL" for table, units in self.by_table.items()}
        )
        self.entity_of, self.graph_of, self.ids, self.dense, self.based, self.steps = (
            {},
            {},
            {},
            set(),
            set(),
            {},
        )
        self.location = str(self.directory)
        self.names = None
        if self.index_path.exists():
            self.restore()
        else:
            self.partition()
            self.persist()
        LOG.info(
            "staged exit seconds=%.3f units=%d nodes=%d relations=%d graphs=%d",
            time.perf_counter() - started,
            len(self.units),
            len(self.nodes),
            len(self.relations),
            len(self.names),
        )

    def pairs(self, table):
        return tuple(dict.fromkeys((unit.source, unit.target) for unit in self.by_table[table]))

    def columns(self, table):
        schemas = [
            {name: kind for name, kind in unit.types().items() if name not in ("from", "to")}
            for unit in self.by_table[table]
        ]
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("inconsistent property schemas for " + table)
        return schemas[0]

    def ids_of(self, table):
        if table not in self.ids:
            (unit,) = self.by_table[table]
            ids = numbers(unit.read(columns=["id"]), "id")
            if len(ids) and np.any(np.diff(ids) <= 0):
                raise ValueError("%s ids must increase down its rows" % table)
            if np.array_equal(ids, np.arange(len(ids))):
                self.dense.add(table)
            self.ids[table] = ids
        return self.ids[table]

    def positions(self, table, wanted):
        if np.any(wanted < 0):
            raise ValueError("rows refer to negative ids in " + table)
        if table == "graph" or table in self.dense:
            (unit,) = self.by_table[table]
            if np.any(wanted >= unit.rows):
                raise ValueError("rows refer to ids missing from " + table)
            return wanted
        if table in self.based:
            (unit,) = self.by_table[table]
            graph, local = wanted >> IDS, wanted & ((1 << IDS) - 1)
            if np.any(graph >= len(unit.bounds) - 1):
                raise ValueError("rows refer to ids missing from %s" % table)
            if np.any(local >= unit.bounds[graph + 1] - unit.bounds[graph]):
                raise ValueError("rows refer to ids missing from %s" % table)
            return unit.bounds[graph] + local
        ids = self.ids_of(table)
        found = np.searchsorted(ids, wanted)
        if np.any(found >= len(ids)) or not np.array_equal(ids[found], wanted):
            raise ValueError("rows refer to ids missing from %s" % table)
        return found

    def graphs_of(self, table):
        if table not in self.graph_of:
            (unit,) = self.by_table[table]
            self.graph_of[table] = numbers(unit.read(columns=["graph"]), "graph")
        return self.graph_of[table]

    def query_steps(self):
        if "query" not in self.steps:
            (unit,) = self.by_table["query"]
            self.steps["query"] = numbers(unit.read(columns=["step"]), "step")
        return self.steps["query"]

    def endpoints(self, unit):
        table = unit.read(columns=["from", "to"])
        return self.positions(unit.source, numbers(table, "from")), self.positions(
            unit.target, numbers(table, "to")
        )

    def partition(self):
        started = time.perf_counter()
        (graph_unit,) = self.by_table["graph"]
        graphs = numbers(graph_unit.read(columns=["id"]), "id")
        if not np.array_equal(graphs, np.arange(len(graphs))):
            raise ValueError("graph ids must number the graphs in order")
        self.names = dict(zip(graphs.tolist(), graph_unit.column("name").to_pylist()))
        count = len(graphs)
        graph_unit.bounds = np.arange(count + 1)
        self.graph_of["graph"] = graphs
        for table in list(self.nodes) + (["query"] if self.queries else []):
            (unit,) = self.by_table[table]
            self.graph_of[table] = numbers(unit.read(columns=["graph"]), "graph")
            if table != "query":
                self.entity_of[table] = numbers(unit.read(columns=["entity"]), "entity")
            unit.bounds = self.bounds_of(self.graph_of[table], count, table)
            self.classify(table, unit)
        for table in self.by_table:
            if self.kinds[table] == "NODE" and table not in self.graph_of:
                self.graph_of[table] = self.derived_graphs(table, count)
                (unit,) = self.by_table[table]
                unit.bounds = self.bounds_of(self.graph_of[table], count, table)
                self.classify(table, unit)
        for table, units in self.by_table.items():
            if self.kinds[table] == "REL":
                for unit in units:
                    source, target = self.endpoints(unit)
                    if not np.array_equal(
                        self.graph_of[unit.source][source], self.graph_of[unit.target][target]
                    ):
                        raise ValueError("relationships cross graph partitions in " + table)
                    unit.bounds = self.bounds_of(self.graph_of[unit.source][source], count, table)
        LOG.info("partition exit seconds=%.3f graphs=%d", time.perf_counter() - started, count)

    def classify(self, table, unit):
        ids = self.ids_of(table)
        if table not in self.dense and np.array_equal(
            ids, (self.graph_of[table] << IDS) + np.arange(len(ids)) - unit.bounds[self.graph_of[table]]
        ):
            self.based.add(table)
            del self.ids[table]

    def derived_graphs(self, table, count):
        graph = np.full(len(self.ids_of(table)), -1, np.int64)
        for candidate, units in self.by_table.items():
            if self.kinds[candidate] != "REL":
                continue
            for unit in units:
                if unit.target == table and unit.source in self.graph_of:
                    source, target = self.endpoints(unit)
                    graph[target] = self.graph_of[unit.source][source]
                elif unit.source == table and unit.target in self.graph_of:
                    source, target = self.endpoints(unit)
                    graph[source] = self.graph_of[unit.target][target]
        if (graph < 0).any():
            raise ValueError("%d rows of %s belong to no graph" % (int((graph < 0).sum()), table))
        return graph

    def persist(self):
        arrays = {"bounds/" + unit.name: unit.bounds for unit in self.units}
        arrays.update({"ids/" + table: ids for table, ids in self.ids.items() if table not in self.dense})
        arrays.update({"entity/" + table: values for table, values in self.entity_of.items()})
        arrays["dense"] = np.asarray(sorted(self.dense))
        arrays["based"] = np.asarray(sorted(self.based))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.index_path, **arrays)

    def restore(self):
        started = time.perf_counter()
        (graph_unit,) = self.by_table["graph"]
        self.names = dict(
            zip(
                numbers(graph_unit.read(columns=["id"]), "id").tolist(), graph_unit.column("name").to_pylist()
            )
        )
        with np.load(self.index_path) as saved:
            for unit in self.units:
                unit.bounds = saved["bounds/" + unit.name]
            self.dense, self.based = set(saved["dense"].tolist()), set(saved["based"].tolist())
            for key in saved.files:
                if key.startswith("ids/"):
                    self.ids[key[4:]] = saved[key]
                elif key.startswith("entity/"):
                    self.entity_of[key[7:]] = saved[key]
        LOG.info("partition restored seconds=%.3f graphs=%d", time.perf_counter() - started, len(self.names))

    @staticmethod
    def bounds_of(graph, count, table):
        if np.any(graph < 0) or np.any(graph >= count):
            raise ValueError("rows of %s name undeclared graphs" % table)
        if np.any(np.diff(graph) < 0):
            raise ValueError("rows of %s are not grouped by graph in order" % table)
        return np.searchsorted(graph, np.arange(count + 1))

    def reader(self, target):
        return Reader(self, target)

    @property
    def schema(self):
        return self

    def close(self):
        for unit in self.units:
            for file in unit.files:
                file.close()

    def graphs(self):
        return pa.table(
            dict(
                id=pa.array(sorted(self.names), pa.int32()),
                name=pa.array([self.names[index] for index in sorted(self.names)]),
            )
        )

    def categories(self):
        (unit,) = self.by_table["query"]
        return sorted(set(unit.column("category").to_pylist()))

    def query_index(self):
        (unit,) = self.by_table["query"]
        return unit.read(columns=["graph", "step", "period", "key"])

    def targets(self, level, target):
        (unit,) = self.by_table[target]
        table = unit.read()
        rows, kinds, destinations = [], [], []
        for link in self.by_table[target + "_of"]:
            source, destination = self.endpoints(link)
            rows.append(source)
            destinations.append(destination)
            kinds.append(
                np.full(
                    len(source), self.nodes.index(link.target) if link.target in self.nodes else -1, np.int64
                )
            )
        rows, kinds, destinations = (np.concatenate(values) for values in (rows, kinds, destinations))
        order = np.argsort(rows, kind="stable")
        rows, kinds, destinations = rows[order], kinds[order], destinations[order]
        if not np.array_equal(rows, np.arange(table.num_rows)):
            raise ValueError("every %s row needs exactly one %s_of link" % (target, target))
        columns = dict(id=table.column("id"), y=table.column("y"), split=table.column("split"))
        if level in ("query", "path"):
            columns.update(
                graph=pa.array(self.graphs_of("query")[destinations], pa.int32()),
                step=pa.array(self.query_steps()[destinations], pa.int32()),
            )
            if level == "path":
                columns["tau"] = table.column("tau")
        elif level == "graph":
            columns.update(graph=pa.array(destinations, pa.int32()))
        else:
            keys = {
                kind: self.by_table[self.nodes[kind]][0].column("key").to_pylist()
                for kind in np.unique(kinds).tolist()
            }
            columns.update(
                step=table.column("step"),
                graph=pa.array(
                    [
                        int(self.graphs_of(self.nodes[k])[d])
                        for k, d in zip(kinds.tolist(), destinations.tolist())
                    ],
                    pa.int32(),
                ),
                entity=pa.array(
                    [
                        int(self.entity_of[self.nodes[k]][d])
                        for k, d in zip(kinds.tolist(), destinations.tolist())
                    ],
                    pa.int32(),
                ),
                node=pa.array([keys[k][d] for k, d in zip(kinds.tolist(), destinations.tolist())]),
            )
        return pa.table(columns)


class Reader:
    def __init__(self, source, target):
        self.source, self.level, self.target_name = source, source.target(target), target

    def graph(self, index):
        started = time.perf_counter()
        source = self.source
        parts = []
        for kind in source.nodes:
            (unit,) = source.by_table[kind]
            rows = unit.graph_rows(index, ["entity", "key"])
            parts.append(rows.append_column("kind", pa.array([kind] * rows.num_rows, pa.string())))
        (observation,) = source.by_table["observation"]
        start, stop = int(observation.bounds[index]), int(observation.bounds[index + 1])
        held = observation.graph_rows(index, ["seq", "step", "x"])
        entities, positions = [], []
        for unit in source.by_table["observed"]:
            rows = unit.graph_rows(index)
            entities.append(
                source.entity_of[unit.source][source.positions(unit.source, numbers(rows, "from"))]
            )
            positions.append(source.positions("observation", numbers(rows, "to")) - start)
        entities, positions = np.concatenate(entities), np.concatenate(positions)
        if len(positions) != stop - start or not np.array_equal(np.sort(positions), np.arange(stop - start)):
            raise ValueError("graph %d observes rows outside its observation range" % index)
        take = pa.array(positions)
        observations = pa.table(
            dict(
                entity=pa.array(entities, pa.int32()),
                seq=held.column("seq").take(take),
                step=held.column("step").take(take),
                x=held.column("x").take(take),
            )
        )
        tables = dict(
            nodes=pa.concat_tables(parts),
            observations=observations,
            edges={relation["name"]: self.edges(index, relation) for relation in source.relations},
        )
        if source.queries:
            tables["queries"] = self.queries(index)
        tables["targets"] = self.targets(index)
        LOG.info(
            "read graph exit graph=%d seconds=%.3f observations=%d",
            index,
            time.perf_counter() - started,
            observations.num_rows,
        )
        return tables

    def edges(self, index, relation):
        parts = []
        for unit in self.source.by_table[EDGE + relation["name"]]:
            rows = unit.graph_rows(index)
            columns = dict(
                source=pa.array(
                    self.source.entity_of[unit.source][
                        self.source.positions(unit.source, numbers(rows, "from"))
                    ],
                    pa.int32(),
                ),
                target=pa.array(
                    self.source.entity_of[unit.target][
                        self.source.positions(unit.target, numbers(rows, "to"))
                    ],
                    pa.int32(),
                ),
            )
            for name in FACTS + tuple(relation["attributes"]):
                columns[name] = rows.column(name)
            parts.append(pa.table(columns))
        return pa.concat_tables(parts)

    def queries(self, index):
        source = self.source
        (query,) = source.by_table["query"]
        rows = query.graph_rows(index)
        start = int(query.bounds[index])
        roles = {}
        for role in ROLES:
            found = np.full(rows.num_rows, -1, np.int64)
            for unit in source.by_table["query_" + role]:
                links = unit.graph_rows(index)
                positions = source.positions("query", numbers(links, "from")) - start
                unique_positions(positions, found, "query_" + role)
                found[positions] = source.entity_of[unit.target][
                    source.positions(unit.target, numbers(links, "to"))
                ]
            if (found < 0).any():
                raise ValueError("graph %d has queries without a %s" % (index, role))
            roles[role] = found
        order = np.argsort(rows.column("step").to_numpy(), kind="stable")
        take = pa.array(order)
        columns = dict(
            step=rows.column("step").take(take),
            period=rows.column("period").take(take),
            key=rows.column("key").take(take),
            category=rows.column("category").take(take),
            actor=pa.array(roles["actor"][order], pa.int32()),
            subject=pa.array(roles["subject"][order], pa.int32()),
            context=pa.array(roles["context"][order], pa.int32()),
            query=rows.column("x").take(take),
        )
        if source.signals:
            columns.update(
                signal=rows.column("signal").take(take),
                signal_observed=rows.column("signal_observed").take(take),
            )
        return pa.table(columns)

    def targets(self, index):
        source = self.source
        (unit,) = source.by_table[self.target_name]
        rows = unit.graph_rows(index)
        start = int(unit.bounds[index])
        kinds, destinations = np.full(rows.num_rows, -1, np.int64), np.full(rows.num_rows, -1, np.int64)
        for link in source.by_table[self.target_name + "_of"]:
            found = link.graph_rows(index)
            position = source.positions(self.target_name, numbers(found, "from")) - start
            unique_positions(position, destinations, self.target_name + "_of")
            kinds[position] = source.nodes.index(link.target) if link.target in source.nodes else -1
            destinations[position] = source.positions(link.target, numbers(found, "to"))
        if (destinations < 0).any():
            raise ValueError("graph %d has %s rows without a link" % (index, self.target_name))
        columns = dict(id=rows.column("id"), y=rows.column("y"), split=rows.column("split"))
        if self.level in ("query", "path"):
            columns["step"] = pa.array(source.query_steps()[destinations], pa.int32())
            if self.level == "path":
                columns["tau"] = rows.column("tau")
        elif self.level == "node":
            columns["step"] = rows.column("step")
            columns["entity"] = pa.array(
                [
                    int(source.entity_of[source.nodes[k]][d])
                    for k, d in zip(kinds.tolist(), destinations.tolist())
                ],
                pa.int32(),
            )
        return pa.table(columns)
