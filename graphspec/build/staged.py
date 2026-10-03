import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import sha256_json
from graphspec.infra.runtime import cache_path
from graphspec.infra.timing import timed
from graphspec.schema.references import SORTS
from graphspec.schema.schema import (
    EDGE,
    FACTS,
    INNER,
    LINK,
    MEMBER,
    OBSERVED,
    TARGET,
    TASK,
    VALUE,
    WITHIN,
    inward,
    outward,
    typename,
)

from .output import manifest

LOG = logging.getLogger(__name__)
INDEX = 4
IDS = 32


def numbers(table, name):
    return table.column(name).to_numpy().astype(np.int64)


@timed("staged.unique_positions")
def unique_positions(positions, assigned, name):
    if np.any(positions < 0) or np.any(positions >= len(assigned)):
        raise ValueError("relationships leave their row partition in " + name)
    if np.any(assigned[positions] >= 0) or len(np.unique(positions)) != len(positions):
        raise ValueError("relationships assign more than one endpoint in " + name)


class Unit:
    @timed("staged.Unit.__init__")
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

    @timed("staged.Unit.types")
    def types(self):
        return {field.name: typename(field.type) for field in self.schema}

    @timed("staged.Unit.read")
    def read(self, columns=None):
        return pa.concat_tables([file.read(columns=columns) for file in self.files])

    @timed("staged.Unit.column")
    def column(self, name):
        return self.read([name]).column(name)

    @timed("staged.Unit.rows_between")
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

    @timed("staged.Unit.graph_rows")
    def graph_rows(self, graph, columns=None):
        return self.rows_between(int(self.bounds[graph]), int(self.bounds[graph + 1]), columns)


class Staged:
    def __init__(self, directory, stated):
        started = time.perf_counter()
        self.directory, self.stated = Path(directory), stated
        entries = manifest(self.directory)
        metadata = [
            (name, (self.directory / name).stat().st_size, (self.directory / name).stat().st_mtime_ns)
            for entry in entries
            for name in entry["files"]
        ]
        self.index_path = (
            cache_path("partitions", sha256_json(dict(version=INDEX, entries=entries, files=metadata)))
            / "partition.npz"
        )
        self.units = [Unit(self.directory, entry) for entry in entries]
        self.by_table = {}
        for unit in self.units:
            self.by_table.setdefault(unit.table, []).append(unit)
        self.kinds = {
            table: "NODE" if units[0].source is None else "REL" for table, units in self.by_table.items()
        }
        self.nodes = tuple(stated.entities())
        stated.conform(self)
        self.entity_of, self.graph_of, self.ids, self.dense, self.based = {}, {}, {}, set(), set()
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
            len(stated.edges),
            len(self.names),
        )

    @timed("staged.Staged.pairs")
    def pairs(self, table):
        return tuple(dict.fromkeys((unit.source, unit.target) for unit in self.by_table[table]))

    @timed("staged.Staged.columns")
    def columns(self, table):
        schemas = [
            {name: kind for name, kind in unit.types().items() if name not in ("from", "to")}
            for unit in self.by_table[table]
        ]
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("inconsistent property schemas for " + table)
        return schemas[0]

    @timed("staged.Staged.ids_of")
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

    @timed("staged.Staged.graphs_of")
    def graphs_of(self, table):
        if table not in self.graph_of:
            (unit,) = self.by_table[table]
            self.graph_of[table] = numbers(unit.read(columns=["graph"]), "graph")
        return self.graph_of[table]

    @timed("staged.Staged.endpoints")
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
        for table in list(self.nodes) + ["task"] + [INNER + kind for kind in self.stated.holds]:
            (unit,) = self.by_table[table]
            self.graph_of[table] = numbers(unit.read(columns=["graph"]), "graph")
            if table in self.nodes:
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

    @timed("staged.Staged.classify")
    def classify(self, table, unit):
        ids = self.ids_of(table)
        if table not in self.dense and np.array_equal(
            ids, (self.graph_of[table] << IDS) + np.arange(len(ids)) - unit.bounds[self.graph_of[table]]
        ):
            self.based.add(table)
            del self.ids[table]

    @timed("staged.Staged.derived_graphs")
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

    @timed("staged.Staged.persist")
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
    @timed("staged.Staged.bounds_of")
    def bounds_of(graph, count, table):
        if np.any(graph < 0) or np.any(graph >= count):
            raise ValueError("rows of %s name undeclared graphs" % table)
        if np.any(np.diff(graph) < 0):
            raise ValueError("rows of %s are not grouped by graph in order" % table)
        return np.searchsorted(graph, np.arange(count + 1))

    @timed("staged.Staged.reader")
    def reader(self):
        return Reader(self)

    @timed("staged.Staged.close")
    def close(self):
        for unit in self.units:
            for file in unit.files:
                file.close()

    @timed("staged.Staged.graphs")
    def graphs(self):
        (unit,) = self.by_table["graph"]
        return unit.read(columns=["id", "name"])

    def asked(self):
        (unit,) = self.by_table["task"]
        started = time.perf_counter()
        stated, names = self.stated.tasks, dict(id="id", graph="graph", kind="kind", split="split")
        names["weight"] = "weight"
        if self.stated.model != "none":
            names["time"] = "time"
        if self.stated.observers:
            names["observer"] = "observer"
        if stated["grouping"] is not None:
            names[stated["grouping"]] = "period"
        names[stated["key"]] = "key"
        result = unit.read(columns=list(names)).rename_columns(list(names.values()))
        LOG.debug("asked exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
        return result

    def aims(self):
        started = time.perf_counter()
        (unit,) = self.by_table[TARGET]
        table = unit.read()
        (link,) = self.by_table[LINK]
        rows, tasks = self.endpoints(link)
        order = np.argsort(rows, kind="stable")
        if not np.array_equal(rows[order], np.arange(table.num_rows)):
            raise ValueError("every target belongs to exactly one task")
        result = table.append_column("task", pa.array(tasks[order], pa.int64()))
        LOG.debug("aims exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
        return result


@timed("staged.valued")
def valued(rows, names):
    return {VALUE + name: rows.column(name) for name in names}


class Reader:
    @timed("staged.Reader.__init__")
    def __init__(self, source):
        self.source = source

    def graph(self, index):
        started = time.perf_counter()
        source, stated = self.source, self.source.stated
        parts = []
        for kind in source.nodes:
            (unit,) = source.by_table[kind]
            rows = unit.graph_rows(index, ["entity", "key"])
            parts.append(rows.append_column("kind", pa.array([kind] * rows.num_rows, pa.string())))
        (observation,) = source.by_table["observation"]
        start, stop = int(observation.bounds[index]), int(observation.bounds[index + 1])
        stamps = list(stated.stamps())
        held = observation.graph_rows(index, ["id"] + stamps)
        observations, seen, entities = {}, np.zeros(stop - start, np.int64), stated.entities()
        for kind in source.nodes:
            (unit,) = source.by_table[OBSERVED + kind]
            rows = unit.graph_rows(index)
            positions = source.positions("observation", numbers(rows, "to")) - start
            if np.any(positions < 0) or np.any(positions >= stop - start):
                raise ValueError("graph %d observes rows outside its observation range" % index)
            np.add.at(seen, positions, 1)
            take = pa.array(positions)
            observations[kind] = pa.table(
                dict(
                    entity=pa.array(
                        source.entity_of[kind][source.positions(kind, numbers(rows, "from"))], pa.int32()
                    ),
                    **{name: held.column(name).take(take) for name in stamps},
                    **valued(rows, entities[kind]),
                )
            )
        if np.any(seen != 1):
            raise ValueError("graph %d has observations that belong to no node or to several" % index)
        (unit,) = source.by_table["graph"]
        described = unit.graph_rows(index)
        named = dict(name=described.column("name"))
        if "unit" in described.column_names:
            named["unit"] = described.column("unit")
        nodes = pa.concat_tables(parts)
        tables = dict(
            graph=pa.table(dict(named, **valued(described, stated.graph))),
            nodes=nodes,
            observations=observations,
            edges={table: self.edges(index, table) for table in stated.edges},
            sets={kind: self.members(index, kind) for kind in stated.sets},
            held={kind: self.inner(index, kind, nodes) for kind in stated.holds},
            tasks=self.tasks(index),
            references=self.references(index, nodes),
            targets=self.targets(index),
        )
        LOG.info(
            "read graph exit graph=%d seconds=%.3f observations=%d",
            index,
            time.perf_counter() - started,
            stop - start,
        )
        return tables

    @timed("staged.Reader.edges")
    def edges(self, index, table):
        entry = self.source.stated.edges[table]
        parts = []
        for unit in self.source.by_table[EDGE + table]:
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
            for name in FACTS + tuple(self.source.stated.stamps()):
                columns[name] = rows.column(name)
            if entry["column"] is not None:
                columns["relation"] = rows.column(entry["column"])
            columns.update(valued(rows, entry["columns"]))
            parts.append(pa.table(columns))
        return pa.concat_tables(parts)

    def members(self, index, kind):
        started = time.perf_counter()
        source, entry = self.source, self.source.stated.sets[kind]
        parts = []
        for unit in source.by_table[MEMBER + kind]:
            rows = unit.graph_rows(index)
            parts.append(
                pa.table(
                    dict(
                        set=pa.array(
                            source.entity_of[kind][source.positions(kind, numbers(rows, "from"))], pa.int32()
                        ),
                        member=pa.array(
                            source.entity_of[unit.target][source.positions(unit.target, numbers(rows, "to"))],
                            pa.int32(),
                        ),
                        role=rows.column("role"),
                        **valued(rows, entry["members"]),
                    )
                )
            )
        result = pa.concat_tables(parts)
        LOG.debug(
            "read members exit graph=%d kind=%s rows=%d seconds=%.4f",
            index,
            kind,
            result.num_rows,
            time.perf_counter() - started,
        )
        return result

    def inner(self, index, kind, nodes):
        started = time.perf_counter()
        source, held = self.source, self.source.stated.holds[kind]
        (unit,) = source.by_table[INNER + kind]
        rows = unit.graph_rows(index)
        found = {
            key: entity
            for entity, key, owned in zip(
                *(nodes.column(name).to_pylist() for name in ("entity", "key", "kind"))
            )
            if owned == kind
        }
        owners = rows.column("owner").to_pylist()
        strangers = sorted({str(owner) for owner in owners if owner not in found})
        if strangers:
            raise ValueError(
                "the graphs held by nodes of %s in graph %d name owners the graph does not hold: %s"
                % (kind, index, ", ".join(strangers))
            )
        start = int(unit.bounds[index])
        parts = []
        for link in source.by_table[WITHIN + kind]:
            links = link.graph_rows(index)
            parts.append(
                pa.table(
                    dict(
                        source=pa.array(
                            source.positions(INNER + kind, numbers(links, "from")) - start, pa.int64()
                        ),
                        target=pa.array(
                            source.positions(INNER + kind, numbers(links, "to")) - start, pa.int64()
                        ),
                        relation=links.column("relation"),
                        **valued(links, outward(held)),
                    )
                )
            )
        result = dict(
            nodes=pa.table(
                dict(
                    owner=pa.array([found[owner] for owner in owners], pa.int32()),
                    kind=rows.column("kind"),
                    key=rows.column("key"),
                    **valued(rows, inward(held)),
                )
            ),
            edges=pa.concat_tables(parts),
        )
        LOG.debug(
            "read held graphs exit graph=%d kind=%s nodes=%d seconds=%.4f",
            index,
            kind,
            rows.num_rows,
            time.perf_counter() - started,
        )
        return result

    def tasks(self, index):
        started = time.perf_counter()
        source, stated = self.source, self.source.stated.tasks
        (task,) = source.by_table["task"]
        rows = task.graph_rows(index)
        columns = dict(
            id=rows.column("id"),
            key=rows.column(stated["key"]),
            kind=rows.column("kind"),
            split=rows.column("split"),
            weight=rows.column("weight"),
        )
        if source.stated.model != "none":
            columns["time"] = rows.column("time")
        if source.stated.observers:
            columns["observer"] = rows.column("observer")
        if stated["grouping"] is not None:
            columns["period"] = rows.column(stated["grouping"])
        columns.update(valued(rows, source.stated.asked_columns()))
        if stated["signals"]:
            columns.update(signal=rows.column("signal"), signal_observed=rows.column("signal_observed"))
        LOG.debug("read tasks exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
        return pa.table(columns)

    def references(self, index, nodes):
        started = time.perf_counter()
        source, stated = self.source, self.source.stated.tasks
        (task,) = source.by_table["task"]
        start = int(task.bounds[index])
        found = {}
        for entity, key, kind in zip(*(nodes.column(name).to_pylist() for name in ("entity", "key", "kind"))):
            found[(kind, key)] = entity
        result = {}
        for name, entry in stated["references"].items():
            stored = SORTS.named(entry["sort"]).columns()
            if stored is None:
                continue
            parts = []
            for unit in source.by_table[TASK + name]:
                links = unit.graph_rows(index)
                columns = dict(
                    task=pa.array(source.positions("task", numbers(links, "from")) - start, pa.int64()),
                    source=pa.array(
                        source.entity_of[unit.target][source.positions(unit.target, numbers(links, "to"))],
                        pa.int32(),
                    ),
                )
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
                parts.append(pa.table(columns))
            result[name] = pa.concat_tables(parts)
        LOG.debug(
            "read references exit graph=%d references=%d seconds=%.4f",
            index,
            len(result),
            time.perf_counter() - started,
        )
        return result

    def targets(self, index):
        started = time.perf_counter()
        source = self.source
        (unit,) = source.by_table[TARGET]
        (task,) = source.by_table["task"]
        rows = unit.graph_rows(index)
        start = int(unit.bounds[index])
        owners = np.full(rows.num_rows, -1, np.int64)
        (link,) = source.by_table[LINK]
        found = link.graph_rows(index)
        position = source.positions(TARGET, numbers(found, "from")) - start
        unique_positions(position, owners, LINK)
        owners[position] = source.positions("task", numbers(found, "to")) - int(task.bounds[index])
        if (owners < 0).any():
            raise ValueError("graph %d has targets that belong to no task" % index)
        LOG.debug("read targets exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
        return rows.append_column("task", pa.array(owners, pa.int64()))
