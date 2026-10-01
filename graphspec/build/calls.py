import logging
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from graphspec.infra.files import write_json
from graphspec.infra.runtime import output_path
from graphspec.schema.references import SORTS
from graphspec.schema.schema import EDGE, LINK, OBSERVED, TARGET, TASK, WRITTEN
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import UNITED

LOG = logging.getLogger(__name__)


def write_task(path, name, staging, schema):
    started = time.perf_counter()
    path = output_path(path)
    staging = Path(staging).resolve()
    location = (
        staging.relative_to(path.parent).as_posix() if staging.is_relative_to(path.parent) else str(staging)
    )
    write_json(path, dict(name=name, staging=location, schema=schema.mapping()))
    LOG.debug("write_task exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return path


class GraphTables:
    def __init__(self, index, name, unit, schema, **attributes):
        asked = schema.tasks
        held = [
            name
            for name, entry in asked["references"].items()
            if not SORTS.named(entry["sort"]).whole
            and (SORTS.named(entry["sort"]).columns() != {} or SORTS.named(entry["sort"]).most != 1)
        ]
        if asked["grouping"] is not None or asked["signals"] or held:
            raise ValueError(
                "graph tables state tasks without a grouping and without signals, whose references are one node "
                "each or the graph"
            )
        if schema.sets or schema.holds:
            raise ValueError("graph tables state no edges joining sets and no held graphs")
        if not isinstance(index, Integral) or index < 0 or index >= 1 << 31:
            raise ValueError("graph index must be a nonnegative int32")
        if (unit is not None) != (schema.model in UNITED) or (unit is not None and unit not in schema.units):
            raise ValueError(
                "a graph states one of the units of time of its schema, or none where the schema has no clock; "
                "found %r" % (unit,)
            )
        self.index, self.schema = index, schema
        described = dict(id=pa.array([index], pa.int32()), name=pa.array([name], pa.string()))
        if unit is not None:
            described["unit"] = pa.array([unit], pa.string())
        described.update(
            self.stated("the graph", schema.graph, {key: [value] for key, value in attributes.items()}, 1)
        )
        self.tables = {("graph", None, None): pa.table(described)}
        self.lookup, self.next = {}, {}
        self.entities, self.sequence = 0, 0

    def stated(self, where, names, columns, count):
        started = time.perf_counter()
        if sorted(columns) != sorted(names):
            raise ValueError(
                "%s states the columns %s, found %s" % (where, ", ".join(names), ", ".join(sorted(columns)))
            )
        result = {}
        for name in names:
            values = pa.array(list(columns[name]), self.schema.kind(name).written)
            if len(values) != count:
                raise ValueError("the column %s of %s needs %d values" % (name, where, count))
            result[name] = values
        LOG.debug(
            "graph tables stated exit columns=%d seconds=%.4f", len(names), time.perf_counter() - started
        )
        return result

    def timed(self, where, when, count):
        started = time.perf_counter()
        stamps = self.schema.stamps()
        if not isinstance(when, dict) or sorted(when) != sorted(stamps):
            raise ValueError(
                "%s states when it holds as exactly %s, found %r"
                % (where, ", ".join(stamps) or "nothing", when)
            )
        result = {}
        for name, kind in stamps.items():
            values = when[name] if isinstance(when[name], list) else [when[name]] * count
            if len(values) != count:
                raise ValueError("the column %s of %s needs %d values" % (name, where, count))
            result[name] = pa.array(
                [None if value is None else list(value) for value in values]
                if name == "observers"
                else values,
                WRITTEN[kind],
            )
        LOG.debug(
            "graph tables timed exit columns=%d seconds=%.4f", len(stamps), time.perf_counter() - started
        )
        return result

    def ids(self, table, count):
        start = self.next.get(table, self.index << 32)
        if start + count > (self.index + 1) << 32:
            raise ValueError("table exceeds its graph id range: " + table)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)

    def put(self, key, table):
        self.tables[key] = pa.concat_tables([self.tables[key], table]) if key in self.tables else table

    def nodes(self, kind, keys):
        if kind not in self.schema.nodes:
            raise ValueError("the schema states no kind of node named " + str(kind))
        keys = list(map(str, keys))
        if kind in self.lookup or len(set(keys)) != len(keys):
            raise ValueError("declare each node type once with unique keys: " + kind)
        ids = self.ids(kind, len(keys))
        self.lookup[kind] = dict(zip(keys, ids.tolist()))
        self.put(
            (kind, None, None),
            pa.table(
                dict(
                    id=ids,
                    graph=pa.array(np.full(len(keys), self.index), pa.int32()),
                    entity=pa.array(np.arange(self.entities, self.entities + len(keys)), pa.int32()),
                    key=pa.array(keys, pa.string()),
                )
            ),
        )
        self.entities += len(keys)
        return self

    def resolve(self, kind, keys):
        if kind not in self.lookup:
            raise ValueError("no nodes are declared of the kind " + str(kind))
        try:
            return np.asarray([self.lookup[kind][str(key)] for key in keys], np.int64)
        except KeyError as error:
            raise ValueError("undeclared node in " + kind + ": " + str(error)) from error

    def observe(self, kind, keys, when, **columns):
        owners = self.resolve(kind, keys)
        values = self.stated("the kind of node " + kind, self.schema.nodes[kind], columns, len(owners))
        ids = self.ids("observation", len(owners))
        self.put(
            ("observation", None, None),
            pa.table(dict(id=ids, **self.timed("the kind of node " + kind, when, len(owners)))),
        )
        self.put((OBSERVED + kind, kind, "observation"), pa.table({"from": owners, "to": ids, **values}))
        return self

    def edges(self, table, source, target, sources, targets, when, present, relation=None, **columns):
        if table not in self.schema.edges or type(present) is not bool:
            raise ValueError("edges name a stated edge table and a boolean presence value: " + str(table))
        entry = self.schema.edges[table]
        if (source, target) not in entry["endpoints"]:
            raise ValueError("the edge table %s is not stated to join %s to %s" % (table, source, target))
        left, right = self.resolve(source, sources), self.resolve(target, targets)
        if len(left) != len(right):
            raise ValueError("edges need matching endpoints")
        count = len(left)
        held = {
            "from": left,
            "to": right,
            "present": pa.array(np.full(count, present), pa.bool_()),
        }
        held.update(self.timed("the edge table " + table, when, count))
        if (entry["column"] is None) != (relation is None):
            raise ValueError(
                "the edge table %s takes its relations from %s"
                % (table, "its statement" if entry["column"] is None else "the column " + entry["column"])
            )
        if relation is not None:
            held[entry["column"]] = pa.array(list(relation), pa.string())
            if len(held[entry["column"]]) != count:
                raise ValueError("edges of %s need one relation for every edge" % table)
        held.update(self.stated("the edge table " + table, entry["columns"], columns, count))
        self.sequence += count
        self.put((EDGE + table, source, target), pa.table(held))
        return self

    def tasks(self, kind, keys, moment, references, splits, targets, known, weights, observers, **columns):
        started = time.perf_counter()
        schema, asked = self.schema, self.schema.tasks
        if kind not in asked["kinds"]:
            raise ValueError("the schema states no kind of task named " + str(kind))
        entry, timed = asked["kinds"][kind], schema.model != "none"
        if timed != isinstance(moment, Integral):
            raise ValueError(
                "tasks state their time as a whole number, or none where the schema has no time; found %r"
                % (moment,)
            )
        keys = list(keys)
        count = len(keys)
        if sorted(references) != sorted(entry["references"]):
            raise ValueError(
                "tasks of the kind %s state the references %s, found %s"
                % (kind, ", ".join(entry["references"]) or "none", ", ".join(sorted(references)) or "none")
            )
        given = dict(splits=splits, targets=targets, known=known, weights=weights, observers=observers)
        wanted = dict(
            splits=True,
            targets=entry["target"] is not None,
            known=entry["target"] is not None and timed,
            weights=entry["weight"],
            observers=schema.observers,
        )
        for name, values in given.items():
            if (values is not None) != wanted[name] or (values is not None and len(list(values)) != count):
                raise ValueError(
                    "tasks of the kind %s state %s"
                    % (kind, "one of %s for every task" % name if wanted[name] else "no " + name)
                )
        ids = self.ids("task", count)
        held = dict(
            id=ids,
            graph=pa.array(np.full(count, self.index), pa.int32()),
            kind=pa.array([kind] * count, pa.string()),
        )
        if timed:
            held["time"] = pa.array(np.full(count, moment), pa.int64())
        if schema.observers:
            held["observer"] = pa.array(list(observers), pa.string())
        held["split"] = pa.array(list(splits), pa.string())
        held["weight"] = pa.array([None] * count if weights is None else list(weights), pa.float32())
        held[asked["key"]] = pa.array(keys, pa.int64())
        for name in schema.asked_columns():
            if name in entry["columns"]:
                held.update(self.stated("the kind of task " + kind, [name], {name: columns[name]}, count))
            else:
                held[name] = pa.array([None] * count, schema.kind(name).written)
        if sorted(columns) != sorted(entry["columns"]):
            raise ValueError(
                "tasks of the kind %s state the columns %s, found %s"
                % (kind, ", ".join(entry["columns"]), ", ".join(sorted(columns)))
            )
        self.put(("task", None, None), pa.table(held))
        for name, (node, members) in references.items():
            if node not in asked["references"][name]["kinds"]:
                raise ValueError("the reference %s is not stated to name nodes of the kind %s" % (name, node))
            self.put((TASK + name, "task", node), pa.table({"from": ids, "to": self.resolve(node, members)}))
        if targets is not None:
            aim = entry["target"]
            target = TARGETS.named(aim["kind"])
            rows = [
                (place, row)
                for place, stated in enumerate(targets)
                for row in target.staged(stated, aim["series"], "a task of the kind %s" % kind)
            ]
            numbers = self.ids(TARGET, len(rows))
            aimed = dict(id=numbers, y=pa.array([row["y"] for _, row in rows], pa.float64()))
            if timed:
                moments = list(known)
                aimed["known"] = pa.array([moments[place] for place, _ in rows], pa.int64())
            for name, written in schema.target_columns().items():
                aimed[name] = pa.array(
                    [row[name] if name in target.columns else None for _, row in rows], WRITTEN[written]
                )
            owners = ids[np.asarray([place for place, _ in rows], np.int64)]
            self.put((TARGET, None, None), pa.table(aimed))
            self.put((LINK, TARGET, "task"), pa.table({"from": numbers, "to": owners}))
        LOG.debug(
            "graph tables tasks exit kind=%s tasks=%d seconds=%.4f",
            kind,
            count,
            time.perf_counter() - started,
        )
        return self

    def completed(self):
        started = time.perf_counter()
        schema = self.schema
        for name, entry in schema.tables().items():
            pairs = entry["pairs"] if entry["kind"] == "REL" else ((None, None),)
            for source, target in pairs:
                if (name, source, target) not in self.tables:
                    required = {held: kind for held, kind in entry["required"].items() if type(kind) is str}
                    required.update(
                        {held: kind[0] for held, kind in entry["required"].items() if type(kind) is not str}
                    )
                    fields = ([("from", pa.int64()), ("to", pa.int64())] if source is not None else []) + [
                        (held, WRITTEN[kind]) for held, kind in required.items()
                    ]
                    fields += [(held, schema.kind(held).written) for held in entry["columns"]]
                    self.tables[(name, source, target)] = pa.table(
                        {held: pa.array([], kind) for held, kind in fields}
                    )
        LOG.debug(
            "graph tables completed exit tables=%d seconds=%.4f",
            len(self.tables),
            time.perf_counter() - started,
        )

    def write(self, stage):
        started = time.perf_counter()
        self.completed()
        stage.write(self.tables)
        LOG.info(
            "graph tables exit graph=%d entities=%d facts=%d seconds=%.3f",
            self.index,
            self.entities,
            self.sequence,
            time.perf_counter() - started,
        )
