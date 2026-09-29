import logging
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from .infra.files import write_json
from .infra.runtime import output_path
from .schema import EDGE, OBSERVED, PREFIXES, TABLES

LOG = logging.getLogger(__name__)


def write_task(path, name, staging, target, schema):
    started = time.perf_counter()
    path = output_path(path)
    staging = Path(staging).resolve()
    location = (
        staging.relative_to(path.parent).as_posix() if staging.is_relative_to(path.parent) else str(staging)
    )
    write_json(path, dict(name=name, staging=location, target=target, schema=schema.mapping()))
    LOG.debug("write_task exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return path


def step_index(step):
    if not isinstance(step, Integral) or not 0 <= step < 1 << 31:
        raise ValueError("graph steps must be nonnegative int32 values")
    return step


class GraphTables:
    def __init__(self, index, name, schema, **attributes):
        if schema.queries is not None:
            raise ValueError("graph tables state no queries; the schema must state none")
        if not isinstance(index, Integral) or index < 0 or index >= 1 << 31:
            raise ValueError("graph index must be a nonnegative int32")
        self.index, self.schema = index, schema
        described = dict(id=pa.array([index], pa.int32()), name=pa.array([name], pa.string()))
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

    def observe(self, kind, keys, step=0, **columns):
        step = step_index(step)
        owners = self.resolve(kind, keys)
        values = self.stated("the kind of node " + kind, self.schema.nodes[kind], columns, len(owners))
        ids = self.ids("observation", len(owners))
        self.put(
            ("observation", None, None),
            pa.table(
                dict(
                    id=ids,
                    seq=pa.array(ids - (self.index << 32), pa.int32()),
                    step=pa.array(np.full(len(ids), step), pa.int32()),
                )
            ),
        )
        self.put((OBSERVED + kind, kind, "observation"), pa.table({"from": owners, "to": ids, **values}))
        return self

    def edges(self, table, source, target, sources, targets, step=0, present=True, relation=None, **columns):
        step = step_index(step)
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
            "seq": pa.array(np.arange(self.sequence, self.sequence + count), pa.int32()),
            "step": pa.array(np.full(count, step), pa.int32()),
            "present": pa.array(np.full(count, present), pa.bool_()),
        }
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

    def labels(self, kind, keys, values, splits, step=0, target="outcome"):
        step = step_index(step)
        owners = self.resolve(kind, keys)
        values, splits = list(values), list(splits)
        if target in self.lookup or target in TABLES or target.startswith(PREFIXES):
            raise ValueError("invalid or reserved target table: " + target)
        if not set(splits) <= {"train", "validation"}:
            raise ValueError("labels need train or validation splits")
        if len(values) != len(owners) or len(splits) != len(owners) or not np.isfinite(values).all():
            raise ValueError("labels need one finite value and one split per node")
        ids = self.ids(target, len(owners))
        self.put(
            (target, None, None),
            pa.table(
                dict(
                    id=ids,
                    y=pa.array(values, pa.float64()),
                    split=pa.array(splits, pa.string()),
                    step=pa.array(np.full(len(ids), step), pa.int32()),
                )
            ),
        )
        self.put((target + "_of", target, kind), pa.table({"from": ids, "to": owners}))
        return self

    def write(self, stage):
        started = time.perf_counter()
        stage.write(self.tables)
        LOG.info(
            "graph tables exit graph=%d entities=%d facts=%d seconds=%.3f",
            self.index,
            self.entities,
            self.sequence,
            time.perf_counter() - started,
        )
