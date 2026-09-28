import logging
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from .infra.files import write_json
from .infra.runtime import output_path
from .schema import EDGE, RESERVED

LOG = logging.getLogger(__name__)


def fixed(values, width):
    values = np.ascontiguousarray(values, np.float32)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError("observation features must be a finite matrix with width %d" % width)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def write_task(path, name, staging, target):
    path = output_path(path)
    staging = Path(staging).resolve()
    location = (
        staging.relative_to(path.parent).as_posix() if staging.is_relative_to(path.parent) else str(staging)
    )
    write_json(path, dict(name=name, staging=location, target=target))
    return path


def step_index(step):
    if not isinstance(step, Integral) or not 0 <= step < 1 << 31:
        raise ValueError("graph steps must be nonnegative int32 values")
    return step


class GraphTables:
    def __init__(self, index, name, features):
        if not isinstance(features, Integral) or features < 2 or features % 2:
            raise ValueError("features must be a positive even width: values, then presence")
        if not isinstance(index, Integral) or index < 0 or index >= 1 << 31:
            raise ValueError("graph index must be a nonnegative int32")
        self.index, self.features = index, features
        self.tables = {("graph", None, None): pa.table(dict(id=pa.array([index], pa.int32()), name=[name]))}
        self.lookup, self.next = {}, {}
        self.entities, self.sequence = 0, 0

    def ids(self, table, count):
        start = self.next.get(table, self.index << 32)
        if start + count > (self.index + 1) << 32:
            raise ValueError("table exceeds its graph id range: " + table)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)

    def put(self, key, table):
        self.tables[key] = pa.concat_tables([self.tables[key], table]) if key in self.tables else table

    def nodes(self, kind, keys):
        if not isinstance(kind, str) or not kind or kind in RESERVED or kind.startswith(EDGE):
            raise ValueError("invalid or reserved node type: " + str(kind))
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
        try:
            return np.asarray([self.lookup[kind][str(key)] for key in keys], np.int64)
        except KeyError as error:
            raise ValueError("undeclared node in " + kind + ": " + str(error)) from error

    def observe(self, kind, keys, values, step=0):
        step = step_index(step)
        owners = self.resolve(kind, keys)
        x = fixed(values, self.features)
        if len(x) != len(owners) or step < 0:
            raise ValueError("observations need one feature vector per node and a nonnegative step")
        ids = self.ids("observation", len(owners))
        self.put(
            ("observation", None, None),
            pa.table(
                dict(
                    id=ids,
                    seq=pa.array(ids - (self.index << 32), pa.int32()),
                    step=pa.array(np.full(len(ids), step), pa.int32()),
                    x=x,
                )
            ),
        )
        self.put(("observed", kind, "observation"), pa.table({"from": owners, "to": ids}))
        return self

    def edges(self, name, source, target, sources, targets, step=0, present=True, **attributes):
        step = step_index(step)
        if not name or ":" in name or type(present) is not bool:
            raise ValueError(
                "edges require a nonempty relation name without colons and a boolean presence value"
            )
        left, right = self.resolve(source, sources), self.resolve(target, targets)
        if len(left) != len(right) or step < 0:
            raise ValueError("edges need matching endpoints and a nonnegative step")
        count = len(left)
        columns = {
            "from": left,
            "to": right,
            "seq": pa.array(np.arange(self.sequence, self.sequence + count), pa.int32()),
            "step": pa.array(np.full(count, step), pa.int32()),
            "present": pa.array(np.full(count, present), pa.bool_()),
        }
        for key, values in attributes.items():
            if key in columns or not np.isfinite(np.asarray(values, np.float32)).all():
                raise ValueError("edge attributes must be finite and have distinct names")
            columns[key] = pa.array(values, pa.float32())
        self.sequence += count
        self.put(("edge_" + name, source, target), pa.table(columns))
        return self

    def labels(self, kind, keys, values, splits, step=0, target="outcome"):
        step = step_index(step)
        owners = self.resolve(kind, keys)
        values, splits = list(values), list(splits)
        if target in self.lookup or target in RESERVED or target.startswith(EDGE):
            raise ValueError("invalid or reserved target table: " + target)
        if step < 0 or not set(splits) <= {"train", "validation"}:
            raise ValueError("labels need a nonnegative step and train or validation splits")
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
