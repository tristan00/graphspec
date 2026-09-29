import math

import numpy as np
import pyarrow as pa

from graphspec.staged import IDS
from graphspec.staging import fixed

from .domain import NODES, RELATIONS, SIGNALS, TARGET


class Counter:
    def __init__(self, index):
        self.base, self.next = index << IDS, {}

    def take(self, table, count):
        start = self.next.get(table, self.base)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def node_id(index, node):
    return (index << IDS) + node


def empty_relation_tables(relation):
    pairs, attributes = RELATIONS[relation]
    tables = {}
    for pair in pairs:
        columns = {
            "from": pa.array([], pa.int64()),
            "to": pa.array([], pa.int64()),
            "seq": pa.array([], pa.int32()),
            "step": pa.array([], pa.int32()),
            "present": pa.array([], pa.bool_()),
        }
        for attribute in attributes:
            columns[attribute] = pa.array([], pa.float32())
        tables[("edge_" + relation, pair[0], pair[1])] = pa.table(columns)
    return tables


def chunk_tables(index, counter, kinds, width, observations, facts, queries, player):
    base = index << IDS
    tables = {}
    observation_ids = counter.take("observation", len(observations))
    tables[("observation", None, None)] = pa.table(
        dict(
            id=pa.array(observation_ids),
            seq=pa.array([row[2] for row in observations], pa.int32()),
            step=pa.array([row[1] for row in observations], pa.int32()),
            x=fixed(
                np.stack([row[3] for row in observations]) if observations else np.zeros((0, width)), width
            ),
        )
    )
    owners = np.asarray([row[0] for row in observations], np.int64)
    owner_kinds = (
        np.asarray([kinds[node] for node in owners], object) if len(owners) else np.asarray([], object)
    )
    for kind in NODES:
        picked = owner_kinds == kind
        tables[("observed", kind, "observation")] = pa.table(
            {"from": pa.array(base + owners[picked], pa.int64()), "to": pa.array(observation_ids[picked])}
        )
    for relation, (pairs, attributes) in RELATIONS.items():
        tables.update(empty_relation_tables(relation))
        rows = facts.get(relation, ())
        if not rows:
            continue
        source = np.asarray([row[0] for row in rows], np.int64)
        target = np.asarray([row[1] for row in rows], np.int64)
        source_kind = np.asarray([kinds[node] for node in source], object)
        target_kind = np.asarray([kinds[node] for node in target], object)
        for pair in pairs:
            picked = np.flatnonzero((source_kind == pair[0]) & (target_kind == pair[1]))
            if not len(picked):
                continue
            chosen = [rows[i] for i in picked]
            columns = {
                "from": pa.array(base + source[picked], pa.int64()),
                "to": pa.array(base + target[picked], pa.int64()),
                "seq": pa.array([row[2] for row in chosen], pa.int32()),
                "step": pa.array([row[3] for row in chosen], pa.int32()),
                "present": pa.array([row[4] for row in chosen], pa.bool_()),
            }
            for offset, attribute in enumerate(attributes):
                columns[attribute] = pa.array([row[5 + offset] for row in chosen], pa.float32())
            tables[("edge_" + relation, pair[0], pair[1])] = pa.table(columns)
        unmatched = len(rows) - sum(
            int(((source_kind == pair[0]) & (target_kind == pair[1])).sum()) for pair in pairs
        )
        if unmatched:
            raise ValueError("%d %s facts join undeclared node type pairs" % (unmatched, relation))
    query_ids = counter.take("query", len(queries))
    tables[("query", None, None)] = pa.table(
        dict(
            id=pa.array(query_ids),
            graph=pa.array(np.full(len(queries), index), pa.int32()),
            step=pa.array([row["step"] for row in queries], pa.int32()),
            period=pa.array([row["period"] for row in queries], pa.int32()),
            key=pa.array([row["key"] for row in queries], pa.int64()),
            category=pa.array([row["category"] for row in queries], pa.string()),
            x=fixed(np.stack([row["x"] for row in queries]) if queries else np.zeros((0, width)), width),
            signal=fixed(
                np.asarray([row["signal"] for row in queries], np.float32).reshape(-1, len(SIGNALS)),
                len(SIGNALS),
            ),
            signal_observed=fixed(
                np.asarray([row["observed"] for row in queries], np.float32).reshape(-1, len(SIGNALS)),
                len(SIGNALS),
            ),
        )
    )
    roles = dict(
        actor=[player] * len(queries),
        subject=[row["subject"] for row in queries],
        context=[player] * len(queries),
    )
    for role, nodes in roles.items():
        nodes = np.asarray(nodes, np.int64)
        node_kinds = np.asarray([kinds[node] for node in nodes], object)
        for kind in NODES:
            picked = np.flatnonzero(node_kinds == kind)
            tables[("query_" + role, "query", kind)] = pa.table(
                {"from": pa.array(query_ids[picked]), "to": pa.array(base + nodes[picked], pa.int64())}
            )
    return tables


def node_tables(index, name, kinds, keys):
    tables = {("graph", None, None): pa.table(dict(id=pa.array([index], pa.int32()), name=pa.array([name])))}
    kinds = np.asarray(kinds, object)
    for kind in NODES:
        picked = np.flatnonzero(kinds == kind)
        tables[(kind, None, None)] = pa.table(
            dict(
                id=pa.array(node_id(index, picked), pa.int64()),
                graph=pa.array(np.full(len(picked), index), pa.int32()),
                entity=pa.array(picked, pa.int32()),
                key=pa.array([keys[i] for i in picked], pa.string()),
            )
        )
    return tables


def label_tables(index, counter, labels, split):
    ids = counter.take(TARGET, len(labels))
    query_base = index << IDS
    return {
        (TARGET, None, None): pa.table(
            dict(
                id=pa.array(ids),
                y=pa.array([row[2] for row in labels], pa.float64()),
                split=pa.array([split] * len(labels), pa.string()),
                tau=pa.array([row[1] for row in labels], pa.float32()),
                series=pa.array([row[3] for row in labels], pa.string()),
            )
        ),
        (TARGET + "_of", TARGET, "query"): pa.table(
            {"from": pa.array(ids), "to": pa.array([query_base + row[0] for row in labels], pa.int64())}
        ),
    }


def removal_values(relation):
    return (math.nan,) * len(RELATIONS[relation][1])
