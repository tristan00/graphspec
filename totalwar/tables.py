import numpy as np
import pyarrow as pa

from .domain import NODES, RELATIONS, SIGNALS, TARGET
from .values import fixed


def tables_of(graph, counter, index, name, observations, fact_rows, queries, targets, held_out):
    node_ids = counter.take("nodes", len(graph.kinds))
    tables = {("graph", None, None): pa.table(dict(id=pa.array([index], pa.int32()), name=pa.array([name])))}
    members = {kind: np.flatnonzero(np.asarray(graph.kinds) == kind) for kind in NODES}
    for kind in NODES:
        picked = members[kind]
        tables[(kind, None, None)] = pa.table(
            dict(
                id=pa.array(node_ids[picked]),
                graph=pa.array(np.full(len(picked), index), pa.int32()),
                entity=pa.array(picked, pa.int32()),
                key=pa.array([graph.keys[i] for i in picked], pa.string()),
            )
        )
    observation_ids = counter.take("observation", len(observations))
    entity = np.asarray([row[0] for row in observations], np.int64)
    tables[("observation", None, None)] = pa.table(
        dict(
            id=pa.array(observation_ids),
            seq=pa.array([row[2] for row in observations], pa.int32()),
            step=pa.array([row[1] for row in observations], pa.int32()),
            x=fixed(np.stack([row[3] for row in observations]), graph.features),
        )
    )
    kinds = np.asarray(graph.kinds)[entity]
    for kind in NODES:
        picked = kinds == kind
        tables[("observed", kind, "observation")] = pa.table(
            {"from": pa.array(node_ids[entity[picked]]), "to": pa.array(observation_ids[picked])}
        )
    query_ids = counter.take("query", len(queries))
    tables[("query", None, None)] = pa.table(
        dict(
            id=pa.array(query_ids),
            graph=pa.array(np.full(len(queries), index), pa.int32()),
            step=pa.array([row[1] for row in queries], pa.int32()),
            period=pa.array([row[2] for row in queries], pa.int32()),
            key=pa.array([row[0] for row in queries], pa.int64()),
            category=pa.array([row[3] for row in queries], pa.string()),
            x=fixed(np.stack([row[7] for row in queries]), graph.features),
            signal=fixed(np.asarray([row[8] for row in queries], np.float32), len(SIGNALS)),
            signal_observed=fixed(np.asarray([row[9] for row in queries], np.float32), len(SIGNALS)),
        )
    )
    for position, role in ((4, "actor"), (5, "subject"), (6, "context")):
        chosen = np.asarray([row[position] for row in queries], np.int64)
        for kind in NODES:
            picked = np.asarray(graph.kinds)[chosen] == kind
            tables[("query_" + role, "query", kind)] = pa.table(
                {"from": pa.array(query_ids[picked]), "to": pa.array(node_ids[chosen[picked]])}
            )
    for relation, (pairs, attributes) in RELATIONS.items():
        rows = fact_rows[relation]
        source = np.asarray([row[0] for row in rows], np.int64)
        target = np.asarray([row[1] for row in rows], np.int64)
        source_kind, target_kind = np.asarray(graph.kinds)[source], np.asarray(graph.kinds)[target]
        for pair in pairs:
            picked = (source_kind == pair[0]) & (target_kind == pair[1])
            chosen = [rows[i] for i in np.flatnonzero(picked)]
            columns = {
                "from": pa.array(node_ids[source[picked]]),
                "to": pa.array(node_ids[target[picked]]),
                "seq": pa.array([row[2] for row in chosen], pa.int32()),
                "step": pa.array([row[3] for row in chosen], pa.int32()),
                "present": pa.array([row[4] for row in chosen], pa.bool_()),
            }
            for offset, attribute in enumerate(attributes):
                columns[attribute] = pa.array([row[5 + offset] for row in chosen], pa.float32())
            tables[("edge_" + relation, pair[0], pair[1])] = pa.table(columns)
    label_ids = counter.take(TARGET, len(targets))
    tables[(TARGET, None, None)] = pa.table(
        dict(
            id=pa.array(label_ids),
            y=pa.array([row[1] for row in targets], pa.float64()),
            split=pa.array(["validation" if held_out else "train"] * len(targets), pa.string()),
        )
    )
    tables[(TARGET + "_of", TARGET, "query")] = pa.table(
        {"from": pa.array(label_ids), "to": pa.array(query_ids[[row[0] for row in targets]])}
    )
    return tables
