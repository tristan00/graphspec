import logging
import time

import numpy as np

from graphspec import GraphTables
from graphspec.infra.features import encoder

from .schema import BY_FIELD, KNOWN_FIELDS, NUMBERS, RELATIONSHIPS, RENEWAL, present, selected_entities
from .snapshot import Snapshot

LOG = logging.getLogger(__name__)


def link(graph, relation, source, target, pairs):
    graph.edges(relation, source, target, [left for left, _ in pairs], [right for _, right in pairs])


def policy_observations(snapshot, encode):
    observations = []
    for start, active, days, cancelled in zip(
        snapshot.starts, snapshot.active, snapshot.ended_days, snapshot.cancellations
    ):
        data = dict(age_days=int((snapshot.cutoff - start).astype(int)), active=int(active))
        if days > 0:
            data.update(ended_days_ago=int(days), cancelled=float(cancelled))
        observations.append(encode("policy", data))
    return np.asarray(observations)


def client_observations(snapshot, encode):
    clients, inverse, counts = np.unique(
        snapshot.columns["client_key"], return_inverse=True, return_counts=True
    )
    active = np.bincount(inverse, weights=snapshot.active, minlength=len(clients))
    cancelled = np.bincount(inverse, weights=snapshot.cancellations == 1, minlength=len(clients))
    earliest = np.full(len(clients), snapshot.cutoff, dtype="datetime64[D]")
    np.minimum.at(earliest, inverse, snapshot.starts)
    tenure = (snapshot.cutoff - earliest).astype(int)
    values = [
        encode(
            "client",
            dict(
                policies=int(count),
                active_policies=int(current),
                past_cancellations=int(past),
                tenure_days=int(days),
            ),
        )
        for count, current, past, days in zip(counts, active, cancelled, tenure)
    ]
    return clients.tolist(), np.asarray(values)


def entity_graph(graph, entity, keys, snapshot, encode):
    members = entity.members(keys)
    if not members:
        return
    graph.nodes(entity.kind, members)
    data = [dict(code=key) for key in members] if entity.categorical else [{}] * len(members)
    graph.observe(entity.kind, members, np.asarray([encode(entity.kind, item) for item in data]))
    pairs = [(policy, key) for policy, key in zip(snapshot.columns["policy_key"], keys) if present(key)]
    link(graph, "policy_" + entity.kind, "policy", entity.kind, pairs)
    if entity.hierarchy_width:
        parents = [
            (code, code[:-1])
            for code in members
            if code.isdigit() and 2 < len(code) <= entity.hierarchy_width
        ]
        link(graph, entity.kind + "_parent", entity.kind, entity.kind, parents)
    if entity.client_link:
        clients = sorted(
            {(client, key) for client, key in zip(snapshot.columns["client_key"], keys) if present(key)}
        )
        link(graph, "client_" + entity.kind, "client", entity.kind, clients)


def renewals(graph, snapshot):
    policies = snapshot.columns["policy_key"]
    positions = dict(zip(policies, range(len(policies))))
    pairs = [
        (policy, previous.strip())
        for policy, previous, start in zip(policies, snapshot.columns[RENEWAL], snapshot.starts)
        if previous.strip() in positions
        and previous.strip() != policy
        and snapshot.starts[positions[previous.strip()]] <= start
    ]
    link(graph, "renews", "policy", "policy", pairs)


def truth(table, prediction, known_fields=KNOWN_FIELDS, features=32):
    started = time.perf_counter()
    entities = selected_entities(known_fields, table.column_names)
    snapshot = Snapshot.at(table, prediction, known_fields)
    graph = GraphTables(0, "churn_" + str(prediction), features)
    encode = encoder(features // 2, NUMBERS, ["code"])
    clients, observations = client_observations(snapshot, encode)
    policies = snapshot.columns["policy_key"]
    graph.nodes("client", clients).nodes("policy", policies)
    graph.observe("policy", policies, policy_observations(snapshot, encode))
    graph.observe("client", clients, observations)
    link(graph, "holds", "client", "policy", list(zip(snapshot.columns["client_key"], policies)))
    keys = {entity.field: entity.keys(snapshot.columns) for entity in entities}
    for entity in entities:
        entity_graph(graph, entity, keys[entity.field], snapshot, encode)
    for left, right, relation in RELATIONSHIPS:
        if left in keys and right in keys:
            pairs = sorted(
                {
                    (source, target)
                    for source, target in zip(keys[left], keys[right])
                    if present(source) and present(target)
                }
            )
            if pairs:
                link(graph, relation, BY_FIELD[left].kind, BY_FIELD[right].kind, pairs)
    if RENEWAL in known_fields:
        renewals(graph, snapshot)
    LOG.info(
        "churn truth exit policies=%d clients=%d seconds=%.3f",
        len(policies),
        len(clients),
        time.perf_counter() - started,
    )
    return graph
