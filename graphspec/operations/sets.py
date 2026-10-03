import logging
import time

import numpy as np

from graphspec.facts.stored import sparse
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.declaration import membership, paired
from graphspec.views.spans import counted

LOG = logging.getLogger(__name__)
STORAGE = Names("form of storage of edges joining sets")
NODE = "node"
PARAMETER = "sets/%s/storage"


class Form:
    def __init__(self, function, nodes, members):
        started = time.perf_counter()
        if type(nodes) is not bool:
            raise ValueError(
                "a form of storage states with true or false whether it makes each set edge a node"
            )
        if type(members) is not bool:
            raise ValueError(
                "a form of storage states with true or false whether it shows the values of the members"
            )
        self.function, self.nodes, self.members = function, nodes, members
        LOG.debug("form exit nodes=%s members=%s seconds=%.4f", nodes, members, time.perf_counter() - started)

    @timed("sets.form.call")
    def __call__(self, facts, registry, kind):
        return self.function(facts, registry, kind)


def members(facts, registry, kind):
    started = time.perf_counter()
    kinds = np.asarray(facts.kinds)
    rows = np.flatnonzero(kinds[np.asarray(facts.set_edge, np.int64)] == registry.kind(kind))
    origin = np.flatnonzero(kinds[np.asarray(facts.obs_entity, np.int64)] == registry.kind(kind))
    LOG.debug(
        "members exit graph=%s kind=%s members=%d facts=%d seconds=%.4f",
        facts.name,
        kind,
        len(rows),
        len(origin),
        time.perf_counter() - started,
    )
    return rows, origin


def repeated(keys, owner):
    started = time.perf_counter()
    first = np.searchsorted(keys, owner, "left")
    counts = np.searchsorted(keys, owner, "right") - first
    result = np.repeat(first, counts) + counted(counts), counts
    LOG.debug("repeated exit rows=%d seconds=%.4f", int(counts.sum()), time.perf_counter() - started)
    return result


def added(facts, origin, source, target, code, values):
    started = time.perf_counter()
    count, origin = len(facts.fact_since), np.asarray(origin, np.int64)
    old, new = facts.stamps("fact"), facts.stamps("obs")
    stamps = {name: np.r_[old[name], new[name][origin]] for name in ("since", "until", "known", "replaced")}
    watched = np.asarray(facts.obs_seen_row, np.int64)
    taken, counts = repeated(watched, origin)
    rows = np.r_[np.asarray(facts.fact_seen_row, np.int64), count + np.repeat(np.arange(len(origin)), counts)]
    observers = np.r_[np.asarray(facts.fact_seen_observer), np.asarray(facts.obs_seen_observer)[taken]]
    padded = np.full((len(origin), facts.fact_values.shape[1]), np.nan, np.float32)
    padded[:, : values.shape[1]] = values
    result = facts.replace(
        fact_source=np.r_[facts.fact_source, source].astype(np.int32),
        fact_target=np.r_[facts.fact_target, target].astype(np.int32),
        fact_code=np.r_[facts.fact_code, code].astype(np.int16),
        fact_present=np.r_[facts.fact_present, np.ones(len(origin), bool)],
        fact_since=stamps["since"],
        fact_until=stamps["until"],
        fact_seq=np.r_[facts.fact_seq, np.asarray(facts.obs_seq)[origin]].astype(np.int32),
        fact_values=np.concatenate((facts.fact_values, padded)),
        **sparse("fact", stamps, rows, observers),
    )
    LOG.debug(
        "added exit graph=%s facts=%d seconds=%.4f", facts.name, len(origin), time.perf_counter() - started
    )
    return result


def dropped(facts, keep):
    started = time.perf_counter()
    rows, place = np.flatnonzero(keep), np.cumsum(keep) - 1
    stamps = {name: values[rows] for name, values in facts.stamps("obs").items()}
    watched = np.asarray(facts.obs_seen_row, np.int64)
    chosen = keep[watched]
    result = facts.replace(
        obs_entity=np.asarray(facts.obs_entity)[rows],
        obs_since=stamps["since"],
        obs_until=stamps["until"],
        obs_seq=np.asarray(facts.obs_seq)[rows],
        obs_values=np.asarray(facts.obs_values)[rows],
        **sparse("obs", stamps, place[watched[chosen]], np.asarray(facts.obs_seen_observer)[chosen]),
    )
    LOG.debug(
        "dropped exit graph=%s kept=%d seconds=%.4f", facts.name, len(rows), time.perf_counter() - started
    )
    return result


def node(facts, registry, kind):
    started = time.perf_counter()
    entry, columns = registry.schema.sets[kind], registry.schema.member_columns()
    rows, origin = members(facts, registry, kind)
    owner = np.asarray(facts.obs_entity, np.int64)[origin]
    taken, counts = repeated(np.asarray(facts.set_edge, np.int64)[rows], owner)
    taken = rows[taken]
    roles = np.asarray([registry.code(membership(kind, role)) for role in entry["roles"]], np.int64)
    result = added(
        facts,
        np.repeat(origin, counts),
        np.repeat(owner, counts),
        np.asarray(facts.set_member)[taken],
        roles[np.asarray(facts.set_role, np.int64)[taken]],
        np.asarray(facts.set_values)[taken][:, [columns.index(name) for name in entry["members"]]],
    )
    LOG.info(
        "node form exit graph=%s kind=%s edges=%d seconds=%.3f",
        facts.name,
        kind,
        len(taken),
        time.perf_counter() - started,
    )
    return result


def pairs(facts, registry, kind):
    started = time.perf_counter()
    schema = registry.schema
    about = [
        table for table, entry in schema.edges.items() if any(kind in pair for pair in entry["endpoints"])
    ]
    if about:
        raise ValueError(
            "the form pairs makes no node of the set edges of %s, which the edge tables %s are about; "
            "state for %s a form that makes each set edge a node" % (kind, ", ".join(about), kind)
        )
    rows, origin = members(facts, registry, kind)
    edge = np.asarray(facts.set_edge, np.int64)[rows]
    member = np.asarray(facts.set_member, np.int64)[rows]
    distinct = np.r_[True, (edge[1:] != edge[:-1]) | (member[1:] != member[:-1])][: len(edge)]
    edge, member = edge[distinct], member[distinct]
    later = np.searchsorted(edge, edge, "right") - np.arange(len(edge)) - 1
    left = np.repeat(np.arange(len(edge)), later)
    right = left + counted(later) + 1
    owner = np.asarray(facts.obs_entity, np.int64)[origin]
    taken, counts = repeated(edge[left], owner)
    chosen = np.repeat(origin, counts)
    values = np.asarray(facts.obs_values)[chosen][
        :, registry.encoding.positions(schema.sets[kind]["columns"])
    ]
    result = added(
        facts,
        chosen,
        member[left][taken],
        member[right][taken],
        np.full(len(taken), registry.code(paired(kind)), np.int64),
        values,
    )
    result = dropped(
        result, np.asarray(facts.kinds)[np.asarray(result.obs_entity, np.int64)] != registry.kind(kind)
    )
    LOG.info(
        "pairs form exit graph=%s kind=%s edges=%d seconds=%.3f",
        facts.name,
        kind,
        len(taken),
        time.perf_counter() - started,
    )
    return result


STORAGE.register(NODE, Form(node, True, True))
STORAGE.register("pairs", Form(pairs, False, False))


def offered(registry, kind):
    started = time.perf_counter()
    schema = registry.schema
    about = any(kind in pair for entry in schema.edges.values() for pair in entry["endpoints"])
    stops = any(
        entry["absent"] == "stop"
        and any(kind in schema.tasks["references"][name]["kinds"] for name in entry["references"])
        for entry in schema.tasks["kinds"].values()
    )
    revealed = any(kind in schema.tasks["references"][name]["kinds"] for name in schema.revealing())
    result = tuple(
        name for name in STORAGE.names() if STORAGE.named(name).nodes or not (about or stops or revealed)
    )
    LOG.debug("offered exit kind=%s forms=%d seconds=%.4f", kind, len(result), time.perf_counter() - started)
    return result


def shown(registry, storage):
    started = time.perf_counter()
    schema = registry.schema
    groups = [schema.graph, schema.layout(), schema.inner_columns()]
    groups += [entry["columns"] for entry in schema.edges.values()]
    groups += [entry["members"] for kind, entry in schema.sets.items() if STORAGE.named(storage[kind]).members]
    found = {name for columns in groups for name in columns}
    result = tuple(name for name in schema.columns if name in found)
    LOG.debug(
        "shown exit columns=%d of=%d seconds=%.4f", len(result), len(schema.columns), time.perf_counter() - started
    )
    return result


def chosen(registry, spec):
    return {kind: spec[PARAMETER % kind] for kind in registry.schema.sets}


def stored(facts, registry, storage):
    started = time.perf_counter()
    if sorted(storage) != sorted(registry.schema.sets):
        raise ValueError(
            "a form of storage is stated for every kind of set edge, %s, and is stated for %s"
            % (", ".join(registry.schema.sets) or "none", ", ".join(storage) or "none")
        )
    for kind in registry.schema.sets:
        facts = STORAGE.named(storage[kind])(facts, registry, kind)
    LOG.debug(
        "stored exit graph=%s kinds=%d seconds=%.4f", facts.name, len(storage), time.perf_counter() - started
    )
    return facts
