import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.features import token
from graphspec.infra.files import Mapped, load_arrays, read_json, save_arrays, write_json
from graphspec.infra.timing import timed
from graphspec.infra.workspace import finite

from . import clock, kernels, spans
from .clock import ALWAYS, NEVER
from .schema import REFERENCES, VALUE

LOG = logging.getLogger(__name__)
VERSION = 8
QUERIES = ("query_key", "query_category", "query_references", "query_x", "query_signals")
LABELS = ("label_entity", "label_step", "label_row", "label_id")
OBSERVATIONS = ("obs_entity", "obs_start", "obs_stop", "obs_x")
FACTS = ("fact_source", "fact_target", "fact_code", "fact_start", "fact_stop", "fact_rank", "fact_values")
ORDER = ("order_step", "order_position", "order_span")
REMOVALS = ("removal_source", "removal_target", "removal_code", "removal_step", "removal_values")
STATED = ("obs_values", "query_values", "graph_values")
ARRAYS = (
    ("kinds", "entity_token", "period") + OBSERVATIONS + FACTS + ORDER + REMOVALS + QUERIES + LABELS + STATED
)
DENSE = (
    ("kinds", "entity_token", "entity_connected", "period", "step_graph")
    + ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
    + ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    + ("x_table", "value_table")
    + QUERIES
    + LABELS
)
CHANGES = ("time", "observations", "facts")
GROUPS = ("obs", "fact")
SPARSE = ("known_row", "known_time", "replaced_row", "replaced_time", "seen_row", "seen_observer")
STORED = (
    ("kinds", "entity_token")
    + ("obs_entity", "obs_since", "obs_until", "obs_seq", "obs_values", "obs_x")
    + ("fact_source", "fact_target", "fact_code", "fact_present")
    + ("fact_since", "fact_until", "fact_seq", "fact_values")
    + tuple(group + "_" + name for group in GROUPS for name in SPARSE)
    + ("query_time", "query_observer", "query_grouping")
    + ("query_key", "query_category", "query_references", "query_values", "query_x", "query_signals")
    + ("label_entity", "label_query", "label_time", "label_row", "label_id")
    + ("graph_values",)
)


def admitted(directory, manifest):
    if manifest["version"] != VERSION:
        raise ValueError(
            "bank %s has version %s; this code reads version %d" % (directory, manifest["version"], VERSION)
        )
    return manifest


class Arrays:
    NAMES = ()

    def __init__(self, name, keys, steps, **arrays):
        missing = sorted(set(self.NAMES) - set(arrays))
        if missing:
            raise ValueError("graph %s lacks arrays: %s" % (name, ", ".join(missing)))
        unknown = sorted(set(arrays) - set(self.NAMES))
        if unknown:
            raise ValueError("graph %s has unknown arrays: %s" % (name, ", ".join(unknown)))
        self.__dict__.update(
            name=name, keys=list(keys), steps=int(steps), arrays={key: arrays[key] for key in self.NAMES}
        )

    def __getattr__(self, key):
        try:
            return self.__dict__["arrays"][key]
        except KeyError:
            raise AttributeError(key) from None

    def replace(self, steps=None, **changes):
        return type(self)(
            self.name, self.keys, self.steps if steps is None else steps, **dict(self.arrays, **changes)
        )

    def save(self, directory):
        directory = Path(directory)
        save_arrays(directory, self.arrays)
        write_json(directory / "graph.json", dict(name=self.name, keys=self.keys, steps=self.steps))

    @classmethod
    def load(cls, directory, mapped=False):
        directory = Path(directory)
        meta = read_json(directory / "graph.json")
        if not mapped:
            return cls(meta["name"], meta["keys"], meta["steps"], **load_arrays(directory, cls.NAMES))
        graph = cls.__new__(cls)
        graph.__dict__.update(
            name=meta["name"],
            keys=meta["keys"],
            steps=int(meta["steps"]),
            arrays=Mapped(directory, cls.NAMES),
        )
        return graph


class Graph(Arrays):
    NAMES = ARRAYS

    def dense(self):
        started = time.perf_counter()
        entity, step, reference = spans.observation_rows(
            np.asarray(self.obs_entity), np.asarray(self.obs_start), np.asarray(self.obs_stop)
        )
        facts = spans.fact_rows(
            {
                name: np.asarray(self.arrays[name])
                for name in FACTS + ORDER + REMOVALS
                if "values" not in name
            },
            self.steps,
        )
        facts["fact_code"] = facts["fact_code"].astype(np.int64)
        result = Dense(
            self.name,
            self.keys,
            self.steps,
            kinds=np.asarray(self.kinds),
            entity_token=np.asarray(self.entity_token),
            entity_connected=np.zeros(len(self.kinds), bool),
            period=np.asarray(self.period),
            step_graph=np.zeros(self.steps, np.int32),
            obs_entity=entity,
            obs_step=step,
            obs_ref=reference,
            obs_anchor=np.zeros(len(step), bool),
            x_table=self.obs_x,
            value_table=np.concatenate((self.fact_values, self.removal_values)),
            **facts,
            **{name: np.asarray(self.arrays[name]) for name in QUERIES + LABELS},
        )
        LOG.debug(
            "dense exit graph=%s observations=%d facts=%d seconds=%.4f",
            self.name,
            len(step),
            len(facts["fact_step"]),
            time.perf_counter() - started,
        )
        return result


class Dense(Arrays):
    NAMES = DENSE


def moment(value):
    value = int(value)
    return None if value <= ALWAYS or value >= NEVER else value


class Facts:
    NAMES = STORED

    @timed("truth.facts")
    def __init__(self, name, keys, unit, **arrays):
        missing = sorted(set(self.NAMES) - set(arrays))
        if missing:
            raise ValueError("graph %s lacks arrays: %s" % (name, ", ".join(missing)))
        unknown = sorted(set(arrays) - set(self.NAMES))
        if unknown:
            raise ValueError("graph %s has unknown arrays: %s" % (name, ", ".join(unknown)))
        self.__dict__.update(
            name=name, keys=list(keys), unit=unit, arrays={key: arrays[key] for key in self.NAMES}
        )

    def __getattr__(self, key):
        try:
            return self.__dict__["arrays"][key]
        except KeyError:
            raise AttributeError(key) from None

    def save(self, directory):
        started = time.perf_counter()
        directory = Path(directory)
        save_arrays(directory, self.arrays)
        write_json(directory / "graph.json", dict(name=self.name, keys=self.keys, unit=self.unit))
        LOG.debug("facts save exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)

    @classmethod
    def load(cls, directory):
        started = time.perf_counter()
        directory = Path(directory)
        meta = read_json(directory / "graph.json")
        result = cls(meta["name"], meta["keys"], meta["unit"], **load_arrays(directory, cls.NAMES))
        LOG.debug("facts load exit graph=%s seconds=%.4f", result.name, time.perf_counter() - started)
        return result

    def stamps(self, group):
        started = time.perf_counter()
        since = np.asarray(self.arrays[group + "_since"], np.int64)
        result = dict(
            since=since,
            until=np.asarray(self.arrays[group + "_until"], np.int64),
            known=clock.filled(
                len(since), since, self.arrays[group + "_known_row"], self.arrays[group + "_known_time"]
            ),
            replaced=clock.filled(
                len(since), NEVER, self.arrays[group + "_replaced_row"], self.arrays[group + "_replaced_time"]
            ),
        )
        LOG.debug(
            "facts stamps exit graph=%s group=%s seconds=%.4f",
            self.name,
            group,
            time.perf_counter() - started,
        )
        return result

    def watchers(self, group, observers):
        started = time.perf_counter()
        found = [None] * len(self.arrays[group + "_since"])
        for row, code in zip(
            np.asarray(self.arrays[group + "_seen_row"]).tolist(),
            np.asarray(self.arrays[group + "_seen_observer"]).tolist(),
        ):
            found[row] = (found[row] or []) + [observers[code]]
        LOG.debug(
            "facts watchers exit graph=%s group=%s seconds=%.4f",
            self.name,
            group,
            time.perf_counter() - started,
        )
        return found

    def timed(self, group, rows, observers, sequenced):
        started = time.perf_counter()
        stamps, watchers = self.stamps(group), self.watchers(group, observers)
        result = dict(
            since=[moment(value) for value in stamps["since"][rows]],
            until=[moment(value) for value in stamps["until"][rows]],
            known=[
                None if known == since else moment(known)
                for known, since in zip(stamps["known"][rows], stamps["since"][rows])
            ],
            replaced=[moment(value) for value in stamps["replaced"][rows]],
            observers=[watchers[row] for row in np.flatnonzero(rows).tolist()],
        )
        if sequenced:
            result["seq"] = np.asarray(self.arrays[group + "_seq"])[rows].tolist()
        LOG.debug(
            "facts timed exit graph=%s group=%s seconds=%.4f",
            self.name,
            group,
            time.perf_counter() - started,
        )
        return result

    def stated(self, registry, categories, observers):
        started = time.perf_counter()
        schema, encoding, keys = registry.schema, registry.encoding, np.asarray(self.keys, object)
        kinds, entity = np.asarray(self.kinds), np.asarray(self.obs_entity)
        sequenced = schema.ties is not None and "seq" in clock.TIES.named(schema.ties).columns
        values = np.asarray(self.obs_values)
        observations = {}
        for kind, columns in schema.nodes.items():
            rows = kinds[entity] == registry.kind(kind)
            observations[kind] = dict(
                key=keys[entity[rows]].tolist(),
                **self.timed("obs", rows, observers, sequenced),
                **prefixed(stated(registry, columns, values[rows][:, encoding.positions(columns)])),
            )
        names = np.asarray([edge.name for edge in registry.edges], object)
        code, held = np.asarray(self.fact_code).astype(np.int64), np.asarray(self.fact_values)
        edges = {}
        for table, entry in schema.edges.items():
            rows = np.isin(code, [registry.code(name) for name in encoding.relations[table]])
            edges[table] = dict(
                source=keys[np.asarray(self.fact_source)[rows]].tolist(),
                target=keys[np.asarray(self.fact_target)[rows]].tolist(),
                present=np.asarray(self.fact_present)[rows].tolist(),
                relation=names[code[rows]].tolist(),
                **self.timed("fact", rows, observers, sequenced),
                **prefixed(stated(registry, entry["columns"], held[rows])),
            )
        queries = None
        if schema.queries is not None:
            asked = schema.queries["columns"]
            references = np.asarray(self.query_references)
            queries = dict(
                time=[
                    None if schema.model == "none" else moment(value) for value in np.asarray(self.query_time)
                ],
                observer=[
                    None if value < 0 else observers[value]
                    for value in np.asarray(self.query_observer).tolist()
                ],
                key=np.asarray(self.query_key).tolist(),
                category=[categories[value] for value in np.asarray(self.query_category).tolist()],
                references={
                    name: keys[references[:, place]].tolist()
                    for place, name in enumerate(schema.queries["references"])
                },
                **prefixed(
                    stated(registry, asked, np.asarray(self.query_values)[:, encoding.positions(asked)])
                ),
            )
            if schema.queries["grouping"] is not None:
                queries["grouping"] = np.asarray(self.query_grouping).tolist()
        result = dict(
            graph=dict(
                unit=self.unit,
                **{
                    name: found[0]
                    for name, found in stated(
                        registry, schema.graph, np.asarray(self.graph_values)[None, :]
                    ).items()
                },
            ),
            nodes={kind: keys[kinds == registry.kind(kind)].tolist() for kind in schema.nodes},
            observations=observations,
            edges=edges,
            queries=queries,
        )
        LOG.debug("stated exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result


def tokens(registry, kinds, keys):
    return np.asarray(
        [token(registry.nodes[kind] + ":" + key) for kind, key in zip(np.asarray(kinds).tolist(), keys)],
        np.uint64,
    )


def column(table, name, dtype):
    return table.column(name).to_numpy().astype(dtype)


def matrix(table, name, dtype):
    values = table.column(name).combine_chunks()
    return values.flatten().to_numpy().reshape(len(values), values.type.list_size).astype(dtype)


def codes(registry, table, names, positions, width):
    started = time.perf_counter()
    schema, dictionaries = registry.schema, registry.encoding.dictionaries
    values = np.full((table.num_rows, width), np.nan, np.float32)
    for name, position in zip(names, positions):
        kind = schema.kind(name)
        values[:, position] = kind.codes(
            kind.read(table.column(VALUE + name)), dictionaries[name] if kind.symbols else None
        )
    LOG.debug(
        "codes exit rows=%d columns=%d seconds=%.4f", len(values), len(names), time.perf_counter() - started
    )
    return values


def stated(registry, names, values):
    started = time.perf_counter()
    schema, dictionaries = registry.schema, registry.encoding.dictionaries
    result = {
        name: schema.kind(name).stated(
            values[:, position], dictionaries[name] if schema.kind(name).symbols else None
        )
        for position, name in enumerate(names)
    }
    LOG.debug(
        "stated exit rows=%d columns=%d seconds=%.4f", len(values), len(names), time.perf_counter() - started
    )
    return result


def gather(schema, tables, until):
    started = time.perf_counter()
    facts = ("since", "known")
    places = [(tables["graph"], schema.graph, clock.reached(tables["graph"], (), until))]
    for kind, columns in schema.nodes.items():
        table = tables["observations"][kind]
        places.append((table, columns, clock.reached(table, facts, until)))
    for held, entry in schema.edges.items():
        table = tables["edges"][held]
        places.append((table, entry["columns"], clock.reached(table, facts, until)))
    if schema.queries is not None:
        table = tables["queries"]
        places.append((table, schema.queries["columns"], clock.reached(table, ("time",), until)))
    found = {name: dict(trained=[], distinct=[]) for name in schema.columns}
    for table, columns, seen in places:
        for name in columns:
            kind = schema.kind(name)
            values = kind.read(table.column(VALUE + name))
            known = ~kind.missing(values)
            found[name]["trained"].append(values[known & seen])
            if kind.symbols:
                found[name]["distinct"].append(np.unique(values[known]))
    result = {
        name: {
            part: np.concatenate(values) if values else np.zeros(0, np.float32)
            for part, values in parts.items()
        }
        for name, parts in found.items()
    }
    LOG.debug("gather exit columns=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def prefixed(columns):
    return {VALUE + name: values for name, values in columns.items()}


def encoded(registry, values):
    return registry.encoding.nodes(values).astype(np.float16)


@timed("truth.sparse")
def sparse(prefix, stamps, rows, observers):
    known = clock.differing(stamps["known"], stamps["since"])
    replaced = clock.differing(stamps["replaced"], NEVER)
    order = np.lexsort((observers, rows))
    return {
        prefix + "_known_row": known[0],
        prefix + "_known_time": known[1],
        prefix + "_replaced_row": replaced[0],
        prefix + "_replaced_time": replaced[1],
        prefix + "_seen_row": np.asarray(rows, np.int64)[order],
        prefix + "_seen_observer": np.asarray(observers, np.int32)[order],
    }


def gathered(parts, width):
    started = time.perf_counter()
    offsets = np.r_[0, np.cumsum([len(part["since"]) for part in parts])].astype(np.int64)
    result = {
        name: np.concatenate([part[name] for part in parts] + [np.zeros(0, np.int64)])
        for name in ("since", "until", "known", "replaced", "seq")
    }
    result["values"] = np.concatenate([part["values"] for part in parts] + [np.zeros((0, width), np.float32)])
    result["watched"] = np.concatenate(
        [part["watched"][0] + offset for part, offset in zip(parts, offsets)] + [np.zeros(0, np.int64)]
    )
    result["watchers"] = np.concatenate([part["watched"][1] for part in parts] + [np.zeros(0, np.int32)])
    LOG.debug("gathered exit parts=%d seconds=%.4f", len(parts), time.perf_counter() - started)
    return result


def arranged(held, order):
    started = time.perf_counter()
    place = np.empty(len(order), np.int64)
    place[order] = np.arange(len(order))
    result = {name: values[order] for name, values in held.items() if name not in ("watched", "watchers")}
    result["watched"], result["watchers"] = place[held["watched"]], held["watchers"]
    LOG.debug("arranged exit rows=%d seconds=%.4f", len(order), time.perf_counter() - started)
    return result


def read(schema, observers, table, label, name):
    started = time.perf_counter()
    sequenced = schema.ties is not None and "seq" in clock.TIES.named(schema.ties).columns
    result = clock.stamped(table, schema.model, label, name)
    result["seq"] = column(table, "seq", np.int64) if sequenced else np.zeros(table.num_rows, np.int64)
    result["watched"] = clock.watched(table, observers, label, name)
    LOG.debug("read exit graph=%s rows=%d seconds=%.4f", name, table.num_rows, time.perf_counter() - started)
    return result


def parse(registry, categories, observers, tables, rows, name):
    started = time.perf_counter()
    schema, encoding = registry.schema, registry.encoding
    unit = None
    if schema.model in clock.UNITED:
        unit = tables["graph"].column("unit").to_pylist()[0]
        if unit not in schema.units:
            raise ValueError(
                "graph %s states the unit of time %r; the schema states %s"
                % (name, unit, ", ".join(schema.units))
            )
    nodes = tables["nodes"]
    numbers = column(nodes, "entity", np.int64)
    order = np.argsort(numbers, kind="stable")
    ordered_numbers = numbers[order]
    if np.any(np.diff(ordered_numbers) == 0):
        raise ValueError("two nodes of graph %s share an entity number" % name)

    def local(values):
        position = np.searchsorted(ordered_numbers, values)
        if np.any(position >= len(ordered_numbers)) or not np.array_equal(ordered_numbers[position], values):
            raise ValueError("a row of graph %s names an entity outside the graph" % name)
        return position.astype(np.int32)

    kinds = np.asarray([registry.kind(kind) for kind in nodes.column("kind").to_pylist()], np.int32)[order]
    names = nodes.column("key").to_pylist()
    keys = [names[i] for i in order.tolist()]
    layout = encoding.layout
    parts = []
    for kind in registry.nodes:
        table = tables["observations"][kind]
        owners = local(column(table, "entity", np.int64))
        if np.any(kinds[owners] != registry.kind(kind)):
            raise ValueError(
                "an observation of graph %s of kind %s names a node of another kind" % (name, kind)
            )
        part = read(schema, observers, table, "fact about a node", name)
        part["entity"] = owners
        part["values"] = codes(
            registry, table, schema.nodes[kind], encoding.positions(schema.nodes[kind]), len(layout)
        )
        parts.append(part)
    seen = gathered(parts, len(layout))
    seen["entity"] = np.concatenate([part["entity"] for part in parts] + [np.zeros(0, np.int32)])
    seen = arranged(seen, np.lexsort((seen["seq"], seen["until"], seen["entity"], seen["since"])))
    width = max([len(registry.attributes(relation)) for relation in registry.extras["base"]] + [0])
    parts = []
    for held, entry in schema.edges.items():
        table = tables["edges"][held]
        count = table.num_rows
        if entry["column"] is None:
            relation = np.full(count, registry.code(held), np.int64)
        else:
            known = {found: registry.code(found) for found in encoding.relations[held]}
            found = table.column("relation").to_pylist()
            strangers = sorted({str(value) for value in found if value not in known})
            if strangers:
                raise ValueError(
                    "graph %s names relations of %s that the bank does not hold: %s"
                    % (name, held, ", ".join(strangers))
                )
            relation = np.asarray([known[value] for value in found], np.int64)
        source = local(column(table, "source", np.int64))
        target = local(column(table, "target", np.int64))
        if not entry["directed"]:
            swap = (kinds[source] == kinds[target]) & (source > target)
            source, target = np.where(swap, target, source), np.where(swap, source, target)
        part = read(schema, observers, table, "fact about an edge", name)
        part.update(source=source, target=target, code=relation, present=column(table, "present", bool))
        part["values"] = codes(registry, table, entry["columns"], range(len(entry["columns"])), width)
        parts.append(part)
    facts = gathered(parts, width)
    for key, kind in (("source", np.int32), ("target", np.int32), ("code", np.int64), ("present", bool)):
        facts[key] = np.concatenate([part[key] for part in parts] + [np.zeros(0, kind)])
    facts = arranged(
        facts,
        np.lexsort(
            (
                facts["until"],
                facts["target"],
                facts["source"],
                facts["code"],
                facts["seq"],
                ~facts["present"],
                facts["since"],
            )
        ),
    )
    for relation in registry.attributed:
        edge = registry.by_name[relation]
        chosen = facts["present"] & (facts["code"] == edge.code)
        if not np.isfinite(facts["values"][chosen, : len(edge.attributes)]).all():
            raise ValueError(
                "graph %s has a missing or infinite %s value on a present fact" % (name, relation)
            )
    if len(facts["code"]) and (facts["code"].min() < 0 or facts["code"].max() > np.iinfo(np.int16).max):
        raise ValueError("fact codes must fit 0..%d" % np.iinfo(np.int16).max)
    level = registry.extras["level"]
    timed = schema.model != "none"
    if registry.extras["queries"]:
        queries = tables["queries"]
        count = queries.num_rows
        if timed and queries.column("time").null_count:
            raise ValueError("a task of graph %s states no time" % name)
        query_time = column(queries, "time", np.int64) if timed else np.zeros(count, np.int64)
        if np.any(np.diff(query_time) < 0):
            raise ValueError("the tasks of graph %s are not read in order of their time" % name)
        if np.any((query_time <= ALWAYS) | (query_time >= NEVER)):
            raise ValueError("a task of graph %s states a time outside the range of times" % name)
        query_observer = np.full(count, -1, np.int32)
        if schema.observers:
            known = {found: code for code, found in enumerate(observers)}
            looking = queries.column("observer").to_pylist()
            strangers = sorted({value for value in looking if value is not None and value not in known})
            if strangers:
                raise ValueError(
                    "tasks of graph %s name observers the bank does not hold: %s"
                    % (name, ", ".join(strangers))
                )
            query_observer = np.asarray(
                [-1 if value is None else known[value] for value in looking], np.int32
            ).reshape(-1)
        position = {category: index for index, category in enumerate(categories)}
        asked = schema.queries["columns"]
        query_values = codes(registry, queries, asked, encoding.positions(asked), len(layout))
        query = dict(
            query_time=query_time,
            query_observer=query_observer,
            query_grouping=column(queries, "period", np.int32)
            if schema.queries["grouping"] is not None
            else np.zeros(count, np.int32),
            query_key=column(queries, "key", np.int64),
            query_category=np.asarray(
                [position[value] for value in queries.column("category").to_pylist()], np.int64
            ),
            query_references=local(matrix(queries, "references", np.int64).reshape(-1)).reshape(
                count, REFERENCES
            ),
            query_x=encoded(registry, query_values),
            query_signals=np.stack(
                (matrix(queries, "signal", np.float32), matrix(queries, "signal_observed", np.float32)), 2
            )
            if registry.extras["signals"]
            else np.zeros((count, 0, 2), np.float32),
        )
    else:
        query_values = np.zeros((0, len(layout)), np.float32)
        query = dict(
            query_time=np.zeros(0, np.int64),
            query_observer=np.zeros(0, np.int32),
            query_grouping=np.zeros(0, np.int32),
            query_key=np.zeros(0, np.int64),
            query_category=np.zeros(0, np.int64),
            query_references=np.zeros((0, REFERENCES), np.int32),
            query_x=np.zeros((0, registry.features), np.float16),
            query_signals=np.zeros((0, 0, 2), np.float32),
        )
    if level == "node":
        targets = tables["targets"]
        entity = local(column(targets, "entity", np.int64))
        if timed and targets.column("time").null_count:
            raise ValueError("a label of graph %s states no time" % name)
        moments = column(targets, "time", np.int64) if timed else np.zeros(len(entity), np.int64)
        order = np.lexsort((entity, moments))
        label_entity, label_time = entity[order], moments[order]
        if np.any((np.diff(label_time) == 0) & (np.diff(label_entity) == 0)):
            raise ValueError("two labels of graph %s sit on one node at one time" % name)
        label_row, label_id = np.asarray(rows, np.int64)[order], column(targets, "id", np.int64)[order]
        label_query = np.full(len(label_entity), -1, np.int32)
    else:
        if level == "path":
            label_query, label_row = (np.asarray(values, np.int64) for values in rows)
            if np.any(np.diff(label_query) < 0):
                raise ValueError("points of graph %s are not ordered by their task" % name)
        else:
            rows = np.asarray(rows, np.int64)
            label_query = np.flatnonzero(rows >= 0)
            label_row = rows[label_query]
        label_id, label_time = query["query_key"][label_query], query["query_time"][label_query]
        label_query = label_query.astype(np.int32)
        label_entity = np.full(len(label_query), -1, np.int32)
    graph = Facts(
        name,
        keys,
        unit,
        kinds=kinds,
        entity_token=tokens(registry, kinds, keys),
        obs_entity=seen["entity"].astype(np.int32),
        obs_since=seen["since"],
        obs_until=seen["until"],
        obs_seq=seen["seq"].astype(np.int32),
        obs_values=seen["values"],
        obs_x=encoded(registry, seen["values"]),
        **sparse("obs", seen, seen["watched"], seen["watchers"]),
        fact_source=facts["source"].astype(np.int32),
        fact_target=facts["target"].astype(np.int32),
        fact_code=facts["code"].astype(np.int16),
        fact_present=facts["present"],
        fact_since=facts["since"],
        fact_until=facts["until"],
        fact_seq=facts["seq"].astype(np.int32),
        fact_values=facts["values"],
        **sparse("fact", facts, facts["watched"], facts["watchers"]),
        label_entity=label_entity,
        label_query=label_query,
        label_time=label_time.astype(np.int64),
        label_row=label_row,
        label_id=label_id,
        query_values=query_values,
        graph_values=codes(
            registry, tables["graph"], schema.graph, range(len(schema.graph)), len(schema.graph)
        )[0],
        **query,
    )
    check(graph, registry, observers)
    LOG.info(
        "truth exit graph=%s seconds=%.3f observations=%d facts=%d tasks=%d labels=%d",
        name,
        time.perf_counter() - started,
        len(graph.obs_since),
        len(graph.fact_since),
        len(graph.query_time),
        len(graph.label_time),
    )
    return graph


@timed("truth.ranks")
def ranks(keys):
    order = sorted(range(len(keys)), key=keys.__getitem__)
    result = np.empty(len(keys), np.int64)
    result[order] = np.arange(len(keys))
    return result


def check(graph, registry, observers):
    started = time.perf_counter()
    name, entities, schema = graph.name, len(graph.kinds), registry.schema
    lengths = dict(
        obs=len(graph.obs_since),
        fact=len(graph.fact_since),
        query=len(graph.query_key),
        label=len(graph.label_time),
    )
    dimensions = dict(
        obs_x=2,
        obs_values=2,
        fact_values=2,
        query_x=2,
        query_values=2,
        query_signals=3,
        query_references=2,
    )
    loose = tuple(group + "_" + part for group in GROUPS for part in SPARSE)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and key not in loose and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if (
            key not in dimensions
            and key not in ("fact_present", "graph_values")
            and values.dtype.kind not in "iu"
        ):
            raise ValueError("graph %s requires whole numbers in %s" % (name, key))
    if (schema.model in clock.UNITED) != (graph.unit is not None):
        raise ValueError("graph %s states a unit of time exactly when its schema has a clock" % name)
    if graph.unit is not None and graph.unit not in schema.units:
        raise ValueError(
            "graph %s states the unit of time %r, which its schema does not" % (name, graph.unit)
        )
    if graph.query_references.shape[1] != REFERENCES:
        raise ValueError("graph %s requires %d references for every query" % (name, REFERENCES))
    if len(graph.keys) != entities or len(graph.entity_token) != entities:
        raise ValueError("graph %s requires one key and token per entity" % name)
    if not np.array_equal(graph.entity_token, tokens(registry, graph.kinds, graph.keys)):
        raise ValueError("entity tokens of graph %s do not match its keys" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.obs_x.shape[1] != registry.features or graph.query_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.query_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if not finite(graph.obs_x) or not finite(graph.query_x):
        raise ValueError("graph %s observation and query features must be finite" % name)
    for given, values in ((graph.obs_x, graph.obs_values), (graph.query_x, graph.query_values)):
        if not np.array_equal(np.asarray(given), encoded(registry, np.asarray(values))):
            raise ValueError("graph %s holds node values that its encoders do not reproduce" % name)
    if graph.graph_values.shape != (len(schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    for label, values in (
        ("fact about a node", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
    ) + tuple(zip(referenced(registry), np.asarray(graph.query_references).T)):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if registry.extras["level"] == "node" and (
        np.any(graph.label_entity < 0) or np.any(graph.label_entity >= entities)
    ):
        raise ValueError("a label of graph %s names an entity outside 0..%d" % (name, entities - 1))
    found = np.asarray(graph.fact_code)
    if np.any(found < 0) or np.any(found >= len(registry.canonical)) or not registry.canonical[found].all():
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    if np.any(graph.label_query >= len(graph.query_key)):
        raise ValueError("a label of graph %s names a task outside the graph" % name)
    if np.any(graph.query_observer < -1) or np.any(graph.query_observer >= len(observers)):
        raise ValueError("a task of graph %s names an observer the bank does not hold" % name)
    if np.any(np.diff(graph.query_time) < 0) or np.any(np.diff(graph.label_time) < 0):
        raise ValueError("tasks and labels of graph %s are not ordered by their time" % name)
    place = ranks(graph.keys)
    for group, label in (("obs", "facts about one node"), ("fact", "facts about edges")):
        stamps = graph.stamps(group)
        if np.any(stamps["until"] <= stamps["since"]) or np.any(stamps["replaced"] <= stamps["known"]):
            raise ValueError("a fact of graph %s holds, or is known, over no time at all" % name)
        rows, codes = graph.arrays[group + "_seen_row"], graph.arrays[group + "_seen_observer"]
        if (
            np.any(rows < 0)
            or np.any(rows >= lengths[group])
            or np.any(codes < 0)
            or np.any(codes >= len(observers))
        ):
            raise ValueError("a fact of graph %s names an observer the bank does not hold" % name)
        if group == "obs":
            owner = np.asarray(graph.obs_entity, np.int64)
            facts = dict(seq=graph.obs_seq, code=graph.kinds[owner], source=place[owner], target=place[owner])
        else:
            owner = np.asarray(graph.fact_present, np.int64)
            facts = dict(
                seq=graph.fact_seq,
                code=graph.fact_code,
                source=place[graph.fact_source],
                target=place[graph.fact_target],
            )
        clock.untied(name, label, schema.ties, owner, stamps["since"], facts)
    LOG.debug("check exit graph=%s seconds=%.4f", name, time.perf_counter() - started)


def conforms(graph, registry, changed):
    started = time.perf_counter()
    unknown = sorted(set(changed) - set(CHANGES))
    if unknown:
        raise ValueError("an operation states changes the schema does not know: " + ", ".join(unknown))
    name, kinds = graph.name, np.asarray(graph.kinds)
    if "time" in changed and len(graph.period) != graph.steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if "observations" in changed:
        if graph.x_table.shape[1] != registry.features:
            raise ValueError("graph %s holds node values of a width its schema does not state" % name)
        if np.any(kinds < 0) or np.any(kinds >= len(registry.nodes)):
            raise ValueError("graph %s names kinds of node its schema does not state" % name)
    if "facts" in changed:
        found = np.asarray(graph.fact_code)
        if np.any(found < 0) or np.any(found >= registry.relations) or not registry.canonical[found].all():
            raise ValueError("graph %s holds facts of relations its schema does not state" % name)
        pairs = kinds[graph.fact_source].astype(np.int64) * len(registry.nodes) + kinds[graph.fact_target]
        for code in np.unique(found).tolist():
            edge = registry.by_code[code]
            allowed = {(registry.kind(source), registry.kind(target)) for source, target in edge.pairs}
            if edge.symmetric:
                allowed |= {(target, source) for source, target in allowed}
            allowed = np.asarray([source * len(registry.nodes) + target for source, target in allowed])
            if not np.isin(pairs[found == code], allowed).all():
                raise ValueError(
                    "graph %s holds a fact of %s between kinds of node the relation does not join"
                    % (name, edge.name)
                )
    LOG.debug(
        "conforms exit graph=%s changed=%s seconds=%.4f",
        name,
        "+".join(changed),
        time.perf_counter() - started,
    )


def verify_dense(graph, registry, known=None):
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_step),
        fact=len(graph.fact_step),
        query=len(graph.query_key),
        label=len(graph.label_step),
    )
    dimensions = dict(x_table=2, value_table=2, query_x=2, query_signals=3, query_references=2)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if (
            key not in dimensions
            and key not in ("obs_anchor", "fact_present", "entity_connected")
            and values.dtype.kind not in "iu"
        ):
            raise ValueError("graph %s requires integer indices in %s" % (name, key))
    if graph.obs_anchor.dtype != np.bool_ or graph.fact_present.dtype != np.bool_:
        raise ValueError("graph %s requires boolean anchor and presence arrays" % name)
    if (
        np.any(graph.obs_ref < 0)
        or np.any(graph.obs_ref >= len(graph.x_table))
        or np.any(graph.fact_ref < -1)
        or np.any(graph.fact_ref >= len(graph.value_table))
    ):
        raise ValueError("graph %s references rows outside its feature or value tables" % name)
    if graph.query_references.dtype.kind not in "iu" or graph.query_references.shape[1] != REFERENCES:
        raise ValueError("graph %s requires %d integer references for every query" % (name, REFERENCES))
    if len(graph.entity_token) != entities or len(graph.entity_connected) != entities or steps < 1:
        raise ValueError("graph %s requires one token and flag per entity and at least one step" % name)
    if len(graph.step_graph) != steps or np.any(np.diff(graph.step_graph) < 0):
        raise ValueError("graph %s needs one joined graph per step, never decreasing" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.x_table.shape[1] != registry.features or graph.query_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.query_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if not finite(graph.x_table) or not finite(graph.query_x):
        raise ValueError("graph %s observation and query features must be finite" % name)
    flags = kernels.audit(
        graph.obs_entity,
        graph.obs_step,
        graph.obs_anchor,
        graph.fact_source,
        graph.fact_target,
        graph.fact_step,
        graph.fact_present,
        graph.fact_code,
        registry.canonical,
        *(np.ascontiguousarray(graph.query_references[:, place]) for place in range(REFERENCES)),
        graph.label_entity,
        graph.label_step,
        registry.extras["level"] == "node",
        steps,
        entities,
        np.zeros(entities, bool) if known is None else known,
    )
    for index, label in enumerate(("observation", "fact", "label")):
        if flags[index]:
            raise ValueError("a %s row of graph %s has a step outside 0..%d" % (label, name, steps - 1))
    for index, label in enumerate(
        ("observation", "fact source", "fact target") + referenced(registry) + ("label",), 3
    ):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if flags[10]:
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    if registry.extras["queries"] and len(graph.query_key) != steps:
        raise ValueError("graph %s needs exactly one query per step" % name)
    for index, message in (
        (11, "observations of graph %s are not ordered by step"),
        (12, "facts of graph %s are not ordered by step with present facts first"),
        (13, "labels of graph %s are not ordered by step"),
        (14, "an entity of graph %s has two observations at one step"),
    ):
        if flags[index]:
            raise ValueError(message % name)
    for index, label, when in (
        (15, "present fact", "outside query anchors at its step"),
        (16, "absent fact", "at or before its step"),
        (17, "query", "at its step"),
        (18, "label", "at its step"),
    ):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity with no observation %s" % (label, name, when))


def referenced(registry):
    if registry.schema.queries is None:
        return ("query reference",) * REFERENCES
    return tuple("query reference " + name for name in registry.schema.queries["references"])


def ordered(*keys):
    if len(keys[0]) < 2:
        return True
    order = np.lexsort(keys[::-1])
    return np.array_equal(order, np.arange(len(order)))


def verify(graph, registry):
    started = time.perf_counter()
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_start),
        fact=len(graph.fact_start),
        removal=len(graph.removal_step),
        query=len(graph.query_key),
        label=len(graph.label_step),
    )
    dimensions = dict(
        obs_x=2,
        obs_values=2,
        fact_values=2,
        removal_values=2,
        query_x=2,
        query_values=2,
        query_signals=3,
        query_references=2,
        graph_values=1,
    )
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if key not in dimensions and values.dtype.kind not in "iu":
            raise ValueError("graph %s requires integer indices in %s" % (name, key))
    if graph.query_references.dtype.kind not in "iu" or graph.query_references.shape[1] != REFERENCES:
        raise ValueError("graph %s requires %d integer references for every query" % (name, REFERENCES))
    if len(graph.keys) != entities or len(graph.entity_token) != entities or steps < 1:
        raise ValueError("graph %s requires one key and token per entity and at least one step" % name)
    if not np.array_equal(graph.entity_token, tokens(registry, graph.kinds, graph.keys)):
        raise ValueError("entity tokens of graph %s do not match its keys" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.obs_x.shape[1] != registry.features or graph.query_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.query_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if graph.fact_values.shape[1] != graph.removal_values.shape[1]:
        raise ValueError("graph %s fact and removal attribute widths differ" % name)
    if not finite(graph.obs_x) or not finite(graph.query_x):
        raise ValueError("graph %s observation and query features must be finite" % name)
    for given, values in ((graph.obs_x, graph.obs_values), (graph.query_x, graph.query_values)):
        if not np.array_equal(np.asarray(given), encoded(registry, np.asarray(values))):
            raise ValueError("graph %s holds node values that its encoders do not reproduce" % name)
    if graph.graph_values.shape != (len(registry.schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    for label, start, stop in (
        ("observation", graph.obs_start, graph.obs_stop),
        ("fact", graph.fact_start, graph.fact_stop),
    ):
        if np.any(start < 0) or np.any(stop <= start) or np.any(stop > steps):
            raise ValueError("a %s span of graph %s lies outside 0..%d or is empty" % (label, name, steps))
    for label, moment in (("removal", graph.removal_step), ("label", graph.label_step)):
        if np.any(moment < 0) or np.any(moment >= steps):
            raise ValueError("a %s row of graph %s has a step outside 0..%d" % (label, name, steps - 1))
    for label, values in (
        ("observation", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
        ("removal source", graph.removal_source),
        ("removal target", graph.removal_target),
    ) + tuple(zip(referenced(registry), np.asarray(graph.query_references).T)):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    labelled = registry.extras["level"] == "node"
    if labelled and (np.any(graph.label_entity < 0) or np.any(graph.label_entity >= entities)):
        raise ValueError("a label of graph %s names an entity outside 0..%d" % (name, entities - 1))
    for label, codes in (("facts", graph.fact_code), ("removals", graph.removal_code)):
        if (
            np.any(codes < 0)
            or np.any(codes >= len(registry.canonical))
            or not registry.canonical[codes].all()
        ):
            raise ValueError(
                "%s of graph %s carry codes that are not canonical relation codes" % (label, name)
            )
    if not ordered(graph.obs_start, graph.obs_entity):
        raise ValueError("observation spans of graph %s are not ordered by start and entity" % name)
    if not np.array_equal(np.sort(graph.fact_rank), np.arange(len(graph.fact_rank))):
        raise ValueError("fact ranks of graph %s are not a permutation" % name)
    if not ordered(graph.fact_start, graph.fact_rank):
        raise ValueError("fact spans of graph %s are not ordered by start and rank" % name)
    if np.any(np.diff(graph.removal_step) < 0):
        raise ValueError("removals of graph %s are not ordered by step" % name)
    moments, positions, moved = (np.asarray(graph.arrays[key]) for key in ORDER)
    change = np.zeros(steps + 1, np.int64)
    np.add.at(change, np.asarray(graph.fact_start), 1)
    np.add.at(change, np.asarray(graph.fact_stop), -1)
    active = np.cumsum(change)[:steps]
    if len(moments) and (
        np.any(np.diff(moments) < 0)
        or np.any(moments < 0)
        or np.any(moments >= steps)
        or np.any(moved < 0)
        or np.any(moved >= len(graph.fact_start))
        or np.any(positions < 0)
        or np.any(positions >= active[moments])
        or np.any(graph.fact_start[moved] > moments)
        or np.any(graph.fact_stop[moved] <= moments)
        or len(np.unique(moments.astype(np.int64) * (1 << 32) + positions)) != len(moments)
        or len(np.unique(moments.astype(np.int64) * (1 << 32) + moved)) != len(moments)
    ):
        raise ValueError("fact order patches of graph %s are malformed" % name)
    if np.any(np.diff(graph.label_step) < 0):
        raise ValueError("labels of graph %s are not ordered by step" % name)
    order = np.lexsort((graph.obs_start, graph.obs_entity))
    entity, start, stop = graph.obs_entity[order], graph.obs_start[order], graph.obs_stop[order]
    same = entity[1:] == entity[:-1]
    if np.any(same & (start[1:] < stop[:-1])):
        raise ValueError("an entity of graph %s has two observations at one step" % name)
    payload = spans.flat(np.asarray(graph.obs_values)[order])
    if np.any(same & (start[1:] == stop[:-1]) & np.all(payload[1:] == payload[:-1], axis=1)):
        raise ValueError("observation spans of graph %s are not maximal" % name)
    offsets, lows, highs = spans.coverage(
        np.asarray(graph.obs_entity), np.asarray(graph.obs_start), np.asarray(graph.obs_stop), entities
    )
    for who in (graph.fact_source, graph.fact_target):
        if spans.all_covered(offsets, lows, highs, np.asarray(who), graph.fact_start, graph.fact_stop) >= 0:
            raise ValueError(
                "a present fact of graph %s names an entity with no observation at its step" % name
            )
    first = np.full(entities, steps, np.int64)
    np.minimum.at(first, graph.obs_entity, graph.obs_start)
    if np.any(first[graph.removal_source] > graph.removal_step) or np.any(
        first[graph.removal_target] > graph.removal_step
    ):
        raise ValueError(
            "an absent fact of graph %s names an entity with no observation at or before its step" % name
        )
    moments = np.arange(len(graph.query_key))
    for who in np.asarray(graph.query_references).T:
        if spans.all_covered(offsets, lows, highs, np.ascontiguousarray(who), moments, moments + 1) >= 0:
            raise ValueError("a query of graph %s names an entity with no observation at its step" % name)
    if labelled:
        moments = np.asarray(graph.label_step)
        if spans.all_covered(offsets, lows, highs, np.asarray(graph.label_entity), moments, moments + 1) >= 0:
            raise ValueError("a label of graph %s names an entity with no observation at its step" % name)
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if registry.extras["queries"] and len(graph.query_key) != steps:
        raise ValueError("graph %s needs exactly one query per step" % name)
    LOG.debug("verify exit graph=%s seconds=%.4f", name, time.perf_counter() - started)
