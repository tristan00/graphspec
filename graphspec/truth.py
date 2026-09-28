import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.features import token
from graphspec.infra.files import Mapped, load_arrays, read_json, save_arrays, write_json
from graphspec.infra.workspace import finite

from . import kernels, spans

LOG = logging.getLogger(__name__)
VERSION = 7
QUERIES = (
    "query_key",
    "query_category",
    "query_actor",
    "query_subject",
    "query_context",
    "query_x",
    "query_signals",
)
LABELS = ("label_entity", "label_step", "label_row", "label_id")
OBSERVATIONS = ("obs_entity", "obs_start", "obs_stop", "obs_x")
FACTS = ("fact_source", "fact_target", "fact_code", "fact_start", "fact_stop", "fact_rank", "fact_values")
ORDER = ("order_step", "order_position", "order_span")
REMOVALS = ("removal_source", "removal_target", "removal_code", "removal_step", "removal_values")
ARRAYS = ("kinds", "entity_token", "period") + OBSERVATIONS + FACTS + ORDER + REMOVALS + QUERIES + LABELS
ROWS = (
    ("kinds", "period", "obs_entity", "obs_step", "obs_x", "obs_anchor")
    + ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_values")
    + QUERIES
    + LABELS
)
DENSE = (
    ("kinds", "entity_token", "entity_connected", "period", "step_campaign")
    + ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
    + ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    + ("x_table", "value_table")
    + QUERIES
    + LABELS
)


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
            step_campaign=np.zeros(self.steps, np.int32),
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

    def rows(self):
        dense = self.dense()
        values = np.full((len(dense.fact_ref), dense.value_table.shape[1]), np.nan, dense.value_table.dtype)
        known = dense.fact_ref >= 0
        values[known] = dense.value_table[dense.fact_ref[known]]
        arrays = {name: dense.arrays[name] for name in ROWS if name in dense.arrays}
        return Rows(
            self.name,
            self.keys,
            self.steps,
            obs_x=np.asarray(dense.x_table)[dense.obs_ref],
            fact_values=values,
            **arrays,
        )


class Dense(Arrays):
    NAMES = DENSE


class Rows(Arrays):
    NAMES = ROWS


def tokens(registry, kinds, keys):
    return np.asarray(
        [token(registry.nodes[kind] + ":" + key) for kind, key in zip(np.asarray(kinds).tolist(), keys)],
        np.uint64,
    )


def encode(rows, registry):
    started = time.perf_counter()
    if rows.obs_anchor.any():
        raise ValueError("truth graph %s carries anchor observations" % rows.name)
    graph = Graph(
        rows.name,
        rows.keys,
        rows.steps,
        kinds=rows.kinds,
        entity_token=tokens(registry, rows.kinds, rows.keys),
        period=rows.period,
        **spans.observation_spans(rows.obs_entity, rows.obs_step, rows.obs_x),
        **spans.fact_spans(
            rows.fact_step,
            rows.fact_source,
            rows.fact_target,
            rows.fact_code,
            rows.fact_present,
            rows.fact_values,
        ),
        **{name: rows.arrays[name] for name in QUERIES + LABELS},
    )
    LOG.info(
        "encode exit graph=%s observations=%d spans=%d facts=%d spans=%d removals=%d seconds=%.3f",
        rows.name,
        len(rows.obs_step),
        len(graph.obs_start),
        len(rows.fact_step),
        len(graph.fact_start),
        len(graph.removal_step),
        time.perf_counter() - started,
    )
    return graph


def column(table, name, dtype):
    return table.column(name).to_numpy().astype(dtype)


def matrix(table, name, dtype):
    values = table.column(name).combine_chunks()
    return values.flatten().to_numpy().reshape(len(values), values.type.list_size).astype(dtype)


def untied(step, seq, label, name):
    if np.any((np.diff(step) == 0) & (np.diff(seq) == 0)):
        raise ValueError("two %s rows of graph %s share a step and seq" % (label, name))


def parse(registry, categories, tables, rows, name):
    started = time.perf_counter()
    nodes, observations = tables["nodes"], tables["observations"]
    numbers = column(nodes, "entity", np.int64)
    order = np.argsort(numbers, kind="stable")
    ordered = numbers[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("two nodes of graph %s share an entity number" % name)

    def local(values):
        position = np.searchsorted(ordered, values)
        if np.any(position >= len(ordered)) or not np.array_equal(ordered[position], values):
            raise ValueError("a row of graph %s names an entity outside the graph" % name)
        return position.astype(np.int32)

    kinds = np.asarray([registry.kind(kind) for kind in nodes.column("kind").to_pylist()], np.int32)[order]
    names = nodes.column("key").to_pylist()
    keys = [names[i] for i in order.tolist()]
    arrangement = np.lexsort((column(observations, "seq", np.int64), column(observations, "step", np.int64)))
    obs_step = column(observations, "step", np.int32)[arrangement]
    untied(obs_step, column(observations, "seq", np.int64)[arrangement], "observation", name)
    obs_entity = local(column(observations, "entity", np.int64)[arrangement])
    obs_x = matrix(observations, "x", np.float16)[arrangement]
    canonical = np.lexsort((obs_entity, obs_step))
    obs_step, obs_entity, obs_x = obs_step[canonical], obs_entity[canonical], obs_x[canonical]
    width = max([len(registry.attributes(relation)) for relation in registry.extras["base"]] + [0])
    parts = dict(
        step=[np.zeros(0, np.int64)],
        seq=[np.zeros(0, np.int64)],
        source=[np.zeros(0, np.int32)],
        target=[np.zeros(0, np.int32)],
        code=[np.zeros(0, np.int64)],
        present=[np.zeros(0, bool)],
        values=[np.zeros((0, width), np.float32)],
    )
    for relation, table in tables["edges"].items():
        edge = registry.by_name[relation]
        count = table.num_rows
        values = np.full((count, width), np.nan, np.float32)
        for position, attribute in enumerate(edge.attributes):
            values[:, position] = column(table, attribute, np.float32)
        for key, value in (
            ("step", column(table, "step", np.int64)),
            ("seq", column(table, "seq", np.int64)),
            ("source", local(column(table, "source", np.int64))),
            ("target", local(column(table, "target", np.int64))),
            ("code", np.full(count, edge.code, np.int64)),
            ("present", column(table, "present", bool)),
            ("values", values),
        ):
            parts[key].append(value)
    facts = {key: np.concatenate(value) for key, value in parts.items()}
    arrangement = np.lexsort((facts["seq"], ~facts["present"], facts["step"]))
    facts = {key: value[arrangement] for key, value in facts.items()}
    for relation in registry.attributed:
        edge = registry.by_name[relation]
        chosen = facts["present"] & (facts["code"] == edge.code)
        if not np.isfinite(facts["values"][chosen, : len(edge.attributes)]).all():
            raise ValueError("graph %s has a non-finite %s attribute on a present fact" % (name, relation))
    for flag, label in ((True, "present fact"), (False, "absent fact")):
        use = facts["present"] == flag
        untied(facts["step"][use], facts["seq"][use], label, name)
    fact_step = facts["step"].astype(np.int32)
    level = registry.extras["level"]
    if registry.extras["queries"]:
        queries = tables["queries"]
        if not np.array_equal(column(queries, "step", np.int64), np.arange(queries.num_rows)):
            raise ValueError("query steps of graph %s do not number 0..n-1" % name)
        steps = queries.num_rows
        position = {category: index for index, category in enumerate(categories)}
        query = dict(
            query_key=column(queries, "key", np.int64),
            query_category=np.asarray(
                [position[value] for value in queries.column("category").to_pylist()], np.int64
            ),
            query_actor=local(column(queries, "actor", np.int64)),
            query_subject=local(column(queries, "subject", np.int64)),
            query_context=local(column(queries, "context", np.int64)),
            query_x=matrix(queries, "query", np.float16),
            query_signals=np.stack(
                (matrix(queries, "signal", np.float32), matrix(queries, "signal_observed", np.float32)), 2
            )
            if registry.extras["signals"]
            else np.zeros((steps, 0, 2), np.float32),
        )
        period = column(queries, "period", np.int32)
    else:
        query = dict(
            query_key=np.zeros(0, np.int64),
            query_category=np.zeros(0, np.int64),
            query_actor=np.zeros(0, np.int32),
            query_subject=np.zeros(0, np.int32),
            query_context=np.zeros(0, np.int32),
            query_x=np.zeros((0, registry.features), np.float16),
            query_signals=np.zeros((0, 0, 2), np.float32),
        )
    if level == "node":
        targets = tables["targets"]
        entity, step = local(column(targets, "entity", np.int64)), column(targets, "step", np.int64)
        arranged = np.lexsort((entity, step))
        label_entity, label_step = entity[arranged], step[arranged].astype(np.int32)
        if np.any((np.diff(label_step) == 0) & (np.diff(label_entity) == 0)):
            raise ValueError("two labels of graph %s sit on one node at one step" % name)
        label_row, label_id = np.asarray(rows, np.int64)[arranged], column(targets, "id", np.int64)[arranged]
    elif level == "path":
        label_step, label_row = (np.asarray(values, np.int64) for values in rows)
        if np.any(np.diff(label_step) < 0):
            raise ValueError("points of graph %s are not ordered by step" % name)
        label_step = label_step.astype(np.int32)
        label_id = query["query_key"][label_step]
        label_entity = np.full(len(label_step), -1, np.int32)
    else:
        rows = np.asarray(rows, np.int64)
        label_step = np.flatnonzero(rows >= 0).astype(np.int32)
        label_row, label_id = rows[label_step], query["query_key"][label_step]
        label_entity = np.full(len(label_step), -1, np.int32)
    if not registry.extras["queries"]:
        steps = 1 + int(max(obs_step.max(initial=-1), fact_step.max(initial=-1), label_step.max(initial=-1)))
        period = np.arange(steps, dtype=np.int32)
    rows = Rows(
        name,
        keys,
        steps,
        kinds=kinds,
        period=period,
        obs_entity=obs_entity,
        obs_step=obs_step,
        obs_x=obs_x,
        obs_anchor=np.zeros(len(obs_step), bool),
        fact_source=facts["source"],
        fact_target=facts["target"],
        fact_code=facts["code"],
        fact_step=fact_step,
        fact_present=facts["present"],
        fact_values=facts["values"],
        label_entity=label_entity,
        label_step=label_step,
        label_row=label_row,
        label_id=label_id,
        **query,
    )
    graph = encode(rows, registry)
    verify(graph, registry)
    LOG.info(
        "truth exit graph=%s seconds=%.3f steps=%d observations=%d facts=%d labels=%d",
        name,
        time.perf_counter() - started,
        steps,
        len(graph.obs_start),
        len(graph.fact_start),
        len(label_step),
    )
    return graph


def verify_dense(graph, registry, known=None):
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_step),
        fact=len(graph.fact_step),
        query=len(graph.query_key),
        label=len(graph.label_step),
    )
    dimensions = dict(x_table=2, value_table=2, query_x=2, query_signals=3)
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
    if len(graph.entity_token) != entities or len(graph.entity_connected) != entities or steps < 1:
        raise ValueError("graph %s requires one token and flag per entity and at least one step" % name)
    if len(graph.step_campaign) != steps or np.any(np.diff(graph.step_campaign) < 0):
        raise ValueError("graph %s needs one nondecreasing campaign per step" % name)
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
        graph.query_actor,
        graph.query_subject,
        graph.query_context,
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
        (
            "observation",
            "fact source",
            "fact target",
            "query actor",
            "query subject",
            "query context",
            "label",
        ),
        3,
    ):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.period) != steps or np.any(np.diff(graph.period) < 0):
        raise ValueError("periods of graph %s must cover every step and never decrease" % name)
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
    dimensions = dict(obs_x=2, fact_values=2, removal_values=2, query_x=2, query_signals=3)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if key not in dimensions and values.dtype.kind not in "iu":
            raise ValueError("graph %s requires integer indices in %s" % (name, key))
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
        ("query actor", graph.query_actor),
        ("query subject", graph.query_subject),
        ("query context", graph.query_context),
    ):
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
    payload = spans.flat(np.asarray(graph.obs_x)[order])
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
    for who in (graph.query_actor, graph.query_subject, graph.query_context):
        if spans.all_covered(offsets, lows, highs, np.asarray(who), moments, moments + 1) >= 0:
            raise ValueError("a query of graph %s names an entity with no observation at its step" % name)
    if labelled:
        moments = np.asarray(graph.label_step)
        if spans.all_covered(offsets, lows, highs, np.asarray(graph.label_entity), moments, moments + 1) >= 0:
            raise ValueError("a label of graph %s names an entity with no observation at its step" % name)
    if len(graph.period) != steps or np.any(np.diff(graph.period) < 0):
        raise ValueError("periods of graph %s must cover every step and never decrease" % name)
    if registry.extras["queries"] and len(graph.query_key) != steps:
        raise ValueError("graph %s needs exactly one query per step" % name)
    LOG.debug("verify exit graph=%s seconds=%.4f", name, time.perf_counter() - started)
