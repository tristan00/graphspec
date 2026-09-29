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
from .references import MEMBERS, SORTS, ends
from .schema import VALUE, absent, inward, outward

LOG = logging.getLogger(__name__)
VERSION = 8
TASKS = ("task_step", "task_key", "task_kind", "task_x", "task_signals")
LABELS = ("label_task", "label_row")
OBSERVATIONS = ("obs_entity", "obs_start", "obs_stop", "obs_x")
FACTS = ("fact_source", "fact_target", "fact_code", "fact_start", "fact_stop", "fact_rank", "fact_values")
ORDER = ("order_step", "order_position", "order_span")
REMOVALS = ("removal_source", "removal_target", "removal_code", "removal_step", "removal_values")
STATED = ("obs_values", "task_values", "graph_values")
ARRAYS = (
    ("kinds", "entity_token", "entity_held", "period")
    + OBSERVATIONS
    + FACTS
    + ORDER
    + REMOVALS
    + TASKS
    + MEMBERS
    + LABELS
    + STATED
)
JOINED = ("set_edge", "set_member", "set_role", "set_values")
HELD = ("inner_owner", "inner_kind", "inner_values")
WITHIN = ("within_source", "within_target", "within_code", "within_values")
DENSE = (
    ("kinds", "entity_token", "entity_connected", "period", "step_graph")
    + ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
    + ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    + ("x_table", "value_table")
    + TASKS
    + MEMBERS
    + LABELS
)
CHANGES = ("time", "observations", "facts")
HALF = int(np.finfo(np.float16).max)
GROUPS = ("obs", "fact")
SPARSE = ("known_row", "known_time", "replaced_row", "replaced_time", "seen_row", "seen_observer")
STORED = (
    ("kinds", "entity_token")
    + ("obs_entity", "obs_since", "obs_until", "obs_seq", "obs_values", "obs_x")
    + ("fact_source", "fact_target", "fact_code", "fact_present")
    + ("fact_since", "fact_until", "fact_seq", "fact_values")
    + tuple(group + "_" + name for group in GROUPS for name in SPARSE)
    + ("task_time", "task_observer", "task_grouping", "task_key", "task_kind", "task_split")
    + ("task_weight", "task_values", "task_x", "task_signals")
    + MEMBERS
    + ("target_task", "target_row", "target_known", "target_y")
    + ("graph_values",)
    + JOINED
    + HELD
    + WITHIN
    + ("entity_held",)
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
            **{name: np.asarray(self.arrays[name]) for name in TASKS + MEMBERS + LABELS},
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
    def __init__(self, name, keys, unit, inner, observers, **arrays):
        missing = sorted(set(self.NAMES) - set(arrays))
        if missing:
            raise ValueError("graph %s lacks arrays: %s" % (name, ", ".join(missing)))
        unknown = sorted(set(arrays) - set(self.NAMES))
        if unknown:
            raise ValueError("graph %s has unknown arrays: %s" % (name, ", ".join(unknown)))
        self.__dict__.update(
            name=name,
            keys=list(keys),
            unit=unit,
            inner=list(inner),
            observers=list(observers),
            arrays={key: arrays[key] for key in self.NAMES},
        )

    @timed("truth.facts.replace")
    def replace(self, **changes):
        return Facts(
            self.name, self.keys, self.unit, self.inner, self.observers, **dict(self.arrays, **changes)
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
        write_json(
            directory / "graph.json",
            dict(name=self.name, keys=self.keys, unit=self.unit, inner=self.inner, observers=self.observers),
        )
        LOG.debug("facts save exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)

    @classmethod
    def load(cls, directory):
        started = time.perf_counter()
        directory = Path(directory)
        meta = read_json(directory / "graph.json")
        result = cls(
            meta["name"],
            meta["keys"],
            meta["unit"],
            meta["inner"],
            meta["observers"],
            **load_arrays(directory, cls.NAMES),
        )
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

    def referred(self, registry):
        started = time.perf_counter()
        schema, keys = registry.schema, self.keys
        names = [edge.name for edge in registry.edges]
        kinds = list(schema.tasks["kinds"])
        held = {name: [None] * len(self.task_key) for name in schema.tasks["references"]}
        owner = np.asarray(self.task_kind).tolist()
        for name, entry in schema.tasks["references"].items():
            for task, kind in enumerate(owner):
                if name in schema.tasks["kinds"][kinds[kind]]["references"]:
                    held[name][task] = []
        listed = list(schema.tasks["references"])
        for task, name, source, target, code in zip(
            *(
                np.asarray(self.arrays[key]).tolist()
                for key in ("member_task", "member_name", "member_source", "member_target", "member_code")
            )
        ):
            held[listed[name]][task].append(
                (keys[source], None if code < 0 else names[code], None if target < 0 else keys[target])
            )
        LOG.debug("facts referred exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return held

    def stated(self, registry, splits, observers):
        started = time.perf_counter()
        schema, encoding, keys = registry.schema, registry.encoding, np.asarray(self.keys, object)
        kinds, entity = np.asarray(self.kinds), np.asarray(self.obs_entity)
        sequenced = schema.ties is not None and "seq" in clock.TIES.named(schema.ties).columns
        values = np.asarray(self.obs_values)
        observations, entities = {}, schema.entities()
        for kind, columns in entities.items():
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
        asked, listed = schema.asked_columns(), list(schema.tasks["kinds"])
        tasks = dict(
            time=[None if schema.model == "none" else moment(value) for value in np.asarray(self.task_time)],
            observer=[
                None if value < 0 else observers[value] for value in np.asarray(self.task_observer).tolist()
            ],
            key=np.asarray(self.task_key).tolist(),
            kind=[listed[value] for value in np.asarray(self.task_kind).tolist()],
            split=[None if value < 0 else splits[value] for value in np.asarray(self.task_split).tolist()],
            weight=[
                None if np.isnan(value) else float(value)
                for value in np.asarray(self.task_weight, np.float32)
            ],
            references=self.referred(registry),
            **prefixed(stated(registry, asked, np.asarray(self.task_values)[:, encoding.positions(asked)])),
        )
        if schema.tasks["grouping"] is not None:
            tasks["grouping"] = np.asarray(self.task_grouping).tolist()
        targets = dict(
            task=np.asarray(self.target_task).tolist(),
            row=np.asarray(self.target_row).tolist(),
            y=np.asarray(self.target_y, np.float64).tolist(),
            known=[moment(value) for value in np.asarray(self.target_known)],
        )
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
            nodes={kind: keys[kinds == registry.kind(kind)].tolist() for kind in entities},
            observations=observations,
            edges=edges,
            sets=self.joined(registry),
            held=self.held(registry),
            tasks=tasks,
            targets=targets,
        )
        LOG.debug("stated exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def joined(self, registry):
        started = time.perf_counter()
        schema, keys = registry.schema, np.asarray(self.keys, object)
        edge, member = np.asarray(self.set_edge, np.int64), np.asarray(self.set_member, np.int64)
        role, values, columns = (
            np.asarray(self.set_role),
            np.asarray(self.set_values),
            schema.member_columns(),
        )
        result = {}
        for kind, entry in schema.sets.items():
            rows = np.asarray(self.kinds)[edge] == registry.kind(kind)
            result[kind] = dict(
                set=keys[edge[rows]].tolist(),
                member=keys[member[rows]].tolist(),
                role=[entry["roles"][code] for code in role[rows].tolist()],
                **prefixed(
                    stated(
                        registry,
                        entry["members"],
                        values[rows][:, [columns.index(name) for name in entry["members"]]],
                    )
                ),
            )
        LOG.debug("facts joined exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def held(self, registry):
        started = time.perf_counter()
        schema, keys, inner = registry.schema, np.asarray(self.keys, object), np.asarray(self.inner, object)
        owner, kinds = np.asarray(self.inner_owner, np.int64), np.asarray(self.kinds)
        source, target = np.asarray(self.within_source, np.int64), np.asarray(self.within_target, np.int64)
        nodes, links = schema.inner_columns(), schema.within_columns()
        result = {}
        for kind, held in schema.holds.items():
            rows, edges = kinds[owner] == registry.kind(kind), kinds[owner[source]] == registry.kind(kind)
            names, relations = list(held["nodes"]), list(held["edges"])
            result[kind] = dict(
                nodes=dict(
                    owner=keys[owner[rows]].tolist(),
                    kind=[names[code] for code in np.asarray(self.inner_kind)[rows].tolist()],
                    key=inner[rows].tolist(),
                    **prefixed(
                        stated(
                            registry,
                            inward(held),
                            np.asarray(self.inner_values)[rows][
                                :, [nodes.index(name) for name in inward(held)]
                            ],
                        )
                    ),
                ),
                edges=dict(
                    owner=keys[owner[source[edges]]].tolist(),
                    source=inner[source[edges]].tolist(),
                    target=inner[target[edges]].tolist(),
                    relation=[relations[code] for code in np.asarray(self.within_code)[edges].tolist()],
                    **prefixed(
                        stated(
                            registry,
                            outward(held),
                            np.asarray(self.within_values)[edges][
                                :, [links.index(name) for name in outward(held)]
                            ],
                        )
                    ),
                ),
            )
        LOG.debug("facts held exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
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
    for kind, columns in schema.entities().items():
        table = tables["observations"][kind]
        places.append((table, columns, clock.reached(table, facts, until)))
    for held, entry in schema.edges.items():
        table = tables["edges"][held]
        places.append((table, entry["columns"], clock.reached(table, facts, until)))
    for kind, entry in schema.sets.items():
        table = tables["sets"][kind]
        places.append((table, entry["members"], clock.reached(table, (), until)))
    for kind, held in schema.holds.items():
        for part, columns in (("nodes", inward(held)), ("edges", outward(held))):
            table = tables["held"][kind][part]
            places.append((table, columns, clock.reached(table, (), until)))
    found = {name: dict(trained=[], distinct=[]) for name in schema.columns}
    if "tasks" in tables:
        table = tables["tasks"]
        places.append((table, schema.asked_columns(), clock.reached(table, ("time",), until)))
        aims = tables["targets"]
        kinds = np.asarray(table.column("kind").to_pylist(), object)[
            aims.column("task").to_numpy().astype(np.int64)
        ]
        seen = clock.reached(aims, ("known",), until)
        if "known" in aims.column_names:
            seen &= ~absent(aims.column("known"))
        else:
            seen[:] = False
        for kind, entry in schema.tasks["kinds"].items():
            if entry["matures"] is not None:
                found[entry["matures"]["column"]]["trained"].append(
                    aims.column("y").to_numpy().astype(np.float32)[seen & (kinds == kind)]
                )
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


def encoded(registry, values, held, named):
    started = time.perf_counter()
    wide = registry.encoding.inputs(values, held)
    with np.errstate(over="ignore"):
        result = wide.astype(np.float16)
    lost = ~np.isfinite(result)
    if lost.any():
        row, place = (int(found[0]) for found in np.nonzero(lost))
        raise ValueError(
            "%s holds %s which gives %s; model inputs are stored at 16 bits, which hold magnitudes up to %d"
            % (
                named(row),
                registry.encoding.origin(place, np.asarray(values)[row]),
                float(wide[row, place]),
                HALF,
            )
        )
    LOG.debug("encoded exit rows=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("truth.unheld")
def unheld(registry, rows):
    return np.zeros((rows, registry.encoding.inner), np.float32)


def inputs(registry, kinds, arrays):
    started = time.perf_counter()
    encoding, kinds = registry.encoding, np.asarray(kinds)
    owner = np.asarray(arrays["inner_owner"], np.int64)
    source, target = (
        np.asarray(arrays["within_source"], np.int64),
        np.asarray(arrays["within_target"], np.int64),
    )
    columns = registry.schema.inner_columns()
    result = unheld(registry, len(kinds))
    for kind, held in registry.schema.holds.items():
        code = registry.kind(kind)
        owners = np.flatnonzero(kinds == code)
        rows = np.flatnonzero(kinds[owner] == code)
        place = np.full(len(owner), -1, np.int64)
        place[rows] = np.arange(len(rows))
        links = kinds[owner[source]] == code
        found = encoding.graphs(
            kind,
            dict(
                owners=len(owners),
                owner=np.searchsorted(owners, owner[rows]),
                values=np.asarray(arrays["inner_values"])[rows][
                    :, [columns.index(name) for name in inward(held)]
                ],
                source=place[source[links]],
                target=place[target[links]],
            ),
        )
        start, width = encoding.held[kind]
        result[owners, start : start + width] = found
    LOG.debug(
        "inputs exit entities=%d held=%d seconds=%.4f", len(kinds), len(owner), time.perf_counter() - started
    )
    return result


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


def asked(registry, splits, observers, tables, name):
    started = time.perf_counter()
    schema, encoding = registry.schema, registry.encoding
    stated, tasks, layout = schema.tasks, tables["tasks"], encoding.layout
    count, timed = tasks.num_rows, schema.model != "none"
    if timed and tasks.column("time").null_count:
        raise ValueError("a task of graph %s states no time" % name)
    moments = column(tasks, "time", np.int64) if timed else np.zeros(count, np.int64)
    if np.any((moments <= ALWAYS) | (moments >= NEVER)):
        raise ValueError("a task of graph %s states a time outside the range of times" % name)
    order = np.argsort(moments, kind="stable")
    place = np.empty(count, np.int64)
    place[order] = np.arange(count)
    listed = {kind: code for code, kind in enumerate(stated["kinds"])}
    found = tasks.column("kind").to_pylist()
    strangers = sorted({str(value) for value in found if value not in listed})
    if strangers:
        raise ValueError(
            "tasks of graph %s are of kinds the schema does not state: %s" % (name, ", ".join(strangers))
        )
    kinds = np.asarray([listed[value] for value in found], np.int32).reshape(-1)
    looking = np.full(count, -1, np.int32)
    if schema.observers:
        known = {held: code for code, held in enumerate(observers)}
        named = tasks.column("observer").to_pylist()
        strangers = sorted({value for value in named if value is not None and value not in known})
        if strangers:
            raise ValueError(
                "tasks of graph %s name observers the bank does not hold: %s" % (name, ", ".join(strangers))
            )
        looking = np.asarray([-1 if value is None else known[value] for value in named], np.int32).reshape(-1)
    known = {held: code for code, held in enumerate(splits)}
    named = tasks.column("split").to_pylist()
    strangers = sorted({value for value in named if value is not None and value not in known})
    if strangers:
        raise ValueError(
            "tasks of graph %s name splits the bank does not hold: %s" % (name, ", ".join(strangers))
        )
    split = np.asarray([-1 if value is None else known[value] for value in named], np.int32).reshape(-1)
    weight = np.asarray(tasks.column("weight").to_numpy(zero_copy_only=False), np.float32)
    keys = column(tasks, "key", np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("two tasks of graph %s state one key" % name)
    columns = schema.asked_columns()
    values = codes(registry, tasks, columns, encoding.positions(columns), len(layout))
    for kind, entry in stated["kinds"].items():
        members = kinds == listed[kind]
        weighted = ~np.isnan(weight[members])
        wrong = np.flatnonzero(members & ~(np.isfinite(weight) & (weight > 0)))
        if entry["weight"] and len(wrong):
            raise ValueError(
                "the task %d of the kind %s of graph %s states the weight %s; its kind states a weight above "
                "zero and finite at float32 precision" % (keys[wrong[0]], kind, name, weight[wrong[0]])
            )
        if not entry["weight"] and weighted.any():
            raise ValueError(
                "a task of graph %s of the kind %s states a weight, which its kind does not" % (name, kind)
            )
        for other in columns:
            if (
                other not in entry["columns"]
                and not np.isnan(values[members][:, encoding.positions([other])]).all()
            ):
                raise ValueError(
                    "a task of graph %s of the kind %s states the column %s, which its kind does not"
                    % (name, kind, other)
                )
    result = dict(
        task_time=moments[order],
        task_observer=looking[order],
        task_grouping=column(tasks, "period", np.int32)[order]
        if stated["grouping"] is not None
        else np.zeros(count, np.int32),
        task_key=keys[order],
        task_kind=kinds[order],
        task_split=split[order],
        task_weight=weight[order],
        task_values=values[order],
        task_x=encoded(
            registry,
            values[order],
            unheld(registry, count),
            lambda row: (
                "the task %d of the kind %s of graph %s"
                % (keys[order][row], list(stated["kinds"])[kinds[order][row]], name)
            ),
        ),
        task_signals=np.stack(
            (matrix(tasks, "signal", np.float32), matrix(tasks, "signal_observed", np.float32)), 2
        )[order]
        if stated["signals"]
        else np.zeros((count, 0, 2), np.float32),
    )
    LOG.debug("asked exit graph=%s tasks=%d seconds=%.4f", name, count, time.perf_counter() - started)
    return result, place


def referred(registry, tables, tasks, place, local, kinds, name):
    started = time.perf_counter()
    stated = registry.schema.tasks
    listed = list(stated["kinds"])
    parts = dict(task=[], name=[], place=[], source=[], target=[], code=[])
    for number, (reference, entry) in enumerate(stated["references"].items()):
        sort = SORTS.named(entry["sort"])
        holders = np.asarray(
            [reference in stated["kinds"][listed[kind]]["references"] for kind in tasks["task_kind"]], bool
        ).reshape(-1)
        where = "the reference %s of graph %s" % (reference, name)
        if sort.columns() is None:
            continue
        table = tables["references"][reference]
        owner = place[column(table, "task", np.int64)]
        source = local(column(table, "source", np.int64))
        allowed = [registry.kind(kind) for kind in entry["kinds"]]
        target, code = np.full(len(owner), -1, np.int64), np.full(len(owner), -1, np.int64)
        if "relation" in table.column_names:
            relations = table.column("relation").to_pylist()
            edge = np.asarray([value is not None for value in relations], bool).reshape(-1)
            strangers = sorted({value for value in relations if value is not None} - set(registry.tables))
            if strangers:
                raise ValueError(
                    "%s names relations the schema does not state: %s" % (where, ", ".join(strangers))
                )
            ended = np.asarray(table.column("target").is_valid().to_numpy(zero_copy_only=False), bool)
            if not np.array_equal(edge, ended):
                raise ValueError("%s states both ends and the relation of every edge it names" % where)
            target[edge] = local(
                np.asarray(table.column("target").fill_null(0).to_numpy(zero_copy_only=False), np.int64)[edge]
            )
            code[edge] = [registry.code(value) for value in relations if value is not None]
            swap = np.zeros(len(owner), bool)
            for row in np.flatnonzero(edge).tolist():
                relation = registry.by_code[int(code[row])]
                pair = (registry.nodes[kinds[source[row]]], registry.nodes[kinds[target[row]]])
                pairs = [tuple(held) for held in relation.pairs]
                if pair not in pairs and not (relation.symmetric and pair[::-1] in pairs):
                    raise ValueError(
                        "%s names an edge of %s between kinds of node the relation does not join"
                        % (where, relation.name)
                    )
                swap[row] = relation.symmetric and pair[0] == pair[1] and source[row] > target[row]
            source, target = np.where(swap, target, source), np.where(swap, source, target)
        if len(owner) and not np.isin(kinds[source[code < 0]], allowed).all():
            raise ValueError("%s names nodes of kinds the reference does not state" % where)
        if np.any(~holders[owner]):
            raise ValueError("%s is stated by a task whose kind does not state it" % where)
        position = (
            column(table, "place", np.int64)
            if "place" in table.column_names
            else np.zeros(len(owner), np.int64)
        )
        order = np.lexsort((target, code, source, position, owner))
        owner, source, target, code, position = (
            values[order] for values in (owner, source, target, code, position)
        )
        if not sort.ordered:
            same = np.r_[False, owner[1:] == owner[:-1]]
            same[1:] &= ((source[1:] == source[:-1]) & (target[1:] == target[:-1]) & (code[1:] == code[:-1]))[
                : len(same) - 1
            ]
            if same.any():
                raise ValueError("%s names a member twice for one task" % where)
            start = np.maximum.accumulate(
                np.where(np.r_[True, owner[1:] != owner[:-1]][: len(owner)], np.arange(len(owner)), 0)
            )
            position = np.arange(len(owner)) - start
        counts = np.bincount(owner, minlength=len(holders))
        wrong = np.flatnonzero(holders & sort.miscounted(counts))
        if len(wrong):
            raise ValueError(
                "the kind %s requires of each of its tasks %s under the reference %s, which is of the sort %s, "
                "and the task %d of graph %s names %d"
                % (
                    listed[tasks["task_kind"][wrong[0]]],
                    sort.required(),
                    reference,
                    entry["sort"],
                    tasks["task_key"][wrong[0]],
                    name,
                    counts[wrong[0]],
                )
            )
        sort.check(
            where,
            counts[holders],
            owner,
            source,
            target,
            code,
            position,
        )
        parts["task"].append(owner)
        parts["name"].append(np.full(len(owner), number, np.int64))
        parts["place"].append(position)
        parts["source"].append(source)
        parts["target"].append(target)
        parts["code"].append(code)
    held = {key: np.concatenate(values + [np.zeros(0, np.int64)]) for key, values in parts.items()}
    rank = np.zeros(len(held["task"]), np.int64)
    for kind, entry in stated["kinds"].items():
        order = {list(stated["references"]).index(found): at for at, found in enumerate(entry["references"])}
        members = np.flatnonzero(tasks["task_kind"][held["task"]] == listed.index(kind))
        rank[members] = [order[value] for value in held["name"][members].tolist()]
    order = np.lexsort((held["place"], rank, held["task"]))
    result = dict(
        member_task=held["task"][order].astype(np.int32),
        member_name=held["name"][order].astype(np.int16),
        member_place=held["place"][order].astype(np.int32),
        member_source=held["source"][order].astype(np.int32),
        member_target=held["target"][order].astype(np.int32),
        member_code=held["code"][order].astype(np.int16),
    )
    LOG.debug(
        "referred exit graph=%s members=%d seconds=%.4f", name, len(order), time.perf_counter() - started
    )
    return result


def aimed(registry, tables, tasks, place, rows, name):
    started = time.perf_counter()
    stated = registry.schema.tasks
    listed = list(stated["kinds"])
    found = tables["targets"]
    owner = place[column(found, "task", np.int64)]
    rows = np.asarray(rows, np.int64)
    if len(rows) != len(owner):
        raise ValueError("graph %s reads different targets than its supervision" % name)
    known = np.full(len(owner), NEVER, np.int64)
    if "known" in found.column_names:
        given = found.column("known")
        held = np.asarray(given.fill_null(0).to_numpy(zero_copy_only=False), np.int64)
        if np.any((held <= ALWAYS) | (held >= NEVER)):
            raise ValueError("a target of graph %s is known from a time outside the range of times" % name)
        known = np.where(absent(given), NEVER, held)
    if np.any(known <= tasks["task_time"][owner]):
        raise ValueError(
            "a target of graph %s is known no later than the time of its task, whose view it would enter"
            % name
        )
    counts = np.bincount(owner, minlength=len(tasks["task_key"]))
    for kind, entry in stated["kinds"].items():
        members = tasks["task_kind"] == listed.index(kind)
        if entry["target"] is None and counts[members].any():
            raise ValueError(
                "a task of graph %s of the kind %s holds a target, which its kind does not state"
                % (name, kind)
            )
        if (
            entry["target"] is not None
            and entry["target"]["kind"] == "number"
            and counts[members].max(initial=0) > 1
        ):
            raise ValueError("a task of graph %s of the kind %s holds more than one target" % (name, kind))
    if np.any(tasks["task_split"][owner] < 0):
        raise ValueError("a task of graph %s holds a target and states no split" % name)
    order = np.lexsort((rows, owner))
    result = dict(
        target_task=owner[order].astype(np.int32),
        target_row=rows[order],
        target_known=known[order],
        target_y=column(found, "y", np.float64)[order],
    )
    LOG.debug("aimed exit graph=%s targets=%d seconds=%.4f", name, len(order), time.perf_counter() - started)
    return result


def parse(registry, splits, observers, tables, rows, name):
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
    layout, entities = encoding.layout, schema.entities()
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
            registry, table, entities[kind], encoding.positions(entities[kind]), len(layout)
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
    tasks, place = asked(registry, splits, observers, tables, name)
    members = referred(registry, tables, tasks, place, local, kinds, name)
    targets = aimed(registry, tables, tasks, place, rows, name)
    joined = belonging(registry, tables, kinds, local, name)
    inner, held = holding(registry, tables, kinds, local, name)
    entity_held = inputs(registry, kinds, held)
    graph = Facts(
        name,
        keys,
        unit,
        inner,
        observers,
        kinds=kinds,
        entity_token=tokens(registry, kinds, keys),
        entity_held=entity_held,
        **joined,
        **held,
        obs_entity=seen["entity"].astype(np.int32),
        obs_since=seen["since"],
        obs_until=seen["until"],
        obs_seq=seen["seq"].astype(np.int32),
        obs_values=seen["values"],
        obs_x=encoded(
            registry,
            seen["values"],
            entity_held[seen["entity"]],
            lambda row: (
                "the node %s of the kind %s of graph %s"
                % (keys[seen["entity"][row]], registry.nodes[kinds[seen["entity"][row]]], name)
            ),
        ),
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
        graph_values=codes(
            registry, tables["graph"], schema.graph, range(len(schema.graph)), len(schema.graph)
        )[0],
        **tasks,
        **members,
        **targets,
    )
    check(graph, registry, splits, observers)
    LOG.info(
        "truth exit graph=%s seconds=%.3f observations=%d facts=%d tasks=%d labels=%d",
        name,
        time.perf_counter() - started,
        len(graph.obs_since),
        len(graph.fact_since),
        len(graph.task_time),
        len(graph.target_task),
    )
    return graph


def belonging(registry, tables, kinds, local, name):
    started = time.perf_counter()
    schema, columns = registry.schema, registry.schema.member_columns()
    parts = dict(edge=[], member=[], role=[], values=[])
    for kind, entry in schema.sets.items():
        table, where = tables["sets"][kind], "a member of a set edge of %s in graph %s" % (kind, name)
        edge, member = local(column(table, "set", np.int64)), local(column(table, "member", np.int64))
        if np.any(kinds[edge] != registry.kind(kind)):
            raise ValueError("%s belongs to an entity of another kind" % where)
        if not np.isin(kinds[member], [registry.kind(held) for held in entry["kinds"]]).all():
            raise ValueError("%s is a node of a kind the set edge does not join" % where)
        roles = table.column("role").to_pylist()
        strangers = sorted({str(role) for role in roles if role not in entry["roles"]})
        if strangers:
            raise ValueError("%s takes roles its kind does not state: %s" % (where, ", ".join(strangers)))
        positions = [columns.index(held) for held in entry["members"]]
        values = codes(registry, table, entry["members"], positions, len(columns))
        if not np.isfinite(values[:, positions]).all():
            raise ValueError("%s carries no value in a column of its members" % where)
        lonely = np.setdiff1d(np.flatnonzero(kinds == registry.kind(kind)), edge)
        if len(lonely):
            raise ValueError("the set edge of %s in graph %s has no member" % (kind, name))
        parts["edge"].append(edge.astype(np.int64))
        parts["member"].append(member.astype(np.int64))
        parts["role"].append(np.asarray([entry["roles"].index(role) for role in roles], np.int64).reshape(-1))
        parts["values"].append(values)
    edge, member, role = (
        np.concatenate(parts[part] + [np.zeros(0, np.int64)]) for part in ("edge", "member", "role")
    )
    values = np.concatenate(parts["values"] + [np.zeros((0, len(columns)), np.float32)])
    order = np.lexsort((role, member, edge))
    edge, member, role, values = edge[order], member[order], role[order], values[order]
    if np.any((edge[1:] == edge[:-1]) & (member[1:] == member[:-1]) & (role[1:] == role[:-1])):
        raise ValueError("a node of graph %s is a member of one set edge twice in one role" % name)
    result = dict(
        set_edge=edge.astype(np.int32),
        set_member=member.astype(np.int32),
        set_role=role.astype(np.int16),
        set_values=values,
    )
    LOG.debug(
        "belonging exit graph=%s members=%d seconds=%.4f", name, len(edge), time.perf_counter() - started
    )
    return result


def holding(registry, tables, kinds, local, name):
    started = time.perf_counter()
    schema = registry.schema
    nodes, links = schema.inner_columns(), schema.within_columns()
    parts = dict(owner=[], kind=[], keys=[], values=[], source=[], target=[], code=[], links=[])
    offset = 0
    for kind, held in schema.holds.items():
        where = "a graph held by a node of %s in graph %s" % (kind, name)
        table, edges = tables["held"][kind]["nodes"], tables["held"][kind]["edges"]
        owner = local(column(table, "owner", np.int64)).astype(np.int64)
        if np.any(kinds[owner] != registry.kind(kind)):
            raise ValueError("%s belongs to a node of another kind" % where)
        names, relations = list(held["nodes"]), list(held["edges"])
        found, keys = table.column("kind").to_pylist(), table.column("key").to_pylist()
        strangers = sorted({str(value) for value in found if value not in names})
        if strangers or None in keys:
            raise ValueError(
                "%s names held nodes of kinds it does not state, %s, or a held node without a key"
                % (where, ", ".join(strangers) or "none")
            )
        inner = np.asarray([names.index(value) for value in found], np.int64).reshape(-1)
        paired = sorted(zip(owner.tolist(), keys))
        if any(first == second for first, second in zip(paired[1:], paired[:-1])):
            raise ValueError("%s names one held node twice" % where)
        values = codes(
            registry, table, inward(held), [nodes.index(part) for part in inward(held)], len(nodes)
        )
        for code, part in enumerate(names):
            others = [nodes.index(other) for other in inward(held) if other not in held["nodes"][part]]
            if not np.isnan(values[inner == code][:, others]).all():
                raise ValueError("%s states a column that its held kind %s does not" % (where, part))
        source, target = column(edges, "source", np.int64), column(edges, "target", np.int64)
        if np.any(owner[source] != owner[target]):
            raise ValueError("%s holds an edge between the graphs of two nodes" % where)
        found = edges.column("relation").to_pylist()
        strangers = sorted({str(value) for value in found if value not in relations})
        if strangers:
            raise ValueError(
                "%s holds edges of relations it does not state: %s" % (where, ", ".join(strangers))
            )
        code = np.asarray([relations.index(value) for value in found], np.int64).reshape(-1)
        joints = codes(
            registry, edges, outward(held), [links.index(part) for part in outward(held)], len(links)
        )
        for number, relation in enumerate(relations):
            entry, chosen = held["edges"][relation], code == number
            pairs = {(names.index(first), names.index(second)) for first, second in entry["endpoints"]}
            if not entry["directed"]:
                pairs |= {(second, first) for first, second in pairs}
            ends = zip(inner[source[chosen]].tolist(), inner[target[chosen]].tolist())
            if not all(pair in pairs for pair in ends):
                raise ValueError(
                    "%s holds an edge of %s between held kinds it does not join" % (where, relation)
                )
            others = [links.index(other) for other in outward(held) if other not in entry["columns"]]
            if not np.isnan(joints[chosen][:, others]).all():
                raise ValueError("%s holds an edge of %s with a column it does not state" % (where, relation))
        order = np.argsort(owner, kind="stable")
        place = np.empty(len(order), np.int64)
        place[order] = np.arange(len(order))
        parts["owner"].append(owner[order])
        parts["kind"].append(inner[order])
        parts["keys"] += [keys[row] for row in order.tolist()]
        parts["values"].append(values[order])
        parts["source"].append(place[source] + offset)
        parts["target"].append(place[target] + offset)
        parts["code"].append(code)
        parts["links"].append(joints)
        offset += len(order)
    empty = np.zeros(0, np.int64)
    result = dict(
        inner_owner=np.concatenate(parts["owner"] + [empty]).astype(np.int32),
        inner_kind=np.concatenate(parts["kind"] + [empty]).astype(np.int16),
        inner_values=np.concatenate(parts["values"] + [np.zeros((0, len(nodes)), np.float32)]),
        within_source=np.concatenate(parts["source"] + [empty]).astype(np.int32),
        within_target=np.concatenate(parts["target"] + [empty]).astype(np.int32),
        within_code=np.concatenate(parts["code"] + [empty]).astype(np.int16),
        within_values=np.concatenate(parts["links"] + [np.zeros((0, len(links)), np.float32)]),
    )
    LOG.debug("holding exit graph=%s held=%d seconds=%.4f", name, offset, time.perf_counter() - started)
    return parts["keys"], result


@timed("truth.ranks")
def ranks(keys):
    order = sorted(range(len(keys)), key=keys.__getitem__)
    result = np.empty(len(keys), np.int64)
    result[order] = np.arange(len(keys))
    return result


def check(graph, registry, splits, observers):
    started = time.perf_counter()
    name, entities, schema = graph.name, len(graph.kinds), registry.schema
    lengths = dict(
        obs=len(graph.obs_since),
        fact=len(graph.fact_since),
        task=len(graph.task_key),
        member=len(graph.member_task),
        target=len(graph.target_task),
        set=len(graph.set_edge),
        inner=len(graph.inner_owner),
        within=len(graph.within_source),
    )
    dimensions = dict(
        obs_x=2,
        obs_values=2,
        fact_values=2,
        task_x=2,
        task_values=2,
        task_signals=3,
        set_values=2,
        inner_values=2,
        within_values=2,
        entity_held=2,
    )
    real = ("fact_present", "graph_values", "task_weight", "target_y")
    loose = tuple(group + "_" + part for group in GROUPS for part in SPARSE)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and key not in loose and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if key not in dimensions and key not in real and values.dtype.kind not in "iu":
            raise ValueError("graph %s requires whole numbers in %s" % (name, key))
    if (schema.model in clock.UNITED) != (graph.unit is not None):
        raise ValueError("graph %s states a unit of time exactly when its schema has a clock" % name)
    if graph.unit is not None and graph.unit not in schema.units:
        raise ValueError(
            "graph %s states the unit of time %r, which its schema does not" % (name, graph.unit)
        )
    if len(graph.keys) != entities or len(graph.entity_token) != entities:
        raise ValueError("graph %s requires one key and token per entity" % name)
    if not np.array_equal(graph.entity_token, tokens(registry, graph.kinds, graph.keys)):
        raise ValueError("entity tokens of graph %s do not match its keys" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.obs_x.shape[1] != registry.features or graph.task_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if not finite(graph.obs_x) or not finite(graph.task_x):
        raise ValueError("graph %s observation and task features must be finite" % name)
    for label, values in (
        ("fact about a node", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
        ("reference", graph.member_source),
        ("reference", np.maximum(graph.member_target, 0)),
        ("set edge", graph.set_edge),
        ("member of a set edge", graph.set_member),
        ("holder of a graph", graph.inner_owner),
    ):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.within_source) and (
        max(graph.within_source.max(), graph.within_target.max()) >= len(graph.inner_owner)
        or min(graph.within_source.min(), graph.within_target.min()) < 0
    ):
        raise ValueError("an edge of a graph held in graph %s names a held node outside it" % name)
    held = inputs(registry, graph.kinds, graph.arrays)
    if len(graph.inner) != len(graph.inner_owner) or not np.array_equal(graph.entity_held, held):
        raise ValueError("graph %s holds encoded graphs that its held graphs do not reproduce" % name)
    for given, values, inner in (
        (graph.obs_x, graph.obs_values, held[np.asarray(graph.obs_entity, np.int64)]),
        (graph.task_x, graph.task_values, unheld(registry, len(graph.task_key))),
    ):
        again = encoded(registry, np.asarray(values), inner, lambda row: "a row of graph %s" % name)
        if not np.array_equal(np.asarray(given), again):
            raise ValueError("graph %s holds node values that its encoders do not reproduce" % name)
    if graph.graph_values.shape != (len(schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    joined = np.asarray([kind in schema.sets for kind in registry.nodes], bool)
    if not joined[graph.kinds[graph.set_edge]].all() or joined[graph.kinds[graph.set_member]].any():
        raise ValueError(
            "a member of a set edge of graph %s joins entities of kinds its schema does not" % name
        )
    found = np.asarray(graph.fact_code)
    if np.any(found < 0) or np.any(found >= len(registry.canonical)) or not registry.canonical[found].all():
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    tasks = len(graph.task_key)
    if np.any(graph.task_kind < 0) or np.any(graph.task_kind >= len(schema.tasks["kinds"])):
        raise ValueError("a task of graph %s is of a kind the schema does not state" % name)
    for label, values in (("reference", graph.member_task), ("target", graph.target_task)):
        if np.any(values < 0) or np.any(values >= tasks):
            raise ValueError("a %s of graph %s names a task outside the graph" % (label, name))
    if np.any(graph.task_observer < -1) or np.any(graph.task_observer >= len(observers)):
        raise ValueError("a task of graph %s names an observer the bank does not hold" % name)
    if np.any(graph.task_split < -1) or np.any(graph.task_split >= len(splits)):
        raise ValueError("a task of graph %s names a split the bank does not hold" % name)
    if np.any(np.diff(graph.task_time) < 0) or np.any(np.diff(graph.target_task) < 0):
        raise ValueError("tasks and targets of graph %s are not ordered by their time" % name)
    if np.any(np.diff(graph.member_task) < 0):
        raise ValueError("references of graph %s are not ordered by their task" % name)
    if np.any(graph.target_known <= graph.task_time[graph.target_task]):
        raise ValueError("a target of graph %s is known no later than the time of its task" % name)
    place = ranks(graph.keys)
    for group in GROUPS:
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

            def named(row):
                entity = int(graph.obs_entity[row])
                return "the node %s of the kind %s" % (graph.keys[entity], registry.nodes[graph.kinds[entity]])

        else:
            identity = np.stack((graph.fact_code, graph.fact_source, graph.fact_target, graph.fact_present), 1)
            owner = np.unique(identity.astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
            facts = dict(
                seq=graph.fact_seq,
                code=graph.fact_code,
                source=place[graph.fact_source],
                target=place[graph.fact_target],
            )

            def named(row):
                return "the edge of %s from %s to %s" % (
                    registry.by_code[int(graph.fact_code[row])].name,
                    graph.keys[int(graph.fact_source[row])],
                    graph.keys[int(graph.fact_target[row])],
                )

        clock.untied(name, named, schema.ties, owner, stamps["since"], facts)
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


@timed("truth.unobserved")
def unobserved(entities, obs_entity, obs_step, who, moments):
    held = np.asarray(obs_step, np.int64) * entities + np.asarray(obs_entity, np.int64)
    return ~np.isin(np.asarray(moments, np.int64) * entities + np.asarray(who, np.int64), held)


def referring(graph, registry):
    started = time.perf_counter()
    member, entity = ends(graph.member_source, graph.member_target)
    task = np.asarray(graph.member_task, np.int64)[member]
    stops = registry.stops[np.asarray(graph.task_kind, np.int64)[task]]
    result = (
        member,
        entity,
        np.asarray(graph.task_step, np.int64)[task],
        stops,
    )
    LOG.debug(
        "referring exit graph=%s ends=%d seconds=%.4f", graph.name, len(member), time.perf_counter() - started
    )
    return result


@timed("truth.unresolved")
def unresolved(graph, registry, member):
    task = int(graph.member_task[member])
    return "the reference %s of the task %d of the kind %s of graph %s" % (
        list(registry.schema.tasks["references"])[int(graph.member_name[member])],
        int(graph.task_key[task]),
        list(registry.schema.tasks["kinds"])[int(graph.task_kind[task])],
        graph.name,
    )


def placed(graph, registry, steps, entities):
    started = time.perf_counter()
    name, tasks = graph.name, len(graph.task_key)
    if (
        np.any(graph.task_step < 0)
        or np.any(graph.task_step >= steps)
        or np.any(np.diff(graph.task_step) < 0)
    ):
        raise ValueError("tasks of graph %s sit at steps outside 0..%d or out of order" % (name, steps - 1))
    if np.any(graph.task_kind < 0) or np.any(graph.task_kind >= len(registry.schema.tasks["kinds"])):
        raise ValueError("a task of graph %s is of a kind the schema does not state" % name)
    for label, values in (("reference", graph.member_task), ("label", graph.label_task)):
        if np.any(values < 0) or np.any(values >= tasks) or np.any(np.diff(values) < 0):
            raise ValueError(
                "a %s of graph %s names a task outside the graph or out of order" % (label, name)
            )
    if (
        np.any(graph.member_source < 0)
        or np.any(graph.member_source >= entities)
        or np.any(graph.member_target < -1)
        or np.any(graph.member_target >= entities)
    ):
        raise ValueError("a reference of graph %s names an entity outside 0..%d" % (name, entities - 1))
    if not registry.aimed[np.asarray(graph.task_kind, np.int64)[graph.label_task]].all():
        raise ValueError("a label of graph %s sits on a task whose kind states no target" % name)
    LOG.debug("placed exit graph=%s tasks=%d seconds=%.4f", name, tasks, time.perf_counter() - started)


def verify_dense(graph, registry, known=None):
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    lengths = dict(
        obs=len(graph.obs_step),
        fact=len(graph.fact_step),
        task=len(graph.task_key),
        member=len(graph.member_task),
        label=len(graph.label_task),
    )
    dimensions = dict(x_table=2, value_table=2, task_x=2, task_signals=3)
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
    if len(graph.step_graph) != steps or np.any(np.diff(graph.step_graph) < 0):
        raise ValueError("graph %s needs one joined graph per step, never decreasing" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    if graph.x_table.shape[1] != registry.features or graph.task_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if not finite(graph.x_table) or not finite(graph.task_x):
        raise ValueError("graph %s observation and task features must be finite" % name)
    placed(graph, registry, steps, entities)
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
        steps,
        entities,
        np.zeros(entities, bool) if known is None else known,
    )
    for index, label in enumerate(("observation", "fact")):
        if flags[index]:
            raise ValueError("a %s row of graph %s has a step outside 0..%d" % (label, name, steps - 1))
    for index, label in enumerate(("observation", "fact source", "fact target"), 2):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    if flags[5]:
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    for index, message in (
        (6, "observations of graph %s are not ordered by step"),
        (7, "facts of graph %s are not ordered by step with present facts first"),
        (8, "an entity of graph %s has two observations at one step"),
    ):
        if flags[index]:
            raise ValueError(message % name)
    for index, label, when in (
        (9, "present fact", "outside the anchors of references at its step"),
        (10, "absent fact", "at or before its step"),
    ):
        if flags[index]:
            raise ValueError("a %s of graph %s names an entity with no observation %s" % (label, name, when))
    member, entity, moments, stops = referring(graph, registry)
    lost = stops & unobserved(entities, graph.obs_entity, graph.obs_step, entity, moments)
    if lost.any():
        raise ValueError(
            "%s names a node not observed at the time of the task, and its kind states that this stops the work"
            % unresolved(graph, registry, member[np.flatnonzero(lost)[0]])
        )


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
        task=len(graph.task_key),
        member=len(graph.member_task),
        label=len(graph.label_task),
    )
    dimensions = dict(
        obs_x=2,
        obs_values=2,
        fact_values=2,
        removal_values=2,
        task_x=2,
        task_values=2,
        task_signals=3,
        graph_values=1,
        entity_held=2,
    )
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
    if graph.obs_x.shape[1] != registry.features or graph.task_x.shape[1] != registry.features:
        raise ValueError("graph %s feature width does not match its registry" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    if graph.fact_values.shape[1] != graph.removal_values.shape[1]:
        raise ValueError("graph %s fact and removal attribute widths differ" % name)
    if not finite(graph.obs_x) or not finite(graph.task_x):
        raise ValueError("graph %s observation and task features must be finite" % name)
    if graph.entity_held.shape != (entities, registry.encoding.inner):
        raise ValueError("graph %s does not hold one encoding of held graphs per entity" % name)
    for given, values, held in (
        (
            graph.obs_x,
            graph.obs_values,
            np.asarray(graph.entity_held)[np.asarray(graph.obs_entity, np.int64)],
        ),
        (graph.task_x, graph.task_values, unheld(registry, len(graph.task_key))),
    ):
        again = encoded(registry, np.asarray(values), held, lambda row: "a row of graph %s" % name)
        if not np.array_equal(np.asarray(given), again):
            raise ValueError("graph %s holds node values that its encoders do not reproduce" % name)
    if graph.graph_values.shape != (len(registry.schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    for label, start, stop in (
        ("observation", graph.obs_start, graph.obs_stop),
        ("fact", graph.fact_start, graph.fact_stop),
    ):
        if np.any(start < 0) or np.any(stop <= start) or np.any(stop > steps):
            raise ValueError("a %s span of graph %s lies outside 0..%d or is empty" % (label, name, steps))
    if np.any(graph.removal_step < 0) or np.any(graph.removal_step >= steps):
        raise ValueError("a removal row of graph %s has a step outside 0..%d" % (name, steps - 1))
    placed(graph, registry, steps, entities)
    for label, values in (
        ("observation", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
        ("removal source", graph.removal_source),
        ("removal target", graph.removal_target),
    ):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
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
        row = spans.all_covered(offsets, lows, highs, np.asarray(who), graph.fact_start, graph.fact_stop)
        if row >= 0:
            end = int(who[row])
            raise ValueError(
                "the edge of %s from %s to %s of graph %s holds over the steps %d to %d, and at one of them "
                "its end %s, a node of the kind %s, has no observation"
                % (
                    registry.by_code[int(graph.fact_code[row])].name,
                    graph.keys[int(graph.fact_source[row])],
                    graph.keys[int(graph.fact_target[row])],
                    name,
                    int(graph.fact_start[row]),
                    int(graph.fact_stop[row]) - 1,
                    graph.keys[end],
                    registry.nodes[int(graph.kinds[end])],
                )
            )
    first = np.full(entities, steps, np.int64)
    np.minimum.at(first, graph.obs_entity, graph.obs_start)
    if np.any(first[graph.removal_source] > graph.removal_step) or np.any(
        first[graph.removal_target] > graph.removal_step
    ):
        raise ValueError(
            "an absent fact of graph %s names an entity with no observation at or before its step" % name
        )
    member, entity, moments, stops = referring(graph, registry)
    chosen = np.flatnonzero(stops)
    lost = spans.all_covered(
        offsets,
        lows,
        highs,
        np.ascontiguousarray(entity[chosen]),
        np.ascontiguousarray(moments[chosen]),
        np.ascontiguousarray(moments[chosen] + 1),
    )
    if lost >= 0:
        raise ValueError(
            "%s names a node not observed at the time of the task, and its kind states that this stops the work"
            % unresolved(graph, registry, member[chosen[lost]])
        )
    if len(graph.period) != steps:
        raise ValueError("graph %s does not state a period for every step" % name)
    LOG.debug("verify exit graph=%s seconds=%.4f", name, time.perf_counter() - started)
