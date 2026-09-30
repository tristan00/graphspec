import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import Mapped, load_arrays, read_json, save_arrays, write_json
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.schema import time as clock
from graphspec.schema.references import MEMBERS
from graphspec.schema.schema import VALUE, inward, outward
from graphspec.schema.time import ALWAYS, NEVER

LOG = logging.getLogger(__name__)
VERSION = 8
JOINED = ("set_edge", "set_member", "set_role", "set_values")
HELD = ("inner_owner", "inner_kind", "inner_values")
WITHIN = ("within_source", "within_target", "within_code", "within_values")
HALF = int(np.finfo(np.float16).max)
GROUPS = ("obs", "fact")
SPARSE = ("known_row", "known_time", "replaced_row", "replaced_time", "seen_row", "seen_observer")
STORED = (
    ("kinds", "entity_token")
    + ("obs_entity", "obs_since", "obs_until", "obs_seq", "obs_values")
    + ("fact_source", "fact_target", "fact_code", "fact_present")
    + ("fact_since", "fact_until", "fact_seq", "fact_values")
    + tuple(group + "_" + name for group in GROUPS for name in SPARSE)
    + ("task_time", "task_observer", "task_grouping", "task_key", "task_kind", "task_split")
    + ("task_weight", "task_values", "task_signals")
    + MEMBERS
    + ("target_task", "target_row", "target_known", "target_y")
    + ("graph_values",)
    + JOINED
    + HELD
    + WITHIN
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


def prefixed(columns):
    return {VALUE + name: values for name, values in columns.items()}


def encoded(registry, values, held, graph, named):
    started = time.perf_counter()
    wide = registry.encoding.inputs(values, held, graph)
    with np.errstate(over="ignore"):
        result = wide.astype(np.float16)
    lost = ~np.isfinite(result)
    if lost.any():
        row, place = (int(found[0]) for found in np.nonzero(lost))
        raise ValueError(
            "%s holds %s which gives %s; model inputs are stored at 16 bits, which hold magnitudes up to %d"
            % (
                named(row),
                registry.encoding.origin(
                    place, np.asarray(values)[row], None if graph is None else np.asarray(graph)[row]
                ),
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


@timed("truth.ranks")
def ranks(keys):
    order = sorted(range(len(keys)), key=keys.__getitem__)
    result = np.empty(len(keys), np.int64)
    result[order] = np.arange(len(keys))
    return result
