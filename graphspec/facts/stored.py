import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import load_arrays, read_json, read_pack, save_arrays, write_json, write_pack
from graphspec.infra.hashing import token
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema import time as clock
from graphspec.schema.decisions import RECORDED
from graphspec.schema.references import MEMBERS
from graphspec.schema.schema import VALUE, inward, outward
from graphspec.schema.time import ALWAYS, NEVER

LOG = logging.getLogger(__name__)
VERSION = 19
MANIFEST = (
    "version",
    "trained",
    "spec",
    "spec_sha256",
    "splits",
    "split_rows",
    "observers",
    "graphs",
    "supervision_sha256",
    "stored_sha256",
)
PACKED = "arrays.pack"
JOINED = ("set_edge", "set_member", "set_role", "set_values")
HELD = ("inner_owner", "inner_kind", "inner_values")
WITHIN = ("within_source", "within_target", "within_code", "within_values")
HALF = int(np.finfo(np.float16).max)
GROUPS = ("obs", "fact")
SPARSE = ("known_row", "known_time", "replaced_row", "replaced_time", "seen_row", "seen_observer")
DECIDED = dict(
    next_task=np.int32,
    next_actor=np.int32,
    next_other=np.int32,
    concurrent_task=np.int32,
    concurrent_other=np.int32,
    option_task=np.int32,
    option_sequence=np.bool_,
    option_available=np.bool_,
    option_chosen=np.bool_,
    option_order=np.int32,
    option_shown=np.float64,
    option_logged=np.float64,
    origin_option=np.int32,
    origin_generator=np.int32,
    origin_seed=np.int64,
    origin_drawn=np.float64,
    action_option=np.int32,
    action_kind=np.int16,
    action_actor=np.int32,
    action_values=np.float32,
    role_action=np.int32,
    role_name=np.int16,
    role_source=np.int32,
    role_target=np.int32,
    role_code=np.int16,
)
STORED = (
    ("kinds", "entity_token")
    + ("obs_entity", "obs_since", "obs_until", "obs_seq", "obs_values")
    + ("fact_source", "fact_target", "fact_code", "fact_present")
    + ("fact_since", "fact_until", "fact_seq", "fact_values")
    + tuple(group + "_" + name for group in GROUPS for name in SPARSE)
    + ("task_time", "task_observer", "task_grouping", "task_key", "task_kind", "task_split")
    + ("task_weight", "task_values", "task_signals", "task_ended", "task_discount")
    + MEMBERS
    + ("target_task", "target_row", "target_known", "target_y", "target_series", "target_node")
    + ("target_option", "target_part")
    + ("graph_values",)
    + JOINED
    + HELD
    + WITHIN
    + tuple(DECIDED)
)


@timed("stored.admitted")
def admitted(directory, manifest):
    if type(manifest) is not dict:
        raise ValueError(
            "the manifest.json of the bank %s holds %r and no mapping; a bank holds the manifest its build wrote"
            % (directory, manifest)
        )
    if "version" in manifest and manifest["version"] != VERSION:
        raise ValueError(
            "bank %s has version %s; this code reads version %d" % (directory, manifest["version"], VERSION)
        )
    lacking = [name for name in MANIFEST if name not in manifest]
    if lacking:
        raise ValueError(
            "the manifest.json of the bank %s lacks %s; a bank holds the manifest its build wrote"
            % (directory, ", ".join(lacking))
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

    @timed("stored.Arrays.replace")
    def replace(self, steps=None, **changes):
        return type(self)(
            self.name, self.keys, self.steps if steps is None else steps, **dict(self.arrays, **changes)
        )

    @timed("stored.Arrays.save")
    def save(self, directory):
        directory = output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_pack(directory / PACKED, self.arrays)
        write_json(directory / "graph.json", dict(name=self.name, keys=self.keys, steps=self.steps))

    @classmethod
    @timed("stored.Arrays.load")
    def load(cls, directory, mapped=False):
        directory = Path(directory)
        meta = read_json(directory / "graph.json")
        return cls(meta["name"], meta["keys"], meta["steps"], **read_pack(directory / PACKED, mapped))


@timed("stored.moment")
def moment(value):
    value = int(value)
    return None if value <= ALWAYS or value >= NEVER else value


@timed("stored.known")
def known(value):
    return None if np.isnan(value) else float(value)


class Facts:
    NAMES = STORED

    @timed("truth.facts")
    def __init__(self, name, keys, unit, inner, observers, options, generators, **arrays):
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
            options=list(options),
            generators=list(generators),
            arrays={key: arrays[key] for key in self.NAMES},
        )

    @timed("truth.facts.replace")
    def replace(self, **changes):
        return Facts(
            self.name,
            self.keys,
            self.unit,
            self.inner,
            self.observers,
            self.options,
            self.generators,
            **dict(self.arrays, **changes),
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
            dict(
                name=self.name,
                keys=self.keys,
                unit=self.unit,
                inner=self.inner,
                observers=self.observers,
                options=self.options,
                generators=self.generators,
            ),
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
            meta["options"],
            meta["generators"],
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
            decisions=self.decided(registry),
        )
        LOG.debug("stated exit graph=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def decided(self, registry):
        started = time.perf_counter()
        schema, keys, names = registry.schema, self.keys, [edge.name for edge in registry.edges]
        kinds, columns, series = list(schema.tasks["kinds"]), schema.parameters(), schema.series()
        task_keys, task_kinds = np.asarray(self.task_key).tolist(), np.asarray(self.task_kind).tolist()
        ended, discount = np.asarray(self.task_ended).tolist(), np.asarray(self.task_discount, np.float64)
        result, options, actions = [None] * len(task_keys), [], []
        for at, task in enumerate(np.asarray(self.option_task).tolist()):
            entry = dict(
                key=self.options[at],
                actions=[],
                together="sequence" if self.option_sequence[at] else "set",
                available=bool(self.option_available[at]),
                chosen=bool(self.option_chosen[at]),
                order=None if self.option_order[at] < 0 else int(self.option_order[at]),
                shown=known(self.option_shown[at]),
                logged=known(self.option_logged[at]),
                origin=RECORDED,
                outcome=None,
            )
            if result[task] is None:
                result[task] = dict(
                    options=[],
                    concurrent=[],
                    next={},
                    ended=None if ended[task] < 0 else bool(ended[task]),
                    discount=known(discount[task]),
                )
            result[task]["options"].append(entry)
            options.append(entry)
        for option, generator, seed, drawn in zip(
            *(np.asarray(self.arrays["origin_" + name]).tolist() for name in ("option", "generator", "seed", "drawn"))
        ):
            options[option]["origin"] = dict(generator=self.generators[generator], seed=seed, drawn=known(drawn))
        values = stated(registry, columns, np.asarray(self.action_values))
        for at, (owner, kind, actor) in enumerate(
            zip(*(np.asarray(self.arrays[name]).tolist() for name in ("action_option", "action_kind", "action_actor")))
        ):
            target = schema.tasks["kinds"][kinds[task_kinds[int(self.option_task[owner])]]]["target"]
            name = list(target["actions"])[kind]
            declared = target["actions"][name]
            entry = dict(
                kind=name,
                actor=keys[actor],
                targets={role: [] for role in declared["roles"]},
                parameters={column: values[column][at] for column in declared["parameters"]},
            )
            options[owner]["actions"].append(entry)
            actions.append((entry, declared))
        for action, role, source, other, code in zip(
            *(
                np.asarray(self.arrays["role_" + name]).tolist()
                for name in ("action", "name", "source", "target", "code")
            )
        ):
            entry, declared = actions[action]
            name = list(declared["roles"])[role]
            edged = declared["roles"][name]["sort"] == "edge"
            entry["targets"][name].append([keys[source], names[code], keys[other]] if edged else keys[source])
        for owner, part, code, y in zip(
            *(np.asarray(self.arrays[name]).tolist() for name in ("target_option", "target_part", "target_series", "target_y"))
        ):
            if owner < 0:
                continue
            entry = options[owner]
            if part < 0:
                entry["outcome"] = entry["outcome"] or {}
                entry["outcome"][series[code]] = float(y)
                continue
            entry["outcome"] = entry["outcome"] or [None] * len(entry["actions"])
            entry["outcome"][part] = entry["outcome"][part] or {}
            entry["outcome"][part][series[code]] = float(y)
        for task, other in zip(np.asarray(self.concurrent_task).tolist(), np.asarray(self.concurrent_other).tolist()):
            result[task]["concurrent"].append(task_keys[other])
        for task, actor, other in zip(
            *(np.asarray(self.arrays["next_" + name]).tolist() for name in ("task", "actor", "other"))
        ):
            result[task]["next"][keys[actor]] = task_keys[other]
        for held in result:
            if held is not None:
                held["concurrent"].sort()
        LOG.debug("facts decided exit graph=%s options=%d seconds=%.4f", self.name, len(options), time.perf_counter() - started)
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


@timed("stored.tokens")
def tokens(registry, kinds, keys):
    return np.asarray(
        [token(registry.nodes[kind] + ":" + key) for kind, key in zip(np.asarray(kinds).tolist(), keys)],
        np.uint64,
    )


@timed("stored.column")
def column(table, name, dtype):
    return table.column(name).to_numpy().astype(dtype)


@timed("stored.matrix")
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


@timed("stored.prefixed")
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
