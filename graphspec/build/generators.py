import copy
import json
import logging
import pickle
import time

import numpy as np
import pyarrow as pa

from graphspec.facts.stored import stated as valued
from graphspec.infra.hashing import token, unit
from graphspec.infra.timing import timed
from graphspec.registries import Names, version
from graphspec.schema.decisions import GENERATORS, PARTS, RECORDED, RULES, TOGETHER, deciding, generator
from graphspec.schema.fields import fields
from graphspec.schema.schema import VALUE
from graphspec.schema.targets import real
from graphspec.views.visible import Looking

LOG = logging.getLogger(__name__)
DISTANCES = Names("distance between states", True)
DRAWN = ("actions", "together", "drawn")
ACTED = ("kind", "actor", "targets", "parameters")


@timed("generators.plain")
def plain(value):
    return float(value) if real(value) else value


@timed("generators.keyed")
def keyed(column):
    if isinstance(column, np.ndarray):
        return column
    if column and all(type(value) is str for value in column):
        return np.asarray([token(value) for value in column], np.uint64)
    return np.asarray(column, np.int64)


def written(option, symmetric):
    started = time.perf_counter()
    actions = [
        dict(
            kind=action["kind"],
            actor=action["actor"],
            targets={
                role: sorted(
                    [min(aim[0], aim[2]), aim[1], max(aim[0], aim[2])] if isinstance(aim, list) and aim[1] in symmetric else aim
                    for aim in aims
                )
                for role, aims in sorted(action["targets"].items())
            },
            parameters={name: plain(value) for name, value in sorted(action["parameters"].items())},
        )
        for action in option["actions"]
    ]
    together = option["together"] if len(actions) > 1 else "set"
    if together == "set":
        actions.sort(key=lambda action: json.dumps(action, sort_keys=True))
    result = json.dumps(dict(actions=actions, together=together), sort_keys=True)
    LOG.debug("written exit actions=%d seconds=%.6f", len(actions), time.perf_counter() - started)
    return result


class World:
    def __init__(self, truth, registry):
        started = time.perf_counter()
        schema = registry.schema
        self.truth, self.registry, self.schema, self.held = truth, registry, schema, {}
        self.listed = list(schema.tasks["kinds"])
        self.decided = truth.decided(registry)
        kinds = np.asarray(truth.kinds).tolist()
        self.lookup = {(kind, key): entity for entity, (kind, key) in enumerate(zip(kinds, truth.keys))}
        self.named = {}
        for kind, key in self.lookup:
            self.named.setdefault(key, []).append(kind)
        self.relations = frozenset(registry.tables)
        self.symmetric = frozenset(name for name in registry.tables if registry.by_code[registry.code(name)].symmetric)
        self.ends = {}
        for edge in (registry.by_code[registry.code(name)] for name in registry.tables):
            pairs = {(registry.kind(first), registry.kind(second)) for first, second in edge.pairs}
            if edge.symmetric:
                pairs |= {(second, first) for first, second in pairs}
            for first, second in pairs:
                self.ends.setdefault(edge.code, {}).setdefault(first, set()).add(second)
        names, self.acting = list(schema.tasks["references"]), {}
        for task, name, source in zip(
            *(np.asarray(truth.arrays[part]).tolist() for part in ("member_task", "member_name", "member_source"))
        ):
            target = self.target(task)
            if deciding(target) and names[name] == target["actors"]:
                self.acting.setdefault(task, {})[truth.keys[source]] = registry.nodes[kinds[source]]
        LOG.debug("world exit graph=%s seconds=%.4f", truth.name, time.perf_counter() - started)

    @timed("generators.World.target")
    def target(self, task):
        return self.schema.tasks["kinds"][self.listed[int(self.truth.task_kind[task])]]["target"]

    @timed("generators.World.where")
    def where(self, task):
        return "the decision %d of the kind %s of graph %s" % (
            int(self.truth.task_key[task]),
            self.listed[int(self.truth.task_kind[task])],
            self.truth.name,
        )

    def shown(self, task):
        started = time.perf_counter()
        if task not in self.held:
            self.held[task] = pickle.dumps(
                dict(
                    key=int(self.truth.task_key[task]),
                    kind=self.listed[int(self.truth.task_kind[task])],
                    time=int(self.truth.task_time[task]),
                    actors=dict(self.acting[task]),
                    options=[
                        dict(key=option["key"], chosen=option["chosen"], **self.canonical(option, self.target(task)))
                        for option in self.decided[task]["options"]
                        if option["origin"] == RECORDED
                    ],
                )
            )
        result = pickle.loads(self.held[task])
        LOG.debug("world shown exit options=%d seconds=%.6f", len(result["options"]), time.perf_counter() - started)
        return result

    @timed("generators.World.rank")
    def rank(self, key, kinds):
        codes = [self.registry.kind(kind) for kind in kinds]
        found = [code for code in codes if (code, key) in self.lookup]
        return codes.index(found[0]) if len(found) == 1 else len(codes)

    @timed("generators.World.aimed")
    def aimed(self, aim, kinds):
        if not isinstance(aim, list):
            return aim
        if aim[1] not in self.symmetric:
            return list(aim)
        first, second = sorted((aim[0], aim[2]), key=lambda end: (self.rank(end, kinds), end))
        return [first, aim[1], second]

    def canonical(self, option, target):
        started = time.perf_counter()
        roles = {kind: declared["roles"] for kind, declared in target["actions"].items()}
        actions = [
            dict(
                kind=action["kind"],
                actor=action["actor"],
                targets={
                    role: sorted(self.aimed(aim, roles[action["kind"]][role]["kinds"]) for aim in aims)
                    for role, aims in sorted(action["targets"].items())
                },
                parameters={name: plain(value) for name, value in sorted(action["parameters"].items())},
            )
            for action in option["actions"]
        ]
        if option["together"] == "set":
            actions.sort(key=lambda action: json.dumps(action, sort_keys=True))
        LOG.debug("world canonical exit actions=%d seconds=%.6f", len(actions), time.perf_counter() - started)
        return dict(actions=actions, together=option["together"])

    @timed("generators.World.content")
    def content(self, option):
        return token(written(option, self.symmetric))


class Frame:
    def __init__(self, world, held, view):
        started = time.perf_counter()
        self.world, self.view = world, view
        self.tasks = np.asarray(held["tasks"], np.int64)
        self.moments = np.asarray(held["moments"], np.int64)
        self.step = dict(zip(self.tasks.tolist(), np.asarray(held["task_step"], np.int64).tolist()))
        self.held = {}
        LOG.debug("frame exit graph=%s tasks=%d seconds=%.4f", world.truth.name, len(self.tasks), time.perf_counter() - started)

    def observed(self, step):
        started = time.perf_counter()
        if ("observed", step) not in self.held:
            view = self.view
            rows = (np.asarray(view.obs_start) <= step) & (np.asarray(view.obs_stop) > step)
            self.held[("observed", step)] = set(np.asarray(view.obs_entity)[rows].tolist())
        LOG.debug("frame observed exit step=%d seconds=%.6f", step, time.perf_counter() - started)
        return self.held[("observed", step)]

    def holding(self, step):
        started = time.perf_counter()
        if ("holding", step) not in self.held:
            view, flipped = self.view, self.world.registry.flipped
            rows = np.flatnonzero((np.asarray(view.fact_start) <= step) & (np.asarray(view.fact_stop) > step))
            found = set()
            for code, source, target in zip(
                *(np.asarray(view.arrays["fact_" + name])[rows].tolist() for name in ("code", "source", "target"))
            ):
                found.add((code, source, target))
                if flipped[code] == code:
                    found.add((code, target, source))
            self.held[("holding", step)] = found
        LOG.debug("frame holding exit step=%d seconds=%.6f", step, time.perf_counter() - started)
        return self.held[("holding", step)]

    def pickled(self, step):
        started = time.perf_counter()
        if ("state", step) not in self.held:
            world, view = self.world, self.view
            registry, schema, keys = world.registry, world.schema, world.truth.keys
            rows = np.flatnonzero((np.asarray(view.obs_start) <= step) & (np.asarray(view.obs_stop) > step))
            entity = np.asarray(view.obs_entity, np.int64)[rows]
            kinds = np.asarray(view.kinds, np.int64)[entity]
            nodes = {}
            for kind, columns in schema.entities().items():
                chosen = kinds == registry.kind(kind)
                values = valued(
                    registry, columns, np.asarray(view.obs_values)[rows[chosen]][:, registry.encoding.positions(columns)]
                )
                nodes[kind] = {
                    keys[owner]: {column: values[column][at] for column in columns}
                    for at, owner in enumerate(entity[chosen].tolist())
                }
            rows = np.flatnonzero((np.asarray(view.fact_start) <= step) & (np.asarray(view.fact_stop) > step))
            code = np.asarray(view.fact_code, np.int64)[rows]
            edges = []
            for table, entry in schema.edges.items():
                chosen = rows[np.isin(code, [registry.code(name) for name in registry.encoding.relations[table]])]
                values = valued(registry, entry["columns"], np.asarray(view.fact_values)[chosen])
                edges += [
                    dict(
                        relation=registry.by_code[int(view.fact_code[row])].name,
                        source=keys[int(view.fact_source[row])],
                        target=keys[int(view.fact_target[row])],
                        values={column: values[column][at] for column in entry["columns"]},
                    )
                    for at, row in enumerate(chosen.tolist())
                ]
            self.held[("state", step)] = pickle.dumps(dict(nodes=nodes, edges=edges))
        LOG.debug("frame pickled exit step=%d seconds=%.4f", step, time.perf_counter() - started)
        return self.held[("state", step)]

    def admitted(self, step, kind):
        started = time.perf_counter()
        if ("admitted", step, kind) not in self.held:
            world = self.world
            registry, keys, named = world.registry, world.truth.keys, world.named
            kinds = np.asarray(world.truth.kinds).tolist()
            observed, holding = self.observed(step), self.holding(step)
            nodes, edges = {}, {}
            for action, declared in world.schema.tasks["kinds"][kind]["target"]["actions"].items():
                for role, entry in declared["roles"].items():
                    codes = {registry.kind(name) for name in entry["kinds"]}
                    if entry["sort"] == "node":
                        nodes[(action, role)] = frozenset(
                            keys[source]
                            for source in observed
                            if kinds[source] in codes and len(codes.intersection(named[keys[source]])) == 1
                        )
                    elif entry["sort"] == "edge":
                        found = set()
                        for code, source, other in holding:
                            edge = registry.by_code[code]
                            if edge.code != code or edge.name not in world.relations or kinds[source] not in codes:
                                continue
                            ends = world.ends[code].get(kinds[source], set())
                            if (
                                len(codes.intersection(named[keys[source]])) == 1
                                and kinds[other] in ends
                                and len(ends.intersection(named[keys[other]])) == 1
                            ):
                                found.add((keys[source], edge.name, keys[other]))
                        edges[(action, role)] = frozenset(found)
            self.held[("admitted", step, kind)] = nodes, edges
        LOG.debug("frame admitted exit step=%d kind=%s seconds=%.4f", step, kind, time.perf_counter() - started)
        return self.held[("admitted", step, kind)]

    @timed("generators.Frame.before")
    def before(self, task):
        return [
            other
            for other in self.tasks.tolist()
            if other != task and self.step[other] <= self.step[task] and self.world.decided[other] is not None
        ]

    def seen(self, task, statement, present):
        started = time.perf_counter()
        world, name = self.world, statement["name"]
        reads, step, target = GENERATORS.named(name).reads, self.step[task], world.target(task)
        where, shown = world.where(task), world.shown(task)
        roles = {action: declared["roles"] for action, declared in target["actions"].items()}
        dictionaries = world.registry.encoding.dictionaries
        columns = {}
        for declared in target["actions"].values():
            for column in declared["parameters"]:
                kind = world.schema.kind(column)
                columns[column] = kind, tuple(dictionaries[column]) if kind.symbols else None
        check = Check(
            name,
            statement,
            world.truth.name,
            shown,
            target,
            *self.admitted(step, shown["kind"]),
            frozenset(
                (roles[action["kind"]][role]["kinds"][0], aim)
                for option in world.decided[task]["options"]
                if option["origin"] == RECORDED
                for action in option["actions"]
                for role, aims in action["targets"].items()
                if roles[action["kind"]][role]["sort"] == "new"
                for aim in aims
            ),
            world.relations,
            world.symmetric,
            columns,
            frozenset(present),
        )
        parts = dict(
            decision={part: value for part, value in shown.items() if part != "options"},
            unit=check.unit,
            closed=check.closed,
            fresh=check.fresh,
            content=check.content,
            formed=check.formed,
            kept=check.kept,
        )
        if "state" in reads:
            parts["state"] = pickle.loads(self.pickled(step))
            parts["at"] = Past(name, where, {int(self.moments[at]): self.pickled(at) for at in range(step + 1)}).at
        if "decisions" in reads:
            concurrent, parts["decisions"] = set(world.decided[task]["concurrent"]), []
            for other in self.before(task):
                found = world.shown(other)
                if found["key"] in concurrent:
                    del found["options"]
                parts["decisions"].append(found)
        if "chosen" in reads:
            parts["chosen"] = [option for option in shown["options"] if option["chosen"]]
        if "rule" in reads:
            parts["rule"] = copy.deepcopy(target["rule"])
        result = check, Seen(name, reads, where, parts)
        LOG.debug("frame seen exit generator=%s task=%d seconds=%.6f", name, task, time.perf_counter() - started)
        return result

    def taken(self, task, statement, check, drawn, present):
        started = time.perf_counter()
        world, name = self.world, statement["name"]
        entry, decision = GENERATORS.named(name), world.where(task)
        if not isinstance(drawn, list):
            raise ValueError("the generator of options %s gives for %s a list of options, found %r" % (name, decision, drawn))
        result = []
        for number, option in enumerate(drawn):
            where = "the option %d the generator of options %s drew for %s" % (number, name, decision)
            check.formed(option, DRAWN, where)
            reason = check.closed(option)
            if reason is not None:
                raise ValueError("%s %s" % (where, reason))
            probability = option["drawn"]
            if entry.probable != (probability is not None) or (
                probability is not None and not (real(probability) and 0 < probability <= 1)
            ):
                raise ValueError(
                    "%s states the probability of its draw as %r, and %s states %s"
                    % (
                        where,
                        probability,
                        name,
                        "the probability of each draw, above 0 and at most 1" if entry.probable else "none, having none",
                    )
                )
            held = world.canonical(option, world.target(task))
            content = world.content(held)
            if not statement["repeats"] and content in present:
                raise ValueError("%s equals an option already present, which its statement does not allow" % where)
            present.append(content)
            result.append(dict(held, drawn=None if probability is None else float(probability)))
        LOG.debug("frame taken exit generator=%s drawn=%d seconds=%.4f", name, len(result), time.perf_counter() - started)
        return result

    @timed("generators.Frame.generated")
    def generated(self, task):
        world = self.world
        target = world.target(task)
        if not deciding(target) or world.decided[task] is None:
            return []
        present = [world.content(option) for option in world.decided[task]["options"]]
        result = []
        for place, statement in enumerate(target["generators"]):
            check, seen = self.seen(task, statement, present)
            drawn = generator(statement["name"])(seen, statement["parameters"])
            for number, option in enumerate(self.taken(task, statement, check, drawn, present)):
                result.append(
                    dict(
                        task=int(world.truth.task_key[task]),
                        option=dict(
                            key="%s.%d.%d" % (statement["name"], place, number),
                            actions=option["actions"],
                            together=option["together"],
                            available=True,
                            chosen=False,
                            order=None,
                            shown=None,
                            logged=None,
                            origin=dict(generator=statement["name"], seed=statement["seed"], drawn=option["drawn"]),
                            outcome=None,
                        ),
                    )
                )
        return result


class Seen:
    def __init__(self, generator, reads, where, parts):
        started = time.perf_counter()
        self.__dict__.update(parts, generator=generator, reads=reads, where=where)
        LOG.debug("seen exit generator=%s parts=%d seconds=%.6f", generator, len(parts), time.perf_counter() - started)

    @timed("generators.Seen.__getattr__")
    def __getattr__(self, name):
        part = "state" if name == "at" else name
        if part not in PARTS:
            raise AttributeError(name)
        held = vars(self)
        raise ValueError(
            "the generator of options %s reads the %s of %s, which it does not state among what it reads: %s"
            % (held["generator"], part, held["where"], ", ".join(held["reads"]) or "nothing")
        )


class Past:
    def __init__(self, generator, where, held):
        started = time.perf_counter()
        self.generator, self.where, self.held = generator, where, held
        LOG.debug("past exit generator=%s steps=%d seconds=%.6f", generator, len(held), time.perf_counter() - started)

    @timed("generators.Past.at")
    def at(self, moment):
        if moment not in self.held:
            raise ValueError(
                "the generator of options %s reads the state of %s at %r, which is no step its task sees at or before "
                "its own" % (self.generator, self.where, moment)
            )
        return pickle.loads(self.held[moment])


class Check:
    def __init__(self, name, statement, graph, shown, target, nodes, edges, created, relations, symmetric, columns, present):
        started = time.perf_counter()
        self.name, self.seed, self.repeats, self.graph = name, statement["seed"], statement["repeats"], graph
        self.key, self.actors, self.actions = shown["key"], frozenset(shown["actors"]), copy.deepcopy(target["actions"])
        self.nothing = target["nothing"] is not None
        self.nodes, self.edges, self.created = nodes, edges, created
        self.relations, self.symmetric, self.columns, self.present = relations, symmetric, columns, present
        LOG.debug("check exit generator=%s decision=%d seconds=%.6f", name, self.key, time.perf_counter() - started)

    @timed("generators.Check.unit")
    def unit(self, *columns):
        count = len(columns[0])
        return unit(
            self.seed,
            self.name,
            np.full(count, token(self.graph), np.uint64),
            np.full(count, self.key, np.int64),
            *(keyed(column) for column in columns),
        )

    @timed("generators.Check.held")
    def held(self, name, value):
        kind, dictionary = self.columns[name]
        if kind.numeric and not real(value):
            return False
        try:
            codes = kind.codes([value], dictionary)
        except (TypeError, ValueError):
            return False
        back = kind.stated(codes, dictionary)
        return (
            back[0] is not None
            and (kind.numeric or type(back[0]) is type(value))
            and np.array_equal(kind.codes(back, dictionary), codes)
        )

    @timed("generators.Check.kept")
    def kept(self, name, value):
        kind, dictionary = self.columns[name]
        return kind.stated(kind.codes([value], dictionary), dictionary)[0]

    def formed(self, option, names, where):
        started = time.perf_counter()
        option = fields(option, names, where)
        if option["together"] not in TOGETHER or not isinstance(option["actions"], list):
            raise ValueError("%s states its actions as a list, and as one of %s" % (where, ", ".join(TOGETHER)))
        for place, action in enumerate(option["actions"]):
            inner = "the action %d of %s" % (place, where)
            action = fields(action, ACTED, inner)
            if (
                type(action["kind"]) is not str
                or type(action["actor"]) is not str
                or not isinstance(action["targets"], dict)
                or not isinstance(action["parameters"], dict)
            ):
                raise ValueError("%s states its kind, its actor, its targets and its parameters" % inner)
            for name, value in action["parameters"].items():
                if name not in self.columns:
                    raise ValueError("%s states the parameter %s, which no kind of action of its kind of task states" % (inner, name))
                if value is not None and not self.held(name, value):
                    raise ValueError(
                        "%s states the parameter %s as %r, not a value of its column the bank holds" % (inner, name, value)
                    )
        LOG.debug("check formed exit actions=%d seconds=%.6f", len(option["actions"]), time.perf_counter() - started)

    @timed("generators.Check.closed")
    def closed(self, option):
        if not option["actions"]:
            return None if self.nothing else "holds no action, and its kind of task states no option that does nothing"
        for place, action in enumerate(option["actions"]):
            reason = self.unfit(action)
            if reason is not None:
                return "names in its action %d %s" % (place, reason)
        return None

    @timed("generators.Check.unfit")
    def unfit(self, action):
        if action["kind"] not in self.actions:
            return "the kind %s, which its kind of task does not state" % action["kind"]
        declared = self.actions[action["kind"]]
        if action["actor"] not in self.actors:
            return "the actor %s, not an actor of its decision" % action["actor"]
        if set(action["targets"]) != set(declared["roles"]) or set(action["parameters"]) != set(declared["parameters"]):
            return "other roles or parameters than its kind %s states" % action["kind"]
        for role, aims in action["targets"].items():
            (least, most), seen = declared["roles"][role]["count"], set()
            if not isinstance(aims, list):
                return "under the role %s %r, not the list of its targets" % (role, aims)
            if len(aims) < least or (most is not None and len(aims) > most):
                return "under the role %s %d targets, and its kind %s states from %d to %s" % (
                    role,
                    len(aims),
                    action["kind"],
                    least,
                    "no bound" if most is None else most,
                )
            for aim in aims:
                reason = self.aimless(action["kind"], role, declared["roles"][role], aim)
                if reason is not None:
                    return "under the role %s %s" % (role, reason)
                content = (
                    aim
                    if type(aim) is str
                    else (min(aim[0], aim[2]), aim[1], max(aim[0], aim[2]))
                    if aim[1] in self.symmetric
                    else tuple(aim)
                )
                if content in seen:
                    return "under the role %s %r twice; the targets of a role are a set" % (role, aim)
                seen.add(content)
        return None

    @timed("generators.Check.aimless")
    def aimless(self, kind, role, declared, aim):
        if declared["sort"] == "new":
            if type(aim) is str and (declared["kinds"][0], aim) in self.created:
                return None
            return "the element %r, which no option recorded for its decision creates" % (aim,)
        if declared["sort"] == "node":
            if type(aim) is str and aim in self.nodes[(kind, role)]:
                return None
            return "the node %r, which is not one node of its kinds observed at the step of its decision" % (aim,)
        if not (
            isinstance(aim, list) and len(aim) == 3 and all(type(part) is str for part in aim) and aim[1] in self.relations
        ):
            return "%r, which is not the first node, a relation stated and the second node of an edge" % (aim,)
        held = self.edges[(kind, role)]
        if tuple(aim) in held or (aim[1] in self.symmetric and (aim[2], aim[1], aim[0]) in held):
            return None
        return "the edge %r, which does not hold at the step of its decision" % (aim,)

    @timed("generators.Check.content")
    def content(self, option):
        return token(written(option, self.symmetric))

    @timed("generators.Check.fresh")
    def fresh(self, option):
        return self.repeats or self.content(option) not in self.present


@timed("generators.looking")
def looking(splits, trained):
    return Looking(splits, trained, (), {name: [] for name in trained})


def regenerated(truth, registry, shown):
    started = time.perf_counter()
    world, result = World(truth, registry), []
    for held, view in shown:
        frame = Frame(world, held, view)
        for task in frame.tasks.tolist():
            result += frame.generated(task)
    LOG.info("regenerated exit graph=%s drawn=%d seconds=%.4f", truth.name, len(result), time.perf_counter() - started)
    return result


def appended(registry, tables, drawn):
    started = time.perf_counter()
    found, columns = tables["decisions"], registry.schema.parameters()
    position = {key: row for row, key in enumerate(tables["tasks"].column("key").to_pylist())}
    options, actions, roles = [], [], []
    for entry in drawn:
        option = entry["option"]
        origin = option["origin"]
        options.append(
            dict(
                key=option["key"],
                sequence=option["together"] == "sequence",
                available=option["available"],
                chosen=option["chosen"],
                generator=origin["generator"],
                order=option["order"],
                shown=option["shown"],
                logged=option["logged"],
                seed=origin["seed"],
                drawn=origin["drawn"],
                task=position[entry["task"]],
            )
        )
        for action in option["actions"]:
            actions.append(
                dict(
                    option=found["option"].num_rows + len(options) - 1,
                    kind=action["kind"],
                    actor=action["actor"],
                    **{VALUE + name: action["parameters"][name] if name in action["parameters"] else None for name in columns},
                )
            )
            for role, aims in action["targets"].items():
                for aim in aims:
                    edged = isinstance(aim, list)
                    roles.append(
                        dict(
                            role=role,
                            key=aim[0] if edged else aim,
                            relation=aim[1] if edged else None,
                            to_key=aim[2] if edged else None,
                            action=found["action"].num_rows + len(actions) - 1,
                        )
                    )
    result = dict(
        found,
        **{
            name: pa.concat_tables([found[name], pa.Table.from_pylist(rows, schema=found[name].schema)])
            for name, rows in (("option", options), ("action", actions), ("role", roles))
        },
    )
    LOG.debug("appended exit options=%d actions=%d seconds=%.4f", len(options), len(actions), time.perf_counter() - started)
    return result


def recorded(schema):
    started = time.perf_counter()
    names = sorted({held["name"] for kind in schema.decided() for held in schema.tasks["kinds"][kind]["target"]["generators"]})
    result = {name: dict(version=version(GENERATORS.named(name)), probabilities=GENERATORS.named(name).probable) for name in names}
    LOG.debug("recorded exit generators=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("generators.counted")
def counted(parameters, names, where):
    parameters = fields(parameters, names, where)
    for name in names:
        if type(parameters[name]) is not int or parameters[name] < 1:
            raise ValueError("%s states %s as a whole number from 1, found %r" % (where, name, parameters[name]))
    return parameters


class Pooled:
    probable = True

    def drawn(self, seen, candidates, count):
        started = time.perf_counter()
        pool = {}
        for option in candidates:
            content = seen.content(option)
            if content not in pool and seen.closed(option) is None and seen.fresh(option):
                pool[content] = dict(actions=option["actions"], together=option["together"])
        tokens = np.asarray(sorted(pool), np.uint64)
        order = np.lexsort((tokens, seen.unit(tokens)))[:count].tolist()
        result = [dict(pool[int(tokens[at])], drawn=min(count, len(tokens)) / len(tokens)) for at in order]
        LOG.debug("pooled drawn exit pool=%d drawn=%d seconds=%.4f", len(tokens), len(result), time.perf_counter() - started)
        return result


class Scripted:
    def __init__(self, function, reads, probable, version):
        started = time.perf_counter()
        if not callable(function):
            raise ValueError("a generator of options of a script is a function of what its decision may see and its parameters")
        self.function, self.reads, self.probable, self.version = function, reads, probable, version
        LOG.debug("scripted exit function=%s seconds=%.6f", function.__name__, time.perf_counter() - started)

    @timed("generators.Scripted.checked")
    def checked(self, parameters, schema, target, references, where):
        if not isinstance(parameters, dict):
            raise ValueError("%s states its parameters as a mapping, found %r" % (where, parameters))
        return parameters

    @timed("generators.Scripted.__call__")
    def __call__(self, seen, parameters):
        return self.function(seen, parameters)


class Others(Pooled):
    version = 9
    reads = ("decisions",)

    @timed("generators.Others.checked")
    def checked(self, parameters, schema, target, references, where):
        return counted(parameters, ("count",), where)

    @timed("generators.Others.__call__")
    def __call__(self, seen, parameters):
        kind = seen.decision["kind"]
        found = [option for other in seen.decisions if other["kind"] == kind and "options" in other for option in other["options"]]
        return self.drawn(seen, found, parameters["count"])


class Euclidean:
    version = 1

    @timed("generators.Euclidean.__call__")
    def __call__(self, left, right):
        return np.sqrt(((np.asarray(right, np.float64) - np.asarray(left, np.float64)) ** 2).sum(1))


class Nearest(Pooled):
    version = 9
    reads = ("state", "decisions")

    def checked(self, parameters, schema, target, references, where):
        started = time.perf_counter()
        parameters = fields(parameters, ("count", "states", "columns", "distance"), where)
        counted(dict(count=parameters["count"], states=parameters["states"]), ("count", "states"), where)
        kinds, columns = references[target["actors"]]["kinds"], parameters["columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or len(set(columns)) != len(columns)
            or any(
                type(column) is not str
                or column not in schema.columns
                or any(kind not in schema.nodes or column not in schema.nodes[kind] for kind in kinds)
                for column in columns
            )
        ):
            raise ValueError(
                "%s states its columns as a list of distinct columns that every kind of the actors, %s, states, found %r"
                % (where, ", ".join(kinds), columns)
            )
        for column in columns:
            if not schema.kind(column).numeric:
                raise ValueError(
                    "%s measures the column %s of the type %s, which is not numeric"
                    % (where, column, schema.columns[column]["type"])
                )
        DISTANCES.named(parameters["distance"])
        LOG.debug("nearest checked exit columns=%d seconds=%.6f", len(columns), time.perf_counter() - started)
        return parameters

    def vector(self, seen, decision, columns):
        started = time.perf_counter()
        nodes, result = seen.at(decision["time"])["nodes"], []
        for column in columns:
            values = [
                nodes[kind][key][column]
                for key, kind in sorted(decision["actors"].items())
                if key in nodes[kind] and nodes[kind][key][column] is not None
            ]
            if not values:
                raise ValueError(
                    "the generator of options nearest finds no value of %s for the actors of the decision %d at %d"
                    % (column, decision["key"], decision["time"])
                )
            result.append(sum(values) / len(values))
        LOG.debug("nearest vector exit columns=%d seconds=%.6f", len(columns), time.perf_counter() - started)
        return result

    @timed("generators.Nearest.__call__")
    def __call__(self, seen, parameters):
        kind, columns = seen.decision["kind"], parameters["columns"]
        others = [other for other in seen.decisions if other["kind"] == kind and "options" in other]
        if not others:
            return []
        distances = DISTANCES.named(parameters["distance"])(
            self.vector(seen, seen.decision, columns), [self.vector(seen, other, columns) for other in others]
        )
        order = np.lexsort(([other["key"] for other in others], distances))[: parameters["states"]].tolist()
        found = [option for at in order for option in others[at]["options"]]
        return self.drawn(seen, found, parameters["count"])


class Changed:
    version = 9
    reads = ("chosen",)
    probable = False

    def checked(self, parameters, schema, target, references, where):
        started = time.perf_counter()
        parameters = fields(parameters, ("count", "ranges"), where)
        counted(dict(count=parameters["count"]), ("count",), where)
        stated = {name for action in target["actions"].values() for name in action["parameters"]}
        ranges = parameters["ranges"]
        if (
            not isinstance(ranges, dict)
            or not ranges
            or any(
                column not in stated
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(real(bound) for bound in bounds)
                or bounds[0] > bounds[1]
                for column, bounds in ranges.items()
            )
        ):
            raise ValueError(
                "%s states its ranges as a mapping from a parameter of its kinds of action to its least and greatest "
                "change, found %r" % (where, ranges)
            )
        for column in ranges:
            if not schema.kind(column).numeric:
                raise ValueError(
                    "%s changes the parameter %s of the type %s, which is not numeric"
                    % (where, column, schema.columns[column]["type"])
                )
        LOG.debug("changed checked exit columns=%d seconds=%.6f", len(ranges), time.perf_counter() - started)
        return parameters

    @timed("generators.Changed.__call__")
    def __call__(self, seen, parameters):
        ranges, chosen = parameters["ranges"], sorted(seen.chosen, key=lambda option: option["key"])
        rows = [
            (place, number, action, column)
            for place, option in enumerate(chosen)
            for number in range(parameters["count"])
            for action, held in enumerate(option["actions"])
            for column in sorted(ranges)
            if column in held["parameters"] and held["parameters"][column] is not None
        ]
        if not rows:
            return []
        draws = seen.unit(*([chosen[row[0]]["key"] for row in rows], *([row[at] for row in rows] for at in (1, 2, 3))))
        found = {}
        for (place, number, action, column), draw in zip(rows, draws.tolist()):
            option = found.setdefault((place, number), copy.deepcopy(chosen[place]))
            low, high = ranges[column]
            held = option["actions"][action]["parameters"]
            held[column] = seen.kept(column, held[column] + low + (high - low) * draw)
        result = [
            dict(actions=option["actions"], together=option["together"], drawn=None)
            for _, option in sorted(found.items())
        ]
        return [option for option in result if seen.fresh(option)]


class Ruled(Pooled):
    version = 9
    reads = ("rule", "state")

    @timed("generators.Ruled.checked")
    def checked(self, parameters, schema, target, references, where):
        rule = RULES.named(target["rule"]["name"])
        if not hasattr(rule, "options") or not callable(rule.options):
            raise ValueError(
                "%s draws from the rule %s, which states no options, the options it gives a decision"
                % (where, target["rule"]["name"])
            )
        return counted(parameters, ("count",), where)

    @timed("generators.Ruled.__call__")
    def __call__(self, seen, parameters):
        rule = seen.rule
        found = RULES.named(rule["name"]).options(seen, rule["parameters"])
        if not isinstance(found, list):
            raise ValueError("the rule of options %s gives for %s a list of options, found %r" % (rule["name"], seen.where, found))
        for place, option in enumerate(found):
            seen.formed(option, ("actions", "together"), "the option %d the rule %s gives %s" % (place, rule["name"], seen.where))
        return self.drawn(seen, found, parameters["count"])


GENERATORS.register("others", Others())
GENERATORS.register("nearest", Nearest())
GENERATORS.register("changed", Changed())
GENERATORS.register("ruled", Ruled())
DISTANCES.register("euclidean", Euclidean())
