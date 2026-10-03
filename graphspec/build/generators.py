import copy
import json
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.facts.stored import stated as valued
from graphspec.infra.hashing import token, unit
from graphspec.infra.timing import timed
from graphspec.registries import version
from graphspec.schema.decisions import GENERATORS, RECORDED, TOGETHER, deciding, generator
from graphspec.schema.fields import fields
from graphspec.schema.schema import VALUE
from graphspec.schema.targets import real

LOG = logging.getLogger(__name__)
DRAWN = ("actions", "together", "drawn")
ACTED = ("kind", "actor", "targets", "parameters")


@timed("generators.plain")
def plain(value):
    return float(value) if real(value) else value


class World:
    def __init__(self, truth, registry):
        started = time.perf_counter()
        schema = registry.schema
        self.truth, self.registry, self.schema = truth, registry, schema
        self.listed = list(schema.tasks["kinds"])
        self.decided = truth.decided(registry)
        kinds = np.asarray(truth.kinds).tolist()
        self.lookup = {(kind, key): entity for entity, (kind, key) in enumerate(zip(kinds, truth.keys))}
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
        result = dict(
            key=int(self.truth.task_key[task]),
            kind=self.listed[int(self.truth.task_kind[task])],
            time=int(self.truth.task_time[task]),
            actors=dict(self.acting[task]),
            options=[
                dict(
                    key=option["key"],
                    actions=copy.deepcopy(option["actions"]),
                    together=option["together"],
                    chosen=option["chosen"],
                )
                for option in self.decided[task]["options"]
                if option["origin"] == RECORDED
            ],
        )
        LOG.debug("world shown exit options=%d seconds=%.6f", len(result["options"]), time.perf_counter() - started)
        return result

    @timed("generators.World.aimed")
    def aimed(self, aim):
        if not isinstance(aim, list):
            return aim
        first, relation, second = aim
        if relation in self.registry.tables and self.registry.by_code[self.registry.code(relation)].symmetric:
            first, second = sorted((first, second))
        return [first, relation, second]

    def canonical(self, option):
        started = time.perf_counter()
        actions = [
            dict(
                kind=action["kind"],
                actor=action["actor"],
                targets={role: self.aimed(aim) for role, aim in sorted(action["targets"].items())},
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
        return json.dumps(self.canonical(option), sort_keys=True)


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

    def state(self, step):
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
            self.held[("state", step)] = dict(nodes=nodes, edges=edges)
        LOG.debug("frame state exit step=%d seconds=%.4f", step, time.perf_counter() - started)
        return copy.deepcopy(self.held[("state", step)])

    @timed("generators.Frame.before")
    def before(self, task):
        return [
            other
            for other in self.tasks.tolist()
            if other != task and self.step[other] <= self.step[task] and self.world.decided[other] is not None
        ]

    @timed("generators.Frame.generated")
    def generated(self, task):
        world = self.world
        target = world.target(task)
        if not deciding(target) or world.decided[task] is None:
            return []
        present = [world.content(option) for option in world.decided[task]["options"]]
        result = []
        for place, statement in enumerate(target["generators"]):
            seen = Seen(self, task, statement, present)
            for number, option in enumerate(seen.taken(generator(statement["name"])(seen, statement["parameters"]))):
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
    def __init__(self, frame, task, statement, present):
        started = time.perf_counter()
        world = frame.world
        self.frame, self.world, self.task, self.statement, self.present = frame, world, task, statement, present
        self.entry, self.step = GENERATORS.named(statement["name"]), frame.step[task]
        self.target, self.where = world.target(task), world.where(task)
        self.decision = {name: value for name, value in world.shown(task).items() if name != "options"}
        roles = {name: action["roles"] for name, action in self.target["actions"].items()}
        self.created = {
            (roles[action["kind"]][role]["kinds"][0], aim)
            for option in world.decided[task]["options"]
            if option["origin"] == RECORDED
            for action in option["actions"]
            for role, aim in action["targets"].items()
            if roles[action["kind"]][role]["sort"] == "new"
        }
        LOG.debug("seen exit generator=%s task=%d seconds=%.6f", statement["name"], task, time.perf_counter() - started)

    def reading(self, part):
        started = time.perf_counter()
        if part not in self.entry.reads:
            raise ValueError(
                "the generator of options %s reads the %s of %s, which it does not state among what it reads: %s"
                % (self.statement["name"], part, self.where, ", ".join(self.entry.reads) or "nothing")
            )
        LOG.debug("seen reading exit part=%s seconds=%.6f", part, time.perf_counter() - started)

    @property
    @timed("generators.Seen.state")
    def state(self):
        self.reading("state")
        return self.frame.state(self.step)

    def at(self, moment):
        started = time.perf_counter()
        self.reading("state")
        moments = self.frame.moments
        step = int(np.searchsorted(moments, moment))
        if step > self.step or step >= len(moments) or moments[step] != moment:
            raise ValueError(
                "the generator of options %s reads the state of %s at %r, which is no step its task sees at or before "
                "its own" % (self.statement["name"], self.where, moment)
            )
        LOG.debug("seen at exit step=%d seconds=%.6f", step, time.perf_counter() - started)
        return self.frame.state(step)

    @property
    @timed("generators.Seen.decisions")
    def decisions(self):
        self.reading("decisions")
        return [self.world.shown(other) for other in self.frame.before(self.task)]

    @property
    @timed("generators.Seen.chosen")
    def chosen(self):
        self.reading("chosen")
        return [option for option in self.world.shown(self.task)["options"] if option["chosen"]]

    @property
    @timed("generators.Seen.rule")
    def rule(self):
        self.reading("rule")
        return copy.deepcopy(self.target["rule"])

    @timed("generators.Seen.unit")
    def unit(self, *columns):
        count = len(columns[0])
        return unit(
            self.statement["seed"],
            self.statement["name"],
            np.full(count, token(self.world.truth.name), np.uint64),
            np.full(count, self.decision["key"], np.int64),
            *columns,
        )

    def formed(self, option, names, where):
        started = time.perf_counter()
        option = fields(option, names, where)
        if option["together"] not in TOGETHER or not isinstance(option["actions"], list):
            raise ValueError("%s states its actions as a list, and as one of %s" % (where, ", ".join(TOGETHER)))
        columns = self.world.schema.columns
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
                kind = columns[name]["type"] if name in columns else None
                if value is not None and not (
                    (kind == "number" and real(value))
                    or (kind == "flag" and type(value) is bool)
                    or (kind == "category" and type(value) is str)
                ):
                    raise ValueError("%s states the parameter %s as %r, not in the type of its column" % (inner, name, value))
        LOG.debug("seen formed exit actions=%d seconds=%.6f", len(option["actions"]), time.perf_counter() - started)

    @timed("generators.Seen.closed")
    def closed(self, option):
        if not option["actions"]:
            return "holds no action"
        for place, action in enumerate(option["actions"]):
            reason = self.unfit(action)
            if reason is not None:
                return "names in its action %d %s" % (place, reason)
        return None

    @timed("generators.Seen.unfit")
    def unfit(self, action):
        if action["kind"] not in self.target["actions"]:
            return "the kind %s, which its kind of task does not state" % action["kind"]
        declared = self.target["actions"][action["kind"]]
        if action["actor"] not in self.decision["actors"]:
            return "the actor %s, not an actor of its decision" % action["actor"]
        if set(action["targets"]) != set(declared["roles"]) or set(action["parameters"]) != set(declared["parameters"]):
            return "other roles or parameters than its kind %s states" % action["kind"]
        dictionaries = self.world.registry.encoding.dictionaries
        for column, value in action["parameters"].items():
            if value is not None and column in dictionaries and value not in dictionaries[column]:
                return "the value %r of %s, which the bank does not hold" % (value, column)
        for role, aim in action["targets"].items():
            reason = self.aimless(declared["roles"][role], aim)
            if reason is not None:
                return "under the role %s %s" % (role, reason)
        return None

    def node(self, codes, key):
        started = time.perf_counter()
        lookup = self.world.lookup
        found = [lookup[(code, key)] for code in codes if type(key) is str and (code, key) in lookup]
        result = found[0] if len(found) == 1 else None
        LOG.debug("seen node exit found=%s seconds=%.6f", result is not None, time.perf_counter() - started)
        return result

    @timed("generators.Seen.aimless")
    def aimless(self, role, aim):
        registry = self.world.registry
        codes = [registry.kind(kind) for kind in role["kinds"]]
        if role["sort"] == "new":
            if (role["kinds"][0], aim) in self.created:
                return None
            return "the element %r, which no option recorded for its decision creates" % (aim,)
        if role["sort"] == "node":
            source = self.node(codes, aim)
            if source is not None and source in self.frame.observed(self.step):
                return None
            return "the node %r, which is not one node of its kinds observed at the step of its decision" % (aim,)
        if not (isinstance(aim, list) and len(aim) == 3 and aim[1] in registry.tables):
            return "%r, which is not the first node, a relation stated and the second node of an edge" % (aim,)
        source, code = self.node(codes, aim[0]), registry.code(aim[1])
        edge = registry.by_code[code]
        pairs = {(registry.kind(first), registry.kind(second)) for first, second in edge.pairs}
        if edge.symmetric:
            pairs |= {(second, first) for first, second in pairs}
        ends = [] if source is None else sorted({second for first, second in pairs if first == int(self.world.truth.kinds[source])})
        other = self.node(ends, aim[2])
        if other is not None and (code, source, other) in self.frame.holding(self.step):
            return None
        return "the edge %r, which does not hold at the step of its decision" % (aim,)

    @timed("generators.Seen.fresh")
    def fresh(self, option):
        return self.statement["repeats"] or self.world.content(option) not in self.present

    def taken(self, drawn):
        started = time.perf_counter()
        name, entry = self.statement["name"], self.entry
        if not isinstance(drawn, list):
            raise ValueError("the generator of options %s gives for %s a list of options, found %r" % (name, self.where, drawn))
        result = []
        for number, option in enumerate(drawn):
            where = "the option %d the generator of options %s drew for %s" % (number, name, self.where)
            self.formed(option, DRAWN, where)
            reason = self.closed(option)
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
            if not self.fresh(option):
                raise ValueError("%s equals an option already present, which its statement does not allow" % where)
            self.present.append(self.world.content(option))
            result.append(dict(self.world.canonical(option), drawn=None if probability is None else float(probability)))
        LOG.debug("seen taken exit generator=%s drawn=%d seconds=%.4f", name, len(result), time.perf_counter() - started)
        return result


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
            for role, aim in action["targets"].items():
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
            if seen.closed(option) is None and seen.fresh(option):
                pool.setdefault(seen.world.content(option), seen.world.canonical(option))
        contents = sorted(pool)
        tokens = np.asarray([token(content) for content in contents], np.uint64)
        order = np.lexsort((tokens, seen.unit(tokens)))[:count].tolist()
        result = [dict(pool[contents[at]], drawn=min(count, len(contents)) / len(contents)) for at in order]
        LOG.debug("pooled drawn exit pool=%d drawn=%d seconds=%.4f", len(contents), len(result), time.perf_counter() - started)
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
    version = 1
    reads = ("decisions",)

    @timed("generators.Others.checked")
    def checked(self, parameters, schema, target, references, where):
        return counted(parameters, ("count",), where)

    @timed("generators.Others.__call__")
    def __call__(self, seen, parameters):
        kind = seen.decision["kind"]
        found = [option for other in seen.decisions if other["kind"] == kind for option in other["options"]]
        return self.drawn(seen, found, parameters["count"])


GENERATORS.register("others", Others())
