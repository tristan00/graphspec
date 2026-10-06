import logging
import time

from graphspec.infra.timing import timed
from graphspec.operations.pipeline import related
from graphspec.registries import Names, ship
from graphspec.schema.added import Added
from graphspec.schema.targets import TARGETS, Sequence

LOG = logging.getLogger(__name__)
HEADS = Names("head of a kind of target", False)
REALISATIONS = Names("realisation of options", True)
FIELD = "field/"
REALISATION = "options/realisation"
DEPTH = "options/depth"
DEPTHS = (0, 1, 2, 3)
NONE = "none"
ONE = "the realisation reads options of one action whose one target is a node"
FIELDS = ("kind", "targets", "parameters")
ELEMENTS = ("action", "target")
STATED = ("reads", "together", "back", "structure", "actor", "refusal")
ENDS = ("source", "target")
EDITED = ("relation", "actor")


class Attached:
    def __init__(self, reads, module):
        started = time.perf_counter()
        if not isinstance(reads, tuple) or any(type(name) is not str for name in reads) or not callable(module):
            raise ValueError(
                "a head states the parameters of the model it reads as a tuple of names, and the function that "
                "builds it"
            )
        self.reads, self.module = reads, module
        LOG.debug("attached exit reads=%d seconds=%.4f", len(reads), time.perf_counter() - started)


@timed("heads.number")
def number(hidden, spec, registry):
    from .layers import NumberHead

    return NumberHead(hidden)


@timed("heads.path")
def path(hidden, spec, registry):
    from .layers import PathHead

    return PathHead(hidden, spec["time/encoding"])


@timed("heads.scores")
def scores(hidden, spec, registry):
    from .layers import ScoreHead

    return ScoreHead(hidden)


@timed("heads.sequence")
def sequence(hidden, spec, registry):
    from .layers import SequenceHead

    longest = max(
        entry["target"]["length"]
        for entry in registry.schema.tasks["kinds"].values()
        if entry["target"] is not None and isinstance(TARGETS.named(entry["target"]["kind"]), Sequence)
    )
    return SequenceHead(hidden, longest)


@timed("heads.nodes")
def nodes(hidden, spec, registry):
    from .layers import NodeHead

    return NodeHead(hidden)


@timed("heads.decision")
def decision(hidden, spec, registry):
    from .layers import NodeHead, NumberHead
    from .model_spec import realised

    name = realised(spec, registry.schema)
    return NumberHead(hidden) if name is None or realisation(name).back else NodeHead(hidden)


def realisation(name):
    started = time.perf_counter()
    entry = REALISATIONS.named(name)
    if (
        not all(hasattr(entry, part) for part in STATED)
        or not isinstance(entry.reads, tuple)
        or not entry.reads
        or any(field not in FIELDS for field in entry.reads)
        or len(set(entry.reads)) != len(entry.reads)
        or any(type(value) is not bool for value in (entry.together, entry.back, entry.structure, entry.actor))
        or (entry.actor and "targets" not in entry.reads)
        or not callable(entry.refusal)
    ):
        raise ValueError(
            "the realisation of options %s states reads, a tuple of the fields of an option it reads among %s; "
            "together, whether the score of an option reads the other options of its decision, true or false; "
            "back, whether messages pass from options back to the state, true or false; structure, whether it puts "
            "options into the graph of the family as structure, true or false; actor, whether it reads the actor "
            "of each action beside its targets, true or false, and true only where it reads targets; refusal, which "
            "gives why it cannot read the options a schema states, or none; and, where it states no back, module, "
            "which builds it" % (name, ", ".join(FIELDS))
        )
    if entry.back and (
        entry.structure
        or entry.together
        or not entry.actor
        or type(getattr(entry, "mapping", None)) is not dict
        or type(getattr(entry, "relations", None)) is not dict
    ):
        raise ValueError(
            "the realisation of options %s states that messages pass from options back to the state, which they do "
            "through an edit of the graph each option makes, encoded afterwards; it states mapping, from every kind "
            "of task that decides and each of its kinds of action to the relation each role becomes and the end of "
            "it the actor is, and relations, those it adds; it reads the actor, reads no option together and puts "
            "no structure beside the graph" % name
        )
    if not entry.back and not callable(getattr(entry, "module", None)):
        raise ValueError(
            "the realisation of options %s states that messages pass from the state to options only, and states no "
            "module, which builds it" % name
        )
    LOG.debug("options realisation exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return entry


def coded(schema):
    started = time.perf_counter()
    actions, roles = [], []
    for task in schema.decided():
        for action, entry in schema.tasks["kinds"][task]["target"]["actions"].items():
            actions.append((task, action))
            roles += [(task, action, role, held["sort"], tuple(held["kinds"])) for role, held in entry["roles"].items()]
    LOG.debug("options coded exit actions=%d roles=%d seconds=%.4f", len(actions), len(roles), time.perf_counter() - started)
    return actions, roles


def fielded(schema, task):
    started = time.perf_counter()
    actions, roles = coded(schema)
    several = len(schema.tasks["kinds"][task]["target"]["actions"]) > 1
    result = [dict(name="kind", sort="kind", code=None)] if several else []
    for code, (owner, action, role, sort, kinds) in enumerate(roles):
        if owner == task:
            named, relations = related(schema, sort, kinds)
            result.append(
                dict(
                    name="target %s of %s" % (role, action),
                    sort="targets",
                    code=code,
                    role=sort,
                    kinds=kinds,
                    action=action,
                    part=role,
                    named=named,
                    relations=relations,
                )
            )
    for code, (owner, action) in enumerate(actions):
        if owner == task:
            for column in schema.tasks["kinds"][task]["target"]["actions"][action]["parameters"]:
                result.append(dict(name="parameter %s of %s" % (column, action), sort="parameters", code=code, column=column))
    LOG.debug("options fielded exit task=%s fields=%d seconds=%.4f", task, len(result), time.perf_counter() - started)
    return result


def refused(name, registry):
    started = time.perf_counter()
    entry, schema = realisation(name), registry.schema
    found, result = entry.refusal(registry), None
    unread = [
        (task, field)
        for task in schema.decided()
        for field in fielded(schema, task)
        if field["sort"] not in entry.reads
    ]
    if found is not None:
        result = "the realisation of options %s cannot read the options of this bank: %s" % (name, found)
    elif unread:
        task, field = unread[0]
        result = (
            "the field %s of the kind of task %s has no route to the prediction under the realisation of options %s: "
            "the realisation reads no %s" % (field["name"], task, name, field["sort"])
        )
    LOG.debug("options refused exit name=%s refused=%s seconds=%.4f", name, result is not None, time.perf_counter() - started)
    return result


class Existing:
    version = 9
    reads = ("targets",)
    together = False
    back = False
    structure = False
    actor = False

    @timed("heads.Existing.refusal")
    def refusal(self, registry):
        schema = registry.schema
        for task in schema.decided():
            target = schema.tasks["kinds"][task]["target"]
            if target["nothing"] is not None:
                return "the kind of task %s states the option %s, which does nothing and holds no action, and %s" % (
                    task,
                    target["nothing"],
                    ONE,
                )
            for action, entry in target["actions"].items():
                if [held["sort"] for held in entry["roles"].values()] != ["node"]:
                    return "the kind of action %s of the kind of task %s states the roles %s, and %s" % (
                        action,
                        task,
                        ", ".join("%s of the sort %s" % (role, held["sort"]) for role, held in entry["roles"].items()),
                        ONE,
                    )
                ((role, held),) = entry["roles"].items()
                if held["count"] != [1, 1]:
                    return "the kind of action %s of the kind of task %s states the role %s holding from %d to %s targets, and %s" % (
                        action,
                        task,
                        role,
                        held["count"][0],
                        "no bound" if held["count"][1] is None else held["count"][1],
                        ONE,
                    )
        if registry.extras["acted"] > 1:
            return "the bank holds an option of %d actions, and %s" % (registry.extras["acted"], ONE)
        return None

    @timed("heads.Existing.module")
    def module(self, hidden, width, spec, registry):
        from .options import Node

        return Node()


class Built:
    version = 8
    reads = ("kind", "targets", "parameters")
    together = False
    back = False
    structure = False
    actor = False

    @timed("heads.Built.refusal")
    def refusal(self, registry):
        return None

    @timed("heads.Built.module")
    def module(self, hidden, width, spec, registry):
        from .options import Vector

        return Vector(hidden, width)


class Structural:
    version = 7
    reads = ("kind", "targets", "parameters")
    together = False
    back = False
    structure = True

    @timed("heads.Structural.__init__")
    def __init__(self, element, actor):
        if element not in ELEMENTS or type(actor) is not bool or (element == "target" and not actor):
            raise ValueError(
                "a structural realisation puts one element into the graph per %s, found %r, and states whether "
                "the actor of each action joins it, found %r, which an element per target, an edge between the "
                "actor and the target, needs" % (" or per ".join(ELEMENTS), element, actor)
            )
        self.element, self.actor = element, actor

    @timed("heads.Structural.refusal")
    def refusal(self, registry):
        if self.element != "target":
            return None
        for task in registry.schema.decided():
            for action, entry in registry.schema.tasks["kinds"][task]["target"]["actions"].items():
                if not entry["roles"]:
                    return (
                        "the kind of action %s of the kind of task %s states no role, and the realisation joins the "
                        "actor of an action to each of its targets by an edge" % (action, task)
                    )
                if not any(held["count"][0] for held in entry["roles"].values()):
                    return (
                        "the kind of action %s of the kind of task %s states no role holding a target in every action, "
                        "and the realisation joins the actor of an action to each of its targets by an edge, so an "
                        "action holding none would put nothing of its kind and parameters before the model" % (action, task)
                    )
        return None

    @timed("heads.Structural.module")
    def module(self, hidden, width, spec, registry):
        from .options import Structured

        return Structured(hidden, width, spec, registry, self.element, self.actor)


def mapped(mapping):
    started = time.perf_counter()
    result = type(mapping) is dict and all(
        type(task) is str
        and type(actions) is dict
        and all(
            type(action) is str
            and type(roles) is dict
            and all(type(role) is str and type(held) is dict and set(held) == set(EDITED) for role, held in roles.items())
            for action, roles in actions.items()
        )
        for task, actions in mapping.items()
    )
    result = result and all(
        type(held["relation"]) is str and held["actor"] in ENDS
        for actions in mapping.values()
        for roles in actions.values()
        for held in roles.values()
    )
    LOG.debug("heads mapped exit stated=%s seconds=%.4f", result, time.perf_counter() - started)
    return result


def joins(pairs, directed, first, second):
    started = time.perf_counter()
    result = (first, second) in pairs or (not directed and (second, first) in pairs)
    LOG.debug("heads joins exit joins=%s seconds=%.6f", result, time.perf_counter() - started)
    return result


def reached(schema, kinds):
    started = time.perf_counter()
    found = set(kinds)
    for entry in schema.edges.values():
        for first, second in entry["endpoints"]:
            if first in kinds:
                found.add(second)
            if not entry["directed"] and second in kinds:
                found.add(first)
    result = tuple(sorted(found))
    LOG.debug("heads reached exit kinds=%d seconds=%.6f", len(result), time.perf_counter() - started)
    return result


class Edited:
    reads = ("kind", "targets", "parameters")
    together = False
    back = True
    structure = False
    actor = True

    def __init__(self, mapping, relations, version):
        started = time.perf_counter()
        if type(version) is not int or version < 1:
            raise ValueError("an edit of the graph states its version, a whole number from 1, found %r" % (version,))
        if not mapped(mapping):
            raise ValueError(
                "an edit of the graph states its mapping from every kind of task that decides, by name, to a mapping "
                "from each of its kinds of action to a mapping from each of its roles to relation, the name of a "
                "relation the schema declares or the edit adds, and actor, source or target, the end of each fact "
                "the actor is; found %r" % (mapping,)
            )
        self.relations = Added({}, {}, {}, relations, {}, {}).relations
        self.mapping = {
            task: {action: {role: dict(held) for role, held in roles.items()} for action, roles in actions.items()}
            for task, actions in mapping.items()
        }
        self.version = version
        LOG.debug("edited exit kinds=%d relations=%d seconds=%.4f", len(mapping), len(relations), time.perf_counter() - started)

    def relation(self, schema, name):
        started = time.perf_counter()
        result = None
        if name in self.relations:
            held = self.relations[name]
            result = held["pairs"], held["directed"], held["columns"]
        elif name in schema.edges and schema.edges[name]["column"] is None:
            held = schema.edges[name]
            result = held["endpoints"], held["directed"], held["columns"]
        LOG.debug("edited relation exit name=%s seconds=%.6f", name, time.perf_counter() - started)
        return result

    def refusal(self, registry):
        started = time.perf_counter()
        result = self.unread(registry.schema) or self.unmapped(registry.schema)
        LOG.debug("edited refusal exit refused=%s seconds=%.4f", result is not None, time.perf_counter() - started)
        return result

    def unread(self, schema):
        started = time.perf_counter()
        taken = set(schema.entities()) | set(schema.edges) | set(schema.columns)
        result = None
        for name, held in self.relations.items():
            kinds = {kind for pair in held["pairs"] for kind in pair}
            if name in taken or ":" in name:
                result = "the edit adds the relation %s, a name without a colon the schema does not hold" % name
            elif not kinds <= set(schema.entities()) or not set(held["columns"]) <= set(schema.columns):
                result = "the edit adds the relation %s between kinds of node or with columns the schema does not state" % name
            if result is not None:
                break
        decided = schema.decided()
        strangers = sorted(set(self.mapping) - set(decided))
        if result is None and strangers:
            result = "the mapping of the edit states the kind of task %s, which decides nothing in the schema" % strangers[0]
        LOG.debug("edited unread exit refused=%s seconds=%.4f", result is not None, time.perf_counter() - started)
        return result

    def unmapped(self, schema):
        started = time.perf_counter()
        result = None
        for task in schema.decided():
            target = schema.tasks["kinds"][task]["target"]
            if task not in self.mapping:
                result = "the mapping of the edit states nothing for the kind of task %s, which decides" % task
                break
            result = self.acted(schema, task, target)
            if result is not None:
                break
        LOG.debug("edited unmapped exit refused=%s seconds=%.4f", result is not None, time.perf_counter() - started)
        return result

    def acted(self, schema, task, target):
        started = time.perf_counter()
        stated, result, made = self.mapping[task], None, {}
        actors = tuple(schema.tasks["references"][target["actors"]]["kinds"])
        strangers = sorted(set(stated) - set(target["actions"]))
        if strangers:
            result = "the mapping of the edit states the kind of action %s of the kind of task %s, which it does not state" % (
                strangers[0],
                task,
            )
        for action, entry in target["actions"].items() if result is None else ():
            result = self.roled(schema, task, action, entry, actors, made)
            if result is not None:
                break
        if result is None:
            result = self.kinded(task, made)
        LOG.debug("edited acted exit task=%s refused=%s seconds=%.4f", task, result is not None, time.perf_counter() - started)
        return result

    def roled(self, schema, task, action, entry, actors, made):
        started = time.perf_counter()
        where = "the kind of action %s of the kind of task %s" % (action, task)
        result = None
        if action not in self.mapping[task]:
            result = "the mapping of the edit states nothing for %s, so an option holding it cannot be made an edit" % where
        elif not entry["roles"]:
            result = "%s states no role, and an edit makes the facts of an action from its targets" % where
        elif not any(held["count"][0] for held in entry["roles"].values()):
            result = (
                "%s states no role holding a target in every action, and an edit makes the facts of an action from its "
                "targets, so an action holding none would make no fact" % where
            )
        elif sorted(self.mapping[task][action]) != sorted(entry["roles"]):
            result = "the mapping of the edit states for %s the roles %s, and it states the roles %s" % (
                where,
                ", ".join(sorted(self.mapping[task][action])) or "none",
                ", ".join(sorted(entry["roles"])),
            )
        carried, relations = set(), []
        for role, held in entry["roles"].items() if result is None else ():
            stated = self.mapping[task][action][role]
            found = self.relation(schema, stated["relation"])
            if found is None:
                result = "the role %s of %s becomes %s, neither a relation the edit adds nor one the schema declares" % (
                    role,
                    where,
                    stated["relation"],
                )
                break
            pairs, directed, columns = found
            ends = reached(schema, held["kinds"]) if held["sort"] == "edge" else tuple(held["kinds"])
            wrong = [
                pair
                for pair in (
                    (actor, kind) if stated["actor"] == "source" else (kind, actor) for actor in actors for kind in ends
                )
                if not joins(pairs, directed, *pair)
            ]
            if wrong:
                result = "the role %s of %s becomes facts of %s from %s to %s, which %s does not join" % (
                    role,
                    where,
                    stated["relation"],
                    wrong[0][0],
                    wrong[0][1],
                    stated["relation"],
                )
                break
            carried |= set(columns)
            relations.append(stated["relation"])
        lacking = [column for column in entry["parameters"] if column not in carried]
        if result is None and lacking:
            result = "the parameter %s of %s becomes the value of no fact: no relation its roles become holds the column %s" % (
                lacking[0],
                where,
                lacking[0],
            )
        if result is None:
            made[action] = (tuple(sorted(relations)), tuple(sorted(entry["parameters"])))
        LOG.debug("edited roled exit action=%s refused=%s seconds=%.4f", action, result is not None, time.perf_counter() - started)
        return result

    def kinded(self, task, made):
        started = time.perf_counter()
        result, seen = None, {}
        for action, facts in made.items():
            if len(made) > 1 and facts in seen:
                result = (
                    "the kinds of action %s and %s of the kind of task %s become facts of the same relations, %s, "
                    "carrying the same parameters, so the field kind has no route to the prediction"
                    % (seen[facts], action, task, ", ".join(facts[0]))
                )
                break
            seen[facts] = action
        LOG.debug("edited kinded exit task=%s refused=%s seconds=%.4f", task, result is not None, time.perf_counter() - started)
        return result


def reading(registry, name):
    started = time.perf_counter()
    result = any(name in HEADS.named(kind).reads for kind in registry.schema.targets())
    LOG.debug("reading exit parameter=%s seconds=%.4f", name, time.perf_counter() - started)
    return result


HEADS.register("number", Attached((), number))
HEADS.register("path", Attached(("time/encoding",), path))
HEADS.register("class", Attached((), scores))
HEADS.register("classes", Attached((), scores))
HEADS.register("proportions", Attached((), scores))
HEADS.register("sequence", Attached((), sequence))
HEADS.register("survival", Attached(("time/encoding",), path))
HEADS.register("ranking", Attached((), nodes))
HEADS.register("nodes", Attached((), nodes))
HEADS.register("link", Attached((), nodes))
HEADS.register("decision", Attached((REALISATION,), decision))
REALISATIONS.register("node", Existing())
REALISATIONS.register("vector", Built())
REALISATIONS.register("action_nodes", Structural("action", False))
REALISATIONS.register("actor_edge", Structural("target", True))
REALISATIONS.register("set_edge", Structural("action", True))
ship([HEADS, REALISATIONS])
