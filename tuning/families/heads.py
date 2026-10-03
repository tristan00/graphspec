import logging
import time

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship
from graphspec.schema.targets import TARGETS, Sequence

LOG = logging.getLogger(__name__)
HEADS = Names("head of a kind of target", False)
REALISATIONS = Names("realisation of options", True)
FIELD = "field/"
REALISATION = "options/realisation"
NONE = "none"
ONE = "the realisation reads options of one action whose one target is a node"
FIELDS = ("kind", "targets", "parameters")
ELEMENTS = ("action", "target")


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

    return NumberHead(hidden) if realised(spec, registry.schema) is None else NodeHead(hidden)


def realisation(name):
    started = time.perf_counter()
    entry = REALISATIONS.named(name)
    if (
        not all(hasattr(entry, part) for part in ("reads", "together", "back", "structure", "actor", "refusal", "module"))
        or not isinstance(entry.reads, tuple)
        or not entry.reads
        or any(field not in FIELDS for field in entry.reads)
        or len(set(entry.reads)) != len(entry.reads)
        or any(type(value) is not bool for value in (entry.together, entry.back, entry.structure, entry.actor))
        or (entry.actor and "targets" not in entry.reads)
        or not callable(entry.refusal)
        or not callable(entry.module)
    ):
        raise ValueError(
            "the realisation of options %s states reads, a tuple of the fields of an option it reads among %s; "
            "together, whether the score of an option reads the other options of its decision, true or false; "
            "back, whether messages pass from options back to the state, true or false; structure, whether it puts "
            "options into the graph of the family as structure, true or false; actor, whether it reads the actor "
            "of each action beside its targets, true or false, and true only where it reads targets; refusal, which "
            "gives why it cannot read the options a schema states, or none; and module, which builds it"
            % (name, ", ".join(FIELDS))
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
            result.append(
                dict(name="target %s of %s" % (role, action), sort="targets", code=code, role=sort, kinds=kinds)
            )
    for code, (owner, action) in enumerate(actions):
        if owner == task:
            for column in schema.tasks["kinds"][task]["target"]["actions"][action]["parameters"]:
                result.append(dict(name="parameter %s of %s" % (column, action), sort="parameters", code=code, column=column))
    LOG.debug("options fielded exit task=%s fields=%d seconds=%.4f", task, len(result), time.perf_counter() - started)
    return result


def refused(name, schema):
    started = time.perf_counter()
    entry = realisation(name)
    found, result = entry.refusal(schema), None
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
    version = 4
    reads = ("targets",)
    together = False
    back = False
    structure = False
    actor = False

    @timed("heads.Existing.refusal")
    def refusal(self, schema):
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
        return None

    @timed("heads.Existing.module")
    def module(self, hidden, width, spec, registry):
        from .options import Node

        return Node(registry)


class Built:
    version = 4
    reads = ("kind", "targets", "parameters")
    together = False
    back = False
    structure = False
    actor = False

    @timed("heads.Built.refusal")
    def refusal(self, schema):
        return None

    @timed("heads.Built.module")
    def module(self, hidden, width, spec, registry):
        from .options import Vector

        return Vector(hidden, width)


class Structural:
    version = 2
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
    def refusal(self, schema):
        if self.element != "target":
            return None
        for task in schema.decided():
            for action, entry in schema.tasks["kinds"][task]["target"]["actions"].items():
                if not entry["roles"]:
                    return (
                        "the kind of action %s of the kind of task %s states no role, and the realisation joins the "
                        "actor of an action to each of its targets by an edge" % (action, task)
                    )
        return None

    @timed("heads.Structural.module")
    def module(self, hidden, width, spec, registry):
        from .options import Structured

        return Structured(hidden, width, spec, registry, self.element, self.actor)


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
