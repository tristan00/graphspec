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


class Existing:
    version = 1
    reads = ("targets",)
    together = False
    back = False

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
    version = 1
    reads = ("kind", "targets", "parameters")
    together = False
    back = False

    @timed("heads.Built.refusal")
    def refusal(self, schema):
        return None

    @timed("heads.Built.module")
    def module(self, hidden, width, spec, registry):
        from .options import Vector

        return Vector(hidden, width)


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
ship([HEADS, REALISATIONS])
