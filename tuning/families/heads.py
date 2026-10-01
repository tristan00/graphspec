import logging
import time

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship

LOG = logging.getLogger(__name__)
HEADS = Names("head of a kind of target")
FIELD = "field/"


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
        if entry["target"] is not None and entry["target"]["kind"] == "sequence"
    )
    return SequenceHead(hidden, longest)


@timed("heads.nodes")
def nodes(hidden, spec, registry):
    from .layers import NodeHead

    return NodeHead(hidden)


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
HEADS.register("ranking", Attached((), nodes))
HEADS.register("nodes", Attached((), nodes))
HEADS.register("link", Attached((), nodes))
ship([HEADS])
