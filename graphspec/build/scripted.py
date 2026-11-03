import logging
import time

from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.decisions import GENERATORS

from .expression_types import FUNCTIONS, Function
from .generators import Scripted
from .mapping_windows import EVERY
from .splits import SPLITS, Divided

LOG = logging.getLogger(__name__)
SCRIPTED = Names("function of a script", False)
ASKED = dict(function=("reads", "gives", "aggregate"), every=(), split=(), generator=("reads", "probable"))
JOINED = dict(function=FUNCTIONS, every=EVERY, split=SPLITS, generator=GENERATORS)


class Ruled:
    @timed("scripted.Ruled")
    def __init__(self, function, version):
        self.function, self.version = function, version

    @timed("scripted.Ruled.call")
    def __call__(self, facts, parameters):
        return self.function(facts, parameters)


@timed("scripted.entry")
def entry(registry, function, version, asked):
    if registry == "function":
        return Function(function, asked["reads"], asked["gives"], asked["aggregate"], version)
    if registry == "every":
        return Ruled(function, version)
    if registry == "split":
        return Divided(function, version)
    return Scripted(function, asked["reads"], asked["probable"], version)


@timed("scripted.scripted")
def scripted(registry, reason, version, **asked):
    where = "scripted(%r)" % (registry,)
    if registry not in ASKED:
        raise ValueError("%s: a function of a script joins one of the registries %s; the registry %r takes none yet"
                         % (where, ", ".join(ASKED), registry))
    if type(reason) is not str or not reason.strip():
        raise ValueError("%s states reason, why no declaration serves, in words" % where)
    if type(version) is not int or version < 1:
        raise ValueError("%s states version, a whole number from 1" % where)
    unknown, lacking = sorted(set(asked) - set(ASKED[registry])), sorted(set(ASKED[registry]) - set(asked))
    if unknown or lacking:
        raise ValueError("%s holds fields it does not know or lacks: %s" % (where, ", ".join(unknown + lacking)))

    @timed("scripted.register")
    def register(function):
        started = time.perf_counter()
        if not callable(function) or not getattr(function, "__name__", "").isidentifier():
            raise ValueError("%s registers a top-level function under its own name" % where)
        name, target = function.__name__, JOINED[registry]
        held = SCRIPTED.named(name) if name in SCRIPTED.names() else None
        if held is not None:
            if (held["registry"], held["version"]) != (registry, version):
                raise ValueError("%s: the function %s is registered by the script as %s version %d already"
                                 % (where, name, held["registry"], held["version"]))
            del target.held[name]
            del SCRIPTED.held[name]
        target.register(name, entry(registry, function, version, asked))
        SCRIPTED.register(name, dict(registry=registry, reason=reason, version=version))
        LOG.info("scripted exit registry=%s name=%s version=%d seconds=%.6f", registry, name, version,
                 time.perf_counter() - started)
        return function

    return register
