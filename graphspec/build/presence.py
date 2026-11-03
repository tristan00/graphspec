from collections.abc import Mapping as Held

from graphspec.infra.timing import timed
from graphspec.registries import Names

PRESENCE = Names("rule of presence", True)
BETWEEN = Names("rule between sightings", True)


class Stop:
    version = 2
    creates = False
    keeps = False

    @timed("presence.Stop")
    def __call__(self, keys, times, parameters):
        raise ValueError("the rule stop meets the key %r, which names no node at the time %r" % (keys[0], times[0]))


class Skip:
    version = 2
    creates = False
    keeps = False

    @timed("presence.Skip")
    def __call__(self, keys, times, parameters):
        return [None] * len(keys)


class Create:
    version = 2
    creates = True
    keeps = False

    @timed("presence.Create")
    def __call__(self, keys, times, parameters):
        return list(keys)


class Keep:
    version = 1
    creates = False
    keeps = True

    @timed("presence.Keep")
    def __call__(self, keys, times, parameters):
        return list(keys)


PRESENCE.register("stop", Stop())
PRESENCE.register("skip", Skip())
PRESENCE.register("create", Create())
PRESENCE.register("keep", Keep())


class Absent:
    version = 1

    @timed("presence.Absent")
    def __call__(self, moments, graphs, parameters, label):
        if parameters:
            raise ValueError("the rule absent takes no fields, found %s" % ", ".join(sorted(parameters)))
        return [(place, moment, moment + 1, None) for place, moment in enumerate(moments)], None


class Carried:
    version = 1

    @timed("presence.Carried")
    def __call__(self, moments, graphs, parameters, label):
        unknown = sorted(set(parameters) - {"age", "names"})
        if unknown:
            raise ValueError("the rule carried holds fields it does not know: %s" % ", ".join(unknown))
        if type(parameters.get("age")) is not str or not parameters["age"]:
            raise ValueError("the rule carried states age, the encoder of the age of a row since it was last seen")
        names = parameters.get("names", {})
        if not isinstance(names, Held) or set(names) - {"age"} or any(type(name) is not str or not name
                                                                       for name in names.values()):
            raise ValueError("the rule carried states names as a mapping from age to the name of its feature")
        sightings = {}
        for moment, graph in zip(moments, graphs):
            sightings.setdefault(graph, set()).add(moment)
        following = {}
        for graph, held in sightings.items():
            ordered = sorted(held)
            following[graph] = dict(zip(ordered, ordered[1:]))
        pieces = []
        for place, (moment, graph) in enumerate(zip(moments, graphs)):
            ends = following[graph].get(moment, moment + 1)
            pieces += [(place, start, start + 1, start - moment) for start in range(moment, ends)]
        return pieces, (names.get("age", label + ".age"), parameters["age"])


BETWEEN.register("absent", Absent())
BETWEEN.register("carried", Carried())
