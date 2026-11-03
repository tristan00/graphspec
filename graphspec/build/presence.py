from graphspec.infra.timing import timed
from graphspec.registries import Names

PRESENCE = Names("rule of presence", True)


class Stop:
    version = 1
    creates = False

    @timed("presence.Stop")
    def __call__(self, keys, times, parameters):
        raise ValueError("the rule stop meets the key %r, which names no node" % keys[0])


class Skip:
    version = 1
    creates = False

    @timed("presence.Skip")
    def __call__(self, keys, times, parameters):
        return [None] * len(keys)


class Create:
    version = 1
    creates = True

    @timed("presence.Create")
    def __call__(self, keys, times, parameters):
        return list(keys)


PRESENCE.register("stop", Stop())
PRESENCE.register("skip", Skip())
PRESENCE.register("create", Create())


@timed("presence.ruled")
def ruled(stated, where):
    if type(stated) is str:
        return stated, {}
    if isinstance(stated, dict) or hasattr(stated, "keys"):
        held = dict(stated)
        if type(held.get("name")) is str:
            return held.pop("name"), held
    raise ValueError("%s states its rule of presence by name, or as a mapping of name and parameters, found %r"
                     % (where, stated))
