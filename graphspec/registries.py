import logging
import pickle
import time

LOG = logging.getLogger(__name__)
REGISTRIES = []


class Names:
    def __init__(self, label):
        if any(registry.label == label for registry in REGISTRIES):
            raise ValueError("a registry is labelled %s twice" % label)
        self.label, self.held, self.shipped = label, {}, frozenset()
        REGISTRIES.append(self)

    def register(self, name, entry):
        started = time.perf_counter()
        if type(name) is not str or not name:
            raise ValueError("a %s is registered under a name, found %r" % (self.label, name))
        if name in self.held:
            raise ValueError("the %s %s is registered twice" % (self.label, name))
        self.held[name] = entry
        LOG.debug(
            "register exit label=%s name=%s seconds=%.4f", self.label, name, time.perf_counter() - started
        )
        return entry

    def named(self, name):
        if type(name) is not str or name not in self.held:
            raise KeyError("no %s is registered under the name %r" % (self.label, name))
        return self.held[name]

    def names(self):
        return tuple(self.held)


def ship():
    started = time.perf_counter()
    for registry in REGISTRIES:
        registry.shipped = frozenset(registry.held)
    LOG.debug("ship exit registries=%d seconds=%.4f", len(REGISTRIES), time.perf_counter() - started)


def carried():
    started = time.perf_counter()
    found = []
    for registry in REGISTRIES:
        for name, entry in registry.held.items():
            if name in registry.shipped:
                continue
            try:
                found.append((registry.label, name, pickle.dumps(entry)))
            except Exception as error:
                raise ValueError(
                    "the %s %s is registered from outside the library and cannot be passed to the processes "
                    "the library starts: %s" % (registry.label, name, error)
                ) from error
    LOG.debug("carried exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return tuple(found)


def restored(entries):
    started = time.perf_counter()
    registries = {registry.label: registry for registry in REGISTRIES}
    for label, name, entry in entries:
        registry = registries[label]
        if name in registry.held:
            continue
        loaded = pickle.loads(entry)
        if name not in registry.held:
            registry.register(name, loaded)
    LOG.debug("restored exit entries=%d seconds=%.4f", len(entries), time.perf_counter() - started)
