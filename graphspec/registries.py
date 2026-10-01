import itertools
import logging
import pickle
import sys
import time
from collections.abc import MutableMapping
from importlib import import_module

LOG = logging.getLogger(__name__)
REGISTRIES = []
CHANGES = itertools.count(1)
GENERATION = 0


def changed():
    global GENERATION
    started = time.perf_counter()
    GENERATION = next(CHANGES)
    LOG.debug("changed exit generation=%d seconds=%.6f", GENERATION, time.perf_counter() - started)


class Held(MutableMapping):
    def __init__(self, entries):
        started = time.perf_counter()
        self.entries = dict(entries)
        LOG.debug("held exit entries=%d seconds=%.6f", len(self.entries), time.perf_counter() - started)

    def __getitem__(self, name):
        return self.entries[name]

    def __setitem__(self, name, entry):
        started = time.perf_counter()
        self.entries[name] = entry
        changed()
        LOG.debug("held set exit name=%s seconds=%.6f", name, time.perf_counter() - started)

    def __delitem__(self, name):
        started = time.perf_counter()
        del self.entries[name]
        changed()
        LOG.debug("held removed exit name=%s seconds=%.6f", name, time.perf_counter() - started)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


class Names:
    def __init__(self, label):
        if any(registry.label == label for registry in REGISTRIES):
            raise ValueError("a registry is labelled %s twice" % label)
        self.label, self.held, self.shipped = label, {}, frozenset()
        self.module = sys._getframe(1).f_globals["__name__"]
        REGISTRIES.append(self)

    def __setattr__(self, name, value):
        if name == "held":
            value = value if type(value) is Held else Held(value)
            changed()
        object.__setattr__(self, name, value)

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


def ship(registries):
    started = time.perf_counter()
    for registry in registries:
        registry.shipped = frozenset(registry.held)
    LOG.debug("ship exit registries=%d seconds=%.4f", len(registries), time.perf_counter() - started)


def carried():
    started = time.perf_counter()
    found = []
    for registry in REGISTRIES:
        for name, entry in registry.held.items():
            if name in registry.shipped:
                continue
            try:
                found.append((registry.label, name, pickle.dumps(entry), registry.module))
            except Exception as error:
                raise ValueError(
                    "the %s %s is registered from outside the library and cannot be passed to the processes "
                    "the library starts: %s" % (registry.label, name, error)
                ) from error
    LOG.debug("carried exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return tuple(found)


def restored(entries):
    started = time.perf_counter()
    for module in sorted({entry[3] for entry in entries}):
        import_module(module)
    registries = {registry.label: registry for registry in REGISTRIES}
    for label, name, entry, _ in entries:
        registry = registries[label]
        if name in registry.held:
            continue
        loaded = pickle.loads(entry)
        if name not in registry.held:
            registry.register(name, loaded)
    LOG.debug("restored exit entries=%d seconds=%.4f", len(entries), time.perf_counter() - started)
