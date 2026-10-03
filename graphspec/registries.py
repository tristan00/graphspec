import itertools
import logging
import pickle
import sys
import time
from collections.abc import MutableMapping
from importlib import import_module

from graphspec.infra.timing import timed

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

    @timed("registries.Held.__len__")
    def __len__(self):
        return len(self.entries)


@timed("registries.version")
def version(entry):
    own = vars(entry) if hasattr(entry, "__dict__") and "version" in vars(entry) else vars(type(entry))
    found = own["version"] if "version" in own else None
    return found if type(found) is int and found >= 1 else None


class Names:
    @timed("registries.Names.__init__")
    def __init__(self, label, versioned):
        if any(registry.label == label for registry in REGISTRIES):
            raise ValueError("a registry is labelled %s twice" % label)
        if type(versioned) is not bool:
            raise ValueError("the registry %s states with true or false whether its entries state versions" % label)
        self.label, self.versioned, self.held, self.shipped = label, versioned, {}, frozenset()
        self.module = sys._getframe(1).f_globals["__name__"]
        REGISTRIES.append(self)

    @timed("registries.Names.__setattr__")
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
        if self.versioned and version(entry) is None:
            raise ValueError(
                "the %s %s states no version: an entry of a registry that derivation, views or packing read "
                "states its version, a whole number from 1 raised whenever its code changes, as the attribute "
                "version of itself or of its own class" % (self.label, name)
            )
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


def versions(read):
    started = time.perf_counter()
    found = sorted({(registry.label, name, version(registry.named(name))) for registry, names in read for name in names})
    LOG.debug("versions exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return [list(entry) for entry in found]


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
