import itertools
import logging
import pickle
import sys
import time
from collections.abc import MutableMapping
from contextlib import contextmanager
from importlib import import_module

from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
REGISTRIES = []
CHANGES = itertools.count(1)
GENERATION = 0
READ = dict(held=None, by=None)


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
    def __init__(self, label, versioned):
        started = time.perf_counter()
        if any(registry.label == label for registry in REGISTRIES):
            raise ValueError("a registry is labelled %s twice" % label)
        if type(versioned) is not bool:
            raise ValueError("the registry %s states with true or false whether its entries state versions" % label)
        self.label, self.versioned, self.held, self.shipped = label, versioned, {}, {}
        self.module = sys._getframe(1).f_globals["__name__"]
        REGISTRIES.append(self)
        LOG.debug("names exit label=%s module=%s seconds=%.4f", label, self.module, time.perf_counter() - started)

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
        if self.versioned and READ["by"] is not None and READ["held"] is not None and (self.label, name) not in READ["held"]:
            raise ValueError(
                "the operation %s reads the %s %s, which the key of the variant it derives does not hold; an "
                "operation reads an entry it chooses by a parameter stated with "
                "graphspec.operations.parameters.registered, and an entry the statement names from its place in the "
                "statement, never by a name of its own" % (READ["by"], self.label, name)
            )
        return self.held[name]

    def names(self):
        return tuple(self.held)


def versions(read):
    started = time.perf_counter()
    found = sorted({(registry.label, name, version(registry.named(name))) for registry, names in read for name in names})
    LOG.debug("versions exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return [list(entry) for entry in found]


@contextmanager
def keyed(held):
    started = time.perf_counter()
    before, READ["held"] = READ["held"], frozenset(held)
    try:
        yield
    finally:
        READ["held"] = before
        LOG.debug("keyed exit entries=%d seconds=%.4f", len(held), time.perf_counter() - started)


def ship(registries):
    started = time.perf_counter()
    for registry in registries:
        registry.shipped = dict(registry.held)
    LOG.debug("ship exit registries=%d seconds=%.4f", len(registries), time.perf_counter() - started)


def carried():
    started = time.perf_counter()
    found = []
    for registry in REGISTRIES:
        if list(registry.held) == list(registry.shipped) and all(
            registry.held[name] is registry.shipped[name] for name in registry.shipped
        ):
            found.append((registry.label, registry.module, None))
            continue
        held = []
        for name, entry in registry.held.items():
            if name in registry.shipped and registry.shipped[name] is entry:
                held.append((name, None, None))
                continue
            try:
                held.append((name, pickle.dumps(entry), version(entry) if registry.versioned else None))
            except Exception as error:
                raise ValueError(
                    "the %s %s is registered from outside the library and cannot be passed to the processes "
                    "the library starts: %s" % (registry.label, name, error)
                ) from error
        found.append((registry.label, registry.module, tuple(held)))
    LOG.debug("carried exit registries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return tuple(found), READ["held"]


def restored(given):
    started = time.perf_counter()
    entries, READ["held"] = given
    for module in sorted({module for _, module, held in entries if held is not None}):
        import_module(module)
    loaded = [
        (label, [(name, None if pickled is None else pickle.loads(pickled), stated) for name, pickled, stated in held])
        for label, _, held in entries
        if held is not None
    ]
    registries = {registry.label: registry for registry in REGISTRIES}
    for label, _, held in entries:
        if held is None and label in registries:
            registries[label].held = dict(registries[label].shipped)
    for label, held in loaded:
        registry = registries[label]
        for name, entry, stated in held:
            if entry is None and name not in registry.shipped:
                raise ValueError(
                    "the %s %s is shipped with the library in the process that called and not in this worker"
                    % (label, name)
                )
            if entry is not None and registry.versioned and version(entry) != stated:
                raise ValueError(
                    "the %s %s is at version %d in the process that called and at %s in this worker, whose code "
                    "differs; the worker imports the code on disk" % (label, name, stated, version(entry))
                )
        registry.held = {name: registry.shipped[name] if entry is None else entry for name, entry, _ in held}
    LOG.debug("restored exit registries=%d seconds=%.4f", len(entries), time.perf_counter() - started)
