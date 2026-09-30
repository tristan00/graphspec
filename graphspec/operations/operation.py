import logging
import time
from collections import Counter

from graphspec.registries import Names
from graphspec.views.graph import CHANGES as CHECKED

LOG = logging.getLogger(__name__)
CHANGES = CHECKED + ("references",)
SPLITS = ("trained", "scored")
TREATMENTS = ("stored", "refuses")
EDGES = dict(
    sets="the edges joining sets of the kind %s", about="the edges of the table %s, which are about edges"
)
PIPELINE = (
    "graphspec.periods",
    "graphspec.window",
    "graphspec.nodes",
    "graphspec.derive",
    "graphspec.relations",
    "graphspec.edges",
    "graphspec.merging",
)


class Count:
    def __init__(self, name, counts, active, sets, about):
        started = time.perf_counter()
        for part, treatment in (("the edges joining sets", sets), ("the edges about edges", about)):
            if treatment not in TREATMENTS:
                raise ValueError(
                    "the count %s counts %s and states no treatment of %s; it states one of %s"
                    % (name, counts, part, ", ".join(TREATMENTS))
                )
        if not callable(active):
            raise ValueError("the count %s states when it is active as a function of a registry and a spec" % name)
        self.name, self.counts, self.active, self.sets, self.about = name, counts, active, sets, about
        LOG.debug("count exit name=%s seconds=%.4f", name, time.perf_counter() - started)

    def declaration(self):
        return dict(name=self.name, counts=self.counts, sets=self.sets, about=self.about)


class Settings:
    def __init__(self, operation, parameters, spec):
        started = time.perf_counter()
        self.operation, self.spec = operation, spec
        self.parameters = {parameter.name: parameter for parameter in parameters}
        LOG.debug(
            "settings exit operation=%s parameters=%d seconds=%.4f",
            operation,
            len(self.parameters),
            time.perf_counter() - started,
        )

    def stated(self, name):
        if name not in self.parameters:
            raise KeyError(
                "the operation %s reads %s, which is none of the parameters it states or reads"
                % (self.operation, name)
            )
        return self.parameters[name]

    def __contains__(self, name):
        active = self.stated(name).active(self.spec)
        if active != (name in self.spec):
            raise ValueError(
                "the parameter %s, read by the operation %s, is %s under its condition and %s by the spec"
                % (
                    name,
                    self.operation,
                    "active" if active else "inactive",
                    "stated" if name in self.spec else "not stated",
                )
            )
        return active

    def __getitem__(self, name):
        self.stated(name)
        if name not in self.spec:
            raise KeyError("the operation %s reads %s, which the spec does not state" % (self.operation, name))
        return self.spec[name]


class Operation:
    def __init__(self, namespace, name, function, parameters, reads, splits, requires, changes, version):
        started = time.perf_counter()
        for label, part in (("namespace", namespace), ("name", name)):
            if type(part) is not str or not part or "." in part:
                raise ValueError("an operation states its %s as a word without a dot, found %r" % (label, part))
        self.name = namespace + "." + name
        stated = dict(
            function=callable(function),
            parameters=callable(parameters),
            reads=type(reads) is tuple and all(type(read) is str for read in reads),
            splits=type(splits) is tuple and all(split in SPLITS for split in splits),
            requires=type(requires) is tuple and all(isinstance(count, Count) for count in requires),
            changes=type(changes) is tuple and all(change in CHANGES for change in changes),
            version=type(version) is int and version >= 1,
        )
        wrong = [label for label, held in stated.items() if not held]
        if wrong:
            raise ValueError(
                "the operation %s states %s as the constructor of an operation asks: a function of a graph, a "
                "registry, settings, a seed, limits and carried state; a function from a registry to its "
                "parameters; a tuple of the names of the operations whose parameters it reads; a tuple of the "
                "splits it may read, among %s; a tuple of counts; a tuple of what it changes, among %s; a whole "
                "version from 1" % (self.name, ", ".join(wrong), ", ".join(SPLITS), ", ".join(CHANGES))
            )
        self.namespace, self.function, self.parameters, self.reads = namespace, function, parameters, reads
        self.splits, self.requires, self.changes, self.version = splits, requires, changes, version
        LOG.debug("operation exit name=%s seconds=%.4f", self.name, time.perf_counter() - started)

    def stated(self, registry):
        started = time.perf_counter()
        found = tuple(self.parameters(registry))
        counted = Counter(parameter.name for parameter in found)
        twice = sorted(name for name, count in counted.items() if count > 1)
        if twice:
            raise ValueError("the operation %s states the parameters %s twice" % (self.name, ", ".join(twice)))
        LOG.debug(
            "stated exit operation=%s parameters=%d seconds=%.4f",
            self.name,
            len(found),
            time.perf_counter() - started,
        )
        return found

    def settings(self, registry, spec):
        started = time.perf_counter()
        parameters = self.stated(registry)
        for read in self.reads:
            parameters += OPERATIONS.named(read).stated(registry)
        result = Settings(self.name, parameters, spec)
        LOG.debug("settings made exit operation=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def check(self, registry, settings):
        started = time.perf_counter()
        met = dict(sets=tuple(registry.schema.sets), about=tuple(registry.schema.about()))
        if any(met.values()):
            for count in self.requires:
                if not count.active(registry, settings):
                    continue
                for part, found in met.items():
                    if found and getattr(count, part) == "refuses":
                        raise ValueError(
                            "the count %s of the operation %s counts %s and refuses %s"
                            % (count.name, self.name, count.counts, EDGES[part] % found[0])
                        )
        LOG.debug("check exit operation=%s seconds=%.4f", self.name, time.perf_counter() - started)

    def __call__(self, graph, registry, spec, seed, limits, carry):
        started = time.perf_counter()
        settings = self.settings(registry, spec)
        self.check(registry, settings)
        result = self.function(graph, registry, settings, seed, limits, carry)
        LOG.debug(
            "operation call exit name=%s graph=%s seconds=%.4f",
            self.name,
            graph.name,
            time.perf_counter() - started,
        )
        return result

    def declaration(self, registry):
        started = time.perf_counter()
        result = dict(
            name=self.name,
            version=self.version,
            parameters=[parameter.declaration() for parameter in self.stated(registry)],
            reads=list(self.reads),
            splits=list(self.splits),
            requires=[count.declaration() for count in self.requires],
            changes=list(self.changes),
        )
        LOG.debug("declaration exit operation=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result


class Operations(Names):
    def register(self, name, entry):
        started = time.perf_counter()
        if not isinstance(entry, Operation) or entry.name != name:
            raise ValueError(
                "an operation is registered under its namespace and name joined by a dot; "
                "%r is registered as %s" % (getattr(entry, "name", entry), name)
            )
        result = super().register(name, entry)
        LOG.debug("operations register exit name=%s seconds=%.4f", name, time.perf_counter() - started)
        return result


OPERATIONS = Operations("operation")


def listed(registry):
    started = time.perf_counter()
    result = tuple(OPERATIONS.named(name).declaration(registry) for name in OPERATIONS.names())
    LOG.debug("listed exit operations=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def ordered(names):
    started = time.perf_counter()
    found, entered, waiting = [], set(), [(name, False) for name in reversed(names)]
    while waiting:
        name, expanded = waiting.pop()
        operation = OPERATIONS.named(name)
        if operation in found:
            continue
        if expanded:
            found.append(operation)
            continue
        if name in entered:
            raise ValueError("the operation %s reads, through the operations it reads, its own parameters" % name)
        entered.add(name)
        waiting.append((name, True))
        waiting.extend((read, False) for read in reversed(operation.reads))
    LOG.debug("ordered exit operations=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return tuple(found)
