import logging
import time
from collections import Counter
from weakref import WeakKeyDictionary

import numpy as np

from graphspec import registries
from graphspec.registries import Names
from graphspec.schema.added import Added
from graphspec.views.graph import CHANGES as CHECKED

LOG = logging.getLogger(__name__)
CHANGES = CHECKED + ("references",)
SPLITS = ("trained", "scored")
TREATMENTS = ("stored", "refuses")
ROLES = ("names", "removes", "fills")
STATED = WeakKeyDictionary()
EDGES = dict(
    sets="the edges joining sets of the kind %s", about="the edges of the table %s, which are about edges"
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
    def __init__(self, operation, parameters, spec, splits, tasks):
        started = time.perf_counter()
        self.operation, self.spec, self.splits, self.tasks = operation, spec, splits, tasks
        self.parameters = {parameter.name: parameter for parameter in parameters}
        LOG.debug(
            "settings exit operation=%s parameters=%d seconds=%.4f",
            operation,
            len(self.parameters),
            time.perf_counter() - started,
        )

    def split(self, name):
        started = time.perf_counter()
        if name not in self.splits:
            raise KeyError(
                "the operation %s reads the split %s, which it does not state it may read; it states %s"
                % (self.operation, name, ", ".join(self.splits) or "none")
            )
        result = self.tasks(name)
        LOG.debug(
            "split exit operation=%s split=%s tasks=%d seconds=%.4f",
            self.operation,
            name,
            len(result),
            time.perf_counter() - started,
        )
        return result

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


class Phase:
    def __init__(self, name, function, carries, parts):
        started = time.perf_counter()
        if type(name) is not str or not name or "." in name or "/" in name:
            raise ValueError("a phase states its name as a word without a dot or a slash, found %r" % (name,))
        stated = dict(
            function=callable(function),
            carries=carries is None or callable(carries),
            parts=parts is None or callable(parts),
        )
        wrong = [label for label, held in stated.items() if not held]
        if wrong:
            raise ValueError(
                "the phase %s states %s as the constructor of a phase asks: a function of a graph, a registry, "
                "settings, a seed, limits, a segment and what the phase before it handed on; none, where the "
                "phase reads no carried state and runs on segments in any order, or a function from a registry "
                "and settings to the state it carries into its first segment; none, or a function from a "
                "registry and settings to the names of the parts it runs apart" % (name, ", ".join(wrong))
            )
        self.name, self.function, self.carries, self.parts = name, function, carries, parts
        LOG.debug("phase exit name=%s seconds=%.4f", name, time.perf_counter() - started)

    def named(self, registry, settings):
        started = time.perf_counter()
        found = self.parts(registry, settings)
        if type(found) is not tuple or len(set(found)) != len(found) or not all(type(part) is str for part in found):
            raise ValueError(
                "the phase %s names its parts as a tuple of distinct words, found %r" % (self.name, found)
            )
        LOG.debug("named exit phase=%s parts=%d seconds=%.4f", self.name, len(found), time.perf_counter() - started)
        return found

    def initial(self, registry, settings):
        started = time.perf_counter()
        if self.parts is None:
            result = self.carries(registry, settings)
        else:
            result = {part: self.carries(registry, settings) for part in self.named(registry, settings)}
        LOG.debug("phase initial exit phase=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def once(self, graph, registry, settings, seed, limits, carry, held, state, part):
        started = time.perf_counter()
        if (state is None) != (self.carries is None):
            raise ValueError(
                "the phase %s %s carried state" % (self.name, "is given no" if state is None else "carries no")
            )
        carry.state, carry.part = None if state is None else dict(state), part
        try:
            result = self.function(graph, registry, settings, seed, limits, carry, held)
            found = carry.state
        finally:
            carry.state, carry.part = None, None
        LOG.debug("phase once exit phase=%s part=%s seconds=%.4f", self.name, part, time.perf_counter() - started)
        return result, found

    def run(self, graph, registry, settings, seed, limits, carry, held, state):
        if self.parts is None:
            return self.once(graph, registry, settings, seed, limits, carry, held, state, None)
        results, states = {}, {}
        for part in self.named(registry, settings):
            results[part], states[part] = self.once(
                graph, registry, settings, seed, limits, carry, held, None if state is None else state[part], part
            )
        return results, None if state is None else states


class Operation:
    def __init__(
        self, namespace, name, phases, parameters, reads, splits, requires, changes, adds, kinds, version, graphs
    ):
        started = time.perf_counter()
        for label, part in (("namespace", namespace), ("name", name)):
            if type(part) is not str or not part or "." in part:
                raise ValueError("an operation states its %s as a word without a dot, found %r" % (label, part))
        self.name = namespace + "." + name
        stated = dict(
            phases=type(phases) is tuple
            and len(phases) > 0
            and all(isinstance(phase, Phase) for phase in phases)
            and len({phase.name for phase in phases}) == len(phases)
            and phases[-1].parts is None
            and not any(
                phase.parts is not None and later.parts is not None for phase, later in zip(phases, phases[1:])
            ),
            parameters=callable(parameters),
            reads=type(reads) is tuple and all(type(read) is str for read in reads),
            splits=type(splits) is tuple and all(split in SPLITS for split in splits),
            requires=type(requires) is tuple and all(isinstance(count, Count) for count in requires),
            changes=type(changes) is tuple and all(change in CHANGES for change in changes),
            adds=adds is None or callable(adds),
            kinds=kinds is None or callable(kinds),
            version=type(version) is int and version >= 1,
            graphs=graphs in (1, 2) and type(graphs) is int,
        )
        wrong = [label for label, held in stated.items() if not held]
        if wrong:
            raise ValueError(
                "the operation %s states %s as the constructor of an operation asks: a tuple of its phases, at "
                "least one, of distinct names, the last not run in parts and none run in parts after one that "
                "is; a function from a registry to its parameters; a tuple of the "
                "names of the operations whose parameters it reads; a tuple of the splits it may read, among %s; "
                "a tuple of counts; a tuple of what it changes, among %s; none, or a function from a registry, "
                "settings and the name of its entry to what it adds; none, or a function from a registry and "
                "settings to what it does with kinds of node, as a mapping from each to one of %s; a whole "
                "version from 1; the number of graphs it takes, 1, or 2 where it reads a second graph beside the "
                "one it changes"
                % (self.name, ", ".join(wrong), ", ".join(SPLITS), ", ".join(CHANGES), ", ".join(ROLES))
            )
        self.namespace, self.phases, self.parameters, self.reads = namespace, phases, parameters, reads
        self.splits, self.requires, self.changes, self.version = splits, requires, changes, version
        self.adds, self.kinds, self.graphs = adds, kinds, graphs
        LOG.debug("operation exit name=%s seconds=%.4f", self.name, time.perf_counter() - started)

    def stated(self, registry):
        held = STATED.setdefault(registry, {})
        if self in held and held[self][0] == registries.GENERATION:
            return held[self][1]
        started = time.perf_counter()
        found = tuple(self.parameters(registry))
        counted = Counter(parameter.name for parameter in found)
        twice = sorted(name for name, count in counted.items() if count > 1)
        if twice:
            raise ValueError("the operation %s states the parameters %s twice" % (self.name, ", ".join(twice)))
        held[self] = registries.GENERATION, found
        LOG.debug(
            "stated exit operation=%s parameters=%d seconds=%.4f",
            self.name,
            len(found),
            time.perf_counter() - started,
        )
        return found

    def settings(self, registry, spec, graph, partitions):
        started = time.perf_counter()
        parameters = self.stated(registry)
        for read in self.reads:
            parameters += OPERATIONS.named(read).stated(registry)

        def tasks(name):
            if graph is None:
                raise ValueError(
                    "the operation %s reads the split %s where it states the state it carries" % (self.name, name)
                )
            if partitions is None:
                raise ValueError(
                    "the derivation of graph %s was given no splits, and the operation %s reads the split %s"
                    % (graph.name, self.name, name)
                )
            rows = np.asarray(graph.label_row, np.int64)
            return np.unique(np.asarray(graph.label_task)[partitions[rows] == SPLITS.index(name)])

        result = Settings(self.name, parameters, spec, self.splits, tasks)
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

    def added(self, registry, spec, entry):
        if self.adds is None:
            return None
        started = time.perf_counter()
        result = self.adds(registry, self.settings(registry, spec, None, None), entry)
        if not isinstance(result, Added):
            raise ValueError(
                "the operation %s states what it adds as graphspec.schema.added.Added, found %r" % (self.name, result)
            )
        LOG.debug("added exit operation=%s entry=%s seconds=%.4f", self.name, entry, time.perf_counter() - started)
        return result

    def roles(self, registry, spec):
        if self.kinds is None:
            return {}
        started = time.perf_counter()
        result = self.kinds(registry, self.settings(registry, spec, None, None))
        if type(result) is not dict or not all(
            kind in registry.declared and role in ROLES for kind, role in result.items()
        ):
            raise ValueError(
                "the operation %s states what it does with kinds of node as a mapping from kinds the registry "
                "declares to one of %s, found %r" % (self.name, ", ".join(ROLES), result)
            )
        LOG.debug("roles exit operation=%s kinds=%d seconds=%.4f", self.name, len(result), time.perf_counter() - started)
        return result

    def initial(self, registry, spec):
        started = time.perf_counter()
        settings = self.settings(registry, spec, None, None)
        result = {phase.name: phase.initial(registry, settings) for phase in self.phases if phase.carries is not None}
        LOG.debug(
            "initial exit operation=%s phases=%d seconds=%.4f", self.name, len(result), time.perf_counter() - started
        )
        return result

    def phase(self, place, graph, registry, spec, seed, limits, carry, held, state, part, partitions):
        started = time.perf_counter()
        settings = self.settings(registry, spec, graph, partitions)
        if place == 0:
            self.check(registry, settings)
        result = self.phases[place].once(graph, registry, settings, seed, limits, carry, held, state, part)
        LOG.debug(
            "operation phase exit name=%s phase=%d part=%s seconds=%.4f",
            self.name,
            place,
            part,
            time.perf_counter() - started,
        )
        return result

    def __call__(self, graph, registry, spec, seed, limits, carry, state, partitions):
        started = time.perf_counter()
        settings = self.settings(registry, spec, graph, partitions)
        self.check(registry, settings)
        held, found = None, {}
        for phase in self.phases:
            given = None if phase.carries is None else state[phase.name]
            held, carried = phase.run(graph, registry, settings, seed, limits, carry, held, given)
            if phase.carries is not None:
                found[phase.name] = carried
        LOG.debug(
            "operation call exit name=%s graph=%s seconds=%.4f",
            self.name,
            graph.name,
            time.perf_counter() - started,
        )
        return held, found

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
            adds=self.adds is not None,
            kinds=self.kinds is not None,
            graphs=self.graphs,
            phases=[
                dict(name=phase.name, carries=phase.carries is not None, parts=phase.parts is not None)
                for phase in self.phases
            ],
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
