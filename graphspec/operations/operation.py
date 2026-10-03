import logging
import time
from collections import Counter
from weakref import WeakKeyDictionary

import numpy as np

from graphspec import registries
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.added import Added
from graphspec.views.graph import CHANGES as CHECKED

from .sets import PARAMETER, STORAGE

LOG = logging.getLogger(__name__)
CHANGES = CHECKED + ("references",)
SPLITS = ("trained", "scored")
TREATMENTS = ("stored", "refuses")
ROLES = ("names", "removes", "fills")
NUMBERS = ("int", "float")
STATED = WeakKeyDictionary()
EDGES = dict(
    sets="the edges joining sets of the kind %s", about="the edges of the table %s, which are about edges"
)
SEGMENT = dict(
    glob="its entities",
    known="what it knows of its entities",
    x_global="its rows of values of nodes",
    value_global="its rows of values of facts",
)


class Refused(ValueError):
    pass


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

    @timed("operation.Count.declaration")
    def declaration(self):
        return dict(name=self.name, counts=self.counts, sets=self.sets, about=self.about)


class Need:
    def __init__(self, parameter, lacking):
        started = time.perf_counter()
        if not (parameter is None or type(parameter) is str and parameter) or not callable(lacking):
            raise ValueError(
                "a need states none, where the operation needs it whatever its parameters, or the name of the "
                "parameter whose every value needs it, and a function from a registry and that value, none where "
                "no parameter is named, to none where the bank holds what it needs and otherwise what it lacks; "
                "found %r and %r" % (parameter, lacking)
            )
        self.parameter, self.lacking = parameter, lacking
        LOG.debug("need exit parameter=%s seconds=%.4f", parameter, time.perf_counter() - started)

    @timed("operation.Need.declaration")
    def declaration(self):
        return dict(need=self.parameter)


class Stored:
    def __init__(self, parameter, kinds):
        started = time.perf_counter()
        if type(parameter) is not str or not parameter or not callable(kinds):
            raise ValueError(
                "a need of set edges stored as nodes states the name of the parameter whose values need them and a "
                "function from a registry and a value of it to the kinds of set edge that value needs stored as "
                "nodes; found %r and %r" % (parameter, kinds)
            )
        self.parameter, self.kinds = parameter, kinds
        LOG.debug("stored exit parameter=%s seconds=%.4f", parameter, time.perf_counter() - started)

    @timed("operation.Stored.declaration")
    def declaration(self):
        return dict(stored=self.parameter)


class Settings:
    def __init__(self, operation, parameters, spec, splits, tasks, numbers):
        self.operation, self.spec, self.splits, self.tasks = operation, spec, splits, tasks
        self.parameters, self.numbers = {parameter.name: parameter for parameter in parameters}, numbers

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
        if not self.numbers and self.parameters[name].kind in NUMBERS:
            raise Refused(
                "the operation %s reads its number %s in what it does with kinds of node; it states that from its "
                "choices and flags alone, since a search offers each value of a parameter by what it does with kinds "
                "and cannot try every number" % (self.operation, name)
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
        return result, found

    @timed("operation.Phase.apart")
    def apart(self, graph, registry, settings, seed, limits, carry, held, state, part):
        before = {name: getattr(carry, name) for name in SEGMENT}
        drawn, carry.candidates = carry.candidates, []
        try:
            result, found = self.once(graph, registry, settings, seed, limits, carry, held, state, part)
        finally:
            drawn, carry.candidates = carry.candidates, drawn
        changed = [SEGMENT[name] for name, value in before.items() if getattr(carry, name) is not value]
        if changed:
            raise ValueError(
                "the part %s of the phase %s of the operation %s changes %s of its segment; a part changes only its "
                "own state and the candidates it draws, which reach the next phase with the results of every part"
                % (part, self.name, settings.operation, " and ".join(changed))
            )
        return (result, drawn), found

    def run(self, graph, registry, settings, seed, limits, carry, held, state):
        if self.parts is None:
            return self.once(graph, registry, settings, seed, limits, carry, held, state, None)
        made, states = {}, {}
        for part in self.named(registry, settings):
            made[part], states[part] = self.apart(
                graph, registry, settings, seed, limits, carry, held, None if state is None else state[part], part
            )
        return gathered(carry, made), None if state is None else states


@timed("operation.gathered")
def gathered(carry, made):
    for _, drawn in made.values():
        carry.candidates.extend(drawn)
    return {part: result for part, (result, _) in made.items()}


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
            requires=type(requires) is tuple and all(isinstance(held, (Count, Need, Stored)) for held in requires),
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
                "a tuple of counts, needs of a bank and needs of set edges stored as nodes; a tuple of what it "
                "changes, among %s; none, or a function from a registry, "
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
        if self not in held or held[self][0] != registries.GENERATION:
            held[self] = registries.GENERATION, self.gathered(registry)
        return held[self][1]

    def gathered(self, registry):
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

    def settings(self, registry, spec, graph, partitions, numbers):
        parameters = self.stated(registry)
        for read in self.reads:
            parameters += OPERATIONS.named(read).stated(registry)

        def tasks(name):
            started = time.perf_counter()
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
            result = np.unique(np.asarray(graph.label_task)[partitions[rows] == SPLITS.index(name)])
            LOG.debug("tasks exit operation=%s split=%s seconds=%.4f", self.name, name, time.perf_counter() - started)
            return result

        return Settings(self.name, parameters, spec, self.splits, tasks, numbers)

    def check(self, registry, settings):
        met = dict(sets=tuple(registry.schema.sets), about=tuple(registry.schema.about()))
        if any(met.values()):
            for count in self.requires:
                if not isinstance(count, Count) or not count.active(registry, settings):
                    continue
                for part, found in met.items():
                    if found and getattr(count, part) == "refuses":
                        raise ValueError(
                            "the count %s of the operation %s counts %s and refuses %s"
                            % (count.name, self.name, count.counts, EDGES[part] % found[0])
                        )

    @timed("operation.Operation.needs")
    def needs(self, parameter):
        return [need for need in self.requires if isinstance(need, Need) and need.parameter == parameter]

    @timed("operation.Operation.lacking")
    def lacking(self, registry):
        found = [need.lacking(registry, None) for need in self.needs(None)]
        return next((held for held in found if held is not None), None)

    @timed("operation.Operation.offered")
    def offered(self, registry, parameter):
        needs = self.needs(parameter.name)
        return tuple(
            option for option in parameter.options if all(need.lacking(registry, option) is None for need in needs)
        )

    @timed("operation.Operation.unstored")
    def unstored(self, registry, values, settings):
        found = [
            (held.parameter, values[held.parameter], kind)
            for held in self.requires
            if isinstance(held, Stored) and held.parameter in values
            for kind in held.kinds(registry, values[held.parameter])
        ]
        return next((each for each in found if not STORAGE.named(settings[PARAMETER % each[2]]).nodes), None)

    @timed("operation.Operation.refused")
    def refused(self, registry, values):
        found = [
            (need.parameter, values[need.parameter], need.lacking(registry, values[need.parameter]))
            for need in self.requires
            if isinstance(need, Need) and need.parameter in values
        ]
        return next((held for held in found if held[2] is not None), None)

    def added(self, registry, spec, entry):
        if self.adds is None:
            return None
        result = self.adds(registry, self.settings(registry, spec, None, None, True), entry)
        if not isinstance(result, Added):
            raise ValueError(
                "the operation %s states what it adds as graphspec.schema.added.Added, found %r" % (self.name, result)
            )
        return result

    def roles(self, registry, spec):
        if self.kinds is None:
            return {}
        result = self.kinds(registry, self.settings(registry, spec, None, None, False))
        if type(result) is not dict or not all(
            kind in registry.declared and role in ROLES for kind, role in result.items()
        ):
            raise ValueError(
                "the operation %s states what it does with kinds of node as a mapping from kinds the registry "
                "declares to one of %s, found %r" % (self.name, ", ".join(ROLES), result)
            )
        return result

    def initial(self, registry, spec):
        started = time.perf_counter()
        settings = self.settings(registry, spec, None, None, True)
        result = {phase.name: phase.initial(registry, settings) for phase in self.phases if phase.carries is not None}
        LOG.debug(
            "initial exit operation=%s phases=%d seconds=%.4f", self.name, len(result), time.perf_counter() - started
        )
        return result

    def phase(self, place, graph, registry, spec, seed, limits, carry, held, state, part, partitions):
        started = time.perf_counter()
        settings = self.settings(registry, spec, graph, partitions, True)
        if place == 0:
            self.check(registry, settings)
        running = self.phases[place].once if part is None else self.phases[place].apart
        result = running(graph, registry, settings, seed, limits, carry, held, state, part)
        LOG.debug(
            "operation phase exit name=%s phase=%d part=%s seconds=%.4f",
            self.name,
            place,
            part,
            time.perf_counter() - started,
        )
        return result

    def __call__(self, graph, registry, spec, seed, limits, carry, state, partitions):
        settings = self.settings(registry, spec, graph, partitions, True)
        self.check(registry, settings)
        held, found = None, {}
        for phase in self.phases:
            given = None if phase.carries is None else state[phase.name]
            held, carried = phase.run(graph, registry, settings, seed, limits, carry, held, given)
            if phase.carries is not None:
                found[phase.name] = carried
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


OPERATIONS = Operations("operation", True)


def listed(registry):
    started = time.perf_counter()
    result = tuple(OPERATIONS.named(name).declaration(registry) for name in OPERATIONS.names())
    LOG.debug("listed exit operations=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result
