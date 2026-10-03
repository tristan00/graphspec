import json
import logging
import time
from collections import ChainMap
from functools import partial
from weakref import WeakKeyDictionary

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.schema import ENCODERS, TYPES
from graphspec.schema.targets import pointed

from .operation import OPERATIONS
from .parameters import Parameter, flag

LOG = logging.getLogger(__name__)
STATED = ("operation", "label", "graph", "parameters")
TEMPLATE = ("operation", "label", "graph")
OPTIONAL = ("label", "graph")
PRESENT = "pipeline/%s"
NAMED = "pipeline/%s/"
HITS = ("names", "removes")
NONE = frozenset()
PROBED = WeakKeyDictionary()
PLANNED = WeakKeyDictionary()


class Step:
    def __init__(self, name, operation, label, graph, parameters, read):
        self.name, self.operation, self.label, self.graph = name, operation, label, graph
        self.parameters, self.read = parameters, read
        self.spec = dict(read, **parameters)

    def __call__(self, graph, registry, seed, limits, carry, state, partitions):
        carry.named(self.graph, self.operation.name, called(self))
        try:
            return self.operation(graph, registry, self.spec, seed, limits, carry, state, partitions)
        finally:
            carry.second, carry.entry = None, None

    @timed("pipeline.Step.initial")
    def initial(self, registry):
        return self.operation.initial(registry, self.spec)

    @timed("pipeline.Step.phase")
    def phase(self, place, graph, registry, seed, limits, carry, held, state, part, partitions):
        carry.named(self.graph, self.operation.name, called(self))
        try:
            return self.operation.phase(
                place, graph, registry, self.spec, seed, limits, carry, held, state, part, partitions
            )
        finally:
            carry.second, carry.entry = None, None

    @timed("pipeline.Step.parts")
    def parts(self, place, registry):
        phase = self.operation.phases[place]
        if phase.parts is None:
            return None
        return phase.named(registry, self.operation.settings(registry, self.spec, None, None))

    def entry(self):
        started = time.perf_counter()
        result = dict(operation=self.operation.name, parameters=self.parameters)
        if self.label is not None:
            result["label"] = self.label
        if self.graph is not None:
            result["graph"] = self.graph
        LOG.debug("entry exit step=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def keyed(self):
        started = time.perf_counter()
        result = dict(
            operation=self.operation.name,
            label=self.label,
            graph=self.graph,
            version=self.operation.version,
            parameters=self.parameters,
        )
        LOG.debug("keyed exit step=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result


class Local:
    __slots__ = ("flat", "prefixes")

    def __init__(self, flat, prefixes):
        self.flat, self.prefixes = flat, prefixes

    def __contains__(self, name):
        return any(prefix + name in self.flat for prefix in self.prefixes)

    def __getitem__(self, name):
        for prefix in self.prefixes:
            if prefix + name in self.flat:
                return self.flat[prefix + name]
        raise KeyError(name)


def checked(entries, fields):
    if type(entries) is not list:
        raise TypeError("a pipeline is a list of its operations in the order they apply, found %r" % (entries,))
    required = tuple(field for field in fields if field not in OPTIONAL)
    found, names = [], set()
    for place, entry in enumerate(entries):
        if type(entry) is not dict or not set(entry) <= set(fields):
            raise ValueError(
                "the entry %d of the pipeline is a mapping of %s, with a label where it states one and the name "
                "of its second graph where its operation takes two; found %r" % (place, " and ".join(required), entry)
            )
        lacking = [field for field in required if field not in entry]
        if lacking:
            raise ValueError("the entry %d of the pipeline states no %s" % (place, " and no ".join(lacking)))
        operation = OPERATIONS.named(entry["operation"])
        label = entry["label"] if "label" in entry else None
        if label is not None and (type(label) is not str or not label or "/" in label or "." in label):
            raise ValueError(
                "the label of the entry %d of the pipeline is a word without a slash or a dot, found %r"
                % (place, label)
            )
        second = entry["graph"] if "graph" in entry else None
        if operation.graphs == 2 and (type(second) is not str or not second):
            raise ValueError(
                "the entry %d of the pipeline, of %s, which takes two graphs, names its second graph under graph, "
                "found %r" % (place, operation.name, second)
            )
        if operation.graphs == 1 and second is not None:
            raise ValueError(
                "the entry %d of the pipeline, of %s, which takes one graph, names a second graph %r"
                % (place, operation.name, second)
            )
        name = operation.name if label is None else label
        if name in names:
            raise ValueError(
                "the pipeline holds %s twice; an operation stated more than once states a label on every entry "
                "of it but one, and no two labels are equal" % name
            )
        names.add(name)
        found.append((name, operation, label, entry))
    return found


def readers(found):
    started = time.perf_counter()
    unlabelled = {operation.name: name for name, operation, label, _ in found if label is None}
    result = {
        name: tuple(unlabelled[read] for read in operation.reads if read in unlabelled)
        for name, operation, _, _ in found
    }
    LOG.debug("readers exit entries=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def walked(found, before, looped):
    started = time.perf_counter()
    order, done, entered = [], set(), set()
    for first, _, _, _ in found:
        waiting = [(first, False)]
        while waiting:
            name, expanded = waiting.pop()
            if name in done:
                continue
            if expanded:
                done.add(name)
                order.append(name)
                continue
            if name in entered:
                raise ValueError(looped % name)
            entered.add(name)
            waiting.append((name, True))
            waiting.extend((read, False) for read in reversed(before[name]))
    LOG.debug("walked exit entries=%d seconds=%.4f", len(order), time.perf_counter() - started)
    return order


@timed("pipeline.drawn")
def drawn(found, reads):
    return walked(found, reads, "the entry %s of the pipeline reads, through the entries it reads, itself")


def ordered(found, reads, waits):
    started = time.perf_counter()
    before = {
        name: tuple(earlier for earlier, other, _, _ in found[:place] if getattr(other, waits) is not None) + reads[name]
        for place, (name, _, _, _) in enumerate(found)
    }
    result = walked(
        found,
        before,
        "the entry %%s of the pipeline comes after itself, through the entries it reads and the entries before them "
        "whose operation states %s" % waits,
    )
    LOG.debug("ordered exit entries=%d waits=%s seconds=%.4f", len(result), waits, time.perf_counter() - started)
    return result


def entered(operation, label):
    return label if label is not None else operation.name.split(".", 1)[1]


@timed("pipeline.specified")
def specified(stated, reads, name):
    return dict(ChainMap(*[stated[read] for read in reads[name]]), **stated[name])


class Growing:
    def __init__(self, registry, found):
        started = time.perf_counter()
        self.found, self.held = found, [registry]
        LOG.debug("growing exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)

    def before(self, place, spec):
        while len(self.held) <= place:
            name, operation, label, _ = self.found[len(self.held) - 1]
            held = self.held[-1]
            if operation.adds is not None:
                added = operation.added(held, spec(name), entered(operation, label))
                if added.holds():
                    held = held.grown(entered(operation, label), added, None)
            self.held.append(held)
        return self.held[place]


def given(name, parameters, values, context):
    if type(values) is not dict:
        raise TypeError("the entry %s of the pipeline states its parameters as a mapping, found %r" % (name, values))
    result = {}
    view = ChainMap(result, context)
    for parameter in parameters:
        if parameter.active(view):
            if parameter.name not in values:
                raise ValueError(
                    "the entry %s of the pipeline states no %s, which is active under what it states"
                    % (name, parameter.name)
                )
            result[parameter.name] = parameter.check(values[parameter.name])
    extra = sorted(set(values) - set(result))
    if extra:
        raise ValueError(
            "the entry %s of the pipeline states %s, which are inactive or unknown" % (name, ", ".join(extra))
        )
    return result


def resolved(registry, entries):
    started = time.perf_counter()
    found = checked(entries, STATED)
    reads, held = readers(found), {name: (operation, entry) for name, operation, _, entry in found}
    places = {name: place for place, (name, _, _, _) in enumerate(found)}
    stated, growing = {}, Growing(registry, found)
    for name in ordered(found, reads, "adds"):
        operation, entry = held[name]
        context = ChainMap(*[stated[read] for read in reads[name]])
        grown = growing.before(places[name], partial(specified, stated, reads))
        lacking = operation.lacking(grown)
        if lacking is not None:
            raise ValueError("the entry %s of the pipeline is of %s, which %s" % (name, operation.name, lacking))
        stated[name] = given(name, operation.stated(grown), entry["parameters"], context)
        refused = operation.refused(grown, stated[name])
        if refused is not None:
            raise ValueError("the entry %s of the pipeline states %s %s, which %s" % ((name,) + refused))
    result = stepped(found, reads, stated)
    emptied(registry, result)
    LOG.debug("resolved exit steps=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def stepped(found, reads, stated):
    started = time.perf_counter()
    result = tuple(
        Step(
            name,
            operation,
            label,
            entry["graph"] if "graph" in entry else None,
            stated[name],
            dict(ChainMap(*[stated[read] for read in reads[name]])),
        )
        for name, operation, label, entry in found
    )
    LOG.debug("stepped exit steps=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def rebuilt(entries):
    started = time.perf_counter()
    found = checked(entries, STATED)
    result = stepped(found, readers(found), {name: entry["parameters"] for name, _, _, entry in found})
    LOG.debug("rebuilt exit steps=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def targeted(registry):
    started = time.perf_counter()
    result = {}
    for task, kinds in pointed(registry.schema).items():
        for kind in kinds:
            result.setdefault(kind, task)
    LOG.debug("targeted exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def emptied(registry, steps):
    started = time.perf_counter()
    removed, named = {}, targeted(registry)
    for step in steps:
        roles = step.operation.roles(registry, step.spec)
        lost = [kind for kind, role in roles.items() if role != "fills" and kind in removed]
        if lost:
            raise ValueError(
                "the entry %s of the pipeline names the kind %s, which the entry %s before it removed; an entry "
                "derives nothing for a kind no node of which is left" % (step.name, lost[0], removed[lost[0]])
            )
        gone = [kind for kind, role in roles.items() if role == "removes" and kind in named]
        if gone:
            raise ValueError(
                "the entry %s of the pipeline removes the kind %s, which the target of the kind of task %s names; a "
                "pipeline keeps every kind of node a target names" % (step.name, gone[0], named[gone[0]])
            )
        for kind, role in roles.items():
            if role == "removes":
                removed[kind] = step.name
            elif role == "fills":
                removed.pop(kind, None)
    LOG.debug("emptied exit steps=%d removed=%d seconds=%.4f", len(steps), len(removed), time.perf_counter() - started)
    return removed


def called(step):
    return entered(step.operation, step.label)


def read(registry, steps):
    started = time.perf_counter()
    held, additions, result = registry, [], []
    for step in steps:
        for parameter in step.operation.stated(held):
            if parameter.registry is not None and parameter.name in step.parameters:
                result.append((parameter.registry, [step.parameters[parameter.name]]))
        added = step.operation.added(held, step.spec, called(step))
        if added is not None and added.holds():
            columns = added.columns.values()
            result += [(TYPES, [column["type"] for column in columns]), (ENCODERS, [column["encoder"] for column in columns])]
            held = held.grown(called(step), added, None)
            additions.append((called(step), added))
    LOG.debug(
        "read exit steps=%d added=%d read=%d seconds=%.4f",
        len(steps),
        len(additions),
        len(result),
        time.perf_counter() - started,
    )
    return additions, result


def grown(registry, steps, states):
    started = time.perf_counter()
    held = registry
    for step in steps:
        added = step.operation.added(held, step.spec, called(step))
        if added is not None and added.holds():
            given = None if states is None else {name: states[name] for name in added.columns}
            held = held.grown(called(step), added, given)
    LOG.debug("grown exit steps=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return held


def sampled(graph, registry):
    started = time.perf_counter()
    encoding, entities = registry.encoding, registry.schema.entities()
    kinds = np.asarray(graph.kinds)[np.asarray(graph.obs_entity, np.int64)]
    rows = np.asarray(graph.x_table)[np.asarray(graph.obs_ref, np.int64)]
    stated = np.asarray(graph.fact_present, bool) & (np.asarray(graph.fact_ref) >= 0)
    result = {}
    for name in registry.schema.added_columns():
        values, flags = encoding.slots(name)
        holders = [registry.kind(kind) for kind, columns in entities.items() if name in columns]
        chosen = np.isin(kinds, holders) & (rows[:, flags[0]] > 0)
        found = [rows[chosen, values[0]].astype(np.float32)]
        for group in registry.valued:
            if name in registry.attributes(group):
                facts = stated & np.isin(graph.fact_code, registry.coded[group])
                column = np.asarray(graph.value_table)[graph.fact_ref[facts], registry.attributes(group).index(name)]
                found.append(np.asarray(column[~np.isnan(column)], np.float32))
        result[name] = np.concatenate(found)
    LOG.debug("sampled exit columns=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def qualified(present, parameter, prefixes):
    return lambda flat: flat[present] and parameter.active(Local(flat, prefixes))


def space(registry, template):
    started = time.perf_counter()
    found = checked(template, TEMPLATE)
    reads, held = readers(found), {name: operation for name, operation, _, _ in found}
    result = []
    for name in drawn(found, reads):
        present, own, operation = PRESENT % name, NAMED % name, held[name]
        prefixes = (own,) + tuple(NAMED % read for read in reads[name])
        offered = None
        if operation.lacking(registry) is None:
            offered = [(parameter, operation.offered(registry, parameter)) for parameter in operation.stated(registry)]
        if offered is None or any(parameter.options and not options for parameter, options in offered):
            result.append(Parameter(present, "bool", False, options=(False,)))
            continue
        result.append(flag(present, True))
        result += [
            Parameter(
                own + parameter.name,
                parameter.kind,
                parameter.default if parameter.default in options or not parameter.options else None,
                parameter.low,
                parameter.high,
                parameter.log,
                options,
                qualified(present, parameter, prefixes),
            )
            for parameter, options in offered
        ]
    LOG.debug("pipeline space exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


def entries(template, flat):
    started = time.perf_counter()
    result = []
    for name, operation, label, stated in checked(template, TEMPLATE):
        if not flat[PRESENT % name]:
            continue
        own = NAMED % name
        entry = dict(
            operation=operation.name,
            parameters={key[len(own) :]: value for key, value in flat.items() if key.startswith(own)},
        )
        if label is not None:
            entry["label"] = label
        if "graph" in stated:
            entry["graph"] = stated["graph"]
        result.append(entry)
    LOG.debug("entries exit entries=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def flattened(template, stated):
    started = time.perf_counter()
    found = checked(template, TEMPLATE)
    named = {name: entry for name, _, _, entry in checked(stated, STATED)}
    strangers = sorted(set(named) - {name for name, _, _, _ in found})
    if strangers:
        raise ValueError("the pipeline holds %s, which its template does not" % ", ".join(strangers))
    order = [name for name, _, _, _ in found if name in named]
    if order != list(named):
        raise ValueError(
            "the pipeline orders its entries %s and its template %s" % (", ".join(named), ", ".join(order))
        )
    result = {}
    for name, _, _, entry in found:
        result[PRESENT % name] = name in named
        if name in named:
            if "graph" in entry and named[name]["graph"] != entry["graph"]:
                raise ValueError(
                    "the entry %s of the pipeline takes the graph %s as its second graph, and its template %s"
                    % (name, named[name]["graph"], entry["graph"])
                )
            result.update({NAMED % name + key: value for key, value in named[name]["parameters"].items()})
    LOG.debug("flattened exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def settled(parameter):
    return parameter.default if parameter.default is not None else parameter.options[0]


def roled(found, roles):
    return frozenset(kind for kind, role in found.items() if role in roles)


def made(operation, registry, context, value):
    result = {}
    view = ChainMap(result, context)
    for parameter in operation.stated(registry):
        if parameter.active(view):
            result[parameter.name] = value(parameter)
    return result


def probed(registry, operation, bounded, context):
    started = time.perf_counter()
    key = (
        operation.name,
        tuple((name, held.default, held.low, held.high, held.options) for name, held in sorted(bounded.items())),
        tuple(sorted(context.items())),
    )
    held = PROBED.setdefault(registry, {})
    if key in held:
        LOG.debug("probed exit operation=%s held seconds=%.4f", operation.name, time.perf_counter() - started)
        return held[key]
    choices = [parameter.name for parameter in operation.stated(registry) if bounded[parameter.name].options]
    defaults = {parameter.name: settled(bounded[parameter.name]) for parameter in operation.stated(registry)}
    drafts, probes = {}, {}

    def spec(chosen):
        key = tuple(chosen.get(name, value) for name, value in defaults.items())
        if key not in drafts:
            drafts[key] = made(
                operation, registry, context, lambda parameter: chosen.get(parameter.name, defaults[parameter.name])
            )
        return drafts[key]

    def probe(chosen):
        key = tuple(chosen.get(name, value) for name, value in defaults.items())
        if key not in probes:
            probes[key] = operation.roles(registry, dict(context, **spec(chosen)))
        return probes[key]

    origin, found = spec({}), probe({})
    must = dict(hits=roled(found, HITS), removes=roled(found, ("removes",)), fills=roled(found, ("fills",)))
    hits, removes = {name: {} for name in choices}, {name: {} for name in choices}
    for name in choices:
        options = bounded[name].options
        starts = [{}] if name in origin else [
            {other: option}
            for other in choices
            if other != name
            for option in bounded[other].options
            if name in spec({other: option})
        ]
        for start in starts:
            found = [probe({**start, name: option}) for option in options]
            for group, roles, target in (("hits", HITS, hits), ("removes", ("removes",), removes)):
                sets = [roled(each, roles) for each in found]
                common = frozenset.intersection(*sets)
                for option, each in zip(options, sets):
                    target[name][option] = target[name].get(option, NONE) | (each - common)
                if not start:
                    must[group] &= common
            if not start:
                must["fills"] &= frozenset.intersection(*[roled(each, ("fills",)) for each in found])
    held[key] = hits, removes, must
    LOG.debug(
        "probed exit operation=%s parameters=%d seconds=%.4f", operation.name, len(choices), time.perf_counter() - started
    )
    return held[key]


class Offered:
    def __init__(self, registry, template, parameters):
        started = time.perf_counter()
        found = checked(template, TEMPLATE)
        self.reads, self.names = readers(found), [name for name, _, _, _ in found]
        self.place = {name: place for place, name in enumerate(self.names)}
        self.operations = {name: operation for name, operation, _, _ in found}
        self.flags, self.owned = {}, {name: [] for name in self.names}
        for parameter in parameters:
            parts = parameter.name.split("/", 2)
            if len(parts) == 2:
                self.flags[parts[1]] = parameter
            else:
                self.owned[parts[1]].append(parameter)
        self.order = ordered(found, self.reads, "kinds")
        bounded = {
            name: {parameter.name[len(NAMED % name) :]: parameter for parameter in self.owned[name]} for name in self.names
        }
        self.probes = {}
        for name in self.names:
            self.probes[name] = None
            if self.operations[name].kinds is None or True not in self.flags[name].options:
                continue
            context = {}
            for read in self.reads[name]:
                if True in self.flags[read].options:
                    context.update(
                        made(self.operations[read], registry, {}, lambda parameter: settled(bounded[read][parameter.name]))
                    )
            self.probes[name] = probed(registry, self.operations[name], bounded[name], context)
        self.protected, ahead = {}, {}
        tasked = {kind: "the target of the kind of task %s" % task for kind, task in targeted(registry).items()}
        for name in reversed(self.names):
            self.protected[name] = dict(ahead, **tasked)
            if self.flags[name].options == (True,) and self.probes[name] is not None:
                must = self.probes[name][2]
                ahead = {kind: held for kind, held in ahead.items() if kind not in must["fills"]}
                ahead.update(dict.fromkeys(must["hits"], "the entry %s after it, held present," % name))
        LOG.debug("offered exit entries=%d seconds=%.4f", len(self.names), time.perf_counter() - started)

    @timed("pipeline.Offered.removed")
    def removed(self, name, roles):
        result = {}
        for earlier in self.names[: self.place[name]]:
            if earlier not in roles:
                if self.probes[earlier] is not None:
                    raise ValueError(
                        "the entry %s of the template is drawn before the entry %s, which comes before it and may "
                        "remove kinds it names" % (name, earlier)
                    )
                continue
            for kind, role in roles[earlier].items():
                if role == "removes":
                    result[kind] = earlier
                elif role == "fills":
                    result.pop(kind, None)
        return result

    def allowed(self, name, parameter, removed, guard):
        probes, own = self.probes[name], parameter.name[len(NAMED % name) :]
        if probes is None or not parameter.options or own not in probes[0]:
            return parameter.options
        hits, removes = probes[0][own], probes[1][own]
        return tuple(
            option
            for option in parameter.options
            if not hits.get(option, NONE) & removed.keys() and not removes.get(option, NONE) & guard.keys()
        )

    @timed("pipeline.Offered.blocked")
    def blocked(self, name, flat, removed, guard):
        must = self.probes[name][2]
        lost = sorted(must["hits"] & removed.keys())
        if lost:
            return "names the kind %s, which the entry %s before it removed" % (lost[0], removed[lost[0]])
        kept = sorted(must["removes"] & guard.keys())
        if kept:
            return "removes the kind %s, which %s names" % (kept[0], guard[kept[0]])
        trial = {**flat, self.flags[name].name: True}
        for parameter in self.owned[name]:
            if parameter.active(trial):
                held = self.allowed(name, parameter, removed, guard)
                if parameter.options and not held:
                    return "is left no value of %s that names no kind an entry before it removed" % parameter.name
                trial[parameter.name] = held[0] if parameter.options else parameter.default
        return None

    @timed("pipeline.Offered.context")
    def context(self, name, flat):
        result = {}
        for read in self.reads[name] + (name,):
            prefix = NAMED % read
            result.update({held.name[len(prefix) :]: flat[held.name] for held in self.owned[read] if held.name in flat})
        return result

    @timed("pipeline.Offered.verified")
    def verified(self, registry, name, flat, removed, guard):
        found = self.operations[name].roles(registry, self.context(name, flat))
        lost = sorted(roled(found, HITS) & removed.keys())
        kept = sorted(roled(found, ("removes",)) & guard.keys())
        if lost or kept:
            raise ValueError(
                "the entry %s of the template %s by values none of which does alone; its operation states what it "
                "does with kinds one parameter at a time"
                % (
                    name,
                    "names the kind %s, which the entry %s before it removed," % (lost[0], removed[lost[0]])
                    if lost
                    else "removes the kind %s, which %s names," % (kept[0], guard[kept[0]]),
                )
            )
        return found


@timed("pipeline.offered")
def offered(registry, template, parameters):
    key = (json.dumps(template, sort_keys=True), parameters)
    held = PLANNED.get(registry)
    if held is None or held[0] != key:
        held = PLANNED[registry] = key, Offered(registry, template, parameters)
    return held[1]


def narrowed(parameter, options):
    if options == parameter.options:
        return parameter
    default = parameter.default if parameter.default in options else None
    return Parameter(parameter.name, parameter.kind, default, options=options, condition=parameter.condition)


def draw(registry, template, parameters, choose):
    started = time.perf_counter()
    plan, flat, roles = offered(registry, template, tuple(parameters)), {}, {}
    for name in plan.order:
        removed, guard, flag = plan.removed(name, roles), plan.protected[name], plan.flags[name]
        options, roles[name] = flag.options, {}
        if plan.probes[name] is not None and True in options:
            reason = plan.blocked(name, flat, removed, guard)
            if reason is not None:
                options = tuple(option for option in options if not option)
                if not options:
                    raise ValueError("the bounds hold the entry %s of the template present, and it %s" % (name, reason))
        flat[flag.name] = choose(narrowed(flag, options))
        if not flat[flag.name]:
            continue
        for parameter in plan.owned[name]:
            if parameter.active(flat):
                held = plan.allowed(name, parameter, removed, guard)
                if parameter.options and not held:
                    raise ValueError(
                        "the entry %s of the template is left no value of %s that names no kind an entry before it "
                        "removed and removes none an entry after it, held present, or a target names"
                        % (name, parameter.name)
                    )
                flat[parameter.name] = choose(narrowed(parameter, held))
        if plan.probes[name] is not None:
            roles[name] = plan.verified(registry, name, flat, removed, guard)
    LOG.debug("pipeline draw exit parameters=%d seconds=%.4f", len(flat), time.perf_counter() - started)
    return flat
