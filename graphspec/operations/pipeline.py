import logging
import time
from collections import ChainMap

import numpy as np

from .operation import OPERATIONS
from .parameters import Parameter, flag

LOG = logging.getLogger(__name__)
STATED = ("operation", "label", "graph", "parameters")
TEMPLATE = ("operation", "label", "graph")
OPTIONAL = ("label", "graph")
PRESENT = "pipeline/%s"
NAMED = "pipeline/%s/"


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

    def initial(self, registry):
        return self.operation.initial(registry, self.spec)

    def phase(self, place, graph, registry, seed, limits, carry, held, state, part, partitions):
        carry.named(self.graph, self.operation.name, called(self))
        try:
            return self.operation.phase(
                place, graph, registry, self.spec, seed, limits, carry, held, state, part, partitions
            )
        finally:
            carry.second, carry.entry = None, None

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
    started = time.perf_counter()
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
    LOG.debug("checked exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
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


def drawn(found, reads):
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
                raise ValueError("the entry %s of the pipeline reads, through the entries it reads, itself" % name)
            entered.add(name)
            waiting.append((name, True))
            waiting.extend((read, False) for read in reversed(reads[name]))
    LOG.debug("drawn exit entries=%d seconds=%.4f", len(order), time.perf_counter() - started)
    return order


def given(name, parameters, values, context):
    started = time.perf_counter()
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
    LOG.debug("given exit entry=%s parameters=%d seconds=%.4f", name, len(result), time.perf_counter() - started)
    return result


def resolved(registry, entries):
    started = time.perf_counter()
    found = checked(entries, STATED)
    reads, held = readers(found), {name: (operation, entry) for name, operation, _, entry in found}
    stated = {}
    for name in drawn(found, reads):
        operation, entry = held[name]
        context = ChainMap(*[stated[read] for read in reads[name]])
        stated[name] = given(name, operation.stated(registry), entry["parameters"], context)
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
    emptied(registry, result)
    LOG.debug("resolved exit steps=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def emptied(registry, steps):
    started = time.perf_counter()
    removed = {}
    for step in steps:
        roles = step.operation.roles(registry, step.spec)
        lost = [kind for kind, role in roles.items() if role != "fills" and kind in removed]
        if lost:
            raise ValueError(
                "the entry %s of the pipeline names the kind %s, which the entry %s before it removed; an entry "
                "derives nothing for a kind no node of which is left" % (step.name, lost[0], removed[lost[0]])
            )
        for kind, role in roles.items():
            if role == "removes":
                removed[kind] = step.name
            elif role == "fills":
                removed.pop(kind, None)
    LOG.debug("emptied exit steps=%d removed=%d seconds=%.4f", len(steps), len(removed), time.perf_counter() - started)
    return removed


def called(step):
    return step.label if step.label is not None else step.operation.name.split(".", 1)[1]


def additions(registry, steps):
    started = time.perf_counter()
    held, result = registry, []
    for step in steps:
        added = step.operation.added(held, step.spec, called(step))
        if added is not None and added.holds():
            held = held.grown(called(step), added, None)
            result.append((called(step), added))
    LOG.debug("additions exit steps=%d added=%d seconds=%.4f", len(steps), len(result), time.perf_counter() - started)
    return result


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
        present, own = PRESENT % name, NAMED % name
        prefixes = (own,) + tuple(NAMED % read for read in reads[name])
        result.append(flag(present, True))
        result += [
            Parameter(
                own + parameter.name,
                parameter.kind,
                parameter.default,
                parameter.low,
                parameter.high,
                parameter.log,
                parameter.options,
                qualified(present, parameter, prefixes),
            )
            for parameter in held[name].stated(registry)
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
