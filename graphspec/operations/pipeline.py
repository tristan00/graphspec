import logging
import time
from collections import ChainMap

from .operation import OPERATIONS
from .parameters import Parameter, flag

LOG = logging.getLogger(__name__)
STATED = ("operation", "label", "parameters")
TEMPLATE = ("operation", "label")
PRESENT = "pipeline/%s"
NAMED = "pipeline/%s/"


class Step:
    def __init__(self, name, operation, label, parameters, read):
        self.name, self.operation, self.label = name, operation, label
        self.parameters, self.read = parameters, read
        self.spec = dict(read, **parameters)

    def __call__(self, graph, registry, seed, limits, carry, partitions):
        return self.operation(graph, registry, self.spec, seed, limits, carry, partitions)

    def entry(self):
        started = time.perf_counter()
        result = dict(operation=self.operation.name, parameters=self.parameters)
        if self.label is not None:
            result["label"] = self.label
        LOG.debug("entry exit step=%s seconds=%.4f", self.name, time.perf_counter() - started)
        return result

    def keyed(self):
        started = time.perf_counter()
        result = dict(
            operation=self.operation.name,
            label=self.label,
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
    required = tuple(field for field in fields if field != "label")
    found, names = [], set()
    for place, entry in enumerate(entries):
        if type(entry) is not dict or not set(entry) <= set(fields):
            raise ValueError(
                "the entry %d of the pipeline is a mapping of %s, with a label where it states one; found %r"
                % (place, " and ".join(required), entry)
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
        Step(name, operation, label, stated[name], dict(ChainMap(*[stated[read] for read in reads[name]])))
        for name, operation, label, _ in found
    )
    LOG.debug("resolved exit steps=%d seconds=%.4f", len(result), time.perf_counter() - started)
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
    for name, operation, label, _ in checked(template, TEMPLATE):
        if not flat[PRESENT % name]:
            continue
        own = NAMED % name
        entry = dict(
            operation=operation.name,
            parameters={key[len(own) :]: value for key, value in flat.items() if key.startswith(own)},
        )
        if label is not None:
            entry["label"] = label
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
    for name, _, _, _ in found:
        result[PRESENT % name] = name in named
        if name in named:
            result.update({NAMED % name + key: value for key, value in named[name]["parameters"].items()})
    LOG.debug("flattened exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result
