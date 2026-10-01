import fnmatch
import itertools
import logging
import math
import time
from collections import Counter

from graphspec.infra.files import sha256_json
from graphspec.schema.time import SEVERAL
from graphspec.views.view import offered

from . import sets

LOG = logging.getLogger(__name__)
GROUPS = ("coupling", "events", "neighbors", "walks", "mixed")
PACKING = ("time/history", "time/context", "neighbors/order", "neighbors/decay")
DIRECTIONS = ("inverse", "symmetric", "directed")


class Parameter:
    __slots__ = ("name", "kind", "default", "low", "high", "log", "options", "condition")

    def __init__(self, name, kind, default, low=None, high=None, log=False, options=(), condition=None):
        if kind not in ("bool", "choice", "int", "float"):
            raise ValueError("unknown parameter kind: " + kind)
        if kind in ("bool", "choice") and not options:
            raise ValueError("categorical parameters require options: " + name)
        if kind in ("int", "float") and (low is None or high is None or low > high):
            raise ValueError("numeric parameters require ordered bounds: " + name)
        if log and (kind != "float" or low <= 0):
            raise ValueError("logarithmic parameters require positive float bounds: " + name)
        if default is None and kind not in ("bool", "choice"):
            raise ValueError("only a choice or a flag may be stated by the pipeline without a default: " + name)
        self.name, self.kind, self.default = name, kind, default
        self.low, self.high, self.log, self.options = low, high, log, tuple(options)
        self.condition = condition
        if default is not None:
            self.check(default)

    def given(self, supplied):
        if self.name in supplied:
            return self.check(supplied[self.name])
        if self.default is None:
            raise ValueError(
                "the pipeline states no %s, which has no default; it is one of %s"
                % (self.name, list(self.options))
            )
        return self.check(self.default)

    def active(self, spec):
        return self.condition is None or self.condition(spec)

    def check(self, value):
        if self.kind in ("bool", "choice"):
            if type(value) is not type(self.options[0]) or value not in self.options:
                raise ValueError("%s must be one of %s, found %r" % (self.name, list(self.options), value))
            return value
        if self.kind == "int":
            if type(value) is not int or not self.low <= value <= self.high:
                raise ValueError(
                    "%s must be an integer in %d..%d, found %r" % (self.name, self.low, self.high, value)
                )
            return value
        if type(value) not in (int, float) or not self.low <= value <= self.high:
            raise ValueError(
                "%s must be a number in %g..%g, found %r" % (self.name, self.low, self.high, value)
            )
        return float(value)

    def uniform(self, rng):
        if self.kind in ("bool", "choice"):
            return self.options[int(rng.integers(len(self.options)))]
        if self.kind == "int":
            return int(rng.integers(self.low, self.high + 1))
        if self.low == self.high:
            return float(self.low)
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def declaration(self):
        return dict(
            name=self.name,
            kind=self.kind,
            default=self.default,
            low=self.low,
            high=self.high,
            log=self.log,
            options=list(self.options),
        )


def flag(name, default, condition=None):
    return Parameter(name, "bool", default, options=(False, True), condition=condition)


def number(name, low, high, default, condition=None):
    return Parameter(name, "int", default, low, high, condition=condition)


def real(name, low, high, default, condition=None, log=False):
    return Parameter(name, "float", default, low, high, log, condition=condition)


def choice(name, options, default, condition=None):
    return Parameter(name, "choice", default, options=options, condition=condition)


def subsets(kinds):
    return tuple(
        "+".join(members)
        for size in range(1, len(kinds) + 1)
        for members in itertools.combinations(kinds, size)
    )


def holds(spec, name, value):
    return name in spec and spec[name] == value


def enabled(spec, name):
    return name in spec and spec[name]


def equals(name, value):
    return lambda spec: holds(spec, name, value)


def among(name, values):
    return lambda spec: name in spec and spec[name] in values


def differs(name, value):
    return lambda spec: name in spec and spec[name] != value


def settings(registry, groups):
    started = time.perf_counter()
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError("unknown representation groups: " + ", ".join(unknown))
    nodes, extras = registry.declared, registry.extras
    parameters = [number("time/history", 1, 32, 16), number("time/context", 0, 3, 0)]
    parameters += [choice("graphs/join", ("separate", "single"), "separate")]
    parameters += [number("graphs/join_size", 2, 4096, 8, equals("graphs/join", "single"))]
    parameters += [flag("graphs/connect/" + kind, True, equals("graphs/join", "single")) for kind in nodes]
    parameters.append(
        real("graphs/connect_fraction", 0.01, 1.0, 1.0, equals("graphs/join", "single"), log=True)
    )
    parameters += [flag("nodes/untyped", False)] if len(nodes) > 1 else []
    parameters += [choice("relations/direction", DIRECTIONS, "inverse")] if registry.groups else []
    tasks = ("task",) if extras["events"] else ()
    if "coupling" in groups:
        parameters += [
            choice("coupling/scheme", ("ordinal", "lags", "window", "all"), "ordinal"),
            choice(
                "coupling/unit",
                ("observation", "step"),
                "observation",
                among("coupling/scheme", ("ordinal", "lags")),
            ),
            number("coupling/max_lag", 2, 64, 64, equals("coupling/scheme", "lags")),
            number(
                "coupling/lag_window",
                1,
                512,
                256,
                lambda spec: spec["coupling/scheme"] == "lags" and spec["coupling/unit"] == "observation",
            ),
            number("coupling/window", 2, 128, 64, equals("coupling/scheme", "window")),
            number("coupling/cap", 1, 128, 64, equals("coupling/scheme", "window")),
            number("coupling/periods", 0, 3, 0),
            number("coupling/period_window", 2, 128, 64, lambda spec: spec["coupling/periods"] > 0),
        ]
    if "events" in groups:
        kinds = subsets(("observation", "relation") + tasks)
        parameters += [
            choice("events/kinds", kinds, kinds[-1]),
            choice("events/link", ("next", "lags", "window"), "lags"),
            number("events/max_lag", 2, 16, 8, equals("events/link", "lags")),
            number("events/window", 1, 1024, 512),
            number("events/cap", 0, 64, 0),
            number("events/periods", 0, 3, 0),
            number("events/period_window", 2, 128, 64, lambda spec: spec["events/periods"] > 0),
        ]
    if "neighbors" in groups or "walks" in groups:
        kinds = subsets(("relation",) + tasks)
        parameters += [
            choice("neighbors/kinds", kinds, kinds[-1]),
            flag("neighbors/removals", True, lambda spec: "relation" in spec["neighbors/kinds"].split("+")),
        ]
    if "neighbors" in groups:
        parameters += [
            choice("neighbors/order", ("recent", "uniform", "decay"), "recent"),
            real("neighbors/decay", 0.001, 1.0, 0.1, equals("neighbors/order", "decay"), log=True),
        ]
    if "walks" in groups:
        parameters += [
            real("walks/bias", 0.0, 1.0, 0.0),
            number("walks/window", 1, 512, 256),
            number("walks/fanout", 1, 128, 64),
        ]
    if "mixed" in groups:
        parameters += [
            flag("mixed/changes", True),
            number("mixed/trace", 0, 8, 4),
            flag("mixed/bundles", True),
        ]
    steps = offered(registry)
    if not steps:
        raise ValueError("no choice of steps serves the graphs of the bank %s" % registry.name)
    parameters.append(choice("time/steps", steps, steps[0]))
    if extras["several"]:
        parameters.append(choice("time/several", SEVERAL.names(), None))
    parameters += [
        choice(sets.PARAMETER % kind, sets.offered(registry, kind), None) for kind in registry.schema.sets
    ]
    counted = Counter(parameter.name for parameter in parameters)
    twice = sorted(name for name, count in counted.items() if count > 1)
    if twice:
        raise ValueError("the parameters %s are stated twice" % ", ".join(twice))
    LOG.debug(
        "settings exit seconds=%.4f parameters=%d groups=%s",
        time.perf_counter() - started,
        len(parameters),
        "+".join(groups),
    )
    return tuple(parameters)


def bound_for(name, bounds):
    if name in bounds:
        return bounds[name]
    patterns = [key for key in bounds if "*" in key and fnmatch.fnmatchcase(name, key)]
    if not patterns:
        return None
    return bounds[max(patterns, key=len)]


def narrow(parameters, bounds):
    parameters = tuple(parameters)
    unmatched = [
        name
        for name in bounds
        if not any(fnmatch.fnmatchcase(parameter.name, name) for parameter in parameters)
    ]
    if unmatched:
        raise ValueError("bounds match no parameters: " + ", ".join(sorted(unmatched)))
    result = []
    for parameter in parameters:
        bound = bound_for(parameter.name, bounds)
        if bound is None:
            result.append(parameter)
            continue
        if parameter.kind in ("bool", "choice"):
            options = tuple(
                parameter.check(option) for option in (bound if isinstance(bound, list) else [bound])
            )
            if not options:
                raise ValueError("parameter bounds cannot be empty: " + parameter.name)
            default = parameter.default if parameter.default in options else None
            result.append(
                Parameter(
                    parameter.name, parameter.kind, default, options=options, condition=parameter.condition
                )
            )
            continue
        low, high = (bound["low"], bound["high"]) if isinstance(bound, dict) else (bound, bound)
        low, high = parameter.check(low), parameter.check(high)
        if low > high:
            raise ValueError("%s bound %r is empty" % (parameter.name, bound))
        result.append(
            Parameter(
                parameter.name,
                parameter.kind,
                min(max(parameter.default, low), high),
                low,
                high,
                parameter.log,
                condition=parameter.condition,
            )
        )
    return tuple(result)


def draw(parameters, choose):
    spec = {}
    for parameter in parameters:
        if parameter.active(spec):
            spec[parameter.name] = choose(parameter)
    return spec


def default(parameters):
    started = time.perf_counter()
    result = draw(parameters, lambda parameter: parameter.given({}))
    LOG.debug("default exit seconds=%.4f parameters=%d", time.perf_counter() - started, len(result))
    return result


def canonical(parameters, spec):
    started = time.perf_counter()

    def read(parameter):
        if parameter.name not in spec:
            raise ValueError("spec lacks the active parameter " + parameter.name)
        return parameter.check(spec[parameter.name])

    result = draw(parameters, read)
    extra = sorted(set(spec) - set(result))
    if extra:
        raise ValueError("spec sets inactive or unknown parameters: " + ", ".join(extra))
    LOG.debug("canonical exit seconds=%.4f parameters=%d", time.perf_counter() - started, len(result))
    return result


def configured(parameters, values):
    started = time.perf_counter()
    supplied = dict(values)
    result = draw(parameters, lambda parameter: parameter.given(supplied))
    extra = sorted(set(supplied) - set(result))
    if extra:
        raise ValueError("spec sets inactive or unknown parameters: " + ", ".join(extra))
    LOG.debug("configured exit seconds=%.4f parameters=%d", time.perf_counter() - started, len(result))
    return result


def materialized(spec):
    return {name: value for name, value in spec.items() if name not in PACKING}


def digest(spec):
    return sha256_json(spec)


def summary(parameters, spec):
    defaults = {parameter.name: parameter.default for parameter in parameters}
    return {name: value for name, value in spec.items() if value != defaults[name]}
