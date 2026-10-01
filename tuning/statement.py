import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field

from graphspec.infra.log import emit

LOG = logging.getLogger(__name__)
VERSION = 3
SEEDS = 2**32


def current(value):
    return type(value) is int and value == VERSION


def device(value):
    import torch

    started = time.perf_counter()
    try:
        parsed = torch.device(value) if type(value) is str else None
    except RuntimeError:
        parsed = None
    result = parsed is not None and (
        parsed.type == "cpu" or (parsed.type == "cuda" and (parsed.index or 0) < torch.cuda.device_count())
    )
    LOG.debug("statement device exit device=%r seconds=%.4f", value, time.perf_counter() - started)
    return result


def seed(value):
    return type(value) is int and 0 <= value < SEEDS


def positive(value):
    return type(value) is int and value >= 1


def processors(value):
    return type(value) is int and 1 <= value <= os.cpu_count()


def named(value):
    return (
        isinstance(value, list)
        and len(set(value)) == len(value)
        and all(type(name) is str and name for name in value)
    )


def splits(value):
    return named(value) and len(value) > 0


def word(value):
    return type(value) is str and len(value) > 0


def mapped(value):
    started = time.perf_counter()
    result = (
        isinstance(value, dict) and len(value) > 0 and all(word(kind) and word(name) for kind, name in value.items())
    )
    LOG.debug("statement mapped exit seconds=%.4f", time.perf_counter() - started)
    return result


def seeing(value):
    from graphspec.views.visible import SEES

    started = time.perf_counter()
    result = isinstance(value, dict) and all(
        type(split) is str and named(rules) and set(rules) <= set(SEES.names())
        for split, rules in value.items()
    )
    LOG.debug("statement seeing exit seconds=%.4f", time.perf_counter() - started)
    return result


def held(check, meaning):
    started = time.perf_counter()
    result = field(metadata=dict(check=check, meaning=meaning))
    LOG.debug("held exit check=%s seconds=%.4f", check.__name__, time.perf_counter() - started)
    return result


@dataclass(frozen=True, init=False)
class Statement:
    version: int = held(current, "the version of the statement, %d" % VERSION)
    device: str = held(device, "cpu, cuda or cuda:N, where N is below the number of GPUs present")
    seed: int = held(seed, "an integer from 0 to %d" % (SEEDS - 1))
    seconds: int = held(positive, "a positive integer")
    preparation_seconds: int = held(positive, "a positive integer")
    workers: int = held(processors, "an integer from 1 to the number of logical processors")
    variant_cache_bytes: int = held(positive, "a positive integer")
    batch_sequences: int = held(positive, "a positive integer")
    trained: list = held(splits, "a list of the distinct names of the splits trained on, at least one")
    scored: list = held(splits, "a list of the distinct names of the splits scored on, at least one")
    sees: dict = held(
        seeing,
        "a mapping from every split trained on or scored on to the list of the registered rules "
        "of what its tasks may see",
    )
    loss: dict = held(
        mapped,
        "a mapping from every kind of target of the bank to the name of a registered loss that reads it, whose "
        "losses a fit sums",
    )
    measures: list = held(
        splits,
        "a list of the distinct names of the registered measures computed, at least one, each reading a kind of "
        "target of the bank, and every kind of target of the bank read by one of them",
    )

    def __init__(self, **mapping):
        started = time.perf_counter()
        absent = [name for name in names() if name not in mapping]
        if absent:
            raise KeyError("the statement lacks " + ", ".join(absent))
        unknown = sorted(set(mapping) - set(names()))
        if unknown:
            raise KeyError("the statement holds fields it does not know: " + ", ".join(unknown))
        for entry in dataclasses.fields(self):
            value = mapping[entry.name]
            if not entry.metadata["check"](value):
                raise ValueError(
                    "the statement's %s is %r and must be %s" % (entry.name, value, entry.metadata["meaning"])
                )
            object.__setattr__(self, entry.name, value)
        both = sorted(set(self.trained) & set(self.scored))
        if both:
            raise ValueError("the statement trains on and scores on the same splits: %s" % ", ".join(both))
        if sorted(self.sees) != sorted(self.trained + self.scored):
            raise ValueError(
                "the statement's sees names the splits %s and must name the splits trained on and scored on, %s"
                % (", ".join(sorted(self.sees)) or "none", ", ".join(sorted(self.trained + self.scored)))
            )
        LOG.debug("statement exit fields=%d seconds=%.4f", len(mapping), time.perf_counter() - started)

    def mapping(self):
        started = time.perf_counter()
        result = dataclasses.asdict(self)
        LOG.debug("statement mapping exit seconds=%.4f", time.perf_counter() - started)
        return result

    def splits(self):
        started = time.perf_counter()
        result = dict(
            trained=list(self.trained),
            scored=list(self.scored),
            sees={name: list(rules) for name, rules in self.sees.items()},
        )
        LOG.debug("statement splits exit seconds=%.4f", time.perf_counter() - started)
        return result

    def replace(self, **changes):
        started = time.perf_counter()
        result = Statement(**dict(self.mapping(), **changes))
        LOG.debug("statement replace exit seconds=%.4f", time.perf_counter() - started)
        return result


def names():
    return tuple(entry.name for entry in dataclasses.fields(Statement))


def state(mapping):
    started = time.perf_counter()
    if not isinstance(mapping, dict):
        raise TypeError("the statement is read from a mapping of its fields, found %r" % (mapping,))
    result = Statement(**mapping)
    LOG.debug("state exit fields=%d seconds=%.4f", len(mapping), time.perf_counter() - started)
    return result


def stated(value):
    started = time.perf_counter()
    if type(value) is not Statement:
        raise TypeError("a statement built by tuning.statement.state is required, found %r" % (value,))
    LOG.debug("stated exit seconds=%.4f", time.perf_counter() - started)
    return value


def seen(given):
    started = time.perf_counter()
    result = None
    if given is not None:
        result = {}
        for item in given:
            name, stated, rules = item.partition("=")
            if not stated or name in result:
                raise ValueError(
                    "what a split may see is stated once for each split as split=rule,rule; found %r" % item
                )
            result[name] = [rule for rule in rules.split(",") if rule]
    emit("sees_parsed", seconds=time.perf_counter() - started)
    return result


def paired(given):
    started = time.perf_counter()
    result = None
    if given is not None:
        result = {}
        for item in given:
            kind, stated, name = item.partition("=")
            if not stated or not kind or not name or kind in result:
                raise ValueError(
                    "the loss of each kind of target is stated once for each kind as kind=loss; found %r" % item
                )
            result[kind] = name
    emit("losses_parsed", seconds=time.perf_counter() - started)
    return result
