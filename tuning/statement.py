import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)
VERSION = 1
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
        parsed.type == "cpu"
        or (parsed.type == "cuda" and (parsed.index or 0) < torch.cuda.device_count())
    )
    LOG.debug("statement device exit device=%r seconds=%.4f", value, time.perf_counter() - started)
    return result


def seed(value):
    return type(value) is int and 0 <= value < SEEDS


def positive(value):
    return type(value) is int and value >= 1


def processors(value):
    return type(value) is int and 1 <= value <= os.cpu_count()


def held(check, meaning):
    return field(metadata=dict(check=check, meaning=meaning))


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
        LOG.debug("statement exit fields=%d seconds=%.4f", len(mapping), time.perf_counter() - started)

    def mapping(self):
        started = time.perf_counter()
        result = dataclasses.asdict(self)
        LOG.debug("statement mapping exit seconds=%.4f", time.perf_counter() - started)
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
