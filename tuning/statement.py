import dataclasses
import logging
import time
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)
VERSION = 1


def current(value):
    return type(value) is int and value == VERSION


def device(value):
    return type(value) is str and value.split(":")[0] in ("cpu", "cuda")


def whole(value):
    return type(value) is int and value >= 0


def positive(value):
    return type(value) is int and value >= 1


def held(check, meaning):
    return field(metadata=dict(check=check, meaning=meaning))


@dataclass(frozen=True)
class Statement:
    version: int = held(current, "the version of the statement, %d" % VERSION)
    device: str = held(device, "cpu, cuda or cuda:N")
    seed: int = held(whole, "a nonnegative integer")
    seconds: int = held(positive, "a positive integer")
    preparation_seconds: int = held(positive, "a positive integer")
    workers: int = held(positive, "a positive integer")
    variant_cache_bytes: int = held(positive, "a positive integer")

    def __post_init__(self):
        started = time.perf_counter()
        for entry in dataclasses.fields(self):
            value = getattr(self, entry.name)
            if not entry.metadata["check"](value):
                raise ValueError(
                    "the statement's %s is %r and must be %s" % (entry.name, value, entry.metadata["meaning"])
                )
        LOG.debug("statement checked exit seconds=%.4f", time.perf_counter() - started)

    def mapping(self):
        started = time.perf_counter()
        result = dataclasses.asdict(self)
        LOG.debug("statement mapping exit seconds=%.4f", time.perf_counter() - started)
        return result

    def replace(self, **changes):
        started = time.perf_counter()
        result = state(dict(self.mapping(), **changes))
        LOG.debug("statement replace exit seconds=%.4f", time.perf_counter() - started)
        return result


def names():
    return tuple(entry.name for entry in dataclasses.fields(Statement))


def state(mapping):
    started = time.perf_counter()
    absent = [name for name in names() if name not in mapping]
    if absent:
        raise KeyError("the statement lacks " + ", ".join(absent))
    unknown = sorted(set(mapping) - set(names()))
    if unknown:
        raise KeyError("the statement holds fields it does not know: " + ", ".join(unknown))
    result = Statement(**mapping)
    LOG.debug("state exit fields=%d seconds=%.4f", len(mapping), time.perf_counter() - started)
    return result
