import logging
import time

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship
from tuning.measures.measures import MEASURES

LOG = logging.getLogger(__name__)
LOSSES = Names("loss")


def standardised(predictions, batch, bank):
    center, scale = predictions.new_tensor(bank.center), predictions.new_tensor(bank.scale)
    return (batch["target"] - center[batch["series"]]) / scale[batch["series"]]


class Squared:
    reads = ("number", "path")

    def __call__(self, predictions, batch, bank):
        error = (predictions - standardised(predictions, batch, bank)).square()
        return (batch["weight"] * error).sum() / batch["weight"].sum()


class Balanced:
    reads = ("number", "path")

    def __call__(self, predictions, batch, bank):
        import torch

        error = (predictions - standardised(predictions, batch, bank)).square()
        count, tasks = len(bank.center), batch["tasks"]
        group = batch["point_row"] * count + batch["series"]
        total = error.new_zeros(tasks * count).index_add_(0, group, error)
        points = error.new_zeros(tasks * count).index_add_(0, group, error.new_ones(len(error)))
        weight = error.new_zeros(tasks * count).index_add_(0, group, batch["weight"]) / points.clamp_min(1)
        kind = torch.arange(tasks * count, device=error.device) % count
        series_total = error.new_zeros(count).index_add_(0, kind, weight * total / points.clamp_min(1))
        series_tasks = error.new_zeros(count).index_add_(0, kind, weight)
        return (series_total / series_tasks.clamp_min(1e-12)).sum() / (series_tasks > 0).sum()


@timed("losses.readers")
def readers(registry, kinds):
    return tuple(name for name in registry.names() if set(kinds) <= set(registry.named(name).reads))


def admitted(kinds, loss, measures):
    started = time.perf_counter()
    named = [("loss", loss, LOSSES)] + [("measure", name, MEASURES) for name in measures]
    for sort, name, registry in named:
        entry = registry.named(name)
        unread = [kind for kind in kinds if kind not in entry.reads]
        if unread:
            raise ValueError(
                "the statement names the %s %s, which reads targets of the kinds %s and cannot read the kind of "
                "target %s of the bank; those that read every kind of target of the bank are %s"
                % (sort, name, ", ".join(entry.reads), unread[0], ", ".join(readers(registry, kinds)) or "none")
            )
    LOG.debug(
        "admitted exit loss=%s measures=%d seconds=%.4f", loss, len(measures), time.perf_counter() - started
    )


LOSSES.register("squared", Squared())
LOSSES.register("balanced", Balanced())
ship([LOSSES])
