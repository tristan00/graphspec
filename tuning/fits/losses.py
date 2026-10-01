import logging
import time

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
    reads = ("path",)

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


def admitted(kinds, loss, measures):
    started = time.perf_counter()
    named = [("loss", loss, LOSSES.named(loss))] + [("measure", name, MEASURES.named(name)) for name in measures]
    for sort, name, entry in named:
        unread = [kind for kind in kinds if kind not in entry.reads]
        if unread:
            raise ValueError(
                "the statement names the %s %s, which reads targets of the kinds %s and cannot read the kind of "
                "target %s of the bank" % (sort, name, ", ".join(entry.reads), unread[0])
            )
    LOG.debug(
        "admitted exit loss=%s measures=%d seconds=%.4f", loss, len(measures), time.perf_counter() - started
    )


LOSSES.register("squared", Squared())
LOSSES.register("balanced", Balanced())
ship([LOSSES])
