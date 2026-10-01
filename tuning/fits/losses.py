import logging
import time

from graphspec.infra.timing import timed
from graphspec.registries import Names, ship
from tuning.families.heads import FIELD
from tuning.measures.measures import MEASURES

LOG = logging.getLogger(__name__)
LOSSES = Names("loss")
POINTED = ("target", "weight", "series", "aim", "point_row")


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
def readers(registry, kind):
    return tuple(name for name in registry.names() if kind in registry.named(name).reads)


def chosen(batch, points):
    held = {key: value[points] for key, value in batch.items() if key in POINTED or key.startswith(FIELD)}
    held["tasks"] = batch["tasks"]
    return held


class Summed:
    def __init__(self, stated, kinds):
        started = time.perf_counter()
        lacking = [kind for kind in kinds if kind not in stated]
        if lacking:
            raise ValueError(
                "the statement names no loss for the kind of target %s of the bank; the losses that read it are %s"
                % (lacking[0], ", ".join(readers(LOSSES, lacking[0])) or "none")
            )
        strangers = sorted(kind for kind in stated if kind not in kinds)
        if strangers:
            raise ValueError(
                "the statement names a loss for the kind of target %s, which the bank does not hold; it holds %s"
                % (strangers[0], ", ".join(kinds))
            )
        for kind in kinds:
            if kind not in LOSSES.named(stated[kind]).reads:
                raise ValueError(
                    "the statement names the loss %s for the kind of target %s, which it does not read; the losses "
                    "that read it are %s" % (stated[kind], kind, ", ".join(readers(LOSSES, kind)) or "none")
                )
        self.losses = [LOSSES.named(stated[kind]) for kind in kinds]
        LOG.debug("summed exit kinds=%d seconds=%.4f", len(kinds), time.perf_counter() - started)

    def __call__(self, predictions, batch, bank):
        import torch

        values = []
        for code, loss in enumerate(self.losses):
            points = (batch["aim"] == code).nonzero().flatten()
            if len(points):
                values.append(loss(predictions[points], chosen(batch, points), bank))
        return torch.stack(values).sum()


def admitted(kinds, loss, measures):
    started = time.perf_counter()
    result = Summed(loss, kinds)
    for name in measures:
        entry = MEASURES.named(name)
        if not set(entry.reads) & set(kinds):
            raise ValueError(
                "the statement names the measure %s, which reads targets of the kinds %s and no kind of target of "
                "the bank, which holds %s" % (name, ", ".join(entry.reads), ", ".join(kinds))
            )
    unread = [kind for kind in kinds if not any(kind in MEASURES.named(name).reads for name in measures)]
    if unread:
        raise ValueError(
            "no measure the statement names reads the kind of target %s of the bank; the measures that read it are %s"
            % (unread[0], ", ".join(readers(MEASURES, unread[0])) or "none")
        )
    LOG.debug(
        "admitted exit kinds=%d measures=%d seconds=%.4f", len(kinds), len(measures), time.perf_counter() - started
    )
    return result


LOSSES.register("squared", Squared())
LOSSES.register("balanced", Balanced())
ship([LOSSES])
