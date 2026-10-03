import logging
import time

import numpy as np
import torch
from torch import nn

from graphspec.infra.timing import timed
from graphspec.infra.workspace import gathered

LOG = logging.getLogger(__name__)


def lookback(cfg):
    return cfg["graph"]["time/context"]


def enabled(cfg):
    return lookback(cfg) > 0


def blocks(bank, original, cfg):
    if not enabled(cfg):
        return original
    started = time.perf_counter()
    output, starts = [], {}
    context = lookback(cfg)
    for graph, start, stop, target, partition, first, last in original:
        if graph not in starts:
            arrays = bank.arrays(graph, cfg)
            starts[graph] = gathered(arrays["period_starts"], np.asarray(arrays["period_membership"]) - context, 0)
        output.append(
            (
                graph,
                int(starts[graph][start]),
                stop,
                target,
                partition,
                first,
                last,
            )
        )
    LOG.info("period blocks exit blocks=%d seconds=%.4f", len(output), time.perf_counter() - started)
    return output


class EndContext(nn.Module):
    @timed("periods.EndContext.__init__")
    def __init__(self, width):
        super().__init__()
        self.value = nn.Linear(width, width, bias=False)
        self.time = nn.Linear(3, width)
        self.gate = nn.Linear(width * 2, width)

    def forward(self, current, previous, gaps, mask):
        weights = mask.to(current.dtype)
        value = self.value(previous) + self.time(gaps.log1p())
        context = (value * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]
        update = self.gate(torch.cat((current, context), -1)).sigmoid() * context
        return current + update * mask.any(1)[:, None]


def gather_end(sequence, entities, steps, time_steps):
    return sequence[steps.clamp_min(0) % time_steps, entities[:, None]]
