import logging
import time

import numpy as np
import torch
from torch import nn

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
            starts[graph] = arrays["period_starts"][np.maximum(0, arrays["period_membership"] - context)]
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


def gather_end(sequence, entities, b):
    times = b["end_steps"].clamp_min(0) % b["time_steps"]
    return sequence[times, entities[:, None]]
