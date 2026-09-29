from collections import deque

import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Head
from .periods import EndContext, enabled


class AffinityNet(nn.Module):
    @timed("affinity.affinitynet")
    def __init__(self, cfg, registry):
        super().__init__()
        spec = cfg["model"]
        hidden, kinds = spec["hidden"], len(registry.events)
        self.buffer_size = spec["affinity/buffer"]
        self.has_periods = enabled(cfg)
        if self.has_periods:
            self.periods = EndContext(kinds * 2)
        self.xh, self.hh, self.xs, self.hs, self.gs = (nn.Linear(kinds, kinds, bias=False) for _ in range(5))
        self.bh, self.bs = nn.Parameter(torch.zeros(kinds)), nn.Parameter(torch.zeros(kinds))
        self.task = nn.Linear(registry.features, hidden)
        self.head = Head(kinds * 2 + hidden, spec, registry)

    def forward(self, b):
        state = b["affinity"].new_zeros(b["entities"], b["affinity"].shape[-1])
        buffer, output, order = deque(maxlen=self.buffer_size), [], []
        for time_step in range(b["time_steps"]):
            step = b["frame_rows"][time_step]
            first = b["first"][step]
            x = b["affinity"][step].log1p()
            global_state = torch.stack(tuple(buffer)).mean(0) if buffer else torch.zeros_like(x)
            keep = (self.xh(x) + self.hh(state[first]) + self.bh).sigmoid()
            hidden = keep * state[first] + (1 - keep) * x
            select = (self.xs(x) + self.hs(hidden) + self.gs(global_state) + self.bs).sigmoid()
            affinity = select * hidden + (1 - select) * x
            pair = torch.cat((affinity, global_state), -1)
            if self.has_periods:
                previous = torch.cat(
                    (b["end_affinity"][step].log1p(), b["end_global_affinity"][step].log1p()), -1
                )
                pair = self.periods(pair, previous, b["end_gaps"][step], b["end_mask"][step])
            output.append(torch.cat((pair, self.task(b["task"][step])), -1))
            order.append(step)
            state = state.index_copy(0, first, hidden)
            events = b["events"]
            choice = events[(events[:, 2] == (b["start"] + time_step) * 2 + 1) & (events[:, 3] == 2)]
            kind = choice[:, 6]
            profile = b["affinity"][step].clone()
            profile[torch.arange(len(step), device=state.device), kind] += 1
            buffer.append(profile.log1p())
        return self.head(torch.cat(output)[torch.argsort(torch.cat(order))][b["selected"]], b)
