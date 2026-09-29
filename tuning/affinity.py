from collections import deque

import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Head
from .turns import EndContext, enabled


class AffinityNet(nn.Module):
    @timed("affinity.affinitynet")
    def __init__(self, cfg, registry, categories):
        super().__init__()
        spec = cfg["model"]
        hidden = spec["hidden"]
        self.buffer_size = spec["affinity/buffer"]
        self.has_turns = enabled(cfg)
        if self.has_turns:
            self.turns = EndContext(categories * 2)
        self.xh, self.hh, self.xs, self.hs, self.gs = (
            nn.Linear(categories, categories, bias=False) for _ in range(5)
        )
        self.bh, self.bs = nn.Parameter(torch.zeros(categories)), nn.Parameter(torch.zeros(categories))
        self.query = nn.Linear(registry.features, hidden)
        self.reward = Head(categories * 2 + hidden, spec, registry)

    def forward(self, b):
        state = b["affinity"].new_zeros(b["entities"], b["affinity"].shape[-1])
        buffer, output, order = deque(maxlen=self.buffer_size), [], []
        for time_step in range(b["time_steps"]):
            step = b["frame_queries"][time_step]
            actor = b["actor"][step]
            x = b["affinity"][step].log1p()
            global_state = torch.stack(tuple(buffer)).mean(0) if buffer else torch.zeros_like(x)
            keep = (self.xh(x) + self.hh(state[actor]) + self.bh).sigmoid()
            hidden = keep * state[actor] + (1 - keep) * x
            select = (self.xs(x) + self.hs(hidden) + self.gs(global_state) + self.bs).sigmoid()
            affinity = select * hidden + (1 - select) * x
            pair = torch.cat((affinity, global_state), -1)
            if self.has_turns:
                previous = torch.cat(
                    (b["end_affinity"][step].log1p(), b["end_global_affinity"][step].log1p()), -1
                )
                pair = self.turns(pair, previous, b["end_gaps"][step], b["end_mask"][step])
            output.append(torch.cat((pair, self.query(b["query"][step])), -1))
            order.append(step)
            state = state.index_copy(0, actor, hidden)
            events = b["events"]
            choice = events[(events[:, 2] == (b["start"] + time_step) * 2 + 1) & (events[:, 3] == 2)]
            category = choice[:, 6]
            profile = b["affinity"][step].clone()
            profile[torch.arange(len(step), device=state.device), category] += 1
            buffer.append(profile.log1p())
        return self.reward(torch.cat(output)[torch.argsort(torch.cat(order))][b["selected"]], b)
