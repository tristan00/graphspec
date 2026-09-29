from collections import deque

import torch
from torch import nn

from graphspec.infra.timing import timed

from .layers import Asked, Head, References, mean
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
        self.register_buffer("seen", torch.as_tensor(registry.seen), persistent=False)
        self.task = nn.Linear(registry.features, hidden)
        self.asked = Asked(registry, hidden)
        self.references = References(registry, kinds * 2, kinds * 2)
        self.head = Head(kinds * 2 + hidden, spec, registry)

    def forward(self, b):
        kinds = b["global_affinity"].shape[-1]
        state = b["global_affinity"].new_zeros(b["entities"], kinds)
        overall = b["global_affinity"].new_zeros(b["predictions"], kinds)
        owner, entity = b["member_row"], b["member_entity"]
        place = torch.zeros_like(b["row_kind"])
        buffer, values, order = deque(maxlen=self.buffer_size), [], []
        for time_step in range(b["time_steps"]):
            step, chosen = b["frame_rows"][time_step], b["frame_members"][time_step]
            place[step] = torch.arange(len(step), device=place.device)
            profile = b["member_affinity"][chosen]
            x = profile.log1p()
            known = torch.stack(tuple(buffer)).mean(0) if buffer else x.new_zeros(len(step), kinds)
            overall[step] = known
            around = known[place[owner[chosen]]]
            held, inverse = torch.unique(entity[chosen], return_inverse=True)
            keep = (self.xh(x) + self.hh(state[entity[chosen]]) + self.bh).sigmoid()
            hidden = keep * state[entity[chosen]] + (1 - keep) * x
            select = (self.xs(x) + self.hs(hidden) + self.gs(around) + self.bs).sigmoid()
            value = torch.cat((select * hidden + (1 - select) * x, around), -1)
            if self.has_periods:
                rows = owner[chosen]
                previous = torch.cat(
                    (
                        b["member_end_affinity"][chosen].log1p(),
                        b["end_global_affinity"][rows].log1p(),
                    ),
                    -1,
                )
                value = self.periods(value, previous, b["end_gaps"][rows], b["end_mask"][rows])
            values.append(value)
            order.append(chosen)
            state = state.index_copy(0, held, mean(hidden, inverse, len(held)))
            row = mean(profile, place[owner[chosen]], len(step))
            code = self.seen[b["row_kind"][step]]
            events = torch.nn.functional.one_hot(code.clamp_min(0), kinds) * (code >= 0)[:, None]
            buffer.append((row + events).log1p())
        members = torch.cat(values)[torch.argsort(torch.cat(order))]
        whole = torch.cat((b["global_affinity"].log1p(), overall), -1)[b["whole_row"]]
        features = torch.cat(
            (self.references(b, members, whole), self.asked(self.task(b["task"]), b)),
            -1,
        )
        return self.head(features[b["selected"]], b)
