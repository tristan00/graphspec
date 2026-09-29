import time
from functools import partial

import torch

from graphspec.infra import compute
from graphspec.infra.bayesian import Search
from graphspec.infra.log import emit

from .space import propose


def build(x, losses, complete, seed, device):
    import botorch

    started = time.perf_counter()
    compute.require(device)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Search(x, losses, complete, device)
    model.sampler, model.phase = "BoTorch", "botorch"
    model.settings = dict(model="MAP-SAAS GP", device=device, botorch_version=botorch.__version__)
    emit("botorch_build", device=device, seconds=time.perf_counter() - started)
    return model


suggest = partial(propose, build)
