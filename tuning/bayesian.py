from functools import partial

import torch

from graphspec.infra.bayesian import Search

from .space import propose

DEVICE = torch.device("cuda")


def build(x, losses, complete, seed):
    import botorch

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Search(x, losses, complete, DEVICE)
    model.sampler, model.phase = "BoTorch", "botorch"
    model.settings = dict(model="MAP-SAAS GP", device=DEVICE.type, botorch_version=botorch.__version__)
    return model


suggest = partial(propose, build)
