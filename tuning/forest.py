import time
from functools import partial

from graphspec.infra.forest import COMPLETION_WEIGHT, Forest
from graphspec.infra.log import emit

from .space import propose


def build(x, losses, complete, seed, device):
    import sklearn

    started = time.perf_counter()
    model = Forest(x, losses, complete, seed)
    model.sampler, model.phase = "SMAC", "forest"
    model.settings = dict(
        model="random forest",
        trees=model.trees,
        transform="log",
        completion_weight=COMPLETION_WEIGHT,
        sklearn_version=sklearn.__version__,
    )
    emit("forest_build", trees=model.trees, seconds=time.perf_counter() - started)
    return model


suggest = partial(propose, build)
