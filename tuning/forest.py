from functools import partial

from graphspec.infra.forest import COMPLETION_WEIGHT, Forest

from .space import propose


def build(x, losses, complete, seed):
    import sklearn

    model = Forest(x, losses, complete, seed)
    model.sampler, model.phase = "SMAC", "forest"
    model.settings = dict(
        model="random forest",
        trees=model.trees,
        transform="log",
        completion_weight=COMPLETION_WEIGHT,
        sklearn_version=sklearn.__version__,
    )
    return model


suggest = partial(propose, build)
