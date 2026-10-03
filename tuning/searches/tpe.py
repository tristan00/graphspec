import warnings
from importlib.metadata import version

import numpy as np

from graphspec.infra.timing import timed

VERSION = version("optuna")


@timed("tpe.categorical")
def categorical(options):
    from optuna.distributions import CategoricalDistribution

    return CategoricalDistribution(list(options))


@timed("tpe.integer")
def integer(low, high):
    from optuna.distributions import IntDistribution

    return IntDistribution(low, high)


@timed("tpe.real")
def real(low, high, log=False):
    from optuna.distributions import FloatDistribution

    return FloatDistribution(low, high, log=log)


@timed("tpe.propose")
def propose(trial, name, domain):
    from optuna.distributions import CategoricalDistribution, IntDistribution

    if isinstance(domain, CategoricalDistribution):
        return trial.suggest_categorical(name, domain.choices)
    if isinstance(domain, IntDistribution):
        return trial.suggest_int(name, domain.low, domain.high, step=domain.step, log=domain.log)
    return trial.suggest_float(name, domain.low, domain.high, log=domain.log)


@timed("tpe.draw")
def draw(domain, rng):
    from optuna.distributions import CategoricalDistribution, IntDistribution

    if isinstance(domain, CategoricalDistribution):
        return domain.choices[int(rng.integers(len(domain.choices)))]
    if isinstance(domain, IntDistribution):
        return int(rng.integers(domain.low, domain.high + 1))
    if domain.log:
        return float(np.exp(rng.uniform(np.log(domain.low), np.log(domain.high))))
    return float(rng.uniform(domain.low, domain.high))


@timed("search.record")
def record(value, params, distributions, **attrs):
    import optuna

    return optuna.trial.create_trial(
        value=value, params=params, distributions=distributions, user_attrs=attrs
    )


@timed("search.study")
def study(seed, settings, records):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        settings = dict(settings)
        good_set = settings.pop("good_set")
        sampler = optuna.samplers.TPESampler(seed=seed, gamma=lambda n: min(n, good_set), **settings)
    result = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.NopPruner())
    for entry in records:
        result.add_trial(entry)
    return result
