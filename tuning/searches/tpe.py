import importlib
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np

from graphspec.infra.timing import timed

VERSION = version("optuna")


@timed("tpe.imported")
def imported(name):
    if name in sys.modules:
        return sys.modules[name]
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(importlib.import_module, name).result()


@timed("tpe.categorical")
def categorical(options):
    return imported("optuna").distributions.CategoricalDistribution(list(options))


@timed("tpe.integer")
def integer(low, high):
    return imported("optuna").distributions.IntDistribution(low, high)


@timed("tpe.real")
def real(low, high, log=False):
    return imported("optuna").distributions.FloatDistribution(low, high, log=log)


@timed("tpe.propose")
def propose(trial, name, domain):
    distributions = imported("optuna").distributions
    if isinstance(domain, distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, domain.choices)
    if isinstance(domain, distributions.IntDistribution):
        return trial.suggest_int(name, domain.low, domain.high, step=domain.step, log=domain.log)
    return trial.suggest_float(name, domain.low, domain.high, log=domain.log)


@timed("tpe.draw")
def draw(domain, rng):
    distributions = imported("optuna").distributions
    if isinstance(domain, distributions.CategoricalDistribution):
        return domain.choices[int(rng.integers(len(domain.choices)))]
    if isinstance(domain, distributions.IntDistribution):
        return int(rng.integers(domain.low, domain.high + 1))
    if domain.log:
        return float(np.exp(rng.uniform(np.log(domain.low), np.log(domain.high))))
    return float(rng.uniform(domain.low, domain.high))


@timed("search.record")
def record(value, params, distributions, **attrs):
    return imported("optuna").trial.create_trial(value=value, params=params, distributions=distributions, user_attrs=attrs)


@timed("search.study")
def study(seed, settings, records):
    optuna = imported("optuna")
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
