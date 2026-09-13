import fnmatch
import math
import time

import numpy as np

from graphspec import spec as specs
from graph_tuning_infra import search
from graph_tuning_infra.files import read_json, sha256_file, write_json
from graph_tuning_infra.log import emit
from .model_spec import TRAINING, describe, groups, space as model_space


SETTINGS = dict(n_startup_trials=10, n_ei_candidates=64, multivariate=True, group=True)
FAILURE_POLICY = dict(version=1, outcomes=['timeout', 'failed'], objective='positive_infinity',
    interpretation='Failure ranks worse than every finite completed score; no prediction metric is assigned.')


def distribution(parameter):
    if parameter.kind in ('bool', 'choice'):
        return search.categorical(parameter.options)
    if parameter.kind == 'int':
        return search.integer(parameter.low, parameter.high)
    return search.real(parameter.low, parameter.high, log=parameter.log)


def graph_space(registry, model, bounds):
    return specs.narrow(specs.space(registry, groups(model)), bounds)


def spaces(registry, bounds):
    return specs.narrow(model_space(registry), bounds), specs.narrow(TRAINING, bounds)


def check_bounds(registry, bounds):
    known = {parameter.name for parameter in model_space(registry) + TRAINING + specs.space(registry, specs.GROUPS)}
    unknown = sorted(key for key in bounds if not (key in known if '*' not in key else any(fnmatch.fnmatchcase(name, key) for name in known)))
    if unknown:
        raise ValueError('bounds name unknown parameters: ' + ', '.join(unknown))
    specs.narrow(model_space(registry) + TRAINING + specs.space(registry, specs.GROUPS), bounds)


def training(config):
    return {parameter.name: config[parameter.name] for parameter in TRAINING}


def observation(registry, config, value, bounds, **attrs):
    params, distributions = {}, {}
    models, trainings = spaces(registry, bounds)

    def enter(prefix, parameters, spec):
        lookup = {parameter.name: parameter for parameter in parameters}
        for name, setting in specs.canonical(parameters, spec).items():
            params[prefix + name], distributions[prefix + name] = setting, distribution(lookup[name])

    enter('model/', models, config['model'])
    enter('', trainings, training(config))
    enter('graph/', graph_space(registry, config['model'], bounds), config['graph'])
    return search.record(value, params, distributions, config=config, **attrs)


def metadata(registry, config, seed, training_seed, bounds, **extra):
    model, graph = spaces(registry, bounds)[0], graph_space(registry, config['model'], bounds)
    return dict(extra, model_digest=specs.digest(config['model']), description=describe(config['model']),
                model_summary=specs.summary(model, config['model']), model_space=[parameter.declaration() for parameter in model],
                graph_digest=specs.digest(config['graph']), graph_summary=specs.summary(graph, config['graph']),
                graph_space=[parameter.declaration() for parameter in graph], selector_seed=seed, training_seed=training_seed)


def configuration(registry, choose, training_seed, bounds):
    models, trainings = spaces(registry, bounds)
    model = specs.draw(models, lambda parameter: choose('model/', parameter))
    config = dict(seed=training_seed, model=model, **specs.draw(trainings, lambda parameter: choose('', parameter)))
    config['graph'] = specs.draw(graph_space(registry, model, bounds), lambda parameter: choose('graph/', parameter))
    return config


def suggest(registry, observations, failures, seed, training_seed, bounds):
    started = time.perf_counter()
    records = [observation(registry, entry['config'], entry['raw_val_mse'], bounds) for entry in observations]
    records += [observation(registry, entry['config'], math.inf, bounds, outcome=entry['status']) for entry in failures]
    study = search.study(seed, SETTINGS, records)
    trial = study.ask()
    config = configuration(registry, lambda prefix, parameter: search.propose(trial, prefix + parameter.name, distribution(parameter)), training_seed, bounds)
    total = len(observations) + len(failures)
    return config, metadata(registry, config, seed, training_seed, bounds, sampler='TPESampler', settings=SETTINGS, params=trial.params,
                            failure_policy=FAILURE_POLICY, observations=len(observations), failed_observations=len(failures),
                            optuna_version=search.VERSION, phase='tpe' if total >= SETTINGS['n_startup_trials'] else 'random_startup',
                            seconds=time.perf_counter() - started)


def random_suggest(registry, seed, training_seed, bounds):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    config = configuration(registry, lambda prefix, parameter: parameter.uniform(rng), training_seed, bounds)
    return config, metadata(registry, config, seed, training_seed, bounds, sampler='numpy.random.Generator', phase='random', seconds=time.perf_counter() - started)


def load_observations(directory, plan):
    observations, history = [], []
    for item in plan['trials']:
        path = directory / item['directory'] / 'result.json'
        if not path.exists():
            continue
        result = read_json(path)
        valid = (result['status'] == 'complete'
            and result['supervision_sha256'] == plan['supervision_sha256']
            and result['training_coverage_complete']
            and result['train_unique_rows'] == result['train_available_rows'] == plan['train_rows']
            and result['holdout_rows'] == plan['holdout_rows']
            and result['training_examples_seen'] == result['epochs'] * plan['train_rows']
            and math.isfinite(result['raw_val_mse']))
        if not valid:
            raise ValueError('incompatible Bayesian observation: ' + str(path))
        observations.append(result)
        history.append(dict(trial=item['directory'], raw_val_mse=result['raw_val_mse'], result_sha256=sha256_file(path)))
    return observations, history


def load_failures(directory, plan):
    failures, history = [], []
    for item in plan['trials']:
        trial = directory / item['directory']
        if (trial / 'result.json').exists():
            continue
        path = next((trial / name for name in ('timeout.json', 'failure.json') if (trial / name).exists()), None)
        if path is None:
            continue
        config_path = trial / 'config.json'
        config = read_json(config_path)
        if specs.digest(config['model']) != item['model']:
            raise ValueError('failure configuration differs from its saved plan: ' + str(path))
        status = 'timeout' if path.name == 'timeout.json' else 'failed'
        failures.append(dict(config=config, status=status))
        history.append(dict(trial=item['directory'], status=status, record=path.name,
            failure_sha256=sha256_file(path), config_sha256=sha256_file(config_path)))
    return failures, history


def append_trial(directory, plan, registry):
    started = time.perf_counter()
    observations, history = load_observations(directory, plan)
    index = len(plan['trials'])
    rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], index, 1]))
    selector_seed, training_seed = (int(v) for v in rng.integers(1, 2**31, size=2))
    selector = 'random' if plan['selector'] == 'random' or index < plan['random_trials'] else 'tpe'
    bounds = plan['bounds']
    if selector == 'random':
        config, selection = random_suggest(registry, selector_seed, training_seed, bounds)
    else:
        failures, failure_history = load_failures(directory, plan)
        config, selection = suggest(registry, observations, failures, selector_seed, training_seed, bounds)
        selection['failure_history'] = failure_history
    name = 'trial_%06d_%s_%s_%s' % (index, selector, selection['model_digest'][:10], selection['graph_digest'][:10])
    trial = directory / name
    trial.mkdir(exist_ok=True)
    if (trial / 'config.json').exists() and read_json(trial / 'config.json') != config:
        raise ValueError('proposal changed on resume: ' + name)
    write_json(trial / 'config.json', config)
    write_json(trial / 'selection.json', dict(selection, history=history, bounds=bounds, bank_manifest_sha256=plan['bank_manifest_sha256'],
        supervision_sha256=plan['supervision_sha256'], spec_sha256=registry.fingerprint(), seconds=time.perf_counter() - started))
    plan['trials'].append(dict(directory=name, model=selection['model_digest'], description=selection['description'],
                               graph=selection['graph_digest'], selector=selector))
    write_json(directory / 'plan.json', plan)
    emit('parameter_selection', trial=name, observations=len(observations), model=selection['description'],
        graph=selection['graph_digest'][:10], phase=selection['phase'], seconds=time.perf_counter() - started)
