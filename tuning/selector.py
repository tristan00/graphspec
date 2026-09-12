import math
import time

import numpy as np

from graphspec import spec as specs
from graph_tuning_infra import search
from graph_tuning_infra.files import read_json, sha256_file, write_json
from graph_tuning_infra.log import emit
from .roster import groups, models, sample


SETTINGS = dict(n_startup_trials=10, n_ei_candidates=64, multivariate=True, group=True)
FAILURE_POLICY = dict(version=1, outcomes=['timeout', 'failed'], objective='positive_infinity',
    interpretation='Failure ranks worse than every finite completed score; no prediction metric is assigned.')


def model_space(model):
    family = int(model[1:])
    result = dict(hidden=search.categorical([16, 24, 32, 48, 64]), epochs=search.integer(2, 12),
        clip=search.real(0.5, 2.0), lr=search.real(0.0001, 0.005, log=True),
        weight_decay=search.real(1e-7, 1e-2, log=True))
    if family in (*range(1, 18), 21, 22, 23, 27, 30):
        result['layers'] = search.categorical([1, 2] if family == 17 else [1, 2, 3])
    if family in (3, 11, 22, 23):
        result['heads'] = search.categorical([1, 2, 4])
    if family in (*range(1, 8), 11, 12, 13, 14, 15, 16, 22, 23):
        result['dropout'] = search.real(0.0, 0.5)
    if family in (17, 18, 20, 21, 22, 23, 24, 26, 27):
        result['neighbors'] = search.categorical([4, 8, 12, 16, 20] if family == 17 else [4, 8, 10, 12, 16, 20, 30, 32, 64])
    if family in (9, 30):
        result['diffusion_hops'] = search.integer(1, 4)
    if family == 22:
        result['patch'] = search.categorical([1, 2, 4, 8, 16])
    if family == 24:
        result.update(projection_hops=search.integer(1, 4),
            projection_width=search.categorical([16, 32, 64, 128, 256]), decay=search.real(1e-7, 1.0, log=True))
    if family == 25:
        result.update(walks=search.categorical([1, 2, 4, 8, 16, 32]), walk_hops=search.integer(1, 4))
    if family == 27:
        result['state_size'] = search.categorical([4, 8, 16, 32, 64])
    if family == 28:
        result['affinity_buffer'] = search.categorical([1, 2, 4, 8, 16, 32])
    if family == 29:
        result.update(integration_steps=search.integer(1, 16), damping=search.real(0.001, 1.0, log=True))
    if family == 30:
        result['signal_topk'] = search.integer(1, 20)
    return result


def distribution(parameter):
    if parameter.kind in ('bool', 'choice'):
        return search.categorical(parameter.options)
    if parameter.kind == 'int':
        return search.integer(parameter.low, parameter.high)
    return search.real(parameter.low, parameter.high, log=parameter.log)


def graph_space(registry, model):
    return specs.space(registry, groups(model))


def observation(registry, config, value, **attrs):
    model = config['model']
    params, distributions = dict(model=model), dict(model=search.categorical(models(registry)))
    for name, domain in model_space(model).items():
        key = model + '/' + name
        params[key], distributions[key] = config[name], domain
    parameters = {parameter.name: parameter for parameter in graph_space(registry, model)}
    for name, setting in specs.canonical(tuple(parameters.values()), config['graph']).items():
        params['graph/' + name], distributions['graph/' + name] = setting, distribution(parameters[name])
    return search.record(value, params, distributions, config=config, **attrs)


def metadata(registry, config, seed, training_seed, **extra):
    parameters = graph_space(registry, config['model'])
    return dict(extra, graph_digest=specs.digest(config['graph']), graph_summary=specs.summary(parameters, config['graph']),
                graph_space=[parameter.declaration() for parameter in parameters], selector_seed=seed, training_seed=training_seed)


def suggest(registry, observations, failures, seed, training_seed):
    started = time.perf_counter()
    records = [observation(registry, entry['config'], entry['raw_val_mse']) for entry in observations]
    records += [observation(registry, entry['config'], math.inf, outcome=entry['status']) for entry in failures]
    study = search.study(seed, SETTINGS, records)
    trial = study.ask()
    model = trial.suggest_categorical('model', list(models(registry)))
    config = sample(model, np.random.default_rng(training_seed), training_seed)
    for name, domain in model_space(model).items():
        config[name] = search.propose(trial, model + '/' + name, domain)
    config['graph'] = specs.draw(graph_space(registry, model), lambda parameter: search.propose(trial, 'graph/' + parameter.name, distribution(parameter)))
    total = len(observations) + len(failures)
    return config, metadata(registry, config, seed, training_seed, sampler='TPESampler', settings=SETTINGS, params=trial.params,
                            failure_policy=FAILURE_POLICY, observations=len(observations), failed_observations=len(failures),
                            optuna_version=search.VERSION, phase='tpe' if total >= SETTINGS['n_startup_trials'] else 'random_startup',
                            seconds=time.perf_counter() - started)


def random_suggest(registry, seed, training_seed):
    rng = np.random.default_rng(seed)
    model = str(rng.choice(list(models(registry))))
    config = sample(model, np.random.default_rng(training_seed), training_seed)
    for name, domain in model_space(model).items():
        config[name] = search.draw(domain, rng)
    config['graph'] = specs.draw(graph_space(registry, model), lambda parameter: parameter.uniform(rng))
    return config, metadata(registry, config, seed, training_seed, sampler='numpy.random.Generator', phase='random')


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
        if config['model'] != item['model']:
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
    if plan['selector'] == 'random':
        config, selection = random_suggest(registry, selector_seed, training_seed)
    else:
        failures, failure_history = load_failures(directory, plan)
        config, selection = suggest(registry, observations, failures, selector_seed, training_seed)
        selection['failure_history'] = failure_history
    name = 'trial_%06d_%s_%s_%s' % (index, plan['selector'], config['model'], selection['graph_digest'][:10])
    trial = directory / name
    trial.mkdir(exist_ok=True)
    if (trial / 'config.json').exists() and read_json(trial / 'config.json') != config:
        raise ValueError('proposal changed on resume: ' + name)
    write_json(trial / 'config.json', config)
    write_json(trial / 'selection.json', dict(selection, history=history, bank_manifest_sha256=plan['bank_manifest_sha256'],
        supervision_sha256=plan['supervision_sha256'], spec_sha256=registry.fingerprint(), seconds=time.perf_counter() - started))
    plan['trials'].append(dict(directory=name, model=config['model'], graph=selection['graph_digest'], selector=plan['selector']))
    write_json(directory / 'plan.json', plan)
    emit('parameter_selection', trial=name, observations=len(observations), model=config['model'],
        graph=selection['graph_digest'][:10], phase=selection['phase'], seconds=time.perf_counter() - started)
