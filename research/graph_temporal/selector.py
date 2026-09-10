import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from . import spec
from .cache import write_json
from .registry import DESIGNS, sample
from .turns import graph_name


CHOICES = tuple(model + '/' + design + suffix for model, designs in DESIGNS.items() for design in designs for suffix in ('', '+turn-end'))
GRAPH_VERSION = 2
MODELS = tuple(DESIGNS)
STRUCTURE = dict(version=2, model='categorical', shared_binary=['turn_end', 'colocated', 'turn_boundary'],
    conditional_layouts=['snapshot_graph', 'temporal_layout'], model_parameters='Shared across compatible graph variants within each model.')
SETTINGS = dict(n_startup_trials=10, n_ei_candidates=64, multivariate=True, group=True)
RANGE_VERSION = 5
RANGE_NAME = 'targeted'
RANDOM_RANGE_VERSION = 1
RANDOM_RANGE_NAME = 'targeted'
FAILURE_POLICY = dict(version=1, outcomes=['timeout', 'failed'], objective='positive_infinity',
    interpretation='Failure ranks worse than every finite completed score; no prediction metric is assigned.',
    interrupted='excluded')
TRAINING = 'Every training decision exactly once per epoch; complete holdout; 2–12 full epochs and 1–32 decision histories.'


def space(model, turn_context='none'):
    family = int(model[1:])
    result = dict(hidden=CategoricalDistribution([16, 24, 32, 48, 64]), epochs=IntDistribution(2, 12),
        history=IntDistribution(1, 32), clip=FloatDistribution(0.5, 2.0),
        lr=FloatDistribution(0.0001, 0.005, log=True), weight_decay=FloatDistribution(1e-7, 1e-2, log=True))
    if family in (*range(1, 18), 21, 22, 23, 27, 30):
        result['layers'] = CategoricalDistribution([1, 2] if family == 17 else [1, 2, 3])
    if family in (3, 11, 22, 23):
        result['heads'] = CategoricalDistribution([1, 2, 4])
    if family in (*range(1, 8), 11, 12, 13, 14, 15, 16, 22, 23):
        result['dropout'] = FloatDistribution(0.0, 0.5)
    if family in (17, 18, 20, 21, 22, 23, 24, 26, 27):
        result['neighbors'] = CategoricalDistribution([4, 8, 12, 16, 20] if family == 17 else [4, 8, 10, 12, 16, 20, 30, 32, 64])
    if family in (9, 30):
        result['diffusion_hops'] = IntDistribution(1, 4)
    if family == 22:
        result['patch'] = CategoricalDistribution([1, 2, 4, 8, 16])
    if family == 24:
        result.update(projection_hops=IntDistribution(1, 4),
            projection_width=CategoricalDistribution([16, 32, 64, 128, 256]), decay=FloatDistribution(1e-7, 1.0, log=True))
    if family == 25:
        result.update(walks=CategoricalDistribution([1, 2, 4, 8, 16, 32]), walk_hops=IntDistribution(1, 4))
    if family == 27:
        result['state_size'] = CategoricalDistribution([4, 8, 16, 32, 64])
    if family == 28:
        result['affinity_buffer'] = CategoricalDistribution([1, 2, 4, 8, 16, 32])
    if family == 29:
        result.update(integration_steps=IntDistribution(1, 16), damping=FloatDistribution(0.001, 1.0, log=True))
    if family == 30:
        result['signal_topk'] = IntDistribution(1, 20)
    if turn_context == 'end':
        result['turn_lookback'] = IntDistribution(1, 3)
    return result


def random_space(model, turn_context='none'):
    result = space(model, turn_context)
    if model == 'M10':
        result.update(hidden=CategoricalDistribution([16, 24, 32]), layers=CategoricalDistribution([1, 2]))
    elif model == 'M26':
        result['neighbors'] = CategoricalDistribution([4, 8, 12, 16])
    elif model == 'M14':
        result.update(hidden=CategoricalDistribution([16, 24]), layers=CategoricalDistribution([1]))
    elif model == 'M29':
        result['integration_steps'] = IntDistribution(1, 4)
    if model in ('M10', 'M14', 'M26', 'M29') and turn_context == 'end':
        result.update(history=IntDistribution(16, 32), turn_lookback=IntDistribution(1, 1))
    return result


def structure(config):
    model, design = config['model'], config['design']
    base = design.split('+')[0]
    values = dict(model=model, turn_end=config.get('turn_context', 'none') == 'end')
    domains = dict(model=CategoricalDistribution(MODELS), turn_end=CategoricalDistribution([False, True]))
    if model in ('M13', 'M14'):
        values['temporal_layout'] = base
        domains['temporal_layout'] = CategoricalDistribution(['X-chain', 'X-skip', 'X-allpast'])
    if model in ('M13', 'M14', 'M15', 'M16'):
        values['turn_boundary'] = design.endswith('+turn-boundary')
        domains['turn_boundary'] = CategoricalDistribution([False, True])
    elif int(model[1:]) <= 11:
        if model == 'M08':
            values['snapshot_graph'] = base
            domains['snapshot_graph'] = CategoricalDistribution(['S-core', 'S-full'])
        values['colocated'] = design.endswith('+colocated')
        domains['colocated'] = CategoricalDistribution([False, True])
    return values, domains


def parameter_key(model, name, turn_context):
    if name == 'turn_lookback' and model not in ('M10', 'M14', 'M26', 'M29'):
        return 'turn_lookback'
    if name == 'history' and turn_context == 'end' and model in ('M10', 'M14', 'M26', 'M29'):
        return model + '/turn_end/history'
    return model + '/' + name


def observation(config, value, targeted=False):
    params, distributions = structure(config)
    domain = (random_space if targeted else space)(config['model'], config.get('turn_context', 'none'))
    for name, distribution in domain.items():
        key = parameter_key(config['model'], name, config.get('turn_context', 'none'))
        supported = config[name] in distribution.choices if isinstance(distribution, CategoricalDistribution) else distribution.low <= config[name] <= distribution.high
        if targeted and not supported:
            continue
        distributions[key] = distribution
        params[key] = config[name]
    return optuna.trial.create_trial(value=value, params=params, distributions=distributions,
        user_attrs=dict(original_params=dict(config), graph_name=graph_name(config)))


def suggest_structure(trial, training_seed):
    model = trial.suggest_categorical('model', MODELS)
    end = trial.suggest_categorical('turn_end', [False, True])
    if model in ('M13', 'M14'):
        design = trial.suggest_categorical('temporal_layout', ['X-chain', 'X-skip', 'X-allpast'])
    elif model == 'M08':
        design = trial.suggest_categorical('snapshot_graph', ['S-core', 'S-full'])
    else:
        design = DESIGNS[model][0]
    if model in ('M13', 'M14', 'M15', 'M16') and trial.suggest_categorical('turn_boundary', [False, True]):
        design += '+turn-boundary'
    elif int(model[1:]) <= 11 and trial.suggest_categorical('colocated', [False, True]):
        design += '+colocated'
    config = sample(model, design, np.random.default_rng(training_seed), training_seed)
    config['turn_context'] = 'end' if end else 'none'
    return config


def suggest(observations, seed, training_seed, failures=()):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(seed=seed, **SETTINGS)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.NopPruner())
    for record in observations:
        study.add_trial(observation(record['config'], record['raw_val_mse'], targeted=True))
    for record in failures:
        imported = observation(record['config'], math.inf, targeted=True)
        imported.user_attrs['outcome'] = record['status']
        study.add_trial(imported)
    trial = study.ask()
    config = suggest_structure(trial, training_seed)
    model, turn_context = config['model'], config['turn_context']
    for name, distribution in random_space(model, turn_context).items():
        key = parameter_key(model, name, turn_context)
        if isinstance(distribution, CategoricalDistribution):
            config[name] = trial.suggest_categorical(key, distribution.choices)
        elif isinstance(distribution, IntDistribution):
            config[name] = trial.suggest_int(key, distribution.low, distribution.high, step=distribution.step, log=distribution.log)
        else:
            config[name] = trial.suggest_float(key, distribution.low, distribution.high, log=distribution.log)
    return config, dict(params=trial.params, sampler='TPESampler', settings=SETTINGS,
        graph_choices_version=GRAPH_VERSION, graph_choices=list(CHOICES), structure_parameterization=STRUCTURE,
        parameter_range=RANGE_NAME, parameter_range_version=RANGE_VERSION,
        history_import='All original scores; model and shared graph flags reconstructed from original configurations. Out-of-range coordinates omitted, full configurations retained in original_params.',
        failure_policy=FAILURE_POLICY, failed_observations=len(failures), total_observations=len(observations) + len(failures),
        distributions={key: json.loads(optuna.distributions.distribution_to_json(value)) for key, value in trial.distributions.items()},
        optuna_version=optuna.__version__, selector_seed=seed, training_seed=training_seed,
        observations=len(observations), phase='tpe' if len(observations) + len(failures) >= SETTINGS['n_startup_trials'] else 'random_startup')


def random_suggest(seed, training_seed):
    rng = np.random.default_rng(seed)
    model = str(rng.choice(list(DESIGNS)))
    design = str(rng.choice(DESIGNS[model]))
    config = sample(model, design, np.random.default_rng(training_seed), training_seed)
    config['turn_context'] = str(rng.choice(['none', 'end']))
    domain = random_space(model, config['turn_context'])
    for name, distribution in domain.items():
        if isinstance(distribution, CategoricalDistribution):
            config[name] = distribution.choices[int(rng.integers(len(distribution.choices)))]
        elif isinstance(distribution, IntDistribution):
            config[name] = int(rng.integers(distribution.low, distribution.high + 1))
        elif distribution.log:
            config[name] = float(np.exp(rng.uniform(np.log(distribution.low), np.log(distribution.high))))
        else:
            config[name] = float(rng.uniform(distribution.low, distribution.high))
    imported = observation(config, 0.0)
    distributions = dict(imported.distributions)
    distributions.update({parameter_key(model, key, config['turn_context']): distribution for key, distribution in domain.items()})
    return config, dict(params=imported.params, sampler='numpy.random.Generator',
        graph_choices_version=GRAPH_VERSION, graph_choices=list(CHOICES), structure_parameterization=STRUCTURE,
        selection='Uniform model, uniform compatible base graph, uniform turn variant; independent random conditional parameters.',
        parameter_range=RANGE_NAME, parameter_range_version=RANGE_VERSION,
        random_parameter_range=RANDOM_RANGE_NAME, random_parameter_range_version=RANDOM_RANGE_VERSION,
        distributions={key: json.loads(optuna.distributions.distribution_to_json(value)) for key, value in distributions.items()},
        selector_seed=seed, training_seed=training_seed, observations=0, phase='random_exploration')


def schedule_random(directory, plan, count):
    if count < 1:
        raise ValueError('random trial count must be positive')
    if 'random_exploration' in plan:
        if plan['random_exploration']['count'] != count:
            raise ValueError('existing random exploration has a different count')
        return
    write_json(directory / 'plan_before_turn_random.json', plan)
    for item in plan['trials']:
        trial = directory / item['directory']
        if not any((trial / name).exists() for name in ('result.json', 'failure.json', 'timeout.json')):
            item['archived_pending'] = True
    plan['random_exploration'] = dict(start_index=len(plan['trials']), count=count,
        counting='Individual attempts; failures count, interrupted attempts resume.', after='tpe')
    plan['selection'] = '%d random model/graph/parameter attempts, then TPE using all compatible completed fits' % count
    write_json(directory / 'plan.json', plan)


def load_observations(directory, plan):
    observations, history = [], []
    for item in plan['trials']:
        path = directory / item['directory'] / 'result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        valid = (result['status'] == 'complete'
            and result['supervision_sha256'] == plan['supervision_sha256']
            and result['training_coverage_complete']
            and result['train_unique_rows'] == result['train_available_rows'] == plan['train_rows']
            and result['holdout_rows'] == plan['holdout_rows']
            and result['training_examples_seen'] == result['epochs'] * plan['train_rows']
            and 1 <= result['config']['history'] <= 32
            and math.isfinite(result['raw_val_mse']))
        if not valid:
            raise ValueError('incompatible Bayesian observation: ' + str(path))
        observations.append(result)
        history.append(dict(trial=item['directory'], raw_val_mse=result['raw_val_mse'],
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return observations, history


def load_failures(directory, plan):
    failures, history = [], []
    for item in plan['trials']:
        trial = directory / item['directory']
        if item.get('archived_pending') or (trial / 'result.json').exists():
            continue
        path = next((trial / name for name in ('timeout.json', 'failure.json') if (trial / name).exists()), None)
        if path is None:
            continue
        failure = json.loads(path.read_text())
        if failure.get('status') == 'interrupted':
            continue
        config_path = trial / 'config.json'
        config = json.loads(config_path.read_text())
        selection_path = trial / 'selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if 'supervision_sha256' in selection and selection['supervision_sha256'] != plan['supervision_sha256']:
                raise ValueError('incompatible Bayesian failure: ' + str(path))
        if config['model'] != item['model'] or config['design'] != item['design']:
            raise ValueError('failure configuration differs from its saved plan: ' + str(path))
        if any(key in failure and failure[key] != config[key] for key in ('model', 'design')):
            raise ValueError('failure record differs from its configuration: ' + str(path))
        status = 'timeout' if path.name == 'timeout.json' else 'failed'
        failures.append(dict(config=config, status=status))
        history.append(dict(trial=item['directory'], status=status, record=path.name,
            failure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
    return failures, history


def activate_graphs(directory, plan):
    if plan.get('graph_choices_version', 1) != GRAPH_VERSION:
        archive = directory / ('plan_before_graph_choices_%d.json' % GRAPH_VERSION)
        if not archive.exists():
            write_json(archive, plan)
        plan.update(graph_choices_version=GRAPH_VERSION, graph_choices=list(CHOICES), structure_parameterization=STRUCTURE,
            selection='TPE selects a model, shared binary graph edits, compatible base layouts and conditional model parameters.')
        write_json(directory / 'plan.json', plan)


def activate(directory, plan):
    if plan.get('parameter_selection') in ('tpe', 'random'):
        if plan.get('parameter_range_version') != RANGE_VERSION:
            archive = directory / ('plan_before_range_%d.json' % RANGE_VERSION)
            if not archive.exists():
                write_json(archive, plan)
            plan.update(parameter_range=RANGE_NAME, parameter_range_version=RANGE_VERSION, histories=list(range(1, 33)), training=TRAINING)
        plan.update(parameter_selection='tpe',
            selection='TPE selects a model, shared binary graph edits, compatible base layouts and conditional model parameters.')
        write_json(directory / 'plan.json', plan)
        activate_graphs(directory, plan)
        return
    archive = directory / 'random_plan_before_tpe.json'
    if not archive.exists():
        write_json(archive, plan)
    started = []
    for item in plan['trials']:
        trial = directory / item['directory']
        if any(trial.glob('worker_*.log')) or any((trial / name).exists() for name in ('result.json', 'failure.json', 'timeout.json')):
            started.append(item)
    manifest = json.loads((Path(plan['cache']) / 'manifest.json').read_text())
    if manifest['supervision_sha256'] != plan['supervision_sha256']:
        raise ValueError('Bayesian migration changed supervision')
    plan.update(parameter_selection='tpe', selection='TPE selects a valid model/graph pair and conditional parameters',
        trials=started, train_rows=manifest['train_rows'], holdout_rows=manifest['validation_rows'],
        random_plan_archive=archive.name, sampler_settings=SETTINGS, parameter_range=RANGE_NAME,
        parameter_range_version=RANGE_VERSION, histories=list(range(1, 33)), training=TRAINING)
    write_json(directory / 'plan.json', plan)
    activate_graphs(directory, plan)


def activate_random(directory, plan):
    activate(directory, plan)
    archive = directory / 'plan_before_continuous_random.json'
    if not archive.exists():
        write_json(archive, plan)
    plan.update(parameter_selection='random',
        selection='Continuous uniform random model, compatible graph structure and targeted conditional parameters.',
        random_parameter_range=RANDOM_RANGE_NAME, random_parameter_range_version=RANDOM_RANGE_VERSION)
    write_json(directory / 'plan.json', plan)


def append_trial(directory, plan):
    from .run import timed_event

    started = time.perf_counter()
    observations, history = load_observations(directory, plan)
    index = len(plan['trials'])
    rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], index, 1]))
    selector_seed, training_seed = (int(v) for v in rng.integers(1, 2**31, size=2))
    exploration = plan.get('random_exploration')
    is_random = (plan.get('parameter_selection') == 'random'
        or exploration is not None and index < exploration['start_index'] + exploration['count'])
    if is_random:
        config, metadata = random_suggest(selector_seed, training_seed)
        plan.update(random_parameter_range=RANDOM_RANGE_NAME, random_parameter_range_version=RANDOM_RANGE_VERSION)
        metadata['observations'] = len(observations)
        if exploration is not None and index < exploration['start_index'] + exploration['count']:
            metadata.update(random_attempt=index - exploration['start_index'] + 1, random_total=exploration['count'])
    else:
        failures, failure_history = load_failures(directory, plan)
        config, metadata = suggest(observations, selector_seed, training_seed, failures)
        metadata['failure_history'] = failure_history
        plan['failure_policy'] = FAILURE_POLICY
    if 'graph_variant_cache' in plan:
        metadata['graph_variant_cache'] = plan['graph_variant_cache']
    selector = 'random' if is_random else 'tpe'
    name = 'trial_%06d_%s_%s_%s' % (index, selector, config['model'], graph_name(config))
    trial = directory / name
    trial.mkdir(exist_ok=True)
    if (trial / 'config.json').exists() and json.loads((trial / 'config.json').read_text()) != config:
        raise ValueError('Bayesian proposal changed on resume: ' + name)
    write_json(trial / 'config.json', config)
    write_json(trial / 'selection.json', dict(metadata, history=history,
        supervision_sha256=plan['supervision_sha256'], spec_sha256=spec.fingerprint(),
        seconds=time.perf_counter() - started))
    plan['trials'].append(dict(directory=name, model=config['model'], design=config['design'],
        turn_context=config['turn_context'], selector=selector))
    write_json(directory / 'plan.json', plan)
    timed_event('parameter_selection', trial=name, observations=len(observations), model=config['model'],
        design=graph_name(config), phase=metadata['phase'], seconds=time.perf_counter() - started)
