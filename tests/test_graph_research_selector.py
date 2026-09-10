from copy import deepcopy
import json

import numpy as np
import pytest

from research.graph_temporal.cache import write_json
from research.graph_temporal.registry import DESIGNS, sample
from research.graph_temporal.run import append_round, report, training_layout
from research.graph_temporal.selector import CHOICES, activate, activate_random, append_trial, load_observations, observation, space, suggest, structure, parameter_key, random_space
from research.graph_temporal.turns import graph_name
from research.graph_temporal import spec


def choice(config):
    return config['model'] + '/' + graph_name(config)


def result(config, value):
    return dict(config=config, model=config['model'], design=config['design'], raw_val_mse=value,
        status='complete', supervision_sha256='frozen', training_coverage_complete=True,
        train_unique_rows=116652, train_available_rows=116652, holdout_rows=20387,
        training_examples_seen=config['epochs'] * 116652, epochs=config['epochs'])


def history(best, repeats=3):
    rng = np.random.default_rng(77)
    records = []
    for choice in CHOICES:
        model, design = choice.split('/')
        for i in range(repeats):
            config = sample(model, design.removesuffix('+turn-end'), rng, len(records) + 1)
            config.update(turn_context='end' if design.endswith('+turn-end') else 'none', turn_lookback=2)
            config['epochs'] = 2 + i % 2
            records.append(result(config, (0.1 if choice == best else 10) + 0.01 * i))
    return records


def selection_history(best, repeats=1):
    records = history(best, repeats)
    records.extend(deepcopy([r for r in records if choice(r['config']) == best] * 10))
    np.random.default_rng(71).shuffle(records)
    return records


def test_tpe_imports_every_family_and_keeps_all_choices_eligible():
    records = history('M13/X-allpast')
    assert len(CHOICES) == 110
    assert len({choice.split('/')[0] for choice in CHOICES}) == 30
    for record in records:
        imported = observation(record['config'], record['raw_val_mse'])
        assert imported.value == record['raw_val_mse']
        assert imported.distributions['model'].choices == tuple(DESIGNS)
        assert imported.params['turn_end'] == (record['config']['turn_context'] == 'end')
        assert 'model_graph' not in imported.params
    config, metadata = suggest(records, 71, 12)
    assert metadata['observations'] == 330 and metadata['phase'] == 'tpe'
    assert config['design'] in DESIGNS[config['model']]
    assert set(metadata['params']) == set(structure(config)[0]) | {
        parameter_key(config['model'], key, config['turn_context']) for key in random_space(config['model'], config['turn_context'])}
    assert suggest(records, 71, 12) == (config, metadata)


def test_observed_scores_change_model_graph_selection():
    left, right = selection_history('M13/X-allpast'), selection_history('M30/V-signals')
    left_choices = [choice(suggest(left, seed, 200 + seed)[0]) for seed in range(16)]
    right_choices = [choice(suggest(right, seed, 200 + seed)[0]) for seed in range(16)]
    assert left_choices.count('M13/X-allpast') >= 8
    assert right_choices.count('M30/V-signals') >= 8
    assert left_choices != right_choices


def setup_run(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    write_json(cache / 'manifest.json', dict(supervision_sha256='frozen', spec_sha256=spec.fingerprint(), train_rows=116652, validation_rows=20387))
    plan = dict(seed=77, rounds=0, trials=[], cache=str(cache), supervision_sha256='frozen', history=16)
    append_round(tmp_path, plan)
    for item in plan['trials'][:12]:
        trial = tmp_path / item['directory']
        config = json.loads((trial / 'config.json').read_text())
        write_json(trial / 'result.json', result(config, 3.0 + config['hidden'] / 100))
    failed = tmp_path / plan['trials'][12]['directory']
    write_json(failed / 'timeout.json', dict(model=plan['trials'][12]['model'], design=plan['trials'][12]['design'], seconds=600))
    return plan


def test_migration_preserves_results_and_resume_preserves_proposal(tmp_path):
    plan = setup_run(tmp_path)
    before = deepcopy(plan)
    original = {path: path.read_bytes() for path in tmp_path.glob('trial_*/*.json')}
    activate(tmp_path, plan)
    assert len(plan['trials']) == 13
    assert json.loads((tmp_path / 'random_plan_before_tpe.json').read_text()) == before
    assert all(path.read_bytes() == value for path, value in original.items())
    activate(tmp_path, plan)
    records, imported = load_observations(tmp_path, plan)
    assert len(records) == len(imported) == 12
    append_trial(tmp_path, plan)
    proposal = plan['trials'][-1]
    config_path = tmp_path / proposal['directory'] / 'config.json'
    config = config_path.read_bytes()
    saved = json.loads((tmp_path / 'plan.json').read_text())
    activate(tmp_path, saved)
    assert saved == plan and config_path.read_bytes() == config
    report(tmp_path)
    assert 'TPE selects' in (tmp_path / 'progress.md').read_text()
    plan['trials'].pop()
    append_trial(tmp_path, plan)
    assert plan['trials'][-1] == proposal and config_path.read_bytes() == config
    write_json(config_path.with_name('result.json'), result(json.loads(config), 2.5))
    append_trial(tmp_path, plan)
    selected = json.loads((tmp_path / plan['trials'][-1]['directory'] / 'selection.json').read_text())
    assert selected['observations'] == 13
    assert selected['history'][-1]['raw_val_mse'] == 2.5


@pytest.mark.parametrize('field,value', [('supervision_sha256', 'different'), ('train_unique_rows', 868), ('holdout_rows', 12)])
def test_incompatible_history_fails_instead_of_silently_dropping_results(tmp_path, field, value):
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    path = tmp_path / plan['trials'][0]['directory'] / 'result.json'
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError, match='incompatible Bayesian observation'):
        load_observations(tmp_path, plan)


def test_medium_space_preserves_old_values_and_records_wider_distributions():
    records = history('M13/X-allpast')
    config, metadata = suggest(records, 19, 44)
    assert metadata['parameter_range'] == 'targeted'
    assert metadata['parameter_range_version'] == 5
    for record in records:
        imported = observation(record['config'], record['raw_val_mse'])
        assert choice(record['config']) in CHOICES
    for model in DESIGNS:
        domain = space(model)
        assert domain['hidden'].choices == (16, 24, 32, 48, 64)
        assert (domain['history'].low, domain['history'].high) == (1, 32)
        assert (domain['epochs'].low, domain['epochs'].high) == (2, 12)
        assert (domain['clip'].low, domain['clip'].high) == (0.5, 2.0)
        if 'dropout' in domain:
            assert (domain['dropout'].low, domain['dropout'].high) == (0.0, 0.5)
        assert domain['lr'].low < 0.0003 and domain['lr'].high > 0.002
        assert domain['weight_decay'].low < 1e-6 and domain['weight_decay'].high > 1e-3
    key = config['model'] + '/hidden'
    assert metadata['distributions'][key]['attributes']['choices'] == [16, 24, 32, 48, 64]
    for hidden in space('M22')['hidden'].choices:
        for heads in space('M22')['heads'].choices:
            assert hidden % heads == 0
    for seed in range(4):
        config, _ = suggest(records, seed, seed + 100)
        assert type(config['history']) is int and 1 <= config['history'] <= 32
        assert type(config['epochs']) is int and 2 <= config['epochs'] <= 12
        assert isinstance(config['clip'], float)


def test_range_upgrade_keeps_existing_proposals_and_original_plan(tmp_path):
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    append_trial(tmp_path, plan)
    plan.pop('parameter_range_version')
    before = deepcopy(plan)
    path = tmp_path / plan['trials'][-1]['directory'] / 'config.json'
    original = path.read_bytes()
    activate(tmp_path, plan)
    assert path.read_bytes() == original
    assert plan['trials'] == before['trials']
    assert json.loads((tmp_path / 'plan_before_range_5.json').read_text()) == before
    assert plan['histories'] == list(range(1, 33))


def test_longer_history_layout_covers_identical_training_rows(tmp_path):
    from research.graph_temporal.data import Bank

    cache = tmp_path / 'cache'
    cache.mkdir()
    campaigns = [dict(index=i, name=str(i), decisions=size) for i, size in enumerate((49, 67, 35))]
    rows = [dict(campaign=c['name'], y=float(j)) for c in campaigns for j in range(c['decisions'])]
    train, held = list(range(116)), list(range(116, 151))
    write_json(cache / 'manifest.json', dict(supervision_sha256='frozen', spec_sha256=spec.fingerprint(), campaigns=campaigns))
    write_json(cache / 'supervision.json', dict(rows=rows, split=dict(train_indices=train, validation_indices=held)))
    plan = dict(cache=str(cache), supervision_sha256='frozen', history=16, histories=list(range(1, 33)), batch_sequences=64)
    bank = Bank(cache)
    starts = [0, 49, 116]
    for history_size in range(1, 33):
        expected_batch = 1024 // history_size
        path, size = training_layout(tmp_path, plan, dict(history=history_size))
        blocks = json.loads(path.read_text())
        assert sorted(map(tuple, blocks)) == sorted(bank.blocks(0, history_size))
        represented = [starts[c] + step for c, start, stop in blocks for step in range(start, stop)]
        assert sorted(represented) == train
        assert size == expected_batch and size * history_size <= 1024
        if history_size > 1:
            assert any(stop - start < history_size for c, start, stop in blocks)


def test_scheduled_random_attempts_then_tpe_and_resume_keeps_counter(tmp_path):
    from research.graph_temporal.selector import schedule_random

    total = 12
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    append_trial(tmp_path, plan)
    pending = plan['trials'][-1]
    schedule_random(tmp_path, plan, total)
    assert pending['archived_pending']
    start = len(plan['trials'])
    for i in range(total):
        append_trial(tmp_path, plan)
        item = plan['trials'][-1]
        selection = json.loads((tmp_path / item['directory'] / 'selection.json').read_text())
        assert item['selector'] == 'random' and selection['random_attempt'] == i + 1
        if i == total // 2:
            restored = json.loads((tmp_path / 'plan.json').read_text())
            schedule_random(tmp_path, restored, total)
            assert restored == plan
        config = json.loads((tmp_path / item['directory'] / 'config.json').read_text())
        if i == total - 1:
            write_json(tmp_path / item['directory'] / 'result.json', result(config, 1.7))
        else:
            write_json(tmp_path / item['directory'] / 'timeout.json', dict(seconds=600))
    append_trial(tmp_path, plan)
    assert len(plan['trials']) == start + total + 1
    assert plan['trials'][-1]['selector'] == 'tpe'
    selection = json.loads((tmp_path / plan['trials'][-1]['directory'] / 'selection.json').read_text())
    assert selection['phase'] == 'tpe' and selection['observations'] == 13
    assert selection['history'][-1]['raw_val_mse'] == 1.7
    before = deepcopy(plan)
    schedule_random(tmp_path, plan, total)
    assert plan == before


def test_continuous_random_can_replace_tpe_and_switch_back(tmp_path):
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    append_trial(tmp_path, plan)
    completed_before = list(plan['trials'])
    activate_random(tmp_path, plan)
    assert plan['parameter_selection'] == 'random'
    assert plan['trials'] == completed_before
    append_trial(tmp_path, plan)
    item = plan['trials'][-1]
    selection = json.loads((tmp_path / item['directory'] / 'selection.json').read_text())
    assert item['selector'] == 'random'
    assert selection['phase'] == 'random_exploration'
    assert selection['observations'] == 12
    assert selection['random_parameter_range'] == 'targeted'
    activate(tmp_path, plan)
    assert plan['parameter_selection'] == 'tpe'
    append_trial(tmp_path, plan)
    assert plan['trials'][-1]['selector'] == 'tpe'


def test_random_selection_is_uniform_over_models_and_supports_every_variant():
    from collections import Counter
    from research.graph_temporal.selector import random_suggest

    configs = [random_suggest(seed, seed + 1)[0] for seed in range(3000)]
    count = Counter(c['model'] for c in configs)
    assert set(count) == set(DESIGNS)
    assert all(60 <= value <= 140 for value in count.values())
    assert {c['model'] + '/' + graph_name(c) for c in configs} == set(CHOICES)
    assert 1300 <= sum(c['turn_context'] == 'end' for c in configs) <= 1700
    assert random_suggest(71, 8) == random_suggest(71, 8)


def test_budget_stopped_scores_keep_original_tpe_parameters(tmp_path):
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    path = tmp_path / plan['trials'][0]['directory'] / 'result.json'
    record = json.loads(path.read_text())
    record['config']['epochs'] = 12
    record.update(epochs=6, requested_epochs=12, training_examples_seen=6 * plan['train_rows'], stop_reason='epoch_budget')
    write_json(path, record)
    original = path.read_bytes()
    records, history = load_observations(tmp_path, plan)
    assert records[0]['config']['epochs'] == 12
    assert records[0]['epochs'] == 6 and records[0]['stop_reason'] == 'epoch_budget'
    trial = observation(records[0]['config'], records[0]['raw_val_mse'])
    assert trial.params[records[0]['config']['model'] + '/epochs'] == 12
    assert path.read_bytes() == original


@pytest.mark.parametrize('model', list(DESIGNS))
@pytest.mark.parametrize('turn_context', ['none', 'end'])
def test_random_reductions_are_limited_to_authorized_parameters(model, turn_context):
    from research.graph_temporal.selector import random_space

    broad = space(model, turn_context)
    reduced = random_space(model, turn_context)
    expected = {'M10': {'hidden', 'layers'}, 'M14': {'hidden', 'layers'},
        'M26': {'neighbors'}, 'M29': {'integration_steps'}}.get(model, set()).copy()
    if expected and turn_context == 'end':
        expected |= {'history', 'turn_lookback'}
    assert set(broad) == set(reduced)
    assert {key for key in broad if broad[key] != reduced[key]} == expected
    assert reduced['epochs'] == broad['epochs']
    assert (reduced['epochs'].low, reduced['epochs'].high) == (2, 12)
    if model == 'M10':
        assert reduced['hidden'].choices == (16, 24, 32) and reduced['layers'].choices == (1, 2)
    elif model == 'M14':
        assert reduced['hidden'].choices == (16, 24) and reduced['layers'].choices == (1,)
    elif model == 'M26':
        assert reduced['neighbors'].choices == (4, 8, 12, 16)
    elif model == 'M29':
        assert (reduced['integration_steps'].low, reduced['integration_steps'].high) == (1, 4)
    if expected and turn_context == 'end':
        assert (reduced['history'].low, reduced['history'].high) == (16, 32)
        assert (reduced['turn_lookback'].low, reduced['turn_lookback'].high) == (1, 1)
    assert space(model, turn_context) == broad


def test_random_selection_records_reduced_domains_but_tpe_still_accepts_wider_history():
    from optuna.distributions import distribution_to_json
    from research.graph_temporal.selector import random_space, random_suggest

    seen = set()
    for seed in range(500):
        config, metadata = random_suggest(seed, seed + 99)
        for key, distribution in random_space(config['model'], config['turn_context']).items():
            assert distribution._contains(distribution.to_internal_repr(config[key]))
            assert metadata['distributions'][parameter_key(config['model'], key, config['turn_context'])] == json.loads(distribution_to_json(distribution))
        if config['model'] in ('M10', 'M14', 'M26', 'M29'):
            seen.add((config['model'], config['turn_context']))
        assert metadata['random_parameter_range'] == 'targeted'
        assert metadata['random_parameter_range_version'] == 1
    assert len(seen) == 8
    config = sample('M10', 'S-full', np.random.default_rng(7), 77)
    config.update(hidden=64, layers=3, epochs=12, history=2, turn_context='end', turn_lookback=3)
    imported = observation(config, 4.0)
    assert imported.params['M10/hidden'] == 64
    assert imported.params['M10/turn_end/history'] == 2
    assert imported.params['M10/turn_lookback'] == 3


@pytest.mark.parametrize('model', ['M10', 'M14', 'M26', 'M29'])
def test_targeted_tpe_keeps_all_scores_without_clamping_old_parameters(model):
    from research.graph_temporal.selector import random_space

    records = history(model + '/' + DESIGNS[model][0] + '+turn-end', 1)
    for record in records:
        cfg = record['config']
        broad = space(cfg['model'], cfg['turn_context'])
        for key, distribution in broad.items():
            cfg[key] = distribution.choices[-1] if hasattr(distribution, 'choices') else distribution.high
    original = deepcopy(records)
    for record in records:
        imported = observation(record['config'], record['raw_val_mse'], targeted=True)
        complete = observation(record['config'], record['raw_val_mse'])
        assert imported.value == record['raw_val_mse']
        assert imported.user_attrs['original_params'] == record['config']
        assert imported.params['model'] == complete.params['model']
        assert imported.params['turn_end'] == complete.params['turn_end']
        for key, value in imported.params.items():
            assert value == complete.params[key]
            distribution = imported.distributions[key]
            assert distribution._contains(distribution.to_internal_repr(value))
    for seed in range(6):
        cfg, metadata = suggest(records, seed, seed + 100)
        assert metadata['observations'] == len(records)
        assert metadata['parameter_range_version'] == 5
        for key, distribution in random_space(cfg['model'], cfg['turn_context']).items():
            assert distribution._contains(distribution.to_internal_repr(cfg[key]))
    assert records == original


def test_failure_import_preserves_records_and_excludes_interruptions_and_recovered_trials(tmp_path):
    from research.graph_temporal.selector import load_failures

    plan = setup_run(tmp_path)
    interrupted, recovered, crashed = plan['trials'][13:16]
    write_json(tmp_path / interrupted['directory'] / 'failure.json', dict(status='interrupted', error='User-requested restart'))
    write_json(tmp_path / recovered['directory'] / 'timeout.json', dict(seconds=600))
    cfg = json.loads((tmp_path / recovered['directory'] / 'config.json').read_text())
    write_json(tmp_path / recovered['directory'] / 'result.json', result(cfg, 3.5))
    write_json(tmp_path / crashed['directory'] / 'failure.json', dict(error='CUDA out of memory'))
    original = {path: path.read_bytes() for path in tmp_path.glob('trial_*/*.json')}
    failed, provenance = load_failures(tmp_path, plan)
    assert [entry['status'] for entry in failed] == ['timeout', 'failed']
    assert [entry['trial'] for entry in provenance] == [plan['trials'][12]['directory'], crashed['directory']]
    assert all('raw_val_mse' not in entry for entry in failed)
    assert all(path.read_bytes() == data for path, data in original.items())


def test_failure_import_rejects_another_split(tmp_path):
    from research.graph_temporal.selector import load_failures

    plan = setup_run(tmp_path)
    failed = tmp_path / plan['trials'][12]['directory']
    write_json(failed / 'selection.json', dict(supervision_sha256='different'))
    with pytest.raises(ValueError, match='incompatible Bayesian failure'):
        load_failures(tmp_path, plan)


def test_timeout_evidence_reduces_model_graph_selection():
    records = selection_history('M13/X-allpast')
    failed = [dict(config=r['config'], status='timeout') for r in records
        if r['model'] == 'M13' and r['design'] == 'X-allpast' and r['config']['turn_context'] == 'none'] * 3
    original = deepcopy(records)
    before = [choice(suggest(records, seed, seed + 100)[0]) for seed in range(16)]
    after = [choice(suggest(records, seed, seed + 100, failed)[0]) for seed in range(16)]
    assert after.count('M13/X-allpast') <= before.count('M13/X-allpast') - 8
    assert records == original


def test_failure_feedback_changes_parameters_within_the_same_family(monkeypatch):
    from research.graph_temporal import selector

    monkeypatch.setattr(selector, 'MODELS', ('M22',))
    records = []
    for i in range(20):
        cfg = sample('M22', 'H-neighbor', np.random.default_rng(7), 9)
        cfg.update(hidden=16 if i % 2 == 0 else 64, epochs=3, history=16, layers=1,
            neighbors=8, patch=2, turn_context='none', dropout=0.1)
        records.append(result(cfg, 3.0))
    failed = [dict(config=deepcopy(records[1]['config']), status='timeout') for _ in range(60)]
    before = [suggest(records, seed, seed + 100)[0]['hidden'] for seed in range(16)]
    after = [suggest(records, seed, seed + 100, failed)[0]['hidden'] for seed in range(16)]
    assert after.count(16) >= before.count(16) + 8


def test_failure_proposals_keep_penalty_separate_from_prediction_scores(tmp_path):
    plan = setup_run(tmp_path)
    activate(tmp_path, plan)
    failed = tmp_path / plan['trials'][-1]['directory']
    original = (failed / 'timeout.json').read_bytes()
    append_trial(tmp_path, plan)
    selection = json.loads((tmp_path / plan['trials'][-1]['directory'] / 'selection.json').read_text())
    assert selection['observations'] == 12
    assert selection['failed_observations'] == 1 and selection['total_observations'] == 13
    assert selection['failure_policy']['objective'] == 'positive_infinity'
    assert selection['failure_history'][0]['trial'] == failed.name
    assert (failed / 'timeout.json').read_bytes() == original
    assert not (failed / 'result.json').exists()
    assert all('raw_val_mse' not in entry for entry in selection['failure_history'])
    rows = report(tmp_path)
    failure = next(row for row in rows if row['trial'] == failed.name)
    assert failure['status'] == 'timeout' and 'raw_val_mse' not in failure
    assert 'Recorded timeouts and crashes also inform TPE' in (tmp_path / 'progress.md').read_text()
    json.dumps(selection, allow_nan=False)


def test_shared_graph_switches_roundtrip_every_supported_combination():
    import optuna
    from research.graph_temporal.selector import suggest_structure

    for graph in CHOICES:
        model, name = graph.split('/')
        config = sample(model, name.removesuffix('+turn-end'), np.random.default_rng(7), 9)
        config['turn_context'] = 'end' if name.endswith('+turn-end') else 'none'
        params, domains = structure(config)
        recreated = suggest_structure(optuna.trial.FixedTrial(params), 9)
        assert choice(recreated) == graph
        assert 'model_graph' not in params
        assert params['turn_end'] is (config['turn_context'] == 'end')
        for key in ('turn_end', 'colocated', 'turn_boundary'):
            if key in domains:
                assert domains[key].choices == (False, True)
        imported = observation(config | {'epochs': 3, 'turn_lookback': 1, 'hidden': 16}, 3.0, targeted=True)
        assert model + '/hidden' in imported.params


@pytest.mark.parametrize('preferred', [False, True])
def test_shared_turn_switch_responds_to_scores(preferred):
    records = history('M13/X-allpast', 1)
    rng = np.random.default_rng(19)
    for entry in records:
        active = entry['config']['turn_context'] == 'end'
        entry['raw_val_mse'] = (1.0 if active == preferred else 10.0) + float(rng.uniform(0, 0.1))
    rng.shuffle(records)
    selected = [suggest(records, seed, seed + 100)[1]['params']['turn_end'] for seed in range(16)]
    assert selected.count(preferred) >= 12
