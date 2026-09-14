import argparse

from graph_tuning_infra.files import read_json, write_json
from tuning.data import Bank
from tuning.run import run
from tuning.selector import random_suggest, suggest


def arguments(bank, output, **overrides):
    values = dict(bank=str(bank), output=str(output), seed=3, batch_sequences=8, continuous=False, trials=None, seconds=10,
                  selector=None, random_trials=None, bounds=None, workers=1, variant_gb=1)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_tpe_learns_from_every_recorded_trial_inside_the_bounds(bank):
    registry = Bank(bank).registry
    records = []
    for index in range(14):
        config, _ = random_suggest(registry, 100 + index, 200 + index, {})
        config['epochs'] = 1 + index % 4
        status = 'complete' if index % 5 else 'failed'
        records.append(dict(config=config, status=status, trial='trial_%06d_random' % index, raw_val_mse=1.0 + index))
    bounds = {'epochs': {'low': 1, 'high': 2}}
    completed = [entry for entry in records if entry['status'] == 'complete']
    failures = [dict(entry, raw_val_mse=None) for entry in records if entry['status'] != 'complete']
    config, selection = suggest(registry, completed, failures, 7, 8, bounds)
    inside = [entry for entry in records if entry['config']['epochs'] <= 2]
    assert selection['observations'] == sum(entry['status'] == 'complete' for entry in inside)
    assert selection['failed_observations'] == sum(entry['status'] != 'complete' for entry in inside)
    assert sorted(entry['trial'] for entry in selection['excluded']) == sorted(entry['trial'] for entry in records if entry['config']['epochs'] > 2)
    assert all(entry['parameters'] == ['epochs'] for entry in selection['excluded'])
    assert 1 <= config['epochs'] <= 2
    assert selection['phase'] == 'random_startup' and len(inside) < 10

    config, selection = suggest(registry, completed, failures, 7, 8, {})
    assert selection['excluded'] == [] and selection['observations'] == len(completed) and selection['failed_observations'] == len(failures)
    assert selection['phase'] == 'tpe'


def test_running_again_continues_and_may_change_selector_random_phase_and_bounds(bank, tmp_path):
    output = tmp_path / 'run'
    run(arguments(bank, output, selector='random'))
    plan = read_json(output / 'plan.json')
    assert plan['selector'] == 'random' and plan['random_trials'] == 0 and plan['bounds'] == {}

    run(arguments(bank, output))
    plan = read_json(output / 'plan.json')
    assert plan['selector'] == 'random' and plan['random_trials'] == 0 and plan['bounds'] == {}

    bounds = tmp_path / 'bounds.json'
    write_json(bounds, {'epochs': {'low': 1, 'high': 2}})
    run(arguments(bank, output, selector='tpe', random_trials=5, bounds=str(bounds), trials=20))
    plan = read_json(output / 'plan.json')
    assert plan['selector'] == 'tpe' and plan['random_trials'] == 5 and plan['bounds'] == {'epochs': {'low': 1, 'high': 2}} and plan['limit'] == 20

    run(arguments(bank, output, selector='random'))
    assert read_json(output / 'plan.json')['selector'] == 'random'

    run(arguments(bank, tmp_path / 'fresh'))
    assert read_json(tmp_path / 'fresh' / 'plan.json')['selector'] == 'tpe'
