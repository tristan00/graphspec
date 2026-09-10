import hashlib
import json
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch

from research.graph_temporal.budget import FitBudget, eligible, wait_worker
from research.graph_temporal.cache import write_json


@pytest.mark.parametrize('durations,expected', [([], False), ([119.99], True), ([120], False),
    ([90, 149], True), ([90, 150], False), ([float('nan')], False), ([float('inf')], False)])
def test_only_completed_epochs_averaging_strictly_under_two_minutes_qualify(durations, expected):
    assert eligible(durations) is expected


def test_grace_remembers_previous_average_even_if_final_epoch_is_slower(tmp_path, monkeypatch):
    from research.graph_temporal import budget as module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock.now))
    budget = FitBudget(tmp_path, 600)
    budget.epoch_seconds = [80.0, 100.0]
    clock.now = 599.0
    assert not budget.check()
    clock.now = 600.0
    assert budget.check()
    budget.epoch_seconds.append(300.0)
    clock.now = 900.0
    assert budget.check()
    budget.pulse('evaluation')
    saved = json.loads((tmp_path / 'fit_progress.json').read_text())
    assert saved['grace_epoch_seconds'] == [80.0, 100.0]
    assert saved['phase'] == 'evaluation'


@pytest.mark.parametrize('durations', [[], [120], [150, 100]])
def test_slow_or_missing_previous_epochs_still_time_out(tmp_path, monkeypatch, durations):
    from research.graph_temporal import budget as module

    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: 601))
    budget = FitBudget(tmp_path, 600, deadline=600)
    budget.epoch_seconds = durations
    with pytest.raises(TimeoutError):
        budget.check()
    assert not budget.extended


@pytest.mark.parametrize('durations,stamp,allowed', [([90], 600, True), ([120], 600, False), ([], 600, False), ([80], 599, False)])
def test_supervisor_respects_the_same_epoch_rule_and_rejects_stale_attempts(tmp_path, monkeypatch, durations, stamp, allowed):
    from research.graph_temporal import budget as module

    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: 600))
    write_json(tmp_path / 'fit_progress.json', dict(deadline=stamp, updated=600, epoch_seconds=durations, phase='training'))

    class Process:
        args = ['worker']
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return 0

    process = Process()
    if allowed:
        assert wait_worker(process, tmp_path, 600) == 0
        assert process.calls == 2
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            wait_worker(process, tmp_path, 600)
        assert process.calls == 1


def test_supervisor_does_not_leave_a_stalled_grace_worker_running_forever(tmp_path, monkeypatch):
    from research.graph_temporal import budget as module

    clock = SimpleNamespace(now=600)
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock.now))
    write_json(tmp_path / 'fit_progress.json', dict(deadline=600, updated=600, epoch_seconds=[90], phase='training'))

    class Process:
        args = ['worker']

        def wait(self, timeout):
            clock.now += 601
            raise subprocess.TimeoutExpired(self.args, timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        wait_worker(Process(), tmp_path, 600)


@pytest.mark.parametrize('requested,limit,training_cost,evaluation_cost,actual,extended', [
    (4, 100, 40, 200, 2, True), (4, 160, 40, 200, 2, True),
    (2, 200, 40, 200, 2, True), (2, 600, 40, 20, 2, False)])
def test_fit_finishes_exact_epoch_then_scores_full_holdout(tmp_path, monkeypatch, requested, limit, training_cost, evaluation_cost, actual, extended):
    from research.graph_temporal import batching, budget, data, registry, run

    clock = SimpleNamespace(now=0.0)
    fake_time = SimpleNamespace(perf_counter=lambda: clock.now)
    monkeypatch.setattr(run, 'time', fake_time)
    monkeypatch.setattr(budget, 'time', fake_time)
    cache = tmp_path / 'cache'
    cache.mkdir()
    supervision = cache / 'supervision.json'
    supervision.write_text('{}')
    digest = hashlib.sha256(supervision.read_bytes()).hexdigest()
    training = [(0, 0, 2), (0, 2, 4)]
    bank = SimpleNamespace(manifest=dict(categories=['test'], supervision_sha256=digest),
        split=dict(train_indices=[0, 1, 2, 3], validation_indices=[4, 5], reward_mean=0.0, reward_sd=1.0),
        targets=np.arange(6, dtype=np.float32), partitions=np.array([0, 0, 0, 0, 1, 1]),
        rows=[dict(campaign='train' if i < 4 else 'held') for i in range(6)],
        blocks=lambda partition, history: training if partition == 0 else [(1, 0, 2)])
    monkeypatch.setattr(data, 'Bank', lambda _: bank)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def to(self, device):
            return self

        def forward(self, batch):
            return self.layer(batch['x']).flatten()

    seen = []

    def pack(bank, blocks, cfg, device):
        partition = blocks[0][0]
        clock.now += evaluation_cost if partition else training_cost
        rows = np.array([campaign * 4 + row for campaign, start, stop in blocks for row in range(start, stop)])
        seen.append((partition, rows.tolist()))
        return dict(rows=rows, x=torch.tensor(rows, dtype=torch.float32)[:, None], target=torch.tensor(bank.targets[rows]))

    monkeypatch.setattr(registry, 'create', lambda *args: Net())
    monkeypatch.setattr(batching, 'pack_batch', pack)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    load = torch.load
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: load(path, map_location='cpu', weights_only=False))
    cfg = dict(model='M01', design='S-full', history=2, seed=7, epochs=requested, lr=0.001, weight_decay=0.0, clip=1.0)
    result = run.fit(cache, tmp_path, cfg, training, seconds=limit, batch_size=1)
    assert result['epochs'] == actual
    assert result['requested_epochs'] == result['config']['epochs'] == requested
    assert result['budget_extended'] is extended
    assert result['stop_reason'] == ('epoch_budget' if actual < requested else 'requested_epochs')
    assert result['training_examples_seen'] == actual * 4
    assert result['train_unique_rows'] == 4 and result['holdout_rows'] == 2
    assert sum(partition == 0 for partition, rows in seen) == 2 * actual
    assert [rows for partition, rows in seen if partition] == [[4, 5]]
    predictions = np.load(tmp_path / 'predictions.npz')
    np.testing.assert_array_equal(predictions['row_index'], [4, 5])
    checkpoint = load(tmp_path / 'checkpoint.pt', weights_only=False)
    assert checkpoint['epochs'] == actual and checkpoint['requested_epochs'] == requested
    assert json.loads((tmp_path / 'fit_progress.json').read_text())['phase'] == 'complete'
