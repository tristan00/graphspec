from collections import Counter
import json
import os
import subprocess
import sys
import time

import psutil

from totalwar.cache import write_json
from totalwar.registry import DESIGNS
from totalwar.run import append_round, report, stop_worker


def test_repeated_random_rounds_preserve_every_family(tmp_path):
    plan = dict(seed=77, rounds=0, trials=[])
    append_round(tmp_path, plan)
    first = json.loads((tmp_path / 'plan.json').read_text())
    append_round(tmp_path, plan)
    assert len(plan['trials']) == 110
    counts = Counter((t['model'], t['design']) for t in plan['trials'])
    assert counts == Counter({(model, design): 2 for model, designs in DESIGNS.items() for design in designs})
    assert plan['trials'][:55] == first['trials']
    seeds = set()
    for trial in plan['trials']:
        cfg = json.loads((tmp_path / trial['directory'] / 'config.json').read_text())
        assert cfg['epochs'] in (2, 3)
        assert cfg['hidden'] in (16, 24, 32)
        seeds.add(cfg['seed'])
    assert len(seeds) == 110


def test_report_retains_failures_and_fixed_baseline(tmp_path):
    baseline = tmp_path / 'baseline.json'
    write_json(baseline, dict(model='greedy_catboost', design='tabular', status='complete', raw_val_mse=2.0,
        val_r2=0.5, train_unique_rows=116652, holdout_rows=20387, config={}, supervision_sha256='same'))
    plan = dict(seed=77, rounds=0, trials=[], baseline=str(baseline), supervision_sha256='same')
    append_round(tmp_path, plan)
    good, bad = plan['trials'][:2]
    write_json(tmp_path / good['directory'] / 'result.json', dict(model=good['model'], design=good['design'],
        status='complete', raw_val_mse=2.5, config={}, training_coverage_complete=True))
    write_json(tmp_path / bad['directory'] / 'timeout.json', dict(model=bad['model'], design=bad['design'], seconds=600))
    results = report(tmp_path)
    assert len(results) == 2
    assert results[0]['mse_minus_catboost'] == 0.5
    assert results[1]['status'] == 'timeout' and 'raw_val_mse' not in results[1]
    coverage = json.loads((tmp_path / 'coverage.json').read_text())
    assert coverage['completed'] == 1 and coverage['failed'] == 1 and not coverage['complete']
    rows = [json.loads(line) for line in (tmp_path / 'results.jsonl').read_text().splitlines()]
    assert rows[0]['model'] == 'greedy_catboost'
    assert rows[0]['raw_val_mse'] == 2.0
    assert 'CatBoost' in (tmp_path / 'progress.md').read_text(encoding='utf-8')


def test_timeout_stops_worker_descendants(tmp_path):
    output = tmp_path / 'child.pid'
    script = tmp_path / 'worker.py'
    script.write_text('import subprocess\nimport sys\nimport time\nfrom pathlib import Path\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
        'Path(sys.argv[1]).write_text(str(child.pid))\n'
        'time.sleep(30)\n')
    process = subprocess.Popen([sys.executable, str(script), str(output)],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        deadline = time.perf_counter() + 10
        while not output.exists() and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert output.exists()
        child = psutil.Process(int(output.read_text()))
        stop_worker(process)
        assert process.poll() is not None
        assert not child.is_running()
    finally:
        if process.poll() is None:
            stop_worker(process)
