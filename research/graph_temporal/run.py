import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

from .cache import write_json


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def timed_event(event, **fields):
    print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)), flush=True)


def stop_worker(process):
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait()
    _, alive = psutil.wait_procs(descendants, timeout=5)
    if alive:
        raise RuntimeError('timed-out worker still has running descendants')


def fit(cache, directory, config, training_blocks, seconds=600, batch_size=64, deadline=None):
    import torch
    from advisor.reward_data import score
    from .data import Bank
    from .batching import batches, pack_batch
    from .registry import create
    from .turns import blocks as turn_blocks
    from .budget import FitBudget, POLICY

    entered = time.perf_counter()
    revision = source_hash()
    cfg, directory = config, Path(directory)
    budget = FitBudget(directory, seconds, deadline)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('this research run requires the available CUDA GPU')
    bank = Bank(cache)
    actual_hash = hashlib.sha256((Path(cache) / 'supervision.json').read_bytes()).hexdigest()
    if actual_hash != bank.manifest['supervision_sha256']:
        raise ValueError('frozen supervision changed')
    model = create(cfg, len(bank.manifest['categories'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    torch.cuda.reset_peak_memory_stats()
    timing = dict(setup=time.perf_counter() - entered, train_pack=0.0, train_compute=0.0, validation_pack=0.0, validation_compute=0.0)
    losses, trained_rows, epoch_losses = [], [], []
    if sorted(map(tuple, training_blocks)) != sorted(bank.blocks(0, cfg['history'])):
        raise ValueError('training blocks must cover the entire frozen training partition')
    epochs = cfg['epochs']
    if epochs < 1:
        raise ValueError('each fit requires at least one complete training epoch')
    context_blocks = turn_blocks(bank, training_blocks, cfg)
    grouped = batches(context_blocks, batch_size, frame_budget=1024)
    training = [(epoch, group) for epoch in range(epochs) for group in batches(context_blocks, batch_size, cfg['seed'] + epoch, frame_budget=1024)]
    timed_event('fit_enter', model=cfg['model'], design=cfg['design'], updates=len(training), epochs=epochs,
        training_rows=len(bank.split['train_indices']), batch_sequences=batch_size, config=cfg)

    model.train()
    epoch_rows, epoch_sum = [], 0.0
    epoch_started = time.perf_counter()
    budget.pulse('training')
    for update, (epoch, blocks) in enumerate(training):
        if budget.check() and not epoch_rows:
            break
        started = time.perf_counter()
        b = pack_batch(bank, blocks, cfg, device)
        timing['train_pack'] += time.perf_counter() - started
        if np.any(bank.partitions[b['rows']] != 0):
            raise ValueError('holdout row reached the optimizer')
        trained_rows.extend(b['rows'].tolist())
        epoch_rows.extend(b['rows'].tolist())
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(b)
        targets = (b['target'] - bank.split['reward_mean']) / bank.split['reward_sd']
        loss = (predictions - targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        timing['train_compute'] += time.perf_counter() - started
        losses.append(float(loss.detach()))
        epoch_sum += losses[-1] * len(b['rows'])
        budget.check()
        if (update + 1) % len(grouped) == 0:
            if not np.array_equal(np.sort(epoch_rows), np.asarray(bank.split['train_indices'])):
                raise ValueError('epoch omitted or duplicated training rows')
            epoch_losses.append(epoch_sum / len(epoch_rows))
            finished = time.perf_counter()
            budget.epoch_seconds.append(finished - epoch_started)
            timed_event('epoch_complete', epoch=epoch + 1, rows=len(epoch_rows), loss=epoch_losses[-1],
                epoch_seconds=budget.epoch_seconds[-1], mean_epoch_seconds=float(np.mean(budget.epoch_seconds)), seconds=finished - entered)
            epoch_rows, epoch_sum = [], 0.0
            epoch_started = finished
            budget.pulse('training')
        elif (update + 1) % 8 == 0:
            budget.pulse('training')
        if (update + 1) % 16 == 0:
            timed_event('training_progress', updates=update + 1, total_updates=len(training), epoch=epoch + 1,
                epoch_rows=len(epoch_rows), loss=float(np.mean(losses[-16:])), seconds=time.perf_counter() - entered)
        del b, predictions, targets, loss
        if not epoch_rows and budget.extended:
            break
    completed_epochs = len(epoch_losses)
    if epoch_rows or completed_epochs < 1:
        raise ValueError('evaluation requires complete training epochs')
    budget.check()
    budget.pulse('evaluation')
    model.eval()
    checkpoint = dict(config=cfg, state_dict=model.state_dict(), reward_mean=bank.split['reward_mean'],
        reward_sd=bank.split['reward_sd'], supervision_sha256=actual_hash, source_sha256=revision,
        train_blocks=training_blocks, train_rows=sorted(set(trained_rows)),
        epochs=completed_epochs, requested_epochs=epochs, epoch_seconds=budget.epoch_seconds)
    torch.save(checkpoint, directory / 'checkpoint.pt')
    values, indices = [], []
    validation = batches(turn_blocks(bank, bank.blocks(1, cfg['history']), cfg), batch_size, frame_budget=1024)
    with torch.inference_mode():
        for index, blocks in enumerate(validation):
            budget.check()
            started = time.perf_counter()
            b = pack_batch(bank, blocks, cfg, device)
            timing['validation_pack'] += time.perf_counter() - started
            started = time.perf_counter()
            predictions = model(b)
            if index == 0:
                loaded = torch.load(directory / 'checkpoint.pt', map_location=device, weights_only=False)
                clone = create(loaded['config'], len(bank.manifest['categories'])).to(device).eval()
                clone.load_state_dict(loaded['state_dict'], strict=True)
                torch.testing.assert_close(predictions, clone(b), atol=2e-5, rtol=2e-5)
                del loaded, clone
            if not torch.isfinite(predictions).all():
                raise FloatingPointError('nonfinite holdout predictions')
            raw = predictions * bank.split['reward_sd'] + bank.split['reward_mean']
            values.extend(raw.cpu().tolist())
            indices.extend(b['rows'].tolist())
            timing['validation_compute'] += time.perf_counter() - started
            if (index + 1) % 8 == 0:
                budget.pulse('evaluation')
            if (index + 1) % 256 == 0:
                timed_event('validation_progress', blocks=index + 1, total_blocks=len(validation), rows=len(indices), seconds=time.perf_counter() - entered)
            del b, predictions
    ordering = np.argsort(indices)
    indices, values = np.asarray(indices, np.int64)[ordering], np.asarray(values, np.float64)[ordering]
    if not np.array_equal(indices, np.asarray(bank.split['validation_indices'])):
        raise ValueError('holdout predictions do not exactly cover the frozen partition')
    metrics = score(bank.targets[indices].astype(np.float64), values)
    np.savez_compressed(directory / 'predictions.npz', row_index=indices, prediction=values, target=bank.targets[indices])
    baseline = float(np.mean((bank.targets[indices].astype(np.float64) - bank.split['reward_mean'])**2))
    budget.check()
    result = dict(status='complete', model=cfg['model'], design=cfg['design'], config=cfg, **metrics,
        baseline_mse=baseline, seconds=time.perf_counter() - entered, timings=timing,
        parameters=sum(p.numel() for p in model.parameters()), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=len(losses), epochs=completed_epochs, requested_epochs=epochs,
        epoch_seconds=budget.epoch_seconds, stop_reason='epoch_budget' if completed_epochs < epochs else 'requested_epochs',
        stopped_early=completed_epochs < epochs,
        budget_extended=budget.extended, grace_epoch_seconds=budget.grace_epoch_seconds, budget_policy=POLICY,
        epoch_losses=epoch_losses, batch_sequences=batch_size,
        training_coverage_complete=True, training_examples_seen=len(trained_rows),
        train_unique_rows=len(set(trained_rows)), train_available_rows=len(bank.split['train_indices']),
        train_campaigns=len({bank.rows[i]['campaign'] for i in trained_rows}), holdout_rows=len(indices),
        training_losses=losses, checkpoint_reload=True, supervision_sha256=actual_hash, source_sha256=revision,
        prediction_rows_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        completed=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'result.json', result)
    budget.pulse('complete')
    timed_event('fit_exit', model=cfg['model'], design=cfg['design'], raw_mse=metrics['raw_val_mse'], seconds=result['seconds'], train_rows=result['train_unique_rows'])
    return result


def worker(args):
    directory = Path(args.output)
    config = json.loads((directory / 'config.json').read_text())
    blocks = json.loads(Path(args.training_blocks).read_text())
    try:
        fit(args.cache, directory, config, blocks, args.seconds, args.batch_sequences, args.deadline)
    except Exception as error:
        write_json(directory / 'failure.json', dict(model=config['model'], design=config['design'],
            error=repr(error), traceback=traceback.format_exc(), source_sha256=source_hash()))
        raise


def report(directory):
    from .registry import DESIGNS, NAMES
    from .turns import graph_name

    directory = Path(directory)
    planned = json.loads((directory / 'plan.json').read_text())
    results = []
    for item in planned['trials']:
        trial = directory / item['directory']
        path = next((trial / name for name in ('result.json', 'failure.json', 'timeout.json') if (trial / name).exists()), None)
        if path is not None:
            result = json.loads(path.read_text())
            result.setdefault('status', 'timeout' if path.name == 'timeout.json' else 'failed')
            result.setdefault('config', json.loads((trial / 'config.json').read_text()))
            result['graph'] = graph_name(dict(result['config'], design=result['design']))
            result['trial'] = item['directory']
            result['selector'] = item.get('selector', 'random')
            if (trial / 'selection.json').exists():
                selection = json.loads((trial / 'selection.json').read_text())
                result['selector_observations'] = selection['observations']
            results.append(result)
    completed = [r for r in results if r['status'] == 'complete']
    baseline = None
    if planned.get('baseline') and Path(planned['baseline']).exists():
        baseline = json.loads(Path(planned['baseline']).read_text())
        if baseline['supervision_sha256'] != planned['supervision_sha256']:
            raise ValueError('CatBoost baseline belongs to another split')
        baseline = dict(baseline, trial='fixed_baseline', selector='fixed')
    for result in results:
        if baseline:
            result['catboost_mse'] = baseline['raw_val_mse']
            if result['status'] == 'complete':
                result['mse_minus_catboost'] = result['raw_val_mse'] - baseline['raw_val_mse']
    table = ([baseline] if baseline else []) + results
    columns = ['trial', 'model', 'name', 'design', 'graph', 'status', 'raw_val_mse', 'val_r2', 'baseline_mse', 'seconds',
        'parameters', 'peak_cuda_bytes', 'optimizer_updates', 'train_unique_rows', 'holdout_rows',
        'epochs', 'requested_epochs', 'stopped_early', 'stop_reason', 'budget_extended', 'training_coverage_complete', 'catboost_mse', 'mse_minus_catboost',
        'supervision_sha256', 'source_sha256', 'selector', 'selector_observations', 'config_json']
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for result in table:
            row = {key: result.get(key) for key in columns}
            row.update(name='Greedy CatBoost' if result['model'] == 'greedy_catboost' else NAMES[int(result['model'][1:]) - 1], config_json=json.dumps(result['config'], sort_keys=True))
            writer.writerow(row)
    with (directory / 'results.jsonl').open('w', encoding='utf-8') as stream:
        for result in table:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
    variants = {model: [design + suffix for design in designs for suffix in ('', '+turn-end')] for model, designs in DESIGNS.items()}
    coverage = {model: {design: sum(r['model'] == model and r['graph'] == design for r in completed) for design in designs} for model, designs in variants.items()}
    write_json(directory / 'coverage.json', dict(complete=all(all(v.values()) for v in coverage.values()), models=coverage,
        completed=len(completed), failed=len(results) - len(completed), planned=len(planned['trials']),
        baseline_mse=baseline['raw_val_mse'] if baseline else None))
    selection_text = ('TPE selects the model/graph combination and its conditional parameters, using previous completed fits.'
        if planned.get('parameter_selection') == 'tpe' else 'Uniform random model, graph switches and targeted parameters continue until stopped.')
    if planned.get('random_exploration') and planned.get('parameter_selection') == 'tpe':
        selection_text = 'The saved schedule runs %d random attempts, then TPE using every completed compatible fit.' % planned['random_exploration']['count']
    if planned.get('failure_policy'):
        selection_text += ' Recorded timeouts and crashes also inform TPE as failures; their prediction metrics remain blank.'
    lines = ['**Continuous graph and temporal model research**', '',
        'Updated ' + datetime.now(timezone.utc).isoformat() + '.', '',
        '%d completed neural fits; %d failed or timed out; %d planned so far. %s' % (len(completed), len(results) - len(completed), len(planned['trials']), selection_text), '',
        'Every completed fit uses the full frozen training partition in each epoch and the complete campaign holdout. Scores are exploratory validation results, not a final test.', '']
    if baseline:
        lines.extend(['Fixed greedy CatBoost baseline: **MSE %.6f**, R² %.4f, %d training rows and %d holdout rows. Lower MSE is better.' % (baseline['raw_val_mse'], baseline['val_r2'], baseline['train_unique_rows'], baseline['holdout_rows']), ''])
    else:
        lines.extend(['The fixed greedy CatBoost baseline has not completed yet.', ''])
    if planned.get('parameter_range'):
        lines.extend(['Parameter range: **%s**, version %d. Earlier fits remain in the TPE history.' %
            (planned['parameter_range'], planned['parameter_range_version']), ''])
    lines.extend(['| Model | Graph | Completed fits | Best MSE | Difference from CatBoost |', '|---|---|---:|---:|---:|'])
    for model, designs in variants.items():
        for design in designs:
            subset = [r for r in completed if r['model'] == model and r['graph'] == design]
            best = min((r['raw_val_mse'] for r in subset), default=None)
            lines.append('| %s %s | %s | %d | %s | %s |' % (model, NAMES[int(model[1:]) - 1], design, len(subset),
                '%.6f' % best if best is not None else 'Pending',
                '%+.6f' % (best - baseline['raw_val_mse']) if best is not None and baseline else '—'))
    lines.extend(['', 'All trial configurations, failure records and individual scores: [results.csv](results.csv). Full records: [results.jsonl](results.jsonl).', ''])
    (directory / 'progress.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


def append_round(directory, plan):
    from .registry import DESIGNS, sample

    round_index = plan['rounds']
    rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], round_index]))
    deck = [(model, design) for model, designs in DESIGNS.items() for design in designs]
    rng.shuffle(deck)
    for model, design in deck:
        config = sample(model, design, rng, int(rng.integers(1, 2**31)))
        config['epochs'] = int(rng.choice([2, 3]))
        name = 'trial_%06d_%s_%s' % (len(plan['trials']), model, design)
        trial = directory / name
        trial.mkdir(exist_ok=True)
        write_json(trial / 'config.json', config)
        plan['trials'].append(dict(directory=name, model=model, design=design, round=round_index))
    plan['rounds'] += 1
    write_json(directory / 'plan.json', plan)


def run(args):
    from .data import Bank

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'plan.json'
    if path.exists():
        if not args.resume:
            raise FileExistsError('run exists; pass --resume to continue its pending trials')
        plan = json.loads(path.read_text())
        if Path(plan['cache']).resolve() != Path(args.cache).resolve():
            raise ValueError('resume requested with a different cache')
    else:
        bank = Bank(args.cache)
        training = bank.blocks(0, 16)
        write_json(directory / 'train_blocks.json', training)
        plan = dict(version=2, seed=args.seed, cache=str(Path(args.cache).resolve()), trials=[], rounds=0,
            batch_sequences=args.batch_sequences, training_blocks=len(training),
            baseline=str(Path(args.baseline).resolve()) if args.baseline else None,
            seconds=args.seconds, history=16, selection='seeded shuffled model/design coverage deck without replacement',
            training='Every training decision exactly once per epoch; complete holdout; 2/3 full epochs sampled per fit.',
            supervision_sha256=bank.manifest['supervision_sha256'], created=datetime.now(timezone.utc).isoformat())
        if args.selector == 'tpe':
            write_json(path, plan)
        else:
            append_round(directory, plan)
        del bank
    if plan.get('version') != 2:
        raise ValueError('sampled-cohort runs cannot be resumed as full-training searches; choose a new output')
    import msvcrt
    with (directory / 'runner.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        running = directory / 'running.json'
        if running.exists():
            previous = json.loads(running.read_text())
            try:
                command = psutil.Process(previous['worker_pid']).cmdline()
            except psutil.NoSuchProcess:
                command = []
            if 'research.graph_temporal.run' in command and 'worker' in command and '--output' in command:
                output = Path(command[command.index('--output') + 1])
                if output.parent.resolve() == directory.resolve():
                    raise RuntimeError('the previous worker is still running; wait for it before resuming')
        if args.selector == 'tpe':
            from .selector import activate
            activate(directory, plan)
        elif args.selector == 'random':
            from .selector import activate_random
            activate_random(directory, plan)
        if args.random_trials:
            from .selector import activate, schedule_random
            activate(directory, plan)
            schedule_random(directory, plan, args.random_trials)
        if plan.get('parameter_range_version', 0) >= 4:
            from .turns import verify
            verify(plan['cache'])
        if plan.get('graph_choices_version', 1) >= 2:
            from .graph_variants import cache_path, verify as verify_variants

            verified = verify_variants(plan['cache'])
            companion = cache_path(plan['cache'])
            plan['graph_variant_cache'] = dict(directory=str(companion.resolve()), version=verified['version'],
                manifest_sha256=hashlib.sha256((companion / 'manifest.json').read_bytes()).hexdigest())
        from .budget import POLICY
        plan['budget_policy'] = POLICY
        write_json(path, plan)
        execute(directory, plan, args.continuous)


def training_layout(directory, plan, config):
    from .data import Bank

    started = time.perf_counter()
    history = config['history']
    if history not in plan.get('histories', [plan['history']]):
        raise ValueError('trial history is outside the saved search range')
    path = directory / ('train_blocks.json' if history == plan['history'] else 'train_blocks_%d.json' % history)
    if not path.exists():
        bank = Bank(plan['cache'])
        if bank.manifest['supervision_sha256'] != plan['supervision_sha256']:
            raise ValueError('training layout changed supervision')
        blocks = bank.blocks(0, history)
        write_json(path, blocks)
        timed_event('training_layout_created', history=history, blocks=len(blocks),
            rows=len(bank.split['train_indices']), seconds=time.perf_counter() - started)
    batch_size = max(1, plan['batch_sequences'] * plan['history'] // history)
    return path, batch_size


def execute(directory, plan, continuous):
    from .budget import wait_worker

    timed_event('run_enter', trials=len(plan['trials']), seconds_per_fit=plan['seconds'], continuous=continuous)
    index = 0
    while True:
        if (directory / 'STOP').exists():
            timed_event('run_stopped', reason='STOP file; current fit finished')
            break
        if index == len(plan['trials']):
            if not continuous:
                break
            if plan.get('parameter_selection') in ('tpe', 'random'):
                from .selector import append_trial
                append_trial(directory, plan)
            else:
                append_round(directory, plan)
                timed_event('round_appended', rounds=plan['rounds'], planned=len(plan['trials']))
        item = plan['trials'][index]
        index += 1
        trial = directory / item['directory']
        if item.get('archived_pending'):
            continue
        if any((trial / name).exists() for name in ('result.json', 'failure.json', 'timeout.json')):
            continue
        config = json.loads((trial / 'config.json').read_text())
        blocks, batch_size = training_layout(directory, plan, config)
        attempt = len(list(trial.glob('worker_*.log')))
        command = [sys.executable, '-u', '-m', 'research.graph_temporal.run', 'worker', '--cache', plan['cache'],
            '--output', str(trial.resolve()), '--training-blocks', str(blocks.resolve()),
            '--seconds', str(plan['seconds']), '--batch-sequences', str(batch_size)]
        timed_event('trial_enter', index=index, total=len(plan['trials']), model=item['model'], design=item['design'], attempt=attempt)
        snapshot = trial / 'source'
        snapshot.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob('*.py'):
            (snapshot / source.name).write_bytes(source.read_bytes())
        with (trial / ('worker_%02d.log' % attempt)).open('w', encoding='utf-8') as stream:
            started = time.perf_counter()
            deadline = started + plan['seconds']
            command.extend(['--deadline', repr(deadline)])
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            write_json(directory / 'running.json', dict(runner_pid=os.getpid(), worker_pid=process.pid, trial=item['directory'], started=datetime.now(timezone.utc).isoformat()))
            try:
                code = wait_worker(process, trial, deadline)
            except subprocess.TimeoutExpired:
                stop_worker(process)
                write_json(trial / 'timeout.json', dict(seconds=time.perf_counter() - started, model=item['model'], design=item['design']))
                code = -1
        if code:
            if not any((trial / name).exists() for name in ('failure.json', 'timeout.json')):
                write_json(trial / 'failure.json', dict(model=item['model'], design=item['design'], error='worker exited %s' % code))
            report(directory)
            timed_event('trial_failed', trial=item['directory'], exit_code=code)
            continue
        result = json.loads((trial / 'result.json').read_text())
        timed_event('trial_exit', index=index, model=item['model'], design=item['design'],
            raw_mse=result['raw_val_mse'], seconds=result['seconds'])
        report(directory)
    results = report(directory)
    completed = [r for r in results if r['status'] == 'complete']
    timed_event('run_exit', completed=len(completed), failed=len(results) - len(completed), model_families=len({r['model'] for r in completed}))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--cache', required=True)
    run_parser.add_argument('--output', required=True)
    run_parser.add_argument('--seed', type=int, default=20260909)
    run_parser.add_argument('--batch-sequences', type=int, default=64)
    run_parser.add_argument('--continuous', action='store_true')
    run_parser.add_argument('--baseline')
    run_parser.add_argument('--seconds', type=int, default=600)
    run_parser.add_argument('--resume', action='store_true')
    run_parser.add_argument('--selector', choices=('random', 'tpe'))
    run_parser.add_argument('--random-trials', type=int, default=0)
    worker_parser = subparsers.add_parser('worker')
    worker_parser.add_argument('--cache', required=True)
    worker_parser.add_argument('--output', required=True)
    worker_parser.add_argument('--training-blocks', required=True)
    worker_parser.add_argument('--seconds', type=int, default=600)
    worker_parser.add_argument('--batch-sequences', type=int, default=64)
    worker_parser.add_argument('--deadline', type=float)
    report_parser = subparsers.add_parser('report')
    report_parser.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args)
    elif args.command == 'run':
        run(args)
    else:
        report(args.directory)


if __name__ == '__main__':
    main()
