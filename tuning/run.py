import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

import graph_tuning_infra
from graphspec.spec import digest
from graph_tuning_infra import compute, sources
from graph_tuning_infra.files import BUDGET, read_json, sha256_bytes, sha256_file, write_csv, write_json, write_jsonl
from graph_tuning_infra.log import configure, emit
from graph_tuning_infra.processes import WORKERS, Lock, command, stop


VERSION = 7
REFERENCE_HISTORY = 16
PACKAGES = (Path(__file__).parent, Path(__file__).parent.parent / 'graphspec', Path(graph_tuning_infra.__file__).parent)


def score(targets, predictions):
    started = time.perf_counter()
    targets = np.asarray(targets, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or targets.shape != values.shape:
        raise ValueError('reward targets and predictions must be matching vectors')
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError('validation predictions must be nonempty and finite')
    mse = float(np.mean((values - targets)**2))
    emit('score', seconds=time.perf_counter() - started, rows=len(values), raw_mse=mse)
    return dict(raw_val_mse=mse, val_r2=1 - mse / float(targets.var()))


def fit(bank_directory, directory, config, training_blocks, batch_size, deadline):
    import torch
    from .data import Bank
    from .batching import batches, pack_batch
    from .roster import create
    from .turns import blocks as turn_blocks
    from .budget import FitBudget, POLICY

    entered = time.perf_counter()
    revision = sources.digest(PACKAGES)
    cfg, directory = config, Path(directory)
    graph, history = digest(cfg['graph']), cfg['graph']['time/history']
    budget = FitBudget(directory, deadline)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = compute.DEVICE
    if not torch.cuda.is_available():
        raise RuntimeError('this research run requires the available CUDA GPU')
    compute.reserve()
    bank = Bank(bank_directory)
    actual_hash = sha256_file(Path(bank_directory) / 'supervision.json')
    if actual_hash != bank.manifest['supervision_sha256']:
        raise ValueError('frozen supervision changed')
    model = create(cfg, bank.registry, len(bank.categories)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    torch.cuda.reset_peak_memory_stats()
    timing = dict(setup=time.perf_counter() - entered, train_pack=0.0, train_compute=0.0, validation_pack=0.0, validation_compute=0.0)
    losses, trained_rows, epoch_losses = [], [], []
    if sorted(map(tuple, training_blocks)) != sorted(bank.blocks(0, history)):
        raise ValueError('training blocks must cover the entire frozen training partition')
    epochs = cfg['epochs']
    if epochs < 1:
        raise ValueError('each fit requires at least one complete training epoch')
    context_blocks = turn_blocks(bank, training_blocks, cfg)
    grouped = batches(context_blocks, batch_size, bank.entities, frame_budget=1024)
    training = [(epoch, group) for epoch in range(epochs) for group in batches(context_blocks, batch_size, bank.entities, cfg['seed'] + epoch, frame_budget=1024)]
    emit('fit_enter', model=cfg['model'], graph=graph[:10], updates=len(training), epochs=epochs,
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
            emit('epoch_complete', epoch=epoch + 1, rows=len(epoch_rows), loss=epoch_losses[-1],
                epoch_seconds=budget.epoch_seconds[-1], mean_epoch_seconds=float(np.mean(budget.epoch_seconds)), seconds=finished - entered)
            epoch_rows, epoch_sum = [], 0.0
            epoch_started = finished
            budget.pulse('training')
        elif (update + 1) % 8 == 0:
            budget.pulse('training')
        if (update + 1) % 16 == 0:
            emit('training_progress', updates=update + 1, total_updates=len(training), epoch=epoch + 1,
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
    validation = batches(turn_blocks(bank, bank.blocks(1, history), cfg), batch_size, bank.entities, frame_budget=1024)
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
                clone = create(loaded['config'], bank.registry, len(bank.categories)).to(device).eval()
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
                emit('validation_progress', blocks=index + 1, total_blocks=len(validation), rows=len(indices), seconds=time.perf_counter() - entered)
            del b, predictions
    ordering = np.argsort(indices)
    indices, values = np.asarray(indices, np.int64)[ordering], np.asarray(values, np.float64)[ordering]
    if not np.array_equal(indices, np.asarray(bank.split['validation_indices'])):
        raise ValueError('holdout predictions do not exactly cover the frozen partition')
    metrics = score(bank.targets[indices].astype(np.float64), values)
    np.savez_compressed(directory / 'predictions.npz', row_index=indices, prediction=values, target=bank.targets[indices])
    baseline = float(np.mean((bank.targets[indices].astype(np.float64) - bank.split['reward_mean'])**2))
    budget.check()
    result = dict(status='complete', model=cfg['model'], graph=graph, config=cfg, **metrics,
        baseline_mse=baseline, seconds=time.perf_counter() - entered, timings=timing,
        parameters=sum(p.numel() for p in model.parameters()), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=len(losses), epochs=completed_epochs, requested_epochs=epochs,
        epoch_seconds=budget.epoch_seconds, stop_reason='epoch_budget' if completed_epochs < epochs else 'requested_epochs',
        stopped_early=completed_epochs < epochs,
        budget_extended=budget.extended, grace_epoch_seconds=budget.grace_epoch_seconds, budget_policy=POLICY,
        epoch_losses=epoch_losses, batch_sequences=batch_size,
        training_coverage_complete=True, training_examples_seen=len(trained_rows),
        train_unique_rows=len(set(trained_rows)), train_available_rows=len(bank.split['train_indices']),
        train_graphs=len({bank.rows[i]['graph'] for i in trained_rows}), holdout_rows=len(indices),
        training_losses=losses, checkpoint_reload=True, supervision_sha256=actual_hash, source_sha256=revision,
        prediction_rows_sha256=sha256_bytes(indices.tobytes()),
        completed=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'result.json', result)
    budget.pulse('complete')
    emit('fit_exit', model=cfg['model'], graph=graph[:10], raw_mse=metrics['raw_val_mse'], seconds=result['seconds'], train_rows=result['train_unique_rows'])
    return result


def worker(args):
    directory = Path(args.output)
    config = read_json(directory / 'config.json')
    blocks = read_json(args.training_blocks)
    try:
        fit(args.bank, directory, config, blocks, args.batch_sequences, args.deadline)
    except Exception as error:
        write_json(directory / 'failure.json', dict(model=config['model'], graph=digest(config['graph']),
            error=repr(error), traceback=traceback.format_exc(), source_sha256=sources.digest(PACKAGES)))
        raise


def report(directory):
    from .roster import NAMES

    started = time.perf_counter()
    directory = Path(directory)
    planned = read_json(directory / 'plan.json')
    results = []
    for item in planned['trials']:
        trial = directory / item['directory']
        path = next((trial / name for name in ('result.json', 'failure.json', 'timeout.json') if (trial / name).exists()), None)
        if path is None:
            continue
        result = read_json(path)
        result.setdefault('status', 'timeout' if path.name == 'timeout.json' else 'failed')
        result.setdefault('config', read_json(trial / 'config.json'))
        result.update(trial=item['directory'], selector=item['selector'], graph=item['graph'], model=item['model'])
        results.append(result)
    completed = [r for r in results if r['status'] == 'complete']
    baseline = None
    if planned['baseline']:
        baseline = read_json(planned['baseline'])
        if baseline['supervision_sha256'] != planned['supervision_sha256']:
            raise ValueError('reference baseline belongs to another split')
        baseline = dict(baseline, trial='fixed_baseline', selector='fixed', graph=None)
    for result in results:
        if baseline:
            result['reference_mse'] = baseline['raw_val_mse']
            if result['status'] == 'complete':
                result['mse_minus_reference'] = result['raw_val_mse'] - baseline['raw_val_mse']
    table = ([baseline] if baseline else []) + results
    columns = ['trial', 'model', 'name', 'graph', 'status', 'raw_val_mse', 'val_r2', 'baseline_mse', 'seconds',
        'parameters', 'peak_cuda_bytes', 'optimizer_updates', 'train_unique_rows', 'holdout_rows',
        'epochs', 'requested_epochs', 'stopped_early', 'stop_reason', 'budget_extended', 'training_coverage_complete', 'reference_mse', 'mse_minus_reference',
        'supervision_sha256', 'source_sha256', 'selector', 'config_json']
    rows = []
    for result in table:
        row = {key: result.get(key) for key in columns}
        row.update(name='Reference' if result['trial'] == 'fixed_baseline' else NAMES[int(result['model'][1:]) - 1], config_json=json.dumps(result['config'], sort_keys=True))
        rows.append(row)
    write_csv(directory / 'results.csv', columns, rows)
    write_jsonl(directory / 'results.jsonl', table)
    models = sorted({r['model'] for r in results})
    write_json(directory / 'coverage.json', dict(models={model: sum(r['model'] == model for r in completed) for model in models},
        completed=len(completed), failed=len(results) - len(completed), planned=len(planned['trials']),
        baseline_mse=baseline['raw_val_mse'] if baseline else None))
    selection_text = ('TPE selects each model, its parameters and its graph-transform spec from completed fits; recorded timeouts and crashes enter as failures.'
                      if planned['selector'] == 'tpe' else 'Each trial draws a uniform model, model parameters and graph-transform spec.')
    lines = ['**Graph transform and temporal model search**', '',
        'Updated ' + datetime.now(timezone.utc).isoformat() + '.', '',
        '%d completed fits; %d failed or timed out; %d planned so far. %s' % (len(completed), len(results) - len(completed), len(planned['trials']), selection_text), '']
    if baseline:
        lines.extend(['Fixed reference baseline: **MSE %.6f**, R² %.4f, %d training rows and %d holdout rows. Lower MSE is better.' % (baseline['raw_val_mse'], baseline['val_r2'], baseline['train_unique_rows'], baseline['holdout_rows']), ''])
    lines.extend(['| Model | Completed fits | Best MSE | Best graph | Difference from reference |', '|---|---:|---:|---|---:|'])
    for model in models:
        subset = [r for r in completed if r['model'] == model]
        best = min(subset, key=lambda r: r['raw_val_mse']) if subset else None
        lines.append('| %s %s | %d | %s | %s | %s |' % (model, NAMES[int(model[1:]) - 1], len(subset),
            '%.6f' % best['raw_val_mse'] if best else 'Pending', best['graph'][:10] if best else '—',
            '%+.6f' % (best['raw_val_mse'] - baseline['raw_val_mse']) if best and baseline else '—'))
    lines.extend(['', 'All trial configurations, failure records and individual scores: [results.csv](results.csv). Full records: [results.jsonl](results.jsonl).', ''])
    (directory / 'progress.md').write_text('\n'.join(lines), encoding='utf-8')
    emit('report', attempts=len(results), completed=len(completed), seconds=time.perf_counter() - started)
    return results


def run(args):
    from .data import Bank
    from .budget import POLICY
    from .selector import FAILURE_POLICY

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'plan.json'
    bank = Bank(args.bank)
    if path.exists():
        if not args.resume:
            raise FileExistsError('run exists; pass --resume to continue its pending trials')
        plan = read_json(path)
        if plan.get('version') != VERSION:
            raise ValueError('this run was written by another plan version and cannot be resumed; choose a new output')
        if Path(plan['bank']).resolve() != Path(args.bank).resolve():
            raise ValueError('resume requested with a different bank')
        if plan['spec_sha256'] != bank.manifest['spec_sha256']:
            raise ValueError('resume requested under a different graph declaration')
        if args.baseline and str(Path(args.baseline).resolve()) != plan['baseline']:
            raise ValueError('resume requested with a different baseline')
        if plan['bank_manifest_sha256'] != sha256_file(Path(args.bank) / 'manifest.json'):
            raise ValueError('resume requested with a rebuilt bank')
        plan.update(workers=args.workers, variant_bytes=args.variant_gb << 30)
    else:
        plan = dict(version=VERSION, seed=args.seed, bank=str(Path(args.bank).resolve()), selector=args.selector, trials=[],
            batch_sequences=args.batch_sequences, baseline=str(Path(args.baseline).resolve()) if args.baseline else None,
            workers=args.workers, variant_bytes=args.variant_gb << 30,
            seconds=args.seconds, train_rows=bank.manifest['train_rows'], holdout_rows=bank.manifest['validation_rows'],
            supervision_sha256=bank.manifest['supervision_sha256'], spec_sha256=bank.manifest['spec_sha256'],
            created=datetime.now(timezone.utc).isoformat())
    if plan['baseline'] and read_json(plan['baseline'])['supervision_sha256'] != plan['supervision_sha256']:
        raise ValueError('reference baseline belongs to another split')
    with Lock(directory / 'runner.lock'):
        running = directory / 'running.json'
        if running.exists():
            previous = read_json(running)
            invocation = command(previous['worker_pid'])
            if 'tuning.run' in invocation and 'worker' in invocation and '--output' in invocation:
                output = Path(invocation[invocation.index('--output') + 1])
                if output.parent.resolve() == directory.resolve():
                    raise RuntimeError('the previous worker is still running; wait for it before resuming')
        plan.update(bank_manifest_sha256=sha256_file(Path(plan['bank']) / 'manifest.json'), budget_policy=POLICY, failure_policy=FAILURE_POLICY)
        write_json(path, plan)
        execute(directory, plan, args.continuous, bank.registry)


def training_layout(directory, plan, config):
    from .data import Bank

    started = time.perf_counter()
    history = config['graph']['time/history']
    path = directory / ('train_blocks_%d.json' % history)
    if not path.exists():
        bank = Bank(plan['bank'])
        if bank.manifest['supervision_sha256'] != plan['supervision_sha256']:
            raise ValueError('training layout changed supervision')
        blocks = bank.blocks(0, history)
        write_json(path, blocks)
        emit('training_layout_created', history=history, blocks=len(blocks),
            rows=len(bank.split['train_indices']), seconds=time.perf_counter() - started)
    return path, max(1, plan['batch_sequences'] * REFERENCE_HISTORY // history)


def execute(directory, plan, continuous, registry):
    from .budget import wait_worker
    from .selector import append_trial
    from .variants import prepare, prune

    emit('run_enter', trials=len(plan['trials']), seconds_per_fit=plan['seconds'], continuous=continuous)
    index = 0
    while True:
        if (directory / 'STOP').exists():
            emit('run_stopped', reason='STOP file; current fit finished')
            break
        if index == len(plan['trials']):
            if not continuous:
                break
            append_trial(directory, plan, registry)
        item = plan['trials'][index]
        index += 1
        trial = directory / item['directory']
        if any((trial / name).exists() for name in ('result.json', 'failure.json', 'timeout.json')):
            continue
        config = read_json(trial / 'config.json')
        blocks, batch_size = training_layout(directory, plan, config)
        try:
            root = prepare(plan['bank'], config, plan['workers'])
            prune(plan['bank'], plan['variant_bytes'], root)
        except Exception as error:
            write_json(trial / 'failure.json', dict(model=item['model'], graph=item['graph'], error=repr(error), traceback=traceback.format_exc()))
            report(directory)
            emit('trial_failed', trial=item['directory'], stage='variant')
            continue
        attempt = len(list(trial.glob('worker_*.log')))
        invocation = [sys.executable, '-u', '-m', 'tuning.run', 'worker', '--bank', plan['bank'],
            '--output', str(trial.resolve()), '--training-blocks', str(blocks.resolve()), '--batch-sequences', str(batch_size)]
        emit('trial_enter', index=index, total=len(plan['trials']), model=item['model'], graph=item['graph'][:10], attempt=attempt)
        sources.snapshot(PACKAGES, trial / 'source')
        with (trial / ('worker_%02d.log' % attempt)).open('w', encoding='utf-8') as stream:
            started = time.perf_counter()
            deadline = started + plan['seconds']
            invocation.extend(['--deadline', repr(deadline)])
            process = subprocess.Popen(invocation, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(directory / 'running.json', dict(runner_pid=os.getpid(), worker_pid=process.pid, trial=item['directory'], started=datetime.now(timezone.utc).isoformat()))
            try:
                code = wait_worker(process, trial, deadline)
            except subprocess.TimeoutExpired:
                stop(process)
                write_json(trial / 'timeout.json', dict(seconds=time.perf_counter() - started, model=item['model'], graph=item['graph']))
                code = -1
        if code:
            if not any((trial / name).exists() for name in ('failure.json', 'timeout.json')):
                write_json(trial / 'failure.json', dict(model=item['model'], graph=item['graph'], error='worker exited %s' % code))
            report(directory)
            emit('trial_failed', trial=item['directory'], exit_code=code)
            continue
        result = read_json(trial / 'result.json')
        emit('trial_exit', index=index, model=item['model'], graph=item['graph'][:10], raw_mse=result['raw_val_mse'], seconds=result['seconds'])
        report(directory)
    results = report(directory)
    completed = [r for r in results if r['status'] == 'complete']
    emit('run_exit', completed=len(completed), failed=len(results) - len(completed), model_families=len({r['model'] for r in completed}))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--bank', required=True)
    run_parser.add_argument('--output', required=True)
    run_parser.add_argument('--seed', type=int, default=20260909)
    run_parser.add_argument('--batch-sequences', type=int, default=64)
    run_parser.add_argument('--continuous', action='store_true')
    run_parser.add_argument('--baseline')
    run_parser.add_argument('--seconds', type=int, default=600)
    run_parser.add_argument('--resume', action='store_true')
    run_parser.add_argument('--selector', choices=('tpe', 'random'), default='tpe')
    run_parser.add_argument('--workers', type=int, default=WORKERS)
    run_parser.add_argument('--variant-gb', type=int, default=BUDGET >> 30)
    worker_parser = subparsers.add_parser('worker')
    worker_parser.add_argument('--bank', required=True)
    worker_parser.add_argument('--output', required=True)
    worker_parser.add_argument('--training-blocks', required=True)
    worker_parser.add_argument('--batch-sequences', type=int, required=True)
    worker_parser.add_argument('--deadline', type=float, required=True)
    report_parser = subparsers.add_parser('report')
    report_parser.add_argument('directory')
    args = parser.parse_args()
    configure(json_lines=True)
    if args.command == 'worker':
        worker(args)
    elif args.command == 'run':
        run(args)
    else:
        report(args.directory)


if __name__ == '__main__':
    main()
