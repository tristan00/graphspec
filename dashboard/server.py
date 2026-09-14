import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from graph_tuning_infra.files import JsonCache
from graph_tuning_infra.log import emit
from graph_tuning_infra.processes import command
from graph_tuning_infra.serve import serve
from tuning.model_spec import describe
from tuning.run import VERSION


class Results:

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = JsonCache()

    def read(self, path):
        return self.cache.read(path)

    def live(self, directory):
        running = self.read(directory / 'running.json') or {}
        invocation = command(running['worker_pid']) if running else []
        return running if 'tuning.run' in invocation and 'worker' in invocation and str((directory / running['trial']).resolve()) in invocation else None

    def runs(self):
        result = []
        for directory in sorted(self.root.iterdir()):
            plan = self.read(directory / 'plan.json') if directory.is_dir() else None
            if plan is None or plan.get('version') != VERSION:
                continue
            manifest = self.read(Path(plan['bank']) / 'manifest.json') or {}
            result.append(dict(name=directory.name, bank=Path(plan['bank']).name, graphs=len(manifest.get('graphs', [])), trials=len(plan['trials']),
                               budget=plan['seconds'], selector=plan['selector'], running=self.live(directory) is not None,
                               updated=datetime.fromtimestamp((directory / 'plan.json').stat().st_mtime, timezone.utc).isoformat()))
        return sorted(result, key=lambda run: run['updated'], reverse=True)

    def listing(self, query):
        started = time.perf_counter()
        runs = self.runs()
        emit('dashboard_runs', runs=len(runs), seconds=round(time.perf_counter() - started, 4))
        return dict(root=str(self.root), runs=runs)

    def snapshot(self, query):
        started = time.perf_counter()
        runs = self.runs()
        chosen = query.get('run') or (runs[0]['name'] if runs else None)
        directory = self.root / chosen if chosen else None
        plan = self.read(directory / 'plan.json') if directory and directory.parent == self.root else None
        if plan is None or plan.get('version') != VERSION:
            emit('dashboard_snapshot', run=chosen, attempts=0, seconds=round(time.perf_counter() - started, 4))
            return dict(updated=datetime.now(timezone.utc).isoformat(), rows=[], complete=0, unscored=0, best_mse=None, best_r2=None, best_model=None,
                        planned=0, limit=None, recorded=0, train_rows=0, holdout_rows=0, stop_requested=False, running=None, run=chosen, runs=runs, bank=None, graphs=None, budget=None)
        running = self.live(directory)
        live = running is not None
        rows = []
        for item in plan['trials']:
            trial = directory / item['directory']
            logs = sorted(trial.glob('worker_*.log'))
            if not logs and not (trial / 'failure.json').exists():
                continue
            logs = logs or [None]
            config = self.read(trial / 'config.json')
            selection = self.read(trial / 'selection.json')
            mode = 'TPE' if item['selector'] == 'tpe' else 'Random'
            result = self.read(trial / 'result.json')
            failure = self.read(trial / 'failure.json')
            timeout = self.read(trial / 'timeout.json')
            record = self.read(trial / 'trial.json') or {}
            for index, log in enumerate(logs):
                last = index == len(logs) - 1
                status = 'Interrupted'
                if last:
                    if result:
                        status = 'Complete'
                    elif timeout:
                        status = 'Timeout'
                    elif failure:
                        status = 'Failed'
                    elif live and running['trial'] == item['directory']:
                        status = 'Running'
                identifier = str(int(item['directory'].split('_')[1])).zfill(3)
                if len(logs) > 1:
                    identifier += chr(97 + index)
                row = self.row(identifier, describe(config['model']), item['graph'][:10], mode,
                    status, config, selection['graph_summary'], result if last and result else None)
                row.update(trial=item['directory'], log=str(log) if log else None, model_id=item['model'][:10])
                if last and record:
                    row.update(seconds=record['seconds'], phases=record['phases'])
                if last and not result:
                    row['error'] = failure
                    row['seconds'] = timeout['seconds'] if timeout else row.get('seconds')
                    if status == 'Running':
                        row['seconds'] = (datetime.now(timezone.utc) - datetime.fromisoformat(running['started'])).total_seconds()
                rows.append(row)
        rows.sort(key=lambda row: (row['mse'] is None, row['mse'] if row['mse'] is not None else 0, row['id']))
        scored = [row for row in rows if row['mse'] is not None]
        for rank, row in enumerate(scored, 1):
            row['rank'] = rank
        manifest = self.read(Path(plan['bank']) / 'manifest.json') or {}
        output = dict(updated=datetime.now(timezone.utc).isoformat(), rows=rows,
            complete=len(scored), unscored=sum(row['mse'] is None and row['status'] != 'Running' for row in rows),
            best_mse=scored[0]['mse'] if scored else None, best_r2=scored[0]['r2'] if scored else None,
            best_model=scored[0]['model'] if scored else None, planned=len(plan['trials']), limit=plan.get('limit'),
            recorded=sum(row['status'] in ('Complete', 'Timeout', 'Failed') for row in rows),
            train_rows=plan['train_rows'], holdout_rows=plan['holdout_rows'],
            stop_requested=(directory / 'STOP').exists(),
            running=running if live else None, run=chosen, runs=runs, bank=plan['bank'],
            graphs=len(manifest.get('graphs', [])), budget=plan['seconds'])
        emit('dashboard_snapshot', run=chosen, attempts=len(rows), seconds=round(time.perf_counter() - started, 4))
        return output

    def row(self, identifier, model, graph, mode, status, config, summary, result=None):
        result = result or {}
        spec = config['model']
        return dict(id=identifier, model=model, view=spec['view'], graph=graph, mode=mode, status=status, config=config,
            model_params=spec, graph_params=summary, training={key: value for key, value in config.items() if key not in ('graph', 'model')},
            mse=result.get('raw_val_mse'), r2=result.get('val_r2'),
            seconds=result.get('seconds'), parameters=result.get('parameters'),
            epochs=result.get('epochs'), requested_epochs=result.get('requested_epochs', config.get('epochs')),
            epoch_seconds=result.get('epoch_seconds'), completed=result.get('completed'),
            train_rows=result.get('train_unique_rows'), holdout_rows=result.get('holdout_rows'),
            timings=result.get('timings'), source_sha256=result.get('source_sha256'),
            supervision_sha256=result.get('supervision_sha256'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='artifacts')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    results = Results(args.root)
    serve(args.port, Path(__file__).with_name('index.html'), {'/api/results': results.snapshot, '/api/runs': results.listing}, root=str(results.root))


if __name__ == '__main__':
    main()
