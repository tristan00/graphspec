import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from graph_tuning_infra.files import JsonCache, read_json
from graph_tuning_infra.log import emit
from graph_tuning_infra.processes import command
from graph_tuning_infra.serve import serve
from tuning.model_spec import describe


class Results:

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.cache = JsonCache()

    def read(self, path):
        return self.cache.read(path)

    def snapshot(self):
        started = time.perf_counter()
        plan = self.read(self.directory / 'plan.json')
        running = self.read(self.directory / 'running.json') or {}
        live = False
        if running:
            invocation = command(running['worker_pid'])
            live = 'tuning.run' in invocation and 'worker' in invocation and str((self.directory / running['trial']).resolve()) in invocation
        rows = []
        baseline = read_json(plan['baseline']) if plan['baseline'] else None
        if baseline:
            if baseline['supervision_sha256'] != plan['supervision_sha256']:
                raise ValueError('Baseline uses a different split')
            rows.append(self.row('REF', 'Reference', '—', 'Fixed', 'Complete', baseline['config'], {}, baseline))
        for item in plan['trials']:
            trial = self.directory / item['directory']
            logs = sorted(trial.glob('worker_*.log'))
            if not logs:
                continue
            config = self.read(trial / 'config.json')
            selection = self.read(trial / 'selection.json')
            mode = 'TPE' if item['selector'] == 'tpe' else 'Random'
            result = self.read(trial / 'result.json')
            failure = self.read(trial / 'failure.json')
            timeout = self.read(trial / 'timeout.json')
            for index, log in enumerate(logs):
                last = index == len(logs) - 1
                status = 'Interrupted'
                if last:
                    if result:
                        status = 'Complete · budget stop' if result.get('stop_reason') == 'epoch_budget' else 'Complete'
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
                row.update(trial=item['directory'], log=str(log), model_id=item['model'][:10])
                if last and not result:
                    row['error'] = failure
                    row['seconds'] = timeout['seconds'] if timeout else None
                    if status == 'Running':
                        row['seconds'] = (datetime.now(timezone.utc) - datetime.fromisoformat(running['started'])).total_seconds()
                rows.append(row)
        rows.sort(key=lambda row: (row['mse'] is None, row['mse'] if row['mse'] is not None else 0, row['id']))
        scored = [row for row in rows if row['mse'] is not None]
        neural = [row for row in scored if row['mode'] != 'Fixed']
        for rank, row in enumerate(scored, 1):
            row['rank'] = rank
        for row in rows:
            row['delta'] = row['mse'] - baseline['raw_val_mse'] if row['mse'] is not None and baseline else None
        output = dict(updated=datetime.now(timezone.utc).isoformat(), rows=rows,
            complete=len(neural), unscored=sum(row['mse'] is None and row['status'] != 'Running' for row in rows),
            baseline_mse=baseline['raw_val_mse'] if baseline else None,
            best_neural_mse=neural[0]['mse'] if neural else None,
            best_neural_model=neural[0]['model'] if neural else None,
            train_rows=plan['train_rows'], holdout_rows=plan['holdout_rows'],
            stop_requested=(self.directory / 'STOP').exists(),
            running=running if live else None, directory=str(self.directory))
        emit('dashboard_snapshot', attempts=len(rows), seconds=round(time.perf_counter() - started, 4))
        return output

    def row(self, identifier, model, graph, mode, status, config, summary, result=None):
        result = result or {}
        spec = config['model'] if isinstance(config['model'], dict) else {}
        return dict(id=identifier, model=model, view=spec.get('view', 'reference'), graph=graph, mode=mode, status=status, config=config,
            model_params=spec, graph_params=summary, training={key: value for key, value in config.items() if key not in ('graph', 'model')},
            mse=result.get('raw_val_mse'), r2=result.get('val_r2'),
            seconds=result.get('seconds'), parameters=result.get('parameters'),
            epochs=result.get('epochs'), requested_epochs=result.get('requested_epochs', config.get('epochs')),
            epoch_seconds=result.get('epoch_seconds'), stop_reason=result.get('stop_reason'),
            stopped_early=result.get('stopped_early', False),
            budget_extended=result.get('budget_extended'), completed=result.get('completed'),
            train_rows=result.get('train_unique_rows'), holdout_rows=result.get('holdout_rows'),
            timings=result.get('timings'), source_sha256=result.get('source_sha256'),
            supervision_sha256=result.get('supervision_sha256'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    results = Results(args.directory)
    serve(args.port, Path(__file__).with_name('index.html'), {'/api/results': results.snapshot}, directory=str(results.directory))


if __name__ == '__main__':
    main()
