import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import psutil

from totalwar.registry import NAMES
from totalwar.selector import space, structure
from totalwar.turns import graph_name


class Results:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.cache = {}

    def read(self, path):
        path = Path(path)
        if not path.exists():
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if path not in self.cache or self.cache[path][0] != key:
            self.cache[path] = (key, json.loads(path.read_text(encoding='utf-8')))
        return self.cache[path][1]

    def snapshot(self):
        started = time.perf_counter()
        plan = self.read(self.directory / 'plan.json')
        running = self.read(self.directory / 'running.json') or {}
        live = False
        if running:
            try:
                command = psutil.Process(running['worker_pid']).cmdline()
                live = ('totalwar.run' in command and 'worker' in command
                    and str((self.directory / running['trial']).resolve()) in command)
            except psutil.NoSuchProcess:
                pass
        rows = []
        baseline = self.read(plan['baseline']) if plan.get('baseline') else None
        if baseline:
            if baseline['supervision_sha256'] != plan['supervision_sha256']:
                raise ValueError('Baseline uses a different split')
            rows.append(self.row('CB', 'Greedy CatBoost', 'Tabular', 'Fixed', 'Complete', baseline['config'], baseline))
            rows[-1]['active_params'] = dict(baseline['config'], fitted_trees=baseline['iterations'],
                early_stopping_rounds=baseline['early_stopping_rounds'])
            log = Path(plan['baseline']).parent.with_suffix('.log')
            if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8'):
                rows.append(self.row('CB-first', 'Greedy CatBoost', 'Tabular', 'Fixed', 'Prediction error', baseline['config']))
                rows[-1]['log'] = str(log)
        for item in plan['trials']:
            trial = self.directory / item['directory']
            logs = sorted(trial.glob('worker_*.log'))
            if not logs:
                continue
            config = self.read(trial / 'config.json')
            selection = self.read(trial / 'selection.json') or {}
            method = 'Random' if item.get('selector', 'random') == 'random' else 'TPE'
            ranges = {1: 'small', 2: 'small–medium', 3: 'expanded', 4: 'expanded + turns', 5: 'targeted ranges'}
            mode = method + ' · ' + ranges[selection.get('parameter_range_version', 1)]
            if selection.get('random_parameter_range') == 'targeted':
                mode = 'Random · targeted ranges'
            if method == 'TPE' and selection.get('structure_parameterization', {}).get('version') == 2:
                mode += ' · graph switches'
            result = self.read(trial / 'result.json')
            failure = self.read(trial / 'failure.json')
            timeout = self.read(trial / 'timeout.json')
            for index, log in enumerate(logs):
                last = index == len(logs) - 1
                actual = config
                with log.open(encoding='utf-8') as stream:
                    first = stream.readline()
                if first.startswith('{') and first.endswith('\n'):
                    actual = json.loads(first).get('config', config)
                status = 'Interrupted'
                if last:
                    if result:
                        status = 'Complete · budget stop' if result.get('stop_reason') == 'epoch_budget' else 'Complete'
                    elif timeout:
                        status = 'Timeout'
                    elif failure:
                        status = 'Interrupted' if failure.get('status') == 'interrupted' else 'Failed'
                    elif live and running['trial'] == item['directory']:
                        status = 'Running'
                identifier = str(int(item['directory'].split('_')[1])).zfill(3)
                if len(logs) > 1:
                    identifier += chr(97 + index)
                row = self.row(identifier, NAMES[int(actual['model'][1:]) - 1], graph_name(actual), mode,
                    status, actual, result if last and result else None)
                row.update(trial=item['directory'], log=str(log), model_id=actual['model'])
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
            train_rows=plan.get('train_rows'), holdout_rows=plan.get('holdout_rows'),
            range=plan.get('parameter_range'), stop_requested=(self.directory / 'STOP').exists(),
            running=running if live else None, directory=str(self.directory))
        exploration = plan.get('random_exploration')
        if exploration:
            attempts = plan['trials'][exploration['start_index']:exploration['start_index'] + exploration['count']]
            finished = sum(any((self.directory / item['directory'] / name).exists()
                for name in ('result.json', 'failure.json', 'timeout.json')) for item in attempts)
            output['exploration'] = dict(finished=finished, total=exploration['count'],
                requested_total=exploration.get('requested_count', exploration['count']), ended_early=exploration.get('ended_early', False),
                phase='Random' if plan.get('parameter_selection') == 'random' or finished < exploration['count'] else 'TPE')
        print(json.dumps(dict(event='dashboard_snapshot', attempts=len(rows), seconds=round(time.perf_counter() - started, 4))), flush=True)
        return output

    def row(self, identifier, model, graph, mode, status, config, result=None):
        result = result or {}
        active = config if mode == 'Fixed' else {key: config[key] for key in space(config['model'], config.get('turn_context', 'none')) if key in config}
        if mode != 'Fixed':
            active.update({key:value for key,value in structure(config)[0].items() if key != 'model'})
        return dict(id=identifier, model=model, graph=graph, mode=mode, status=status, config=config,
            active_params=active, mse=result.get('raw_val_mse'), r2=result.get('val_r2'),
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
    page = Path(__file__).with_name('index.html')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path in ('/', '/index.html'):
                    body, content_type = page.read_bytes(), 'text/html; charset=utf-8'
                elif path == '/api/results':
                    body = json.dumps(results.snapshot(), allow_nan=False).encode('utf-8')
                    content_type = 'application/json; charset=utf-8'
                elif path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404)
                    return
            except Exception as error:
                print(json.dumps(dict(event='dashboard_error', error=repr(error))), flush=True)
                self.send_error(503, 'Results could not be read; retry shortly')
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps(dict(event='dashboard_listening', url='http://127.0.0.1:%d' % args.port,
        directory=str(results.directory))), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
