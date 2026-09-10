import json
from pathlib import Path
import subprocess
import time

from .cache import write_json


EPOCH_LIMIT = 120.0
STALL_LIMIT = 600.0
POLICY = dict(version=1, previous_epoch_mean_below_seconds=EPOCH_LIMIT,
    after_deadline='Finish the current epoch, then evaluate the complete holdout.',
    no_progress_timeout_seconds=STALL_LIMIT)


def eligible(durations):
    return bool(durations) and all(0 <= value < float('inf') for value in durations) and sum(durations) / len(durations) < EPOCH_LIMIT


class FitBudget:
    def __init__(self, directory, seconds, deadline=None):
        self.path = Path(directory) / 'fit_progress.json'
        self.deadline = time.perf_counter() + seconds if deadline is None else deadline
        self.epoch_seconds = []
        self.extended = False
        self.grace_epoch_seconds = []
        self.phase = 'setup'
        self.pulse('setup')

    def check(self):
        if time.perf_counter() < self.deadline:
            return False
        if not self.extended:
            if not eligible(self.epoch_seconds):
                raise TimeoutError('fit deadline reached without completed epochs averaging under 120 seconds')
            self.extended = True
            self.grace_epoch_seconds = list(self.epoch_seconds)
            self.pulse(self.phase)
            print(json.dumps(dict(event='epoch_budget_extension', epoch_seconds=self.grace_epoch_seconds,
                mean_epoch_seconds=sum(self.grace_epoch_seconds) / len(self.grace_epoch_seconds))), flush=True)
        return True

    def pulse(self, phase):
        self.phase = phase
        write_json(self.path, dict(deadline=self.deadline, updated=time.perf_counter(), phase=phase,
            epoch_seconds=self.epoch_seconds, budget_extended=self.extended,
            grace_epoch_seconds=self.grace_epoch_seconds))


def wait_worker(process, directory, deadline):
    try:
        return process.wait(timeout=max(0.0, deadline - time.perf_counter()))
    except subprocess.TimeoutExpired:
        pass
    path = Path(directory) / 'fit_progress.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    durations = state.get('grace_epoch_seconds') if state.get('budget_extended') else state.get('epoch_seconds', [])
    if state.get('deadline') != deadline or not eligible(durations):
        raise subprocess.TimeoutExpired(process.args, 0)
    print(json.dumps(dict(event='supervisor_epoch_extension', mean_epoch_seconds=sum(durations) / len(durations),
        completed_epochs=len(durations), phase=state['phase'])), flush=True)
    while True:
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state = json.loads(path.read_text())
            if state['deadline'] != deadline or time.perf_counter() - state['updated'] >= STALL_LIMIT:
                raise
