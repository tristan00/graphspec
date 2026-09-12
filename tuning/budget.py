from pathlib import Path
import time

from graph_tuning_infra.log import emit
from graph_tuning_infra.processes import Heartbeat, wait


EPOCH_LIMIT = 120.0
STALL_LIMIT = 600.0
POLICY = dict(version=1, previous_epoch_mean_below_seconds=EPOCH_LIMIT,
    after_deadline='Finish the current epoch, then evaluate the complete holdout.',
    no_progress_timeout_seconds=STALL_LIMIT)


def eligible(durations):
    return bool(durations) and all(0 <= value < float('inf') for value in durations) and sum(durations) / len(durations) < EPOCH_LIMIT


class FitBudget:

    def __init__(self, directory, deadline):
        self.deadline = deadline
        self.heartbeat = Heartbeat(Path(directory) / 'fit_progress.json', deadline)
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
            emit('epoch_budget_extension', epoch_seconds=self.grace_epoch_seconds,
                 mean_epoch_seconds=sum(self.grace_epoch_seconds) / len(self.grace_epoch_seconds))
        return True

    def pulse(self, phase):
        self.phase = phase
        self.heartbeat.pulse(phase=phase, epoch_seconds=self.epoch_seconds, budget_extended=self.extended,
                             grace_epoch_seconds=self.grace_epoch_seconds)


def wait_worker(process, directory, deadline):

    def may_extend(state):
        durations = state['grace_epoch_seconds'] if state['budget_extended'] else state['epoch_seconds']
        if not eligible(durations):
            return False
        emit('supervisor_epoch_extension', mean_epoch_seconds=sum(durations) / len(durations), completed_epochs=len(durations), phase=state['phase'])
        return True

    return wait(process, deadline, Path(directory) / 'fit_progress.json', may_extend, STALL_LIMIT)
