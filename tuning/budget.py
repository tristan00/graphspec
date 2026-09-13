from pathlib import Path
import time

from graph_tuning_infra.processes import Heartbeat, wait


POLICY = dict(version=2, at_deadline='The trial is stopped. Variant preparation, packing, fitting and scoring all count.')


class FitBudget:

    def __init__(self, directory, deadline):
        self.deadline = deadline
        self.heartbeat = Heartbeat(Path(directory) / 'fit_progress.json', deadline)
        self.epoch_seconds = []
        self.phase = 'setup'
        self.pulse('setup')

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('trial deadline reached')

    def pulse(self, phase):
        self.phase = phase
        self.heartbeat.pulse(phase=phase, epoch_seconds=self.epoch_seconds)


def wait_worker(process, deadline):
    return wait(process, deadline)
