from dataclasses import dataclass

from graphspec.infra.timing import timed


@dataclass(frozen=True)
class BuildPlan:
    memory_bytes: int
    workers: int
    shards: int = 1

    @timed("planning.checked")
    def __post_init__(self):
        if type(self.memory_bytes) is not int or self.memory_bytes < 1:
            raise ValueError("a build plan states memory_bytes as a positive integer")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("a build plan states workers as a positive integer")
        if type(self.shards) is not int or self.shards < 1:
            raise ValueError("a build plan states shards as a positive integer")
        if self.memory_bytes < self.workers * 4096:
            raise ValueError("a build plan provides at least 4096 bytes per worker")

    @property
    @timed("planning.worker_bytes")
    def worker_bytes(self):
        return self.memory_bytes // self.workers

    @timed("planning.mapping")
    def mapping(self):
        return dict(memory_bytes=self.memory_bytes, workers=self.workers, shards=self.shards)


@timed("planning.stated")
def stated(plan):
    if isinstance(plan, BuildPlan):
        return plan
    if not isinstance(plan, dict) or not {"memory_bytes", "workers"} <= set(plan) or set(plan) - {"memory_bytes", "workers", "shards"}:
        raise ValueError("state a BuildPlan with memory_bytes and workers")
    return BuildPlan(**plan)
