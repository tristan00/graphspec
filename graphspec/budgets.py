from dataclasses import dataclass

from graphspec.infra.timing import timed


class GraphCapacityError(ValueError):
    pass


class Refused(ValueError):
    pass


@dataclass(frozen=True)
class GraphLimits:
    device: str
    max_facts: int = 200_000_000
    max_working_bytes: int = 1 << 30

    @timed("resources.graphlimits.post_init")
    def __post_init__(self):
        if type(self.device) is not str or self.device.split(":")[0] not in ("cpu", "cuda"):
            raise ValueError("the device of graph limits is %r and must be cpu, cuda or cuda:N" % self.device)
        if (
            type(self.max_facts) is not int
            or type(self.max_working_bytes) is not int
            or self.max_facts < 1
            or self.max_working_bytes < 1
        ):
            raise ValueError("graph resource limits must be positive")


class GraphBudget:
    @timed("resources.graphbudget")
    def __init__(self, limits, fact_bytes=21):
        self.limits = limits
        self.fact_bytes = fact_bytes
        self.facts = 0

    @property
    def pair_limit(self):
        remaining_bytes = self.limits.max_working_bytes - self.facts * self.fact_bytes * 2
        return max(0, min(self.limits.max_facts - self.facts, remaining_bytes // (96 + self.fact_bytes * 2)))

    def pairs(self, count, operation):
        if count > self.pair_limit:
            raise GraphCapacityError(
                "%s requires %d candidate edges; the remaining graph budget permits %d"
                % (operation, count, self.pair_limit)
            )

    def retain(self, count, operation):
        self.pairs(count, operation)
        self.facts += int(count)

    def workspace(self, size, operation):
        available = self.limits.max_working_bytes - self.facts * self.fact_bytes * 2
        if size > available:
            raise GraphCapacityError(
                "%s requires %d working bytes; the remaining graph budget permits %d"
                % (operation, size, available)
            )

    @timed("budgets.GraphBudget.dense")
    def dense(self, nodes, operation):
        self.workspace(int(nodes) * int(nodes) * 80, operation)
