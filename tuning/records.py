import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from graphspec.infra.files import JsonCache, sha256_json
from graphspec.infra.processes.processes import command
from graphspec.infra.timing import timed

from . import objective

VERSION = 9


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"


OUTCOMES = {
    Status.COMPLETE: "result.json",
    Status.FAILED: "failure.json",
    Status.TIMEOUT: "timeout.json",
    Status.INTERRUPTED: "interrupted.json",
}


def child(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError(f"trial directory must be a direct child of its run: {name}")
    return path


@timed("records.worker_alive")
def worker_alive(directory: Path, running: dict) -> bool:
    invocation = command(running["worker_pid"])
    expected = child(directory, running["trial"])
    if "tuning.run" not in invocation or "worker" not in invocation or "--output" not in invocation:
        return False
    position = invocation.index("--output") + 1
    return position < len(invocation) and Path(invocation[position]).resolve() == expected


@timed("records.validate_result")
def validate_result(result: dict, item: dict, config: dict, plan: dict):
    identity = (
        result["config"] == config
        and result["model"] == item["model"]
        and result["graph"] == item["graph"]
        and result["supervision_sha256"] == plan["supervision_sha256"]
    )
    epochs = result["epochs"] == result["requested_epochs"] == config["epochs"] and config["epochs"] > 0
    coverage = (
        result["training_coverage_complete"]
        and result["train_unique_rows"] == result["train_available_rows"] == plan["train_rows"]
        and result["holdout_rows"] == plan["holdout_rows"]
        and result["training_examples_seen"] == result["epochs"] * plan["train_rows"]
    )
    score = objective.value(result, plan)
    metrics = math.isfinite(score) and score >= 0
    if not (identity and epochs and coverage and metrics):
        raise ValueError(
            "completed trial violates its configuration or scoring contract: " + item["directory"]
        )


@dataclass(frozen=True)
class Outcome:
    status: Status
    path: Path
    data: dict

    @classmethod
    @timed("records.outcome.read")
    def read(cls, directory: Path, cache: JsonCache):
        found = []
        for status, name in OUTCOMES.items():
            path = directory / name
            data = cache.read(path)
            if data is not None:
                found.append(cls(status, path, data))
        if len(found) > 1:
            raise ValueError(f"conflicting trial outcomes in {directory}")
        if not found:
            return None
        outcome = found[0]
        if outcome.status == Status.COMPLETE and outcome.data["status"] != Status.COMPLETE:
            raise ValueError(f"result is not complete: {outcome.path}")
        return outcome


@dataclass(frozen=True)
class Trial:
    index: int
    directory: Path
    item: dict
    config: dict
    selection: dict
    record: dict | None
    outcome: Outcome | None
    status: Status
    running: dict | None

    @property
    def parameters(self):
        values = dict(self.config)
        budget = self.record.get("budget") if self.record else None
        if budget is not None:
            if "max_seconds" in values and values["max_seconds"] != budget:
                raise ValueError("trial time limit differs from its recorded budget: " + str(self.directory))
            values["max_seconds"] = budget
        if "max_seconds" in values and (type(values["max_seconds"]) is not int or values["max_seconds"] < 1):
            raise ValueError("max_seconds must be a positive integer: " + str(self.directory))
        return values


class Run:
    @timed("records.run")
    def __init__(self, directory: Path, cache: JsonCache):
        self.directory = Path(directory).resolve()
        self.cache = cache
        self.plan = cache.require(self.directory / "plan.json")
        if self.plan["version"] != VERSION:
            raise ValueError(f"unsupported run version {self.plan['version']}: {self.directory}")
        required = {
            "bank",
            "trials",
            "limit",
            "seconds",
            "selector",
            "train_rows",
            "holdout_rows",
            "objective",
        }
        missing = required - self.plan.keys()
        if missing:
            raise ValueError(f"run plan lacks {sorted(missing)}: {self.directory}")
        names = [item["directory"] for item in self.plan["trials"]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate trial directories in {self.directory}")
        for name in names:
            child(self.directory, name)

    @timed("records.run.live")
    def live(self) -> dict | None:
        running = self.cache.read(self.directory / "running.json")
        if running is None or not worker_alive(self.directory, running):
            return None
        return running

    @timed("records.run.runner")
    def runner(self) -> dict | None:
        state = self.cache.read(self.directory / "runner.json")
        if state is None or state["status"] != Status.RUNNING:
            return state
        invocation = command(state["pid"])
        module = "tuning.run" in invocation or ("graphspec" in invocation and "search" in invocation)
        if module and "run" in invocation and "--output" in invocation:
            index = invocation.index("--output") + 1
            if index < len(invocation) and Path(invocation[index]).resolve() == self.directory:
                return state
        return dict(state, status=Status.INTERRUPTED, error="Runner exited without a terminal record")

    @timed("records.run.trials")
    def trials(self, *, selection=True, name=None) -> list[Trial]:
        running = self.live()
        result = []
        for index, item in enumerate(self.plan["trials"]):
            if name is not None and item["directory"] != name:
                continue
            directory = child(self.directory, item["directory"])
            config = self.cache.require(directory / "config.json")
            if sha256_json(config["model"]) != item["model"] or sha256_json(config["graph"]) != item["graph"]:
                raise ValueError("trial configuration differs from its saved plan: " + str(directory))
            outcome = Outcome.read(directory, self.cache)
            if outcome is not None and outcome.status == Status.COMPLETE:
                validate_result(outcome.data, item, config, self.plan)
            record = self.cache.read(directory / "trial.json")
            active = running is not None and running["trial"] == item["directory"]
            if outcome is not None:
                status = outcome.status
            elif active:
                status = Status.RUNNING
            elif record is not None or any(directory.glob("worker*.log")):
                status = Status.INTERRUPTED
            else:
                status = Status.PENDING
            result.append(
                Trial(
                    index,
                    directory,
                    item,
                    config,
                    self.cache.require(directory / "selection.json") if selection else {},
                    record,
                    outcome,
                    status,
                    running if active else None,
                )
            )
        return result
