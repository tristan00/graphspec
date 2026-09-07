import json
import os
from pathlib import Path
import time

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("project tests")
    group.addoption("--run-adhoc", action="store_true", help="Include data-heavy and model checks")
    group.addoption("--run-integration", action="store_true", help="Include database checks")
    group.addoption("--run-gpu-tests", action="store_true", help="Include CUDA checks")


def pytest_configure(config):
    selections = (("--run-adhoc", "adhoc", None),
                  ("--run-integration", "integration", "TW_RUN_INTEGRATION"),
                  ("--run-gpu-tests", "integration/test_gnn_gpu.py", "TW_RUN_GPU_TESTS"))
    explicit = [Path(arg.split("::", 1)[0]).resolve() for arg in config.args]
    base = Path(__file__).parent.resolve()
    for option, category, environment in selections:
        path = base / category
        if config.getoption(option):
            if environment:
                os.environ[environment] = "1"
            if not any(p == path or path in p.parents or p in path.parents for p in explicit):
                config.args.append(str(path))
    for path in explicit:
        if base / "adhoc" in path.parents or path == base / "adhoc":
            if not config.getoption("--run-adhoc"):
                raise pytest.UsageError("Ad hoc tests require --run-adhoc")
        if base / "integration" in path.parents or path == base / "integration":
            gpu_only = path.name == "test_gnn_gpu.py" and config.getoption("--run-gpu-tests")
            if not config.getoption("--run-integration") and not gpu_only:
                raise pytest.UsageError("Integration tests require --run-integration (or --run-gpu-tests for the GPU file)")
    config.pluginmanager.register(TestTimings(config), "project-test-timings")


def pytest_ignore_collect(collection_path, config):
    base = Path(__file__).parent.resolve()
    if collection_path == base / "adhoc" and not config.getoption("--run-adhoc"):
        return True
    if collection_path == base / "integration":
        return not (config.getoption("--run-integration") or config.getoption("--run-gpu-tests"))


def pytest_collection_modifyitems(items):
    for item in items:
        parts = item.path.parts
        if "adhoc" in parts:
            item.add_marker(pytest.mark.adhoc)
        if "integration" in parts:
            item.add_marker(pytest.mark.gpu if item.path.name == "test_gnn_gpu.py" else pytest.mark.integration)


class TestTimings:
    def __init__(self, config):
        self.config = config
        self.started = None
        self.collection_seconds = None
        self.tests = {}
        self.collection = []

    def pytest_sessionstart(self, session):
        self.started = time.perf_counter()

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection.append({"nodeid": report.nodeid, "outcome": report.outcome})

    def pytest_collection_finish(self, session):
        self.collection_seconds = time.perf_counter() - self.started

    def pytest_runtest_logreport(self, report):
        row = self.tests.setdefault(report.nodeid, {"nodeid": report.nodeid, "phases": {}})
        row["phases"][report.when] = {"seconds": report.duration, "outcome": report.outcome}

    def pytest_sessionfinish(self, session, exitstatus):
        wall = time.perf_counter() - self.started
        rows = list(self.tests.values())
        for row in rows:
            row["total_seconds"] = sum(phase["seconds"] for phase in row["phases"].values())
        result = {"exit_code": int(exitstatus), "session_wall_seconds": wall,
                  "collection_seconds": self.collection_seconds,
                  "test_phase_seconds": sum(row["total_seconds"] for row in rows),
                  "collected": session.testscollected, "collection_issues": self.collection,
                  "tests": sorted(rows, key=lambda row: row["total_seconds"], reverse=True)}
        out = self.config.rootpath / "tests" / ".results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "timings.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def pytest_terminal_summary(self, terminalreporter):
        collection = self.collection_seconds if self.collection_seconds is not None else 0
        terminalreporter.write_line("Project timings: collection %.3fs; per-test setup/call/teardown in tests/.results/timings.json" % collection)
