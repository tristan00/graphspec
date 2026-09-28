import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from graphspec.infra.files import read_json, sha256_json
from graphspec.infra.serve import HttpError
from tuning.dashboard import Dashboard, create_server
from tuning.records import VERSION
from tuning.selector import load_history


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, name="search-alpha", count=32):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    plan = dict(
        version=VERSION,
        bank="fixture-bank",
        trials=[],
        limit=40,
        seconds=120,
        selector="tpe",
        train_rows=10,
        holdout_rows=4,
        objective=None,
        supervision_sha256="fixture",
        workers=2,
        random_trials=10,
        bounds={},
        created="2026-09-15T12:00:00+00:00",
    )
    for index in range(count):
        config = dict(
            model=dict(
                view=["snapshot", "sequence", "event_graph"][index % 3], hidden=8 + index, dropout=0.1
            ),
            graph={"time/history": 4 + index % 4},
            epochs=3,
            seed=index,
            lr=0.001,
        )
        if index % 5 != 4:
            config["graph"]["derive/hops"] = [0, 2, 10, 100][index % 4]
        config["graph"]["derive/enabled"] = index % 2 == 0
        name = f"trial_{index:06d}"
        item = dict(
            directory=name,
            model=sha256_json(config["model"]),
            graph=sha256_json(config["graph"]),
            selector="tpe",
            description=f"Model {index}",
        )
        plan["trials"].append(item)
        trial = directory / name
        write(trial / "config.json", config)
        write(trial / "selection.json", dict(phase="random_startup" if index < 10 else "tpe"))
        if index == count - 1:
            continue
        status = "timeout" if index % 7 == 5 else "failed" if index % 7 == 6 else "complete"
        write(
            trial / "trial.json",
            dict(
                seconds=20 + index,
                started="2026-09-15T12:00:00+00:00",
                completed="2026-09-15T12:01:00+00:00",
                phases={"worker": 19 + index},
            ),
        )
        if status == "complete":
            write(
                trial / "result.json",
                dict(
                    status=status,
                    model=item["model"],
                    graph=item["graph"],
                    config=config,
                    supervision_sha256="fixture",
                    epochs=3,
                    requested_epochs=3,
                    training_coverage_complete=True,
                    train_unique_rows=10,
                    train_available_rows=10,
                    holdout_rows=4,
                    training_examples_seen=30,
                    raw_val_mse=0 if index == 3 else 1 / (index + 1),
                    val_r2=-0.5 if index == 0 else 0.9 - index / 100,
                    parameters=100 + index,
                    epoch_losses=[1, 0.5, 0.2],
                    fit_phases={"preparation": 2, "training": 16, "scoring": 2},
                    training_losses=list(range(100)),
                    peak_cuda_bytes=0,
                    optimizer_updates=30,
                ),
            )
        else:
            write(
                trial / ("timeout.json" if status == "timeout" else "failure.json"),
                dict(
                    error="Deadline exceeded" if status == "timeout" else '<img src=x onerror="alert(1)">',
                    traceback="Fixture worker failure",
                    fit_phases={"preparation": 20},
                ),
            )
        (trial / "worker.log").write_text("Fixture worker log\n<unsafe & text>\n", encoding="utf-8")
    write(directory / "plan.json", plan)
    write(directory / "runner.json", dict(status="stopped", seconds=400))
    return directory


def test_optimizer_history_reads_actual_limits_without_rewriting_configs(tmp_path):
    directory = fixture(tmp_path, count=8)
    original = {path: path.read_bytes() for path in directory.glob("trial_*/*.json")}
    for index in range(7):
        path = directory / f"trial_{index:06d}" / "trial.json"
        write(path, dict(read_json(path), budget=700 + index * 100))
    plan = read_json(directory / "plan.json")
    plan["seconds"] = 9999
    observations, failures, history, failure_history = load_history(directory, plan)
    assert len(observations) == len(history) == 5
    assert len(failures) == len(failure_history) == 2
    assert [entry["config"]["max_seconds"] for entry in observations + failures] == list(
        range(700, 1400, 100)
    )
    assert all(
        path.read_bytes() == content for path, content in original.items() if path.name != "trial.json"
    )


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dashboard-")
        self.root = Path(self.temporary.name)
        self.run = fixture(self.root, count=8)
        self.dashboard = Dashboard(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_catalogue_ignores_non_runs(self):
        (self.root / "bank").mkdir()
        self.assertEqual([r["id"] for r in self.dashboard.runs({})["runs"]], ["search-alpha"])

    def test_single_run_root(self):
        self.assertEqual(Dashboard(self.run).snapshot({"id": "search-alpha"})["path"], str(self.run))

    def test_empty_root(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(Dashboard(empty).runs({})["runs"], [])

    def test_statuses_metrics_and_no_invented_scores(self):
        snapshot = self.dashboard.snapshot({"id": "search-alpha"})
        rows = snapshot["trials"]
        self.assertEqual([entry["key"] for entry in snapshot["measures"]], ["mse", "r2"])
        self.assertEqual(snapshot["columns"], [])
        self.assertEqual(rows[0]["scores"], {})
        self.assertEqual([row["status"] for row in rows], ["complete"] * 5 + ["timeout", "failed", "pending"])
        self.assertEqual(rows[0]["r2"], -0.5)
        self.assertEqual(rows[3]["mse"], 0)
        self.assertIsNone(rows[5]["mse"])
        self.assertIsNone(rows[7]["seconds"])
        self.assertEqual(rows[0]["metrics"]["peak_cuda_bytes"], 0)
        self.assertEqual(rows[0]["metrics"]["fit_phases"]["training"], 16)
        self.assertNotIn("training_losses", rows[0]["metrics"])
        self.assertNotIn("config", rows[0]["metrics"])

    def test_stale_runner_is_interrupted(self):
        write(self.run / "runner.json", dict(status="running", pid=2147483647))
        self.assertEqual(self.dashboard.snapshot({"id": "search-alpha"})["runner"]["status"], "interrupted")

    def test_live_progress_and_stale_worker(self):
        write(
            self.run / "running.json",
            dict(trial="trial_000007", worker_pid=2147483647, started="2026-09-15T12:00:00Z"),
        )
        write(
            self.run / "trial_000007" / "fit_progress.json", dict(phase="training", recorded_at=time.time())
        )
        with patch("tuning.records.worker_alive", return_value=True):
            trial = self.dashboard.snapshot({"id": "search-alpha"})["trials"][-1]
            self.assertEqual(trial["status"], "running")
            self.assertEqual(trial["progress"]["phase"], "training")
        trial = self.dashboard.snapshot({"id": "search-alpha"})["trials"][-1]
        self.assertNotEqual(trial["status"], "running")
        self.assertIsNone(trial["progress"])

    def test_updates_are_seen(self):
        self.dashboard.snapshot({"id": "search-alpha"})
        write(self.run / "trial_000007" / "failure.json", dict(error="new failure"))
        self.assertEqual(self.dashboard.snapshot({"id": "search-alpha"})["trials"][-1]["status"], "failed")

    def test_removed_optional_records_are_not_cached(self):
        self.dashboard.snapshot({"id": "search-alpha"})
        (self.run / "runner.json").unlink()
        self.assertIsNone(self.dashboard.snapshot({"id": "search-alpha"})["runner"])

    def test_historical_trial_limits_do_not_follow_the_current_run_limit(self):
        path = self.run / "trial_000000" / "trial.json"
        record = json.loads(path.read_text())
        write(path, dict(record, budget=600))
        path = self.run / "trial_000001" / "trial.json"
        record = json.loads(path.read_text())
        write(path, dict(record, budget=1200))
        snapshot = self.dashboard.snapshot({"id": "search-alpha"})
        self.assertEqual(snapshot["plan"]["seconds"], 120)
        self.assertEqual(snapshot["trials"][0]["config"]["max_seconds"], 600)
        self.assertEqual(snapshot["trials"][1]["config"]["max_seconds"], 1200)
        self.assertNotIn("max_seconds", snapshot["trials"][2]["config"])
        detail = self.dashboard.detail({"run": "search-alpha", "id": "trial_000000"})
        self.assertEqual(detail["config"]["max_seconds"], 600)
        original = json.loads((self.run / "trial_000000" / "config.json").read_text())
        self.assertNotIn("max_seconds", original)

    def test_conflicting_recorded_limit_is_an_error(self):
        path = self.run / "trial_000007" / "config.json"
        write(path, dict(json.loads(path.read_text()), max_seconds=600))
        write(self.run / "trial_000007" / "trial.json", dict(seconds=5, budget=1200))
        with self.assertRaises(HttpError):
            self.dashboard.routes()["/api/run"]({"id": "search-alpha"})

    def test_report_exports_each_recorded_time_limit(self):
        from tuning.report import report

        for index, budget in ((0, 600), (5, 1200)):
            path = self.run / f"trial_{index:06d}" / "trial.json"
            write(path, dict(json.loads(path.read_text()), budget=budget))
        with patch("tuning.report.describe", return_value="fixture"):
            report(self.run)
        with (self.run / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["max_seconds"], "600")
        self.assertEqual(rows[5]["max_seconds"], "1200")
        self.assertEqual(json.loads(rows[0]["config_json"])["max_seconds"], 600)
        self.assertEqual(rows[2]["max_seconds"], "")

    def test_bad_run_does_not_hide_healthy_runs(self):
        write(self.root / "broken" / "plan.json", dict(version=0))
        rows = self.dashboard.runs({})["runs"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "unreadable")
        self.assertIsNone(rows[1]["error"])

    def test_unknown_and_traversal_run(self):
        for identity in (None, "missing", "../search-alpha", str(self.run)):
            with self.assertRaises(HttpError) as caught:
                self.dashboard.snapshot({"id": identity})
            self.assertEqual(caught.exception.status, 404)

    def test_unknown_and_traversal_trial(self):
        for identity in (None, "missing", "../plan.json"):
            with self.assertRaises(HttpError) as caught:
                self.dashboard.detail({"run": "search-alpha", "id": identity})
            self.assertEqual(caught.exception.status, 404)

    def test_conflicting_outcomes_are_an_explicit_error(self):
        write(self.run / "trial_000000" / "failure.json", dict(error="conflict"))
        with self.assertRaises(HttpError) as caught:
            self.dashboard.routes()["/api/run"]({"id": "search-alpha"})
        self.assertEqual(caught.exception.status, 422)

    def test_changed_configuration_is_an_explicit_error(self):
        path = self.run / "trial_000000" / "config.json"
        config = json.loads(path.read_text())
        config["model"]["hidden"] = 999
        write(path, config)
        with self.assertRaises(HttpError):
            self.dashboard.routes()["/api/run"]({"id": "search-alpha"})

    def test_missing_record_is_an_explicit_error(self):
        (self.run / "trial_000000" / "selection.json").unlink()
        with self.assertRaises(HttpError) as caught:
            self.dashboard.routes()["/api/trial"]({"run": "search-alpha", "id": "trial_000000"})
        self.assertIn("required JSON", str(caught.exception))

    def test_overview_does_not_read_proposal_histories(self):
        (self.run / "trial_000000" / "selection.json").write_text("unreadable proposal history")
        self.assertEqual(len(self.dashboard.snapshot({"id": "search-alpha"})["trials"]), 8)

    def test_details_read_only_the_selected_proposal(self):
        (self.run / "trial_000001" / "selection.json").write_text("unreadable proposal history")
        result = self.dashboard.detail({"run": "search-alpha", "id": "trial_000000"})
        self.assertEqual(result["selection"]["phase"], "random_startup")

    def test_detail_excludes_unbounded_batch_losses(self):
        result = self.dashboard.detail({"run": "search-alpha", "id": "trial_000000"})
        self.assertNotIn("training_losses", result["outcome"])
        self.assertEqual(result["outcome"]["epoch_losses"], [1, 0.5, 0.2])
        self.assertIn("<unsafe & text>", result["log"]["text"])

    def test_optimizer_reads_botorch_proposals_without_large_histories(self):
        path = self.run / "plan.json"
        plan = json.loads(path.read_text())
        plan["trials"][0]["selector"] = "botorch"
        plan["random_params"] = ["model/view"]
        write(path, plan)
        prediction = dict(predicted_mse=0.4, uncertainty=0.2, completion_probability=0.75)
        write(
            self.run / "trial_000000" / "selection.json",
            dict(
                fixed_params={"model/view": "snapshot"},
                observations=100,
                failed_observations=12,
                diagnostics=dict(prediction=prediction, rounds=[]),
                history=["large history"],
            ),
        )
        write(self.run / "selection_progress.json", dict(stage="Fitting models", trial_number=9))
        result = self.dashboard.optimizer({"id": "search-alpha"})["proposals"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["number"], 1)
        self.assertEqual(result[0]["diagnostics"]["prediction"], prediction)
        self.assertEqual(result[0]["failed_observations"], 12)
        self.assertNotIn("history", result[0])
        snapshot = self.dashboard.snapshot({"id": "search-alpha"})
        self.assertEqual(snapshot["plan"]["random_params"], ["model/view"])
        self.assertEqual(snapshot["selection_progress"]["stage"], "Fitting models")

    def test_non_botorch_run_has_empty_optimizer_history(self):
        self.assertEqual(self.dashboard.optimizer({"id": "search-alpha"}), {"proposals": []})

    def test_optimizer_reads_tpe_proposals(self):
        path = self.run / "plan.json"
        plan = json.loads(path.read_text())
        plan["trials"][0]["selector"] = "tpe"
        write(path, plan)
        write(
            self.run / "trial_000000" / "selection.json",
            dict(
                params={"model/view": "snapshot", "lr": 0.01},
                observations=30,
                failed_observations=5,
                excluded=[dict(trial="trial_000009", status="complete", parameters=["epochs"])],
                phase="tpe",
                history=["large history"],
            ),
        )
        result = self.dashboard.optimizer({"id": "search-alpha"})["proposals"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["selector"], "tpe")
        self.assertEqual(result[0]["params"], {"model/view": "snapshot", "lr": 0.01})
        self.assertEqual(result[0]["excluded"][0]["trial"], "trial_000009")
        self.assertNotIn("history", result[0])

    def test_log_tail_is_bounded(self):
        (self.run / "trial_000000" / "worker_01.log").write_text("old\n" * 10000 + "last line\n")
        log = self.dashboard.detail({"run": "search-alpha", "id": "trial_000000"})["log"]
        self.assertTrue(log["truncated"])
        self.assertLessEqual(len(log["text"]), 24000)
        self.assertEqual(log["text"].splitlines()[-1], "last line")

    def test_read_only(self):
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.run.rglob("*") if p.is_file()}
        self.dashboard.runs({})
        self.dashboard.snapshot({"id": "search-alpha"})
        self.dashboard.detail({"run": "search-alpha", "id": "trial_000000"})
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in before})

    def test_http_assets_routes_and_errors(self):
        with create_server(self.root) as instance:
            thread = threading.Thread(target=instance.serve_forever, args=(0.01,), daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{instance.server_address[1]}"
            try:
                for path, content_type in (
                    ("/", "text/html"),
                    ("/app.js", "text/javascript"),
                    ("/parameters.js", "text/javascript"),
                    ("/explore.js", "text/javascript"),
                    ("/discovery.js", "text/javascript"),
                    ("/discovery-worker.js", "text/javascript"),
                    ("/discover-view.js", "text/javascript"),
                    ("/run-view.js", "text/javascript"),
                    ("/style.css", "text/css"),
                    ("/api/runs", "application/json"),
                    ("/api/optimizer?id=search-alpha", "application/json"),
                ):
                    with urlopen(base + path, timeout=3) as response:
                        self.assertIn(content_type, response.headers["Content-Type"])
                        self.assertEqual(response.headers["Cache-Control"], "no-store")
                        self.assertTrue(response.read())
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + "/api/run?id=missing", timeout=3)
                self.assertEqual(caught.exception.code, 404)
                self.assertIn("error", json.loads(caught.exception.read()))
                caught.exception.close()
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + "/../pyproject.toml", timeout=3)
                self.assertEqual(caught.exception.code, 404)
                caught.exception.close()
            finally:
                instance.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
