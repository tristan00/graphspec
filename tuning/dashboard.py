import argparse
import logging
import time
from pathlib import Path

from graphspec.infra.files import JsonCache
from graphspec.infra.log import configure
from graphspec.infra.runtime import home
from graphspec.infra.serve import HttpError, serve, server

from .records import Run, Status

LOG = logging.getLogger(__name__)
WEB = Path(__file__).with_name("web")


class Dashboard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Run directory does not exist: " + str(self.root))
        self.cache = JsonCache()

    def directories(self):
        if (self.root / "plan.json").is_file():
            return {self.root.name: self.root}
        return {
            path.name: path
            for path in sorted(self.root.iterdir(), key=lambda value: value.name.casefold())
            if path.is_dir() and not path.is_symlink() and (path / "plan.json").is_file()
        }

    def open(self, identity):
        directory = self.directories().get(identity)
        if directory is None:
            raise HttpError(404, "This run is no longer available. Choose another run.")
        return Run(directory, self.cache)

    def runs(self, query):
        rows = []
        for identity, path in self.directories().items():
            try:
                run = Run(path, self.cache)
                runner = run.runner()
                rows.append(
                    dict(
                        id=identity,
                        trials=len(run.plan["trials"]),
                        status=runner["status"] if runner else "unknown",
                        error=None,
                    )
                )
            except (ValueError, KeyError, TypeError, OSError) as error:
                rows.append(dict(id=identity, trials=None, status="unreadable", error=str(error)))
        return dict(root=str(self.root), runs=rows)

    def snapshot(self, query):
        run = self.open(query.get("id"))
        trials = []
        for trial in run.trials(selection=False):
            outcome = trial.outcome.data if trial.outcome else {}
            record = trial.record or {}
            progress = (
                self.cache.read(trial.directory / "fit_progress.json")
                if trial.status == Status.RUNNING
                else None
            )
            trials.append(
                dict(
                    id=trial.item["directory"],
                    number=trial.index + 1,
                    status=trial.status,
                    description=trial.item.get("description", ""),
                    family=trial.config["model"]["view"],
                    selector=trial.item["selector"],
                    mse=outcome.get("raw_val_mse"),
                    r2=outcome.get("val_r2"),
                    seconds=record.get("seconds"),
                    parameters=outcome.get("parameters"),
                    started=record.get("started") if not trial.running else trial.running["started"],
                    completed=record.get("completed"),
                    config=trial.parameters,
                    metrics={
                        key: value
                        for key, value in outcome.items()
                        if key
                        not in (
                            "config",
                            "training_losses",
                            "epoch_losses",
                            "error",
                            "traceback",
                            "raw_val_mse",
                            "val_r2",
                            "parameters",
                        )
                    },
                    error=outcome.get("error"),
                    progress=progress if trial.status == Status.RUNNING else None,
                )
            )
        plan = run.plan
        return dict(
            id=query["id"],
            path=str(run.directory),
            recorded_at=time.time(),
            runner=run.runner(),
            stop_requested=(run.directory / "STOP").exists(),
            selection_progress=self.cache.read(run.directory / "selection_progress.json"),
            plan={
                key: plan.get(key)
                for key in (
                    "created",
                    "bank",
                    "limit",
                    "seconds",
                    "selector",
                    "random_trials",
                    "random_params",
                    "mix",
                    "workers",
                    "train_rows",
                    "holdout_rows",
                    "bounds",
                    "budget_policy",
                    "failure_policy",
                )
            },
            trials=trials,
        )

    def optimizer(self, query):
        run = self.open(query.get("id"))
        proposals = []
        for index, item in enumerate(run.plan["trials"]):
            if item["selector"] not in ("botorch", "forest", "tpe"):
                continue
            selection = self.cache.require(run.directory / item["directory"] / "selection.json")
            if item["selector"] == "tpe" and selection.get("phase") != "tpe":
                continue
            proposals.append(
                dict(
                    id=item["directory"],
                    number=index + 1,
                    selector=item["selector"],
                    **{
                        key: selection.get(key)
                        for key in (
                            "fixed_params",
                            "observations",
                            "failed_observations",
                            "seconds",
                            "acquisition",
                            "acquisition_value",
                            "settings",
                            "diagnostics",
                            "params",
                            "excluded",
                            "phase",
                        )
                    },
                )
            )
        return dict(proposals=proposals)

    def detail(self, query):
        run = self.open(query.get("run"))
        trial = next(iter(run.trials(name=query.get("id") or "")), None)
        if trial is None:
            raise HttpError(404, "This trial is no longer available.")
        logs = sorted(trial.directory.glob("worker*.log"))
        log = None
        if logs:
            with logs[-1].open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 24000))
                raw = stream.read()
            if size > 24000:
                raw = raw.partition(b"\n")[2]
            log = dict(name=logs[-1].name, text=raw.decode("utf-8", errors="replace"), truncated=size > 24000)
        outcome = trial.outcome.data if trial.outcome else None
        if outcome is not None:
            outcome = {key: value for key, value in outcome.items() if key != "training_losses"}
        return dict(
            id=trial.item["directory"],
            status=trial.status,
            config=trial.parameters,
            selection=trial.selection,
            record=trial.record,
            outcome=outcome,
            progress=self.cache.read(trial.directory / "fit_progress.json"),
            log=log,
        )

    def route(self, function):
        def execute(query):
            started = time.perf_counter()
            try:
                return function(query)
            except HttpError:
                raise
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise HttpError(422, "Cannot read the saved tuning records: " + str(error)) from error
            finally:
                LOG.info("dashboard %s exit seconds=%.3f", function.__name__, time.perf_counter() - started)

        return execute

    def routes(self):
        return {
            "/api/runs": self.route(self.runs),
            "/api/run": self.route(self.snapshot),
            "/api/trial": self.route(self.detail),
            "/api/optimizer": self.route(self.optimizer),
        }


def assets():
    return {
        "/app.js": (WEB / "app.js", "text/javascript; charset=utf-8"),
        "/parameters.js": (WEB / "parameters.js", "text/javascript; charset=utf-8"),
        "/explore.js": (WEB / "explore.js", "text/javascript; charset=utf-8"),
        "/discovery.js": (WEB / "discovery.js", "text/javascript; charset=utf-8"),
        "/discovery-worker.js": (WEB / "discovery-worker.js", "text/javascript; charset=utf-8"),
        "/discover-view.js": (WEB / "discover-view.js", "text/javascript; charset=utf-8"),
        "/run-view.js": (WEB / "run-view.js", "text/javascript; charset=utf-8"),
        "/style.css": (WEB / "style.css", "text/css; charset=utf-8"),
    }


def create_server(root, port=0):
    return server(port, WEB / "index.html", Dashboard(root).routes(), assets())


def main():
    parser = argparse.ArgumentParser(description="Track and explore saved tuning runs.")
    parser.add_argument("--runs", type=Path, default=home() / "runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    configure()
    dashboard = Dashboard(args.runs)
    serve(args.port, WEB / "index.html", dashboard.routes(), assets(), root=str(dashboard.root))


if __name__ == "__main__":
    main()
