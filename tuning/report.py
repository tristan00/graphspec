import json
import time
from datetime import datetime, timezone

from graphspec.infra.files import JsonCache, write_csv, write_json, write_jsonl
from graphspec.infra.log import emit
from graphspec.infra.runtime import output_path

from . import objective
from .model_spec import describe
from .records import Run, Status


def report(directory):

    started = time.perf_counter()
    directory = output_path(directory)
    run = Run(directory, JsonCache())
    planned = run.plan
    results = []
    for trial in run.trials():
        if trial.status in (Status.PENDING, Status.RUNNING):
            continue
        record = trial.record
        outcome = (
            trial.outcome.data
            if trial.outcome is not None
            else {"error": "worker exited without a terminal record"}
        )
        results.append(
            dict(
                outcome,
                status=trial.status,
                config=trial.parameters,
                max_seconds=trial.parameters.get("max_seconds"),
                trial=trial.item["directory"],
                selector=trial.item["selector"],
                graph=trial.item["graph"],
                model=trial.item["model"],
                trial_seconds=record["seconds"] if record is not None else None,
                phases=record["phases"] if record is not None else None,
                **(dict(objective=objective.value(outcome, planned)) if trial.status == Status.COMPLETE else {}),
            )
        )
    completed = [r for r in results if r["status"] == "complete"]
    columns = [
        "trial",
        "view",
        "description",
        "model",
        "graph",
        "status",
        "objective",
        "val_r2",
        "trial_seconds",
        "max_seconds",
        "seconds",
        "parameters",
        "peak_cuda_bytes",
        "optimizer_updates",
        "train_unique_rows",
        "holdout_rows",
        "epochs",
        "requested_epochs",
        "training_coverage_complete",
        "supervision_sha256",
        "source_sha256",
        "selector",
        "config_json",
    ]
    rows = []
    for result in results:
        row = {key: result.get(key) for key in columns}
        row.update(
            view=result["config"]["model"]["view"],
            description=describe(result["config"]["model"]),
            config_json=json.dumps(result["config"], sort_keys=True),
        )
        rows.append(row)
    write_csv(directory / "results.csv", columns, rows)
    write_jsonl(directory / "results.jsonl", results)
    views = sorted({r["config"]["model"]["view"] for r in results})
    write_json(
        directory / "coverage.json",
        dict(
            views={view: sum(r["config"]["model"]["view"] == view for r in completed) for view in views},
            completed=len(completed),
            failed=len(results) - len(completed),
            planned=len(planned["trials"]),
        ),
    )
    drawn = sum(item["selector"] == "random" for item in planned["trials"])
    selection_text = (
        "%d trials drew a uniform model spec, training parameters and graph-transform spec; %d were proposed by TPE from every trial recorded before them, timeouts and crashes entering as failures."
        % (drawn, len(planned["trials"]) - drawn)
    )
    lines = [
        "**Graph transform and temporal model search**",
        "",
        "Updated " + datetime.now(timezone.utc).isoformat() + ".",
        "",
        "%d completed fits; %d failed, timed out or interrupted; %d planned so far. %s"
        % (len(completed), len(results) - len(completed), len(planned["trials"]), selection_text),
        "",
    ]
    lines.extend(
        [
            "| View | Completed fits | Best MSE | Best RÂ² | Best model | Best graph |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for view in views:
        subset = [r for r in completed if r["config"]["model"]["view"] == view]
        best = min(subset, key=lambda r: r["objective"]) if subset else None
        lines.append(
            "| %s | %d | %s | %s | %s | %s |"
            % (
                view,
                len(subset),
                "%.6f" % best["objective"] if best else "Pending",
                "%.4f" % best["val_r2"] if best else "â€”",
                describe(best["config"]["model"]) if best else "â€”",
                best["graph"][:10] if best else "â€”",
            )
        )
    lines.extend(
        [
            "",
            "All trial configurations, failure records and individual scores: [results.csv](results.csv). Full records: [results.jsonl](results.jsonl).",
            "",
        ]
    )
    (directory / "progress.md").write_text("\n".join(lines), encoding="utf-8")
    emit("report", attempts=len(results), completed=len(completed), seconds=time.perf_counter() - started)
    return results
