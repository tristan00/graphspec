import argparse
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from graphspec.infra import sources
from graphspec.infra.files import (
    BUDGET,
    JsonCache,
    read_json,
    sha256_file,
    write_json,
)
from graphspec.infra.log import configure, emit
from graphspec.infra.processes import WORKERS, Lock, WorkerProcess
from graphspec.infra.runtime import exclusive, output_path
from graphspec.infra.timing import Timeline, measure, timed
from graphspec.spec import digest

from .budget import TIMEOUT_EXIT_CODE
from .records import VERSION, Outcome, worker_alive
from .report import report
from .training import PACKAGES, fit

REFERENCE_HISTORY = 16


def fit_timings(directory):
    progress = JsonCache().read(Path(directory) / "fit_progress.json")
    if progress is None:
        return {}
    return dict(
        timeline=progress.get("timeline"),
        stage=progress["phase"],
        timings=progress.get("timings", {}),
        fit_phases=progress.get("phase_seconds", {}),
        timing_recorded_at=progress.get("recorded_at"),
        graph_preparation=progress.get("graph_preparation"),
        timeout_reason=progress.get("timeout_reason"),
    )


def worker(args):
    directory = Path(args.output)
    config = read_json(directory / "config.json")
    blocks = read_json(args.training_blocks)
    try:
        fit(args.bank, directory, config, blocks, args.batch_sequences, args.deadline, args.workers)
    except TimeoutError as error:
        emit("worker_timeout", error=str(error), deadline=args.deadline)
        raise SystemExit(TIMEOUT_EXIT_CODE) from error
    except Exception as error:
        write_json(
            directory / "failure.json",
            dict(
                model=digest(config["model"]),
                graph=digest(config["graph"]),
                error=repr(error),
                traceback=traceback.format_exc(),
                source_sha256=sources.digest(PACKAGES),
                **fit_timings(directory),
            ),
        )
        raise


@exclusive
def run(args):
    from .budget import POLICY
    from .data import Bank
    from .selector import FAILURE_POLICY, SURROGATES, check_bounds, check_mix

    for name in ("workers", "batch_sequences", "variant_gb"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(name + " must be a positive integer")
    for name in ("trials", "random_trials"):
        value = getattr(args, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(name + " must be a nonnegative integer")
    if args.seconds is not None and (type(args.seconds) is not int or args.seconds < 1):
        raise ValueError("seconds must be a positive integer")
    directory = output_path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    bank = Bank(args.bank)
    bounds = read_json(args.bounds) if args.bounds else {}
    mix = None
    if args.mix is not None:
        pairs = [item.split("=") for item in args.mix]
        mix = check_mix({name: float(share) for name, share in pairs})
    check_bounds(bank.registry, bounds)
    if path.exists():
        plan = read_json(path)
        if plan.get("version") != VERSION:
            raise ValueError("this run was written by another plan version; choose a new output")
        if plan["bank_manifest_sha256"] != sha256_file(Path(args.bank) / "manifest.json"):
            raise ValueError("this run belongs to a different bank")
        plan["bank"] = str(Path(args.bank).resolve())
        plan.update(workers=args.workers, variant_bytes=args.variant_gb << 30, limit=args.trials)
        plan.setdefault("random_params", [])
        plan.setdefault("mix", {plan["selector"]: 100})
        for name, value in (
            ("selector", args.selector),
            ("random_trials", args.random_trials),
            ("mix", mix),
            ("random_params", args.random_params),
            ("seconds", args.seconds),
            ("bounds", bounds if args.bounds else None),
        ):
            if value is not None and value != plan[name]:
                emit(name + "_replaced", previous=plan[name], **{name: value})
                plan[name] = value
    else:
        plan = dict(
            version=VERSION,
            seed=args.seed,
            bank=str(Path(args.bank).resolve()),
            selector=args.selector or "botorch",
            trials=[],
            batch_sequences=args.batch_sequences,
            workers=args.workers,
            variant_bytes=args.variant_gb << 30,
            random_trials=100 if args.random_trials is None else args.random_trials,
            random_params=args.random_params or [],
            mix=mix or {args.selector or "botorch": 100},
            bounds=bounds,
            limit=args.trials,
            seconds=600 if args.seconds is None else args.seconds,
            train_rows=bank.manifest["train_rows"],
            holdout_rows=bank.manifest["validation_rows"],
            supervision_sha256=bank.manifest["supervision_sha256"],
            spec_sha256=bank.manifest["spec_sha256"],
            created=datetime.now(timezone.utc).isoformat(),
        )
    if plan["selector"] in SURROGATES:
        from .space import Space

        Space(bank.registry, plan["bounds"], plan["seconds"]).random_names(plan["random_params"])
    with Lock(directory / "runner.lock"):
        running = directory / "running.json"
        if running.exists():
            previous = read_json(running)
            if worker_alive(directory, previous):
                raise RuntimeError("the previous worker is still running; wait for it before continuing")
        plan.update(
            bank_manifest_sha256=sha256_file(Path(plan["bank"]) / "manifest.json"),
            budget_policy=POLICY,
            failure_policy=FAILURE_POLICY,
        )
        discard_unfinished(directory, plan)
        write_json(path, plan)
        execute(directory, plan, args.continuous, bank.registry)


def discard_unfinished(directory, plan):
    started = time.perf_counter()
    finished = []
    for item in plan["trials"]:
        if Outcome.read(directory / item["directory"], JsonCache()) is None:
            shutil.rmtree(directory / item["directory"], ignore_errors=True)
            emit("trial_discarded", directory=item["directory"])
        else:
            finished.append(item)
    plan["trials"] = finished
    emit("unfinished_discarded", trials=len(finished), seconds=time.perf_counter() - started)


@timed("layout")
def training_layout(directory, plan, config):
    from .data import Bank

    started = time.perf_counter()
    history = config["graph"]["time/history"]
    from graphspec.join import parameters

    path = directory / (
        "train_blocks_%s.json" % digest(dict(history=history, join=parameters(config["graph"])))[:12]
    )
    if not path.exists():
        bank = Bank(plan["bank"])
        if bank.manifest["supervision_sha256"] != plan["supervision_sha256"]:
            raise ValueError("training layout changed supervision")
        blocks = bank.blocks(0, history, config)
        write_json(path, blocks)
        emit(
            "training_layout_created",
            history=history,
            blocks=len(blocks),
            rows=len(bank.split["train_indices"]),
            seconds=time.perf_counter() - started,
        )
    return path, max(1, plan["batch_sequences"] * REFERENCE_HISTORY // history)


def mark_timeout(trial, max_seconds, started, phases, error):
    for name in ("result.json", "failure.json"):
        path = trial / name
        if path.exists():
            path.replace(trial / ("late_" + name))
    write_json(
        trial / "timeout.json",
        dict(
            seconds=time.perf_counter() - started,
            budget=max_seconds,
            phases=phases,
            error=str(error),
            **fit_timings(trial),
        ),
    )


def execute_trial(directory, plan, item, index, started, phases):
    with Timeline(directory / item["directory"] / "runner_timing.json"), measure("trial_runner"):
        return execute_trial_body(directory, plan, item, index, started, phases)


def execute_trial_body(directory, plan, item, index, started, phases):
    from .budget import wait_worker

    trial = directory / item["directory"]
    opened = datetime.now(timezone.utc).isoformat()
    config = read_json(trial / "config.json")
    if "max_seconds" not in config:
        config["max_seconds"] = plan["seconds"]
        write_json(trial / "config.json", config)
    max_seconds = config["max_seconds"]
    if type(max_seconds) is not int or max_seconds < 1:
        raise ValueError("max_seconds must be a positive integer")
    deadline = started + max_seconds
    timeout = None
    try:
        stage = time.perf_counter()
        blocks, batch_size = training_layout(directory, plan, config)
        phases["layout"] = time.perf_counter() - stage
        arguments = [
            "worker",
            "--bank",
            plan["bank"],
            "--output",
            str(trial.resolve()),
            "--training-blocks",
            str(blocks.resolve()),
            "--batch-sequences",
            str(batch_size),
            "--workers",
            str(plan["workers"]),
            "--deadline",
            repr(deadline),
        ]
        emit(
            "trial_enter",
            index=index,
            total=len(plan["trials"]),
            model=item["model"],
            graph=item["graph"][:10],
        )
        stage = time.perf_counter()
        with measure("source_snapshot"):
            sources.snapshot(PACKAGES, trial / "source")
        phases["snapshot"] = time.perf_counter() - stage
        if time.perf_counter() >= deadline:
            mark_timeout(trial, max_seconds, started, phases, "deadline reached during trial preparation")
            return config
        with (trial / "worker.log").open("w", encoding="utf-8") as stream:
            stage = time.perf_counter()
            with WorkerProcess("tuning.run", arguments, stream) as process:
                write_json(
                    directory / "running.json",
                    dict(
                        runner_pid=os.getpid(),
                        worker_pid=process.pid,
                        trial=item["directory"],
                        started=opened,
                        max_seconds=max_seconds,
                    ),
                )
                try:
                    with measure("worker_wait"):
                        code = wait_worker(process, deadline, trial)
                except subprocess.TimeoutExpired as error:
                    timeout = error
                    mark_timeout(trial, max_seconds, started, phases, error)
            phases["worker"] = time.perf_counter() - stage
        if timeout is not None:
            mark_timeout(trial, max_seconds, started, phases, timeout)
        elif code == TIMEOUT_EXIT_CODE:
            mark_timeout(
                trial,
                max_seconds,
                started,
                phases,
                fit_timings(trial).get("timeout_reason") or "worker reached its deadline",
            )
        elif Outcome.read(trial, JsonCache()) is None:
            write_json(
                trial / "failure.json",
                dict(error=f"worker exited {code} without an outcome", **fit_timings(trial)),
            )
    except BaseException as error:
        if Outcome.read(trial, JsonCache()) is None:
            filename = (
                "interrupted.json" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failure.json"
            )
            write_json(trial / filename, dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        seconds = time.perf_counter() - started
        write_json(
            trial / "trial.json",
            dict(
                seconds=seconds,
                budget=max_seconds,
                phases=phases,
                started=opened,
                completed=datetime.now(timezone.utc).isoformat(),
            ),
        )
        (directory / "running.json").unlink(missing_ok=True)
    outcome = Outcome.read(trial, JsonCache())
    emit("trial_exit", index=index, status=outcome.status, seconds=seconds, phases=phases)
    return config


def execute(directory, plan, continuous, registry):
    from .data import Bank
    from .selector import append_trial
    from .variants import prune

    started = time.perf_counter()
    state = dict(status="running", pid=os.getpid(), started=datetime.now(timezone.utc).isoformat())
    write_json(directory / "runner.json", state)
    emit(
        "run_enter",
        trials=len(plan["trials"]),
        limit=plan["limit"],
        seconds_per_trial=plan["seconds"],
        continuous=continuous,
    )
    index = 0
    try:
        while not (directory / "STOP").exists():
            trial_started, phases = time.perf_counter(), {}
            if plan["limit"] is not None and index >= plan["limit"]:
                break
            if index == len(plan["trials"]):
                if not continuous:
                    break
                append_trial(directory, plan, registry)
                phases["select"] = time.perf_counter() - trial_started
            item = plan["trials"][index]
            index += 1
            if Outcome.read(directory / item["directory"], JsonCache()) is not None:
                continue
            config = execute_trial(directory, plan, item, index, trial_started, phases)
            stage = time.perf_counter()
            with Timeline(directory / item["directory"] / "maintenance_timing.json"), measure("maintenance"):
                with measure("cache_prune"):
                    prune(plan["bank"], plan["variant_bytes"], Bank(plan["bank"]).stamp(config))
                emit("trial_prune", seconds=time.perf_counter() - stage)
                with measure("report"):
                    report(directory)
        report(directory)
        state["status"] = "stopped" if (directory / "STOP").exists() else "complete"
    except BaseException as error:
        state.update(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        state.update(completed=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started)
        write_json(directory / "runner.json", state)
        emit("run_exit", status=state["status"], seconds=state["seconds"])


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--bank", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--seed", type=int, default=20260909)
    run_parser.add_argument("--batch-sequences", type=int, default=64)
    run_parser.add_argument("--continuous", action="store_true")
    run_parser.add_argument("--trials", type=int)
    run_parser.add_argument(
        "--seconds",
        type=int,
        help="time limit for new trials; defaults to 600 for a new run, otherwise keeps the saved limit",
    )
    run_parser.add_argument(
        "--selector",
        choices=("forest", "botorch", "tpe", "random"),
        help="how trials are chosen; botorch for a new run, otherwise keep the saved selector",
    )
    run_parser.add_argument(
        "--startup-trials",
        "--random-trials",
        dest="random_trials",
        type=int,
        help="minimum finished trials before a surrogate selector starts, including failures and any previous selector",
    )
    run_parser.add_argument(
        "--random-params",
        nargs="*",
        help="parameters to draw randomly and hold fixed during surrogate selection",
    )
    run_parser.add_argument(
        "--mix",
        nargs="*",
        help="percent of trials after startup for each selector, as name=share pairs summing to 100",
    )
    run_parser.add_argument("--bounds")
    run_parser.add_argument("--workers", type=int, default=WORKERS)
    run_parser.add_argument("--variant-gb", type=int, default=BUDGET >> 30)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--bank", required=True)
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--training-blocks", required=True)
    worker_parser.add_argument("--batch-sequences", type=int, required=True)
    worker_parser.add_argument("--deadline", type=float, required=True)
    worker_parser.add_argument("--workers", type=int, required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("directory")
    args = parser.parse_args()
    configure(json_lines=True)
    if args.command == "worker":
        worker(args)
    elif args.command == "run":
        run(args)
    else:
        report(args.directory)


if __name__ == "__main__":
    main()
