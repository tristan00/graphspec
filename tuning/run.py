import argparse
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from graphspec.infra import files, sources
from graphspec.infra.files import (
    JsonCache,
    read_json,
    sha256_file,
    write_json,
)
from graphspec.infra.log import configure, emit
from graphspec.infra.processes import Lock, WorkerProcess
from graphspec.infra.runtime import exclusive, output_path
from graphspec.infra.timing import Timeline, measure, timed
from graphspec.spec import digest

from . import objective
from . import statement as statements
from .budget import TIMEOUT_EXIT_CODE
from .records import VERSION, Outcome, worker_alive
from .report import report
from .training import PACKAGES, fit

REFERENCE_HISTORY = 16

FLAGS = dict(
    selector="--selector", random_trials="--startup-trials", mix="--mix", random_params="--random-params"
)
COPIES = dict(
    seed="seed",
    seconds="seconds",
    workers="workers",
    variant_bytes="variant_cache_bytes",
    batch_sequences="batch_sequences",
)


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
        fit(
            args.bank,
            directory,
            config,
            blocks,
            args.batch_sequences,
            args.deadline,
            statements.state(read_json(args.statement)),
        )
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


def continued(plan, stated, given):
    started = time.perf_counter()
    lacking = [name for name in (*COPIES, *given) if name not in plan]
    if lacking:
        raise KeyError("plan.json lacks " + ", ".join(lacking))
    for name, field in COPIES.items():
        if plan[name] != getattr(stated, field):
            raise ValueError(
                "plan.json holds %s %r and the statement's %s is %r"
                % (name, plan[name], field, getattr(stated, field))
            )
    for name, value in given.items():
        if value is not None and value != plan[name]:
            emit(name + "_replaced", previous=plan[name], **{name: value})
            plan[name] = value
    emit("plan_continued", trials=len(plan["trials"]), seconds=time.perf_counter() - started)
    return plan


def opened(stated, given, bank_path, manifest, limit):
    started = time.perf_counter()
    absent = [FLAGS[name] for name in FLAGS if given[name] is None]
    if absent:
        raise ValueError("a new search states " + ", ".join(absent))
    plan = dict(
        version=VERSION,
        seed=stated.seed,
        bank=bank_path,
        selector=given["selector"],
        trials=[],
        batch_sequences=stated.batch_sequences,
        workers=stated.workers,
        variant_bytes=stated.variant_cache_bytes,
        random_trials=given["random_trials"],
        random_params=given["random_params"],
        mix=given["mix"],
        bounds=given["bounds"],
        objective=given["objective"],
        limit=limit,
        seconds=stated.seconds,
        train_rows=manifest["train_rows"],
        holdout_rows=manifest["validation_rows"],
        supervision_sha256=manifest["supervision_sha256"],
        spec_sha256=manifest["spec_sha256"],
        created=datetime.now(timezone.utc).isoformat(),
    )
    emit("plan_opened", selector=plan["selector"], seconds=time.perf_counter() - started)
    return plan


@exclusive
def run(args):
    stated = statements.state(flags(args))
    directory = output_path(args.output)
    path = directory / "plan.json"
    from .budget import policy
    from .data import Bank
    from .selector import FAILURE_POLICY, SURROGATES, check_bounds, check_mix

    for name in ("trials", "random_trials"):
        value = getattr(args, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(name + " must be a nonnegative integer")
    bank = Bank(args.bank)
    mix = None
    if args.mix is not None:
        pairs = [item.split("=") for item in args.mix]
        mix = check_mix({name: float(share) for name, share in pairs})
    if path.exists():
        plan = read_json(path)
        if plan.get("version") != VERSION:
            raise ValueError("this run was written by another plan version; choose a new output")
        if plan["bank_manifest_sha256"] != sha256_file(Path(args.bank) / "manifest.json"):
            raise ValueError("this run belongs to a different bank")
        given = dict(
            selector=args.selector,
            random_trials=args.random_trials,
            mix=mix,
            random_params=args.random_params,
            bounds=read_json(args.bounds) if args.bounds else None,
            objective=objective.weights(bank, args.objective) if args.objective else None,
        )
        plan = continued(plan, stated, given)
        plan["bank"] = str(Path(args.bank).resolve())
        plan["limit"] = args.trials
    else:
        given = dict(
            selector=args.selector,
            random_trials=args.random_trials,
            mix=mix,
            random_params=args.random_params,
            bounds=read_json(args.bounds) if args.bounds else {},
            objective=objective.weights(bank, args.objective),
        )
        plan = opened(stated, given, str(Path(args.bank).resolve()), bank.manifest, args.trials)
    check_bounds(bank.registry, plan["bounds"])
    if plan["selector"] in SURROGATES:
        from .space import Space

        Space(bank.registry, plan["bounds"], stated.seconds, ()).random_names(plan["random_params"])
    directory.mkdir(parents=True, exist_ok=True)
    with Lock(directory / "runner.lock"):
        running = directory / "running.json"
        if running.exists():
            previous = read_json(running)
            if worker_alive(directory, previous):
                raise RuntimeError("the previous worker is still running; wait for it before continuing")
        plan.update(
            bank_manifest_sha256=sha256_file(Path(plan["bank"]) / "manifest.json"),
            budget_policy=policy(stated.preparation_seconds),
            failure_policy=FAILURE_POLICY,
        )
        discard_unfinished(directory, plan)
        write_json(path, plan)
        execute(directory, plan, args.continuous, bank.registry, stated)


def flags(args):
    started = time.perf_counter()
    given = dict(
        version=statements.VERSION,
        device=args.device,
        seed=args.seed,
        seconds=args.seconds,
        preparation_seconds=args.preparation_seconds,
        workers=args.workers,
        variant_cache_bytes=None if args.variant_gb is None else args.variant_gb << 30,
        batch_sequences=args.batch_sequences,
    )
    result = {name: value for name, value in given.items() if value is not None}
    emit("statement_flags", fields=sorted(result), seconds=time.perf_counter() - started)
    return result


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
def training_layout(directory, plan, config, stated):
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
    return path, max(1, stated.batch_sequences * REFERENCE_HISTORY // history)


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


def execute_trial(directory, plan, item, index, started, phases, stated):
    with Timeline(directory / item["directory"] / "runner_timing.json"), measure("trial_runner"):
        return execute_trial_body(directory, plan, item, index, started, phases, stated)


def execute_trial_body(directory, plan, item, index, started, phases, stated):
    from .budget import wait_worker
    from .selector import seeds

    trial = directory / item["directory"]
    opened = datetime.now(timezone.utc).isoformat()
    config = read_json(trial / "config.json")
    stated = stated.replace(seed=seeds(stated.seed, plan["trials"].index(item))[1])
    for name, value in (("seed", stated.seed), ("max_seconds", stated.seconds)):
        if name not in config:
            raise KeyError("config.json of %s lacks %s" % (item["directory"], name))
        if config[name] != value:
            raise ValueError(
                "config.json of %s holds %s %r and the statement of the trial says %r"
                % (item["directory"], name, config[name], value)
            )
    max_seconds = stated.seconds
    write_json(trial / "statement.json", stated.mapping())
    deadline = started + max_seconds
    timeout = None
    try:
        stage = time.perf_counter()
        blocks, batch_size = training_layout(directory, plan, config, stated)
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
            "--statement",
            str((trial / "statement.json").resolve()),
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
                        code = wait_worker(process, deadline, trial, stated.preparation_seconds)
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


def execute(directory, plan, continuous, registry, stated):
    from graphspec.infra.arrays import RESIDENT

    from .data import Deriver
    from .selector import append_trial
    from .variants import prune

    started = time.perf_counter()
    state = dict(status="running", pid=os.getpid(), started=datetime.now(timezone.utc).isoformat())
    write_json(directory / "runner.json", state)
    emit(
        "run_enter",
        trials=len(plan["trials"]),
        limit=plan["limit"],
        seconds_per_trial=stated.seconds,
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
                append_trial(directory, plan, registry, stated)
                phases["select"] = time.perf_counter() - trial_started
            item = plan["trials"][index]
            index += 1
            if Outcome.read(directory / item["directory"], JsonCache()) is not None:
                continue
            config = execute_trial(directory, plan, item, index, trial_started, phases, stated)
            stage = time.perf_counter()
            with Timeline(directory / item["directory"] / "maintenance_timing.json"), measure("maintenance"):
                with measure("cache_prune"):
                    keep = Deriver(plan["bank"], stated.device, RESIDENT).stamp(config)
                    prune(plan["bank"], stated.variant_cache_bytes, keep)
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
        files.drain()
        state.update(completed=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started)
        write_json(directory / "runner.json", state)
        emit("run_exit", status=state["status"], seconds=state["seconds"])


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--bank", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--preparation-seconds", type=int, required=True)
    run_parser.add_argument("--batch-sequences", type=int, required=True)
    run_parser.add_argument("--continuous", action="store_true")
    run_parser.add_argument("--trials", type=int)
    run_parser.add_argument("--seconds", type=int, required=True, help="time limit of every trial")
    run_parser.add_argument(
        "--selector",
        choices=("forest", "botorch", "tpe", "random"),
        help="how trials are chosen; stated for a new run, otherwise keeps the saved selector when omitted",
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
    run_parser.add_argument(
        "--objective",
        nargs="*",
        help="series=weight pairs that blend per-series validation losses into the trial score; every series weighs 1 when omitted on a new run",
    )
    run_parser.add_argument("--bounds")
    run_parser.add_argument("--workers", type=int, required=True)
    run_parser.add_argument("--variant-gb", type=int, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--bank", required=True)
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--training-blocks", required=True)
    worker_parser.add_argument("--batch-sequences", type=int, required=True)
    worker_parser.add_argument("--deadline", type=float, required=True)
    worker_parser.add_argument("--statement", required=True)
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
