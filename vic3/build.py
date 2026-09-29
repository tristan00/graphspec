import argparse
import hashlib
import logging
import math
import random
import time
from contextlib import closing
from dataclasses import dataclass
from multiprocessing.util import Finalize

from graphspec.infra.features import encoder
from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure
from graphspec.infra.postgres import connect as postgres
from graphspec.infra.processes import parallel
from graphspec.infra.runtime import exclusive, output_path
from graphspec.infra.staging import Stage
from graphspec.staging import write_task

from . import redraw, source
from .domain import HELD, SALT, TARGET
from .graph import Run, allowed_fields
from .tables import Counter

LOG = logging.getLogger(__name__)
TIMEOUT_MS = 60000


@dataclass(frozen=True)
class Worker:
    connection: object
    names: dict
    columns: dict
    parameters: dict
    encode: object
    allowed: frozenset
    features: int
    address: str


WORKER = None


def drawn(*parts):
    text = ":".join(str(part) for part in (SALT,) + parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def held(run):
    return drawn(run) % 1000 < HELD


def sampled(summary, decisions):
    started = time.perf_counter()
    kept = [item for item in summary if item["split"] == "validation"]
    share = math.ceil(decisions / len(kept))
    chosen = {
        item["run"]: sorted(
            random.Random(drawn(item["run"], "spread")).sample(
                item["labelled"], min(share, len(item["labelled"]))
            )
        )
        for item in kept
    }
    LOG.info(
        "sampled exit seconds=%.3f runs=%d decisions=%d",
        time.perf_counter() - started,
        len(chosen),
        sum(len(steps) for steps in chosen.values()),
    )
    return chosen


def attach(host, port, database, user, features):
    global WORKER
    configure()
    connection = postgres(host, port, database, user, TIMEOUT_MS)
    Finalize(connection, connection.close, exitpriority=0)
    names = source.names_of(connection)
    columns = source.columns_of(connection)
    parameters = source.parameters_of(connection)
    numbers, categories = allowed_fields(columns, parameters, names, source.modifier_names(connection))
    WORKER = Worker(
        connection,
        names,
        columns,
        parameters,
        encoder(features // 2, numbers, categories),
        frozenset(numbers + categories),
        features,
        "host=%s port=%d dbname=%s user=%s" % (host, port, database, user),
    )


def extract(job):
    if WORKER is None:
        raise RuntimeError("run worker must be initialized before extraction")
    directory, chosen = job
    started = time.perf_counter()
    summary = []
    with Stage(directory) as stage:
        for index, run in chosen:
            clock = time.perf_counter()
            data = source.load(WORKER.connection, run)
            graph = Run(
                WORKER.encode,
                WORKER.features,
                run,
                data,
                WORKER.names,
                WORKER.columns,
                WORKER.parameters,
                WORKER.allowed,
            )
            counts = graph.build(stage, Counter(index), index, "run-%d" % run["run_id"], held(run["run_id"]))
            summary.append(
                dict(
                    index=index,
                    run=run["run_id"],
                    kind=run["kind"],
                    end_reason=run["end_reason"],
                    split="validation" if held(run["run_id"]) else "train",
                    seconds=time.perf_counter() - clock,
                    **counts,
                )
            )
    LOG.info(
        "shard exit directory=%s runs=%d seconds=%.1f", directory, len(chosen), time.perf_counter() - started
    )
    return summary


def offer(job):
    if WORKER is None:
        raise RuntimeError("run worker must be initialized before extraction")
    directory, chosen, project, blocks = job
    started = time.perf_counter()
    summary = []
    with Stage(directory) as stage:
        for first, run, steps in chosen:
            clock = time.perf_counter()
            graph = Run(
                WORKER.encode,
                WORKER.features,
                run,
                source.load(WORKER.connection, run),
                WORKER.names,
                WORKER.columns,
                WORKER.parameters,
                WORKER.allowed,
            )
            moments = [[graph.steps[step], drawn(run["run_id"], graph.steps[step])] for step in steps]
            advisor, found = redraw.drawn(
                project, WORKER.address, run["run_id"], moments, blocks - 1, TIMEOUT_MS
            )
            written = graph.offered(stage, first, "run-%d" % run["run_id"], dict(zip(steps, found)), advisor)
            if len(written) != len(steps) * blocks:
                raise ValueError("run %d wrote %d of its offered graphs" % (run["run_id"], len(written)))
            share = (time.perf_counter() - clock) / len(written)
            summary.extend(
                dict(
                    item,
                    run=run["run_id"],
                    kind=run["kind"],
                    end_reason=run["end_reason"],
                    split="spread",
                    seconds=share,
                    entities=len(graph.kinds),
                    absent=graph.absent,
                    labelled=[item["steps"] - 1],
                )
                for item in written
            )
    LOG.info(
        "offer exit directory=%s graphs=%d seconds=%.1f",
        directory,
        len(summary),
        time.perf_counter() - started,
    )
    return summary


def offers(staging, picked, summary, spread, workers, settings):
    started = time.perf_counter()
    project, decisions, blocks = spread
    if decisions < 1 or blocks < 2:
        raise ValueError("spread needs a decision and two blocks")
    chosen = sampled(summary, decisions)
    rows = {run["run_id"]: run for _, run in picked}
    first, planned = len(picked), []
    for run, steps in sorted(chosen.items()):
        if steps:
            planned.append((first, rows[run], steps))
            first += len(steps) * blocks
    shards = min(workers, len(planned))
    jobs = [
        (
            str(staging / ("offer_%02d" % shard)),
            planned[shard * len(planned) // shards : (shard + 1) * len(planned) // shards],
            project,
            blocks,
        )
        for shard in range(shards)
    ]
    found = [item for part in parallel(offer, jobs, shards, attach, settings) for item in part]
    LOG.info("offers exit seconds=%.1f graphs=%d", time.perf_counter() - started, len(found))
    return [directory for directory, *_ in jobs], found


@exclusive
def build(
    output, name, host, port, database, user, runs, chosen, features, workers, staged=False, spread=None
):
    started = time.perf_counter()
    if runs < 1 or workers < 1:
        raise ValueError("runs and workers must be positive")
    if features < 2 or features % 2:
        raise ValueError("features must be a positive even width")
    output = output_path(output)
    staging = output / (name + ".staging")
    provenance = dict(host=host, port=port, database=database, runs=runs, chosen=chosen, features=features)
    if staged:
        stage = Stage.open(staging)
        summary = read_json(staging / "summary.json")
    else:
        if staging.exists() or (output / (name + ".task.json")).exists():
            raise FileExistsError("output exists; choose another name or output")
        with closing(postgres(host, port, database, user, TIMEOUT_MS)) as connection:
            picked = list(enumerate(source.runs(connection, runs, chosen)))
        if not picked:
            raise ValueError("no finished runs are available")
        shards = min(workers, len(picked))
        jobs = [
            (
                str(staging / ("shard_%02d" % shard)),
                picked[shard * len(picked) // shards : (shard + 1) * len(picked) // shards],
            )
            for shard in range(shards)
        ]
        summary = [
            item
            for part in parallel(extract, jobs, shards, attach, (host, port, database, user, features))
            for item in part
        ]
        folders = [directory for directory, _ in jobs]
        if spread is not None:
            extra, found = offers(
                staging, picked, summary, spread, workers, (host, port, database, user, features)
            )
            folders += extra
            summary += found
        stage = Stage.merge(staging, folders)
        write_json(staging / "summary.json", summary)
    task = write_task(output / (name + ".task.json"), name, staging, TARGET)
    write_json(
        output / (name + ".build.json"),
        dict(provenance, graphs=summary, seconds=time.perf_counter() - started),
    )
    LOG.info(
        "build exit seconds=%.1f graphs=%d steps=%d labels=%d units=%d",
        time.perf_counter() - started,
        len(summary),
        sum(item["steps"] for item in summary),
        sum(item["labels"] for item in summary),
        len(stage.files),
    )
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="vic3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55433)
    parser.add_argument("--database", default="vic3")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--run", type=int, action="append", default=[])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--redraw", help="the vic3 project whose redraw draws blocks for stored pauses")
    parser.add_argument("--spread-decisions", type=int, default=60)
    parser.add_argument("--spread-blocks", type=int, default=8)
    args = parser.parse_args()
    configure()
    print(
        build(
            args.output,
            args.name,
            args.host,
            args.port,
            args.database,
            args.user,
            args.runs,
            args.run,
            args.features,
            args.workers,
            args.staged,
            None if args.redraw is None else (args.redraw, args.spread_decisions, args.spread_blocks),
        )
    )


if __name__ == "__main__":
    main()
