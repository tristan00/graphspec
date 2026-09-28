import argparse
import hashlib
import logging
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

from . import source
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


WORKER = None


def held(run):
    digest = hashlib.blake2b(("%s:%d" % (SALT, run)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 1000 < HELD


def attach(host, port, database, user, features):
    global WORKER
    configure()
    connection = postgres(host, port, database, user, TIMEOUT_MS)
    Finalize(connection, connection.close, exitpriority=0)
    names = source.names_of(connection)
    columns = source.columns_of(connection)
    parameters = source.parameters_of(connection)
    numbers, categories = allowed_fields(columns, parameters, names)
    WORKER = Worker(
        connection,
        names,
        columns,
        parameters,
        encoder(features // 2, numbers, categories),
        frozenset(numbers + categories),
        features,
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


@exclusive
def build(output, name, host, port, database, user, runs, chosen, features, workers, staged=False):
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
        stage = Stage.merge(staging, [directory for directory, _ in jobs])
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
        )
    )


if __name__ == "__main__":
    main()
