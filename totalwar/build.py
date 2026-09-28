import argparse
import logging
import time
from contextlib import closing
from dataclasses import dataclass
from multiprocessing.util import Finalize

import numpy as np

from graphspec.infra.features import encoder
from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure
from graphspec.infra.postgres import connect as postgres
from graphspec.infra.processes import WORKERS, parallel
from graphspec.infra.runtime import exclusive, output_path
from graphspec.infra.staging import Stage
from graphspec.staged import IDS
from graphspec.staging import write_task

from . import corpus
from .domain import CATEGORIES, NUMBERS, TARGET
from .graph import Campaign
from .reference import Reference

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Worker:
    connection: object
    reference: Reference
    encode: object
    features: int


WORKER = None


class Counter:
    def __init__(self, base):
        self.base, self.next = base, {}

    def take(self, table, count):
        start = self.next.get(table, self.base)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def attach(host, port, database, user, features):
    global WORKER
    configure()
    connection = postgres(host, port, database, user)
    Finalize(connection, connection.close, exitpriority=0)
    WORKER = Worker(
        connection, Reference(connection), encoder(features // 2, list(NUMBERS), list(CATEGORIES)), features
    )


def extract(job):
    if WORKER is None:
        raise RuntimeError("campaign worker must be initialized before extraction")
    directory, chosen = job
    started = time.perf_counter()
    summary = []
    with Stage(directory) as stage:
        for index, campaign in chosen:
            clock = time.perf_counter()
            rows = corpus.load(WORKER.connection, campaign["campaign_id"])
            tables = Campaign(WORKER.reference, WORKER.encode, WORKER.features, campaign, rows).build(
                Counter(index << IDS), index, campaign["campaign_key"]
            )
            stage.write(tables)
            summary.append(
                dict(
                    index=index,
                    campaign=campaign["campaign_key"],
                    faction=campaign["faction"],
                    steps=tables[("query", None, None)].num_rows,
                    entities=sum(
                        table.num_rows
                        for (kind, source, _), table in tables.items()
                        if source is None and kind not in ("graph", "observation", "query", TARGET)
                    ),
                    observations=tables[("observation", None, None)].num_rows,
                    facts=sum(
                        table.num_rows for (kind, _, _), table in tables.items() if kind.startswith("edge_")
                    ),
                    labels=tables[(TARGET, None, None)].num_rows,
                    seconds=time.perf_counter() - clock,
                )
            )
            LOG.info(
                "campaign exit index=%d key=%s seconds=%.2f",
                index,
                campaign["campaign_key"],
                summary[-1]["seconds"],
            )
    LOG.info(
        "shard exit directory=%s campaigns=%d seconds=%.1f",
        directory,
        len(chosen),
        time.perf_counter() - started,
    )
    return summary


@exclusive
def build(output, name, host, port, database, user, campaigns, features, workers, staged=False):
    started = time.perf_counter()
    if campaigns < 1 or workers < 1:
        raise ValueError("campaigns and workers must be positive")
    if features < 2 or features % 2:
        raise ValueError("features must be a positive even width")
    output = output_path(output)
    staging = output / (name + ".staging")
    provenance = dict(host=host, port=port, database=database, campaigns=campaigns, features=features)
    if staged:
        stage = Stage.open(staging)
        summary = read_json(staging / "summary.json")
        LOG.info("staging reused directory=%s units=%d graphs=%d", staging, len(stage.files), len(summary))
    else:
        if staging.exists() or (output / (name + ".task.json")).exists():
            raise FileExistsError("output exists; choose another name or output")
        with closing(postgres(host, port, database, user)) as connection:
            chosen = list(enumerate(corpus.campaigns(connection, campaigns)))
        if not chosen:
            raise ValueError("no campaigns are available")
        shards = min(workers, len(chosen))
        jobs = [
            (
                str(staging / ("shard_%02d" % shard)),
                chosen[shard * len(chosen) // shards : (shard + 1) * len(chosen) // shards],
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
        "build exit seconds=%.1f graphs=%d observations=%d facts=%d units=%d",
        time.perf_counter() - started,
        len(summary),
        sum(item["observations"] for item in summary),
        sum(item["facts"] for item in summary),
        len(stage.files),
    )
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="totalwar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55433)
    parser.add_argument("--database", default="tw_stack")
    parser.add_argument("--user", default="tw")
    parser.add_argument("--campaigns", type=int, default=100)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--workers", type=int, default=WORKERS)
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
            args.campaigns,
            args.features,
            args.workers,
            args.staged,
        )
    )


if __name__ == "__main__":
    main()
