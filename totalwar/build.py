import argparse
import logging
from pathlib import Path
import shutil
import time

import numpy as np

from graph_tuning_infra.database import Stage, connect, copy
from graph_tuning_infra.features import encoder
from graph_tuning_infra.files import write_json
from graph_tuning_infra.log import configure
from graph_tuning_infra.postgres import connect as postgres

from . import corpus
from .domain import CATEGORIES, NUMBERS, SIGNALS, TARGET, declaration
from .graph import Campaign
from .reference import Reference


LOG = logging.getLogger(__name__)


class Counter:

    def __init__(self):
        self.next = {}

    def take(self, table, count):
        start = self.next.get(table, 0)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def build(output, name, host, port, database, user, campaigns, features):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database_path = output / (name + '.lbug')
    if database_path.exists() or output.joinpath(name + '.staging').exists():
        raise FileExistsError('database exists; choose another name or output')
    connection = postgres(host, port, database, user)
    reference = Reference(connection)
    encode = encoder(features // 2, list(NUMBERS), list(CATEGORIES))
    chosen = corpus.campaigns(connection, campaigns)
    counter = Counter()
    stage = Stage(output / (name + '.staging'))
    summary = []
    for index, campaign in enumerate(chosen):
        clock = time.perf_counter()
        rows = corpus.load(connection, campaign['campaign_id'])
        tables = Campaign(reference, encode, features, campaign, rows).build(counter, index, campaign['campaign_key'])
        stage.write(tables)
        summary.append(dict(index=index, campaign=campaign['campaign_key'], faction=campaign['faction'], steps=tables[('query', None, None)].num_rows,
                            entities=sum(table.num_rows for (kind, source, _), table in tables.items() if source is None and kind not in ('graph', 'observation', 'query', TARGET)),
                            observations=tables[('observation', None, None)].num_rows,
                            facts=sum(table.num_rows for (kind, _, _), table in tables.items() if kind.startswith('edge_')),
                            labels=tables[(TARGET, None, None)].num_rows, seconds=time.perf_counter() - clock))
        LOG.info('campaign exit index=%d key=%s seconds=%.2f', index, campaign['campaign_key'], summary[-1]['seconds'])
    stage.close()
    connection.close()
    store, ladybug = connect(database_path, read_only=False)
    for statement in declaration(features, len(SIGNALS)):
        ladybug.execute(statement)
    copy(ladybug, stage)
    ladybug.close()
    store.close()
    shutil.rmtree(stage.directory)
    write_json(output / (name + '.task.json'), dict(name=name, database=name + '.lbug', target=TARGET))
    write_json(output / (name + '.build.json'), dict(host=host, port=port, database=database, campaigns=campaigns, features=features, graphs=summary,
                                                    seconds=time.perf_counter() - started))
    LOG.info('build exit seconds=%.1f graphs=%d observations=%d facts=%d', time.perf_counter() - started, len(summary),
             sum(item['observations'] for item in summary), sum(item['facts'] for item in summary))
    return output / (name + '.task.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', default='totalwar')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=55433)
    parser.add_argument('--database', default='tw_stack')
    parser.add_argument('--user', default='tw')
    parser.add_argument('--campaigns', type=int, default=100)
    parser.add_argument('--features', type=int, default=128)
    args = parser.parse_args()
    configure()
    print(build(args.output, args.name, args.host, args.port, args.database, args.user, args.campaigns, args.features))


if __name__ == '__main__':
    main()
