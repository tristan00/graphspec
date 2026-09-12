import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
import time

import numpy as np

from graphspec import Registry, graphdb, vocabulary
from graphspec.truth import VERSION, parse
from graph_tuning_infra.database import connect, query
from graph_tuning_infra.files import read_json, sha256_file, write_json
from graph_tuning_infra.log import configure


LOG = logging.getLogger(__name__)
SPLITS = ('train', 'validation')
WORKER = {}


def supervision(connection, level, target, names):
    started = time.perf_counter()
    if level == 'node':
        found = query(connection, 'MATCH (t:`%s`)-[:`%s_of`]->(n) RETURN t.id AS id, t.y AS y, t.split AS split, t.step AS step, '
                                  'n.graph AS graph, n.entity AS entity, n.key AS node ORDER BY id' % (target, target))
        owners, steps = found.column('graph').to_numpy().astype(np.int64), found.column('step').to_numpy().astype(np.int64)
        if np.any(steps < 0):
            raise ValueError('target steps must not be negative')
        if len(np.unique(np.stack((owners, found.column('entity').to_numpy().astype(np.int64), steps), 1), axis=0)) != len(owners):
            raise ValueError('a node carries more than one target at one step')
        described = [dict(graph=names[owner], step=step, node=node)
                     for owner, step, node in zip(owners.tolist(), steps.tolist(), found.column('node').to_pylist())]
    else:
        queries = query(connection, 'MATCH (q:query) RETURN q.graph AS graph, q.step AS step, q.period AS period, q.key AS key')
        counts = dict(zip(*(values.tolist() for values in np.unique(queries.column('graph').to_numpy(), return_counts=True))))
        empty = sorted(set(names) - set(counts))
        if empty:
            raise ValueError('%d graphs have no queries, for example graph %d' % (len(empty), empty[0]))
        lookup = {(g, s): (p, k) for g, s, p, k in zip(*(queries.column(name).to_pylist() for name in ('graph', 'step', 'period', 'key')))}
        if level == 'query':
            found = query(connection, 'MATCH (t:`%s`)-[:`%s_of`]->(q:query) RETURN t.id AS id, t.y AS y, t.split AS split, '
                                      'q.graph AS graph, q.step AS step ORDER BY id' % (target, target))
            owners, steps = found.column('graph').to_numpy().astype(np.int64), found.column('step').to_numpy().astype(np.int64)
        else:
            found = query(connection, 'MATCH (t:`%s`)-[:`%s_of`]->(g:graph) RETURN t.id AS id, t.y AS y, t.split AS split, g.id AS graph ORDER BY id' % (target, target))
            owners = found.column('graph').to_numpy().astype(np.int64)
            steps = np.asarray([counts[owner] - 1 for owner in owners.tolist()], np.int64)
        if len(np.unique(np.stack((owners, steps), 1), axis=0)) != len(owners):
            raise ValueError('a query or graph carries more than one target')
        described = [dict(graph=names[owner], decision=lookup[(owner, step)][1], step=step, turn=lookup[(owner, step)][0])
                     for owner, step in zip(owners.tolist(), steps.tolist())]
    split = found.column('split').to_pylist()
    unknown = sorted(set(split) - set(SPLITS))
    if unknown:
        raise ValueError('target splits must be train or validation, found ' + ', '.join(map(str, unknown)))
    unlabelled = sorted(set(names) - set(owners.tolist()))
    if unlabelled:
        raise ValueError('%d graphs have no target, for example graph %d' % (len(unlabelled), unlabelled[0]))
    held = np.asarray([value == 'validation' for value in split])
    y = found.column('y').to_numpy().astype(np.float64)
    train, validation = np.flatnonzero(~held), np.flatnonzero(held)
    stored = y.astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError('targets must be finite at float32 precision')
    if len(train) < 2 or len(validation) < 2 or np.ptp(stored[train]) == 0 or np.ptp(stored[validation]) == 0:
        raise ValueError('need at least two varying training targets and two varying validation targets')
    rows = [dict(entry, y=value) for entry, value in zip(described, y.tolist())]
    result = dict(rows=rows, split=dict(train_indices=train.tolist(), validation_indices=validation.tolist(),
                                        reward_mean=float(y[train].mean()), reward_sd=float(y[train].std(ddof=1))))
    ids = found.column('id').to_numpy().astype(np.int64)
    order = np.argsort(owners, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(owners[order])) + 1, len(order)]
    groups = [(int(owners[order[lo]]), steps[order[lo:hi]], ids[order[lo:hi]], order[lo:hi].astype(np.int64))
              for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    LOG.info('supervision exit seconds=%.3f rows=%d train=%d validation=%d level=%s', time.perf_counter() - started,
             len(rows), len(train), len(validation), level)
    return result, groups


def attach(database, declared, categories, target, output, level):
    configure(level)
    store, connection = connect(database, read_only=True, buffer_pool_size=4 << 30)
    WORKER.update(reader=graphdb.Reader(store, connection, graphdb.Schema(connection), target),
                  registry=Registry.from_declaration(declared), categories=categories, output=Path(output))


def graph(job):
    index, name, steps, ids, rows = job
    started = time.perf_counter()
    registry = WORKER['registry']
    tables = WORKER['reader'].graph(index)
    if registry.extras['level'] == 'node':
        found = tables['targets'].column('id').to_numpy().astype(np.int64)
        position = np.searchsorted(ids, found)
        if len(found) != len(ids) or not np.array_equal(ids[np.minimum(position, len(ids) - 1)], found):
            raise ValueError('graph %s reads different targets than its supervision' % name)
        labels = rows[position]
    else:
        labels = np.full(tables['queries'].num_rows, -1, np.int64)
        labels[steps] = rows
    truth = parse(registry, WORKER['categories'], tables, labels, name)
    truth.save(WORKER['output'] / 'truth' / ('graph_%04d' % index))
    metrics = dict(index=index, name=name, steps=truth.steps, entities=len(truth.kinds), observations=len(truth.obs_step),
                   facts=len(truth.fact_step), labels=len(truth.label_step), seconds=time.perf_counter() - started)
    LOG.info('bank graph exit graph=%d seconds=%.3f', index, metrics['seconds'])
    return metrics


def build(task_path, output, workers):
    started = time.perf_counter()
    task, output = read_json(task_path), Path(output)
    if output.exists():
        raise FileExistsError('bank exists; choose a new output')
    database = (Path(task_path).parent / task['database']).resolve()
    store, connection = connect(database, read_only=True, buffer_pool_size=4 << 30)
    schema = graphdb.Schema(connection)
    registry = vocabulary.registry(schema, task)
    graphdb.check(registry)
    level = registry.extras['level']
    graphs = query(connection, 'MATCH (g:graph) RETURN g.id AS id, g.name AS name ORDER BY id')
    names = dict(zip(graphs.column('id').to_pylist(), graphs.column('name').to_pylist()))
    if len(set(names.values())) != len(names):
        raise ValueError('graph names must be unique')
    categories = []
    if schema.queries:
        categories = query(connection, 'MATCH (q:query) RETURN DISTINCT q.category AS category ORDER BY category').column('category').to_pylist()
    rows, groups = supervision(connection, level, task['target'], names)
    connection.close()
    store.close()
    (output / 'truth').mkdir(parents=True)
    write_json(output / 'supervision.json', rows)
    jobs = [(index, names[index], steps, ids, members) for index, steps, ids, members in groups]
    metrics = []
    with ProcessPoolExecutor(max_workers=workers, initializer=attach, initargs=(str(database), registry.declaration(), categories,
                                                                              task['target'], str(output), LOG.getEffectiveLevel())) as pool:
        for done, entry in enumerate(pool.map(graph, jobs, chunksize=4), 1):
            metrics.append(entry)
            if done % 100 == 0 or done == len(jobs):
                LOG.info('bank progress graphs=%d/%d seconds=%.1f', done, len(jobs), time.perf_counter() - started)
    manifest = dict(version=VERSION, database=str(database), task=task, target=dict(table=task['target'], level=level),
                    spec=registry.declaration(), spec_sha256=registry.fingerprint(), categories=categories, signals=registry.extras['signals'],
                    graphs=metrics, rows=len(rows['rows']), train_rows=len(rows['split']['train_indices']),
                    validation_rows=len(rows['split']['validation_indices']), supervision_sha256=sha256_file(output / 'supervision.json'),
                    seconds=time.perf_counter() - started)
    write_json(output / 'manifest.json', manifest)
    LOG.info('bank exit seconds=%.3f graphs=%d rows=%d level=%s', manifest['seconds'], len(metrics), manifest['rows'], level)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    configure()
    manifest = build(args.task, args.output, args.workers)
    print(json.dumps({key: value for key, value in manifest.items() if key not in ('graphs', 'spec')}, indent=2))


if __name__ == '__main__':
    main()
