import argparse
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa

from graph_tuning_infra.database import Stage, connect, copy
from graph_tuning_infra.files import write_json
from graph_tuning_infra.log import configure


LOG = logging.getLogger(__name__)
NODES = ('person', 'place', 'item', 'team')
SHARES = (0.5, 0.2, 0.2, 0.1)
RELATIONS = (('visits', 'person', 'place', ('weight',)), ('owns', 'person', 'item', ()), ('member', 'person', 'team', ()),
             ('near', 'place', 'place', ()), ('located', 'team', 'place', ('distance',)))
ROLES = dict(actor='person', subject='place', context='team')
CATEGORIES = ('buy', 'sell', 'move')
TARGET = 'outcome'


def schema(features, signals, level):
    statements = ['CREATE NODE TABLE graph(id INT32 PRIMARY KEY, name STRING)']
    statements += ['CREATE NODE TABLE %s(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING)' % kind for kind in NODES]
    statements.append('CREATE NODE TABLE observation(id INT64 PRIMARY KEY, seq INT32, step INT32, x FLOAT[%d])' % features)
    signal = ', signal FLOAT[%d], signal_observed FLOAT[%d]' % (signals, signals) if signals else ''
    statements.append('CREATE NODE TABLE query(id INT64 PRIMARY KEY, graph INT32, step INT32, period INT32, key INT64, category STRING, x FLOAT[%d]%s)' % (features, signal))
    statements.append('CREATE NODE TABLE %s(id INT64 PRIMARY KEY, y DOUBLE, split STRING%s)' % (TARGET, ', step INT32' if level == 'node' else ''))
    statements.append('CREATE REL TABLE observed(%s)' % ', '.join('FROM %s TO observation' % kind for kind in NODES))
    statements += ['CREATE REL TABLE query_%s(FROM query TO %s)' % (role, kind) for role, kind in ROLES.items()]
    for name, source, target, attributes in RELATIONS:
        columns = ''.join(', %s FLOAT' % attribute for attribute in attributes)
        statements.append('CREATE REL TABLE edge_%s(FROM %s TO %s, seq INT32, step INT32, present BOOLEAN%s)' % (name, source, target, columns))
    statements.append('CREATE REL TABLE %s_of(FROM %s TO %s)' % (TARGET, TARGET, dict(query='query', graph='graph', node='person')[level]))
    return statements


def vectors(rng, rows, features):
    values = rng.normal(size=(rows, features // 2)).astype(np.float32)
    present = (rng.random((rows, features // 2)) < 0.9).astype(np.float32)
    return np.concatenate((values * present, present), 1)


def fixed(values):
    values = np.ascontiguousarray(values, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


class Counter:

    def __init__(self):
        self.next = {}

    def take(self, table, count):
        start = self.next.get(table, 0)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def graph_rows(rng, index, name, settings, counter, level):
    started = time.perf_counter()
    entities, steps, features, signals = settings['entities'], settings['steps'], settings['features'], settings['signals']
    counts = np.maximum(np.floor(np.asarray(SHARES) * entities).astype(np.int64), 2)
    offsets = np.r_[0, np.cumsum(counts)]
    total = int(offsets[-1])
    kinds = np.repeat(np.arange(len(NODES)), counts)
    birth = rng.integers(0, max(1, steps // 4), total)
    observed = (rng.random((steps, total)) < settings['alive']) & (np.arange(steps)[:, None] >= birth[None, :])
    roles = {}
    for role, kind in ROLES.items():
        members = np.arange(offsets[NODES.index(kind)], offsets[NODES.index(kind) + 1])
        roles[role] = rng.choice(members, steps)
        observed[np.arange(steps), roles[role]] = True
    facts = {}
    for relation, source_kind, target_kind, attributes in RELATIONS:
        sources = np.arange(offsets[NODES.index(source_kind)], offsets[NODES.index(source_kind) + 1])
        targets = np.arange(offsets[NODES.index(target_kind)], offsets[NODES.index(target_kind) + 1])
        degree = np.minimum(rng.poisson(settings['degree'], len(sources)), settings['max_degree'])
        source = np.repeat(sources, degree)
        target = rng.choice(targets, len(source))
        keep = source != target
        source, target = source[keep], target[keep]
        pairs = np.unique(source.astype(np.int64) * total + target)
        source, target = pairs // total, pairs % total
        begin = rng.integers(0, steps, len(source))
        length = np.minimum(rng.geometric(1.0 / settings['lifetime'], len(source)), steps - begin)
        pair = np.repeat(np.arange(len(source)), length)
        step = begin[pair] + np.arange(len(pair)) - np.repeat(np.cumsum(length) - length, length)
        both = observed[step, source[pair]] & observed[step, target[pair]]
        present = np.stack((source[pair][both], target[pair][both], step[both]), 1)
        stated = np.zeros(len(source), bool)
        stated[pair[both]] = True
        ended = stated & (begin + length < steps)
        absent = np.stack((source[ended], target[ended], (begin + length)[ended]), 1)
        earlier = observed.cumsum(0) > 0
        good = earlier[absent[:, 2], absent[:, 0]] & earlier[absent[:, 2], absent[:, 1]]
        facts[relation] = (present, absent[good], attributes)
    obs_step, obs_entity = np.nonzero(observed)
    order = np.lexsort((obs_entity, obs_step))
    obs_step, obs_entity = obs_step[order], obs_entity[order]
    seq = np.arange(len(obs_step)) - np.searchsorted(obs_step, obs_step)
    base = vectors(rng, total, features)
    x = base[obs_entity] + np.concatenate((rng.normal(scale=0.3, size=(len(obs_step), features // 2)).astype(np.float32), np.zeros((len(obs_step), features // 2), np.float32)), 1)
    x[:, :features // 2] *= x[:, features // 2:]
    node_ids = {kind: counter.take(kind, int(counts[position])) for position, kind in enumerate(NODES)}
    entity_id = np.concatenate([node_ids[kind] for kind in NODES])
    observation_ids = counter.take('observation', len(obs_step))
    query_ids = counter.take('query', steps)
    tables = {('graph', None, None): pa.table(dict(id=pa.array([index], pa.int32()), name=pa.array([name])))}
    for position, kind in enumerate(NODES):
        members = np.arange(offsets[position], offsets[position + 1])
        tables[(kind, None, None)] = pa.table(dict(id=pa.array(node_ids[kind]), graph=pa.array(np.full(len(members), index), pa.int32()),
                                                   entity=pa.array(members, pa.int32()), key=pa.array(['%s_%d' % (kind, member) for member in members])))
        picked = kinds[obs_entity] == position
        tables[('observed', kind, 'observation')] = pa.table({'from': pa.array(entity_id[obs_entity[picked]]), 'to': pa.array(observation_ids[picked])})
    tables[('observation', None, None)] = pa.table(dict(id=pa.array(observation_ids), seq=pa.array(seq, pa.int32()), step=pa.array(obs_step, pa.int32()), x=fixed(x)))
    category = rng.integers(0, len(CATEGORIES), steps)
    query = dict(id=pa.array(query_ids), graph=pa.array(np.full(steps, index), pa.int32()), step=pa.array(np.arange(steps), pa.int32()),
                 period=pa.array(np.arange(steps) // settings['period'], pa.int32()), key=pa.array(query_ids * 7 + 3),
                 category=pa.array([CATEGORIES[value] for value in category]), x=fixed(vectors(rng, steps, features)))
    if signals:
        query.update(signal=fixed(rng.normal(size=(steps, signals))), signal_observed=fixed(rng.random((steps, signals)) < 0.7))
    tables[('query', None, None)] = pa.table(query)
    for role, kind in ROLES.items():
        tables[('query_' + role, 'query', kind)] = pa.table({'from': pa.array(query_ids), 'to': pa.array(entity_id[roles[role]])})
    fact_seq = {}
    for relation, source_kind, target_kind, attributes in RELATIONS:
        present, absent, _ = facts[relation]
        rows = np.concatenate((present, absent))
        flags = np.r_[np.ones(len(present), bool), np.zeros(len(absent), bool)]
        sequence = np.empty(len(rows), np.int64)
        for flag in (True, False):
            picked = np.flatnonzero(flags == flag)
            for step in np.unique(rows[picked, 2]):
                members = picked[rows[picked, 2] == step]
                start = fact_seq.get((flag, step), 0)
                sequence[members] = np.arange(start, start + len(members))
                fact_seq[(flag, step)] = start + len(members)
        columns = {'from': pa.array(entity_id[rows[:, 0]]), 'to': pa.array(entity_id[rows[:, 1]]), 'seq': pa.array(sequence, pa.int32()),
                   'step': pa.array(rows[:, 2], pa.int32()), 'present': pa.array(flags)}
        for attribute in attributes:
            values = rng.random(len(rows)).astype(np.float32)
            values[~flags] = np.nan
            columns[attribute] = pa.array(values, pa.float32())
        tables[('edge_' + relation, source_kind, target_kind)] = pa.table(columns)
    actor_x = base[roles['actor'], :features // 2]
    if level == 'node':
        moments = np.arange(settings['period'] - 1, steps, settings['period'])
        persons = np.arange(offsets[0], offsets[1])
        label_step, label_person = np.nonzero(observed[moments][:, persons])
        label_step, label_person = moments[label_step], persons[label_person]
        y = base[label_person, :3].sum(1) + 0.1 * label_step + rng.normal(scale=0.5, size=len(label_step))
        split = np.where(label_step >= steps * 0.8, 'validation', 'train')
        label_ids = counter.take(TARGET, len(y))
        tables[(TARGET, None, None)] = pa.table(dict(id=pa.array(label_ids), y=pa.array(y, pa.float64()), split=pa.array(split), step=pa.array(label_step, pa.int32())))
        tables[(TARGET + '_of', TARGET, 'person')] = pa.table({'from': pa.array(label_ids), 'to': pa.array(entity_id[label_person])})
    elif level == 'query':
        y = actor_x[:, :3].sum(1) + (category == 0) + rng.normal(scale=0.5, size=steps)
        split = np.where(np.arange(steps) >= steps * 0.8, 'validation', 'train')
        label_ids = counter.take(TARGET, steps)
        tables[(TARGET, None, None)] = pa.table(dict(id=pa.array(label_ids), y=pa.array(y, pa.float64()), split=pa.array(split)))
        tables[(TARGET + '_of', TARGET, 'query')] = pa.table({'from': pa.array(label_ids), 'to': pa.array(query_ids)})
    else:
        label_ids = counter.take(TARGET, 1)
        tables[(TARGET, None, None)] = pa.table(dict(id=pa.array(label_ids), y=pa.array([float(actor_x[:, 0].mean() + rng.normal(scale=0.2))], pa.float64()),
                                                     split=pa.array(['validation' if index % 2 == 1 else 'train'])))
        tables[(TARGET + '_of', TARGET, 'graph')] = pa.table({'from': pa.array(label_ids), 'to': pa.array([index], pa.int64())})
    LOG.info('synthetic graph exit graph=%d seconds=%.3f entities=%d observations=%d facts=%d', index, time.perf_counter() - started, total, len(obs_step),
             sum(len(present) + len(absent) for present, absent, _ in facts.values()))
    return tables


def generate(output, name, graphs, level, seed, **settings):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database_path = output / (name + '.lbug')
    if database_path.exists() or output.joinpath(name + '.staging').exists():
        raise FileExistsError('synthetic database exists; choose another name or output')
    rng = np.random.default_rng(seed)
    counter = Counter()
    stage = Stage(output / (name + '.staging'))
    for index in range(graphs):
        stage.write(graph_rows(rng, index, 'graph_%03d' % index, settings, counter, level))
    stage.close()
    database, connection = connect(database_path, read_only=False)
    for statement in schema(settings['features'], settings['signals'], level):
        connection.execute(statement)
    copy(connection, stage)
    connection.close()
    database.close()
    shutil.rmtree(stage.directory)
    write_json(output / (name + '.task.json'), dict(name=name, database=name + '.lbug', target=TARGET))
    LOG.info('synthetic exit seconds=%.3f database=%s graphs=%d level=%s', time.perf_counter() - started, database_path, graphs, level)
    return output / (name + '.task.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--graphs', type=int, default=4)
    parser.add_argument('--entities', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--signals', type=int, default=2)
    parser.add_argument('--degree', type=float, default=3.0)
    parser.add_argument('--max-degree', type=int, default=12)
    parser.add_argument('--alive', type=float, default=0.8)
    parser.add_argument('--lifetime', type=float, default=6.0)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--level', choices=('query', 'node', 'graph'), default='query')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    configure()
    task = generate(args.output, args.name, args.graphs, args.level, args.seed, entities=args.entities, steps=args.steps, features=args.features,
                    signals=args.signals, degree=args.degree, max_degree=args.max_degree, alive=args.alive, lifetime=args.lifetime, period=args.period)
    print(task)


if __name__ == '__main__':
    main()
