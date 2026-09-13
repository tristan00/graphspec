import logging
from pathlib import Path
import time

import numpy as np

from graph_tuning_infra.files import load_arrays, read_json, save_arrays, write_json

from . import kernels


LOG = logging.getLogger(__name__)
VERSION = 6
ARRAYS = ('kinds', 'period', 'obs_entity', 'obs_step', 'obs_x', 'obs_anchor', 'fact_source', 'fact_target', 'fact_code', 'fact_step',
          'fact_present', 'fact_values', 'query_key', 'query_category', 'query_actor', 'query_subject', 'query_context',
          'query_x', 'query_signals', 'label_entity', 'label_step', 'label_row', 'label_id')


class Graph:

    def __init__(self, name, keys, steps, **arrays):
        missing = sorted(set(ARRAYS) - set(arrays))
        if missing:
            raise ValueError('graph %s lacks arrays: %s' % (name, ', '.join(missing)))
        self.__dict__.update(name=name, keys=list(keys), steps=int(steps), arrays={key: arrays[key] for key in ARRAYS})

    def __getattr__(self, key):
        try:
            return self.__dict__['arrays'][key]
        except KeyError:
            raise AttributeError(key) from None

    def replace(self, steps=None, **changes):
        return Graph(self.name, self.keys, self.steps if steps is None else steps, **dict(self.arrays, **changes))

    def save(self, directory):
        directory = Path(directory)
        save_arrays(directory, self.arrays)
        write_json(directory / 'graph.json', dict(name=self.name, keys=self.keys, steps=self.steps))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = read_json(directory / 'graph.json')
        return cls(meta['name'], meta['keys'], meta['steps'], **load_arrays(directory, ARRAYS))


def column(table, name, dtype):
    return table.column(name).to_numpy().astype(dtype)


def matrix(table, name, dtype):
    values = table.column(name).combine_chunks()
    return values.flatten().to_numpy().reshape(len(values), values.type.list_size).astype(dtype)


def untied(step, seq, label, name):
    if np.any((np.diff(step) == 0) & (np.diff(seq) == 0)):
        raise ValueError('two %s rows of graph %s share a step and seq' % (label, name))


def parse(registry, categories, tables, rows, name):
    started = time.perf_counter()
    nodes, observations = tables['nodes'], tables['observations']
    numbers = column(nodes, 'entity', np.int64)
    order = np.argsort(numbers, kind='stable')
    ordered = numbers[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError('two nodes of graph %s share an entity number' % name)

    def local(values):
        position = np.minimum(np.searchsorted(ordered, values), len(ordered) - 1)
        if not np.array_equal(ordered[position], values):
            raise ValueError('a row of graph %s names an entity outside the graph' % name)
        return position.astype(np.int32)

    kinds = np.asarray([registry.kind(kind) for kind in nodes.column('kind').to_pylist()], np.int8)[order]
    names = nodes.column('key').to_pylist()
    keys = [names[i] for i in order.tolist()]
    arrangement = np.lexsort((column(observations, 'seq', np.int64), column(observations, 'step', np.int64)))
    obs_step = column(observations, 'step', np.int32)[arrangement]
    untied(obs_step, column(observations, 'seq', np.int64)[arrangement], 'observation', name)
    obs_entity = local(column(observations, 'entity', np.int64)[arrangement])
    obs_x = matrix(observations, 'x', np.float16)[arrangement]
    width = max([len(registry.attributes(relation)) for relation in registry.extras['base']] + [0])
    parts = dict(step=[np.zeros(0, np.int64)], seq=[np.zeros(0, np.int64)], source=[np.zeros(0, np.int32)], target=[np.zeros(0, np.int32)],
                 code=[np.zeros(0, np.int64)], present=[np.zeros(0, bool)], values=[np.zeros((0, width), np.float32)])
    for relation, table in tables['edges'].items():
        edge = registry.by_name[relation]
        count = table.num_rows
        values = np.full((count, width), np.nan, np.float32)
        for position, attribute in enumerate(edge.attributes):
            values[:, position] = column(table, attribute, np.float32)
        for key, value in (('step', column(table, 'step', np.int64)), ('seq', column(table, 'seq', np.int64)),
                           ('source', local(column(table, 'source', np.int64))), ('target', local(column(table, 'target', np.int64))),
                           ('code', np.full(count, edge.code, np.int64)), ('present', column(table, 'present', bool)), ('values', values)):
            parts[key].append(value)
    facts = {key: np.concatenate(value) for key, value in parts.items()}
    arrangement = np.lexsort((facts['seq'], ~facts['present'], facts['step']))
    facts = {key: value[arrangement] for key, value in facts.items()}
    for relation in registry.attributed:
        edge = registry.by_name[relation]
        chosen = facts['present'] & (facts['code'] == edge.code)
        if not np.isfinite(facts['values'][chosen, :len(edge.attributes)]).all():
            raise ValueError('graph %s has a non-finite %s attribute on a present fact' % (name, relation))
    for flag, label in ((True, 'present fact'), (False, 'absent fact')):
        use = facts['present'] == flag
        untied(facts['step'][use], facts['seq'][use], label, name)
    fact_step = facts['step'].astype(np.int32)
    level = registry.extras['level']
    if registry.extras['queries']:
        queries = tables['queries']
        if not np.array_equal(column(queries, 'step', np.int64), np.arange(queries.num_rows)):
            raise ValueError('query steps of graph %s do not number 0..n-1' % name)
        steps = queries.num_rows
        position = {category: index for index, category in enumerate(categories)}
        query = dict(query_key=column(queries, 'key', np.int64),
                     query_category=np.asarray([position[value] for value in queries.column('category').to_pylist()], np.int64),
                     query_actor=local(column(queries, 'actor', np.int64)), query_subject=local(column(queries, 'subject', np.int64)),
                     query_context=local(column(queries, 'context', np.int64)), query_x=matrix(queries, 'query', np.float16),
                     query_signals=np.stack((matrix(queries, 'signal', np.float32), matrix(queries, 'signal_observed', np.float32)), 2)
                     if 'signal' in queries.column_names else np.zeros((steps, 0, 2), np.float32))
        period = column(queries, 'period', np.int32)
    else:
        query = dict(query_key=np.zeros(0, np.int64), query_category=np.zeros(0, np.int64), query_actor=np.zeros(0, np.int32),
                     query_subject=np.zeros(0, np.int32), query_context=np.zeros(0, np.int32),
                     query_x=np.zeros((0, registry.features), np.float16), query_signals=np.zeros((0, 0, 2), np.float32))
    if level == 'node':
        targets = tables['targets']
        entity, step = local(column(targets, 'entity', np.int64)), column(targets, 'step', np.int64)
        arranged = np.lexsort((entity, step))
        label_entity, label_step = entity[arranged], step[arranged].astype(np.int32)
        if np.any((np.diff(label_step) == 0) & (np.diff(label_entity) == 0)):
            raise ValueError('two labels of graph %s sit on one node at one step' % name)
        label_row, label_id = np.asarray(rows, np.int64)[arranged], column(targets, 'id', np.int64)[arranged]
    else:
        rows = np.asarray(rows, np.int64)
        label_step = np.flatnonzero(rows >= 0).astype(np.int32)
        label_row, label_id = rows[label_step], query['query_key'][label_step]
        label_entity = np.full(len(label_step), -1, np.int32)
    if not registry.extras['queries']:
        steps = 1 + int(max(obs_step.max(initial=-1), fact_step.max(initial=-1), label_step.max(initial=-1)))
        period = np.arange(steps, dtype=np.int32)
    graph = Graph(name, keys, steps, kinds=kinds, period=period, obs_entity=obs_entity, obs_step=obs_step, obs_x=obs_x,
                  obs_anchor=np.zeros(len(obs_step), bool), fact_source=facts['source'], fact_target=facts['target'], fact_code=facts['code'],
                  fact_step=fact_step, fact_present=facts['present'], fact_values=facts['values'], label_entity=label_entity, label_step=label_step,
                  label_row=label_row, label_id=label_id, **query)
    verify(graph, registry)
    LOG.info('truth exit graph=%s seconds=%.3f steps=%d observations=%d facts=%d labels=%d', name, time.perf_counter() - started,
             steps, len(obs_step), len(fact_step), len(label_step))
    return graph


def verify(graph, registry):
    name, steps, entities = graph.name, graph.steps, len(graph.kinds)
    flags = kernels.audit(graph.obs_entity, graph.obs_step, graph.obs_anchor, graph.fact_source, graph.fact_target, graph.fact_step, graph.fact_present,
                          graph.fact_code, registry.canonical, graph.query_actor, graph.query_subject, graph.query_context, graph.label_entity,
                          graph.label_step, registry.extras['level'] == 'node', steps, entities)
    for index, label in enumerate(('observation', 'fact', 'label')):
        if flags[index]:
            raise ValueError('a %s row of graph %s has a step outside 0..%d' % (label, name, steps - 1))
    for index, label in enumerate(('observation', 'fact source', 'fact target', 'query actor', 'query subject', 'query context', 'label'), 3):
        if flags[index]:
            raise ValueError('a %s of graph %s names an entity outside 0..%d' % (label, name, entities - 1))
    if len(graph.period) != steps or np.any(np.diff(graph.period) < 0):
        raise ValueError('periods of graph %s must cover every step and never decrease' % name)
    if flags[10]:
        raise ValueError('facts of graph %s carry codes that are not canonical relation codes' % name)
    if registry.extras['queries'] and len(graph.query_key) != steps:
        raise ValueError('graph %s needs exactly one query per step' % name)
    for index, message in ((11, 'observations of graph %s are not ordered by step'), (12, 'facts of graph %s are not ordered by step with present facts first'),
                           (13, 'labels of graph %s are not ordered by step'), (14, 'an entity of graph %s has two observations at one step')):
        if flags[index]:
            raise ValueError(message % name)
    for index, label, when in ((15, 'present fact', 'outside query anchors at its step'), (16, 'absent fact', 'at or before its step'),
                               (17, 'query', 'at its step'), (18, 'label', 'at its step')):
        if flags[index]:
            raise ValueError('a %s of graph %s names an entity with no observation %s' % (label, name, when))
