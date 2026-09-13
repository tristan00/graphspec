import logging
import time

import numpy as np

from graph_tuning_infra.features import token

from .truth import ARRAYS, Graph, verify


LOG = logging.getLogger(__name__)
SCALE = float(1 << 53)


def sampled(kind, key, fraction):
    return token(kind + ':' + key) % (1 << 53) / SCALE < fraction


def parameters(spec):
    return {name: value for name, value in spec.items() if name.startswith('graphs/')}


def joined(graphs, registry, spec):
    started = time.perf_counter()
    connect = {registry.kind(kind) for kind in registry.nodes if spec['graphs/connect/' + kind]}
    fraction = spec['graphs/connect_fraction']
    shared, kinds, keys, maps = {}, [], [], []
    for graph in graphs:
        table = np.empty(len(graph.kinds), np.int64)
        for local, (kind, key) in enumerate(zip(graph.kinds.tolist(), graph.keys)):
            name = registry.nodes[kind]
            if kind in connect and sampled(name, key, fraction):
                merged = shared.get((kind, key))
                if merged is None:
                    merged = shared[(kind, key)] = len(kinds)
                    kinds.append(kind)
                    keys.append(key)
            else:
                merged = len(kinds)
                kinds.append(kind)
                keys.append(key)
            table[local] = merged
        maps.append(table)
    step_offsets = np.r_[0, np.cumsum([graph.steps for graph in graphs])]
    period_offsets = np.r_[0, np.cumsum([int(graph.period.max(initial=-1)) + 1 for graph in graphs])]
    pieces = {name: [] for name in ARRAYS}
    for graph, table, step_offset, period_offset in zip(graphs, maps, step_offsets, period_offsets):
        remap = lambda values: np.where(values >= 0, table[np.maximum(values, 0)], -1).astype(np.int32)
        pieces['kinds'].append(np.zeros(0, np.int8))
        pieces['period'].append((graph.period + period_offset).astype(np.int32))
        pieces['obs_entity'].append(remap(graph.obs_entity))
        pieces['obs_step'].append((graph.obs_step + step_offset).astype(np.int32))
        pieces['obs_x'].append(graph.obs_x)
        pieces['obs_anchor'].append(graph.obs_anchor)
        pieces['fact_source'].append(remap(graph.fact_source))
        pieces['fact_target'].append(remap(graph.fact_target))
        pieces['fact_code'].append(graph.fact_code)
        pieces['fact_step'].append((graph.fact_step + step_offset).astype(np.int32))
        pieces['fact_present'].append(graph.fact_present)
        pieces['fact_values'].append(graph.fact_values)
        pieces['query_key'].append(graph.query_key)
        pieces['query_category'].append(graph.query_category)
        for role in ('actor', 'subject', 'context'):
            pieces['query_' + role].append(remap(graph.arrays['query_' + role]))
        pieces['query_x'].append(graph.query_x)
        pieces['query_signals'].append(graph.query_signals)
        pieces['label_entity'].append(remap(graph.label_entity))
        pieces['label_step'].append((graph.label_step + step_offset).astype(np.int32))
        pieces['label_row'].append(graph.label_row)
        pieces['label_id'].append(graph.label_id)
    arrays = {name: np.concatenate(values) for name, values in pieces.items() if name != 'kinds'}
    arrays['kinds'] = np.asarray(kinds, np.int8)
    result = Graph('joined', keys, int(step_offsets[-1]), **arrays)
    verify(result, registry)
    LOG.info('join exit seconds=%.3f graphs=%d entities=%d shared=%d steps=%d facts=%d', time.perf_counter() - started, len(graphs), len(kinds),
             len(shared), result.steps, len(result.fact_step))
    return result


def layout(graphs, level):
    offsets = np.r_[0, np.cumsum([steps for steps, _, _ in graphs])]
    if level == 'node':
        rows = np.concatenate([row_index for _, row_index, _ in graphs])
        steps = np.concatenate([(row_steps + offset).astype(np.int32) for (_, _, row_steps), offset in zip(graphs, offsets)])
        return rows, steps
    rows = np.concatenate([row_index for _, row_index, _ in graphs])
    return rows, np.arange(int(offsets[-1]), dtype=np.int32)
