import logging
import time

from graphspec import spec as specs
from graphspec.spec import GROUPS, among, choice, draw, equals, flag, number, real


LOG = logging.getLogger(__name__)
VIEWS = ('snapshot', 'mixed', 'event_graph', 'memory', 'projection', 'sequence', 'tree', 'walks', 'affinity', 'signal')
SPATIAL = ('none', 'mean', 'normalised', 'relational', 'attention')
TEMPORAL = ('graph_sequence', 'entity_gru', 'entity_tcn', 'entity_attention', 'graph_cell', 'evolving', 'coupling_edges')
MESSAGES = ('gated', 'attention')
UPDATES = ('mailbox_gru', 'mutual_rnn', 'intensity', 'antisymmetric')
OPERATORS = ('mixer', 'transformer', 'ssm')
STRUCTURES = ('none', 'occurrence', 'overlap')
HIDDEN = (16, 24, 32, 48, 64)
HEADS = (1, 2, 4)
NEIGHBOURS = (4, 8, 10, 12, 16, 20, 30, 32, 64)
REPRESENTATIONS = dict(lifted='coupling', dag='events', tree='neighbors', history='neighbors', walks='walks', mixed='mixed')
TRAINING = (number('epochs', 2, 12, 4), real('clip', 0.5, 2.0, 1.0), real('lr', 0.0001, 0.005, 0.001, log=True),
            real('weight_decay', 1e-7, 1e-2, 1e-5, log=True))


def views(registry):
    extras = registry.extras
    excluded = set()
    if not extras['signals']:
        excluded.add('signal')
    if extras['level'] == 'node' or not extras['queries']:
        excluded.add('affinity')
    return tuple(view for view in VIEWS if view not in excluded)


def view(*names):
    return among('view', names)


def attends(spec):
    return spec.get('snapshot/sequence') == 'attention' or spec.get('snapshot/temporal') == 'entity_attention'


def memory_state(spec):
    return spec.get('view') == 'memory' and spec['memory/update'] != 'mutual_rnn'


def reads_dropout(spec):
    if spec['view'] == 'snapshot':
        return spec['snapshot/spatial'] != 'none' or attends(spec) or spec['snapshot/temporal'] == 'coupling_edges'
    return spec['view'] in ('mixed', 'event_graph') or spec.get('sequence/operator') == 'transformer'


def space(registry):
    started = time.perf_counter()
    offered = views(registry)
    updates = UPDATES if registry.extras['queries'] else tuple(update for update in UPDATES if update != 'mutual_rnn')
    parameters = [choice('view', offered, offered[0]), choice('hidden', HIDDEN, 32),
                  choice('snapshot/spatial', SPATIAL, 'relational', view('snapshot')),
                  number('snapshot/layers', 1, 3, 2, lambda spec: spec.get('view') == 'snapshot' and spec['snapshot/spatial'] != 'none'),
                  choice('snapshot/temporal', TEMPORAL, 'entity_gru', view('snapshot')),
                  choice('snapshot/sequence', ('gru', 'lstm', 'attention'), 'gru', equals('snapshot/temporal', 'graph_sequence')),
                  number('snapshot/temporal_layers', 1, 4, 2, among('snapshot/temporal', ('entity_tcn', 'graph_cell', 'evolving', 'coupling_edges'))),
                  number('snapshot/diffusion_hops', 0, 4, 0, equals('snapshot/temporal', 'graph_cell')),
                  choice('snapshot/heads', HEADS, 2, attends),
                  choice('snapshot/message', MESSAGES, 'gated', equals('snapshot/temporal', 'coupling_edges')),
                  choice('mixed/aggregation', ('mean', 'attention'), 'mean', view('mixed')),
                  choice('mixed/update', ('gru', 'residual'), 'gru', view('mixed')),
                  number('mixed/layers', 1, 3, 2, view('mixed')),
                  choice('event_graph/message', MESSAGES, 'gated', view('event_graph')),
                  number('event_graph/layers', 1, 3, 2, view('event_graph')),
                  choice('memory/update', updates, 'mailbox_gru', view('memory')),
                  choice('memory/embedding', ('state', 'attention'), 'attention', memory_state),
                  flag('memory/pair_terms', False, memory_state),
                  choice('memory/neighbors', NEIGHBOURS, 8, lambda spec: spec.get('memory/embedding') == 'attention' or spec.get('memory/pair_terms') is True),
                  number('memory/integration_steps', 1, 16, 2, equals('memory/update', 'antisymmetric')),
                  real('memory/damping', 0.001, 1.0, 0.1, equals('memory/update', 'antisymmetric'), log=True),
                  number('projection/hops', 1, 4, 2, view('projection')),
                  choice('projection/width', (16, 32, 64, 128, 256), 16, view('projection')),
                  real('projection/decay', 1e-7, 1.0, 0.05, view('projection'), log=True),
                  choice('projection/neighbors', NEIGHBOURS, 8, view('projection')),
                  choice('sequence/operator', OPERATORS, 'transformer', view('sequence')),
                  choice('sequence/neighbors', NEIGHBOURS, 8, view('sequence')),
                  number('sequence/layers', 1, 3, 2, view('sequence')),
                  choice('sequence/structure', STRUCTURES, 'none', view('sequence')),
                  choice('sequence/patch', (1, 2, 4, 8, 16), 2, equals('sequence/operator', 'transformer')),
                  choice('sequence/heads', HEADS, 2, equals('sequence/operator', 'transformer')),
                  choice('sequence/state', (4, 8, 16, 32, 64), 4, equals('sequence/operator', 'ssm')),
                  number('tree/depth', 1, 2, 2, view('tree')),
                  choice('tree/neighbors', (4, 8, 12, 16, 20), 8, view('tree')),
                  choice('walks/count', (1, 2, 4, 8, 16, 32), 4, view('walks')),
                  number('walks/hops', 1, 4, 3, view('walks')),
                  choice('affinity/buffer', (1, 2, 4, 8, 16, 32), 4, view('affinity')),
                  number('signal/topk', 1, 20, 3, view('signal')),
                  number('signal/hops', 1, 4, 1, view('signal')),
                  number('signal/layers', 1, 3, 2, view('signal')),
                  real('dropout', 0.0, 0.5, 0.0, reads_dropout)]
    LOG.debug('model space exit seconds=%.4f parameters=%d views=%d', time.perf_counter() - started, len(parameters), len(offered))
    return tuple(parameters)


def blocks(spec):
    kind = spec['view']
    if kind == 'snapshot':
        return frozenset(('snapshot', 'lifted') if spec['snapshot/temporal'] == 'coupling_edges' else ('snapshot',))
    if kind == 'mixed':
        return frozenset(('mixed',))
    if kind == 'event_graph':
        return frozenset(('events', 'dag'))
    if kind == 'memory':
        result = {'events'}
        if spec.get('memory/embedding') == 'attention' or spec.get('memory/pair_terms'):
            result.add('history')
        if spec['memory/update'] in ('intensity', 'antisymmetric'):
            result.add('snapshot')
        return frozenset(result)
    if kind == 'projection':
        return frozenset(('events', 'history'))
    if kind == 'sequence':
        result = {'history', 'history_targets', 'turn_history'}
        if spec['sequence/structure'] == 'overlap':
            result.add('structure')
        return frozenset(result)
    if kind == 'tree':
        return frozenset(('tree', 'turn_history'))
    if kind == 'walks':
        return frozenset(('walks',))
    if kind == 'affinity':
        return frozenset(('events',))
    return frozenset()


def groups(spec):
    wanted = {REPRESENTATIONS[block] for block in blocks(spec) if block in REPRESENTATIONS}
    return tuple(group for group in GROUPS if group in wanted)


def neighbors(spec):
    return spec[spec['view'] + '/neighbors']


def describe(spec):
    parts = [spec['view']]
    for name, value in spec.items():
        if name != 'view':
            parts.append('%s %s' % (name.split('/', 1)[-1], '%.3g' % value if isinstance(value, float) else value))
    return ' '.join(parts)


def configured(registry, values):
    parameters = space(registry)
    unknown = sorted(set(values) - {parameter.name for parameter in parameters})
    if unknown:
        raise ValueError('unknown model parameters: ' + ', '.join(unknown))
    return draw(parameters, lambda parameter: values.get(parameter.name, parameter.default))


def configuration(registry, values, graph_values, seed):
    spec = configured(registry, values)
    parameters = specs.space(registry, groups(spec))
    graph = specs.canonical(parameters, draw(parameters, lambda parameter: graph_values.get(parameter.name, parameter.default)))
    return dict(seed=seed, model=spec, graph=graph, **specs.default(TRAINING))


def samples(registry):
    started = time.perf_counter()
    parameters = space(registry)
    switches = [parameter for parameter in parameters if parameter.kind in ('choice', 'bool') and isinstance(parameter.options[0], (str, bool))
                and parameter.name != 'view']
    result = []

    def add(values):
        spec = configured(registry, values)
        if spec not in result:
            result.append(spec)
        return spec

    for name in views(registry):
        base = add(dict(view=name))
        first = [add({'view': name, parameter.name: option}) for parameter in switches for option in parameter.options]
        for spec in first:
            for parameter in switches:
                if parameter.name in spec:
                    for option in parameter.options:
                        add(dict(spec, **{parameter.name: option}))
    LOG.debug('model samples exit seconds=%.4f samples=%d', time.perf_counter() - started, len(result))
    return result
