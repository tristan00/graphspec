import itertools
import logging
import math
import time

from graph_tuning_infra.files import sha256_json

from .vocabulary import HOPS


LOG = logging.getLogger(__name__)
GROUPS = ('coupling', 'events', 'neighbors', 'walks', 'mixed')
PACKING = ('time/history', 'time/context', 'neighbors/order', 'neighbors/decay')
FILTERS = ('none', 'degree', 'total_degree', 'core', 'activity', 'sample')
LAGS = (2, 4, 8, 16, 32, 64)


class Parameter:

    __slots__ = ('name', 'kind', 'default', 'low', 'high', 'log', 'options', 'condition')

    def __init__(self, name, kind, default, low=None, high=None, log=False, options=(), condition=None):
        self.name, self.kind, self.default = name, kind, default
        self.low, self.high, self.log, self.options = low, high, log, tuple(options)
        self.condition = condition

    def active(self, spec):
        return self.condition is None or self.condition(spec)

    def check(self, value):
        if self.kind in ('bool', 'choice'):
            if type(value) is not type(self.options[0]) or value not in self.options:
                raise ValueError('%s must be one of %s, found %r' % (self.name, list(self.options), value))
            return value
        if self.kind == 'int':
            if type(value) is not int or not self.low <= value <= self.high:
                raise ValueError('%s must be an integer in %d..%d, found %r' % (self.name, self.low, self.high, value))
            return value
        if type(value) not in (int, float) or not self.low <= value <= self.high:
            raise ValueError('%s must be a number in %g..%g, found %r' % (self.name, self.low, self.high, value))
        return float(value)

    def uniform(self, rng):
        if self.kind in ('bool', 'choice'):
            return self.options[int(rng.integers(len(self.options)))]
        if self.kind == 'int':
            return int(rng.integers(self.low, self.high + 1))
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def declaration(self):
        return dict(name=self.name, kind=self.kind, default=self.default, low=self.low, high=self.high, log=self.log,
                    options=list(self.options))


def flag(name, default, condition=None):
    return Parameter(name, 'bool', default, options=(False, True), condition=condition)


def number(name, low, high, default, condition=None):
    return Parameter(name, 'int', default, low, high, condition=condition)


def real(name, low, high, default, condition=None, log=False):
    return Parameter(name, 'float', default, low, high, log, condition=condition)


def choice(name, options, default, condition=None):
    return Parameter(name, 'choice', default, options=options, condition=condition)


def subsets(kinds):
    return tuple('+'.join(members) for size in range(1, len(kinds) + 1) for members in itertools.combinations(kinds, size))


def equals(name, value):
    return lambda spec: spec.get(name) == value


def among(name, values):
    return lambda spec: spec.get(name) in values


def differs(name, value):
    return lambda spec: spec.get(name, value) != value


def counterparts(relation, position, kind):
    return {pair[1 - position] for pair in relation.pairs if pair[position] == kind}


def space(registry, groups):
    started = time.perf_counter()
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError('unknown representation groups: ' + ', '.join(unknown))
    nodes, extras = registry.nodes, registry.extras
    protected = set(extras['protected'])
    base = [registry.by_name[name] for name in extras['base']]

    def kept(spec, kind):
        return kind in protected or spec['nodes/%s/keep' % kind]

    def periodic(spec):
        return (extras['queries'] and spec['time/periods'] == 'truth') or spec.get('time/period_length', 1) > 1

    def projectable(spec, edge):
        relation, position, first, second = registry.by_name[edge.origin[1]], ('sources', 'targets').index(edge.origin[2]), edge.origin[3], edge.origin[4]
        shared = counterparts(relation, position, first) & counterparts(relation, position, second)
        return kept(spec, first) and kept(spec, second) and any(kept(spec, kind) for kind in shared)

    def filtering(kind, modes):
        return lambda spec: spec.get('nodes/%s/filter' % kind) in modes

    def linked(spec):
        return any(spec.get('relations/%s/keep' % edge.name) for edge in base)

    def joins(spec, first, second):
        return any(spec.get('relations/%s/keep' % edge.name) and ((first, second) in edge.pairs or (second, first) in edge.pairs) for edge in base)

    def anchored(spec, first, second):
        anchors = [kind for kind in nodes if spec.get('derive/complete_within/' + kind)]
        return not anchors or any(joins(spec, kind, first) and joins(spec, kind, second) for kind in anchors)

    emits = dict(project=lambda spec, edge: spec.get('derive/' + edge.name),
                 complete=lambda spec, edge: (spec.get('derive/complete/' + edge.origin[1]) and spec.get('derive/complete/' + edge.origin[2])
                                              and anchored(spec, edge.origin[1], edge.origin[2])),
                 hop=lambda spec, edge: spec.get('derive/hops', 1) >= int(edge.origin[1]),
                 diffusion=lambda spec, edge: spec.get('derive/diffusion', 'none') != 'none',
                 rewire=lambda spec, edge: spec.get('derive/rewiring', 'none') != 'none')

    def filters(name, condition, attributes):
        prefix = 'edges/%s/' % name
        orders = ('recent', 'random') + tuple('largest:' + attribute for attribute in attributes)
        return [real(prefix + 'drop', 0.0, 0.9, 0.0, condition), number(prefix + 'persistence', 1, 4, 1, condition),
                number(prefix + 'cap', 0, 64, 0, condition), choice(prefix + 'cap_order', orders, 'recent', lambda spec: condition(spec) and spec[prefix + 'cap'] > 0)]

    projected = [edge for edge in registry.edges if edge.origin[0] == 'project']
    unprotected = [kind for kind in nodes if kind not in protected]
    parameters = [number('time/history', 1, 32, 16), choice('time/periods', ('truth', 'every'), 'truth'),
                  number('time/period_length', 1 if extras['queries'] else 2, 16, 4, equals('time/periods', 'every')),
                  choice('time/scheme', ('window', 'state'), 'window'),
                  choice('time/unit', ('step', 'period'), 'step', lambda spec: spec['time/scheme'] == 'window' and periodic(spec)),
                  number('time/window', 1, 8, 1, lambda spec: spec['time/scheme'] == 'window' and spec.get('time/unit', 'step') == 'step'),
                  number('time/past_periods', 0, 3, 0, equals('time/unit', 'period')), number('time/context', 0, 3, 0)]
    parameters += [flag('nodes/%s/keep' % kind, True) for kind in unprotected]
    parameters += [flag('nodes/untyped', False)] if len(nodes) > 1 else []
    for kind in unprotected:
        prefix = 'nodes/%s/' % kind
        parameters += [choice(prefix + 'filter', FILTERS, 'none', lambda spec, kind=kind: spec['nodes/%s/keep' % kind]),
                       number(prefix + 'min_degree', 1, 16, 1, filtering(kind, ('degree', 'total_degree'))),
                       number(prefix + 'core', 2, 16, 2, filtering(kind, ('core',))),
                       number(prefix + 'min_activity', 2, 16, 2, filtering(kind, ('activity',))),
                       real(prefix + 'sample', 0.1, 0.9, 0.5, filtering(kind, ('sample',)))]
    parameters += [flag('relations/%s/keep' % edge.name, True, lambda spec, edge=edge: any(kept(spec, source) and kept(spec, target) for source, target in edge.pairs))
                   for edge in base]
    parameters += [flag('derive/' + edge.name, False, lambda spec, edge=edge: projectable(spec, edge)) for edge in projected]
    parameters.append(choice('derive/project_cap', (0, 2, 4, 8, 16, 32, 64), 0, lambda spec: any(spec.get('derive/' + edge.name) for edge in projected)))
    parameters += [flag('derive/complete/' + kind, False, lambda spec, kind=kind: kept(spec, kind)) for kind in nodes]
    parameters += [flag('derive/complete_within/' + kind, False, lambda spec, kind=kind: kept(spec, kind)
                        and any(spec.get('derive/complete/' + other) and joins(spec, kind, other) for other in nodes)) for kind in nodes]
    parameters += [number('derive/hops', 1, HOPS[-1], 1, linked), choice('derive/diffusion', ('none', 'ppr', 'heat'), 'none', linked),
                   number('derive/diffusion_top_k', 1, 16, 4, differs('derive/diffusion', 'none')),
                   real('derive/ppr_alpha', 0.05, 0.5, 0.15, equals('derive/diffusion', 'ppr')),
                   real('derive/heat_time', 0.5, 5.0, 2.0, equals('derive/diffusion', 'heat')),
                   choice('derive/rewiring', ('none', 'sdrf', 'fosr'), 'none', linked),
                   number('derive/rewiring_budget', 1, 64, 8, differs('derive/rewiring', 'none')),
                   real('derive/sdrf_temperature', 1.0, 500.0, 115.0, equals('derive/rewiring', 'sdrf'), log=True),
                   real('derive/sdrf_threshold', 0.1, 20.0, 1.0, equals('derive/rewiring', 'sdrf'), log=True)]
    parameters.append(choice('relations/direction', ('inverse', 'symmetric', 'directed'), 'inverse', linked))
    for edge in base:
        parameters += filters(edge.name, lambda spec, edge=edge: bool(spec.get('relations/%s/keep' % edge.name)), edge.attributes)
    for edge in registry.edges:
        if edge.origin[0] in emits:
            parameters += filters(edge.name, lambda spec, edge=edge: bool(emits[edge.origin[0]](spec, edge)), ())
    parameters += [choice('relations/types', ('typed', 'untyped'), 'typed', linked), choice('relations/parallel', ('keep', 'merge'), 'keep', linked)]
    queries = ('query',) if extras['queries'] else ()
    if 'coupling' in groups:
        parameters += [choice('coupling/scheme', ('ordinal', 'lags', 'window', 'all'), 'ordinal'),
                       choice('coupling/unit', ('observation', 'step'), 'observation', among('coupling/scheme', ('ordinal', 'lags'))),
                       choice('coupling/max_lag', LAGS, 64, equals('coupling/scheme', 'lags')),
                       number('coupling/lag_window', 1, 512, 256, lambda spec: spec['coupling/scheme'] == 'lags' and spec['coupling/unit'] == 'observation'),
                       number('coupling/window', 2, 128, 64, equals('coupling/scheme', 'window')),
                       number('coupling/cap', 1, 128, 64, equals('coupling/scheme', 'window')),
                       number('coupling/periods', 0, 3, 0),
                       number('coupling/period_window', 2, 128, 64, lambda spec: spec['coupling/periods'] > 0)]
    if 'events' in groups:
        kinds = subsets(('observation', 'relation') + queries)
        parameters += [choice('events/kinds', kinds, kinds[-1]), choice('events/link', ('next', 'lags', 'window'), 'lags'),
                       choice('events/max_lag', LAGS[:4], 8, equals('events/link', 'lags')), number('events/window', 1, 1024, 512),
                       number('events/cap', 0, 64, 0), number('events/periods', 0, 3, 0),
                       number('events/period_window', 2, 128, 64, lambda spec: spec['events/periods'] > 0)]
    if 'neighbors' in groups or 'walks' in groups:
        kinds = subsets(('relation',) + queries)
        parameters += [choice('neighbors/kinds', kinds, kinds[-1]),
                       flag('neighbors/removals', True, lambda spec: 'relation' in spec['neighbors/kinds'].split('+'))]
    if 'neighbors' in groups:
        parameters += [choice('neighbors/order', ('recent', 'uniform', 'decay'), 'recent'),
                       real('neighbors/decay', 0.001, 1.0, 0.1, equals('neighbors/order', 'decay'), log=True)]
    if 'walks' in groups:
        parameters += [real('walks/bias', 0.0, 1.0, 0.0), number('walks/window', 1, 512, 256), number('walks/fanout', 1, 128, 64)]
    if 'mixed' in groups:
        parameters += [flag('mixed/changes', True), number('mixed/trace', 0, 8, 4), flag('mixed/bundles', True)]
    LOG.debug('space exit seconds=%.4f parameters=%d groups=%s', time.perf_counter() - started, len(parameters), '+'.join(groups))
    return tuple(parameters)


def draw(parameters, choose):
    spec = {}
    for parameter in parameters:
        if parameter.active(spec):
            spec[parameter.name] = choose(parameter)
    return spec


def default(parameters):
    return draw(parameters, lambda parameter: parameter.default)


def canonical(parameters, spec):
    started = time.perf_counter()

    def read(parameter):
        if parameter.name not in spec:
            raise ValueError('spec lacks the active parameter ' + parameter.name)
        return parameter.check(spec[parameter.name])

    result = draw(parameters, read)
    extra = sorted(set(spec) - set(result))
    if extra:
        raise ValueError('spec sets inactive or unknown parameters: ' + ', '.join(extra))
    LOG.debug('canonical exit seconds=%.4f parameters=%d', time.perf_counter() - started, len(result))
    return result


def materialized(spec):
    return {name: value for name, value in spec.items() if name not in PACKING}


def digest(spec):
    return sha256_json(spec)


def summary(parameters, spec):
    defaults = {parameter.name: parameter.default for parameter in parameters}
    return {name: value for name, value in spec.items() if value != defaults[name]}
