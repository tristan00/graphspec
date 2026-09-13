import numpy as np
import pytest

from graphspec import Registry, operators, spec as specs, transform

from . import graphs
from .graphs import categories, chain, gapped, layered, line, links, mesh, periodic, random_graph, relation, tiny, vector, weighted
from .support import adjacency, built, configured, connected, derived, equal, observations, orphans, per, randomized, rows, same, truncate, walked


def test_space():
    registry, _ = random_graph(3, 'node')
    parameters = specs.space(registry, specs.GROUPS)
    names = {parameter.name for parameter in parameters}
    defaults = specs.default(parameters)
    assert not {'nodes/place/min_degree', 'edges/hop:2/drop', 'edges/visits/cap_order', 'time/past_periods', 'time/unit', 'derive/project_cap'} & set(defaults)
    assert specs.canonical(parameters, defaults) == defaults
    for broken in (dict(defaults, **{'nodes/place/filter': 'degree'}), dict(defaults, **{'nodes/place/min_degree': 3}), dict(defaults, **{'time/window': 99}),
                   dict(defaults, **{'coupling/scheme': 'lags', 'coupling/max_lag': 3, 'coupling/lag_window': 256})):
        try:
            specs.canonical(parameters, broken)
        except ValueError:
            continue
        raise AssertionError('canonical accepted an invalid spec')
    assert specs.digest(dict(reversed(list(defaults.items())))) == specs.digest(defaults)
    assert specs.materialized(dict(defaults, **{'neighbors/decay': 0.5, 'time/history': 2})) == specs.materialized(defaults)
    assert 'nodes/person/keep' not in names and 'nodes/person/filter' not in names
    assert {'nodes/place/keep', 'nodes/place/filter', 'relations/near/keep', 'edges/visits/cap', 'edges/near/persistence', 'edges/hop:3/drop',
            'edges/diffusion/cap', 'edges/rewire/persistence', 'edges/complete:place:item/drop', 'derive/complete_within/place'} <= names
    placeless = specs.draw(parameters, lambda parameter: False if parameter.name == 'nodes/place/keep' else parameter.default)
    assert 'derive/project:visits:sources:person:person' not in placeless and 'derive/project:owns:sources:person:person' in placeless
    assert not {'nodes/place/filter', 'relations/visits/keep', 'edges/visits/drop'} & set(placeless) and 'edges/owns/drop' in placeless

    def drawn(values, space=parameters):
        return specs.draw(space, lambda parameter: values.get(parameter.name, parameter.default))

    by_period = drawn({'time/periods': 'every', 'time/period_length': 2, 'time/unit': 'period'})
    assert 'time/window' not in by_period and 'time/past_periods' in by_period and 'time/unit' not in drawn({'time/periods': 'every', 'time/scheme': 'state'})
    query_parameters = specs.space(random_graph(3, 'query')[0], ())
    assert 'time/unit' in specs.default(query_parameters) and 'time/unit' not in drawn({'time/periods': 'every', 'time/period_length': 1}, query_parameters)
    projection = 'project:visits:sources:person:person'
    for name, values in (('edges/hop:2/drop', {'derive/hops': 2}), ('edges/hop:3/drop', {'derive/hops': 3}), ('edges/diffusion/drop', {'derive/diffusion': 'ppr'}),
                         ('edges/rewire/drop', {'derive/rewiring': 'fosr'}), ('edges/%s/drop' % projection, {'derive/' + projection: True}),
                         ('edges/complete:place:item/drop', {'derive/complete/place': True, 'derive/complete/item': True}),
                         ('edges/visits/cap_order', {'edges/visits/cap': 2}), ('derive/project_cap', {'derive/' + projection: True}),
                         ('derive/complete_within/person', {'derive/complete/place': True}), ('derive/complete_within/item', {'derive/complete/person': True})):
        assert name in drawn(values) and name not in defaults, name
    assert 'edges/hop:3/drop' not in drawn({'derive/hops': 2}) and 'edges/complete:place:item/drop' not in drawn({'derive/complete/place': True})
    assert 'derive/complete_within/item' not in drawn({'derive/complete/place': True})
    assert 'derive/complete_within/item' not in drawn({'derive/complete/person': True, 'relations/owns/keep': False})
    assert 'derive/complete_within/unit' in drawn({'derive/complete/actor': True}, query_parameters)
    assert 'derive/complete_within/unit' not in drawn({'derive/complete/actor': True, 'nodes/unit/keep': False}, query_parameters)
    assert 'derive/complete_within/unit' in drawn({'derive/complete/region': True}, query_parameters)
    assert 'derive/complete_within/unit' not in drawn({'derive/complete/region': True, 'relations/located/keep': False}, query_parameters)
    assert 'derive/complete_within/place' in drawn({'derive/complete/place': True})
    assert 'edges/complete:person:place/drop' in drawn({'derive/complete/person': True, 'derive/complete/place': True, 'derive/complete_within/place': True,
                                                        'derive/complete_within/item': True})
    assert 'time/window' not in drawn({'time/scheme': 'state'}) and 'time/past_periods' not in drawn({'time/periods': 'every', 'time/period_length': 2})
    multi = graphs.registry(('a', 'b', 'c', 'd'), [dict(name='r', pairs=(('a', 'b'), ('c', 'd')), attributes=())])
    assert 'project:r:sources:a:a' in multi.by_name and 'project:r:sources:a:c' not in multi.by_name
    assert Registry.from_declaration(multi.declaration()).fingerprint() == multi.fingerprint()
    multi_parameters = specs.space(multi, ())
    assert 'derive/complete_within/c' in drawn({'derive/complete/d': True}, multi_parameters)
    assert 'derive/complete_within/a' not in drawn({'derive/complete/d': True}, multi_parameters)
    assert not {'relations/r/keep', 'derive/hops', 'relations/direction'} & set(drawn({'nodes/b/keep': False, 'nodes/c/keep': False}, multi_parameters))
    within = drawn({'derive/complete/person': True, 'derive/complete/item': True, 'derive/complete_within/place': True})
    assert 'edges/complete:person:person/drop' in within and not {'edges/complete:item:item/drop', 'edges/complete:person:item/drop'} & set(within)
    unlinked = drawn(dict(per('relations', registry.extras['base'], keep=False), **{'derive/complete/place': True, 'derive/hops': 3, 'derive/diffusion': 'ppr',
                                                                                     'derive/rewiring': 'fosr'}))
    assert not {'derive/hops', 'derive/diffusion', 'derive/diffusion_top_k', 'derive/rewiring', 'derive/rewiring_budget', 'derive/complete_within/place',
                'edges/hop:2/drop', 'edges/diffusion/drop', 'edges/rewire/drop', 'relations/direction', 'relations/types', 'relations/parallel'} & set(unlinked)
    assert 'derive/' + projection in unlinked and 'edges/complete:place:place/drop' in unlinked
    stepped = specs.draw(specs.space(registry, ('coupling',)), lambda parameter: {'coupling/scheme': 'lags', 'coupling/unit': 'step'}.get(parameter.name, parameter.default))
    assert 'coupling/lag_window' not in stepped and 'coupling/max_lag' in stepped
    walks = {parameter.name for parameter in specs.space(registry, ('walks',))}
    assert {'neighbors/kinds', 'neighbors/removals', 'walks/fanout'} <= walks and 'neighbors/order' not in walks
    options = {parameter.name: parameter.options for parameter in parameters}
    assert options['edges/visits/cap_order'] == ('recent', 'random', 'largest:weight') and options['edges/owns/cap_order'] == ('recent', 'random')
    assert options['edges/hop:2/cap_order'] == ('recent', 'random') and options['derive/project_cap'] == (0, 2, 4, 8, 16, 32, 64)
    line_registry, _ = line()
    line_space = {parameter.name: parameter for parameter in specs.space(line_registry, ())}
    assert 'nodes/untyped' not in line_space and line_space['time/period_length'].low == 2
    query_registry, _ = random_graph(3, 'query')
    assert {'nodes/actor/keep', 'nodes/unit/filter', 'nodes/region/keep', 'nodes/site/filter'} <= {parameter.name for parameter in specs.space(query_registry, ())}
    queried = specs.draw(specs.space(query_registry, ('neighbors',)), lambda parameter: 'query' if parameter.name == 'neighbors/kinds' else parameter.default)
    assert 'neighbors/removals' not in queried
    rng = np.random.default_rng(1)
    for _ in range(50):
        drawn = randomized(registry, rng)
        assert specs.canonical(parameters, drawn) == drawn
        for kind in ('place', 'item'):
            mode = drawn.get('nodes/%s/filter' % kind)
            assert ('nodes/%s/min_degree' % kind in drawn) == (mode in ('degree', 'total_degree')) and ('nodes/%s/core' % kind in drawn) == (mode == 'core')


def test_identity():
    for level in ('node', 'query'):
        registry, graph = random_graph(3, level)
        assert same(derived(graph, registry, {}), graph)


def test_determinism():
    rng = np.random.default_rng(17)
    for level in ('node', 'query'):
        registry, graph = random_graph(5, level)
        for _ in range(4):
            spec = randomized(registry, rng)
            first = transform.materialize(graph, registry, spec, specs.GROUPS, categories(registry))
            second = transform.materialize(graph, registry, spec, specs.GROUPS, categories(registry))
            assert first.keys() == second.keys() and all(equal(first[key], second[key]) for key in first)
            result = transform.derive(graph, registry, spec)
            assert equal(result.label_row, graph.label_row) and equal(result.query_key, graph.query_key)
            assert orphans(result) <= orphans(graph), sorted(orphans(result) - orphans(graph))[:3]


def test_seeds():
    for level in ('node', 'query'):
        registry, graph = random_graph(9, level)
        kinds = [kind for kind in registry.nodes if kind not in registry.extras['protected']]
        relations = registry.extras['base']
        values = dict(per('nodes', kinds, filter='sample', sample=0.6), **per('edges', relations + ['rewire'], drop=0.3, cap=1, cap_order='random'),
                      **{'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 2, 'walks/fanout': 4, 'walks/bias': 0.5})
        spec = configured(registry, values)
        arrays = transform.materialize(graph, registry, spec, specs.GROUPS, categories(registry))
        packed = dict(spec, **{'time/history': 3, 'time/context': 2, 'neighbors/order': 'decay', 'neighbors/decay': 0.5})
        assert all(equal(arrays[key], value) for key, value in transform.materialize(graph, registry, packed, specs.GROUPS, categories(registry)).items())
        represented = configured(registry, dict(values, **{'coupling/scheme': 'window', 'mixed/trace': 2, 'neighbors/order': 'uniform', 'nodes/untyped': True}))
        assert same(transform.derive(graph, registry, spec), transform.derive(graph, registry, represented))

        def sampled(extra):
            return set(transform.derive(graph, registry, configured(registry, dict(per('nodes', kinds, filter='sample', sample=0.5), **extra))).obs_entity.tolist())

        assert sampled({}) == sampled(dict(per('edges', relations, cap=8), **{'relations/parallel': 'merge'})) and sampled({}) <= sampled(per('nodes', kinds, sample=0.8))
    registry, graph = random_graph(0, 'query')
    trimmed = graph.replace(**{key: graph.arrays[key][graph.fact_step > 0] for key in operators.FACTS})
    rewired = [[row for row in rows(derived(source, registry, {'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 3}), registry, 'rewire') if row[2] > 0]
               for source in (graph, trimmed)]
    assert rewired[0] and rewired[0] == rewired[1]


def test_protection():
    registry, graph = random_graph(5, 'query')
    split = 0
    for kind in ('actor', 'unit', 'region'):
        members = set(np.flatnonzero(graph.kinds == registry.kind(kind)).tolist())
        needed = {(entity, step) for role in (graph.query_actor, graph.query_subject, graph.query_context)
                  for step, entity in enumerate(role.tolist()) if entity in members}
        result = derived(graph, registry, {'nodes/%s/keep' % kind: False})
        assert observations(result, members) == needed, kind
        assert not np.isin(result.fact_source, list(members)).any() and not np.isin(result.fact_target, list(members)).any()
        assert equal(result.obs_anchor, np.isin(result.obs_entity, list(members))), kind
        entities = np.asarray(sorted(members))
        passing = set(entities[operators.unit(transform.SEED, operators.SAMPLE, entities) < 0.5].tolist())
        sampled = derived(graph, registry, {'nodes/%s/filter' % kind: 'sample', 'nodes/%s/sample' % kind: 0.5})
        touched = set(sampled.fact_source.tolist()) | set(sampled.fact_target.tolist())
        assert touched & members <= passing and observations(sampled, passing) == observations(graph, passing), kind
        assert observations(sampled, members - passing) == {row for row in needed if row[0] not in passing}, kind
        cored = derived(graph, registry, {'nodes/%s/filter' % kind: 'core', 'nodes/%s/core' % kind: 16, 'derive/complete/' + kind: True})
        assert observations(cored, members) == needed and not np.isin(np.r_[cored.fact_source, cored.fact_target], list(members)).any(), kind
        assert equal(cored.obs_anchor, np.isin(cored.obs_entity, list(members))), kind
        split += bool(passing) and bool(members - passing)
    assert split
    registry, graph = tiny()
    corrupted = [('anchor source', graph.replace(obs_anchor=graph.obs_entity == 0), registry),
                 ('anchor target', graph.replace(obs_anchor=graph.obs_entity == 3), registry),
                 ('duplicate observation', graph.replace(**{name: np.insert(graph.arrays[name], 1, graph.arrays[name][0], 0)
                                                            for name in operators.OBSERVATIONS}), registry)]
    swapped_registry, swapped = links([(0, 1, 'link', 1, True, ()), (0, 1, 'link', 1, False, ())], 2, 3)
    corrupted.append(('unordered facts', swapped.replace(**{name: swapped.arrays[name][::-1] for name in operators.FACTS}), swapped_registry))
    for build, drop in ((line, lambda source: source.obs_step != 3), (lambda: links([(0, 1, 'link', 2, False, ())], 2, 3), lambda source: source.obs_entity != 1),
                        (lambda: periodic([0, 0, 1], [(0, 1, 'link', 0, True, ())]), lambda source: (source.obs_entity != 1) | (source.obs_step != 2))):
        schema, source = build()
        keep = drop(source)
        corrupted.append(('missing observation', source.replace(**{name: source.arrays[name][keep] for name in operators.OBSERVATIONS}), schema))
    for label, broken, schema in corrupted:
        try:
            transform.verify(broken, schema)
        except ValueError:
            continue
        raise AssertionError('verify accepted ' + label)
    result = derived(graph, registry, {'nodes/place/keep': False})
    assert not rows(result, registry, 'visits') and rows(result, registry, 'owns') == rows(graph, registry, 'owns')
    for filtered in (per('nodes', ('place', 'item'), filter='sample', sample=0.1), per('nodes', ('place', 'item'), filter='core', core=16)):
        assert observations(derived(graph, registry, filtered), {0, 1, 2}) == observations(graph, {0, 1, 2})
    registry, graph = random_graph(7, 'node')
    fewer = graph.replace(**{key: graph.arrays[key][1:] for key in ('label_entity', 'label_step', 'label_row', 'label_id')})
    spec = configured(registry, {'walks/fanout': 3}, ('walks',))
    whole, part = (transform.materialize(source, registry, spec, ('walks',), []) for source in (graph, fewer))
    index = {key: position for position, key in enumerate(zip(whole['actor'].tolist(), whole['query_step'].tolist()))}
    for position, key in enumerate(zip(part['actor'].tolist(), part['query_step'].tolist())):
        assert equal(whole['walks'][index[key]], part['walks'][position]) and equal(whole['turn_walks'][index[key]], part['turn_walks'][position])
    rng = np.random.default_rng(29)
    for level in ('node', 'query'):
        registry, graph = random_graph(7, level)
        early = graph.label_step < np.median(graph.label_step)
        fewer = graph.replace(**{key: graph.arrays[key][early] for key in ('label_entity', 'label_step', 'label_row', 'label_id')})
        for _ in range(6):
            spec = randomized(registry, rng)
            assert same(transform.derive(graph, registry, spec), transform.derive(fewer, registry, spec), operators.FACTS + operators.OBSERVATIONS + ('period',))


def test_causality():
    rng = np.random.default_rng(23)
    for level in ('node', 'query'):
        registry, graph = random_graph(7, level)
        for _ in range(4):
            spec = randomized(registry, rng)
            last = int(rng.integers(3, graph.steps - 2))
            full = transform.materialize(graph, registry, spec, specs.GROUPS, categories(registry))
            short = transform.materialize(truncate(graph, last), registry, spec, specs.GROUPS, categories(registry))
            observed, events, predictions, spatial, mixed = (len(short[key]) for key in ('step', 'events', 'query_step', 'spatial', 'mixed_x'))
            for key in ('x', 'entity', 'step', 'turn'):
                assert equal(full[key][:observed], short[key]), key
            for key in ('spatial', 'spatial_step', 'spatial_attribute_row'):
                assert equal(full[key][:spatial], short[key]), key
            assert full['spatial_step'][spatial:].min(initial=last + 1) > last
            assert equal(full['events'][:events], short['events']) and full['events'][events:, 2].min(initial=2 * last + 2) > 2 * last + 1
            for key in ('actor', 'subject', 'query_context', 'query_step', 'row_index', 'decision', 'affinity', 'global_affinity', 'signals', 'walks', 'turn_walks'):
                assert equal(full[key][:predictions], short[key]), key
            for key in [key for key in short if key.startswith('attributes_')]:
                assert equal(full[key][:len(short[key])], short[key]), key
            for key, limit in (('coupling', observed), ('coupling_boundaries', observed), ('event_dag', events), ('event_boundaries', events)):
                assert equal(full[key][full[key][:, 1] < limit], short[key]), key
            assert equal(full['event_member'][:events], short['event_member'])
            for entity in range(len(short['entity_kind'])):
                whole = full['neighbor_ids'][full['neighbor_offsets'][entity]:full['neighbor_offsets'][entity + 1]]
                assert equal(whole[whole < events], short['neighbor_ids'][short['neighbor_offsets'][entity]:short['neighbor_offsets'][entity + 1]])
            for key in ('mixed_x', 'mixed_kind', 'mixed_tick', 'mixed_entity'):
                assert equal(full[key][:mixed], short[key]), key
            edges = full['mixed_edges']
            assert equal(edges[(edges[:, 0] < mixed) & (edges[:, 1] < mixed)], short['mixed_edges'])
            periods = len(short['turn_starts'])
            for key in ('turn_membership', 'turn_action_category'):
                assert equal(full[key][:last + 1], short[key]), key
            assert equal(full['turn_starts'][:periods], short['turn_starts']) and equal(full['turn_numbers'][:periods], short['turn_numbers'])
            for key in ('turn_ends', 'turn_observations', 'turn_actor_decision'):
                assert equal(full[key][:periods - 1], short[key][:-1]), key


def test_time():
    registry, graph = tiny()
    owns = rows(derived(graph, registry, {'time/window': 2}), registry, 'owns')
    assert (0, 5, 0) in owns and (0, 5, 1) in owns and (0, 5, 2) not in owns
    assert derived(graph, registry, {'time/periods': 'every', 'time/period_length': 2}).period.tolist() == [0, 0, 1, 1]
    assert len(derived(graph, registry, {'time/scheme': 'state'}).fact_step) == len(graph.fact_step) + 3
    registry, graph = weighted()
    copied = derived(graph, registry, {'time/window': 3})
    assert copied.fact_step.tolist() == [0, 0, 1, 1, 2, 2] and copied.fact_values[:, 0].tolist() == [1.0, 2.0] * 3
    merged = derived(graph, registry, {'time/window': 3, 'relations/parallel': 'merge'})
    assert merged.fact_step.tolist() == [0, 1, 2] and merged.fact_values[:, 0].tolist() == [2.0] * 3
    registry, sparse = line()
    carried = derived(sparse, registry, {'time/window': 2})
    assert sorted(zip(carried.obs_entity.tolist(), carried.obs_step.tolist())) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    facts = [(0, 1, 'link', 0, True, ()), (0, 1, 'link', 1, True, ()), (0, 1, 'link', 2, False, ()), (0, 1, 'link', 4, True, ()), (1, 2, 'link', 0, True, ())]
    registry, graph = links(facts, 3, 6)
    for values, expected in (({'time/window': 3}, [(0, 1, 0), (0, 1, 1), (0, 1, 4), (0, 1, 5), (1, 2, 0), (1, 2, 1), (1, 2, 2)]),
                             ({'time/scheme': 'state'}, [(0, 1, 0), (0, 1, 1), (0, 1, 4), (0, 1, 5)] + [(1, 2, step) for step in range(6)])):
        result = derived(graph, registry, values)
        assert rows(result, registry, 'link') == sorted(expected) and rows(result, registry, 'link', False) == [(0, 1, 2)], values
    registry, graph = links([(0, 1, 'link', 0, True, ()), (1, 2, 'link', 1, True, ())], 3, 6)
    by_period = {'time/periods': 'every', 'time/period_length': 2, 'time/unit': 'period'}
    assert rows(derived(graph, registry, dict(by_period, **{'time/past_periods': 0})), registry, 'link') == [(0, 1, 0), (0, 1, 1), (1, 2, 1)]
    assert rows(derived(graph, registry, dict(by_period, **{'time/past_periods': 1})), registry, 'link') == [(0, 1, s) for s in range(4)] + [(1, 2, s) for s in (1, 2, 3)]
    registry, graph = periodic([0, 0, 2, 2, 3, 3], [(0, 1, 'link', 0, True, ())])
    for past, last in ((0, 1), (1, 3), (2, 5)):
        assert rows(derived(graph, registry, {'time/unit': 'period', 'time/past_periods': past}), registry, 'link') == [(0, 1, s) for s in range(last + 1)], past
    registry, graph = links([(0, 1, 'link', 1, True, ()), (0, 1, 'link', 1, False, ())], 2, 4)
    removed = derived(graph, registry, {'time/window': 3})
    assert rows(removed, registry, 'link') == [(0, 1, 1)] and rows(removed, registry, 'link', False) == [(0, 1, 1)]


def test_nodes():
    registry, graph = tiny()
    result = derived(graph, registry, {'nodes/item/keep': False})
    assert not np.isin(result.obs_entity, [5, 6]).any() and not rows(result, registry, 'owns')
    others = ('place', 'item')
    result = derived(graph, registry, per('nodes', others, filter='total_degree', min_degree=2))
    assert rows(result, registry, 'visits') == sorted([(0, 3, s) for s in range(4)] + [(1, 3, s) for s in range(4)] + [(2, 4, 2), (2, 4, 3)])
    assert rows(result, registry, 'owns') == [(1, 6, 3)] and len(observations(result, {0, 1, 2})) == 12
    result = derived(graph, registry, per('nodes', others, filter='degree', min_degree=2))
    assert rows(result, registry, 'visits') == sorted([(0, 3, s) for s in range(4)] + [(1, 3, s) for s in range(4)]) and not rows(result, registry, 'owns')
    result = derived(graph, registry, per('nodes', others, filter='activity', min_activity=2))
    assert rows(result, registry, 'visits') == sorted([(0, 3, s) for s in (1, 2, 3)] + [(1, 3, s) for s in (1, 2, 3)] + [(2, 4, s) for s in (1, 2, 3)])
    result = derived(graph, registry, per('nodes', ('place',), filter='degree', min_degree=2))
    assert rows(result, registry, 'visits') == sorted([(0, 3, s) for s in range(4)] + [(1, 3, s) for s in range(4)])
    assert rows(result, registry, 'owns') == rows(graph, registry, 'owns')
    registry, graph = chain()
    assert rows(derived(graph, registry, per('nodes', ('place',), filter='degree', min_degree=2)), registry, 'near') == [(2, 3, 0)]
    cored = derived(graph, registry, per('nodes', ('place',), filter='core', core=2))
    assert not rows(cored, registry, 'near') and rows(cored, registry, 'visits') == [(0, 2, 0), (1, 2, 0)]
    registry, graph = random_graph(11, 'node')
    result = derived(graph, registry, per('nodes', ('place', 'item'), filter='sample', sample=0.5))
    persons = set(np.flatnonzero(graph.kinds == registry.kind('person')).tolist())
    for entity in range(len(graph.kinds)):
        total, kept = observations(graph, {entity}), observations(result, {entity})
        assert kept == total if entity in persons else kept in (total, set())
    assert 0 < len(result.obs_step) < len(graph.obs_step)
    registry, graph = layered()
    for values, kept in (({'nodes/hub/filter': 'core', 'nodes/hub/core': 2, 'nodes/leaf/filter': 'core', 'nodes/leaf/core': 16}, {2}),
                         ({'nodes/hub/filter': 'core', 'nodes/hub/core': 3, 'nodes/leaf/filter': 'degree', 'nodes/leaf/min_degree': 2}, {2, 4, 6}),
                         ({'nodes/hub/keep': False, 'nodes/leaf/filter': 'degree', 'nodes/leaf/min_degree': 2}, {6}),
                         ({'nodes/leaf/filter': 'degree', 'nodes/leaf/min_degree': 1}, {2, 3, 4, 5, 6, 7}),
                         ({'nodes/leaf/filter': 'core', 'nodes/leaf/core': 2}, {2, 3, 4, 6})):
        assert set(derived(graph, registry, values).obs_entity.tolist()) == {0, 1} | kept, values
    registry, graph = mesh(('anchor', 'thing'), [relation('link', ('thing',), ('thing',))], [0, 1, 1], 3,
                           [(1, 2, 'link', 0, True, ()), (1, 2, 'link', 1, True, ()), (1, 2, 'link', 2, False, ())])
    late = derived(graph, registry, {'nodes/thing/filter': 'activity', 'nodes/thing/min_activity': 3})
    assert observations(late, {1, 2}) == {(1, 2), (2, 2)} and not rows(late, registry, 'link') and not rows(late, registry, 'link', False)


def test_derived_relations():
    registry, graph = tiny()
    name = 'project:visits:sources:person:person'
    assert rows(derived(graph, registry, {'derive/' + name: True}), registry, name) == [(0, 1, step) for step in range(4)]
    assert rows(derived(graph, registry, {'derive/' + name: True, 'derive/project_cap': 2}), registry, name) == [(0, 1, step) for step in range(4)]
    cap_registry, cap_graph = links([(0, 3, 'link', 0, True, ()), (1, 3, 'link', 0, True, ()), (2, 3, 'link', 0, True, ())], 4, 1)
    projection = 'project:link:sources:thing:thing'
    assert rows(derived(cap_graph, cap_registry, {'derive/' + projection: True}), cap_registry, projection) == [(0, 1, 0), (0, 2, 0), (1, 2, 0)]
    assert rows(derived(cap_graph, cap_registry, {'derive/' + projection: True, 'derive/project_cap': 2}), cap_registry, projection) == [(1, 2, 0)]
    stale = [(0, 3, 'link', 0, True, ()), (1, 3, 'link', 1, True, ()), (2, 3, 'link', 1, True, ()), (1, 3, 'link', 1, True, ())]
    cap_registry, cap_graph = links(stale, 4, 2)
    fresh = derived(cap_graph, cap_registry, {'time/window': 2, 'derive/' + projection: True, 'derive/project_cap': 2})
    assert rows(fresh, cap_registry, projection) == [(1, 2, 1)]
    scaffold = derived(graph, registry, {'derive/' + name: True, 'relations/visits/keep': False})
    assert rows(scaffold, registry, name) == [(0, 1, step) for step in range(4)] and not rows(scaffold, registry, 'visits')
    complete = 'complete:person:person'
    assert rows(derived(graph, registry, {'derive/complete/person': True}), registry, complete) == sorted((a, b, s) for s in range(4) for a, b in ((0, 1), (0, 2), (1, 2)))
    assert rows(derived(graph, registry, {'derive/complete/person': True, 'derive/complete_within/place': True}), registry, complete) == [(0, 1, step) for step in range(4)]
    capped = derived(graph, registry, {'derive/complete/person': True, 'edges/%s/cap' % complete: 1})
    assert rows(capped, registry, complete) == [(0, 1, step) for step in range(4)] and rows(capped, registry, 'visits') == rows(graph, registry, 'visits')
    assert not rows(derived(graph, registry, dict(per('nodes', ('place',), filter='degree', min_degree=3), **{'derive/hops': 2})), registry, 'hop:2')
    values = {'derive/' + name: True, 'derive/complete/person': True, 'derive/hops': 3, 'derive/diffusion': 'heat', 'derive/diffusion_top_k': 16,
              'derive/rewiring': 'fosr', 'derive/rewiring_budget': 2}
    plain = derived(graph, registry, values)
    produced = [name, complete, 'hop:2', 'hop:3', 'diffusion', 'rewire']
    for target in produced:
        stated = rows(plain, registry, target)
        steady = sorted({(a, b, 3) for a, b, _ in stated if all((a, b, s) in stated for s in range(4))})
        result = derived(graph, registry, dict(values, **{'edges/%s/persistence' % target: 4}))
        assert stated and rows(result, registry, target) == steady != stated, target
        assert all(rows(result, registry, other) == rows(plain, registry, other) for other in produced + ['visits', 'owns'] if other != target), target
    layered_registry, layered_graph = layered()
    completions = [edge.name for edge in layered_registry.edges if edge.origin[0] == 'complete']
    for anchors, expected in (((), [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]), (('hub',), [(4, 5)]), (('anchor',), [(4, 6)]),
                              (('hub', 'anchor'), [(4, 5), (4, 6)])):
        settings = dict({'derive/complete/leaf': True}, **{'derive/complete_within/' + kind: True for kind in anchors})
        completed = derived(layered_graph, layered_registry, settings)
        assert [edge for edge in completions if rows(completed, layered_registry, edge)] == ['complete:leaf:leaf'], anchors
        assert rows(completed, layered_registry, 'complete:leaf:leaf') == [pair + (0,) for pair in expected], anchors
    parallel_registry, parallel_graph = links([(1, 2, 'link', 0, True, ()), (0, 2, 'link', 0, True, ()), (0, 2, 'link', 0, True, ())], 3, 1)
    capped_pairs = rows(derived(parallel_graph, parallel_registry, {'derive/' + projection: True, 'derive/project_cap': 2}), parallel_registry, projection)
    assert capped_pairs == [(0, 1, 0)], capped_pairs
    kinds_registry, kinds_graph = mesh(('a', 'b'), [relation('link', ('a',), ('b',))], [0, 1, 1, 0], 1, [(0, 1, 'link', 0, True, ())])
    both = derived(kinds_graph, kinds_registry, {'derive/complete/a': True, 'derive/complete/b': True})
    assert rows(both, kinds_registry, 'complete:a:a') == [(0, 3, 0)] and rows(both, kinds_registry, 'complete:b:b') == [(1, 2, 0)]
    assert rows(both, kinds_registry, 'complete:a:b') == [(0, 1, 0), (0, 2, 0), (3, 1, 0), (3, 2, 0)]
    clique = [(a, b, 'link', 0, True, ()) for a in range(4) for b in range(a + 1, 4)] + [(0, 1, 'link', 1, False, ())]
    clique_registry, clique_graph = links(clique, 4, 2)
    thinned = derived(clique_graph, clique_registry, {'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 1, 'derive/sdrf_threshold': 0.1})
    assert rows(thinned, clique_registry, 'link') == [(a, b, 0) for a in range(4) for b in range(a + 1, 4) if (a, b) != (0, 1)]
    assert not rows(thinned, clique_registry, 'link', False)
    split_registry, split_graph = links([(a, 2, 'link', step, True, ()) for step in (0, 2) for a in (0, 1)], 3, 4)
    assert rows(derived(split_graph, split_registry, {'derive/' + projection: True}), split_registry, projection, False) == [(0, 1, 1), (0, 1, 3)]
    result = derived(graph, registry, {'derive/hops': 3})
    assert rows(result, registry, 'hop:2') == sorted([(0, 1, 0), (3, 5, 0), (0, 1, 1), (0, 1, 2), (3, 6, 2), (0, 1, 3), (3, 6, 3)])
    assert rows(result, registry, 'hop:3') == [(0, 6, 2), (0, 6, 3), (1, 5, 0)]
    result = derived(graph, registry, {'derive/hops': 3, 'relations/owns/keep': False})
    assert rows(result, registry, 'hop:2') == [(0, 1, step) for step in range(4)] and not rows(result, registry, 'hop:3')
    found = rows(derived(graph, registry, {'derive/diffusion': 'ppr', 'derive/diffusion_top_k': 1}), registry, 'diffusion')
    assert found and all(tuple(sorted((a, b))) not in adjacency(graph, registry, s) and a != b for a, b, s in found)
    assert max(sum(1 for a, _, s in found if (a, s) == key) for key in {(a, s) for a, _, s in found}) == 1
    heat = rows(derived(graph, registry, {'derive/diffusion': 'heat', 'derive/diffusion_top_k': 16}), registry, 'diffusion')
    within = {0: [(0, 1), (1, 5), (3, 5)], 1: [(0, 1)], 2: [(0, 1), (0, 6), (3, 6)], 3: [(0, 1), (0, 6), (3, 6)]}
    assert heat == sorted((x, y, s) for s, pairs in within.items() for a, b in pairs for x, y in ((a, b), (b, a)))
    for seed in (3, 4):
        query_registry, query_graph = random_graph(seed, 'query')
        for method in ('ppr', 'heat'):
            found = rows(derived(query_graph, query_registry, {'derive/diffusion': method, 'derive/diffusion_top_k': 16}), query_registry, 'diffusion')
            assert found and all(connected(adjacency(query_graph, query_registry, s), a, b) for a, b, s in found)
    found = rows(derived(graph, registry, {'derive/rewiring': 'fosr', 'derive/rewiring_budget': 2}), registry, 'rewire')
    assert [sum(1 for *_, s in found if s == step) for step in range(4)] == [2, 1, 2, 2] and (0, 1, 1) in found
    assert all(a < b and tuple(sorted((a, b))) not in adjacency(graph, registry, s) for a, b, s in found)
    path = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (5, 7)]
    for order in (list(range(9)), list(range(8, -1, -1))):
        path_registry, path_graph = links([(order[a], order[b], 'link', 0, True, ()) for a, b in path], 9, 1)
        added = rows(derived(path_graph, path_registry, {'derive/rewiring': 'fosr', 'derive/rewiring_budget': 1}), path_registry, 'rewire')
        assert added == [tuple(sorted((order[0], order[4]))) + (0,)], added
    gap_registry, gap_graph = gapped()
    rewired = derived(gap_graph, gap_registry, {'time/window': 2, 'derive/rewiring': 'fosr', 'derive/rewiring_budget': 1})
    assert rows(rewired, gap_registry, 'rewire') == [(0, 4, 0), (0, 4, 1), (0, 4, 2)] and not rows(rewired, gap_registry, 'rewire', False)
    first = derived(graph, registry, {'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 3})
    assert same(first, derived(graph, registry, {'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 3}))
    assert all(adjacency(first, registry, s) <= adjacency(graph, registry, s) and len(adjacency(graph, registry, s) - adjacency(first, registry, s)) <= 3 for s in range(4))
    calm = derived(graph, registry, {'derive/rewiring': 'sdrf', 'derive/rewiring_budget': 3, 'derive/sdrf_threshold': 20.0})
    added = rows(calm, registry, 'rewire')
    assert added and all(adjacency(calm, registry, s) == adjacency(graph, registry, s) for s in range(4))
    assert all(sum(1 for *_, s in added if s == step) <= 3 for step in range(4))
    assert all(tuple(sorted((a, b))) not in adjacency(graph, registry, s) for a, b, s in added)
    complete4, cycle4, path4 = ~np.eye(4, dtype=bool), np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    for i in range(4):
        cycle4[i, (i + 1) % 4] = cycle4[(i + 1) % 4, i] = True
    for i in range(3):
        path4[i, i + 1] = path4[i + 1, i] = True
    assert np.isclose(operators.curvature(complete4, 0, 1), 4 / 3) and np.isclose(operators.curvature(cycle4, 0, 1), 1.0)
    assert operators.curvature(path4, 1, 2) == 0 and operators.curvature(path4, 0, 1) == 0


def test_replay():
    from graphspec import kernels

    schema = graphs.registry(('thing',), [relation('link', ('thing',), ('thing',))])
    observations = ([(0, step, vector(0, step)) for step in range(4)]
                    + [(1, 0, vector(7, 7)), (1, 1, vector(7, 7)), (1, 2, vector(9, 9)), (1, 3, vector(9, 9))])
    facts = [(0, 1, 'link', 0, True, ()), (0, 1, 'link', 1, True, ()), (0, 1, 'link', 2, False, ()), (0, 1, 'link', 3, False, ())]
    graph = graphs.assemble(schema, [0, 0], 4, observations, facts, [(0, 3, 0)])
    events = built(graph, schema, {}, ())['events']
    assert [(row[3], row[2], row[0], row[7], row[8]) for row in events.tolist()] == [
        (0, 0, 0, -1, -1), (0, 0, 1, -1, -1), (1, 0, 0, -1, 1), (0, 2, 0, -1, -1), (0, 4, 0, -1, -1), (0, 4, 1, -1, -1), (1, 4, 0, 1, 0), (0, 6, 0, -1, -1)]
    facts = [(0, 1, 'link', 1, True, ()), (0, 1, 'link', 1, False, ())]
    graph = graphs.assemble(schema, [0, 0], 3, [(entity, step, vector(entity, step)) for entity in range(2) for step in range(3)], facts, [(0, 2, 0)])
    events = built(graph, schema, {}, ())['events']
    assert [(row[2], row[7], row[8]) for row in events.tolist() if row[3] == 1] == [(2, -1, 1), (2, 1, 0)]
    empty = np.zeros(0, np.int64)
    rows, status = kernels.replayed(np.zeros(1, np.int64), np.ones(1, np.int64), np.ones(1, bool), empty, empty, empty, empty,
                                    np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64),
                                    empty, empty, np.zeros((0, 3), np.int64), empty, 2, 1)
    assert status < 0 and len(rows) == 0


def test_relations():
    registry, graph = tiny()
    assert not rows(derived(graph, registry, {'relations/owns/keep': False}), registry, 'owns')
    present = int(graph.fact_present.sum())
    for direction, count in (('inverse', 2 * present), ('symmetric', 2 * present), ('directed', present)):
        spatial = built(graph, registry, {'relations/direction': direction}, ())['spatial']
        assert len(spatial) == count
        if direction != 'directed':
            codes = spatial[1::2, 2]
            assert equal(codes, spatial[0::2, 2] if direction == 'symmetric' else registry.flipped[spatial[0::2, 2]])
    assert equal(built(graph, registry, {}, ())['entity_kind'], graph.kinds) and not built(graph, registry, {'nodes/untyped': True}, ())['entity_kind'].any()
    duplicated_registry, duplicated = tiny(duplicate=True)
    assert rows(derived(duplicated, duplicated_registry, {}), duplicated_registry, 'visits').count((0, 3, 0)) == 2
    assert rows(derived(duplicated, duplicated_registry, {'relations/parallel': 'merge'}), duplicated_registry, 'visits').count((0, 3, 0)) == 1
    untyped = derived(graph, registry, {'relations/types': 'untyped'})
    assert not rows(untyped, registry, 'visits') and not rows(untyped, registry, 'owns')
    assert rows(untyped, registry, 'untyped') == sorted(rows(graph, registry, 'visits') + rows(graph, registry, 'owns'))
    assert not np.isfinite(untyped.fact_values).any()
    filtered = derived(graph, registry, {'relations/types': 'untyped', 'edges/owns/drop': 0.9, 'edges/visits/persistence': 4})
    assert rows(filtered, registry, 'untyped') == sorted([(0, 3, 3), (1, 3, 3)] + [row for row in rows(graph, registry, 'owns') if row in rows(derived(graph, registry, {'edges/owns/drop': 0.9}), registry, 'owns')])
    facts = [(0, 1, 'a', 0, True, ()), (0, 1, 'b', 0, True, ()), (0, 1, 'a', 1, True, ()), (0, 1, 'b', 1, True, ()),
             (0, 1, 'a', 2, False, ()), (0, 1, 'b', 2, True, ()), (0, 1, 'b', 3, False, ())]
    registry, graph = links(facts, 2, 5, ('a', 'b'))
    merged = derived(graph, registry, {'relations/types': 'untyped'})
    assert rows(merged, registry, 'untyped', False) == [(0, 1, 3)]
    assert rows(merged, registry, 'untyped') == [(0, 1, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 1, 2)]
    events = built(graph, registry, {'relations/types': 'untyped'}, ())['events']
    assert events[events[:, 3] == 1][:, [2, 8]].tolist() == [[0, 1], [6, 0]]
    registry, graph = links([(0, 1, 'link', 1, True, ()), (0, 1, 'link', 1, False, ())], 2, 3)
    merged = derived(graph, registry, {'relations/parallel': 'merge'})
    assert rows(merged, registry, 'link') == [(0, 1, 1)] and rows(merged, registry, 'link', False) == [(0, 1, 1)]
    registry, graph = tiny()
    projection = 'project:visits:sources:person:person'
    result = derived(graph, registry, {'derive/' + projection: True})
    for step in range(graph.steps):
        block = (result.fact_step == step) & result.fact_present
        assert np.all(np.diff((result.fact_code[block] == registry.code(projection)).astype(np.int8)) >= 0), step


def test_edge_filters():
    run_registry, run_graph = links([(0, 1, 'link', s, True, ()) for s in (0, 1, 3, 4)], 2, 5)
    assert rows(derived(run_graph, run_registry, {'edges/link/persistence': 2}), run_registry, 'link') == [(0, 1, 1), (0, 1, 4)]
    registry, graph = tiny()
    result = derived(graph, registry, per('edges', ('visits', 'owns'), persistence=2))
    assert rows(result, registry, 'visits') == sorted([(0, 3, s) for s in (1, 2, 3)] + [(1, 3, s) for s in (1, 2, 3)] + [(2, 4, 2), (2, 4, 3)])
    assert rows(result, registry, 'owns') == [(1, 6, 3)]
    result = derived(graph, registry, {'edges/visits/persistence': 2})
    assert rows(result, registry, 'owns') == rows(graph, registry, 'owns') and len(rows(result, registry, 'visits')) == 8
    assert rows(derived(graph, registry, {'edges/visits/cap': 1}), registry, 'visits') == sorted([(0, 3, s) for s in range(4)] + [(2, 4, s) for s in (1, 2, 3)])
    largest = derived(graph, registry, {'edges/visits/cap': 1, 'edges/visits/cap_order': 'largest:weight'})
    assert rows(largest, registry, 'visits') == sorted([(1, 3, s) for s in range(4)] + [(2, 4, s) for s in (1, 2, 3)])
    shuffled = rows(derived(graph, registry, {'edges/visits/cap': 1, 'edges/visits/cap_order': 'random'}), registry, 'visits')
    assert sorted({(t, s) for _, t, s in shuffled}) == sorted({(t, s) for _, t, s in rows(graph, registry, 'visits')}) and len(shuffled) == 7
    registry, graph = links([(a, b, 'link', s, True, ()) for s in range(4) for a, b in ((0, 1), (1, 0))], 2, 4)
    result = derived(graph, registry, {'relations/direction': 'symmetric', 'edges/link/persistence': 2})
    assert rows(result, registry, 'link') == sorted((a, b, s) for s in (1, 2, 3) for a, b in ((0, 1), (1, 0)))
    registry, graph = links([(0, 1, 'link', s, True, ()) for s in range(4) for _ in range(2)], 2, 4)
    assert rows(derived(graph, registry, {'edges/link/persistence': 3}), registry, 'link') == [(0, 1, s) for s in (2, 3) for _ in range(2)]
    registry, graph = links([(0, 1, 'link', 0, True, ()), (1, 2, 'link', 0, True, ())], 3, 1)
    assert rows(derived(graph, registry, {'relations/direction': 'symmetric', 'edges/link/cap': 1}), registry, 'link') == [(0, 1, 0)]
    registry, graph = links([(0, 1, 'link', 0, True, ()), (2, 0, 'link', 0, True, ()), (0, 2, 'link', 0, True, ()), (1, 0, 'link', 0, True, ())], 3, 1)
    assert rows(derived(graph, registry, {'relations/direction': 'symmetric', 'edges/link/cap': 1}), registry, 'link') == [(0, 1, 0), (1, 0, 0)]
    registry, graph = links([(0, 1, 'link', 0, True, ()), (0, 2, 'link', 0, True, ()), (0, 3, 'link', 0, True, ())], 4, 1)
    assert rows(derived(graph, registry, {'edges/link/cap': 1}), registry, 'link') == [(0, 1, 0)]
    assert len(rows(derived(graph, registry, {'edges/link/cap': 1, 'relations/direction': 'directed'}), registry, 'link')) == 3
    registry, graph = links([(2, 1, 'link', 0, True, ()), (0, 1, 'link', 0, True, ())], 3, 1)
    assert rows(derived(graph, registry, {'edges/link/cap': 1}), registry, 'link') == [(2, 1, 0)]
    registry, graph = links([(0, 1, 'link', 0, True, ()), (0, 1, 'link', 0, True, ()), (2, 1, 'link', 0, True, ())], 3, 1)
    for direction in ('directed', 'inverse', 'symmetric'):
        assert len(rows(derived(graph, registry, {'edges/link/cap': 2, 'relations/direction': direction}), registry, 'link')) == 3, direction
    registry, graph = links([(0, 0, 'link', 0, True, ()), (0, 1, 'link', 0, True, ())], 2, 1)
    for direction in ('inverse', 'symmetric'):
        assert rows(derived(graph, registry, {'edges/link/cap': 2, 'relations/direction': direction}), registry, 'link') == [(0, 0, 0), (0, 1, 0)], direction
    registry, graph = random_graph(13, 'node')
    result = derived(graph, registry, per('edges', ('visits', 'owns', 'near'), drop=0.5))
    for key in {(a, b, c) for a, b, c in zip(graph.fact_source.tolist(), graph.fact_target.tolist(), graph.fact_code.tolist())}:
        mask = lambda g: (g.fact_source == key[0]) & (g.fact_target == key[1]) & (g.fact_code == key[2])
        assert int(mask(result).sum()) in (0, int(mask(graph).sum()))
    assert 0 < len(result.fact_step) < len(graph.fact_step)
    only = derived(graph, registry, {'edges/near/drop': 0.9})
    assert rows(only, registry, 'visits') == rows(graph, registry, 'visits') and len(rows(only, registry, 'near')) < len(rows(graph, registry, 'near'))
    registry, graph = mesh(('thing',), [relation('link', ('thing',), ('thing',), ('low', 'high'))], [0, 0, 0], 1,
                           [(0, 2, 'link', 0, True, (1.0, 9.0)), (1, 2, 'link', 0, True, (5.0, 2.0))])
    for attribute, expected in (('low', [(1, 2, 0)]), ('high', [(0, 2, 0)])):
        assert rows(derived(graph, registry, {'edges/link/cap': 1, 'edges/link/cap_order': 'largest:' + attribute}), registry, 'link') == expected, attribute
    registry, graph = links([(0, 2, 'link', 0, True, ()), (0, 2, 'link', 1, True, ()), (1, 2, 'link', 1, True, ())], 3, 2)
    assert rows(derived(graph, registry, {'edges/link/cap': 1}), registry, 'link') == [(0, 2, 0), (1, 2, 1)]
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)]
    registry, graph = links([(a, b, name, 0, True, ()) for name in ('a', 'b') for a, b in pairs], 8, 1, ('a', 'b'))
    dropped = derived(graph, registry, {'edges/a/drop': 0.5, 'edges/b/drop': 0.5})
    kept = [[row[:2] for row in rows(dropped, registry, name)] for name in ('a', 'b')]
    assert 0 < len(kept[0]) < len(pairs) and kept[0] != kept[1]


def test_coupling():
    registry, graph = line()
    expectations = [({'coupling/scheme': 'ordinal'}, [(0, 2), (1, 3), (3, 4), (2, 5), (4, 6)]),
                    ({'coupling/scheme': 'ordinal', 'coupling/unit': 'step'}, [(0, 2), (1, 3), (3, 4), (4, 6)]),
                    ({'coupling/scheme': 'lags', 'coupling/max_lag': 2, 'coupling/lag_window': 8}, [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5), (4, 6), (3, 6)]),
                    ({'coupling/scheme': 'lags', 'coupling/max_lag': 2, 'coupling/unit': 'step'}, [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (4, 6), (3, 6)]),
                    ({'coupling/scheme': 'window', 'coupling/window': 2, 'coupling/cap': 8}, [(0, 2), (1, 3), (3, 4), (4, 6)]),
                    ({'coupling/scheme': 'all'}, [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5), (4, 6), (3, 6), (1, 6)]),
                    ({'coupling/scheme': 'window', 'coupling/window': 64, 'coupling/cap': 1}, [(0, 2), (1, 3), (3, 4), (2, 5), (4, 6)]),
                    ({'coupling/scheme': 'window', 'coupling/window': 64, 'coupling/cap': 2},
                     [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5), (4, 6), (3, 6)]),
                    ({'coupling/scheme': 'window', 'coupling/window': 64, 'coupling/cap': 8},
                     [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5), (4, 6), (3, 6), (1, 6)]),
                    ({'coupling/scheme': 'lags', 'coupling/max_lag': 2, 'coupling/lag_window': 1}, [(0, 2), (1, 3), (3, 4), (4, 6)]),
                    ({'coupling/scheme': 'lags', 'coupling/max_lag': 2, 'coupling/lag_window': 2},
                     [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (4, 6), (3, 6)]),
                    ({'coupling/scheme': 'lags', 'coupling/max_lag': 4, 'coupling/lag_window': 512},
                     [(0, 2), (1, 3), (3, 4), (1, 4), (2, 5), (0, 5), (4, 6), (3, 6)])]
    for values, expected in expectations:
        assert built(graph, registry, values, ('coupling',))['coupling'].tolist() == [list(pair) for pair in expected], values
    first, last = registry.code('boundary:first'), registry.code('boundary:last')
    periods = {'time/periods': 'every', 'time/period_length': 2}
    boundaries = [[1, 4, first, 1], [3, 4, last, 1], [0, 5, first, 1], [2, 5, last, 1], [1, 6, first, 1], [3, 6, last, 1]]
    for window, expected in ((64, boundaries), (2, [[3, 4, last, 1]])):
        values = dict(periods, **{'coupling/periods': 1, 'coupling/period_window': window})
        assert built(graph, registry, values, ('coupling',))['coupling_boundaries'].tolist() == expected, window
    values = dict(periods, **{'events/kinds': 'observation', 'events/periods': 1})
    assert built(graph, registry, values, ('events',))['event_boundaries'].tolist() == boundaries
    facts = [(0, 1, 'link', 0, True, ()), (0, 2, 'link', 1, True, ()), (0, 1, 'link', 3, False, ()), (0, 2, 'link', 4, False, ()),
             (0, 1, 'link', 6, True, ()), (0, 1, 'link', 9, False, ()), (0, 2, 'link', 10, True, ())]
    registry, graph = links(facts, 3, 12)
    spanning = {'time/periods': 'every', 'time/period_length': 3, 'events/kinds': 'relation'}
    deep = built(graph, registry, dict(spanning, **{'events/periods': 3, 'events/period_window': 64}), ('events',))['event_boundaries']
    shallow = built(graph, registry, dict(spanning, **{'events/periods': 1, 'events/period_window': 64}), ('events',))['event_boundaries']
    narrow = built(graph, registry, dict(spanning, **{'events/periods': 3, 'events/period_window': 2}), ('events',))['event_boundaries']
    assert deep.tolist() == [[3, 14, 12, 1], [7, 14, 11, 1], [3, 18, 10, 1], [7, 18, 12, 1], [3, 25, 12, 2], [7, 25, 11, 2], [14, 25, 12, 1],
                             [18, 25, 11, 1], [3, 35, 12, 3], [7, 35, 11, 3], [14, 35, 12, 2], [18, 35, 11, 2], [25, 35, 12, 1], [3, 39, 10, 3],
                             [7, 39, 12, 3], [14, 39, 10, 2], [18, 39, 12, 2], [25, 39, 12, 1]]
    assert shallow.tolist() == [[3, 14, 12, 1], [7, 14, 11, 1], [3, 18, 10, 1], [7, 18, 12, 1], [14, 25, 12, 1], [18, 25, 11, 1], [25, 35, 12, 1],
                                [25, 39, 12, 1]]
    assert not len(narrow)
    registry, graph = gapped()
    spans = {'time/periods': 'every', 'time/period_length': 2, 'coupling/periods': 1}
    wide = built(graph, registry, dict(spans, **{'coupling/period_window': 64}), ('coupling',))['coupling_boundaries']
    tight = built(graph, registry, dict(spans, **{'coupling/period_window': 2}), ('coupling',))['coupling_boundaries']
    assert wide.tolist() == [[0, 9, 12, 1], [1, 10, 10, 1], [5, 10, 11, 1], [2, 11, 10, 1], [6, 11, 11, 1], [3, 12, 10, 1], [7, 12, 11, 1],
                             [4, 13, 10, 1], [8, 13, 11, 1]]
    assert tight.tolist() == [[5, 10, 11, 1], [6, 11, 11, 1], [7, 12, 11, 1], [8, 13, 11, 1]]


def test_events_and_neighbours():
    registry, graph = line()
    for values, expected in (({'events/link': 'next'}, [(0, 2), (1, 3), (2, 5), (3, 4), (4, 6)]),
                             ({'events/link': 'lags', 'events/max_lag': 2}, [(0, 2), (0, 5), (1, 3), (1, 4), (2, 5), (3, 4), (3, 6), (4, 6)]),
                             ({'events/link': 'window', 'events/window': 2}, [(0, 2), (1, 3), (3, 4), (4, 6)])):
        dag = built(graph, registry, dict(values, **{'events/kinds': 'observation'}), ('events',))['event_dag']
        assert sorted(map(tuple, dag.tolist())) == expected, values
    registry, graph = random_graph(19, 'query')
    counts = []
    for link in ('next', 'lags', 'window'):
        arrays = built(graph, registry, {'events/kinds': 'relation', 'events/link': link, 'events/window': 1024}, ('events',))
        dag = arrays['event_dag']
        assert len(dag) and (arrays['events'][dag.ravel(), 3] == 1).all() and len(np.unique(dag, axis=0)) == len(dag)
        counts.append(len(dag))
    assert counts[0] <= counts[1] <= counts[2]
    capped = built(graph, registry, {'events/kinds': 'relation', 'events/link': 'window', 'events/window': 1024, 'events/cap': 1}, ('events',))
    assert len(capped['event_dag']) <= counts[2]
    for kinds, removals, allowed in (('relation', False, {(1, 1)}), ('relation', True, {(1, 1), (1, 0)}), ('query', True, {(2, -1)})):
        arrays = built(graph, registry, {'neighbors/kinds': kinds, 'neighbors/removals': removals}, ('neighbors',))
        found = arrays['events'][arrays['neighbor_ids']]
        assert len(found) and {tuple(row) for row in found[:, [3, 8]].tolist()} <= allowed
    facts = [(0, 1, 'link', 0, True, ()), (0, 2, 'link', 0, True, ()), (1, 2, 'link', 0, True, ()),
             (0, 1, 'link', 1, False, ()), (0, 2, 'link', 1, False, ()),
             (0, 1, 'link', 2, True, ()), (0, 2, 'link', 2, True, ()), (1, 2, 'link', 2, True, ()),
             (0, 1, 'link', 5, False, ()), (0, 2, 'link', 5, False, ())]
    registry, graph = links(facts, 3, 7)
    settings = {'events/kinds': 'relation', 'events/window': 1024, 'events/cap': 0}
    whole = built(graph, registry, dict(settings, **{'events/link': 'lags', 'events/max_lag': 8}), ('events',))['event_dag']
    capped = built(graph, registry, dict(settings, **{'events/link': 'lags', 'events/max_lag': 8, 'events/cap': 1}), ('events',))['event_dag']
    wide = built(graph, registry, dict(settings, **{'events/link': 'window'}), ('events',))['event_dag']
    tight = built(graph, registry, dict(settings, **{'events/link': 'window', 'events/window': 3}), ('events',))['event_dag']
    narrow = built(graph, registry, dict(settings, **{'events/link': 'lags', 'events/max_lag': 8, 'events/window': 6}), ('events',))
    ticks = narrow['events'][:, 2]
    assert (len(whole), len(capped), len(wide), len(tight), len(narrow['event_dag'])) == (24, 17, 30, 10, 20)
    assert max(ticks[after] - ticks[before] for before, after in narrow['event_dag'].tolist()) == 6
    assert {tuple(edge) for edge in narrow['event_dag'].tolist()} < {tuple(edge) for edge in whole.tolist()}
    assert {tuple(edge) for edge in capped.tolist()} < {tuple(edge) for edge in whole.tolist()}
    assert {tuple(edge) for edge in tight.tolist()} < {tuple(edge) for edge in wide.tolist()}
    for dag in (whole, capped, wide, tight):
        repeated = np.flatnonzero(dag[1:, 1] == dag[:-1, 1]) + 1
        assert np.all(dag[repeated, 0] > dag[repeated - 1, 0])


def test_walks():
    registry, graph = random_graph(29, 'query')
    for fanout in (1, 3):
        arrays = built(graph, registry, {'walks/fanout': fanout, 'walks/window': 4, 'walks/bias': 0.5}, ('walks',))
        walks, turns = arrays['walks'], arrays['turn_walks']
        assert (walks[..., 0] >= 0).any() and (turns[..., 0] >= 0).any()
        for row in range(len(walks)):
            moment = int(arrays['query_step'][row])
            membership = int(arrays['turn_membership'][moment])
            for side, root in enumerate((arrays['actor'][row], arrays['subject'][row])):
                for sample in range(walks.shape[2]):
                    walked(arrays, walks[row, side, sample], root, max(0, (moment - 3) * 2), 2 * moment + 1, fanout)
                    for back in range(1, 4):
                        prior = membership - back
                        if prior < 0:
                            assert (turns[row, side, back - 1, sample] < 0).all()
                            continue
                        walked(arrays, turns[row, side, back - 1, sample], root, 2 * int(arrays['turn_starts'][prior]), 2 * int(arrays['turn_ends'][prior]) + 2, fanout)
    facts = ([(0, 1 + i, 'link', 6 + i, True, ()) for i in range(4)]
             + [(1, 5, 'link', 1, True, ()), (1, 6, 'link', 3, True, ()), (2, 5, 'link', 0, True, ()), (2, 6, 'link', 2, True, ()),
                (3, 5, 'link', 4, True, ()), (4, 6, 'link', 5, True, ())])
    registry, graph = links(facts, 7, 10)
    settings = {'walks/fanout': 8, 'walks/window': 512}
    spread = built(graph, registry, dict(settings, **{'walks/bias': 0.0}), ('walks',))
    tilted = built(graph, registry, dict(settings, **{'walks/bias': 1.0}), ('walks',))
    assert spread['walks'][0, 0, :8].tolist() == [[71, 39, 7, -1], [79, 47, 23, 7], [79, 47, 31, 15], [63, 23, -1, -1], [79, 47, 23, 7],
                                                  [71, 39, 15, -1], [55, 15, 7, -1], [71, 39, 15, -1]]
    assert tilted['walks'][0, 0, :8].tolist() == [[79, 47, 31, 15], [79, 47, 31, 15], [79, 47, 31, 15], [79, 47, 31, 15], [79, 47, 23, 7],
                                                  [79, 47, 31, 15], [63, 23, -1, -1], [79, 47, 31, 15]]
    means = []
    for arrays in (spread, tilted):
        chosen = arrays['walks'][0, 0, :, 0]
        means.append(arrays['events'][chosen[chosen >= 0], 2].mean())
    assert len(np.unique(spread['walks'][0, 0, :, 0])) == 4 and means[1] > means[0]


def test_mixed():
    registry, graph = random_graph(31, 'query')
    kinds = built(graph, registry, {'mixed/changes': False, 'mixed/trace': 0, 'mixed/bundles': False}, ('mixed',))['mixed_kind']
    assert set(kinds.tolist()) == {0, 4}
    kinds = built(graph, registry, {'mixed/trace': 3}, ('mixed',))['mixed_kind']
    assert {1, 2, 3, 4} <= set(kinds.tolist()) and int((kinds == 2).sum()) == graph.steps
    registry, graph = periodic([0, 0, 1, 1], [(0, 1, 'link', 0, True, ()), (0, 1, 'link', 2, False, ())])
    plain = built(graph, registry, {'mixed/changes': False, 'mixed/trace': 0, 'mixed/bundles': False}, ('mixed',))
    assert plain['mixed_kind'].tolist() == [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4]
    assert plain['mixed_tick'].tolist() == [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]
    assert plain['mixed_entity'].tolist() == [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    assert plain['mixed_edges'].tolist() == [[0, 3, 4], [1, 4, 4], [2, 5, 1], [3, 6, 4], [4, 7, 4], [5, 8, 1], [6, 9, 4], [7, 10, 4], [8, 11, 1]]
    assert plain['mixed_member_offsets'].tolist() == [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8]
    assert plain['mixed_member_ids'].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    full = built(graph, registry, {'mixed/changes': True, 'mixed/trace': 2, 'mixed/bundles': True}, ('mixed',))
    assert full['mixed_kind'].tolist() == [0, 0, 2, 4, 0, 1, 3, 0, 1, 3, 2, 4, 0, 1, 0, 1, 2, 4, 0, 1, 3, 0, 1, 3, 2, 4]
    assert full['mixed_tick'].tolist() == [0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7]
    assert full['mixed_entity'].tolist() == [0, 1, -1, 0, 0, 0, 0, 1, 1, 1, -1, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, -1, 0]
    assert full['mixed_edges'].tolist() == [[0, 2, 512], [1, 2, 512], [2, 3, 8], [0, 4, 4], [0, 5, 32], [5, 4, 32], [0, 6, 512], [4, 6, 512],
                                            [6, 4, 1024], [1, 7, 4], [1, 8, 32], [8, 7, 32], [1, 9, 512], [7, 9, 512], [9, 7, 1024], [4, 10, 512],
                                            [7, 10, 512], [10, 11, 8], [3, 11, 1], [4, 12, 4], [4, 13, 32], [13, 12, 32], [7, 14, 4], [7, 15, 32],
                                            [15, 14, 32], [12, 16, 512], [14, 16, 512], [16, 17, 8], [11, 17, 1], [12, 18, 4], [12, 19, 32],
                                            [19, 18, 32], [12, 20, 512], [18, 20, 512], [20, 18, 1024], [14, 21, 4], [14, 22, 32], [22, 21, 32],
                                            [14, 23, 512], [21, 23, 512], [23, 21, 1024], [18, 24, 512], [21, 24, 512], [24, 25, 8], [17, 25, 1]]
    assert full['mixed_member_offsets'].tolist() == [0, 1, 2, 4, 4, 5, 7, 9, 10, 12, 14, 16, 16, 17, 19, 20, 22, 24, 24, 25, 27, 29, 30, 32, 34, 36, 36]
    assert full['mixed_member_ids'].tolist() == [0, 1, 0, 1, 2, 0, 2, 0, 2, 3, 1, 3, 1, 3, 2, 3, 4, 2, 4, 5, 3, 5, 4, 5, 6, 4, 6, 4, 6, 7, 5, 7, 5, 7, 6, 7]
    result = graphs.registry(('thing',), [])
    observations = [(entity, step, vector(entity, step)) for entity in range(2) for step in (0, 1, 3)]
    empty = graphs.assemble(result, [0, 0], 4, observations, [], [(0, 3, 0), (1, 1, 1)])
    gap = built(empty, result, {'mixed/changes': False, 'mixed/trace': 0, 'mixed/bundles': True}, ('mixed',))
    assert gap['mixed_kind'].tolist() == [0, 0, 2, 0, 0, 2, 2, 0, 0, 2]
    assert gap['mixed_tick'].tolist() == [0, 0, 0, 2, 2, 2, 4, 6, 6, 6]
    assert gap['mixed_entity'].tolist() == [0, 1, -1, 0, 1, -1, -1, 0, 1, -1]
    assert gap['mixed_edges'].tolist() == [[0, 2, 512], [1, 2, 512], [0, 3, 4], [1, 4, 4], [3, 5, 512], [4, 5, 512], [3, 7, 4], [4, 8, 4],
                                           [7, 9, 512], [8, 9, 512]]
    assert gap['mixed_member_offsets'].tolist() == [0, 1, 2, 4, 5, 6, 8, 8, 9, 10, 12]
    assert gap['mixed_member_ids'].tolist() == [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
    assert np.isfinite(gap['mixed_x']).all() and not gap['mixed_x'][6].any()


def test_derivation_stops_at_the_fact_budget(monkeypatch):
    from graph_tuning_infra import compute

    registry, graph = tiny()
    monkeypatch.setattr(compute, 'FACTS', 4)
    with pytest.raises(ValueError, match='exceed 4 facts'):
        derived(graph, registry, {'derive/complete/person': True, 'derive/complete/place': True})
    monkeypatch.setattr(compute, 'FACTS', 1000)
    assert len(derived(graph, registry, {'derive/complete/person': True, 'derive/complete/place': True}).fact_step) > len(graph.fact_step)
