import numpy as np

from graphspec import operators, spec as specs, transform
from tuning import model_spec

from . import graphs
from .graphs import WIDTH, categories, chain, gapped, layered, line, links, mesh, periodic, random_graph, relation, tiny, vector, weighted


def configured(registry, values, groups=specs.GROUPS):
    parameters = specs.space(registry, groups)
    return specs.canonical(parameters, specs.draw(parameters, lambda parameter: values.get(parameter.name, parameter.default)))


def config(registry, model_values, graph_values, seed=5):
    return model_spec.configuration(registry, model_values, graph_values, seed)


def derived(graph, registry, values):
    return transform.derive(graph, registry, configured(registry, values))


def built(graph, registry, values, groups):
    return transform.materialize(graph, registry, configured(registry, values, groups), groups, categories(registry))


def randomized(registry, rng, groups=specs.GROUPS):
    return specs.draw(specs.space(registry, groups), lambda parameter: parameter.uniform(rng))


def per(prefix, names, **settings):
    return {'%s/%s/%s' % (prefix, name, key): value for name in names for key, value in settings.items()}


def rows(graph, registry, name, present=True):
    use = (graph.fact_code == registry.code(name)) & (graph.fact_present == present)
    return sorted(zip(graph.fact_source[use].tolist(), graph.fact_target[use].tolist(), graph.fact_step[use].tolist()))


def equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and np.array_equal(left, right, equal_nan=left.dtype.kind == 'f')


def same(left, right, keys=None):
    return left.steps == right.steps and all(equal(left.arrays[key], right.arrays[key]) for key in (keys or left.arrays))


def adjacency(graph, registry, step):
    use = graph.fact_present & (graph.fact_step == step) & np.isin(graph.fact_code, [registry.code(name) for name in registry.extras['base']])
    return {tuple(sorted(pair)) for pair in zip(graph.fact_source[use].tolist(), graph.fact_target[use].tolist())}


def observations(graph, members):
    return {(entity, step) for entity, step in zip(graph.obs_entity.tolist(), graph.obs_step.tolist()) if entity in members}


def orphans(graph):
    stated, found = set(), set()
    for step, absent, source, target, code in sorted(zip(graph.fact_step.tolist(), (~graph.fact_present).tolist(), graph.fact_source.tolist(),
                                                         graph.fact_target.tolist(), graph.fact_code.tolist())):
        if not absent:
            stated.add((source, target, code))
        elif (source, target, code) not in stated:
            found.add((source, target, step))
    return found


def connected(pairs, first, second):
    reached, frontier = {first}, [first]
    while frontier:
        node = frontier.pop()
        for left, right in pairs:
            for here, there in ((left, right), (right, left)):
                if here == node and there not in reached:
                    reached.add(there)
                    frontier.append(there)
    return second in reached


def truncate(graph, last):
    observed, stated, labelled = graph.obs_step <= last, graph.fact_step <= last, graph.label_step <= last
    queries = slice(0, last + 1 if len(graph.query_key) else 0)
    changes = {key: graph.arrays[key][observed] for key in operators.OBSERVATIONS}
    changes.update({key: graph.arrays[key][stated] for key in ('fact_source', 'fact_target', 'fact_code', 'fact_step', 'fact_present', 'fact_values')})
    changes.update({key: graph.arrays[key][labelled] for key in ('label_entity', 'label_step', 'label_row', 'label_id')})
    changes.update({key: graph.arrays[key][queries] for key in ('query_key', 'query_category', 'query_actor', 'query_subject', 'query_context', 'query_x', 'query_signals')})
    return graph.replace(steps=last + 1, period=graph.period[:last + 1], **changes)


def walked(arrays, walk, root, floor, cutoff, fanout):
    events, current = arrays['events'], int(root)
    for hop, event in enumerate(walk.tolist()):
        if event < 0:
            assert (walk[hop:] < 0).all()
            return
        ids = arrays['neighbor_ids'][arrays['neighbor_offsets'][current]:arrays['neighbor_offsets'][current + 1]]
        lo, hi = np.searchsorted(events[ids, 2], (floor, cutoff))
        assert event in ids[max(lo, hi - fanout):hi]
        current = int(events[event, 1] if events[event, 0] == current else events[event, 0])
        cutoff = int(events[event, 2])
