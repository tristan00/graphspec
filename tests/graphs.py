import numpy as np

from graphspec import vocabulary
from graphspec.truth import Graph, verify


WIDTH = 8
CATEGORIES = ('build', 'move', 'trade')


class Schema:

    def __init__(self, nodes, relations, queries, level, signals, labels):
        self.nodes, self.relations, self.queries, self.signals = tuple(nodes), tuple(relations), queries, signals
        self.features, self.level, self.labels = 2 * WIDTH, level, tuple(labels)

    def target(self, name):
        return self.level

    def destinations(self, table):
        return self.labels


def relation(name, sources, targets, attributes=()):
    return dict(name=name, pairs=tuple((source, target) for source in sources for target in targets), attributes=tuple(attributes))


def registry(nodes, relations, queries=False, level='node', signals=0):
    schema = Schema(nodes, relations, queries, level, signals, nodes[:1])
    return vocabulary.registry(schema, dict(name='check', database='check.lbug', target='label'))


def vector(*values):
    head = np.zeros(WIDTH, np.float32)
    head[:len(values)] = values
    return np.concatenate((head, np.ones(WIDTH, np.float32)))


def categories(registry):
    return list(CATEGORIES) if registry.extras['queries'] else []


def assemble(registry, kinds, steps, observations, facts, labels, queries=None):
    width = max([len(registry.attributes(name)) for name in registry.extras['base']] + [0])
    observations = sorted(observations, key=lambda row: (row[1], row[0]))
    ordered = [fact for _, fact in sorted(enumerate(facts), key=lambda item: (item[1][3], not item[1][4], item[0]))]
    values = np.full((len(ordered), width), np.nan, np.float32)
    for row, fact in enumerate(ordered):
        values[row, :len(fact[5])] = fact[5]
    arrays = dict(kinds=np.asarray(kinds, np.int8), obs_entity=np.asarray([row[0] for row in observations], np.int32),
                  obs_step=np.asarray([row[1] for row in observations], np.int32),
                  obs_x=np.asarray([row[2] for row in observations], np.float16).reshape(-1, registry.features), obs_anchor=np.zeros(len(observations), bool),
                  fact_source=np.asarray([fact[0] for fact in ordered], np.int32), fact_target=np.asarray([fact[1] for fact in ordered], np.int32),
                  fact_code=np.asarray([registry.code(fact[2]) for fact in ordered], np.int64), fact_step=np.asarray([fact[3] for fact in ordered], np.int32),
                  fact_present=np.asarray([fact[4] for fact in ordered], bool), fact_values=values)
    if queries is None:
        labels = sorted(labels, key=lambda row: (row[1], row[0]))
        arrays.update(period=np.arange(steps, dtype=np.int32), query_key=np.zeros(0, np.int64), query_category=np.zeros(0, np.int64),
                      query_actor=np.zeros(0, np.int32), query_subject=np.zeros(0, np.int32), query_context=np.zeros(0, np.int32),
                      query_x=np.zeros((0, registry.features), np.float16), query_signals=np.zeros((0, 0, 2), np.float32),
                      label_entity=np.asarray([row[0] for row in labels], np.int32), label_step=np.asarray([row[1] for row in labels], np.int32),
                      label_row=np.asarray([row[2] for row in labels], np.int64), label_id=np.asarray([row[2] for row in labels], np.int64))
    else:
        labels = sorted(labels)
        arrays.update(period=np.asarray([query[0] for query in queries], np.int32), query_key=np.asarray([query[1] for query in queries], np.int64),
                      query_category=np.asarray([query[2] for query in queries], np.int64), query_actor=np.asarray([query[3] for query in queries], np.int32),
                      query_subject=np.asarray([query[4] for query in queries], np.int32), query_context=np.asarray([query[5] for query in queries], np.int32),
                      query_x=np.asarray([query[6] for query in queries], np.float16), query_signals=np.asarray([query[7] for query in queries], np.float32),
                      label_entity=np.full(len(labels), -1, np.int32), label_step=np.asarray([row[0] for row in labels], np.int32),
                      label_row=np.asarray([row[1] for row in labels], np.int64), label_id=np.asarray([queries[row[0]][1] for row in labels], np.int64))
    graph = Graph('check', ['entity_%d' % i for i in range(len(kinds))], steps, **arrays)
    verify(graph, registry)
    return graph


def tiny(duplicate=False):
    result = registry(('person', 'place', 'item'), [relation('visits', ('person',), ('place',), ('weight',)), relation('owns', ('person',), ('item',))])
    kinds = [0, 0, 0, 1, 1, 2, 2]
    observations = [(entity, step, vector(entity, step % 2)) for entity in range(7) for step in range(4)]
    facts = ([(0, 3, 'visits', step, True, (1.0,)) for step in range(4)] + [(1, 3, 'visits', step, True, (5.0,)) for step in range(4)]
             + [(2, 4, 'visits', step, True, (2.0,)) for step in (1, 2, 3)] + [(0, 5, 'owns', 0, True, ())]
             + [(1, 6, 'owns', step, True, ()) for step in (2, 3)])
    if duplicate:
        facts.append((0, 3, 'visits', 0, True, (9.0,)))
    return result, assemble(result, kinds, 4, observations, facts, [(person, 3, person) for person in range(3)])


def line():
    result = registry(('thing',), [])
    observations = [(0, step, vector(0, step)) for step in (0, 1, 3)] + [(1, step, vector(1, step)) for step in range(4)]
    return result, assemble(result, [0, 0], 4, observations, [], [(0, 3, 0), (1, 2, 1)])


def mesh(nodes, relations, kinds, steps, facts):
    result = registry(nodes, relations)
    observations = [(entity, step, vector(entity, step)) for entity in range(len(kinds)) for step in range(steps)]
    return result, assemble(result, kinds, steps, observations, facts, [(0, steps - 1, 0)])


def links(facts, entities, steps, names=('link',)):
    return mesh(('thing',), [relation(name, ('thing',), ('thing',)) for name in names], [0] * entities, steps, facts)


def layered():
    relations = [relation('grip', ('anchor',), ('hub',)), relation('tie', ('hub',), ('leaf',)), relation('touch', ('anchor',), ('leaf',)),
                 relation('loop', ('leaf',), ('leaf',))]
    facts = [(0, 2, 'grip', 0, True, ()), (1, 2, 'grip', 0, True, ()), (0, 3, 'grip', 0, True, ()), (2, 4, 'tie', 0, True, ()),
             (3, 4, 'tie', 0, True, ()), (3, 5, 'tie', 0, True, ()), (0, 4, 'touch', 0, True, ()), (0, 6, 'touch', 0, True, ()),
             (1, 6, 'touch', 0, True, ()), (7, 7, 'loop', 0, True, ())]
    return mesh(('anchor', 'hub', 'leaf'), relations, [0, 0, 1, 1, 2, 2, 2, 2], 1, facts)


def periodic(periods, facts, entities=2):
    result = registry(('thing',), [relation('link', ('thing',), ('thing',))], True, 'query')
    steps = len(periods)
    observations = [(entity, step, vector(entity, step)) for entity in range(entities) for step in range(steps)]
    queries = [(period, 100 + step, 0, 0, 1, 0, vector(step), np.zeros((0, 2), np.float32)) for step, period in enumerate(periods)]
    return result, assemble(result, [0] * entities, steps, observations, facts, [(0, 0)], queries)


def chain():
    result = registry(('person', 'place'), [relation('visits', ('person',), ('place',)), relation('near', ('place',), ('place',))])
    observations = [(entity, 0, vector(entity)) for entity in range(5)]
    facts = [(0, 2, 'visits', 0, True, ()), (1, 2, 'visits', 0, True, ()), (2, 3, 'near', 0, True, ()), (3, 4, 'near', 0, True, ())]
    return result, assemble(result, [0, 0, 1, 1, 1], 1, observations, facts, [(0, 0, 0)])


def weighted():
    result = registry(('thing',), [relation('link', ('thing',), ('thing',), ('weight',))])
    observations = [(entity, step, vector(entity, step)) for entity in range(2) for step in range(3)]
    facts = [(0, 1, 'link', 0, True, (1.0,)), (0, 1, 'link', 0, True, (2.0,))]
    return result, assemble(result, [0, 0], 3, observations, facts, [(0, 2, 0)])


def gapped():
    result = registry(('thing',), [relation('link', ('thing',), ('thing',))])
    observations = [(entity, step, vector(entity, step)) for entity in range(5) for step in range(3) if (entity, step) != (0, 1)]
    facts = [(a, a + 1, 'link', step, True, ()) for step in range(3) for a in range(1, 4)] + [(0, 1, 'link', step, True, ()) for step in (0, 2)]
    return result, assemble(result, [0] * 5, 3, observations, facts, [(1, 2, 0)])


def random_graph(seed, level, steps=12):
    rng = np.random.default_rng(seed)
    if level == 'node':
        relations = [relation('visits', ('person',), ('place',), ('weight',)), relation('owns', ('person',), ('item',)), relation('near', ('place',), ('place',))]
        result = registry(('person', 'place', 'item'), relations)
        counts = (6, 4, 5)
        pairs = [('visits', 0, 1), ('owns', 0, 2), ('near', 1, 1)]
    else:
        relations = [relation('controls', ('actor',), ('unit', 'region'), ('strength',)), relation('located', ('unit',), ('region',)),
                     relation('allied', ('actor',), ('actor',)), relation('hosts', ('region',), ('site',))]
        result = registry(('actor', 'unit', 'region', 'site'), relations, True, 'query', signals=2)
        counts = (3, 6, 4, 3)
        pairs = [('controls', 0, 1), ('controls', 0, 2), ('located', 1, 2), ('allied', 0, 0), ('hosts', 2, 3)]
    kinds = [kind for kind, count in enumerate(counts) for _ in range(count)]
    members = [[entity for entity, value in enumerate(kinds) if value == kind] for kind in range(len(counts))]
    first = {entity: 0 if level == 'query' and kinds[entity] < 3 else int(rng.integers(0, 3)) for entity in range(len(kinds))}
    state = {entity: vector(rng.normal(), rng.normal()) for entity in range(len(kinds))}
    observations, facts, active = [], [], {}
    for step in range(steps):
        present = [entity for entity in range(len(kinds)) if first[entity] <= step]
        for entity in present:
            if rng.random() < 0.4:
                state[entity] = vector(rng.normal(), rng.normal())
            observations.append((entity, step, state[entity]))
        for name, source_kind, target_kind in pairs:
            sources = [entity for entity in members[source_kind] if entity in present]
            targets = [entity for entity in members[target_kind] if entity in present]
            if sources and targets and rng.random() < 0.7:
                source, target = int(rng.choice(sources)), int(rng.choice(targets))
                if source != target:
                    active[(source, target, name)] = True
        for (source, target, name), live in list(active.items()):
            if not live:
                continue
            width = len(result.attributes(name))
            if rng.random() < 0.15:
                facts.append((source, target, name, step, False, ()))
                active[(source, target, name)] = False
            else:
                facts.append((source, target, name, step, True, tuple(rng.random(width).tolist())))
    if level == 'node':
        labels = [(person, step, row) for row, (person, step) in enumerate((p, s) for s in (5, steps - 1) for p in members[0] if first[p] <= s)]
        return result, assemble(result, kinds, steps, observations, facts, labels)
    queries = [(step // 3, 100 + step, int(rng.integers(3)), members[0][step % 3], int(rng.choice(members[1])), int(rng.choice(members[2])),
                vector(rng.normal()), rng.normal(size=(2, 2)).astype(np.float32)) for step in range(steps)]
    labels = [(step, row) for row, step in enumerate(range(0, steps, 2))]
    return result, assemble(result, kinds, steps, observations, facts, labels, queries)
