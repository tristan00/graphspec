import logging
import time

import pyarrow as pa

from graph_tuning_infra.database import columns, connections, query, tables, width


LOG = logging.getLogger(__name__)
EDGE = 'edge_'
ROLES = ('actor', 'subject', 'context')
FACTS = ('seq', 'step', 'present')
RESERVED = ('graph', 'observation', 'observed', 'query') + tuple('query_' + role for role in ROLES)


def check(registry):
    clash = sorted(kind for kind in registry.nodes if kind in RESERVED or kind.startswith(EDGE))
    if clash:
        raise ValueError('node types collide with reserved table names: ' + ', '.join(clash))


class Schema:

    def __init__(self, connection):
        started = time.perf_counter()
        self.connection = connection
        listed = tables(connection)
        self.kinds = {row['name']: row['type'] for row in listed}
        for name, kind in (('graph', 'NODE'), ('observation', 'NODE'), ('observed', 'REL')):
            if self.kinds.get(name) != kind:
                raise ValueError('the database has no %s table named %s' % (kind.lower(), name))
        decisions = [('query', 'NODE')] + [('query_' + role, 'REL') for role in ROLES]
        found = [name for name, _ in decisions if name in self.kinds]
        self.queries = bool(found)
        if self.queries and any(self.kinds.get(name) != kind for name, kind in decisions):
            raise ValueError('a database with queries needs the query node table and the query_actor, query_subject and query_context relationship tables')
        observed = {source for source, _ in self.pairs('observed')}
        self.nodes = tuple(row['name'] for row in listed if row['name'] in observed)
        relations = []
        for row in listed:
            if row['type'] != 'REL' or not row['name'].startswith(EDGE):
                continue
            pairs, typed = self.pairs(row['name']), self.columns(row['name'])
            missing = [name for name in FACTS if name not in typed]
            if missing:
                raise ValueError('%s lacks the columns %s' % (row['name'], ', '.join(missing)))
            attributes = tuple(name for name in typed if name not in FACTS)
            if any(typed[name] != 'FLOAT' for name in attributes):
                raise ValueError('attributes of %s must be FLOAT columns' % row['name'])
            if not {kind for pair in pairs for kind in pair} <= set(self.nodes):
                raise ValueError('%s connects tables that are not observed node types' % row['name'])
            relations.append(dict(name=row['name'][len(EDGE):], pairs=pairs, attributes=attributes))
        self.relations = tuple(relations)
        self.features = width(self.columns('observation')['x'])
        self.signals = 0
        if self.queries:
            typed = self.columns('query')
            if width(typed['x']) != self.features:
                raise ValueError('query.x and observation.x must have the same width')
            self.signals = width(typed['signal']) if 'signal' in typed else 0
        LOG.info('schema exit seconds=%.3f nodes=%d relations=%d features=%d signals=%d queries=%s', time.perf_counter() - started,
                 len(self.nodes), len(self.relations), self.features, self.signals, self.queries)

    def pairs(self, table):
        return connections(self.connection, table)

    def columns(self, table):
        return columns(self.connection, table)

    def destinations(self, table):
        return tuple(dict.fromkeys(target for _, target in self.pairs(table)))

    def target(self, name):
        if self.kinds.get(name) != 'NODE' or self.kinds.get(name + '_of') != 'REL':
            raise ValueError('the database has no target table %s with a %s_of relationship' % (name, name))
        destinations = set(self.destinations(name + '_of'))
        if destinations in ({'query'}, {'graph'}):
            return destinations.pop()
        if not destinations <= set(self.nodes):
            raise ValueError('%s_of must point to query, to graph, or only to observed node types' % name)
        if self.columns(name).get('step') != 'INT32':
            raise ValueError('%s labels node types, so it needs a step INT32 column naming the step each label is predicted at' % name)
        return 'node'


class Reader:

    def __init__(self, database, connection, schema, target):
        started = time.perf_counter()
        self.database, self.connection, self.schema = database, connection, schema
        self.level = schema.target(target)
        signal = ', q.signal AS signal, q.signal_observed AS signal_observed' if schema.signals else ''
        self.statements = dict(
            observations='MATCH (e)-[:observed]->(o:observation) WHERE e.graph = $graph '
                         'RETURN e.entity AS entity, o.seq AS seq, o.step AS step, o.x AS x')
        if schema.queries:
            self.statements['queries'] = (
                'MATCH (a)<-[:query_actor]-(q:query)-[:query_subject]->(s), (q)-[:query_context]->(c) WHERE q.graph = $graph '
                'RETURN q.step AS step, q.period AS period, q.key AS key, q.category AS category, a.entity AS actor, '
                's.entity AS subject, c.entity AS context, q.x AS query%s ORDER BY step' % signal)
        if self.level == 'query':
            self.statements['targets'] = ('MATCH (t:`%s`)-[:`%s_of`]->(q:query) WHERE q.graph = $graph '
                                          'RETURN q.step AS step, t.id AS id, t.y AS y, t.split AS split' % (target, target))
        elif self.level == 'graph':
            self.statements['targets'] = ('MATCH (t:`%s`)-[:`%s_of`]->(g:graph) WHERE g.id = $graph '
                                          'RETURN t.id AS id, t.y AS y, t.split AS split' % (target, target))
        else:
            self.statements['targets'] = ('MATCH (t:`%s`)-[:`%s_of`]->(n) WHERE n.graph = $graph '
                                          'RETURN t.id AS id, t.y AS y, t.split AS split, t.step AS step, n.entity AS entity' % (target, target))
        for kind in schema.nodes:
            self.statements['node:' + kind] = 'MATCH (n:`%s`) WHERE n.graph = $graph RETURN n.entity AS entity, n.key AS key' % kind
        for relation in schema.relations:
            self.statements['edge:' + relation['name']] = (
                'MATCH (a)-[r:`%s%s`]->(b) WHERE a.graph = $graph RETURN a.entity AS source, b.entity AS target, '
                'r.seq AS seq, r.step AS step, r.present AS present%s' % (
                    EDGE, relation['name'], ''.join(', r.`%s` AS `%s`' % (name, name) for name in relation['attributes'])))
        LOG.info('reader exit seconds=%.3f level=%s relations=%d', time.perf_counter() - started, self.level, len(schema.relations))

    def run(self, name, graph):
        return query(self.connection, self.statements[name], dict(graph=graph))

    def graph(self, index):
        started = time.perf_counter()
        parts = []
        for kind in self.schema.nodes:
            part = self.run('node:' + kind, index)
            parts.append(part.append_column('kind', pa.array([kind] * part.num_rows, pa.string())))
        tables = dict(nodes=pa.concat_tables(parts), observations=self.run('observations', index), targets=self.run('targets', index),
                      edges={relation['name']: self.run('edge:' + relation['name'], index) for relation in self.schema.relations})
        if self.schema.queries:
            tables['queries'] = self.run('queries', index)
        LOG.info('read graph exit graph=%d seconds=%.3f observations=%d', index, time.perf_counter() - started, tables['observations'].num_rows)
        return tables
