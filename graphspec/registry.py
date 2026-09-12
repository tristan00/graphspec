import numpy as np

from graph_tuning_infra.files import sha256_json

from .declarations import EdgeType


class Registry:

    def __init__(self, name, nodes, edges, width, markers, extras):
        self.name, self.width = name, width
        self.nodes, self.edges = tuple(nodes), tuple(edges)
        self.features = 2 * width
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.relations = self._allocate()
        self.by_name = {edge.name: edge for edge in self.edges}
        if len(self.by_name) != len(self.edges):
            raise ValueError('duplicate edge type name')
        self.by_code = {}
        for edge in self.edges:
            self.by_code[edge.code] = edge
            if not edge.symmetric:
                self.by_code[edge.reverse] = edge
            for endpoint in edge.sources + edge.targets:
                if endpoint not in self.node_index:
                    raise ValueError('edge %s names an undeclared node type %s' % (edge.name, endpoint))
        self.canonical = np.zeros(self.relations, bool)
        for edge in self.edges:
            self.canonical[edge.code] = True
        self.flipped = np.arange(self.relations, dtype=np.int64)
        for edge in self.edges:
            if not edge.symmetric:
                self.flipped[edge.code], self.flipped[edge.reverse] = edge.reverse, edge.code
        self.attributed = tuple(edge.name for edge in self.edges if edge.attributes)
        self.markers = tuple(markers)
        self.marker_index = {marker: self.relations + i for i, marker in enumerate(self.markers)}
        self.relation_slots = self.relations + len(self.markers)
        self.extras = dict(extras)

    @classmethod
    def from_declaration(cls, declared):
        edges = [EdgeType(e['name'], e['origin'], e['pairs'], e['attributes'], symmetric=e['symmetric'])
                 for e in declared['edges']]
        return cls(declared['name'], declared['nodes'], edges, declared['width'], declared['markers'], declared['extras'])

    def _allocate(self):
        code = 0
        for edge in self.edges:
            edge.code = code
            code += 1
        for edge in self.edges:
            if not edge.symmetric:
                edge.reverse = code
                code += 1
        return code

    def code(self, name):
        return self.by_name[name].code

    def attributes(self, name):
        return self.by_name[name].attributes

    def kind(self, name):
        return self.node_index[name]

    def marker(self, name):
        return self.marker_index[name]

    def declaration(self):
        return dict(name=self.name, width=self.width, features=self.features, nodes=list(self.nodes), relations=self.relations,
                    markers=list(self.markers), edges=[edge.declaration() for edge in self.edges], extras=self.extras)

    def fingerprint(self):
        return sha256_json(self.declaration())
