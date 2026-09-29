import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed

from .declarations import EdgeType
from .schema import LEVELS


class Registry:
    @timed("registry.registry")
    def __init__(self, name, nodes, edges, width, markers, extras):
        if type(width) is not int or width < 1:
            raise ValueError("registry feature width must be a positive integer")
        self.name, self.width = name, width
        self.nodes, self.edges = tuple(nodes), tuple(edges)
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("registry node types must be nonempty and unique")
        self.features = 2 * width
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.relations = self._allocate()
        self.by_name = {edge.name: edge for edge in self.edges}
        if len(self.by_name) != len(self.edges):
            raise ValueError("duplicate edge type name")
        self.by_code = {}
        for edge in self.edges:
            self.by_code[edge.code] = edge
            if not edge.symmetric:
                self.by_code[edge.reverse] = edge
            for endpoint in edge.sources + edge.targets:
                if endpoint not in self.node_index:
                    raise ValueError("edge %s names an undeclared node type %s" % (edge.name, endpoint))
        self.canonical = np.zeros(self.relations, bool)
        for edge in self.edges:
            self.canonical[edge.code] = True
        self.flipped = np.arange(self.relations, dtype=np.int64)
        for edge in self.edges:
            if not edge.symmetric:
                self.flipped[edge.code], self.flipped[edge.reverse] = edge.reverse, edge.code
        self.attributed = tuple(edge.name for edge in self.edges if edge.attributes)
        self.markers = tuple(markers)
        if len(set(self.markers)) != len(self.markers):
            raise ValueError("duplicate relation marker name")
        self.marker_index = {marker: self.relations + i for i, marker in enumerate(self.markers)}
        self.relation_slots = self.relations + len(self.markers)
        self.extras = dict(extras)
        if self.extras["level"] not in LEVELS:
            raise ValueError(
                "registry level must be one of %s, found %s" % (", ".join(LEVELS), self.extras["level"])
            )
        if (self.extras["level"] == "path") != bool(self.extras.get("series")):
            raise ValueError("path registries, and only they, name their target series")
        if not set(self.extras["protected"]) <= set(self.nodes):
            raise ValueError("protected node types must be declared")
        if not set(self.extras["base"]) <= set(self.by_name):
            raise ValueError("base relation types must be declared")

    @classmethod
    @timed("registry.registry.from_declaration")
    def from_declaration(cls, declared):
        edges = [
            EdgeType(e["name"], e["origin"], e["pairs"], e["attributes"], symmetric=e["symmetric"])
            for e in declared["edges"]
        ]
        return cls(
            declared["name"],
            declared["nodes"],
            edges,
            declared["width"],
            declared["markers"],
            declared["extras"],
        )

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

    @timed("registry.registry.declaration")
    def declaration(self):
        return dict(
            name=self.name,
            width=self.width,
            features=self.features,
            nodes=list(self.nodes),
            relations=self.relations,
            markers=list(self.markers),
            edges=[edge.declaration() for edge in self.edges],
            extras=self.extras,
        )

    @timed("registry.registry.fingerprint")
    def fingerprint(self):
        return sha256_json(self.declaration())
