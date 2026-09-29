import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed

from .declarations import EdgeType
from .schema import LEVELS, Encoding, Schema


class Registry:
    @timed("registry.registry")
    def __init__(self, name, encoding, edges, markers, extras):
        self.name, self.encoding, self.schema = name, encoding, encoding.schema
        self.nodes, self.edges = tuple(self.schema.nodes), tuple(edges)
        self.features, self.values = encoding.features, encoding.values
        self.widths = {kind: encoding.width(columns) for kind, columns in self.schema.nodes.items()}
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
        path = self.extras["level"] == "path"
        if path != ("series" in self.extras) or (path and not self.extras["series"]):
            raise ValueError("path registries, and only they, name their target series")
        if not set(self.extras["protected"]) <= set(self.nodes):
            raise ValueError("protected node types must be declared")
        if not set(self.extras["base"]) <= set(self.by_name):
            raise ValueError("base relation types must be declared")
        self.tables = {}
        for name in self.extras["base"]:
            table = self.by_name[name].origin[1]
            if table not in self.schema.edges or name not in encoding.relations[table]:
                raise ValueError("the relation %s belongs to no stated edge table" % name)
            self.tables[name] = table

    @classmethod
    @timed("registry.registry.from_declaration")
    def from_declaration(cls, declared):
        schema = Schema(declared["schema"])
        edges = [
            EdgeType(e["name"], e["origin"], e["pairs"], e["attributes"], symmetric=e["symmetric"])
            for e in declared["edges"]
        ]
        return cls(
            declared["name"],
            Encoding(schema, **declared["encoding"]),
            edges,
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

    def ordered(self, name):
        return tuple(column for column in self.attributes(name) if self.schema.kind(column).ordered)

    def attribute_width(self, name):
        return self.encoding.width(self.attributes(name))

    def kind(self, name):
        return self.node_index[name]

    def marker(self, name):
        return self.marker_index[name]

    @timed("registry.registry.declaration")
    def declaration(self):
        return dict(
            name=self.name,
            schema=self.schema.mapping(),
            encoding=self.encoding.declaration(),
            features=self.features,
            values=self.values,
            widths=self.widths,
            nodes=list(self.nodes),
            relations=self.relations,
            markers=list(self.markers),
            edges=[edge.declaration() for edge in self.edges],
            extras=self.extras,
        )

    @timed("registry.registry.fingerprint")
    def fingerprint(self):
        return sha256_json(self.declaration())
