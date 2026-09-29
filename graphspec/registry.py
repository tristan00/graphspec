import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed

from .declarations import EdgeType
from .schema import TARGETS, Encoding, Schema

EXTRAS = ("target", "series", "signals", "base", "events", "single", "several")
STORED = ("member", "pairs")


class Registry:
    @timed("registry.registry")
    def __init__(self, name, encoding, edges, markers, extras):
        self.name, self.encoding, self.schema = name, encoding, encoding.schema
        self.nodes, self.edges = tuple(self.schema.entities()), tuple(edges)
        self.features, self.values = encoding.features, encoding.values
        self.widths = {
            kind: encoding.width(columns) + (encoding.held[kind][1] if kind in encoding.held else 0)
            for kind, columns in self.schema.entities().items()
        }
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
        lacking = [name for name in EXTRAS if name not in self.extras]
        if lacking or len(self.extras) != len(EXTRAS):
            raise ValueError(
                "a registry records exactly %s; it lacks %s" % (", ".join(EXTRAS), ", ".join(lacking))
            )
        if self.extras["target"] not in TARGETS or self.extras["target"] != self.schema.aim():
            raise ValueError(
                "the registry records targets of the kind %r and its schema states %s"
                % (self.extras["target"], self.schema.aim())
            )
        if list(self.extras["series"]) != list(self.schema.series()):
            raise ValueError("the registry and its schema name different series")
        kinds = self.schema.tasks["kinds"]
        self.tasks = tuple(kinds)
        self.stops = np.asarray([entry["absent"] == "stop" for entry in kinds.values()], bool)
        self.aimed = np.asarray([entry["target"] is not None for entry in kinds.values()], bool)
        self.events = self.schema.events()
        self.seen = np.asarray(
            [self.events.index(kind) if kind in self.events else -1 for kind in kinds], np.int64
        )
        self.references = tuple(self.schema.tasks["references"])
        if not set(self.extras["base"]) <= set(self.by_name):
            raise ValueError("base relation types must be declared")
        self.tables, self.stored, self.about = {}, {}, {}
        rules = self.schema.about()
        for name in self.extras["base"]:
            origin = self.by_name[name].origin
            if origin[0] == "base":
                if origin[1] not in self.schema.edges or name not in encoding.relations[origin[1]]:
                    raise ValueError("the relation %s belongs to no stated edge table" % name)
                self.tables[name] = origin[1]
                if origin[1] in rules:
                    self.about[self.code(name)] = rules[origin[1]]
            elif origin[0] in STORED and origin[1] in self.schema.sets:
                self.stored[name] = origin[1]
            else:
                raise ValueError(
                    "the relation %s belongs to no stated edge table and no kind of set edge" % name
                )

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
