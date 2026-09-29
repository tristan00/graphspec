import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed

from .declarations import EdgeType
from .schema import TARGETS, Encoding, Schema

EXTRAS = ("target", "series", "signals", "base", "events", "several", "untasked")
STORED = ("member", "pairs")
FITTED = ("features", "values", "widths")


class Registry:
    @timed("registry.registry")
    def __init__(self, name, encoding, edges, markers, extras):
        self.name, self.encoding, self.schema = name, encoding, encoding.schema
        self.nodes, self.edges = tuple(self.schema.entities()), tuple(edges)
        if encoding.states is not None:
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
        self.grouping()

    @timed("registry.registry.grouping")
    def grouping(self):
        members = {}
        for name in self.extras["base"]:
            members.setdefault(self.group(name), []).append(name)
        self.groups = {group: tuple(names) for group, names in members.items()}
        self.named = {group: self.by_name[names[0]] for group, names in self.groups.items()}
        self.coded = {group: self.codes(names) for group, names in self.groups.items()}
        derived = [edge for edge in self.edges if edge.name not in self.tables and edge.name not in self.stored]
        for edge in derived:
            if edge.origin[0] == "project" and edge.origin[1] not in self.groups:
                raise ValueError(
                    "the derived relation %s projects %s, which is no table of relations; build the bank again"
                    % (edge.name, edge.origin[1])
                )
            self.named[edge.name], self.coded[edge.name] = edge, self.codes([edge.name])
        keys = {}
        self.grouped = np.zeros(self.relations, np.int64)
        for edge in self.edges:
            self.grouped[edge.code] = keys.setdefault((self.group(edge.name), False), len(keys))
            if not edge.symmetric:
                self.grouped[edge.reverse] = keys.setdefault((self.group(edge.name), True), len(keys))
        self.group_codes = len(keys)
        self.valued = tuple(group for group in self.groups if self.named[group].attributes)
        self.value_group = np.full(self.relations, -1, np.int64)
        for place, group in enumerate(self.valued):
            chosen = [self.by_name[name] for name in self.groups[group]]
            self.value_group[[edge.code for edge in chosen]] = place
            self.value_group[[edge.reverse for edge in chosen if not edge.symmetric]] = place

    def group(self, name):
        return self.tables.get(name, name)

    @timed("registry.registry.codes")
    def codes(self, names):
        return np.asarray([self.by_name[name].code for name in names], np.int64)

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

    def __getattr__(self, key):
        if key in FITTED:
            raise ValueError(
                "the encoders of the columns are fitted when the views of a statement of splits are built, and "
                "the registry %s holds the dictionaries of its bank only; %s needs the fitted encoders"
                % (self.__dict__.get("name"), key)
            )
        raise AttributeError(key)

    @timed("registry.registry.fitted")
    def fitted(self, encoding):
        edges = [EdgeType(e.name, e.origin, e.pairs, e.attributes, symmetric=e.symmetric) for e in self.edges]
        return Registry(self.name, encoding, edges, self.markers, self.extras)

    def code(self, name):
        return self.by_name[name].code

    def attributes(self, name):
        return self.named[name].attributes

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
            nodes=list(self.nodes),
            relations=self.relations,
            markers=list(self.markers),
            edges=[edge.declaration() for edge in self.edges],
            extras=self.extras,
        )

    @timed("registry.registry.fingerprint")
    def fingerprint(self):
        return sha256_json(self.declaration())
