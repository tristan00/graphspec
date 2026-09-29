from graphspec.infra.timing import timed


class EdgeType:
    __slots__ = (
        "name",
        "origin",
        "pairs",
        "sources",
        "targets",
        "attributes",
        "symmetric",
        "code",
        "reverse",
    )

    @timed("declarations.edgetype")
    def __init__(self, name, origin, pairs, attributes=(), symmetric=False):
        if not name or not origin:
            raise ValueError("edge declarations require a name and origin")
        self.name, self.origin = name, tuple(origin)
        self.pairs = tuple((source, target) for source, target in pairs)
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError("edge endpoint pairs must be nonempty and unique: " + name)
        self.sources = tuple(dict.fromkeys(source for source, _ in self.pairs))
        self.targets = tuple(dict.fromkeys(target for _, target in self.pairs))
        self.attributes, self.symmetric = tuple(attributes), symmetric
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("edge attribute names must be unique: " + name)
        self.code = self.reverse = -1

    @timed("declarations.edgetype.declaration")
    def declaration(self):
        return dict(
            name=self.name,
            origin=list(self.origin),
            pairs=[list(pair) for pair in self.pairs],
            attributes=list(self.attributes),
            symmetric=self.symmetric,
            code=self.code,
            reverse=self.reverse,
        )
