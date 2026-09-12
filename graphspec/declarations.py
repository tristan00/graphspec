class EdgeType:

    __slots__ = ('name', 'origin', 'pairs', 'sources', 'targets', 'attributes', 'symmetric', 'code', 'reverse')

    def __init__(self, name, origin, pairs, attributes=(), symmetric=False):
        self.name, self.origin = name, tuple(origin)
        self.pairs = tuple((source, target) for source, target in pairs)
        self.sources = tuple(dict.fromkeys(source for source, _ in self.pairs))
        self.targets = tuple(dict.fromkeys(target for _, target in self.pairs))
        self.attributes, self.symmetric = tuple(attributes), symmetric
        self.code = self.reverse = -1

    def declaration(self):
        return dict(name=self.name, origin=list(self.origin), pairs=[list(pair) for pair in self.pairs], attributes=list(self.attributes),
                    symmetric=self.symmetric, code=self.code, reverse=self.reverse)
