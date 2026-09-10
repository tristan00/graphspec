import numpy as np


class EdgeType:

    __slots__ = ('name', 'family', 'sources', 'targets', 'attributes', 'default', 'symmetric',
                 'collection', 'subject_field', 'field', 'column', 'code', 'reverse')

    def __init__(self, name, family, sources, targets, attributes=(), default=(), symmetric=False,
                 collection=None, subject_field='key', field=None, column=None):
        self.name, self.family = name, family
        self.sources, self.targets = tuple(sources), tuple(targets)
        self.attributes, self.symmetric = tuple(attributes), symmetric
        self.default = tuple(default) if default else (0.0,) * len(self.attributes)
        if len(self.default) != len(self.attributes):
            raise ValueError('edge %s default does not match its attribute schema' % name)
        self.collection, self.subject_field, self.field = collection, subject_field, field
        self.column = column
        if collection and len(self.targets) != 1:
            raise ValueError('collection edge %s needs exactly one target kind' % name)
        if column and len(self.targets) != 1:
            raise ValueError('column edge %s needs exactly one target kind' % name)
        if sum(binding is not None for binding in (collection, field, column)) > 1:
            raise ValueError('edge %s declares more than one extraction binding' % name)
        self.code = self.reverse = -1

    def declaration(self):
        return dict(name=self.name, family=self.family, sources=list(self.sources),
                    targets=list(self.targets), attributes=list(self.attributes),
                    default=list(self.default), symmetric=self.symmetric, collection=self.collection,
                    subject_field=self.subject_field, field=self.field, column=self.column,
                    code=self.code, reverse=self.reverse)


class Modifier:

    __slots__ = ('name', 'family', 'bases', 'parameter')

    def __init__(self, name, family, bases, parameter):
        self.name, self.family = name, family
        self.bases, self.parameter = tuple(bases), parameter

    def declaration(self):
        return dict(name=self.name, family=self.family, bases=list(self.bases), parameter=self.parameter)


class Recipe:

    __slots__ = ('name', 'view', 'nodes', 'families', 'lifted', 'registry')

    def __init__(self, name, view, nodes, families, lifted=None, registry=None):
        self.name, self.view, self.nodes = name, view, tuple(nodes)
        self.families, self.lifted = tuple(families), lifted
        self.registry = registry

    def kinds(self):
        return np.asarray(sorted(self.registry.kind(n) for n in self.nodes), np.int64)

    def kind_mask(self):
        mask = np.zeros(len(self.registry.nodes), bool)
        mask[self.kinds()] = True
        return mask

    def codes(self):
        return np.asarray(self.registry.family_codes(self.families), np.int64)

    def attributed(self):
        selected = set(self.families)
        return tuple(edge.name for edge in self.registry.edges if edge.attributes and edge.family in selected)

    def declaration(self):
        return dict(name=self.name, view=self.view, nodes=list(self.nodes),
                    families=list(self.families), lifted=self.lifted)
