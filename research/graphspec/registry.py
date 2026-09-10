import hashlib
import itertools
import json

import numpy as np

from .declarations import Recipe


class Registry:

    def __init__(self, domain, version, nodes, edges, width=64, markers=(), layouts=None,
                 modifiers=(), recipes=(), construction=None, extras=None):
        self.domain, self.version, self.width = domain, version, width
        self.nodes, self.edges = tuple(nodes), tuple(edges)
        self.features = 2 * width
        self.node_index = {name: i for i, name in enumerate(self.nodes)}
        self.relations = self._allocate()
        self.by_name = {edge.name: edge for edge in self.edges}
        self.by_code = {}
        for edge in self.edges:
            self.by_code[edge.code] = edge
            if not edge.symmetric:
                self.by_code[edge.reverse] = edge
        self.canonical = np.zeros(self.relations, bool)
        for edge in self.edges:
            self.canonical[edge.code] = True
        self.families = tuple(dict.fromkeys(edge.family for edge in self.edges))
        self.attributed = tuple(edge.name for edge in self.edges if edge.attributes)
        self.collections = tuple(edge for edge in self.edges if edge.collection)
        self.flagged = tuple(edge for edge in self.edges if edge.field)
        self.keyed = tuple(edge for edge in self.edges if edge.column)
        self.markers = tuple(markers)
        self.marker_index = {name: self.relations + i for i, name in enumerate(self.markers)}
        self.relation_slots = self.relations + len(self.markers)
        self.layouts = dict(layouts or {})
        self.modifiers = tuple(modifiers)
        self.modifier_by_name = {modifier.name: modifier for modifier in self.modifiers}
        self.construction = dict(construction or {})
        self.extras = dict(extras or {})
        self.recipes = {}
        for recipe in recipes:
            recipe.registry = self
            self.recipes[recipe.name] = recipe
        self._check()

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

    def _check(self):
        if len(self.by_name) != len(self.edges):
            raise ValueError('duplicate edge type name')
        for edge in self.edges:
            for endpoint in edge.sources + edge.targets:
                if endpoint not in self.node_index:
                    raise ValueError('edge %s names an undeclared node type %s' % (edge.name, endpoint))
        for group in self.layouts.values():
            for modifier in self.modifiers:
                covered = set(group) & set(modifier.bases)
                if covered and covered != set(group):
                    raise ValueError('modifier %s covers part of a layout group' % modifier.name)
        for group in self.layouts.values():
            for base in group:
                if base not in self.recipes:
                    raise ValueError('layout group names an undeclared recipe ' + base)
        for modifier in self.modifiers:
            for base in modifier.bases:
                if base not in self.recipes:
                    raise ValueError('modifier %s names an undeclared recipe %s' % (modifier.name, base))

    def code(self, name):
        return self.by_name[name].code

    def reverse(self, name):
        return self.by_name[name].reverse

    def attributes(self, name):
        return self.by_name[name].attributes

    def default(self, name):
        return self.by_name[name].default

    def kind(self, name):
        return self.node_index[name]

    def marker(self, name):
        return self.marker_index[name]

    def family_codes(self, families):
        selected = set(families)
        result = []
        for edge in self.edges:
            if edge.family in selected:
                result.append(edge.code)
                if not edge.symmetric:
                    result.append(edge.reverse)
        return sorted(result)

    def validate(self, name, source_kind, target_kind):
        edge = self.by_name[name]
        if self.nodes[source_kind] not in edge.sources or self.nodes[target_kind] not in edge.targets:
            raise ValueError('edge %s rejects %s -> %s' % (name, self.nodes[source_kind], self.nodes[target_kind]))

    def modifiers_for(self, bases):
        selected = set(bases)
        return tuple(modifier for modifier in self.modifiers if selected & set(modifier.bases))

    def designs_for(self, bases):
        result = list(bases)
        for base in bases:
            active = [modifier for modifier in self.modifiers if base in modifier.bases]
            for size in range(1, len(active) + 1):
                for combination in itertools.combinations(active, size):
                    result.append('+'.join([base] + [modifier.name for modifier in combination]))
        return tuple(result)

    def resolve(self, design):
        parts = design.split('+')
        base = self.recipes[parts[0]]
        families = list(base.families)
        for name in parts[1:]:
            if name not in self.modifier_by_name:
                raise ValueError('unknown graph modifier ' + name)
            families.append(self.modifier_by_name[name].family)
        return Recipe(design, base.view, base.nodes, families, base.lifted, self)

    def declaration(self):
        return dict(domain=self.domain, version=self.version, width=self.width, features=self.features,
                    nodes=list(self.nodes), relations=self.relations, markers=list(self.markers),
                    edges=[edge.declaration() for edge in self.edges],
                    layouts={k: list(v) for k, v in self.layouts.items()},
                    modifiers=[modifier.declaration() for modifier in self.modifiers],
                    recipes=[recipe.declaration() for recipe in self.recipes.values()],
                    construction={name: {k: list(v) if isinstance(v, tuple) else v for k, v in recipe.items()}
                                  for name, recipe in self.construction.items()},
                    extras=self.extras)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.declaration(), sort_keys=True).encode()).hexdigest()
