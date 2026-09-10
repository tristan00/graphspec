import hashlib
import json

import numpy as np


VERSION = 1
WIDTH = 64
FEATURES = 2 * WIDTH

NODES = ('faction', 'character', 'region', 'settlement', 'province', 'slot', 'building',
         'unit', 'tech', 'skill', 'item', 'ritual', 'context')
CORE = ('faction', 'character', 'region', 'settlement', 'province')

ACTORS = ('faction', 'character', 'settlement')
CATALOGUE = ('count', 'level', 'strength', 'cost')
CATALOGUE_DEFAULT = (1.0, 0.0, 0.0, 0.0)

TOPOLOGY = 'topology'
OWNERSHIP = 'ownership'
MEMBERSHIP = 'membership'
DIPLOMATIC = 'diplomatic'
CONTEXT = 'context'
COLOCATED = 'colocated'
BOUNDARY = 'boundary'
LIFTED = 'lifted'

SNAPSHOT = (TOPOLOGY, OWNERSHIP, MEMBERSHIP, DIPLOMATIC, CONTEXT)


class EdgeType:

    __slots__ = ('name', 'family', 'sources', 'targets', 'attributes', 'default', 'symmetric', 'code', 'reverse')

    def __init__(self, name, family, sources, targets, attributes=(), default=(), symmetric=False):
        self.name, self.family = name, family
        self.sources, self.targets = tuple(sources), tuple(targets)
        self.attributes, self.symmetric = tuple(attributes), symmetric
        self.default = tuple(default) if default else (0.0,) * len(self.attributes)
        if len(self.default) != len(self.attributes):
            raise ValueError('edge %s default does not match its attribute schema' % name)
        self.code = self.reverse = -1

    def declaration(self):
        return dict(name=self.name, family=self.family, sources=list(self.sources),
                    targets=list(self.targets), attributes=list(self.attributes),
                    default=list(self.default), symmetric=self.symmetric,
                    code=self.code, reverse=self.reverse)


EDGES = (
    EdgeType('owns', OWNERSHIP, ('faction',), ('region', 'settlement', 'character')),
    EdgeType('located', TOPOLOGY, ('region', 'settlement', 'character'), ('region',)),
    EdgeType('contains', TOPOLOGY, ('province', 'region') + ACTORS, ('region', 'settlement', 'character', 'slot')),
    EdgeType('adjacent', TOPOLOGY, ('region',), ('region',)),
    EdgeType('unit', MEMBERSHIP, ACTORS, ('unit',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('skill', MEMBERSHIP, ACTORS, ('skill',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('item', MEMBERSHIP, ACTORS, ('item',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('building', MEMBERSHIP, ACTORS + ('slot',), ('building',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('tech', MEMBERSHIP, ACTORS, ('tech',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('ritual', MEMBERSHIP, ACTORS, ('ritual',), CATALOGUE, CATALOGUE_DEFAULT),
    EdgeType('war', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('allied', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('trade', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('vassal', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('nap', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('access', DIPLOMATIC, ('faction',), ('faction',)),
    EdgeType('context', CONTEXT, ('faction',), ('context',)),
    EdgeType('colocated', COLOCATED, ('character', 'settlement'), ('character', 'settlement'), symmetric=True),
    EdgeType('temporal', LIFTED, NODES, NODES, symmetric=True),
    EdgeType('boundary_first', BOUNDARY, NODES, NODES, symmetric=True),
    EdgeType('boundary_last', BOUNDARY, NODES, NODES, symmetric=True),
    EdgeType('boundary_both', BOUNDARY, NODES, NODES, symmetric=True),
)


def _allocate(types):
    code = 0
    for edge in types:
        edge.code = code
        code += 1
    for edge in types:
        if not edge.symmetric:
            edge.reverse = code
            code += 1
    return code


RELATIONS = _allocate(EDGES)
BY_NAME = {edge.name: edge for edge in EDGES}
BY_CODE = {}
for _edge in EDGES:
    BY_CODE[_edge.code] = _edge
    if not _edge.symmetric:
        BY_CODE[_edge.reverse] = _edge

CANONICAL = np.zeros(RELATIONS, bool)
for _edge in EDGES:
    CANONICAL[_edge.code] = True

NODE_INDEX = {name: i for i, name in enumerate(NODES)}
FAMILIES = tuple(dict.fromkeys(edge.family for edge in EDGES))
ATTRIBUTED = tuple(edge.name for edge in EDGES if edge.attributes)
BOUNDARY_CODES = tuple(BY_NAME[name].code for name in ('boundary_first', 'boundary_last', 'boundary_both'))
MARKERS = ('non_relation', 'turn_context')
NON_RELATION = RELATIONS
TURN_CONTEXT = RELATIONS + 1
RELATION_SLOTS = RELATIONS + len(MARKERS)


def code(name):
    return BY_NAME[name].code


def reverse(name):
    return BY_NAME[name].reverse


def attributes(name):
    return BY_NAME[name].attributes


def default(name):
    return BY_NAME[name].default


def kind(name):
    return NODE_INDEX[name]


def family_codes(families):
    selected = set(families)
    result = []
    for edge in EDGES:
        if edge.family in selected:
            result.append(edge.code)
            if not edge.symmetric:
                result.append(edge.reverse)
    return sorted(result)


def validate(name, source_kind, target_kind):
    edge = BY_NAME[name]
    if NODES[source_kind] not in edge.sources or NODES[target_kind] not in edge.targets:
        raise ValueError('edge %s rejects %s -> %s' % (name, NODES[source_kind], NODES[target_kind]))


class Recipe:

    __slots__ = ('name', 'view', 'nodes', 'families', 'lifted')

    def __init__(self, name, view, nodes, families, lifted=None):
        self.name, self.view, self.nodes = name, view, tuple(nodes)
        self.families, self.lifted = tuple(families), lifted

    def kinds(self):
        return np.asarray(sorted(NODE_INDEX[n] for n in self.nodes), np.int64)

    def kind_mask(self):
        mask = np.zeros(len(NODES), bool)
        mask[self.kinds()] = True
        return mask

    def codes(self):
        return np.asarray(family_codes(self.families), np.int64)

    def attributed(self):
        selected = set(self.families)
        return tuple(edge.name for edge in EDGES if edge.attributes and edge.family in selected)

    def declaration(self):
        return dict(name=self.name, view=self.view, nodes=list(self.nodes),
                    families=list(self.families), lifted=self.lifted)


SNAPSHOT_VIEW = 'snapshot'

RECIPES = {
    'S-core': Recipe('S-core', SNAPSHOT_VIEW, CORE, SNAPSHOT),
    'S-full': Recipe('S-full', SNAPSHOT_VIEW, NODES, SNAPSHOT),
    'X-chain': Recipe('X-chain', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'chain'),
    'X-skip': Recipe('X-skip', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'skip'),
    'X-allpast': Recipe('X-allpast', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'allpast'),
    'O-mixed': Recipe('O-mixed', 'mixed', NODES, ()),
    'D-event': Recipe('D-event', 'event', NODES, ()),
    'E-entity': Recipe('E-entity', 'entity', NODES, ()),
    'B-role': Recipe('B-role', 'role', NODES, ()),
    'H-neighbor': Recipe('H-neighbor', 'neighbor', NODES, ()),
    'W-temporal': Recipe('W-temporal', 'walk', NODES, ()),
    'N-affinity': Recipe('N-affinity', 'affinity', NODES, ()),
    'V-signals': Recipe('V-signals', 'signals', NODES, ()),
}

MODIFIERS = {'colocated': COLOCATED, 'turn-boundary': BOUNDARY}


def resolve(design):
    parts = design.split('+')
    base = RECIPES[parts[0]]
    families = list(base.families)
    for modifier in parts[1:]:
        if modifier not in MODIFIERS:
            raise ValueError('unknown graph modifier ' + modifier)
        families.append(MODIFIERS[modifier])
    return Recipe(design, base.view, base.nodes, families, base.lifted)


def declaration():
    return dict(version=VERSION, width=WIDTH, features=FEATURES, nodes=list(NODES),
                relations=RELATIONS, edges=[edge.declaration() for edge in EDGES],
                markers=list(MARKERS),
                recipes=[recipe.declaration() for recipe in RECIPES.values()],
                modifiers=dict(MODIFIERS))


def fingerprint():
    return hashlib.sha256(json.dumps(declaration(), sort_keys=True).encode()).hexdigest()
