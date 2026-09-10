from ..graphspec import EdgeType, Modifier, Recipe, Registry


VERSION = 2
WIDTH = 64
FEATURES = 2 * WIDTH

NODES = ('faction', 'character', 'region', 'settlement', 'province', 'slot', 'building',
         'unit', 'tech', 'skill', 'item', 'ritual', 'context')
CORE = ('faction', 'character', 'region', 'settlement', 'province')

ACTORS = ('faction', 'character', 'settlement')
CATALOGUE = ('count', 'level', 'strength_pct', 'cost')
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
SNAPSHOT_VIEW = 'snapshot'

EDGES = (
    EdgeType('owns', OWNERSHIP, ('faction',), ('region', 'settlement', 'character')),
    EdgeType('located', TOPOLOGY, ('region', 'settlement', 'character'), ('region',)),
    EdgeType('contains', TOPOLOGY, ('province', 'region') + ACTORS, ('region', 'settlement', 'character', 'slot')),
    EdgeType('adjacent', TOPOLOGY, ('region',), ('region',)),
    EdgeType('unit', MEMBERSHIP, ACTORS, ('unit',), CATALOGUE, CATALOGUE_DEFAULT, collection='unit_cards'),
    EdgeType('skill', MEMBERSHIP, ACTORS, ('skill',), CATALOGUE, CATALOGUE_DEFAULT, collection='skills'),
    EdgeType('item', MEMBERSHIP, ACTORS, ('item',), CATALOGUE, CATALOGUE_DEFAULT, collection='equipped'),
    EdgeType('building', MEMBERSHIP, ACTORS + ('slot',), ('building',), CATALOGUE, CATALOGUE_DEFAULT,
             collection='built', subject_field='value'),
    EdgeType('tech', MEMBERSHIP, ACTORS, ('tech',), CATALOGUE, CATALOGUE_DEFAULT, collection='tech'),
    EdgeType('ritual', MEMBERSHIP, ACTORS, ('ritual',), CATALOGUE, CATALOGUE_DEFAULT, collection='rites'),
    EdgeType('war', DIPLOMATIC, ('faction',), ('faction',), field='at_war'),
    EdgeType('allied', DIPLOMATIC, ('faction',), ('faction',), field='allied'),
    EdgeType('trade', DIPLOMATIC, ('faction',), ('faction',), field='trade'),
    EdgeType('vassal', DIPLOMATIC, ('faction',), ('faction',), field='their_vassal'),
    EdgeType('nap', DIPLOMATIC, ('faction',), ('faction',), field='nap'),
    EdgeType('access', DIPLOMATIC, ('faction',), ('faction',), field='mil_access'),
    EdgeType('context', CONTEXT, ('faction',), ('context',)),
    EdgeType('colocated', COLOCATED, ('character', 'settlement'), ('character', 'settlement'), symmetric=True),
    EdgeType('temporal', LIFTED, NODES, NODES, symmetric=True),
    EdgeType('boundary_first', BOUNDARY, NODES, NODES, symmetric=True),
    EdgeType('boundary_last', BOUNDARY, NODES, NODES, symmetric=True),
    EdgeType('boundary_both', BOUNDARY, NODES, NODES, symmetric=True),
)

RECIPE_LIST = (
    Recipe('S-core', SNAPSHOT_VIEW, CORE, SNAPSHOT),
    Recipe('S-full', SNAPSHOT_VIEW, NODES, SNAPSHOT),
    Recipe('X-chain', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'chain'),
    Recipe('X-skip', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'skip'),
    Recipe('X-allpast', SNAPSHOT_VIEW, NODES, SNAPSHOT, 'allpast'),
    Recipe('O-mixed', 'mixed', NODES, ()),
    Recipe('D-event', 'event', NODES, ()),
    Recipe('E-entity', 'entity', NODES, ()),
    Recipe('B-role', 'role', NODES, ()),
    Recipe('H-neighbor', 'neighbor', NODES, ()),
    Recipe('W-temporal', 'walk', NODES, ()),
    Recipe('N-affinity', 'affinity', NODES, ()),
    Recipe('V-signals', 'signals', NODES, ()),
)

LAYOUTS = {
    'snapshot_graph': ('S-core', 'S-full'),
    'temporal_layout': ('X-chain', 'X-skip', 'X-allpast'),
}

MODIFIERS = (
    Modifier('colocated', COLOCATED, ('S-core', 'S-full'), 'colocated'),
    Modifier('turn-boundary', BOUNDARY, ('X-chain', 'X-skip', 'X-allpast', 'D-event'), 'turn_boundary'),
)

CONSTRUCTION = dict(
    chain=dict(),
    skip=dict(lags=(1, 2, 4, 8, 16, 32, 64), window=256),
    allpast=dict(fanout=64, window=64),
    event_dag=dict(lags=(1, 2, 4, 8), window=512),
    walk=dict(seed=741, samples=8, hops=4, fanout=64, window=256),
    expanded_walk=dict(samples=32),
    turn_walk=dict(samples=32, hops=4, lookback=3),
    boundary=dict(lookback=3, window=64),
    trace=dict(size=4),
)

MIXED_NODES = ('observation', 'change', 'bundle', 'trace', 'action')
MIXED_EDGES = dict(previous_action=1, previous_observation=4, bundle_action=8,
                   change_link=32, member_group=512, trace_current=1024)

MARKERS = ('non_relation', 'turn_context')

REGISTRY = Registry('total_war', VERSION, NODES, EDGES, WIDTH, MARKERS, LAYOUTS, MODIFIERS,
                    RECIPE_LIST, CONSTRUCTION,
                    extras=dict(mixed_nodes=list(MIXED_NODES), mixed_edges=dict(MIXED_EDGES)))

RELATIONS = REGISTRY.relations
BY_NAME = REGISTRY.by_name
BY_CODE = REGISTRY.by_code
CANONICAL = REGISTRY.canonical
NODE_INDEX = REGISTRY.node_index
FAMILIES = REGISTRY.families
ATTRIBUTED = REGISTRY.attributed
COLLECTIONS = REGISTRY.collections
FLAGGED = REGISTRY.flagged
RECIPES = REGISTRY.recipes
MODIFIER_BY_NAME = REGISTRY.modifier_by_name
NON_RELATION = REGISTRY.marker('non_relation')
TURN_CONTEXT = REGISTRY.marker('turn_context')
RELATION_SLOTS = REGISTRY.relation_slots
BOUNDARY_CODES = tuple(BY_NAME[name].code for name in ('boundary_first', 'boundary_last', 'boundary_both'))

code = REGISTRY.code
reverse = REGISTRY.reverse
attributes = REGISTRY.attributes
default = REGISTRY.default
kind = REGISTRY.kind
family_codes = REGISTRY.family_codes
validate = REGISTRY.validate
modifiers_for = REGISTRY.modifiers_for
designs_for = REGISTRY.designs_for
resolve = REGISTRY.resolve
declaration = REGISTRY.declaration
fingerprint = REGISTRY.fingerprint


def mixed_kind(name):
    return MIXED_NODES.index(name)
