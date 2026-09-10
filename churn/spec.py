from graphspec import EdgeType, Modifier, Recipe, Registry


VERSION = 1
WIDTH = 64

NODES = ('policy', 'client', 'producer', 'placement_specialist', 'account_executive', 'branch',
         'billing_company', 'issuing_company', 'servicer', 'policy_type', 'policy_type_group',
         'city', 'naics', 'sic', 'state', 'platform')
CORE = ('policy', 'client', 'producer', 'branch')

HOLDER = 'holder'
PLACEMENT = 'placement'
SERVICING = 'servicing'
CLASSIFICATION = 'classification'
GEOGRAPHY = 'geography'
INDUSTRY = 'industry'
RENEWAL = 'renewal'

BOOK = (HOLDER, PLACEMENT, SERVICING, CLASSIFICATION)
TERM = ('term_days', 'cancelled')
GAP = ('gap_days',)

EDGES = (
    EdgeType('held_by', HOLDER, ('policy',), ('client',), TERM, column='client_key'),
    EdgeType('placed_by', PLACEMENT, ('policy',), ('producer',), column='producer_key'),
    EdgeType('specialist', PLACEMENT, ('policy',), ('placement_specialist',), column='placement_specialist_key'),
    EdgeType('executive', PLACEMENT, ('policy',), ('account_executive',), column='account_executive_key'),
    EdgeType('booked_at', SERVICING, ('policy',), ('branch',), column='branch_key'),
    EdgeType('billed_by', SERVICING, ('policy',), ('billing_company',), column='billing_company_key'),
    EdgeType('issued_by', SERVICING, ('policy',), ('issuing_company',), column='issuing_company_key'),
    EdgeType('serviced_by', SERVICING, ('policy',), ('servicer',), column='servicer_key'),
    EdgeType('platform', SERVICING, ('policy',), ('platform',), column='integrated_platform'),
    EdgeType('of_type', CLASSIFICATION, ('policy',), ('policy_type',), column='policy_type_key'),
    EdgeType('in_group', CLASSIFICATION, ('policy_type',), ('policy_type_group',), column='policy_type_group_code'),
    EdgeType('located_in', GEOGRAPHY, ('client',), ('city',), column='client_city'),
    EdgeType('issued_in', GEOGRAPHY, ('policy',), ('state',), column='issuing_state_code'),
    EdgeType('naics', INDUSTRY, ('client',), ('naics',), column='naics_code'),
    EdgeType('sic', INDUSTRY, ('client',), ('sic',), column='sic_code'),
    EdgeType('renews', RENEWAL, ('policy',), ('policy',), GAP, column='expiring_policy_key'),
)

RECIPE_LIST = (
    Recipe('P-core', 'snapshot', CORE, BOOK),
    Recipe('P-full', 'snapshot', NODES, BOOK),
    Recipe('R-chain', 'snapshot', NODES, BOOK, 'renewal'),
)

LAYOUTS = {'book_graph': ('P-core', 'P-full')}

MODIFIERS = (
    Modifier('geography', GEOGRAPHY, ('P-core', 'P-full', 'R-chain'), 'geography'),
    Modifier('industry', INDUSTRY, ('P-core', 'P-full', 'R-chain'), 'industry'),
)

CONSTRUCTION = dict(
    renewal=dict(window=1095),
    client_history=dict(fanout=32, window=1095),
    label=dict(grace=90, definition='client_continuity'),
)

MARKERS = ('non_relation',)

VIEWS = {
    'snapshot': ('x', 'entity', 'entity_kind', 'snapshot_offsets', 'spatial', 'spatial_step',
                 'spatial_attribute_row', 'query_context'),
}


REGISTRY = Registry('churn_tx', VERSION, NODES, EDGES, WIDTH, MARKERS, LAYOUTS, MODIFIERS,
                    RECIPE_LIST, CONSTRUCTION)

RELATIONS = REGISTRY.relations
RECIPES = REGISTRY.recipes
code = REGISTRY.code
kind = REGISTRY.kind
resolve = REGISTRY.resolve
designs_for = REGISTRY.designs_for
modifiers_for = REGISTRY.modifiers_for
validate = REGISTRY.validate
declaration = REGISTRY.declaration
fingerprint = REGISTRY.fingerprint
