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
GAP = ('gap_days',)

EDGES = (
    EdgeType('held_by', HOLDER, ('policy',), ('client',), column='client_key'),
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
    Recipe('R-chain', 'snapshot', NODES, BOOK + (RENEWAL,)),
)

LAYOUTS = {'book_graph': ('P-core', 'P-full')}

MODIFIERS = (
    Modifier('geography', GEOGRAPHY, ('P-core', 'P-full', 'R-chain'), 'geography'),
    Modifier('industry', INDUSTRY, ('P-core', 'P-full', 'R-chain'), 'industry'),
)

CONSTRUCTION = dict(
    client_history=dict(fanout=32, window=1095),
    label=dict(horizon_days=365, tenure_days=365, min_policies=3, definition='customer_continuity'),
)

MARKERS = ('non_relation',)

VIEWS = {
    'snapshot': ('x', 'entity', 'entity_kind', 'snapshot_offsets', 'spatial', 'spatial_step',
                 'spatial_attribute_row', 'query_context'),
}

NUMBERS = ('policies_in_force', 'policies_known', 'tenure_days', 'age_days', 'gap_days')
CATEGORIES = ('policy_type_key', 'policy_type_group_code', 'client_city', 'issuing_state_code',
              'naics_code', 'sic_code', 'integrated_platform', 'pol_income_type', 'branch_key',
              'producer_key', 'servicer_key', 'billing_company_key', 'issuing_company_key', 'key')


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
