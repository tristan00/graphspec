from collections import Counter

import optuna
import pytest

from research.churn import spec as churn
from research.churn.domain import DOMAIN as CHURN_DOMAIN
from research.graph_temporal.registry import DOMAIN as TW_DOMAIN
from research.graphspec import search
from research.graph_temporal import spec as total_war
from research.graphspec import EdgeType, Modifier, Recipe, Registry

DOMAINS = [total_war.REGISTRY, churn.REGISTRY]
IDS = [registry.domain for registry in DOMAINS]


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_codes_are_unique_and_dense(registry):
    codes = [e.code for e in registry.edges] + [e.reverse for e in registry.edges if not e.symmetric]
    assert sorted(codes) == list(range(registry.relations))
    assert max(Counter(codes).values()) == 1


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_canonical_marks_the_forward_direction_only(registry):
    assert registry.canonical.sum() == len(registry.edges)
    for edge in registry.edges:
        assert registry.canonical[edge.code]
        if not edge.symmetric:
            assert not registry.canonical[edge.reverse]


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_markers_sit_above_every_relation_code(registry):
    assert registry.relation_slots == registry.relations + len(registry.markers)
    for name in registry.markers:
        assert registry.marker(name) >= registry.relations


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_every_endpoint_names_a_declared_node_type(registry):
    for edge in registry.edges:
        for endpoint in edge.sources + edge.targets:
            assert endpoint in registry.node_index


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_every_design_resolves_and_keeps_declared_families(registry):
    for group in list(registry.layouts.values()) + [(name,) for name in registry.recipes]:
        for design in registry.designs_for(group):
            recipe = registry.resolve(design)
            assert recipe.name == design
            assert set(recipe.families) <= set(registry.families)
            assert len(recipe.codes()) == len(set(recipe.codes()))


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_modifiers_only_add_their_own_family_codes(registry):
    for modifier in registry.modifiers:
        for base in modifier.bases:
            plain = set(registry.resolve(base).codes())
            with_modifier = set(registry.resolve(base + '+' + modifier.name).codes())
            assert with_modifier - plain == set(registry.family_codes([modifier.family]))


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_designs_cover_the_modifier_powerset(registry):
    for group in registry.layouts.values():
        designs = registry.designs_for(group)
        for base in group:
            applicable = [m for m in registry.modifiers if base in m.bases]
            assert len(set(designs) & {d for d in designs if d.split('+')[0] == base}) == 2 ** len(applicable)


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_unknown_modifier_is_refused(registry):
    base = next(iter(registry.recipes))
    with pytest.raises(ValueError):
        registry.resolve(base + '+invented')


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_endpoint_validation_rejects_undeclared_pairs(registry):
    edge = registry.edges[0]
    registry.validate(edge.name, registry.kind(edge.sources[0]), registry.kind(edge.targets[0]))
    outside = [n for n in registry.nodes if n not in edge.sources]
    if outside:
        with pytest.raises(ValueError):
            registry.validate(edge.name, registry.kind(outside[0]), registry.kind(edge.targets[0]))


@pytest.mark.parametrize('registry', DOMAINS, ids=IDS)
def test_fingerprint_is_stable_and_declaration_round_trips(registry):
    assert registry.fingerprint() == registry.fingerprint()
    declared = registry.declaration()
    assert declared['domain'] == registry.domain
    assert {e['name'] for e in declared['edges']} == {e.name for e in registry.edges}
    assert {r['name'] for r in declared['recipes']} == set(registry.recipes)
    assert declared['relations'] == registry.relations


def test_domains_are_independent():
    assert total_war.fingerprint() != churn.fingerprint()
    assert total_war.REGISTRY.relations != churn.REGISTRY.relations
    assert set(total_war.NODES).isdisjoint(churn.NODES)


def test_core_rejects_a_malformed_declaration():
    node = ('a', 'b')
    with pytest.raises(ValueError):
        Registry('bad', 1, node, (EdgeType('x', 'f', ('a',), ('missing',)),))
    with pytest.raises(ValueError):
        Registry('bad', 1, node, (EdgeType('x', 'f', ('a',), ('b',)), EdgeType('x', 'f', ('b',), ('a',))))
    with pytest.raises(ValueError):
        EdgeType('x', 'f', ('a',), ('b',), ('one',), (1.0, 2.0))
    with pytest.raises(ValueError):
        EdgeType('x', 'f', ('a',), ('b',), collection='c', column='d')


def test_core_rejects_partial_layout_group_coverage():
    edges = (EdgeType('link', 'f', ('a',), ('b',)), EdgeType('extra', 'g', ('a',), ('b',)))
    recipes = (Recipe('L', 'v', ('a', 'b'), ('f',)), Recipe('R', 'v', ('a', 'b'), ('f',)))
    modifier = Modifier('only-left', 'g', ('L',), 'only_left')
    with pytest.raises(ValueError):
        Registry('bad', 1, ('a', 'b'), edges, layouts={'group': ('L', 'R')},
                 modifiers=(modifier,), recipes=recipes)


def test_churn_declares_a_foreign_key_binding_for_every_edge():
    assert len(churn.REGISTRY.keyed) == len(churn.REGISTRY.edges)
    assert not churn.REGISTRY.collections and not churn.REGISTRY.flagged
    assert {e.column for e in churn.REGISTRY.keyed} >= {
        'client_key', 'producer_key', 'branch_key', 'expiring_policy_key'}


def test_total_war_keeps_its_collection_and_flag_bindings():
    assert not total_war.REGISTRY.keyed
    assert len(total_war.REGISTRY.collections) == 6 and len(total_war.REGISTRY.flagged) == 6


PIPELINES = [TW_DOMAIN, CHURN_DOMAIN]
PIPELINE_IDS = [d.registry.domain for d in PIPELINES]


def categorical(options):
    return tuple(options)


def rebuild(domain, model, values):
    parameter = domain.layout_parameter(model)
    base = values[parameter] if parameter is not None else domain.bases(model)[0]
    active = [m.name for m in domain.modifiers(model) if values[m.parameter]]
    return '+'.join([base] + active)


@pytest.mark.parametrize('domain', PIPELINES, ids=PIPELINE_IDS)
def test_every_declared_design_round_trips_through_structure(domain):
    for model, designs in domain.designs.items():
        for design in designs:
            values, domains = search.structure(domain, dict(model=model, design=design), categorical)
            assert set(values) == set(domains)
            assert values['model'] == model
            assert rebuild(domain, model, values) == design


@pytest.mark.parametrize('domain', PIPELINES, ids=PIPELINE_IDS)
def test_proposals_are_always_declared_designs(domain):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for seed in range(8):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
        trial = study.ask()
        model, design, context = search.suggest_structure(domain, trial)
        assert design in domain.designs[model]
        assert set(context) == {switch.key for switch in domain.switches}
        expected = search.structure(domain, dict(model=model, design=design, **context), categorical)[0]
        assert set(trial.params) == set(expected)


@pytest.mark.parametrize('domain', PIPELINES, ids=PIPELINE_IDS)
def test_choices_cover_every_model_and_design(domain):
    entries = search.choices(domain)
    assert len(entries) == sum(len(d) for d in domain.designs.values()) * 2 ** len(domain.switches)
    assert len(set(entries)) == len(entries)


@pytest.mark.parametrize('domain', PIPELINES, ids=PIPELINE_IDS)
def test_every_model_declares_a_view_and_its_blocks(domain):
    known = set().union(*domain.blocks.values())
    for model in domain.models:
        assert domain.view(model) in {r.view for r in domain.registry.recipes.values()}
        assert isinstance(domain.blocks[model], frozenset)
        assert domain.blocks[model] <= known


def test_switches_are_declared_per_domain_not_baked_in():
    assert [s.parameter for s in TW_DOMAIN.switches] == ['turn_end']
    assert CHURN_DOMAIN.switches == ()
    assert 'turn_end' in search.structure(TW_DOMAIN, dict(model='M01', design='S-full'), categorical)[0]
    assert 'turn_end' not in search.structure(CHURN_DOMAIN, dict(model='C02', design='P-full'), categorical)[0]


def test_domain_rejects_a_model_naming_an_undeclared_graph():
    from research.graphspec import Domain

    with pytest.raises(ValueError):
        Domain(CHURN_DOMAIN.registry, dict(C99='not-a-recipe'))
