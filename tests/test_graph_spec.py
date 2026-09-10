from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from totalwar import spec
from totalwar.registry import DESIGNS


def test_every_code_is_unique_and_dense():
    codes = [edge.code for edge in spec.EDGES] + [edge.reverse for edge in spec.EDGES if not edge.symmetric]
    assert sorted(codes) == list(range(spec.RELATIONS))
    assert max(Counter(codes).values()) == 1


def test_markers_sit_above_every_relation_code():
    assert spec.NON_RELATION >= spec.RELATIONS
    assert spec.TURN_CONTEXT >= spec.RELATIONS
    assert spec.RELATION_SLOTS == spec.RELATIONS + len(spec.MARKERS)
    assert len({spec.NON_RELATION, spec.TURN_CONTEXT}) == len(spec.MARKERS)


def test_canonical_marks_forward_direction_only():
    assert spec.CANONICAL.sum() == len(spec.EDGES)
    for edge in spec.EDGES:
        assert spec.CANONICAL[edge.code]
        if not edge.symmetric:
            assert not spec.CANONICAL[edge.reverse]


def test_symmetric_types_have_no_reverse_code():
    for edge in spec.EDGES:
        assert (edge.reverse == -1) == edge.symmetric


def test_every_declared_type_has_a_matching_default():
    for edge in spec.EDGES:
        assert len(edge.default) == len(edge.attributes)


def test_attributed_types_are_exactly_those_with_a_schema():
    assert spec.ATTRIBUTED == tuple(edge.name for edge in spec.EDGES if edge.attributes)
    for name in spec.ATTRIBUTED:
        assert len(spec.attributes(name)) and len(spec.default(name)) == len(spec.attributes(name))


def test_every_registry_design_resolves_to_a_recipe():
    for designs in DESIGNS.values():
        for design in designs:
            recipe = spec.resolve(design)
            assert recipe.name == design
            assert set(recipe.families) <= set(spec.FAMILIES)


def test_unknown_modifier_is_rejected():
    with pytest.raises(ValueError):
        spec.resolve('S-full+invented')


def test_core_recipe_selects_the_documented_node_types():
    recipe = spec.resolve('S-core')
    assert set(recipe.nodes) == set(spec.CORE)
    mask = recipe.kind_mask()
    assert mask.sum() == len(spec.CORE)
    for name in spec.CORE:
        assert mask[spec.kind(name)]


def test_core_recipe_excludes_catalogue_and_context_edges():
    core = set(spec.CORE)
    for edge in spec.EDGES:
        reachable = set(edge.sources) & core and set(edge.targets) & core
        if edge.family == spec.MEMBERSHIP or edge.family == spec.CONTEXT:
            assert not reachable


def test_modifiers_add_only_their_own_family():
    base = spec.resolve('S-full')
    with_colocated = spec.resolve('S-full+colocated')
    added = set(with_colocated.codes()) - set(base.codes())
    assert added == {spec.code('colocated')}


def test_turn_boundary_adds_exactly_the_boundary_codes():
    base = spec.resolve('X-allpast')
    boundary = spec.resolve('X-allpast+turn-boundary')
    added = set(boundary.codes()) - set(base.codes())
    assert added == set(spec.BOUNDARY_CODES)


def test_lifted_layout_comes_from_the_recipe():
    assert spec.resolve('X-chain').lifted == 'chain'
    assert spec.resolve('X-skip').lifted == 'skip'
    assert spec.resolve('X-allpast').lifted == 'allpast'
    assert spec.resolve('S-full').lifted is None


def test_endpoint_validation_rejects_undeclared_kinds():
    spec.validate('adjacent', spec.kind('region'), spec.kind('region'))
    with pytest.raises(ValueError):
        spec.validate('adjacent', spec.kind('faction'), spec.kind('region'))
    with pytest.raises(ValueError):
        spec.validate('skill', spec.kind('skill'), spec.kind('skill'))


def test_fingerprint_tracks_the_declaration():
    before = spec.fingerprint()
    assert before == spec.fingerprint()
    edge = spec.BY_NAME['adjacent']
    original = edge.targets
    edge.targets = original + ('province',)
    try:
        assert spec.fingerprint() != before
    finally:
        edge.targets = original
    assert spec.fingerprint() == before


def test_declaration_round_trips_every_type_and_recipe():
    declared = spec.declaration()
    assert {entry['name'] for entry in declared['edges']} == {edge.name for edge in spec.EDGES}
    assert {entry['name'] for entry in declared['recipes']} == set(spec.RECIPES)
    assert declared['relations'] == spec.RELATIONS
    assert declared['features'] == 2 * declared['width']


def test_construction_recipes_cover_every_generated_structure():
    assert set(spec.CONSTRUCTION) == {'chain', 'skip', 'allpast', 'event_dag', 'walk',
                                      'expanded_walk', 'turn_walk', 'boundary', 'trace'}
    assert spec.CONSTRUCTION['skip']['lags'] and spec.CONSTRUCTION['event_dag']['lags']
    for name in ('skip', 'allpast', 'event_dag', 'walk'):
        assert spec.CONSTRUCTION[name]['window'] > 0
    assert spec.CONSTRUCTION['expanded_walk']['samples'] >= spec.CONSTRUCTION['walk']['samples']


def test_builders_read_their_windows_from_the_spec():
    from totalwar import graph_variants, turns, walk_cache

    assert graph_variants.LOOKBACK == spec.CONSTRUCTION['boundary']['lookback']
    assert graph_variants.WINDOW == spec.CONSTRUCTION['boundary']['window']
    assert turns.walks.__defaults__ == (spec.CONSTRUCTION['turn_walk']['samples'],
                                        spec.CONSTRUCTION['turn_walk']['hops'],
                                        spec.CONSTRUCTION['turn_walk']['lookback'])
    assert walk_cache.extend.__defaults__ == (spec.CONSTRUCTION['expanded_walk']['samples'],)
    assert walk_cache.cache_path(Path('bank')).name.endswith(str(spec.CONSTRUCTION['expanded_walk']['samples']))


def test_mixed_view_codes_are_declared_and_distinct():
    assert len(set(spec.MIXED_EDGES.values())) == len(spec.MIXED_EDGES)
    assert spec.mixed_kind('observation') == 0 and spec.mixed_kind('action') == len(spec.MIXED_NODES) - 1
    assert set(spec.MIXED_NODES) == {'observation', 'change', 'bundle', 'trace', 'action'}


def test_construction_recipes_reach_the_fingerprint():
    before = spec.fingerprint()
    original = spec.CONSTRUCTION['walk']['hops']
    spec.CONSTRUCTION['walk']['hops'] = original + 1
    try:
        assert spec.fingerprint() != before
    finally:
        spec.CONSTRUCTION['walk']['hops'] = original
    assert spec.fingerprint() == before
