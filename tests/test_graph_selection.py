import optuna
import pytest

from totalwar import selector, spec
from totalwar.registry import DESIGNS, bases, layout_parameter


def rebuild(model, values):
    parameter = layout_parameter(model)
    base = values[parameter] if parameter is not None else bases(model)[0]
    active = [m.name for m in spec.modifiers_for(bases(model)) if values[m.parameter]]
    return '+'.join([base] + active)


@pytest.mark.parametrize('model', list(DESIGNS))
def test_structure_round_trips_every_design_of_every_model(model):
    for design in DESIGNS[model]:
        for turn in ('none', 'end'):
            values, domains = selector.structure(dict(model=model, design=design, turn_context=turn))
            assert set(values) == set(domains)
            assert values['model'] == model and values['turn_end'] == (turn == 'end')
            assert rebuild(model, values) == design


def test_modifier_switches_reach_only_models_whose_layouts_carry_them():
    for model in DESIGNS:
        offered = {m.parameter for m in spec.modifiers_for(bases(model))}
        recorded = set(selector.structure(dict(model=model, design=DESIGNS[model][0]))[0])
        assert offered <= recorded
        for modifier in spec.MODIFIERS:
            carried = bool(set(bases(model)) & set(modifier.bases))
            assert (modifier.parameter in recorded) == carried


def test_layout_coordinate_is_shared_by_models_with_the_same_options():
    assert layout_parameter('M13') == layout_parameter('M14') == 'temporal_layout'
    assert layout_parameter('M08') == 'snapshot_graph'
    assert layout_parameter('M01') is None and layout_parameter('M30') is None


def propose(model, seed):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    trial = study.ask()
    config = selector.suggest_structure(trial, 5)
    return config, trial.params


@pytest.mark.parametrize('model', list(DESIGNS))
def test_every_proposal_is_a_declared_design(model, monkeypatch):
    monkeypatch.setattr(selector, 'MODELS', (model,))
    for seed in range(6):
        config, params = propose(model, seed)
        assert config['design'] in DESIGNS[config['model']]
        assert set(params) == set(selector.structure(config)[0])


def register(monkeypatch, extra):
    monkeypatch.setattr(spec.REGISTRY, 'modifiers', spec.REGISTRY.modifiers + (extra,))
    monkeypatch.setattr(spec.REGISTRY, 'modifier_by_name',
                        dict(spec.REGISTRY.modifier_by_name, **{extra.name: extra}))


def test_a_new_modifier_extends_designs_without_touching_the_selector(monkeypatch):
    extra = spec.Modifier('extra-links', spec.LIFTED, ('S-full',), 'extra_links')
    assert 'S-full+extra-links' not in spec.designs_for(('S-full',))
    assert extra.parameter not in selector.structure(dict(model='M01', design='S-full'))[0]
    register(monkeypatch, extra)
    assert 'S-full+extra-links' in spec.designs_for(('S-full',))
    values, domains = selector.structure(dict(model='M01', design='S-full+extra-links'))
    assert values[extra.parameter] is True and extra.parameter in domains
    assert selector.structure(dict(model='M01', design='S-full'))[0][extra.parameter] is False
    assert spec.resolve('S-full+extra-links').families[-1] == spec.LIFTED


def test_a_new_modifier_becomes_a_search_dimension(monkeypatch):
    extra = spec.Modifier('extra-links', spec.LIFTED, ('S-full',), 'extra_links')
    monkeypatch.setattr(selector, 'MODELS', ('M01',))
    assert extra.parameter not in propose('M01', 3)[1]
    register(monkeypatch, extra)
    proposed = [propose('M01', seed) for seed in range(10)]
    assert all(extra.parameter in params for _, params in proposed)
    assert {config['design'].endswith('+extra-links') for config, _ in proposed} == {True, False}


def test_a_new_modifier_is_refused_where_its_layouts_do_not_apply(monkeypatch):
    extra = spec.Modifier('extra-links', spec.LIFTED, ('S-full',), 'extra_links')
    register(monkeypatch, extra)
    assert extra.parameter not in selector.structure(dict(model='M13', design='X-chain'))[0]
    assert 'X-chain+extra-links' not in spec.designs_for(bases('M13'))


def test_partial_layout_group_coverage_is_rejected():
    covered = set(spec.LAYOUTS['snapshot_graph'])
    for modifier in spec.MODIFIERS:
        overlap = covered & set(modifier.bases)
        assert not overlap or overlap == covered


def test_declared_extraction_bindings_cover_the_regular_emitters():
    assert {edge.collection for edge in spec.COLLECTIONS} == {
        'unit_cards', 'skills', 'equipped', 'tech', 'rites', 'built'}
    assert {edge.field for edge in spec.FLAGGED} == {
        'at_war', 'allied', 'trade', 'their_vassal', 'nap', 'mil_access'}
    for edge in spec.COLLECTIONS:
        assert len(edge.targets) == 1 and edge.attributes[0] == 'count'
    for edge in spec.FLAGGED:
        assert edge.family == spec.DIPLOMATIC
