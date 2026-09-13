import json
import shutil

import numpy as np
import pytest
import torch

from graph_tuning_infra.database import Stage, connect, copy
from graph_tuning_infra.features import encoder
from totalwar import corpus
from totalwar.build import Counter
from totalwar.domain import CATEGORIES, NUMBERS, RELATIONS, SIGNALS, TARGET, declaration
from totalwar.graph import Campaign
from totalwar.reference import Reference
from totalwar.reward import held, label, turn_open
from tuning.build import build as build_bank
from tuning.data import Bank, pack
from tuning.models import build
from tuning.turns import blocks as turn_blocks

from .support import config


FEATURES = 32


def reference():
    ref = Reference.__new__(Reference)
    families = dict(faction={1: 'cathay', 2: 'norsca'}, region={1: 'north', 2: 'south'}, province={1: 'plain'}, skill={1: 'march', 2: 'shield'},
                    unit={1: 'spear', 2: 'horse'}, building={1: 'barracks_1', 2: 'barracks_2'}, tech_node={1: 'growth_1', 2: 'growth_2'},
                    agent_subtype={1: 'general'}, agent_type={1: 'general'}, ancillary={1: 'axe'}, ritual={}, trait={1: 'brave'}, trait_level={},
                    mission={1: 'quest'}, mission_issuer={}, edict={1: 'tax'}, effect_bundle={1: 'blessing'}, pooled_resource={1: 'gold', 2: 'chaos'},
                    stance={1: 'march', 2: 'none'}, effect={1: 'growth'}, effect_scope={}, dilemma={1: 'plague'}, incident={}, action_type={}, unit_category={})
    ref.keys = families
    ref.ids = {family: {key: identifier for identifier, key in table.items()} for family, table in families.items()}
    ref.enum = {1: 'decision', 2: 'interrupt', 14: 'active', 12: 'inactive', 50: 'pre_battle', 60: 'panel', 62: 'deal', 27: 'declare_war'}
    ref.actions = {1: ('skills', 'march'), 2: ('research', 'growth_2'), 3: ('end_turn', 'end_turn'), 4: ('building', 'barracks_2')}
    ref.factions = dict(cathay=dict(subculture='sc', category='major', is_rebel=False, culture='cu'), norsca=dict(subculture='sn', category='minor', is_rebel=False, culture='cn'))
    ref.tech_sets = dict(cth=dict(key='cth', faction_key='cathay', culture=None, subculture=None))
    ref.tech_nodes = {'growth_1': dict(set='cth', technology_key='growth_1', tier=0, indent=0, research_points_required=100, cost_per_round=0, food_cost=0,
                                       resource_cost=0, required_parents=0, is_civil=True, is_engineering=False, is_military=False),
                      'growth_2': dict(set='cth', technology_key='growth_2', tier=1, indent=0, research_points_required=200, cost_per_round=0, food_cost=0,
                                       resource_cost=0, required_parents=1, is_civil=True, is_engineering=False, is_military=False)}
    ref.tech_links = dict(cth=[dict(parent_key='growth_1', child_key='growth_2', set='cth')])
    ref.tech_effects = dict(growth_1=[dict(effect='growth', value=5.0, effect_scope='faction')])
    ref.tech_buildings = {}
    ref.skill_sets = dict(general={'': 'general_set'})
    ref.skill_nodes = dict(general_set=[dict(key='n1', skill='march', tier=0, indent=0, required_num_parents=0, points_on_creation=0),
                                        dict(key='n2', skill='shield', tier=1, indent=0, required_num_parents=1, points_on_creation=0)])
    ref.skill_links = dict(general_set=[dict(parent_key='n1', child_key='n2', link_type='REQUIRED', initial_descent_tiers=0)])
    ref.skills = dict(march=dict(unlocked_at_rank=1, is_background_skill=False, influence_cost=0), shield=dict(unlocked_at_rank=3, is_background_skill=False, influence_cost=0))
    ref.skill_effects = dict(march=[dict(level=1, effect='growth', value=1.0, effect_scope='character')])
    ref.buildings = dict(barracks_1=dict(chain='barracks', level=1, create_cost=500, create_time=2, upkeep_cost=0, food_cost=0, development_point_cost=0,
                                         only_in_capital=False, can_be_damaged=True, superchain='military', chain_category='military'),
                         barracks_2=dict(chain='barracks', level=2, create_cost=1500, create_time=3, upkeep_cost=0, food_cost=0, development_point_cost=0,
                                         only_in_capital=False, can_be_damaged=True, superchain='military', chain_category='military'))
    ref.building_upgrades = dict(barracks_1=[dict(source='barracks_1', target='barracks_2')])
    ref.building_requires = {}
    ref.building_effects = dict(barracks_2=[dict(effect='growth', value=2.0, effect_scope='region', value_damaged=1.0, value_ruined=0.0)])
    ref.building_units = dict(barracks_1=[dict(building='barracks_1', unit='spear')])
    ref.units = dict(spear=dict(recruitment_cost=400, upkeep_cost=100, create_time=1, campaign_cap=-1, tier=1, num_men=120, is_monstrous=False, can_siege=False,
                                caste='melee', category='inf', class_='inf', morale=50, bonus_hit_points=60, armour='light', shield='none', melee_attack=20,
                                melee_defence=25, charge_bonus=10, accuracy=0, campaign_action_points=2000, num_mounts=0, weapon='spear'),
                     horse=dict(recruitment_cost=800, upkeep_cost=200, create_time=2, campaign_cap=-1, tier=2, num_men=60, is_monstrous=False, can_siege=False,
                                caste='cavalry', category='cav', class_='cav', morale=60, bonus_hit_points=90, armour='medium', shield='none', melee_attack=30,
                                melee_defence=20, charge_bonus=30, accuracy=0, campaign_action_points=2400, num_mounts=1, weapon='lance'))
    ref.items = dict(axe=dict(type='weapon', category='weapon', subcategory='axe', legendary_item=False, transferrable=True, randomly_dropped=True, uniqueness_score=1, can_be_stolen=False))
    ref.item_effects = dict(axe=[dict(effect='growth', value=1.0, effect_scope='character')])
    ref.item_skills = {}
    ref.traits = dict(brave=dict(hidden=False, precedence=1, no_going_back_level=0))
    ref.trait_effects = {}
    ref.effects = dict(growth=dict(category='economy', is_positive_value_good=True))
    ref.bundles = dict(blessing=dict(bundle_target='faction', is_global_effect=False))
    ref.bundle_effects = dict(blessing=[dict(effect='growth', value=3.0, effect_scope='faction')])
    ref.rituals, ref.missions, ref.edicts = {}, dict(quest=dict(mission_type='quest', quest_mission=True, event_category='q')), dict(tax=dict(effect_bundle='blessing'))
    ref.resources = dict(gold=dict(scope='faction'), chaos=dict(scope='region'))
    ref.subtypes = dict(general=dict(is_caster=False, cap=-1, cost=1000, recruitable=True, can_equip_ancillaries=True))
    ref.agents = dict(general=dict(move_points=2000, faction_total_cap=-1))
    ref.regions = dict(north=dict(province='plain', is_capital=True), south=dict(province='plain', is_capital=False))
    ref.dilemmas, ref.incidents, ref.scopes = dict(plague=dict(event_category='d', is_large=False)), {}, dict(faction=0, character=1, region=2)
    ref.agent_actions = dict(general=[dict(unique_id='agent_action_general_assist_army_training', agent='general', ability='assist_army', attribute='zeal',
                                          chance_of_success=80, success='train_ok', failure=None, critical_success=None, critical_failure=None, subculture=None)])
    ref.action_results = dict(train_ok=dict(actor_effect_bundle='blessing', target_effect_bundle=None, actor_effect_bundle_turns=2, target_effect_bundle_turns=None))
    ref.tech_requires = dict(growth_2=[dict(technology='growth_2', required_technology='growth_1')])
    ref.skill_items = dict(shield=[dict(skill='shield', granted_ancillary='axe', level=1)])
    ref.mission_payloads = dict(quest=[dict(mission_key='quest', status_key='SUCCESS', payload_key='EFFECT_BUNDLE', value='KEY[blessing];DURATION[5]'),
                                      dict(mission_key='quest', status_key='SUCCESS', payload_key='TREASURY', value='AMOUNT[800]')])
    return ref


def rows_for(faction_id):
    rows = {name: [] for name in TABLES}
    ticks = (100.0, 101.0, 101.5, 102.0)
    turns = (1, 1, 1, 2)
    kinds = (1, 1, 2, 1)
    settlements = (2, 2, 2, 3)
    for snapshot, (ts, turn, kind, owned) in enumerate(zip(ticks, turns, kinds, settlements), 1):
        rows['snapshots'].append(dict(snapshot_id=snapshot, kind_id=kind, ts=ts, turn=turn, faction_id=faction_id, income=100 + snapshot, settlements=owned,
                                      treasury=5000, is_researching=True, armies=1, lord_level=3, allies=0, vassals=0, power_rank=-5, ll_wounded=False,
                                      defeated=False, difficulty=-1, net_income=None, expenditure=None, upkeep=None, trade_value=None, tax_level=None,
                                      influence=None, food_production=None, food_consumption=None, imperium_level=None, settlement_income=None,
                                      province_income=None, raiding_income=None))
        rows['world'].append(dict(snapshot_id=snapshot, citizenry=[], diplo_unseen=[]))
        rows['armies'].append(dict(snapshot_id=snapshot, ord=0, cqi=11, subtype_id=1, agent_type_id=1, is_leader=True, has_army=True, is_general=True, rank=3,
                                   x=10, y=20, ap_pct=1.0, stance_id=1, hp=0.9, units=2, region_owner_id=faction_id, region_id=1, province_id=1,
                                   in_own_territory=True, ap_remaining=100, ap_per_turn=100, raiding_income=0, upkeep=300))
        rows['hostiles'].append(dict(snapshot_id=snapshot, ord=0, kind_id=7, faction_id=2, visible=True, cqi=99, subtype_id=1, agent_type_id=1, province_id=None,
                                     region_id=2, x=30, y=40, dist=12, is_armed_citizenry=False, units=4, hp=0.5, stance_id=2))
        rows['regions'].append(dict(snapshot_id=snapshot, set_id=1, ord=0, region_id=1, x=10, y=20, province_id=1, owner_id=faction_id, capital=True, abandoned=False, adjacent=[2]))
        rows['regions'].append(dict(snapshot_id=snapshot, set_id=1, ord=1, region_id=2, x=30, y=40, province_id=1, owner_id=2, capital=False, abandoned=False, adjacent=[1]))
        rows['relations'].append(dict(snapshot_id=snapshot, set_id=2, ord=0, faction_id=2, at_war=snapshot > 2, allied=False, trade=snapshot <= 2, their_vassal=False,
                                      standing=-10 * snapshot, excluded=False, mil_ally=False, def_ally=False, nap=False, mil_access=False, our_master=False))
        rows['resources'].append(dict(snapshot_id=snapshot, set_id=3, ord=0, resource_id=1, value=7.0))
        rows['faction_bundles'].append(dict(snapshot_id=snapshot, set_id=4, ord=0, effect_bundle_id=1, turns_remaining=3))
        if kind != 1:
            rows['interrupts'].append(dict(snapshot_id=snapshot, prev_decision_id=snapshot - 1, state_at_id=60, kind_id=50, root='popup_pre_battle', dilemma_id=None,
                                           incident_id=None, root_context=None, region_id=2, chosen='button_autoresolve', answer=None))
            rows['battle_panels'].append(dict(snapshot_id=snapshot, interrupt_id=snapshot, ally_cqi=11, enemy_cqi=99, n_ally_armies=1, n_enemy_armies=1,
                                              result_state='victory', result_text='', casualties_state='light', casualties_text=''))
            continue
        rows['entities'] += [dict(snapshot_id=snapshot, entity_seq=0, kind_id=4, character_id=7, region_id=None, cqi=11),
                             dict(snapshot_id=snapshot, entity_seq=1, kind_id=6, character_id=None, region_id=1, cqi=None),
                             dict(snapshot_id=snapshot, entity_seq=2, kind_id=3, character_id=None, region_id=None, cqi=None)]
        rows['characters'].append(dict(snapshot_id=snapshot, character_id=7, entity_seq=0, is_hero=False, rank=3, skill_points=1, units=2, pending_recruits=0,
                                       ap_pct=1.0, ap_remaining=100, ap_per_turn=100, hp=0.9, loyalty=8, stance_id=1, subtype_id=1, region_id=1, x=10, y=20,
                                       garrisoned=False, besieging=False, acted=False, is_leader=True, wounded=False, is_agent=None, can_embed=None, agent_type_id=1,
                                       skill_set_id=5, hidden_skill_ids=[], stance_set_id=6, recruitable_set_id=7, unit_card_set_id=8, pending_queue_set_id=None,
                                       pending_recruit_unit_ids=[], equipped_set_id=9, horde_slot_set_id=None, merc_pool_set_id=None, reach_chars_true=[99],
                                       reach_setts_true=[2], move_x=[], move_y=[], reach_rays=[1, 2, 3, 4, 5, 6, 7, 8], reach_max=8, skill_prereq_set_id=None, cqi=11))
        rows['extensions'].append(dict(snapshot_id=snapshot, character_id=7, trait_set_id=None, trait_progress_set_id=None, xp=120, xp_next_level=300, subterfuge=1, zeal=1,
                                       authority=1, resurrection_turns=0, upkeep=300, background_skill_id=1, hidden_skill_state_set_id=None, effect_bundle_set_id=None,
                                       force_effect_bundle_set_id=None, armory_item_ids=None, cqi=11))
        level = 1 if snapshot >= 2 else 0
        rows['skills'] += [dict(snapshot_id=snapshot, character_id=7, set_id=5, ord=0, skill_id=1, status_id=14, level=level, total_levels=3, tier=0, rank_required=1),
                           dict(snapshot_id=snapshot, character_id=7, set_id=5, ord=1, skill_id=2, status_id=12, level=0, total_levels=1, tier=1, rank_required=3)]
        rows['stances'].append(dict(snapshot_id=snapshot, character_id=7, set_id=6, ord=0, stance_id=1, active=True, can_activate=True, can_afford=True))
        rows['recruitable'].append(dict(snapshot_id=snapshot, character_id=7, set_id=7, ord=0, unit_id=2, state='ok', cost=800, disabled=False))
        rows['unit_cards'] += [dict(snapshot_id=snapshot, character_id=7, set_id=8, ord=0, unit_id=1, strength_pct=100.0, category_id=1, xp=0),
                               dict(snapshot_id=snapshot, character_id=7, set_id=8, ord=1, unit_id=1, strength_pct=80.0, category_id=1, xp=1)]
        rows['equipped'].append(dict(snapshot_id=snapshot, character_id=7, set_id=9, ord=0, index=1, ancillary_id=1, name='Axe'))
        rows['provinces'].append(dict(snapshot_id=snapshot, region_id=1, entity_seq=1, settlement_present=True, province_id=1, complete_owner=False, max_slots=2,
                                      free_slots=1, can_set_edict=True, selected_edict_id=1, active_edict_id=None, public_order=5, buildings=1, is_capital=True,
                                      settlement_level=2, growth_per_turn=10, gross_income=200, development_points=0, income=150, has_port=False, has_walls=True,
                                      built_set_id=10, locked_slots=[], building_now_set_id=None, corruption_set_id=11, buildable_set_id=12, edict_ids=[1],
                                      slot_state_set_id=13, effect_bundle_set_id=None, plague_bundle_set_id=None, income_breakdown_set_id=None))
        rows['built'].append(dict(snapshot_id=snapshot, region_id=1, set_id=10, ord=0, slot_index=0, building_id=1))
        rows['corruption'].append(dict(snapshot_id=snapshot, region_id=1, set_id=11, ord=0, resource_id=2, value=12.0))
        rows['buildable'].append(dict(snapshot_id=snapshot, region_id=1, set_id=12, ord=0, slot_index=0, building_id=2, active=False, empty=False, can_upgrade=True,
                                      cost=1500, upkeep=0, level=2, can_afford_resources=True))
        rows['slot_states'].append(dict(snapshot_id=snapshot, region_id=1, set_id=13, ord=0, index=0, damaged=False, can_repair=False, repairing=False, can_dismantle=True,
                                        refund=100, queued=False, queued_building_id=None, empty=False, building_id=1, health=5, max_health=5, ruined=False,
                                        repair_cost=0, upgrading=False, dismantling=False))
        rows['states'].append(dict(snapshot_id=snapshot, entity_seq=2, tech_set_id=14, rite_set_id=15, lord_pool_set_id=16, anc_pool_set_id=17, equipped_all_set_id=9,
                                   mission_set_id=18, current_research_id=2 if snapshot > 1 else None, research_points=100))
        rows['tech'] += [dict(snapshot_id=snapshot, set_id=14, ord=0, tech_node_id=1, researched=True, can_research=False, cost=0),
                         dict(snapshot_id=snapshot, set_id=14, ord=1, tech_node_id=2, researched=False, can_research=True, cost=200)]
        rows['lord_pools'].append(dict(snapshot_id=snapshot, set_id=16, ord=0, subtype_id=1, n=1))
        rows['candidates'].append(dict(snapshot_id=snapshot, set_id=16, subtype_id=1, ord=0, can=True, agent=False, bg_skill_id=1, cand_subtype_id=1, trait_ids=[1]))
        rows['missions'].append(dict(snapshot_id=snapshot, set_id=18, ord=0, mission_id=1, status_id=17, turns_remaining=4, is_quest=True, is_victory=False,
                                     completed=False, cancelled=False, pending=False, category='main', issuer_id=None))
        rows['taken'].append(dict(snapshot_id=snapshot, entity_seq=(0, 2, 2)[min(snapshot, 3) - 1], action_id=(1, 2, 3)[min(snapshot, 3) - 1], slot_index=None))
    rows['battles'].append(dict(battle_id=1, campaign_id=faction_id, turn=1, ts=101.2, battle_type='land_normal', autoresolved=True, attacker_result='decisive_victory',
                                defender_result='decisive_defeat', attacker_casualties=10, defender_casualties=200, attacker_kills=200, defender_kills=10,
                                attacker_hp_lost=100, defender_hp_lost=900, attacker_routed=0.0, defender_routed=1.0, siege=False, naval=False, ambush=False,
                                night=False, region_id=2))
    rows['participants'] += [dict(battle_id=1, ord=0, side='attacker', char_cqi=11, mf_cqi=1, faction_id=faction_id),
                             dict(battle_id=1, ord=0, side='defender', char_cqi=99, mf_cqi=2, faction_id=2)]
    rows['battle_units'].append(dict(battle_id=1, ord=0, side='attacker', part_ord=0, unit_id=1, unit_cqi=5))
    rows['diplomacy'].append(dict(event_id=1, campaign_id=faction_id, turn=1, ts=101.9, ts_recorded=101.9, kind_id=62, channel_id=65, faction_id=2, term_ids=[27],
                                  gift_id=None, ok=True, failed_at=None, success_chance=0.9, accepted=True, chosen=None, answer=None, executed=True, confirmed=True,
                                  policy_id=None, proposer=None, speech=None, attitude=None, pair_at_war=None, pair_allied=None, pair_trade=None,
                                  pair_our_master=None, pair_their_vassal=None, pair_standing=None, turns_played=None, ended_by=None, tracked_faction_ids=None))
    rows['events'].append(dict(event_id=1, campaign_id=faction_id, turn=1, ts=101.9, kind='ancillary_gained', incident_id=None, dilemma_id=None, choice=None,
                               faction_id=faction_id, ancillary_id=1, character_cqi=11, region_id=None))
    return rows


TABLES = tuple(corpus.TABLES) + tuple(name for name, _, _, _ in corpus.SETS)


def test_reward_follows_the_future_maximum_and_the_split_hashes_campaigns():
    snapshots = [dict(snapshot_id=1, kind_id=1, turn=1, settlements=2, lord_level=1, allies=0, vassals=0),
                 dict(snapshot_id=2, kind_id=1, turn=1, settlements=3, lord_level=1, allies=0, vassals=0),
                 dict(snapshot_id=3, kind_id=1, turn=2, settlements=4, lord_level=2, allies=1, vassals=1),
                 dict(snapshot_id=4, kind_id=2, turn=2, settlements=9, lord_level=9, allies=9, vassals=9)]
    opened = turn_open(snapshots, 1)
    assert opened == {1: dict(settlements=2, lord_level=1, allies=0, vassals=0), 2: dict(settlements=4, lord_level=2, allies=1, vassals=1)}
    assert label(opened, 1, snapshots[1]) == (4 - 3) + (2 - 1) + (1 - 0) + 3 * (1 - 0)
    assert label(opened, 2, snapshots[2]) == 0.0
    assert label(opened, 3, snapshots[2]) is None
    assert label(opened, 1, dict(settlements=None, lord_level=None, allies=None, vassals=None)) is None
    keys = ['campaign_%d' % index for index in range(400)]
    fraction = sum(held(key) for key in keys) / len(keys)
    assert 0.08 < fraction < 0.22 and held('campaign_0') == held('campaign_0')


def test_a_campaign_builds_a_contract_database_and_a_bank(tmp_path):
    ref = reference()
    encode = encoder(FEATURES // 2, list(NUMBERS), list(CATEGORIES))
    keys = [key for key in ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa') if held(key)][:1]
    keys += [key for key in ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa') if not held(key)][:1]
    assert len(keys) == 2
    counter = Counter()
    stage = Stage(tmp_path / 'staging')
    for index, key in enumerate(keys):
        campaign = dict(campaign_id=index + 1, campaign_key=key, faction='cathay')
        tables = Campaign(ref, encode, FEATURES, campaign, rows_for(1)).build(counter, index, key)
        assert tables[('query', None, None)].num_rows == 4 and tables[(TARGET, None, None)].num_rows == 4
        assert tables[('query', None, None)].column('category').to_pylist() == ['skills', 'research', 'pre_battle', 'end_turn']
        assert set(tables[(TARGET, None, None)].column('split').to_pylist()) == {'validation' if held(key) else 'train'}
        present = {name[0]: table for name, table in tables.items() if name[0].startswith('edge_')}
        assert {'edge_skill', 'edge_tech_state', 'edge_tech_prerequisite', 'edge_grants', 'edge_adjacent', 'edge_holds', 'edge_can_build', 'edge_unit',
                'edge_battle', 'edge_proposed', 'edge_gained_item', 'edge_battle_report', 'edge_trade', 'edge_at_war', 'edge_candidate_trait',
                'edge_can_act', 'edge_action_result', 'edge_mission_payload', 'edge_grants_item'} <= set(present)
        removed = {name: table.column('present').to_pylist().count(False) for name, table in present.items()}
        assert removed['edge_trade'] == 1 and removed['edge_at_war'] == 0
        stage.write(tables)
    stage.close()
    store, connection = connect(tmp_path / 'tw.lbug', read_only=False)
    for statement in declaration(FEATURES, len(SIGNALS)):
        connection.execute(statement)
    copy(connection, stage)
    connection.close()
    store.close()
    (tmp_path / 'tw.task.json').write_text(json.dumps(dict(name='tw', database='tw.lbug', target=TARGET)))
    manifest = build_bank(tmp_path / 'tw.task.json', tmp_path / 'bank', 1)
    assert manifest['rows'] == 8 and manifest['validation_rows'] == 4 and len(manifest['graphs']) == 2
    bank = Bank(tmp_path / 'bank')
    assert set(bank.registry.nodes) == set(name for name in bank.registry.nodes) and bank.registry.extras['queries'] and bank.registry.extras['signals'] == len(SIGNALS)
    for values in ({'view': 'snapshot'}, {'view': 'event_graph'}, {'view': 'sequence', 'sequence/operator': 'mixer'}):
        cfg = config(bank.registry, dict(values, hidden=16), {'time/history': 2})
        block = turn_blocks(bank, [bank.blocks(0, 2, cfg)[-1]], cfg)[0]
        b = pack(bank, block, cfg, 'cuda')
        net = build(cfg, bank.registry, len(bank.categories)).to('cuda')
        output = net(b)
        assert output.shape == b['target'].shape and torch.isfinite(output).all()
    shutil.rmtree(tmp_path / 'bank', ignore_errors=True)
