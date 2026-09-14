import bisect
import logging
import math
import re
import time

import numpy as np
import pyarrow as pa

from graph_tuning_infra.features import token

from .domain import BATTLE, CATEGORIES, NODES, NUMBERS, RELATIONS, SIGNALS, TARGET, TERMS, TREATIES
from .reward import held, label, turn_open


LOG = logging.getLogger(__name__)
DECISION, INTERRUPT = 1, 2
STATUS = {'active': 'active', 'locked_due_to_rank': 'locked_rank', 'locked_by_skill': 'locked_skill', 'locked_by_item': 'locked_item'}
HOSTILE_KINDS = {7: 'army', 8: 'neutral_army', 9: 'settlement', 10: 'hero', 11: 'neutral_hero'}
MERC = {20: 1.0, 21: 2.0, 22: 3.0, 23: 4.0}
GIFTS = {24: 1.0, 25: 2.0, 26: 3.0}
RESULTS = {'decisive_victory': 3.0, 'heroic_victory': 3.0, 'close_victory': 2.0, 'victory': 2.0, 'win': 2.0, 'pyrrhic_victory': 1.0,
           'valiant_defeat': -1.0, 'close_defeat': -2.0, 'defeat': -2.0, 'loss': -2.0, 'decisive_defeat': -3.0, 'crushing_defeat': -3.0}
ALLOWED = frozenset(NUMBERS + CATEGORIES)


def by_snapshot(rows):
    result = {}
    for row in rows:
        result.setdefault(row['snapshot_id'], []).append(row)
    return result


def by_key(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def number(value):
    if value is None:
        return 0.0
    return float(value)


def fixed(values, width):
    values = np.ascontiguousarray(values, np.float32).reshape(-1, width)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


class Campaign:

    def __init__(self, reference, encode, features, campaign, rows):
        self.ref, self.encode, self.features, self.campaign, self.rows = reference, encode, features, campaign, rows
        self.entities, self.kinds, self.keys = {}, [], []
        self.static_features, self.static_facts = {}, {}
        self.tables = {name: by_snapshot(table) for name, table in rows.items() if table and 'snapshot_id' in table[0]}
        self.player = reference.keys['faction'][rows['snapshots'][0]['faction_id']]
        self.snapshots = rows['snapshots']
        self.decision_index = {row['snapshot_id']: index for index, row in enumerate(self.snapshots) if row['kind_id'] == DECISION}
        self.ordinal = {row['snapshot_id']: index for index, row in enumerate(self.snapshots)}
        self.times = [row['ts'] for row in self.snapshots]
        self.opened = turn_open(self.snapshots, DECISION)

    def entity(self, kind, key):
        found = self.entities.get((kind, key))
        if found is None:
            found = len(self.kinds)
            self.entities[(kind, key)] = found
            self.kinds.append(kind)
            self.keys.append(str(key))
        return found

    def key(self, family, identifier):
        return self.ref.keys[family].get(identifier) if identifier is not None else None

    def vector(self, kind, data):
        unknown = set(data) - ALLOWED
        if unknown:
            raise ValueError('undeclared features for %s: %s' % (kind, ', '.join(sorted(unknown))))
        return self.encode(kind, data)

    def step_at(self, ts):
        return min(bisect.bisect_left(self.times, ts), len(self.snapshots) - 1)

    def static(self, features, facts, kind, key, data):
        node = self.entity(kind, key)
        if node not in self.static_features:
            self.static_features[node] = dict(data, key=key)
        return node

    def catalogue(self):
        started = time.perf_counter()
        ref, features, facts = self.ref, self.static_features, self.static_facts

        def add(relation, source, target, *values):
            numbers = tuple(number(value) for value in values)
            facts.setdefault(relation, {})[(source, target, numbers)] = numbers

        def effect(key):
            row = ref.effects.get(key)
            return self.static(features, facts, 'effect', key, dict(category=row['category'], is_positive_good=row['is_positive_value_good']) if row else {})

        def grants(source, granted):
            for row in granted:
                add('grants', source, effect(row['effect']), row['value'], ref.scope(row['effect_scope']), row.get('level'), row.get('value_damaged'), row.get('value_ruined'))

        techs = set()
        for row in self.rows['tech']:
            techs.add(self.key('tech_node', row['tech_node_id']))
        for row in self.rows['states']:
            techs.add(self.key('tech_node', row['current_research_id']))
        techs.discard(None)
        chosen = ref.tech_set(self.player)
        techs |= {key for key, node in ref.tech_nodes.items() if node['set'] == chosen}
        for key in sorted(techs):
            node = ref.tech_nodes.get(key, {})
            tech = self.static(features, facts, 'tech', key, {name: node.get(name) for name in ('tier', 'indent', 'research_points_required', 'cost_per_round',
                                                                                              'food_cost', 'resource_cost', 'required_parents', 'is_civil', 'is_engineering', 'is_military')})
            grants(tech, ref.tech_effects.get(node.get('technology_key'), ()))
            for row in ref.tech_buildings.get(node.get('technology_key'), ()):
                if row['required_building_level'] in ref.buildings:
                    add('requires_building', tech, self.building(row['required_building_level']))
        for link in ref.tech_links.get(chosen, ()):
            if link['parent_key'] in techs and link['child_key'] in techs:
                add('tech_prerequisite', self.entity('tech', link['parent_key']), self.entity('tech', link['child_key']))
        by_technology = {ref.tech_nodes[key]['technology_key']: key for key in sorted(techs) if key in ref.tech_nodes}
        for key in sorted(techs):
            for row in ref.tech_requires.get(ref.tech_nodes.get(key, {}).get('technology_key'), ()):
                parent = by_technology.get(row['required_technology'])
                if parent:
                    add('tech_prerequisite', self.entity('tech', parent), self.entity('tech', key))
        subtypes = {self.key('agent_subtype', row['subtype_id']) for row in self.rows['characters'] + self.rows['armies']}
        subtypes |= {self.key('agent_subtype', row['subtype_id']) for row in self.rows['lord_pools']}
        subtypes |= {self.key('agent_subtype', row['cand_subtype_id']) for row in self.rows['candidates']}
        subtypes.discard(None)
        for subtype in sorted(subtypes):
            row = ref.subtypes.get(subtype, {})
            node = self.static(features, facts, 'subtype', subtype, dict(is_caster=row.get('is_caster'), cap=row.get('cap'), cost=row.get('cost'),
                                                                        recruitable=row.get('recruitable'), can_equip=row.get('can_equip_ancillaries')))
            chosen_set = ref.skill_set(subtype, self.player)
            nodes = {item['key']: item for item in ref.skill_nodes.get(chosen_set, ())}
            for item in nodes.values():
                skill = self.skill(item['skill'])
                add('permits_skill', node, skill, item['tier'], item['indent'], item['required_num_parents'])
            for link in ref.skill_links.get(chosen_set, ()):
                parent, child = nodes.get(link['parent_key']), nodes.get(link['child_key'])
                if parent and child:
                    add('skill_link', self.skill(parent['skill']), self.skill(child['skill']), link['link_type'] == 'SUBSET_REQUIRED', link['initial_descent_tiers'])
        for row in self.rows['skills'] + self.rows['hidden_skills'] + self.rows['skill_prereqs']:
            self.skill(self.key('skill', row['skill_id']))
        for row in self.rows['skill_prereqs']:
            self.skill(self.key('skill', row['parent_skill_id']))
        for row in self.rows['candidates']:
            self.skill(self.key('skill', row['bg_skill_id']))
            for trait in row['trait_ids'] or ():
                self.trait(self.key('trait', trait))
        for row in self.rows['extensions']:
            self.skill(self.key('skill', row['background_skill_id']))
        for row in self.rows['traits'] + self.rows['trait_progress']:
            self.trait(self.key('trait', row['trait_id']))
        for skill in [key for kind, key in self.entities if kind == 'skill']:
            grants(self.entity('skill', skill), ref.skill_effects.get(skill, ()))
            for row in ref.skill_items.get(skill, ()):
                if row['granted_ancillary'] in ref.items:
                    add('grants_item', self.entity('skill', skill), self.item(row['granted_ancillary']), row['level'])
        subculture = ref.factions.get(self.player, {}).get('subculture')
        agents = {self.key('agent_type', row['agent_type_id']) for row in self.rows['characters'] + self.rows['armies']}
        agents.discard(None)
        for agent in agents:
            for row in ref.agent_actions.get(agent, ()):
                if row['subculture'] and row['subculture'] != subculture:
                    continue
                action = self.static(features, facts, 'action', self.action_name(row), dict(ability=row['ability'], attribute=row['attribute'], chance=row['chance_of_success']))
                for outcome, success in (('success', 1), ('critical_success', 1), ('failure', 0), ('critical_failure', 0)):
                    result = ref.action_results.get(row[outcome])
                    if not result:
                        continue
                    for side, actor in (('actor', 1), ('target', 0)):
                        bundle = result[side + '_effect_bundle']
                        if bundle:
                            add('action_result', action, self.bundle(bundle), success, actor, result[side + '_effect_bundle_turns'])
        for trait in [key for kind, key in self.entities if kind == 'trait']:
            grants(self.entity('trait', trait), ref.trait_effects.get(trait, ()))
        buildings = set()
        for name in ('built', 'buildable', 'building_now', 'horde_slots'):
            buildings |= {self.key('building', row['building_id']) for row in self.rows[name]}
        buildings |= {self.key('building', row['queued_building_id']) for row in self.rows['slot_states']}
        buildings |= {self.key('building', row['building_id']) for row in self.rows['slot_states']}
        buildings.discard(None)
        frontier = sorted(buildings)
        while frontier:
            key = frontier.pop()
            for row in list(ref.building_upgrades.get(key, ())) + list(ref.building_requires.get(key, ())):
                target = row.get('target') or row.get('required')
                if target in ref.buildings and target not in buildings:
                    buildings.add(target)
                    frontier.append(target)
        for key in sorted(buildings):
            building = self.building(key)
            grants(building, ref.building_effects.get(key, ()))
            for row in ref.building_upgrades.get(key, ()):
                if row['target'] in buildings:
                    add('upgrades_to', building, self.building(row['target']))
            for row in ref.building_requires.get(key, ()):
                if row['required'] in buildings:
                    add('requires_building', building, self.building(row['required']))
            for row in ref.building_units.get(key, ()):
                if row['unit'] in ref.units:
                    add('unlocks_unit', building, self.unit(row['unit']))
        for name in ('unit_cards', 'recruitable', 'pending_queue', 'merc_pools', 'faction_mercenaries', 'battle_units'):
            for row in self.rows[name]:
                if row['unit_id'] is not None:
                    self.unit(self.key('unit', row['unit_id']))
        for name in ('equipped', 'item_pool'):
            for row in self.rows[name]:
                if row['ancillary_id'] is not None:
                    self.item(self.key('ancillary', row['ancillary_id']))
        for row in self.rows['events']:
            if row['ancillary_id'] is not None:
                self.item(self.key('ancillary', row['ancillary_id']))
        for item in [key for kind, key in self.entities if kind == 'item']:
            node = self.entity('item', item)
            grants(node, ref.item_effects.get(item, ()))
            for row in ref.item_skills.get(item, ()):
                add('requires_skill', node, self.skill(row['required_skill']), row['required_skill_level'])
        for name in ('faction_bundles', 'character_bundles', 'force_bundles', 'province_bundles', 'plague_bundles'):
            for row in self.rows[name]:
                self.bundle(self.key('effect_bundle', row['effect_bundle_id']))
        for row in self.rows['faction_effects']:
            self.bundle(self.key('effect_bundle', row['effect_bundle_id']))
        for bundle in [key for kind, key in self.entities if kind == 'bundle']:
            grants(self.entity('bundle', bundle), ref.bundle_effects.get(bundle, ()))
        adjacency = {}
        for row in self.rows['regions']:
            adjacency.setdefault(row['region_id'], set()).update(row['adjacent'] or ())
        for region, neighbours in adjacency.items():
            source = self.region(region)
            for neighbour in sorted(neighbours):
                add('adjacent', source, self.region(neighbour))
        LOG.info('catalogue exit campaign=%d seconds=%.3f entities=%d static_facts=%d', self.campaign['campaign_id'], time.perf_counter() - started,
                 len(self.kinds), sum(len(items) for items in facts.values()))

    def skill(self, key):
        row = self.ref.skills.get(key, {})
        return self.static(self.static_features, self.static_facts, 'skill', key,
                           dict(unlocked_at_rank=row.get('unlocked_at_rank'), is_background=row.get('is_background_skill'), influence_cost=row.get('influence_cost')))

    def trait(self, key):
        row = self.ref.traits.get(key, {})
        return self.static(self.static_features, self.static_facts, 'trait', key,
                           dict(hidden=row.get('hidden'), precedence=row.get('precedence'), no_going_back_level=row.get('no_going_back_level')))

    def building(self, key):
        row = self.ref.buildings.get(key, {})
        return self.static(self.static_features, self.static_facts, 'building', key, dict(
            level=row.get('level'), create_cost=row.get('create_cost'), create_time=row.get('create_time'), upkeep_cost=row.get('upkeep_cost'),
            food_cost=row.get('food_cost'), development_point_cost=row.get('development_point_cost'), only_in_capital=row.get('only_in_capital'),
            can_be_damaged=row.get('can_be_damaged'), chain=row.get('chain'), superchain=row.get('superchain'), category=row.get('chain_category')))

    def unit(self, key):
        row = self.ref.units.get(key, {})
        return self.static(self.static_features, self.static_facts, 'unit', key, {name: row.get(name) for name in (
            'recruitment_cost', 'upkeep_cost', 'create_time', 'campaign_cap', 'tier', 'num_men', 'is_monstrous', 'can_siege', 'caste', 'category',
            'class', 'morale', 'bonus_hit_points', 'armour', 'shield', 'melee_attack', 'melee_defence', 'charge_bonus', 'accuracy',
            'campaign_action_points', 'num_mounts', 'weapon')})

    def item(self, key):
        row = self.ref.items.get(key, {})
        return self.static(self.static_features, self.static_facts, 'item', key, dict(
            type=row.get('type'), category=row.get('category'), subcategory=row.get('subcategory'), legendary=row.get('legendary_item'),
            transferrable=row.get('transferrable'), randomly_dropped=row.get('randomly_dropped'), uniqueness_score=row.get('uniqueness_score'),
            can_be_stolen=row.get('can_be_stolen')))

    def bundle(self, key):
        row = self.ref.bundles.get(key, {})
        return self.static(self.static_features, self.static_facts, 'bundle', key, dict(bundle_target=row.get('bundle_target'), is_global=row.get('is_global_effect')))

    def region(self, identifier):
        key = self.key('region', identifier)
        row = self.ref.regions.get(key, {})
        node = self.static(self.static_features, self.static_facts, 'region', key, dict(capital=row.get('is_capital')))
        if row.get('province'):
            province = self.static(self.static_features, self.static_facts, 'province', row['province'], {})
            self.static_facts.setdefault('in_province', {})[(node, province, None)] = ()
        return node

    def faction(self, key):
        row = self.ref.factions.get(key, {})
        return self.static(self.static_features, self.static_facts, 'faction', key, dict(
            faction=key, culture=row.get('culture'), subculture=row.get('subculture'), category=row.get('category'), is_rebel=row.get('is_rebel')))

    def character(self, cqi):
        return self.entity('character', '%s:%s' % (self.campaign['campaign_key'], cqi))

    def slot(self, owner, index):
        return self.entity('slot', '%s#%s' % (owner, 'new' if index is None else index))

    def candidate(self, subtype, index):
        return self.entity('candidate', '%s:%s@%s' % (self.campaign['campaign_key'], subtype, index))

    def world(self, snapshot, features, facts):
        ref, keys = self.ref, self.ref.keys
        player = self.faction(self.player)
        at = lambda name: self.tables.get(name, {}).get(snapshot['snapshot_id'], ())

        def fact(relation, source, target, *values, tag=None):
            facts.setdefault(relation, {})[(source, target, tag)] = tuple(number(value) for value in values)

        scalars = {name: snapshot.get(name) for name in ('turn', 'income', 'settlements', 'treasury', 'is_researching', 'armies', 'lord_level', 'allies', 'vassals',
                                                         'power_rank', 'll_wounded', 'defeated', 'difficulty', 'net_income', 'expenditure', 'upkeep', 'trade_value',
                                                         'tax_level', 'influence', 'food_production', 'food_consumption', 'imperium_level', 'settlement_income',
                                                         'province_income', 'raiding_income')}
        scalars['hero_count'] = sum(row['n'] or 0 for row in at('hero_counts'))
        features[player] = dict(self.static_features[player], **scalars)
        for row in at('resources'):
            fact('resource', player, self.entity('resource', keys['pooled_resource'][row['resource_id']]), row['value'])
        for row in at('faction_bundles'):
            fact('bundle', player, self.bundle(keys['effect_bundle'][row['effect_bundle_id']]), row['turns_remaining'], 0, 0)
        for row in at('faction_mercenaries'):
            fact('faction_mercenary', player, self.unit(keys['unit'][row['unit_id']]), row['base_max'], row['current_max'])
        for row in at('regions'):
            region = self.region(row['region_id'])
            features[region] = dict(self.static_features[region], x=row['x'], y=row['y'], capital=row['capital'], abandoned=row['abandoned'])
            if row['owner_id'] is not None:
                fact('owns', self.faction(keys['faction'][row['owner_id']]), region)
        for row in at('settlements'):
            region = self.region(row['region_id'])
            features.setdefault(region, dict(self.static_features[region])).update(garrison=row['units'], x=row['x'], y=row['y'])
        for row in at('ruins'):
            region = self.region(row['region_id'])
            features.setdefault(region, dict(self.static_features[region])).update(ruin=1, x=row['x'], y=row['y'])
        for row in at('armies'):
            if row['cqi'] is None:
                continue
            character = self.character(row['cqi'])
            data = {name: row.get(name) for name in ('is_leader', 'has_army', 'is_general', 'rank', 'x', 'y', 'ap_pct', 'hp', 'units', 'in_own_territory',
                                                     'ap_remaining', 'ap_per_turn', 'raiding_income', 'upkeep')}
            data.update(subtype=self.key('agent_subtype', row['subtype_id']), agent_type=self.key('agent_type', row['agent_type_id']), stance=self.key('stance', row['stance_id']))
            features.setdefault(character, {}).update({name: value for name, value in data.items() if value is not None})
            fact('controls', player, character)
            if row['region_id'] is not None:
                fact('located', character, self.region(row['region_id']))
            if row['stance_id'] is not None:
                fact('in_stance', character, self.entity('stance', keys['stance'][row['stance_id']]))
        for row in at('hostiles'):
            faction = self.faction(keys['faction'][row['faction_id']])
            features.setdefault(faction, dict(self.static_features[faction]))
            if row['kind_id'] == 9 or row['cqi'] is None:
                if row['region_id'] is not None:
                    region = self.region(row['region_id'])
                    features.setdefault(region, dict(self.static_features[region])).update(garrison=row['units'], x=row['x'], y=row['y'])
                    fact('owns', faction, region)
                continue
            character = self.character(row['cqi'])
            data = dict(category=HOSTILE_KINDS.get(row['kind_id']), faction=keys['faction'][row['faction_id']], visible=row['visible'], x=row['x'], y=row['y'],
                        dist=row['dist'], armed_citizenry=row['is_armed_citizenry'], units=row['units'], hp=row['hp'],
                        subtype=self.key('agent_subtype', row['subtype_id']), agent_type=self.key('agent_type', row['agent_type_id']), stance=self.key('stance', row['stance_id']))
            features.setdefault(character, {}).update({name: value for name, value in data.items() if value is not None})
            fact('controls', faction, character)
            if row['region_id'] is not None:
                fact('located', character, self.region(row['region_id']))
        for row in at('enemy_agents'):
            faction = self.faction(keys['faction'][row['faction_id']])
            character = self.character(row['cqi'])
            features.setdefault(character, {}).update(x=row['x'], y=row['y'], at_war=row['at_war'], faction=keys['faction'][row['faction_id']], category='agent')
            features.setdefault(faction, dict(self.static_features[faction]))
            fact('controls', faction, character)
        for row in at('relations'):
            other = self.faction(keys['faction'][row['faction_id']])
            features.setdefault(other, dict(self.static_features[other]))
            fact('standing', player, other, row['standing'])
            for column, relation in TREATIES.items():
                if row[column]:
                    fact(relation, player, other)
        for row in at('war_graph'):
            source = self.faction(keys['faction'][row['faction_id']])
            features.setdefault(source, dict(self.static_features[source]))
            for enemy in row['at_war_with'] or ():
                target = self.faction(keys['faction'][enemy])
                features.setdefault(target, dict(self.static_features[target]))
                fact('war', source, target)
        for row in at('stationed'):
            if row['cqi'] is not None:
                fact('stationed', self.character(row['cqi']), self.region(row['region_id']))
        for row in at('world'):
            for cqi in row['citizenry'] or ():
                fact('citizenry', player, self.character(cqi))

    def entities_block(self, snapshot, features, facts):
        ref, keys = self.ref, self.ref.keys
        player = self.faction(self.player)
        at = lambda name: self.tables.get(name, {}).get(snapshot['snapshot_id'], ())

        def fact(relation, source, target, *values, tag=None):
            facts.setdefault(relation, {})[(source, target, tag)] = tuple(number(value) for value in values)

        characters = {}
        for row in at('characters'):
            character = self.character(row['cqi'])
            characters[row['character_id']] = character
            data = {name: row.get(name) for name in ('is_hero', 'rank', 'skill_points', 'units', 'pending_recruits', 'ap_pct', 'ap_remaining', 'ap_per_turn', 'hp',
                                                     'loyalty', 'x', 'y', 'garrisoned', 'besieging', 'acted', 'is_leader', 'wounded', 'is_agent', 'can_embed', 'reach_max')}
            data.update(subtype=self.key('agent_subtype', row['subtype_id']), agent_type=self.key('agent_type', row['agent_type_id']), stance=self.key('stance', row['stance_id']))
            for index, ray in enumerate(row['reach_rays'] or ()):
                data['ray%d' % index] = ray
            features.setdefault(character, {}).update({name: value for name, value in data.items() if value is not None})
            if row['subtype_id'] is not None:
                fact('of_subtype', character, self.entity('subtype', keys['agent_subtype'][row['subtype_id']]))
            for region in row['reach_setts_true'] or ():
                fact('can_reach', character, self.region(region))
            for other in row['reach_chars_true'] or ():
                fact('can_reach_character', character, self.character(other))
            for position, unit in enumerate(row['pending_recruit_unit_ids'] or ()):
                fact('recruiting', character, self.unit(keys['unit'][unit]), None, position, tag=('pending', position))
            for skill in row['hidden_skill_ids'] or ():
                fact('hidden_skill', character, self.skill(keys['skill'][skill]), None, None)
            agent = self.key('agent_type', row['agent_type_id'])
            for action in ref.agent_actions.get(agent, ()) if agent else ():
                name = self.action_name(action)
                if ('action', name) in self.entities:
                    fact('can_act', character, self.entity('action', name), action['chance_of_success'], tag=name)
        for row in at('extensions'):
            character = characters[row['character_id']]
            data = {name: row.get(name) for name in ('xp', 'xp_next_level', 'subterfuge', 'zeal', 'authority', 'resurrection_turns', 'upkeep')}
            features[character].update({name: value for name, value in data.items() if value is not None})
            if row['background_skill_id'] is not None:
                fact('background', character, self.skill(keys['skill'][row['background_skill_id']]))
            for item in row['armory_item_ids'] or ():
                fact('armory', character, self.item(str(item)))
        collections = (('skills', 'skill'), ('hidden_skills', 'hidden_skill'), ('traits', 'trait'), ('trait_progress', 'trait_progress'), ('equipped', 'equipped'),
                       ('stances', 'stance_option'), ('recruitable', 'recruitable'), ('unit_cards', 'unit'), ('pending_queue', 'recruiting'),
                       ('merc_pools', 'mercenary'), ('skill_prereqs', 'skill_prerequisite'), ('character_bundles', 'bundle'), ('force_bundles', 'bundle'),
                       ('horde_slots', 'holds'))
        for name, relation in collections:
            for row in at(name):
                character = characters.get(row['character_id'])
                if character is None:
                    continue
                if name == 'skills':
                    status = ref.enum.get(row['status_id'])
                    flags = {value: 1.0 for key, value in STATUS.items() if key == status}
                    fact(relation, character, self.skill(keys['skill'][row['skill_id']]), row['level'], row['total_levels'], row['tier'], row['rank_required'],
                         flags.get('active', 0.0), flags.get('locked_rank', 0.0), flags.get('locked_skill', 0.0), flags.get('locked_item', 0.0))
                elif name == 'hidden_skills':
                    fact(relation, character, self.skill(keys['skill'][row['skill_id']]), row['level'], row['total_levels'])
                elif name == 'traits':
                    fact(relation, character, self.trait(keys['trait'][row['trait_id']]), row['level'], row['points'], row['threshold_points'], row['chaos_realm'])
                elif name == 'trait_progress':
                    fact(relation, character, self.trait(keys['trait'][row['trait_id']]), row['points'])
                elif name == 'equipped':
                    if row['ancillary_id'] is not None:
                        fact(relation, character, self.item(keys['ancillary'][row['ancillary_id']]), row['index'], tag=row['index'])
                elif name == 'stances':
                    fact(relation, character, self.entity('stance', keys['stance'][row['stance_id']]), row['active'], row['can_activate'], row['can_afford'])
                elif name == 'recruitable':
                    fact(relation, character, self.unit(keys['unit'][row['unit_id']]), row['cost'], row['disabled'])
                elif name == 'unit_cards':
                    fact(relation, character, self.unit(keys['unit'][row['unit_id']]), row['strength_pct'], row['xp'], row['ord'], tag=row['ord'])
                elif name == 'pending_queue':
                    fact(relation, character, self.unit(keys['unit'][row['unit_id']]), row['turns_left'], row['ord'], tag=('queue', row['ord']))
                elif name == 'merc_pools':
                    fact(relation, character, self.unit(keys['unit'][row['unit_id']]), row['avail'], row['cost'], row['can'], MERC.get(row['action_id']), tag=row['action_id'])
                elif name == 'skill_prereqs':
                    fact(relation, self.skill(keys['skill'][row['parent_skill_id']]), self.skill(keys['skill'][row['skill_id']]))
                elif name in ('character_bundles', 'force_bundles'):
                    fact(relation, character, self.bundle(keys['effect_bundle'][row['effect_bundle_id']]), row['turns_remaining'], name == 'force_bundles', 0, tag=name)
                else:
                    slot = self.slot('%s:%s' % (self.campaign['campaign_key'], row['character_id']), row['slot_index'])
                    features[slot] = dict(index=row['slot_index'], empty=row['empty'])
                    fact('has_slot', character, slot)
                    if not row['empty'] and row['building_id'] is not None:
                        fact(relation, slot, self.building(keys['building'][row['building_id']]), None, None, None, None, row['available'])
        provinces = {}
        for row in at('provinces'):
            region = self.region(row['region_id'])
            provinces[row['region_id']] = region
            data = {name: row.get(name) for name in ('settlement_present', 'complete_owner', 'max_slots', 'free_slots', 'can_set_edict', 'public_order', 'buildings',
                                                     'settlement_level', 'growth_per_turn', 'gross_income', 'development_points', 'income', 'has_port', 'has_walls')}
            data['capital'] = row['is_capital']
            features.setdefault(region, dict(self.static_features[region])).update({name: value for name, value in data.items() if value is not None})
            fact('owns', player, region)
            locked = set(row['locked_slots'] or ())
            province = self.ref.regions.get(keys['region'][row['region_id']], {}).get('province')
            if province:
                target = self.entity('province', province)
                features.setdefault(target, {})
                for edict in row['edict_ids'] or ():
                    fact('edict_option', target, self.entity('edict', keys['edict'][edict]), edict == row['selected_edict_id'], edict == row['active_edict_id'], row['can_set_edict'])
            for index in range(int(row['max_slots'] or 0)):
                slot = self.slot(keys['region'][row['region_id']], index)
                features[slot] = dict(index=index, locked=index in locked)
                fact('has_slot', region, slot)
        for row in at('slot_states'):
            slot = self.slot(keys['region'][row['region_id']], row['index'])
            features.setdefault(slot, dict(index=row['index'])).update({name: row[name] for name in ('damaged', 'repairing', 'upgrading', 'dismantling', 'queued', 'refund',
                                                                                                  'repair_cost', 'health', 'max_health', 'ruined', 'empty')})
            fact('has_slot', provinces[row['region_id']], slot)
            if row['building_id'] is not None:
                fact('holds', slot, self.building(keys['building'][row['building_id']]), row['health'], row['max_health'], row['damaged'], row['ruined'], None)
            if row['queued_building_id'] is not None:
                fact('constructing', slot, self.building(keys['building'][row['queued_building_id']]), None, None, 1, tag='queued')
        for row in at('built'):
            slot = self.slot(keys['region'][row['region_id']], row['slot_index'])
            features.setdefault(slot, dict(index=row['slot_index']))
            fact('has_slot', provinces[row['region_id']], slot)
            fact('holds', slot, self.building(keys['building'][row['building_id']]), None, None, None, None, None)
        for row in at('building_now'):
            slot = self.slot(keys['region'][row['region_id']], row['slot_index'])
            features.setdefault(slot, dict(index=row['slot_index']))
            fact('has_slot', provinces[row['region_id']], slot)
            fact('constructing', slot, self.building(keys['building'][row['building_id']]), row['turns_left'], row['paused'], 0, tag='now')
        for row in at('buildable'):
            slot = self.slot(keys['region'][row['region_id']], row['slot_index'])
            features.setdefault(slot, dict(index=row['slot_index']))
            fact('has_slot', provinces[row['region_id']], slot)
            fact('can_build', slot, self.building(keys['building'][row['building_id']]), row['cost'], row['upkeep'], row['level'], not row['empty'], row['can_afford_resources'])
        for row in at('corruption'):
            fact('corruption', provinces[row['region_id']], self.entity('resource', keys['pooled_resource'][row['resource_id']]), row['value'])
        for row in at('income_breakdown'):
            fact('income', provinces[row['region_id']], self.entity('resource', 'income:' + row['label']), row['amount'])
        for name, plague in (('province_bundles', 0), ('plague_bundles', 1)):
            for row in at(name):
                fact('bundle', provinces[row['region_id']], self.bundle(keys['effect_bundle'][row['effect_bundle_id']]), row['turns_remaining'], 0, plague, tag=name)
        for row in at('states'):
            features.setdefault(player, {})['research_points'] = row['research_points']
            if row['current_research_id'] is not None:
                fact('researching', player, self.entity('tech', keys['tech_node'][row['current_research_id']]), row['research_points'])
        for row in at('tech'):
            fact('tech_state', player, self.entity('tech', keys['tech_node'][row['tech_node_id']]), row['researched'], row['can_research'], row['cost'])
        for row in at('rites'):
            rite = self.entity('rite', keys['ritual'][row['ritual_id']])
            ritual = ref.rituals.get(keys['ritual'][row['ritual_id']], {})
            features[rite] = dict(category=ritual.get('category'), cast_time=ritual.get('cast_time'), cooldown_time=ritual.get('cooldown_time'),
                                  influence_cost=ritual.get('influence_cost'), slave_cost=ritual.get('slave_cost'))
            fact('rite', player, rite, row['can_perform'], row['reason_id'] or 0)
        for row in at('lord_pools'):
            fact('lord_pool', player, self.entity('subtype', keys['agent_subtype'][row['subtype_id']]), row['n'])
        for row in at('candidates'):
            subtype = keys['agent_subtype'][row['subtype_id']]
            candidate = self.candidate(subtype, row['ord'])
            features[candidate] = dict(candidate_index=row['ord'], is_agent=row['agent'], subtype=self.key('agent_subtype', row['cand_subtype_id']))
            fact('candidate_of', candidate, self.entity('subtype', subtype), row['can'], row['agent'])
            if row['bg_skill_id'] is not None:
                fact('candidate_skill', candidate, self.skill(keys['skill'][row['bg_skill_id']]))
            for trait in row['trait_ids'] or ():
                fact('candidate_trait', candidate, self.trait(keys['trait'][trait]))
        for row in at('item_pool'):
            if row['ancillary_id'] is not None:
                fact('item_pool', player, self.item(keys['ancillary'][row['ancillary_id']]), row['index'], tag=row['index'])
        for row in at('missions'):
            mission = self.entity('mission', keys['mission'][row['mission_id']])
            info = ref.missions.get(keys['mission'][row['mission_id']], {})
            features[mission] = dict(mission_type=info.get('mission_type'), event_category=info.get('event_category'), is_quest=info.get('quest_mission'),
                                     issuer=self.key('mission_issuer', row['issuer_id']), category=row['category'])
            fact('mission', player, mission, row['status_id'], row['turns_remaining'], row['is_quest'], row['is_victory'], row['completed'], row['cancelled'], row['pending'])
            for payload in ref.mission_payloads.get(keys['mission'][row['mission_id']], ()):
                self.payload(mission, payload, features[mission], fact)

    @staticmethod
    def action_name(row):
        prefix = 'agent_action_%s_%s_' % (row['agent'], row['ability'])
        return row['unique_id'].split(prefix, 1)[1] if prefix in row['unique_id'] else row['unique_id']

    def payload(self, mission, payload, data, fact):
        fields = dict(re.findall(r'([A-Z_]+)\[([^\]]*)\]', payload['value'] or ''))
        success = payload['status_key'] == 'SUCCESS'
        kind, amount = payload['payload_key'], fields.get('AMOUNT')
        if kind == 'EFFECT_BUNDLE' and fields.get('KEY') in self.ref.bundles:
            fact('mission_payload', mission, self.bundle(fields['KEY']), fields.get('DURATION'), success, tag=('bundle', fields['KEY']))
        elif kind == 'ADD_ANCILLARY' and fields.get('ANCILLARY_KEY') in self.ref.items:
            fact('mission_payload', mission, self.item(fields['ANCILLARY_KEY']), None, success, tag=('item', fields['ANCILLARY_KEY']))
        elif kind in ('FACTION_POOLED_RESOURCE_TRANSACTION', 'FACTION_PROVINCE_POOLED_RESOURCE_TRANSACTION') and fields.get('RESOURCE'):
            fact('mission_payload', mission, self.entity('resource', fields['RESOURCE']), amount, success, tag=('resource', fields['RESOURCE']))
        elif kind == 'GIVE_TRAIT' and fields.get('TRAIT_LEVEL_KEY'):
            trait = next((key for key in self.ref.traits if fields['TRAIT_LEVEL_KEY'].startswith(key)), None)
            if trait:
                fact('mission_payload', mission, self.trait(trait), None, success, tag=('trait', trait))
        elif kind == 'TREASURY' and amount:
            data['treasury'] = float(amount)
        elif kind == 'DIPLOMATIC_STANDING' and amount:
            data['standing'] = float(amount)
        elif kind == 'INFLUENCE' and amount:
            data['influence'] = float(amount)

    def consequences(self):
        keys, placed = self.ref.keys, {}

        def fact(index, relation, source, target, *values, tag=None):
            placed.setdefault(index, {}).setdefault(relation, {})[(source, target, tag)] = tuple(number(value) for value in values)

        participants = by_key(self.rows['participants'], 'battle_id')
        units = by_key(self.rows['battle_units'], 'battle_id')
        for row in self.rows['battles']:
            index = self.step_at(row['ts'])
            sides = {}
            for part in participants.get(row['battle_id'], ()):
                if part['char_cqi'] is not None:
                    sides.setdefault(part['side'], []).append(self.character(part['char_cqi']))
            for attacker in sides.get('attacker', ()):
                for defender in sides.get('defender', ()):
                    fact(index, 'battle', attacker, defender, RESULTS.get(row['attacker_result'], 0), RESULTS.get(row['defender_result'], 0),
                         *[row[name] for name in BATTLE[2:10]], row['autoresolved'], row['siege'], row['naval'], row['ambush'], row['night'],
                         token(row['battle_type'] or '') % 97, tag=row['battle_id'])
                if row['region_id'] is not None:
                    fact(index, 'battle_at', attacker, self.region(row['region_id']), tag=row['battle_id'])
            for unit in units.get(row['battle_id'], ()):
                members = participants.get(row['battle_id'], ())
                owner = next((part for part in members if part['side'] == unit['side'] and part['ord'] == unit['part_ord']), None)
                if owner and owner['char_cqi'] is not None and unit['unit_id'] is not None:
                    fact(index, 'fought_with', self.character(owner['char_cqi']), self.unit(keys['unit'][unit['unit_id']]), unit['side'] == 'attacker',
                         tag=(row['battle_id'], unit['ord']))
        player = self.faction(self.player)
        for row in self.rows['diplomacy']:
            if self.ref.enum.get(row['kind_id']) != 'deal' or row['faction_id'] is None:
                continue
            index = self.step_at(row['ts'])
            terms = {self.ref.enum.get(term) for term in row['term_ids'] or ()}
            gift = GIFTS.get(row['gift_id'])
            if gift:
                terms.add(('gift_small', 'gift_medium', 'gift_large')[int(gift) - 1])
            fact(index, 'proposed', player, self.faction(keys['faction'][row['faction_id']]), row['accepted'], row['ok'], row['success_chance'], gift, None, None,
                 *[term in terms for term in TERMS], tag=row['event_id'])
        for row in self.rows['events']:
            index = self.step_at(row['ts'])
            if row['kind'] == 'ancillary_gained' and row['ancillary_id'] is not None and row['character_cqi'] is not None:
                fact(index, 'gained_item', self.character(row['character_cqi']), self.item(keys['ancillary'][row['ancillary_id']]), tag=row['event_id'])
            elif row['dilemma_id'] is not None or row['incident_id'] is not None:
                key = keys['dilemma'].get(row['dilemma_id']) or keys['incident'].get(row['incident_id'])
                node = self.entity('dilemma', key)
                self.static_features.setdefault(node, dict(key=key, **{name: (self.ref.dilemmas.get(key) or self.ref.incidents.get(key) or {}).get(name)
                                                                            for name in ('event_category', 'is_large')}))
                fact(index, 'faced', player, node, row['kind'] == 'dilemma_choice_made', tag=row['event_id'])
        return placed

    def query(self, snapshot, index, features, facts):
        keys, player = self.ref.keys, self.faction(self.player)
        data = {}
        if snapshot['kind_id'] == DECISION:
            taken = self.tables.get('taken', {}).get(snapshot['snapshot_id'], ())
            if not taken:
                return 'no_action', player, player, player, dict(action_type='no_action')
            row = taken[0]
            action_type, key = self.ref.actions[row['action_id']]
            entity = next((item for item in self.tables.get('entities', {}).get(snapshot['snapshot_id'], ()) if item['entity_seq'] == row['entity_seq']), None)
            actor, context = player, player
            if entity and entity['cqi'] is not None:
                actor = self.character(entity['cqi'])
                region = next((target for (source, target, _) in facts.get('located', {}) if source == actor), None)
                context = region if region is not None else player
            elif entity and entity['region_id'] is not None:
                actor = self.region(entity['region_id'])
                province = self.ref.regions.get(keys['region'][entity['region_id']], {}).get('province')
                context = self.entity('province', province) if province else player
            data = dict(action_type=action_type, key=key)
            subject = self.subject(action_type, key, actor, row, snapshot, facts, data)
            for node in (actor, subject, context):
                features.setdefault(node, dict(self.static_features.get(node, {})))
            return action_type, actor, subject, context, data
        interrupt = self.tables.get('interrupts', {}).get(snapshot['snapshot_id'], [None])[0]
        kind = self.ref.enum.get(interrupt['kind_id']) if interrupt else 'interrupt'
        data = dict(action_type=kind, option=interrupt['chosen'] if interrupt else None, root=interrupt['root'] if interrupt else None, answer=interrupt['answer'] if interrupt else None)
        actor = subject = context = player
        panel = self.tables.get('battle_panels', {}).get(snapshot['snapshot_id'], [None])[0]
        diplo = self.tables.get('diplo_panels', {}).get(snapshot['snapshot_id'], [None])[0]
        if panel:
            if panel['ally_cqi'] is not None:
                actor = self.character(panel['ally_cqi'])
            if panel['enemy_cqi'] is not None:
                subject = self.character(panel['enemy_cqi'])
            data.update(n_ally_armies=panel['n_ally_armies'], n_enemy_armies=panel['n_enemy_armies'], result=panel['result_state'])
            if actor != player and subject != player:
                facts.setdefault('battle_report', {})[(actor, subject, snapshot['snapshot_id'])] = (
                    number(RESULTS.get(panel['result_state'], 0)), number(token(panel['casualties_state'] or '') % 97))
        if diplo:
            data.update(attitude=diplo['attitude'], race=diplo['race'], amount_offered=diplo['amount_offered'], amount_demanded=diplo['amount_demanded'],
                        settlements_offered=diplo['settlements'])
        if interrupt:
            if interrupt['region_id'] is not None:
                context = self.region(interrupt['region_id'])
                if subject == player:
                    subject = context
            if interrupt['root_context'] in self.ref.ids['faction']:
                subject = self.faction(interrupt['root_context'])
            if interrupt['dilemma_id'] is not None or interrupt['incident_id'] is not None:
                key = keys['dilemma'].get(interrupt['dilemma_id']) or keys['incident'].get(interrupt['incident_id'])
                subject = self.entity('dilemma', key)
                self.static_features.setdefault(subject, dict(key=key))
        for node in (actor, subject, context):
            features.setdefault(node, dict(self.static_features.get(node, {})))
        return kind, actor, subject, context, data

    def subject(self, action_type, key, actor, taken, snapshot, facts, data):
        keys, ids, player = self.ref.keys, self.ref.ids, self.faction(self.player)
        located = {source: target for (source, target, _) in facts.get('located', {})}
        if action_type == 'skills' and key in ids['skill']:
            return self.skill(key)
        if action_type == 'stance' and key in ids['stance']:
            return self.entity('stance', key)
        if action_type in ('recruit_unit', 'raise_dead', 'recruit_ror', 'recruit_blessed', 'recruit_imperial', 'cancel_recruit'):
            unit = key.split('@')[0]
            if '@' in key:
                data['queue_index'] = int(key.split('@')[1])
            return self.unit(unit) if unit in self.ref.units else actor
        if action_type == 'attack_army' or (action_type == 'hero_action' and '@cqi:' in key):
            data['action'] = key.split('@')[0] if action_type == 'hero_action' else None
            target = self.character(int(key.split('cqi:')[1]))
            return target
        if action_type == 'hero_action':
            data['action'] = key.split('@')[0]
            region = key.split('@')[1] if '@' in key else None
            return self.region(ids['region'][region]) if region in ids['region'] else actor
        if action_type in ('attack_settlement', 'colonize'):
            return self.region(ids['region'][key]) if key in ids['region'] else actor
        if action_type == 'garrison':
            region = key.split(':')[-1]
            return self.region(ids['region'][region]) if region in ids['region'] else actor
        if action_type in ('leave_garrison', 'move'):
            if action_type == 'move' and key.startswith('xy:'):
                x, y = key[3:].split(',')
                data.update(x=float(x), y=float(y))
            return located.get(actor, actor)
        if action_type in ('items', 'item_unequip'):
            return self.item(key) if key in self.ref.items else actor
        if action_type == 'horde_building':
            building = key.split('@')[-1]
            return self.building(building) if building in self.ref.buildings else actor
        if action_type == 'building':
            data['slot_index'] = taken['slot_index']
            return self.building(key) if key in self.ref.buildings else actor
        if action_type in ('building_repair', 'building_dismantle'):
            region, _, slot = key.partition('@')
            data['slot_index'] = int(slot) if slot.isdigit() else None
            return self.slot(region, int(slot)) if slot.isdigit() and ('slot', '%s#%s' % (region, slot)) in self.entities else actor
        if action_type == 'edict':
            return self.entity('edict', key) if key in ids['edict'] else actor
        if action_type in ('recruit_lord', 'recruit_hero'):
            subtype, _, index = key.partition('@')
            data['candidate_index'] = int(index) if index.isdigit() else None
            candidate = ('candidate', '%s:%s@%s' % (self.campaign['campaign_key'], subtype, index))
            return self.entities[candidate] if candidate in self.entities else (self.entity('subtype', subtype) if subtype in ids['agent_subtype'] else actor)
        if action_type == 'research':
            if key in ids['tech_node']:
                return self.entity('tech', key)
            node = next((name for name, item in self.ref.tech_nodes.items() if item['technology_key'] == key and ('tech', name) in self.entities), None)
            return self.entity('tech', node) if node else actor
        if action_type == 'rites':
            return self.entity('rite', key) if key in ids['ritual'] else actor
        if action_type == 'diplomacy':
            faction, _, term = key.rpartition(':')
            data['term'] = term
            return self.faction(faction) if faction in self.ref.factions or faction in ids['faction'] else player
        return actor

    def build(self, counter, index, name):
        started = time.perf_counter()
        self.catalogue()
        placed = self.consequences()
        held_out = held(self.campaign['campaign_key'])
        observations, fact_rows, queries, targets = [], {relation: [] for relation in RELATIONS}, [], []
        previous, carried = {}, None
        width = self.features // 2
        for step, snapshot in enumerate(self.snapshots):
            features, facts = {}, {}
            self.world(snapshot, features, facts)
            if snapshot['kind_id'] == DECISION:
                entity_features, entity_facts = {}, {}
                self.entities_block(snapshot, entity_features, entity_facts)
                carried = (entity_features, entity_facts)
            if carried:
                for node, data in carried[0].items():
                    features.setdefault(node, {}).update(data)
                for relation, items in carried[1].items():
                    facts.setdefault(relation, {}).update(items)
            for relation, items in placed.get(step, {}).items():
                facts.setdefault(relation, {}).update(items)
            category, actor, subject, context, data = self.query(snapshot, step, features, facts)
            for relation, items in self.static_facts.items():
                facts.setdefault(relation, {}).update(items)
            present = {relation: set(items) for relation, items in facts.items()}
            removed = {relation: keys - present.get(relation, set()) for relation, keys in previous.items()}
            for relation, items in facts.items():
                for (source, target, _), values in items.items():
                    features.setdefault(source, dict(self.static_features.get(source, {})))
                    features.setdefault(target, dict(self.static_features.get(target, {})))
            for node in list(features):
                features[node] = dict(self.static_features.get(node, {}), **features[node])
            observed = sorted(features)
            for seq, node in enumerate(observed):
                observations.append((node, step, seq, self.vector(self.kinds[node], features[node])))
            seq = 0
            for relation in RELATIONS:
                for (source, target, _), values in facts.get(relation, {}).items():
                    fact_rows[relation].append((source, target, seq, step, True) + values)
                    seq += 1
            seq = 0
            for relation, keys in removed.items():
                width_r = len(RELATIONS[relation][1])
                for source, target, _ in sorted(keys):
                    fact_rows[relation].append((source, target, seq, step, False) + (math.nan,) * width_r)
                    seq += 1
            previous = present
            signal = [number(snapshot.get(part)) for part in SIGNALS]
            queries.append((snapshot['snapshot_id'], step, snapshot['turn'], category, actor, subject, context, self.vector('query', data),
                            signal, [0.0 if math.isnan(value) else 1.0 for value in signal]))
            baseline = self.baseline(snapshot)
            if baseline is not None:
                value = label(self.opened, baseline[0], baseline[1])
                if value is not None:
                    targets.append((step, value))
        LOG.info('campaign build exit campaign=%d seconds=%.3f steps=%d entities=%d observations=%d facts=%d labels=%d', self.campaign['campaign_id'],
                 time.perf_counter() - started, len(self.snapshots), len(self.kinds), len(observations), sum(len(rows) for rows in fact_rows.values()), len(targets))
        return self.tables_of(counter, index, name, observations, fact_rows, queries, targets, held_out)

    def baseline(self, snapshot):
        if snapshot['kind_id'] == DECISION:
            return snapshot['turn'], snapshot
        interrupt = self.tables.get('interrupts', {}).get(snapshot['snapshot_id'], [None])[0]
        if interrupt is None:
            return None
        if self.ref.enum.get(interrupt['state_at_id']) == 'panel':
            return snapshot['turn'], snapshot
        previous = next((row for row in self.snapshots if row['snapshot_id'] == interrupt['prev_decision_id']), None)
        return (previous['turn'], previous) if previous else None

    def tables_of(self, counter, index, name, observations, fact_rows, queries, targets, held_out):
        node_ids = counter.take('nodes', len(self.kinds))
        tables = {('graph', None, None): pa.table(dict(id=pa.array([index], pa.int32()), name=pa.array([name])))}
        members = {kind: np.flatnonzero(np.asarray(self.kinds) == kind) for kind in NODES}
        for kind in NODES:
            picked = members[kind]
            tables[(kind, None, None)] = pa.table(dict(id=pa.array(node_ids[picked]), graph=pa.array(np.full(len(picked), index), pa.int32()),
                                                       entity=pa.array(picked, pa.int32()), key=pa.array([self.keys[i] for i in picked], pa.string())))
        observation_ids = counter.take('observation', len(observations))
        entity = np.asarray([row[0] for row in observations], np.int64)
        tables[('observation', None, None)] = pa.table(dict(id=pa.array(observation_ids), seq=pa.array([row[2] for row in observations], pa.int32()),
                                                            step=pa.array([row[1] for row in observations], pa.int32()),
                                                            x=fixed(np.stack([row[3] for row in observations]), self.features)))
        kinds = np.asarray(self.kinds)[entity]
        for kind in NODES:
            picked = kinds == kind
            tables[('observed', kind, 'observation')] = pa.table({'from': pa.array(node_ids[entity[picked]]), 'to': pa.array(observation_ids[picked])})
        query_ids = counter.take('query', len(queries))
        tables[('query', None, None)] = pa.table(dict(
            id=pa.array(query_ids), graph=pa.array(np.full(len(queries), index), pa.int32()), step=pa.array([row[1] for row in queries], pa.int32()),
            period=pa.array([row[2] for row in queries], pa.int32()), key=pa.array([row[0] for row in queries], pa.int64()),
            category=pa.array([row[3] for row in queries], pa.string()), x=fixed(np.stack([row[7] for row in queries]), self.features),
            signal=fixed(np.asarray([row[8] for row in queries], np.float32), len(SIGNALS)),
            signal_observed=fixed(np.asarray([row[9] for row in queries], np.float32), len(SIGNALS))))
        for position, role in ((4, 'actor'), (5, 'subject'), (6, 'context')):
            chosen = np.asarray([row[position] for row in queries], np.int64)
            for kind in NODES:
                picked = np.asarray(self.kinds)[chosen] == kind
                tables[('query_' + role, 'query', kind)] = pa.table({'from': pa.array(query_ids[picked]), 'to': pa.array(node_ids[chosen[picked]])})
        for relation, (pairs, attributes) in RELATIONS.items():
            rows = fact_rows[relation]
            source = np.asarray([row[0] for row in rows], np.int64)
            target = np.asarray([row[1] for row in rows], np.int64)
            source_kind, target_kind = np.asarray(self.kinds)[source], np.asarray(self.kinds)[target]
            for pair in pairs:
                picked = (source_kind == pair[0]) & (target_kind == pair[1])
                chosen = [rows[i] for i in np.flatnonzero(picked)]
                columns = {'from': pa.array(node_ids[source[picked]]), 'to': pa.array(node_ids[target[picked]]),
                           'seq': pa.array([row[2] for row in chosen], pa.int32()), 'step': pa.array([row[3] for row in chosen], pa.int32()),
                           'present': pa.array([row[4] for row in chosen], pa.bool_())}
                for offset, attribute in enumerate(attributes):
                    columns[attribute] = pa.array([row[5 + offset] for row in chosen], pa.float32())
                tables[('edge_' + relation, pair[0], pair[1])] = pa.table(columns)
        label_ids = counter.take(TARGET, len(targets))
        tables[(TARGET, None, None)] = pa.table(dict(id=pa.array(label_ids), y=pa.array([row[1] for row in targets], pa.float64()),
                                                     split=pa.array(['validation' if held_out else 'train'] * len(targets), pa.string())))
        tables[(TARGET + '_of', TARGET, 'query')] = pa.table({'from': pa.array(label_ids), 'to': pa.array(query_ids[[row[0] for row in targets]])})
        return tables
