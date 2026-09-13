import logging
import time

from graph_tuning_infra.postgres import records


LOG = logging.getLogger(__name__)
FAMILIES = ('faction', 'region', 'province', 'skill', 'unit', 'building', 'tech_node', 'agent_subtype', 'agent_type', 'ancillary', 'ritual',
            'trait', 'trait_level', 'mission', 'mission_issuer', 'edict', 'effect_bundle', 'pooled_resource', 'stance', 'effect', 'effect_scope',
            'dilemma', 'incident', 'action_type', 'unit_category')


def grouped(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def keyed(rows, key):
    return {row[key]: row for row in rows}


class Reference:

    def __init__(self, connection):
        started = time.perf_counter()
        read = lambda statement: records(connection, statement)
        self.keys = {family: {row['id']: row['key'] for row in read('SELECT id, key FROM dict.%s' % family)} for family in FAMILIES}
        self.ids = {family: {key: identifier for identifier, key in table.items()} for family, table in self.keys.items()}
        self.enum = {row['enum_id']: row['key'] for row in read('SELECT enum_id, key FROM dict.enum')}
        self.actions = {row['action_id']: (row['type'], row['action_key']) for row in read(
            'SELECT a.action_id, t.key AS type, a.action_key FROM dict.action a JOIN dict.action_type t ON t.id = a.action_type_id')}
        self.factions = keyed(read('SELECT f.key, f.subculture, f.category, f.is_rebel, cs.culture FROM ref.factions f '
                                   'LEFT JOIN ref.cultures_subcultures cs ON cs.subculture = f.subculture'), 'key')
        self.tech_sets = keyed(read('SELECT key, faction_key, culture, subculture FROM ref.technology_node_sets'), 'key')
        self.tech_nodes = keyed(read('SELECT n.key, n.technology_node_set AS set, n.technology_key, n.tier, n.indent, n.research_points_required, '
                                     'n.cost_per_round, n.food_cost, n.resource_cost, n.required_parents, t.is_civil, t.is_engineering, t.is_military '
                                     'FROM ref.technology_nodes n LEFT JOIN ref.technologies t ON t.key = n.technology_key'), 'key')
        self.tech_links = grouped(read('SELECT l.parent_key, l.child_key, n.technology_node_set AS set FROM ref.technology_node_links l '
                                       'JOIN ref.technology_nodes n ON n.key = l.child_key'), 'set')
        self.tech_effects = grouped(read('SELECT technology, effect, value, effect_scope FROM ref.technology_effects_junction'), 'technology')
        self.tech_buildings = grouped(read('SELECT technology, required_building_level FROM ref.technology_required_building_levels_junctions'), 'technology')
        self.skill_sets = {}
        for row in read('SELECT key, agent_subtype_key, faction_key FROM ref.character_skill_node_sets WHERE agent_subtype_key IS NOT NULL'):
            self.skill_sets.setdefault(row['agent_subtype_key'], {})[row['faction_key'] or ''] = row['key']
        self.skill_nodes = grouped(read('SELECT i.set, n.key, n.character_skill_key AS skill, n.tier, n.indent, n.required_num_parents, n.points_on_creation '
                                        'FROM ref.character_skill_node_set_items i JOIN ref.character_skill_nodes n ON n.key = i.item'), 'set')
        self.skill_links = grouped(read('SELECT i.set, l.parent_key, l.child_key, l.link_type, l.initial_descent_tiers FROM ref.character_skill_node_links l '
                                        'JOIN ref.character_skill_node_set_items i ON i.item = l.child_key'), 'set')
        self.skills = keyed(read('SELECT key, unlocked_at_rank, is_background_skill, influence_cost FROM ref.character_skills'), 'key')
        self.skill_effects = grouped(read('SELECT character_skill_key AS skill, level, effect_key AS effect, value, effect_scope '
                                          'FROM ref.character_skill_level_to_effects_junctions'), 'skill')
        self.buildings = keyed(read('SELECT b.level_name AS key, b.chain, b.level, b.create_cost, b.create_time, b.upkeep_cost, b.food_cost, '
                                    'b.development_point_cost, b.only_in_capital, b.can_be_damaged, c.building_superchain AS superchain, c.chain_category '
                                    'FROM ref.building_levels b LEFT JOIN ref.building_chains c ON c.key = b.chain'), 'key')
        self.building_upgrades = grouped(read('SELECT "from" AS source, "to" AS target FROM ref.building_upgrades_junction'), 'source')
        self.building_requires = grouped(read('SELECT building_level, required FROM ref.building_level_required_buildings'), 'building_level')
        self.building_effects = grouped(read('SELECT building, effect, value, effect_scope, value_damaged, value_ruined FROM ref.building_effects_junction'), 'building')
        self.building_units = grouped(read('SELECT building, unit FROM ref.building_units_allowed'), 'building')
        self.units = keyed(read('SELECT m.unit AS key, m.recruitment_cost, m.upkeep_cost, m.create_time, m.campaign_cap, m.tier, m.num_men, m.is_monstrous, '
                                'm.can_siege, m.caste, l.category, l.class, l.morale, l.bonus_hit_points, l.armour, l.shield, l.melee_attack, l.melee_defence, '
                                'l.charge_bonus, l.accuracy, l.campaign_action_points, l.num_mounts, l.primary_melee_weapon AS weapon '
                                'FROM ref.main_units m LEFT JOIN ref.land_units l ON l.key = m.land_unit'), 'key')
        self.items = keyed(read('SELECT key, type, category, subcategory, legendary_item, transferrable, randomly_dropped, uniqueness_score, can_be_stolen '
                                'FROM ref.ancillaries'), 'key')
        self.item_effects = grouped(read('SELECT ancillary, effect, value, effect_scope FROM ref.ancillary_to_effects'), 'ancillary')
        self.item_skills = grouped(read('SELECT ancillary, required_skill, required_skill_level FROM ref.ancillaries_required_skills'), 'ancillary')
        self.traits = keyed(read('SELECT key, hidden, precedence, no_going_back_level FROM ref.character_traits'), 'key')
        self.trait_effects = grouped(read('SELECT l.trait, l.level, e.effect, e.value, e.effect_scope FROM ref.character_trait_levels l '
                                          'JOIN ref.trait_level_effects e ON e.trait_level = l.key'), 'trait')
        self.effects = keyed(read('SELECT effect AS key, category, is_positive_value_good FROM ref.effects'), 'key')
        self.bundles = keyed(read('SELECT key, bundle_target, is_global_effect FROM ref.effect_bundles'), 'key')
        self.bundle_effects = grouped(read('SELECT effect_bundle_key AS bundle, effect_key AS effect, value, effect_scope FROM ref.effect_bundles_to_effects_junctions'), 'bundle')
        self.rituals = keyed(read('SELECT key, cast_time, cooldown_time, category, influence_cost, slave_cost FROM ref.rituals'), 'key')
        self.missions = keyed(read('SELECT key, mission_type, quest_mission, event_category FROM ref.missions'), 'key')
        self.edicts = keyed(read('SELECT key, effect_bundle FROM ref.provincial_initiative_records'), 'key')
        self.resources = keyed(read('SELECT key, scope FROM ref.pooled_resources'), 'key')
        self.subtypes = keyed(read('SELECT key, is_caster, cap, cost, recruitable, can_equip_ancillaries FROM ref.agent_subtypes'), 'key')
        self.agents = keyed(read('SELECT key, move_points, faction_total_cap FROM ref.agents'), 'key')
        self.regions = keyed(read('SELECT region AS key, province, is_capital FROM ref.region_to_province_junctions'), 'key')
        self.dilemmas = keyed(read('SELECT key, event_category, is_large_dilemma AS is_large FROM ref.dilemmas'), 'key')
        self.incidents = keyed(read('SELECT key, event_category, is_large_incident AS is_large FROM ref.incidents'), 'key')
        self.scopes = {key: index for index, key in enumerate(sorted(row['key'] for row in read('SELECT key FROM ref.campaign_effect_scopes')))}
        LOG.info('reference exit seconds=%.3f tech_nodes=%d skill_sets=%d buildings=%d units=%d effects=%d', time.perf_counter() - started,
                 len(self.tech_nodes), len(self.skill_sets), len(self.buildings), len(self.units), len(self.effects))

    def tech_set(self, faction):
        row = self.factions.get(faction)
        for candidate in self.tech_sets.values():
            if candidate['faction_key'] == faction:
                return candidate['key']
        if row is None:
            return None
        for candidate in self.tech_sets.values():
            if candidate['subculture'] and candidate['subculture'] == row['subculture'] and not candidate['faction_key']:
                return candidate['key']
        for candidate in self.tech_sets.values():
            if candidate['culture'] and candidate['culture'] == row['culture'] and not candidate['faction_key'] and not candidate['subculture']:
                return candidate['key']
        return None

    def skill_set(self, subtype, faction):
        sets = self.skill_sets.get(subtype)
        if not sets:
            return None
        return sets.get(faction) or sets.get('') or next(iter(sets.values()))

    def scope(self, key):
        return self.scopes.get(key, -1)
