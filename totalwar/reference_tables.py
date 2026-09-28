from dataclasses import dataclass

from graphspec.infra.postgres import records


def grouped(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def keyed(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError("duplicate reference identity: " + str(identity))
        result[identity] = row
    return result


@dataclass(frozen=True)
class ReferenceTable:
    name: str
    statement: str
    key: str
    index: object

    def load(self, connection):
        return self.index(records(connection, self.statement), self.key)


TABLES = (
    ReferenceTable("trait_levels", "SELECT key, trait FROM ref.character_trait_levels", "key", keyed),
    ReferenceTable(
        "factions",
        "SELECT f.key, f.subculture, f.category, f.is_rebel, cs.culture FROM ref.factions f LEFT JOIN "
        "ref.cultures_subcultures cs ON cs.subculture = f.subculture",
        "key",
        keyed,
    ),
    ReferenceTable(
        "tech_sets",
        "SELECT key, faction_key, culture, subculture FROM ref.technology_node_sets",
        "key",
        keyed,
    ),
    ReferenceTable(
        "tech_nodes",
        "SELECT n.key, n.technology_node_set AS set, n.technology_key, n.tier, n.indent, n.research_points_required, "
        "n.cost_per_round, n.food_cost, n.resource_cost, n.required_parents, t.is_civil, t.is_engineering, "
        "t.is_military FROM ref.technology_nodes n LEFT JOIN ref.technologies t ON t.key = n.technology_key",
        "key",
        keyed,
    ),
    ReferenceTable(
        "tech_links",
        "SELECT l.parent_key, l.child_key, n.technology_node_set AS set FROM ref.technology_node_links l JOIN "
        "ref.technology_nodes n ON n.key = l.child_key",
        "set",
        grouped,
    ),
    ReferenceTable(
        "tech_effects",
        "SELECT technology, effect, value, effect_scope FROM ref.technology_effects_junction",
        "technology",
        grouped,
    ),
    ReferenceTable(
        "tech_buildings",
        "SELECT technology, required_building_level FROM ref.technology_required_building_levels_junctions",
        "technology",
        grouped,
    ),
    ReferenceTable(
        "skill_nodes",
        "SELECT i.set, n.key, n.character_skill_key AS skill, n.tier, n.indent, n.required_num_parents, "
        "n.points_on_creation FROM ref.character_skill_node_set_items i JOIN ref.character_skill_nodes n ON n.key = "
        "i.item",
        "set",
        grouped,
    ),
    ReferenceTable(
        "skill_links",
        "SELECT i.set, l.parent_key, l.child_key, l.link_type, l.initial_descent_tiers FROM "
        "ref.character_skill_node_links l JOIN ref.character_skill_node_set_items i ON i.item = l.child_key",
        "set",
        grouped,
    ),
    ReferenceTable(
        "skills",
        "SELECT key, unlocked_at_rank, is_background_skill, influence_cost FROM ref.character_skills",
        "key",
        keyed,
    ),
    ReferenceTable(
        "skill_effects",
        "SELECT character_skill_key AS skill, level, effect_key AS effect, value, effect_scope FROM "
        "ref.character_skill_level_to_effects_junctions",
        "skill",
        grouped,
    ),
    ReferenceTable(
        "buildings",
        "SELECT b.level_name AS key, b.chain, b.level, b.create_cost, b.create_time, b.upkeep_cost, b.food_cost, "
        "b.development_point_cost, b.only_in_capital, b.can_be_damaged, c.building_superchain AS superchain, "
        "c.chain_category FROM ref.building_levels b LEFT JOIN ref.building_chains c ON c.key = b.chain",
        "key",
        keyed,
    ),
    ReferenceTable(
        "building_upgrades",
        'SELECT "from" AS source, "to" AS target FROM ref.building_upgrades_junction',
        "source",
        grouped,
    ),
    ReferenceTable(
        "building_requires",
        "SELECT building_level, required FROM ref.building_level_required_buildings",
        "building_level",
        grouped,
    ),
    ReferenceTable(
        "building_effects",
        "SELECT building, effect, value, effect_scope, value_damaged, value_ruined FROM ref.building_effects_junction",
        "building",
        grouped,
    ),
    ReferenceTable(
        "building_units", "SELECT building, unit FROM ref.building_units_allowed", "building", grouped
    ),
    ReferenceTable(
        "units",
        "SELECT m.unit AS key, m.recruitment_cost, m.upkeep_cost, m.create_time, m.campaign_cap, m.tier, m.num_men, "
        "m.is_monstrous, m.can_siege, m.caste, l.category, l.class, l.morale, l.bonus_hit_points, l.armour, l.shield, "
        "l.melee_attack, l.melee_defence, l.charge_bonus, l.accuracy, l.campaign_action_points, l.num_mounts, "
        "l.primary_melee_weapon AS weapon FROM ref.main_units m LEFT JOIN ref.land_units l ON l.key = m.land_unit",
        "key",
        keyed,
    ),
    ReferenceTable(
        "items",
        "SELECT key, type, category, subcategory, legendary_item, transferrable, randomly_dropped, uniqueness_score, "
        "can_be_stolen FROM ref.ancillaries",
        "key",
        keyed,
    ),
    ReferenceTable(
        "item_effects",
        "SELECT ancillary, effect, value, effect_scope FROM ref.ancillary_to_effects",
        "ancillary",
        grouped,
    ),
    ReferenceTable(
        "item_skills",
        "SELECT ancillary, required_skill, required_skill_level FROM ref.ancillaries_required_skills",
        "ancillary",
        grouped,
    ),
    ReferenceTable(
        "traits",
        "SELECT key, hidden, precedence, no_going_back_level FROM ref.character_traits",
        "key",
        keyed,
    ),
    ReferenceTable(
        "trait_effects",
        "SELECT l.trait, l.level, e.effect, e.value, e.effect_scope FROM ref.character_trait_levels l JOIN "
        "ref.trait_level_effects e ON e.trait_level = l.key",
        "trait",
        grouped,
    ),
    ReferenceTable(
        "effects", "SELECT effect AS key, category, is_positive_value_good FROM ref.effects", "key", keyed
    ),
    ReferenceTable(
        "bundles", "SELECT key, bundle_target, is_global_effect FROM ref.effect_bundles", "key", keyed
    ),
    ReferenceTable(
        "bundle_effects",
        "SELECT effect_bundle_key AS bundle, effect_key AS effect, value, effect_scope FROM "
        "ref.effect_bundles_to_effects_junctions",
        "bundle",
        grouped,
    ),
    ReferenceTable(
        "rituals",
        "SELECT key, cast_time, cooldown_time, category, influence_cost, slave_cost FROM ref.rituals",
        "key",
        keyed,
    ),
    ReferenceTable(
        "missions", "SELECT key, mission_type, quest_mission, event_category FROM ref.missions", "key", keyed
    ),
    ReferenceTable(
        "edicts", "SELECT key, effect_bundle FROM ref.provincial_initiative_records", "key", keyed
    ),
    ReferenceTable("resources", "SELECT key, scope FROM ref.pooled_resources", "key", keyed),
    ReferenceTable(
        "subtypes",
        "SELECT key, is_caster, cap, cost, recruitable, can_equip_ancillaries FROM ref.agent_subtypes",
        "key",
        keyed,
    ),
    ReferenceTable("agents", "SELECT key, move_points, faction_total_cap FROM ref.agents", "key", keyed),
    ReferenceTable(
        "regions",
        "SELECT region AS key, province, is_capital FROM ref.region_to_province_junctions",
        "key",
        keyed,
    ),
    ReferenceTable(
        "dilemmas", "SELECT key, event_category, is_large_dilemma AS is_large FROM ref.dilemmas", "key", keyed
    ),
    ReferenceTable(
        "incidents",
        "SELECT key, event_category, is_large_incident AS is_large FROM ref.incidents",
        "key",
        keyed,
    ),
    ReferenceTable(
        "agent_actions",
        "SELECT unique_id, agent, ability, attribute, chance_of_success, success, failure, critical_success, "
        "critical_failure, subculture FROM ref.agent_actions",
        "agent",
        grouped,
    ),
    ReferenceTable(
        "action_results",
        "SELECT key, actor_effect_bundle, target_effect_bundle, actor_effect_bundle_turns, target_effect_bundle_turns "
        "FROM ref.action_results",
        "key",
        keyed,
    ),
    ReferenceTable(
        "tech_requires",
        "SELECT technology, required_technology FROM ref.technology_required_technology_junctions",
        "technology",
        grouped,
    ),
    ReferenceTable(
        "skill_items",
        "SELECT skill, granted_ancillary, level FROM ref.character_skill_level_to_ancillaries_junctions",
        "skill",
        grouped,
    ),
    ReferenceTable(
        "mission_payloads",
        "SELECT mission_key, status_key, payload_key, value FROM ref.cdir_events_mission_payloads",
        "mission_key",
        grouped,
    ),
)
