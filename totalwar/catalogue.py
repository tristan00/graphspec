import logging
import time

from .state import action_name
from .values import number

LOG = logging.getLogger(__name__)


class Catalogue:
    def __init__(self, graph):
        self.graph = graph

    def add(self, relation, source, target, *values):
        numbers = tuple(number(value) for value in values)
        self.graph.static_facts.setdefault(relation, {})[(source, target, numbers)] = numbers

    def effect(self, key):
        row = self.graph.ref.effects.get(key)
        data = dict(category=row["category"], is_positive_good=row["is_positive_value_good"]) if row else {}
        return self.graph.static("effect", key, data)

    def grants(self, source, rows):
        for row in rows:
            self.add(
                "grants",
                source,
                self.effect(row["effect"]),
                row["value"],
                self.graph.ref.scope(row["effect_scope"]),
                row.get("level"),
                row.get("value_damaged"),
                row.get("value_ruined"),
            )


def catalogue(graph):
    started = time.perf_counter()
    builder = Catalogue(graph)
    for populate in (
        technologies,
        skill_trees,
        skills,
        agent_actions,
        traits,
        buildings,
        equipment,
        bundles,
        geography,
    ):
        populate(builder)
    LOG.info(
        "catalogue exit campaign=%d seconds=%.3f entities=%d static_facts=%d",
        graph.campaign["campaign_id"],
        time.perf_counter() - started,
        len(graph.kinds),
        sum(len(items) for items in graph.static_facts.values()),
    )


def technologies(builder):
    graph, ref = builder.graph, builder.graph.ref
    add, grants = builder.add, builder.grants
    techs = set()
    for row in graph.rows["tech"]:
        techs.add(graph.key("tech_node", row["tech_node_id"]))
    for row in graph.rows["states"]:
        techs.add(graph.key("tech_node", row["current_research_id"]))
    techs.discard(None)
    chosen = ref.tech_set(graph.player)
    if chosen is not None:
        techs |= {key for key, node in ref.tech_nodes.items() if node["set"] == chosen}
    for key in sorted(techs):
        node = ref.tech_nodes.get(key, {})
        tech = graph.static(
            "tech",
            key,
            {
                name: node.get(name)
                for name in (
                    "tier",
                    "indent",
                    "research_points_required",
                    "cost_per_round",
                    "food_cost",
                    "resource_cost",
                    "required_parents",
                    "is_civil",
                    "is_engineering",
                    "is_military",
                )
            },
        )
        grants(tech, ref.tech_effects.get(node.get("technology_key"), ()))
        for row in ref.tech_buildings.get(node.get("technology_key"), ()):
            if row["required_building_level"] in ref.buildings:
                add("requires_building", tech, graph.reference("building", row["required_building_level"]))
    for link in ref.tech_links.get(chosen, ()):
        if link["parent_key"] in techs and link["child_key"] in techs:
            add(
                "tech_prerequisite",
                graph.entity("tech", link["parent_key"]),
                graph.entity("tech", link["child_key"]),
            )
    by_technology = {
        ref.tech_nodes[key]["technology_key"]: key for key in sorted(techs) if key in ref.tech_nodes
    }
    for key in sorted(techs & ref.tech_nodes.keys()):
        technology = ref.tech_nodes[key]["technology_key"]
        for row in ref.tech_requires.get(technology, ()):
            parent = by_technology.get(row["required_technology"])
            if parent:
                add("tech_prerequisite", graph.entity("tech", parent), graph.entity("tech", key))


def skill_trees(builder):
    graph, ref = builder.graph, builder.graph.ref
    add = builder.add
    subtypes = {
        graph.key("agent_subtype", row["subtype_id"])
        for row in graph.rows["characters"] + graph.rows["armies"]
    }
    subtypes |= {graph.key("agent_subtype", row["subtype_id"]) for row in graph.rows["lord_pools"]}
    subtypes |= {graph.key("agent_subtype", row["cand_subtype_id"]) for row in graph.rows["candidates"]}
    subtypes.discard(None)
    for subtype in sorted(subtypes):
        row = ref.subtypes.get(subtype, {})
        node = graph.static(
            "subtype",
            subtype,
            dict(
                is_caster=row.get("is_caster"),
                cap=row.get("cap"),
                cost=row.get("cost"),
                recruitable=row.get("recruitable"),
                can_equip=row.get("can_equip_ancillaries"),
            ),
        )
        chosen_set = ref.skill_set(subtype, graph.player)
        nodes = {item["key"]: item for item in ref.skill_nodes.get(chosen_set, ())}
        for item in nodes.values():
            skill = graph.reference("skill", item["skill"])
            add("permits_skill", node, skill, item["tier"], item["indent"], item["required_num_parents"])
        for link in ref.skill_links.get(chosen_set, ()):
            parent, child = nodes.get(link["parent_key"]), nodes.get(link["child_key"])
            if parent and child:
                add(
                    "skill_link",
                    graph.reference("skill", parent["skill"]),
                    graph.reference("skill", child["skill"]),
                    link["link_type"] == "SUBSET_REQUIRED",
                    link["initial_descent_tiers"],
                )


def skills(builder):
    graph, ref = builder.graph, builder.graph.ref
    add, grants = builder.add, builder.grants
    for row in graph.rows["skills"] + graph.rows["hidden_skills"] + graph.rows["skill_prereqs"]:
        graph.reference("skill", graph.key("skill", row["skill_id"]))
    for row in graph.rows["skill_prereqs"]:
        graph.reference("skill", graph.key("skill", row["parent_skill_id"]))
    for row in graph.rows["candidates"]:
        if row["bg_skill_id"] is not None:
            graph.reference("skill", graph.key("skill", row["bg_skill_id"]))
        for trait in row["trait_ids"] or ():
            graph.reference("trait", graph.key("trait", trait))
    for row in graph.rows["extensions"]:
        if row["background_skill_id"] is not None:
            graph.reference("skill", graph.key("skill", row["background_skill_id"]))
    for row in graph.rows["traits"] + graph.rows["trait_progress"]:
        graph.reference("trait", graph.key("trait", row["trait_id"]))
    for skill in [key for kind, key in graph.entities if kind == "skill"]:
        grants(graph.entity("skill", skill), ref.skill_effects.get(skill, ()))
        for row in ref.skill_items.get(skill, ()):
            if row["granted_ancillary"] in ref.items:
                add(
                    "grants_item",
                    graph.entity("skill", skill),
                    graph.reference("item", row["granted_ancillary"]),
                    row["level"],
                )


def agent_actions(builder):
    graph, ref = builder.graph, builder.graph.ref
    add = builder.add
    faction = ref.factions.get(graph.player)
    subculture = None if faction is None else faction["subculture"]
    agents = {
        graph.key("agent_type", row["agent_type_id"])
        for row in graph.rows["characters"] + graph.rows["armies"]
    }
    agents.discard(None)
    for agent in sorted(agents):
        for row in ref.agent_actions.get(agent, ()):
            if row["subculture"] and row["subculture"] != subculture:
                continue
            action = graph.static(
                "action",
                action_name(row),
                dict(ability=row["ability"], attribute=row["attribute"], chance=row["chance_of_success"]),
            )
            for outcome, success in (
                ("success", 1),
                ("critical_success", 1),
                ("failure", 0),
                ("critical_failure", 0),
            ):
                result = ref.action_results.get(row[outcome])
                if not result:
                    continue
                for side, actor in (("actor", 1), ("target", 0)):
                    bundle = result[side + "_effect_bundle"]
                    if bundle:
                        add(
                            "action_result",
                            action,
                            graph.reference("bundle", bundle),
                            success,
                            actor,
                            result[side + "_effect_bundle_turns"],
                        )


def traits(builder):
    graph, ref = builder.graph, builder.graph.ref
    grants = builder.grants
    for trait in [key for kind, key in graph.entities if kind == "trait"]:
        grants(graph.entity("trait", trait), ref.trait_effects.get(trait, ()))


def buildings(builder):
    graph, ref = builder.graph, builder.graph.ref
    add, grants = builder.add, builder.grants
    buildings = set()
    for name in ("built", "buildable", "building_now", "horde_slots"):
        buildings |= {graph.key("building", row["building_id"]) for row in graph.rows[name]}
    buildings |= {graph.key("building", row["queued_building_id"]) for row in graph.rows["slot_states"]}
    buildings |= {graph.key("building", row["building_id"]) for row in graph.rows["slot_states"]}
    buildings.discard(None)
    frontier = sorted(buildings)
    while frontier:
        key = frontier.pop()
        targets = [row["target"] for row in ref.building_upgrades.get(key, ())]
        targets.extend(row["required"] for row in ref.building_requires.get(key, ()))
        for target in targets:
            if target in ref.buildings and target not in buildings:
                buildings.add(target)
                frontier.append(target)
    for key in sorted(buildings):
        building = graph.reference("building", key)
        grants(building, ref.building_effects.get(key, ()))
        for row in ref.building_upgrades.get(key, ()):
            if row["target"] in buildings:
                add("upgrades_to", building, graph.reference("building", row["target"]))
        for row in ref.building_requires.get(key, ()):
            if row["required"] in buildings:
                add("requires_building", building, graph.reference("building", row["required"]))
        for row in ref.building_units.get(key, ()):
            if row["unit"] in ref.units:
                add("unlocks_unit", building, graph.reference("unit", row["unit"]))


def equipment(builder):
    graph, ref = builder.graph, builder.graph.ref
    add, grants = builder.add, builder.grants
    for name in (
        "unit_cards",
        "recruitable",
        "pending_queue",
        "merc_pools",
        "faction_mercenaries",
        "battle_units",
    ):
        for row in graph.rows[name]:
            if row["unit_id"] is not None:
                graph.reference("unit", graph.key("unit", row["unit_id"]))
    for name in ("equipped", "item_pool"):
        for row in graph.rows[name]:
            if row["ancillary_id"] is not None:
                graph.reference("item", graph.key("ancillary", row["ancillary_id"]))
    for row in graph.rows["events"]:
        if row["ancillary_id"] is not None:
            graph.reference("item", graph.key("ancillary", row["ancillary_id"]))
    for item in [key for kind, key in graph.entities if kind == "item"]:
        node = graph.entity("item", item)
        grants(node, ref.item_effects.get(item, ()))
        for row in ref.item_skills.get(item, ()):
            add(
                "requires_skill",
                node,
                graph.reference("skill", row["required_skill"]),
                row["required_skill_level"],
            )


def bundles(builder):
    graph, ref = builder.graph, builder.graph.ref
    grants = builder.grants
    for name in (
        "faction_bundles",
        "character_bundles",
        "force_bundles",
        "province_bundles",
        "plague_bundles",
    ):
        for row in graph.rows[name]:
            graph.reference("bundle", graph.key("effect_bundle", row["effect_bundle_id"]))
    for row in graph.rows["faction_effects"]:
        graph.reference("bundle", graph.key("effect_bundle", row["effect_bundle_id"]))
    for bundle in [key for kind, key in graph.entities if kind == "bundle"]:
        grants(graph.entity("bundle", bundle), ref.bundle_effects.get(bundle, ()))


def geography(builder):
    graph = builder.graph
    add = builder.add
    adjacency = {}
    for row in graph.rows["regions"]:
        adjacency.setdefault(row["region_id"], set()).update(row["adjacent"] or ())
    for region, neighbours in adjacency.items():
        source = graph.region(region)
        for neighbour in sorted(neighbours):
            add("adjacent", source, graph.region(neighbour))
