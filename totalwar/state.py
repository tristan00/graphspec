from functools import partial

from .collections import character_collections
from .payloads import payload
from .values import put_fact


def entities_block(graph, snapshot, features, facts):
    at = partial(graph.at, snapshot=snapshot)
    fact = partial(put_fact, facts)
    character_state(graph, at, features, fact)
    province_state(graph, at, features, fact)
    campaign_state(graph, at, features, fact)


def character_state(graph, at, features, fact):
    ref, keys = graph.ref, graph.ref.keys
    characters = {}
    for row in at("characters"):
        character = graph.character(row["cqi"])
        characters[row["character_id"]] = character
        data = {
            name: row.get(name)
            for name in (
                "is_hero",
                "rank",
                "skill_points",
                "units",
                "pending_recruits",
                "ap_pct",
                "ap_remaining",
                "ap_per_turn",
                "hp",
                "loyalty",
                "x",
                "y",
                "garrisoned",
                "besieging",
                "acted",
                "is_leader",
                "wounded",
                "is_agent",
                "can_embed",
                "reach_max",
            )
        }
        data.update(
            subtype=graph.key("agent_subtype", row["subtype_id"]),
            agent_type=graph.key("agent_type", row["agent_type_id"]),
            stance=graph.key("stance", row["stance_id"]),
        )
        for index, ray in enumerate(row["reach_rays"] or ()):
            data["ray%d" % index] = ray
        features.setdefault(character, {}).update(
            {name: value for name, value in data.items() if value is not None}
        )
        if row["subtype_id"] is not None:
            fact("of_subtype", character, graph.entity("subtype", keys["agent_subtype"][row["subtype_id"]]))
        for region in row["reach_setts_true"] or ():
            fact("can_reach", character, graph.region(region))
        for other in row["reach_chars_true"] or ():
            fact("can_reach_character", character, graph.character(other))
        for position, unit in enumerate(row["pending_recruit_unit_ids"] or ()):
            fact(
                "recruiting",
                character,
                graph.reference("unit", keys["unit"][unit]),
                None,
                position,
                tag=("pending", position),
            )
        for skill in row["hidden_skill_ids"] or ():
            fact("hidden_skill", character, graph.reference("skill", keys["skill"][skill]), None, None)
        agent = graph.key("agent_type", row["agent_type_id"])
        for action in ref.agent_actions.get(agent, ()) if agent else ():
            name = action_name(action)
            if ("action", name) in graph.entities:
                fact(
                    "can_act", character, graph.entity("action", name), action["chance_of_success"], tag=name
                )
    for row in at("extensions"):
        character = characters[row["character_id"]]
        data = {
            name: row.get(name)
            for name in (
                "xp",
                "xp_next_level",
                "subterfuge",
                "zeal",
                "authority",
                "resurrection_turns",
                "upkeep",
            )
        }
        features[character].update({name: value for name, value in data.items() if value is not None})
        if row["background_skill_id"] is not None:
            fact("background", character, graph.reference("skill", keys["skill"][row["background_skill_id"]]))
        for item in row["armory_item_ids"] or ():
            fact("armory", character, graph.reference("item", str(item)))
    character_collections(graph, at, characters, features, fact)


def province_state(graph, at, features, fact):
    keys, player = graph.ref.keys, graph.faction(graph.player)
    provinces = {}
    for row in at("provinces"):
        region = graph.region(row["region_id"])
        provinces[row["region_id"]] = region
        data = {
            name: row.get(name)
            for name in (
                "settlement_present",
                "complete_owner",
                "max_slots",
                "free_slots",
                "can_set_edict",
                "public_order",
                "buildings",
                "settlement_level",
                "growth_per_turn",
                "gross_income",
                "development_points",
                "income",
                "has_port",
                "has_walls",
            )
        }
        data["capital"] = row["is_capital"]
        features.setdefault(region, dict(graph.static_features[region])).update(
            {name: value for name, value in data.items() if value is not None}
        )
        fact("owns", player, region)
        locked = set(row["locked_slots"] or ())
        province = graph.province(keys["region"][row["region_id"]])
        if province:
            target = graph.entity("province", province)
            features.setdefault(target, {})
            for edict in row["edict_ids"] or ():
                fact(
                    "edict_option",
                    target,
                    graph.entity("edict", keys["edict"][edict]),
                    edict == row["selected_edict_id"],
                    edict == row["active_edict_id"],
                    row["can_set_edict"],
                )
        for index in range(int(row["max_slots"] or 0)):
            slot = graph.slot(keys["region"][row["region_id"]], index)
            features[slot] = dict(index=index, locked=index in locked)
            fact("has_slot", region, slot)
    for row in at("slot_states"):
        slot = graph.slot(keys["region"][row["region_id"]], row["index"])
        features.setdefault(slot, dict(index=row["index"])).update(
            {
                name: row[name]
                for name in (
                    "damaged",
                    "repairing",
                    "upgrading",
                    "dismantling",
                    "queued",
                    "refund",
                    "repair_cost",
                    "health",
                    "max_health",
                    "ruined",
                    "empty",
                )
            }
        )
        fact("has_slot", provinces[row["region_id"]], slot)
        if row["building_id"] is not None:
            fact(
                "holds",
                slot,
                graph.reference("building", keys["building"][row["building_id"]]),
                row["health"],
                row["max_health"],
                row["damaged"],
                row["ruined"],
                None,
            )
        if row["queued_building_id"] is not None:
            fact(
                "constructing",
                slot,
                graph.reference("building", keys["building"][row["queued_building_id"]]),
                None,
                None,
                1,
                tag="queued",
            )
    for row in at("built"):
        slot = graph.slot(keys["region"][row["region_id"]], row["slot_index"])
        features.setdefault(slot, dict(index=row["slot_index"]))
        fact("has_slot", provinces[row["region_id"]], slot)
        fact(
            "holds",
            slot,
            graph.reference("building", keys["building"][row["building_id"]]),
            None,
            None,
            None,
            None,
            None,
        )
    for row in at("building_now"):
        slot = graph.slot(keys["region"][row["region_id"]], row["slot_index"])
        features.setdefault(slot, dict(index=row["slot_index"]))
        fact("has_slot", provinces[row["region_id"]], slot)
        fact(
            "constructing",
            slot,
            graph.reference("building", keys["building"][row["building_id"]]),
            row["turns_left"],
            row["paused"],
            0,
            tag="now",
        )
    for row in at("buildable"):
        slot = graph.slot(keys["region"][row["region_id"]], row["slot_index"])
        features.setdefault(slot, dict(index=row["slot_index"]))
        fact("has_slot", provinces[row["region_id"]], slot)
        fact(
            "can_build",
            slot,
            graph.reference("building", keys["building"][row["building_id"]]),
            row["cost"],
            row["upkeep"],
            row["level"],
            not row["empty"],
            row["can_afford_resources"],
        )
    for row in at("corruption"):
        fact(
            "corruption",
            provinces[row["region_id"]],
            graph.entity("resource", keys["pooled_resource"][row["resource_id"]]),
            row["value"],
        )
    for row in at("income_breakdown"):
        fact(
            "income",
            provinces[row["region_id"]],
            graph.entity("resource", "income:" + row["label"]),
            row["amount"],
        )
    for name, plague in (("province_bundles", 0), ("plague_bundles", 1)):
        for row in at(name):
            fact(
                "bundle",
                provinces[row["region_id"]],
                graph.reference("bundle", keys["effect_bundle"][row["effect_bundle_id"]]),
                row["turns_remaining"],
                0,
                plague,
                tag=name,
            )


def campaign_state(graph, at, features, fact):
    ref, keys, player = graph.ref, graph.ref.keys, graph.faction(graph.player)
    for row in at("states"):
        features.setdefault(player, {})["research_points"] = row["research_points"]
        if row["current_research_id"] is not None:
            fact(
                "researching",
                player,
                graph.entity("tech", keys["tech_node"][row["current_research_id"]]),
                row["research_points"],
            )
    for row in at("tech"):
        fact(
            "tech_state",
            player,
            graph.entity("tech", keys["tech_node"][row["tech_node_id"]]),
            row["researched"],
            row["can_research"],
            row["cost"],
        )
    for row in at("rites"):
        rite = graph.entity("rite", keys["ritual"][row["ritual_id"]])
        ritual = ref.rituals.get(keys["ritual"][row["ritual_id"]], {})
        features[rite] = dict(
            category=ritual.get("category"),
            cast_time=ritual.get("cast_time"),
            cooldown_time=ritual.get("cooldown_time"),
            influence_cost=ritual.get("influence_cost"),
            slave_cost=ritual.get("slave_cost"),
        )
        fact("rite", player, rite, row["can_perform"], row["reason_id"] or 0)
    for row in at("lord_pools"):
        fact("lord_pool", player, graph.entity("subtype", keys["agent_subtype"][row["subtype_id"]]), row["n"])
    for row in at("candidates"):
        subtype = keys["agent_subtype"][row["subtype_id"]]
        candidate = graph.candidate(subtype, row["ord"])
        features[candidate] = dict(
            candidate_index=row["ord"],
            is_agent=row["agent"],
            subtype=graph.key("agent_subtype", row["cand_subtype_id"]),
        )
        fact("candidate_of", candidate, graph.entity("subtype", subtype), row["can"], row["agent"])
        if row["bg_skill_id"] is not None:
            fact("candidate_skill", candidate, graph.reference("skill", keys["skill"][row["bg_skill_id"]]))
        for trait in row["trait_ids"] or ():
            fact("candidate_trait", candidate, graph.reference("trait", keys["trait"][trait]))
    for row in at("item_pool"):
        if row["ancillary_id"] is not None:
            fact(
                "item_pool",
                player,
                graph.reference("item", keys["ancillary"][row["ancillary_id"]]),
                row["index"],
                tag=row["index"],
            )
    for row in at("missions"):
        mission = graph.entity("mission", keys["mission"][row["mission_id"]])
        info = ref.missions.get(keys["mission"][row["mission_id"]], {})
        features[mission] = dict(
            mission_type=info.get("mission_type"),
            event_category=info.get("event_category"),
            is_quest=info.get("quest_mission"),
            issuer=graph.key("mission_issuer", row["issuer_id"]),
            category=row["category"],
        )
        fact(
            "mission",
            player,
            mission,
            row["status_id"],
            row["turns_remaining"],
            row["is_quest"],
            row["is_victory"],
            row["completed"],
            row["cancelled"],
            row["pending"],
        )
        for record in ref.mission_payloads.get(keys["mission"][row["mission_id"]], ()):
            payload(graph, mission, record, features[mission], fact)


def action_name(row):
    prefix = "agent_action_%s_%s_" % (row["agent"], row["ability"])
    return row["unique_id"].split(prefix, 1)[1] if prefix in row["unique_id"] else row["unique_id"]
