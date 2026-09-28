from functools import partial

from .domain import TREATIES
from .values import HOSTILE_KINDS, put_fact


def world(graph, snapshot, features, facts):
    at = partial(graph.at, snapshot=snapshot)
    fact = partial(put_fact, facts)
    faction_state(graph, snapshot, at, features, fact)
    region_state(graph, at, features, fact)
    army_state(graph, at, features, fact)
    diplomacy_state(graph, at, features, fact)


def faction_state(graph, snapshot, at, features, fact):
    keys, player = graph.ref.keys, graph.faction(graph.player)
    scalars = {
        name: snapshot.get(name)
        for name in (
            "turn",
            "income",
            "settlements",
            "treasury",
            "is_researching",
            "armies",
            "lord_level",
            "allies",
            "vassals",
            "power_rank",
            "ll_wounded",
            "defeated",
            "difficulty",
            "net_income",
            "expenditure",
            "upkeep",
            "trade_value",
            "tax_level",
            "influence",
            "food_production",
            "food_consumption",
            "imperium_level",
            "settlement_income",
            "province_income",
            "raiding_income",
        )
    }
    scalars["hero_count"] = sum(row["n"] or 0 for row in at("hero_counts"))
    features[player] = dict(graph.static_features[player], **scalars)
    for row in at("resources"):
        fact(
            "resource",
            player,
            graph.entity("resource", keys["pooled_resource"][row["resource_id"]]),
            row["value"],
        )
    for row in at("faction_bundles"):
        fact(
            "bundle",
            player,
            graph.reference("bundle", keys["effect_bundle"][row["effect_bundle_id"]]),
            row["turns_remaining"],
            0,
            0,
        )
    for row in at("faction_mercenaries"):
        fact(
            "faction_mercenary",
            player,
            graph.reference("unit", keys["unit"][row["unit_id"]]),
            row["base_max"],
            row["current_max"],
        )


def region_state(graph, at, features, fact):
    keys = graph.ref.keys
    for row in at("regions"):
        region = graph.region(row["region_id"])
        features[region] = dict(
            graph.static_features[region],
            x=row["x"],
            y=row["y"],
            capital=row["capital"],
            abandoned=row["abandoned"],
        )
        if row["owner_id"] is not None:
            fact("owns", graph.faction(keys["faction"][row["owner_id"]]), region)
    for row in at("settlements"):
        region = graph.region(row["region_id"])
        features.setdefault(region, dict(graph.static_features[region])).update(
            garrison=row["units"], x=row["x"], y=row["y"]
        )
    for row in at("ruins"):
        region = graph.region(row["region_id"])
        features.setdefault(region, dict(graph.static_features[region])).update(
            ruin=1, x=row["x"], y=row["y"]
        )


def army_state(graph, at, features, fact):
    keys, player = graph.ref.keys, graph.faction(graph.player)
    for row in at("armies"):
        if row["cqi"] is None:
            continue
        character = graph.character(row["cqi"])
        data = {
            name: row.get(name)
            for name in (
                "is_leader",
                "has_army",
                "is_general",
                "rank",
                "x",
                "y",
                "ap_pct",
                "hp",
                "units",
                "in_own_territory",
                "ap_remaining",
                "ap_per_turn",
                "raiding_income",
                "upkeep",
            )
        }
        data.update(
            subtype=graph.key("agent_subtype", row["subtype_id"]),
            agent_type=graph.key("agent_type", row["agent_type_id"]),
            stance=graph.key("stance", row["stance_id"]),
        )
        features.setdefault(character, {}).update(
            {name: value for name, value in data.items() if value is not None}
        )
        fact("controls", player, character)
        if row["region_id"] is not None:
            fact("located", character, graph.region(row["region_id"]))
        if row["stance_id"] is not None:
            fact("in_stance", character, graph.entity("stance", keys["stance"][row["stance_id"]]))
    for row in at("hostiles"):
        faction = graph.faction(keys["faction"][row["faction_id"]])
        features.setdefault(faction, dict(graph.static_features[faction]))
        if row["kind_id"] == 9 or row["cqi"] is None:
            if row["region_id"] is not None:
                region = graph.region(row["region_id"])
                features.setdefault(region, dict(graph.static_features[region])).update(
                    garrison=row["units"], x=row["x"], y=row["y"]
                )
                fact("owns", faction, region)
            continue
        character = graph.character(row["cqi"])
        data = dict(
            category=HOSTILE_KINDS.get(row["kind_id"]),
            faction=keys["faction"][row["faction_id"]],
            visible=row["visible"],
            x=row["x"],
            y=row["y"],
            dist=row["dist"],
            armed_citizenry=row["is_armed_citizenry"],
            units=row["units"],
            hp=row["hp"],
            subtype=graph.key("agent_subtype", row["subtype_id"]),
            agent_type=graph.key("agent_type", row["agent_type_id"]),
            stance=graph.key("stance", row["stance_id"]),
        )
        features.setdefault(character, {}).update(
            {name: value for name, value in data.items() if value is not None}
        )
        fact("controls", faction, character)
        if row["region_id"] is not None:
            fact("located", character, graph.region(row["region_id"]))
    for row in at("enemy_agents"):
        faction = graph.faction(keys["faction"][row["faction_id"]])
        character = graph.character(row["cqi"])
        features.setdefault(character, {}).update(
            x=row["x"],
            y=row["y"],
            at_war=row["at_war"],
            faction=keys["faction"][row["faction_id"]],
            category="agent",
        )
        features.setdefault(faction, dict(graph.static_features[faction]))
        fact("controls", faction, character)


def diplomacy_state(graph, at, features, fact):
    keys, player = graph.ref.keys, graph.faction(graph.player)
    for row in at("relations"):
        other = graph.faction(keys["faction"][row["faction_id"]])
        features.setdefault(other, dict(graph.static_features[other]))
        fact("standing", player, other, row["standing"])
        for column, relation in TREATIES.items():
            if row[column]:
                fact(relation, player, other)
    for row in at("war_graph"):
        source = graph.faction(keys["faction"][row["faction_id"]])
        features.setdefault(source, dict(graph.static_features[source]))
        for enemy in row["at_war_with"] or ():
            target = graph.faction(keys["faction"][enemy])
            features.setdefault(target, dict(graph.static_features[target]))
            fact("war", source, target)
    for row in at("stationed"):
        if row["cqi"] is not None:
            fact("stationed", graph.character(row["cqi"]), graph.region(row["region_id"]))
    for row in at("world"):
        for cqi in row["citizenry"] or ():
            fact("citizenry", player, graph.character(cqi))
