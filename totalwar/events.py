from graphspec.infra.features import token

from .domain import BATTLE, TERMS
from .values import GIFTS, RESULTS, by_key, number


def consequences(graph):
    keys, placed = graph.ref.keys, {}

    def fact(index, relation, source, target, *values, tag=None):
        placed.setdefault(index, {}).setdefault(relation, {})[(source, target, tag)] = tuple(
            number(value) for value in values
        )

    participants = by_key(graph.rows["participants"], "battle_id")
    units = by_key(graph.rows["battle_units"], "battle_id")
    for row in graph.rows["battles"]:
        index = graph.step_at(row["ts"])
        if index == len(graph.snapshots):
            continue
        sides = {}
        for part in participants.get(row["battle_id"], ()):
            if part["char_cqi"] is not None:
                sides.setdefault(part["side"], []).append(graph.character(part["char_cqi"]))
        for attacker in sides.get("attacker", ()):
            for defender in sides.get("defender", ()):
                fact(
                    index,
                    "battle",
                    attacker,
                    defender,
                    RESULTS.get(row["attacker_result"], 0),
                    RESULTS.get(row["defender_result"], 0),
                    *[row[name] for name in BATTLE[2:10]],
                    row["autoresolved"],
                    row["siege"],
                    row["naval"],
                    row["ambush"],
                    row["night"],
                    token(row["battle_type"] or "") % 97,
                    tag=row["battle_id"],
                )
            if row["region_id"] is not None:
                fact(index, "battle_at", attacker, graph.region(row["region_id"]), tag=row["battle_id"])
        for unit in units.get(row["battle_id"], ()):
            members = participants.get(row["battle_id"], ())
            owner = next(
                (
                    part
                    for part in members
                    if part["side"] == unit["side"] and part["ord"] == unit["part_ord"]
                ),
                None,
            )
            if owner and owner["char_cqi"] is not None and unit["unit_id"] is not None:
                fact(
                    index,
                    "fought_with",
                    graph.character(owner["char_cqi"]),
                    graph.reference("unit", keys["unit"][unit["unit_id"]]),
                    unit["side"] == "attacker",
                    tag=(row["battle_id"], unit["ord"]),
                )
    player = graph.faction(graph.player)
    for row in graph.rows["diplomacy"]:
        if graph.ref.enum.get(row["kind_id"]) != "deal" or row["faction_id"] is None:
            continue
        index = graph.step_at(row["ts"])
        if index == len(graph.snapshots):
            continue
        terms = {graph.ref.enum.get(term) for term in row["term_ids"] or ()}
        gift = GIFTS.get(row["gift_id"])
        if gift:
            terms.add(("gift_small", "gift_medium", "gift_large")[int(gift) - 1])
        fact(
            index,
            "proposed",
            player,
            graph.faction(keys["faction"][row["faction_id"]]),
            row["accepted"],
            row["ok"],
            row["success_chance"],
            gift,
            None,
            None,
            *[term in terms for term in TERMS],
            tag=row["event_id"],
        )
    for row in graph.rows["events"]:
        index = graph.step_at(row["ts"])
        if index == len(graph.snapshots):
            continue
        if (
            row["kind"] == "ancillary_gained"
            and row["ancillary_id"] is not None
            and row["character_cqi"] is not None
        ):
            fact(
                index,
                "gained_item",
                graph.character(row["character_cqi"]),
                graph.reference("item", keys["ancillary"][row["ancillary_id"]]),
                tag=row["event_id"],
            )
        elif row["dilemma_id"] is not None or row["incident_id"] is not None:
            node = graph.dilemma(row)
            fact(index, "faced", player, node, row["kind"] == "dilemma_choice_made", tag=row["event_id"])
    return placed
