from dataclasses import dataclass

from .values import MERC, STATUS


@dataclass(frozen=True)
class CharacterRelation:
    table: str
    relation: str
    kind: str
    family: str
    identity: str
    attributes: tuple[str, ...]
    tag: str | None = None
    tag_prefix: str | None = None

    def add(self, graph, row, character, fact):
        identifier = row[self.identity]
        if identifier is None:
            return
        key = graph.key(self.family, identifier)
        target = graph.entity("stance", key) if self.kind == "stance" else graph.reference(self.kind, key)
        tag = row[self.tag] if self.tag is not None else None
        if self.tag_prefix is not None:
            tag = (self.tag_prefix, tag)
        fact(self.relation, character, target, *[row[field] for field in self.attributes], tag=tag)


CHARACTER_RELATIONS = (
    CharacterRelation(
        "hidden_skills", "hidden_skill", "skill", "skill", "skill_id", ("level", "total_levels")
    ),
    CharacterRelation(
        "traits",
        "trait",
        "trait",
        "trait",
        "trait_id",
        ("level", "points", "threshold_points", "chaos_realm"),
    ),
    CharacterRelation("trait_progress", "trait_progress", "trait", "trait", "trait_id", ("points",)),
    CharacterRelation("equipped", "equipped", "item", "ancillary", "ancillary_id", ("index",), tag="index"),
    CharacterRelation(
        "stances", "stance_option", "stance", "stance", "stance_id", ("active", "can_activate", "can_afford")
    ),
    CharacterRelation("recruitable", "recruitable", "unit", "unit", "unit_id", ("cost", "disabled")),
    CharacterRelation(
        "unit_cards", "unit", "unit", "unit", "unit_id", ("strength_pct", "xp", "ord"), tag="ord"
    ),
    CharacterRelation(
        "pending_queue",
        "recruiting",
        "unit",
        "unit",
        "unit_id",
        ("turns_left", "ord"),
        tag="ord",
        tag_prefix="queue",
    ),
)


def character_collections(graph, at, characters, features, fact):
    for row in at("skills"):
        status = graph.ref.enum[row["status_id"]]
        flags = [float(status == key) for key in STATUS]
        fact(
            "skill",
            characters[row["character_id"]],
            graph.reference("skill", graph.key("skill", row["skill_id"])),
            row["level"],
            row["total_levels"],
            row["tier"],
            row["rank_required"],
            *flags,
        )
    for spec in CHARACTER_RELATIONS:
        for row in at(spec.table):
            spec.add(graph, row, characters[row["character_id"]], fact)
    for row in at("merc_pools"):
        fact(
            "mercenary",
            characters[row["character_id"]],
            graph.reference("unit", graph.key("unit", row["unit_id"])),
            row["avail"],
            row["cost"],
            row["can"],
            MERC[row["action_id"]],
            tag=row["action_id"],
        )
    for row in at("skill_prereqs"):
        fact(
            "skill_prerequisite",
            graph.reference("skill", graph.key("skill", row["parent_skill_id"])),
            graph.reference("skill", graph.key("skill", row["skill_id"])),
        )
    for table, force in (("character_bundles", False), ("force_bundles", True)):
        for row in at(table):
            fact(
                "bundle",
                characters[row["character_id"]],
                graph.reference("bundle", graph.key("effect_bundle", row["effect_bundle_id"])),
                row["turns_remaining"],
                force,
                False,
                tag=table,
            )
    for row in at("horde_slots"):
        slot = graph.slot("%s:%s" % (graph.campaign["campaign_key"], row["character_id"]), row["slot_index"])
        features[slot] = dict(index=row["slot_index"], empty=row["empty"])
        fact("has_slot", characters[row["character_id"]], slot)
        if not row["empty"] and row["building_id"] is not None:
            fact(
                "holds",
                slot,
                graph.reference("building", graph.key("building", row["building_id"])),
                None,
                None,
                None,
                None,
                row["available"],
            )
