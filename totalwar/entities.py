from dataclasses import dataclass

from .domain import NODES
from .values import ALLOWED


@dataclass(frozen=True)
class ReferenceNode:
    table: str
    fields: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...] = ()

    def read(self, reference, key):
        row = getattr(reference, self.table).get(key)
        if row is None:
            return {}
        return {
            **{name: row[name] for name in self.fields},
            **{output: row[source] for output, source in self.aliases},
        }


REFERENCE_NODES = {
    "skill": ReferenceNode(
        "skills", ("unlocked_at_rank", "influence_cost"), (("is_background", "is_background_skill"),)
    ),
    "trait": ReferenceNode("traits", ("hidden", "precedence", "no_going_back_level")),
    "building": ReferenceNode(
        "buildings",
        (
            "level",
            "create_cost",
            "create_time",
            "upkeep_cost",
            "food_cost",
            "development_point_cost",
            "only_in_capital",
            "can_be_damaged",
            "chain",
            "superchain",
        ),
        (("category", "chain_category"),),
    ),
    "unit": ReferenceNode(
        "units",
        (
            "recruitment_cost",
            "upkeep_cost",
            "create_time",
            "campaign_cap",
            "tier",
            "num_men",
            "is_monstrous",
            "can_siege",
            "caste",
            "category",
            "class",
            "morale",
            "bonus_hit_points",
            "armour",
            "shield",
            "melee_attack",
            "melee_defence",
            "charge_bonus",
            "accuracy",
            "campaign_action_points",
            "num_mounts",
            "weapon",
        ),
    ),
    "item": ReferenceNode(
        "items",
        (
            "type",
            "category",
            "subcategory",
            "transferrable",
            "randomly_dropped",
            "uniqueness_score",
            "can_be_stolen",
        ),
        (("legendary", "legendary_item"),),
    ),
    "bundle": ReferenceNode("bundles", ("bundle_target",), (("is_global", "is_global_effect"),)),
}


class Entities:
    def __init__(self, reference, encode, features, campaign):
        self.ref, self.encode, self.features, self.campaign = reference, encode, features, campaign
        self.entities, self.kinds, self.keys = {}, [], []
        self.static_features, self.static_facts = {}, {}

    def entity(self, kind, key):
        if kind not in NODES or key is None:
            raise ValueError("node requires a declared kind and a non-null key: %s %r" % (kind, key))
        found = self.entities.get((kind, key))
        if found is None:
            found = len(self.kinds)
            self.entities[(kind, key)] = found
            self.kinds.append(kind)
            self.keys.append(str(key))
        return found

    def key(self, family, identifier):
        return self.ref.keys[family][identifier] if identifier is not None else None

    def vector(self, kind, data):
        unknown = set(data) - ALLOWED
        if unknown:
            raise ValueError("undeclared features for %s: %s" % (kind, ", ".join(sorted(unknown))))
        return self.encode(kind, data)

    def static(self, kind, key, data):
        node = self.entity(kind, key)
        if node not in self.static_features:
            self.static_features[node] = dict(data, key=key)
        return node

    def reference(self, kind, key):
        return self.static(kind, key, REFERENCE_NODES[kind].read(self.ref, key))

    def region(self, identifier):
        key = self.key("region", identifier)
        row = self.ref.regions.get(key, {})
        node = self.static("region", key, dict(capital=row.get("is_capital")))
        if row.get("province"):
            province = self.static("province", row["province"], {})
            self.static_facts.setdefault("in_province", {})[(node, province, None)] = ()
        return node

    def province(self, region_key):
        row = self.ref.regions.get(region_key)
        return None if row is None else row["province"]

    def dilemma(self, row):
        families = [("dilemma", row["dilemma_id"]), ("incident", row["incident_id"])]
        selected = [(family, identifier) for family, identifier in families if identifier is not None]
        if len(selected) != 1:
            raise ValueError("event must identify exactly one dilemma or incident")
        family, identifier = selected[0]
        key = self.key(family, identifier)
        metadata = getattr(self.ref, family + "s").get(key)
        data = {} if metadata is None else {name: metadata[name] for name in ("event_category", "is_large")}
        return self.static("dilemma", key, data)

    def faction(self, key):
        row = self.ref.factions.get(key, {})
        return self.static(
            "faction",
            key,
            dict(
                faction=key,
                culture=row.get("culture"),
                subculture=row.get("subculture"),
                category=row.get("category"),
                is_rebel=row.get("is_rebel"),
            ),
        )

    def character(self, cqi):
        return self.entity("character", "%s:%s" % (self.campaign["campaign_key"], cqi))

    def slot(self, owner, index):
        return self.entity("slot", "%s#%s" % (owner, "new" if index is None else index))

    def candidate(self, subtype, index):
        return self.entity("candidate", "%s:%s@%s" % (self.campaign["campaign_key"], subtype, index))
