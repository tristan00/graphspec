import logging
import time
from functools import partial

from graphspec.infra.postgres import records

from .reference_tables import TABLES

LOG = logging.getLogger(__name__)
FAMILIES = (
    "faction",
    "region",
    "province",
    "skill",
    "unit",
    "building",
    "tech_node",
    "agent_subtype",
    "agent_type",
    "ancillary",
    "ritual",
    "trait",
    "trait_level",
    "mission",
    "mission_issuer",
    "edict",
    "effect_bundle",
    "pooled_resource",
    "stance",
    "effect",
    "effect_scope",
    "dilemma",
    "incident",
    "action_type",
    "unit_category",
)


class Reference:
    def __init__(self, connection):
        started = time.perf_counter()
        read = partial(records, connection)
        self.keys = {
            family: {row["id"]: row["key"] for row in read("SELECT id, key FROM dict.%s" % family)}
            for family in FAMILIES
        }
        self.ids = {
            family: {key: identifier for identifier, key in table.items()}
            for family, table in self.keys.items()
        }
        self.enum = {row["enum_id"]: row["key"] for row in read("SELECT enum_id, key FROM dict.enum")}
        self.actions = {
            row["action_id"]: (row["type"], row["action_key"])
            for row in read(
                "SELECT a.action_id, t.key AS type, a.action_key FROM dict.action a JOIN dict.action_type t ON t.id = a.action_type_id"
            )
        }
        for table in TABLES:
            setattr(self, table.name, table.load(connection))
        self.skill_sets = {}
        for row in read(
            "SELECT key, agent_subtype_key, faction_key FROM ref.character_skill_node_sets WHERE agent_subtype_key IS NOT NULL"
        ):
            choices = self.skill_sets.setdefault(row["agent_subtype_key"], {})
            faction = row["faction_key"] or ""
            if faction in choices:
                raise ValueError("ambiguous skill sets for %s and %s" % (row["agent_subtype_key"], faction))
            choices[faction] = row["key"]
        self.scopes = {
            key: index
            for index, key in enumerate(
                sorted(row["key"] for row in read("SELECT key FROM ref.campaign_effect_scopes"))
            )
        }
        LOG.info(
            "reference exit seconds=%.3f tech_nodes=%d skill_sets=%d buildings=%d units=%d effects=%d",
            time.perf_counter() - started,
            len(self.tech_nodes),
            len(self.skill_sets),
            len(self.buildings),
            len(self.units),
            len(self.effects),
        )

    def tech_set(self, faction):
        row = self.factions.get(faction)
        scopes = [("faction_key", faction)]
        if row is not None:
            scopes.extend((("subculture", row["subculture"]), ("culture", row["culture"])))
        constrained = []
        for field, value in scopes:
            choices = [
                candidate["key"]
                for candidate in self.tech_sets.values()
                if value and candidate[field] == value and all(not candidate[name] for name in constrained)
            ]
            if len(choices) > 1:
                raise ValueError(
                    "ambiguous technology sets for %s=%s: %s" % (field, value, ", ".join(sorted(choices)))
                )
            if choices:
                return choices[0]
            constrained.append(field)
        return None

    def skill_set(self, subtype, faction):
        sets = self.skill_sets.get(subtype)
        if not sets:
            return None
        return sets[faction] if faction in sets else sets.get("")

    def scope(self, key):
        return self.scopes[key]
