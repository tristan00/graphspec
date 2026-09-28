import re
from dataclasses import dataclass

FIELDS = re.compile(r"([A-Z_]+)\[([^\]]*)\]")
SCALARS = {"TREASURY": "treasury", "DIPLOMATIC_STANDING": "standing", "INFLUENCE": "influence"}


@dataclass(frozen=True)
class PayloadRelation:
    kind: str
    key_field: str
    amount_field: str | None = None

    def add(self, graph, mission, fields, success, fact):
        key = fields[self.key_field]
        amount = fields.get(self.amount_field) if self.amount_field is not None else None
        target = graph.entity("resource", key) if self.kind == "resource" else graph.reference(self.kind, key)
        fact("mission_payload", mission, target, amount, success, tag=(self.kind, key))


RELATIONS = {
    "EFFECT_BUNDLE": PayloadRelation("bundle", "KEY", "DURATION"),
    "ADD_ANCILLARY": PayloadRelation("item", "ANCILLARY_KEY"),
    "FACTION_POOLED_RESOURCE_TRANSACTION": PayloadRelation("resource", "RESOURCE", "AMOUNT"),
    "FACTION_PROVINCE_POOLED_RESOURCE_TRANSACTION": PayloadRelation("resource", "RESOURCE", "AMOUNT"),
    "GIVE_TRAIT": PayloadRelation("trait", "TRAIT_LEVEL_KEY"),
}


def payload(graph, mission, row, data, fact):
    fields = dict(FIELDS.findall(row["value"] or ""))
    kind = row["payload_key"]
    if kind in SCALARS:
        data[SCALARS[kind]] = float(fields["AMOUNT"])
        return
    if kind not in RELATIONS:
        return
    if kind == "GIVE_TRAIT":
        fields["TRAIT_LEVEL_KEY"] = graph.ref.trait_levels[fields["TRAIT_LEVEL_KEY"]]["trait"]
    RELATIONS[kind].add(graph, mission, fields, row["status_key"] == "SUCCESS", fact)
