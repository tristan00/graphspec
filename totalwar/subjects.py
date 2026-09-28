class ActionSubject:
    def __init__(self, graph, actor, taken, facts, data):
        self.graph, self.actor, self.taken, self.data = graph, actor, taken, data
        self.located = {source: target for source, target, _ in facts.get("located", {})}

    def resolve(self, action, key):
        try:
            handler = SUBJECTS[action]
        except KeyError as error:
            raise ValueError("undeclared Total War action type: " + action) from error
        return handler(self, key)

    def unchanged(self, key):
        return self.actor

    def reference(self, kind, key):
        return self.graph.reference(kind, key)

    def skill(self, key):
        return self.reference("skill", key)

    def stance(self, key):
        return self.graph.entity("stance", key)

    def unit(self, key):
        unit, separator, index = key.partition("@")
        if separator:
            self.data["queue_index"] = int(index)
        return self.reference("unit", unit)

    def character(self, key):
        return self.graph.character(int(key.split("cqi:", 1)[1]))

    def hero(self, key):
        action, _, target = key.partition("@")
        self.data["action"] = action
        if target.startswith("cqi:"):
            return self.character(target)
        return self.region(target)

    def region(self, key):
        return self.graph.region(self.graph.ref.ids["region"][key])

    def garrison(self, key):
        return self.region(key.rsplit(":", 1)[-1])

    def location(self, key):
        return self.located.get(self.actor, self.actor)

    def move(self, key):
        if key.startswith("xy:"):
            x, y = key[3:].split(",")
            self.data.update(x=float(x), y=float(y))
        return self.location(key)

    def item(self, key):
        return self.reference("item", key)

    def horde_building(self, key):
        return self.reference("building", key.rsplit("@", 1)[-1])

    def building(self, key):
        self.data["slot_index"] = self.taken["slot_index"]
        return self.reference("building", key)

    def slot(self, key):
        region, _, index = key.partition("@")
        self.data["slot_index"] = int(index)
        return self.graph.slot(region, int(index))

    def edict(self, key):
        return self.graph.entity("edict", key)

    def candidate(self, key):
        subtype, _, index = key.partition("@")
        self.data["candidate_index"] = int(index) if index.isdigit() else None
        identity = ("candidate", "%s:%s@%s" % (self.graph.campaign["campaign_key"], subtype, index))
        if identity in self.graph.entities:
            return self.graph.entities[identity]
        return self.graph.entity("subtype", subtype)

    def research(self, key):
        if key in self.graph.ref.ids["tech_node"]:
            return self.graph.entity("tech", key)
        nodes = [
            name
            for name, row in self.graph.ref.tech_nodes.items()
            if row["technology_key"] == key and ("tech", name) in self.graph.entities
        ]
        if len(nodes) != 1:
            raise ValueError("research action must identify one visible technology node: " + key)
        return self.graph.entity("tech", nodes[0])

    def rite(self, key):
        return self.graph.entity("rite", key)

    def diplomacy(self, key):
        faction, _, term = key.rpartition(":")
        self.data["term"] = term
        return self.graph.faction(faction)


SUBJECTS = {
    "skills": ActionSubject.skill,
    "stance": ActionSubject.stance,
    "recruit_unit": ActionSubject.unit,
    "raise_dead": ActionSubject.unit,
    "recruit_ror": ActionSubject.unit,
    "recruit_blessed": ActionSubject.unit,
    "recruit_imperial": ActionSubject.unit,
    "cancel_recruit": ActionSubject.unit,
    "attack_army": ActionSubject.character,
    "hero_action": ActionSubject.hero,
    "attack_settlement": ActionSubject.region,
    "colonize": ActionSubject.region,
    "garrison": ActionSubject.garrison,
    "leave_garrison": ActionSubject.location,
    "move": ActionSubject.move,
    "items": ActionSubject.item,
    "item_unequip": ActionSubject.item,
    "horde_building": ActionSubject.horde_building,
    "building": ActionSubject.building,
    "building_repair": ActionSubject.slot,
    "building_dismantle": ActionSubject.slot,
    "edict": ActionSubject.edict,
    "recruit_lord": ActionSubject.candidate,
    "recruit_hero": ActionSubject.candidate,
    "research": ActionSubject.research,
    "rites": ActionSubject.rite,
    "diplomacy": ActionSubject.diplomacy,
    "end_turn": ActionSubject.unchanged,
    "noop": ActionSubject.unchanged,
}
