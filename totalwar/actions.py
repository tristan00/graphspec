from graphspec.infra.features import token

from .subjects import ActionSubject
from .values import DECISION, RESULTS, number


def query(graph, snapshot, features, facts):
    keys, player = graph.ref.keys, graph.faction(graph.player)
    if snapshot["kind_id"] == DECISION:
        row = graph.one("taken", snapshot)
        if row is None:
            return "no_action", player, player, player, dict(action_type="no_action")
        action_type, key = graph.ref.actions[row["action_id"]]
        entity = next(
            (item for item in graph.at("entities", snapshot) if item["entity_seq"] == row["entity_seq"]), None
        )
        actor, context = player, player
        if entity and entity["cqi"] is not None:
            actor = graph.character(entity["cqi"])
            region = next(
                (target for (source, target, _) in facts.get("located", {}) if source == actor), None
            )
            context = region if region is not None else player
        elif entity and entity["region_id"] is not None:
            actor = graph.region(entity["region_id"])
            province = graph.province(keys["region"][entity["region_id"]])
            context = graph.entity("province", province) if province else player
        data = dict(action_type=action_type, key=key)
        subject = ActionSubject(graph, actor, row, facts, data).resolve(action_type, key)
        for node in (actor, subject, context):
            features.setdefault(node, dict(graph.static_features.get(node, {})))
        return action_type, actor, subject, context, data
    interrupt = graph.one("interrupts", snapshot)
    if interrupt is None:
        raise ValueError("interrupt snapshot is missing its interrupt record")
    kind = graph.ref.enum[interrupt["kind_id"]]
    data = dict(
        action_type=kind, option=interrupt["chosen"], root=interrupt["root"], answer=interrupt["answer"]
    )
    actor = subject = context = player
    panel = graph.one("battle_panels", snapshot)
    diplo = graph.one("diplo_panels", snapshot)
    if panel:
        if panel["ally_cqi"] is not None:
            actor = graph.character(panel["ally_cqi"])
        if panel["enemy_cqi"] is not None:
            subject = graph.character(panel["enemy_cqi"])
        data.update(
            n_ally_armies=panel["n_ally_armies"],
            n_enemy_armies=panel["n_enemy_armies"],
            result=panel["result_state"],
        )
        if actor != player and subject != player:
            facts.setdefault("battle_report", {})[(actor, subject, snapshot["snapshot_id"])] = (
                number(RESULTS.get(panel["result_state"], 0)),
                number(token(panel["casualties_state"] or "") % 97),
            )
    if diplo:
        data.update(
            attitude=diplo["attitude"],
            race=diplo["race"],
            amount_offered=diplo["amount_offered"],
            amount_demanded=diplo["amount_demanded"],
            settlements_offered=diplo["settlements"],
        )
    if interrupt["region_id"] is not None:
        context = graph.region(interrupt["region_id"])
        if subject == player:
            subject = context
    if interrupt["root_context"] in graph.ref.ids["faction"]:
        subject = graph.faction(interrupt["root_context"])
    if interrupt["dilemma_id"] is not None or interrupt["incident_id"] is not None:
        subject = graph.dilemma(interrupt)
    for node in (actor, subject, context):
        features.setdefault(node, dict(graph.static_features.get(node, {})))
    return kind, actor, subject, context, data
