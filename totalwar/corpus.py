import logging
import time
from dataclasses import dataclass

from graphspec.infra.postgres import records

LOG = logging.getLogger(__name__)
SPINE = "JOIN corpus.snapshot s ON s.snapshot_id = h.snapshot_id WHERE s.campaign_id = %(campaign)s"


@dataclass(frozen=True)
class Holder:
    table: str
    identity: tuple[str, ...] = ()


HOLDERS = {
    "world": Holder("corpus.snapshot_world"),
    "campaign": Holder("corpus.snapshot_campaign"),
    "char": Holder("corpus.char_state", ("character_id",)),
    "ext": Holder("corpus.char_state_ext", ("character_id",)),
    "province": Holder("corpus.province_state", ("region_id",)),
    "state": Holder("corpus.campaign_state"),
}
SETS = (
    ("regions", "world", "region_set_id", "region_set_member"),
    ("settlements", "world", "settlement_set_id", "settlement_set_member"),
    ("ruins", "world", "ruin_set_id", "ruin_set_member"),
    ("enemy_agents", "world", "enemy_agent_set_id", "enemy_agent_set_member"),
    ("war_graph", "world", "war_graph_set_id", "war_graph_set_member"),
    ("relations", "world", "relation_set_id", "relation_set_member"),
    ("stationed", "world", "stationed_set_id", "stationed_set_member"),
    ("resources", "campaign", "resource_set_id", "resource_set_member"),
    ("hero_counts", "campaign", "hero_count_set_id", "hero_count_set_member"),
    ("faction_bundles", "campaign", "effect_bundle_set_id", "effect_bundle_set_member"),
    ("faction_mercenaries", "campaign", "merc_pool_set_id", "faction_merc_set_member"),
    ("faction_effects", "campaign", "effect_set_id", "effect_set_member"),
    ("skills", "char", "skill_set_id", "skill_set_member"),
    ("stances", "char", "stance_set_id", "stance_set_member"),
    ("recruitable", "char", "recruitable_set_id", "recruitable_set_member"),
    ("unit_cards", "char", "unit_card_set_id", "unit_card_set_member"),
    ("pending_queue", "char", "pending_queue_set_id", "pending_queue_set_member"),
    ("equipped", "char", "equipped_set_id", "item_slot_set_member"),
    ("horde_slots", "char", "horde_slot_set_id", "horde_slot_set_member"),
    ("merc_pools", "char", "merc_pool_set_id", "merc_pool_set_member"),
    ("skill_prereqs", "char", "skill_prereq_set_id", "skill_prereq_set_member"),
    ("traits", "ext", "trait_set_id", "trait_set_member"),
    ("trait_progress", "ext", "trait_progress_set_id", "trait_progress_set_member"),
    ("hidden_skills", "ext", "hidden_skill_state_set_id", "hidden_skill_state_set_member"),
    ("character_bundles", "ext", "effect_bundle_set_id", "effect_bundle_set_member"),
    ("force_bundles", "ext", "force_effect_bundle_set_id", "effect_bundle_set_member"),
    ("built", "province", "built_set_id", "built_set_member"),
    ("building_now", "province", "building_now_set_id", "building_now_set_member"),
    ("corruption", "province", "corruption_set_id", "resource_set_member"),
    ("buildable", "province", "buildable_set_id", "buildable_set_member"),
    ("slot_states", "province", "slot_state_set_id", "slot_state_set_member"),
    ("province_bundles", "province", "effect_bundle_set_id", "effect_bundle_set_member"),
    ("plague_bundles", "province", "plague_bundle_set_id", "effect_bundle_set_member"),
    ("income_breakdown", "province", "income_breakdown_set_id", "income_breakdown_set_member"),
    ("tech", "state", "tech_set_id", "tech_set_member"),
    ("rites", "state", "rite_set_id", "rite_set_member"),
    ("lord_pools", "state", "lord_pool_set_id", "lord_pool_set_member"),
    ("candidates", "state", "lord_pool_set_id", "lord_pool_candidate"),
    ("item_pool", "state", "anc_pool_set_id", "item_slot_set_member"),
    ("missions", "state", "mission_set_id", "mission_set_member"),
)
TABLES = dict(
    snapshots="SELECT s.snapshot_id, s.kind_id, s.ts, s.turn, h.* FROM corpus.snapshot_campaign h "
    + SPINE
    + " ORDER BY s.ts, s.snapshot_id",
    world="SELECT h.* FROM corpus.snapshot_world h " + SPINE,
    armies="SELECT h.* FROM corpus.world_army h " + SPINE,
    hostiles="SELECT h.* FROM corpus.world_hostile h " + SPINE,
    entities="SELECT h.*, c.cqi FROM corpus.snapshot_entity h LEFT JOIN corpus.character c ON c.character_id = h.character_id "
    + SPINE,
    characters="SELECT h.*, c.cqi FROM corpus.char_state h JOIN corpus.character c ON c.character_id = h.character_id "
    + SPINE,
    extensions="SELECT h.*, c.cqi FROM corpus.char_state_ext h JOIN corpus.character c ON c.character_id = h.character_id "
    + SPINE,
    provinces="SELECT h.* FROM corpus.province_state h " + SPINE,
    states="SELECT h.* FROM corpus.campaign_state h " + SPINE,
    taken="SELECT h.decision_id AS snapshot_id, h.entity_seq, h.action_id, o.slot_index FROM corpus.taken h "
    "LEFT JOIN corpus.offer o ON o.decision_id = h.decision_id AND o.offer_seq = h.offer_seq WHERE h.campaign_id = %(campaign)s",
    interrupts="SELECT h.interrupt_id AS snapshot_id, h.prev_decision_id, h.state_at_id, h.kind_id, h.root, h.dilemma_id, h.incident_id, "
    "h.root_context, h.region_id, h.chosen, h.answer FROM corpus.interrupt h JOIN corpus.snapshot s ON s.snapshot_id = h.interrupt_id "
    "WHERE s.campaign_id = %(campaign)s",
    battle_panels="SELECT h.interrupt_id AS snapshot_id, h.* FROM corpus.interrupt_battle_panel h JOIN corpus.snapshot s ON s.snapshot_id = h.interrupt_id "
    "WHERE s.campaign_id = %(campaign)s",
    diplo_panels="SELECT h.interrupt_id AS snapshot_id, h.* FROM corpus.interrupt_diplo_panel h JOIN corpus.snapshot s ON s.snapshot_id = h.interrupt_id "
    "WHERE s.campaign_id = %(campaign)s",
    diplomacy="SELECT h.* FROM corpus.diplomacy_event h WHERE h.campaign_id = %(campaign)s ORDER BY h.ts, h.event_id",
    battles="SELECT h.* FROM corpus.battle h WHERE h.campaign_id = %(campaign)s ORDER BY h.ts, h.battle_id",
    participants="SELECT p.* FROM corpus.battle_participant p JOIN corpus.battle b ON b.battle_id = p.battle_id WHERE b.campaign_id = %(campaign)s",
    battle_units="SELECT u.* FROM corpus.battle_unit u JOIN corpus.battle b ON b.battle_id = u.battle_id WHERE b.campaign_id = %(campaign)s",
    events="SELECT h.* FROM corpus.event h WHERE h.campaign_id = %(campaign)s ORDER BY h.ts, h.event_id",
)


def campaigns(connection, count):
    return records(
        connection,
        "SELECT * FROM (SELECT c.campaign_id, c.campaign_key, c.first_snapshot_id, f.key AS faction, c.n_decisions, c.n_interrupts, c.turns "
        "FROM corpus.campaign c JOIN dict.faction f ON f.id = c.faction_id WHERE c.first_snapshot_id IS NOT NULL AND c.n_decisions > 0 "
        "ORDER BY c.first_snapshot_id DESC LIMIT %(count)s) newest ORDER BY first_snapshot_id",
        dict(count=count),
    )


def members(connection, campaign, holder, column, table):
    source = HOLDERS[holder]
    identity = ", ".join("h.%s" % name for name in source.identity)
    return records(
        connection,
        "SELECT h.snapshot_id%s, m.* FROM %s h JOIN corpus.%s m ON m.set_id = h.%s %s"
        % (", " + identity if identity else "", source.table, table, column, SPINE),
        dict(campaign=campaign),
    )


def load(connection, campaign):
    started = time.perf_counter()
    parameters = dict(campaign=campaign)
    result = {name: records(connection, statement, parameters) for name, statement in TABLES.items()}
    for name, holder, column, table in SETS:
        result[name] = members(connection, campaign, holder, column, table)
    LOG.info(
        "corpus load exit campaign=%d seconds=%.3f snapshots=%d rows=%d",
        campaign,
        time.perf_counter() - started,
        len(result["snapshots"]),
        sum(len(rows) for rows in result.values()),
    )
    return result
