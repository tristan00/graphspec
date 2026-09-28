import logging
import time

from graphspec.infra.postgres import records, rows

from .domain import APPROVAL, DISALLOWS, PLAYER, REPORTS, REQUIREMENTS, UNLOCKS, VOCABULARIES

LOG = logging.getLogger(__name__)
KEYS = {
    "country_history": ("country",),
    "state_history": ("state",),
    "building_history": ("building",),
    "company_history": ("company",),
    "market_history": ("market",),
    "market_area_history": ("market_area",),
    "market_area_state": ("market_area", "state"),
    "market_goods": ("market", "goods"),
    "state_goods": ("state", "goods"),
    "building_goods": ("building", "goods"),
    "building_production_method_active": ("building", "production_method"),
    "building_flow": ("building",),
    "state_economy": ("state",),
    "state_population": ("state",),
    "state_status": ("state",),
    "player_policy": (),
    "player_tariff": ("direction", "goods"),
    "player_adjacent_country": ("country",),
    "player_construction_queue_item": ("private", "position"),
    "law_history": ("law",),
    "country_research_history": ("country_research",),
    "country_research_acquired": ("country_research", "technology"),
    "decree_history": ("decree",),
    "institution_history": ("institution",),
    "building_ownership_history": ("building_ownership",),
    "state_pop_need": ("state", "culture_type", "need", "goods"),
    "pop_history": ("pop",),
    "pop_status": ("pop",),
    "culture_history": ("culture",),
    "state_migration": ("state", "direction", "other_state"),
    "world_market_hub": ("state",),
    "shipping_lane_history": ("shipping_lane",),
    "strait_history": ("strait",),
    "strait_endpoint": ("strait",),
    "interest_group_history": ("interest_group",),
    "interest_group_stats": ("interest_group",),
    "character_history": ("character",),
    "character_assigned_role": ("character", "character_role"),
    "political_movement_history": ("political_movement",),
    "interest_group_movement_support": ("interest_group", "political_movement"),
    "power_bloc_history": ("power_bloc",),
    "power_bloc_adopted_principle": ("power_bloc", "power_bloc_principle"),
    "war_history": ("war",),
    "war_participant": ("war", "country"),
    "occupation_history": ("occupation",),
    "occupation_country": ("occupation", "country"),
    "country_ai_history": ("country_ai",),
    "country_ai_goal": ("country_ai", "target_country"),
    "country_ai_region_stance": ("country_ai", "strategic_region"),
    "law_private": ("law",),
    "interest_group_ideology": ("interest_group", "ideology"),
    "interest_group_held_trait": ("interest_group", "interest_group_trait"),
    "interest_group_pop": ("interest_group", "pop"),
    "party_history": ("party",),
}
KEYS.update({table: ("country",) for table in REPORTS["country"]})
KEYS.update({table: ("state",) for table in REPORTS["state"]})
KEYS.update({table: () for table in PLAYER})
OWNED = ("run_id", "seq", "present", "country", "state")
PROVINCES = (
    "SELECT color, state_region, terrain, area_pixels, x, y FROM world_province WHERE world_id = %(world)s"
)
ADJACENCY = (
    "SELECT first_color, second_color, shared_border_pixels FROM world_adjacency WHERE world_id = %(world)s"
)
DEFINITIONS = (
    "SELECT folder, name, field, value FROM game_definition WHERE ruleset_id = %(ruleset)s "
    "AND (field IN ('unlocking_technologies', 'unlocking_laws') OR folder = 'common/laws' "
    "AND field IN ('disallowing_laws', 'group', 'modifier') OR folder = 'common/ideologies' AND field = 'stance') "
    "ORDER BY folder, name, field, position"
)
REQUIRED = (
    "SELECT production_method, kind, key FROM production_method_requirement WHERE ruleset_id = %(ruleset)s "
    "AND kind = ANY(%(kinds)s) ORDER BY production_method, kind, key"
)
MODIFIERS = "SELECT DISTINCT value FROM game_definition WHERE folder = 'common/laws' AND field = 'modifier'"
RUNS = (
    "SELECT run_id, kind, world_id, ruleset_id, player_country, game_speed, end_reason "
    "FROM run WHERE ended_at IS NOT NULL ORDER BY run_id"
)
SNAPSHOTS = "SELECT seq, phase, game_date FROM snapshot WHERE run_id = %(run)s ORDER BY seq"
ACTIONS = (
    "SELECT l.n, l.turn, l.before_seq, a.name, v.name AS advisor FROM loop_action l "
    "JOIN action a ON a.action = l.action JOIN advisor v ON v.advisor = l.advisor "
    "WHERE l.run_id = %(run)s ORDER BY l.n"
)
PARAMS = (
    "SELECT n, name, integer_value, text_value FROM loop_action_param WHERE run_id = %(run)s ORDER BY n, name"
)
PARAMETERS = "SELECT name, value_kind FROM action_parameter ORDER BY name, value_kind"


class Versions:
    def __init__(self, table, keys, items):
        self.table, self.keys = table, keys
        self.items = sorted(items, key=lambda row: row["seq"])
        self.position, self.current = 0, {}

    def advance(self, seq):
        while self.position < len(self.items) and self.items[self.position]["seq"] <= seq:
            row = self.items[self.position]
            key = tuple(row[name] for name in self.keys)
            if row.get("present", True) is False:
                self.current.pop(key, None)
            else:
                self.current[key] = row
            self.position += 1
        return self.current


def columns_of(connection):
    found = {}
    for row in records(
        connection,
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' "
        "AND table_name = ANY(%(tables)s) ORDER BY table_name, ordinal_position",
        dict(tables=list(REPORTS["country"] + REPORTS["state"] + PLAYER)),
    ):
        found.setdefault(row["table_name"], []).append(row["column_name"])
    missing = set(REPORTS["country"] + REPORTS["state"] + PLAYER) - set(found)
    if missing:
        raise ValueError("report tables missing from the database: " + ", ".join(sorted(missing)))
    return {table: [name for name in names if name not in OWNED] for table, names in found.items()}


def names_of(connection):
    return {
        code: {row[0]: row[1] for row in rows(connection, "SELECT %s, name FROM %s" % (code, code))}
        for code in VOCABULARIES
    }


def world_of(connection, world):
    return dict(
        provinces=records(connection, PROVINCES, dict(world=world)),
        adjacency=records(connection, ADJACENCY, dict(world=world)),
    )


def modifiers(text):
    tokens = text.split()
    if len(tokens) % 3 or any(tokens[i + 1] != "=" for i in range(0, len(tokens), 3)):
        raise ValueError("law modifier is not a list of key = value pairs: %r" % text)
    return {
        tokens[i]: 1.0 if tokens[i + 2] == "yes" else 0.0 if tokens[i + 2] == "no" else float(tokens[i + 2])
        for i in range(0, len(tokens), 3)
    }


def modifier_names(connection):
    return sorted({name for (text,) in rows(connection, MODIFIERS) for name in modifiers(text)})


def definitions_of(connection, ruleset):
    started = time.perf_counter()
    found = dict(unlocks=[], disallows=[], stances={}, laws={})
    for row in records(connection, DEFINITIONS, dict(ruleset=ruleset)):
        key, name, value = (row["folder"], row["field"]), row["name"], row["value"]
        if key in UNLOCKS:
            source, target = UNLOCKS[key]
            found["unlocks"].append((source, value, target, name))
        elif key in DISALLOWS:
            source, target = DISALLOWS[key]
            found["disallows"].append((source, name, target, value))
        elif key == ("common/ideologies", "stance"):
            law, stance = value.split()
            if stance not in APPROVAL:
                raise ValueError("unknown ideology stance %r" % stance)
            found["stances"].setdefault(name, []).append((law, stance))
        elif row["field"] == "group":
            found["laws"].setdefault(name, {})["law_type.group"] = value
        elif row["field"] == "modifier":
            found["laws"].setdefault(name, {}).update(
                ("modifier." + modifier, amount) for modifier, amount in modifiers(value).items()
            )
    for row in records(connection, REQUIRED, dict(ruleset=ruleset, kinds=list(REQUIREMENTS))):
        relation, kind = REQUIREMENTS[row["kind"]]
        found[relation].append((kind, row["key"], "production_method", row["production_method"]))
    LOG.info(
        "definitions exit ruleset=%d seconds=%.2f unlocks=%d disallows=%d ideologies=%d laws=%d",
        ruleset,
        time.perf_counter() - started,
        len(found["unlocks"]),
        len(found["disallows"]),
        len(found["stances"]),
        len(found["laws"]),
    )
    return found


def parameters_of(connection):
    kinds = {}
    for row in records(connection, PARAMETERS):
        kinds.setdefault(row["name"], set()).add(row["value_kind"])
    return kinds


def runs(connection, count, chosen):
    found = records(connection, RUNS)
    if chosen:
        by_id = {row["run_id"]: row for row in found}
        missing = [run for run in chosen if run not in by_id]
        if missing:
            raise ValueError("runs are not finished or do not exist: %s" % missing)
        return [by_id[run] for run in chosen]
    return found[:count]


def load(connection, run):
    started = time.perf_counter()
    tables = {}
    for table, keys in KEYS.items():
        items = records(
            connection, "SELECT * FROM %s WHERE run_id = %%(run)s" % table, dict(run=run["run_id"])
        )
        tables[table] = Versions(table, keys, items)
    snapshots = records(connection, SNAPSHOTS, dict(run=run["run_id"]))
    actions = records(connection, ACTIONS, dict(run=run["run_id"]))
    by_n = {row["n"]: row for row in actions}
    for row in actions:
        row["params"] = {}
    for row in records(connection, PARAMS, dict(run=run["run_id"])):
        by_n[row["n"]]["params"][row["name"]] = (
            row["text_value"] if row["integer_value"] is None else row["integer_value"]
        )
    LOG.info(
        "load exit run=%d seconds=%.2f snapshots=%d actions=%d rows=%d",
        run["run_id"],
        time.perf_counter() - started,
        len(snapshots),
        len(actions),
        sum(len(table.items) for table in tables.values()),
    )
    return dict(
        snapshots=snapshots,
        actions=actions,
        tables=tables,
        world=world_of(connection, run["world_id"]),
        definitions=definitions_of(connection, run["ruleset_id"]),
    )
