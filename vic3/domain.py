TARGET = "future"
SERIES = ("gdp", "population", "sol")
SIGNALS = ("log_gdp", "log_population", "sol")
STRATA = ("lower_strata", "middle_strata", "upper_strata")
SALT, HELD = "vic3", 250
HOURS_PER_DAY = 24
HORIZON_DAYS = 5 * 365
LOOP_ADVISOR = "loop"
PRE_ACTION = ("start", "post_run")
CODES = (
    "goods",
    "building_type",
    "production_method",
    "law_type",
    "technology",
    "decree_type",
    "institution_type",
)
ATTITUDES = tuple(range(14))
STANCES = ("stance_colonize_region", "stance_conquer_region", "stance_none", "stance_protect_region")
ENACTING_PHASES = (1, 2)
OWNER_KINDS = {
    "country": ("country", "owner_country"),
    "self": ("building", "owner_building"),
    "manor_house": ("building", "owner_building"),
    "financial_district": ("building", "owner_building"),
    "company": ("building", "owner_building"),
    "company_regional_headquarter": ("building", "owner_building"),
}
LAND = "land"
NODES = (
    "country",
    "state",
    "market",
    "building",
    "queue_entry",
    "company",
    "block",
    "action",
    "stratum",
    "world_market_hub",
    "interest_group",
    "character",
    "political_movement",
    "power_bloc",
    "war",
    "law",
    "strategic_region",
    "ideology",
    "party",
) + CODES
RELATIONS = {
    "in_block": ((("action", "block"),), ()),
    "targets": (
        tuple(
            ("action", kind)
            for kind in (
                "country",
                "state",
                "building",
                "company",
                "character",
                "interest_group",
                "political_movement",
                "war",
            )
            + CODES
        ),
        (),
    ),
    "owns": ((("country", "state"),), ()),
    "adjacent_country": ((("country", "country"),), ()),
    "import_tariff": ((("country", "goods"),), ("level",)),
    "export_tariff": ((("country", "goods"),), ("level",)),
    "market_owner": ((("market", "country"),), ()),
    "market_goods": ((("market", "goods"),), ("price", "buy_orders", "sell_orders")),
    "in_market": ((("state", "market"),), ("port",)),
    "local_price": ((("state", "goods"),), ("local_price",)),
    "produces_state": ((("state", "goods"),), ("production",)),
    "consumes_state": ((("state", "goods"),), ("consumption",)),
    "produces": ((("building", "goods"),), ("output",)),
    "consumes": ((("goods", "building"),), ("input",)),
    "uses": ((("building", "production_method"),), ()),
    "in_state": ((("building", "state"),), ()),
    "queued": ((("queue_entry", "state"),), ()),
    "will_be": ((("queue_entry", "building_type"),), ()),
    "has_law": ((("country", "law_type"),), ("progress",)),
    "enacting": ((("country", "law_type"),), ("progress",)),
    "researched": ((("country", "technology"),), ()),
    "researching": ((("country", "technology"),), ()),
    "decreed": ((("state", "decree_type"),), ()),
    "institution": ((("country", "institution_type"),), ("investment",)),
    "runs": ((("company", "building"),), ()),
    "owned_by": ((("building", "building"), ("building", "country")), ("levels",)),
    "needs": ((("state", "goods"),), ("weight",)),
    "unlocks": (
        (
            ("technology", "building_type"),
            ("technology", "law_type"),
            ("technology", "decree_type"),
            ("technology", "technology"),
            ("technology", "production_method"),
            ("law_type", "law_type"),
            ("law_type", "decree_type"),
            ("law_type", "production_method"),
            ("production_method", "production_method"),
        ),
        (),
    ),
    "disallows": ((("law_type", "law_type"), ("law_type", "production_method")), ()),
    "stance": ((("ideology", "law_type"),), ("approval",)),
    "holds": ((("interest_group", "ideology"),), ()),
    "governs": ((("interest_group", "country"),), ()),
    "party_of": ((("party", "country"),), ()),
    "member_of": ((("interest_group", "party"),), ()),
    "backs": ((("stratum", "interest_group"),), ("members",)),
    "lives_in": ((("stratum", "state"),), ()),
    "works_in": ((("stratum", "building_type"),), ("share",)),
    "migrates_to": ((("state", "state"),), ()),
    "adjacent": ((("state", "state"),), ("border",)),
    "hub": ((("state", "world_market_hub"),), ()),
    "lane": ((("state", "state"),), ("convoy_cost",)),
    "strait": ((("state", "state"),), ("toll",)),
    "represents": ((("interest_group", "country"),), ()),
    "leads": ((("character", "interest_group"),), ()),
    "rules": ((("character", "country"),), ()),
    "supports": ((("political_movement", "interest_group"),), ("clout",)),
    "movement_of": ((("political_movement", "country"),), ()),
    "in_bloc": ((("country", "power_bloc"),), ()),
    "in_war": ((("country", "war"),), ("war_support",)),
    "occupies": ((("country", "state"),), ("fraction",)),
    "pending": ((("country", "law"),), ()),
    "of_type": ((("law", "law_type"),), ()),
}
RELATIONS.update(
    {"attitude_%d" % code: ((("country", "country"),), ("neighbour", "relevance")) for code in ATTITUDES}
)
RELATIONS.update({stance: ((("country", "strategic_region"),), ()) for stance in STANCES})
APPROVAL = dict(strongly_disapprove=-2.0, disapprove=-1.0, neutral=0.0, approve=1.0, strongly_approve=2.0)
REQUIREMENTS = {
    "unlocking_technologies": ("unlocks", "technology"),
    "unlocking_laws": ("unlocks", "law_type"),
    "unlocking_production_methods": ("unlocks", "production_method"),
    "disallowing_laws": ("disallows", "law_type"),
}
DISALLOWS = {("common/laws", "disallowing_laws"): ("law_type", "law_type")}
UNLOCKS = {
    ("common/buildings", "unlocking_technologies"): ("technology", "building_type"),
    ("common/laws", "unlocking_technologies"): ("technology", "law_type"),
    ("common/decrees", "unlocking_technologies"): ("technology", "decree_type"),
    ("common/technology/technologies", "unlocking_technologies"): ("technology", "technology"),
    ("common/laws", "unlocking_laws"): ("law_type", "law_type"),
    ("common/decrees", "unlocking_laws"): ("law_type", "decree_type"),
}
VOCABULARIES = CODES + (
    "terrain",
    "character_role",
    "power_bloc_principle",
    "strategic_region",
    "pop_type",
    "culture_type",
    "religion",
    "ideology",
    "interest_group_type",
    "political_movement_type",
    "power_bloc_identity",
    "commander_rank",
    "state_region",
    "ai_region_stance",
    "interest_group_trait",
)
REPORTS = {
    "country": (
        "country_economy_report",
        "country_construction_report",
        "country_population_report",
        "country_diplomacy_report",
        "country_standard_of_living_report",
        "country_politics_report",
        "country_military_report",
    ),
    "state": (
        "state_economy_report",
        "state_population_report",
        "state_status_report",
        "state_standard_of_living_report",
    ),
}
PLAYER = (
    "player_treasury",
    "player_economy",
    "player_construction",
    "player_politics",
    "player_research",
    "player_spending",
    "player_population",
    "player_diplomacy",
)
SUBJECTS = {
    "state": {
        "state_economy": (
            "arable_land",
            "infrastructure",
            "infrastructure_usage",
            "trade_capacity",
            "trade_capacity_usage",
            "treaty_port",
        ),
        "state_population": (
            "lower_strata",
            "middle_strata",
            "upper_strata",
            "radicals",
            "loyalists",
            "subsisting_workforce",
            "unemployed_workforce",
            "last_week_emigration",
            "last_week_immigration",
        ),
        "state_status": ("incorporation", "colony_progress", "devastation"),
    },
    "interest_group": {
        "interest_group_stats": (
            "approval",
            "political_strength",
            "radicals_political_strength",
            "loyalists_political_strength",
            "clout",
        ),
    },
    "building": {
        "building_flow": (
            "input_goods_shortage",
            "cash_reserves",
            "staffing",
            "throughput",
            "salary_rate",
            "dividends",
            "goods_cost",
            "goods_sales",
            "balance_subsidies",
        ),
    },
}
HISTORY = {
    "country": (("in_civil_war", "bloc_leader"), ("government", "country_rank", "country_type", "religion")),
    "state": (("treaty_port_potential",), ("state_region",)),
    "building": (("levels", "active", "subsidized", "dead"), ("building_type",)),
    "company": (("prosperity", "investment_charter", "trade_charter"), ("company_type",)),
    "interest_group": (("in_government",), ("interest_group_type", "approval_state", "influence_type")),
    "character": (("noble", "adult", "prominence"), ("ideology", "rank", "religion")),
    "political_movement": (("radicalism",), ("political_movement_type", "ideology", "managed_status")),
    "power_bloc": (("cohesion", "mandate_progress"), ("power_bloc_identity",)),
    "war": (("days_since_exhaustion", "dead"), ()),
    "party": (("votes",), ("party_type",)),
    "law": (("progress", "phase"), ()),
    "world_market_hub": (("trade_centers",), ()),
}
STRATUM_SUMS = ("workforce", "dependents", "political_strength", "num_literate", "loyalists_and_radicals")
STRATUM_MEANS = ("wealth", "job_satisfaction", "food_security")
STATE_SHAPE = ("x", "y", "area", "coastal", "provinces")
LAW_PRIVATE = (("chance",), ("checkpoint_result",))
POLICY = ("tax_level", "government_wage_level", "military_wage_level", "supply_ship_construction_level")
QUEUE = ("queue_work", "queue_speed", "queue_items")
CHOICE_PARAMS = ("option", "index", "level", "flag", "button", "days", "levels", "slot")
UNMODELLED_PARAMS = (
    "military_formation",
    "journal_entry",
    "treaty",
    "diplomatic_play",
    "amendment",
    "front",
    "ship",
)
GROUPS = {
    "construction": (
        "build",
        "cancel_construction",
        "downsize",
        "pause_construction",
        "resume_construction",
        "pause_all_construction",
        "resume_all_construction",
        "pause_construction_output",
        "resume_construction_output",
        "cancel_queue_item",
        "raise_queue_item",
        "lower_queue_item",
        "first_queue_item",
        "last_queue_item",
    ),
    "buildings": (
        "nationalize",
        "toggle_privatization",
        "toggle_monopoly",
        "subsidies_on",
        "subsidies_off",
        "switch_production_method",
        "subset_subsidies_on_state",
        "subset_subsidies_off_state",
        "bulk_switch_production_method_state",
        "bulk_nationalize_state",
    ),
    "budget": (
        "set_tax_level",
        "set_government_wages",
        "set_military_wages",
        "toggle_consumption_tax",
        "set_import_tariff",
        "set_export_tariff",
        "set_import_subvention",
        "set_export_subvention",
        "declare_bankruptcy",
        "set_strait_toll",
        "set_strait_access",
        "set_strait_allowed",
    ),
    "politics": (
        "enact_law",
        "cancel_enactment",
        "impose_law",
        "repeal_amendment",
        "toggle_bolster_movement",
        "toggle_suppress_movement",
        "negotiate",
        "reform_government",
        "reform_government_party",
        "set_institution_level",
        "stop_institution_change",
        "issue_decree",
        "remove_decree",
    ),
    "research": ("research", "queue_research", "dequeue_research"),
    "army": (
        "defend_invasion",
        "set_formation_flag",
        "station_selected",
        "deploy_selected_to_front",
        "mobilize_selected",
        "demobilize_selected",
        "raise_conscripts_selected",
        "cancel_strategic_objective_selected",
        "set_borrow_setting",
        "transfer_battalions",
        "upgrade_all_units",
        "create_army",
        "recruit_from_formation",
        "activate_conscription_center",
        "disembark",
        "deploy_to_front",
        "station_formation",
        "rehome_formation",
        "set_strategic_objective",
        "cancel_invasion",
        "recruit",
        "mobilize",
        "demobilize",
        "raise_conscripts",
        "disband_formation",
        "mobilize_all",
        "demobilize_all",
        "toggle_mobilization_option",
        "set_supply_priority",
        "disband_regulars",
        "disband_conscripts",
        "upgrade_regulars",
        "upgrade_conscripts",
        "set_default_unit_type",
    ),
    "navy": (
        "recall_fleet",
        "raise_ship_item",
        "lower_ship_item",
        "first_ship_item",
        "last_ship_item",
        "cancel_ship_item",
        "set_default_construction_fleet",
        "clear_default_construction_fleet",
        "decommission_ship",
        "build_ship",
        "build_ship_for",
        "cancel_ship",
        "start_naval_mission",
        "end_naval_mission",
        "start_naval_mission_selected",
        "end_naval_mission_selected",
        "pause_ship_construction",
        "resume_ship_construction",
        "set_supply_ship_construction",
        "design_ship_template",
        "edit_ship_template",
        "retrofit_fleet",
        "cancel_retrofit_fleet",
        "retrofit_all",
        "cancel_retrofit_all",
        "set_default_ship_template",
        "set_fleet_ship_template",
        "delete_ship_template",
    ),
    "characters": (
        "commander_order",
        "hire_commander",
        "hire_commander_for",
        "promote_commander",
        "assign_commander",
        "unassign_commander",
        "character_interaction",
    ),
    "diplomacy": (
        "diplomatic_action",
        "diplomatic_action_obligation",
        "diplomatic_action_state",
        "start_play_confirmed",
        "withdraw_treaty",
        "accept_treaty",
        "decline_treaty",
        "accept_demand",
        "reject_demand",
        "accept_action_proposal",
        "decline_action_proposal",
        "quick_decline_proposal",
        "request_join_power_bloc",
        "form_power_bloc",
        "kick_from_power_bloc",
        "leave_power_bloc",
        "upgrade_principle",
        "pick_principle",
    ),
    "plays": (
        "start_play_country",
        "start_play_state",
        "support_initiator",
        "support_target",
        "lean_initiator",
        "lean_target",
        "stay_on_fence",
        "declare_neutrality",
        "back_down",
        "accept_sway",
        "reject_sway",
        "toggle_press_war_goal",
        "make_primary_demand",
        "add_war_goal",
    ),
    "war": ("propose_peace", "white_peace", "capitulate", "toggle_accept_peace"),
    "companies": ("create_company", "remove_company", "toggle_charter"),
    "expansion": ("colonize", "toggle_pause_colonization", "abandon_colony", "incorporate", "unincorporate"),
    "journal": ("press_journal_button", "choose_event_option", "take_decision", "toggle_ignore_decision"),
    "time": ("pause", "unpause", "set_speed", "run_days"),
}
GROUP = {name: group for group, names in GROUPS.items() for name in names}
EVENT = "choose_event_option"
WAIT = "run_days"
TEXT_TARGETS = {
    "building_type": "building_type",
    "production_method": "production_method",
    "technology": "technology",
    "goods": "goods",
    "decree": "decree_type",
}
ENTITY_TARGETS = {
    "country": "country",
    "state": "state",
    "building": "building",
    "company": "company",
    "character": "character",
    "interest_group": "interest_group",
    "member": "interest_group",
    "political_movement": "political_movement",
    "war": "war",
}
RESOLVED_TARGETS = {"law": "law_type", "institution": "institution_type"}


def field(table, column):
    return table + "." + column


def declared():
    numbers = set(QUEUE) | {
        "position",
        "turn",
        "actions",
        "events",
        "days",
        "queue_private",
        "queue_left",
        "queue_rate",
    }
    categories = {"name", "group", "advisor", "key"}
    for tables in SUBJECTS.values():
        for table, columns in tables.items():
            numbers.update(field(table, column) for column in columns)
    for kind, (values, codes) in HISTORY.items():
        numbers.update(field(kind, column) for column in values)
        categories.update(field(kind, column) for column in codes)
    categories.update(field("player_policy", column) for column in POLICY)
    numbers.update(field("stratum", column) for column in STRATUM_SUMS + STRATUM_MEANS)
    categories.update(field("stratum", column) for column in ("pop_type", "culture_type", "religion"))
    numbers.update(field("state", column) for column in STATE_SHAPE)
    numbers.update(field("law", column) for column in LAW_PRIVATE[0])
    categories.update(field("law", column) for column in LAW_PRIVATE[1])
    numbers.add(field("character", "ruler"))
    categories.add(field("law_type", "group"))
    return numbers, categories


NUMBERS, CATEGORIES = declared()
