NODES = ('faction', 'character', 'region', 'province', 'slot', 'tech', 'skill', 'building', 'unit', 'item', 'trait', 'effect', 'bundle',
         'rite', 'mission', 'edict', 'subtype', 'candidate', 'stance', 'resource', 'dilemma')
TARGET = 'reward'
SIGNALS = ('settlements', 'lord_level', 'allies', 'vassals', 'income', 'treasury', 'armies', 'power_rank')
PARTS = dict(settlements=1.0, lord_level=1.0, allies=1.0, vassals=3.0)
SALT, HELD = 'gnn3', 1500
TERMS = ('declare_war', 'peace', 'nonaggression_pact', 'trade_agreement', 'defensive_alliance', 'soft_access', 'military_alliance', 'vassal',
         'confederation', 'gift_small', 'gift_medium', 'gift_large')
TREATIES = dict(at_war='at_war', allied='allied', trade='trade', nap='non_aggression', mil_access='military_access', mil_ally='military_ally',
                def_ally='defensive_ally', their_vassal='vassal_of', our_master='master_of', excluded='excluded')
GRANTERS = ('tech', 'skill', 'building', 'item', 'trait', 'bundle')
BATTLE = ('attacker_result', 'defender_result', 'attacker_casualties', 'defender_casualties', 'attacker_kills', 'defender_kills',
          'attacker_hp_lost', 'defender_hp_lost', 'attacker_routed', 'defender_routed', 'autoresolved', 'siege', 'naval', 'ambush', 'night', 'battle_type')
RELATIONS = {
    'controls': ((('faction', 'character'),), ()), 'owns': ((('faction', 'region'),), ()), 'in_province': ((('region', 'province'),), ()),
    'adjacent': ((('region', 'region'),), ()), 'located': ((('character', 'region'),), ()), 'stationed': ((('character', 'region'),), ()),
    'can_reach': ((('character', 'region'),), ()), 'can_reach_character': ((('character', 'character'),), ()),
    'has_slot': ((('region', 'slot'), ('character', 'slot')), ()),
    'standing': ((('faction', 'faction'),), ('standing',)), 'war': ((('faction', 'faction'),), ()),
    'at_war': ((('faction', 'faction'),), ()), 'allied': ((('faction', 'faction'),), ()), 'trade': ((('faction', 'faction'),), ()),
    'non_aggression': ((('faction', 'faction'),), ()), 'military_access': ((('faction', 'faction'),), ()), 'military_ally': ((('faction', 'faction'),), ()),
    'defensive_ally': ((('faction', 'faction'),), ()), 'vassal_of': ((('faction', 'faction'),), ()), 'master_of': ((('faction', 'faction'),), ()),
    'excluded': ((('faction', 'faction'),), ()),
    'proposed': ((('faction', 'faction'),), ('accepted', 'ok', 'success_chance', 'gift', 'amount_offered', 'amount_demanded') + TERMS),
    'skill': ((('character', 'skill'),), ('level', 'total_levels', 'tier', 'rank_required', 'active', 'locked_rank', 'locked_skill', 'locked_item')),
    'hidden_skill': ((('character', 'skill'),), ('level', 'total_levels')), 'background': ((('character', 'skill'),), ()),
    'skill_prerequisite': ((('skill', 'skill'),), ()), 'trait': ((('character', 'trait'),), ('level', 'points', 'threshold_points', 'chaos_realm')),
    'trait_progress': ((('character', 'trait'),), ('points',)), 'equipped': ((('character', 'item'),), ('position',)), 'armory': ((('character', 'item'),), ()),
    'unit': ((('character', 'unit'),), ('strength_pct', 'xp', 'position')), 'recruitable': ((('character', 'unit'),), ('cost', 'disabled')),
    'mercenary': ((('character', 'unit'),), ('available', 'cost', 'can', 'pool')), 'recruiting': ((('character', 'unit'),), ('turns_left', 'position')),
    'stance_option': ((('character', 'stance'),), ('active', 'can_activate', 'can_afford')), 'in_stance': ((('character', 'stance'),), ()),
    'of_subtype': ((('character', 'subtype'),), ()),
    'bundle': ((('faction', 'bundle'), ('character', 'bundle'), ('region', 'bundle')), ('turns_remaining', 'force', 'plague')),
    'holds': ((('slot', 'building'),), ('health', 'max_health', 'damaged', 'ruined', 'available')),
    'constructing': ((('slot', 'building'),), ('turns_left', 'paused', 'queued')),
    'can_build': ((('slot', 'building'),), ('cost', 'upkeep', 'level', 'is_upgrade', 'can_afford_resources')),
    'upgrades_to': ((('building', 'building'),), ()), 'requires_building': ((('building', 'building'), ('tech', 'building')), ()),
    'unlocks_unit': ((('building', 'unit'),), ()),
    'edict_option': ((('province', 'edict'),), ('selected', 'active', 'can_set')), 'corruption': ((('region', 'resource'),), ('value',)),
    'income': ((('region', 'resource'),), ('value',)),
    'tech_state': ((('faction', 'tech'),), ('researched', 'can_research', 'cost')), 'researching': ((('faction', 'tech'),), ('research_points',)),
    'tech_prerequisite': ((('tech', 'tech'),), ()), 'rite': ((('faction', 'rite'),), ('can_perform', 'reason')),
    'mission': ((('faction', 'mission'),), ('status', 'turns_remaining', 'is_quest', 'is_victory', 'completed', 'cancelled', 'pending')),
    'issued_by': ((('mission', 'faction'),), ()), 'resource': ((('faction', 'resource'),), ('value',)), 'item_pool': ((('faction', 'item'),), ('position',)),
    'lord_pool': ((('faction', 'subtype'),), ('n',)), 'candidate_of': ((('candidate', 'subtype'),), ('can', 'is_agent')),
    'candidate_skill': ((('candidate', 'skill'),), ()), 'candidate_trait': ((('candidate', 'trait'),), ()),
    'faction_mercenary': ((('faction', 'unit'),), ('base_max', 'current_max')), 'citizenry': ((('faction', 'character'),), ()),
    'grants': (tuple((kind, 'effect') for kind in GRANTERS), ('value', 'scope', 'level', 'value_damaged', 'value_ruined')),
    'skill_link': ((('skill', 'skill'),), ('subset', 'tiers')), 'permits_skill': ((('subtype', 'skill'),), ('tier', 'indent', 'required_parents')),
    'requires_skill': ((('item', 'skill'),), ('level',)),
    'battle': ((('character', 'character'),), BATTLE), 'battle_at': ((('character', 'region'),), ()), 'fought_with': ((('character', 'unit'),), ('side',)),
    'battle_report': ((('character', 'character'),), ('victory', 'casualties')),
    'gained_item': ((('character', 'item'),), ()), 'faced': ((('faction', 'dilemma'),), ('choice',)),
}
NUMBERS = ('turn', 'income', 'treasury', 'settlements', 'armies', 'lord_level', 'allies', 'vassals', 'power_rank', 'is_researching', 'll_wounded',
           'defeated', 'difficulty', 'research_points', 'net_income', 'expenditure', 'upkeep', 'trade_value', 'tax_level', 'influence',
           'food_production', 'food_consumption', 'imperium_level', 'settlement_income', 'province_income', 'raiding_income', 'hero_count',
           'is_rebel', 'is_hero', 'rank', 'xp', 'xp_next_level', 'skill_points', 'hp', 'units', 'loyalty', 'authority',
           'subterfuge', 'zeal', 'resurrection_turns', 'ap_pct', 'ap_remaining', 'ap_per_turn', 'pending_recruits', 'x', 'y', 'acted',
           'besieging', 'garrisoned', 'wounded', 'is_leader', 'is_agent', 'can_embed', 'is_general', 'has_army', 'in_own_territory', 'reach_max',
           'ray0', 'ray1', 'ray2', 'ray3', 'ray4', 'ray5', 'ray6', 'ray7', 'dist', 'visible', 'armed_citizenry', 'at_war', 'capital', 'abandoned',
           'ruin', 'settlement_present', 'settlement_level', 'public_order', 'buildings', 'growth_per_turn', 'gross_income', 'development_points',
           'max_slots', 'free_slots', 'has_port', 'has_walls', 'can_set_edict', 'garrison', 'complete_owner', 'index', 'empty', 'locked', 'health',
           'max_health', 'damaged', 'ruined', 'repairing', 'upgrading', 'dismantling', 'queued', 'refund', 'repair_cost', 'tier', 'indent',
           'research_points_required', 'cost_per_round', 'food_cost', 'resource_cost', 'required_parents', 'is_civil', 'is_engineering',
           'is_military', 'points_on_creation', 'unlocked_at_rank', 'is_background', 'influence_cost', 'level', 'create_cost', 'create_time',
           'upkeep_cost', 'development_point_cost', 'only_in_capital', 'can_be_damaged', 'recruitment_cost', 'campaign_cap', 'num_men',
           'morale', 'bonus_hit_points', 'melee_attack', 'melee_defence', 'charge_bonus', 'accuracy', 'campaign_action_points',
           'num_mounts', 'is_monstrous', 'can_siege', 'legendary', 'transferrable', 'randomly_dropped', 'uniqueness_score', 'can_be_stolen',
           'hidden', 'precedence', 'no_going_back_level', 'is_positive_good', 'is_global', 'cast_time', 'cooldown_time', 'slave_cost',
           'is_quest', 'is_caster', 'cap', 'cost', 'recruitable', 'can_equip', 'move_points', 'faction_total_cap', 'is_large', 'n',
           'sample_index', 'candidate_index', 'total_levels', 'in_progress', 'points_available', 'slot_index', 'is_upgrade', 'can_upgrade',
           'can_afford_resources', 'pool_index', 'queue_index', 'turns_left', 'chance', 'standing', 'allied', 'trade', 'n_ally_armies',
           'n_enemy_armies', 'amount_offered', 'amount_demanded', 'attitude', 'settlements_offered', 'threshold_points', 'points', 'strength_pct')
CATEGORIES = ('faction', 'culture', 'subculture', 'category', 'agent_type', 'subtype', 'stance', 'chain', 'superchain', 'class', 'caste',
              'type', 'subcategory', 'key', 'issuer', 'mission_type', 'event_category', 'bundle_target', 'armour', 'shield', 'weapon',
              'action_type', 'term', 'option', 'action', 'result', 'battle_type', 'root', 'answer', 'race', 'reason', 'status', 'source')


def declaration(features, signals):
    statements = ['CREATE NODE TABLE graph(id INT32 PRIMARY KEY, name STRING)']
    statements += ['CREATE NODE TABLE %s(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING)' % kind for kind in NODES]
    statements.append('CREATE NODE TABLE observation(id INT64 PRIMARY KEY, seq INT32, step INT32, x FLOAT[%d])' % features)
    statements.append('CREATE NODE TABLE query(id INT64 PRIMARY KEY, graph INT32, step INT32, period INT32, key INT64, category STRING, x FLOAT[%d], '
                      'signal FLOAT[%d], signal_observed FLOAT[%d])' % (features, signals, signals))
    statements.append('CREATE NODE TABLE %s(id INT64 PRIMARY KEY, y DOUBLE, split STRING)' % TARGET)
    statements.append('CREATE REL TABLE observed(%s)' % ', '.join('FROM %s TO observation' % kind for kind in NODES))
    statements += ['CREATE REL TABLE query_%s(%s)' % (role, ', '.join('FROM query TO %s' % kind for kind in NODES)) for role in ('actor', 'subject', 'context')]
    for name, (pairs, attributes) in RELATIONS.items():
        columns = ''.join(', %s FLOAT' % attribute for attribute in attributes)
        statements.append('CREATE REL TABLE edge_%s(%s, seq INT32, step INT32, present BOOLEAN%s)' % (
            name, ', '.join('FROM %s TO %s' % pair for pair in pairs), columns))
    statements.append('CREATE REL TABLE %s_of(FROM %s TO query)' % (TARGET, TARGET))
    return statements
