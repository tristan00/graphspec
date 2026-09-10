import hashlib
from functools import lru_cache

import numpy as np

from .spec import NODES as KINDS, RECIPES, WIDTH
NUMBERS = tuple('turn income settlements treasury is_researching armies lord_level allies vassals power_rank ll_wounded defeated difficulty net_income expenditure upkeep trade_value tax_level influence food_production food_consumption imperium_level settlement_income province_income raiding_income rank skill_points units pending_recruits ap_pct ap_remaining ap_per_turn hp loyalty x y garrisoned besieging acted is_leader wounded is_agent can_embed xp xp_next_level subterfuge zeal authority resurrection_turns settlement_present complete_owner max_slots free_slots can_set_edict public_order buildings is_capital settlement_level growth_per_turn gross_income development_points has_port has_walls research_points visible dist is_armed_citizenry in_own_territory has_army is_general abandoned capital standing level rank_required tier total_levels strength_pct cost repair_cost pool_avail turns_left count slot_index empty available researched progress'.split())
CATEGORIES = tuple('faction game_version campaign_map subtype agent_type stance region province region_owner kind background_skill selected_edict active_edict current_research key category owner action_type actor_kind'.split())
SIGNALS = tuple('treasury income upkeep net_income settlements armies lord_level allies vassals power_rank research_points public_order growth_per_turn units hp ap_pct skill_points trade_value food_production food_consumption'.split())
DESIGNS = tuple(RECIPES)


@lru_cache(maxsize=200000)
def token(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), 'little')


def encode(kind, data):
    fields = tuple(sorted((key, value) for key, value in data.items() if key in ALLOWED and isinstance(value, (str, int, float))))
    return encoded(kind, fields)


ALLOWED = frozenset(NUMBERS + CATEGORIES)


@lru_cache(maxsize=250000)
def encoded(kind, fields):
    data = dict(fields)
    values = np.zeros(WIDTH, np.float32)
    present = np.zeros(WIDTH, np.float32)
    for field in NUMBERS:
        value = data.get(field)
        if isinstance(value, (int, float)) and np.isfinite(value):
            value = float(value)
            code = token(kind + '.' + field)
            values[code % WIDTH] += (1 if code & 128 else -1) * np.sign(value) * np.log1p(abs(value))
            present[code % WIDTH] = 1
    for field in CATEGORIES:
        value = data.get(field)
        if isinstance(value, (str, int)):
            code = token(kind + '.' + field + '=' + str(value))
            values[code % WIDTH] += 1 if code & 128 else -1
            present[code % WIDTH] = 1
    code = token('kind=' + kind)
    values[code % WIDTH] += 1
    present[code % WIDTH] = 1
    return np.concatenate((np.sign(values) * np.log1p(np.abs(values)), present)).astype(np.float16)


def items(value):
    if isinstance(value, dict):
        return [(str(k), dict(v, key=v.get('key', str(k))) if isinstance(v, dict) else {'key': str(k), 'value': v}) for k, v in sorted(value.items())]
    if isinstance(value, list):
        return [(str(i), v if isinstance(v, dict) else {'key': str(v)}) for i, v in enumerate(value)]
    return []


def identity(kind, data, key):
    if kind == 'character':
        return str(data.get('cqi', key))
    if kind in ('region', 'settlement'):
        return str(data.get('region', key))
    if kind == 'province':
        return str(data.get('province', key))
    if kind == 'faction':
        return str(data.get('faction', key))
    return str(key)
