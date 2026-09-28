import numpy as np
import pyarrow as pa

from .domain import CATEGORIES, NUMBERS

DECISION, INTERRUPT = 1, 2
STATUS = {
    "active": "active",
    "locked_due_to_rank": "locked_rank",
    "locked_by_skill": "locked_skill",
    "locked_by_item": "locked_item",
}
HOSTILE_KINDS = {7: "army", 8: "neutral_army", 9: "settlement", 10: "hero", 11: "neutral_hero"}
MERC = {20: 1.0, 21: 2.0, 22: 3.0, 23: 4.0}
GIFTS = {24: 1.0, 25: 2.0, 26: 3.0}
RESULTS = {
    "decisive_victory": 3.0,
    "heroic_victory": 3.0,
    "close_victory": 2.0,
    "victory": 2.0,
    "win": 2.0,
    "pyrrhic_victory": 1.0,
    "valiant_defeat": -1.0,
    "close_defeat": -2.0,
    "defeat": -2.0,
    "loss": -2.0,
    "decisive_defeat": -3.0,
    "crushing_defeat": -3.0,
}
ALLOWED = frozenset(NUMBERS + CATEGORIES)


def by_snapshot(rows):
    return by_key(rows, "snapshot_id")


def by_key(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def number(value):
    if value is None:
        return 0.0
    return float(value)


def fixed(values, width):
    values = np.ascontiguousarray(values, np.float32).reshape(-1, width)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), width)


def put_fact(facts, relation, source, target, *values, tag=None):
    facts.setdefault(relation, {})[(source, target, tag)] = tuple(number(value) for value in values)
