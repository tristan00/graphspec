import hashlib

from .domain import HELD, PARTS, SALT


def held(campaign_key):
    digest = hashlib.sha1(("%s|%s" % (SALT, campaign_key)).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10000 < HELD


def turn_open(snapshots, decision_kind):
    opened = {}
    for row in sorted(snapshots, key=lambda row: row["snapshot_id"]):
        if row["kind_id"] == decision_kind and row["turn"] not in opened:
            opened[row["turn"]] = {part: row[part] or 0.0 for part in PARTS}
    return opened


def label(opened, turn, baseline):
    future = [values for observed_turn, values in opened.items() if observed_turn >= turn]
    if not future:
        return None
    total, found = 0.0, False
    for part, weight in PARTS.items():
        current = baseline.get(part)
        if current is None:
            continue
        total += weight * (max(values[part] for values in future) - float(current))
        found = True
    return total if found else None
