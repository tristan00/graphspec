import bisect
import logging
import math
import time

from .actions import query
from .catalogue import catalogue
from .domain import RELATIONS, SIGNALS
from .entities import Entities
from .events import consequences
from .reward import held, label, turn_open
from .state import entities_block
from .tables import tables_of
from .values import DECISION, by_snapshot, number
from .world import world

LOG = logging.getLogger(__name__)


class Campaign(Entities):
    def __init__(self, reference, encode, features, campaign, rows):
        super().__init__(reference, encode, features, campaign)
        self.rows = rows
        self.tables = {
            name: by_snapshot(table) for name, table in rows.items() if table and "snapshot_id" in table[0]
        }
        self.snapshots = rows["snapshots"]
        if not self.snapshots:
            raise ValueError("campaign needs at least one snapshot")
        self.player = reference.keys["faction"][self.snapshots[0]["faction_id"]]
        self.by_id = {row["snapshot_id"]: row for row in self.snapshots}
        self.times = [row["ts"] for row in self.snapshots]
        if self.times != sorted(self.times) or len(self.by_id) != len(self.snapshots):
            raise ValueError("campaign snapshots must have unique ids and chronological order")
        self.opened = turn_open(self.snapshots, DECISION)

    def at(self, name, snapshot):
        return self.tables.get(name, {}).get(snapshot["snapshot_id"], ())

    def one(self, name, snapshot):
        records = self.at(name, snapshot)
        if len(records) > 1:
            raise ValueError("expected one %s row for snapshot %s" % (name, snapshot["snapshot_id"]))
        return next(iter(records), None)

    def step_at(self, ts):
        return bisect.bisect_left(self.times, ts)

    def build(self, counter, index, name):
        started = time.perf_counter()
        catalogue(self)
        placed = consequences(self)
        held_out = held(self.campaign["campaign_key"])
        observations, fact_rows, queries, targets = [], {relation: [] for relation in RELATIONS}, [], []
        previous, carried = {}, None
        for step, snapshot in enumerate(self.snapshots):
            features, facts = {}, {}
            world(self, snapshot, features, facts)
            if snapshot["kind_id"] == DECISION:
                entity_features, entity_facts = {}, {}
                entities_block(self, snapshot, entity_features, entity_facts)
                carried = (entity_features, entity_facts)
            if carried:
                for node, data in carried[0].items():
                    features.setdefault(node, {}).update(data)
                for relation, items in carried[1].items():
                    facts.setdefault(relation, {}).update(items)
            for relation, items in placed.get(step, {}).items():
                facts.setdefault(relation, {}).update(items)
            category, actor, subject, context, data = query(self, snapshot, features, facts)
            for relation, items in self.static_facts.items():
                facts.setdefault(relation, {}).update(items)
            present = {relation: set(items) for relation, items in facts.items()}
            removed = {relation: keys - present.get(relation, set()) for relation, keys in previous.items()}
            for relation, items in facts.items():
                for (source, target, _), values in items.items():
                    features.setdefault(source, dict(self.static_features.get(source, {})))
                    features.setdefault(target, dict(self.static_features.get(target, {})))
            for node in list(features):
                features[node] = dict(self.static_features.get(node, {}), **features[node])
            observed = sorted(features)
            for seq, node in enumerate(observed):
                observations.append((node, step, seq, self.vector(self.kinds[node], features[node])))
            seq = 0
            for relation in RELATIONS:
                for (source, target, _), values in facts.get(relation, {}).items():
                    fact_rows[relation].append((source, target, seq, step, True) + values)
                    seq += 1
            seq = 0
            for relation, keys in removed.items():
                width_r = len(RELATIONS[relation][1])
                for source, target, _ in sorted(keys):
                    fact_rows[relation].append((source, target, seq, step, False) + (math.nan,) * width_r)
                    seq += 1
            previous = present
            signal = [number(snapshot.get(part)) for part in SIGNALS]
            queries.append(
                (
                    snapshot["snapshot_id"],
                    step,
                    snapshot["turn"],
                    category,
                    actor,
                    subject,
                    context,
                    self.vector("query", data),
                    signal,
                    [
                        float(snapshot.get(part) is not None and math.isfinite(float(snapshot[part])))
                        for part in SIGNALS
                    ],
                )
            )
            baseline = self.baseline(snapshot)
            if baseline is not None:
                value = label(self.opened, baseline[0], baseline[1])
                if value is not None:
                    targets.append((step, value))
        LOG.info(
            "campaign build exit campaign=%d seconds=%.3f steps=%d entities=%d observations=%d facts=%d labels=%d",
            self.campaign["campaign_id"],
            time.perf_counter() - started,
            len(self.snapshots),
            len(self.kinds),
            len(observations),
            sum(len(rows) for rows in fact_rows.values()),
            len(targets),
        )
        return tables_of(self, counter, index, name, observations, fact_rows, queries, targets, held_out)

    def baseline(self, snapshot):
        if snapshot["kind_id"] == DECISION:
            return snapshot["turn"], snapshot
        interrupt = self.one("interrupts", snapshot)
        if interrupt is None:
            return None
        if self.ref.enum.get(interrupt["state_at_id"]) == "panel":
            return snapshot["turn"], snapshot
        previous = self.by_id.get(interrupt["prev_decision_id"])
        return (previous["turn"], previous) if previous else None
