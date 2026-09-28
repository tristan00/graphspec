import argparse
import logging
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure, emit
from graphspec.infra.postgres import connect as postgres
from graphspec.infra.postgres import records, rows
from graphspec.infra.staging import Stage
from graphspec.staged import IDS

from .domain import PARTS, TARGET
from .reward import held

LOG = logging.getLogger(__name__)
COUNTS = {
    "skill": "SELECT count(*) FROM corpus.char_state h JOIN corpus.skill_set_member m ON m.set_id = h.skill_set_id WHERE h.snapshot_id = %(snapshot)s",
    "unit": "SELECT count(*) FROM corpus.char_state h JOIN corpus.unit_card_set_member m ON m.set_id = h.unit_card_set_id WHERE h.snapshot_id = %(snapshot)s",
    "stance_option": "SELECT count(*) FROM corpus.char_state h JOIN corpus.stance_set_member m ON m.set_id = h.stance_set_id WHERE h.snapshot_id = %(snapshot)s",
    "recruitable": "SELECT count(*) FROM corpus.char_state h JOIN corpus.recruitable_set_member m ON m.set_id = h.recruitable_set_id WHERE h.snapshot_id = %(snapshot)s",
    "mercenary": "SELECT count(*) FROM corpus.char_state h JOIN corpus.merc_pool_set_member m ON m.set_id = h.merc_pool_set_id WHERE h.snapshot_id = %(snapshot)s",
    "tech_state": "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.tech_set_member m ON m.set_id = h.tech_set_id WHERE h.snapshot_id = %(snapshot)s",
    "rite": "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.rite_set_member m ON m.set_id = h.rite_set_id WHERE h.snapshot_id = %(snapshot)s",
    "mission": "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.mission_set_member m ON m.set_id = h.mission_set_id WHERE h.snapshot_id = %(snapshot)s",
    "lord_pool": "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.lord_pool_set_member m ON m.set_id = h.lord_pool_set_id WHERE h.snapshot_id = %(snapshot)s",
    "can_build": "SELECT count(*) FROM corpus.province_state h JOIN corpus.buildable_set_member m ON m.set_id = h.buildable_set_id WHERE h.snapshot_id = %(snapshot)s",
    "corruption": "SELECT count(*) FROM corpus.province_state h JOIN corpus.resource_set_member m ON m.set_id = h.corruption_set_id WHERE h.snapshot_id = %(snapshot)s",
    "standing": "SELECT count(*) FROM corpus.snapshot_world h JOIN corpus.relation_set_member m ON m.set_id = h.relation_set_id WHERE h.snapshot_id = %(snapshot)s",
    "at_war": "SELECT count(*) FROM corpus.snapshot_world h JOIN corpus.relation_set_member m ON m.set_id = h.relation_set_id WHERE h.snapshot_id = %(snapshot)s AND m.at_war",
    "war": "SELECT coalesce(sum(cardinality(m.at_war_with)), 0) FROM corpus.snapshot_world h JOIN corpus.war_graph_set_member m ON m.set_id = h.war_graph_set_id WHERE h.snapshot_id = %(snapshot)s",
    "resource": "SELECT count(*) FROM corpus.snapshot_campaign h JOIN corpus.resource_set_member m ON m.set_id = h.resource_set_id WHERE h.snapshot_id = %(snapshot)s",
    "located": "SELECT (SELECT count(*) FROM corpus.world_army a WHERE a.snapshot_id = %(snapshot)s AND a.cqi IS NOT NULL AND a.region_id IS NOT NULL) + "
    "(SELECT count(*) FROM corpus.world_hostile w WHERE w.snapshot_id = %(snapshot)s AND w.cqi IS NOT NULL AND w.kind_id <> 9 AND w.region_id IS NOT NULL)",
}
CARRIED = (
    "skill",
    "unit",
    "stance_option",
    "recruitable",
    "mercenary",
    "tech_state",
    "rite",
    "mission",
    "lord_pool",
    "can_build",
    "corruption",
)
LABEL = (
    "SELECT b.turn, b.settlements, b.lord_level, b.allies, b.vassals FROM corpus.snapshot s JOIN corpus.snapshot_campaign b ON b.snapshot_id = s.snapshot_id "
    "WHERE s.snapshot_id = %(snapshot)s"
)
OPEN = "SELECT turn, settlements, lord_level, allies, vassals FROM corpus.turn_open WHERE campaign_id = %(campaign)s"


def expected_label(connection, campaign, snapshot):
    base = records(connection, LABEL, dict(snapshot=snapshot))[0]
    opened = records(connection, OPEN, dict(campaign=campaign))
    total = 0.0
    for part, weight in PARTS.items():
        future = [row[part] or 0.0 for row in opened if row["turn"] >= base["turn"]]
        if base[part] is None or not future:
            continue
        total += weight * (max(future) - float(base[part]))
    return total


class StagedChecks:
    def __init__(self, directory):
        self.stage = Stage.open(directory)
        self.queries = self.read(("query", None, None)).to_pylist()
        targets = {row["id"]: row for row in self.read((TARGET, None, None)).to_pylist()}
        self.labels = {
            row["to"]: targets[row["from"]]
            for row in self.read((TARGET + "_of", TARGET, "query")).to_pylist()
        }

    def read(self, key):
        return pa.concat_tables([pq.read_table(path) for path in self.stage.files[key]])

    def count(self, relation, graph, step):
        filters = [
            ("from", ">=", graph << IDS),
            ("from", "<", (graph + 1) << IDS),
            ("step", "=", step),
            ("present", "=", True),
        ]
        return sum(
            pq.read_table(path, columns=["from"], filters=filters).num_rows
            for key, paths in self.stage.files.items()
            if key[0] == "edge_" + relation
            for path in paths
        )


def compare(source, corpus, build, chosen):
    campaigns = {
        row["campaign_key"]: row["campaign_id"]
        for row in records(corpus, "SELECT campaign_id, campaign_key FROM corpus.campaign")
    }
    kinds = {
        row["snapshot"]: row["kind"]
        for row in records(
            corpus,
            "SELECT snapshot_id AS snapshot, kind_id AS kind FROM corpus.snapshot WHERE snapshot_id = ANY(%(ids)s)",
            dict(ids=[row["key"] for row in chosen]),
        )
    }
    mismatches, checked = [], 0
    graphs = {row["index"]: row["campaign"] for row in build["graphs"]}
    for row in chosen:
        snapshot = row["key"]
        name = graphs[row["graph"]]
        decision = kinds[snapshot] == 1
        for relation, statement in COUNTS.items():
            if not decision and relation in CARRIED:
                continue
            expected = int(rows(corpus, statement, dict(snapshot=snapshot))[0][0])
            found = source.count(relation, row["graph"], row["step"])
            checked += 1
            if expected != found:
                mismatches.append(
                    dict(
                        graph=row["graph"],
                        step=row["step"],
                        snapshot=snapshot,
                        relation=relation,
                        expected=expected,
                        found=found,
                    )
                )
        label = source.labels.get(row["id"])
        if decision:
            expected = expected_label(corpus, campaigns[name], snapshot)
            checked += 1
            if label is None or abs(label["y"] - expected) > 1e-9:
                mismatches.append(
                    dict(
                        graph=row["graph"],
                        step=row["step"],
                        snapshot=snapshot,
                        relation="label",
                        expected=expected,
                        found=label,
                    )
                )
        partition = "validation" if held(name) else "train"
        if label is not None and label["split"] != partition:
            mismatches.append(
                dict(
                    graph=row["graph"],
                    step=row["step"],
                    relation="split",
                    expected=partition,
                    found=label["split"],
                )
            )
    return mismatches, checked


def verify(task_path, host, port, database, user, samples, seed):
    started = time.perf_counter()
    if samples < 1:
        raise ValueError("samples must be positive")
    task = read_json(task_path)
    root = Path(task_path).parent
    build = read_json(root / (task["name"] + ".build.json"))
    source = StagedChecks(root / task["staging"])
    if not source.queries:
        raise ValueError("cannot verify a graph without query rows")
    rng = np.random.default_rng(seed)
    chosen = [
        source.queries[i]
        for i in rng.choice(len(source.queries), min(samples, len(source.queries)), replace=False)
    ]
    with closing(postgres(host, port, database, user)) as corpus:
        mismatches, checked = compare(source, corpus, build, chosen)
    result = dict(
        samples=len(chosen), checks=checked, mismatches=mismatches, seconds=time.perf_counter() - started
    )
    write_json(root / (task["name"] + ".verify.json"), result)
    emit(
        "verify_exit",
        samples=len(chosen),
        checks=checked,
        mismatches=len(mismatches),
        seconds=result["seconds"],
    )
    if mismatches:
        raise SystemExit(
            "%d of %d checks disagree with the corpus; see %s"
            % (len(mismatches), checked, root / (task["name"] + ".verify.json"))
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55433)
    parser.add_argument("--database", default="tw_stack")
    parser.add_argument("--user", default="tw")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    configure()
    verify(args.task, args.host, args.port, args.database, args.user, args.samples, args.seed)


if __name__ == "__main__":
    main()
