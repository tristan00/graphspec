import argparse
import logging
from pathlib import Path
import time

import numpy as np

from graph_tuning_infra.database import connect, query
from graph_tuning_infra.files import read_json, write_json
from graph_tuning_infra.log import configure, emit
from graph_tuning_infra.postgres import connect as postgres, records, rows

from .domain import PARTS, TARGET
from .reward import held


LOG = logging.getLogger(__name__)
COUNTS = {
    'skill': "SELECT count(*) FROM corpus.char_state h JOIN corpus.skill_set_member m ON m.set_id = h.skill_set_id WHERE h.snapshot_id = %(snapshot)s",
    'unit': "SELECT count(*) FROM corpus.char_state h JOIN corpus.unit_card_set_member m ON m.set_id = h.unit_card_set_id WHERE h.snapshot_id = %(snapshot)s",
    'stance_option': "SELECT count(*) FROM corpus.char_state h JOIN corpus.stance_set_member m ON m.set_id = h.stance_set_id WHERE h.snapshot_id = %(snapshot)s",
    'recruitable': "SELECT count(*) FROM corpus.char_state h JOIN corpus.recruitable_set_member m ON m.set_id = h.recruitable_set_id WHERE h.snapshot_id = %(snapshot)s",
    'mercenary': "SELECT count(*) FROM corpus.char_state h JOIN corpus.merc_pool_set_member m ON m.set_id = h.merc_pool_set_id WHERE h.snapshot_id = %(snapshot)s",
    'tech_state': "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.tech_set_member m ON m.set_id = h.tech_set_id WHERE h.snapshot_id = %(snapshot)s",
    'rite': "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.rite_set_member m ON m.set_id = h.rite_set_id WHERE h.snapshot_id = %(snapshot)s",
    'mission': "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.mission_set_member m ON m.set_id = h.mission_set_id WHERE h.snapshot_id = %(snapshot)s",
    'lord_pool': "SELECT count(*) FROM corpus.campaign_state h JOIN corpus.lord_pool_set_member m ON m.set_id = h.lord_pool_set_id WHERE h.snapshot_id = %(snapshot)s",
    'can_build': "SELECT count(*) FROM corpus.province_state h JOIN corpus.buildable_set_member m ON m.set_id = h.buildable_set_id WHERE h.snapshot_id = %(snapshot)s",
    'corruption': "SELECT count(*) FROM corpus.province_state h JOIN corpus.resource_set_member m ON m.set_id = h.corruption_set_id WHERE h.snapshot_id = %(snapshot)s",
    'standing': "SELECT count(*) FROM corpus.snapshot_world h JOIN corpus.relation_set_member m ON m.set_id = h.relation_set_id WHERE h.snapshot_id = %(snapshot)s",
    'at_war': "SELECT count(*) FROM corpus.snapshot_world h JOIN corpus.relation_set_member m ON m.set_id = h.relation_set_id WHERE h.snapshot_id = %(snapshot)s AND m.at_war",
    'war': "SELECT coalesce(sum(cardinality(m.at_war_with)), 0) FROM corpus.snapshot_world h JOIN corpus.war_graph_set_member m ON m.set_id = h.war_graph_set_id WHERE h.snapshot_id = %(snapshot)s",
    'resource': "SELECT count(*) FROM corpus.snapshot_campaign h JOIN corpus.resource_set_member m ON m.set_id = h.resource_set_id WHERE h.snapshot_id = %(snapshot)s",
    'located': "SELECT (SELECT count(*) FROM corpus.world_army a WHERE a.snapshot_id = %(snapshot)s AND a.cqi IS NOT NULL AND a.region_id IS NOT NULL) + "
               "(SELECT count(*) FROM corpus.world_hostile w WHERE w.snapshot_id = %(snapshot)s AND w.cqi IS NOT NULL AND w.kind_id <> 9 AND w.region_id IS NOT NULL)",
}
CARRIED = ('skill', 'unit', 'stance_option', 'recruitable', 'mercenary', 'tech_state', 'rite', 'mission', 'lord_pool', 'can_build', 'corruption')
LABEL = ("SELECT b.turn, b.settlements, b.lord_level, b.allies, b.vassals FROM corpus.snapshot s JOIN corpus.snapshot_campaign b ON b.snapshot_id = s.snapshot_id "
         "WHERE s.snapshot_id = %(snapshot)s")
OPEN = "SELECT turn, settlements, lord_level, allies, vassals FROM corpus.turn_open WHERE campaign_id = %(campaign)s"


def expected_label(connection, campaign, snapshot):
    base = records(connection, LABEL, dict(snapshot=snapshot))[0]
    opened = records(connection, OPEN, dict(campaign=campaign))
    total = 0.0
    for part, weight in PARTS.items():
        future = [row[part] or 0.0 for row in opened if row['turn'] >= base['turn']]
        if base[part] is None or not future:
            continue
        total += weight * (max(future) - float(base[part]))
    return total


def verify(task_path, host, port, database, user, samples, seed):
    started = time.perf_counter()
    task = read_json(task_path)
    root = Path(task_path).parent
    build = read_json(root / (task['name'] + '.build.json'))
    store, ladybug = connect(root / task['database'], read_only=True)
    corpus = postgres(host, port, database, user)
    campaigns = {row['campaign_key']: row['campaign_id'] for row in records(corpus, 'SELECT campaign_id, campaign_key FROM corpus.campaign')}
    rng = np.random.default_rng(seed)
    queries = query(ladybug, 'MATCH (q:query) RETURN q.graph AS graph, q.step AS step, q.key AS snapshot, q.category AS category').to_pylist()
    kinds = {row['snapshot']: row['kind'] for row in records(corpus, 'SELECT snapshot_id AS snapshot, kind_id AS kind FROM corpus.snapshot WHERE snapshot_id = ANY(%(ids)s)',
                                                              dict(ids=[row['snapshot'] for row in queries]))}
    chosen = [queries[i] for i in rng.choice(len(queries), min(samples, len(queries)), replace=False)]
    mismatches, checked = [], 0
    for row in chosen:
        name = build['graphs'][row['graph']]['campaign']
        campaign = campaigns[name]
        decision = kinds[row['snapshot']] == 1
        for relation, statement in COUNTS.items():
            if not decision and relation in CARRIED:
                continue
            expected = int(rows(corpus, statement, dict(snapshot=row['snapshot']))[0][0])
            found = query(ladybug, 'MATCH (a)-[r:`edge_%s`]->(b) WHERE a.graph = $graph AND r.step = $step AND r.present RETURN count(r) AS n' % relation,
                          dict(graph=row['graph'], step=row['step'])).to_pylist()[0]['n']
            checked += 1
            if expected != found:
                mismatches.append(dict(graph=row['graph'], step=row['step'], snapshot=row['snapshot'], relation=relation, expected=expected, found=found))
        label = query(ladybug, 'MATCH (t:`%s`)-[:`%s_of`]->(q:query) WHERE q.graph = $graph AND q.step = $step RETURN t.y AS y, t.split AS split' % (TARGET, TARGET),
                      dict(graph=row['graph'], step=row['step'])).to_pylist()
        if decision:
            expected = expected_label(corpus, campaign, row['snapshot'])
            checked += 1
            if not label or abs(label[0]['y'] - expected) > 1e-9:
                mismatches.append(dict(graph=row['graph'], step=row['step'], snapshot=row['snapshot'], relation='label', expected=expected, found=label))
        if label and label[0]['split'] != ('validation' if held(name) else 'train'):
            mismatches.append(dict(graph=row['graph'], step=row['step'], relation='split', expected=held(name), found=label[0]['split']))
    ladybug.close()
    store.close()
    corpus.close()
    result = dict(samples=len(chosen), checks=checked, mismatches=mismatches, seconds=time.perf_counter() - started)
    write_json(root / (task['name'] + '.verify.json'), result)
    emit('verify_exit', samples=len(chosen), checks=checked, mismatches=len(mismatches), seconds=result['seconds'])
    if mismatches:
        raise SystemExit('%d of %d checks disagree with the corpus; see %s' % (len(mismatches), checked, root / (task['name'] + '.verify.json')))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=55433)
    parser.add_argument('--database', default='tw_stack')
    parser.add_argument('--user', default='tw')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    configure()
    verify(args.task, args.host, args.port, args.database, args.user, args.samples, args.seed)


if __name__ == '__main__':
    main()
