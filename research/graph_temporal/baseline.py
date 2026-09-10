import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .cache import write_json
from .data import Bank
from .run import timed_event


def battle_history(connection, campaigns):
    from advisor import memory

    sql = '''
        SELECT c.campaign_key, s.ts, t.decision_id, ty.key, a.action_key,
               i.chosen, bp.result_state, bp.casualties_text, ch.cqi
        FROM corpus.interrupt i
        JOIN corpus.snapshot s ON s.snapshot_id = i.interrupt_id
        JOIN corpus.campaign c ON c.campaign_id = s.campaign_id
        JOIN dict.enum k ON k.enum_id = i.kind_id AND k.domain = 'interrupt_kind' AND k.key = 'pre_battle'
        JOIN LATERAL (
            SELECT decision_id, action_id, entity_seq, ts FROM corpus.taken
            WHERE campaign_id = s.campaign_id AND ts <= s.ts ORDER BY ts DESC LIMIT 1
        ) t ON TRUE
        JOIN dict.action a ON a.action_id = t.action_id
        JOIN dict.action_type ty ON ty.id = a.action_type_id
        LEFT JOIN corpus.interrupt_battle_panel bp ON bp.interrupt_id = i.interrupt_id
        LEFT JOIN corpus.snapshot_entity se ON se.snapshot_id = t.decision_id AND se.entity_seq = t.entity_seq
        LEFT JOIN corpus.character ch ON ch.character_id = se.character_id
        WHERE c.campaign_key = ANY(%s) AND ty.key IN ('attack_army', 'attack_settlement')
          AND s.ts - t.ts <= %s ORDER BY c.campaign_key, s.ts
    '''
    rows = connection.execute(sql, (campaigns, memory.PB_WINDOW_S)).fetchall()
    army = memory._army_targets(connection, [(r[2], int(r[4].split(':')[1])) for r in rows if r[3] == 'attack_army' and r[4].startswith('cqi:')])
    coords, zones = memory._settlement_targets(connection, [(r[2], r[4]) for r in rows if r[3] == 'attack_settlement'])
    result = defaultdict(list)
    for campaign, stamp, did, kind, key, chosen, outcome, casualties, actor in rows:
        if kind == 'attack_army':
            target = int(key.split(':')[1])
            hit = army.get((did, target))
            params = dict(target_cqi=target)
            zone = None
            if hit:
                params.update(x=hit[0], y=hit[1])
                zone = hit[2]
        else:
            hit = coords.get((did, key))
            params = dict(x=hit[0], y=hit[1]) if hit else {}
            zone = zones.get((did, key)) or key
        result[campaign].append((stamp, str(actor), kind, key, params, chosen, outcome, casualties, zone))
    return result


def prepare(cache, directory, only_holdout=False):
    from advisor.model import F as features, MEM as memory
    from advisor.training_rows import TrainingRows
    from decisions import hydrate
    from decisions.store import DecisionStore

    started = time.perf_counter()
    bank = Bank(cache)
    selected_rows = [bank.rows[i] for i in bank.split['validation_indices']] if only_holdout else bank.rows
    holdout_ids = {bank.rows[i]['decision'] for i in bank.split['validation_indices']}
    prediction_rows = []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    campaigns = sorted({row['campaign'] for row in selected_rows})
    store = DecisionStore(readonly=True)
    timed_event('catboost_features_enter', rows=len(selected_rows), campaigns=len(campaigns), only_holdout=only_holdout)
    try:
        store.con.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY')
        stamps = dict(store.con.execute('SELECT snapshot_id, ts FROM corpus.snapshot WHERE snapshot_id = ANY(%s)', ([r['decision'] for r in selected_rows],)).fetchall())
        history = battle_history(store.con, campaigns)
        with TrainingRows(features) as packed:
            previous_campaign = None
            for offset in range(0, len(selected_rows), hydrate.PREFETCH_CHUNK):
                chunk = selected_rows[offset:offset + hydrate.PREFETCH_CHUNK]
                pre = hydrate.Prefetch(store.con, [r['decision'] for r in chunk], include_offers=False)
                for row in chunk:
                    record = hydrate.record(store.con, row['decision'], legacy=True, pre=pre)
                    if record['campaign_id'] != row['campaign']:
                        raise ValueError('frozen decision changed campaign')
                    if row['campaign'] != previous_campaign:
                        previous_campaign = row['campaign']
                        mem, actions, counts, turns, moves, battle_index = memory.CampaignMemory(), [], {}, {}, {}, 0
                    turn, camp, action = row['turn'], record['campaign'], row['action']
                    turns[turn] = turns.get(turn, 0) + 1
                    camp['act_index'], camp['move_index'] = turns[turn], moves.get(turn, 0) + 1
                    features.stamp_prev_actions(camp, actions)
                    features.stamp_action_counts(camp, counts)
                    mem.begin_turn(turn)
                    battles = history[row['campaign']]
                    while battle_index < len(battles) and battles[battle_index][0] < stamps[row['decision']]:
                        _, actor, kind, key, params, chosen, outcome, casualties, zone = battles[battle_index]
                        mem.note_prebattle('lord', actor, kind, key, params, None, chosen, outcome, casualties, zone=zone)
                        battle_index += 1
                    mem.stamp(camp)
                    entity_seq = action['entity_seq']
                    if entity_seq is None:
                        entity_seq = len(record['entities'])
                        record['entities'].append(dict(context_kind='campaign', context_id=camp['faction'], state=camp, offers=[]))
                    hydrate.attach_taken(store.con, record, entity_seq, action['action_type'], action['key'])
                    selected_entity = record['entities'][entity_seq]
                    if action['slot_index'] is not None:
                        selected_entity['offers'][0]['params']['slot_index'] = action['slot_index']
                    triples = features.decision_rows(record)
                    if len(triples) != 1:
                        raise ValueError('frozen CatBoost decision must produce exactly one feature row')
                    feature = triples[0][2]
                    feature = {k: v for k, v in feature.items() if k in features.MODEL_COLUMNS}
                    if not only_holdout:
                        packed.append(feature)
                    if row['decision'] in holdout_ids:
                        prediction_rows.append(feature)
                    mem.note_pick(selected_entity['context_kind'], selected_entity['context_id'], action['action_type'], selected_entity['state'])
                    if action['action_type'] != 'noop':
                        actions.append(action['action_type'])
                        del actions[:-features.PREV_ACTIONS]
                        features.bump_action_counts(counts, action['action_type'])
                    if action['action_type'] == 'move':
                        moves[turn] = moves.get(turn, 0) + 1
                if offset % (hydrate.PREFETCH_CHUNK * 16) == 0:
                    timed_event('catboost_features_progress', rows=offset + len(chunk), seconds=time.perf_counter() - started)
            if not only_holdout:
                pool, numeric, categorical = packed.pool(bank.targets)
        store.con.execute('COMMIT')
    finally:
        store.close()
    if len(prediction_rows) != len(holdout_ids):
        raise ValueError('raw CatBoost holdout features omitted rows')
    with (directory / 'holdout.pickle').open('wb') as stream:
        pickle.dump(prediction_rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if only_holdout:
        timed_event('catboost_holdout_features_exit', rows=len(prediction_rows), seconds=time.perf_counter() - started)
        return
    quantization = pool.slice(bank.split['train_indices'])
    quantization.quantize(border_count=64, random_seed=0)
    quantization.save_quantization_borders(str(directory / 'borders.tsv'))
    del quantization
    pool.quantize(input_borders=str(directory / 'borders.tsv'))
    pool.save(str(directory / 'features.quantized'))
    write_json(directory / 'features.json', dict(rows=len(bank.rows), numeric=numeric, categorical=categorical,
        supervision_sha256=bank.manifest['supervision_sha256'], seconds=time.perf_counter() - started,
        history='Chosen actions and strictly earlier battle panels, without execution-status filters.'))
    timed_event('catboost_features_exit', rows=len(bank.rows), seconds=time.perf_counter() - started)


def fit(cache, directory):
    from catboost import CatBoostRegressor, Pool
    from advisor.base_model import CB_PARAMS, CB_ITERATIONS, CB_LOSS, CB_EARLY_STOPPING
    from advisor.reward_data import score
    from advisor.model import F

    started = time.perf_counter()
    directory, bank = Path(directory), Bank(cache)
    if (directory / 'result.json').exists():
        raise FileExistsError('fixed baseline already fitted; it must not be sampled repeatedly')
    metadata = json.loads((directory / 'features.json').read_text())
    if metadata['supervision_sha256'] != bank.manifest['supervision_sha256']:
        raise ValueError('baseline features belong to different supervision')
    pool = Pool('quantized://' + str((directory / 'features.quantized').resolve()))
    train, validation = bank.split['train_indices'], bank.split['validation_indices']
    if pool.num_row() != len(bank.rows):
        raise ValueError('baseline is missing frozen rows')
    config = dict(CB_PARAMS, iterations=CB_ITERATIONS, loss_function=CB_LOSS, random_seed=0,
        thread_count=8, allow_writing_files=False)
    model = CatBoostRegressor(**config)
    timed_event('catboost_fit_enter', train_rows=len(train), holdout_rows=len(validation), config=config)
    if (directory / 'model.cbm').exists():
        model.load_model(str(directory / 'model.cbm'))
    else:
        model.fit(pool.slice(train), eval_set=pool.slice(validation), early_stopping_rounds=CB_EARLY_STOPPING,
            use_best_model=True, verbose=100)
        model.save_model(str(directory / 'model.cbm'))
    with (directory / 'holdout.pickle').open('rb') as stream:
        raw = pickle.load(stream)
    validation_pool = Pool(F.matrix(raw, metadata['numeric'], metadata['categorical']),
        cat_features=list(range(len(metadata['numeric']), len(metadata['numeric']) + len(metadata['categorical']))))
    predictions = model.predict(validation_pool)
    clone = CatBoostRegressor()
    clone.load_model(str(directory / 'model.cbm'))
    np.testing.assert_allclose(predictions, clone.predict(validation_pool), atol=1e-12, rtol=1e-12)
    ids = np.asarray(validation, np.int64)
    np.savez_compressed(directory / 'predictions.npz', row_index=ids, prediction=predictions, target=bank.targets[ids])
    result = dict(status='complete', model='greedy_catboost', design='tabular', config=config,
        **score(bank.targets[ids], predictions), train_unique_rows=len(train), train_available_rows=len(train),
        training_coverage_complete=True, holdout_rows=len(validation), seconds=time.perf_counter() - started,
        feature_seconds=metadata['seconds'], iterations=model.tree_count_, early_stopping_rounds=CB_EARLY_STOPPING,
        checkpoint_reload=True, supervision_sha256=bank.manifest['supervision_sha256'],
        prediction_rows_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    write_json(directory / 'result.json', result)
    timed_event('catboost_fit_exit', raw_mse=result['raw_val_mse'], seconds=result['seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not (Path(args.output) / 'features.json').exists():
        prepare(args.cache, args.output)
    elif not (Path(args.output) / 'holdout.pickle').exists():
        prepare(args.cache, args.output, only_holdout=True)
    fit(args.cache, args.output)


if __name__ == '__main__':
    main()
