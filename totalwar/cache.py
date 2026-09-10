import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from . import spec
from .schema import KINDS, SIGNALS, encode, identity, items


LOG = logging.getLogger(__name__)
VERSION = 2


def write_json(path, value):
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(directory / (name + '.npy'), value, allow_pickle=False)


class Campaign:

    def __init__(self, name, categories):
        self.name, self.categories = name, categories
        self.ids, self.node_keys, self.node_kinds = {}, [], []
        self.x, self.entity, self.steps, self.turns, self.offset = [], [], [], [], [0]
        self.edges, self.edge_steps, self.attribute_rows = [], [], []
        self.type_attributes = {name: [] for name in spec.ATTRIBUTED}
        self.q, self.actors, self.subjects, self.query_context, self.rows = [], [], [], [], []
        self.events, self.affinity, self.global_affinity, self.signals = [], [], [], []
        self.last_obs, self.associations, self.known_status = {}, {}, {}
        self.counts, self.global_counts = defaultdict(lambda: np.zeros(len(categories), np.float32)), np.zeros(len(categories), np.float32)
        self.unresolved = defaultdict(int)

    def eid(self, kind, key):
        key = (kind, str(key))
        if key not in self.ids:
            self.ids[key] = len(self.ids)
            self.node_keys.append(key)
            self.node_kinds.append(KINDS.index(kind))
        return self.ids[key]

    def append(self, row, record):
        step = len(self.rows)
        states, spatial, explicit_status, attributes = {}, set(), {}, {}
        faction = record['campaign']['faction']

        def node(kind, key, data=None):
            entity = self.eid(kind, key)
            if entity not in states:
                states[entity] = dict(data or {})
            elif data:
                states[entity].update(data)
            return entity

        def edge(source, target, relation):
            if source != target:
                spatial.add((source, target, spec.code(relation)))

        def place(entity, data, own=False):
            if data.get('region'):
                region = node('region', data['region'], {'region': data['region']})
                edge(entity, region, 'located')
            if data.get('province'):
                province = node('province', data['province'], {'province': data['province']})
                edge(province, entity, 'contains')
            owner = faction if own else data.get('owner', data.get('faction', data.get('region_owner')))
            if owner:
                owner = node('faction', owner, {'faction': owner})
                edge(owner, entity, 'owns')

        def catalogue(entity, data):
            for declared in spec.COLLECTIONS:
                for key, value in items(data.get(declared.collection)):
                    subject = str(value.get(declared.subject_field, value.get('key', key)))
                    other = node(declared.targets[0], subject, {'key': subject})
                    edge(entity, other, declared.name)
                    pair = entity, other, declared.code
                    attr = attributes.setdefault(pair, [0] * len(declared.attributes))
                    attr[0] += 1
                    for position, field in enumerate(declared.attributes[1:], 1):
                        attr[position] += float(value.get(field) or 0)
            for collection in ('slot_states', 'horde_slots'):
                for key, value in items(data.get(collection)):
                    slot_key = (entity, value.get('slot_index', value.get('slot_id', key)))
                    slot = node('slot', str(slot_key), {k: v for k, v in value.items() if k != 'key'})
                    edge(entity, slot, 'contains')
                    if not value.get('empty', False) and value.get('key'):
                        building = node('building', value['key'], {'key': value['key']})
                        edge(slot, building, 'building')

        camp = node('faction', faction, record['campaign'])
        for scope, kind in (('regions', 'region'), ('settlements', 'settlement'), ('ruins', 'settlement'),
                            ('armies', 'character'), ('hostiles', 'character'), ('enemy_agents', 'character')):
            for key, data in items(record['world'].get(scope)):
                actual = 'settlement' if data.get('kind') == 'settlement' else kind
                if actual == 'character' and data.get('cqi') is None:
                    continue
                entity = node(actual, identity(actual, data, key), data)
                place(entity, data)
                if actual == 'settlement':
                    edge(node('region', identity(actual, data, key), {'region': identity(actual, data, key)}), entity, 'contains')
                if actual == 'region':
                    for adjacent in data.get('adjacent', []):
                        edge(entity, node('region', adjacent, {'region': adjacent}), 'adjacent')
        actor_map = {None: camp}
        for entity in record['entities']:
            scope, data, key = entity['context_kind'], entity['state'], entity['context_id']
            if scope == 'campaign':
                eid = node('faction', faction, data)
            elif scope == 'province':
                eid = node('settlement', data.get('region', key), data)
                place(eid, data, True)
            elif scope in ('lord', 'hero'):
                eid = node('character', identity('character', data, key), data)
                place(eid, data, True)
            else:
                raise ValueError('unknown actor scope ' + scope)
            actor_map[entity['snapshot_id'] % 64] = eid
            catalogue(eid, data)
        for _, data in items(record['world'].get('relations')):
            other = node('faction', data['faction'], data)
            for declared in spec.FLAGGED:
                if declared.field in data:
                    pair = (camp, other, declared.code)
                    explicit_status[pair] = int(bool(data[declared.field]))
                    if data[declared.field]:
                        spatial.add(pair)
        for _, data in items(record['world'].get('war_graph')):
            own = node('faction', data['faction'], {'faction': data['faction']})
            for opponent in data.get('at_war_with', []):
                edge(own, node('faction', opponent, {'faction': opponent}), 'war')
        for source, target, relation in spatial:
            if relation in (spec.code('located'), spec.code('building')):
                key = source, relation
            elif relation == spec.code('owns'):
                key = target, relation
            else:
                continue
            previous = self.associations.get(key)
            if previous is not None and previous != (source, target, relation):
                explicit_status[previous] = 0
            self.associations[key] = source, target, relation
        action = row['action']
        actor = actor_map[action['entity_seq']]
        at, key = action['action_type'], str(action['key'])
        subject_kind, subject_key = 'context', 'campaign'
        if at in ('attack_settlement', 'colonize', 'garrison'):
            subject_kind, subject_key = 'settlement', key.removeprefix('settlement:')
        elif at == 'attack_army':
            subject_kind, subject_key = 'character', key.removeprefix('cqi:').removeprefix('army:').removeprefix('character:')
        elif at == 'diplomacy':
            subject_kind, subject_key = 'faction', key.split(':', 1)[0]
        elif at in ('building', 'building_repair', 'building_dismantle', 'horde_building'):
            subject_kind, subject_key = 'slot', str((actor, action.get('slot_index')))
        elif at in ('research', 'skills', 'items', 'item_unequip', 'recruit_unit', 'raise_dead', 'recruit_ror', 'recruit_imperial', 'recruit_blessed', 'rites'):
            subject_kind = {'research': 'tech', 'skills': 'skill', 'items': 'item', 'item_unequip': 'item', 'rites': 'ritual'}.get(at, 'unit')
            subject_key = key
        elif at not in ('end_turn', 'stance', 'cancel_recruit', 'recruit_lord', 'recruit_hero', 'edict', 'move', 'leave_garrison', 'hero_action'):
            raise ValueError('unknown action type ' + at)
        elif at == 'hero_action':
            target = key.rsplit('@', 1)[-1]
            if target.startswith('cqi:'):
                subject_kind, subject_key = 'character', target[4:]
            elif target.startswith('settlement:') or target.startswith('wh'):
                subject_kind, subject_key = 'settlement', target.removeprefix('settlement:')
            else:
                self.unresolved[at] += 1
        elif at == 'move' and key.startswith('xy:'):
            coordinates = [float(v) for v in key[3:].split(',')]
            locations = [(sum((float(data[k]) - coordinates[i])**2 for i, k in enumerate(('x', 'y'))), eid)
                         for eid, data in states.items() if self.node_keys[eid][0] == 'region' and all(isinstance(data.get(k), (int, float)) for k in ('x', 'y'))]
            if locations:
                subject_kind, subject_key = self.node_keys[min(locations)[1]]
            else:
                subject_kind, subject_key = 'context', key
        elif at == 'leave_garrison' and states[actor].get('region'):
            subject_kind, subject_key = 'region', states[actor]['region']
        subject = self.eid(subject_kind, subject_key)
        context = self.eid('context', 'campaign')
        if subject not in states:
            states[subject] = {'key': subject_key}
        if context not in states:
            states[context] = {}
        edge(camp, context, 'context')
        mapping = {}
        for entity, data in states.items():
            kind = self.node_keys[entity][0]
            ix = len(self.x)
            mapping[entity] = ix
            self.x.append(encode(kind, data))
            self.entity.append(entity)
            self.steps.append(step)
            self.turns.append(row['turn'])
            previous = self.last_obs.get(entity, -1)
            if previous < 0 or not np.array_equal(self.x[ix], self.x[previous]):
                self.events.append((entity, entity, step * 2, 0, ix, ix, previous, -1, -1))
            self.last_obs[entity] = ix
        for source, target, relation in sorted(spatial):
            declared = spec.BY_CODE[relation]
            spec.validate(declared.name, self.node_kinds[source], self.node_kinds[target])
            self.edges.append((mapping[source], mapping[target], relation))
            self.edges.append((mapping[target], mapping[source], declared.reverse))
            self.edge_steps.extend((step, step))
            if declared.attributes:
                block = self.type_attributes[declared.name]
                attribute_row = len(block)
                block.append(attributes.get((source, target, relation), declared.default))
                self.attribute_rows.extend((attribute_row, attribute_row))
            else:
                self.attribute_rows.extend((-1, -1))
            pair = source, target, relation
            old = self.known_status.get(pair, -1)
            if old != 1:
                self.events.append((source, target, step * 2, 1, mapping[source], mapping[target], relation, old, 1))
                self.known_status[pair] = 1
        for pair, status in explicit_status.items():
            old = self.known_status.get(pair, -1)
            if old != status:
                source, target, relation = pair
                self.events.append((source, target, step * 2, 1, self.last_obs[source], self.last_obs[target], relation, old, status))
                self.known_status[pair] = status
        action_data = dict(action, actor_kind=self.node_keys[actor][0])
        if at == 'attack_army':
            action_data.pop('key')
        elif at == 'hero_action':
            action_data['key'] = key.split('@', 1)[0]
        elif at == 'move' and key.startswith('xy:'):
            action_data.pop('key')
            action_data['x'], action_data['y'] = [float(v) for v in key[3:].split(',')]
        action_features = encode('action', action_data)
        self.q.append(action_features)
        self.actors.append(actor)
        self.subjects.append(subject)
        self.query_context.append([mapping[actor], mapping[subject], mapping[camp]])
        self.affinity.append(self.counts[actor].copy())
        self.global_affinity.append(self.global_counts.copy())
        category = self.categories.index(at)
        self.events.append((actor, subject, step * 2 + 1, 2, mapping[actor], mapping[subject], category, -1, -1))
        self.counts[actor][category] += 1
        self.global_counts[category] += 1
        signal_data = dict(record['campaign'])
        for ent in record['entities']:
            if ent['context_kind'] == 'campaign':
                signal_data.update(ent['state'])
        for name in ('units', 'hp', 'ap_pct', 'skill_points', 'public_order', 'growth_per_turn'):
            vals = [ent['state'][name] for ent in record['entities'] if isinstance(ent['state'].get(name), (int, float))]
            if vals:
                signal_data[name] = float(np.mean(vals))
        self.signals.append([[float(signal_data[s]), 1] if isinstance(signal_data.get(s), (int, float)) else [0, 0] for s in SIGNALS])
        self.rows.append(row)
        self.offset.append(len(self.x))

    def finish(self):
        started = time.perf_counter()
        x = np.asarray(self.x, dtype=np.float16)
        entity = np.asarray(self.entity, np.int32)
        step = np.asarray(self.steps, np.int32)
        turn = np.asarray(self.turns, np.int32)
        offsets = np.asarray(self.offset, np.int64)
        lags, lag_window = spec.CONSTRUCTION['skip']['lags'], spec.CONSTRUCTION['skip']['window']
        fanout, past_window = spec.CONSTRUCTION['allpast']['fanout'], spec.CONSTRUCTION['allpast']['window']
        by_entity = defaultdict(list)
        chain, skip, allpast = [], [], []
        for ix, eid in enumerate(entity):
            history = by_entity[int(eid)]
            if history:
                chain.append((history[-1], ix))
            for lag in lags:
                if len(history) >= lag and step[ix] - step[history[-lag]] <= lag_window:
                    skip.append((history[-lag], ix))
            for previous in reversed(history[-fanout:]):
                if step[ix] - step[previous] >= past_window:
                    break
                allpast.append((previous, ix))
            history.append(ix)
        observations = np.argsort(entity, kind='stable').astype(np.int32)
        obs_offsets = np.r_[0, np.cumsum(np.bincount(entity, minlength=len(self.ids)))].astype(np.int64)
        events = np.asarray(self.events, np.int32)
        event_order = np.argsort(events[:, 2], kind='stable')
        events = events[event_order]
        neighbors = defaultdict(list)
        dag, prior_groups = [], defaultdict(list)
        ticks = events[:, 2]
        for start, stop in zip(np.r_[0, np.flatnonzero(np.diff(ticks)) + 1], np.r_[np.flatnonzero(np.diff(ticks)) + 1, len(events)]):
            members = defaultdict(list)
            for ix in range(int(start), int(stop)):
                source, target = events[ix, :2]
                participants = {int(source), int(target)}
                for participant in participants:
                    for lag in spec.CONSTRUCTION['event_dag']['lags']:
                        groups = prior_groups[participant]
                        if len(groups) >= lag:
                            for previous in groups[-lag]:
                                if ticks[ix] - ticks[previous] <= spec.CONSTRUCTION['event_dag']['window']:
                                    dag.append((previous, ix))
                    members[participant].append(ix)
                    if events[ix, 3] != 0:
                        neighbors[participant].append(ix)
            for participant, group in members.items():
                prior_groups[participant].append(group)
        neighbor_ids = np.asarray([i for eid in range(len(self.ids)) for i in neighbors[eid]], np.int32)
        neighbor_offsets = np.r_[0, np.cumsum([len(neighbors[eid]) for eid in range(len(self.ids))])].astype(np.int64)
        walk = spec.CONSTRUCTION['walk']
        rng = np.random.default_rng(walk['seed'])
        walks = np.full((len(self.rows), 2, walk['samples'], walk['hops']), -1, np.int32)
        for query, (actor, subject) in enumerate(zip(self.actors, self.subjects)):
            for side, root in enumerate((actor, subject)):
                for sample in range(walk['samples']):
                    current, cutoff = root, query * 2 + 1
                    for hop in range(walk['hops']):
                        ids = neighbor_ids[neighbor_offsets[current]:neighbor_offsets[current + 1]]
                        end = np.searchsorted(ticks[ids], cutoff)
                        start = np.searchsorted(ticks[ids], max(0, (query - walk['window'] + 1) * 2))
                        if end <= start:
                            break
                        event = int(ids[rng.integers(max(start, end - walk['fanout']), end)])
                        walks[query, side, sample, hop] = event
                        source, target, cutoff = events[event, :3]
                        current = int(target if current == source else source)
        mixed_x, mixed_kind, mixed_step, mixed_entity, mixed_members, mixed_edges = [], [], [], [], [], []
        last, traces = {}, defaultdict(list)
        for decision in range(len(self.rows)):
            current = []
            for ix in range(offsets[decision], offsets[decision + 1]):
                eid = int(entity[ix])
                j = len(mixed_x)
                mixed_x.append(x[ix]); mixed_kind.append(spec.mixed_kind('observation')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append([ix])
                current.append(j)
                if eid in last:
                    previous, prior_ix = last[eid]
                    mixed_edges.append((previous, j, spec.MIXED_EDGES['previous_observation']))
                    if not np.array_equal(x[ix], x[prior_ix]):
                        delta = np.concatenate((x[ix, :64].astype(np.float32) - x[prior_ix, :64], x[ix, 64:])).astype(np.float16)
                        change = len(mixed_x)
                        mixed_x.append(delta); mixed_kind.append(spec.mixed_kind('change')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append([prior_ix, ix])
                        mixed_edges.extend(((previous, change, spec.MIXED_EDGES['change_link']), (change, j, spec.MIXED_EDGES['change_link'])))
                last[eid] = (j, ix)
                traces[eid].append((j, ix))
                if len(traces[eid]) == spec.CONSTRUCTION['trace']['size']:
                    trace = len(mixed_x)
                    members = [v for _, v in traces[eid]]
                    mixed_x.append(x[members].mean(0)); mixed_kind.append(spec.mixed_kind('trace')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append(members)
                    mixed_edges.extend((v, trace, spec.MIXED_EDGES['member_group']) for v, _ in traces[eid])
                    mixed_edges.append((trace, j, spec.MIXED_EDGES['trace_current']))
                    traces[eid].clear()
            bundle = len(mixed_x)
            members = list(range(int(offsets[decision]), int(offsets[decision + 1])))
            mixed_x.append(x[members].mean(0)); mixed_kind.append(spec.mixed_kind('bundle')); mixed_step.append(decision * 2); mixed_entity.append(-1); mixed_members.append(members)
            mixed_edges.extend((v, bundle, spec.MIXED_EDGES['member_group']) for v in current)
            action = len(mixed_x)
            mixed_x.append(self.q[decision]); mixed_kind.append(spec.mixed_kind('action')); mixed_step.append(decision * 2 + 1); mixed_entity.append(self.actors[decision]); mixed_members.append([])
            mixed_edges.append((bundle, action, spec.MIXED_EDGES['bundle_action']))
            if decision:
                mixed_edges.append((previous_action, action, spec.MIXED_EDGES['previous_action']))
            previous_action = action
        arrays = dict(x=x, entity=entity, step=step, turn=turn, snapshot_offsets=offsets,
            entity_kind=np.asarray(self.node_kinds, np.int8), spatial=np.asarray(self.edges, np.int32).reshape(-1, 3),
            spatial_attribute_row=np.asarray(self.attribute_rows, np.int32),
            spatial_step=np.asarray(self.edge_steps, np.int32), chain=np.asarray(chain, np.int32).reshape(-1, 2),
            skip=np.asarray(skip, np.int32).reshape(-1, 2), allpast=np.asarray(allpast, np.int32).reshape(-1, 2),
            observation_ids=observations, observation_offsets=obs_offsets,
            query=np.asarray(self.q, np.float16), actor=np.asarray(self.actors, np.int32), subject=np.asarray(self.subjects, np.int32),
            query_context=np.asarray(self.query_context, np.int32), events=events, event_dag=np.asarray(dag, np.int32).reshape(-1, 2),
            neighbor_ids=neighbor_ids, neighbor_offsets=neighbor_offsets, walks=walks,
            affinity=np.asarray(self.affinity, np.float32), global_affinity=np.asarray(self.global_affinity, np.float32),
            signals=np.asarray(self.signals, np.float32), mixed_x=np.asarray(mixed_x, np.float16),
            mixed_kind=np.asarray(mixed_kind, np.int8), mixed_tick=np.asarray(mixed_step, np.int32),
            mixed_entity=np.asarray(mixed_entity, np.int32), mixed_edges=np.asarray(mixed_edges, np.int32).reshape(-1, 3),
            mixed_member_ids=np.asarray([m for group in mixed_members for m in group], np.int32),
            mixed_member_offsets=np.r_[0, np.cumsum([len(group) for group in mixed_members])].astype(np.int64),
            role_left=events[events[:, 3] == 2, 0], role_right=events[events[:, 3] == 2, 1],
            propagation_ids=np.flatnonzero(events[:, 3] != 0).astype(np.int32))
        for name, block in self.type_attributes.items():
            arrays['attributes_' + name] = np.asarray(block, np.float16).reshape(-1, len(spec.attributes(name)))
        LOG.info('campaign cache exit campaign=%s seconds=%.3f decisions=%d nodes=%d events=%d', self.name, time.perf_counter() - started, len(self.rows), len(x), len(events))
        return arrays


def validate(arrays):
    x, step, events = arrays['x'], arrays['step'], arrays['events']
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observation features')
    codes, rows = arrays['spatial'][:, 2], arrays['spatial_attribute_row']
    for edge in spec.EDGES:
        selected = rows[codes == edge.code]
        if not edge.attributes:
            if np.any(selected != -1):
                raise ValueError('unattributed edge %s carries an attribute row' % edge.name)
            continue
        block = arrays['attributes_' + edge.name]
        if np.any(selected < 0) or (len(selected) and selected.max() >= len(block)):
            raise ValueError('attribute row out of range for edge ' + edge.name)
        if not np.array_equal(rows[codes == edge.reverse], selected):
            raise ValueError('reverse edge attribute rows diverge for ' + edge.name)
    for name in ('chain', 'skip', 'allpast'):
        source, destination = arrays[name].T
        if np.any(step[source] >= step[destination]) or np.any(arrays['entity'][source] != arrays['entity'][destination]):
            raise ValueError('invalid lifted time edges ' + name)
    source, destination = arrays['event_dag'].T
    if np.any(events[source, 2] >= events[destination, 2]):
        raise ValueError('event DAG crosses a simultaneous or future group')
    walks = arrays['walks']
    times = np.where(walks >= 0, events[np.maximum(walks, 0), 2], -1)
    if np.any(times[:, :, :, 0] > np.arange(len(walks))[:, None, None] * 2):
        raise ValueError('walk includes current choice or future event')
    if np.any((np.diff(times, axis=-1) >= 0) & (times[:, :, :, 1:] >= 0)):
        raise ValueError('walk is not strictly causal')


def build_campaign(directory, index, name, records, categories, row_indices):
    started = time.perf_counter()
    bank = Campaign(name, categories)
    for row, record in records:
        bank.append(row, record)
    arrays = bank.finish()
    validate(arrays)
    arrays['row_index'] = np.asarray(row_indices, np.int64)
    arrays['decision'] = np.asarray([row['decision'] for row in bank.rows], np.int64)
    arrays['decision_turn'] = np.asarray([row['turn'] for row in bank.rows], np.int32)
    path = Path(directory) / ('campaign_%04d' % index)
    save_arrays(path, arrays)
    write_json(path / 'entities.json', bank.node_keys)
    return dict(index=index, name=name, decisions=len(bank.rows), nodes=len(arrays['x']), events=len(arrays['events']),
        bytes=sum(a.nbytes for a in arrays.values()), seconds=time.perf_counter() - started,
        unresolved_subjects=dict(bank.unresolved), counts={k: len(v) for k, v in arrays.items()})


def build(directory, window=2500, workers=4):
    from .tw_source import open_corpus

    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('completed cache exists; choose a new directory')
    if any(directory.glob('campaign_*')):
        raise FileExistsError('partial cache exists; choose a new directory to preserve its snapshot')
    store, corpus = open_corpus(window)
    try:
        categories = sorted({row['action']['action_type'] for row in corpus.heads})
        row_keys = {(row['campaign'], row['step']): i for i, row in enumerate(corpus.rows)}
        split = corpus.split
        write_json(directory / 'supervision.json', dict(rows=corpus.rows, split=split))
        fingerprint = hashlib.sha256((directory / 'supervision.json').read_bytes()).hexdigest()
        write_json(directory / 'building.json', dict(started=datetime.now(timezone.utc).isoformat(), window=window, supervision_sha256=fingerprint))
        campaigns = []
        def completed(future):
            metrics = future.result()
            campaigns.append(metrics)
            with (directory / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics) + '\n')
            if len(campaigns) % 10 == 0:
                LOG.warning('cache progress campaigns=%d decisions=%d seconds=%.1f GiB=%.3f', len(campaigns), sum(c['decisions'] for c in campaigns), time.perf_counter() - started, sum(c['bytes'] for c in campaigns) / 2**30)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = []
            for index, (name, records) in enumerate(corpus.campaigns()):
                records = list(records)
                rows = [row_keys.get((name, row['step']), -1) for row, _ in records]
                pending.append(pool.submit(build_campaign, directory, index, name, records, categories, rows))
                if len(pending) >= workers * 2:
                    completed(pending.pop(0))
            for future in pending:
                completed(future)
        store.con.execute('COMMIT')
        walk_recipe = spec.CONSTRUCTION['walk']
        manifest = dict(version=VERSION, completed=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
            window=window, rows=len(corpus.rows), designs=list(spec.RECIPES), histories=[16, 64, 256], allpast_history=spec.CONSTRUCTION['allpast']['window'],
            supervision_sha256=fingerprint, holdout_salt='gnn3', validation_fraction=0.15,
            train_rows=len(split['train_indices']), validation_rows=len(split['validation_indices']),
            categories=categories, signals=list(SIGNALS), campaigns=campaigns,
            spec=spec.declaration(), spec_sha256=spec.fingerprint(),
            feature_policy='Explicit scalar allowlist; no execution validation, policy scores or outcomes.',
            observation_clock='pre-state tick=2*step; chosen action tick=2*step+1; query cutoff=2*step',
            mixed_recipe='Causal observation/change, four-observation traces, decision bundles and prior-choice links; adapted cache topology for the existing message network.',
            walk_seed=walk_recipe['seed'], walk_samples=walk_recipe['samples'], walk_hops=walk_recipe['hops'])
        write_json(directory / 'manifest.json', manifest)
        LOG.warning('cache complete seconds=%.3f campaigns=%d rows=%d bytes=%d', manifest['seconds'], len(campaigns), len(corpus.rows), sum(c['bytes'] for c in campaigns))
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, default=2500)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    build(args.output, args.window, args.workers)


if __name__ == '__main__':
    main()
