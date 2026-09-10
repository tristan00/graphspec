import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

from graphspec import builder
from graphspec.builder import Builder, build_campaign, write_json
from . import spec
from .schema import KINDS, SIGNALS, encode, identity, items


LOG = logging.getLogger(__name__)
VERSION = 2


def validate(arrays):
    return builder.validate(spec.REGISTRY, arrays)


class Campaign(Builder):

    registry = spec.REGISTRY
    views = spec.VIEWS

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
                pending.append(pool.submit(build_campaign, Campaign, directory, index, name, records, categories, rows))
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
