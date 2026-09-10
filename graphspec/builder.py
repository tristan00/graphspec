from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import time

import numpy as np


LOG = logging.getLogger(__name__)


def write_json(path, value):
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(directory / (name + '.npy'), value, allow_pickle=False)


class Builder:

    registry = None
    views = {}

    def __init__(self, name, categories):
        self.name, self.categories = name, categories
        self.construction = self.registry.construction
        self.mixed_nodes = tuple(self.registry.extras.get('mixed_nodes', ()))
        self.mixed_codes = dict(self.registry.extras.get('mixed_edges', {}))
        self.ids, self.node_keys, self.node_kinds = {}, [], []
        self.x, self.entity, self.steps, self.turns, self.offset = [], [], [], [], [0]
        self.edges, self.edge_steps, self.attribute_rows = [], [], []
        self.type_attributes = {name: [] for name in self.registry.attributed}
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
            self.node_kinds.append(self.registry.kind(kind))
        return self.ids[key]
    def finish(self):
        started = time.perf_counter()
        x = np.asarray(self.x, dtype=np.float16)
        entity = np.asarray(self.entity, np.int32)
        step = np.asarray(self.steps, np.int32)
        turn = np.asarray(self.turns, np.int32)
        offsets = np.asarray(self.offset, np.int64)
        skip_recipe, past_recipe = self.construction.get('skip'), self.construction.get('allpast')
        chained = 'chain' in self.construction
        by_entity = defaultdict(list)
        chain, skip, allpast = [], [], []
        for ix, eid in enumerate(entity):
            history = by_entity[int(eid)]
            if chained and history:
                chain.append((history[-1], ix))
            if skip_recipe:
                for lag in skip_recipe['lags']:
                    if len(history) >= lag and step[ix] - step[history[-lag]] <= skip_recipe['window']:
                        skip.append((history[-lag], ix))
            if past_recipe:
                for previous in reversed(history[-past_recipe['fanout']:]):
                    if step[ix] - step[previous] >= past_recipe['window']:
                        break
                    allpast.append((previous, ix))
            history.append(ix)
        observations = np.argsort(entity, kind='stable').astype(np.int32)
        obs_offsets = np.r_[0, np.cumsum(np.bincount(entity, minlength=len(self.ids)))].astype(np.int64)
        events = np.asarray(self.events, np.int32)
        event_order = np.argsort(events[:, 2], kind='stable')
        events = events[event_order]
        neighbors = defaultdict(list)
        dag_recipe = self.construction.get('event_dag')
        dag, prior_groups = [], defaultdict(list)
        ticks = events[:, 2]
        for start, stop in zip(np.r_[0, np.flatnonzero(np.diff(ticks)) + 1], np.r_[np.flatnonzero(np.diff(ticks)) + 1, len(events)]):
            members = defaultdict(list)
            for ix in range(int(start), int(stop)):
                source, target = events[ix, :2]
                participants = {int(source), int(target)}
                for participant in participants:
                    for lag in dag_recipe['lags'] if dag_recipe else ():
                        groups = prior_groups[participant]
                        if len(groups) >= lag:
                            for previous in groups[-lag]:
                                if ticks[ix] - ticks[previous] <= dag_recipe['window']:
                                    dag.append((previous, ix))
                    members[participant].append(ix)
                    if events[ix, 3] != 0:
                        neighbors[participant].append(ix)
            for participant, group in members.items():
                prior_groups[participant].append(group)
        neighbor_ids = np.asarray([i for eid in range(len(self.ids)) for i in neighbors[eid]], np.int32)
        neighbor_offsets = np.r_[0, np.cumsum([len(neighbors[eid]) for eid in range(len(self.ids))])].astype(np.int64)
        walk = self.construction.get('walk')
        walks = np.full((len(self.rows), 2, walk['samples'], walk['hops']), -1, np.int32) if walk else np.zeros((0, 2, 0, 0), np.int32)
        rng = np.random.default_rng(walk['seed']) if walk else None
        for query, (actor, subject) in enumerate(zip(self.actors, self.subjects) if walk else ()):
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
        for decision in range(len(self.rows) if self.mixed_nodes else 0):
            current = []
            for ix in range(offsets[decision], offsets[decision + 1]):
                eid = int(entity[ix])
                j = len(mixed_x)
                mixed_x.append(x[ix]); mixed_kind.append(self.mixed_nodes.index('observation')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append([ix])
                current.append(j)
                if eid in last:
                    previous, prior_ix = last[eid]
                    mixed_edges.append((previous, j, self.mixed_codes['previous_observation']))
                    if not np.array_equal(x[ix], x[prior_ix]):
                        delta = np.concatenate((x[ix, :self.registry.width].astype(np.float32) - x[prior_ix, :self.registry.width], x[ix, self.registry.width:])).astype(np.float16)
                        change = len(mixed_x)
                        mixed_x.append(delta); mixed_kind.append(self.mixed_nodes.index('change')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append([prior_ix, ix])
                        mixed_edges.extend(((previous, change, self.mixed_codes['change_link']), (change, j, self.mixed_codes['change_link'])))
                last[eid] = (j, ix)
                traces[eid].append((j, ix))
                if len(traces[eid]) == self.construction['trace']['size']:
                    trace = len(mixed_x)
                    members = [v for _, v in traces[eid]]
                    mixed_x.append(x[members].mean(0)); mixed_kind.append(self.mixed_nodes.index('trace')); mixed_step.append(decision * 2); mixed_entity.append(eid); mixed_members.append(members)
                    mixed_edges.extend((v, trace, self.mixed_codes['member_group']) for v, _ in traces[eid])
                    mixed_edges.append((trace, j, self.mixed_codes['trace_current']))
                    traces[eid].clear()
            bundle = len(mixed_x)
            members = list(range(int(offsets[decision]), int(offsets[decision + 1])))
            mixed_x.append(x[members].mean(0)); mixed_kind.append(self.mixed_nodes.index('bundle')); mixed_step.append(decision * 2); mixed_entity.append(-1); mixed_members.append(members)
            mixed_edges.extend((v, bundle, self.mixed_codes['member_group']) for v in current)
            action = len(mixed_x)
            mixed_x.append(self.q[decision]); mixed_kind.append(self.mixed_nodes.index('action')); mixed_step.append(decision * 2 + 1); mixed_entity.append(self.actors[decision]); mixed_members.append([])
            mixed_edges.append((bundle, action, self.mixed_codes['bundle_action']))
            if decision:
                mixed_edges.append((previous_action, action, self.mixed_codes['previous_action']))
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
            arrays['attributes_' + name] = np.asarray(block, np.float16).reshape(-1, len(self.registry.attributes(name)))
        missing = sorted({name for arrays_of in self.views.values() for name in arrays_of} - set(arrays))
        if missing:
            raise ValueError('declared views name arrays the builder did not emit: ' + ', '.join(missing))
        LOG.info('campaign cache exit campaign=%s seconds=%.3f decisions=%d nodes=%d events=%d', self.name, time.perf_counter() - started, len(self.rows), len(x), len(events))
        return arrays

def validate(registry, arrays):
    x, step, events = arrays['x'], arrays['step'], arrays['events']
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observation features')
    codes, rows = arrays['spatial'][:, 2], arrays['spatial_attribute_row']
    for edge in registry.edges:
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
    for name in sorted({recipe.lifted for recipe in registry.recipes.values() if recipe.lifted}):
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

def build_campaign(builder_class, directory, index, name, records, categories, row_indices):
    started = time.perf_counter()
    bank = builder_class(name, categories)
    for row, record in records:
        bank.append(row, record)
    arrays = bank.finish()
    validate(bank.registry, arrays)
    arrays['row_index'] = np.asarray(row_indices, np.int64)
    arrays['decision'] = np.asarray([row['decision'] for row in bank.rows], np.int64)
    arrays['decision_turn'] = np.asarray([row['turn'] for row in bank.rows], np.int32)
    path = Path(directory) / ('campaign_%04d' % index)
    save_arrays(path, arrays)
    write_json(path / 'entities.json', bank.node_keys)
    return dict(index=index, name=name, decisions=len(bank.rows), nodes=len(arrays['x']), events=len(arrays['events']),
        bytes=sum(a.nbytes for a in arrays.values()), seconds=time.perf_counter() - started,
        unresolved_subjects=dict(bank.unresolved), counts={k: len(v) for k, v in arrays.items()})
