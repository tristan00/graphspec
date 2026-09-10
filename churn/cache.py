import argparse
from datetime import date, timedelta
import hashlib
import logging
from pathlib import Path
import time

import numpy as np

from graphspec import Builder, build_campaign, write_json

from . import spec
from .schema import encode
from .source import load, population, resolve_customers


LOG = logging.getLogger(__name__)
VERSION = 2
IDENTITY = ('policy', 'client')
HISTORY = spec.CONSTRUCTION['client_history']
LABEL = spec.CONSTRUCTION['label']


class Book(Builder):

    registry = spec.REGISTRY
    views = spec.VIEWS

    def append(self, row, record):
        step = len(self.rows)
        moment = record['moment']
        states, spatial, attributes = {}, set(), {}

        def node(kind, key, data=None):
            entity = self.eid(kind, key)
            if entity not in states:
                states[entity] = dict(data or {}) if kind in IDENTITY else dict(data or {}, key=str(key))
            elif data:
                states[entity].update(data)
            return entity

        def edge(source, target, relation):
            if source != target:
                spatial.add((source, target, self.registry.code(relation)))

        client = node('client', record['customer'], record['client_state'])
        observed = {}
        for policy in record['policies']:
            entity = node('policy', policy['policy_key'],
                          dict(policy, age_days=(moment - policy['effective']).days))
            observed[policy['policy_key']] = entity
            context = {'policy': entity, 'client': client,
                       'policy_type': node('policy_type', policy['policy_type_key'])}
            for declared in self.registry.keyed:
                value = policy.get(declared.column)
                if not value or declared.name == 'renews' or declared.sources[0] not in context:
                    continue
                edge(context[declared.sources[0]], node(declared.targets[0], value), declared.name)
        renews = self.registry.code('renews')
        for policy in record['policies']:
            prior = observed.get(policy['expiring_policy_key'])
            entity = observed[policy['policy_key']]
            if prior is None or prior == entity:
                continue
            edge(prior, entity, 'renews')
            previous = record['by_key'][policy['expiring_policy_key']]
            gap = (policy['effective'] - previous['expiration']).days if previous['expiration'] <= moment else 0
            attributes[(prior, entity, renews)] = [float(gap)]
        subject = observed[record['policies'][-1]['policy_key']]
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
            declared = self.registry.by_code[relation]
            self.registry.validate(declared.name, self.node_kinds[source], self.node_kinds[target])
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
        self.q.append(encode('client', record['client_state']))
        self.actors.append(client)
        self.subjects.append(subject)
        self.query_context.append([mapping[client], mapping[subject], mapping[client]])
        self.affinity.append(self.counts[client].copy())
        self.global_affinity.append(self.global_counts.copy())
        self.events.append((client, subject, step * 2 + 1, 2, mapping[client], mapping[subject], 0, -1, -1))
        self.counts[client][0] += 1
        self.global_counts[0] += 1
        self.signals.append([])
        self.rows.append(row)
        self.offset.append(len(self.x))


def cohort(policies, customer, moment):
    started = time.perf_counter()
    by_owner, by_key = {}, {}
    floor = moment - timedelta(days=HISTORY['window'])
    for policy in policies:
        if floor <= policy['effective'] <= moment:
            by_owner.setdefault(customer[policy['client_key']], []).append(policy)
            by_key[policy['policy_key']] = policy
    at_risk = population(policies, customer, moment, LABEL['min_policies'], LABEL['tenure_days'])
    records = []
    for owner, held, churned in at_risk:
        history = sorted(by_owner.get(owner, held), key=lambda p: p['effective'])[-HISTORY['fanout']:]
        if not history:
            continue
        state = dict(policies_in_force=len(held), policies_known=len(history),
                     tenure_days=(moment - min(p['effective'] for p in history)).days,
                     client_city=held[0].get('client_city'), naics_code=held[0].get('naics_code'),
                     sic_code=held[0].get('sic_code'), integrated_platform=held[0].get('integrated_platform'),
                     pol_income_type=held[0].get('pol_income_type'))
        records.append((owner, history, churned, state))
    LOG.info('churn cohort exit seconds=%.3f moment=%s customers=%d churn=%.4f',
             time.perf_counter() - started, moment, len(records),
             sum(record[2] for record in records) / max(len(records), 1))
    return records, by_key


def build(directory, source, first=date(2023, 1, 1), count=8, spacing=91):
    started = time.perf_counter()
    directory = Path(directory)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('completed cache exists; choose a new directory')
    directory.mkdir(parents=True, exist_ok=True)
    policies = load(source)
    customer = resolve_customers(policies)
    coverage = max(policy['effective'] for policy in policies if policy['effective'] <= date.today())
    last = first + timedelta(days=spacing * (count - 1) + LABEL['horizon_days'])
    if last > coverage:
        raise ValueError('label horizon %s runs past the observable end of the data %s' % (last, coverage))
    rows, blocks = [], []
    for index in range(count):
        moment = first + timedelta(days=spacing * index)
        records, by_key = cohort(policies, customer, moment)
        block = []
        for owner, history, churned, state in records:
            row = dict(campaign=str(moment), decision=len(rows), step=len(block), turn=index,
                       y=float(churned), moment=str(moment))
            block.append((row, dict(customer=owner, policies=history, moment=moment,
                                    by_key=by_key, client_state=state)))
            rows.append(row)
        blocks.append((str(moment), block))
    holdout = {name for index, (name, _) in enumerate(blocks) if index >= count - 2}
    embargo = min(date.fromisoformat(name) for name in holdout)
    trained = {name for name, _ in blocks
               if name not in holdout and date.fromisoformat(name) + timedelta(days=LABEL['horizon_days']) <= embargo}
    split = dict(validation_indices=[i for i, row in enumerate(rows) if row['campaign'] in holdout],
                 train_indices=[i for i, row in enumerate(rows) if row['campaign'] in trained])
    write_json(directory / 'supervision.json', dict(rows=rows, split=split))
    fingerprint = hashlib.sha256((directory / 'supervision.json').read_bytes()).hexdigest()
    campaigns, offset = [], 0
    for index, (name, block) in enumerate(blocks):
        indices = list(range(offset, offset + len(block)))
        offset += len(block)
        campaigns.append(build_campaign(Book, directory, index, name, block, ['churn'], indices))
    manifest = dict(version=VERSION, rows=len(rows), campaigns=campaigns, categories=['churn'],
                    signals=[], designs=list(spec.RECIPES), histories=[16, 64, 256],
                    supervision_sha256=fingerprint, spec=spec.declaration(), spec_sha256=spec.fingerprint(),
                    train_rows=len(split['train_indices']), validation_rows=len(split['validation_indices']),
                    coverage_end=str(coverage), label=dict(LABEL), moments=[name for name, _ in blocks], holdout_moments=sorted(holdout),
                    train_moments=sorted(trained), embargo=str(embargo),
                    seconds=time.perf_counter() - started)
    write_json(directory / 'manifest.json', manifest)
    LOG.info('churn cache exit seconds=%.3f campaigns=%d rows=%d', manifest['seconds'], len(campaigns), len(rows))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--moments', type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    build(args.output, args.source, count=args.moments)


if __name__ == '__main__':
    main()
