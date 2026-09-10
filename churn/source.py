import csv
from datetime import date, timedelta
import io
import logging
import time
import zipfile


LOG = logging.getLogger(__name__)

KEY_COLUMNS = ('producer_key', 'placement_specialist_key', 'account_executive_key', 'billing_company_key',
               'expiring_policy_key', 'servicer_key', 'issuing_company_key')
DROP = ('client_state_code', 'client_state_name', 'line_of_business_code', 'db_branch_key', 'ds_servicer_key',
        'pol_no_longer_in_force_date', 'pol_policy_cancelled_flag', 'fact_policy_key',
        'naics_name', 'sic_name', 'policy_type_group_name', 'servicer_name')
HORIZON = 365
SENTINEL = date(2050, 1, 1)


def parse_date(text):
    year, month, day = (int(part) for part in text.split('-'))
    return date(year, month, day) if year < 2050 else SENTINEL


def clean_key(value):
    value = value.strip()
    return value[:-2] if value.endswith('.0') else value


def load(path):
    started = time.perf_counter()
    policies = []
    with zipfile.ZipFile(path) as archive:
        with archive.open('TX.csv') as handle:
            for raw in csv.DictReader(io.TextIOWrapper(handle, 'utf-8-sig')):
                policy = {k: v for k, v in raw.items() if k not in DROP}
                for column in KEY_COLUMNS:
                    if column in policy:
                        policy[column] = clean_key(policy[column])
                policy['policy_key'] = clean_key(policy['policy_key'])
                policy['client_key'] = clean_key(policy['client_key'])
                policy['effective'] = parse_date(raw['pol_policy_effective_date'])
                policy['expiration'] = parse_date(raw['pol_expiration_date'])
                policies.append(policy)
    LOG.info('churn load exit seconds=%.3f policies=%d', time.perf_counter() - started, len(policies))
    return policies


def resolve_customers(policies):
    started = time.perf_counter()
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    by_policy = {policy['policy_key']: policy for policy in policies}
    for policy in policies:
        find(policy['client_key'])
    links = 0
    for policy in policies:
        prior = by_policy.get(policy['expiring_policy_key'])
        if prior is not None:
            links += 1
            union(policy['client_key'], prior['client_key'])
    customer = {key: find(key) for key in parent}
    LOG.info('churn resolve exit seconds=%.3f client_keys=%d customers=%d links=%d',
             time.perf_counter() - started, len(customer), len(set(customer.values())), links)
    return customer


def in_force(policy, moment):
    return policy['effective'] <= moment < policy['expiration']


def population(policies, customer, moment, min_policies, tenure_days=365):
    held, first_seen = {}, {}
    horizon = moment + timedelta(days=HORIZON)
    later = {}
    for policy in policies:
        owner = customer[policy['client_key']]
        if policy['effective'] < first_seen.get(owner, SENTINEL):
            first_seen[owner] = policy['effective']
        if in_force(policy, moment):
            held.setdefault(owner, []).append(policy)
        if in_force(policy, horizon):
            later[owner] = True
    result = []
    for owner, group in held.items():
        if len(group) >= min_policies and first_seen[owner] <= moment - timedelta(days=tenure_days):
            result.append((owner, group, 0 if owner in later else 1))
    return result
