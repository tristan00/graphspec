from copy import deepcopy

import numpy as np

from totalwar.cache import Campaign, validate
from totalwar.schema import encode


def records(count=8):
    result = []
    for step in range(count):
        row = dict(campaign='example', decision=step + 10, step=step, turn=step // 2 + 1, y=float(step),
                   action=dict(entity_seq=0, action_type='attack_army', key='cqi:2', slot_index=None, offer_seq=0))
        record = dict(campaign_id='example', decision_id=row['decision'], turn=row['turn'],
            campaign=dict(faction='player', treasury=1000 - step * 10, turn=row['turn']),
            entities=[dict(snapshot_id=(step + 10) * 64, context_kind='lord', context_id=1,
                state=dict(cqi=1, units=10, hp=10 - step / 10, region='r1', unit_cards=[{'key': 'spear', 'strength_pct': 99}]))],
            world=dict(armies=[dict(cqi=2, units=5, faction='enemy', region='r2')],
                regions=[dict(region='r1', owner='player', province='p', adjacent=['r2']), dict(region='r2', owner='enemy', province='p')],
                relations=[dict(faction='enemy', at_war=step > 0)]))
        result.append((row, record))
    return result


def make(source):
    campaign = Campaign('example', ['attack_army'])
    for row, record in source:
        campaign.append(row, record)
    return campaign, campaign.finish()


def test_cache_causality_and_exact_allpast():
    campaign, arrays = make(records())
    validate(arrays)
    actor = campaign.ids['character', '1']
    target = campaign.ids['character', '2']
    assert np.all(arrays['subject'] == target)
    sources, destinations = arrays['allpast'].T
    assert np.sum(arrays['entity'][destinations] == actor) == 8 * 7 // 2
    assert np.all(arrays['affinity'][:, 0] == np.arange(8))
    assert np.all(arrays['global_affinity'] == arrays['affinity'])
    assert arrays['signals'][0, 2, 1] == 0


def test_future_and_outcomes_cannot_change_cached_prefix():
    original = records()
    modified = deepcopy(original)
    for row, record in modified:
        row['y'] = 999999
        record['campaign'].update(validated=True, selector='future', raw_reward=123, execution_succeeded=True)
        record['entities'][0]['state'].update(validated=True, invalid_reason='invented', action_result='success')
    modified[-1][1]['entities'][0]['state']['hp'] = 9000
    _, before = make(original)
    _, after = make(modified)
    stop = before['snapshot_offsets'][-2]
    np.testing.assert_array_equal(before['x'][:stop], after['x'][:stop])
    np.testing.assert_array_equal(before['query'], after['query'])
    np.testing.assert_array_equal(before['walks'][:-1], after['walks'][:-1])
    np.testing.assert_array_equal(encode('character', {'hp': 2, 'validated': False}), encode('character', {'hp': 2, 'validated': True}))


def test_relation_removal_requires_observed_status():
    source = records(4)
    source[2][1]['world']['relations'][0]['at_war'] = False
    source[3][1]['world']['relations'] = []
    _, arrays = make(source)
    relation_events = arrays['events'][arrays['events'][:, 3] == 1]
    removals = relation_events[relation_events[:, 8] == 0]
    assert np.any((removals[:, 2] == 4) & (removals[:, 7] == 1))
    assert not np.any(removals[:, 2] == 6)


def test_long_cache_retains_exact_window_and_skip_links():
    campaign, arrays = make(records(70))
    actor = campaign.ids['character', '1']
    observations = np.flatnonzero(arrays['entity'] == actor)
    last = observations[-1]
    earlier = arrays['allpast'][arrays['allpast'][:, 1] == last, 0]
    assert len(earlier) == 63
    assert set(arrays['step'][earlier]) == set(range(6, 69))
    skip = arrays['skip'][arrays['skip'][:, 1] == last, 0]
    assert set(69 - arrays['step'][skip]) == {1, 2, 4, 8, 16, 32, 64}


def test_observed_movement_removes_previous_location():
    source = records(3)
    source[-1][1]['entities'][0]['state']['region'] = 'r2'
    campaign, arrays = make(source)
    actor, previous = campaign.ids['character', '1'], campaign.ids['region', 'r1']
    changes = arrays['events']
    removal = changes[(changes[:, 0] == actor) & (changes[:, 1] == previous) & (changes[:, 3] == 1) & (changes[:, 8] == 0)]
    assert len(removal) == 1 and removal[0, 2] == 4 and removal[0, 7] == 1


def test_rows_keep_their_supervision_dicts():
    source = records()
    campaign, _ = make(source)
    assert campaign.rows == [row for row, _ in source]
    assert [row['decision'] for row in campaign.rows] == [row['decision'] for row, _ in source]
