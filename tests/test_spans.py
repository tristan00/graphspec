import numpy as np

from graphspec.views import spans

NAN = np.nan
ROWS = [
    (0, 0, 1, 0, True, 1.0),
    (0, 1, 2, 0, True, 2.0),
    (0, 2, 3, 1, True, 5.0),
    (0, 2, 3, 1, True, 5.0),
    (1, 1, 2, 0, True, 2.0),
    (1, 0, 1, 0, True, 1.0),
    (1, 2, 3, 1, True, 5.0),
    (1, 2, 3, 1, True, 5.0),
    (1, 3, 0, 1, False, NAN),
    (2, 0, 1, 0, True, 1.0),
    (2, 1, 2, 0, True, 3.0),
    (2, 2, 3, 1, True, 5.0),
    (3, 0, 1, 0, True, 1.0),
    (3, 1, 2, 0, True, 3.0),
    (3, 2, 3, 1, False, NAN),
    (3, 2, 3, 1, False, 4.0),
]


def test_fact_spans_reproduce_rows_with_duplicates_changes_and_reordering():
    step, source, target, code, present, value = (np.asarray(column) for column in zip(*ROWS))
    values = np.stack((value, np.full(len(value), NAN)), 1).astype(np.float32)
    encoded = spans.fact_spans(
        step.astype(np.int32),
        source.astype(np.int32),
        target.astype(np.int32),
        code.astype(np.int64),
        present.astype(bool),
        values,
    )
    assert len(encoded["fact_start"]) == 5 and encoded["order_step"].tolist() == [1]
    rows = spans.fact_rows(encoded, 4)
    assert np.array_equal(rows["fact_step"], step)
    assert np.array_equal(rows["fact_source"], source) and np.array_equal(rows["fact_target"], target)
    assert np.array_equal(rows["fact_code"], code) and np.array_equal(rows["fact_present"], present)
    table = np.concatenate((encoded["fact_values"], encoded["removal_values"]))
    assert np.array_equal(table[rows["fact_ref"]].view(np.uint32), values.view(np.uint32))


def test_observation_spans_split_on_changes_and_gaps():
    entity = np.asarray([0, 1, 0, 1, 0, 1, 1], np.int32)
    step = np.asarray([0, 0, 1, 1, 2, 3, 4], np.int32)
    x = np.asarray([[1], [2], [1], [2], [7], [2], [2]], np.float16)
    encoded = spans.observation_spans(entity, step, x)
    assert encoded["obs_entity"].tolist() == [0, 1, 0, 1]
    assert encoded["obs_start"].tolist() == [0, 0, 2, 3] and encoded["obs_stop"].tolist() == [2, 2, 3, 5]
    owner, moment, reference = spans.observation_rows(
        encoded["obs_entity"], encoded["obs_start"], encoded["obs_stop"]
    )
    assert np.array_equal(owner, entity) and np.array_equal(moment, step)
    assert np.array_equal(encoded["obs_x"][reference], x)
