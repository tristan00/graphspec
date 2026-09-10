import numpy as np

from research.graph_temporal.walk_cache import extend
from tests.test_graph_research_cache import make, records


def test_expanded_walks_preserve_originals_and_follow_causal_participant_paths():
    _, arrays = make(records(12))
    walks = extend(arrays)
    np.testing.assert_array_equal(walks[:, :, :8], arrays['walks'])
    np.testing.assert_array_equal(walks, extend(arrays))
    assert walks.shape == (12, 2, 32, 4) and np.any(walks[:, :, 8:] >= 0)
    for query in range(len(walks)):
        for side, root in enumerate((arrays['actor'][query], arrays['subject'][query])):
            for walk in walks[query, side]:
                current, cutoff = root, query * 2 + 1
                ended = False
                for event_id in walk:
                    if event_id < 0:
                        ended = True
                        continue
                    assert not ended
                    source, target, tick = arrays['events'][event_id, :3]
                    assert current in (source, target) and tick < cutoff
                    current, cutoff = (target if current == source else source), tick


def test_more_future_decisions_cannot_change_expanded_walk_prefix():
    _, before = make(records(6))
    _, after = make(records(12))
    np.testing.assert_array_equal(extend(before)[:, :, 8:], extend(after)[:6, :, 8:])
