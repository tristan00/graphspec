import numpy as np
import support

from conftest import SPLITS
from graphspec.budgets import GraphLimits
from graphspec.engine import join, segments
from graphspec.engine.segments import Derivation
from graphspec.engine.transform import SEED
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.hashing import hashed, token
from graphspec.operations.periods import PERIODS
from tuning.bank.bank import Bank, Deriver, distinct
from tuning.bank.variants import prepare
from tuning.batches.batching import Shuffle, batches, identities
from tuning.batches.inputs import recipe

configuration = support.configuration


def identical(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def test_single_join_builds_one_graph_per_group_and_keeps_every_block(bank):
    registry = Bank(bank).registry
    separate = configuration(registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS)
    single = configuration(
        registry,
        dict(view="snapshot"),
        {"time/history": 4, "graphs/join": "single", "graphs/join_size": 3},
        5,
        SPLITS,
    )
    prepare(bank, single, 1, "cpu")
    opened = Deriver(bank, "cpu", RESIDENT)
    assert opened.groups(single) == [[0, 1, 2], [3]]
    assert opened.graphs(single) == [0, 1]
    assert sorted(opened.sizes(single)) == [0, 1]
    keys = identities(opened, single)
    names = [graph["name"] for graph in opened.viewed(single).graphs]
    assert keys[1] == int(hashed(np.asarray([token(names[3])], np.uint64))[0]) and len(set(keys.values())) == 2
    for partition in (0, 1):
        blocks = opened.blocks(partition, 4, single)
        assert {block[0] for block in blocks} == {0, 1}
        assert len(blocks) == len(opened.blocks(partition, 4, separate))
        shuffled = batches(blocks, 2, opened.sizes(single), Shuffle(5, 0, keys), frame_budget=1024)
        assert sorted(block for group in shuffled for block in group) == sorted(blocks)


def test_single_join_groups_graphs_within_their_split(bank):
    single = configuration(
        Bank(bank).registry,
        dict(view="snapshot"),
        {"graphs/join": "single", "graphs/join_size": 3},
        5,
        SPLITS,
    )
    opened = Deriver(bank, "cpu", RESIDENT)
    partitions = opened.partitions(single)
    partitions[:] = 0
    held = opened.viewed(single).row_index[1]
    partitions[held[held >= 0]] = 1
    assert opened.groups(single) == [[0, 2, 3], [1]]


def joined_spec(opened, **graph):
    values = {"graphs/join": "single", "graphs/join_size": 4, **graph}
    return opened.variant(configuration(opened.registry, dict(view="snapshot"), values, 5, SPLITS))[1:3]


def test_unconnected_entities_end_with_their_graph(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    unlinked = {"graphs/connect/%s" % kind: False for kind in opened.registry.nodes}
    steps, spec = joined_spec(opened, **{"time/scheme": "state"}, **unlinked)
    graphs = [support.shown(opened, index).dense() for index in range(4)]
    combined = join.joined(graphs, support.fitted(opened), spec, SEED)
    derived = segments.whole(combined, {}, support.fitted(opened), steps, SEED, GraphLimits("cpu"), None)
    home = np.full(len(derived.kinds), -1)
    home[combined.obs_entity] = combined.step_graph[combined.obs_step]
    assert not derived.entity_connected.any()
    assert np.array_equal(home[derived.obs_entity], derived.step_graph[derived.obs_step])
    assert np.array_equal(home[derived.fact_source], derived.step_graph[derived.fact_step])
    assert len(derived.obs_step) > len(combined.obs_step)


def test_every_periods_restart_at_each_graph(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    steps, spec = joined_spec(opened, **{"time/periods": "every", "time/period_length": 5})
    graphs = [support.shown(opened, index).dense() for index in range(4)]
    registry = support.fitted(opened)
    periods = tuple(step for step in steps if step.operation is PERIODS)
    combined = segments.whole(
        join.joined(graphs, registry, spec, SEED), {}, registry, periods, SEED, GraphLimits("cpu"), None
    )
    starts = np.r_[0, np.cumsum([graph.steps for graph in graphs])]
    for first, last in zip(starts[:-1], starts[1:]):
        local = combined.period[first:last] - combined.period[first]
        assert np.array_equal(local, np.arange(last - first) // 5)
    assert np.all(np.diff(combined.period) >= 0)


def test_a_join_of_one_graph_equals_the_graph(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    values = {"graphs/join": "single", "graphs/join_size": 4, "time/scheme": "state"}
    _, steps, spec, chosen = opened.variant(
        configuration(opened.registry, dict(view="mixed"), values, 5, SPLITS)
    )
    graph, registry = support.shown(opened, 2).dense(), support.fitted(opened)
    separate = dict(spec, **{"graphs/join": "separate"})

    def variant(members, stated):
        derivation = Derivation.begun(
            members, {}, registry, steps, stated, chosen, SEED, GraphLimits("cpu"), None, 0
        )
        return derivation.continued(graph.steps, None).arrays()

    alone, together = variant([graph], separate), variant([graph], spec)
    assert sorted(alone) == sorted(together)
    assert all(identical(np.asarray(alone[name]), np.asarray(together[name])) for name in alone)


def test_window_counts_match_brute_force():
    rng = np.random.default_rng(3)
    values = rng.integers(0, 40, 500)
    lo = rng.integers(0, 500, 300)
    hi = np.minimum(lo + rng.integers(0, 120, 300), 500)
    counts = distinct(values, lo, hi, 40)
    assert counts.tolist() == [len(np.unique(values[a:b])) for a, b in zip(lo, hi)]


def test_joined_windows_use_their_own_entities(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    cfg = configuration(
        opened.registry,
        dict(view="memory"),
        {
            "time/history": 4,
            "graphs/join": "single",
            "graphs/join_size": 3,
            **{"graphs/connect/" + kind: False for kind in opened.registry.nodes},
        },
        5,
        SPLITS,
    )
    prepare(bank, cfg, 1, "cpu")
    opened = Deriver(bank, "cpu", RESIDENT)
    for block in opened.blocks(0, 4, cfg)[:12]:
        b = recipe(opened, block, cfg)
        ids = np.concatenate([np.asarray(b["member_entity"]), b["events"][:, :2].reshape(-1)])
        assert (
            b["entities"] == len(np.unique(ids[ids >= 0])) < len(opened.arrays(block[0], cfg)["entity_kind"])
        )
        assert ids.max() < b["entities"] and len(b["entity_batch"]) == b["entities"]
