import logging
import time

import numpy as np
import pytest
import support
import torch
from keyed import Keys, dense_keyed, differing, variant_part

from graphspec.budgets import GraphLimits
from graphspec.engine import transform
from graphspec.engine.segments import Derivation
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.state import ENTITIES
from graphspec.views.graph import Dense
from tuning.bank.bank import Deriver

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("draws")
GROUPS = ("walks",)
LIMITS = GraphLimits("cpu")
DRAWN = {
    "time/periods": "every",
    "time/period_length": 3,
    "nodes/place/filter": "sample",
    "nodes/place/sample": 0.6,
    "nodes/item/filter": "sample",
    "nodes/item/sample": 0.7,
    "derive/hops": 2,
    "derive/rewiring": "sdrf",
    "derive/rewiring_budget": 3,
    "edges/visits/drop": 0.3,
    "edges/visits/persistence": 2,
    "edges/near/cap": 1,
    "edges/near/cap_order": "random",
    "edges/hop:2/drop": 0.3,
    "edges/rewire/drop": 0.2,
}
WALKS = {"walks/bias": 0.5, "walks/fanout": 3, "walks/window": 6}
JOINED = {"graphs/join": "single", "graphs/join_size": 2}
GENERATORS = (
    (np.random, ("default_rng", "Generator", "RandomState", "seed", "random", "rand", "choice", "permutation")),
    (torch, ("Generator", "manual_seed", "rand", "randn", "randint", "randperm", "multinomial", "bernoulli")),
)


class Unread:
    max_facts, max_working_bytes = LIMITS.max_facts, LIMITS.max_working_bytes

    @property
    def device(self):
        raise AssertionError("a derivation read the device")


def stated(registry, values, graph):
    started = time.perf_counter()
    entries, rest = support.pipelined(registry, values, support.SEVEN)
    assert rest == {}
    result = pipeline.resolved(registry, entries), specs.configured(
        specs.settings(registry, GROUPS), dict(WALKS, **graph)
    )
    LOG.debug("stated exit settings=%d seconds=%.4f", len(graph), time.perf_counter() - started)
    return result


def permuted(graph, seed):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    arrays = dict(graph.arrays)
    place = rng.permutation(len(graph.kinds))
    for name in ("kinds", "entity_token", "entity_connected"):
        arrays[name] = np.empty_like(np.asarray(arrays[name]))
        arrays[name][place] = np.asarray(graph.arrays[name])
    for name in ENTITIES:
        held = np.asarray(arrays[name])
        arrays[name] = np.where(held >= 0, place[np.maximum(held, 0)], -1).astype(held.dtype)
    keys = [None] * len(place)
    for entity, key in enumerate(graph.keys):
        keys[place[entity]] = key
    for table, reference in (("x_table", "obs_ref"), ("value_table", "fact_ref")):
        rows = rng.permutation(len(arrays[table]))
        arrays[table] = np.empty_like(np.asarray(arrays[table]))
        arrays[table][rows] = np.asarray(graph.arrays[table])
        held = np.asarray(arrays[reference])
        arrays[reference] = np.where(held >= 0, rows[np.maximum(held, 0)], -1).astype(held.dtype)
    order = np.lexsort((arrays["obs_entity"], arrays["obs_step"]))
    for name in ("obs_entity", "obs_step", "obs_ref", "obs_anchor"):
        arrays[name] = np.asarray(arrays[name])[order]
    result = Dense(graph.name, keys, graph.steps, **arrays)
    LOG.debug("permuted exit graph=%s seconds=%.4f", graph.name, time.perf_counter() - started)
    return result


def derived(members, registry, steps, spec, cuts, limits):
    started = time.perf_counter()
    derivation = Derivation.begun(members, registry, steps, spec, GROUPS, transform.SEED, limits, None, 0)
    for high in cuts + [sum(member.steps for member in members)]:
        derivation = derivation.continued(high, None)
    dense, arrays = derivation.derived(), derivation.arrays()
    keys = Keys(dense, registry)
    variant = variant_part(arrays, keys, list(arrays))
    result = dict(dense_keyed(dense, keys), **{"variant/" + name: held for name, held in variant.items()})
    LOG.info("keyed exit graphs=%d cuts=%d seconds=%.3f", len(members), len(cuts), time.perf_counter() - started)
    return result


@pytest.fixture(scope="module")
def drawn(bank):
    opened = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(opened)
    graphs = [support.shown(opened, index).dense() for index in range(2)]
    steps, spec = stated(registry, DRAWN, {})
    return registry, graphs, derived(graphs[:1], registry, steps, spec, [], LIMITS)


def test_permuting_entities_and_graphs_leaves_every_derived_array_equal_after_sorting_by_key(drawn):
    started = time.perf_counter()
    registry, graphs, reference = drawn
    moved = [permuted(graph, seed) for seed, graph in enumerate(graphs)]
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(moved[:1], registry, steps, spec, [4, 9], LIMITS), reference) == []
    assert len(reference["variant/walks"]) and registry.code("rewire") in reference["facts"][:, 2]
    unlinked = dict(JOINED, **{"graphs/connect/" + kind: False for kind in registry.nodes})
    steps, spec = stated(registry, DRAWN, unlinked)
    whole = derived(graphs, registry, steps, spec, [], LIMITS)
    turned = derived(moved[::-1], registry, steps, spec, [5, graphs[1].steps + 2], LIMITS)
    assert differing(turned, whole) == []
    steps, spec = stated(registry, DRAWN, dict(JOINED, **{"graphs/connect_fraction": 0.5}))
    linked = derived(graphs, registry, steps, spec, [], LIMITS)
    assert differing(derived(moved, registry, steps, spec, [graphs[0].steps + 3], LIMITS), linked) == []
    assert set(linked["entities"][:, -1].tolist()) == {0, 1}
    steps, spec = stated(registry, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    fosr = derived(graphs[1:], registry, steps, spec, [], LIMITS)
    assert differing(derived(moved[1:], registry, steps, spec, [7], LIMITS), fosr) == []
    LOG.info("permuted draws exit seconds=%.3f", time.perf_counter() - started)


def test_no_draw_reads_a_random_generator_or_a_device(drawn, monkeypatch):
    registry, graphs, reference = drawn
    for module, names in GENERATORS:
        for name in names:
            monkeypatch.setattr(module, name, lambda *args, name=name, **kwargs: pytest.fail("a draw used " + name))
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(graphs[:1], registry, steps, spec, [6], Unread()), reference) == []


@pytest.mark.gpu
def test_the_arrays_of_every_draw_are_equal_on_cpu_and_on_gpu(drawn):
    registry, graphs, reference = drawn
    steps, spec = stated(registry, DRAWN, {})
    assert differing(derived(graphs[:1], registry, steps, spec, [], GraphLimits("cuda")), reference) == []
    steps, spec = stated(registry, dict(DRAWN, **{"derive/rewiring": "fosr"}), {})
    on = [derived(graphs[1:], registry, steps, spec, [], GraphLimits(device)) for device in ("cpu", "cuda")]
    assert differing(on[1], on[0]) == []
