from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graphspec import kernels, operators, spans, spec
from graphspec.declarations import EdgeType
from graphspec.infra import compute
from graphspec.registry import Registry
from graphspec.resources import GraphBudget, GraphCapacityError, GraphLimits
from graphspec.staging import GraphTables
from graphspec.truth import verify


@pytest.mark.parametrize("width", [0, 1, 4, 8, 20])
def test_diffusion_top_matches_stable_sort_with_ties(width):
    scores = np.array(
        [[2, 2, 3, 1, 3, -np.inf, np.inf, 3], [np.nan, 0, np.nan, np.inf, 0, -np.inf, 0, 0]],
        np.float64,
    )
    values, indices = torch.sort(torch.from_numpy(scores), dim=1, descending=True, stable=True)
    actual_indices, actual_values = compute.top(scores, width, "cpu")
    np.testing.assert_array_equal(actual_indices, indices[:, :width].numpy())
    np.testing.assert_array_equal(actual_values, values[:, :width].numpy())


def test_completion_rejects_expansion_before_pair_allocation(monkeypatch):
    members = np.arange(100_000, dtype=np.int32)
    offsets = np.array([0, len(members)], np.int64)
    kinds = np.zeros(len(members), np.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation happened before capacity validation")

    monkeypatch.setattr(kernels.np, "empty", forbidden)
    with pytest.raises(ValueError, match="completion candidate edges"):
        kernels.within_pairs.py_func(members, offsets, kinds, 1, 100)


def test_projection_preserves_pairs_and_rejects_oversized_candidates():
    members = np.array([0, 1, 2, 3], np.int32)
    owners = np.array([4, 4, 4, 4], np.int32)
    kinds = np.zeros(5, np.int32)
    source, target = kernels.project_pairs.py_func(members, owners, kinds, 0, 0, 0, 6)
    assert list(zip(source.tolist(), target.tolist())) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    with pytest.raises(ValueError, match="projection candidate edges"):
        kernels.project_pairs.py_func(members, owners, kinds, 0, 0, 0, 5)


def test_rewiring_rejects_dense_workspace_before_allocating(monkeypatch):
    present = np.arange(100_000, dtype=np.int32)

    def forbidden(*args):
        raise AssertionError("dense graph was allocated")

    monkeypatch.setattr(operators, "dense", forbidden)
    with pytest.raises(GraphCapacityError, match="fosr rewiring"):
        operators.rewire(
            {"derive/rewiring": "fosr", "derive/rewiring_budget": 1},
            present,
            present[:-1],
            present[1:],
            len(present),
            np.random.default_rng(0),
            GraphBudget(GraphLimits("cpu")),
        )


@pytest.mark.parametrize("mode", ["fosr", "sdrf"])
def test_rewiring_does_not_allocate_for_isolated_nodes(mode):
    present = np.arange(100_000, dtype=np.int32)
    empty = np.empty(0, np.int32)
    added, removed = operators.rewire(
        {"derive/rewiring": mode, "derive/rewiring_budget": 8},
        present,
        empty,
        empty,
        len(present),
        np.random.default_rng(0),
        GraphBudget(GraphLimits("cpu", max_working_bytes=2048)),
    )
    assert not len(added[0]) and not len(added[1]) and not len(removed)


def test_resource_budget_is_shared_across_expansions():
    budget = GraphBudget(GraphLimits("cpu", max_facts=10, max_working_bytes=1 << 20))
    budget.retain(6, "projection")
    budget.retain(4, "completion")
    with pytest.raises(GraphCapacityError, match="remaining graph budget permits 0"):
        budget.retain(1, "diffusion")


def test_configuration_rejects_unused_overrides():
    parameters = (spec.flag("enabled", False), spec.number("width", 1, 8, 2, spec.equals("enabled", True)))
    assert spec.configured(parameters, {"enabled": True}) == {"enabled": True, "width": 2}
    with pytest.raises(ValueError, match="inactive or unknown"):
        spec.configured(parameters, {"width": 3})
    with pytest.raises(ValueError, match="match no parameters"):
        spec.narrow(parameters, {"widht": 3})
    with pytest.raises(ValueError, match="cannot be empty"):
        spec.narrow(parameters, {"enabled": []})


def test_graph_registry_rejects_duplicate_types():
    edge = EdgeType("rel", ("base",), [("person", "person")])
    with pytest.raises(ValueError, match="unique"):
        Registry("test", ["person", "person"], [edge], 1, [], {"protected": [], "base": ["rel"]})


def test_persistence_eight_requires_eight_consecutive_observations(monkeypatch):
    relation = EdgeType("rel", ("base",), [("person", "person")])
    registry = Registry(
        "test", ["person"], [relation], 1, [], {"protected": [], "base": ["rel"], "level": "node"}
    )
    steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 10, 11], np.int32)
    graph = SimpleNamespace(
        fact_step=steps,
        fact_code=np.zeros(len(steps), np.int32),
        fact_present=np.ones(len(steps), bool),
        fact_source=np.zeros(len(steps), np.int32),
        fact_target=np.ones(len(steps), np.int32),
    )
    monkeypatch.setattr(kernels, "runs", kernels.runs.py_func)
    settings = {"edges/rel/drop": 0.0, "edges/rel/persistence": 8, "edges/rel/cap": 0}
    retained = operators.filtered(graph, registry, settings, 1, "rel", np.arange(len(steps)))
    assert steps[retained].tolist() == [7, 7, 8]


def test_persistence_bounds_widen_only_the_three_named_relations():
    edges = [
        EdgeType("adjacent", ("base",), [("region", "region")]),
        EdgeType("can_act", ("base",), [("character", "region")]),
        EdgeType(
            "project:adjacent:sources:region:region",
            ("project", "adjacent", "sources", "region", "region"),
            [("region", "region")],
        ),
        EdgeType(
            "project:can_act:sources:character:character",
            ("project", "can_act", "sources", "character", "character"),
            [("character", "character")],
        ),
        EdgeType("diffusion", ("diffusion",), [("character", "region")]),
        EdgeType("rewire", ("rewire",), [("character", "region")]),
    ]
    registry = Registry(
        "test",
        ["character", "region"],
        edges,
        1,
        [],
        {"protected": [], "base": ["adjacent", "can_act"], "queries": [], "level": "node", "signals": 0},
    )
    selected = {
        "edges/project:adjacent:sources:region:region/persistence",
        "edges/project:can_act:sources:character:character/persistence",
        "edges/diffusion/persistence",
    }
    bounds = {
        "edges/*/persistence": {"low": 1, "high": 4},
        **{name: {"low": 1, "high": 8} for name in selected},
    }
    parameters = spec.narrow(spec.space(registry, ()), bounds)
    persistence = [parameter for parameter in parameters if parameter.name.endswith("/persistence")]
    assert len(persistence) == 6
    for parameter in persistence:
        maximum = 8 if parameter.name in selected else 4
        rng = np.random.default_rng(23)
        assert {parameter.uniform(rng) for _ in range(200)} == set(range(1, maximum + 1))
        assert all(parameter.check(value) == value for value in range(1, maximum + 1))
        with pytest.raises(ValueError):
            parameter.check(maximum + 1)


def test_graph_tables_reject_invalid_features_and_labels():
    tables = GraphTables(0, "test", 2).nodes("client", ["a", "b"])
    with pytest.raises(ValueError, match="finite matrix"):
        tables.observe("client", ["a", "b"], [[1, 1, 1, 1]])
    with pytest.raises(ValueError, match="one finite value"):
        tables.labels("client", ["a", "b"], [0, np.nan], ["train", "validation"])
    with pytest.raises(ValueError, match="nonnegative int32"):
        tables.observe("client", ["a"], [[1, 1]], step=0.5)


def test_truth_shape_is_checked_before_compiled_audit(monkeypatch):
    graph = SimpleNamespace(
        name="invalid",
        steps=1,
        kinds=np.zeros(1, np.int32),
        obs_start=np.zeros(2, np.int32),
        fact_start=np.empty(0, np.int32),
        removal_step=np.empty(0, np.int32),
        query_key=np.empty(0, np.int64),
        label_step=np.empty(0, np.int32),
        arrays={"obs_entity": np.zeros(1, np.int32)},
    )

    def forbidden(*args):
        raise AssertionError("invalid arrays entered compiled indexing")

    monkeypatch.setattr(spans, "coverage", forbidden)
    with pytest.raises(ValueError, match="inconsistent obs row counts"):
        verify(graph, None)
