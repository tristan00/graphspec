from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support import VALUES, asked, stated
from support import registry as registered

from graphspec import kernels, operators, spans, spec
from graphspec.declarations import EdgeType
from graphspec.infra import compute
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


def test_bounds_that_leave_out_a_default_leave_the_parameter_without_one():
    parameters = (spec.choice("view", ("snapshot", "memory", "tree"), "snapshot"), spec.flag("enabled", False))
    kept = spec.narrow(parameters, {"view": ["snapshot", "tree"]})
    assert kept[0].default == "snapshot" and kept[0].options == ("snapshot", "tree")
    left = spec.narrow(parameters, {"view": ["memory", "tree"], "enabled": True})
    assert left[0].default is None and left[1].default is None and left[1].options == (True,)
    with pytest.raises(
        ValueError, match=r"the pipeline states no view, which has no default; it is one of \['memory', 'tree'\]"
    ):
        spec.default(left)
    assert spec.configured(left, {"view": "tree", "enabled": True}) == {"view": "tree", "enabled": True}


def test_graph_registry_rejects_duplicate_relations():
    edge = EdgeType("rel", ("base", "rel"), [("person", "person")])
    with pytest.raises(ValueError, match="duplicate edge type name"):
        registered("test", ["person"], [edge, EdgeType("rel", ("rewire",), [("person", "person")])])


def test_persistence_eight_requires_eight_consecutive_observations(monkeypatch):
    relation = EdgeType("rel", ("base", "rel"), [("person", "person")])
    registry = registered("test", ["person"], [relation])
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
        EdgeType("beside", ("base", "beside"), [("place", "place")]),
        EdgeType("visits", ("base", "visits"), [("person", "place")]),
        EdgeType(
            "project:beside:sources:place:place",
            ("project", "beside", "sources", "place", "place"),
            [("place", "place")],
        ),
        EdgeType(
            "project:visits:sources:person:person",
            ("project", "visits", "sources", "person", "person"),
            [("person", "person")],
        ),
        EdgeType("diffusion", ("diffusion",), [("person", "place")]),
        EdgeType("rewire", ("rewire",), [("person", "place")]),
    ]
    registry = registered("test", ["person", "place"], edges)
    selected = {
        "edges/project:beside:sources:place:place/persistence",
        "edges/project:visits:sources:person:person/persistence",
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


def test_graph_tables_reject_unstated_columns_and_invalid_tasks():
    schema = stated(dict(client=["value"]), {}, VALUES, asked(["client"]))
    tables = GraphTables(0, "test", "step", schema).nodes("client", ["a", "b"])
    when = dict(since=0, until=None, seq=0)
    with pytest.raises(ValueError, match="states the columns value, found other"):
        tables.observe("client", ["a", "b"], when, other=[1, 1])
    with pytest.raises(ValueError, match="needs 2 values"):
        tables.observe("client", ["a", "b"], when, value=[1])
    holder, splits = dict(holder=("client", ["a", "b"])), ["train", "validation"]
    with pytest.raises(ValueError, match="one finite target"):
        tables.tasks("outcome", [1, 2], 0, holder, splits, [0, np.nan], [None, None], None, None)
    for call in (
        lambda: tables.observe("client", ["a", "b"], value=[1, 1]),
        lambda: tables.tasks("outcome", [1, 2], 0, holder, splits, [0, 1], [None, None], None),
        lambda: tables.tasks("outcome", [1, 2], 0, holder, splits),
    ):
        with pytest.raises(TypeError, match="when|observers|targets"):
            call()
    with pytest.raises(ValueError, match="states when it holds as exactly since, until, seq"):
        tables.observe("client", ["a"], dict(since=0), value=[1])
    with pytest.raises(ValueError, match="tasks state their time as a whole number"):
        tables.tasks("outcome", [1, 2], 0.5, holder, splits, [0, 1], [None, None], None, None)
    with pytest.raises(ValueError, match="the schema states no kind of task named income"):
        tables.tasks("income", [1, 2], 0, holder, splits, [0, 1], [None, None], None, None)
    with pytest.raises(ValueError, match="tasks of the kind outcome state the references holder, found none"):
        tables.tasks("outcome", [1, 2], 0, {}, splits, [0, 1], [None, None], None, None)
    with pytest.raises(ValueError, match="tasks of the kind outcome state no weights"):
        tables.tasks("outcome", [1, 2], 0, holder, splits, [0, 1], [None, None], [1.0, 2.0], None)
    with pytest.raises(ValueError, match="tasks of the kind outcome state one of targets for every task"):
        tables.tasks("outcome", [1, 2], 0, holder, splits, None, [None, None], None, None)
    with pytest.raises(ValueError, match="a graph states one of the units of time of its schema"):
        GraphTables(0, "test", "hour", schema)


def test_truth_shape_is_checked_before_compiled_audit(monkeypatch):
    graph = SimpleNamespace(
        name="invalid",
        steps=1,
        kinds=np.zeros(1, np.int32),
        obs_start=np.zeros(2, np.int32),
        fact_start=np.empty(0, np.int32),
        removal_step=np.empty(0, np.int32),
        task_key=np.empty(0, np.int64),
        member_task=np.empty(0, np.int32),
        label_task=np.empty(0, np.int32),
        arrays={"obs_entity": np.zeros(1, np.int32)},
    )

    def forbidden(*args):
        raise AssertionError("invalid arrays entered compiled indexing")

    monkeypatch.setattr(spans, "coverage", forbidden)
    with pytest.raises(ValueError, match="inconsistent obs row counts"):
        verify(graph, None)
