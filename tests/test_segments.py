import numpy as np
import pytest
import support

from graphspec.budgets import GraphLimits
from graphspec.engine import join, segments, transform
from graphspec.infra.arrays import RESIDENT
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.representations import groups
from graphspec.views.graph import DENSE
from tuning.bank.bank import Deriver, identical

pytestmark = pytest.mark.xdist_group("segments")


def shortened(graph, stop):
    arrays = dict(graph.arrays)
    tasks = int((np.asarray(graph.task_step) < stop).sum())
    kept = dict(
        obs=np.asarray(graph.obs_step) < stop,
        fact=np.asarray(graph.fact_step) < stop,
        task=np.asarray(graph.task_step) < stop,
        member=np.asarray(graph.member_task) < tasks,
        label=np.asarray(graph.label_task) < tasks,
    )
    for name in arrays:
        group = name.split("_", 1)[0]
        if group in kept:
            arrays[name] = np.asarray(arrays[name])[kept[group]]
        elif name in ("period", "step_graph"):
            arrays[name] = np.asarray(arrays[name])[:stop]
    return type(graph)(graph.name, graph.keys, stop, **arrays)


@pytest.mark.parametrize("draw", range(6))
def test_segment_engine_equals_the_physical_join(bank, draw):
    opened = Deriver(bank, "cpu", RESIDENT)
    registry = support.fitted(opened)
    rng = np.random.default_rng(draw)
    template = [dict(operation=name) for name in support.SEVEN]
    stated = specs.draw(
        pipeline.space(registry, template),
        lambda parameter: True if parameter.name.count("/") == 1 else parameter.uniform(rng),
    )
    steps = pipeline.resolved(registry, pipeline.entries(template, stated))
    spec = specs.draw(
        specs.settings(registry, specs.GROUPS),
        lambda parameter: "single" if parameter.name == "graphs/join" else parameter.uniform(rng),
    )
    graphs = [support.shown(opened, index).dense() for index in range(4)]
    graphs = [shortened(graph, int(rng.integers(1, graph.steps + 1))) for graph in graphs]
    derived = transform.derive(join.joined(graphs, registry, spec), registry, steps, GraphLimits("cpu"), None)
    reference = groups.build(
        derived, registry, spec, specs.GROUPS, transform.SEED, GraphLimits("cpu")
    )
    plan = segments.Plan(graphs, registry, steps, spec)
    outputs = segments.derived(plan, graphs, registry, steps, GraphLimits("cpu"), None)
    stitched = segments.stitch(plan, graphs, outputs)
    assert all(
        identical(np.asarray(derived.arrays[name]), np.asarray(stitched.arrays[name])) for name in DENSE
    )
    engine = segments.represented(
        plan, outputs, registry, spec, specs.GROUPS, transform.SEED, GraphLimits("cpu")
    )
    assert sorted(engine) == sorted(reference)
    assert [
        name for name in reference if not identical(np.asarray(reference[name]), np.asarray(engine[name]))
    ] == []


def test_appender_writes_loadable_arrays_of_every_shape(tmp_path):
    pieces = [
        dict(
            flags=np.asarray([True, False]),
            wide=np.arange(6, dtype=np.float16).reshape(2, 3),
            hollow=np.zeros((2, 0, 2), np.float32),
            later=np.zeros(0, np.int64),
        ),
        dict(
            flags=np.asarray([True]),
            wide=np.arange(3, dtype=np.float16).reshape(1, 3),
            hollow=np.zeros((3, 0, 2), np.float32),
            later=np.asarray([7, 8], np.int64),
        ),
    ]
    appender = segments.Appender(tmp_path)
    for piece in pieces:
        appender.add(piece)
    appender.close()
    for name in pieces[0]:
        expected = np.concatenate([piece[name] for piece in pieces])
        loaded = np.load(tmp_path / (name + ".npy"), mmap_mode="r")
        assert loaded.dtype == expected.dtype and loaded.shape == expected.shape
        assert np.array_equal(np.asarray(loaded), expected)
    (tmp_path / "other").mkdir()
    strict = segments.Appender(tmp_path / "other")
    strict.add(dict(flags=np.zeros(1, bool)))
    with pytest.raises(ValueError, match="changed dtype or shape"):
        strict.add(dict(flags=np.zeros(1, np.int8)))
    strict.close()
