import numpy as np
import pytest

from graphspec import join, representations, segments, transform
from graphspec import spec as specs
from graphspec.resources import GraphLimits
from graphspec.truth import DENSE, Graph
from tuning.data import Bank, identical


def shortened(graph, stop):
    arrays = dict(graph.arrays)
    for group, step in (("obs", "obs_step"), ("fact", "fact_step"), ("label", "label_step")):
        keep = np.asarray(graph.arrays[step]) < stop
        for name in arrays:
            if name.split("_", 1)[0] == group:
                arrays[name] = np.asarray(arrays[name])[keep]
    for name in arrays:
        if name.startswith("query_") or name in ("period", "step_campaign"):
            arrays[name] = np.asarray(arrays[name])[:stop]
    return type(graph)(graph.name, graph.keys, stop, **arrays)


@pytest.mark.parametrize("draw", range(6))
def test_segment_engine_equals_the_physical_join(bank, draw):
    opened = Bank(bank, device="cpu")
    registry = opened.registry
    rng = np.random.default_rng(draw)
    spec = specs.draw(
        specs.space(registry, specs.GROUPS),
        lambda parameter: "single" if parameter.name == "graphs/join" else parameter.uniform(rng),
    )
    graphs = [Graph.load(opened.truth(index)).dense() for index in range(4)]
    graphs = [shortened(graph, int(rng.integers(1, graph.steps + 1))) for graph in graphs]
    derived = transform.derive(join.joined(graphs, registry, spec), registry, spec, GraphLimits("cpu"))
    reference = representations.build(
        derived, registry, spec, specs.GROUPS, opened.categories, transform.SEED, GraphLimits("cpu")
    )
    plan = segments.Plan(graphs, registry, spec)
    outputs = segments.derived(plan, graphs, registry, spec, GraphLimits("cpu"))
    stitched = segments.stitch(plan, graphs, outputs)
    assert all(
        identical(np.asarray(derived.arrays[name]), np.asarray(stitched.arrays[name])) for name in DENSE
    )
    engine = segments.represented(
        plan, outputs, registry, spec, specs.GROUPS, opened.categories, transform.SEED, GraphLimits("cpu")
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
