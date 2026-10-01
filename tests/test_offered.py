import copy
import logging
import time

import pytest
import support
import torch

from conftest import SPLITS
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from tuning.bank.bank import Deriver
from tuning.batches.batching import pack_batch
from tuning.families.model_spec import VIEWS, configuration, groups, space, views
from tuning.families.models import build
from tuning.families.periods import blocks as period_blocks
from tuning.fits.losses import Summed
from tuning.searches.selector import graph_space

LOG = logging.getLogger(__name__)
TINY = {"hidden": 4, "time/history": 2}
pytestmark = pytest.mark.xdist_group("offered")
ADDING = [
    dict(
        operation="graphspec.relation_forms",
        parameters={
            "relation_forms/types": "typed",
            "relation_forms/parallel": "sum",
            "relation_forms/reverse": True,
            "relation_forms/loops": True,
        },
    ),
    dict(
        operation="graphspec.nearby",
        parameters={"nearby/kinds": "person:place", "nearby/rule": "nearest", "nearby/k": 2, "nearby/encoder": "standard"},
    ),
    dict(
        operation="graphspec.measures",
        parameters={
            "measures/degree": True,
            "measures/components": True,
            "measures/cores": True,
            "measures/encoder": "logarithm",
        },
    ),
    dict(
        operation="graphspec.contract",
        parameters={"contract/grouping": "item:v0", "contract/aggregate": "mean", "contract/parallel": "keep"},
    ),
    dict(
        operation="graphspec.product",
        parameters={
            "product/path": "visits:forward:weight,visits:backward:weight",
            "product/length": 2,
            "product/aggregate": "product",
            "product/cap": 2,
            "product/encoder": "standard",
        },
    ),
    dict(
        operation="graphspec.negatives",
        parameters={"negatives/of": "link:visits:person", "negatives/rule": "degree", "negatives/count": 2},
    ),
]


@pytest.fixture(scope="module")
def held(bank):
    started = time.perf_counter()
    opened = Deriver(bank, "cpu", RESIDENT)
    result = dict(
        bank=opened,
        space=space(opened.registry),
        pipeline=support.pipelined(opened.registry, {})[0],
        graphs={},
    )
    LOG.info("offered bank exit seconds=%.4f", time.perf_counter() - started)
    return result


@timed("offered.ends")
def ends(parameter):
    return parameter.options if parameter.options else tuple(dict.fromkeys((parameter.low, parameter.high)))


@timed("offered.drawn")
def drawn(held, stated):
    model = specs.draw(held["space"], lambda parameter: stated.get(parameter.name, parameter.default))
    key = groups(model)
    if key not in held["graphs"]:
        held["graphs"][key] = graph_space(held["bank"].registry, model, {})
    graph = specs.draw(held["graphs"][key], lambda parameter: stated.get(parameter.name, parameter.default))
    return model, graph


@timed("offered.chosen")
def chosen(held, model, graph):
    return [parameter for parameter in held["space"] if parameter.name in model and parameter.name != "view"] + [
        parameter
        for parameter in held["graphs"][groups(model)]
        if parameter.name in graph
        and (parameter.name in specs.PACKING or parameter.name.split("/")[0] in specs.GROUPS)
    ]


@timed("offered.conditions")
def conditions(held, family):
    found, frontier = {}, [({}, set())]
    while frontier:
        following = []
        for stated, before in frontier:
            model, graph = drawn(held, dict(TINY, view=family, **stated))
            spec, offered = dict(model, **graph), chosen(held, model, graph)
            for parameter in offered:
                found.setdefault(parameter.name, (parameter, stated))
            following += [
                (dict(stated, **{parameter.name: value}), {parameter.name for parameter in offered})
                for parameter in offered
                if parameter.name not in before and parameter.name not in TINY
                for value in ends(parameter)
                if value != spec[parameter.name]
            ]
        frontier = following
    return found


@timed("offered.cases")
def cases(held, family, found):
    result, seen = [], []
    for name, (parameter, stated) in found.items():
        for value in ends(parameter):
            drawn_case = drawn(held, dict(TINY, view=family, **stated, **{name: value}))
            if drawn_case in seen:
                continue
            seen.append(drawn_case)
            where = " where " + ", ".join("%s=%s" % item for item in stated.items()) if stated else ""
            result.append(("%s=%s%s" % (name, value, where), *drawn_case))
    return result


@timed("offered.read")
def read(held, model, graph, pipeline):
    bank = held["bank"]
    cfg = configuration(bank.registry, model, pipeline, graph, 5, copy.deepcopy(SPLITS))
    block = bank.blocks(0, graph["time/history"], cfg)[-1]
    b = pack_batch(bank, period_blocks(bank, [block], cfg), cfg, "cpu")
    torch.manual_seed(5)
    return cfg, b, build(cfg, bank.fitted(cfg))


@timed("offered.passed")
def passed(held, model, graph, pipeline):
    bank = held["bank"]
    cfg, b, net = read(held, model, graph, pipeline)
    predicted = net(b)
    lost = Summed(dict(number="squared"), bank.registry.schema.targets())(predicted, b, bank)
    lost.backward()
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    if predicted.shape != b["series"].shape or not torch.isfinite(predicted).all():
        raise ValueError("predictions of shape %s are not finite or not of the targets' shape" % (predicted.shape,))
    if not torch.isfinite(lost) or not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise ValueError("the loss or a gradient is not finite")
    return cfg, b, predicted.detach()


@pytest.mark.parametrize("family", VIEWS)
def test_every_value_a_search_offers_a_family_packs_a_batch_and_passes_forward_and_backward(held, family):
    assert family in views(held["bank"].registry)
    found = conditions(held, family)
    assert {parameter.name for parameter in held["space"] if parameter.name.startswith(family + "/")} <= set(found)
    listed, failures = cases(held, family, found), []
    for label, model, graph in listed:
        try:
            passed(held, model, graph, held["pipeline"])
        except Exception as error:
            failures.append("%s: %r" % (label, error))
    LOG.info("offered family=%s cases=%d failures=%d", family, len(listed), len(failures))
    assert failures == [], "\n".join(failures)


@pytest.mark.parametrize("family", VIEWS)
def test_every_family_reads_the_kinds_relations_and_columns_a_pipeline_adds(held, family):
    started = time.perf_counter()
    model, graph = drawn(held, dict(TINY, view=family))
    pipeline = held["pipeline"][:-1] + ADDING
    cfg, b, predicted = passed(held, model, graph, pipeline)
    registry = held["bank"].fitted(cfg)
    assert registry.nodes[-1] == "contract" and "relation_forms.loops" in registry.by_name
    assert "product" in registry.by_name and registry.encoding.states["product.value"]
    assert "negatives" in registry.by_candidate and "negatives" not in registry.by_name
    assert registry.features > held["bank"].viewed(cfg).registry.features
    variant = held["bank"].arrays(0, cfg)
    assert len(variant["candidate_step"])
    assert set(variant["candidate_code"].tolist()) == {registry.by_candidate["negatives"].code}
    plain, unread, alone = read(held, model, graph, pipeline[:-1])
    assert "candidate_step" not in held["bank"].arrays(0, plain) and support.same(b, unread)
    with torch.no_grad():
        assert torch.equal(predicted, alone(unread))
    LOG.info("added read family=%s seconds=%.3f", family, time.perf_counter() - started)
