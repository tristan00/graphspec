import copy
import logging
import time

import generated
import pytest
import support

from conftest import shared
from graphspec.engine.transform import validate
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema.schema import Schema
from tuning.bank.bank import Deriver

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("generators")
OPENED = {}


@pytest.fixture(scope="module")
def banks():
    generated.register()
    root = shared("generated", generated.made)
    yield {name: root / name for name in generated.CASES}
    generated.unregister()
    OPENED.clear()


@timed("generators.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", support.SEED, RESIDENT)
    return OPENED[name]


def decisions(case):
    started = time.perf_counter()
    result = {}
    for graph in case["graphs"]:
        tasks = graph["tasks"]
        for at, (kind, key, moment, target) in enumerate(zip(tasks["kind"], tasks["key"], tasks["time"], tasks["target"])):
            if kind == generated.KIND and target is not None:
                result[(graph["name"], key)] = dict(
                    graph=graph, time=moment, actors=tasks["references"]["agents"][at], options=target["options"]
                )
    LOG.debug("generators decisions exit decisions=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def pool(found, place):
    started = time.perf_counter()
    graph, own = found[place]["graph"], found[place]
    present = {generated.content(option) for option in own["options"]}
    result = {}
    for other, held in found.items():
        if other == place or other[0] != place[0] or held["time"] > own["time"]:
            continue
        for option in held["options"]:
            if (
                option["origin"] == "recorded"
                and option["actions"]
                and all(action["actor"] in own["actors"] and action["kind"] != "put" for action in option["actions"])
                and generated.observed(graph, option, own["time"])
                and generated.content(option) not in present
            ):
                result[generated.content(option)] = option
    LOG.debug("generators pool exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def test_others_draws_the_options_of_earlier_decisions_whose_targets_resolve_with_the_probability_of_each_draw(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    found, drawn = decisions(generated.case("base")), generated.drawn(support.read_back(held))
    counted = 0
    for place, own in found.items():
        options = [option for option in drawn[place] if option["origin"]["generator"] == "others"]
        candidates = pool(found, place)
        assert len(options) == min(2, len(candidates)), place
        for option in options:
            assert generated.content(option) in candidates and generated.observed(own["graph"], option, own["time"])
            assert option["origin"] == dict(generator="others", seed=11, drawn=len(options) / len(candidates))
            assert (option["available"], option["chosen"], option["shown"], option["logged"], option["outcome"]) == (
                True,
                False,
                None,
                None,
                None,
            )
        counted += len(options)
    assert counted > 10 and held.manifest["generators"] == dict(others=dict(version=1, probabilities=True))
    for index in range(len(held.manifest["graphs"])):
        validate(Facts.load(held.truth(index)), held.registry, held.splits)
    LOG.info("others exit drawn=%d seconds=%.3f", counted, time.perf_counter() - started)


def test_a_statement_of_generators_is_checked_with_the_schema_naming_what_it_refuses():
    stated = generated.case("base")["schema"]
    wrong = (
        (dict(name="unknown"), "no generator of options is registered under the name 'unknown'"),
        (dict(seed=-1), "states its seed as a whole number from 0 to"),
        (dict(repeats=None), "states whether an option it draws may equal an option already present"),
        (dict(parameters=dict(count=0)), "states count as a whole number from 1, found 0"),
        (dict(parameters=dict(count=1, other=2)), "holds fields it does not know: other"),
    )
    for change, message in wrong:
        held = copy.deepcopy(stated)
        held["tasks"]["kinds"][generated.KIND]["target"]["generators"][0].update(change)
        with pytest.raises((ValueError, KeyError), match=message):
            Schema(held)
