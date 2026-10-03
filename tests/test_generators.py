import copy
import logging
import time

import generated
import pytest
import support

from conftest import shared
from graphspec import GENERATORS, Registry
from graphspec.build.generators import looking, regenerated
from graphspec.engine.transform import shown, validate
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema import decisions
from graphspec.schema.schema import Schema
from graphspec.views.view import TASKS, sequences
from tuning.bank.bank import Deriver

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("generators")
OPENED = {}
NAMES = [held["name"] for held in generated.STATED]


class Bare:
    version = 1

    @timed("generators.Bare.__call__")
    def __call__(self, seen, parameters):
        return []


@pytest.fixture(scope="module")
def entries():
    generated.register()
    yield
    generated.unregister()
    OPENED.clear()


@pytest.fixture(scope="module")
def banks(entries):
    root = shared("generated", generated.made)
    return {name: root / name for name in generated.CASES}


@timed("generators.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", support.SEED, RESIDENT)
        OPENED[name].back = support.read_back(OPENED[name])
    return OPENED[name]


def turns(case):
    started = time.perf_counter()
    result = {}
    for place, graph in enumerate(case["graphs"]):
        tasks = graph["tasks"]
        for at, (kind, key, moment, target) in enumerate(zip(tasks["kind"], tasks["key"], tasks["time"], tasks["target"])):
            if kind == generated.KIND and target is not None:
                result[(graph["name"], key)] = dict(
                    graph=graph, place=place, time=moment, actors=tasks["references"]["agents"][at], options=target["options"]
                )
    LOG.debug("generators decisions exit decisions=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("generators.of")
def of(options, name):
    return [option for option in options if option["origin"]["generator"] == name]


@timed("generators.before")
def before(found, drawn, place, name):
    return [generated.content(option) for option in found[place]["options"]] + [
        generated.content(option) for option in drawn[place] if NAMES.index(option["origin"]["generator"]) < NAMES.index(name)
    ]


@timed("generators.bare")
def bare(option):
    return generated.content(dict(option, actions=[dict(action, parameters={}) for action in option["actions"]]))


def pool(found, drawn, place, chosen, name):
    started = time.perf_counter()
    graph, own = found[place]["graph"], found[place]
    present = set(before(found, drawn, place, name))
    result = {}
    for other in chosen:
        for option in found[other]["options"]:
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


@timed("generators.earlier")
def earlier(found, place):
    return [other for other, held in found.items() if other != place and other[0] == place[0] and held["time"] <= found[place]["time"]]


def nearest(found, place):
    started = time.perf_counter()

    @timed("generators.nearest.state")
    def state(held):
        return sum(generated.load(generated.synthetic.AGENTS.index(actor), held["time"], held["place"]) for actor in held["actors"]) / len(held["actors"])

    others = earlier(found, place)
    result = sorted(others, key=lambda other: (abs(state(found[other]) - state(found[place])), other[1]))[:2]
    LOG.debug("generators nearest exit decisions=%d seconds=%.4f", len(others), time.perf_counter() - started)
    return result


def checked(found, drawn, place, name, candidates, count):
    started = time.perf_counter()
    options = of(drawn[place], name)
    assert len(options) == min(count, len(candidates)), (place, name)
    for option in options:
        assert generated.content(option) in candidates and generated.observed(found[place]["graph"], option, found[place]["time"])
        assert option["origin"]["drawn"] == len(options) / len(candidates)
    LOG.debug("generators checked exit drawn=%d seconds=%.4f", len(options), time.perf_counter() - started)
    return len(options)


def test_each_shipped_generator_draws_options_whose_targets_resolve_and_are_observed_at_the_step_of_their_decision(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    found, drawn = turns(generated.case("base")), generated.drawn(held.back)
    counts = dict.fromkeys(NAMES, 0)
    for place, own in found.items():
        for name in ("others", "nearest"):
            chosen = earlier(found, place) if name == "others" else nearest(found, place)
            counts[name] += checked(found, drawn, place, name, pool(found, drawn, place, chosen, name), 2)
        spots = [dict(actions=[dict(kind="go", actor=actor, targets=dict(to=spot), parameters=dict(speed=1.25))], together="set") for actor in own["actors"] for spot in generated.synthetic.SPOTS]
        counts["ruled"] += checked(found, drawn, place, "ruled", {generated.content(option): option for option in spots}, 2)
        changed, recorded = of(drawn[place], "changed"), [option for option in own["options"] if option["chosen"] and any(action["parameters"].get("speed") is not None for action in option["actions"])]
        assert len(changed) == len(recorded)
        for option, chosen in zip(changed, sorted(recorded, key=lambda option: option["key"])):
            assert option["origin"]["drawn"] is None and bare(option) == bare(chosen)
            for new, old in zip(option["actions"], chosen["actions"]):
                assert {name: value for name, value in new["parameters"].items() if name != "speed"} == {
                    name: value for name, value in old["parameters"].items() if name != "speed"
                }
                if old["parameters"].get("speed") is not None:
                    assert abs(new["parameters"]["speed"] - old["parameters"]["speed"]) <= 0.5 + 1e-6
        counts["changed"] += len(changed)
        for name in ("around", "crossings"):
            for option in of(drawn[place], name):
                assert generated.observed(own["graph"], option, own["time"]) and option["origin"]["drawn"] in (1.0, 0.5, 0.25)
                counts[name] += 1
        present = [generated.content(option) for option in own["options"]]
        for option in drawn[place]:
            assert generated.content(option) not in present or option["origin"]["generator"] == "crossings"
            present.append(generated.content(option))
            assert (option["available"], option["chosen"], option["order"], option["shown"], option["logged"], option["outcome"]) == (True, False, None, None, None, None)
    assert min(counts.values()) > 5, counts
    assert held.manifest["generators"] == dict(
        around=dict(version=1, probabilities=True),
        changed=dict(version=1, probabilities=False),
        crossings=dict(version=1, probabilities=True),
        nearest=dict(version=1, probabilities=True),
        others=dict(version=1, probabilities=True),
        ruled=dict(version=1, probabilities=True),
    )
    for index in range(len(held.manifest["graphs"])):
        validate(Facts.load(held.truth(index)), held.registry, held.splits)
    decisions.drawn(held.registry.schema)
    LOG.info("generators exit counts=%s seconds=%.3f", counts, time.perf_counter() - started)


def test_a_drawn_option_reads_back_as_drawn_and_nothing_else_of_the_bank_changes(banks):
    held = opened(banks, "base")
    case = generated.case("base")
    for graph in case["graphs"]:
        for key, target in zip(graph["tasks"]["key"], graph["tasks"]["target"]):
            if target is not None:
                target["options"] += copy.deepcopy(generated.drawn(held.back).get((graph["name"], key), []))
    meant = support.meant(case)
    for name, found in held.back.items():
        assert support.differing(found, meant[name], held.registry.schema) == [], name


def test_no_drawn_option_depends_on_anything_after_its_decision(banks):
    started = time.perf_counter()
    base, later = (generated.drawn(opened(banks, name).back) for name in ("base", "later"))
    times = {place: held["time"] for place, held in turns(generated.case("base")).items()}
    assert sorted(base) == sorted(later)
    assert all(base[place] == later[place] for place in base if times.get(place, 0) <= generated.LATER)
    assert any(base[place] != later[place] for place in base if times.get(place, 0) > generated.LATER)
    LOG.info("causality of generators exit seconds=%.3f", time.perf_counter() - started)


def test_a_generator_that_reads_what_it_does_not_state_or_draws_what_its_statement_refuses_stops_naming_it(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    truth = Facts.load(held.truth(0))
    where = "the generator of options faulty "
    faults = (
        ("unread", where + "reads the state of the decision 10 of the kind turn of graph g0, which it does not state among what it reads: chosen"),
        ("unobserved", "names in its action 0 under the role along the edge \\['s0', 'near', 's4'\\], which does not hold at the step of its decision"),
        ("probability", "states the probability of its draw as 0.5, and faulty states none, having none"),
        ("repeated", "the option 0 the generator of options faulty drew for the decision 10 of the kind turn of graph g0 equals an option already present"),
    )
    trained = tuple(held.splits.index(name) for name in held.manifest["trained"])
    assert held.manifest["trained"] == ["train"] and set(held.splits) == {"train", "validation"}
    for steps in sequences(truth, held.registry, TASKS, looking(held.splits, held.manifest["trained"])):
        assert steps["matures"] == trained and steps["rules"] == ()
    for fault, message in faults:
        declared = copy.deepcopy(held.manifest["spec"])
        declared["schema"]["tasks"]["kinds"][generated.KIND]["target"]["generators"] = [
            dict(name="faulty", seed=1, repeats=False, parameters=dict(fault=fault))
        ]
        registry = Registry.from_declaration(declared)
        with pytest.raises(ValueError, match=message):
            regenerated(truth, registry, shown(truth, registry, looking(held.splits, held.manifest["trained"]))[1])
    LOG.info("faulty generators exit seconds=%.3f", time.perf_counter() - started)


def test_a_statement_of_generators_is_checked_with_the_schema_and_a_rule_is_drawn_only_by_a_generator_that_reads_it(
    entries,
):
    stated = generated.case("base")["schema"]
    target = ("tasks", "kinds", generated.KIND, "target")
    wrong = (
        (0, dict(name="unknown"), "no generator of options is registered under the name 'unknown'"),
        (0, dict(seed=-1), "states its seed as a whole number from 0 to"),
        (0, dict(repeats=None), "states whether an option it draws may equal an option already present"),
        (0, dict(parameters=dict(count=0)), "states count as a whole number from 1, found 0"),
        (0, dict(parameters=dict(count=1, other=2)), "holds fields it does not know: other"),
        (1, dict(parameters=dict(count=1, states=1, columns=["speed"], distance="euclidean")), "states its columns as a list"),
        (1, dict(parameters=dict(count=1, states=1, columns=["load"], distance="far")), "no distance between states is registered"),
        (2, dict(parameters=dict(count=1, ranges=dict(mode=[0, 1]))), "states its ranges as a mapping"),
        (2, dict(parameters=dict(count=1, ranges=dict(speed=[1, 0]))), "states its ranges as a mapping"),
    )
    for place, change, message in wrong:
        held = copy.deepcopy(stated)
        entry = held
        for name in target:
            entry = entry[name]
        entry["generators"][place].update(change)
        with pytest.raises((ValueError, KeyError), match=message):
            Schema(held)
    held = copy.deepcopy(stated)
    held["tasks"]["kinds"][generated.KIND]["target"]["rule"] = None
    with pytest.raises(ValueError, match="the generator 3 of the options of the target of the kind of task turn, ruled, draws from"):
        Schema(held)
    held = copy.deepcopy(stated)
    held["tasks"]["references"]["road"] = dict(sort="edge", kinds=["spot"])
    held["tasks"]["kinds"][generated.KIND]["references"].append("road")
    held["tasks"]["kinds"][generated.KIND]["target"]["about"] = "road"
    with pytest.raises(ValueError, match="states generators of options and is about road, whose facts a split's rule keeps"):
        Schema(held)
    held["tasks"]["kinds"][generated.KIND]["target"].update(generators=[], rule=None)
    Schema(held)
    GENERATORS.register("bare", Bare())
    try:
        held = copy.deepcopy(stated)
        held["tasks"]["kinds"][generated.KIND]["target"]["generators"][0]["name"] = "bare"
        with pytest.raises(ValueError, match="the generator of options bare states reads, a tuple of what it reads"):
            Schema(held)
    finally:
        del GENERATORS.held["bare"]
    held["tasks"]["kinds"][generated.KIND]["target"].update(rule=copy.deepcopy(generated.RULE), generators=generated.STATED[:3])
    with pytest.raises(ValueError, match="the kind of task turn states its options by the rule spots; the options"):
        decisions.drawn(Schema(held))
