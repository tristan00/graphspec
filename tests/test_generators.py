import copy
import logging
import re
import time
import types
from collections import Counter

import generated
import numpy as np
import pytest
import support

from conftest import shared
from graphspec import GENERATORS, RULES, TYPES, Registry
from graphspec.build.generators import Check, Past, Seen, World, looking, regenerated, written
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


class Scalar:
    version = 1

    @timed("generators.Scalar.checked")
    def checked(self, parameters, schema, where):
        return parameters

    @timed("generators.Scalar.options")
    def options(self, seen, parameters):
        actor = sorted(seen.decision["actors"])[0]
        return [dict(actions=[dict(kind="go", actor=actor, targets=dict(to="s0"), parameters=dict(speed=1.25))], together="set")]


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
                    graph=graph,
                    place=place,
                    time=moment,
                    observer=tasks["observer"][at],
                    actors=tasks["references"]["agents"][at],
                    options=target["options"],
                    concurrent=target["concurrent"],
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
                and all(action["actor"] in own["actors"] and not (action["kind"] == "put" and action["targets"]["made"]) for action in option["actions"])
                and generated.observed(graph, option, own["time"], own["observer"])
                and generated.content(option) not in present
            ):
                result[generated.content(option)] = option
    LOG.debug("generators pool exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("generators.earlier")
def earlier(found, place):
    own = found[place]
    return [
        other
        for other, held in found.items()
        if other != place
        and other[0] == place[0]
        and held["observer"] == own["observer"]
        and held["time"] <= own["time"]
        and other[1] not in own["concurrent"]
    ]


def nearest(found, place):
    started = time.perf_counter()

    @timed("generators.nearest.state")
    def state(held):
        return sum(generated.value(held["graph"], actor, held["time"], held["observer"]) for actor in held["actors"]) / len(held["actors"])

    others = earlier(found, place)
    result = sorted(others, key=lambda other: (abs(state(found[other]) - state(found[place])), other[1]))[:2]
    LOG.debug("generators nearest exit decisions=%d seconds=%.4f", len(others), time.perf_counter() - started)
    return result


def checked(found, drawn, place, name, candidates, count):
    started = time.perf_counter()
    options = of(drawn[place], name)
    assert len(options) == min(count, len(candidates)), (place, name)
    for option in options:
        own = found[place]
        assert generated.content(option) in candidates and generated.observed(own["graph"], option, own["time"], own["observer"])
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
        spotted = generated.spotted(own["graph"], own["time"], own["observer"])
        spots = [dict(actions=[dict(kind="go", actor=actor, targets=dict(to=[spot]), parameters=dict(speed=1.25))], together="set") for actor in own["actors"] for spot in spotted]
        spots += [
            dict(actions=[dict(kind="hold", actor=actor, targets=dict(of=[[generated.HOME[actor], "base", actor]]), parameters=dict(speed=1.25))], together="set")
            for actor in own["actors"]
        ]
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
        pairs, crossed = generated.edged(own["graph"], own["time"], own["observer"]), of(drawn[place], "crossings")
        assert len(crossed) == min(1, len(pairs))
        for option in crossed:
            assert tuple(sorted(option["actions"][0]["targets"]["along"][0][::2])) in pairs and option["origin"]["drawn"] == 1 / len(pairs)
        for name in ("around", "crossings"):
            for option in of(drawn[place], name):
                assert generated.observed(own["graph"], option, own["time"], own["observer"])
                assert option["origin"]["drawn"] in {1 / count for count in range(1, 10)}
                counts[name] += 1
        idle = [option for option in own["options"] + drawn[place] if not option["actions"]]
        assert len(idle) == 1 and (idle[0]["origin"] == "recorded") == ("wait" in [option["key"] for option in own["options"]])
        counts["idle"] += len(idle) - len([option for option in own["options"] if not option["actions"]])
        present = [generated.content(option) for option in own["options"]]
        for option in drawn[place]:
            assert generated.content(option) not in present or option["origin"]["generator"] == "crossings"
            present.append(generated.content(option))
            assert (option["available"], option["chosen"], option["order"], option["shown"], option["logged"], option["outcome"]) == (True, False, None, None, None, None)
    mirrored = {
        (option["origin"]["generator"], action["kind"], [aims[0][0] in generated.HOME for aims in action["targets"].values()] == [action["kind"] == "tie"])
        for options in drawn.values()
        for option in options
        for action in option["actions"]
        if action["kind"] in generated.MIRRORED
    }
    wanted = {("others", "tie"), ("others", "hold"), ("nearest", "tie"), ("nearest", "hold"), ("changed", "hold"), ("ruled", "hold")}
    assert {(name, kind, True) for name, kind in wanted} <= mirrored and all(oriented for *_, oriented in mirrored), mirrored
    shapes = {
        (option["origin"]["generator"], len(aims))
        for options in drawn.values()
        for option in options
        for action in option["actions"]
        for aims in action["targets"].values()
    }
    assert {(name, count) for name in ("others", "nearest") for count in (0, 2)} <= shapes, shapes
    assert min(counts.values()) > 5, counts
    assert held.manifest["generators"] == dict(
        around=dict(version=1, probabilities=True),
        changed=dict(version=10, probabilities=False),
        crossings=dict(version=1, probabilities=True),
        idle=dict(version=1, probabilities=False),
        nearest=dict(version=10, probabilities=True),
        others=dict(version=10, probabilities=True),
        ruled=dict(version=10, probabilities=True),
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
            if isinstance(target, dict):
                target["options"] += copy.deepcopy(generated.drawn(held.back).get((graph["name"], key), []))
    meant = support.meant(case)
    for name, found in held.back.items():
        assert support.differing(found, meant[name], held.registry.schema) == [], name


@timed("generators.aims")
def aims(options):
    return {
        aim if isinstance(aim, str) else tuple(sorted(aim[::2]))
        for option in options
        for action in option["actions"]
        for aims in action["targets"].values()
        for aim in aims
    }


def test_no_drawn_option_depends_on_anything_known_after_its_decision_or_seen_by_another_observer(banks):
    started = time.perf_counter()
    base, later = (generated.drawn(opened(banks, name).back) for name in ("base", "later"))
    found = turns(generated.case("base"))
    assert sorted(base) == sorted(later) and set(found) <= set(base)
    early = [place for place in found if found[place]["time"] <= generated.LATER]
    north = [place for place in early if found[place]["observer"] == generated.NORTH]
    assert north and all(base[place] == later[place] for place in north)
    assert any(base[place] != later[place] for place in early if found[place]["observer"] == generated.SOUTH)
    assert any(base[place] != later[place] for place in found if found[place]["time"] > generated.LATER)
    for place, own in found.items():
        unseen = {"s6", ("s1", "s3")} if own["time"] <= generated.LATER else set()
        unseen |= {"s7", ("s0", "s3")} if 2 <= own["time"] <= generated.LATER else set()
        unseen |= {"s8", ("s0", "s2")} if own["observer"] == generated.NORTH else set()
        assert not aims(base[place]) & unseen, place
    LOG.info("causality of generators exit seconds=%.3f", time.perf_counter() - started)


def test_drawn_options_read_the_matured_targets_of_the_splits_trained_on_alone(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    seen, changed = looking(held.splits, held.manifest["trained"]), dict(train=False, validation=False)
    visit = list(held.registry.schema.tasks["kinds"]).index("visit")
    for entry in held.manifest["graphs"]:
        truth = generated.undrawn(Facts.load(held.truth(entry["index"])))
        drawn = regenerated(truth, held.registry, shown(truth, held.registry, seen)[1])
        owner = np.asarray(truth.target_task, np.int64)
        for split in changed:
            chosen = (np.asarray(truth.task_kind)[owner] == visit) & (np.asarray(truth.task_split)[owner] == held.splits.index(split))
            assert chosen.any()
            moved = generated.retargeted(truth, chosen, -3.0)
            changed[split] |= regenerated(moved, held.registry, shown(moved, held.registry, seen)[1]) != drawn
    assert changed == dict(train=True, validation=False)
    LOG.info("matured targets of generators exit seconds=%.3f", time.perf_counter() - started)


def test_an_option_recorded_is_made_canonical_at_most_twice_per_graph_however_many_decisions_read_it(banks, monkeypatch):
    started = time.perf_counter()
    held, calls, canonical = opened(banks, "base"), Counter(), World.canonical

    @timed("generators.counted")
    def counted(self, option, target):
        calls["recorded" if "origin" in option else "read" if "chosen" in option else "drawn", id(option)] += 1
        return canonical(self, option, target)

    monkeypatch.setattr(World, "canonical", counted)
    seen = looking(held.splits, held.manifest["trained"])
    for entry in held.manifest["graphs"]:
        calls.clear()
        truth = generated.undrawn(Facts.load(held.truth(entry["index"])))
        assert regenerated(truth, held.registry, shown(truth, held.registry, seen)[1])
        assert {sort for sort, _ in calls} == {"recorded", "drawn"}
        assert max(count for (sort, _), count in calls.items() if sort == "recorded") <= 2
    LOG.info("canonical options exit seconds=%.3f", time.perf_counter() - started)


def test_a_generator_that_reads_what_it_does_not_state_or_draws_what_its_statement_refuses_stops_naming_it(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    truth = generated.undrawn(Facts.load(held.truth(0)))
    where = "the generator of options faulty "
    drew = "the option %d the generator of options faulty drew for the decision 11 of the kind turn of graph g0 "
    faults = (
        ("unread", where + "reads the rule of the decision 11 of the kind turn of graph g0, which it does not state among what it reads: chosen, decisions, state"),
        ("unobserved", "names in its action 0 under the role along the edge \\['s0', 'near', 's4'\\], which does not hold at the step of its decision"),
        ("probability", "states the probability of its draw as 0.5, and faulty states none, having none"),
        ("repeated", drew % 0 + "equals an option already present"),
        ("twice", drew % 1 + "equals an option already present"),
        ("idle", drew % 0 + "holds no action, and its kind of task states no option that does nothing"),
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
        if fault == "idle":
            declared["schema"]["tasks"]["kinds"][generated.KIND]["target"]["nothing"] = None
        registry = Registry.from_declaration(declared)
        with pytest.raises(ValueError, match=message):
            regenerated(truth, registry, shown(truth, registry, looking(held.splits, held.manifest["trained"]))[1])
    RULES.register("scalar", Scalar())
    try:
        declared = copy.deepcopy(held.manifest["spec"])
        declared["schema"]["tasks"]["kinds"][generated.KIND]["target"].update(
            rule=dict(name="scalar", parameters={}), generators=[dict(name="ruled", seed=1, repeats=False, parameters=dict(count=1))]
        )
        registry = Registry.from_declaration(declared)
        scalar = "^the action 0 of the option 0 the rule scalar gives the decision \\d+ of the kind turn of graph g0 " + re.escape(
            "states its targets as a mapping from every role of its kind to the list of its targets, empty where it holds none, found {'to': 's0'}"
        )
        with pytest.raises(ValueError, match=scalar):
            regenerated(truth, registry, shown(truth, registry, looking(held.splits, held.manifest["trained"]))[1])
    finally:
        del RULES.held["scalar"]
    LOG.info("faulty generators exit seconds=%.3f", time.perf_counter() - started)


def test_the_content_of_an_option_reads_the_targets_of_a_role_as_a_set_and_a_role_holding_none_apart():
    started = time.perf_counter()
    cut = dict(kind="cut", actor="a0", targets=dict(along=[["s1", "near", "s0"], ["s2", "near", "s3"]]), parameters=dict(mode="fast"))
    put = dict(kind="put", actor="a0", targets=dict(at=["s3", "s1"], made=[]), parameters=dict(lit=True))
    stated = dict(actions=[cut, put], together="set")
    turned = dict(
        actions=[dict(put, targets=dict(at=["s1", "s3"], made=[])), dict(cut, targets=dict(along=[["s3", "near", "s2"], ["s0", "near", "s1"]]))],
        together="set",
    )
    found = {written(option, frozenset({"near"})) for option in (stated, turned)}
    others = [
        dict(stated, actions=[cut, dict(put, targets=dict(at=["s3"], made=[]))]),
        dict(stated, actions=[cut, dict(put, targets=dict(at=["s3", "s1"], made=["m1"]))]),
        dict(stated, actions=[dict(cut, targets=dict(along=[["s0", "near", "s1"]])), put]),
    ]
    assert len(found) == 1 and not found & {written(option, frozenset({"near"})) for option in others}
    assert len({written(option, frozenset({"near"})) for option in others}) == len(others)
    LOG.info("content of drawn options exit seconds=%.3f", time.perf_counter() - started)


@timed("generators.reached")
def reached(start):
    found, waiting, visited = [], [start], set()
    while waiting:
        held = waiting.pop()
        if id(held) in visited or isinstance(held, (type, types.ModuleType, logging.Logger, str, bytes, int, float, np.ndarray, np.generic)) or held is None:
            continue
        visited.add(id(held))
        found.append(held)
        if isinstance(held, dict):
            waiting += list(held) + list(held.values())
        elif isinstance(held, (list, tuple, set, frozenset)):
            waiting += list(held)
        elif isinstance(held, types.MethodType):
            waiting += [held.__self__, held.__func__]
        elif isinstance(held, types.FunctionType):
            waiting += [cell.cell_contents for cell in held.__closure__ or ()] + list(held.__defaults__ or ()) + [vars(held)]
        elif hasattr(held, "__dict__"):
            waiting.append(vars(held))
    return found


@timed("generators.scalars")
def scalars(entries):
    found = set()
    for held in entries:
        items = list(held) + list(held.values()) if isinstance(held, dict) else held if isinstance(held, (list, tuple, set, frozenset)) else ()
        found.update(item for item in items if isinstance(item, (str, float)))
    return found


def test_a_generator_holds_the_parts_it_states_and_reaches_nothing_else_of_the_build(banks):
    started = time.perf_counter()
    held = opened(banks, "base")
    truth = generated.undrawn(Facts.load(held.truth(0)))
    declared = copy.deepcopy(held.manifest["spec"])
    declared["schema"]["tasks"]["kinds"][generated.KIND]["target"]["generators"] = [
        dict(name="faulty", seed=1, repeats=False, parameters=dict(fault="kept"))
    ]
    registry = Registry.from_declaration(declared)
    generated.KEPT.clear()
    assert regenerated(truth, registry, shown(truth, registry, looking(held.splits, held.manifest["trained"]))[1]) == []
    assert len(generated.KEPT) > 5
    parts = ("decision", "unit", "closed", "fresh", "content", "formed", "kept", "chosen", "decisions", "state", "at")
    typed = {type(TYPES.named(name)) for name in TYPES.names()}
    back = held.back[truth.name]
    concurrent = {key: set(decision["concurrent"]) for key, decision in zip(back["tasks"]["key"], back["decisions"]) if decision}
    unknown, marked = Counter(), {}
    for seen in generated.KEPT:
        assert sorted(vars(seen)) == sorted(parts + ("generator", "reads", "where"))
        assert not any(hasattr(seen, name) for name in ("world", "frame", "task", "statement", "present"))
        with pytest.raises(ValueError, match="reads the rule of the decision"):
            seen.rule
        entries = reached(seen)
        found = {type(entry) for entry in entries if type(entry).__module__.split(".")[0] in ("graphspec", "tuning")}
        assert {Seen, Check, Past} <= found <= {Seen, Check, Past} | typed
        past = seen.at.__self__.held
        assert max(past) == seen.decision["time"] and all(type(state) is bytes for state in past.values())
        for other in seen.decisions:
            hidden = other["key"] in concurrent[seen.decision["key"]]
            unknown[hidden] += 1
            assert sorted(other) == (["actors", "key", "kind", "time"] if hidden else ["actors", "key", "kind", "options", "time"])
            assert hidden or all("chosen" in option for option in other["options"])
        if seen.decision["key"] != generated.LONE:
            marked.setdefault(seen.decision["key"] in concurrent[generated.LONE], set()).update({"lone", generated.SPEED} & scalars(entries))
    assert unknown[True] and unknown[False], unknown
    assert marked == {True: set(), False: {"lone", generated.SPEED}}, marked
    generated.KEPT.clear()
    LOG.info("parts of generators exit seconds=%.3f", time.perf_counter() - started)


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
        (2, dict(parameters=dict(count=1, ranges=dict(mode=[0, 1]))), "changes the parameter mode of the type category, which is not numeric"),
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
    for typed, encoder in (("flag", "value"), ("amount", "plain")):
        held = copy.deepcopy(stated)
        held["columns"]["load"] = dict(type=typed, encoder=encoder)
        held["tasks"]["kinds"]["visit"]["matures"] = None
        if typed == "flag":
            with pytest.raises(ValueError, match="measures the column load of the type flag, which is not numeric"):
                Schema(held)
        else:
            assert Schema(held).kind("load").numeric
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
