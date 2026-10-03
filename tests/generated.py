import copy
import logging
import time

import numpy as np

from graphspec import GENERATORS, Scripted
from graphspec.facts.stored import Facts
from graphspec.infra.timing import timed
from graphspec.sources import synthetic
from tuning.bank.build import build

LOG = logging.getLogger(__name__)
LATER = 3
KIND = "turn"
STATED = [dict(name="others", seed=11, repeats=False, parameters=dict(count=2))]
ORIGIN = "around"
CASES = ("base",)


@timed("generated.around")
def around(seen, parameters):
    state, found = seen.state, []
    for option in seen.chosen:
        for action in option["actions"]:
            if action["kind"] != "go":
                continue
            spot = action["targets"]["to"]
            near = sorted(
                {
                    edge["target"] if edge["source"] == spot else edge["source"]
                    for edge in state["edges"]
                    if edge["relation"] == "near" and spot in (edge["source"], edge["target"])
                }
            )
            made = [dict(actions=[dict(action, targets=dict(to=key))], together="set") for key in near]
            made = [held for held in made if seen.fresh(held)]
            if made:
                draws = seen.unit([held["actions"][0]["targets"]["to"] for held in made])
                found.append(dict(made[int(np.argmin(draws))], drawn=1 / len(made)))
    return found


ENTRIES = ((GENERATORS, ORIGIN, Scripted(around, ("state", "chosen"), True, 1)),)


def register():
    started = time.perf_counter()
    for names, name, entry in ENTRIES:
        names.register(name, entry)
    LOG.debug("generated register exit seconds=%.4f", time.perf_counter() - started)


def unregister():
    started = time.perf_counter()
    for names, name, _ in ENTRIES:
        del names.held[name]
    LOG.debug("generated unregister exit seconds=%.4f", time.perf_counter() - started)


@timed("generated.load")
def load(at, moment, place):
    return 1.0 + at + place + 0.5 * ((moment * (at + 1)) % 3)


def agents(place):
    started = time.perf_counter()
    rows = dict(key=[], since=[], until=[], load=[])
    for at, key in enumerate(synthetic.AGENTS):
        for moment in range(synthetic.TURNS[-1] + 1):
            rows["key"].append(key)
            rows["since"].append(moment)
            rows["until"].append(None if moment == synthetic.TURNS[-1] else moment + 1)
            rows["load"].append(load(at, moment, place))
    LOG.debug("generated agents exit rows=%d seconds=%.4f", len(rows["key"]), time.perf_counter() - started)
    return rows


def later(graph):
    started = time.perf_counter()
    rows = graph["values"][0]["rows"]
    rows["load"] = [value + 10.0 if since > LATER else value for value, since in zip(rows["load"], rows["since"])]
    spots = graph["values"][1]["rows"]
    spots["until"][1] = LATER + 1
    for name, value in (("key", "s1"), ("since", LATER + 1), ("until", None), ("load", 7.5)):
        spots[name].append(value)
    graph["nodes"]["spot"].append("s5")
    for name, value in (("key", "s5"), ("since", LATER + 1), ("until", None), ("load", 3.0)):
        spots[name].append(value)
    near = graph["edges"][0]["rows"]
    near["until"][0] = LATER + 2
    for name, value in (("source", "s4"), ("target", "s5"), ("present", True), ("since", LATER + 1), ("until", None)):
        near[name].append(value)
    tasks = graph["tasks"]
    for at, (moment, target) in enumerate(zip(tasks["time"], tasks["target"])):
        if target is None:
            if moment > LATER:
                tasks["references"]["at"][at] = "s0"
            continue
        for option in target["options"]:
            held = option["outcome"]
            for part in held if isinstance(held, list) else [held]:
                for name in part or {}:
                    part[name] = 2.0 * part[name] + 1.0
            if moment <= LATER:
                continue
            for action in option["actions"]:
                parameters = action["parameters"]
                if "speed" in parameters and parameters["speed"] is not None:
                    parameters["speed"] += 3.0
                if "mode" in parameters:
                    parameters["mode"] = "slow" if parameters["mode"] == "fast" else "fast"
    LOG.debug("generated later exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


def turned(graph):
    started = time.perf_counter()
    for kind in graph["nodes"]:
        graph["nodes"][kind].reverse()
    for statement in graph["values"] + graph["edges"]:
        for values in statement["rows"].values():
            values.reverse()
    tasks = graph["tasks"]
    for name, values in tasks.items():
        if name == "references":
            for held in values.values():
                held.reverse()
        else:
            values.reverse()
    for target in filter(None, tasks["target"]):
        target["options"].reverse()
    LOG.debug("generated turned exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


def case(name):
    started = time.perf_counter()
    result = synthetic.decisions(ORIGIN)
    result["schema"]["tasks"]["kinds"][KIND]["target"]["generators"] = copy.deepcopy(STATED)
    for place, graph in enumerate(result["graphs"]):
        graph["values"][0]["rows"] = agents(place)
        if name == "later":
            later(graph)
        if name == "turned":
            turned(graph)
    if name == "turned":
        result["graphs"].reverse()
    LOG.debug("generated case exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return result


def made(root):
    started = time.perf_counter()
    for name in CASES:
        build(synthetic.stage(root, name, case(name)), root / name, 1, ("train",))
    LOG.info("banks of generated options exit seconds=%.3f", time.perf_counter() - started)


def holds(rows, key, moment):
    started = time.perf_counter()
    result = any(
        found == key and since <= moment and (until is None or moment < until)
        for found, since, until in zip(rows["key"], rows["since"], rows["until"])
    )
    LOG.debug("generated holds exit key=%s seconds=%.6f", key, time.perf_counter() - started)
    return result


def linked(rows, first, second, moment):
    started = time.perf_counter()
    result = any(
        {source, target} == {first, second} and present and since <= moment and (until is None or moment < until)
        for source, target, present, since, until in zip(*(rows[name] for name in ("source", "target", "present", "since", "until")))
    )
    LOG.debug("generated linked exit seconds=%.6f", time.perf_counter() - started)
    return result


def observed(graph, option, moment):
    started = time.perf_counter()
    spots, near = graph["values"][1]["rows"], graph["edges"][0]["rows"]
    result = all(
        holds(spots, aim, moment) if role in ("to", "at") else linked(near, aim[0], aim[2], moment) if role == "along" else True
        for action in option["actions"]
        for role, aim in action["targets"].items()
    )
    LOG.debug("generated observed exit seconds=%.6f", time.perf_counter() - started)
    return result


@timed("generated.content")
def content(option):
    actions = []
    for action in option["actions"]:
        targets = {role: sorted(aim[::2]) + [aim[1]] if isinstance(aim, list) else aim for role, aim in action["targets"].items()}
        actions.append(repr((action["kind"], action["actor"], sorted(targets.items()), sorted(action["parameters"].items()))))
    return (option["together"], tuple(sorted(actions) if option["together"] == "set" else actions))


@timed("generated.ours")
def ours(key):
    return any(key.startswith("%s.%d." % (held["name"], place)) for place, held in enumerate(STATED))


def drawn(back):
    started = time.perf_counter()
    result = {}
    for name, graph in back.items():
        for key, decision in zip(graph["tasks"]["key"], graph["decisions"]):
            if decision is not None:
                result[(name, key)] = [option for option in decision["options"] if ours(option["key"])]
    LOG.debug("generated drawn exit decisions=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def undrawn(truth):
    started = time.perf_counter()
    keep = np.asarray([not ours(key) for key in truth.options], bool).reshape(-1)
    placed = np.cumsum(keep) - 1
    acted = keep[np.asarray(truth.action_option, np.int64)]
    numbered = np.cumsum(acted) - 1
    roled = acted[np.asarray(truth.role_action, np.int64)]
    names = [truth.generators[code] for code in np.asarray(truth.option_generator)[keep].tolist() if code >= 0]
    generators = list(dict.fromkeys(names))
    codes = [-1 if code < 0 else generators.index(truth.generators[code]) for code in np.asarray(truth.option_generator)[keep].tolist()]
    arrays = {name: np.asarray(values)[keep] for name, values in truth.arrays.items() if name.startswith("option_")}
    arrays.update({name: np.asarray(values)[acted] for name, values in truth.arrays.items() if name.startswith("action_")})
    arrays.update({name: np.asarray(values)[roled] for name, values in truth.arrays.items() if name.startswith("role_")})
    arrays["option_generator"] = np.asarray(codes, np.int32).reshape(-1)
    arrays["action_option"] = placed[arrays["action_option"]].astype(np.int32)
    arrays["role_action"] = numbered[arrays["role_action"]].astype(np.int32)
    target = np.asarray(truth.target_option, np.int64)
    arrays["target_option"] = np.where(target >= 0, placed[np.maximum(target, 0)], -1).astype(np.int32)
    result = Facts(
        truth.name,
        truth.keys,
        truth.unit,
        truth.inner,
        truth.observers,
        [key for key, kept in zip(truth.options, keep.tolist()) if kept],
        generators,
        **dict(truth.arrays, **arrays),
    )
    LOG.debug("generated undrawn exit options=%d seconds=%.4f", int(keep.sum()), time.perf_counter() - started)
    return result
