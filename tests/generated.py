import copy
import logging
import time

import numpy as np

from graphspec import ENCODERS, GENERATORS, RULES, TYPES, Scripted
from graphspec.facts.stored import Facts
from graphspec.infra.timing import timed
from graphspec.schema.schema import Number, Value
from graphspec.sources import synthetic
from tuning.bank.build import build

LOG = logging.getLogger(__name__)
LATER = 3
KIND = "turn"
RULE = dict(name="spots", parameters=dict(speed=1.25))
STATED = [
    dict(name="others", seed=11, repeats=False, parameters=dict(count=2)),
    dict(
        name="nearest",
        seed=12,
        repeats=False,
        parameters=dict(count=2, states=2, columns=["load"], distance="euclidean"),
    ),
    dict(name="changed", seed=13, repeats=False, parameters=dict(count=1, ranges=dict(speed=[-0.5, 0.5]))),
    dict(name="ruled", seed=14, repeats=False, parameters=dict(count=2)),
    dict(name="around", seed=15, repeats=False, parameters={}),
    dict(name="crossings", seed=16, repeats=True, parameters=dict(count=1, mode="fast")),
    dict(name="idle", seed=17, repeats=False, parameters={}),
]
ORIGIN = "around"
LONE, SPEED = 41, 9.25
CASES = ("base", "later", "turned")
KEPT = []
NORTH, SOUTH = "north", "south"
WATCHED = "a0"
GAUGE = dict(kind="number", series="level", about=None, utility=synthetic.valued(["level"]))
SEEN = (
    ("spot", dict(key="s6", since=1, until=None, load=4.0, known=LATER + 1, replaced=None, observers=None)),
    ("spot", dict(key="s7", since=0, until=2, load=5.0, known=None, replaced=LATER + 1, observers=None)),
    ("spot", dict(key="s7", since=1, until=None, load=5.0, known=LATER + 1, replaced=None, observers=None)),
    ("spot", dict(key="s8", since=0, until=None, load=6.0, known=None, replaced=None, observers=[SOUTH])),
    ("near", dict(source="s1", target="s3", since=1, until=None, known=LATER + 1, replaced=None, observers=None)),
    ("near", dict(source="s0", target="s3", since=0, until=2, known=None, replaced=LATER + 1, observers=None)),
    ("near", dict(source="s0", target="s3", since=1, until=None, known=LATER + 1, replaced=None, observers=None)),
    ("near", dict(source="s0", target="s2", since=0, until=None, known=None, replaced=None, observers=[SOUTH])),
)


class Amount(Number):
    version = 1


class Plain(Value):
    version = 1
    types = ("amount",)


class Spots:
    version = 1

    @timed("generated.Spots.checked")
    def checked(self, parameters, schema, where):
        if not isinstance(parameters, dict) or set(parameters) != {"speed"} or type(parameters["speed"]) is not float:
            raise ValueError("%s reads speed, found %r" % (where, parameters))
        return parameters

    @timed("generated.Spots.options")
    def options(self, seen, parameters):
        state = seen.state
        return [
            dict(
                actions=[dict(kind="go", actor=actor, targets=dict(to=spot), parameters=dict(speed=parameters["speed"]))],
                together="set",
            )
            for actor in sorted(seen.decision["actors"])
            for spot in sorted(state["nodes"]["spot"])
        ] + [
            dict(
                actions=[dict(kind="hold", actor=actor, targets=dict(of=[actor, "base", spot]), parameters=dict(speed=parameters["speed"]))],
                together="set",
            )
            for actor in sorted(seen.decision["actors"])
            for spot in sorted(
                edge["source"] if edge["target"] == actor else edge["target"]
                for edge in state["edges"]
                if edge["relation"] == "base" and actor in (edge["source"], edge["target"])
            )
        ]


class Crossings:
    version = 1
    reads = ("state",)
    probable = True

    @timed("generated.Crossings.checked")
    def checked(self, parameters, schema, target, references, where):
        if not isinstance(parameters, dict) or set(parameters) != {"count", "mode"}:
            raise ValueError("%s reads count and mode, found %r" % (where, parameters))
        return parameters

    @timed("generated.Crossings.__call__")
    def __call__(self, seen, parameters):
        actor = sorted(seen.decision["actors"])[0]
        edges = sorted(
            {tuple(sorted((edge["source"], edge["target"]))) for edge in seen.state["edges"] if edge["relation"] == "near"}
        )
        made = [
            dict(
                actions=[
                    dict(kind="cut", actor=actor, targets=dict(along=[first, "near", second]), parameters=dict(mode=parameters["mode"]))
                ],
                together="set",
            )
            for first, second in edges
        ]
        order = np.argsort(seen.unit([first for first, _ in edges], [second for _, second in edges]), kind="stable")
        return [dict(made[at], drawn=min(parameters["count"], len(made)) / len(made)) for at in order[: parameters["count"]].tolist()]


class Faulty:
    version = 1
    reads = ("chosen", "decisions", "state")
    probable = False

    @timed("generated.Faulty.checked")
    def checked(self, parameters, schema, target, references, where):
        return parameters

    @timed("generated.Faulty.__call__")
    def __call__(self, seen, parameters):
        fault = parameters["fault"]
        if fault == "kept":
            KEPT.append(seen)
            return []
        if fault in ("idle", "twice"):
            return [dict(actions=[], together=together, drawn=None) for together in ("set", "sequence")[: 1 + (fault == "twice")]]
        if fault == "unread":
            return seen.rule
        if fault == "unobserved":
            cut = dict(kind="cut", actor=sorted(seen.decision["actors"])[0], targets=dict(along=["s0", "near", "s4"]), parameters=dict(mode="fast"))
            return [dict(actions=[cut], together="set", drawn=None)]
        copied = [dict(actions=option["actions"], together=option["together"], drawn=None) for option in seen.chosen if option["actions"]]
        return [dict(option, drawn=0.5) for option in copied] if fault == "probability" else copied


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


@timed("generated.idle")
def idle(seen, parameters):
    return [option for option in [dict(actions=[], together="sequence", drawn=None)] if seen.fresh(option)]


ENTRIES = (
    (GENERATORS, ORIGIN, Scripted(around, ("state", "chosen"), True, 1)),
    (GENERATORS, "idle", Scripted(idle, (), False, 1)),
    (GENERATORS, "crossings", Crossings()),
    (GENERATORS, "faulty", Faulty()),
    (RULES, "spots", Spots()),
    (TYPES, "amount", Amount()),
    (ENCODERS, "plain", Plain()),
)


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


SPAN = 2
HOME = dict(zip(synthetic.AGENTS, ("s0", "s2", "s4")))
MIRRORED = dict(
    tie=dict(roles=dict(by=dict(sort="edge", kinds=["agent"])), parameters=["speed"]),
    hold=dict(roles=dict(of=dict(sort="edge", kinds=["spot"])), parameters=["speed"]),
)
MIRRORED_ROLES = ("by", "of")


def tied(graph, place):
    started = time.perf_counter()
    rows = graph["edges"][1]["rows"]
    rows.update({name: [None] * len(rows["since"]) for name in ("known", "replaced", "observers")})
    tasks = graph["tasks"]
    for kind, moment, target in zip(tasks["kind"], tasks["time"], tasks["target"]):
        if kind != KIND or target is None:
            continue
        actor = target["options"][1]["actions"][0]["actor"]
        made = [
            synthetic.offered("tie", [synthetic.acting("tie", actor, dict(by=[actor, "base", HOME[actor]]), dict(speed=0.25 * moment))], "set", "recorded"),
            synthetic.offered("hold", [synthetic.acting("hold", actor, dict(of=[HOME[actor], "base", actor]), dict(speed=0.5 * moment + place))], "set", "recorded"),
        ]
        if (moment + place) % 3 == 0:
            for option in target["options"]:
                option.update(chosen=False, order=None)
            made[1]["chosen"] = True
        if moment % 2:
            target["options"] = [option for option in target["options"] if option["chosen"] or option["actions"]]
        target["options"] += made
    LOG.debug("generated tied exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


def lone(graph, place):
    started = time.perf_counter()
    tasks = graph["tasks"]
    (at,) = [at for at, (kind, key) in enumerate(zip(tasks["kind"], tasks["key"])) if kind == KIND and key == LONE]
    target, actor = tasks["target"][at], tasks["references"]["agents"][at][0]
    chosen = synthetic.offered("lone", [synthetic.acting("go", actor, dict(to="s3"), dict(speed=SPEED))], "set", "recorded")
    chosen.update(chosen=True, outcome=synthetic.worth(tasks["time"][at], place, 0))
    target["options"] = [chosen] + [option for option in target["options"] if option["origin"] != "recorded"]
    LOG.debug("generated lone exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


@timed("generated.load")
def load(at, moment, place):
    start = moment - moment % SPAN
    return 1.0 + at + place + 0.5 * ((start * (at + 1)) % 3)


@timed("generated.stated")
def stated(rows, **values):
    if not set(values) <= set(rows):
        raise ValueError("a row states %s, which its table does not" % ", ".join(sorted(set(values) - set(rows))))
    for name, held in rows.items():
        held.append(copy.deepcopy(values.get(name)))


def agents(place):
    started = time.perf_counter()
    rows = dict(key=[], since=[], until=[], load=[], known=[], replaced=[], observers=[])
    for at, key in enumerate(synthetic.AGENTS):
        for start in range(0, synthetic.TURNS[-1] + 1, SPAN) if key != WATCHED else ():
            until = None if start + SPAN > synthetic.TURNS[-1] else start + SPAN
            stated(rows, key=key, since=start, until=until, load=load(at, start, place))
    first, second, last = (load(0, start, place) for start in (0, SPAN, 3 * SPAN))
    for key, since, until, value, seen in (
        (WATCHED, 0, LATER, first, {}),
        (WATCHED, LATER, 3 * SPAN, second, dict(replaced=LATER + 1)),
        (WATCHED, 1, 3 * SPAN, second + 1.5, dict(known=LATER + 1)),
        (WATCHED, 3 * SPAN, None, last, {}),
        ("a1", 1, LATER, load(1, 0, place) + 0.75, dict(observers=[SOUTH])),
    ):
        stated(rows, key=key, since=since, until=until, load=value, **seen)
    LOG.debug("generated agents exit rows=%d seconds=%.4f", len(rows["key"]), time.perf_counter() - started)
    return rows


def watched(graph, place):
    started = time.perf_counter()
    tasks = graph["tasks"]
    tasks["observer"] = [
        SOUTH if kind == KIND and actors == [WATCHED] and moment % 2 else NORTH
        for kind, actors, moment in zip(tasks["kind"], tasks["references"]["agents"], tasks["time"])
    ]
    for at, (kind, moment) in enumerate(zip(tasks["kind"], tasks["time"])):
        if kind == "visit":
            tasks["target"][at] = 2.0 + moment + 0.5 * place
            tasks["split"][at] = "train" if moment % 2 else "validation"
            tasks["known"][at] = moment + 1
    statements = dict(spot=graph["values"][1]["rows"], near=graph["edges"][0]["rows"])
    for rows in statements.values():
        rows.update({name: [None] * len(rows["since"]) for name in ("known", "replaced", "observers")})
    for table, held in SEEN:
        stated(statements[table], **(dict(held, present=True) if table == "near" else held))
    graph["nodes"]["spot"] += sorted({held["key"] for table, held in SEEN if table == "spot"})
    LOG.debug("generated watched exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


@timed("generated.unseen")
def unseen(rows, at):
    return rows["observers"][at] == [SOUTH] or (rows["known"][at] or 0) > LATER


def later(graph):
    started = time.perf_counter()
    rows = graph["values"][0]["rows"]
    rows["load"] = [
        value + 10.0 if since > LATER else value + 5.0 if unseen(rows, at) else value
        for at, (value, since) in enumerate(zip(rows["load"], rows["since"]))
    ]
    spots, near = graph["values"][1]["rows"], graph["edges"][0]["rows"]
    for held in (spots, near):
        held["since"] = [LATER + 1 if unseen(held, at) else since for at, since in enumerate(held["since"])]
    spots["until"][1] = LATER + 1
    stated(spots, key="s1", since=LATER + 1, until=None, load=7.5)
    graph["nodes"]["spot"].append("s5")
    stated(spots, key="s5", since=LATER + 1, until=None, load=3.0)
    near["until"][0] = LATER + 2
    stated(near, source="s4", target="s5", present=True, since=LATER + 1, until=None)
    tasks = graph["tasks"]
    for at, (kind, moment, target) in enumerate(zip(tasks["kind"], tasks["time"], tasks["target"])):
        if kind == "visit":
            if moment > LATER:
                tasks["references"]["at"][at] = "s0"
            if tasks["split"][at] != "train" or tasks["known"][at] > LATER:
                tasks["target"][at] += 7.0
            continue
        if target is None:
            continue
        for option in target["options"]:
            held = option["outcome"]
            for part in held if isinstance(held, list) else [held]:
                for name in part or {}:
                    part[name] = 2.0 * part[name] + 1.0
            if moment <= LATER and tasks["observer"][at] != SOUTH:
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
    for target in (held for held in tasks["target"] if isinstance(held, dict)):
        target["options"].reverse()
        for option in target["options"]:
            if option["together"] == "set":
                option["actions"].reverse()
    LOG.debug("generated turned exit graph=%s seconds=%.4f", graph["name"], time.perf_counter() - started)


def case(name):
    started = time.perf_counter()
    result = synthetic.decisions(ORIGIN)
    schema = result["schema"]
    schema.update(time=dict(schema["time"], model="clocks"), observers=True)
    kinds = schema["tasks"]["kinds"]
    kinds[KIND]["target"].update(generators=copy.deepcopy(STATED), rule=copy.deepcopy(RULE))
    kinds[KIND]["target"]["actions"].update(copy.deepcopy(MIRRORED))
    kinds["visit"].update(target=copy.deepcopy(GAUGE), matures=dict(reference="who", column="load"))
    schema["edges"]["base"] = synthetic.edge("agent", "spot", [], directed=False)
    schema["columns"]["speed"] = dict(type="amount", encoder="plain")
    for place, graph in enumerate(result["graphs"]):
        graph["values"][0]["rows"] = agents(place)
        watched(graph, place)
        tied(graph, place)
        lone(graph, place)
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


@timed("generated.shows")
def shows(rows, at, moment, observer):
    since, until, known, replaced = (rows[name][at] for name in ("since", "until", "known", "replaced"))
    return (
        max(since, since if known is None else known) <= moment
        and (until is None or moment < until)
        and (replaced is None or moment < replaced)
        and rows["observers"][at] in (None, [observer])
    )


def holds(rows, key, moment, observer):
    started = time.perf_counter()
    result = any(found == key and shows(rows, at, moment, observer) for at, found in enumerate(rows["key"]))
    LOG.debug("generated holds exit key=%s seconds=%.6f", key, time.perf_counter() - started)
    return result


def linked(rows, first, second, moment, observer):
    started = time.perf_counter()
    result = any(
        {source, target} == {first, second} and present and shows(rows, at, moment, observer)
        for at, (source, target, present) in enumerate(zip(rows["source"], rows["target"], rows["present"]))
    )
    LOG.debug("generated linked exit seconds=%.6f", time.perf_counter() - started)
    return result


def observed(graph, option, moment, observer):
    started = time.perf_counter()
    spots, near, base = graph["values"][1]["rows"], graph["edges"][0]["rows"], graph["edges"][1]["rows"]
    result = all(
        holds(spots, aim, moment, observer)
        if role in ("to", "at")
        else linked(near, aim[0], aim[2], moment, observer)
        if role == "along"
        else linked(base, aim[0], aim[2], moment, observer)
        if role in MIRRORED_ROLES
        else True
        for action in option["actions"]
        for role, aim in action["targets"].items()
    )
    LOG.debug("generated observed exit seconds=%.6f", time.perf_counter() - started)
    return result


def spotted(graph, moment, observer):
    started = time.perf_counter()
    spots = graph["values"][1]["rows"]
    result = sorted({key for key in spots["key"] if holds(spots, key, moment, observer)})
    LOG.debug("generated spotted exit spots=%d seconds=%.6f", len(result), time.perf_counter() - started)
    return result


def edged(graph, moment, observer):
    started = time.perf_counter()
    near = graph["edges"][0]["rows"]
    result = sorted(
        {
            tuple(sorted((source, target)))
            for at, (source, target, present) in enumerate(zip(near["source"], near["target"], near["present"]))
            if present and shows(near, at, moment, observer)
        }
    )
    LOG.debug("generated edged exit edges=%d seconds=%.6f", len(result), time.perf_counter() - started)
    return result


def value(graph, key, moment, observer):
    started = time.perf_counter()
    tasks, rows = graph["tasks"], graph["values"][0]["rows"]
    ripe = [
        (known, since, y)
        for kind, who, since, y, split, known, seer in zip(
            tasks["kind"],
            tasks["references"]["who"],
            *(tasks[name] for name in ("time", "target", "split", "known", "observer")),
        )
        if kind == "visit" and who == key and split == "train" and seer == observer and known <= moment
    ]
    (found,) = [load for at, (owner, load) in enumerate(zip(rows["key"], rows["load"])) if owner == key and shows(rows, at, moment, observer)]
    result = max(ripe)[2] if ripe else found
    LOG.debug("generated value exit key=%s seconds=%.6f", key, time.perf_counter() - started)
    return result


@timed("generated.content")
def content(option):
    actions = []
    for action in option["actions"]:
        targets = {role: sorted(aim[::2]) + [aim[1]] if isinstance(aim, list) else aim for role, aim in action["targets"].items()}
        actions.append(repr((action["kind"], action["actor"], sorted(targets.items()), sorted(action["parameters"].items()))))
    together = option["together"] if len(actions) > 1 else "set"
    return (together, tuple(sorted(actions) if together == "set" else actions))


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


def retargeted(truth, chosen, shift):
    started = time.perf_counter()
    y = np.asarray(truth.target_y, np.float64)
    arrays = dict(truth.arrays, target_y=np.where(chosen, y + shift, y))
    result = Facts(truth.name, truth.keys, truth.unit, truth.inner, truth.observers, truth.options, truth.generators, **arrays)
    LOG.debug("generated retargeted exit targets=%d seconds=%.4f", int(np.sum(chosen)), time.perf_counter() - started)
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
