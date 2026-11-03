import copy
import json
from collections import Counter

import pytest
import support
from test_mappings import PLAN

from conftest import shared
from graphspec import GENERATORS, SCRIPTED, Builder, decisions, edges, nodes, scripted, source
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema.decisions import RECORDED
from graphspec.sources import synthetic
from tuning import build_bank
from tuning.bank.bank import Deriver
from tuning.bank.build import build

pytestmark = pytest.mark.builder
TICK = dict(type="time", unit="tick")
OMITTED = dict(type="number", missing="omit")
KEYS = dict(type="list", of="identifier")
SHIFTED = dict(name="shifted", seed=5, repeats=False, parameters=dict(count=2, by=0.5))
STATED = dict(turn=[dict(name="others", seed=11, repeats=False, parameters=dict(count=2))], show=[SHIFTED])
UTILITY = dict(name="weighted", better="higher", weights=dict(gain=1.0, cost=-1.0))
ROUTES = ("case", "script")


@timed("routes.shifted")
def shifted(seen, parameters):
    pool = {}
    for option in seen.chosen:
        for action in option["actions"]:
            speed = action["parameters"]["speed"] + parameters["by"]
            for spot in sorted(seen.state["nodes"]["spot"]):
                drawn = dict(actions=[dict(action, targets=dict(to=[spot]), parameters=dict(speed=speed))],
                             together="set", drawn=None)
                if seen.closed(drawn) is None and seen.fresh(drawn):
                    pool[seen.content(drawn)] = drawn
    tokens = sorted(pool)
    return [pool[token] for _, token in sorted(zip(seen.unit(tokens).tolist(), tokens))[: parameters["count"]]]


@pytest.fixture(scope="module")
def registered():
    scripted("generator", reason="no shipped generator moves a chosen action to every spot at another speed",
             reads=("state", "chosen"), probable=False, version=1)(shifted)
    yield
    del GENERATORS.held["shifted"]
    del SCRIPTED.held["shifted"]


@timed("routes.renumbered")
def renumbered(graph):
    tasks = graph["tasks"]
    order = sorted(range(len(tasks["key"])), key=lambda at: (("turn", "show", "visit").index(tasks["kind"][at]),
                                                             tasks["key"][at]))
    return {tasks["key"][at]: number for number, at in enumerate(order)}


@timed("routes.case")
def case():
    result = synthetic.decisions("unused")
    for kind, stated in STATED.items():
        result["schema"]["tasks"]["kinds"][kind]["target"]["generators"] = copy.deepcopy(stated)
    for graph in result["graphs"]:
        numbers, tasks = renumbered(graph), graph["tasks"]
        tasks["key"] = [numbers[key] for key in tasks["key"]]
        for target in filter(None, tasks["target"]):
            target["options"] = [option for option in target["options"] if option["origin"] == RECORDED]
            target["concurrent"] = [numbers[key] for key in target["concurrent"]]
            target["next"] = {actor: numbers[key] for actor, key in target["next"].items()}
    return result


@timed("routes.written")
def written(folder, name, rows, columns):
    (folder / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
    return source(name + ".json", "json", columns)


@timed("routes.tabled")
def tabled(given):
    found = dict(agent=[], spot=[], mark=[], near=[], base=[], decided=[], offered=[], acted=[])
    for graph in given["graphs"]:
        name = graph["name"]
        for statement in graph["values"]:
            rows = statement["rows"]
            for at, key in enumerate(rows["key"]):
                found[statement["kind"]].append(dict(graph=name, key=key, since=rows["since"][at],
                                                     **({"load": rows["load"][at]} if "load" in rows else {})))
        for statement in graph["edges"]:
            rows = statement["rows"]
            found[statement["table"]] += [dict(graph=name, source=first, target=second, since=since)
                                          for first, second, since in zip(rows["source"], rows["target"], rows["since"])]
        tasks = graph["tasks"]
        for at, (kind, key, moment, target) in enumerate(zip(tasks["kind"], tasks["key"], tasks["time"], tasks["target"])):
            if target is None:
                continue
            found["decided"].append(dict(graph=name, key=key, kind=kind, at=moment, split=tasks["split"][at],
                                         actors=tasks["references"]["agents"][at], ended=target["ended"],
                                         discount=target["discount"]))
            for option in target["options"]:
                outcome = option["outcome"] if isinstance(option["outcome"], dict) else {}
                found["offered"].append(dict(
                    graph=name, decision=key, key=option["key"], chosen=option["chosen"],
                    available=option["available"], together=option["together"], order=option["order"],
                    shown=option["shown"], logged=option["logged"], gain=outcome.get("gain"),
                    cost=outcome.get("cost"), known=moment + 1 if option["outcome"] is not None else None))
                parts = option["outcome"] if isinstance(option["outcome"], list) else [{}] * len(option["actions"])
                for place, (action, part) in enumerate(zip(option["actions"], parts)):
                    aims = action["targets"]
                    along = aims.get("along", [])
                    found["acted"].append(dict(
                        graph=name, decision=key, option=option["key"], kind=action["kind"], actor=action["actor"],
                        place=place, to=aims.get("to", []), at=aims.get("at", []), made=aims.get("made", []),
                        sources=[aim[0] for aim in along], targets=[aim[2] for aim in along],
                        speed=action["parameters"].get("speed"), mode=action["parameters"].get("mode"),
                        lit=action["parameters"].get("lit"), gain=part.get("gain"), cost=part.get("cost")))
    return found


@timed("routes.script")
def script(folder, given):
    rows = tabled(given)
    held = dict(graph="identifier", key="identifier", since=TICK)
    agents = written(folder, "agents", rows["agent"], dict(held, load="number"))
    spots = written(folder, "spots", rows["spot"], dict(held, load="number"))
    marks = written(folder, "marks", rows["mark"], held)
    ends = dict(graph="identifier", source="identifier", target="identifier", since=TICK)
    near, base = (written(folder, name, rows[name], ends) for name in ("near", "base"))
    decided = written(folder, "decided", rows["decided"], dict(
        graph="identifier", key="number", kind="identifier", at=TICK, split="identifier", actors=KEYS, ended="flag",
        discount=OMITTED))
    offered = written(folder, "offered", rows["offered"], dict(
        graph="identifier", decision="number", key="identifier", chosen="flag", available="flag",
        together="identifier", order=OMITTED, shown=OMITTED, logged=OMITTED, gain=OMITTED, cost=OMITTED,
        known=dict(TICK, missing="omit")))
    acted = written(folder, "acted", rows["acted"], dict(
        graph="identifier", decision="number", option="identifier", kind="identifier", actor="identifier",
        place="number", to=KEYS, at=KEYS, made=KEYS, sources=KEYS, targets=KEYS, speed=OMITTED,
        mode=dict(type="category", missing="omit"), lit=dict(type="flag", missing="omit"), gain=OMITTED,
        cost=OMITTED))
    holding = dict(holds=dict(name="state", since="since", until=None), known=None)
    agent = nodes(agents, key="key", graph="graph", features=dict(load="value"), **holding)
    spot = nodes(spots, key="key", graph="graph", features=dict(load="value"), **holding)
    mark = nodes(marks, key="key", graph="graph", **holding)
    ended = dict(graph="graph", source=dict(of=spot, key="source", absent="stop"), **holding)
    found = dict(agents=agents, spots=spots, marks=marks, near_rows=near, base_rows=base, decided=decided,
                 offered=offered, acted=acted, agent=agent, spot=spot, mark=mark,
                 near=edges(near, target=dict(of=spot, key="target", absent="stop"), directed=False, **ended),
                 base=edges(base, **dict(ended, source=dict(of=agent, key="source", absent="stop")),
                            target=dict(of=spot, key="target", absent="stop"), directed=True))
    kinds = dict(
        go=dict(roles=dict(to=dict(sort="node", of=spot, keys="to", count=[1, 1])), parameters=dict(speed="value")),
        cut=dict(roles=dict(along=dict(sort="edge", of=found["near"], sources="sources", targets="targets",
                                       count=[1, None])), parameters=dict(mode="onehot")),
        put=dict(roles=dict(at=dict(sort="node", of=spot, keys="at", count=[1, 2]),
                            made=dict(sort="new", of=mark, keys="made", count=[0, 1])), parameters=dict(lit="value")))
    for kind, fields in (("turn", dict(nothing="'wait'", concurrent="True")), ("show", {})):
        found[kind] = decisions(
            decided, key=["graph", "key"], at="at", graph="graph", where="kind == '%s'" % kind,
            references=dict(agents=dict(sort="nodes", of=agent, keys="actors", absent="stop")), actors="agents",
            series=["gain", "cost"], about=None, utility=UTILITY, split="split", ended="ended", discount="discount",
            generators=copy.deepcopy(STATED[kind]),
            options=dict(rows=offered, decision=["graph", "decision"], key="key", chosen="chosen",
                         available="available", together="together", order="order", shown="shown", logged="logged",
                         outcome=dict(gain="gain", cost="cost"), known="known"),
            actions=dict(rows=acted, decision=["graph", "decision"], option="option", kind="kind", actor="actor",
                         place="place", outcome=dict(gain="gain", cost="cost"), kinds=kinds),
            **fields)
    return found


@timed("routes.made")
def made(root):
    given = case()
    build(synthetic.stage(root, "staged", given), root / "case", 1, ("train",))
    sources = root / "sources"
    sources.mkdir()
    build_bank(Builder(script(sources, given), sources), root / "script", 1, ("train",), plan=PLAN)


@pytest.fixture(scope="module")
def banks(registered):
    root = shared("routes", made)
    return {name: support.read_back(Deriver(root / name, "cpu", support.SEED, RESIDENT)) for name in ROUTES}


@timed("routes.options")
def options(back):
    found = {}
    for name, graph in back.items():
        tasks = graph["tasks"]
        for kind, key, target in zip(tasks["kind"], tasks["key"], graph["decisions"]):
            if target is not None:
                found[(name, kind, key)] = sorted(target["options"], key=lambda option: option["key"])
    return found


def test_tables_and_a_named_generator_give_the_options_of_the_synthetic_decisions(banks):
    stated, staged, mapped = options(support.meant(case())), *(options(banks[name]) for name in ROUTES)
    assert stated.keys() == staged.keys() == mapped.keys()
    for decision, wanted in staged.items():
        assert mapped[decision] == wanted, decision
        assert [option for option in wanted if option["origin"] == RECORDED] == stated[decision], decision
    origins = Counter(option["origin"] if option["origin"] == RECORDED else option["origin"]["generator"]
                      for held in mapped.values() for option in held)
    assert origins == dict(recorded=114, others=36, shifted=12)
