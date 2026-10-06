import json
import logging
import shutil
import sys
import time

import numpy as np
import pytest

from conftest import SPLITS, shared
from graphspec import OPERATIONS, Operation, Phase, describe
from graphspec.descriptions import describe as describing
from graphspec.descriptions import effect
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.timing import timed
from graphspec.operations import nodes as filtering
from graphspec.operations import parameters as specs
from graphspec.operations.operation import Need
from graphspec.operations.parameters import choice, equals, number
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.sources import synthetic
from tuning import REALISATIONS, Edited, configuration, reach
from tuning import describe as variant
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.descriptions import reach as reaching
from tuning.descriptions import variant as varying
from tuning.statement import VERSION as STATEMENT

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("descriptions")
NUMBER = dict(type="number", encoder="value")
CHANGED = dict(name="changed", seed=13, repeats=False, parameters=dict(count=1, ranges=dict(speed=[0.25, 0.5])))
HELD = dict(
    near=dict(table="near", ends=[["spot", "spot"]], columns=["weight"]),
    base=dict(table="base", ends=[["agent", "spot"]], columns=[]),
)
NONE = dict(least=None, most=None, mean=None)
FIRST = dict(
    name="g0",
    unit="tick",
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    kinds=dict(
        agent=dict(nodes=2, observations=3, nodes_at=[2, 2, 2, 2, 2, 2, 2, 2], observations_at=[2, 2, 2, 3, 2, 2, 2, 2]),
        spot=dict(nodes=3, observations=3, nodes_at=[2, 2, 3, 3, 3, 2, 2, 2], observations_at=[2, 2, 3, 3, 3, 2, 2, 2]),
    ),
    relations=dict(
        near=dict(
            HELD["near"], edges=2, facts=3, absent=1, edges_at=[1, 1, 2, 2, 1, 1, 0, 0], runs=2, bounded=2, least=2, most=6, mean=4.0
        ),
        base=dict(
            HELD["base"], edges=2, facts=3, absent=0, edges_at=[1, 2, 1, 1, 2, 2, 1, 1], runs=3, bounded=2, least=1, most=2, mean=1.5
        ),
    ),
    tasks=dict(
        turn=dict(tasks=3, splits=dict(train=2, validation=1), unsplit=0),
        visit=dict(tasks=4, splits=dict(train=2, validation=1), unsplit=1),
    ),
    decisions=dict(
        turn=dict(
            decisions=3,
            options=10,
            recorded=7,
            regenerated=3,
            generators=dict(changed=3),
            each=[
                dict(key=10, time=1, options=3, recorded=2, regenerated=1, generators=dict(changed=1)),
                dict(key=30, time=3, options=4, recorded=3, regenerated=1, generators=dict(changed=1)),
                dict(key=50, time=5, options=3, recorded=2, regenerated=1, generators=dict(changed=1)),
            ],
        )
    ),
)
SECOND = dict(
    name="g1",
    unit="tick",
    times=[1, 2, 3],
    kinds=dict(
        agent=dict(nodes=1, observations=1, nodes_at=[1, 1, 1], observations_at=[1, 1, 1]),
        spot=dict(nodes=2, observations=2, nodes_at=[2, 2, 1], observations_at=[2, 2, 1]),
    ),
    relations=dict(
        near=dict(HELD["near"], ends=[], edges=0, facts=0, absent=0, edges_at=[0, 0, 0], runs=0, bounded=0, **NONE),
        base=dict(HELD["base"], edges=1, facts=1, absent=0, edges_at=[1, 1, 0], runs=1, bounded=0, **NONE),
    ),
    tasks=dict(
        turn=dict(tasks=1, splits=dict(train=1, validation=0), unsplit=0),
        visit=dict(tasks=1, splits=dict(train=0, validation=1), unsplit=0),
    ),
    decisions=dict(
        turn=dict(
            decisions=1,
            options=3,
            recorded=2,
            regenerated=1,
            generators=dict(changed=1),
            each=[dict(key=10, time=2, options=3, recorded=2, regenerated=1, generators=dict(changed=1))],
        )
    ),
)
TEXT = """truth graphs of the bank stored as %s, described at version %d
time: one clock, units tick
splits: train, validation
%s

graph g0, unit tick, times 8 from 0 to 7
  kind agent: nodes 2, observations 3
  kind spot: nodes 3, observations 3
  relation near of the table near: ends spot to spot; columns weight; edges 2, facts 3, absent 1; runs 2, bounded 2, lasting 2 to 6, mean 4.0
  relation base of the table base: ends agent to spot; no columns; edges 2, facts 3, absent 0; runs 3, bounded 2, lasting 1 to 2, mean 1.5
  held at each time:
    time  agent nodes  agent observations  spot nodes  spot observations  near edges  base edges
    0     2            2                   2           2                  1           1
    1     2            2                   2           2                  1           2
    2     2            2                   3           3                  2           1
    3     2            3                   3           3                  2           1
    4     2            2                   3           3                  1           2
    5     2            2                   2           2                  1           2
    6     2            2                   2           2                  0           1
    7     2            2                   2           2                  0           1
  kind of task turn: tasks 3; train 2, validation 1, no split 0
  kind of task visit: tasks 4; train 2, validation 1, no split 1
  decisions of turn: decisions 3, options 10, recorded 7, regenerated 3; changed 3
    task  time  options  recorded  regenerated  changed
    10    1     3        2         1            1
    30    3     4        3         1            1
    50    5     3        2         1            1

graph g1, unit tick, times 3 from 1 to 3
  kind agent: nodes 1, observations 1
  kind spot: nodes 2, observations 2
  relation near of the table near: no ends; columns weight; edges 0, facts 0, absent 0; runs 0, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0; runs 1, bounded 0
  held at each time:
    time  agent nodes  agent observations  spot nodes  spot observations  near edges  base edges
    1     1            1                   2           2                  0           1
    2     1            1                   2           2                  0           1
    3     1            1                   1           1                  0           0
  kind of task turn: tasks 1; train 1, validation 0, no split 0
  kind of task visit: tasks 1; train 0, validation 1, no split 0
  decisions of turn: decisions 1, options 3, recorded 2, regenerated 1; changed 1
    task  time  options  recorded  regenerated  changed
    10    2     3        2         1            1
"""


@timed("descriptions.schema")
def schema():
    roles = dict(
        to=dict(sort="node", kinds=["spot"], count=[1, 2]),
        along=dict(sort="edge", kinds=["agent"], count=[0, None]),
        made=dict(sort="new", kinds=["spot"], count=[0, 1]),
    )
    go = dict(roles=roles, parameters=["speed"])
    decided = dict(
        kind="decision",
        series=["gain"],
        about=None,
        utility=synthetic.valued(["gain"]),
        actors="agents",
        actions=dict(go=go),
        nothing=None,
        rule=None,
        generators=[CHANGED],
    )
    level = dict(kind="number", series="level", about=None, utility=synthetic.valued(["level"]))
    return dict(
        version=SCHEMA,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(load=NUMBER, speed=NUMBER, weight=NUMBER),
        graph=dict(columns=[]),
        nodes=dict(agent=dict(columns=["load"], holds=None), spot=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(
            near=synthetic.edge("spot", "spot", ["weight"], directed=False), base=synthetic.edge("agent", "spot", [])
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(agents=dict(sort="nodes", kinds=["agent"]), at=dict(sort="node", kinds=["spot"])),
            kinds=dict(turn=synthetic.kind(["agents"], [], decided, None), visit=synthetic.kind(["at"], [], level, None)),
        ),
    )


UNAIMED = dict(along=[], made=[])
AIMED = dict(to=["s0", "s1"], along=[["a1", "base", "s1"]], made=["s2"])


@timed("descriptions.aiming")
def aiming(spot):
    return dict(UNAIMED, to=[spot])


@timed("descriptions.turned")
def turned(actor, aims, chosen, gain):
    options = [
        synthetic.offered("o%d" % at, [synthetic.acting("go", actor, targets, dict(speed=speed))], "set", "recorded")
        for at, (targets, speed) in enumerate(aims)
    ]
    options[chosen].update(chosen=True, outcome=dict(gain=gain))
    return dict(options=options, concurrent=[], next={}, ended=None, discount=None)


@timed("descriptions.rows")
def rows(names, *stated):
    return {name: [row[at] for row in stated] for at, name in enumerate(names)}


def asked(stated):
    started = time.perf_counter()
    result = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], references=dict(agents=[], at=[]))
    for key, kind, moment, split, referred, target in stated:
        for name, value in dict(key=key, kind=kind, time=moment, split=split, weight=None, target=target).items():
            result[name].append(value)
        result["known"].append(None if target is None else moment + 1)
        result["references"]["agents"].append(referred if kind == "turn" else None)
        result["references"]["at"].append(referred if kind == "visit" else None)
    LOG.debug("descriptions asked exit tasks=%d seconds=%.4f", len(stated), time.perf_counter() - started)
    return result


@timed("descriptions.graph")
def graph(name, nodes, agents, spots, edges, stated):
    values = [
        dict(form="state", parameters={}, rows=rows(("key", "since", "until", "load"), *held), kind=kind)
        for kind, held in (("agent", agents), ("spot", spots))
    ]
    facts = [
        dict(form="state", parameters={}, rows=rows(names, *held), table=table, ends=ends)
        for table, ends, names, held in edges
    ]
    return dict(name=name, unit="tick", columns={}, nodes=nodes, values=values, edges=facts, tasks=asked(stated))


NEAR = ("source", "target", "present", "since", "until", "weight")
BASE = ("source", "target", "present", "since", "until")


@timed("descriptions.case")
def case():
    first = graph(
        "g0",
        dict(agent=["a0", "a1"], spot=["s0", "s1", "s2"]),
        [("a0", 0, None, 1.0), ("a1", 0, 4, 2.0), ("a1", 3, None, 3.0)],
        [("s0", 0, None, 1.0), ("s1", 0, None, 2.0), ("s2", 2, 5, 3.0)],
        [
            (
                "near",
                ("spot", "spot"),
                NEAR,
                [
                    ("s0", "s1", True, 0, 3, 1.0),
                    ("s0", "s1", True, 3, 6, 2.0),
                    ("s0", "s1", False, 6, 7, None),
                    ("s1", "s2", True, 2, 4, 0.5),
                ],
            ),
            (
                "base",
                ("agent", "spot"),
                BASE,
                [("a0", "s0", True, 1, 2), ("a0", "s0", True, 4, 6), ("a1", "s1", True, 0, None)],
            ),
        ],
        [
            (10, "turn", 1, "train", ["a0"], turned("a0", [(aiming("s0"), 1.0), (AIMED, 2.0)], 0, 1.0)),
            (20, "visit", 2, "train", "s2", 1.5),
            (
                30,
                "turn",
                3,
                "train",
                ["a1"],
                turned("a1", [(aiming("s1"), 2.0), (aiming("s1"), 1.0), (aiming("s1"), 3.0)], 1, 3.0),
            ),
            (40, "visit", 4, "train", "s0", 2.5),
            (50, "turn", 5, "validation", ["a0"], turned("a0", [(aiming("s0"), 1.0), (aiming("s0"), 2.0)], 0, 2.0)),
            (51, "visit", 5, "validation", "s1", 0.5),
            (60, "visit", 6, None, "s0", None),
        ],
    )
    second = graph(
        "g1",
        dict(agent=["a0"], spot=["s0", "s1"]),
        [("a0", None, None, 1.0)],
        [("s0", None, 3, 1.0), ("s1", 1, None, 2.0)],
        [("base", ("agent", "spot"), BASE, [("a0", "s0", True, None, 3)])],
        [
            (20, "visit", 1, "validation", "s1", 1.0),
            (10, "turn", 2, "train", ["a0"], turned("a0", [(aiming("s0"), 1.0), (aiming("s1"), 2.0)], 0, 4.0)),
        ],
    )
    return dict(schema=schema(), graphs=[first, second])


@timed("descriptions.made")
def made(root):
    build(synthetic.stage(root, "described", case()), root / "bank", 1, ("train",))


@pytest.fixture(scope="module")
def bank():
    return shared("described", made) / "bank"


@timed("descriptions.expected")
def expected(stored):
    return dict(
        version=describing.VERSION,
        stored_sha256=stored,
        time=dict(model="clock", units=["tick"]),
        splits=["train", "validation"],
        graphs=[FIRST, SECOND],
    )


@timed("descriptions.files")
def files(bank, stored):
    stem = bank / "descriptions" / "truth" / sha256_json(dict(stored=stored, version=describing.VERSION))
    return stem.with_suffix(".json"), stem.with_suffix(".txt")


def test_every_number_of_the_description_of_a_synthetic_bank_equals_the_number_written_out_here(bank):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    record = describe(bank)
    assert record == expected(stored)
    assert json.loads(json.dumps(record)) == record


def test_the_text_holds_every_number_of_the_record_and_the_verb_prints_what_the_call_writes(bank, capsys, monkeypatch):
    from tuning.__main__ import main

    stored = read_json(bank / "manifest.json")["stored_sha256"]
    record = describe(bank)
    written = TEXT % (stored, describing.VERSION, describing.READING)
    assert describing.text(record) == written
    found, shown = files(bank, stored)
    assert read_json(found) == record and shown.read_text(encoding="utf-8") == written
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", str(bank)])
    main()
    assert capsys.readouterr().out == written


@timed("descriptions.refused")
def refused(*args):
    raise AssertionError("the description was computed again")


def test_a_second_call_reads_the_file_of_the_first_and_a_bank_of_other_stored_content_is_not_served_it(bank, tmp_path, monkeypatch):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    record = describe(bank)
    found, _ = files(bank, stored)
    assert found.exists()
    monkeypatch.setattr(describing, "recorded", refused)
    assert describe(bank) == record
    other = tmp_path / "bank"
    shutil.copytree(bank, other)
    assert files(other, stored)[0].exists()
    truth = other / "truth"
    (truth / "graph_0000").rename(truth / "held")
    (truth / "graph_0001").rename(truth / "graph_0000")
    (truth / "held").rename(truth / "graph_0001")
    manifest = read_json(other / "manifest.json")
    write_json(other / "manifest.json", dict(manifest, stored_sha256="0" * 64))
    with pytest.raises(AssertionError, match="the description was computed again"):
        describe(other)
    monkeypatch.undo()
    assert describe(other) == dict(record, stored_sha256="0" * 64, graphs=[SECOND, FIRST])
    assert files(other, "0" * 64)[0].exists() and describe(bank) == record


NODES = {
    "nodes/agent/keep": True,
    "nodes/spot/keep": True,
    "nodes/referenced": "keep",
    "nodes/agent/filter": "degree",
    "nodes/agent/min_degree": 1,
    "nodes/spot/filter": "none",
}
FORMS = {
    "relation_forms/types": "typed",
    "relation_forms/parallel": "keep",
    "relation_forms/reverse": False,
    "relation_forms/loops": True,
}
CUTTING = {"cut/relation": "base", "cut/kind": "turn"}
SETTINGS = {
    "time/history": 4,
    "time/context": 0,
    "graphs/join": "separate",
    "nodes/untyped": False,
    "relations/direction": "inverse",
    "time/steps": "changes",
    "time/several": "last",
}
MODEL = {"view": "snapshot", "options/realisation": "none"}
STATED = dict(
    version=STATEMENT,
    device="cpu",
    seed=5,
    derivation_seed=7,
    seconds=60,
    preparation_seconds=60,
    workers=1,
    variant_cache_bytes=1 << 30,
    batch_sequences=4,
    trained=["train"],
    scored=["validation"],
    sees=dict(train=[], validation=[]),
    loss=dict(number="squared", decision="squared"),
    measures=["mse"],
    alike=1,
)
LOOPS = "relation_forms.loops"
ZERO = [0] * 8
EMPTY = [0] * 3


@timed("descriptions.pipelined")
def pipelined(cutting):
    return [
        dict(operation="graphspec.nodes", parameters=NODES),
        dict(operation="test.cut", parameters=cutting),
        dict(operation="graphspec.relation_forms", parameters=FORMS),
    ]


@timed("descriptions.choices")
def choices(registry):
    return (
        choice("cut/relation", tuple(registry.groups), tuple(registry.groups)[0]),
        choice("cut/kind", registry.tasks, registry.tasks[0]),
    )


def unheld(graph, registry, settings, seed, limits, carry, held):
    started = time.perf_counter()
    gone = np.isin(graph.fact_code, registry.coded[settings["cut/relation"]]) & np.asarray(graph.fact_present, bool)
    if "cut/kind" in settings:
        tasked = np.asarray(graph.task_step)[np.asarray(graph.task_kind) == registry.tasks.index(settings["cut/kind"])]
        gone &= np.isin(graph.fact_step, tasked)
    fields = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    result = graph.replace(**{name: np.asarray(graph.arrays[name])[~gone] for name in fields})
    LOG.debug("descriptions unheld exit removed=%d seconds=%.4f", int(gone.sum()), time.perf_counter() - started)
    return result


CUT = Operation("test", "cut", (Phase("cut", unheld, None, None),), choices, (), (), (), ("facts",), None, None, 1, 1)
UNDECIDED = "decides, and the entry cuts only at the times of the tasks of kinds that decide nothing"


@timed("descriptions.undecided")
def undecided(registry, value):
    return UNDECIDED if value in registry.schema.decided() else None


SEVER = Operation(
    "test",
    "sever",
    (Phase("sever", unheld, None, None),),
    choices,
    (),
    (),
    (Need("cut/kind", undecided),),
    ("facts",),
    None,
    None,
    1,
    1,
)


@timed("descriptions.timed")
def timed_choices(registry):
    return (
        choice("cut/relation", tuple(registry.groups), tuple(registry.groups)[0]),
        choice("cut/times", ("all", "tasks"), "all"),
        choice("cut/kind", registry.tasks, registry.tasks[0], equals("cut/times", "tasks")),
    )


PART = Operation(
    "test",
    "part",
    (Phase("part", unheld, None, None),),
    timed_choices,
    (),
    (),
    (Need("cut/kind", undecided),),
    ("facts",),
    None,
    None,
    1,
    1,
)


@timed("descriptions.levelled")
def levelled(registry):
    return (number("read/level", 1, 3, 1, equals("cut/times", "tasks")),)


@timed("descriptions.passed")
def passed(graph, registry, settings, seed, limits, carry, held):
    return graph


READ = Operation("test", "read", (Phase("read", passed, None, None),), levelled, ("test.part",), (), (), (), None, None, 1, 1)


@pytest.fixture
def cutting():
    started = time.perf_counter()
    for held in (CUT, SEVER, PART, READ):
        OPERATIONS.register(held.name, held)
    LOG.debug("descriptions cutting registered seconds=%.4f", time.perf_counter() - started)
    yield CUT
    for held in (CUT, SEVER, PART, READ):
        del OPERATIONS.held[held.name]


@timed("descriptions.configured")
def configured(bank, cut):
    return configuration(bank, MODEL, None, pipeline=pipelined(cut), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.tally")
def tally(before, removed, removed_share, added, added_share, after, **held):
    return dict(
        before=before,
        removed=removed,
        removed_share=removed_share,
        added=added,
        added_share=added_share,
        after=after,
        **held,
    )


@timed("descriptions.holding")
def holding(agent, spot, near, base, loops):
    zero = [0] * len(agent)
    return dict(
        kinds=dict(agent=dict(nodes=agent, anchors=zero), spot=dict(nodes=spot, anchors=zero)),
        relations={"near": dict(edges=near), "base": dict(edges=base), LOOPS: dict(edges=loops)},
    )


@timed("descriptions.effected")
def effected(entry, kinds, relations, cuts):
    return dict(entry=entry, operation=entry, kinds=kinds, relations=relations, cuts=cuts)


VARIED = dict(
    name="g0",
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    tasks=dict(turn=[1, 3, 5], visit=[2, 4, 5, 6]),
    held=holding([2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 3, 3, 3, 2, 2, 2], [1, 1, 2, 2, 1, 1, 0, 0], [1, 2, 1, 1, 2, 2, 1, 1], ZERO),
    effects=[
        effected(
            "graphspec.nodes",
            dict(
                agent=tally(16, 5, 0.3125, 0, 0.0, 11, nodes=[1, 2, 1, 1, 2, 2, 1, 1], anchors=ZERO),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=[2, 2, 3, 3, 3, 2, 2, 2], anchors=ZERO),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=[1, 1, 2, 2, 1, 1, 0, 0]),
                "base": tally(11, 0, 0.0, 0, 0.0, 11, edges=[1, 2, 1, 1, 2, 2, 1, 1]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=ZERO),
            },
            [],
        ),
        effected(
            "test.cut",
            dict(
                agent=tally(11, 0, 0.0, 0, 0.0, 11, nodes=[1, 2, 1, 1, 2, 2, 1, 1], anchors=ZERO),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=[2, 2, 3, 3, 3, 2, 2, 2], anchors=ZERO),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=[1, 1, 2, 2, 1, 1, 0, 0]),
                "base": tally(11, 5, 0.4545, 0, 0.0, 6, edges=[1, 0, 1, 0, 2, 0, 1, 1]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=ZERO),
            },
            [dict(sort="relation", name="base", task="turn", times=[1, 3, 5], everywhere=False)],
        ),
        effected(
            "graphspec.relation_forms",
            dict(
                agent=tally(11, 0, 0.0, 0, 0.0, 11, nodes=[1, 2, 1, 1, 2, 2, 1, 1], anchors=ZERO),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=[2, 2, 3, 3, 3, 2, 2, 2], anchors=ZERO),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=[1, 1, 2, 2, 1, 1, 0, 0]),
                "base": tally(6, 0, 0.0, 0, 0.0, 6, edges=[1, 0, 1, 0, 2, 0, 1, 1]),
                LOOPS: tally(0, 0, None, 30, None, 30, edges=[3, 4, 4, 4, 5, 4, 3, 3]),
            },
            [],
        ),
    ],
)
VARIED_SECOND = dict(
    name="g1",
    times=[1, 2, 3],
    tasks=dict(turn=[2], visit=[1]),
    held=holding([1, 1, 1], [2, 2, 1], EMPTY, [1, 1, 0], EMPTY),
    effects=[
        effected(
            "graphspec.nodes",
            dict(
                agent=tally(3, 1, 0.3333, 0, 0.0, 2, nodes=[1, 1, 0], anchors=EMPTY),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[2, 2, 1], anchors=EMPTY),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=EMPTY),
                "base": tally(2, 0, 0.0, 0, 0.0, 2, edges=[1, 1, 0]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=EMPTY),
            },
            [],
        ),
        effected(
            "test.cut",
            dict(
                agent=tally(2, 0, 0.0, 0, 0.0, 2, nodes=[1, 1, 0], anchors=EMPTY),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[2, 2, 1], anchors=EMPTY),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=EMPTY),
                "base": tally(2, 1, 0.5, 0, 0.0, 1, edges=[1, 0, 0]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=EMPTY),
            },
            [dict(sort="relation", name="base", task="turn", times=[2], everywhere=False)],
        ),
        effected(
            "graphspec.relation_forms",
            dict(
                agent=tally(2, 0, 0.0, 0, 0.0, 2, nodes=[1, 1, 0], anchors=EMPTY),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[2, 2, 1], anchors=EMPTY),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=EMPTY),
                "base": tally(1, 0, 0.0, 0, 0.0, 1, edges=[1, 0, 0]),
                LOOPS: tally(0, 0, None, 7, None, 7, edges=[3, 3, 1]),
            },
            [],
        ),
    ],
)
STEPS = "    time  agent nodes  agent anchors  spot nodes  spot anchors  near edges  base edges  relation_forms.loops edges"
EFFECT = "    held                        before  removed  share    added  share    after"
VARIANT = """variant %s of the bank stored as %s, described at version %d
derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
%s
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree 1, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none

graph g0, 8 steps from time 0 to 7
  tasks: turn at 1, 3, 5; visit at 2, 4, 5, 6
  held at each time before the first entry:
%(steps)s
    0     2            0              2           0             1           1           0
    1     2            0              2           0             1           2           0
    2     2            0              3           0             2           1           0
    3     2            0              3           0             2           1           0
    4     2            0              3           0             1           2           0
    5     2            0              2           0             1           2           0
    6     2            0              2           0             0           1           0
    7     2            0              2           0             0           1           0
  entry graphspec.nodes, of the operation graphspec.nodes:
%(effect)s
    agent nodes                 16      5        0.3125   0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  11      0        0.0      0      0.0      11
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  held at each time after it:
%(steps)s
    0     1            0              2           0             1           1           0
    1     2            0              2           0             1           2           0
    2     1            0              3           0             2           1           0
    3     1            0              3           0             2           1           0
    4     2            0              3           0             1           2           0
    5     2            0              2           0             1           2           0
    6     1            0              2           0             0           1           0
    7     1            0              2           0             0           1           0
  entry test.cut, of the operation test.cut:
%(effect)s
    agent nodes                 11      0        0.0      0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  11      5        0.4545   0      0.0      6
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  held at each time after it:
%(steps)s
    0     1            0              2           0             1           1           0
    1     2            0              2           0             1           0           0
    2     1            0              3           0             2           1           0
    3     1            0              3           0             2           0           0
    4     2            0              3           0             1           2           0
    5     2            0              2           0             1           0           0
    6     1            0              2           0             0           1           0
    7     1            0              2           0             0           1           0
  entry test.cut, of the operation test.cut, removes every edge of the relation base at the times of the tasks of turn: 1, 3, 5; not at every time
  entry graphspec.relation_forms, of the operation graphspec.relation_forms:
%(effect)s
    agent nodes                 11      0        0.0      0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  6       0        0.0      0      0.0      6
    relation_forms.loops edges  0       0        nothing  30     nothing  30
  held at each time after it:
%(steps)s
    0     1            0              2           0             1           1           3
    1     2            0              2           0             1           0           4
    2     1            0              3           0             2           1           4
    3     1            0              3           0             2           0           4
    4     2            0              3           0             1           2           5
    5     2            0              2           0             1           0           4
    6     1            0              2           0             0           1           3
    7     1            0              2           0             0           1           3

graph g1, 3 steps from time 1 to 3
  tasks: turn at 2; visit at 1
  held at each time before the first entry:
%(steps)s
    1     1            0              2           0             0           1           0
    2     1            0              2           0             0           1           0
    3     1            0              1           0             0           0           0
  entry graphspec.nodes, of the operation graphspec.nodes:
%(effect)s
    agent nodes                 3       1        0.3333   0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  2       0        0.0      0      0.0      2
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  held at each time after it:
%(steps)s
    1     1            0              2           0             0           1           0
    2     1            0              2           0             0           1           0
    3     0            0              1           0             0           0           0
  entry test.cut, of the operation test.cut:
%(effect)s
    agent nodes                 2       0        0.0      0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  2       1        0.5      0      0.0      1
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  held at each time after it:
%(steps)s
    1     1            0              2           0             0           1           0
    2     1            0              2           0             0           0           0
    3     0            0              1           0             0           0           0
  entry test.cut, of the operation test.cut, removes every edge of the relation base at the times of the tasks of turn: 2; not at every time
  entry graphspec.relation_forms, of the operation graphspec.relation_forms:
%(effect)s
    agent nodes                 2       0        0.0      0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  1       0        0.0      0      0.0      1
    relation_forms.loops edges  0       0        nothing  7      nothing  7
  held at each time after it:
%(steps)s
    1     1            0              2           0             0           1           3
    2     1            0              2           0             0           0           3
    3     0            0              1           0             0           0           1
"""


SETTLED = {
    "graphs/join": "separate",
    "nodes/untyped": False,
    "relations/direction": "inverse",
    "time/steps": "changes",
    "time/several": "last",
}


@timed("descriptions.entered")
def entered(pipeline):
    return [
        dict(
            name=entry["operation"],
            operation=entry["operation"],
            version=OPERATIONS.named(entry["operation"]).version,
            graph=None,
            parameters=entry["parameters"],
            reads={},
            splits={},
            storage={},
        )
        for entry in pipeline
    ]


@timed("descriptions.entries")
def entries(cutter, cut):
    return entered([dict(entry, operation=cutter) if entry["operation"] == "test.cut" else entry for entry in pipelined(cut)])


@timed("descriptions.keyed")
def keyed(bank, config):
    return specs.digest(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT).identity(config))


@timed("descriptions.varied")
def varied(bank, config):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    key = keyed(bank, config)
    record = dict(
        version=describing.VERSION,
        variant=key,
        stored_sha256=stored,
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=SETTLED,
        entries=entries("test.cut", CUTTING),
        graphs=[VARIED, VARIED_SECOND],
    )
    written = VARIANT.replace("%(steps)s", STEPS).replace("%(effect)s", EFFECT) % (
        key,
        stored,
        describing.VERSION,
        effect.READING,
        OPERATIONS.named("graphspec.nodes").version,
        OPERATIONS.named("graphspec.relation_forms").version,
    )
    return record, written, bank / "descriptions" / "variants" / sha256_json(dict(variant=key, version=describing.VERSION))


def test_every_count_and_share_of_a_variant_equals_the_number_written_out_here_and_names_the_entry_that_cuts(bank, cutting):
    started = time.perf_counter()
    config = configured(bank, CUTTING)
    record = variant(bank, config, STATED)
    assert record == varied(bank, config)[0]
    assert json.loads(json.dumps(record)) == record
    cut = [found for graph in record["graphs"] for held in graph["effects"] for found in held["cuts"]]
    assert [held["entry"] for graph in record["graphs"] for held in graph["effects"] if held["cuts"]] == ["test.cut"] * 2
    assert cut == [
        dict(sort="relation", name="base", task="turn", times=[1, 3, 5], everywhere=False),
        dict(sort="relation", name="base", task="turn", times=[2], everywhere=False),
    ]
    LOG.info("descriptions variant numbers exit seconds=%.3f", time.perf_counter() - started)


def test_the_text_of_a_variant_holds_every_number_of_its_record_and_the_verb_prints_what_the_call_writes(
    bank, cutting, tmp_path, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    config = configured(bank, CUTTING)
    record = variant(bank, config, STATED)
    expected, written, stem = varied(bank, config)
    assert effect.text(record) == written
    assert read_json(stem.with_suffix(".json")) == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "statement.json", STATED)
    monkeypatch.setattr(
        sys, "argv", ["graphspec", "describe", str(bank), str(tmp_path / "config.json"), str(tmp_path / "statement.json")]
    )
    main()
    assert capsys.readouterr().out == written
    LOG.info("descriptions variant text exit seconds=%.3f", time.perf_counter() - started)


def test_a_second_call_reads_the_file_of_a_variant_and_a_statement_differing_in_one_parameter_is_not_served_it(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    config = configured(bank, CUTTING)
    record = variant(bank, config, STATED)
    assert varied(bank, config)[2].with_suffix(".json").exists()
    monkeypatch.setattr(varying, "recorded", refused)
    assert variant(bank, config, STATED) == record
    other = configured(bank, dict(CUTTING, **{"cut/kind": "visit"}))
    assert varied(bank, other)[2] != varied(bank, config)[2]
    with pytest.raises(AssertionError, match="the description was computed again"):
        variant(bank, other, STATED)
    LOG.info("descriptions variant served exit seconds=%.3f", time.perf_counter() - started)


JOINED = {
    "graphs/join": "single",
    "graphs/join_size": 2,
    "graphs/connect/agent": True,
    "graphs/connect/spot": True,
    "graphs/connect_fraction": 1.0,
}


def test_a_joined_group_is_described_graph_by_graph_as_its_graphs_derived_apart_are(bank, cutting):
    started = time.perf_counter()
    config = configuration(
        bank, MODEL, None, pipeline=pipelined(CUTTING), graph=dict(SETTINGS, **JOINED), seed=STATED["seed"], splits=SPLITS
    )
    record = variant(bank, config, STATED)
    assert record["settings"]["graphs/join"] == "single" and record["graphs"] == [VARIED, VARIED_SECOND]
    LOG.info("descriptions variant joined exit seconds=%.3f", time.perf_counter() - started)


GO = ("target to of go", "target along of go", "target made of go", "parameter speed of go")
THREE = ["changed.0.0", "o0", "o1"]
FOUR = ["changed.0.0", "o0", "o1", "o2"]
DECIDED = (
    ("g0", ((10, 1, "a0", THREE, "s0", True), (30, 3, "a1", FOUR, "s1", True), (50, 5, "a0", THREE, "s0", False))),
    ("g1", ((10, 2, "a0", THREE, "s0", True),)),
)
STEP = dict(relation="went", actor="source")
STEPS_OF_GO = {role: STEP for role in ("to", "along", "made")}
WENT_ADDED = dict(went=dict(pairs=(("agent", "spot"), ("agent", "agent")), directed=True, columns=("speed",)))
WALKED = Edited(dict(turn=dict(go=STEPS_OF_GO)), WENT_ADDED, 1)
STEPPED = Edited(dict(turn=dict(go=STEPS_OF_GO, stay=dict(by=STEP))), WENT_ADDED, 1)
LOAD = ["load"]
APART = dict(state="apart", hops=None, relations=None)
ABSENT = dict(state="absent", hops=None, relations=None)
ADJACENT = dict(state="connected", hops=1, relations=[["base"]])
WENT = dict(state="connected", hops=1, relations=[["went"]])
FARTHER = dict(state="connected", hops=2, relations=[["base"], ["near"]])
NOWHERE = dict(entry=None, operation=None, settings=[])
EDGE = "a1 base s1"
SEVERING = {"cut/relation": "base", "cut/kind": "visit"}
PARTING = {"cut/relation": "base", "cut/times": "all"}
REFUSING = "the entry %s of the pipeline states cut/kind turn, which " + UNDECIDED
REFUSAL = REFUSING % "test.sever"
SEVERED = (
    "apart; cut by the entry test.cut, of the operation test.cut; cut/relation base of the entry test.cut gives it "
    "back under near, not under none, refused under none; cut/kind turn of the entry test.cut gives it back under %s, "
    "not under %s, refused under none"
)
AIMED_TEXT = """    option o1
      target to of go: %(severed)s
        s0: apart, features load
        s1: apart, features load
      target along of go: absent; cut by the entry test.cut, of the operation test.cut; cut/relation base of the entry test.cut gives it back under none, not under near, refused under none; cut/kind turn of the entry test.cut gives it back under visit, not under none, refused under none
        a1 base s1: absent, features none
      target made of go: absent; cut by no entry: connected in no state, the view before the first entry included
        s2: absent, features none
      parameter speed of go: %(severed)s"""
APART_TEXT = """    option o1
      target to of go: apart; cut by no entry: connected in no state, the view before the first entry included
        s1: apart, features load
      parameter speed of go: apart; cut by no entry: connected in no state, the view before the first entry included"""
REFUSED_TEXT = (
    "      target to of go: apart; cut by the entry test.sever, of the operation test.sever; cut/relation base of the "
    "entry test.sever gives it back under near, not under none, refused under none; cut/kind visit of the entry "
    "test.sever gives it back under none, not under none, refused under turn (the entry test.sever of the pipeline "
    "states cut/kind turn, which decides, and the entry cuts only at the times of the tasks of kinds that decide "
    "nothing)\n"
)
PARTED_TEXT = (
    "      target to of go: apart; cut by the entry test.part, of the operation test.part; cut/relation base of the "
    "entry test.part gives it back under near, not under none, refused under none; cut/times all of the entry "
    "test.part gives it back under %s, not under %s, refused under tasks with cut/kind turn (the entry test.part of "
    "the pipeline states cut/kind turn, which decides, and the entry cuts only at the times of the tasks of kinds "
    "that decide nothing)\n"
)
UNTRIED_TEXT = (
    "cut by the entry test.part, of the operation test.part; cut/times all of the entry test.part gives it back under "
    "none, not under none, refused under tasks with cut/kind turn (the entry test.part of the pipeline states "
    "cut/kind turn, which decides, and the entry cuts only at the times of the tasks of kinds that decide nothing), "
    "not tried under tasks with cut/kind visit"
)
REACH = """reach of the options of the variant %s of the bank stored as %s, described at version %d
realisation of options vector at version %d, reading kind, targets, parameters; options read beside the variant
derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
%s
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree 1, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none
kind of task turn: fields target to of go, target along of go, target made of go, parameter speed of go; not read by the realisation: none
kinds of task that decide nothing, which hold no option: visit"""


@pytest.fixture
def walking():
    started = time.perf_counter()
    REALISATIONS.register("walked", WALKED)
    REALISATIONS.register("stepped", STEPPED)
    LOG.debug("descriptions walking registered seconds=%.4f", time.perf_counter() - started)
    yield WALKED
    del REALISATIONS.held["walked"]
    del REALISATIONS.held["stepped"]


@timed("descriptions.realising")
def realising(bank, cut, name):
    model = dict(MODEL, **{"options/realisation": name})
    return configuration(bank, model, None, pipeline=pipelined(cut), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.role")
def role(held, cut, *targets):
    return dict(held, targets=[dict(found, target=target, features=features) for target, found, features in targets], cut=cut)


@timed("descriptions.under")
def under(value, *stating):
    return dict(value=value, stating=[dict(entry=entry, parameter=parameter, value=held) for entry, parameter, held in stating])


@timed("descriptions.setting")
def setting(entry, parameter, value, back, still, refused):
    return dict(
        entry=entry,
        parameter=parameter,
        value=value,
        back=[held if type(held) is dict else under(held) for held in back],
        still=[held if type(held) is dict else under(held) for held in still],
        refused=[held if type(held) is dict else dict(under(held[0]), reason=held[1]) for held in refused],
        untried=[],
    )


@timed("descriptions.severing")
def severing(entry, relation, kind):
    settings = [setting(entry, "cut/relation", *relation), setting(entry, kind[0], *kind[1:])]
    return dict(entry=entry, operation=entry, settings=settings)


@timed("descriptions.aimed")
def aimed(held, cut, target):
    return {GO[0]: role(held, cut, (target, held, LOAD)), GO[3]: dict(held, cut=cut)}


@timed("descriptions.cut_at")
def cut_at(given, back, still):
    kind = ("cut/kind", "turn", ["visit"], [], []) if given else ("cut/kind", "turn", [], ["visit"], [])
    return severing("test.cut", ("base", back, still, []), kind)


@timed("descriptions.parted_at")
def parted_at(given, back, still):
    visit = under("tasks", ("test.part", "cut/kind", "visit"))
    turn = dict(under("tasks", ("test.part", "cut/kind", "turn")), reason=REFUSING % "test.part")
    times = ("cut/times", "all", [visit], [], [turn]) if given else ("cut/times", "all", [], [visit], [turn])
    return severing("test.part", ("base", back, still, []), times)


@timed("descriptions.unreached")
def unreached(target):
    return {GO[0]: role(APART, NOWHERE, (target, APART, LOAD)), GO[3]: dict(APART, cut=NOWHERE)}


@timed("descriptions.lost")
def lost(cut, graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        return {
            GO[0]: role(APART, cut(given, ["near"], []), ("s0", APART, LOAD), ("s1", APART, LOAD)),
            GO[1]: role(ABSENT, cut(True, [], ["near"]), (EDGE, ABSENT, [])),
            GO[2]: role(ABSENT, NOWHERE, ("s2", ABSENT, [])),
            GO[3]: dict(APART, cut=cut(given, ["near"], [])),
        }
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    return aimed(APART, cut(given, ["near"], []), target)


@timed("descriptions.severed")
def severed(graph, task, key, target, given):
    return lost(cut_at, graph, task, key, target, given)


@timed("descriptions.parted")
def parted(graph, task, key, target, given):
    return lost(parted_at, graph, task, key, target, given)


@timed("descriptions.given_back")
def given_back(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        along = severing("test.cut", ("near", [], ["base"], []), ("cut/kind", "turn", ["visit"], [], []))
        return {
            GO[0]: role(ADJACENT, None, ("s0", ADJACENT, LOAD), ("s1", APART, LOAD)),
            GO[1]: role(APART, along, (EDGE, APART, LOAD)),
            GO[2]: role(ABSENT, NOWHERE, ("s2", ABSENT, [])),
            GO[3]: dict(ADJACENT, cut=None),
        }
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    return aimed(ADJACENT, None, target)


@timed("descriptions.walked")
def walked(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        return {
            GO[0]: role(WENT, None, ("s0", WENT, LOAD), ("s1", WENT, LOAD)),
            GO[1]: role(WENT, None, (EDGE, WENT, LOAD)),
            GO[2]: role(WENT, None, ("s2", WENT, [])),
            GO[3]: dict(WENT, cut=None),
        }
    if (graph, task, key) == ("g1", 10, "o1"):
        return aimed(WENT, None, "s1")
    return aimed(WENT, None, target)


@timed("descriptions.refusing")
def refusing(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        return {
            GO[0]: role(ADJACENT, None, ("s0", ADJACENT, LOAD), ("s1", FARTHER, LOAD)),
            GO[1]: role(FARTHER, None, (EDGE, FARTHER, LOAD)),
            GO[2]: role(ABSENT, NOWHERE, ("s2", ABSENT, [])),
            GO[3]: dict(ADJACENT, cut=None),
        }
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    if (graph, task) == ("g0", 50):
        refused = ("cut/kind", "visit", [], [], [["turn", REFUSAL]])
        return aimed(APART, severing("test.sever", ("base", ["near"], [], []), refused), target)
    return aimed(ADJACENT, None, target)


@timed("descriptions.reached")
def reached(fields):
    return [
        dict(
            name=graph,
            decisions=[
                dict(
                    task=task,
                    kind="turn",
                    time=moment,
                    references=dict(agents=[actor]),
                    options=[dict(key=key, fields=fields(graph, task, key, target, given)) for key in options],
                )
                for task, moment, actor, options, target, given in decisions
            ],
        )
        for graph, decisions in DECIDED
    ]


@timed("descriptions.expected_reach")
def expected_reach(bank, config, name, fields):
    key, entry = keyed(bank, config), REALISATIONS.named(name)
    stem = sha256_json(dict(variant=key, realisation=dict(name=name, version=entry.version), version=describing.VERSION))
    cutter = config["pipeline"][1]
    record = dict(
        version=describing.VERSION,
        variant=key,
        stored_sha256=read_json(bank / "manifest.json")["stored_sha256"],
        realisation=dict(name=name, version=entry.version, reads=["kind", "targets", "parameters"], back=entry.back),
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=SETTLED,
        entries=entries(cutter["operation"], cutter["parameters"]),
        tasks=dict(turn=dict(fields=list(GO), unread=[])),
        undecided=["visit"],
        graphs=reached(fields),
    )
    return record, bank / "descriptions" / "reach" / stem


@timed("descriptions.spelled")
def spelled(record):
    lines = [
        REACH
        % (
            record["variant"],
            record["stored_sha256"],
            describing.VERSION,
            REALISATIONS.named("vector").version,
            reaching.READING,
            OPERATIONS.named("graphspec.nodes").version,
            OPERATIONS.named("graphspec.relation_forms").version,
        )
    ]
    for graph, decisions in DECIDED:
        lines += ["", "graph %s, decisions %d" % (graph, len(decisions))]
        for task, moment, actor, options, target, given in decisions:
            cut = SEVERED % (("visit", "none") if given else ("none", "visit"))
            lines.append("  decision %d of turn at time %d, references agents %s" % (task, moment, actor))
            for key in options:
                if (graph, task, key) == ("g0", 10, "o1"):
                    lines.append(AIMED_TEXT % dict(severed=cut))
                elif (graph, task, key) == ("g1", 10, "o1"):
                    lines.append(APART_TEXT)
                else:
                    lines += ["    option " + key, "      %s: %s" % (GO[0], cut), "        %s: apart, features load" % target]
                    lines.append("      %s: %s" % (GO[3], cut))
    return "\n".join(lines) + "\n"


@timed("descriptions.cutter")
def cutter(bank, operation, cut):
    pipeline = [dict(entry, operation=operation) if entry["operation"] == "test.cut" else entry for entry in pipelined(cut)]
    model = dict(MODEL, **{"options/realisation": "vector"})
    return configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


def test_the_reach_of_every_option_names_the_entry_its_operation_and_the_setting_that_cut_its_edges_at_its_decision(bank, cutting):
    started = time.perf_counter()
    config = realising(bank, CUTTING, "vector")
    record = reach(bank, config, STATED)
    assert record == expected_reach(bank, config, "vector", severed)[0]
    assert json.loads(json.dumps(record)) == record
    LOG.info("descriptions reach cut exit seconds=%.3f", time.perf_counter() - started)


def test_with_the_setting_changed_the_reach_lists_every_field_of_every_option_at_one_hop_by_the_relation_joining_it(bank, cutting):
    started = time.perf_counter()
    config = realising(bank, dict(CUTTING, **{"cut/relation": "near"}), "vector")
    record = reach(bank, config, STATED)
    assert record == expected_reach(bank, config, "vector", given_back)[0]
    LOG.info("descriptions reach given back exit seconds=%.3f", time.perf_counter() - started)


def test_the_text_of_reach_holds_its_record_and_the_verb_prints_what_the_call_writes(bank, cutting, tmp_path, capsys, monkeypatch):
    from tuning.__main__ import main

    started = time.perf_counter()
    config = realising(bank, CUTTING, "vector")
    record = reach(bank, config, STATED)
    expected, stem = expected_reach(bank, config, "vector", severed)
    written = spelled(record)
    assert reaching.text(record) == written
    assert read_json(stem.with_suffix(".json")) == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "statement.json", STATED)
    arguments = [str(bank), str(tmp_path / "config.json"), str(tmp_path / "statement.json")]
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--reach"] + arguments)
    main()
    assert capsys.readouterr().out == written
    LOG.info("descriptions reach text exit seconds=%.3f", time.perf_counter() - started)


def test_a_second_call_reads_the_file_of_reach_and_another_realisation_or_setting_is_not_served_it(bank, cutting, monkeypatch):
    started = time.perf_counter()
    config = realising(bank, CUTTING, "vector")
    record = reach(bank, config, STATED)
    assert expected_reach(bank, config, "vector", severed)[1].with_suffix(".json").exists()
    monkeypatch.setattr(reaching, "recorded", refused)
    assert reach(bank, config, STATED) == record
    model = dict(MODEL, **{"options/realisation": "actor_edge", "options/depth": 1})
    structured = configuration(bank, model, None, pipeline=pipelined(CUTTING), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    for other in (structured, realising(bank, dict(CUTTING, **{"cut/kind": "visit"}), "vector")):
        with pytest.raises(AssertionError, match="the description was computed again"):
            reach(bank, other, STATED)
    with pytest.raises(ValueError, match="states options/realisation none, which puts no option before it"):
        reach(bank, configured(bank, CUTTING), STATED)
    LOG.info("descriptions reach served exit seconds=%.3f", time.perf_counter() - started)


def test_under_an_edit_the_reach_of_each_option_is_read_in_its_own_fork_where_its_facts_join_the_actor_to_its_target(
    bank, cutting, walking
):
    started = time.perf_counter()
    config = realising(bank, CUTTING, "walked")
    record = reach(bank, config, STATED)
    assert record == expected_reach(bank, config, "walked", walked)[0]
    LOG.info("descriptions reach edit exit seconds=%.3f", time.perf_counter() - started)


def test_the_reach_of_a_joined_group_is_described_graph_by_graph_as_its_graphs_derived_apart_are(bank, cutting):
    started = time.perf_counter()
    model = dict(MODEL, **{"options/realisation": "vector"})
    graph = dict(SETTINGS, **JOINED)
    config = configuration(bank, model, None, pipeline=pipelined(CUTTING), graph=graph, seed=STATED["seed"], splits=SPLITS)
    record = reach(bank, config, STATED)
    assert record["settings"]["graphs/join"] == "single" and record["graphs"] == reached(severed)
    LOG.info("descriptions reach joined exit seconds=%.3f", time.perf_counter() - started)


def test_a_value_the_statement_refuses_is_recorded_with_its_refusal_and_the_reach_is_written_with_edges_at_two_hops(
    bank, cutting
):
    started = time.perf_counter()
    config = cutter(bank, "test.sever", SEVERING)
    record = reach(bank, config, STATED)
    expected, stem = expected_reach(bank, config, "vector", refusing)
    assert record == expected
    assert read_json(stem.with_suffix(".json")) == record
    written = stem.with_suffix(".txt").read_text(encoding="utf-8")
    assert written == reaching.text(record) and written.count(REFUSED_TEXT) == 3
    LOG.info("descriptions reach refused exit seconds=%.3f", time.perf_counter() - started)


def test_a_value_that_makes_a_parameter_active_is_tried_with_each_value_of_it_and_refused_only_where_the_whole_statement_is(
    bank, cutting
):
    started = time.perf_counter()
    config = cutter(bank, "test.part", PARTING)
    record = reach(bank, config, STATED)
    expected, stem = expected_reach(bank, config, "vector", parted)
    assert record == expected
    assert json.loads(json.dumps(record)) == record
    written = stem.with_suffix(".txt").read_text(encoding="utf-8")
    assert written == reaching.text(record)
    assert written.count(PARTED_TEXT % ("tasks with cut/kind visit", "none")) == 9
    assert written.count(PARTED_TEXT % ("none", "tasks with cut/kind visit")) == 3
    LOG.info("descriptions reach made active exit seconds=%.3f", time.perf_counter() - started)


def test_past_the_bound_of_statements_tried_for_a_value_the_rest_are_recorded_as_not_tried_and_never_as_refused(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    held = reaching.Reached(deriver, cutter(bank, "test.part", PARTING), "vector")
    monkeypatch.setattr(reaching, "BOUND", 1)
    (decision,) = [found for found in held.decided(held.derived(None, 3, 0), 0) if found["key"] == 50]
    option = int(held.found.first[decision["decision"]])
    times = [(holder, parameter) for holder, parameter in held.settable(1) if parameter["name"] == "cut/times"]
    found = [held.tried(decision, option, GO[0], holder, parameter) for holder, parameter in times]
    turn = dict(under("tasks", ("test.part", "cut/kind", "turn")), reason=REFUSING % "test.part")
    untried = [under("tasks", ("test.part", "cut/kind", "visit"))]
    assert found == [dict(setting("test.part", "cut/times", "all", [], [], [turn]), untried=untried)]
    assert reaching.cut_text(dict(entry="test.part", operation="test.part", settings=found)) == UNTRIED_TEXT
    LOG.info("descriptions reach bound exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.reading")
def reading(bank, cut, read):
    pipeline = [
        dict(operation="graphspec.nodes", parameters=NODES),
        dict(operation="test.part", parameters=cut),
        dict(operation="test.read", parameters=read),
    ]
    model = dict(MODEL, **{"options/realisation": "vector"})
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "vector")
    (times,) = [parameter for _, parameter in held.settable(1) if parameter["name"] == "cut/times"]
    return held, times


def test_a_value_is_stated_with_what_it_makes_active_in_an_entry_that_reads_it_and_without_what_it_makes_inactive_there(
    bank, cutting
):
    started = time.perf_counter()
    held, times = reading(bank, PARTING, {})
    tried, untried = held.restating(held.steps[1], times, "tasks")
    visit = [("test.part", "cut/kind", "visit")]
    assert untried == [] and [(added, reason) for added, _, reason in tried] == [
        (under("tasks", ("test.part", "cut/kind", "turn"))["stating"], REFUSING % "test.part"),
        (under("tasks", *visit, ("test.read", "read/level", 1))["stating"], None),
        (under("tasks", *visit, ("test.read", "read/level", 3))["stating"], None),
    ]
    assert [held.variants[key][0]["pipeline"][1:] for _, key, _ in tried[1:]] == [
        [
            dict(operation="test.part", parameters={"cut/relation": "base", "cut/times": "tasks", "cut/kind": "visit"}),
            dict(operation="test.read", parameters={"read/level": level}),
        ]
        for level in (1, 3)
    ]
    stated = dict(entry="test.part", back=[under("tasks", *visit, ("test.read", "read/level", 3))])
    assert reaching.trials(stated, "back") == "tasks with cut/kind visit and read/level 3 of the entry test.read"
    held, times = reading(bank, {"cut/relation": "base", "cut/times": "tasks", "cut/kind": "visit"}, {"read/level": 2})
    ((added, key, reason),), untried = held.restating(held.steps[1], times, "all")
    assert (added, reason, untried) == ([], None, []) and held.variants[key][0]["pipeline"][1:] == [
        dict(operation="test.part", parameters=PARTING),
        dict(operation="test.read", parameters={}),
    ]
    LOG.info("descriptions reach across entries exit seconds=%.3f", time.perf_counter() - started)


def test_reach_stops_naming_a_node_no_graph_of_the_view_holds_and_a_decision_whose_task_its_graph_does_not_hold(
    bank, cutting
):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    held = reaching.Reached(deriver, realising(bank, CUTTING, "vector"), "vector")
    view = held.loaded(0)[1][0]
    where = "the reference agents of the decision 10 of the kind turn of graph g0"
    assert [held.keyed(token, where) for token in reaching.masked(view.entity_token).tolist()] == list(view.keys)
    with pytest.raises(ValueError, match="^%s names the node of the token 7, which no graph of the view holds; " % where):
        held.keyed(7, where)
    with pytest.raises(
        ValueError,
        match="^reach reads the references of the decision 10 of the kind turn of graph g0 from its task at the step 0, "
        "and the graph derived for it holds 0 tasks of that key and kind there$",
    ):
        held.referred(view, dict(step=0, key=10, kind=0, graph="g0"))
    LOG.info("descriptions reach unheld exit seconds=%.3f", time.perf_counter() - started)


FIELDS = ("kind", "target to of go", "target along of go", "target made of go", "target by of stay", "parameter speed of go")
OVER = [["a0", "base", "s0"]]
REFERRED = dict(agents=["a0"], over=["a0 base s0"])
BOTH = "s0 near s1"
HERE = dict(state="connected", hops=0, relations=[])
NEARBY = dict(state="connected", hops=1, relations=[["near"]])
EITHER = dict(state="connected", hops=1, relations=[["near", "went"]])
LONE = {name: value for name, value in SETTINGS.items() if name != "time/several"}
CROSSED = {name: value for name, value in SETTLED.items() if name != "time/several"}
KEPT = {
    "nodes/agent/keep": True,
    "nodes/spot/keep": True,
    "nodes/referenced": "keep",
    "nodes/agent/filter": "none",
    "nodes/spot/filter": "none",
}
THINNED = dict(KEPT, **{"nodes/spot/filter": "degree", "nodes/spot/min_degree": 2})
THINNER = "graphspec.nodes"
CROSSED_TEXT = """kind of task turn: fields kind, target to of go, target along of go, target made of go, target by of stay, parameter speed of go; not read by the realisation: none
kinds of task that decide nothing, which hold no option: none

graph h0, decisions 3
  decision 10 of turn at time 0, references agents a0; over a0 base s0
    option o0
      kind: apart; cut by no entry: connected in no state, the view before the first entry included
      target to of go: apart; cut by no entry: connected in no state, the view before the first entry included
        s2: apart, features load
      parameter speed of go: apart; cut by no entry: connected in no state, the view before the first entry included
    option o1
      kind: connected at hop 1 by near
      target by of stay: connected at hop 1 by near
        s1: connected at hop 1 by near, features load
  decision 20 of turn at time 1, references agents a0; over a0 base s0
    option o0
      kind: connected at hop 0 by nothing
      target to of go: connected at hop 0 by nothing
        s0: connected at hop 0 by nothing, features load
      target along of go: connected at hop 0 by nothing
        a0 base s0: connected at hop 0 by nothing, features load
        s0 near s1: connected at hop 0 by nothing, features load
      parameter speed of go: connected at hop 0 by nothing
    option o1
      kind: connected at hop 0 by nothing
      target to of go: connected at hop 1 by near
        s1: connected at hop 1 by near, features load
      target along of go: connected at hop 0 by nothing
        s0 base a0: connected at hop 0 by nothing, features load
        s0 near s1: connected at hop 0 by nothing, features load
      parameter speed of go: connected at hop 0 by nothing
  decision 30 of turn at time 2, references agents a0; over a0 base s0
    option o0
      kind: connected at hop 0 by nothing
      target by of stay: connected at hop 0 by nothing
        s0: connected at hop 0 by nothing, features load
"""
THINNED_TEXT = (
    "      target to of go: absent; cut by the entry graphspec.nodes, of the operation graphspec.nodes; "
    "nodes/agent/keep True of the entry graphspec.nodes gives it back under none, not under False, refused under none; "
    "nodes/spot/keep True of the entry graphspec.nodes gives it back under none, not under False, refused under none; "
    "nodes/referenced keep of the entry graphspec.nodes gives it back under none, not under none, refused under none; "
    "nodes/agent/filter none of the entry graphspec.nodes gives it back under none, not under "
    "degree with nodes/agent/min_degree 1, degree with nodes/agent/min_degree 16, "
    "total_degree with nodes/agent/min_degree 1, total_degree with nodes/agent/min_degree 16, "
    "core with nodes/agent/core 2, core with nodes/agent/core 16, "
    "activity with nodes/agent/min_activity 2, activity with nodes/agent/min_activity 16, "
    "sample with nodes/agent/sample 0.1, sample with nodes/agent/sample 0.9, refused under none; "
    "nodes/spot/filter degree of the entry graphspec.nodes gives it back under none, sample with nodes/spot/sample 0.9, "
    "not under total_degree, core with nodes/spot/core 2, core with nodes/spot/core 16, "
    "activity with nodes/spot/min_activity 2, activity with nodes/spot/min_activity 16, "
    "sample with nodes/spot/sample 0.1, refused under none; "
    "nodes/spot/min_degree 2 of the entry graphspec.nodes gives it back under 1, not under 16, refused under none\n"
    "        s2: absent, features none\n"
)


@timed("descriptions.crossing")
def crossing():
    held = schema()
    turn = held["tasks"]["kinds"]["turn"]["target"]
    roles = dict(turn["actions"]["go"]["roles"], along=dict(sort="edge", kinds=["agent", "spot"], count=[0, None]))
    stay = dict(roles=dict(by=dict(sort="node", kinds=["spot"], count=[1, 1])), parameters=[])
    decided = dict(turn, actions=dict(go=dict(turn["actions"]["go"], roles=roles), stay=stay), generators=[])
    references = dict(agents=dict(sort="nodes", kinds=["agent"]), over=dict(sort="edge", kinds=["agent"]))
    kinds = dict(turn=synthetic.kind(["agents", "over"], [], decided, None))
    edges = dict(held["edges"], base=synthetic.edge("agent", "spot", [], directed=False))
    return dict(held, edges=edges, tasks=dict(held["tasks"], references=references, kinds=kinds))


@timed("descriptions.acted")
def acted(key, kind, targets, parameters):
    return synthetic.offered(key, [synthetic.acting(kind, "a0", targets, parameters)], "set", "recorded")


def over(stated):
    started = time.perf_counter()
    result = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], references=dict(agents=[], over=[]))
    for key, moment, split, gain, options in stated:
        options[0].update(chosen=True, outcome=dict(gain=gain))
        target = dict(options=options, concurrent=[], next={}, ended=None, discount=None)
        held = dict(key=key, kind="turn", time=moment, split=split, weight=None, target=target, known=moment + 1)
        for name, value in held.items():
            result[name].append(value)
        result["references"]["agents"].append(["a0"])
        result["references"]["over"].append(OVER)
    LOG.debug("descriptions over exit tasks=%d seconds=%.4f", len(stated), time.perf_counter() - started)
    return result


@timed("descriptions.crossed_case")
def crossed_case():
    tasks = over(
        [
            (10, 0, "train", 1.0, [acted("o0", "go", dict(UNAIMED, to=["s2"]), dict(speed=3.0)), acted("o1", "stay", dict(by=["s1"]), {})]),
            (
                20,
                1,
                "train",
                2.0,
                [
                    acted("o0", "go", dict(to=["s0"], along=[["s0", "near", "s1"], ["a0", "base", "s0"]], made=[]), dict(speed=1.0)),
                    acted("o1", "go", dict(to=["s1"], along=[["s1", "near", "s0"], ["s0", "base", "a0"]], made=[]), dict(speed=2.0)),
                ],
            ),
            (30, 2, "validation", 3.0, [acted("o0", "stay", dict(by=["s0"]), {})]),
        ]
    )
    held = graph(
        "h0",
        dict(agent=["a0"], spot=["s0", "s1", "s2"]),
        [("a0", 0, None, 1.0)],
        [("s0", 0, None, 1.0), ("s1", 0, None, 2.0), ("s2", 0, None, 2.0)],
        [
            ("near", ("spot", "spot"), NEAR, [("s1", "s0", True, 0, None, 1.0)]),
            ("base", ("agent", "spot"), BASE, [("a0", "s0", True, 0, None)]),
        ],
        [],
    )
    return dict(schema=crossing(), graphs=[dict(held, tasks=tasks)])


@timed("descriptions.crossed_made")
def crossed_made(root):
    build(synthetic.stage(root, "crossed", crossed_case()), root / "bank", 1, ("train",))


@pytest.fixture(scope="module")
def crossed():
    started = time.perf_counter()
    result = shared("crossed", crossed_made) / "bank"
    LOG.debug("descriptions crossed exit seconds=%.4f", time.perf_counter() - started)
    return result


@timed("descriptions.noded")
def noded(bank, parameters, name):
    model = dict(MODEL, **{"options/realisation": name})
    pipeline = [dict(operation="graphspec.nodes", parameters=parameters)]
    return configuration(bank, model, None, pipeline=pipeline, graph=LONE, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.gone")
def gone(held, cut, features):
    return {FIELDS[0]: dict(held, cut=cut), FIELDS[1]: role(held, cut, ("s2", held, features)), FIELDS[5]: dict(held, cut=cut)}


@timed("descriptions.along")
def along(aim, target, based):
    return {
        FIELDS[0]: dict(HERE, cut=None),
        FIELDS[1]: role(aim, None, (target, aim, LOAD)),
        FIELDS[2]: role(HERE, None, (based, HERE, LOAD), (BOTH, HERE, LOAD)),
        FIELDS[5]: dict(HERE, cut=None),
    }


@timed("descriptions.stayed")
def stayed(held, target):
    return {FIELDS[0]: dict(held, cut=None), FIELDS[4]: role(held, None, (target, held, LOAD))}


@timed("descriptions.crossed_reach")
def crossed_reach(bank, config, name, lone, near):
    entry = REALISATIONS.named(name)
    decisions = [
        (10, 0, [("o0", lone), ("o1", stayed(near, "s1"))]),
        (20, 1, [("o0", along(HERE, "s0", "a0 base s0")), ("o1", along(near, "s1", "s0 base a0"))]),
        (30, 2, [("o0", stayed(HERE, "s0"))]),
    ]
    return dict(
        version=describing.VERSION,
        variant=keyed(bank, config),
        stored_sha256=read_json(bank / "manifest.json")["stored_sha256"],
        realisation=dict(name=name, version=entry.version, reads=["kind", "targets", "parameters"], back=entry.back),
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=CROSSED,
        entries=entered(config["pipeline"]),
        tasks=dict(turn=dict(fields=list(FIELDS), unread=[])),
        undecided=[],
        graphs=[
            dict(
                name="h0",
                decisions=[
                    dict(
                        task=task,
                        kind="turn",
                        time=moment,
                        references=REFERRED,
                        options=[dict(key=option, fields=fields) for option, fields in options],
                    )
                    for task, moment, options in decisions
                ],
            )
        ],
    )


@timed("descriptions.each")
def each(value, parameter, *ends):
    return [under(value, (THINNER, parameter, end)) for end in ends]


@timed("descriptions.thinning")
def thinning():
    agent = (
        each("degree", "nodes/agent/min_degree", 1, 16)
        + each("total_degree", "nodes/agent/min_degree", 1, 16)
        + each("core", "nodes/agent/core", 2, 16)
        + each("activity", "nodes/agent/min_activity", 2, 16)
        + each("sample", "nodes/agent/sample", 0.1, 0.9)
    )
    kept = [under("none")] + each("sample", "nodes/spot/sample", 0.9)
    lost = (
        [under("total_degree")]
        + each("core", "nodes/spot/core", 2, 16)
        + each("activity", "nodes/spot/min_activity", 2, 16)
        + each("sample", "nodes/spot/sample", 0.1)
    )
    settings = [
        setting(THINNER, "nodes/agent/keep", True, [], [False], []),
        setting(THINNER, "nodes/spot/keep", True, [], [False], []),
        setting(THINNER, "nodes/referenced", "keep", [], [], []),
        setting(THINNER, "nodes/agent/filter", "none", [], agent, []),
        setting(THINNER, "nodes/spot/filter", "degree", kept, lost, []),
        setting(THINNER, "nodes/spot/min_degree", 2, [1], [16], []),
    ]
    return dict(entry=THINNER, operation=THINNER, settings=settings)


@timed("descriptions.halved")
def halved(seed, salt, tokens, steps):
    return np.full(len(tokens), 0.5)


def test_reach_reads_an_edge_without_direction_stated_either_way_a_reference_that_is_an_edge_a_target_at_no_hop_and_the_kind(
    crossed,
):
    started = time.perf_counter()
    config = noded(crossed, KEPT, "vector")
    record = reach(crossed, config, STATED)
    assert record == crossed_reach(crossed, config, "vector", gone(APART, NOWHERE, LOAD), NEARBY)
    assert json.loads(json.dumps(record)) == record
    written = reaching.text(record)
    assert written[written.index("kind of task turn") :] == CROSSED_TEXT
    first, second = (option["fields"][FIELDS[2]]["targets"] for option in record["graphs"][0]["decisions"][1]["options"])
    assert first[1] == second[1] and dict(first[0], target="s0 base a0") == second[0]
    LOG.info("descriptions reach crossed exit seconds=%.3f", time.perf_counter() - started)


def test_under_an_edit_the_entry_that_removes_the_node_an_option_targets_in_its_own_fork_is_named_with_what_gives_it_back(
    crossed, walking, monkeypatch
):
    started = time.perf_counter()
    monkeypatch.setattr(filtering, "unit", halved)
    config = noded(crossed, THINNED, "stepped")
    record = reach(crossed, config, STATED)
    assert record == crossed_reach(crossed, config, "stepped", gone(ABSENT, thinning(), []), EITHER)
    assert THINNED_TEXT in reaching.text(record)
    LOG.info("descriptions reach edit cut exit seconds=%.3f", time.perf_counter() - started)
