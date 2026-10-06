import json
import logging
import shutil
import sys
import time
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import SPLITS, shared
from graphspec import OPERATIONS, Operation, Phase, Registry, describe
from graphspec import differences as between
from graphspec.descriptions import describe as describing
from graphspec.descriptions import differences as differing
from graphspec.descriptions import effect
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.timing import timed
from graphspec.operations import nodes as filtering
from graphspec.operations import parameters as specs
from graphspec.operations.operation import Need
from graphspec.operations.parameters import choice, equals, number
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.schema.time import ALWAYS as EARLIEST
from graphspec.schema.time import NEVER as LATEST
from graphspec.sources import synthetic
from tuning import REALISATIONS, Edited, configuration, reach, reach_over, reach_search
from tuning import describe as variant
from tuning import differences as against
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.descriptions import differences as diverging
from tuning.descriptions import reach as reaching
from tuning.descriptions import reaches
from tuning.descriptions import variant as varying
from tuning.families import heads
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
CONTRACTING = "graphspec.contract"
CONTRACTED = {"contract/grouping": "spot:load", "contract/aggregate": "mean", "contract/parallel": "keep"}
GROUPED = "contract in place of a0"
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


@timed("descriptions.lined")
def lined(bank, operation, parameters, name):
    model = dict(MODEL, **{"options/realisation": name})
    pipeline = [dict(operation=operation, parameters=parameters)]
    return configuration(bank, model, None, pipeline=pipeline, graph=LONE, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.gone")
def gone(held, cut, features):
    return {FIELDS[0]: dict(held, cut=cut), FIELDS[1]: role(held, cut, ("s2", held, features)), FIELDS[5]: dict(held, cut=cut)}


@timed("descriptions.along")
def along(aim, target, based, edge, features):
    return {
        FIELDS[0]: dict(HERE, cut=None),
        FIELDS[1]: role(aim, None, (target, aim, LOAD)),
        FIELDS[2]: role(HERE, None, (based, edge, features), (BOTH, HERE, LOAD)),
        FIELDS[5]: dict(HERE, cut=None),
    }


@timed("descriptions.stayed")
def stayed(held, target):
    return {FIELDS[0]: dict(held, cut=None), FIELDS[4]: role(held, None, (target, held, LOAD))}


@timed("descriptions.held_whole")
def held_whole(lone, near, edge, features):
    return [
        lone,
        stayed(near, "s1"),
        along(HERE, "s0", "a0 base s0", edge, features),
        along(near, "s1", "s0 base a0", edge, features),
        stayed(HERE, "s0"),
    ]


@timed("descriptions.crossed_reach")
def crossed_reach(bank, config, name, references, fields):
    entry = REALISATIONS.named(name)
    decisions = [
        (10, 0, [("o0", fields[0]), ("o1", fields[1])]),
        (20, 1, [("o0", fields[2]), ("o1", fields[3])]),
        (30, 2, [("o0", fields[4])]),
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
                        references=references,
                        options=[dict(key=option, fields=found) for option, found in options],
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
    config = lined(crossed, THINNER, KEPT, "vector")
    record = reach(crossed, config, STATED)
    fields = held_whole(gone(APART, NOWHERE, LOAD), NEARBY, HERE, LOAD)
    assert record == crossed_reach(crossed, config, "vector", REFERRED, fields)
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
    config = lined(crossed, THINNER, THINNED, "stepped")
    record = reach(crossed, config, STATED)
    fields = held_whole(gone(ABSENT, thinning(), []), EITHER, HERE, LOAD)
    assert record == crossed_reach(crossed, config, "stepped", REFERRED, fields)
    assert THINNED_TEXT in reaching.text(record)
    LOG.info("descriptions reach edit cut exit seconds=%.3f", time.perf_counter() - started)


def test_a_reference_the_pipeline_puts_in_a_node_it_adds_is_named_by_the_kind_of_that_node_and_the_node_in_whose_place_it_stands(
    crossed,
):
    started = time.perf_counter()
    config = lined(crossed, CONTRACTING, dict(CONTRACTED, **{"contract/grouping": "agent:load"}), "vector")
    record = reach(crossed, config, STATED)
    grouped = dict(agents=[GROUPED], over=[GROUPED + " base s0"])
    fields = held_whole(gone(APART, NOWHERE, LOAD), NEARBY, ABSENT, [])
    assert record == crossed_reach(crossed, config, "vector", grouped, fields)
    assert "  decision 10 of turn at time 0, references agents %s; over %s base s0\n" % (GROUPED, GROUPED) in reaching.text(record)
    config = lined(crossed, CONTRACTING, CONTRACTED, "vector")
    held = reaching.Reached(Deriver(crossed, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "vector")
    trunk = held.derived(None, 1, 0)
    found = held.decided(trunk, 0)
    assert [held.references(trunk, decision) for decision in found] == [dict(agents=["a0"], over=["a0 base contract in place of s0"])] * 3
    decision = found[0]
    edge = [member for member in held.referred(trunk, decision) if member[2] >= 0]
    where = "the reference over of the decision 10 of the kind turn of graph h0"
    assert held.ended(trunk, decision, edge, 0, 1, where) == "a0"
    with pytest.raises(
        ValueError,
        match="^%s names a node of the kind contract, which the pipeline adds, and the references of its task hold 1 "
        "members after the pipeline and 2 in the view, or the same in another order, so reach cannot say in place of "
        "which node of the view that node stands$" % where,
    ):
        held.ended(trunk, decision, edge, 0, 2, where)
    LOG.info("descriptions reach contracted exit seconds=%.3f", time.perf_counter() - started)


NEAR_TURN = {"cut/relation": "near", "cut/kind": "turn"}
NEAR_VISIT = {"cut/relation": "near", "cut/kind": "visit"}
PART_NEAR = {"cut/relation": "near", "cut/times": "all"}
PART_VISIT = {"cut/relation": "base", "cut/times": "tasks", "cut/kind": "visit"}
STARVED = dict(NODES, **{"nodes/agent/min_degree": 16})
HALTED = "the entry test.halt derives no graph"
UNSTARTED = "the trial holds no statement.json"
UNREALISED = (
    "the model of the configuration states options/realisation none, which puts no option before it; reach describes "
    "the options of decisions as a realisation puts them before a model"
)
SEEDED = "the seed of the configuration is 6 and the seed of the statement is 5"


@timed("descriptions.halted")
def halted(graph, registry, settings, seed, limits, carry, held):
    raise ValueError(HALTED)


@timed("descriptions.unstated")
def unstated(registry):
    return ()


HALT = Operation("test", "halt", (Phase("halt", halted, None, None),), unstated, (), (), (), ("facts",), None, None, 1, 1)


class Aimed(heads.Built):
    version = 1
    reads = ("targets",)


@pytest.fixture
def searching(cutting, walking):
    started = time.perf_counter()
    OPERATIONS.register(HALT.name, HALT)
    REALISATIONS.register("aimed", Aimed())
    LOG.debug("descriptions searching registered seconds=%.4f", time.perf_counter() - started)
    yield HALT
    del OPERATIONS.held[HALT.name]
    del REALISATIONS.held["aimed"]


@timed("descriptions.entry")
def entry(operation, parameters):
    return dict(operation=operation, parameters=parameters)


@timed("descriptions.modelled")
def modelled(bank, name):
    depth = {"options/depth": 1} if name == "actor_edge" else {}
    model = dict(MODEL, **{"options/realisation": name}, **depth)
    return configuration(bank, model, None, pipeline=pipelined(CUTTING), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


def tried(bank):
    started = time.perf_counter()
    models = {name: modelled(bank, name) for name in ("vector", "walked", "actor_edge", "aimed", "none")}
    nodes, forms = entry("graphspec.nodes", NODES), entry("graphspec.relation_forms", FORMS)
    joined = dict(models["vector"]["graph"], **JOINED)
    stated = [
        ("vector", [nodes, entry("test.cut", CUTTING), forms], {}),
        ("vector", [nodes, entry("test.cut", CUTTING), forms], dict(seed=6)),
        ("vector", [nodes, entry("test.cut", NEAR_TURN), forms], {}),
        ("vector", [nodes, entry("test.cut", SEVERING), forms], {}),
        ("vector", [nodes, entry("test.cut", NEAR_VISIT), forms], {}),
        ("vector", [nodes, forms], {}),
        ("walked", [nodes, entry("test.cut", CUTTING), forms], {}),
        ("actor_edge", [nodes, entry("test.cut", CUTTING), forms], {}),
        ("vector", [nodes, entry("test.sever", CUTTING), forms], {}),
        ("vector", [nodes, entry("test.sever", SEVERING), forms], {}),
        ("vector", [nodes, entry("test.part", PARTING), forms], {}),
        ("vector", [nodes, entry("test.part", PART_NEAR), forms], {}),
        ("vector", [nodes, entry("test.part", PART_VISIT), forms], {}),
        ("vector", [entry("graphspec.nodes", STARVED), forms], {}),
        ("none", [nodes, entry("test.cut", CUTTING), forms], {}),
        ("vector", [nodes, forms], dict(seed=8)),
        ("aimed", [nodes, entry("test.cut", SEVERING), forms], {}),
        ("vector", [nodes, entry("test.cut", CUTTING), forms], dict(graph=joined)),
        ("walked", [nodes, entry("test.cut", NEAR_TURN), forms], {}),
        ("vector", [nodes, entry("test.halt", {})], {}),
    ]
    result = [dict(models[name], pipeline=pipeline, **changes) for name, pipeline, changes in stated]
    LOG.debug("descriptions tried exit trials=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("descriptions.trial")
def trial(place):
    return "trial_%06d" % place


def laid(folder, bank, configs, unstarted):
    started = time.perf_counter()
    for place, config in enumerate(configs):
        (folder / trial(place)).mkdir(parents=True)
        write_json(folder / trial(place) / "config.json", config)
        if place not in unstarted:
            write_json(folder / trial(place) / "statement.json", dict(STATED, seed=config["seed"]))
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    trials = [dict(directory=trial(place)) for place in range(len(configs))]
    write_json(folder / "plan.json", dict(bank_stored_sha256=stored, trials=trials))
    LOG.debug("descriptions laid exit trials=%d seconds=%.4f", len(configs), time.perf_counter() - started)
    return folder


@pytest.fixture
def searched(bank, searching):
    started = time.perf_counter()
    result = shared("searched", lambda root: laid(root / "search", bank, tried(bank), (15,))) / "search"
    LOG.debug("descriptions searched exit seconds=%.4f", time.perf_counter() - started)
    return result


@timed("descriptions.tallying")
def tallying(held, connected, apart, absent, least, most, *cut):
    return dict(
        held=held,
        connected=connected,
        apart=apart,
        absent=absent,
        least=least,
        most=most,
        cut=[dict(entry=name, operation=name, options=options) for name, options in cut],
    )


NEVER = (None, 1)
UNMADE = tallying(1, 0, 0, 1, None, None, NEVER)
FAR = tallying(1, 1, 0, 0, 2, 2)
MOSTLY = tallying(13, 12, 1, 0, 1, 1, NEVER)
EVERY = tallying(13, 13, 0, 0, 1, 1)
ONE = tallying(1, 1, 0, 0, 1, 1)
WHOLE = [MOSTLY, FAR, UNMADE, MOSTLY]
EDITED = [EVERY, ONE, ONE, EVERY]
BASE_TURN = ("relation", "base", "turn", 2)
BASE_VISIT = ("relation", "base", "visit", 2)
NEAR_AT_TURN = ("relation", "near", "turn", 1)
NEAR_AT_VISIT = ("relation", "near", "visit", 1)
AGENTS = (("kind", "agent", "turn", 2), ("kind", "agent", "visit", 2))


@timed("descriptions.gone_by")
def gone_by(name):
    every = tallying(13, 0, 13, 0, None, None, (name, 12), NEVER)
    return [every, tallying(1, 0, 0, 1, None, None, (name, 1)), UNMADE, every]


@timed("descriptions.near_by")
def near_by(name):
    return [MOSTLY, tallying(1, 0, 1, 0, None, None, (name, 1)), UNMADE, MOSTLY]


@timed("descriptions.visited_by")
def visited_by(name):
    some = tallying(13, 9, 4, 0, 1, 1, (name, 3), NEVER)
    return [some, FAR, UNMADE, some]


@timed("descriptions.left_by")
def left_by(name, *found):
    return [
        dict(entry=name, operation=name, sort=sort, name=held, task=task, graphs=graphs) for sort, held, task, graphs in found
    ]


SEARCHED = (
    ("vector", gone_by("test.cut"), left_by("test.cut", BASE_TURN)),
    ("vector", gone_by("test.cut"), left_by("test.cut", BASE_TURN)),
    ("vector", near_by("test.cut"), left_by("test.cut", NEAR_AT_TURN)),
    ("vector", visited_by("test.cut"), left_by("test.cut", BASE_VISIT)),
    ("vector", WHOLE, left_by("test.cut", NEAR_AT_VISIT)),
    ("vector", WHOLE, []),
    ("walked", EDITED, left_by("test.cut", BASE_TURN)),
    ("actor_edge", gone_by("test.cut"), left_by("test.cut", BASE_TURN)),
    (None, REFUSAL),
    ("vector", visited_by("test.sever"), left_by("test.sever", BASE_VISIT)),
    ("vector", gone_by("test.part"), left_by("test.part", BASE_TURN, BASE_VISIT)),
    ("vector", near_by("test.part"), left_by("test.part", NEAR_AT_TURN, NEAR_AT_VISIT)),
    ("vector", visited_by("test.part"), left_by("test.part", BASE_VISIT)),
    ("vector", gone_by("graphspec.nodes"), left_by("graphspec.nodes", *AGENTS, BASE_TURN, BASE_VISIT)),
    (None, UNREALISED),
    (None, UNSTARTED),
    ("aimed", visited_by("test.cut")[:3] + [None], left_by("test.cut", BASE_VISIT)),
    ("vector", gone_by("test.cut"), left_by("test.cut", BASE_TURN)),
    ("walked", EDITED, left_by("test.cut", NEAR_AT_TURN)),
    ("vector", HALTED),
)
SEARCHED_TEXT = """reach over 20 trials on the bank stored as %(stored)s, described at version %(version)d
described 16, of 15 variants and realisations; not described 4
one row per trial, in the order of the trials; %(reading)s
    trial         variant     realisation  turn: options in decisions  turn: target to of go                                      turn: target along of go                 turn: target made of go           turn: parameter speed of go                                nothing left at the steps of tasks
    trial_000000  %(v0)s  vector       13 in 4                     0/13, apart 13, cut by test.cut 12 and no entry 1          0/1, absent 1, cut by test.cut 1         0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by test.cut 12 and no entry 1          test.cut: relation base at turn 2/2
    trial_000001  %(v1)s  vector       13 in 4                     0/13, apart 13, cut by test.cut 12 and no entry 1          0/1, absent 1, cut by test.cut 1         0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by test.cut 12 and no entry 1          test.cut: relation base at turn 2/2
    trial_000002  %(v2)s  vector       13 in 4                     12/13, hops 1, apart 1, cut by no entry 1                  0/1, apart 1, cut by test.cut 1          0/1, absent 1, cut by no entry 1  12/13, hops 1, apart 1, cut by no entry 1                  test.cut: relation near at turn 1/2
    trial_000003  %(v3)s  vector       13 in 4                     9/13, hops 1, apart 4, cut by test.cut 3 and no entry 1    1/1, hops 2                              0/1, absent 1, cut by no entry 1  9/13, hops 1, apart 4, cut by test.cut 3 and no entry 1    test.cut: relation base at visit 2/2
    trial_000004  %(v4)s  vector       13 in 4                     12/13, hops 1, apart 1, cut by no entry 1                  1/1, hops 2                              0/1, absent 1, cut by no entry 1  12/13, hops 1, apart 1, cut by no entry 1                  test.cut: relation near at visit 1/2
    trial_000005  %(v5)s  vector       13 in 4                     12/13, hops 1, apart 1, cut by no entry 1                  1/1, hops 2                              0/1, absent 1, cut by no entry 1  12/13, hops 1, apart 1, cut by no entry 1                  none
    trial_000006  %(v6)s  walked       13 in 4                     13/13, hops 1                                              1/1, hops 1                              1/1, hops 1                       13/13, hops 1                                              test.cut: relation base at turn 2/2
    trial_000007  %(v7)s  actor_edge   13 in 4                     0/13, apart 13, cut by test.cut 12 and no entry 1          0/1, absent 1, cut by test.cut 1         0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by test.cut 12 and no entry 1          test.cut: relation base at turn 2/2
    trial_000008  -           -            not described: the entry test.sever of the pipeline states cut/kind turn, which decides, and the entry cuts only at the times of the tasks of kinds that decide nothing
    trial_000009  %(v9)s  vector       13 in 4                     9/13, hops 1, apart 4, cut by test.sever 3 and no entry 1  1/1, hops 2                              0/1, absent 1, cut by no entry 1  9/13, hops 1, apart 4, cut by test.sever 3 and no entry 1  test.sever: relation base at visit 2/2
    trial_000010  %(v10)s  vector       13 in 4                     0/13, apart 13, cut by test.part 12 and no entry 1         0/1, absent 1, cut by test.part 1        0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by test.part 12 and no entry 1         test.part: relation base at turn 2/2, relation base at visit 2/2
    trial_000011  %(v11)s  vector       13 in 4                     12/13, hops 1, apart 1, cut by no entry 1                  0/1, apart 1, cut by test.part 1         0/1, absent 1, cut by no entry 1  12/13, hops 1, apart 1, cut by no entry 1                  test.part: relation near at turn 1/2, relation near at visit 1/2
    trial_000012  %(v12)s  vector       13 in 4                     9/13, hops 1, apart 4, cut by test.part 3 and no entry 1   1/1, hops 2                              0/1, absent 1, cut by no entry 1  9/13, hops 1, apart 4, cut by test.part 3 and no entry 1   test.part: relation base at visit 2/2
    trial_000013  %(v13)s  vector       13 in 4                     0/13, apart 13, cut by graphspec.nodes 12 and no entry 1   0/1, absent 1, cut by graphspec.nodes 1  0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by graphspec.nodes 12 and no entry 1   graphspec.nodes: kind agent at turn 2/2, kind agent at visit 2/2, relation base at turn 2/2, relation base at visit 2/2
    trial_000014  -           -            not described: the model of the configuration states options/realisation none, which puts no option before it; reach describes the options of decisions as a realisation puts them before a model
    trial_000015  -           -            not described: the trial holds no statement.json
    trial_000016  %(v16)s  aimed        13 in 4                     9/13, hops 1, apart 4, cut by test.cut 3 and no entry 1    1/1, hops 2                              0/1, absent 1, cut by no entry 1  unread                                                     test.cut: relation base at visit 2/2
    trial_000017  %(v17)s  vector       13 in 4                     0/13, apart 13, cut by test.cut 12 and no entry 1          0/1, absent 1, cut by test.cut 1         0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by test.cut 12 and no entry 1          test.cut: relation base at turn 2/2
    trial_000018  %(v18)s  walked       13 in 4                     13/13, hops 1                                              1/1, hops 1                              1/1, hops 1                       13/13, hops 1                                              test.cut: relation near at turn 1/2
    trial_000019  %(v19)s  vector       not described: the entry test.halt derives no graph
"""
ROW_TEXT = """reach of the options of the variant %s of the bank stored as %s in one row, described at version %d
%s
    variant     realisation  turn: options in decisions  turn: target to of go                                     turn: target along of go                 turn: target made of go           turn: parameter speed of go                               nothing left at the steps of tasks
    %s  vector       13 in 4                     0/13, apart 13, cut by graphspec.nodes 12 and no entry 1  0/1, absent 1, cut by graphspec.nodes 1  0/1, absent 1, cut by no entry 1  0/13, apart 13, cut by graphspec.nodes 12 and no entry 1  graphspec.nodes: kind agent at turn 2/2, kind agent at visit 2/2, relation base at turn 2/2, relation base at visit 2/2
"""
STOPPED_TEXT = """reach of the options of the variant %s of the bank stored as %s in one row, described at version %d
%s
    variant     realisation  turn: options in decisions  turn: target to of go  turn: target along of go  turn: target made of go  turn: parameter speed of go  nothing left at the steps of tasks
    %s  vector       not described: the entry test.halt derives no graph
"""
CROSSED_ROW = dict(
    turn=dict(
        decisions=3,
        options=5,
        fields={
            FIELDS[0]: tallying(5, 4, 1, 0, 0, 1, NEVER),
            FIELDS[1]: tallying(3, 2, 1, 0, 0, 1, NEVER),
            FIELDS[2]: tallying(2, 2, 0, 0, 0, 0),
            FIELDS[3]: tallying(0, 0, 0, 0, None, None),
            FIELDS[4]: tallying(2, 2, 0, 0, 0, 1),
            FIELDS[5]: tallying(3, 2, 1, 0, 0, 0, NEVER),
        },
    )
)


def over_search(bank, configs, stated):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    rows = []
    for place, (config, held) in enumerate(zip(configs, stated)):
        key = None if held[0] is None else specs.digest(deriver.identity(config))
        row = dict(trial=trial(place), variant=key, realisation=held[0], reason=None, graphs=None, tasks=None, cuts=None)
        if len(held) == 2:
            row["reason"] = held[1]
        else:
            fields = dict(zip(GO, held[1]))
            row.update(graphs=2, tasks=dict(turn=dict(decisions=4, options=13, fields=fields)), cuts=held[2])
        rows.append(row)
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    result = dict(version=describing.VERSION, stored_sha256=stored, fields=dict(turn=list(GO)), trials=rows)
    LOG.debug("descriptions over search exit trials=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return result


def tabled(folder, record):
    started = time.perf_counter()
    trials = [
        dict(
            trial=row["trial"],
            row=None if row["variant"] is None else reaching.stemmed(row["variant"], row["realisation"]),
            reason=row["reason"] if row["variant"] is None else None,
        )
        for row in record["trials"]
    ]
    key = sha256_json(dict(stored=record["stored_sha256"], version=describing.VERSION, trials=trials))
    LOG.debug("descriptions tabled exit trials=%d seconds=%.4f", len(trials), time.perf_counter() - started)
    return folder / "descriptions" / "reaches" / key


def test_reach_over_the_twenty_trials_of_a_search_equals_the_table_written_out_here_and_keeps_the_row_of_each_variant(
    bank, searched
):
    started = time.perf_counter()
    record = reach_search(bank, searched)
    assert record == over_search(bank, tried(bank), SEARCHED)
    assert json.loads(json.dumps(record)) == record
    rows = record["trials"]
    assert rows[0]["variant"] == rows[1]["variant"] == rows[7]["variant"] != rows[17]["variant"]
    assert rows[3]["variant"] == rows[16]["variant"] and rows[5]["variant"] is not None and rows[15]["variant"] is None
    described = [row for row in rows if row["reason"] is None]
    assert len(described) == 16 and len({(row["variant"], row["realisation"]) for row in described}) == 15
    for place, written in ((13, ROW_TEXT), (19, STOPPED_TEXT)):
        key = rows[place]["variant"]
        stem = bank / "descriptions" / "reach_rows" / reaching.stemmed(key, "vector")
        kept = {name: value for name, value in rows[place].items() if name != "trial"}
        assert read_json(stem.with_suffix(".json")) == dict(
            kept, version=describing.VERSION, stored_sha256=record["stored_sha256"], fields=record["fields"]
        )
        assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written % (
            key,
            record["stored_sha256"],
            describing.VERSION,
            reaches.READING,
            key[:10],
        )
    LOG.info("descriptions reach over a search exit seconds=%.3f", time.perf_counter() - started)


def test_the_text_of_the_table_holds_one_line_per_trial_of_its_record_and_the_verb_prints_what_the_call_writes(
    bank, searched, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    record = reach_search(bank, searched)
    shown = {"v%d" % place: row["variant"][:10] for place, row in enumerate(record["trials"]) if row["variant"] is not None}
    written = SEARCHED_TEXT % dict(
        shown, stored=record["stored_sha256"], version=describing.VERSION, reading=reaches.READING
    )
    assert reaches.text(record) == written
    stem = tabled(searched, record)
    assert read_json(stem.with_suffix(".json")) == record
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--reach", str(bank), str(searched)])
    main()
    assert capsys.readouterr().out == written
    LOG.info("descriptions reach over a search text exit seconds=%.3f", time.perf_counter() - started)


def test_a_second_call_reads_the_file_of_the_table_and_a_search_that_gains_a_trial_is_described_anew_from_the_rows_kept(
    bank, searching, tmp_path, monkeypatch
):
    started = time.perf_counter()
    configs = tried(bank)
    folder = laid(tmp_path / "search", bank, configs[:2], ())
    record = reach_search(bank, folder)
    assert record == over_search(bank, configs[:2], SEARCHED[:2])
    assert tabled(folder, record).with_suffix(".json").exists()
    monkeypatch.setattr(reaches, "recorded", refused)
    assert reach_search(bank, folder) == record
    (folder / trial(2)).mkdir()
    write_json(folder / trial(2) / "config.json", configs[5])
    write_json(folder / trial(2) / "statement.json", STATED)
    plan = read_json(folder / "plan.json")
    write_json(folder / "plan.json", dict(plan, trials=plan["trials"] + [dict(directory=trial(2))]))
    with pytest.raises(AssertionError, match="the description was computed again"):
        reach_search(bank, folder)
    monkeypatch.undo()
    rowed = Mock(wraps=reaches.rowed)
    monkeypatch.setattr(reaches, "rowed", rowed)
    gained = reach_search(bank, folder)
    assert gained == over_search(bank, configs[:2] + [configs[5]], SEARCHED[:2] + (SEARCHED[5],))
    assert [call.args[2] for call in rowed.call_args_list] in ([], [gained["trials"][2]["variant"]])
    assert tabled(folder, gained) != tabled(folder, record) and tabled(folder, gained).with_suffix(".txt").exists()
    monkeypatch.setattr(reaches, "recorded", refused)
    assert reach_search(bank, folder) == gained
    LOG.info("descriptions reach over a search served exit seconds=%.3f", time.perf_counter() - started)


CROSSED_OVER = """reach over 3 trials on the bank stored as %s, described at version %d
described 2, of 1 variants and realisations; not described 1
one row per trial, in the order of the trials; %s
    trial  variant     realisation  turn: options in decisions  turn: kind                                    turn: target to of go                         turn: target along of go  turn: target made of go  turn: target by of stay  turn: parameter speed of go              nothing left at the steps of tasks
    0      %s  vector       5 in 3                      4/5, hops 0 to 1, apart 1, cut by no entry 1  2/3, hops 0 to 1, apart 1, cut by no entry 1  2/2, hops 0               held by none             2/2, hops 0 to 1         2/3, hops 0, apart 1, cut by no entry 1  none
    1      %s  vector       5 in 3                      4/5, hops 0 to 1, apart 1, cut by no entry 1  2/3, hops 0 to 1, apart 1, cut by no entry 1  2/2, hops 0               held by none             2/2, hops 0 to 1         2/3, hops 0, apart 1, cut by no entry 1  none
    2      -           -            not described: the seed of the configuration is 6 and the seed of the statement is 5
"""


def test_a_plain_list_of_configurations_is_described_beside_its_bank_and_variants_alike_are_derived_once(
    crossed, monkeypatch
):
    started = time.perf_counter()
    config = lined(crossed, THINNER, KEPT, "vector")
    wider = dict(config, model=dict(config["model"], hidden=config["model"]["hidden"] + 1))
    rowed = Mock(wraps=reaches.rowed)
    monkeypatch.setattr(reaches, "rowed", rowed)
    record = reach_over(crossed, [config, wider, dict(config, seed=6)], STATED)
    stored, key = read_json(crossed / "manifest.json")["stored_sha256"], keyed(crossed, config)
    described = dict(variant=key, realisation="vector", reason=None, graphs=1, tasks=CROSSED_ROW, cuts=[])
    unread = dict(variant=None, realisation=None, reason=SEEDED, graphs=None, tasks=None, cuts=None)
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=stored,
        fields=dict(turn=list(FIELDS)),
        trials=[dict(described, trial=0), dict(described, trial=1), dict(unread, trial=2)],
    )
    assert json.loads(json.dumps(record)) == record and rowed.call_count == 1
    written = CROSSED_OVER % (stored, describing.VERSION, reaches.READING, key[:10], key[:10])
    stem = tabled(crossed, record)
    assert reaches.text(record) == written == stem.with_suffix(".txt").read_text(encoding="utf-8")
    assert read_json(stem.with_suffix(".json")) == record
    assert (crossed / "descriptions" / "reach_rows" / reaching.stemmed(key, "vector")).with_suffix(".json").exists()
    monkeypatch.setattr(reaches, "recorded", refused)
    assert reach_over(crossed, [config, wider, dict(config, seed=6)], STATED) == record
    with pytest.raises(AssertionError, match="the description was computed again"):
        reach_over(crossed, [wider, config], STATED)
    LOG.info("descriptions reach over a list exit seconds=%.3f", time.perf_counter() - started)


def test_reach_over_a_search_stops_naming_a_folder_without_a_plan_a_plan_of_another_bank_and_a_set_that_is_no_list(
    bank, path_bank, tmp_path, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    with pytest.raises(FileNotFoundError, match="holds no plan.json; reach over a search reads the folder of a search"):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(trials=[]))
    with pytest.raises(KeyError, match="lacks bank_stored_sha256"):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(bank_stored_sha256="0" * 64, trials=[]))
    with pytest.raises(ValueError, match="searches the bank whose build stored 0{64}, and the bank .* stored %s$" % stored):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(bank_stored_sha256=stored, trials=[]))
    empty = dict(version=describing.VERSION, stored_sha256=stored, fields=dict(turn=list(GO)), trials=[])
    assert reach_search(bank, tmp_path) == empty == reach_over(bank, [], STATED)
    with pytest.raises(TypeError, match="reads a list of the configurations of fits, found {}"):
        reach_over(bank, {}, STATED)
    with pytest.raises(KeyError, match="the statement lacks alike"):
        reach_over(bank, [], {name: value for name, value in STATED.items() if name != "alike"})
    with pytest.raises(ValueError, match="holds no kind of task that decides; reach describes the options of decisions$"):
        reach_over(path_bank, [], STATED)
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--reach", str(bank), str(tmp_path / "plan.json")])
    with pytest.raises(SystemExit):
        main()
    assert "plan.json is no folder; describe --reach reads the folder of a search" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--reach", str(bank)])
    with pytest.raises(SystemExit):
        main()
    assert "found 0 paths after the bank" in capsys.readouterr().err
    LOG.info("descriptions reach over a search refused exit seconds=%.3f", time.perf_counter() - started)


DEGREE = "nodes/agent/min_degree"
NODAL = ("first", "second", "first_anchors", "second_anchors")
EDGED = ("first", "second", "valued")
SWAPPED = dict(first="second", second="first", first_anchors="second_anchors", second_anchors="first_anchors")
UNSHARED = dict(first=[], second=[])


@timed("descriptions.unlike")
def unlike(labels, stated, totals, *rows):
    return dict(
        stated=stated,
        **dict(zip(labels, totals)),
        at=[dict(time=row[0], **dict(zip(labels, row[1:]))) for row in rows],
    )


LEANER = dict(
    name="g0",
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    alone=UNSHARED,
    kinds=dict(
        agent=unlike(
            NODAL,
            "both",
            (11, 0, 0, 3),
            (0, 1, 0, 0, 0),
            (1, 2, 0, 0, 1),
            (2, 1, 0, 0, 0),
            (3, 1, 0, 0, 1),
            (4, 2, 0, 0, 0),
            (5, 2, 0, 0, 1),
            (6, 1, 0, 0, 0),
            (7, 1, 0, 0, 0),
        )
    ),
    relations={
        "base": unlike(EDGED, "both", (6, 0, 0), (0, 1, 0, 0), (2, 1, 0, 0), (4, 2, 0, 0), (6, 1, 0, 0), (7, 1, 0, 0)),
        LOOPS: unlike(
            EDGED,
            "both",
            (11, 0, 0),
            (0, 1, 0, 0),
            (1, 2, 0, 0),
            (2, 1, 0, 0),
            (3, 1, 0, 0),
            (4, 2, 0, 0),
            (5, 2, 0, 0),
            (6, 1, 0, 0),
            (7, 1, 0, 0),
        ),
    },
    equal=dict(kinds=1, relations=1),
)
LEANER_SECOND = dict(
    name="g1",
    times=[1, 2, 3],
    alone=UNSHARED,
    kinds=dict(agent=unlike(NODAL, "both", (2, 0, 0, 1), (1, 1, 0, 0, 0), (2, 1, 0, 0, 1))),
    relations={
        "base": unlike(EDGED, "both", (1, 0, 0), (1, 1, 0, 0)),
        LOOPS: unlike(EDGED, "both", (2, 0, 0), (1, 1, 0, 0), (2, 1, 0, 0)),
    },
    equal=dict(kinds=1, relations=1),
)
TRUTHS = dict(
    first="g0",
    second="g1",
    unit="tick",
    differs=True,
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    kinds=dict(
        agent=unlike(
            NODAL,
            "both",
            (8, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (2, 1, 0, 0, 0),
            (3, 1, 0, 0, 0),
            (4, 1, 0, 0, 0),
            (5, 1, 0, 0, 0),
            (6, 1, 0, 0, 0),
            (7, 1, 0, 0, 0),
        ),
        spot=unlike(
            NODAL,
            "both",
            (9, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (2, 1, 0, 0, 0),
            (3, 2, 0, 0, 0),
            (4, 2, 0, 0, 0),
            (5, 1, 0, 0, 0),
            (6, 1, 0, 0, 0),
            (7, 1, 0, 0, 0),
        ),
    ),
    relations=dict(
        near=unlike(
            EDGED, "both", (8, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (2, 2, 0, 0), (3, 2, 0, 0), (4, 1, 0, 0), (5, 1, 0, 0)
        ),
        base=unlike(
            EDGED,
            "both",
            (10, 2, 0),
            (0, 1, 1, 0),
            (1, 1, 0, 0),
            (2, 1, 1, 0),
            (3, 1, 0, 0),
            (4, 2, 0, 0),
            (5, 2, 0, 0),
            (6, 1, 0, 0),
            (7, 1, 0, 0),
        ),
    ),
    equal=dict(kinds=0, relations=0),
)
SIDE_TEXT = """derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree %d, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none"""
HEAD_TEXT = """difference of the variants %s, the first, and %s, the second, of the bank stored as %s, described at version %d
%s
the first:
%s
the second:
%s
"""
DIFFERENCE = """
graph g0, times 8 from 0 to 7; times only the first shows: none; times only the second shows: none
  kind agent: nodes only the first holds 11, only the second 0; anchors only the first holds 0, only the second 3
    time  first nodes  second nodes  first anchors  second anchors
    0     1            0             0              0
    1     2            0             0              1
    2     1            0             0              0
    3     1            0             0              1
    4     2            0             0              0
    5     2            0             0              1
    6     1            0             0              0
    7     1            0             0              0
  relation base: edges only the first holds 6, only the second 0, held by both with other values 0
    time  first  second  other values
    0     1      0       0
    2     1      0       0
    4     2      0       0
    6     1      0       0
    7     1      0       0
  relation relation_forms.loops: edges only the first holds 11, only the second 0, held by both with other values 0
    time  first  second  other values
    0     1      0       0
    1     2      0       0
    2     1      0       0
    3     1      0       0
    4     2      0       0
    5     2      0       0
    6     1      0       0
    7     1      0       0
  equal at every time: kinds 1, relations 1

graph g1, times 3 from 1 to 3; times only the first shows: none; times only the second shows: none
  kind agent: nodes only the first holds 2, only the second 0; anchors only the first holds 0, only the second 1
    time  first nodes  second nodes  first anchors  second anchors
    1     1            0             0              0
    2     1            0             0              1
  relation base: edges only the first holds 1, only the second 0, held by both with other values 0
    time  first  second  other values
    1     1      0       0
  relation relation_forms.loops: edges only the first holds 2, only the second 0, held by both with other values 0
    time  first  second  other values
    1     1      0       0
    2     1      0       0
  equal at every time: kinds 1, relations 1
"""
SAME_TEXT = "nothing differs: the two hold the same nodes, anchors and edges at every time of each of the 2 graphs of the view\n"
TRUTHS_TEXT = """difference of the truth graphs g0, the first, and g1, the second, of the bank stored as %s, described at version %d
unit tick, times 8 from 0 to 7
%s
  kind agent: nodes only the first holds 8, only the second 0; anchors only the first holds 0, only the second 0
    time  first nodes  second nodes  first anchors  second anchors
    0     1            0             0              0
    1     1            0             0              0
    2     1            0             0              0
    3     1            0             0              0
    4     1            0             0              0
    5     1            0             0              0
    6     1            0             0              0
    7     1            0             0              0
  kind spot: nodes only the first holds 9, only the second 0; anchors only the first holds 0, only the second 0
    time  first nodes  second nodes  first anchors  second anchors
    0     1            0             0              0
    2     1            0             0              0
    3     2            0             0              0
    4     2            0             0              0
    5     1            0             0              0
    6     1            0             0              0
    7     1            0             0              0
  relation near: edges only the first holds 8, only the second 0, held by both with other values 0
    time  first  second  other values
    0     1      0       0
    1     1      0       0
    2     2      0       0
    3     2      0       0
    4     1      0       0
    5     1      0       0
  relation base: edges only the first holds 10, only the second 2, held by both with other values 0
    time  first  second  other values
    0     1      1       0
    1     1      0       0
    2     1      1       0
    3     1      0       0
    4     2      0       0
    5     2      0       0
    6     1      0       0
    7     1      0       0
  equal at every time: kinds 0, relations 0
"""
ALONE_TEXT = """difference of the truth graphs g1, the first, and g1, the second, of the bank stored as %s, described at version %d
unit tick, times 3 from 1 to 3
%s
  nothing differs; equal at every time: kinds 2, relations 2
"""
TIMELESS_TEXT = """unit tick, no time
%s
  kind agent: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0
    time  first nodes  second nodes  first anchors  second anchors
    none  1            0             0              0
  kind spot: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0
    time  first nodes  second nodes  first anchors  second anchors
    none  1            0             0              0
  relation near: edges only the first holds 2, only the second 0, held by both with other values 0
    time  first  second  other values
    none  2      0       0
  relation base: edges only the first holds 1, only the second 0, held by both with other values 0
    time  first  second  other values
    none  1      0       0
  equal at every time: kinds 0, relations 0
"""


@timed("descriptions.turned_over")
def turned_over(value):
    if type(value) is dict:
        return {
            SWAPPED.get(name, name): SWAPPED.get(held, held) if name == "stated" else turned_over(held)
            for name, held in value.items()
        }
    return [turned_over(held) for held in value] if type(value) is list else value


@timed("descriptions.leaning")
def leaning(degree):
    return [dict(entry, parameters=dict(entry["parameters"], **{DEGREE: degree})) if DEGREE in entry["parameters"] else entry for entry in pipelined(CUTTING)]


@timed("descriptions.leaned")
def leaned(bank, degree):
    return configuration(bank, MODEL, None, pipeline=leaning(degree), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.diverged")
def diverged(bank, degrees, differs, graphs):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    sides = [
        dict(
            variant=specs.digest(deriver.identity(leaned(bank, degree))),
            seed=STATED["derivation_seed"],
            device="cpu",
            splits=SPLITS,
            settings=SETTLED,
            entries=entered(leaning(degree)),
        )
        for degree in degrees
    ]
    record = dict(
        version=describing.VERSION, stored_sha256=stored, first=sides[0], second=sides[1], differs=differs, graphs=graphs
    )
    versions = [OPERATIONS.named(name).version for name in ("graphspec.nodes", "graphspec.relation_forms")]
    written = HEAD_TEXT % (
        sides[0]["variant"],
        sides[1]["variant"],
        stored,
        describing.VERSION,
        differing.READING,
        *[SIDE_TEXT % (versions[0], degree, versions[1]) for degree in degrees],
    )
    key = dict(version=describing.VERSION, first=dict(variant=sides[0]["variant"]), second=dict(variant=sides[1]["variant"]))
    return record, written, bank / "descriptions" / "differences" / sha256_json(key)


@timed("descriptions.truths")
def truths(bank, first, second, found):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    key = dict(stored=stored, version=describing.VERSION, first=dict(graph=first), second=dict(graph=second))
    return (
        dict(version=describing.VERSION, stored_sha256=stored, **found),
        bank / "descriptions" / "differences" / sha256_json(key),
    )


def test_two_variants_that_differ_by_one_setting_differ_in_the_description_by_what_that_setting_removes_and_nothing_else(
    bank, cutting
):
    assert {DEGREE: 1}.items() <= NODES.items() and leaning(2)[0]["parameters"] == dict(NODES, **{DEGREE: 2})
    started = time.perf_counter()
    first, second = leaned(bank, 1), leaned(bank, 2)
    record = against(bank, first, STATED, second, STATED)
    assert record == diverged(bank, (1, 2), True, [LEANER, LEANER_SECOND])[0]
    assert json.loads(json.dumps(record)) == record
    assert against(bank, second, STATED, first, STATED) == turned_over(record)
    assert turned_over(record) == diverged(bank, (2, 1), True, [turned_over(LEANER), turned_over(LEANER_SECOND)])[0]
    assert turned_over(LEANER)["kinds"]["agent"]["at"][1] == dict(time=1, first=0, second=2, first_anchors=1, second_anchors=0)
    same = [
        dict(name=name, times=times, alone=UNSHARED, kinds={}, relations={}, equal=dict(kinds=2, relations=3))
        for name, times in (("g0", LEANER["times"]), ("g1", LEANER_SECOND["times"]))
    ]
    expected, written, _ = diverged(bank, (1, 1), False, same)
    assert against(bank, first, STATED, first, STATED) == expected
    assert diverging.text(expected) == written + SAME_TEXT
    joined = dict(pipeline=leaning(2), graph=dict(SETTINGS, **JOINED), seed=STATED["seed"], splits=SPLITS)
    assert against(bank, first, STATED, configuration(bank, MODEL, None, **joined), STATED)["graphs"] == record["graphs"]
    reseeded = against(bank, first, STATED, first, dict(STATED, derivation_seed=8))
    assert (reseeded["first"]["seed"], reseeded["second"]["seed"], reseeded["differs"]) == (7, 8, False)
    LOG.info("descriptions difference of variants exit seconds=%.3f", time.perf_counter() - started)


def test_two_truth_graphs_of_a_bank_differ_by_what_is_written_out_here_and_a_graph_beside_itself_in_nothing(bank):
    started = time.perf_counter()
    expected, _ = truths(bank, "g0", "g1", TRUTHS)
    record = between(bank, "g0", "g1")
    assert record == expected and json.loads(json.dumps(record)) == record
    assert between(bank, "g1", "g0") == turned_over(expected)
    assert turned_over(expected)["relations"]["base"]["at"][0] == dict(time=0, first=1, second=1, valued=0)
    assert turned_over(expected)["relations"]["base"]["at"][1] == dict(time=1, first=0, second=1, valued=0)
    alone = dict(first="g1", second="g1", unit="tick", differs=False, times=[1, 2, 3], kinds={}, relations={})
    alone, stem = truths(bank, "g1", "g1", dict(alone, equal=dict(kinds=2, relations=2)))
    assert between(bank, "g1", "g1") == alone
    written = ALONE_TEXT % (alone["stored_sha256"], describing.VERSION, differing.READING)
    assert differing.text(alone) == written == stem.with_suffix(".txt").read_text(encoding="utf-8")
    with pytest.raises(KeyError, match="holds no graph named g9; its graphs are g0, g1"):
        between(bank, "g0", "g9")
    LOG.info("descriptions difference of truth graphs exit seconds=%.3f", time.perf_counter() - started)


def test_the_text_of_a_difference_holds_its_record_and_the_verb_prints_what_the_call_writes(
    bank, cutting, tmp_path, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    first, second = leaned(bank, 1), leaned(bank, 2)
    record = against(bank, first, STATED, second, STATED)
    expected, head, stem = diverged(bank, (1, 2), True, [LEANER, LEANER_SECOND])
    assert diverging.text(record) == head + DIFFERENCE
    assert read_json(stem.with_suffix(".json")) == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == head + DIFFERENCE
    names = ("config.json", "statement.json", "other.json")
    for name, held in zip(names, (first, STATED, second)):
        write_json(tmp_path / name, held)
    files = [str(tmp_path / names[place]) for place in (0, 1, 2, 1)]
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--difference", str(bank), *files])
    main()
    assert capsys.readouterr().out == head + DIFFERENCE
    record = between(bank, "g0", "g1")
    expected, stem = truths(bank, "g0", "g1", TRUTHS)
    written = TRUTHS_TEXT % (expected["stored_sha256"], describing.VERSION, differing.READING)
    assert differing.text(record) == written
    assert read_json(stem.with_suffix(".json")) == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--difference", str(bank), "g0", "g1"])
    main()
    assert capsys.readouterr().out == written
    LOG.info("descriptions difference text exit seconds=%.3f", time.perf_counter() - started)


def test_a_second_call_reads_the_file_of_a_difference_and_another_pair_is_not_served_it(bank, cutting, monkeypatch):
    started = time.perf_counter()
    first, second = leaned(bank, 1), leaned(bank, 2)
    record = against(bank, first, STATED, second, STATED)
    stems = [diverged(bank, degrees, True, [])[2] for degrees in ((1, 2), (1, 3), (2, 1))]
    assert stems[0].with_suffix(".json").exists() and len(set(stems)) == 3
    monkeypatch.setattr(diverging, "recorded", refused)
    assert against(bank, first, STATED, second, STATED) == record
    with pytest.raises(AssertionError, match="the description was computed again"):
        against(bank, first, STATED, leaned(bank, 3), STATED)
    record = between(bank, "g0", "g1")
    assert truths(bank, "g0", "g1", {})[1].with_suffix(".json").exists()
    monkeypatch.setattr(differing, "recorded", refused)
    assert between(bank, "g0", "g1") == record
    with pytest.raises(AssertionError, match="the description was computed again"):
        between(bank, "g0", "g0")
    LOG.info("descriptions difference served exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.timeless")
def timeless(facts):
    return facts.replace(
        obs_since=np.full(len(facts.obs_since), EARLIEST, np.int64),
        obs_until=np.full(len(facts.obs_until), LATEST, np.int64),
        fact_since=np.full(len(facts.fact_since), EARLIEST, np.int64),
        fact_until=np.full(len(facts.fact_until), LATEST, np.int64),
        task_time=np.full(len(facts.task_time), LATEST, np.int64),
    )


def test_edges_both_hold_with_other_values_graphs_that_hold_no_time_and_steps_only_one_side_shows_are_told_apart(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    manifest = read_json(bank / "manifest.json")
    registry = Registry.from_declaration(manifest["spec"])
    held = [describing.stored(bank, graph) for graph in manifest["graphs"]]
    moments = describing.clocked(registry, held[0])
    values = np.array(held[0].fact_values)
    values[values[:, 0] == 2.0, 0] = 5.0
    sides = [differing.lasting(registry, facts, moments) for facts in (held[0], held[0].replace(fact_values=values))]
    assert differing.differed(*sides, moments.tolist()) == dict(
        kinds={},
        relations=dict(near=unlike(EDGED, "both", (0, 0, 3), (3, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1))),
        equal=dict(kinds=2, relations=1),
    )
    still = [timeless(facts) for facts in held]
    monkeypatch.setattr(describing, "stored", lambda path, graph: still[graph["index"]])
    record = differing.recorded(bank, manifest, *manifest["graphs"])
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=manifest["stored_sha256"],
        first="g0",
        second="g1",
        unit="tick",
        differs=True,
        times=[None],
        kinds=dict(
            agent=unlike(NODAL, "both", (1, 0, 0, 0), (None, 1, 0, 0, 0)),
            spot=unlike(NODAL, "both", (1, 0, 0, 0), (None, 1, 0, 0, 0)),
        ),
        relations=dict(
            near=unlike(EDGED, "both", (2, 0, 0), (None, 2, 0, 0)), base=unlike(EDGED, "both", (1, 0, 0), (None, 1, 0, 0))
        ),
        equal=dict(kinds=0, relations=0),
    )
    assert differing.text(record).split("\n", 1)[1] == TIMELESS_TEXT % differing.READING
    config = leaned(bank, 1)
    name, graph, grown, low, high = diverging.ended(Deriver(bank, "cpu", STATED["derivation_seed"], RESIDENT), config)[0]
    assert (name, low, high) == ("g0", 0, 8)
    assert diverging.paired((name, graph, grown, 0, 8), (name, graph, grown, 1, 7)) == dict(
        name="g0",
        times=[0, 1, 2, 3, 4, 5, 6, 7],
        alone=dict(first=[0, 7], second=[]),
        kinds=dict(
            agent=unlike(NODAL, "both", (2, 0, 0, 0), (0, 1, 0, 0, 0), (7, 1, 0, 0, 0)),
            spot=unlike(NODAL, "both", (4, 0, 0, 0), (0, 2, 0, 0, 0), (7, 2, 0, 0, 0)),
        ),
        relations={
            "near": unlike(EDGED, "both", (1, 0, 0), (0, 1, 0, 0)),
            "base": unlike(EDGED, "both", (2, 0, 0), (0, 1, 0, 0), (7, 1, 0, 0)),
            LOOPS: unlike(EDGED, "both", (6, 0, 0), (0, 3, 0, 0), (7, 3, 0, 0)),
        },
        equal=dict(kinds=0, relations=0),
    )
    unlooped = leaning(1)[:2] + [dict(operation="graphspec.relation_forms", parameters=dict(FORMS, **{"relation_forms/loops": False}))]
    unlooped = configuration(bank, MODEL, None, pipeline=unlooped, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    found = against(bank, config, STATED, unlooped, STATED)["graphs"]
    assert [(graph["kinds"], graph["equal"]) for graph in found] == [({}, dict(kinds=2, relations=2))] * 2
    assert found[1]["relations"] == {LOOPS: unlike(EDGED, "first", (7, 0, 0), (1, 3, 0, 0), (2, 3, 0, 0), (3, 1, 0, 0))}
    assert "  relation relation_forms.loops, which only the first states: edges only the first holds 7," in "\n".join(
        differing.graph_text(found[1])
    )
    LOG.info("descriptions difference apart exit seconds=%.3f", time.perf_counter() - started)


def test_a_difference_stops_naming_what_it_cannot_meet(bank, capsys, monkeypatch):
    from tuning.__main__ import main

    started = time.perf_counter()
    manifest = read_json(bank / "manifest.json")
    registry = Registry.from_declaration(manifest["spec"])
    held = [describing.stored(bank, graph) for graph in manifest["graphs"]]
    moments = describing.clocked(registry, held[0])
    undirected = dict(
        manifest["spec"], edges=[dict(edge, symmetric=True) if edge["name"] == "base" else edge for edge in manifest["spec"]["edges"]]
    )
    sides = [differing.lasting(found, held[0], moments) for found in (registry, Registry.from_declaration(undirected))]
    with pytest.raises(ValueError, match="the relation base has a direction on one side and none on the other"):
        differing.differed(*sides, moments.tolist())
    with pytest.raises(ValueError, match="the second shows the graph g0 at two steps of one time, among 0, 1, 1"):
        differing.met("g0", [0, 1], [0, 1, 1])
    last = held[1]
    hourly = Facts(last.name, last.keys, "hour", last.inner, last.observers, last.options, last.generators, **last.arrays)
    monkeypatch.setattr(describing, "stored", lambda path, graph: (held[0], hourly)[graph["index"]])
    with pytest.raises(ValueError, match="the graph g0 counts time in tick and the graph g1 in hour"):
        differing.recorded(bank, manifest, *manifest["graphs"])
    monkeypatch.setattr(diverging, "ended", Mock(side_effect=[[("g0",)], [("g1",)]]))
    with pytest.raises(ValueError, match="the first variant shows the graphs g0 and the view of the second g1"):
        diverging.recorded([(None, None), (None, None)], (None, None))
    for stated, refusal in (
        (["--reach", "--difference", str(bank), "g0", "g1"], "describe takes --reach or --difference, not both"),
        (["--difference", str(bank), "g0"], "found 1 paths after the bank"),
        (["--difference", str(bank), "g0", "g1", "g0"], "found 3 paths after the bank"),
    ):
        monkeypatch.setattr(sys, "argv", ["graphspec", "describe", *stated])
        with pytest.raises(SystemExit):
            main()
        assert refusal in capsys.readouterr().err
    LOG.info("descriptions difference refused exit seconds=%.3f", time.perf_counter() - started)
