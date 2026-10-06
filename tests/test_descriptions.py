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
from graphspec.operations import parameters as specs
from graphspec.operations.parameters import choice
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.sources import synthetic
from tuning import configuration
from tuning import describe as variant
from tuning.bank.bank import Deriver
from tuning.bank.build import build
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
    go = dict(roles=dict(to=dict(sort="node", kinds=["spot"], count=[1, 1])), parameters=["speed"])
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


@timed("descriptions.turned")
def turned(actor, aims, chosen, gain):
    options = [
        synthetic.offered(
            "o%d" % at, [synthetic.acting("go", actor, dict(to=[spot]), dict(speed=speed))], "set", "recorded"
        )
        for at, (spot, speed) in enumerate(aims)
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
            (10, "turn", 1, "train", ["a0"], turned("a0", [("s0", 1.0), ("s1", 2.0)], 0, 1.0)),
            (20, "visit", 2, "train", "s2", 1.5),
            (30, "turn", 3, "train", ["a1"], turned("a1", [("s0", 2.0), ("s1", 1.0), ("s2", 3.0)], 1, 3.0)),
            (40, "visit", 4, "train", "s0", 2.5),
            (50, "turn", 5, "validation", ["a0"], turned("a0", [("s0", 1.0), ("s1", 2.0)], 0, 2.0)),
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
            (10, "turn", 2, "train", ["a0"], turned("a0", [("s0", 1.0), ("s1", 2.0)], 0, 4.0)),
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
    tasked = np.asarray(graph.task_step)[np.asarray(graph.task_kind) == registry.tasks.index(settings["cut/kind"])]
    gone = (
        np.isin(graph.fact_code, registry.coded[settings["cut/relation"]])
        & np.isin(graph.fact_step, tasked)
        & np.asarray(graph.fact_present, bool)
    )
    fields = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
    result = graph.replace(**{name: np.asarray(graph.arrays[name])[~gone] for name in fields})
    LOG.debug("descriptions unheld exit removed=%d seconds=%.4f", int(gone.sum()), time.perf_counter() - started)
    return result


CUT = Operation("test", "cut", (Phase("cut", unheld, None, None),), choices, (), (), (), ("facts",), None, None, 1, 1)


@pytest.fixture
def cutting():
    started = time.perf_counter()
    OPERATIONS.register(CUT.name, CUT)
    LOG.debug("descriptions cutting registered seconds=%.4f", time.perf_counter() - started)
    yield CUT
    del OPERATIONS.held[CUT.name]


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


@timed("descriptions.varied")
def varied(bank, config):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    key = specs.digest(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT).identity(config))
    record = dict(
        version=describing.VERSION,
        variant=key,
        stored_sha256=stored,
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings={
            "graphs/join": "separate",
            "nodes/untyped": False,
            "relations/direction": "inverse",
            "time/steps": "changes",
            "time/several": "last",
        },
        entries=[
            dict(
                name=name,
                operation=name,
                version=OPERATIONS.named(name).version,
                graph=None,
                parameters=parameters,
                reads={},
                splits={},
                storage={},
            )
            for name, parameters in (("graphspec.nodes", NODES), ("test.cut", CUTTING), ("graphspec.relation_forms", FORMS))
        ],
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


def test_a_joined_group_is_described_graph_by_graph_as_its_graphs_derived_apart_are(bank, cutting):
    started = time.perf_counter()
    joined = dict(
        SETTINGS,
        **{
            "graphs/join": "single",
            "graphs/join_size": 2,
            "graphs/connect/agent": True,
            "graphs/connect/spot": True,
            "graphs/connect_fraction": 1.0,
        },
    )
    config = configuration(
        bank, MODEL, None, pipeline=pipelined(CUTTING), graph=joined, seed=STATED["seed"], splits=SPLITS
    )
    record = variant(bank, config, STATED)
    assert record["settings"]["graphs/join"] == "single" and record["graphs"] == [VARIED, VARIED_SECOND]
    LOG.info("descriptions variant joined exit seconds=%.3f", time.perf_counter() - started)
