import json
import logging
import shutil
import sys
import time

import pytest

from conftest import shared
from graphspec import describe
from graphspec.descriptions import describe as describing
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.sources import synthetic
from tuning.bank.build import build

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
TEXT = """truth graphs of the bank stored as %s, described at version 1
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
        version=1,
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
    written = TEXT % (stored, describing.READING)
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
