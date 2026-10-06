import io
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import ANY, Mock

import numpy as np
import pytest

from conftest import SPLITS, shared
from graphspec import ENCODERS, OPERATIONS, RULES, Operation, Phase, Registry, describe, registries
from graphspec import counts as held_at
from graphspec import differences as between
from graphspec.budgets import GraphCapacityError, Refused
from graphspec.descriptions import describe as describing
from graphspec.descriptions import differences as differing
from graphspec.descriptions import effect
from graphspec.engine import transform
from graphspec.facts.stored import Facts
from graphspec.infra import runtime
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.runtime import cache_path
from graphspec.infra.timing import timed
from graphspec.operations import nodes as filtering
from graphspec.operations import parameters as specs
from graphspec.operations.operation import Need
from graphspec.operations.parameters import choice, equals, number
from graphspec.operations.rows import made_rows
from graphspec.representations import groups as representing
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.schema.time import ALWAYS as EARLIEST
from graphspec.schema.time import NEVER as LATEST
from graphspec.sources import synthetic
from graphspec.views import view as viewing
from tuning import REALISATIONS, Edited, configuration, reach, reach_over, reach_search
from tuning import describe as variant
from tuning import differences as against
from tuning.bank import forks
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.bank.options import offered
from tuning.bank.routes import routed
from tuning.batches import batching
from tuning.batches.tensors import Host
from tuning.descriptions import differences as diverging
from tuning.descriptions import reach as reaching
from tuning.descriptions import reaches
from tuning.descriptions import restated as restating
from tuning.descriptions import variant as varying
from tuning.families import heads
from tuning.families.options import slots
from tuning.families.periods import blocks as windows
from tuning.statement import VERSION as STATEMENT
from tuning.statement import agreed, state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("descriptions")
NUMBER = dict(type="number", encoder="value")
CHANGED = dict(name="changed", seed=13, repeats=False, parameters=dict(count=1, ranges=dict(speed=[0.25, 0.5])))
HELD = dict(
    near=dict(table="near", ends=[["spot", "spot"]], columns=["weight"], about=None, lapsed=0),
    base=dict(table="base", ends=[["agent", "spot"]], columns=[], about=None, lapsed=0),
)
NONE = dict(least=None, most=None, mean=None)
CLOCKED = dict(model="clock", units=["tick"])
UNHELD = [[None, 0]]
WHOLE_KIND = dict(created=0, unstated=0)
UNREAD = dict(
    later=dict(nodes=0, edges=0, gap=None, last=None),
    watched=dict(nodes=0, edges=0, observers={}),
    sets={},
    held={},
    entities=dict(agent=WHOLE_KIND, spot=WHOLE_KIND),
)
FIRST = dict(
    name="g0",
    unit="tick",
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    kinds=dict(
        agent=dict(
            nodes=2,
            observations=3,
            replaced=0,
            nodes_held=[[None, 0], [0, 2]],
            observations_held=[[None, 0], [0, 2], [3, 3], [4, 2]],
        ),
        spot=dict(
            nodes=3,
            observations=3,
            replaced=0,
            nodes_held=[[None, 0], [0, 2], [2, 3], [5, 2]],
            observations_held=[[None, 0], [0, 2], [2, 3], [5, 2]],
        ),
    ),
    relations=dict(
        near=dict(
            HELD["near"],
            edges=2,
            facts=3,
            absent=1,
            replaced=0,
            edges_held=[[None, 0], [0, 1], [2, 2], [4, 1], [6, 0]],
            runs=2,
            bounded=2,
            least=2,
            most=6,
            mean=4.0,
        ),
        base=dict(
            HELD["base"],
            edges=2,
            facts=3,
            absent=0,
            replaced=0,
            edges_held=[[None, 0], [0, 1], [1, 2], [2, 1], [4, 2], [6, 1]],
            runs=3,
            bounded=2,
            least=1,
            most=2,
            mean=1.5,
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
            rule=None,
            each=[
                dict(key=10, time=1, options=3, recorded=2, regenerated=1, generators=dict(changed=1)),
                dict(key=30, time=3, options=4, recorded=3, regenerated=1, generators=dict(changed=1)),
                dict(key=50, time=5, options=3, recorded=2, regenerated=1, generators=dict(changed=1)),
            ],
        )
    ),
    unread=UNREAD,
)
SECOND = dict(
    name="g1",
    unit="tick",
    times=[1, 2, 3],
    kinds=dict(
        agent=dict(nodes=1, observations=1, replaced=0, nodes_held=[[None, 1]], observations_held=[[None, 1]]),
        spot=dict(
            nodes=2,
            observations=2,
            replaced=0,
            nodes_held=[[None, 1], [1, 2], [3, 1]],
            observations_held=[[None, 1], [1, 2], [3, 1]],
        ),
    ),
    relations=dict(
        near=dict(HELD["near"], ends=[], edges=0, facts=0, absent=0, replaced=0, edges_held=UNHELD, runs=0, bounded=0, **NONE),
        base=dict(
            HELD["base"], edges=1, facts=1, absent=0, replaced=0, edges_held=[[None, 1], [3, 0]], runs=1, bounded=0, **NONE
        ),
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
            rule=None,
            each=[dict(key=10, time=2, options=3, recorded=2, regenerated=1, generators=dict(changed=1))],
        )
    ),
    unread=UNREAD,
)
TEXT = """truth graphs of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/truth
time: one clock, units tick
splits: train, validation
%s

graph g0, unit tick, times 8 from 0 to 7
  kind agent: nodes 2, observations 3, replaced 0
  kind spot: nodes 3, observations 3, replaced 0
  relation near of the table near: ends spot to spot; columns weight; edges 2, facts 3, absent 1, replaced 0; runs 2, bounded 2, lasting 2 to 6, mean 4.0
  relation base of the table base: ends agent to spot; no columns; edges 2, facts 3, absent 0, replaced 0; runs 3, bounded 2, lasting 1 to 2, mean 1.5
  held before every time and at each time:
    time                before  0  1  2  3  4  5  6  7
    agent nodes         0       2  2  2  2  2  2  2  2
    agent observations  0       2  2  2  3  2  2  2  2
    spot nodes          0       2  2  3  3  3  2  2  2
    spot observations   0       2  2  3  3  3  2  2  2
    near edges          0       1  1  2  2  1  1  0  0
    base edges          0       1  2  1  1  2  2  1  1
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 3; train 2, validation 1, no split 0
  kind of task visit: tasks 4; train 2, validation 1, no split 1
  decisions of turn: decisions 3, options 10, recorded 7, regenerated 3; changed 3; no rule of options
    task  time  options  recorded  regenerated  changed
    10    1     3        2         1            1
    30    3     4        3         1            1
    50    5     3        2         1            1

graph g1, unit tick, times 3 from 1 to 3
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 2, observations 2, replaced 0
  relation near of the table near: no ends; columns weight; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  held before every time and at each time:
    time                before  1  2  3
    agent nodes         1       1  1  1
    agent observations  1       1  1  1
    spot nodes          1       2  2  1
    spot observations   1       2  2  1
    near edges          0       0  0  0
    base edges          1       1  1  0
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 1; train 1, validation 0, no split 0
  kind of task visit: tasks 1; train 0, validation 1, no split 0
  decisions of turn: decisions 1, options 3, recorded 2, regenerated 1; changed 1; no rule of options
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
LATER = BASE + ("known", "replaced")


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


@timed("descriptions.kept_as")
def kept_as(part, beside):
    return dict(beside=beside, away=None, folder="descriptions/" + part, file=ANY)


@timed("descriptions.located")
def located(folder, beside, part):
    return describing.located(folder / "descriptions", beside, part, read_json(folder / "manifest.json")["stored_sha256"])


@timed("descriptions.expected")
def expected(stored):
    return dict(
        version=describing.VERSION,
        stored_sha256=stored,
        time=dict(model="clock", units=["tick"]),
        splits=["train", "validation"],
        graphs=[FIRST, SECOND],
        kept=kept_as("truth", "bank"),
    )


@timed("descriptions.files")
def files(bank, stored):
    key = describing.keyed(read_json(bank / "manifest.json"), dict(located(bank, "bank", "truth"), stored=stored))
    stem = bank / "descriptions" / "truth" / describing.named(key)
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
    written = TEXT % (stored, describing.VERSION, describing.READING["clock"])
    assert describing.text(record) == written
    found, shown = files(bank, stored)
    assert read_json(found)["record"] == record and shown.read_text(encoding="utf-8") == written
    assert record["kept"] == dict(beside="bank", away=None, folder="descriptions/truth", file=found.name)
    assert found.read_text(encoding="utf-8") == json.dumps(dict(key=read_json(found)["key"], record=record), separators=(",", ":"))
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
    elsewhere = dict(record["kept"], file=files(other, "0" * 64)[0].name)
    assert describe(other) == dict(record, stored_sha256="0" * 64, graphs=[SECOND, FIRST], kept=elsewhere)
    assert elsewhere["file"] != record["kept"]["file"] and files(other, "0" * 64)[0].exists() and describe(bank) == record


@timed("descriptions.kept_in")
def kept_in(folder):
    return sorted(path.name for path in folder.iterdir())


def test_a_file_is_served_only_where_it_parses_and_holds_its_key_and_the_files_other_code_made_are_removed(
    bank, tmp_path, monkeypatch, caplog
):
    started = time.perf_counter()
    other = tmp_path / "bank"
    shutil.copytree(bank, other, ignore=shutil.ignore_patterns("descriptions"))
    record, key = describe(other), describing.keyed(read_json(other / "manifest.json"), located(other, "bank", "truth"))
    folder = other / "descriptions" / "truth"
    held, text = [(folder / describing.named(key)).with_suffix(suffix) for suffix in (".json", ".txt")]
    whole, written = dict(key=key, record=record), describing.text(record)
    assert read_json(held) == whole and text.read_text(encoding="utf-8") == written
    earlier = dict(key, code=dict(key["code"], description=key["code"]["description"] - 1))
    broken = [
        (json.dumps(whole)[: len(json.dumps(whole)) // 2], "it is no JSON"),
        ("", "it is no JSON"),
        ("{}", "it holds no key and record"),
        (json.dumps(record), "it holds no key and record"),
        (json.dumps(dict(key=dict(key, stored="0" * 64), record=record)), "it holds another key"),
        (json.dumps(dict(key=earlier, record=dict(record, graphs=[]))), "it holds another key"),
    ]
    for content, why in broken:
        held.write_text(content, encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=describing.LOG.name):
            assert describe(other) == record
        assert [(str(held) in message, why in message) for message in caplog.messages] == [(True, True)]
        assert read_json(held) == whole
    text.unlink()
    monkeypatch.setattr(describing, "recorded", refused)
    assert describe(other) == record and text.read_text(encoding="utf-8") == written
    monkeypatch.undo()
    code, marked = key["code"], held.name.split("-")[0] + ".json"
    mine = sorted([marked, held.name, text.name])
    version = code["description"]
    older = ["0" * 64 + ".json", "0" * 64 + ".txt"]
    moved = ["a" * 16 + ".json", "a" * 16 + "-" + "1" * 64 + ".json", "a" * 16 + "-" + "1" * 64 + ".txt"]
    absent = ["b" * 16 + ".json", "b" * 16 + "-" + "2" * 64 + ".json"]
    lone = ["c" * 16 + "-" + "3" * 64 + ".txt"]
    assert kept_in(folder) == mine and read_json(folder / marked) == code
    for name in older + moved[1:] + absent[1:] + lone:
        (folder / name).write_text("{}", encoding="utf-8")
    later = dict(code, description=version + 1)
    write_json(folder / moved[0], dict(later, entries=code["entries"] + [["encoder", "standard", 99]]))
    write_json(folder / absent[0], dict(later, entries=code["entries"] + [["operation", "test.absent", 1]]))
    between(other, "g0", "g1")
    apart = kept_in(other / "descriptions" / "differences")
    assert describe(other) == record and kept_in(folder) == sorted(mine + older + moved + absent + lone)
    caplog.clear()
    with raised(describing), caplog.at_level(logging.INFO, logger=describing.LOG.name):
        later_key = describing.keyed(read_json(other / "manifest.json"), located(other, "bank", "truth"))
        later_kept = dict(record["kept"], file=describing.named(later_key) + ".json")
        assert describe(other) == dict(record, version=version + 1, kept=later_kept) and later_kept != record["kept"]
        newer = kept_in(folder)
        assert len(newer) == 5 and set(absent) < set(newer) and not set(newer) & set(mine + older + moved + lone)
    removed = [message for message in caplog.messages if "removed" in message]
    assert len(removed) == 4 and all(str(folder) in message for message in removed)
    for reason in (
        "the encoder standard at version 99, now %d" % registries.version(ENCODERS.named("standard")),
        "description at version %d, now %d" % (version, version + 1),
        "it holds no record of the code that made it",
        "its record of the code that made it is not one this code writes",
    ):
        assert sum(reason in message for message in removed) == 1, reason
    assert describe(other) == record and kept_in(folder) == mine
    assert kept_in(other / "descriptions" / "differences") == apart and len(apart) == 3
    LOG.info("descriptions served and removed exit seconds=%.3f", time.perf_counter() - started)


MANIFEST = "the manifest.json of the bank %s lacks %s; a bank holds the manifest its build wrote"
UNFOLDERED = "%s is a file; descriptions are kept in a folder of that name"


def test_a_bank_or_a_file_lacking_a_part_stops_the_call_naming_the_file_and_the_part(bank, tmp_path, capsys, monkeypatch):
    from tuning.__main__ import main

    started = time.perf_counter()
    other = tmp_path / "bank"
    shutil.copytree(bank, other, ignore=shutil.ignore_patterns("descriptions"))
    manifest = read_json(other / "manifest.json")
    config = configuration(bank, MODEL, None, pipeline=[], graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    calls = (lambda: describe(other), lambda: between(other, "g0", "g1"), lambda: variant(other, config, STATED))
    for lacking in ("version", "splits", "graphs", "spec", "stored_sha256", "spec_sha256", "trained"):
        write_json(other / "manifest.json", without(manifest, lacking))
        for call in calls:
            with pytest.raises(ValueError) as raised:
                call()
            assert str(raised.value) == MANIFEST % (other, lacking)
    write_json(other / "manifest.json", {})
    with pytest.raises(ValueError, match="lacks version, trained, spec, spec_sha256, splits, split_rows, observers, graphs, "):
        describe(other)
    (other / "manifest.json").write_text("this is no json", encoding="utf-8")
    for call in calls:
        with pytest.raises(ValueError) as raised:
            call()
        assert str(raised.value) == "%s is no JSON: Expecting value: line 1 column 1 (char 0)" % (other / "manifest.json")
    write_json(other / "manifest.json", manifest)
    (other / "descriptions").write_text("not a folder", encoding="utf-8")
    for call in calls:
        with pytest.raises(NotADirectoryError) as raised:
            call()
        assert str(raised.value) == UNFOLDERED % (other / "descriptions")
    (other / "descriptions").unlink()
    (other / "supervision.json").rename(other / "held.json")
    with pytest.raises(FileNotFoundError) as raised:
        variant(other, config, STATED)
    assert raised.value.filename == str(other / "supervision.json") and repr(str(other / "supervision.json")) in str(raised.value)
    (other / "held.json").rename(other / "supervision.json")
    shutil.rmtree(other / "truth" / "graph_0001")
    for call in calls[:2]:
        with pytest.raises(FileNotFoundError) as raised:
            call()
        assert raised.value.filename == str(other / "truth" / "graph_0001" / "graph.json")
        assert repr(str(other / "truth" / "graph_0001" / "graph.json")) in str(raised.value)
    assert not (other / "descriptions").exists()
    with pytest.raises(ValueError) as raised:
        between(bank, "g0", "g9")
    assert type(raised.value) is ValueError
    assert str(raised.value) == "the bank %s holds no graph named g9; its graphs are g0, g1" % bank
    (tmp_path / "not.json").write_text("{", encoding="utf-8")
    write_json(tmp_path / "statement.json", STATED)
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", str(bank), str(tmp_path / "not.json"), str(tmp_path / "statement.json")])
    with pytest.raises(ValueError, match="^%s is no JSON: " % re.escape(str(tmp_path / "not.json"))):
        main()
    write_json(tmp_path / "plan.json", dict(bank_stored_sha256=manifest["stored_sha256"], trials=[dict(folder="trial_000000")]))
    with pytest.raises(ValueError, match="^the plan.json of .* lists a trial that names no directory: "):
        reach_search(bank, tmp_path)
    LOG.info("descriptions lacking parts exit seconds=%.3f", time.perf_counter() - started)


AWAY = (
    "kept under the runtime home, in its folder cache/descriptions/%s/%s: the library writes nothing beside the %s, "
    "which is %s"
)
INSIDE, ELSEWHERE = "inside a code checkout", "on another volume than the runtime home"


def test_a_bank_the_library_may_not_write_beside_is_described_under_the_home_and_every_text_says_where_it_is_kept(
    bank, tmp_path, monkeypatch
):
    started = time.perf_counter()
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    (tmp_path / "repository" / ".git").mkdir(parents=True)
    other = tmp_path / "repository" / "bank"
    shutil.copytree(bank, other, ignore=shutil.ignore_patterns("descriptions"))
    assert runtime.refusal(bank / "descriptions") is None and runtime.refusal(other / "descriptions") == "code"
    config = configuration(bank, MODEL, None, pipeline=[], graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    (tmp_path / "repository" / "search").mkdir()
    planned = json.dumps(dict(bank_stored_sha256=stored, trials=[]))
    (tmp_path / "repository" / "search" / "plan.json").write_text(planned, encoding="utf-8")
    described = [
        ("truth", "bank", lambda: describe(other), describing.text),
        ("differences", "bank", lambda: between(other, "g0", "g1"), differing.text),
        ("variants", "bank", lambda: variant(other, config, STATED), effect.text),
        ("reaches", "bank", lambda: reach_over(other, [], STATED), reaches.text),
        ("reaches", "search", lambda: reach_search(bank, tmp_path / "repository" / "search"), reaches.text),
    ]
    home = cache_path("descriptions", stored)
    for part, beside, call, text in described:
        record = call()
        assert text(record).split("\n")[1] == AWAY % (stored, part, beside, INSIDE)
        written = [path for path in (home / part).glob("*-*.txt") if path.read_text(encoding="utf-8") == text(record)]
        assert len(written) == 1 and read_json(written[0].with_suffix(".json"))["record"] == record
        folder = "cache/descriptions/%s/%s" % (stored, part)
        assert record["kept"] == dict(beside=beside, away="code", folder=folder, file=written[0].with_suffix(".json").name)
        assert call() == record
    model = dict(MODEL, **{"options/realisation": "vector"})
    aimed = configuration(bank, model, None, pipeline=[], graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    undone = [
        ("reach", lambda: reach(other, aimed, STATED)),
        ("differences", lambda: against(other, config, STATED, config, STATED)),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(describing, "kept", lambda where, key, made, written: (where, key))
        for part, call in undone:
            where, key = call()
            assert (where["folder"], where["beside"], where["away"]) == (home / part, "bank", "code")
            assert {name: key[name] for name in describing.BASED} == describing.based(where)
    assert reaches.row_placed(other, stored)["folder"] == home / "reach_rows"
    assert sorted(path.name for path in home.iterdir()) == ["differences", "reaches", "truth", "variants"]
    assert kept_in(tmp_path / "repository") == [".git", "bank", "search"] and kept_in(tmp_path / "repository" / "search") == ["plan.json"]
    beside = describe(bank)
    assert kept_in(other) == kept_in(bank)[1:] and kept_in(bank)[0] == "descriptions"
    assert beside["kept"] == dict(beside="bank", away=None, folder="descriptions/truth", file=files(bank, stored)[0].name)
    assert describing.text(beside).split("\n")[1] == "kept beside the bank, in its folder descriptions/truth"
    assert without(describe(other), "kept") == without(beside, "kept")
    far = dict(beside, kept=dict(beside["kept"], away="volume", folder="cache/descriptions/%s/truth" % stored))
    assert describing.text(far).split("\n")[1] == AWAY % (stored, "truth", "bank", ELSEWHERE)
    if os.name == "nt":
        monkeypatch.setattr(runtime, "configured_home", lambda: Path("Q:/home"))
        assert runtime.refusal(bank / "descriptions") == "volume"
        with pytest.raises(ValueError, match="^runtime output must use the GRAPHSPEC_HOME volume: "):
            runtime.output_path(bank / "descriptions")
    LOG.info("descriptions kept under the home exit seconds=%.3f", time.perf_counter() - started)


BEYOND = dict(g0="图一", near="喜欢", base="repère")


@timed("descriptions.renamed")
def renamed(record):
    graphs = [
        dict(
            graph,
            name=BEYOND.get(graph["name"], graph["name"]),
            relations={BEYOND.get(name, name): held for name, held in graph["relations"].items()},
        )
        for graph in record["graphs"]
    ]
    return dict(record, graphs=graphs)


def piped(call, monkeypatch, encoding):
    started = time.perf_counter()
    reading, writing = os.pipe()
    stream = io.TextIOWrapper(os.fdopen(writing, "wb"), encoding=encoding)
    with ThreadPoolExecutor(1) as pool, os.fdopen(reading, "rb") as source, monkeypatch.context() as patch:
        found = pool.submit(source.read)
        patch.setattr(sys, "stdout", stream)
        try:
            call()
        finally:
            stream.close()
            result = found.result()
    LOG.debug("descriptions piped exit bytes=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def test_the_verb_writes_names_beyond_ascii_through_a_pipe_as_utf_8_the_bytes_of_its_file_whatever_the_encoding_of_the_pipe(
    bank, tmp_path, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    record = renamed(describe(bank))
    written = describing.text(record)
    assert all(written.count(name) for name in BEYOND.values())
    with pytest.raises(UnicodeEncodeError):
        written.encode("cp1252")
    (tmp_path / "kept.txt").write_text(written, encoding="utf-8")
    monkeypatch.setattr(describing, "describe", lambda path: record)
    for encoding in ("cp1252", "ascii", "utf-8"):
        monkeypatch.setattr(sys, "argv", ["graphspec", "describe", str(bank)])
        sent = piped(main, monkeypatch, encoding)
        assert sent == (tmp_path / "kept.txt").read_bytes()
        assert sent.decode("utf-8").replace(os.linesep, "\n") == written
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", str(bank)])
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    main()
    assert sys.stdout.getvalue() == written
    LOG.info("descriptions piped verb exit seconds=%.3f", time.perf_counter() - started)


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
NIL = [[0, 0]]
SPOTS = [[0, 2], [2, 3], [5, 2]]
NEARS = [[0, 1], [2, 2], [4, 1], [6, 0]]
BASES = [[0, 1], [1, 2], [2, 1], [4, 2], [6, 1]]
BASES_CUT = [[0, 1], [1, 0], [2, 1], [3, 0], [4, 2], [5, 0], [6, 1]]
LOOPS_HELD = [[0, 3], [1, 4], [4, 5], [5, 4], [6, 3]]


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
HEAVY = "test.lean requires 10000000000 candidate edges; the remaining graph budget permits 200000000"
LONELY = "leans on the entry test.gate"
LEANING = "the entry test.leans of the pipeline states leans/alone never, which " + LONELY


@timed("descriptions.inclined")
def inclined(registry):
    return (choice("lean/mode", ("cut", "heavy"), "cut"), choice("lean/kind", registry.tasks, registry.tasks[0]))


@timed("descriptions.leant")
def leant(graph, registry, settings, seed, limits, carry, held):
    if settings["lean/mode"] == "heavy":
        raise GraphCapacityError(HEAVY)
    if settings["cut/times"] == "tasks":
        return graph
    return unheld(graph, registry, {"cut/relation": "base", "cut/kind": settings["lean/kind"]}, seed, limits, carry, held)


@timed("descriptions.gating")
def gating(registry):
    return (number("gate/level", 1, 2, 2),)


@timed("descriptions.gated")
def gated(graph, registry, settings, seed, limits, carry, held):
    if settings["gate/level"] == 1:
        return graph
    return unheld(graph, registry, CUTTING, seed, limits, carry, held)


@timed("descriptions.lonely")
def lonely(registry):
    return (choice("leans/alone", ("never",), "never", lambda spec: "gate/level" not in spec),)


@timed("descriptions.unleant")
def unleant(registry, value):
    return LONELY


LEAN = Operation("test", "lean", (Phase("lean", leant, None, None),), inclined, ("test.part",), (), (), ("facts",), None, None, 1, 1)
GATE = Operation("test", "gate", (Phase("gate", gated, None, None),), gating, (), (), (), ("facts",), None, None, 1, 1)
LEANS = Operation(
    "test", "leans", (Phase("leans", passed, None, None),), lonely, ("test.gate",), (), (Need("leans/alone", unleant),), (), None, None, 1, 1
)
@timed("descriptions.lifting_kind")
def lifting_kind(registry):
    return (choice("lift/kind", registry.tasks, registry.tasks[0]),)


def lifted(graph, registry, settings, seed, limits, carry, held):
    started = time.perf_counter()
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    tasked = np.asarray(graph.task_step)[np.asarray(graph.task_kind) == registry.tasks.index(settings["lift/kind"])]
    chosen = np.flatnonzero((np.asarray(graph.kinds)[entity] == registry.kind("spot")) & np.isin(step, tasked))
    reference = np.asarray(graph.obs_ref, np.int64)
    rows = np.array(np.asarray(graph.x_table)[reference[chosen]], np.float32)
    rows[:, registry.encoding.slots("load")[0][0]] += 1.0
    graph, made = made_rows(graph, carry, rows.astype(graph.x_table.dtype))
    reference = reference.copy()
    reference[chosen] = made
    LOG.debug("descriptions lifted exit nodes=%d seconds=%.4f", len(chosen), time.perf_counter() - started)
    return graph.replace(obs_ref=reference.astype(graph.obs_ref.dtype))


LIFT = Operation(
    "test", "lift", (Phase("lift", lifted, None, None),), lifting_kind, (), (), (), ("observations",), None, None, 1, 1
)
CUTTERS = (CUT, SEVER, PART, READ, LEAN, GATE, LEANS, LIFT)


@pytest.fixture
def cutting():
    started = time.perf_counter()
    for held in CUTTERS:
        OPERATIONS.register(held.name, held)
    LOG.debug("descriptions cutting registered seconds=%.4f", time.perf_counter() - started)
    yield CUT
    for held in CUTTERS:
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
    return dict(
        kinds=dict(agent=dict(nodes=agent, anchors=NIL), spot=dict(nodes=spot, anchors=NIL)),
        relations={"near": dict(edges=near), "base": dict(edges=base), LOOPS: dict(edges=loops)},
    )


UNVALUED = dict(kinds={}, relations={})


@timed("descriptions.effected")
def effected(entry, kinds, relations, cuts):
    return dict(entry=entry, operation=entry, kinds=kinds, relations=relations, values=UNVALUED, cuts=cuts)


EVERY = ["train", "validation"]
VIEWED = {name: dict(graph=name, observer=None, splits=EVERY) for name in ("g0", "g1", "h0", "w0")}
SEEN_BY = "the graph %s as the tasks of the splits train, validation see it"
UNSHOWN = "graphs of the bank the view does not show: none"
VARIED = dict(
    name="g0",
    view=VIEWED["g0"],
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    tasks=dict(turn=[1, 3, 5], visit=[2, 4, 5, 6]),
    held=holding([[0, 2]], SPOTS, NEARS, BASES, NIL),
    effects=[
        effected(
            "graphspec.nodes",
            dict(
                agent=tally(16, 5, 0.3125, 0, 0.0, 11, nodes=BASES, anchors=NIL),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=SPOTS, anchors=NIL),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=NEARS),
                "base": tally(11, 0, 0.0, 0, 0.0, 11, edges=BASES),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=NIL),
            },
            [],
        ),
        effected(
            "test.cut",
            dict(
                agent=tally(11, 0, 0.0, 0, 0.0, 11, nodes=BASES, anchors=NIL),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=SPOTS, anchors=NIL),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=NEARS),
                "base": tally(11, 5, 0.4545, 0, 0.0, 6, edges=BASES_CUT),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=NIL),
            },
            [dict(sort="relation", name="base", task="turn", times=[1, 3, 5], everywhere=False, without=[])],
        ),
        effected(
            "graphspec.relation_forms",
            dict(
                agent=tally(11, 0, 0.0, 0, 0.0, 11, nodes=BASES, anchors=NIL),
                spot=tally(19, 0, 0.0, 0, 0.0, 19, nodes=SPOTS, anchors=NIL),
            ),
            {
                "near": tally(8, 0, 0.0, 0, 0.0, 8, edges=NEARS),
                "base": tally(6, 0, 0.0, 0, 0.0, 6, edges=BASES_CUT),
                LOOPS: tally(0, 0, None, 30, None, 30, edges=LOOPS_HELD),
            },
            [],
        ),
    ],
)
VARIED_SECOND = dict(
    name="g1",
    view=VIEWED["g1"],
    times=[1, 2, 3],
    tasks=dict(turn=[2], visit=[1]),
    held=holding([[0, 1]], [[0, 2], [2, 1]], NIL, [[0, 1], [2, 0]], NIL),
    effects=[
        effected(
            "graphspec.nodes",
            dict(
                agent=tally(3, 1, 0.3333, 0, 0.0, 2, nodes=[[0, 1], [2, 0]], anchors=NIL),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[[0, 2], [2, 1]], anchors=NIL),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=NIL),
                "base": tally(2, 0, 0.0, 0, 0.0, 2, edges=[[0, 1], [2, 0]]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=NIL),
            },
            [],
        ),
        effected(
            "test.cut",
            dict(
                agent=tally(2, 0, 0.0, 0, 0.0, 2, nodes=[[0, 1], [2, 0]], anchors=NIL),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[[0, 2], [2, 1]], anchors=NIL),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=NIL),
                "base": tally(2, 1, 0.5, 0, 0.0, 1, edges=[[0, 1], [1, 0]]),
                LOOPS: tally(0, 0, None, 0, None, 0, edges=NIL),
            },
            [dict(sort="relation", name="base", task="turn", times=[2], everywhere=False, without=[])],
        ),
        effected(
            "graphspec.relation_forms",
            dict(
                agent=tally(2, 0, 0.0, 0, 0.0, 2, nodes=[[0, 1], [2, 0]], anchors=NIL),
                spot=tally(5, 0, 0.0, 0, 0.0, 5, nodes=[[0, 2], [2, 1]], anchors=NIL),
            ),
            {
                "near": tally(0, 0, None, 0, None, 0, edges=NIL),
                "base": tally(1, 0, 0.0, 0, 0.0, 1, edges=[[0, 1], [1, 0]]),
                LOOPS: tally(0, 0, None, 7, None, 7, edges=[[0, 3], [2, 1]]),
            },
            [],
        ),
    ],
)
EFFECT = "    held                        before  removed  share    added  share    after"
VARIANT = """variant %s of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/variants
derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
graphs of the bank the view does not show: none
%s
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree 1, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none

graph g0, the graph g0 as the tasks of the splits train, validation see it, 8 steps from time 0 to 7
  tasks: turn at 1, 3, 5; visit at 2, 4, 5, 6
  held at each time before the first entry:
    time                        0  1  2  3  4  5  6  7
    agent nodes                 2  2  2  2  2  2  2  2
    agent anchors               0  0  0  0  0  0  0  0
    spot nodes                  2  2  3  3  3  2  2  2
    spot anchors                0  0  0  0  0  0  0  0
    near edges                  1  1  2  2  1  1  0  0
    base edges                  1  2  1  1  2  2  1  1
    relation_forms.loops edges  0  0  0  0  0  0  0  0
  entry graphspec.nodes, of the operation graphspec.nodes:
%(effect)s
    agent nodes                 16      5        0.3125   0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  11      0        0.0      0      0.0      11
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        0  1  2  3  4  5  6  7
    agent nodes                 1  2  1  1  2  2  1  1
    agent anchors               0  0  0  0  0  0  0  0
    spot nodes                  2  2  3  3  3  2  2  2
    spot anchors                0  0  0  0  0  0  0  0
    near edges                  1  1  2  2  1  1  0  0
    base edges                  1  2  1  1  2  2  1  1
    relation_forms.loops edges  0  0  0  0  0  0  0  0
  entry test.cut, of the operation test.cut:
%(effect)s
    agent nodes                 11      0        0.0      0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  11      5        0.4545   0      0.0      6
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        0  1  2  3  4  5  6  7
    agent nodes                 1  2  1  1  2  2  1  1
    agent anchors               0  0  0  0  0  0  0  0
    spot nodes                  2  2  3  3  3  2  2  2
    spot anchors                0  0  0  0  0  0  0  0
    near edges                  1  1  2  2  1  1  0  0
    base edges                  1  0  1  0  2  0  1  1
    relation_forms.loops edges  0  0  0  0  0  0  0  0
  entry test.cut, of the operation test.cut, removes every edge of the relation base at the times of the tasks of turn: 1, 3, 5; not at every time
  entry graphspec.relation_forms, of the operation graphspec.relation_forms:
%(effect)s
    agent nodes                 11      0        0.0      0      0.0      11
    spot nodes                  19      0        0.0      0      0.0      19
    near edges                  8       0        0.0      0      0.0      8
    base edges                  6       0        0.0      0      0.0      6
    relation_forms.loops edges  0       0        nothing  30     nothing  30
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        0  1  2  3  4  5  6  7
    agent nodes                 1  2  1  1  2  2  1  1
    agent anchors               0  0  0  0  0  0  0  0
    spot nodes                  2  2  3  3  3  2  2  2
    spot anchors                0  0  0  0  0  0  0  0
    near edges                  1  1  2  2  1  1  0  0
    base edges                  1  0  1  0  2  0  1  1
    relation_forms.loops edges  3  4  4  4  5  4  3  3

graph g1, the graph g1 as the tasks of the splits train, validation see it, 3 steps from time 1 to 3
  tasks: turn at 2; visit at 1
  held at each time before the first entry:
    time                        1  2  3
    agent nodes                 1  1  1
    agent anchors               0  0  0
    spot nodes                  2  2  1
    spot anchors                0  0  0
    near edges                  0  0  0
    base edges                  1  1  0
    relation_forms.loops edges  0  0  0
  entry graphspec.nodes, of the operation graphspec.nodes:
%(effect)s
    agent nodes                 3       1        0.3333   0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  2       0        0.0      0      0.0      2
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        1  2  3
    agent nodes                 1  1  0
    agent anchors               0  0  0
    spot nodes                  2  2  1
    spot anchors                0  0  0
    near edges                  0  0  0
    base edges                  1  1  0
    relation_forms.loops edges  0  0  0
  entry test.cut, of the operation test.cut:
%(effect)s
    agent nodes                 2       0        0.0      0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  2       1        0.5      0      0.0      1
    relation_forms.loops edges  0       0        nothing  0      nothing  0
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        1  2  3
    agent nodes                 1  1  0
    agent anchors               0  0  0
    spot nodes                  2  2  1
    spot anchors                0  0  0
    near edges                  0  0  0
    base edges                  1  0  0
    relation_forms.loops edges  0  0  0
  entry test.cut, of the operation test.cut, removes every edge of the relation base at the times of the tasks of turn: 2; not at every time
  entry graphspec.relation_forms, of the operation graphspec.relation_forms:
%(effect)s
    agent nodes                 2       0        0.0      0      0.0      2
    spot nodes                  5       0        0.0      0      0.0      5
    near edges                  0       0        nothing  0      nothing  0
    base edges                  1       0        0.0      0      0.0      1
    relation_forms.loops edges  0       0        nothing  7      nothing  7
  columns of values and values of nodes: none added, removed, filled or changed
  held at each time after it:
    time                        1  2  3
    agent nodes                 1  1  0
    agent anchors               0  0  0
    spot nodes                  2  2  1
    spot anchors                0  0  0
    near edges                  0  0  0
    base edges                  1  0  0
    relation_forms.loops edges  3  3  1
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


def named(bank, config, part, name, refused):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    key, place = specs.digest(deriver.identity(config)), located(bank, "bank", part)
    if part == "variants":
        held = varying.keyed(deriver, config, key, place)
    elif part == "reach":
        held = reaching.keyed(deriver, config, key, name, refused, place)
    else:
        held = reaches.row_keyed(deriver, config, key, name, place)
    LOG.debug("descriptions named exit part=%s seconds=%.4f", part, time.perf_counter() - started)
    return held


@timed("descriptions.stemmed")
def stemmed(bank, config, part, name, refused):
    return bank / "descriptions" / part / describing.named(named(bank, config, part, name, refused))


@timed("descriptions.varied")
def varied(bank, config):
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    key = keyed(bank, config)
    record = dict(
        version=describing.VERSION,
        variant=key,
        stored_sha256=stored,
        time=CLOCKED,
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=SETTLED,
        unshown=[],
        entries=entries("test.cut", CUTTING),
        graphs=[VARIED, VARIED_SECOND],
        kept=kept_as("variants", "bank"),
    )
    written = VARIANT.replace("%(effect)s", EFFECT) % (
        key,
        stored,
        describing.VERSION,
        effect.READING,
        OPERATIONS.named("graphspec.nodes").version,
        OPERATIONS.named("graphspec.relation_forms").version,
    )
    return record, written, stemmed(bank, config, "variants", None, None)


def test_every_count_and_share_of_a_variant_equals_the_number_written_out_here_and_names_the_entry_that_cuts(bank, cutting):
    started = time.perf_counter()
    config = configured(bank, CUTTING)
    record = variant(bank, config, STATED)
    assert record == varied(bank, config)[0]
    assert json.loads(json.dumps(record)) == record
    cut = [found for graph in record["graphs"] for held in graph["effects"] for found in held["cuts"]]
    assert [held["entry"] for graph in record["graphs"] for held in graph["effects"] if held["cuts"]] == ["test.cut"] * 2
    assert cut == [
        dict(sort="relation", name="base", task="turn", times=[1, 3, 5], everywhere=False, without=[]),
        dict(sort="relation", name="base", task="turn", times=[2], everywhere=False, without=[]),
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
    assert read_json(stem.with_suffix(".json"))["record"] == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    assert record["kept"]["file"] == stem.with_suffix(".json").name
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


PRODUCED = {"product/path": "near:both:weight", "product/length": 1, "product/aggregate": "product", "product/encoder": "value"}
MEASURED = {"measures/degree": True, "measures/components": False, "measures/cores": False, "measures/encoder": "value"}
VALUING = [
    dict(operation="graphspec.product", parameters=PRODUCED),
    dict(operation="graphspec.measures", parameters=MEASURED),
    dict(operation="test.lift", parameters={"lift/kind": "turn"}),
]


def test_a_relation_an_entry_adds_is_listed_in_a_graph_where_it_holds_no_edge_and_a_joined_group_is_described_as_its_graphs_apart(
    bank, cutting
):
    started = time.perf_counter()
    assert [held["table"] for held in case()["graphs"][1]["edges"]] == ["base"]
    found = []
    for graph in (SETTINGS, dict(SETTINGS, **JOINED)):
        config = configuration(bank, MODEL, None, pipeline=VALUING, graph=graph, seed=STATED["seed"], splits=SPLITS)
        found.append(variant(bank, config, STATED))
    apart, joined = found
    assert (apart["settings"]["graphs/join"], joined["settings"]["graphs/join"]) == ("separate", "single")
    assert [list(graph["held"]["kinds"]) for graph in apart["graphs"]] == [["agent", "spot"]] * 2
    assert [list(graph["held"]["relations"]) for graph in apart["graphs"]] == [["near", "base", "product"]] * 2
    made = [graph["effects"][0]["relations"]["product"] for graph in apart["graphs"]]
    assert made == [tally(0, 0, None, 8, None, 8, edges=NEARS), tally(0, 0, None, 0, None, 0, edges=NIL)]
    assert joined["graphs"] == apart["graphs"]
    degree = dict(added=["measures.degree"], removed=[], filled=["measures.degree"])
    valued = dict(added=["product.value"], removed=[], filled=["product.value"])
    assert [[held["values"] for held in graph["effects"]] for graph in apart["graphs"]] == [
        [
            dict(kinds={}, relations=dict(product=valued)),
            dict(kinds=dict(agent=dict(degree, changed=16), spot=dict(degree, changed=19)), relations={}),
            dict(kinds=dict(spot=dict(added=[], removed=[], filled=[], changed=7)), relations={}),
        ],
        [
            dict(kinds={}, relations=dict(product=dict(valued, filled=[]))),
            dict(kinds=dict(agent=dict(degree, changed=3), spot=dict(degree, changed=5)), relations={}),
            dict(kinds=dict(spot=dict(added=[], removed=[], filled=[], changed=2)), relations={}),
        ],
    ]
    written = effect.text(apart)
    assert [line for line in written.split("\n") if line.startswith("  columns of values")] == [
        "  columns of values and values of nodes: relation product adds product.value, fills product.value",
        "  columns of values and values of nodes: kind agent adds measures.degree, fills measures.degree, changes the "
        "values of 16 nodes; kind spot adds measures.degree, fills measures.degree, changes the values of 19 nodes",
        "  columns of values and values of nodes: kind spot changes the values of 7 nodes",
        "  columns of values and values of nodes: relation product adds product.value",
        "  columns of values and values of nodes: kind agent adds measures.degree, fills measures.degree, changes the "
        "values of 3 nodes; kind spot adds measures.degree, fills measures.degree, changes the values of 5 nodes",
        "  columns of values and values of nodes: kind spot changes the values of 2 nodes",
    ]
    LOG.info("descriptions variant added relation exit seconds=%.3f", time.perf_counter() - started)


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
GROUND = dict(STEPS_OF_GO, to=dict(relation="base", actor="source"))
GROUNDED = Edited(dict(turn=dict(go=GROUND)), WENT_ADDED, 1)
LOAD = ["load"]
UNDIRECTED = dict(along=None, anchor=False)
APART = dict(UNDIRECTED, state="unjoined", hops=None, relations=None)
CREATED = dict(UNDIRECTED, state="created", hops=None, relations=None)
ADJACENT = dict(UNDIRECTED, state="joined", hops=1, relations=[["base"]])
WENT = dict(UNDIRECTED, state="joined", hops=1, relations=[["went"]])
FARTHER = dict(UNDIRECTED, state="joined", hops=2, relations=[["base"], ["near"]])
SKIRTED = dict(UNDIRECTED, state="joined", hops=2, relations=[["went"], ["near"]])
EDGE = "a1 base s1"
SEVERING = {"cut/relation": "base", "cut/kind": "visit"}
PARTING = {"cut/relation": "base", "cut/times": "all"}
REFUSING = "the entry %s of the pipeline states cut/kind turn, which " + UNDECIDED
REFUSAL = REFUSING % "test.sever"
RESTATED = "    %s, its parameters re-stated: statements giving it back %d, not giving it back %d, refused %d, stopped %d, not tried %d"
GIVEN = "      given back by %s"
CUT_BY = "  cut %d, of being %s: by the entry %s"
CUT_BY_NONE = "  cut %d, of being %s: by no entry, %s in no state, the view before the first entry included"
UNJOINED = "    the truth joins it to no reference at the time of the task"
A_PATH, A_FIT, THE_MODEL = "joined to a reference", "held by a fit", "put before the model"
NEAR_OR_VISIT = "cut/relation near, cut/kind visit"
SEVERED_TEXT = dict(
    g0=[
        CUT_BY % (1, A_PATH, "test.cut"),
        RESTATED % ("test.cut", 2, 0, 0, 0, 0),
        GIVEN % NEAR_OR_VISIT,
        CUT_BY % (2, A_FIT, "test.cut"),
        RESTATED % ("test.cut", 2, 0, 0, 0, 0),
        GIVEN % NEAR_OR_VISIT,
        CUT_BY % (3, A_PATH, "test.cut"),
        RESTATED % ("test.cut", 1, 1, 0, 0, 0),
        GIVEN % "cut/relation near",
    ],
    g1=[
        CUT_BY % (1, A_PATH, "test.cut"),
        RESTATED % ("test.cut", 2, 0, 0, 0, 0),
        GIVEN % NEAR_OR_VISIT,
        CUT_BY_NONE % (2, A_PATH, A_PATH),
        UNJOINED,
    ],
)
HELD_TEXT = """    option %(key)s, held by a fit
      target to of go: put before the model as the targets the variant holds
        %(target)s: joined to a reference by no path, features load; %(cut)s
      parameter speed of go: put before the model with the option"""
LEFT_TEXT = (
    "    option o1, left out by a fit, lacking a1 base s1, target along of go: no present fact of its relation joins its "
    "ends at the step; cut 2, by test.cut; nothing is put before the model of the fields it states, target to of go, "
    "target along of go, target made of go, parameter speed of go"
)
REFUSED_TEXT = [CUT_BY % (1, A_PATH, "test.sever"), RESTATED % ("test.sever", 1, 0, 1, 0, 0), GIVEN % "cut/relation near"]
PARTED_TEXT = [
    CUT_BY % (1, A_PATH, "test.part"),
    RESTATED % ("test.part", 2, 0, 1, 0, 0),
    GIVEN % "cut/relation near, cut/times tasks with cut/kind visit",
    CUT_BY % (2, A_FIT, "test.part"),
    RESTATED % ("test.part", 2, 0, 1, 0, 0),
    GIVEN % "cut/relation near, cut/times tasks with cut/kind visit",
    CUT_BY % (3, A_PATH, "test.part"),
    RESTATED % ("test.part", 1, 1, 1, 0, 0),
    GIVEN % "cut/relation near",
]
UNTRIED_TEXT = [
    CUT_BY % (1, A_PATH, "test.part"),
    RESTATED % ("test.part", 1, 0, 1, 0, 1),
    GIVEN % "cut/relation near",
    "      not tried past the first 1 statements of cut/times tasks",
]
EVERY_FIELD = ["kind", "targets", "parameters"]
WENT_STATED = dict(went=dict(pairs=[["agent", "spot"], ["agent", "agent"]], directed=True, columns=["speed"]))
BESIDE = dict(reads=EVERY_FIELD, back=False, structure=False, actor=False, mapping=None, relations=None)
EDITING = dict(reads=EVERY_FIELD, back=True, structure=False, actor=True, relations=WENT_STATED)
REALISED = dict(
    vector=BESIDE,
    aimed=dict(BESIDE, reads=["targets"]),
    actor_edge=dict(BESIDE, structure=True, actor=True),
    walked=dict(EDITING, mapping=dict(turn=dict(go=STEPS_OF_GO))),
    stepped=dict(EDITING, mapping=dict(turn=dict(go=STEPS_OF_GO, stay=dict(by=STEP)))),
)
DIRECTED = (
    "the field target to of go of the kind of task turn has no route to the prediction under the realisation of options "
    "walked: an edit makes it facts of the relation went, which has a direction, from the actor, and the setting "
    "relations/direction directed reads a relation from its source to its target alone, so the actor reads none of them"
)
WALKED_TEXT = [
    "realisation of options walked at version 1, reading kind, targets, parameters; each option an edit of the graph, "
    "read in its own fork",
    "the edit makes to of go of turn a fact of went from the actor; along of go of turn a fact of went from the actor; "
    "made of go of turn a fact of went from the actor; it adds the relations went from agent to spot or agent to agent, "
    "directed, with speed",
]
GROUND_CUTS = [
    CUT_BY % (1, A_FIT, "test.cut"),
    RESTATED % ("test.cut", 2, 0, 0, 0, 0),
    GIVEN % NEAR_OR_VISIT,
    CUT_BY % (2, THE_MODEL, "test.cut"),
    RESTATED % ("test.cut", 2, 0, 0, 0, 0),
    GIVEN % NEAR_OR_VISIT,
]
GROUNDED_TEXT = """    option o0, left out by a fit, lacking a0 base s0 of s0, target to of go: not in its fork; cut 1, by test.cut; nothing is put before the model of the fields it states, target to of go, parameter speed of go
    option o1, held by a fit
      target to of go: not put before the model, its fork holding none of the facts its edit makes of it
        s0: fact a0 base s0 not in its fork; joined to a reference at hop 2 by went then near and by no path along the direction of its facts, features load; cut 2, by test.cut
        s1: fact a0 base s1 not in its fork; joined to a reference at hop 1 by went and by no path along the direction of its facts, features load; cut 2, by test.cut
      target along of go: put before the model as the facts its edit makes of it
        a1 base s1: facts a0 went a1, a0 went s1 in its fork; joined to a reference at hop 1 by went and by no path along the direction of its facts, features of a1 load, of s1 load, of the edge none
      target made of go: put before the model as the facts its edit makes of it
        s2: fact a0 went s2 in its fork; joined to a reference at hop 1 by went and by no path along the direction of its facts, features none
      parameter speed of go: put before the model as the facts its edit makes of it; facts a0 went a1, a0 went s1, a0 went s2 in its fork
"""
KEPT_REACH = "kept beside the bank, in its folder descriptions/reach"
REACH = """reach of the options of the variant %s of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/reach
realisation of options vector at version %d, reading kind, targets, parameters; options read beside the variant, without the actor of each action
derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
graphs of the bank the view does not show: none
%s
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree 1, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none
kind of task turn: fields target to of go, target along of go, target made of go, parameter speed of go; not read by the realisation: none
kinds of task that decide nothing, which hold no option: visit"""


@pytest.fixture
def walking():
    started = time.perf_counter()
    for name, entry in (("walked", WALKED), ("stepped", STEPPED), ("grounded", GROUNDED)):
        REALISATIONS.register(name, entry)
    LOG.debug("descriptions walking registered seconds=%.4f", time.perf_counter() - started)
    yield WALKED
    for name in ("walked", "stepped", "grounded"):
        del REALISATIONS.held[name]


@timed("descriptions.realising")
def realising(bank, cut, name):
    model = dict(MODEL, **{"options/realisation": name})
    return configuration(bank, model, None, pipeline=pipelined(cut), graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


@timed("descriptions.told")
def told(facts):
    return [dict(fact=fact, held=held) for fact, held in facts]


@timed("descriptions.aim")
def aim(target, held, columns, cut, *facts):
    put = not facts or any(found for _, found in facts)
    nodes = target.split(" ")[::2] if held["state"] in ("joined", "unjoined") else []
    features = [dict(node=key, columns=columns) for key in nodes]
    valued = [] if " " in target else None
    return dict(target=target, put=put, facts=told(facts), **held, features=features, values=valued, cut=cut)


@timed("descriptions.role")
def role(route, *targets):
    return dict(put=any(target["put"] for target in targets), route=route, targets=list(targets))


@timed("descriptions.own")
def own(route, cut, *facts):
    return dict(put=not facts or any(found for _, found in facts), route=route, facts=told(facts), cut=cut)


@timed("descriptions.kept")
def kept(fields):
    return dict(held=True, lacking=[], cut=None, fields=fields)


@timed("descriptions.left")
def left(lacking, cut, *names):
    return dict(held=False, lacking=lacking, cut=cut, fields={name: None for name in names})


@timed("descriptions.lack")
def lack(field, target, lacks, fact, *unobserved):
    return dict(field=field, target=target, lacks=lacks, fact=fact, unobserved=list(unobserved))


@timed("descriptions.nowhere")
def nowhere(on):
    return dict(on=on, entries=[], truth=[dict(target=None, held=False, hidden=[])])


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
        stopped=[],
        untried=[],
    )


@timed("descriptions.lifting")
def lifting(how, back):
    return dict(how=how, parameter=None, value=None, stating=[], back=back)


@timed("descriptions.cutting_entry")
def cutting_entry(entry, operation, settings, without):
    return dict(entry=entry, operation=operation, restated=[entry], spared=[], settings=settings, without=without)


@timed("descriptions.severing")
def severing(on, entry, relation, kind):
    settings = [setting(entry, "cut/relation", *relation), setting(entry, kind[0], *kind[1:])]
    return dict(on=on, entries=[cutting_entry(entry, entry, settings, None)], truth=None)


@timed("descriptions.placed")
def placed(cuts, held):
    if held is None:
        return None
    if held not in cuts:
        cuts.append(held)
    return cuts.index(held)


@timed("descriptions.numbered")
def numbered(cuts, option):
    fields = {}
    for name, field in option["fields"].items():
        if field is None:
            fields[name] = None
        elif "targets" in field:
            fields[name] = dict(field, targets=[dict(target, cut=placed(cuts, target["cut"])) for target in field["targets"]])
        else:
            fields[name] = dict(field, cut=placed(cuts, field["cut"]))
    return dict(option, cut=placed(cuts, option["cut"]), fields=fields)


@timed("descriptions.aimed")
def aimed(held, cut, target):
    return kept({GO[0]: role("target", aim(target, held, LOAD, cut)), GO[3]: own("option", None)})


@timed("descriptions.cut_at")
def cut_at(on, given, back, still):
    kind = ("cut/kind", "turn", ["visit"], [], []) if given else ("cut/kind", "turn", [], ["visit"], [])
    return severing(on, "test.cut", ("base", back, still, []), kind)


@timed("descriptions.parted_at")
def parted_at(on, given, back, still):
    visit = under("tasks", ("test.part", "cut/kind", "visit"))
    turn = dict(under("tasks", ("test.part", "cut/kind", "turn")), reason=REFUSING % "test.part")
    times = ("cut/times", "all", [visit], [], [turn]) if given else ("cut/times", "all", [], [visit], [turn])
    return severing(on, "test.part", ("base", back, still, []), times)


@timed("descriptions.unreached")
def unreached(target):
    return aimed(APART, nowhere("joined"), target)


@timed("descriptions.lost")
def lost(cut, graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        return left([lack(GO[1], EDGE, "fact", None)], cut("held", True, ["near"], []), *GO)
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    return aimed(APART, cut("joined", given, ["near"], []), target)


@timed("descriptions.severed")
def severed(graph, task, key, target, given):
    return lost(cut_at, graph, task, key, target, given)


@timed("descriptions.parted")
def parted(graph, task, key, target, given):
    return lost(parted_at, graph, task, key, target, given)


@timed("descriptions.given_back")
def given_back(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        along = severing("joined", "test.cut", ("near", [], ["base"], []), ("cut/kind", "turn", ["visit"], [], []))
        return kept(
            {
                GO[0]: role("target", aim("s0", ADJACENT, LOAD, None), aim("s1", APART, LOAD, along)),
                GO[1]: role("target", aim(EDGE, APART, LOAD, along)),
                GO[2]: role("role", aim("s2", CREATED, [], None)),
                GO[3]: own("option", None),
            }
        )
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    return aimed(ADJACENT, None, target)


@timed("descriptions.walked")
def walked(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        every = [("a0 went a1", True), ("a0 went s0", True), ("a0 went s1", True), ("a0 went s2", True)]
        return kept(
            {
                GO[0]: role("fact", aim("s0", WENT, LOAD, None, every[1]), aim("s1", WENT, LOAD, None, every[2])),
                GO[1]: role("fact", aim(EDGE, WENT, LOAD, None, every[0], every[2])),
                GO[2]: role("fact", aim("s2", WENT, [], None, every[3])),
                GO[3]: own("fact", None, *every),
            }
        )
    aimed_at = "s1" if (graph, task, key) == ("g1", 10, "o1") else target
    made = ("%s went %s" % ("a1" if task == 30 else "a0", aimed_at), True)
    return kept({GO[0]: role("fact", aim(aimed_at, WENT, LOAD, None, made)), GO[3]: own("fact", None, made)})


@timed("descriptions.grounded")
def grounded(key):
    if key != "o1":
        return left([lack(GO[0], "s0", "fact", "a0 base s0")], cut_at("held", True, ["near"], []), GO[0], GO[3])
    cut = cut_at("put", True, ["near"], [])
    went = [("a0 went a1", True), ("a0 went s1", True), ("a0 went s2", True)]
    based = [aim("s0", SKIRTED, LOAD, cut, ("a0 base s0", False)), aim("s1", WENT, LOAD, cut, ("a0 base s1", False))]
    return kept(
        {
            GO[0]: role("fact", *based),
            GO[1]: role("fact", aim(EDGE, WENT, LOAD, None, went[0], went[1])),
            GO[2]: role("fact", aim("s2", WENT, [], None, went[2])),
            GO[3]: own("fact", None, *went),
        }
    )


@timed("descriptions.refusing")
def refusing(graph, task, key, target, given):
    if (graph, task, key) == ("g0", 10, "o1"):
        return kept(
            {
                GO[0]: role("target", aim("s0", ADJACENT, LOAD, None), aim("s1", FARTHER, LOAD, None)),
                GO[1]: role("target", aim(EDGE, FARTHER, LOAD, None)),
                GO[2]: role("role", aim("s2", CREATED, [], None)),
                GO[3]: own("option", None),
            }
        )
    if (graph, task, key) == ("g1", 10, "o1"):
        return unreached("s1")
    if (graph, task) == ("g0", 50):
        refused = ("cut/kind", "visit", [], [], [["turn", REFUSAL]])
        return aimed(APART, severing("joined", "test.sever", ("base", ["near"], [], []), refused), target)
    return aimed(ADJACENT, None, target)


@timed("descriptions.reached")
def reached(stated):
    result = []
    for (graph, decisions), count in zip(DECIDED, (3, 1)):
        cuts = []
        found = [
            dict(
                task=task,
                kind="turn",
                time=moment,
                references=dict(agents=[actor]),
                options=[numbered(cuts, dict(key=key, **stated(graph, task, key, target, given))) for key in options],
            )
            for task, moment, actor, options, target, given in decisions
        ]
        decided = dict(turn=dict(decisions=count, described=count))
        result.append(dict(name=graph, view=VIEWED[graph], decided=decided, cuts=cuts, decisions=found))
    return result


@timed("descriptions.realised")
def realised(name):
    return dict(name=name, version=REALISATIONS.named(name).version, **REALISED[name])


@timed("descriptions.expected_reach")
def expected_reach(bank, config, name, fields):
    key = keyed(bank, config)
    cutter = config["pipeline"][1]
    record = dict(
        version=describing.VERSION,
        variant=key,
        stored_sha256=read_json(bank / "manifest.json")["stored_sha256"],
        time=CLOCKED,
        realisation=realised(name),
        refused=None,
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=SETTLED,
        unshown=[],
        entries=entries(cutter["operation"], cutter["parameters"]),
        tasks=dict(turn=dict(fields=list(GO), unread=[])),
        undecided=["visit"],
        graphs=reached(fields),
        kept=kept_as("reach", "bank"),
    )
    return record, stemmed(bank, config, "reach", name, None)


@timed("descriptions.spelled")
def spelled(record):
    lines = [
        REACH
        % (
            record["variant"],
            record["stored_sha256"],
            describing.VERSION,
            REALISATIONS.named("vector").version,
            reaching.READING["clock"],
            OPERATIONS.named("graphspec.nodes").version,
            OPERATIONS.named("graphspec.relation_forms").version,
        )
    ]
    for (graph, decisions), counted in zip(DECIDED, ("decisions of turn 3, described 3", "decisions of turn 1, described 1")):
        lines += ["", "graph %s, %s, %s" % (graph, SEEN_BY % graph, counted)] + SEVERED_TEXT[graph]
        for task, moment, actor, options, target, given in decisions:
            cut = "cut 1, by test.cut" if given else "cut 3, by test.cut"
            lines.append("  decision %d of turn at time %d, references agents %s" % (task, moment, actor))
            for key in options:
                if (graph, task, key) == ("g0", 10, "o1"):
                    lines.append(LEFT_TEXT)
                elif (graph, task, key) == ("g1", 10, "o1"):
                    lines.append(HELD_TEXT % dict(key=key, target="s1", cut="cut 2, by no entry"))
                else:
                    lines.append(HELD_TEXT % dict(key=key, target=target, cut=cut))
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
    assert read_json(stem.with_suffix(".json"))["record"] == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "statement.json", STATED)
    arguments = [str(bank), str(tmp_path / "config.json"), str(tmp_path / "statement.json")]
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--reach"] + arguments)
    main()
    assert capsys.readouterr().out == written
    ordered = reaching.text(dict(record, time=ORDERED))
    assert "\n  decision 10 of turn at position 1, references agents a0\n" in ordered
    assert "\n    the truth joins it to no reference at the position of the task\n" in ordered
    assert "time" not in reaching.READING["order"] + ordered[ordered.index("\ngraph ") :]
    timeless = [dict(graph, decisions=[dict(held, time=None) for held in graph["decisions"]]) for graph in record["graphs"]]
    still = reaching.text(dict(record, time=STILL, graphs=timeless))
    assert "\n  decision 10 of turn, references agents a0\n" in still and "\n    the truth joins it to no reference\n" in still
    assert "time" not in reaching.READING["none"] + still[still.index("\ngraph ") :]
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


def test_an_edit_of_another_mapping_at_the_same_version_is_not_served_the_reach_or_the_row_of_the_first(bank, cutting, monkeypatch):
    started = time.perf_counter()
    added = dict(WENT_ADDED, gone=WENT_ADDED["went"])
    stated = dict(EDITING, relations=dict(WENT_STATED, gone=WENT_STATED["went"]))
    monkeypatch.setattr(reaching, "recorded", refused)
    keys, stems, held, rows = [], [], [], []
    for relation in ("went", "gone"):
        mapping = {role: dict(relation=relation, actor="source") for role in ("to", "along", "made")}
        REALISATIONS.register("moving", Edited(dict(turn=dict(go=mapping)), added, 1))
        try:
            config = realising(bank, CUTTING, "moving")
            keys.append(named(bank, config, "reach", "moving", None))
            rows.append(named(bank, config, "reach_rows", "moving", None))
            stems.append(stemmed(bank, config, "reach", "moving", None))
            held.append(dict(stated, name="moving", version=1, mapping=dict(turn=dict(go=mapping))))
            served = stems[0].with_suffix(".json")
            if relation == "went":
                served.parent.mkdir(parents=True, exist_ok=True)
                write_json(served, dict(key=keys[0], record=dict(served="the edit that makes facts of went")))
                served.with_suffix(".txt").write_text("the edit that makes facts of went", encoding="utf-8")
                assert reach(bank, config, STATED) == read_json(served)["record"]
            else:
                with pytest.raises(AssertionError, match="the description was computed again"):
                    reach(bank, config, STATED)
            assert reaching.stated("moving") == held[-1]
        finally:
            del REALISATIONS.held["moving"]
    assert stems[0] != stems[1] and [key["realisation"] for key in keys] == held == [key["realisation"] for key in rows]
    assert without(keys[0], "realisation") == without(keys[1], "realisation") and keys[0] != keys[1]
    assert without(rows[0], "realisation") == without(rows[1], "realisation") and rows[0] != rows[1]
    LOG.info("descriptions reach of another mapping exit seconds=%.3f", time.perf_counter() - started)


def test_under_an_edit_the_reach_of_each_option_is_read_in_its_own_fork_where_its_facts_join_the_actor_to_its_target(
    bank, cutting, walking
):
    started = time.perf_counter()
    config = realising(bank, CUTTING, "walked")
    record = reach(bank, config, STATED)
    assert record == expected_reach(bank, config, "walked", walked)[0]
    assert reaching.text(record).split("\n")[1:4] == [KEPT_REACH] + WALKED_TEXT
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    directed = dict(config, graph=dict(config["graph"], **{"relations/direction": "directed"}))
    assert reaching.unrouted(deriver, config) is None and reaching.unrouted(deriver, directed) == DIRECTED
    refused, routed = (named(bank, directed, "reach", "walked", reason) for reason in (DIRECTED, None))
    assert refused == dict(routed, refused=DIRECTED) and routed["refused"] is None
    refused_text = reaching.text(dict(record, refused=DIRECTED)).split("\n")[1:4]
    assert refused_text == [reaching.REFUSAL % DIRECTED, KEPT_REACH, WALKED_TEXT[0]]
    LOG.info("descriptions reach edit exit seconds=%.3f", time.perf_counter() - started)


def test_under_an_edit_a_field_is_put_before_the_model_only_where_the_fork_of_its_option_holds_the_fact_its_edit_makes_of_it(
    bank, cutting, walking, monkeypatch
):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    held = reaching.Reached(deriver, realising(bank, CUTTING, "grounded"), "grounded")
    assert [found["time"] for found in held.decided(held.derived(None, 3, 0), 0)] == [1, 3, 5]
    monkeypatch.setattr(deriver.registry.schema, "model", "none")
    assert [found["time"] for found in held.decided(held.derived(None, 3, 0), 0)] == [None] * 3
    monkeypatch.undo()
    (decision,) = [found for found in held.decided(held.derived(None, 3, 0), 0) if found["key"] == 10]
    asked, cuts, found = [], {}, []
    options = [held.option(decision, option, asked) for option in held.optioned(decision)]
    held.searched(0, asked, cuts)
    assert options == [numbered(found, dict(key=key, **grounded(key))) for key in THREE]
    assert cuts == {0: found} and found == [cut_at("held", True, ["near"], []), cut_at("put", True, ["near"], [])]
    assert "".join(line + "\n" for option in options[1:] for line in reaching.option_text(option, found)) == GROUNDED_TEXT
    assert [line for place, cut in enumerate(found) for line in reaching.cut_text(place, cut, describing.NAMED["clock"])] == GROUND_CUTS
    LOG.info("descriptions reach edit facts exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.cutless")
def cutless(found):
    fields = {
        name: dict(field, targets=[dict(target, cut=None) for target in field["targets"]]) if "targets" in field else dict(field, cut=None)
        for name, field in found["fields"].items()
    }
    return dict(found, cut=None, fields=fields)


def test_a_target_the_fork_of_its_option_does_not_hold_reads_as_not_held_and_its_fact_as_not_in_the_fork(bank, walking):
    started = time.perf_counter()
    thinned = dict(NODES, **{"nodes/spot/filter": "degree", "nodes/spot/min_degree": 2})
    model = dict(MODEL, **{"options/realisation": "walked"})
    pipeline = [dict(operation="graphspec.nodes", parameters=thinned)]
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "walked")
    (decision,) = [found for found in held.decided(held.derived(None, 1, 0), 0) if found["key"] == 10]
    option = held.optioned(decision)[2]
    both = dict(UNDIRECTED, state="joined", hops=1, relations=[["base", "went"]])
    went = [("a0 went a1", True), ("a0 went s0", True), ("a0 went s1", True), ("a0 went s2", False)]
    fields = {
        GO[0]: role("fact", aim("s0", both, LOAD, None, went[1]), aim("s1", WENT, LOAD, None, went[2])),
        GO[1]: role("fact", aim(EDGE, WENT, LOAD, None, went[0], went[2])),
        GO[2]: role("fact", aim("s2", dict(UNDIRECTED, state="unheld", hops=None, relations=None), [], None, went[3])),
        GO[3]: own("fact", None, *went),
    }
    assert held.found.keys[option] == "o1" and cutless(held.read(None, 1, decision, option)) == kept(fields)
    assert reaching.field_text(GO[2], fields[GO[2]], []) + reaching.field_text(GO[3], fields[GO[3]], []) == [
        "      target made of go: not put before the model, its fork holding none of the facts its edit makes of it",
        "        s2: fact a0 went s2 not in its fork; not held at the step",
        "      parameter speed of go: put before the model as the facts its edit makes of it; facts a0 went a1, a0 went s0, "
        "a0 went s1 in its fork, fact a0 went s2 not in its fork",
    ]
    LOG.info("descriptions reach unheld exit seconds=%.3f", time.perf_counter() - started)


def test_an_option_whose_only_fact_the_variant_holds_without_its_edit_is_left_out_naming_that_fact(bank, walking):
    started = time.perf_counter()
    model = dict(MODEL, **{"options/realisation": "grounded"})
    pipeline = [dict(operation="graphspec.relation_forms", parameters=dict(FORMS, **{"relation_forms/parallel": "first"}))]
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "grounded")
    (decision,) = [found for found in held.decided(held.derived(None, 1, 0), 0) if found["key"] == 30]
    lacking = lack(GO[0], "s1", "edit", "a1 base s1")
    found = [dict(held.read(None, 1, decision, option), cut=None) for option in held.optioned(decision)]
    assert found == [left([lacking], None, GO[0], GO[3])] * 4
    assert reaching.lacking_text(lacking) == "a1 base s1 of s1, target to of go: in its fork and in the variant without the edit"
    LOG.info("descriptions reach unshown exit seconds=%.3f", time.perf_counter() - started)


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
    assert read_json(stem.with_suffix(".json"))["record"] == record
    written = stem.with_suffix(".txt").read_text(encoding="utf-8")
    assert written == reaching.text(record) and "\n".join(REFUSED_TEXT) + "\n  decision 10 " in written
    assert written.count("        s0: joined to a reference by no path, features load; cut 1, by test.sever\n") == 3
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
    assert written == reaching.text(record) and "\n".join(PARTED_TEXT) + "\n  decision 10 " in written
    assert [written.count("; cut %d, by test.part" % place) for place in (1, 2, 3)] == [8, 1, 3]
    LOG.info("descriptions reach made active exit seconds=%.3f", time.perf_counter() - started)


def test_past_the_bound_of_statements_tried_for_a_value_the_rest_are_recorded_as_not_tried_and_never_as_refused(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    held = reaching.Reached(deriver, cutter(bank, "test.part", PARTING), "vector")
    monkeypatch.setattr(restating, "BOUND", 1)
    (decision,) = [found for found in held.decided(held.derived(None, 3, 0), 0) if found["key"] == 50]
    option = int(held.found.first[decision["decision"]])
    asked = [dict(decision=decision, option=option, site=("joined", GO[0], 0), entries=[])]
    assert held.tried(None, 1, asked) == []
    turn = dict(under("tasks", ("test.part", "cut/kind", "turn")), reason=REFUSING % "test.part")
    untried = [under("tasks", ("test.part", "cut/kind", "visit"))]
    settings = [
        setting("test.part", "cut/relation", "base", ["near"], [], []),
        dict(setting("test.part", "cut/times", "all", [], [], [turn]), untried=untried),
    ]
    assert asked[0]["entries"] == [cutting_entry("test.part", "test.part", settings, None)]
    untried = dict(on="joined", entries=asked[0]["entries"], truth=None)
    assert reaching.cut_text(0, untried, describing.NAMED["clock"]) == UNTRIED_TEXT
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
    (times,) = [parameter for _, found in held.settable(None, 1) for parameter in found if parameter["name"] == "cut/times"]
    return held, times


def test_a_value_is_stated_with_what_it_makes_active_in_an_entry_that_reads_it_and_without_what_it_makes_inactive_there(
    bank, cutting
):
    started = time.perf_counter()
    held, times = reading(bank, PARTING, {})
    tried, untried = held.restating(None, held.steps[1], times, "tasks")
    visit = [("test.part", "cut/kind", "visit")]
    assert untried == [] and [(added, why) for added, _, why in tried] == [
        (under("tasks", ("test.part", "cut/kind", "turn"))["stating"], ("refused", dict(reason=REFUSING % "test.part"))),
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
    stated = under("tasks", *visit, ("test.read", "read/level", 3))
    worded = reaching.worded(dict(entry="test.part", parameter="cut/times"), stated, "test.part")
    assert worded == "cut/times tasks with cut/kind visit and read/level 3 of the entry test.read"
    held, times = reading(bank, {"cut/relation": "base", "cut/times": "tasks", "cut/kind": "visit"}, {"read/level": 2})
    ((added, key, reason),), untried = held.restating(None, held.steps[1], times, "all")
    assert (added, reason, untried) == ([], None, []) and held.variants[key][0]["pipeline"][1:] == [
        dict(operation="test.part", parameters=PARTING),
        dict(operation="test.read", parameters={}),
    ]
    LOG.info("descriptions reach across entries exit seconds=%.3f", time.perf_counter() - started)


LEANT_TEXT = [
    CUT_BY % (1, A_PATH, "test.lean"),
    RESTATED % ("test.lean", 1, 0, 0, 1, 0) + "; the parameters of test.part, which it reads, not re-stated",
    GIVEN % "lean/kind visit",
    CUT_BY % (2, A_FIT, "test.lean"),
    RESTATED % ("test.lean", 1, 0, 0, 1, 0) + "; the parameters of test.part, which it reads, not re-stated",
    GIVEN % "lean/kind visit",
    CUT_BY % (3, A_PATH, "test.lean"),
    "    test.lean, its parameters and those of test.part re-stated: statements giving it back 1, not giving it back 2, "
    "refused 1, stopped 1, not tried 0",
    GIVEN % "cut/times tasks of the entry test.part with cut/kind visit",
]
DOUBLED_TEXT = [
    "  cut 1, of being joined to a reference: by the entries test.cut and again, of the operation test.cut, each alone",
    RESTATED % ("test.cut", 0, 2, 0, 0, 0) + "; without the entry it is still missing",
    RESTATED % ("again, with test.cut absent", 2, 0, 0, 0, 0),
    GIVEN % NEAR_OR_VISIT,
    "  cut 2, of being held by a fit: by the entries test.cut and again, of the operation test.cut, each alone",
    RESTATED % ("test.cut", 0, 2, 0, 0, 0) + "; without the entry it is still missing",
    RESTATED % ("again, with test.cut absent", 2, 0, 0, 0, 0),
    GIVEN % NEAR_OR_VISIT,
    "  cut 3, of being joined to a reference: by the entries test.cut and again, of the operation test.cut, each alone",
    RESTATED % ("test.cut", 0, 2, 0, 0, 0) + "; without the entry it is still missing",
    RESTATED % ("again, with test.cut absent", 1, 1, 0, 0, 0),
    GIVEN % "cut/relation near",
]
LEAST_TEXT = [
    "  cut 1, of being joined to a reference: by the entries test.gate and test.cut, each alone",
    RESTATED % ("test.gate", 0, 1, 0, 0, 0)
    + "; the pipeline is refused or stops without the entry; under gate/level 1, after which the graph holds the most, "
    "it is still missing",
    RESTATED % ("test.cut, with test.gate at gate/level 1", 2, 0, 0, 0, 0),
    GIVEN % NEAR_OR_VISIT,
]
LIFTED_TEXT = (
    "  entry test.cut, of the operation test.cut, removes every edge of the relation base at the times of the tasks of "
    "turn: 1, 3, 5; not at every time; alone, with one parameter re-stated of the entry test.gate"
)
SEEING = dict(SPLITS, sees=dict(train=["without_scored"], validation=[]))
HIDDEN = dict(on="held", entries=[], truth=[dict(target="s1", held=True, hidden=["sees without_scored"])])
HIDDEN_TEXT = [
    CUT_BY_NONE % (1, A_FIT, A_FIT),
    "    s1: the truth holds it at the time of the task and the view does not show it to the task: the rule "
    "without_scored of what its split may see",
]
HIDDEN_PATH = (
    "    the truth joins it to a reference at the time of the task, and the view keeps facts of that time from the task "
    "by: the rule without_scored of what its split may see"
)


@timed("descriptions.leaned_at")
def leaned_at(on, given, back, still):
    heavy = dict(under("heavy"), error="GraphCapacityError", reason=HEAVY)
    mode = dict(setting("test.lean", "lean/mode", "cut", [], [], []), stopped=[heavy])
    if given:
        settings = [mode, setting("test.lean", "lean/kind", "turn", ["visit"], [], [])]
        return dict(on=on, entries=[dict(cutting_entry("test.lean", "test.lean", settings, None), spared=["test.part"])], truth=None)
    turn = dict(under("tasks", ("test.part", "cut/kind", "turn")), reason=REFUSING % "test.part")
    settings = [
        mode,
        setting("test.lean", "lean/kind", "turn", [], ["visit"], []),
        setting("test.part", "cut/relation", "near", [], ["base"], []),
        setting("test.part", "cut/times", "all", [under("tasks", ("test.part", "cut/kind", "visit"))], [], [turn]),
    ]
    found = dict(cutting_entry("test.lean", "test.lean", settings, None), restated=["test.lean", "test.part"])
    return dict(on=on, entries=[found], truth=None)


@timed("descriptions.leaned_on")
def leaned_on(graph, task, key, target, given):
    return lost(leaned_at, graph, task, key, target, given)


def test_a_cut_is_written_once_its_own_parameters_restated_first_and_a_derivation_the_library_stops_recorded_apart(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    pipeline = [
        dict(operation="graphspec.nodes", parameters=NODES),
        dict(operation="test.part", parameters={"cut/relation": "near", "cut/times": "all"}),
        dict(operation="test.lean", parameters={"lean/mode": "cut", "lean/kind": "turn"}),
    ]
    model = dict(MODEL, **{"options/realisation": "vector"})
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    begun = Mock(wraps=effect.begun)
    monkeypatch.setattr(effect, "begun", begun)
    record = reach(bank, config, STATED)
    assert record["graphs"] == reached(leaned_on) and record["refused"] is None and begun.call_count == 12
    assert json.loads(json.dumps(record)) == record
    assert [len(graph["cuts"]) for graph in record["graphs"]] == [3, 2]
    written = reaching.text(record)
    assert "\n".join(LEANT_TEXT) + "\n  decision 10 " in written
    lines = [line for graph in record["graphs"] for place, cut in enumerate(graph["cuts"]) for line in reaching.cut_text(place, cut, describing.NAMED["clock"])]
    assert len(lines) == 14 and max(map(len, lines)) < 200
    LOG.info("descriptions reach restated exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.twice_at")
def twice_at(on, given, back, still):
    first = dict(cut_at(on, False, [], ["near"])["entries"][0], without=lifting("absent", False))
    second = cut_at(on, given, back, still)["entries"][0]
    again = dict(second, entry="again", restated=["again"], settings=[dict(held, entry="again") for held in second["settings"]])
    return dict(on=on, entries=[first, again], truth=None)


@timed("descriptions.doubled")
def doubled(graph, task, key, target, given):
    return lost(twice_at, graph, task, key, target, given)


def test_of_two_entries_that_each_cut_both_are_named_each_alone_by_reach_and_among_those_leaving_nothing_at_the_task_steps(
    bank, cutting
):
    started = time.perf_counter()
    pipeline = pipelined(CUTTING)
    pipeline.insert(2, dict(operation="test.cut", label="again", parameters=CUTTING))
    model = dict(MODEL, **{"options/realisation": "vector"})
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "vector")
    asked, cuts, counted, expected = [], {}, {}, reached(doubled)[0]
    found = [
        [held.option(decision, option, asked) for option in held.optioned(decision)]
        for decision in held.decided(held.derived(None, 4, 0), 0)
    ]
    held.searched(0, asked, cuts)
    assert cuts == {0: expected["cuts"]} and found == [decision["options"] for decision in expected["decisions"]]
    assert [line for place, cut in enumerate(cuts[0]) for line in reaching.cut_text(place, cut, describing.NAMED["clock"])] == DOUBLED_TEXT
    assert reaching.cited(cuts[0], 0) == "cut 1, by test.cut and again"
    alone = (("test.cut", "absent"),)
    assert reaches.severed(held, 0, counted) == 1 and counted == {
        ("test.cut", "test.cut", "relation", "base", "turn", ()): 1,
        ("again", "test.cut", "relation", "base", "turn", alone): 1,
    }
    cut = dict(left_by("test.cut", ("relation", "base", "turn", 1))[0], entry="again", without=[dict(entry="test.cut", how="absent")])
    assert reaches.cut_cell(left_by("test.cut", ("relation", "base", "turn", 1)) + [cut], 1) == (
        "test.cut: relation base at turn 1/1; again: relation base at turn 1/1 alone without the entry test.cut"
    )
    LOG.info("descriptions reach of two cutting entries exit seconds=%.3f", time.perf_counter() - started)


def test_an_entry_the_pipeline_is_refused_without_is_restated_at_the_value_that_cuts_least_and_the_next_entry_named(
    bank, cutting
):
    started = time.perf_counter()
    pipeline = [
        dict(operation="graphspec.nodes", parameters=NODES),
        dict(operation="test.gate", parameters={"gate/level": 2}),
        dict(operation="test.leans", parameters={}),
        dict(operation="test.cut", parameters=CUTTING),
    ]
    model = dict(MODEL, **{"options/realisation": "vector"})
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "vector")
    (decision,) = [found for found in held.decided(held.derived(None, 4, 0), 0) if found["key"] == 10]
    asked, cuts = [], {}
    option = held.option(decision, held.optioned(decision)[1], asked)
    held.searched(0, asked, cuts)
    absent = held.settled((None, "test.gate", "absent"), dict(config, pipeline=pipeline[:1] + pipeline[2:]))
    assert absent == ([([dict(entry="test.leans", parameter="leans/alone", value="never")], None, ("refused", dict(reason=LEANING)))], [])
    least = dict(how="least", parameter="gate/level", value=1, stating=[], back=False)
    gate = cutting_entry("test.gate", "test.gate", [setting("test.gate", "gate/level", 2, [], [1], [])], least)
    assert cuts == {0: [dict(on="joined", entries=[gate, cut_at("joined", True, ["near"], [])["entries"][0]], truth=None)]}
    assert option == dict(key="o0", **aimed(APART, 0, "s0"))
    assert reaching.cut_text(0, cuts[0][0], describing.NAMED["clock"]) == LEAST_TEXT
    told = held.effects(0)[0]["effects"]
    turned = dict(sort="relation", name="base", task="turn", times=[1, 3, 5], everywhere=False)
    assert [found["cuts"] for found in told] == [
        [],
        [dict(turned, without=[])],
        [],
        [dict(turned, without=[dict(entry="test.gate", how="least")])],
    ]
    assert effect.cut_text(told[3], told[3]["cuts"][0], describing.NAMED["clock"]) == LIFTED_TEXT
    LOG.info("descriptions reach least exit seconds=%.3f", time.perf_counter() - started)


def test_where_no_entry_cuts_reach_reads_the_truth_and_names_the_rule_by_which_the_view_keeps_it_from_the_task(
    bank, crossed
):
    started = time.perf_counter()
    model = dict(MODEL, **{"options/realisation": "vector"})
    pipeline = [dict(operation="graphspec.nodes", parameters=NODES), dict(operation="graphspec.relation_forms", parameters=FORMS)]
    config = configuration(bank, model, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SEEING)
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    held = reaching.Reached(deriver, config, "vector")
    assert [graph["name"] for graph in deriver.viewed(config).graphs] == ["g0", "g0 for split 0", "g1", "g1 for split 0"]
    (decision,) = [found for found in held.decided(held.derived(None, 2, 1), 1) if found["key"] == 30]
    asked, cuts = [], {}
    options = [held.option(decision, option, asked) for option in held.optioned(decision)]
    held.searched(1, asked, cuts)
    lacking = lack(GO[0], "s1", "observation", None, "s1")
    assert options == [dict(key=key, **left([lacking], 0, GO[0], GO[3])) for key in FOUR] and cuts == {1: [HIDDEN]}
    assert reaching.cut_text(0, HIDDEN, describing.NAMED["clock"]) == HIDDEN_TEXT
    tokens = {key: token for token, key in held.names.items()}
    path = held.told_truly(
        held.truthful(decision), dict(site=("joined",), decision=decision), dict(target=None, ends=(tokens["s2"],), relation=None)
    )
    assert path == dict(target=None, held=True, hidden=["sees without_scored"])
    assert reaching.truth_text("joined", path, describing.NAMED["clock"]) == HIDDEN_PATH
    assert reaching.truth_text("joined", path, None) == (
        "    the truth joins it to a reference, and the view keeps facts from the task by: the rule without_scored of what "
        "its split may see"
    )
    other = reaching.Reached(Deriver(crossed, STATED["device"], STATED["derivation_seed"], RESIDENT), lined(crossed, THINNER, KEPT, "vector"), "vector")
    (first,) = [found for found in other.decided(other.derived(None, 1, 0), 0) if found["key"] == 10]
    seen = other.truthful(first)
    flags = other.hidden(seen, "fact", np.arange(len(seen["held"]["fact"])))
    assert sorted(flags) == ["known", "observer"] and int(flags["known"].sum()) == 3 and not flags["observer"].any()
    LOG.info("descriptions reach truth exit seconds=%.3f", time.perf_counter() - started)


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
        Refused,
        match="^reach reads the references of the decision 10 of the kind turn of graph g0 from its task at the step 0, "
        "and the graph derived for it holds 0 tasks of that key and kind there$",
    ):
        held.referred(view, dict(step=0, key=10, kind=0, graph="g0"))
    LOG.info("descriptions reach unheld exit seconds=%.3f", time.perf_counter() - started)


FIELDS = ("kind", "target to of go", "target along of go", "target made of go", "target by of stay", "parameter speed of go")
OVER = [["a0", "base", "s0"]]
REFERRED = dict(agents=["a0"], over=["a0 base s0"])
BOTH = "s0 near s1"
HERE = dict(state="joined", hops=0, relations=[], along=0, anchor=False)
NEARBY = dict(state="joined", hops=1, relations=[["near"]], along=1, anchor=False)
EITHER = dict(state="joined", hops=1, relations=[["near", "went"]], along=1, anchor=False)
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
LOOPED, BASED, NEARED = ("a0 went a0", True), ("a0 went s0", True), ("a0 went s1", True)
UNSEEN_TEXT = [
    CUT_BY_NONE % (1, A_PATH, A_PATH),
    UNJOINED,
    CUT_BY % (2, A_FIT, "graphspec.contract"),
    RESTATED % ("graphspec.contract", 0, 10, 0, 0, 0) + "; without the entry the pipeline gives it back",
    "  decision 10 of turn at time 0, references agents %s; over %s base s0" % (GROUPED, GROUPED),
]
UNROUTED = (
    "the field target along of go of the kind of task turn has no route to the prediction under the realisation of "
    "options vector: the pipeline removes the kind of node agent, which it may name, the entry graphspec.contract "
    "removing it"
)
UNSEEN_OPTION = (
    "    option o1, left out by a fit, lacking s0 base a0, target along of go: a0 not observed at the step; cut 2, by "
    "graphspec.contract; nothing is put before the model of the fields it states, kind, target to of go, target along of "
    "go, parameter speed of go\n"
)
CROSSED_TEXT = """kind of task turn: fields kind, target to of go, target along of go, target made of go, target by of stay, parameter speed of go; not read by the realisation: none
kinds of task that decide nothing, which hold no option: none

graph h0, the graph h0 as the tasks of the splits train, validation see it, decisions of turn 4, described 3, 1 holding no point a fit reads
  cut 1, of being joined to a reference: by no entry, joined to a reference in no state, the view before the first entry included
    the truth joins it to no reference at the time of the task
  decision 10 of turn at time 0, references agents a0; over a0 base s0
    option o0, held by a fit
      kind: put before the model with the option
      target to of go: put before the model as the targets the variant holds
        s2: joined to a reference by no path, features load; cut 1, by no entry
      parameter speed of go: put before the model with the option
    option o1, held by a fit
      kind: put before the model with the option
      target by of stay: put before the model as the targets the variant holds
        s1: joined to a reference at hop 1 by near and at hop 1 along the direction of its facts, features load
  decision 20 of turn at time 1, references agents a0; over a0 base s0
    option o0, held by a fit
      kind: put before the model with the option
      target to of go: put before the model as the targets the variant holds
        s0: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features load
      target along of go: put before the model as the targets the variant holds
        a0 base s0: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of a0 load, of s0 load, of the edge none
        s0 near s1: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of s0 load, of s1 load, of the edge none
      parameter speed of go: put before the model with the option
    option o1, held by a fit
      kind: put before the model with the option
      target to of go: put before the model as the targets the variant holds
        s1: joined to a reference at hop 1 by near and at hop 1 along the direction of its facts, features load
      target along of go: put before the model as the targets the variant holds
        s0 base a0: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of s0 load, of a0 load, of the edge none
        s0 near s1: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of s0 load, of s1 load, of the edge none
      parameter speed of go: put before the model with the option
  decision 30 of turn at time 2, references agents a0; over a0 base s0
    option o0, held by a fit
      kind: put before the model with the option
      target by of stay: put before the model as the targets the variant holds
        s0: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features load
"""
STEPPED_TEXT = """  decision 20 of turn at time 1, references agents a0; over a0 base s0
    option o0, held by a fit
      kind: put before the model as the facts its edit makes of it; facts a0 went a0, a0 went s0, a0 went s1 in its fork
      target to of go: put before the model as the facts its edit makes of it
        s0: fact a0 went s0 in its fork; joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features load
      target along of go: put before the model as the facts its edit makes of it
        a0 base s0: facts a0 went a0, a0 went s0 in its fork; joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of a0 load, of s0 load, of the edge none
        s0 near s1: facts a0 went s0, a0 went s1 in its fork; joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features of s0 load, of s1 load, of the edge none
      parameter speed of go: put before the model as the facts its edit makes of it; facts a0 went a0, a0 went s0, a0 went s1 in its fork
"""
THINNED_TEXT = (
    "\n".join(
        [
            CUT_BY % (1, A_FIT, "graphspec.nodes"),
            RESTATED % ("graphspec.nodes", 3, 19, 0, 0, 0),
            GIVEN % "nodes/spot/filter none, nodes/spot/filter sample with nodes/spot/sample 0.9, nodes/spot/min_degree 1",
            "  decision 10 of turn at time 0, references agents a0; over a0 base s0",
            "    option o0, left out by a fit, lacking a0 went s2 of s2, target to of go: not in its fork; cut 1, by "
            "graphspec.nodes; nothing is put before the model of the fields it states, kind, target to of go, parameter "
            "speed of go",
        ]
    )
    + "\n"
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
    either = dict(synthetic.edge("agent", "spot", [], directed=False), endpoints=[["agent", "spot"], ["spot", "agent"]])
    edges = dict(near=synthetic.edge("spot", "spot", [], directed=False), base=either)
    columns, clocks = dict(load=NUMBER, speed=NUMBER), dict(held["time"], model="clocks")
    return dict(held, time=clocks, columns=columns, edges=edges, tasks=dict(held["tasks"], references=references, kinds=kinds))


@timed("descriptions.acted")
def acted(key, kind, targets, parameters):
    return synthetic.offered(key, [synthetic.acting(kind, "a0", targets, parameters)], "set", "recorded")


def over(stated):
    started = time.perf_counter()
    result = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], references=dict(agents=[], over=[]))
    for key, moment, split, gain, options in stated:
        options[0].update(chosen=True, outcome=None if gain is None else dict(gain=gain))
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
            (40, 2, "validation", None, [acted("o0", "stay", dict(by=["s2"]), {})]),
        ]
    )
    held = graph(
        "h0",
        dict(agent=["a0"], spot=["s0", "s1", "s2"]),
        [("a0", 0, None, 1.0)],
        [("s0", 0, None, 1.0), ("s1", 0, None, 2.0), ("s2", 0, None, 2.0)],
        [
            ("near", ("spot", "spot"), BASE, [("s1", "s0", True, 0, None)]),
            ("near", ("spot", "spot"), LATER, [("s1", "s2", True, 0, None, 3, 4), ("s1", "s2", False, 0, None, 4, None)]),
            ("base", ("agent", "spot"), BASE, [("a0", "s0", True, 0, None)]),
            ("base", ("spot", "agent"), BASE, [("s0", "a0", True, 3, None)]),
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
def gone(held, cut):
    lone = role("target", aim("s2", held, LOAD, cut))
    return kept({FIELDS[0]: own("option", None), FIELDS[1]: lone, FIELDS[5]: own("option", None)})


@timed("descriptions.along")
def along(held, target, based):
    return kept(
        {
            FIELDS[0]: own("option", None),
            FIELDS[1]: role("target", aim(target, held, LOAD, None)),
            FIELDS[2]: role("target", aim(based, HERE, LOAD, None), aim(BOTH, HERE, LOAD, None)),
            FIELDS[5]: own("option", None),
        }
    )


@timed("descriptions.stayed")
def stayed(held, target):
    return kept({FIELDS[0]: own("option", None), FIELDS[4]: role("target", aim(target, held, LOAD, None))})


@timed("descriptions.unseen")
def unseen(edge):
    settings = [
        setting(CONTRACTING, "contract/grouping", "agent:load", [], ["spot:load"], []),
        setting(CONTRACTING, "contract/aggregate", "mean", [], ["sum", "min", "max"], []),
        setting(CONTRACTING, "contract/parallel", "keep", [], ["first", "last", "sum", "mean", "min", "max"], []),
    ]
    found = cutting_entry(CONTRACTING, CONTRACTING, settings, lifting("absent", True))
    cut = dict(on="held", entries=[found], truth=None)
    return left([lack(FIELDS[2], edge, "observation", None, "a0")], cut, FIELDS[0], FIELDS[1], FIELDS[2], FIELDS[5])


@timed("descriptions.stepping")
def stepping(held, target, made, based):
    return kept(
        {
            FIELDS[0]: own("fact", None, LOOPED, BASED, NEARED),
            FIELDS[1]: role("fact", aim(target, held, LOAD, None, made)),
            FIELDS[2]: role("fact", aim(based, HERE, LOAD, None, LOOPED, BASED), aim(BOTH, HERE, LOAD, None, BASED, NEARED)),
            FIELDS[5]: own("fact", None, LOOPED, BASED, NEARED),
        }
    )


@timed("descriptions.staying")
def staying(held, target, made):
    return kept({FIELDS[0]: own("fact", None, made), FIELDS[4]: role("fact", aim(target, held, LOAD, None, made))})


@timed("descriptions.held_whole")
def held_whole(lone, first, second):
    return [lone, stayed(NEARBY, "s1"), first, second, stayed(HERE, "s0")]


@timed("descriptions.crossed_reach")
def crossed_reach(bank, config, name, references, fields):
    decisions = [
        (10, 0, [("o0", fields[0]), ("o1", fields[1])]),
        (20, 1, [("o0", fields[2]), ("o1", fields[3])]),
        (30, 2, [("o0", fields[4])]),
    ]
    cuts = []
    described = [
        dict(
            task=task,
            kind="turn",
            time=moment,
            references=references,
            options=[numbered(cuts, dict(key=option, **found)) for option, found in options],
        )
        for task, moment, options in decisions
    ]
    return dict(
        version=describing.VERSION,
        variant=keyed(bank, config),
        stored_sha256=read_json(bank / "manifest.json")["stored_sha256"],
        time=dict(model="clocks", units=["tick"]),
        realisation=realised(name),
        refused=None,
        seed=STATED["derivation_seed"],
        device="cpu",
        splits=SPLITS,
        settings=CROSSED,
        unshown=[],
        entries=entered(config["pipeline"]),
        tasks=dict(turn=dict(fields=list(FIELDS), unread=[])),
        undecided=[],
        kept=kept_as("reach", "bank"),
        graphs=[
            dict(
                name="h0",
                view=VIEWED["h0"],
                decided=dict(turn=dict(decisions=4, described=3)),
                cuts=cuts,
                decisions=described,
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
    return dict(on="held", entries=[cutting_entry(THINNER, THINNER, settings, None)], truth=None)


@timed("descriptions.halved")
def halved(seed, salt, tokens, steps):
    return np.full(len(tokens), 0.5)


def test_reach_reads_an_edge_without_direction_stated_either_way_a_reference_that_is_an_edge_a_target_at_no_hop_and_the_kind(
    crossed,
):
    started = time.perf_counter()
    config = lined(crossed, THINNER, KEPT, "vector")
    record = reach(crossed, config, STATED)
    fields = held_whole(gone(APART, nowhere("joined")), along(HERE, "s0", "a0 base s0"), along(NEARBY, "s1", "s0 base a0"))
    assert record == crossed_reach(crossed, config, "vector", REFERRED, fields)
    assert json.loads(json.dumps(record)) == record
    held = describe(crossed)["graphs"][0]["decisions"]["turn"]
    assert (held["decisions"], [found["key"] for found in held["each"]]) == (4, [10, 20, 30, 40])
    assert record["graphs"][0]["decided"] == dict(turn=dict(decisions=4, described=3))
    assert [decision["task"] for decision in record["graphs"][0]["decisions"]] == [10, 20, 30]
    written = reaching.text(record)
    assert written[written.index("kind of task turn") :] == CROSSED_TEXT
    first, second = (option["fields"][FIELDS[2]]["targets"] for option in record["graphs"][0]["decisions"][1]["options"])
    assert first[1] == second[1] and dict(first[0], target="s0 base a0", features=first[0]["features"][::-1]) == second[0]
    LOG.info("descriptions reach crossed exit seconds=%.3f", time.perf_counter() - started)


def test_under_an_edit_on_a_bank_whose_relations_hold_no_column_the_entry_that_removes_a_node_in_a_fork_is_named_with_what_gives_it_back(
    crossed, walking, monkeypatch
):
    started = time.perf_counter()
    stated = Registry.from_declaration(read_json(crossed / "manifest.json")["spec"]).schema.edges
    assert [list(entry["columns"]) for entry in stated.values()] == [[], []] and WENT_ADDED["went"]["columns"] == ("speed",)
    monkeypatch.setattr(filtering, "unit", halved)
    config = lined(crossed, THINNER, THINNED, "stepped")
    begun = Mock(wraps=effect.begun)
    monkeypatch.setattr(effect, "begun", begun)
    record = reach(crossed, config, STATED)
    assert begun.call_count == 25
    fields = [
        left([lack(FIELDS[1], "s2", "fact", "a0 went s2")], thinning(), FIELDS[0], FIELDS[1], FIELDS[5]),
        staying(EITHER, "s1", NEARED),
        stepping(HERE, "s0", BASED, "a0 base s0"),
        stepping(EITHER, "s1", NEARED, "s0 base a0"),
        staying(HERE, "s0", BASED),
    ]
    assert record == crossed_reach(crossed, config, "stepped", REFERRED, fields)
    written = reaching.text(record)
    assert THINNED_TEXT in written and STEPPED_TEXT in written
    LOG.info("descriptions reach edit cut exit seconds=%.3f", time.perf_counter() - started)


def test_a_reference_the_pipeline_puts_in_a_node_it_adds_is_named_by_the_kind_of_that_node_and_the_node_in_whose_place_it_stands(
    crossed,
):
    started = time.perf_counter()
    config = lined(crossed, CONTRACTING, dict(CONTRACTED, **{"contract/grouping": "agent:load"}), "vector")
    record = reach(crossed, config, STATED)
    grouped = dict(agents=[GROUPED], over=[GROUPED + " base s0"])
    fields = held_whole(gone(APART, nowhere("joined")), unseen("a0 base s0"), unseen("s0 base a0"))
    assert record == dict(crossed_reach(crossed, config, "vector", grouped, fields), refused=UNROUTED)
    written = reaching.text(record)
    assert written.split("\n")[1] == reaching.REFUSAL % UNROUTED
    assert "\n".join(UNSEEN_TEXT) + "\n" in written and written.count(UNSEEN_OPTION) == 1
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
        Refused,
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
HALTED = "the entry test.halt derives no graph\nand says so on a second line"
STOPPED = (
    "the derivation of its variant stopped with GraphCapacityError: the entry test.halt derives no graph and says so on "
    "a second line"
)
UNSTARTED = "the trial holds no statement.json"
UNREALISED = (
    "the model of the configuration states options/realisation none, which puts no option before it; reach describes "
    "the options of decisions as a realisation puts them before a model"
)
SEEDED = "the seed of the configuration is 6 and the seed of the statement is 5"
AIMLESS = (
    "the field parameter speed of go of the kind of task turn has no route to the prediction under the realisation of "
    "options aimed: the realisation reads no parameters"
)


@timed("descriptions.halted")
def halted(graph, registry, settings, seed, limits, carry, held):
    raise GraphCapacityError(HALTED)


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


@timed("descriptions.stopped")
def stopped(part, found):
    return [{"entry": name, "operation": name, part: count} for name, count in found]


@timed("descriptions.owned")
def owned(stated, held, put, *unput):
    return dict(
        stated=stated,
        held=held,
        put=put,
        unput=stopped("options", unput),
        targets=None,
        joined=None,
        least=None,
        most=None,
        parted=None,
    )


@timed("descriptions.roled")
def roled(stated, held, put, targets, joined, least, most, *parted):
    return dict(
        stated=stated,
        held=held,
        put=put,
        unput=[],
        targets=targets,
        joined=joined,
        least=least,
        most=most,
        parted=stopped("targets", parted),
    )


@timed("descriptions.tasked")
def tasked(held, left, *fields):
    return dict(decisions=4, described=4, options=13, held=held, left=stopped("options", left), fields=dict(zip(GO, fields)))


NEVER = (None, 1)
SPED = owned(13, 13, 13)
UNMADE = roled(1, 1, 1, 1, 0, None, None)
UNSTATED = roled(1, 0, 0, 0, 0, None, None)
FAR = roled(1, 1, 1, 1, 1, 2, 2)
ONE = roled(1, 1, 1, 1, 1, 1, 1)
WHOLE = tasked(13, [], roled(13, 13, 13, 14, 13, 1, 2, NEVER), FAR, UNMADE, SPED)
EDITED = tasked(13, [], roled(13, 13, 13, 14, 14, 1, 1), ONE, ONE, SPED)
BASE_TURN = ("relation", "base", "turn", 2)
BASE_VISIT = ("relation", "base", "visit", 2)
NEAR_AT_TURN = ("relation", "near", "turn", 1)
NEAR_AT_VISIT = ("relation", "near", "visit", 1)
AGENTS = (("kind", "agent", "turn", 2), ("kind", "agent", "visit", 2))


@timed("descriptions.gone_by")
def gone_by(name):
    every = roled(13, 12, 12, 12, 0, None, None, (name, 11), NEVER)
    return tasked(12, [(name, 1)], every, UNSTATED, UNSTATED, owned(13, 12, 12))


@timed("descriptions.near_by")
def near_by(name):
    most = roled(13, 13, 13, 14, 12, 1, 1, (name, 1), NEVER)
    return tasked(13, [], most, roled(1, 1, 1, 1, 0, None, None, (name, 1)), UNMADE, SPED)


@timed("descriptions.visited_by")
def visited_by(name):
    return tasked(13, [], roled(13, 13, 13, 14, 10, 1, 2, (name, 3), NEVER), FAR, UNMADE, SPED)


@timed("descriptions.unsped")
def unsped(held):
    return dict(held, fields=dict(held["fields"], **{GO[3]: None}))


@timed("descriptions.left_by")
def left_by(name, *found):
    return [
        dict(entry=name, operation=name, sort=sort, name=held, task=task, without=[], graphs=graphs)
        for sort, held, task, graphs in found
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
    ("aimed", unsped(visited_by("test.cut")), left_by("test.cut", BASE_VISIT), AIMLESS),
    ("vector", gone_by("test.cut"), left_by("test.cut", BASE_TURN)),
    ("walked", EDITED, left_by("test.cut", NEAR_AT_TURN)),
    ("vector", STOPPED),
)
SEARCHED_TEXT = """reach over 20 trials on the bank stored as %(stored)s, described at version %(version)d
kept beside the search, in its folder descriptions/reaches
described 16, of 15 variants and realisations; not described 4
one row per trial, in the order of the trials; %(reading)s
    trial         variant     realisation  a fit        turn: options held/stated in decisions described/shown  turn: target to of go, put/held/stated                                           turn: target along of go, put/held/stated      turn: target made of go, put/held/stated  turn: parameter speed of go, put/held/stated  nothing left at the steps of tasks    a fit is refused because
    trial_000000  %(v0)s  vector       not refused  12/13 in 4/4, left out cut by test.cut 1                12/12/13, joined 0/12, no path cut by test.cut 11 and no entry 1                 0/0/1                                          0/0/1                                     12/12/13                                      test.cut: relation base at turn 2/2
    trial_000001  %(v1)s  vector       not refused  12/13 in 4/4, left out cut by test.cut 1                12/12/13, joined 0/12, no path cut by test.cut 11 and no entry 1                 0/0/1                                          0/0/1                                     12/12/13                                      test.cut: relation base at turn 2/2
    trial_000002  %(v2)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 12/14, hops 1, no path cut by test.cut 1 and no entry 1         1/1/1, joined 0/1, no path cut by test.cut 1   1/1/1, joined 0/1                         13/13/13                                      test.cut: relation near at turn 1/2
    trial_000003  %(v3)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 10/14, hops 1 to 2, no path cut by test.cut 3 and no entry 1    1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         13/13/13                                      test.cut: relation base at visit 2/2
    trial_000004  %(v4)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 13/14, hops 1 to 2, no path cut by no entry 1                   1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         13/13/13                                      test.cut: relation near at visit 1/2
    trial_000005  %(v5)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 13/14, hops 1 to 2, no path cut by no entry 1                   1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         13/13/13                                      none
    trial_000006  %(v6)s  walked       not refused  13/13 in 4/4                                            13/13/13, joined 14/14, hops 1                                                   1/1/1, joined 1/1, hops 1                      1/1/1, joined 1/1, hops 1                 13/13/13                                      test.cut: relation base at turn 2/2
    trial_000007  %(v7)s  actor_edge   not refused  12/13 in 4/4, left out cut by test.cut 1                12/12/13, joined 0/12, no path cut by test.cut 11 and no entry 1                 0/0/1                                          0/0/1                                     12/12/13                                      test.cut: relation base at turn 2/2
    trial_000008  -           -            not described: the entry test.sever of the pipeline states cut/kind turn, which decides, and the entry cuts only at the times of the tasks of kinds that decide nothing
    trial_000009  %(v9)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 10/14, hops 1 to 2, no path cut by test.sever 3 and no entry 1  1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         13/13/13                                      test.sever: relation base at visit 2/2
    trial_000010  %(v10)s  vector       not refused  12/13 in 4/4, left out cut by test.part 1               12/12/13, joined 0/12, no path cut by test.part 11 and no entry 1                0/0/1                                          0/0/1                                     12/12/13                                      test.part: relation base at turn 2/2, relation base at visit 2/2
    trial_000011  %(v11)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 12/14, hops 1, no path cut by test.part 1 and no entry 1        1/1/1, joined 0/1, no path cut by test.part 1  1/1/1, joined 0/1                         13/13/13                                      test.part: relation near at turn 1/2, relation near at visit 1/2
    trial_000012  %(v12)s  vector       not refused  13/13 in 4/4                                            13/13/13, joined 10/14, hops 1 to 2, no path cut by test.part 3 and no entry 1   1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         13/13/13                                      test.part: relation base at visit 2/2
    trial_000013  %(v13)s  vector       not refused  12/13 in 4/4, left out cut by graphspec.nodes 1         12/12/13, joined 0/12, no path cut by graphspec.nodes 11 and no entry 1          0/0/1                                          0/0/1                                     12/12/13                                      graphspec.nodes: kind agent at turn 2/2, kind agent at visit 2/2, relation base at turn 2/2, relation base at visit 2/2
    trial_000014  -           -            not described: the model of the configuration states options/realisation none, which puts no option before it; reach describes the options of decisions as a realisation puts them before a model
    trial_000015  -           -            not described: the trial holds no statement.json
    trial_000016  %(v16)s  aimed        refused      13/13 in 4/4                                            13/13/13, joined 10/14, hops 1 to 2, no path cut by test.cut 3 and no entry 1    1/1/1, joined 1/1, hops 2                      1/1/1, joined 0/1                         unread                                        test.cut: relation base at visit 2/2  %(aimless)s
    trial_000017  %(v17)s  vector       not refused  12/13 in 4/4, left out cut by test.cut 1                12/12/13, joined 0/12, no path cut by test.cut 11 and no entry 1                 0/0/1                                          0/0/1                                     12/12/13                                      test.cut: relation base at turn 2/2
    trial_000018  %(v18)s  walked       not refused  13/13 in 4/4                                            13/13/13, joined 14/14, hops 1                                                   1/1/1, joined 1/1, hops 1                      1/1/1, joined 1/1, hops 1                 13/13/13                                      test.cut: relation near at turn 1/2
    trial_000019  %(v19)s  vector       not described: %(stopped)s
"""
ROW_TEXT = """reach of the options of the variant %s of the bank stored as %s in one row, described at version %d
kept beside the bank, in its folder descriptions/reach_rows
%s
    variant     realisation  turn: options held/stated in decisions described/shown  turn: target to of go, put/held/stated                                   turn: target along of go, put/held/stated  turn: target made of go, put/held/stated  turn: parameter speed of go, put/held/stated  nothing left at the steps of tasks
    %s  vector       12/13 in 4/4, left out cut by graphspec.nodes 1         12/12/13, joined 0/12, no path cut by graphspec.nodes 11 and no entry 1  0/0/1                                      0/0/1                                     12/12/13                                      graphspec.nodes: kind agent at turn 2/2, kind agent at visit 2/2, relation base at turn 2/2, relation base at visit 2/2
"""
CROSSED_ROW = dict(
    turn=dict(
        decisions=4,
        described=3,
        options=5,
        held=5,
        left=[],
        fields={
            FIELDS[0]: owned(5, 5, 5),
            FIELDS[1]: roled(3, 3, 3, 3, 2, 0, 1, NEVER),
            FIELDS[2]: roled(2, 2, 2, 4, 4, 0, 0),
            FIELDS[3]: roled(0, 0, 0, 0, 0, None, None),
            FIELDS[4]: roled(2, 2, 2, 2, 2, 0, 1),
            FIELDS[5]: owned(3, 3, 3),
        },
    )
)
STEPPED_ROW = dict(CROSSED_ROW["turn"], fields=dict(CROSSED_ROW["turn"]["fields"], **{FIELDS[1]: roled(3, 3, 3, 3, 3, 0, 1)}))


def over_search(bank, configs, stated):
    started = time.perf_counter()
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    rows = []
    for place, (config, held) in enumerate(zip(configs, stated)):
        key = None if held[0] is None else specs.digest(deriver.identity(config))
        row = dict(trial=trial(place), variant=key, realisation=held[0], refused=None, reason=None, graphs=None, tasks=None, cuts=None)
        if len(held) == 2:
            row["reason"] = held[1]
        else:
            row.update(graphs=2, tasks=dict(turn=held[1]), cuts=held[2], refused=held[3] if len(held) == 4 else None)
        rows.append(row)
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    result = dict(
        version=describing.VERSION, stored_sha256=stored, fields=dict(turn=list(GO)), trials=rows, kept=kept_as("reaches", "search")
    )
    LOG.debug("descriptions over search exit trials=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return result


def tabled(folder, record):
    started = time.perf_counter()
    kept = sorted((folder / "descriptions" / "reaches").glob("*-*.json"))
    (found,) = [path for path in kept if read_json(path)["record"] == record]
    held = read_json(found)["key"]
    assert held["part"] == "reaches" and held["stored"] == record["stored_sha256"] and found.stem == describing.named(held)
    assert [(trial["trial"], trial["refused"], trial["row"] is None) for trial in held["trials"]] == [
        (row["trial"], row["refused"], row["variant"] is None) for row in record["trials"]
    ]
    assert [trial["reason"] for trial in held["trials"]] == [
        row["reason"] if row["variant"] is None else None for row in record["trials"]
    ]
    LOG.debug("descriptions tabled exit tables=%d seconds=%.4f", len(kept), time.perf_counter() - started)
    return found.with_suffix("")


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
    key, configs = rows[13]["variant"], tried(bank)
    stem = stemmed(bank, configs[13], "reach_rows", "vector", None)
    kept = {name: value for name, value in rows[13].items() if name not in ("trial", "refused", "reason")}
    assert read_json(stem.with_suffix(".json"))["record"] == dict(
        kept,
        version=describing.VERSION,
        stored_sha256=record["stored_sha256"],
        fields=record["fields"],
        kept=kept_as("reach_rows", "bank"),
    )
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == ROW_TEXT % (
        key,
        record["stored_sha256"],
        describing.VERSION,
        reaches.READING,
        key[:10],
    )
    assert rows[19]["variant"] is not None and rows[19]["reason"] == STOPPED and "\n" not in rows[19]["reason"]
    halted = stemmed(bank, configs[19], "reach_rows", "vector", None)
    assert not halted.with_suffix(".json").exists() and not halted.with_suffix(".txt").exists()
    LOG.info("descriptions reach over a search exit seconds=%.3f", time.perf_counter() - started)


def test_the_text_of_the_table_holds_one_line_per_trial_of_its_record_and_the_verb_prints_what_the_call_writes(
    bank, searched, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    record = reach_search(bank, searched)
    shown = {"v%d" % place: row["variant"][:10] for place, row in enumerate(record["trials"]) if row["variant"] is not None}
    written = SEARCHED_TEXT % dict(
        shown, stored=record["stored_sha256"], version=describing.VERSION, reading=reaches.READING, aimless=AIMLESS, stopped=STOPPED
    )
    assert reaches.text(record) == written and len(written.split("\n")) == 5 + len(record["trials"]) + 1
    stem = tabled(searched, record)
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
kept beside the bank, in its folder descriptions/reaches
described 2, of 1 variants and realisations; not described 1
one row per trial, in the order of the trials; %s
    trial  variant     realisation  a fit        turn: options held/stated in decisions described/shown  turn: kind, put/held/stated  turn: target to of go, put/held/stated                     turn: target along of go, put/held/stated  turn: target made of go, put/held/stated  turn: target by of stay, put/held/stated  turn: parameter speed of go, put/held/stated  nothing left at the steps of tasks  a fit is refused because
    0      %s  vector       not refused  5/5 in 3/4                                              5/5/5                        3/3/3, joined 2/3, hops 0 to 1, no path cut by no entry 1  2/2/2, joined 4/4, hops 0                  stated by none                            2/2/2, joined 2/2, hops 0 to 1            3/3/3                                         none
    1      %s  vector       not refused  5/5 in 3/4                                              5/5/5                        3/3/3, joined 2/3, hops 0 to 1, no path cut by no entry 1  2/2/2, joined 4/4, hops 0                  stated by none                            2/2/2, joined 2/2, hops 0 to 1            3/3/3                                         none
    2      -           -            not described: the seed of the configuration is 6 and the seed of the statement is 5
"""
STEPPED_OVER = """    trial  variant     realisation  a fit        turn: options held/stated in decisions described/shown  turn: kind, put/held/stated  turn: target to of go, put/held/stated  turn: target along of go, put/held/stated  turn: target made of go, put/held/stated  turn: target by of stay, put/held/stated  turn: parameter speed of go, put/held/stated  nothing left at the steps of tasks  a fit is refused because
    0      %s  stepped      not refused  5/5 in 3/4                                              5/5/5                        3/3/3, joined 3/3, hops 0 to 1          2/2/2, joined 4/4, hops 0                  stated by none                            2/2/2, joined 2/2, hops 0 to 1            3/3/3                                         none
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
    described = dict(variant=key, realisation="vector", refused=None, reason=None, graphs=1, tasks=CROSSED_ROW, cuts=[])
    unread = dict(variant=None, realisation=None, refused=None, reason=SEEDED, graphs=None, tasks=None, cuts=None)
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=stored,
        fields=dict(turn=list(FIELDS)),
        trials=[dict(described, trial=0), dict(described, trial=1), dict(unread, trial=2)],
        kept=kept_as("reaches", "bank"),
    )
    assert json.loads(json.dumps(record)) == record and rowed.call_count == 1
    written = CROSSED_OVER % (stored, describing.VERSION, reaches.READING, key[:10], key[:10])
    stem = tabled(crossed, record)
    assert reaches.text(record) == written == stem.with_suffix(".txt").read_text(encoding="utf-8")
    assert stemmed(crossed, config, "reach_rows", "vector", None).with_suffix(".json").exists()
    monkeypatch.setattr(reaches, "recorded", refused)
    assert reach_over(crossed, [config, wider, dict(config, seed=6)], STATED) == record
    with pytest.raises(AssertionError, match="the description was computed again"):
        reach_over(crossed, [wider, config], STATED)
    LOG.info("descriptions reach over a list exit seconds=%.3f", time.perf_counter() - started)


def test_a_trial_under_an_edit_on_a_bank_whose_relations_hold_no_column_has_its_row_in_the_table(crossed, walking):
    started = time.perf_counter()
    config = lined(crossed, THINNER, KEPT, "stepped")
    record = reach_over(crossed, [config], STATED)
    key = keyed(crossed, config)
    row = dict(
        trial=0, variant=key, realisation="stepped", refused=None, reason=None, graphs=1, tasks=dict(turn=STEPPED_ROW), cuts=[]
    )
    assert record["trials"] == [row]
    assert reaches.text(record).split("\n", 4)[4] == STEPPED_OVER % key[:10]
    LOG.info("descriptions reach over an edit without columns exit seconds=%.3f", time.perf_counter() - started)


VIEWLESS = (
    "the model of the configuration states no view, the family of its model, one of snapshot, mixed, event_graph, "
    "memory, projection, sequence, tree, walks, affinity, signal"
)
STEPLESS = "the graph of the configuration states no time/steps, which is active under what it states"
UNKNOWN = "the entry 0 of the pipeline names the operation graphspec.nope, and no operation is registered under that name"
GHOST = (
    "the model of the family snapshot states options/realisation ghost, and the realisations of options registered in "
    "tuning.REALISATIONS are %s"
)
MODELLESS = "the configuration states no model"
HIDDENLESS = "the model of the configuration states no hidden, which is active under what it states"
UNSTATED_REALISATION = (
    "the model of the family snapshot states no options/realisation, which a model of a family that reads options "
    "states on a bank that decides: none or a realisation registered in tuning.REALISATIONS"
)
UNLISTED = "a pipeline is a list of its operations in the order they apply, found 'graphspec.nodes'"
LACKING = "the statement lacks alike"
UNMAPPED = "the statement is read from a mapping of its fields, found [1, 2]"
FAULTY = "a fault in the code of an operation"
KEYLESS = "a lookup of the library that fails"


@timed("descriptions.faulted")
def faulted(graph, registry, settings, seed, limits, carry, held):
    raise ValueError(FAULTY)


FAULT = Operation("test", "fault", (Phase("fault", faulted, None, None),), unstated, (), (), (), ("facts",), None, None, 1, 1)


@timed("descriptions.keyless")
def keyless(*args):
    raise KeyError(KEYLESS)


@timed("descriptions.without")
def without(held, name):
    return {key: value for key, value in held.items() if key != name}


def test_a_trial_is_not_described_only_for_what_the_checks_of_its_statement_and_its_configuration_refuse(
    bank, searching, tmp_path, monkeypatch
):
    started = time.perf_counter()
    good = dict(modelled(bank, "vector"), pipeline=[entry("graphspec.nodes", NODES)])
    model = good["model"]
    refused = [
        (dict(good, model=without(model, "view")), VIEWLESS),
        (dict(good, graph=without(good["graph"], "time/steps")), STEPLESS),
        (dict(good, pipeline=[entry("graphspec.nope", {})]), UNKNOWN),
        (dict(good, model=dict(model, **{"options/realisation": "ghost"})), GHOST % ", ".join(REALISATIONS.names())),
        (without(good, "model"), MODELLESS),
        (dict(good, model=without(model, "hidden")), HIDDENLESS),
        (dict(good, model=without(model, "options/realisation")), UNSTATED_REALISATION),
        (dict(good, pipeline="graphspec.nodes"), UNLISTED),
    ]
    for config, message in refused:
        with pytest.raises(Refused) as raised:
            reach(bank, config, STATED)
        assert str(raised.value) == message
    folder = laid(tmp_path / "search", bank, [good] + [config for config, _ in refused] + [good, good], ())
    write_json(folder / trial(9) / "statement.json", without(STATED, "alike"))
    write_json(folder / trial(10) / "statement.json", [1, 2])
    record = reach_search(bank, folder)
    assert [row["reason"] for row in record["trials"]] == [None] + [message for _, message in refused] + [LACKING, UNMAPPED]
    assert [row["variant"] is None for row in record["trials"]] == [False] + [True] * 10
    assert len(reaches.text(record).split("\n")) == 5 + len(record["trials"]) + 1
    OPERATIONS.register(FAULT.name, FAULT)
    try:
        faulty = dict(good, pipeline=good["pipeline"] + [entry("test.fault", {})])
        with pytest.raises(ValueError, match="^%s$" % FAULTY) as raised:
            reach_over(bank, [good, faulty], STATED)
        assert type(raised.value) is ValueError
    finally:
        del OPERATIONS.held[FAULT.name]
    monkeypatch.setattr(reaching, "realising", keyless)
    with pytest.raises(KeyError, match=KEYLESS):
        reach_over(bank, [good], STATED)
    LOG.info("descriptions a trial's own exit seconds=%.3f", time.perf_counter() - started)


SCHEMA_READ = [
    ["column type", "number"],
    ["encoder", "value"],
    ["kind of target", "decision"],
    ["kind of target", "number"],
    ["operation", "graphspec.difference"],
    ["rule for ties", "key"],
    ["sort of reference", "node"],
    ["sort of reference", "nodes"],
]
VARIANT_READ = sorted(
    SCHEMA_READ
    + [
        ["choice among several facts", "last"],
        ["choice of steps", "changes"],
        ["operation", "graphspec.nodes"],
        ["operation", "graphspec.relation_forms"],
        ["operation", "test.cut"],
    ]
)
REACH_READ = sorted(VARIANT_READ + [["realisation of options", "vector"]])
ALL = ("truth", "between", "variant", "reach", "row", "table", "against")
VARIANTS = ALL[2:]
REACHES = ("reach", "row", "table")


def every_key(bank, configs):
    started = time.perf_counter()
    manifest = read_json(bank / "manifest.json")
    (deriver, name, key, unrouted), (other, _, second, _) = (reaching.named(bank, {}, config, STATED) for config in configs)
    stated = [("only", configs[0], (deriver, name, key, unrouted), None), ("none", None, None, "absent")]
    result = dict(
        truth=describing.keyed(manifest, located(bank, "bank", "truth")),
        between=differing.keyed(manifest, "g0", "g1", located(bank, "bank", "differences")),
        variant=varying.keyed(deriver, configs[0], key, located(bank, "bank", "variants")),
        reach=reaching.keyed(deriver, configs[0], key, name, unrouted, located(bank, "bank", "reach")),
        row=reaches.row_keyed(deriver, configs[0], key, name, located(bank, "bank", "reach_rows")),
        table=reaches.keyed(stated, located(bank, "bank", "reach_rows"), located(bank, "search", "reaches")),
        against=diverging.keyed([(deriver, key), (other, second)], configs, located(bank, "bank", "differences")),
    )
    LOG.debug("descriptions every key exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@contextmanager
def raised(held):
    started = time.perf_counter()
    name = "VERSION" if hasattr(held, "VERSION") else "version"
    own, before = name in vars(held), getattr(held, name)
    setattr(held, name, before + 1)
    try:
        yield
    finally:
        if own:
            setattr(held, name, before)
        else:
            delattr(held, name)
        LOG.debug("descriptions raised exit version=%d seconds=%.4f", before, time.perf_counter() - started)


def test_the_key_of_every_description_holds_the_version_of_every_unit_and_entry_its_code_runs_and_changes_with_each(
    bank, cutting
):
    started = time.perf_counter()
    labelled = {registry.label: registry for registry in registries.REGISTRIES}
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    configs = (realising(bank, CUTTING, "vector"), realising(bank, SEVERING, "vector"))
    before = every_key(bank, configs)
    key, second = keyed(bank, configs[0]), keyed(bank, configs[1])

    @timed("descriptions.every_key.code")
    def code(read):
        return dict(
            description=describing.VERSION,
            views=viewing.VERSION,
            derivation=transform.VERSION,
            representations=representing.VERSION,
            entries=[[label, name, registries.version(labelled[label].named(name))] for label, name in read],
        )

    @timed("descriptions.every_key.based")
    def based(part, beside):
        return dict(part=part, beside=beside, away=None, stored=stored)

    row = dict(based("reach_rows", "bank"), variant=key, realisation=realised("vector"), code=code(REACH_READ))
    assert before == dict(
        truth=dict(based("truth", "bank"), code=code(SCHEMA_READ)),
        between=dict(based("differences", "bank"), first=dict(graph="g0"), second=dict(graph="g1"), code=code(SCHEMA_READ)),
        variant=dict(based("variants", "bank"), variant=key, code=code(VARIANT_READ)),
        reach=dict(based("reach", "bank"), variant=key, realisation=realised("vector"), refused=None, code=code(REACH_READ)),
        row=row,
        table=dict(
            based("reaches", "search"),
            trials=[
                dict(trial="only", row=sha256_json(row), refused=None, reason=None),
                dict(trial="none", row=None, refused=None, reason="absent"),
            ],
            code=code(REACH_READ),
        ),
        against=dict(based("differences", "bank"), first=dict(variant=key), second=dict(variant=second), code=code(VARIANT_READ)),
    )
    units = [
        (describing, ALL),
        (viewing, ALL),
        (transform, ALL),
        (representing, ALL),
        (OPERATIONS.named("graphspec.difference"), ALL),
        (labelled["encoder"].named("value"), ALL),
        (labelled["sort of reference"].named("nodes"), ALL),
        (OPERATIONS.named("graphspec.nodes"), VARIANTS),
        (CUT, VARIANTS),
        (REALISATIONS.named("vector"), REACHES),
        (labelled["encoder"].named("standard"), ()),
        (OPERATIONS.named("graphspec.window"), ()),
        (REALISATIONS.named("node"), ()),
    ]
    for held, changed in units:
        with raised(held):
            after = every_key(bank, configs)
        assert tuple(kind for kind in ALL if after[kind] != before[kind]) == changed, held
    assert every_key(bank, configs) == before
    LOG.info("descriptions keys exit units=%d seconds=%.3f", len(units), time.perf_counter() - started)


def test_every_check_of_a_statement_a_configuration_a_pipeline_and_a_realisation_refuses_with_the_one_class(bank, searching):
    started = time.perf_counter()
    held, stated = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT), state(STATED)
    config = modelled(bank, "vector")
    graph, turned = config["graph"], dict(trained=["validation"], scored=["train"], sees=dict(train=[], validation=[]))
    absent = dict(SPLITS, scored=["absent"], sees=dict(train=[], absent=[]))
    cases = [
        (lambda: state(without(STATED, "alike")), LACKING),
        (lambda: state(dict(STATED, colour="red")), "the statement holds fields it does not know: colour"),
        (lambda: state(dict(STATED, seed=-1)), "the statement's seed is -1 and must be an integer from 0 to 4294967295"),
        (lambda: state([1, 2]), UNMAPPED),
        (lambda: agreed([1, 2], stated), "a configuration is a mapping of its fields, found [1, 2]"),
        (lambda: agreed(without(config, "seed"), stated), "the configuration states no seed"),
        (lambda: agreed(dict(config, seed=6), stated), SEEDED),
        (lambda: agreed(without(config, "splits"), stated), "the configuration states no splits to train on and to score on"),
        (lambda: held.stated(without(config, "graph")), "the configuration states no graph"),
        (lambda: held.stated(without(without(config, "graph"), "pipeline")), "the configuration states no pipeline and no graph"),
        (lambda: held.stated(dict(config, graph=dict(graph, **{"time/history": 0}))), "time/history must be an integer in 1..32, found 0"),
        (
            lambda: held.stated(dict(config, graph=dict(graph, colour=1))),
            "the graph of the configuration states colour, which are inactive or unknown",
        ),
        (
            lambda: held.stated(dict(config, pipeline=[dict(operation="graphspec.nodes", parameters=[1])])),
            "the entry graphspec.nodes of the pipeline states its parameters as a mapping, found [1]",
        ),
        (
            lambda: held.looking(dict(config, splits=absent)),
            "the splits scored on name absent, and the bank holds the splits train, validation",
        ),
        (
            lambda: held.looking(dict(config, splits=turned)),
            "the scaling of the targets of the bank was fitted on the splits train, and the fit trains on validation",
        ),
        (lambda: held.viewed(config).second(held, 0, "absent"), "the bank holds no graph named absent to show beside graph g0"),
        (lambda: routed(held, modelled(bank, "aimed")), AIMLESS),
        (lambda: routed(held, without(config, "model")), MODELLESS),
    ]
    for call, message in cases:
        with pytest.raises(Refused) as raised:
            call()
        assert str(raised.value) == message
    LOG.info("descriptions the one class exit cases=%d seconds=%.3f", len(cases), time.perf_counter() - started)


@timed("descriptions.plain")
def plain(value):
    return np.asarray(value.value if isinstance(value, Host) else value)


def grouped(held, cfg, partition):
    started = time.perf_counter()
    blocks = windows(held, held.blocks(partition, cfg["graph"]["time/history"], cfg), cfg)
    result = batching.batches(blocks, len(blocks), held.sizes(cfg), None, frame_budget=1024) if blocks else []
    LOG.debug("descriptions grouped exit partition=%d batches=%d seconds=%.4f", partition, len(result), time.perf_counter() - started)
    return result


def batched(held, cfg, name, partition):
    started = time.perf_counter()
    found, registry, schema = offered(held), held.fitted(cfg), held.registry.schema
    columns, kinds, reads = schema.parameters(), list(schema.tasks["kinds"]), REALISATIONS.named(name).reads
    fields = {task: [field for field in heads.fielded(schema, task) if field["sort"] in reads] for task in schema.decided()}
    stated, kept = set(), {}
    for group in grouped(held, cfg, partition):
        b = batching.recipe_batch(held, group, cfg)
        rows = np.r_[plain(b["rows"]), plain(b["drawn"])][plain(b["placed"])]
        decided = dict(zip(plain(b["point_row"]).tolist(), found.decision[rows].tolist()))
        for decision in {decision for decision in decided.values() if decision >= 0}:
            stated |= set(range(int(found.first[decision]), int(found.first[decision] + found.count[decision])))
        options = b["options"]
        owner, acted, values = (plain(options[part]) for part in ("action_option", "action_kind", "action_values"))
        roled, codes = plain(options["role_action"]), plain(options["role_code"])
        for at, (row, content) in enumerate(zip(plain(options["row"]).tolist(), plain(options["content"]).tolist())):
            first, count = int(found.first[decided[row]]), int(found.count[decided[row]])
            (option,) = [first + place for place in range(count) if int(found.option["content"][first + place]) == content]
            acts = np.flatnonzero(owner == at)
            aimed, held_fields = codes[np.isin(roled, acts)], set()
            for field in fields[kinds[int(found.option["kind"][option])]]:
                if field["sort"] == "kind":
                    holds = len(acts) > 0
                elif field["sort"] == "targets":
                    holds = bool((aimed == field["code"]).any())
                else:
                    assert registry.encoding.sizes[field["column"]][1] == 1
                    flag = slots(registry.encoding, columns, field["column"])[-1]
                    holds = bool(((acted[acts] == field["code"]) & (values[acts, flag] > 0)).any())
                if holds:
                    held_fields.add(field["name"])
            kept[option] = held_fields
    LOG.debug("descriptions batched exit options=%d held=%d seconds=%.4f", len(stated), len(kept), time.perf_counter() - started)
    return stated, kept


def branched(held, cfg, partition):
    started = time.perf_counter()
    registry, result = held.fitted(cfg), {}
    with forks.Forks(held):
        for group in grouped(held, cfg, partition):
            kept = set(plain(batching.recipe_batch(held, group, cfg)["options"]["option"]).tolist())
            for part in [part for part, *_ in batching.parted(held, group, cfg) if part.option >= 0]:
                fork, seen = held.resident(part, cfg, "cpu"), {}
                entity, tokens = fork.numpy("entity").astype(np.int64), fork.numpy("entity_token")
                rows = np.flatnonzero(fork.numpy("spatial_step") == part.step)
                for at, (source, target, code) in zip(rows.tolist(), fork.numpy("spatial")[rows].tolist()):
                    place = int(registry.value_group[code])
                    columns = registry.attributes(registry.valued[place]) if place >= 0 else ()
                    table = fork.numpy("attributes_" + registry.valued[place]) if place >= 0 else None
                    values = table[int(fork.numpy("spatial_attribute_row")[at])] if place >= 0 else ()
                    key = (int(tokens[entity[source]]), int(tokens[entity[target]]), code)
                    seen.setdefault(key, set()).update(column for column, value in zip(columns, values) if not np.isnan(value))
                history = forks.trunk(held, cfg, part.graph, part.step)
                plan = forks.planned(held, cfg, history, part, forks.ordered(held, part.option))
                marks = np.r_[reaching.masked(plan.graph.entity_token), reaching.masked([made for made, _ in plan.made])].tolist()
                facts = []
                for code, first, second, *_ in plan.facts:
                    ends = (marks[first], marks[second])
                    keys = [(*ends, code)] + ([(*ends[::-1], code)] if registry.by_code[code].symmetric else [])
                    shown = [seen[key] for key in keys if key in seen]
                    facts.append((bool(shown), sorted(set().union(*shown))))
                result[part.option] = part.option in kept, facts
    LOG.debug("descriptions branched exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def said(held, cfg, name):
    started = time.perf_counter()
    told = reaching.Reached(held, cfg, name)
    last, result = len(told.steps), {}
    for group in held.graphs(cfg):
        for decision in told.decided(told.derived(None, last, group), group):
            for option in told.optioned(decision):
                found = told.read(None, last, decision, option)
                put = {field for field, entry in found["fields"].items() if entry is not None and entry["put"]}
                facts = None
                if told.entry.back:
                    facts = [(fact["held"], sorted(fact["valued"])) for fact in told.edits(None, last, decision, option)[0]]
                result[option] = found["held"], put, facts
            told.decision_done()
        told.group_done()
    LOG.debug("descriptions said exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def agreeing(held, cfg, name, partitions):
    started = time.perf_counter()
    told, counts = said(held, cfg, name), dict(stated=0, held=0, put=0, facts=0, shown=0)
    for partition in partitions:
        if REALISATIONS.named(name).back:
            packed = branched(held, cfg, partition)
            assert {option: (told[option][0], told[option][2]) for option in packed} == packed
            stated, kept = set(packed), {option for option, (found, _) in packed.items() if found}
            counts["facts"] += sum(len(facts) for _, facts in packed.values())
            counts["shown"] += sum(found for _, facts in packed.values() for found, _ in facts)
        else:
            stated, kept = batched(held, cfg, name, partition)
            assert {option for option in stated if told[option][0]} == set(kept)
            assert {option: told[option][1] for option in kept} == kept
        counts["stated"] += len(stated)
        counts["held"] += len(kept)
        counts["put"] += sum(len(told[option][1]) for option in kept)
    LOG.info("descriptions agreeing exit realisation=%s counts=%s seconds=%.3f", name, counts, time.perf_counter() - started)
    return counts


EDGED = {
    "edges/direction": "inverse",
    "edges/near/drop": 0.0,
    "edges/near/persistence": 1,
    "edges/near/cap": 0,
    "edges/base/drop": 0.0,
    "edges/base/persistence": 2,
    "edges/base/cap": 0,
}
LASTING = [dict(operation="graphspec.edges", parameters=EDGED)]
BESIDE_HELD = dict(stated=13, held=12, put=24, facts=0, shown=0)
PACKED = (
    ("vector", None, BESIDE_HELD),
    ("action_nodes", None, BESIDE_HELD),
    ("actor_edge", None, BESIDE_HELD),
    ("set_edge", None, BESIDE_HELD),
    ("aimed", None, dict(BESIDE_HELD, put=12)),
    ("grounded", LASTING, dict(stated=13, held=10, put=12, facts=17, shown=12)),
)


def test_reach_says_of_every_option_and_field_what_the_batches_and_forks_a_fit_packs_hold_under_each_realisation(
    bank, searching
):
    started = time.perf_counter()
    held = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    for name, pipeline, counts in PACKED:
        model = dict(MODEL, **{"options/realisation": name}, **({"options/depth": 1} if REALISATIONS.named(name).structure else {}))
        stated = pipelined(CUTTING) if pipeline is None else pipeline
        config = configuration(bank, model, None, pipeline=stated, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
        assert agreeing(held, config, name, (0, 1)) == counts, name
    LOG.info("descriptions reach against the packing exit seconds=%.3f", time.perf_counter() - started)


def test_reach_over_a_search_stops_naming_a_folder_without_a_plan_a_plan_of_another_bank_and_a_set_that_is_no_list(
    bank, path_bank, tmp_path, capsys, monkeypatch
):
    from tuning.__main__ import main

    started = time.perf_counter()
    stored = read_json(bank / "manifest.json")["stored_sha256"]
    with pytest.raises(FileNotFoundError, match="holds no plan.json; reach over a search reads the folder of a search"):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(trials=[]))
    with pytest.raises(ValueError, match="^the plan.json of .* lacks bank_stored_sha256$"):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(bank_stored_sha256="0" * 64, trials=[]))
    with pytest.raises(ValueError, match="searches the bank whose build stored 0{64}, and the bank .* stored %s$" % stored):
        reach_search(bank, tmp_path)
    write_json(tmp_path / "plan.json", dict(bank_stored_sha256=stored, trials=[]))
    empty = dict(version=describing.VERSION, stored_sha256=stored, fields=dict(turn=list(GO)), trials=[])
    assert reach_search(bank, tmp_path) == dict(empty, kept=kept_as("reaches", "search"))
    assert reach_over(bank, [], STATED) == dict(empty, kept=kept_as("reaches", "bank"))
    with pytest.raises(TypeError, match="reads a list of the configurations of fits, found {}"):
        reach_over(bank, {}, STATED)
    with pytest.raises(Refused, match="^the statement lacks alike$"):
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
NODAL = ("first", "second", "first_anchors", "second_anchors", "valued")
EDGED = ("first", "second", "valued")
UNCOLUMNED = dict(kinds={}, relations={})
SWAPPED = dict(first="second", second="first", first_anchors="second_anchors", second_anchors="first_anchors")
UNSHARED = dict(first=[], second=[])


@timed("descriptions.unlike")
def unlike(labels, stated, totals, *held):
    return dict(stated=stated, **dict(zip(labels, totals)), at=dict(zip(labels, held, strict=True)))


ANCHORED = [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0], [5, 1], [6, 0]]
LEANER = dict(
    name="g0",
    view=VIEWED["g0"],
    times=[0, 1, 2, 3, 4, 5, 6, 7],
    alone=UNSHARED,
    kinds=dict(agent=unlike(NODAL, "both", (11, 0, 0, 3, 0), BASES, NIL, NIL, ANCHORED, NIL)),
    relations={
        "base": unlike(EDGED, "both", (6, 0, 0), BASES_CUT, NIL, NIL),
        LOOPS: unlike(EDGED, "both", (11, 0, 0), BASES, NIL, NIL),
    },
    equal=dict(kinds=1, relations=1),
)
LEANER_SECOND = dict(
    name="g1",
    view=VIEWED["g1"],
    times=[1, 2, 3],
    alone=UNSHARED,
    kinds=dict(agent=unlike(NODAL, "both", (2, 0, 0, 1, 0), [[0, 1], [2, 0]], NIL, NIL, [[0, 0], [1, 1], [2, 0]], NIL)),
    relations={
        "base": unlike(EDGED, "both", (1, 0, 0), [[0, 1], [1, 0]], NIL, NIL),
        LOOPS: unlike(EDGED, "both", (2, 0, 0), [[0, 1], [2, 0]], NIL, NIL),
    },
    equal=dict(kinds=1, relations=1),
)
TRUTHS = dict(
    first="g0",
    second="g1",
    unit="tick",
    differs=True,
    times=[None, 0, 1, 2, 3, 4, 5, 6, 7],
    kinds=dict(
        agent=unlike(NODAL, "both", (8, 1, 0, 0, 0), [[None, 0], [0, 1]], [[None, 1], [0, 0]], UNHELD, UNHELD, UNHELD),
        spot=unlike(
            NODAL,
            "both",
            (9, 1, 0, 0, 0),
            [[None, 0], [0, 1], [1, 0], [2, 1], [3, 2], [5, 1]],
            [[None, 1], [0, 0]],
            UNHELD,
            UNHELD,
            UNHELD,
        ),
    ),
    relations=dict(
        near=unlike(EDGED, "both", (8, 0, 0), [[None, 0], [0, 1], [2, 2], [4, 1], [6, 0]], UNHELD, UNHELD),
        base=unlike(
            EDGED, "both", (10, 3, 0), [[None, 0], [0, 1], [4, 2], [6, 1]], [[None, 1], [1, 0], [2, 1], [3, 0]], UNHELD
        ),
    ),
    equal=dict(kinds=0, relations=0),
)
SIDE_TEXT = """derivation seed 7, device cpu
splits trained on train, scored on validation; rules of what each sees: train none, validation none
settings: graphs/join separate, nodes/untyped False, relations/direction inverse, time/steps changes, time/several last
graphs of the bank the view does not show: none
entry graphspec.nodes, of the operation graphspec.nodes at version %d: parameters nodes/agent/keep True, nodes/spot/keep True, nodes/referenced keep, nodes/agent/filter degree, nodes/agent/min_degree %d, nodes/spot/filter none; reads none; splits none; storage none; second graph none
entry test.cut, of the operation test.cut at version 1: parameters cut/relation base, cut/kind turn; reads none; splits none; storage none; second graph none
entry graphspec.relation_forms, of the operation graphspec.relation_forms at version %d: parameters relation_forms/types typed, relation_forms/parallel keep, relation_forms/reverse False, relation_forms/loops True; reads none; splits none; storage none; second graph none"""
HEAD_TEXT = """difference of the variants %s, the first, and %s, the second, of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/differences
%s
the first:
%s
the second:
%s
columns of values only one side holds: none
"""
DIFFERENCE = """
graph g0, the graph g0 as the tasks of the splits train, validation see it, times 8 from 0 to 7; times only the first shows: none; times only the second shows: none
  kind agent: nodes only the first holds 11, only the second 0; anchors only the first holds 0, only the second 3; nodes both hold with other values 0
    time  first nodes  second nodes  first anchors  second anchors  other values
    0     1            0             0              0               0
    1     2            0             0              1               0
    2     1            0             0              0               0
    3     1            0             0              1               0
    4     2            0             0              0               0
    5     2            0             0              1               0
    6     1            0             0              0               0
    7     1            0             0              0               0
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

graph g1, the graph g1 as the tasks of the splits train, validation see it, times 3 from 1 to 3; times only the first shows: none; times only the second shows: none
  kind agent: nodes only the first holds 2, only the second 0; anchors only the first holds 0, only the second 1; nodes both hold with other values 0
    time  first nodes  second nodes  first anchors  second anchors  other values
    1     1            0             0              0               0
    2     1            0             0              1               0
  relation base: edges only the first holds 1, only the second 0, held by both with other values 0
    time  first  second  other values
    1     1      0       0
  relation relation_forms.loops: edges only the first holds 2, only the second 0, held by both with other values 0
    time  first  second  other values
    1     1      0       0
    2     1      0       0
  equal at every time: kinds 1, relations 1
"""
SAME_TEXT = """
graph g0, the graph g0 as the tasks of the splits train, validation see it, times 8 from 0 to 7; times only the first shows: none; times only the second shows: none
  nothing differs of what is compared; equal at every time: kinds 2, relations 3

graph g1, the graph g1 as the tasks of the splits train, validation see it, times 3 from 1 to 3; times only the first shows: none; times only the second shows: none
  nothing differs of what is compared; equal at every time: kinds 2, relations 3
"""
TRUTHS_TEXT = """difference of the truth graphs g0, the first, and g1, the second, of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/differences
unit tick, times 8 from 0 to 7
%s
  kind agent: nodes only the first holds 8, only the second 1; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    time    first nodes  second nodes  first anchors  second anchors  other values
    before  0            1             0              0               0
    0       1            0             0              0               0
    1       1            0             0              0               0
    2       1            0             0              0               0
    3       1            0             0              0               0
    4       1            0             0              0               0
    5       1            0             0              0               0
    6       1            0             0              0               0
    7       1            0             0              0               0
  kind spot: nodes only the first holds 9, only the second 1; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    time    first nodes  second nodes  first anchors  second anchors  other values
    before  0            1             0              0               0
    0       1            0             0              0               0
    2       1            0             0              0               0
    3       2            0             0              0               0
    4       2            0             0              0               0
    5       1            0             0              0               0
    6       1            0             0              0               0
    7       1            0             0              0               0
  relation near: edges only the first holds 8, only the second 0, held by both with other values 0
    time  first  second  other values
    0     1      0       0
    1     1      0       0
    2     2      0       0
    3     2      0       0
    4     1      0       0
    5     1      0       0
  relation base: edges only the first holds 10, only the second 3, held by both with other values 0
    time    first  second  other values
    before  0      1       0
    0       1      1       0
    1       1      0       0
    2       1      1       0
    3       1      0       0
    4       2      0       0
    5       2      0       0
    6       1      0       0
    7       1      0       0
  equal at every time: kinds 0, relations 0
"""
ALONE_TEXT = """difference of the truth graphs g1, the first, and g1, the second, of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/differences
unit tick, times 3 from 1 to 3
%s
  nothing differs of what is compared; equal at every time: kinds 2, relations 2
"""
TIMELESS_TEXT = """unit tick, no time
%s
  kind agent: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    time    first nodes  second nodes  first anchors  second anchors  other values
    before  1            0             0              0               0
  kind spot: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    time    first nodes  second nodes  first anchors  second anchors  other values
    before  1            0             0              0               0
  relation near: edges only the first holds 2, only the second 0, held by both with other values 0
    time    first  second  other values
    before  2      0       0
  relation base: edges only the first holds 1, only the second 0, held by both with other values 0
    time    first  second  other values
    before  1      0       0
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
            unshown=[],
            entries=entered(leaning(degree)),
        )
        for degree in degrees
    ]
    record = dict(
        version=describing.VERSION,
        stored_sha256=stored,
        time=CLOCKED,
        first=sides[0],
        second=sides[1],
        columns=UNCOLUMNED,
        differs=differs,
        graphs=graphs,
        kept=kept_as("differences", "bank"),
    )
    versions = [OPERATIONS.named(name).version for name in ("graphspec.nodes", "graphspec.relation_forms")]
    written = HEAD_TEXT % (
        sides[0]["variant"],
        sides[1]["variant"],
        stored,
        describing.VERSION,
        differing.READING["clock"],
        *[SIDE_TEXT % (versions[0], degree, versions[1]) for degree in degrees],
    )
    configs = [leaned(bank, degree) for degree in degrees]
    key = diverging.keyed([(deriver, side["variant"]) for side in sides], configs, located(bank, "bank", "differences"))
    assert (key["part"], key["first"], key["second"]) == ("differences", *[dict(variant=side["variant"]) for side in sides])
    return record, written, bank / "descriptions" / "differences" / describing.named(key)


@timed("descriptions.truths")
def truths(bank, first, second, found):
    manifest, place = read_json(bank / "manifest.json"), located(bank, "bank", "differences")
    stored, key = manifest["stored_sha256"], differing.keyed(manifest, first, second, place)
    assert key == dict(describing.keyed(manifest, place), first=dict(graph=first), second=dict(graph=second))
    return (
        dict(version=describing.VERSION, stored_sha256=stored, time=CLOCKED, kept=kept_as("differences", "bank"), **found),
        bank / "descriptions" / "differences" / describing.named(key),
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
    mirrored, _, stem = diverged(bank, (2, 1), True, [turned_over(LEANER), turned_over(LEANER_SECOND)])
    kept = dict(record["kept"], file=stem.with_suffix(".json").name)
    assert against(bank, second, STATED, first, STATED) == dict(turned_over(record), kept=kept) == mirrored
    assert kept["file"] != record["kept"]["file"]
    turned = dict(first=NIL, second=BASES, first_anchors=ANCHORED, second_anchors=NIL, valued=NIL)
    assert turned_over(LEANER)["kinds"]["agent"]["at"] == turned
    assert held_at(turned["second"], [1]) == [2] and held_at(turned["first_anchors"], [1]) == [1]
    same = [
        dict(
            name=name, view=VIEWED[name], times=times, alone=UNSHARED, kinds={}, relations={}, equal=dict(kinds=2, relations=3)
        )
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


CORED = dict(MEASURED, **{"measures/degree": False, "measures/cores": True})
UNNEARED = {"relations/near/keep": False, "relations/base/keep": True}
COLUMNS_TEXT = (
    "\ncolumns of values only one side holds: kind agent, the first measures.degree, the second measures.cores; kind "
    "spot, the first measures.degree, the second measures.cores\n"
)
EQUAL_TEXT = "  nothing differs of what is compared; equal at every time: kinds 2, relations 2\n"
UNLIKE_VALUES = """  kind spot: nodes only the first holds 0, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 14
    time  first nodes  second nodes  first anchors  second anchors  other values
    0     0            0             0              0               2
    1     0            0             0              0               2
    2     0            0             0              0               3
    3     0            0             0              0               3
    4     0            0             0              0               2
    5     0            0             0              0               2
"""
UNCOMPARED = ("periods", "tasks", "targets", "options", "negatives")


@timed("descriptions.measuring")
def measuring(bank, *entries):
    pipeline = [dict(operation=operation, parameters=parameters) for operation, parameters in entries]
    return configuration(bank, MODEL, None, pipeline=pipeline, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)


def test_two_variants_differ_by_the_columns_of_values_one_alone_holds_and_by_the_nodes_both_hold_with_other_values(bank):
    started = time.perf_counter()
    degree = measuring(bank, ("graphspec.measures", MEASURED))
    record = against(bank, degree, STATED, measuring(bank, ("graphspec.measures", CORED)), STATED)
    only = dict(first=["measures.degree"], second=["measures.cores"])
    assert record["differs"] and record["columns"] == dict(kinds=dict(agent=only, spot=only), relations={})
    found = [(graph["kinds"], graph["relations"], graph["equal"]) for graph in record["graphs"]]
    assert found == [({}, {}, dict(kinds=2, relations=2))] * 2
    written = diverging.text(record)
    assert COLUMNS_TEXT in written and written.count(EQUAL_TEXT) == 2
    unneared = measuring(bank, ("graphspec.relations", UNNEARED), ("graphspec.measures", MEASURED))
    record = against(bank, degree, STATED, unneared, STATED)
    first, second = record["graphs"]
    assert record["differs"] and record["columns"] == UNCOLUMNED
    unlike_values = [[0, 2], [2, 3], [4, 2], [6, 0]]
    assert first["kinds"] == dict(spot=unlike(NODAL, "both", (0, 0, 0, 0, 14), NIL, NIL, NIL, NIL, unlike_values))
    assert first["relations"] == dict(near=unlike(EDGED, "both", (8, 0, 0), NEARS, NIL, NIL))
    assert (first["equal"], second["kinds"], second["relations"]) == (dict(kinds=1, relations=1), {}, {})
    assert second["equal"] == dict(kinds=2, relations=2) and UNLIKE_VALUES in diverging.text(record)
    for legend in (effect.READING, differing.READING["clock"], differing.TRUTH["clock"]):
        assert [word for word in UNCOMPARED if word not in legend] == []
    sized = [Mock(encoding=Mock(table=("load",), sizes=dict(load=held))) for held in ((1, 1), (2, 1))]
    with pytest.raises(ValueError, match="^the first variant encodes the column load as 1 values and 1 flags and the second as 2 and 1; "):
        diverging.shared(sized)
    LOG.info("descriptions difference of values exit seconds=%.3f", time.perf_counter() - started)


def test_two_truth_graphs_of_a_bank_differ_by_what_is_written_out_here_and_a_graph_beside_itself_in_nothing(bank):
    started = time.perf_counter()
    expected, _ = truths(bank, "g0", "g1", TRUTHS)
    record = between(bank, "g0", "g1")
    assert record == expected and json.loads(json.dumps(record)) == record
    assert between(bank, "g1", "g0") == turned_over(expected)
    turned = turned_over(expected)["relations"]["base"]["at"]
    assert turned == dict(first=[[None, 1], [1, 0], [2, 1], [3, 0]], second=[[None, 0], [0, 1], [4, 2], [6, 1]], valued=UNHELD)
    assert [held_at(turned[label], [None, 0, 1]) for label in EDGED] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    alone = dict(first="g1", second="g1", unit="tick", differs=False, times=[None, 1, 2, 3], kinds={}, relations={})
    alone, stem = truths(bank, "g1", "g1", dict(alone, equal=dict(kinds=2, relations=2)))
    assert between(bank, "g1", "g1") == alone
    written = ALONE_TEXT % (alone["stored_sha256"], describing.VERSION, differing.TRUTH["clock"])
    assert differing.text(alone) == written == stem.with_suffix(".txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="holds no graph named g9; its graphs are g0, g1$"):
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
    assert read_json(stem.with_suffix(".json"))["record"] == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == head + DIFFERENCE
    assert record["kept"]["file"] == stem.with_suffix(".json").name
    names = ("config.json", "statement.json", "other.json")
    for name, held in zip(names, (first, STATED, second)):
        write_json(tmp_path / name, held)
    files = [str(tmp_path / names[place]) for place in (0, 1, 2, 1)]
    monkeypatch.setattr(sys, "argv", ["graphspec", "describe", "--difference", str(bank), *files])
    main()
    assert capsys.readouterr().out == head + DIFFERENCE
    record = between(bank, "g0", "g1")
    expected, stem = truths(bank, "g0", "g1", TRUTHS)
    written = TRUTHS_TEXT % (expected["stored_sha256"], describing.VERSION, differing.TRUTH["clock"])
    assert differing.text(record) == written
    assert read_json(stem.with_suffix(".json"))["record"] == record == expected
    assert stem.with_suffix(".txt").read_text(encoding="utf-8") == written
    assert record["kept"]["file"] == stem.with_suffix(".json").name
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
        relations=dict(near=unlike(EDGED, "both", (0, 0, 3), NIL, NIL, [[0, 0], [3, 1], [6, 0]])),
        equal=dict(kinds=2, relations=1),
    )
    still = [timeless(facts) for facts in held]
    monkeypatch.setattr(describing, "stored", lambda path, graph: still[graph["index"]])
    record = differing.recorded(bank, manifest, *manifest["graphs"])
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=CLOCKED,
        first="g0",
        second="g1",
        unit="tick",
        differs=True,
        times=[None],
        kinds=dict(
            agent=unlike(NODAL, "both", (1, 0, 0, 0, 0), [[None, 1]], UNHELD, UNHELD, UNHELD, UNHELD),
            spot=unlike(NODAL, "both", (1, 0, 0, 0, 0), [[None, 1]], UNHELD, UNHELD, UNHELD, UNHELD),
        ),
        relations=dict(
            near=unlike(EDGED, "both", (2, 0, 0), [[None, 2]], UNHELD, UNHELD),
            base=unlike(EDGED, "both", (1, 0, 0), [[None, 1]], UNHELD, UNHELD),
        ),
        equal=dict(kinds=0, relations=0),
    )
    kept = dict(record, kept=kept_as("differences", "bank"))
    assert differing.text(kept).split("\n", 2)[2] == TIMELESS_TEXT % differing.TRUTH["clock"]
    config = leaned(bank, 1)
    name, graph, grown, low, high = diverging.ended(Deriver(bank, "cpu", STATED["derivation_seed"], RESIDENT), config)[0]
    assert (name, low, high) == ("g0", 0, 8)
    assert diverging.paired((name, graph, grown, 0, 8), (name, graph, grown, 1, 7)) == dict(
        name="g0",
        times=[0, 1, 2, 3, 4, 5, 6, 7],
        alone=dict(first=[0, 7], second=[]),
        kinds=dict(
            agent=unlike(NODAL, "both", (2, 0, 0, 0, 0), [[0, 1], [1, 0], [7, 1]], NIL, NIL, NIL, NIL),
            spot=unlike(NODAL, "both", (4, 0, 0, 0, 0), [[0, 2], [1, 0], [7, 2]], NIL, NIL, NIL, NIL),
        ),
        relations={
            "near": unlike(EDGED, "both", (1, 0, 0), [[0, 1], [1, 0]], NIL, NIL),
            "base": unlike(EDGED, "both", (2, 0, 0), [[0, 1], [1, 0], [7, 1]], NIL, NIL),
            LOOPS: unlike(EDGED, "both", (6, 0, 0), [[0, 3], [1, 0], [7, 3]], NIL, NIL),
        },
        equal=dict(kinds=0, relations=0),
    )
    unlooped = leaning(1)[:2] + [dict(operation="graphspec.relation_forms", parameters=dict(FORMS, **{"relation_forms/loops": False}))]
    unlooped = configuration(bank, MODEL, None, pipeline=unlooped, graph=SETTINGS, seed=STATED["seed"], splits=SPLITS)
    found = against(bank, config, STATED, unlooped, STATED)["graphs"]
    assert [(graph["kinds"], graph["equal"]) for graph in found] == [({}, dict(kinds=2, relations=2))] * 2
    assert found[1]["relations"] == {LOOPS: unlike(EDGED, "first", (7, 0, 0), [[0, 3], [2, 1]], NIL, NIL)}
    assert "  relation relation_forms.loops, which only the first states: edges only the first holds 7," in "\n".join(
        differing.graph_text(found[1], describing.NAMED["clock"], kept_as("differences", "bank"))
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


RETRACTED = dict(
    table="near",
    ends=[["spot", "spot"]],
    columns=[],
    about=None,
    lapsed=0,
    edges=1,
    facts=1,
    absent=1,
    replaced=1,
    edges_held=[[None, 0], [0, 1]],
    runs=1,
    bounded=0,
    **NONE,
)
RETRACTED_TEXT = (
    "  relation near of the table near: ends spot to spot; no columns; edges 1, facts 1, absent 1, replaced 1; runs 1, bounded 0\n"
)
EITHER_WAY = dict(
    table="base",
    ends=[["agent", "spot"], ["spot", "agent"]],
    columns=[],
    about=None,
    lapsed=0,
    edges=1,
    facts=2,
    absent=0,
    replaced=0,
    edges_held=[[None, 0], [0, 1]],
    runs=1,
    bounded=0,
    **NONE,
)
THIRD = dict(
    name="h0",
    unit="tick",
    times=[0, 1, 2, 3],
    kinds=dict(
        agent=dict(nodes=1, observations=1, replaced=0, nodes_held=[[None, 0], [0, 1]], observations_held=[[None, 0], [0, 1]]),
        spot=dict(nodes=3, observations=3, replaced=0, nodes_held=[[None, 0], [0, 3]], observations_held=[[None, 0], [0, 3]]),
    ),
    relations=dict(near=RETRACTED, base=EITHER_WAY),
    tasks=dict(turn=dict(tasks=4, splits=dict(train=2, validation=2), unsplit=0)),
    decisions=dict(
        turn=dict(
            decisions=4,
            options=6,
            recorded=6,
            regenerated=0,
            generators={},
            rule=None,
            each=[
                dict(key=10, time=0, options=2, recorded=2, regenerated=0, generators={}),
                dict(key=20, time=1, options=2, recorded=2, regenerated=0, generators={}),
                dict(key=30, time=2, options=1, recorded=1, regenerated=0, generators={}),
                dict(key=40, time=2, options=1, recorded=1, regenerated=0, generators={}),
            ],
        )
    ),
    unread=dict(UNREAD, later=dict(nodes=0, edges=1, gap=4, last=4)),
)
THIRD_TEXT = """
graph h0, unit tick, times 4 from 0 to 3
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 3, observations 3, replaced 0
  relation near of the table near: ends spot to spot; no columns; edges 1, facts 1, absent 1, replaced 1; runs 1, bounded 0
  relation base of the table base: ends agent to spot, spot to agent; no columns; edges 1, facts 2, absent 0, replaced 0; runs 1, bounded 0
  held before every time and at each time:
    time                before  0  1  2  3
    agent nodes         0       1  1  1  1
    agent observations  0       1  1  1  1
    spot nodes          0       3  3  3  3
    spot observations   0       3  3  3  3
    near edges          0       1  1  1  1
    base edges          0       1  1  1  1
  shown to some tasks only: facts known later than they hold from, about nodes 0 and about edges 1, by 4 at most, the last known from time 4, before which a task is not shown them; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 4; train 2, validation 2, no split 0
  decisions of turn: decisions 4, options 6, recorded 6, regenerated 0; no generator; no rule of options
    task  time  options  recorded  regenerated
    10    0     2        2         0
    20    1     2        2         0
    30    2     1        1         0
    40    2     1        1         0
"""


def test_an_edge_without_direction_stated_from_both_ends_is_one_edge_and_a_truth_graph_holds_at_a_time_what_its_view_shows(
    crossed,
):
    started = time.perf_counter()
    record = describe(crossed)
    stored = read_json(crossed / "manifest.json")["stored_sha256"]
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=stored,
        time=dict(model="clocks", units=["tick"]),
        splits=["train", "validation"],
        graphs=[THIRD],
        kept=kept_as("truth", "bank"),
    )
    assert describing.text(record).split("\n", 5)[5] == THIRD_TEXT
    (told,) = record["graphs"]
    (shown,) = variant(crossed, lined(crossed, THINNER, KEPT, "vector"), STATED)["graphs"]
    assert shown["times"] == [0, 1, 2, 3, 4]
    steps = [shown["times"].index(moment) for moment in told["times"]]
    edges = {name: held_at(held["edges"], steps) for name, held in shown["held"]["relations"].items()}
    nodes = {name: held_at(held["nodes"], steps) for name, held in shown["held"]["kinds"].items()}
    assert nodes == {name: held_at(held["nodes_held"], told["times"]) for name, held in told["kinds"].items()}
    assert nodes == dict(agent=[1, 1, 1, 1], spot=[3, 3, 3, 3])
    assert edges["base"] == held_at(told["relations"]["base"]["edges_held"], told["times"]) == [1, 1, 1, 1]
    assert (edges["near"], held_at(told["relations"]["near"]["edges_held"], told["times"])) == ([1, 1, 1, 2], [1, 1, 1, 1])
    assert shown["held"]["relations"]["near"]["edges"] == [[0, 1], [3, 2], [4, 1]]
    LOG.info("descriptions either way exit seconds=%.3f", time.perf_counter() - started)


def test_a_fact_a_correction_replaced_holds_nothing_is_counted_apart_and_alone_tells_a_graph_from_its_uncorrected_twin(
    crossed, monkeypatch
):
    started = time.perf_counter()
    record = describe(crossed)
    (told,) = record["graphs"]
    assert told["times"] == [0, 1, 2, 3] and told["relations"]["near"] == RETRACTED
    assert [held["replaced"] for held in told["kinds"].values()] == [0, 0] and told["relations"]["base"]["replaced"] == 0
    assert RETRACTED_TEXT in describing.text(record)
    manifest = read_json(crossed / "manifest.json")
    held = describing.stored(crossed, manifest["graphs"][0])
    assert np.asarray(held.fact_replaced_time).tolist() == [4] and sorted(np.asarray(held.fact_known_time).tolist()) == [3, 4]
    none = np.zeros(0, np.int64)
    arrays = dict(held.arrays, fact_replaced_row=none, fact_replaced_time=none)
    twin = Facts("uncorrected", held.keys, held.unit, held.inner, held.observers, held.options, held.generators, **arrays)
    monkeypatch.setattr(describing, "stored", lambda path, graph: (twin, held)[graph["index"]])
    assert differing.recorded(crossed, manifest, dict(index=0), dict(index=1)) == dict(
        version=describing.VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=dict(model="clocks", units=["tick"]),
        first="uncorrected",
        second="h0",
        unit="tick",
        differs=True,
        times=[None, 0, 1, 2, 3],
        kinds={},
        relations=dict(near=unlike(EDGED, "both", (4, 0, 0), [[None, 0], [0, 1]], UNHELD, UNHELD)),
        equal=dict(kinds=2, relations=1),
    )
    LOG.info("descriptions corrected exit seconds=%.3f", time.perf_counter() - started)


ORDERED = dict(model="order", units=[])
STILL = dict(model="none", units=[])
ORDERED_SECOND = """graph g1, no unit, positions 3 from 1 to 3
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 2, observations 2, replaced 0
  relation near of the table near: no ends; columns weight; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  held before every position and at each position:
    position            before  1  2  3
    agent nodes         1       1  1  1
    agent observations  1       1  1  1
    spot nodes          1       2  2  1
    spot observations   1       2  2  1
    near edges          0       0  0  0
    base edges          1       1  1  0
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 1; train 1, validation 0, no split 0
  kind of task visit: tasks 1; train 0, validation 1, no split 0
  decisions of turn: decisions 1, options 3, recorded 2, regenerated 1; changed 1; no rule of options
    task  position  options  recorded  regenerated  changed
    10    2         3        2         1            1
"""
ORDERED_ALONE = """no unit, positions 3 from 1 to 3
%s
  nothing differs of what is compared; equal at every position: kinds 2, relations 2
"""
ORDERED_CUT = (
    "  entry test.cut, of the operation test.cut, removes every edge of the relation base at the positions of the tasks "
    "of turn: 1, 3, 5; not at every position\n"
)
UNMET = (
    "^the graphs g0 and g1 state when their facts hold as positions, each in an order of its own; the positions of two "
    "graphs are never met, so under the model order a graph is compared only with itself$"
)
STILL_SECOND = """graph g1, no unit, no time
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 2, observations 2, replaced 0
  relation near of the table near: no ends; columns weight; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  held always:
    held                always
    agent nodes         1
    agent observations  1
    spot nodes          2
    spot observations   2
    near edges          0
    base edges          1
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 1; train 1, validation 0, no split 0
  kind of task visit: tasks 1; train 0, validation 1, no split 0
  decisions of turn: decisions 1, options 3, recorded 2, regenerated 1; changed 1; no rule of options
    task  time  options  recorded  regenerated  changed
    10    none  3        2         1            1
"""
STILL_APART = """no unit, no time
%s
  kind agent: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    first nodes  second nodes  first anchors  second anchors  other values
    1            0             0              0               0
  kind spot: nodes only the first holds 1, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    first nodes  second nodes  first anchors  second anchors  other values
    1            0             0              0               0
  relation near: edges only the first holds 2, only the second 0, held by both with other values 0
    first  second  other values
    2      0       0
  relation base: edges only the first holds 1, only the second 0, held by both with other values 0
    first  second  other values
    1      0       0
  equal: kinds 0, relations 0
"""
STILL_VARIANT = """
graph g1, the graph g1 as the tasks of the splits train, validation see it, 3 steps, no time
  tasks: turn at the step; visit at the step
  held at each step before the first entry:
    step                        0  1  2
    agent nodes                 1  1  0
    agent anchors               0  0  0
    spot nodes                  2  2  1
    spot anchors                0  0  0
    near edges                  0  0  0
    base edges                  1  0  0
    relation_forms.loops edges  3  3  1
"""


@timed("descriptions.restated")
def restated(manifest, stated, ties):
    spec = manifest["spec"]
    return dict(manifest, spec=dict(spec, schema=dict(spec["schema"], time=dict(stated, ties=ties))))


@timed("descriptions.unclocked")
def unclocked(facts):
    return Facts(facts.name, facts.keys, None, facts.inner, facts.observers, facts.options, facts.generators, **facts.arrays)


def test_under_order_a_description_states_positions_and_no_length_and_the_orders_of_two_graphs_are_not_met(
    bank, cutting, monkeypatch
):
    started = time.perf_counter()
    manifest = restated(read_json(bank / "manifest.json"), ORDERED, "key")
    held = [unclocked(describing.stored(bank, graph)) for graph in manifest["graphs"]]
    monkeypatch.setattr(describing, "stored", lambda path, graph: held[graph["index"]])
    record = describing.recorded(bank, manifest)
    first, second = record["graphs"]
    unmeasured = {name: dict(found, **NONE) for name, found in FIRST["relations"].items()}
    assert record["time"] == ORDERED and second == dict(SECOND, unit=None)
    assert first == dict(FIRST, unit=None, relations=unmeasured) and FIRST["relations"]["near"]["mean"] == 4.0
    written = describing.text(dict(record, kept=kept_as("truth", "bank")))
    assert written.split("\n")[2:5] == ["time: order without a clock, no unit", "splits: train, validation", describing.READING["order"]]
    assert written.split("\n\n")[2] == ORDERED_SECOND
    assert not {"time", "lasting", "unit"} & set(describing.READING["order"].split() + written.split("\n\n", 1)[1].split()) - {"unit,"}
    with pytest.raises(ValueError, match=UNMET):
        differing.recorded(bank, manifest, *manifest["graphs"])
    alone = differing.recorded(bank, manifest, manifest["graphs"][1], manifest["graphs"][1])
    assert alone == dict(
        version=describing.VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=ORDERED,
        first="g1",
        second="g1",
        unit=None,
        differs=False,
        times=[None, 1, 2, 3],
        kinds={},
        relations={},
        equal=dict(kinds=2, relations=2),
    )
    kept = dict(alone, kept=kept_as("differences", "bank"))
    assert differing.text(kept).split("\n", 2)[2] == ORDERED_ALONE % differing.TRUTH["order"]
    assert "time" not in differing.TRUTH["order"] + differing.READING["order"]
    config = configured(bank, CUTTING)
    stated = effect.text(dict(variant(bank, config, STATED), time=ORDERED))
    body = stated[stated.index("\ngraph ") :]
    head = "\ngraph g0, %s, 8 steps from position 0 to 7\n  tasks: turn at 1, 3, 5; visit at 2, 4, 5, 6\n" % (SEEN_BY % "g0")
    assert body.startswith(head)
    opening = (
        "  held at each position before the first entry:\n"
        "    position                    0  1  2  3  4  5  6  7\n"
        "    agent nodes                 2  2  2  2  2  2  2  2\n"
    )
    assert opening in body and ORDERED_CUT in body and "time" not in body
    LOG.info("descriptions under order exit seconds=%.3f", time.perf_counter() - started)


def test_without_time_a_description_states_no_time_and_counts_once_what_holds_always(bank, cutting, monkeypatch):
    started = time.perf_counter()
    manifest = restated(read_json(bank / "manifest.json"), STILL, None)
    held = [unclocked(timeless(describing.stored(bank, graph))) for graph in manifest["graphs"]]
    monkeypatch.setattr(describing, "stored", lambda path, graph: held[graph["index"]])
    record = describing.recorded(bank, manifest)
    second, turn = record["graphs"][1], SECOND["decisions"]["turn"]
    assert record["time"] == STILL and second == dict(
        SECOND,
        unit=None,
        times=[],
        kinds=dict(
            agent=dict(nodes=1, observations=1, replaced=0, nodes_held=[[None, 1]], observations_held=[[None, 1]]),
            spot=dict(nodes=2, observations=2, replaced=0, nodes_held=[[None, 2]], observations_held=[[None, 2]]),
        ),
        relations=dict(near=SECOND["relations"]["near"], base=dict(SECOND["relations"]["base"], edges_held=[[None, 1]])),
        decisions=dict(turn=dict(turn, each=[dict(turn["each"][0], time=None)])),
    )
    written = describing.text(dict(record, kept=kept_as("truth", "bank")))
    assert written.split("\n")[2:5] == ["time: no time, no unit", "splits: train, validation", describing.READING["none"]]
    assert written.split("\n\n")[2] == STILL_SECOND
    apart = differing.recorded(bank, manifest, *manifest["graphs"])
    assert (apart["time"], apart["unit"], apart["times"], apart["differs"]) == (STILL, None, [None], True)
    assert apart["relations"]["near"] == unlike(EDGED, "both", (2, 0, 0), [[None, 2]], UNHELD, UNHELD)
    kept = dict(apart, kept=kept_as("differences", "bank"))
    assert differing.text(kept).split("\n", 2)[2] == STILL_APART % differing.TRUTH["none"]
    name, graph, grown, low, high = diverging.ended(Deriver(bank, "cpu", STATED["derivation_seed"], RESIDENT), leaned(bank, 1))[1]
    assert (name, low, high, np.asarray(graph.step_time).tolist()) == ("g1", 0, 3, [1, 2, 3])
    monkeypatch.setattr(grown.schema, "model", "none")
    (told,) = effect.told([graph], [graph], grown, ())
    assert (told["times"], told["tasks"]) == ([None, None, None], dict(turn=[None], visit=[None]))
    still = dict(variant(bank, configured(bank, CUTTING), STATED), time=STILL, graphs=[dict(told, view=VIEWED["g1"])])
    stated = effect.text(still)
    assert stated[stated.index("\ngraph ") :] == STILL_VARIANT
    monkeypatch.undo()
    first, second = (diverging.ended(Deriver(bank, "cpu", STATED["derivation_seed"], RESIDENT), leaned(bank, degree))[1] for degree in (1, 2))
    for side in (first, second):
        monkeypatch.setattr(side[2].schema, "model", "none")
    unlike_steps = diverging.paired(first, second)
    assert (unlike_steps["times"], unlike_steps["alone"]) == ([None, None, None], UNSHARED)
    LOG.info("descriptions without time exit seconds=%.3f", time.perf_counter() - started)


SEEN = ("key", "since", "until", "load", "known", "observers")
OWNED = ("key", "since", "until", "weight", "known", "observers")
WEIGHED = ("source", "target", "present", "since", "until", "weight", "known", "observers")
JOINING = ("source", "target", "present", "since", "until", "known", "observers")
NOTED = ("source", "target", "present", "since", "until", "mark", "known", "observers")
MEMBERS = ("set", "member", "role", "share")
PIECES = ("owner", "kind", "key", "mass")
JOINTS = ("owner", "source", "target", "relation")
UNWATCHED = dict(north=dict(nodes=0, edges=0), south=dict(nodes=0, edges=0))
WATCHED_FIRST = dict(
    name="w0",
    unit="tick",
    times=[0, 1, 2, 3, 4, 5, 6],
    kinds=dict(
        agent=dict(nodes=2, observations=2, replaced=0, nodes_held=[[None, 0], [0, 2]], observations_held=[[None, 0], [0, 2]]),
        spot=dict(
            nodes=5,
            observations=5,
            replaced=0,
            nodes_held=[[None, 0], [0, 4], [2, 5]],
            observations_held=[[None, 0], [0, 4], [2, 5]],
        ),
        group=dict(
            nodes=2,
            observations=2,
            replaced=0,
            nodes_held=[[None, 0], [0, 2], [4, 1]],
            observations_held=[[None, 0], [0, 2], [4, 1]],
        ),
    ),
    relations=dict(
        near=dict(
            table="near",
            ends=[["spot", "spot"]],
            columns=["weight"],
            about=None,
            lapsed=0,
            edges=3,
            facts=3,
            absent=0,
            replaced=0,
            edges_held=[[None, 0], [0, 3]],
            runs=3,
            bounded=0,
            **NONE,
        ),
        base=dict(
            table="base",
            ends=[["agent", "spot"]],
            columns=[],
            about=None,
            lapsed=0,
            edges=3,
            facts=3,
            absent=0,
            replaced=0,
            edges_held=[[None, 0], [0, 2], [3, 3]],
            runs=3,
            bounded=0,
            **NONE,
        ),
        back=dict(
            table="back",
            ends=[["spot", "agent"]],
            columns=[],
            about=None,
            lapsed=0,
            edges=1,
            facts=1,
            absent=0,
            replaced=0,
            edges_held=[[None, 0], [0, 1]],
            runs=1,
            bounded=0,
            **NONE,
        ),
        note=dict(
            table="note",
            ends=[["group", "spot"]],
            columns=["mark"],
            about="lapse",
            lapsed=2,
            edges=2,
            facts=2,
            absent=0,
            replaced=0,
            edges_held=[[None, 0], [0, 1], [1, 2], [4, 1]],
            runs=2,
            bounded=1,
            least=3,
            most=3,
            mean=3.0,
        ),
    ),
    tasks=dict(turn=dict(tasks=3, splits=dict(train=2, validation=1), unsplit=0)),
    decisions=dict(
        turn=dict(
            decisions=3,
            options=4,
            recorded=4,
            regenerated=0,
            generators={},
            rule=None,
            each=[
                dict(key=10, time=1, options=2, recorded=2, regenerated=0, generators={}),
                dict(key=20, time=3, options=1, recorded=1, regenerated=0, generators={}),
                dict(key=30, time=5, options=1, recorded=1, regenerated=0, generators={}),
            ],
        )
    ),
    unread=dict(
        later=dict(nodes=1, edges=1, gap=3, last=5),
        watched=dict(nodes=1, edges=2, observers=dict(north=dict(nodes=1, edges=1), south=dict(nodes=0, edges=1))),
        sets=dict(group=dict(edges=2, members=5, least=2, most=3)),
        held=dict(agent=dict(graphs=2, nodes=2, edges=1, least=0, most=2)),
        entities=dict(agent=WHOLE_KIND, spot=dict(created=1, unstated=1), group=WHOLE_KIND),
    ),
)
UNHELD_RELATION = dict(
    ends=[], about=None, lapsed=0, edges=0, facts=0, absent=0, replaced=0, edges_held=UNHELD, runs=0, bounded=0, **NONE
)
WATCHED_SECOND = dict(
    name="w1",
    unit="tick",
    times=[],
    kinds=dict(
        agent=dict(nodes=1, observations=1, replaced=0, nodes_held=[[None, 1]], observations_held=[[None, 1]]),
        spot=dict(nodes=1, observations=1, replaced=0, nodes_held=[[None, 1]], observations_held=[[None, 1]]),
        group=dict(nodes=0, observations=0, replaced=0, nodes_held=UNHELD, observations_held=UNHELD),
    ),
    relations=dict(
        near=dict(UNHELD_RELATION, table="near", columns=["weight"]),
        base=dict(
            UNHELD_RELATION, table="base", ends=[["agent", "spot"]], columns=[], edges=1, facts=1, edges_held=[[None, 1]], runs=1
        ),
        back=dict(UNHELD_RELATION, table="back", columns=[]),
        note=dict(UNHELD_RELATION, table="note", columns=["mark"], about="lapse"),
    ),
    tasks=dict(turn=dict(tasks=0, splits=dict(train=0, validation=0), unsplit=0)),
    decisions=dict(turn=dict(decisions=0, options=0, recorded=0, regenerated=0, generators={}, rule=None, each=[])),
    unread=dict(
        later=dict(nodes=0, edges=0, gap=None, last=None),
        watched=dict(nodes=0, edges=0, observers=UNWATCHED),
        sets=dict(group=dict(edges=0, members=0, least=None, most=None)),
        held=dict(agent=dict(graphs=1, nodes=0, edges=0, least=0, most=0)),
        entities=dict(agent=WHOLE_KIND, spot=WHOLE_KIND, group=WHOLE_KIND),
    ),
)
UNSAID_TEXT = """truth graphs of the bank stored as %s, described at version %d
kept beside the bank, in its folder descriptions/truth
time: two clocks, units tick
splits: train, validation
%s

graph w0, unit tick, times 7 from 0 to 6
  kind agent: nodes 2, observations 2, replaced 0
  kind spot: nodes 5, observations 5, replaced 0
  set edge group: set edges 2, facts 2, replaced 0
  relation near of the table near: ends spot to spot; columns weight; edges 3, facts 3, absent 0, replaced 0; runs 3, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 3, facts 3, absent 0, replaced 0; runs 3, bounded 0
  relation back of the table back: ends spot to agent; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  relation note of the table note: ends group to spot; columns mark; edges 2, facts 2, absent 0, replaced 0; runs 2, bounded 1, lasting 3 to 3, mean 3.0; about a set edge under the rule lapse, lapsed 2
  held before every time and at each time:
    time                before  0  1  2  3  4  5  6
    agent nodes         0       2  2  2  2  2  2  2
    agent observations  0       2  2  2  2  2  2  2
    spot nodes          0       4  4  5  5  5  5  5
    spot observations   0       4  4  5  5  5  5  5
    group set edges     0       2  2  2  2  1  1  1
    group facts         0       2  2  2  2  1  1  1
    near edges          0       3  3  3  3  3  3  3
    base edges          0       2  2  2  3  3  3  3
    back edges          0       1  1  1  1  1  1  1
    note edges          0       1  2  2  2  1  1  1
  shown to some tasks only: facts known later than they hold from, about nodes 1 and about edges 1, by 3 at most, the last known from time 5, before which a task is not shown them; facts naming observers, which a task naming another observer is not shown, about nodes 1 and about edges 2, naming north 1 and 1, south 0 and 1
  not counted above: members of set edges, group 5 in 2 set edges of 2 to 3; graphs nodes hold, agent 2 of 0 to 2 nodes, 2 nodes and 1 edges in all; entities no fact is about, spot 1 an option would create and 1 other
  kind of task turn: tasks 3; train 2, validation 1, no split 0
  decisions of turn: decisions 3, options 4, recorded 4, regenerated 0; no generator; no rule of options
    task  time  options  recorded  regenerated
    10    1     2        2         0
    20    3     1        1         0
    30    5     1        1         0

graph w1, unit tick, no time
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 1, observations 1, replaced 0
  set edge group: set edges 0, facts 0, replaced 0
  relation near of the table near: no ends; columns weight; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  relation back of the table back: no ends; no columns; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0
  relation note of the table note: no ends; columns mark; edges 0, facts 0, absent 0, replaced 0; runs 0, bounded 0; about a set edge under the rule lapse, lapsed 0
  held before every time and at each time:
    time                before
    agent nodes         1
    agent observations  1
    spot nodes          1
    spot observations   1
    group set edges     0
    group facts         0
    near edges          0
    base edges          1
    back edges          0
    note edges          0
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: members of set edges, group 0 in 0 set edges; graphs nodes hold, agent 1 of 0 to 0 nodes, 0 nodes and 0 edges in all; no entity without a fact
  kind of task turn: tasks 0; train 0, validation 0, no split 0
  decisions of turn: decisions 0, options 0, recorded 0, regenerated 0; no generator; no rule of options
"""


@timed("descriptions.unsaid_schema")
def unsaid_schema():
    roles = dict(
        to=dict(sort="node", kinds=["spot"], count=[1, 2]),
        along=dict(sort="edge", kinds=["spot"], count=[0, None]),
        made=dict(sort="new", kinds=["spot"], count=[0, 1]),
    )
    decided = dict(
        kind="decision",
        series=["gain"],
        about=None,
        utility=synthetic.valued(["gain"]),
        actors="agents",
        actions=dict(go=dict(roles=roles, parameters=["speed"])),
        nothing=None,
        rule=None,
        generators=[],
    )
    pieces = dict(
        nodes=dict(piece=dict(columns=["mass"])),
        edges=dict(joint=dict(endpoints=[["piece", "piece"]], directed=False, columns=[])),
        encoder="pooled",
    )
    noting = dict(endpoints=[["group", "spot"]], directed=True, columns=["mark"], relation="declared", about="lapse")
    return dict(
        version=SCHEMA,
        time=dict(model="clocks", units=["tick"], ties="key"),
        observers=True,
        columns=dict(load=NUMBER, speed=NUMBER, weight=NUMBER, share=NUMBER, mass=NUMBER, mark=NUMBER),
        graph=dict(columns=[]),
        nodes=dict(agent=dict(columns=["load"], holds=pieces), spot=dict(columns=["load"], holds=None)),
        sets=dict(group=dict(kinds=["spot"], roles=["lead", "other"], members=["share"], columns=["weight"])),
        edges=dict(
            near=synthetic.edge("spot", "spot", ["weight"], directed=False),
            base=synthetic.edge("agent", "spot", []),
            back=synthetic.edge("spot", "agent", []),
            note=noting,
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(agents=dict(sort="nodes", kinds=["agent"])),
            kinds=dict(turn=synthetic.kind(["agents"], [], decided, None)),
        ),
    )


@timed("descriptions.valued_rows")
def valued_rows(kind, names, *stated):
    return dict(form="state", parameters={}, rows=rows(names, *stated), kind=kind)


@timed("descriptions.joined_rows")
def joined_rows(table, ends, names, *stated):
    return dict(form="state", parameters={}, rows=rows(names, *stated), table=table, ends=ends)


@timed("descriptions.going")
def going(key, to, along, made, speed):
    targets = dict(to=to, along=along, made=made)
    return synthetic.offered(key, [synthetic.acting("go", "a0", targets, dict(speed=speed))], "set", "recorded")


def unwatched(stated):
    started = time.perf_counter()
    names = ("key", "kind", "time", "observer", "split", "weight", "target", "known")
    result = dict({name: [] for name in names}, references=dict(agents=[]))
    for key, moment, split, gain, options in stated:
        options[0].update(chosen=True, outcome=dict(gain=gain))
        target = dict(options=options, concurrent=[], next={}, ended=None, discount=None)
        for name, value in zip(names, (key, "turn", moment, None, split, None, target, moment + 1)):
            result[name].append(value)
        result["references"]["agents"].append(["a0"])
    LOG.debug("descriptions unwatched exit tasks=%d seconds=%.4f", len(stated), time.perf_counter() - started)
    return result


@timed("descriptions.unsaid_first")
def unsaid_first():
    values = [
        valued_rows("agent", SEEN, ("a0", 0, None, 1.0, None, None), ("a1", 0, None, 2.0, None, None)),
        valued_rows(
            "spot",
            SEEN,
            ("s0", 0, None, 1.0, None, None),
            ("s1", 0, None, None, None, None),
            ("s2", 0, None, 3.0, None, None),
            ("s3", 2, None, 4.0, 5, None),
            ("s4", 0, None, 5.0, None, ["north"]),
        ),
        valued_rows("group", OWNED, ("g0", 0, None, 1.0, None, None), ("g1", 0, 4, 2.0, None, None)),
    ]
    edges = [
        joined_rows(
            "near",
            ("spot", "spot"),
            WEIGHED,
            ("s0", "s1", True, 0, None, 0.5, None, None),
            ("s1", "s2", True, 0, None, 0.75, None, None),
            ("s0", "s2", True, 0, None, 0.25, None, ["south"]),
        ),
        joined_rows(
            "base",
            ("agent", "spot"),
            JOINING,
            ("a0", "s0", True, 0, None, None, None),
            ("a0", "s3", True, 3, None, 5, None),
            ("a1", "s4", True, 0, None, None, ["north"]),
        ),
        joined_rows("back", ("spot", "agent"), JOINING, ("s1", "a0", True, 0, None, None, None)),
        joined_rows(
            "note",
            ("group", "spot"),
            NOTED,
            ("g0", "s0", True, 0, None, 1.0, None, None),
            ("g1", "s2", True, 1, 6, 2.0, None, None),
        ),
    ]
    members = rows(
        MEMBERS,
        ("g0", "s0", "lead", 0.5),
        ("g0", "s1", "other", 0.3),
        ("g0", "s2", "other", 0.2),
        ("g1", "s0", "lead", 0.6),
        ("g1", "s1", "other", 0.4),
    )
    inner = dict(
        nodes=rows(PIECES, ("a0", "piece", "p0", 1.0), ("a0", "piece", "p1", 2.0)),
        edges=rows(JOINTS, ("a0", "p0", "p1", "joint")),
    )
    tasks = unwatched(
        [
            (10, 1, "train", 1.0, [going("o0", ["s0"], [], [], 1.0), going("o1", ["s2"], [["s0", "near", "s1"]], ["s9"], 2.0)]),
            (20, 3, "train", 2.0, [going("o0", ["s1"], [["s1", "back", "a0"]], [], 1.0)]),
            (30, 5, "validation", 3.0, [going("o0", ["s3"], [], [], 3.0)]),
        ]
    )
    return dict(
        name="w0",
        unit="tick",
        columns={},
        nodes=dict(agent=["a0", "a1"], spot=["s0", "s1", "s2", "s3", "s4", "s8"], group=["g0", "g1"]),
        values=values,
        edges=edges,
        sets=dict(group=members),
        held=dict(agent=inner),
        tasks=tasks,
    )


@timed("descriptions.unsaid_second")
def unsaid_second():
    values = [
        valued_rows("agent", SEEN, ("a0", None, None, 1.0, None, None)),
        valued_rows("spot", SEEN, ("s0", None, None, 1.0, None, None)),
    ]
    edges = [joined_rows("base", ("agent", "spot"), JOINING, ("a0", "s0", True, None, None, None, None))]
    return dict(
        name="w1",
        unit="tick",
        columns={},
        nodes=dict(agent=["a0"], spot=["s0"], group=[]),
        values=values,
        edges=edges,
        sets=dict(group=rows(MEMBERS)),
        held=dict(agent=dict(nodes=rows(PIECES), edges=rows(JOINTS))),
        tasks=unwatched([]),
    )


@timed("descriptions.unsaid_made")
def unsaid_made(root):
    case = dict(schema=unsaid_schema(), graphs=[unsaid_first(), unsaid_second()])
    build(synthetic.stage(root, "unsaid", case), root / "bank", 1, ("train",))


@pytest.fixture(scope="module")
def unsaid():
    started = time.perf_counter()
    result = shared("unsaid", unsaid_made) / "bank"
    LOG.debug("descriptions unsaid exit seconds=%.4f", time.perf_counter() - started)
    return result


def test_a_truth_description_counts_what_it_leaves_out_and_holds_an_edge_about_a_set_edge_as_a_view_does(unsaid):
    started = time.perf_counter()
    manifest = read_json(unsaid / "manifest.json")
    record = describe(unsaid)
    assert record == dict(
        version=describing.VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=dict(model="clocks", units=["tick"]),
        splits=["train", "validation"],
        graphs=[WATCHED_FIRST, WATCHED_SECOND],
        kept=kept_as("truth", "bank"),
    )
    assert json.loads(json.dumps(record)) == record
    assert describing.text(record) == UNSAID_TEXT % (manifest["stored_sha256"], describing.VERSION, describing.READING["clocks"])
    registry = Registry.from_declaration(manifest["spec"])
    held = describing.stored(unsaid, manifest["graphs"][0])
    side = differing.lasting(registry, held, describing.placed(describing.clocked(registry, held)))
    noted = side["facts"][side["facts"][:, 1] == registry.code("note")]
    assert sorted(map(tuple, noted[:, [0, 2, 3]].tolist())) == sorted(
        [(place, held.keys.index("g0"), held.keys.index("s0")) for place in range(1, 8)]
        + [(place, held.keys.index("g1"), held.keys.index("s2")) for place in (2, 3, 4)]
    )
    LOG.info("descriptions unsaid truth exit seconds=%.3f", time.perf_counter() - started)


WATCHING = dict(LONE, **{"sets/group/storage": "node"})
UNSHOWN_TEXT = (
    "\ngraphs of the bank the view does not show: w1, which holds 0 tasks, 0 of them of a split trained on or scored "
    "on, and to which the choice of steps changes gives 0 sequences of steps\n"
)
RULED_VIEWS = {
    0: dict(graph="g0", observer=None, splits=["validation"]),
    1: dict(graph="g0", observer=None, splits=["train"]),
    2: dict(graph="g1", observer=None, splits=["validation"]),
    3: dict(graph="g1", observer=None, splits=["train"]),
}


def test_a_variant_names_the_graphs_of_the_bank_its_view_does_not_show_and_the_splits_whose_tasks_see_each_view(
    unsaid, bank
):
    started = time.perf_counter()
    kept = {"relations/%s/keep" % name: name != "near" for name in ("near", "base", "back", "note", "group.lead", "group.other", "group")}
    pipeline = [dict(operation="graphspec.relations", parameters=kept)]
    config = configuration(unsaid, MODEL, None, pipeline=pipeline, graph=WATCHING, seed=STATED["seed"], splits=SPLITS)
    record = variant(unsaid, config, STATED)
    assert record["unshown"] == [dict(name="w1", tasks=0, stated=0, sequences=0)]
    (shown,) = record["graphs"]
    assert (shown["name"], shown["view"], shown["times"]) == ("w0", VIEWED["w0"], [0, 1, 2, 3, 4, 5, 6])
    written = effect.text(record)
    assert UNSHOWN_TEXT in written and "\ngraph w0, %s, 7 steps from time 0 to 6\n" % (SEEN_BY % "w0") in written
    unweighed = dict(kinds={}, relations=dict(near=dict(added=[], removed=["weight"], filled=[])))
    assert [held["values"] for held in shown["effects"]] == [unweighed]
    assert "\n  columns of values and values of nodes: relation near removes weight\n" in written
    assert shown["effects"][0]["relations"]["near"] == tally(21, 21, 1.0, 0, 0.0, 0, edges=NIL)
    steps, moments = list(range(7)), WATCHED_FIRST["times"]
    noted = held_at(WATCHED_FIRST["relations"]["note"]["edges_held"], moments)
    assert held_at(shown["held"]["relations"]["note"]["edges"], steps) == noted == [1, 2, 2, 2, 1, 1, 1]
    assert shown["held"]["relations"]["note"]["edges"] == [[0, 1], [1, 2], [4, 1]]
    grouped_held = held_at(WATCHED_FIRST["kinds"]["group"]["nodes_held"], moments)
    assert held_at(shown["held"]["kinds"]["group"]["nodes"], steps) == grouped_held == [2, 2, 2, 2, 1, 1, 1]
    assert (shown["held"]["relations"]["base"]["edges"], WATCHED_FIRST["relations"]["base"]["edges_held"]) == (
        [[0, 2], [5, 3]],
        [[None, 0], [0, 2], [3, 3]],
    )
    deriver = Deriver(bank, STATED["device"], STATED["derivation_seed"], RESIDENT)
    ruled = configuration(bank, MODEL, None, pipeline=[], graph=SETTINGS, seed=STATED["seed"], splits=SEEING)
    assert restating.viewed(deriver, ruled) == RULED_VIEWS and restating.unviewed(deriver, ruled) == []
    assert describing.viewed_text(RULED_VIEWS[1]) == "the graph g0 as the tasks of the split train see it"
    north = dict(graph="w0", observer="north", splits=EVERY)
    assert describing.viewed_text(north) == SEEN_BY % "w0" + ", those that name the observer north"
    LOG.info("descriptions unsaid variant exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.end")
def end(key, *columns):
    return dict(node=key, columns=list(columns))


@timed("descriptions.joined_at")
def joined_at(target, hops, relations, along, features, values):
    held = dict(state="joined", hops=hops, relations=relations, along=along, anchor=False)
    return dict(target=target, put=True, facts=[], **held, features=features, values=values)


BESIDE_BACK = [["back", "base"]]
WATCHED_AIMS = {
    (10, "o0"): {GO[0]: [joined_at("s0", 1, [["base"]], 2, [end("s0", "load")], None)]},
    (10, "o1"): {
        GO[0]: [joined_at("s2", 2, BESIDE_BACK + [["near"]], 2, [end("s2", "load")], None)],
        GO[1]: [joined_at("s0 near s1", 1, BESIDE_BACK, 1, [end("s0", "load"), end("s1")], ["weight"])],
        GO[2]: [dict(target="s9", put=True, facts=[], **CREATED, features=[], values=None)],
    },
    (20, "o0"): {
        GO[0]: [joined_at("s1", 1, [["back"]], 1, [end("s1")], None)],
        GO[1]: [joined_at("s1 back a0", 0, [], 0, [end("s1"), end("a0", "load")], [])],
    },
    (30, "o0"): {GO[0]: [joined_at("s3", 1, [["base"]], None, [end("s3", "load")], None)]},
}
WATCHED_AIMS_TEXT = [
    "        s0: joined to a reference at hop 1 by base and at hop 2 along the direction of its facts, features load",
    "        s2: joined to a reference at hop 2 by back or base then near and at hop 2 along the direction of its facts, "
    "features load",
    "        s0 near s1: joined to a reference at hop 1 by back or base and at hop 1 along the direction of its facts, "
    "features of s0 load, of s1 none, of the edge weight",
    "        s9: an element the option creates, no node of the variant",
    "        s1: joined to a reference at hop 1 by back and at hop 1 along the direction of its facts, features none",
    "        s1 back a0: joined to a reference at hop 0 by nothing and at hop 0 along the direction of its facts, features "
    "of s1 none, of a0 load, of the edge none",
    "        s3: joined to a reference at hop 1 by base and by no path along the direction of its facts, features load",
]
ANCHORED_TEXT = (
    "        s0: an anchor at the step, joined to a reference at hop 0 by nothing and at hop 0 along the direction of its "
    "facts, features load"
)


def test_reach_says_which_end_of_an_edge_holds_a_feature_its_own_values_the_hops_along_its_facts_and_what_is_an_anchor(
    unsaid, crossed
):
    started = time.perf_counter()
    model = dict(MODEL, **{"options/realisation": "vector"})
    config = configuration(unsaid, model, None, pipeline=[], graph=WATCHING, seed=STATED["seed"], splits=SPLITS)
    held = reaching.Reached(Deriver(unsaid, STATED["device"], STATED["derivation_seed"], RESIDENT), config, "vector")
    found, lines = {}, []
    for decision in held.decided(held.derived(None, 0, 0), 0):
        for option in held.optioned(decision):
            read = held.read(None, 0, decision, option)
            aims = {name: field["targets"] for name, field in read["fields"].items() if "targets" in field}
            found[decision["key"], held.found.keys[option]] = aims
            lines += [reaching.target_text(dict(target, cut=None), []) for targets in aims.values() for target in targets]
            assert read["held"]
        held.decision_done()
    assert found == WATCHED_AIMS and lines == WATCHED_AIMS_TEXT
    thinnest = dict(KEPT, **{"nodes/spot/filter": "degree", "nodes/spot/min_degree": 3})
    other = reaching.Reached(
        Deriver(crossed, STATED["device"], STATED["derivation_seed"], RESIDENT), lined(crossed, THINNER, thinnest, "vector"), "vector"
    )
    (last,) = [decision for decision in other.decided(other.derived(None, 1, 0), 0) if decision["key"] == 30]
    (anchored,) = other.read(None, 1, last, other.optioned(last)[0])["fields"][FIELDS[4]]["targets"]
    assert anchored == dict(target="s0", put=True, facts=[], **dict(HERE, anchor=True), features=[end("s0", "load")], values=None)
    assert reaching.target_text(dict(anchored, cut=None), []) == ANCHORED_TEXT
    for legend in (effect.READING, reaching.READING["clock"]):
        assert "observed at the step other than as an anchor" in legend
    LOG.info("descriptions reach ends exit seconds=%.3f", time.perf_counter() - started)


class Listed:
    version = 1

    @timed("descriptions.listed.checked")
    def checked(self, stated, schema, where):
        return stated


RULED_TEXT = (
    "  decisions of turn: decisions 3, options 10, recorded 7, regenerated 3; changed 3; its options are given by the rule "
    "test.listed with within 2, of which those counted here are the options listed or drawn\n"
)


def test_a_kind_of_task_whose_options_a_rule_gives_names_the_rule_beside_the_options_it_counts(bank):
    started = time.perf_counter()
    manifest = read_json(bank / "manifest.json")
    schema = json.loads(json.dumps(manifest["spec"]["schema"]))
    rule = dict(name="test.listed", parameters=dict(within=2))
    schema["tasks"]["kinds"]["turn"]["target"]["rule"] = rule
    RULES.register("test.listed", Listed())
    try:
        record = describing.recorded(bank, dict(manifest, spec=dict(manifest["spec"], schema=schema)))
    finally:
        del RULES.held["test.listed"]
    assert [graph["decisions"]["turn"]["rule"] for graph in record["graphs"]] == [rule, rule]
    written = describing.text(dict(record, kept=kept_as("truth", "bank")))
    assert RULED_TEXT in written and "no rule of options" not in written
    LOG.info("descriptions ruled exit seconds=%.3f", time.perf_counter() - started)


@timed("descriptions.stretched")
def stretched(values, step, shift):
    values = np.asarray(values, np.int64)
    stated = (values > EARLIEST) & (values < LATEST)
    return np.where(stated, values * 1000 + step * np.arange(len(values)) + shift, values)


def brute(registry, facts, places):
    started = time.perf_counter()
    kinds = np.asarray(facts.kinds).tolist()
    names = ("obs_entity", "obs_since", "obs_until")
    seen = list(zip(*[np.asarray(facts.arrays[name]).tolist() for name in names]))
    names = ("fact_code", "fact_source", "fact_target", "fact_since", "fact_until", "fact_present")
    stated = [row for row in zip(*[np.asarray(facts.arrays[name]).tolist() for name in names]) if row[5]]
    found = {}
    for place in places:
        moment = EARLIEST if place is None else place
        held = [entity for entity, since, until in seen if since <= moment < until]
        joined = {
            (code, *(sorted((source, target)) if registry.by_code[code].symmetric else (source, target)))
            for code, source, target, since, until, _ in stated
            if since <= moment < until
        }
        for name in registry.nodes:
            mine = [entity for entity in held if kinds[entity] == registry.kind(name)]
            found.setdefault(name + " nodes", []).append(len(set(mine)))
            found.setdefault(name + " observations", []).append(len(mine))
        for name in ("near", "base"):
            found.setdefault(name + " edges", []).append(sum(edge[0] == registry.code(name) for edge in joined))
    LOG.debug("descriptions brute exit places=%d seconds=%.4f", len(places), time.perf_counter() - started)
    return found


def test_what_holds_over_time_is_recorded_as_runs_exact_at_every_time_and_no_larger_than_the_facts_that_change_it(bank):
    started = time.perf_counter()
    held = [[None, 0], [0, 2], [3, 3], [4, 2]]
    assert held_at(held, [None, 0, 1, 2, 3, 4, 7]) == [0, 2, 2, 2, 3, 2, 2] and held_at(held, [3]) == [3]
    assert held_at(held, [-5, 100]) == [0, 2] and held_at(NIL, [0, 6]) == [0, 0] and held_at(held, []) == []
    with pytest.raises(ValueError, match="^a count is asked at -1, before 0, the first place from which its runs hold it$"):
        held_at(NIL, [-1])
    manifest = read_json(bank / "manifest.json")
    registry = Registry.from_declaration(manifest["spec"])
    facts = describing.stored(bank, manifest["graphs"][0])
    fine = facts.replace(
        obs_since=stretched(facts.obs_since, 7, 0),
        obs_until=stretched(facts.obs_until, 13, 1),
        fact_since=stretched(facts.fact_since, 11, 2),
        fact_until=stretched(facts.fact_until, 17, 5),
        task_time=np.asarray(facts.task_time, np.int64) * 1000 + 3,
    )
    record = describing.graphed(registry, fine, manifest["splits"])
    places = [None] + record["times"]
    told = {name + " edges": entry["edges_held"] for name, entry in record["relations"].items()}
    for name, entry in record["kinds"].items():
        told.update({name + " nodes": entry["nodes_held"], name + " observations": entry["observations_held"]})
    assert len(places) > 2 * describing.SHOWN and sorted(places[1:]) == places[1:]
    counted = brute(registry, fine, places)
    assert {label: held_at(runs, places) for label, runs in told.items()} == counted
    assert sorted(counted) == sorted(told) and all(len(set(found)) > 1 for found in counted.values())
    assert all(runs[0][0] is None and runs[0][1] == 0 for runs in told.values())
    assert all(first[1] != second[1] and second[0] in places for runs in told.values() for first, second in zip(runs, runs[1:]))
    changes = 2 * (2 * len(facts.obs_since) + len(facts.fact_since))
    assert sum(len(runs) - 1 for runs in told.values()) <= changes < len(told) * len(places)
    between_two = [moment + 1 for moment in record["times"] if moment + 1 not in record["times"]]
    assert {label: held_at(runs, between_two) for label, runs in told.items()} == brute(registry, fine, between_two)
    assert json.loads(json.dumps(record)) == record
    whole = dict(version=describing.VERSION, stored_sha256="stored", time=CLOCKED, splits=manifest["splits"], graphs=[record])
    lines = describing.text(dict(whole, kept=dict(kept_as("truth", "bank"), file="fine.json"))).split("\n")
    header = lines.index("  held before every time and at each time:") + 1
    assert lines[header].split()[:2] == ["time", "before"] and len(lines[header].split()) == 1 + describing.SHOWN
    assert lines[header].split()[-1] == str(places[-1]) and max(map(len, lines[5:])) < 300
    assert [line.split()[:2] for line in lines[header + 1 : header + 7]] == [label.split() for label in (
        "agent nodes", "agent observations", "spot nodes", "spot observations", "near edges", "base edges"
    )]
    assert lines[header + 7].startswith("    %d of %d columns shown, " % (describing.SHOWN, len(places)))
    assert lines[header + 7].endswith(", in the file descriptions/truth/fine.json")
    LOG.info("descriptions runs exit times=%d seconds=%.3f", len(record["times"]), time.perf_counter() - started)


OMITTED = (
    "    %d of %d %s shown, the first, the last and those between at which the counts change most; a count changes at %d "
    "of the %d left out; the record holds every one, in the file descriptions/%s/long.json"
)
CLIMBING = [[None, 0], [11, 1], [12, 2], [13, 3], [14, 4], [15, 5], [16, 6], [17, 7], [18, 8], [19, 9], [21, 8], [22, 7], [23, 6]]
ASKED = [2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 7, 2, 2, 2, 3]
LONG = dict(
    name="long",
    unit="tick",
    times=list(range(10, 30)),
    kinds=dict(
        spot=dict(
            nodes=9, observations=9, replaced=0, nodes_held=[[None, 0], [10, 4], [20, 9]], observations_held=[[None, 0], [10, 4], [20, 9]]
        )
    ),
    relations=dict(
        near=dict(HELD["near"], edges=9, facts=9, absent=0, replaced=0, edges_held=CLIMBING, runs=9, bounded=3, least=4, most=12, mean=8.0),
        base=dict(HELD["near"], columns=[], edges=3, facts=3, absent=0, replaced=0, edges_held=[[None, 3], [24, 0]], runs=3, bounded=0, **NONE),
    ),
    tasks=dict(turn=dict(tasks=15, splits=dict(train=15, validation=0), unsplit=0)),
    decisions=dict(
        turn=dict(
            decisions=15,
            options=39,
            recorded=39,
            regenerated=0,
            generators={},
            rule=None,
            each=[
                dict(key=100 + at, time=10 + at, options=count, recorded=count, regenerated=0, generators={})
                for at, count in enumerate(ASKED)
            ],
        )
    ),
    unread=dict(UNREAD, entities=dict(spot=WHOLE_KIND)),
)
LONG_TEXT = """
graph long, unit tick, times 20 from 10 to 29
  kind spot: nodes 9, observations 9, replaced 0
  relation near of the table near: ends spot to spot; columns weight; edges 9, facts 9, absent 0, replaced 0; runs 9, bounded 3, lasting 4 to 12, mean 8.0
  relation base of the table near: ends spot to spot; no columns; edges 3, facts 3, absent 0, replaced 0; runs 3, bounded 0
  held before every time and at each time:
    time               before  10  11  12  13  14  15  16  17  20  24  29
    spot nodes         0       4   4   4   4   4   4   4   4   9   9   9
    spot observations  0       4   4   4   4   4   4   4   4   9   9   9
    near edges         0       0   1   2   3   4   5   6   7   9   6   6
    base edges         3       3   3   3   3   3   3   3   3   3   0   0
%s
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 15; train 15, validation 0, no split 0
  decisions of turn: decisions 15, options 39, recorded 39, regenerated 0; no generator; no rule of options
    task  time  options  recorded  regenerated
    100   10    2        2         0
    103   13    5        5         0
    104   14    2        2         0
    110   20    7        7         0
    111   21    2        2         0
    114   24    3        3         0
%s"""


@timed("descriptions.filed")
def filed(part):
    return dict(kept_as(part, "bank"), file="long.json")


def test_a_table_over_more_than_the_bound_shows_the_first_the_last_and_those_between_at_which_its_counts_change_most(bank):
    started = time.perf_counter()
    assert describing.SHOWN == 12 and sum(ASKED) == 39
    written = LONG_TEXT % (OMITTED % (12, 21, "columns", 5, 9, "truth"), OMITTED % (6, 15, "rows", 0, 9, "truth"))
    assert describing.graph_text(LONG, describing.NAMED["clock"], filed("truth")) == written.split("\n")
    within = dict(near=dict(LONG["relations"]["near"], edges_held=CLIMBING[:10]), base=dict(LONG["relations"]["base"], edges_held=[[None, 3]]))
    short = describing.graph_text(dict(LONG, times=LONG["times"][:11], relations=within), describing.NAMED["clock"], filed("truth"))
    assert short[6].split() == ["time", "before"] + [str(moment) for moment in range(10, 21)]
    assert short[9].split() == ["near", "edges", "0"] + [str(count) for count in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9)]
    assert short[10].split() == ["base", "edges"] + ["3"] * 12 and short[11].startswith("  shown to some tasks only: ")
    stated = "at most %d" % describing.SHOWN
    legends = [describing.READING[model] for model in ("none", "order", "clock", "clocks")] + [effect.READING]
    legends += [held[model] for held in (differing.READING, differing.TRUTH) for model in ("order", "clock", "clocks")]
    assert [legend.count(stated) for legend in legends] == [1] * 11
    assert "decisions, as rows" in describing.READING["none"] and "as columns" not in describing.READING["none"]
    relations = {
        "r%03d" % at: dict(LONG["relations"]["base"], edges_held=[[None, 0], [1000 + at, 1], [1005 + at, 0]]) for at in range(200)
    }
    spots = dict(LONG["kinds"]["spot"], nodes_held=[[None, 0], [1000, 4]], observations_held=[[None, 0], [1000, 4]])
    long = dict(LONG, times=list(range(1000, 1300)), kinds=dict(spot=spots), relations=relations)
    wide = describing.graph_text(long, describing.NAMED["clock"], filed("truth"))
    assert len(wide) == 3 + 200 + 1 + 1 + 2 + 200 + 1 + 2 + 2 + 1 + 6 + 1 and max(map(len, wide)) < 300
    assert wide[204].split() == ["time", "before", "1000"] + [str(moment) for moment in range(1005, 1014)] + ["1299"]
    assert wide[407] == OMITTED % (12, 301, "columns", 195, 289, "truth")
    LOG.info("descriptions bounded truth exit seconds=%.3f", time.perf_counter() - started)


TURNS = list(range(100, 140))
WAVING = [[step, step % 3] for step in range(40)]
LONG_VARIANT = dict(
    name="long",
    view=VIEWED["g0"],
    times=TURNS,
    tasks=dict(turn=TURNS, visit=[]),
    held=dict(kinds=dict(spot=dict(nodes=[[0, 1], [5, 2], [30, 1]], anchors=NIL)), relations=dict(near=dict(edges=WAVING))),
    effects=[
        dict(
            entry="test.cut",
            operation="test.cut",
            kinds=dict(spot=tally(55, 0, 0.0, 0, 0.0, 55, nodes=[[0, 1], [5, 2], [30, 1]], anchors=NIL)),
            relations=dict(near=tally(39, 39, 1.0, 0, 0.0, 0, edges=NIL)),
            values=UNVALUED,
            cuts=[dict(sort="relation", name="near", task="turn", times=TURNS[1:], everywhere=True, without=[])],
        )
    ],
)
LONG_STEPS = """  held at each time before the first entry:
    time          100  103  105  106  109  112  115  118  121  124  130  139
    spot nodes    1    1    2    2    2    2    2    2    2    2    1    1
    spot anchors  0    0    0    0    0    0    0    0    0    0    0    0
    near edges    0    0    2    0    0    0    0    0    0    0    0    0
%s""" % (OMITTED % (12, 40, "columns", 28, 28, "variants"))
LONG_CUT = (
    "  entry test.cut, of the operation test.cut, removes every edge of the relation near at the times of the tasks of turn: "
    "39 times from 101 to 139; and at every time"
)
WAVERING = [[None, 0]] + [[moment, 1 + moment % 2] for moment in range(50)]
LONG_DIFFERENCE = """  kind spot: nodes only the first holds 75, only the second 0; anchors only the first holds 0, only the second 0; nodes both hold with other values 0
    time  first nodes  second nodes  first anchors  second anchors  other values
    0     1            0             0              0               0
    1     2            0             0              0               0
    2     1            0             0              0               0
    3     2            0             0              0               0
    4     1            0             0              0               0
    5     2            0             0              0               0
    6     1            0             0              0               0
    7     2            0             0              0               0
    8     1            0             0              0               0
    9     2            0             0              0               0
    10    1            0             0              0               0
    49    2            0             0              0               0
%s
  equal at every time: kinds 0, relations 2""" % (OMITTED % (12, 50, "rows", 38, 38, "differences"))


def test_the_texts_of_a_variant_and_of_a_difference_are_bounded_as_that_of_a_truth_graph_is_whatever_their_steps(bank):
    started = time.perf_counter()
    lines = effect.graph_text(LONG_VARIANT, describing.NAMED["clock"], filed("variants"))
    assert lines[1] == "graph long, %s, 40 steps from time 100 to 139" % (SEEN_BY % "g0")
    assert lines[2] == "  tasks: turn at 40 times from 100 to 139; visit at no time"
    assert lines[3:9] == LONG_STEPS.split("\n") and lines[-1] == LONG_CUT
    after = lines.index("  held at each time after it:")
    assert lines[after + 1].split() == ["time", "100", "105", "130", "139"] and lines[after + 3].split() == ["spot", "anchors"] + ["0"] * 4
    assert lines[after + 5] == OMITTED % (4, 40, "columns", 0, 36, "variants") and max(map(len, lines)) < 300
    still = effect.graph_text(dict(LONG_VARIANT, times=[None] * 40, tasks=dict(turn=[None] * 40, visit=[])), None, filed("variants"))
    assert still[2] == "  tasks: turn at the step; visit at no step"
    assert still[4].split() == ["step", "0", "3", "5", "6", "9", "12", "15", "18", "21", "24", "30", "39"]
    found = dict(
        times=[None] + list(range(50)),
        kinds=dict(spot=unlike(NODAL, "both", (75, 0, 0, 0, 0), WAVERING, UNHELD, UNHELD, UNHELD, UNHELD)),
        relations={},
        equal=dict(kinds=0, relations=2),
    )
    heads = [describing.BEFORE] + list(range(50))
    told = differing.found_text(found, describing.NAMED["clock"], found["times"], heads, filed("differences"))
    assert told == LONG_DIFFERENCE.split("\n")
    stepped = dict(found, kinds=dict(spot=unlike(NODAL, "both", (75, 0, 0, 0, 0), WAVERING[1:], NIL, NIL, NIL, NIL)))
    shown = dict(stepped, name="long", view=VIEWED["g0"], times=TURNS + list(range(140, 150)), alone=dict(first=TURNS, second=[]))
    lines = differing.graph_text(shown, describing.NAMED["clock"], filed("differences"))
    assert lines[1].endswith("; times only the first shows: 40 times from 100 to 139; times only the second shows: none")
    assert [line.split()[0] for line in lines[4:16]] == [str(moment) for moment in (*range(100, 111), 149)]
    assert lines[16] == LONG_DIFFERENCE.split("\n")[14]
    LOG.info("descriptions bounded variant and difference exit seconds=%.3f", time.perf_counter() - started)
