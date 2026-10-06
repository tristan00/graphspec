import logging
import time

import numpy as np
import pytest
from test_decisions import banks, opened, registered
from test_descriptions import (
    APART,
    LOAD,
    NIL,
    NONE,
    UNREAD,
    aim,
    batched,
    branched,
    cutting_entry,
    effected,
    entered,
    grouped,
    kept,
    kept_as,
    nowhere,
    own,
    realised,
    role,
    said,
    setting,
    tally,
    under,
)
from test_edited import edits
from test_realisations import PLAIN, SAMPLED, configured, modelled

from conftest import SPLITS, inline
from graphspec import describe
from graphspec.descriptions import describe as describing
from graphspec.descriptions import effect
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.schema.schema import VERSION as SCHEMA
from graphspec.sources import synthetic
from tuning import configuration, reach
from tuning import describe as variant
from tuning.bank import forks
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.bank.options import offered
from tuning.batches import batching
from tuning.descriptions import reach as reaching
from tuning.families.heads import REALISATION, fielded
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("decisions")
SHARED = (banks, registered, edits)
EDGES = "graphspec.edges"
WHOLE = dict(trained=dict(stated=68, held=68, put=200), scored=dict(stated=68, held=68, put=198))
SPARSE = dict(trained=dict(stated=68, held=33, put=73), scored=dict(stated=68, held=31, put=72))
CHOSEN = dict(trained=dict(stated=12, held=12, put=12), scored=dict(stated=18, held=18, put=18))
CHOSEN_SPARSE = dict(trained=dict(stated=12, held=2, put=2), scored=dict(stated=18, held=5, put=5))
LASTING = dict(PLAIN, **{"edges/base/persistence": 2})
AT = "target at of put"
FORKED = dict(stated=136, held=136, put=375, facts=197, shown=171)
CUT_TEXT = [
    "  cut 1, of being put before the model: by the entry graphspec.edges",
    "    graphspec.edges, its parameters re-stated: statements giving it back 1, not giving it back 10, refused 0, "
    "stopped 0, not tried 0; the parameters of graphspec.derive, which it reads, not re-stated",
    "      given back by edges/base/persistence 1",
]
SEEN = [
    "kind of show",
    "kind of turn",
    "parameter lit of put of turn",
    "parameter mode of cut of turn",
    "parameter speed of go of show",
    "parameter speed of go of turn",
    "target along of cut of turn",
    "target at of put of turn",
    "target made of put of turn",
    "target to of go of show",
    "target to of go of turn",
]
NUMBER = dict(type="number", encoder="value")
MODEL = {"view": "snapshot", REALISATION: "vector"}
FILTERED = {"edges/direction": "inverse", "edges/base/drop": 0.0, "edges/base/persistence": 2, "edges/base/cap": 0}
PIPELINE = [dict(operation=EDGES, parameters=FILTERED)]
SETTLED = {"graphs/join": "separate", "nodes/untyped": False, "relations/direction": "inverse"}
FIELDS = ("target to of go", "parameter speed of go")
CASES = dict(
    still=("none", None, "n0", None, None, [("s0", None)], "tasks"),
    placed=("order", "key", "p0", "state", (1, 3, 3), [("s0", (1, 3)), ("s1", (2, 3))], "changes"),
)
STILL_TRUTH = """
graph n0, no unit, no time
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 2, observations 2, replaced 0
  relation base of the table base: ends agent to spot; no columns; edges 1, facts 1, absent 0, replaced 0; runs 1, bounded 0
  held always:
    held                always
    agent nodes         1
    agent observations  1
    spot nodes          2
    spot observations   2
    base edges          1
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 3; train 2, validation 1, no split 0
  decisions of turn: decisions 3, options 6, recorded 6, regenerated 0; no generator; no rule of options
    task  time  options  recorded  regenerated
    10    none  2        2         0
    20    none  2        2         0
    30    none  2        2         0
"""
STILL_VARIANT = """
graph n0, the graph n0 as the tasks of the splits train, validation see it, 1 steps, no time
  tasks: turn at the step
  held at each step before the first entry:
    step           0
    agent nodes    1
    agent anchors  0
    spot nodes     2
    spot anchors   0
    base edges     1
  entry graphspec.edges, of the operation graphspec.edges:
    held         before  removed  share  added  share  after
    agent nodes  1       0        0.0    0      0.0    1
    spot nodes   2       0        0.0    0      0.0    2
    base edges   1       1        1.0    0      0.0    0
  columns of values and values of nodes: none added, removed, filled or changed
  held at each step after it:
    step           0
    agent nodes    1
    agent anchors  0
    spot nodes     2
    spot anchors   0
    base edges     0
  entry graphspec.edges, of the operation graphspec.edges, removes every edge of the relation base at the step of the tasks of turn; and at every step
"""
PLACED_TRUTH = """
graph p0, no unit, positions 4 from 0 to 3
  kind agent: nodes 1, observations 1, replaced 0
  kind spot: nodes 2, observations 2, replaced 0
  relation base of the table base: ends agent to spot; no columns; edges 2, facts 2, absent 0, replaced 0; runs 2, bounded 2
  held before every position and at each position:
    position            before  0  1  2  3
    agent nodes         0       1  1  1  1
    agent observations  0       1  1  1  1
    spot nodes          0       2  2  2  2
    spot observations   0       2  2  2  2
    base edges          0       0  1  2  0
  shown to some tasks only: no fact known later than it holds from; no fact naming observers
  not counted above: no member of a set edge; no graph a node holds; no entity without a fact
  kind of task turn: tasks 3; train 2, validation 1, no split 0
  decisions of turn: decisions 3, options 6, recorded 6, regenerated 0; no generator; no rule of options
    task  position  options  recorded  regenerated
    10    1         2        2         0
    20    3         2        2         0
    30    3         2        2         0
"""
PLACED_VARIANT = """
graph p0, the graph p0 as the tasks of the splits train, validation see it, 4 steps from position 0 to 3
  tasks: turn at 1, 3
  held at each position before the first entry:
    position       0  1  2  3
    agent nodes    1  1  1  1
    agent anchors  0  0  0  0
    spot nodes     2  2  2  2
    spot anchors   0  0  0  0
    base edges     0  1  2  0
  entry graphspec.edges, of the operation graphspec.edges:
    held         before  removed  share   added  share  after
    agent nodes  4       0        0.0     0      0.0    4
    spot nodes   8       0        0.0     0      0.0    8
    base edges   3       2        0.6667  0      0.0    1
  columns of values and values of nodes: none added, removed, filled or changed
  held at each position after it:
    position       0  1  2  3
    agent nodes    1  1  1  1
    agent anchors  0  0  0  0
    spot nodes     2  2  2  2
    spot anchors   0  0  0  0
    base edges     0  0  1  0
  entry graphspec.edges, of the operation graphspec.edges, removes every edge of the relation base at the positions of the tasks of turn: 1; not at every position
"""
CUT_BY = [
    "  cut 1, of being joined to a reference: by the entry graphspec.edges",
    "    graphspec.edges, its parameters re-stated: statements giving it back 1, not giving it back 6, refused 0, "
    "stopped 0, not tried 0",
    "      given back by edges/base/persistence 1",
    "  cut 2, of being joined to a reference: by no entry, joined to a reference in no state, the view before the "
    "first entry included",
]
STILL_REACH = CUT_BY + [
    "    the truth joins it to no reference",
    "  decision 10 of turn, references agents a0",
    "  decision 20 of turn, references agents a0",
    "  decision 30 of turn, references agents a0",
]
PLACED_REACH = CUT_BY + [
    "    the truth joins it to no reference at the position of the task",
    "  decision 10 of turn at position 1, references agents a0",
    "  decision 20 of turn at position 3, references agents a0",
    "  decision 30 of turn at position 3, references agents a0",
]
TEXTS = dict(still=(STILL_TRUTH, STILL_VARIANT, STILL_REACH), placed=(PLACED_TRUTH, PLACED_VARIANT, PLACED_REACH))


@timed("reach_fit.filtered")
def filtered(relation, persistence, back):
    capped = [under(64, (EDGES, "edges/%s/cap_order" % relation, order)) for order in ("recent", "random")]
    return [
        setting(EDGES, "edges/%s/drop" % relation, 0.0, [], [0.9], []),
        setting(EDGES, "edges/%s/persistence" % relation, persistence, back, [8], []),
        setting(EDGES, "edges/%s/cap" % relation, 0, [], capped, []),
    ]


@timed("reach_fit.cut")
def cut(on, spared, *relations):
    settings = [setting(EDGES, "edges/direction", "inverse", [], ["symmetric", "directed"], [])]
    for relation, persistence, back in relations:
        settings += filtered(relation, persistence, back)
    return dict(on=on, entries=[dict(cutting_entry(EDGES, EDGES, settings, None), spared=spared)], truth=None)


def compared(held, cfg, name):
    started = time.perf_counter()
    told, counts, every = said(held, cfg, name), {}, set()
    for split, partition in (("trained", 0), ("scored", 1)):
        stated, packed = batched(held, cfg, name, partition)
        assert {option for option in stated if told[option][0]} == set(packed), (name, split)
        assert {option: told[option][1] for option in packed} == packed, (name, split)
        counts[split] = dict(stated=len(stated), held=len(packed), put=sum(map(len, packed.values())))
        every |= stated
    assert set(told) == every, name
    LOG.info("reach_fit compared exit realisation=%s counts=%s seconds=%.3f", name, counts, time.perf_counter() - started)
    return counts


def test_reach_names_the_options_and_fields_the_batches_of_a_fit_hold_of_the_stated_bank_whole_and_with_spots_sampled(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    for name in ("vector", "actor_edge"):
        for graph, counts in ((PLAIN, WHOLE), (SAMPLED, SPARSE)):
            assert compared(held, configured(held, name, modelled("snapshot"), graph), name) == counts, (name, graph)
    LOG.info("reach_fit stated against the packing exit seconds=%.3f", time.perf_counter() - started)


def test_reach_names_the_options_and_fields_the_batches_of_a_fit_hold_of_the_picked_bank_under_the_node_realisation(banks):
    started = time.perf_counter()
    held = opened(banks, "picked")
    for graph, counts in ((PLAIN, CHOSEN), (SAMPLED, CHOSEN_SPARSE)):
        assert compared(held, configured(held, "node", modelled("snapshot"), graph), "node") == counts, graph
    LOG.info("reach_fit picked against the packing exit seconds=%.3f", time.perf_counter() - started)


def roled(held, cfg):
    started = time.perf_counter()
    found, result = offered(held), {}
    with forks.Forks(held):
        for group in [group for partition in (0, 1) for group in grouped(held, cfg, partition)]:
            for part in [part for part, *_ in batching.parted(held, group, cfg) if part.option >= 0]:
                history = forks.trunk(held, cfg, part.graph, part.step)
                plan = forks.planned(held, cfg, history, part, forks.ordered(held, part.option))
                result[part.option] = [int(found.role["code"][fact[5]]) for fact in plan.facts]
    LOG.debug("reach_fit roled exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def test_under_the_edit_and_the_setting_of_28_september_reach_names_the_facts_the_forks_of_a_fit_hold_and_the_value_that_gives_a_field_back(
    banks, statement
):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = configured(held, "edited", modelled("snapshot"), LASTING)
    packed = {option: found for partition in (0, 1) for option, found in branched(held, cfg, partition).items()}
    told, codes = said(held, cfg, "edited"), roled(held, cfg)
    assert {option: (found, facts) for option, (found, _, facts) in told.items()} == packed
    assert FORKED == dict(
        stated=len(packed),
        held=sum(found for found, _ in packed.values()),
        put=sum(len(put) for _, put, _ in told.values()),
        facts=sum(len(facts) for _, facts in packed.values()),
        shown=sum(shown for _, facts in packed.values() for shown, _ in facts),
    )
    (at,) = [field["code"] for field in fielded(held.registry.schema, "turn") if field["name"] == AT]
    shown = {
        option: [found for (found, _), code in zip(packed[option][1], made, strict=True) if code == at]
        for option, made in codes.items()
        if at in made
    }
    lacking = {option for option, found in shown.items() if not any(found)}
    assert (len(shown), len(lacking)) == (22, 15)
    assert {option for option in shown if AT not in told[option][1]} == lacking
    record = reach(banks["stated"], cfg, dict(statement, loss=dict(decision="squared"), measures=["mse"]))
    options = [option for graph in record["graphs"] for decision in graph["decisions"] for option in decision["options"]]
    stating = [option["fields"][AT] for option in options if AT in option["fields"]]
    unput = [field for field in stating if not field["put"]]
    assert (len(options), len(stating), len(unput)) == (136, 22, 15)
    assert {target["cut"] for field in unput for target in field["targets"]} == {0}
    for graph in record["graphs"]:
        assert graph["cuts"][0] == cut("put", ["graphspec.derive"], ("near", 1, []), ("base", 2, [1])), graph["name"]
        assert reaching.cut_text(0, graph["cuts"][0], describing.NAMED["clock"]) == CUT_TEXT, graph["name"]
    LOG.info("reach_fit edit against the forks exit seconds=%.3f", time.perf_counter() - started)


def test_the_fields_the_guards_of_the_smallest_fit_see_are_those_reach_puts_before_the_model_for_an_option_held_in_the_scored_splits(
    banks, tmp_path, statement, monkeypatch
):
    started = time.perf_counter()
    held = opened(banks, "stated")
    inline(banks["stated"], (), monkeypatch)
    cfg = dict(configured(held, "vector", modelled("snapshot"), SAMPLED), epochs=1)
    given = state(dict(statement, loss=dict(decision="squared"), measures=["mse"]))
    blocks = held.blocks(0, cfg["graph"]["time/history"], cfg)
    fit(banks["stated"], tmp_path, cfg, blocks, 16, time.perf_counter() + statement["seconds"], given)
    guards = read_json(tmp_path / "result.json")["options"]
    found, schema, told = offered(held), held.registry.schema, said(held, cfg, "vector")
    kinds = list(schema.tasks["kinds"])
    decision = found.decision[held.members(cfg, 1)]
    scored = np.flatnonzero(np.isin(found.option["decision"], decision[decision >= 0])).tolist()
    packed = [option for option in scored if told[option][0]]
    put = {"%s of %s" % (name, kinds[int(found.option["kind"][option])]) for option in packed for name in told[option][1]}
    every = {"%s of %s" % (field["name"], task) for task in schema.decided() for field in fielded(schema, task)}
    assert every - set(guards["unseen"]) == put == set(SEEN)
    assert (len(every), len(scored), len(packed), guards["options"]) == (16, 68, 31, 31)
    LOG.info("reach_fit against the guards exit seconds=%.3f", time.perf_counter() - started)


@timed("reach_fit.schema")
def schema(model, ties):
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
        generators=[],
    )
    return dict(
        version=SCHEMA,
        time=dict(model=model, units=[], ties=ties),
        observers=False,
        columns=dict(load=NUMBER, speed=NUMBER),
        graph=dict(columns=[]),
        nodes=dict(agent=dict(columns=["load"], holds=None), spot=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(base=synthetic.edge("agent", "spot", [])),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(agents=dict(sort="nodes", kinds=["agent"])),
            kinds=dict(turn=synthetic.kind(["agents"], [], decided, None)),
        ),
    )


@timed("reach_fit.turned")
def turned(gain):
    options = [
        synthetic.offered("o%d" % at, [synthetic.acting("go", "a0", dict(to=[spot]), dict(speed=1.0 + at))], "set", "recorded")
        for at, spot in enumerate(("s0", "s1"))
    ]
    options[0].update(chosen=True, outcome=dict(gain=gain))
    return dict(options=options, concurrent=[], next={}, ended=None, discount=None)


@timed("reach_fit.asked")
def asked(moments):
    tasks = dict(
        key=[10, 20, 30],
        kind=["turn"] * 3,
        split=["train", "train", "validation"],
        weight=[None] * 3,
        references=dict(agents=[["a0"]] * 3),
        target=[turned(1.0), turned(2.0), turned(3.0)],
        known=[None] * 3,
    )
    return tasks if moments is None else dict(tasks, time=list(moments), known=[moment + 1 for moment in moments])


@timed("reach_fit.stated")
def stated(form, spans, rows, **named):
    held = {} if form is None else dict(since=[since for since, _ in spans], until=[until for _, until in spans])
    return dict(named, form=form, parameters={}, rows=dict(rows, **held))


@timed("reach_fit.case")
def case(model, ties, name, form, moments, based):
    values = [
        stated(form, [(0, None)], dict(key=["a0"], load=[1.0]), kind="agent"),
        stated(form, [(0, None)] * 2, dict(key=["s0", "s1"], load=[1.0, 2.0]), kind="spot"),
    ]
    joined = dict(source=["a0"] * len(based), target=[spot for spot, _ in based], present=[True] * len(based))
    edges = [stated(form, [span for _, span in based], joined, table="base", ends=("agent", "spot"))]
    nodes = dict(agent=["a0"], spot=["s0", "s1"])
    graph = dict(name=name, unit=None, columns={}, nodes=nodes, values=values, edges=edges, tasks=asked(moments))
    return dict(schema=schema(model, ties), graphs=[graph])


@timed("reach_fit.truthful")
def truthful(name, times, agents, spots, based, moments):
    each = [
        dict(key=key, time=moment, options=2, recorded=2, regenerated=0, generators={})
        for key, moment in zip((10, 20, 30), moments, strict=True)
    ]
    joined = dict(table="base", ends=[["agent", "spot"]], columns=[], about=None, lapsed=0, absent=0, replaced=0)
    return dict(
        name=name,
        unit=None,
        times=times,
        kinds=dict(
            agent=dict(nodes=1, observations=1, replaced=0, nodes_held=agents, observations_held=agents),
            spot=dict(nodes=2, observations=2, replaced=0, nodes_held=spots, observations_held=spots),
        ),
        relations=dict(base=dict(joined, **based, **NONE)),
        tasks=dict(turn=dict(tasks=3, splits=dict(train=2, validation=1), unsplit=0)),
        decisions=dict(turn=dict(decisions=3, options=6, recorded=6, regenerated=0, generators={}, rule=None, each=each)),
        unread=UNREAD,
    )


@timed("reach_fit.viewed")
def viewed(name):
    return dict(graph=name, observer=None, splits=["train", "validation"])


@timed("reach_fit.varied")
def varied(name, times, tasked, agents, spots, before, based, after, left, everywhere):
    kinds = dict(agent=dict(nodes=[[0, 1]], anchors=NIL), spot=dict(nodes=[[0, 2]], anchors=NIL))
    removed = dict(sort="relation", name="base", task="turn", times=left, everywhere=everywhere, without=[])
    return dict(
        name=name,
        view=viewed(name),
        times=times,
        tasks=dict(turn=tasked),
        held=dict(kinds=kinds, relations=dict(base=dict(edges=before))),
        effects=[
            effected(
                EDGES,
                dict(
                    agent=tally(agents, 0, 0.0, 0, 0.0, agents, **kinds["agent"]),
                    spot=tally(spots, 0, 0.0, 0, 0.0, spots, **kinds["spot"]),
                ),
                dict(base=tally(*based, edges=after)),
                [removed],
            )
        ],
    )


@timed("reach_fit.reached")
def reached(name, decisions):
    return dict(
        name=name,
        view=viewed(name),
        decided=dict(turn=dict(decisions=3, described=3)),
        cuts=[cut("joined", [], ("base", 2, [1])), nowhere("joined")],
        decisions=[
            dict(
                task=task,
                kind="turn",
                time=moment,
                references=dict(agents=["a0"]),
                options=[
                    dict(key=key, **kept({FIELDS[0]: role("target", aim(spot, APART, LOAD, place)), FIELDS[1]: own("option", None)}))
                    for key, spot, place in (("o0", "s0", first), ("o1", "s1", 1))
                ],
            )
            for task, moment, first in decisions
        ],
    )


TRUTHS = dict(
    still=truthful("n0", [], [[None, 1]], [[None, 2]], dict(edges=1, facts=1, edges_held=[[None, 1]], runs=1, bounded=0), [None] * 3),
    placed=truthful(
        "p0",
        [0, 1, 2, 3],
        [[None, 0], [0, 1]],
        [[None, 0], [0, 2]],
        dict(edges=2, facts=2, edges_held=[[None, 0], [1, 1], [2, 2], [3, 0]], runs=2, bounded=2),
        [1, 3, 3],
    ),
)
VARIANTS = dict(
    still=varied("n0", [None], [None], 1, 2, [[0, 1]], (1, 1, 1.0, 0, 0.0, 0), NIL, [None], True),
    placed=varied(
        "p0",
        [0, 1, 2, 3],
        [1, 3],
        4,
        8,
        [[0, 0], [1, 1], [2, 2], [3, 0]],
        (3, 2, 0.6667, 0, 0.0, 1),
        [[0, 0], [2, 1], [3, 0]],
        [1],
        False,
    ),
)
REACHES = dict(
    still=reached("n0", [(10, None, 0), (20, None, 0), (30, None, 0)]),
    placed=reached("p0", [(10, 1, 0), (20, 3, 1), (30, 3, 1)]),
)


def test_a_bank_without_time_and_one_of_order_built_as_such_are_described_in_the_words_of_their_model_as_the_records_written_out_here(
    tmp_path, statement
):
    started = time.perf_counter()
    given = dict(statement, loss=dict(decision="squared"), measures=["mse"])
    for name, (model, ties, graph, form, moments, based, steps) in CASES.items():
        root = tmp_path / name
        root.mkdir()
        bank = root / "bank"
        build(synthetic.stage(root, name, case(model, ties, graph, form, moments, based)), bank, 1, ("train",))
        settings = dict(SETTLED, **{"time/steps": steps})
        shown = dict(settings, **{"time/history": 1, "time/context": 0})
        config = configuration(bank, MODEL, None, pipeline=PIPELINE, graph=shown, seed=statement["seed"], splits=SPLITS)
        seed = statement["derivation_seed"]
        head = dict(
            version=describing.VERSION,
            stored_sha256=read_json(bank / "manifest.json")["stored_sha256"],
            time=dict(model=model, units=[]),
        )
        derived = dict(
            head,
            variant=specs.digest(Deriver(bank, "cpu", seed, RESIDENT).identity(config)),
            seed=seed,
            device="cpu",
            splits=SPLITS,
            settings=settings,
            unshown=[],
            entries=entered(PIPELINE),
        )
        read = dict(realisation=realised("vector"), refused=None, tasks=dict(turn=dict(fields=list(FIELDS), unread=[])), undecided=[])
        truth, changed, told = describe(bank), variant(bank, config, given), reach(bank, config, given)
        assert truth == dict(head, splits=["train", "validation"], graphs=[TRUTHS[name]], kept=kept_as("truth", "bank")), name
        assert changed == dict(derived, graphs=[VARIANTS[name]], kept=kept_as("variants", "bank")), name
        assert told == dict(derived, **read, graphs=[REACHES[name]], kept=kept_as("reach", "bank")), name
        written = [describing.text(truth), effect.text(changed), reaching.text(told)]
        bodies = [found[found.index("\ngraph ") :] for found in written]
        lines = bodies[2].split("\n")
        assert (bodies[0], bodies[1]) == TEXTS[name][:2], name
        assert lines[2:7] + [line for line in lines if line.startswith("  decision ")] == TEXTS[name][2], name
        assert "time" not in bodies[2] and "lasting" not in bodies[0], name
    LOG.info("reach_fit other models of time exit seconds=%.3f", time.perf_counter() - started)
