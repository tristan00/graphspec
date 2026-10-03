import copy
import logging
import re
import time
from functools import partial

import numpy as np
import pytest
import support

from conftest import SPLITS, Inline
from graphspec import ABOUT, HELD, NEGATIVES, STORAGE, inspect
from graphspec.budgets import GraphLimits
from graphspec.engine import segments
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.hashing import hashed, token
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline, sets
from graphspec.operations.nodes import NODES
from graphspec.operations.operation import Count, Refused
from graphspec.schema.schema import VERSION, Encoding, Schema, fitted
from graphspec.sources import synthetic
from graphspec.views.graph import Graph
from graphspec.views.view import sequences, view
from graphspec.views.visible import unstated
from tuning.bank.bank import Deriver
from tuning.fits.training import fit
from tuning.searches.selector import check_bounds
from tuning.statement import state

LOG = logging.getLogger(__name__)
configuration = support.configuration
pytestmark = pytest.mark.xdist_group("sets")
POINTS = tuple("p%d" % number for number in range(16))
PLACES = tuple("q%d" % number for number in range(8))
SIZES = (2, 3, 5, 8, 12, 20)
LINKS = (("l0", "p0", "p1", 0, 4), ("l1", "p2", "p3", 2, None), ("l2", "p4", "p5", 0, None))
NOTES = (("l0", "q0", 1, 7), ("l1", "q1", 0, 3), ("l2", "q2", 3, None))
OWNERS = dict(box=("b0", "b1", "b2"), tray=("t0", "t1"))
INNER = (("n0", "piece"), ("n1", "piece"), ("n2", "piece"), ("n3", "corner"), ("n4", "corner"))
JOINTS = (("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n0", "n4"))
KINDS = ("total", "around", "inside")
FORMS = dict(node=dict(gathering="node", link="node"), pairs=dict(gathering="pairs", link="node"))
STORED = {
    form: {"sets/%s/storage" % kind: name for kind, name in chosen.items()} for form, chosen in FORMS.items()
}
VALUE = dict(type="number", encoder="value")
OFFERED = "pipeline/graphspec.negatives/negatives/of"


@timed("sets.held")
def held(encoder):
    return dict(
        nodes=dict(piece=dict(columns=["mass"]), corner=dict(columns=["mass", "tint"])),
        edges=dict(
            joint=dict(endpoints=[["piece", "piece"], ["piece", "corner"]], directed=False, columns=["tie"])
        ),
        encoder=encoder,
    )


@timed("sets.stated")
def stated():
    kinds = {
        kind: dict(
            references=[name],
            columns=[],
            target=dict(kind="number", series=kind, about=None, utility=synthetic.valued([kind])),
            weight=False,
            absent="keep",
            event=None,
            matures=None,
        )
        for kind, name in zip(KINDS, ("whole", "group", "crate"))
    }
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(
            level=VALUE,
            load=VALUE,
            share=dict(type="number", encoder="standard"),
            scale=VALUE,
            strength=VALUE,
            mark=VALUE,
            tie=VALUE,
            mass=dict(type="number", encoder="standard"),
            tint=dict(type="category", encoder="onehot"),
        ),
        graph=dict(columns=[]),
        nodes=dict(
            point=dict(columns=["level"], holds=None),
            place=dict(columns=["load"], holds=None),
            box=dict(columns=["level"], holds=held("propagated")),
            tray=dict(columns=["load"], holds=held("pooled")),
        ),
        sets=dict(
            gathering=dict(
                kinds=["point", "place"], roles=["lead", "other"], members=["share"], columns=["scale"]
            ),
            link=dict(kinds=["point"], roles=["head", "tail"], members=[], columns=["strength"]),
        ),
        edges=dict(
            near=synthetic.edge("point", "place", []),
            note=dict(
                endpoints=[["link", "place"]],
                directed=True,
                columns=["mark"],
                relation="declared",
                about="lapse",
            ),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(
                whole=dict(sort="graph", kinds=[]),
                group=dict(sort="node", kinds=["gathering"]),
                crate=dict(sort="node", kinds=["box", "tray"]),
            ),
            kinds=kinds,
        ),
    )


@timed("sets.lasting")
def lasting(kind, keys, column, values, since, until):
    rows = dict(key=list(keys), since=list(since), until=list(until), **{column: list(values)})
    return dict(form="state", parameters={}, rows=rows, kind=kind)


@timed("sets.gathered")
def gathered(place):
    everyone = list(POINTS) + list(PLACES)
    rows = dict(set=[], member=[], role=[], share=[])
    for number, size in enumerate(SIZES):
        chosen = (everyone[3 * number :] + everyone[: 3 * number])[:size]
        rows["set"] += ["g%d" % number] * size
        rows["member"] += chosen
        rows["role"] += ["lead"] + ["other"] * (size - 1)
        rows["share"] += [0.1 * (at + 1) + place for at in range(size)]
    return rows


@timed("sets.inner")
def inner(owners, place):
    nodes = dict(owner=[], kind=[], key=[], mass=[], tint=[])
    edges = dict(owner=[], source=[], target=[], relation=[], tie=[])
    for number, owner in enumerate(owners):
        for at, (key, kind) in enumerate(INNER):
            nodes["owner"].append(owner)
            nodes["kind"].append(kind)
            nodes["key"].append(key)
            nodes["mass"].append(1.0 + at + number + place)
            nodes["tint"].append(None if kind == "piece" else ("dark", "light")[(at + number) % 2])
        for at, (source, target) in enumerate(JOINTS[: 3 + number % 2]):
            edges["owner"].append(owner)
            edges["source"].append(source)
            edges["target"].append(target)
            edges["relation"].append("joint")
            edges["tie"].append(0.5 * at + number)
    return dict(nodes=nodes, edges=edges)


@timed("sets.graph")
def graph(place, name):
    count = len(SIZES)
    values = [
        lasting("point", POINTS, "level", [1.0 + at + place for at in range(16)], [0] * 16, [None] * 16),
        lasting("place", PLACES, "load", [2.0 + at for at in range(8)], [0] * 8, [None] * 8),
        lasting("box", OWNERS["box"], "level", [3.0, 4.0, 5.0 + place], [0] * 3, [None] * 3),
        lasting("tray", OWNERS["tray"], "load", [6.0, 7.0 + place], [0] * 2, [None] * 2),
        lasting(
            "gathering",
            ["g%d" % at for at in range(count)],
            "scale",
            [0.5 * at + place for at in range(count)],
            [at % 2 for at in range(count)],
            [None] * count,
        ),
        lasting(
            "link",
            [link[0] for link in LINKS],
            "strength",
            [1.0 + at for at in range(len(LINKS))],
            [link[3] for link in LINKS],
            [link[4] for link in LINKS],
        ),
    ]
    near = dict(
        source=list(POINTS),
        target=[PLACES[at % 8] for at in range(16)],
        present=[True] * 16,
        since=[0] * 16,
        until=[None] * 16,
    )
    notes = dict(
        source=[note[0] for note in NOTES],
        target=[note[1] for note in NOTES],
        present=[True] * len(NOTES),
        mark=[0.25 * at + place for at in range(len(NOTES))],
        since=[note[2] for note in NOTES],
        until=[note[3] for note in NOTES],
    )
    edges = [
        dict(form="state", parameters={}, rows=near, table="near", ends=("point", "place")),
        dict(form="state", parameters={}, rows=notes, table="note", ends=("link", "place")),
    ]
    links = dict(set=[], member=[], role=[])
    for key, head, tail, _, _ in LINKS:
        links["set"] += [key, key]
        links["member"] += [head, tail]
        links["role"] += ["head", "tail"]
    times = [moment for moment in range(8) for _ in range(2)]
    kinds = [KINDS[at % 3] for at in range(16)]
    crates = list(OWNERS["box"]) + list(OWNERS["tray"])
    tasks = dict(
        key=[100 * place + at for at in range(16)],
        kind=kinds,
        time=times,
        split=["train" if moment < 6 else "validation" for moment in times],
        weight=[None] * 16,
        references=dict(
            whole=[None] * 16,
            group=["g%d" % (at % count) if kind == "around" else None for at, kind in enumerate(kinds)],
            crate=[crates[at % 5] if kind == "inside" else None for at, kind in enumerate(kinds)],
        ),
        target=[1.0 + 0.5 * at + place + (at % 3) for at in range(16)],
        known=[None] * 16,
    )
    return dict(
        name=name,
        unit="tick",
        columns={},
        nodes=dict(
            point=list(POINTS),
            place=list(PLACES),
            box=list(OWNERS["box"]),
            tray=list(OWNERS["tray"]),
            gathering=["g%d" % at for at in range(count)],
            link=[link[0] for link in LINKS],
        ),
        values=values,
        edges=edges,
        sets=dict(gathering=gathered(place), link=links),
        held={kind: inner(owners, place) for kind, owners in OWNERS.items()},
        tasks=tasks,
    )


@timed("sets.case")
def case():
    return dict(schema=stated(), graphs=[graph(place, name) for place, name in enumerate(("one", "two"))])


@pytest.fixture(scope="module")
def bank(sets_bank):
    started = time.perf_counter()
    LOG.info("bank of sets exit seconds=%.3f", time.perf_counter() - started)
    return sets_bank


@timed("sets.shown")
def shown(held, facts, form):
    stored = sets.stored(facts, held.registry, FORMS[form])
    (sequence,) = sequences(stored, held.registry, "tasks", unstated(held.splits))
    return view(stored, held.registry, sequence, "last")


@timed("sets.present")
def present(held, graph, relation):
    dense = graph.dense()
    rows = dense.fact_present & (dense.fact_code == held.registry.code(relation))
    return sorted(
        (int(step), graph.keys[source], graph.keys[target])
        for step, source, target in zip(
            dense.fact_step[rows].tolist(), dense.fact_source[rows].tolist(), dense.fact_target[rows].tolist()
        )
    )


def test_sets_edges_about_edges_and_held_graphs_read_back_from_the_bank_equal_to_what_was_stated(bank):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    schema = held.registry.schema
    assert held.manifest["version"] == 13 and schema.mapping() == stated()
    found, wanted = support.read_back(held), support.meant(case())
    assert list(found) == list(wanted) == ["one", "two"]
    for entry in held.manifest["graphs"]:
        read = found[entry["name"]]
        assert support.differing(read, wanted[entry["name"]], schema) == [], entry["name"]
        members = read["sets"]["gathering"]["set"]
        assert sorted(members.count(key) for key in set(members)) == list(SIZES)
        assert entry["members"] == sum(SIZES) + 2 * len(LINKS) and entry["held"] == 5 * 5
        for kind, owners in OWNERS.items():
            nodes = read["held"][kind]["nodes"]
            assert [nodes["owner"].count(owner) for owner in owners] == [5] * len(owners)
        notes = read["edges"]["note"]
        assert sorted(zip(notes["source"], notes["since"], notes["until"])) == sorted(
            (note[0], note[2], note[3]) for note in NOTES
        )
        links = read["observations"]["link"]
        assert sorted(zip(links["key"], links["since"], links["until"])) == sorted(
            (link[0], link[3], link[4]) for link in LINKS
        )


def test_the_schema_states_sets_edges_about_edges_and_held_graphs_and_refuses_what_it_cannot_read():
    mapping = stated()
    schema = Schema(mapping)
    assert schema.layout() == ("level", "load", "scale", "strength")
    assert schema.member_columns() == ("share",) and schema.inner_columns() == ("mass", "tint")
    assert schema.about() == dict(note="lapse") and schema.tables()["edge_note"]["pairs"] == (
        ("link", "place"),
    )
    assert schema.tables()["member_gathering"]["pairs"] == (("gathering", "point"), ("gathering", "place"))
    assert schema.tables()["held_box"]["columns"] == ("mass", "tint")
    assert schema.tables()["held_edge_tray"]["pairs"] == (("held_tray", "held_tray"),)
    broken = copy.deepcopy(mapping)
    broken["edges"]["note"]["about"] = None
    with pytest.raises(
        ValueError, match="edge table note joins a kind of set edge and states by name the rule"
    ):
        Schema(broken)
    broken["edges"]["note"]["about"] = "forever"
    with pytest.raises(
        KeyError, match="no rule for an edge about an edge is registered under the name 'forever'"
    ):
        Schema(broken)
    broken = copy.deepcopy(mapping)
    broken["sets"]["gathering"]["kinds"] = ["box", "hall"]
    with pytest.raises(
        ValueError, match="the kind of set edge gathering joins kinds of node that are stated"
    ):
        Schema(broken)
    broken = copy.deepcopy(mapping)
    broken["nodes"]["box"]["holds"]["encoder"] = "unknown"
    with pytest.raises(KeyError, match="no encoder of held graphs is registered under the name 'unknown'"):
        Schema(broken)
    broken = copy.deepcopy(mapping)
    del broken["sets"]
    with pytest.raises(ValueError, match="the schema lacks sets"):
        Schema(broken)
    with pytest.raises(ValueError, match="the rule for an edge about an edge lapse is registered twice"):
        ABOUT.register("lapse", ABOUT.named("lapse"))
    with pytest.raises(ValueError, match="the encoder of held graphs pooled is registered twice"):
        HELD.register("pooled", HELD.named("pooled"))
    with pytest.raises(
        ValueError, match="the form of storage of edges joining sets node is registered twice"
    ):
        STORAGE.register("node", STORAGE.named("node"))


def test_a_set_edge_and_an_edge_about_it_are_nodes_and_edges_under_the_node_form(bank):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = held.registry
    facts = Facts.load(held.truth(0))
    graph = shown(held, facts, "node")
    dense = graph.dense()
    gatherings = np.flatnonzero(np.asarray(graph.kinds) == registry.kind("gathering"))
    for number, entity in enumerate(gatherings.tolist()):
        steps = sorted(set(dense.obs_step[dense.obs_entity == entity].tolist()))
        assert steps == list(range(number % 2, 8))
    found = present(held, graph, "gathering.lead") + present(held, graph, "gathering.other")
    assert len(found) == sum(size * (8 - number % 2) for number, size in enumerate(SIZES))
    largest = [row for row in found if row[1] == "g5"]
    assert len(largest) == 20 * 7 and len({row[2] for row in largest}) == 20
    assert [row[2] for row in present(held, graph, "gathering.lead") if row[:2] == (1, "g5")] == ["p15"]
    notes = present(held, graph, "note")
    expected = []
    for (key, _, _, since, until), (_, place, begun, ended) in zip(LINKS, NOTES):
        low, high = max(since, begun), min(8 if until is None else until, 8 if ended is None else ended)
        expected += [(step, key, place) for step in range(low, high)]
    assert notes == sorted(expected)
    assert present(held, graph, "link.head")[0] == (0, "l0", "p0")
    references = np.flatnonzero(graph.member_name == list(registry.schema.tasks["references"]).index("group"))
    assert set(np.asarray(graph.kinds)[graph.member_source[references]].tolist()) == {
        registry.kind("gathering")
    }
    registry.about[registry.code("note")] = "stop"
    with pytest.raises(
        ValueError,
        match="the edge of note from l1 to q1 of graph one at the time 0 holds at a time when the edge it is "
        "about does not, and the rule stop states that this stops the work",
    ):
        shown(held, facts, "node")


def test_a_set_edge_is_one_edge_per_pair_of_members_under_the_pairs_form(bank):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    registry = held.registry
    facts = Facts.load(held.truth(1))
    graph = shown(held, facts, "pairs")
    dense = graph.dense()
    gatherings = np.asarray(graph.kinds)[dense.obs_entity] == registry.kind("gathering")
    assert not gatherings.any()
    pairs = present(held, graph, "gathering")
    per_step = [
        sum(size * (size - 1) // 2 for number, size in enumerate(SIZES) if number % 2 <= step)
        for step in range(8)
    ]
    assert [sum(1 for row in pairs if row[0] == step) for step in range(8)] == per_step
    assert present(held, graph, "gathering.lead") == []
    names = {parameter.name: parameter for parameter in specs.settings(registry, ())}
    assert names["sets/gathering/storage"].options == ("node", "pairs")
    assert names["sets/link/storage"].options == ("node",) and names["sets/link/storage"].default is None
    with pytest.raises(
        ValueError, match="the form pairs makes no node of the set edges of link, which the edge tables note"
    ):
        sets.stored(facts, registry, dict(gathering="node", link="pairs"))
    with pytest.raises(ValueError, match="a form of storage is stated for every kind of set edge"):
        sets.stored(facts, registry, dict(gathering="node"))
    with pytest.raises(
        ValueError, match="the pipeline states no sets/gathering/storage, which has no default"
    ):
        configuration(registry, dict(view="snapshot"), {"sets/link/storage": "node"}, 5, SPLITS)


def test_a_node_holds_a_graph_of_five_nodes_encoded_by_its_stated_encoder(bank, monkeypatch):
    monkeypatch.setattr(HELD, "held", support.entries(HELD))
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), STORED["node"], 5, SPLITS)
    registry = held.fitted(cfg)
    encoding = registry.encoding
    facts = Facts.load(held.truth(0))
    (index,) = [entry["index"] for entry in held.viewed(cfg).graphs if entry["source"] == 0]
    view = Graph.load(held.view(index, cfg))
    inner, columns = np.asarray(facts.inner_values), registry.schema.inner_columns()
    owner = np.asarray(facts.inner_owner)
    source, target = np.asarray(facts.within_source), np.asarray(facts.within_target)
    for kind in OWNERS:
        start, width = encoding.held[kind]
        for entity in np.flatnonzero(np.asarray(facts.kinds) == registry.kind(kind)).tolist():
            rows = np.flatnonzero(owner == entity)
            assert len(rows) == 5
            own = encoding.vectors(columns, inner[rows])
            if kind == "tray":
                wanted = np.r_[own.sum(0), 5.0]
            else:
                local = {row: at for at, row in enumerate(rows.tolist())}
                summed, degree = np.zeros_like(own), np.zeros(5)
                for first, second in zip(source.tolist(), target.tolist()):
                    if first in local:
                        summed[local[first]] += own[local[second]]
                        summed[local[second]] += own[local[first]]
                        degree[[local[first], local[second]]] += 1
                mean = summed / np.maximum(degree, 1)[:, None]
                wanted = np.r_[own.sum(0), mean.sum(0), 5.0]
            assert np.asarray(view.entity_held)[entity, start : start + width] == pytest.approx(
                wanted, abs=1e-5
            )
    rows = np.asarray(view.kinds)[np.asarray(view.obs_entity)] == registry.kind("box")
    assert np.asarray(view.obs_x)[rows][:, encoding.features - encoding.inner :].any()
    assert registry.widths["box"] == encoding.width(("level",)) + encoding.held["box"][1]
    assert "holds graphs encoded by propagated: " in inspect(bank)

    class Counted:
        version = 1

        def width(self, encoding, held):
            return 1

        def encode(self, encoding, held, graphs):
            return np.bincount(graphs["owner"], minlength=graphs["owners"]).astype(np.float32)[:, None]

    HELD.register("counted", Counted())
    mapping = stated()
    mapping["nodes"]["tray"]["holds"]["encoder"] = "counted"
    other = Encoding(Schema(mapping), **copy.deepcopy(encoding.declaration()))
    graphs = dict(
        owners=3, owner=np.asarray([0, 0, 2]), values=np.zeros((3, 2)), source=owner[:0], target=owner[:0]
    )
    assert other.held["tray"][1] == 1 and other.graphs("tray", graphs)[:, 0].tolist() == [2.0, 0.0, 1.0]


def test_a_column_no_view_of_the_statement_shows_is_not_encoded_and_a_shown_one_without_two_values_encodes_0(bank):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    for form, width in (("node", 2), ("pairs", 0)):
        cfg = configuration(held.registry, dict(view="snapshot"), STORED[form], 5, SPLITS)
        registry = held.fitted(cfg)
        states = read_json(held.viewed(cfg).folder / "encoders.json")["states"]
        assert (states["share"] is None) == (form == "pairs") and states["tie"] is None
        assert states["mass"]["sd"] > 0 and registry.attribute_width("gathering") == 2
        assert registry.attribute_width("gathering.lead") == width
    samples = dict.fromkeys(held.registry.schema.columns, [])
    samples.update(mass=[np.asarray([1.0, 2.0], np.float32)], share=None, tie=None)
    assert fitted(held.registry.encoding, samples).width(("share", "tie")) == 0
    shown = fitted(held.registry.encoding, dict(samples, share=[np.asarray([3.0], np.float32)]))
    assert shown.states["share"] == dict(mean=None, sd=None) and shown.width(("share", "tie")) == 2


def test_an_operation_that_counts_states_how_it_treats_sets_and_edges_about_edges(bank, monkeypatch):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), STORED["node"], 5, SPLITS)
    registry = held.fitted(cfg)
    (index,) = [entry["index"] for entry in held.viewed(cfg).graphs if entry["source"] == 0]
    dense = Graph.load(held.view(index, cfg)).dense()
    values = {"nodes/point/filter": "degree", "nodes/point/min_degree": 2, **STORED["node"]}
    steps = support.steps(registry, values)[0]
    derived = segments.whole(dense, {}, registry, steps, support.SEED, GraphLimits("cpu"), None)
    points = registry.kind("point")
    before = int((np.asarray(dense.kinds)[dense.obs_entity] == points).sum())
    after = int((np.asarray(derived.kinds)[derived.obs_entity] == points).sum())
    assert 0 < after < before
    counts = {count.name for step in steps for count in step.operation.requires}
    assert counts >= {"nodes/degree", "nodes/core", "derive/hops", "derive/diffusion", "edges/cap"}
    (degree,) = [count for count in NODES.requires if count.name == "nodes/degree"]
    with pytest.raises(
        ValueError,
        match="the count nodes/degree counts degree and states no treatment of the edges joining sets; "
        "it states one of stored, refuses",
    ):
        Count("nodes/degree", "degree", degree.active, None, "stored")
    refusing = Count("nodes/degree", "degree", degree.active, "refuses", "stored")
    monkeypatch.setattr(NODES, "requires", (refusing,))
    with pytest.raises(
        ValueError,
        match="the count nodes/degree of the operation graphspec.nodes counts degree and refuses the edges "
        "joining sets of the kind gathering",
    ):
        segments.whole(dense, {}, registry, steps, support.SEED, GraphLimits("cpu"), None)
    monkeypatch.setattr(NODES, "requires", (Count("nodes/degree", "degree", degree.active, "stored", "refuses"),))
    with pytest.raises(
        ValueError,
        match="nodes/degree of the operation graphspec.nodes counts degree and refuses the edges of the table "
        "note, which are about edges",
    ):
        segments.whole(dense, {}, registry, steps, support.SEED, GraphLimits("cpu"), None)
    monkeypatch.setattr(NODES, "requires", (Count("nodes/degree", "degree", peeking, "stored", "stored"),))
    with pytest.raises(
        ValueError,
        match="the code test_sets.peeking reads the rule of negative sampling hops during a derivation whose registry",
    ):
        segments.whole(dense, {}, registry, steps, support.SEED, GraphLimits("cpu"), None)


@timed("sets.peeking")
def peeking(registry, settings):
    NEGATIVES.named("hops")
    return True


@timed("sets.members_of")
def members_of(graph, registry, kind, codes, step):
    found = {}
    chosen = (graph.fact_step == step) & graph.fact_present & np.isin(graph.fact_code, list(codes))
    for source, target, code in zip(graph.fact_source[chosen].tolist(), graph.fact_target[chosen].tolist(), graph.fact_code[chosen].tolist()):
        found.setdefault(source, set()).add((codes[code], target))
    return found


@timed("sets.drawn_members")
def drawn_members(drawn, codes, step):
    found = {}
    chosen = (drawn["step"] == step) & np.isin(drawn["code"], list(codes))
    for source, target, code in zip(drawn["source"][chosen].tolist(), drawn["target"][chosen].tolist(), drawn["code"][chosen].tolist()):
        found.setdefault(source, set()).add((codes[code], target))
    return found


@timed("sets.entered")
def entered(rng, parameter):
    return True if parameter.name.count("/") == 1 else parameter.uniform(rng)


def test_negative_sets_replace_one_member_by_a_node_of_its_kind_and_are_never_a_present_set(bank):
    started = time.perf_counter()
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), STORED["node"], 5, SPLITS)
    registry = held.fitted(cfg)
    (index,) = [entry["index"] for entry in held.viewed(cfg).graphs if entry["source"] == 0]
    dense = Graph.load(held.view(index, cfg)).dense()
    stated = {"negatives/of": "set:gathering", "negatives/rule": "uniform", "negatives/count": 3}
    steps = pipeline.resolved(registry, [dict(operation="graphspec.negatives", parameters=stated)])
    grown = pipeline.grown(registry, steps, None)
    lead = grown.by_candidate["negatives.lead"]
    assert grown.nodes[-1] == "negatives" and lead.pairs == (("negatives", "point"), ("negatives", "place"))
    assert "negatives.lead" not in grown.by_name and support.unoffered(registry, grown) == []
    derived, candidates = segments.apart(dense, {}, grown, steps, support.SEED, GraphLimits("cpu"), None)
    roles = ("lead", "other")
    stored = {grown.code("gathering." + role): role for role in roles}
    made = {grown.by_candidate["negatives." + role].code: role for role in roles}
    negative, kinds, drawn = grown.kind("negatives"), np.asarray(derived.kinds), 0
    for step in range(dense.steps):
        present = members_of(derived, grown, "gathering", stored, step)
        found = drawn_members(candidates, made, step)
        shown = derived.obs_entity[(derived.obs_step == step) & ~derived.obs_anchor].tolist()
        assert all(kinds[node] == negative for node in found) and all(kinds[node] != negative for node in shown)
        tokens = np.asarray(derived.entity_token)
        owners = {}
        for node in found:
            (owner,) = [
                owner
                for owner, members in present.items()
                for at in range(3)
                if int(tokens[node]) == int(hashed(np.asarray([token("negatives")], np.uint64), tokens[[owner]], np.asarray([at]))[0])
            ]
            owners.setdefault(owner, []).append(node)
            gone, came = present[owner] - found[node], found[node] - present[owner]
            assert len(gone) == len(came) == 1 and next(iter(gone))[0] == next(iter(came))[0]
            assert kinds[next(iter(gone))[1]] == kinds[next(iter(came))[1]]
            assert next(iter(came))[1] not in {member for _, member in present[owner]}
            assert found[node] not in present.values()
        for owner, members in present.items():
            held_nodes = {member for _, member in members}
            allowed = sum(
                1
                for role, member in members
                for node in shown
                if kinds[node] == kinds[member]
                and node not in held_nodes
                and (members - {(role, member)}) | {(role, node)} not in present.values()
            )
            assert len(owners.get(owner, [])) == min(3, allowed), (step, owner)
            assert len({frozenset(found[node]) for node in owners.get(owner, [])}) == len(owners.get(owner, []))
        drawn += len(found)
    assert drawn > 0
    paired = configuration(held.registry, dict(view="snapshot"), STORED["pairs"], 5, SPLITS)
    (index,) = [entry["index"] for entry in held.viewed(paired).graphs if entry["source"] == 0]
    flat = Graph.load(held.view(index, paired)).dense()
    with pytest.raises(ValueError, match="draws negative sets of gathering among the sets stored as nodes"):
        segments.whole(flat, {}, grown, steps, support.SEED, GraphLimits("cpu"), None)
    refused = dict(paired, pipeline=[dict(operation="graphspec.negatives", parameters=stated)])
    message = (
        "the entry graphspec.negatives of the pipeline states negatives/of set:gathering, which needs the set edges "
        "of gathering stored as nodes, and sets/gathering/storage states pairs, which makes none"
    )
    with pytest.raises(Refused, match=re.escape(message)):
        held.stated(refused)
    template, rng = [dict(operation="graphspec.negatives")], np.random.default_rng(5)
    space, offered = pipeline.space(held.registry, template), {}
    for form in FORMS:
        graph = configuration(held.registry, dict(view="snapshot"), STORED[form], 5, SPLITS)["graph"]
        for _ in range(12):
            flat = pipeline.draw(held.registry, template, space, partial(entered, rng), graph)
            offered.setdefault(form, set()).add(flat[OFFERED].split(":")[0] + ":" + flat[OFFERED].split(":")[1])
    assert "set:gathering" in offered["node"] and "set:gathering" not in offered["pairs"] and len(offered["pairs"]) > 1
    message = (
        "the bounds hold the entry graphspec.negatives of the template present with negatives/of among set:gathering, "
        "each of which needs set edges stored as nodes, and leave sets/gathering/storage the form pairs, which makes none"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        check_bounds(held.registry, {"pipeline/graphspec.negatives": True, OFFERED: ["set:gathering"]}, template)
    check_bounds(held.registry, {"pipeline/graphspec.negatives": True, OFFERED: ["set:gathering"], **STORED["node"]}, template)
    LOG.info("negative sets exit drawn=%d seconds=%.3f", drawn, time.perf_counter() - started)


@pytest.fixture
def packed(monkeypatch):
    import torch

    started = time.perf_counter()
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
    LOG.debug("packed exit seconds=%.4f", time.perf_counter() - started)


@pytest.mark.parametrize("form", sorted(FORMS))
def test_each_form_of_storage_completes_one_fit_on_cpu(bank, packed, tmp_path, statement, form):
    held = Deriver(bank, "cpu", support.SEED, RESIDENT)
    cfg = dict(
        configuration(held.registry, dict(view="snapshot"), {"time/history": 4, **STORED[form]}, 5, SPLITS),
        epochs=1,
    )
    blocks = held.blocks(0, 4, cfg)
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], state(statement))
    LOG.info("fit of sets exit form=%s seconds=%.3f", form, time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(held.members(cfg, 0))
    assert np.isfinite(saved["measures"]["mse"]["value"]) and saved["holdout_rows"] == len(held.members(cfg, 1))
    assert saved["config"]["graph"]["sets/gathering/storage"] == FORMS[form]["gathering"]
