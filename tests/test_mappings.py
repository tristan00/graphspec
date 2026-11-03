import json

import pytest

from graphspec import (
    MAPPINGS,
    PRESENCE,
    Builder,
    BuildPlan,
    decisions,
    edges,
    graphs,
    mapping,
    nested,
    nodes,
    sets,
    source,
    tasks,
)
from graphspec.schema.schema import VALUE, VERSION

pytestmark = pytest.mark.builder
PLAN = BuildPlan(16 << 20, 1)
DAY = dict(type="time", unit="day")
UTILITY = dict(name="weighted", better="higher", weights=1.0)


def table(folder, name, rows, columns):
    (folder / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
    return source(name + ".json", "json", columns)


def schema(nodes=None, edges=None, sets=None, columns=None, kinds=None, references=None, model="none", units=(),
           ties=None, graph=()):
    return dict(version=VERSION, time=dict(model=model, units=list(units), ties=ties), observers=False,
                columns=columns or {}, graph=dict(columns=list(graph)),
                nodes={kind: dict(columns=list(held), holds=None) for kind, held in (nodes or {}).items()},
                sets=sets or {}, edges=edges or {},
                tasks=dict(key="key", grouping=None, signals=0, references=references or {}, kinds=kinds or {}))


def edge(pairs, directed=True, columns=(), relation="declared", about=None):
    return dict(endpoints=[list(pair) for pair in pairs], directed=directed, columns=list(columns), relation=relation,
                about=about)


def kind(references, target=None, columns=()):
    return dict(references=list(references), columns=list(columns), target=target, weight=False, absent="stop",
                event=None, matures=None)


def weighted(*series):
    return dict(UTILITY, weights={name: 1.0 for name in series})


def number(series):
    return dict(kind="number", series=series, about=None, utility=weighted(series))


def mapped(folder, declarations, name="one"):
    return Builder(declarations, folder).map(PLAN, name).records()


def called(folder, stated, graphs=("one",), unit=None):
    builder = Builder({}, folder).schema(stated)
    for name in graphs:
        builder.graph(name, unit)
    return builder


def looking(rows, of, key="id", **fields):
    return tasks(rows, references=dict(at=dict(sort="node", of=of, key=key, absent="stop")), **fields)


def asked(kinds):
    return dict(kinds=dict(ask=kind(["at"])), references=dict(at=dict(sort="node", kinds=list(kinds))))


def test_rows_to_nodes_edges_and_tasks_by_a_single_key_equal_the_same_graph_stated_by_calls(tmp_path):
    people = table(tmp_path, "people", [dict(id="a", age=3, born=1, grade="x"), dict(id="b", age=5, born=2, grade="y")],
                   dict(id="identifier", age="number", born=DAY, grade="category"))
    knows = table(tmp_path, "knows", [dict(source="a", target="b", day=4)],
                  dict(source="identifier", target="identifier", day=DAY))
    person = nodes(people, key="id", features=dict(age="standard", grade="onehot"),
                   holds=dict(name="state", since="born", until=None), known=None)
    knew = edges(knows, source=dict(of=person, key="source", absent="stop"),
                 target=dict(of=person, key="target", absent="stop"), directed=True,
                 holds=dict(name="event", at="day"), known=None)
    ask = tasks(people, at="born", references=dict(who=dict(sort="node", of=person, key="id", absent="stop")),
                target=dict(kind="number", series="age", value="age", known="born + 1", about=None, utility=UTILITY),
                split="'train' if grade == 'x' else 'test'")
    found = mapped(tmp_path, dict(people=people, knows=knows, person=person, knew=knew, ask=ask))
    columns = dict(age=dict(type="number", encoder="standard"), grade=dict(type="category", encoder="onehot"))
    builder = called(tmp_path, schema(dict(person=["age", "grade"]), dict(knew=edge([("person", "person")])),
                                      columns=columns, kinds=dict(ask=kind(["who"], number("age"))),
                                      references=dict(who=dict(sort="node", kinds=["person"])), model="clock",
                                      units=["day"]), unit="day")
    builder.node("person", "a", 1, None, known=None, age=3.0, grade="x")
    builder.node("person", "b", 2, None, known=None, age=5.0, grade="y")
    builder.event("knew", ("person", "a"), ("person", "b"), 4, known=None)
    builder.task("ask", 0, 1, dict(who="a"), "train", 3.0, 2)
    builder.task("ask", 1, 2, dict(who="b"), "test", 5.0, 3)
    assert found == builder.records()


def test_rows_to_tasks_carry_observers_groupings_weights_signals_and_features(tmp_path):
    people = table(tmp_path, "people", [dict(id="a", age=3, born=1, viewer="v", lot=3, w=2, s=1, t=None)],
                   dict(id="identifier", age="number", born=DAY, viewer="identifier", lot="number", w="number",
                        s="number", t=dict(type="number", missing="omit")))
    person = nodes(people, key="id", holds=dict(name="state", since="born", until=None), known=None, seen="viewer")
    ask = looking(people, person, at="born", observer="viewer", grouping="lot", weight="w",
                  signals="vector(s, coalesce(t, 0))", features=dict(age="standard"))
    found = mapped(tmp_path, dict(people=people, person=person, ask=ask))
    stated = schema(dict(person=[]), columns=dict(age=dict(type="number", encoder="standard")), model="clock",
                    units=["day"], kinds=dict(ask=dict(kind(["at"], columns=["age"]), weight=True)),
                    references=dict(at=dict(sort="node", kinds=["person"])))
    stated["observers"] = True
    stated["tasks"].update(grouping="grouping", signals=2)
    builder = called(tmp_path, stated, unit="day")
    builder.node("person", "a", 1, None, known=None, seen=["v"])
    builder.task("ask", 0, 1, dict(at="a"), None, None, None, weight=2.0, observer="v", grouping=3,
                 signal=[1.0, 0.0], signal_observed=[True, True], age=3.0)
    assert found == builder.records()


def test_a_composite_key_is_its_parts_and_two_keys_the_bank_cannot_tell_apart_stop(tmp_path):
    rows = [dict(firm="f", quarter="1", sales=2, closes=1, reported=2, revised=None),
            dict(firm="f", quarter="1", sales=3, closes=1, reported=4, revised=None),
            dict(firm="f", quarter="2", sales=4, closes=3, reported=5, revised=7)]
    versions = table(tmp_path, "versions", rows, dict(
        firm="identifier", quarter="identifier", sales="number", closes=DAY, reported=DAY,
        revised=dict(type="time", unit="day", missing="omit")))
    figure = nodes(versions, key=["firm", "quarter"], features=dict(sales="value"),
                   holds=dict(name="state", since="closes", until=None), known="reported", replaced="revised",
                   ties=dict(name="sequence", seq="row()"))
    ask = looking(versions, figure, ["firm", "quarter"], at="closes")
    found = mapped(tmp_path, dict(versions=versions, figure=figure, ask=ask))
    builder = called(tmp_path, schema(dict(figure=["sales"]), columns=dict(sales=dict(type="number", encoder="value")),
                                      model="clocks", units=["day"], ties="sequence", **asked(["figure"])), unit="day")
    for seq, (key, sales, since, known, replaced) in enumerate([("f|1", 2.0, 1, 2, None), ("f|1", 3.0, 1, 4, None),
                                                                ("f|2", 4.0, 3, 5, 7)]):
        builder.node("figure", key, since, None, known=known, replaced=replaced, seq=seq, sales=sales)
    for key, (name, since) in enumerate([("f|1", 1), ("f|1", 1), ("f|2", 3)]):
        builder.task("ask", key, since, dict(at=name), None, None, None)
    assert found == builder.records()
    alike = table(tmp_path, "alike", [dict(a="x|y", b="z"), dict(a="x", b="y|z")], dict(a="identifier", b="identifier"))
    thing = nodes(alike, key=["a", "b"])
    with pytest.raises(ValueError, match="cannot tell apart"):
        mapped(tmp_path, dict(alike=alike, thing=thing, ask=looking(alike, thing, ["a", "b"])))


def test_keys_are_unique_within_their_kind_and_their_graph(tmp_path):
    rows = [dict(id="1", molecule="m", role="atom"), dict(id="1", molecule="n", role="atom"),
            dict(id="2", molecule="m", role="ion")]
    parts = table(tmp_path, "parts", rows, dict(id="identifier", molecule="identifier", role="category"))
    labs = table(tmp_path, "labs", [dict(id="1")], dict(id="identifier"))
    part = nodes(parts, key="id", graph="molecule", kind="role")
    lab = nodes(labs, key="id")
    found = mapped(tmp_path, dict(parts=parts, labs=labs, part=part, lab=lab, ask=looking(parts, part,
                                                                                          graph="molecule")))
    builder = called(tmp_path, schema(dict(atom=[], ion=[], lab=[]), **asked(["atom", "ion"])), ("m", "n"))
    builder.node("atom", "1", graph="m").node("atom", "1", graph="n").node("ion", "2", graph="m")
    builder.node("lab", "1", graph="m").node("lab", "1", graph="n")
    for graph, key, name in (("m", 0, "1"), ("n", 0, "1"), ("m", 1, "2")):
        builder.task("ask", key, references=dict(at=name), split=None, target=None, known=None, graph=graph)
    assert found == builder.records()
    clash = table(tmp_path, "clash", [dict(id="1", role="atom"), dict(id="1", role="ion")],
                  dict(id="identifier", role="category"))
    both = nodes(clash, key="id", kind="role")
    with pytest.raises(ValueError, match="unique across them"):
        mapped(tmp_path, dict(clash=clash, both=both, ask=looking(clash, both)))


def test_rows_to_edges_take_relations_from_a_column_and_tie_by_a_stated_sequence(tmp_path):
    readings = table(tmp_path, "readings", [dict(id="s", day=1, load=2), dict(id="s", day=3, load=4)],
                     dict(id="identifier", day=DAY, load="number"))
    triples = table(tmp_path, "triples", [dict(head="s", tail="s", rel="near", day=2),
                                          dict(head="s", tail="s", rel="far", day=2)],
                    dict(head="identifier", tail="identifier", rel="category", day=DAY))
    sensor = nodes(readings, key="id", features=dict(load="value"), holds=dict(name="change", at="day"), known=None)
    fact = edges(triples, source=dict(of=sensor, key="head", absent="stop"),
                 target=dict(of=sensor, key="tail", absent="stop"), relation="rel", directed=False,
                 holds=dict(name="event", at="day"), known=None, ties=dict(name="sequence", seq="row()"))
    found = mapped(tmp_path, dict(readings=readings, triples=triples, sensor=sensor, fact=fact,
                                  ask=looking(readings, sensor, at="day")))
    builder = called(tmp_path, schema(dict(sensor=["load"]), dict(fact=edge([("sensor", "sensor")], False,
                                                                             relation=dict(column="relation"))),
                                      columns=dict(load=dict(type="number", encoder="value")), model="clock",
                                      units=["day"], ties="sequence", **asked(["sensor"])), unit="day")
    builder.change("sensor", "s", 1, known=None, seq=0, load=2.0).change("sensor", "s", 3, known=None, seq=0, load=4.0)
    builder.event(("fact", "near"), ("sensor", "s"), ("sensor", "s"), 2, known=None, seq=0)
    builder.event(("fact", "far"), ("sensor", "s"), ("sensor", "s"), 2, known=None, seq=1)
    builder.task("ask", 0, 1, dict(at="s"), None, None, None).task("ask", 1, 3, dict(at="s"), None, None, None)
    assert found == builder.records()
    ruled = nodes(readings, key="id", holds=dict(name="change", at="day"), known=None,
                  ties=dict(name="sequence", seq="row()"))
    more = nodes(readings, of=ruled, key="id", features=dict(load="value"), holds=dict(name="change", at="day"),
                 known=None)
    with pytest.raises(ValueError, match="more states no ties"):
        mapped(tmp_path, dict(readings=readings, ruled=ruled, more=more, ask=looking(readings, ruled, at="day")))


def test_the_key_of_an_edge_holds_the_edges_an_end_names_as_set_edges(tmp_path):
    rows = [dict(id="t1", head="a", tail="b", rel="r", about=None),
            dict(id="t2", head="b", tail="a", rel="r", about="t1")]
    stated = table(tmp_path, "stated", rows, dict(id="identifier", head="identifier", tail="identifier",
                                                  rel="category", about=dict(type="identifier", missing="omit")))
    entity = nodes(stated, distinct=["head", "tail"])
    fact = edges(stated, where="about is missing", key="id", source=dict(of=entity, key="head", absent="stop"),
                 target=dict(of=entity, key="tail", absent="stop"), relation="rel", directed=True)
    qualifier = edges(stated, where="about is not missing", source=dict(of=fact, key="about", absent="stop"),
                      target=dict(of=entity, key="tail", absent="stop"), relation="rel", directed=True, about="lapse")
    found = mapped(tmp_path, dict(stated=stated, entity=entity, fact=fact, qualifier=qualifier,
                                  ask=looking(stated, entity, "head")))
    column = dict(column="relation")
    builder = called(tmp_path, schema(
        dict(entity=[]), dict(fact=edge([("entity", "entity")], relation=column),
                              qualifier=edge([("fact.edge", "entity")], relation=column, about="lapse")),
        sets={"fact.edge": dict(kinds=["entity"], roles=["source", "target"], members=[], columns=[])},
        **asked(["entity"])))
    builder.node("entity", "a").node("entity", "b").node("fact.edge", "t1")
    builder.member("fact.edge", "t1", ("entity", "a"), "source").member("fact.edge", "t1", ("entity", "b"), "target")
    builder.edge(("fact", "r"), ("entity", "a"), ("entity", "b"))
    builder.edge(("qualifier", "r"), ("fact.edge", "t1"), ("entity", "a"))
    builder.task("ask", 0, references=dict(at="a"), split=None, target=None, known=None)
    builder.task("ask", 1, references=dict(at="b"), split=None, target=None, known=None)
    assert found == builder.records()


def test_keys_made_from_the_position_in_a_nested_record(tmp_path):
    rows = [dict(program="p", root=dict(form="a", children=[dict(form="b", children=[]),
                                                             dict(form="c", children=[])]))]
    programs = table(tmp_path, "programs", rows, dict(program="identifier", root="nested"))
    spread = nested(programs, record="root", children="children", columns=dict(form="category"), depth=None)
    part = nodes(spread, key="path", graph="program", features=dict(form="onehot"))
    child = edges(spread, where="parent is not missing", graph="program",
                  source=dict(of=part, key="parent", absent="stop"), target=dict(of=part, key="path", absent="stop"),
                  directed=True, features=dict(place="value"))
    ask = tasks(programs, graph="program", references=dict(whole=dict(sort="graph")))
    found = mapped(tmp_path, dict(programs=programs, spread=spread, part=part, child=child, ask=ask))
    builder = called(tmp_path, schema(
        dict(part=["form"]), dict(child=edge([("part", "part")], columns=["place"])),
        columns=dict(form=dict(type="category", encoder="onehot"), place=dict(type="number", encoder="value")),
        kinds=dict(ask=kind(["whole"])), references=dict(whole=dict(sort="graph", kinds=[]))), ("p",))
    for key, form in (("0", "a"), ("0.0", "b"), ("0.1", "c")):
        builder.node("part", key, graph="p", form=form)
    builder.edge("child", ("part", "0"), ("part", "0.0"), graph="p", place=0.0)
    builder.edge("child", ("part", "0"), ("part", "0.1"), graph="p", place=1.0)
    builder.task("ask", 0, references=dict(whole=None), split=None, target=None, known=None, graph="p")
    assert found == builder.records()


def test_keys_from_the_row_number_of_a_file_and_edges_in_every_graph_holding_both_ends(tmp_path):
    atoms = table(tmp_path, "atoms", [dict(molecule="m", element="C"), dict(molecule="m", element="O"),
                                      dict(molecule="n", element="N")], dict(molecule="identifier", element="category"))
    bonds = table(tmp_path, "bonds", [dict(one=1, other=0)], dict(one="number", other="number"))
    atom = nodes(atoms, key="row()", graph="molecule", features=dict(element="onehot"))
    bond = edges(bonds, source=dict(of=atom, key="one", absent="stop"),
                 target=dict(of=atom, key="other", absent="stop"), directed=False)
    found = mapped(tmp_path, dict(atoms=atoms, bonds=bonds, atom=atom, bond=bond,
                                  ask=looking(atoms, atom, "row()", graph="molecule")))
    builder = called(tmp_path, schema(dict(atom=["element"]), dict(bond=edge([("atom", "atom")], False)),
                                      columns=dict(element=dict(type="category", encoder="onehot")),
                                      **asked(["atom"])), ("m", "n"))
    builder.node("atom", "0", graph="m", element="C").node("atom", "1", graph="m", element="O")
    builder.node("atom", "2", graph="n", element="N")
    builder.edge("bond", ("atom", "1"), ("atom", "0"), graph="m")
    for graph, key, name in (("m", 0, "0"), ("m", 1, "1"), ("n", 0, "2")):
        builder.task("ask", key, references=dict(at=name), split=None, target=None, known=None, graph=graph)
    assert found == builder.records()


def test_nodes_from_the_distinct_values_of_a_column_and_from_links(tmp_path):
    orders = table(tmp_path, "orders", [dict(number="o1", customer="c1", placed=1),
                                        dict(number="o2", customer="c1", placed=2),
                                        dict(number="o3", customer=None, placed=3)],
                   dict(number="identifier", customer=dict(type="identifier", missing="omit"), placed=DAY))
    buyer = nodes(orders, distinct=["customer"], holds="always", known=None)
    order = nodes(orders, key="number", links=dict(customer="customer",
                                                   by=dict(of=buyer, key="customer", absent="stop")),
                  holds=dict(name="state", since="placed", until=None), known=None)
    shipment = edges(orders, where="number == 'o1'", source=dict(of=order, key="number", absent="stop"),
                     target=dict(of=order.customer, key="customer", absent="stop"), directed=True,
                     holds=dict(name="event", at="placed"), known=None)
    found = mapped(tmp_path, dict(orders=orders, buyer=buyer, order=order, shipment=shipment,
                                  ask=looking(orders, order, "number", at="placed")))
    builder = called(tmp_path, schema(
        dict(buyer=[], order=[], customer=[]),
        {"order.customer": edge([("order", "customer")]), "order.by": edge([("order", "buyer")]),
         "shipment": edge([("order", "customer")])}, model="clock", units=["day"], **asked(["order"])), unit="day")
    builder.node("buyer", "c1", None, None, known=None)
    for since, key in enumerate(("o1", "o2", "o3"), 1):
        builder.node("order", key, since, None, known=None)
    builder.node("customer", "c1", None, None, known=None)
    for relation, other in (("order.customer", "customer"), ("order.by", "buyer")):
        for since, key in ((1, "o1"), (2, "o2")):
            builder.edge(relation, ("order", key), (other, "c1"), since, None, known=None)
    builder.event("shipment", ("order", "o1"), ("customer", "c1"), 1, known=None)
    for key, name in enumerate(("o1", "o2", "o3")):
        builder.task("ask", key, key + 1, dict(at=name), None, None, None)
    assert found == builder.records()


def requiring(tmp_path, rule):
    topics = table(tmp_path, "topics", [dict(exercise="e1", topic="t1")],
                   dict(exercise="identifier", topic="identifier"))
    prerequisites = table(tmp_path, "prerequisites", [dict(topic="t1", needed="t2"), dict(topic="t3", needed="t1")],
                          dict(topic="identifier", needed="identifier"))
    exercise = nodes(topics, key="exercise", links=dict(topic="topic"))
    requires = edges(prerequisites, source=dict(of=exercise.topic, key="topic", absent=rule),
                     target=dict(of=exercise.topic, key="needed", absent="create"), directed=True)
    return Builder(dict(topics=topics, prerequisites=prerequisites, exercise=exercise, requires=requires,
                        ask=looking(topics, exercise, "exercise")), tmp_path).map(PLAN, "one")


def test_nodes_are_created_on_first_reference_only_where_the_end_states_it(tmp_path):
    made = requiring(tmp_path, "create")
    builder = called(tmp_path, schema(dict(exercise=[], topic=[]), {"exercise.topic": edge([("exercise", "topic")]),
                                                                    "requires": edge([("topic", "topic")])},
                                      **asked(["exercise"])))
    builder.node("exercise", "e1").node("topic", "t1").node("topic", "t2").node("topic", "t3")
    builder.edge("exercise.topic", ("exercise", "e1"), ("topic", "t1"))
    builder.edge("requires", ("topic", "t1"), ("topic", "t2")).edge("requires", ("topic", "t3"), ("topic", "t1"))
    builder.task("ask", 0, references=dict(at="e1"), split=None, target=None, known=None)
    assert made.records() == builder.records()
    assert made.counts == dict(requires=dict(source=dict(create=1), target=dict(create=1)),
                               ask={"references.at": dict(stop=0)})
    skipped = requiring(tmp_path, "skip")
    assert skipped.counts["requires"]["source"] == dict(skip=1)
    assert len(skipped.records()["graphs"][0]["edges"]) == 2
    with pytest.raises(ValueError, match="requires field source row 1 key t3: the rule stop"):
        requiring(tmp_path, "stop")
    topics = table(tmp_path, "topics", [dict(exercise="e1", topic="t1")],
                   dict(exercise="identifier", topic="identifier"))
    alone = nodes(topics, key="exercise")
    with pytest.raises(ValueError, match="rule create, which makes a node only for an end of edges"):
        mapped(tmp_path, dict(topics=topics, alone=alone, ask=tasks(topics, references=dict(
            at=dict(sort="node", of=alone, key="topic", absent="create")))))


def test_rows_to_edges_joining_sets_with_their_members_and_roles(tmp_path):
    works = table(tmp_path, "works", [dict(id="p1", day=1), dict(id="p2", day=2)], dict(id="identifier", day=DAY))
    bylines = table(tmp_path, "bylines", [dict(id="p1", author="x", lead=True), dict(id="p1", author="y", lead=False),
                                          dict(id="p2", author="x", lead=True)],
                    dict(id="identifier", author="identifier", lead="flag"))
    author = nodes(bylines, distinct=["author"], holds="always", known=None)
    paper = sets(works, key="id", holds=dict(name="event", at="day"), known=None,
                 members=[dict(rows=bylines, set="id", node=dict(of=author, key="author", absent="stop"),
                               role="'author' if lead else 'helper'", features=dict(lead="value"))])
    declared = dict(works=works, bylines=bylines, author=author, paper=paper, ask=looking(works, paper, at="day"))
    made = Builder(declared, tmp_path).map(PLAN, "one")
    builder = called(tmp_path, schema(
        dict(author=[]), sets=dict(paper=dict(kinds=["author"], roles=["author", "helper"], members=["lead"],
                                              columns=[])),
        columns=dict(lead=dict(type="flag", encoder="value")), model="clock", units=["day"], **asked(["paper"])),
        unit="day")
    builder.node("author", "x", None, None, known=None).node("author", "y", None, None, known=None)
    builder.node("paper", "p1", 1, 2, known=None).node("paper", "p2", 2, 3, known=None)
    builder.member("paper", "p1", ("author", "x"), "author", lead=True)
    builder.member("paper", "p1", ("author", "y"), "helper", lead=False)
    builder.member("paper", "p2", ("author", "x"), "author", lead=True)
    builder.task("ask", 0, 1, dict(at="p1"), None, None, None).task("ask", 1, 2, dict(at="p2"), None, None, None)
    assert made.records() == builder.records()
    with pytest.raises(ValueError, match="names the node x twice in the role author"):
        builder.member("paper", "p1", ("author", "x"), "author", lead=False)
    builder.member("paper", "p1", ("author", "x"), "helper", lead=False)
    made.stage(tmp_path / "staged", "one", PLAN)
    assert (tmp_path / "staged" / "one.task.json").exists()


def test_rows_to_graphs_state_their_attributes_and_a_graph_no_row_states_stops(tmp_path):
    labels = table(tmp_path, "labels", [dict(molecule="m", solvent="w"), dict(molecule="n", solvent="o")],
                   dict(molecule="identifier", solvent="category"))
    atoms = table(tmp_path, "atoms", [dict(molecule="m", id="a"), dict(molecule="n", id="a")],
                  dict(molecule="identifier", id="identifier"))
    molecule = graphs(labels, key="molecule", features=dict(solvent="onehot"))
    atom = nodes(atoms, key="id", graph="molecule")
    ask = tasks(labels, graph="molecule", references=dict(whole=dict(sort="graph")))
    found = mapped(tmp_path, dict(labels=labels, atoms=atoms, molecule=molecule, atom=atom, ask=ask))
    builder = Builder({}, tmp_path).schema(schema(
        dict(atom=[]), columns=dict(solvent=dict(type="category", encoder="onehot")), graph=["solvent"],
        kinds=dict(ask=kind(["whole"])), references=dict(whole=dict(sort="graph", kinds=[]))))
    builder.graph("m", None, solvent="w").graph("n", None, solvent="o")
    builder.node("atom", "a", graph="m").node("atom", "a", graph="n")
    for graph in ("m", "n"):
        builder.task("ask", 0, references=dict(whole=None), split=None, target=None, known=None, graph=graph)
    assert found == builder.records()
    stray = table(tmp_path, "stray", [dict(molecule="z", id="b")], dict(molecule="identifier", id="identifier"))
    with pytest.raises(ValueError, match="names the graph z, which no row of graphs states"):
        mapped(tmp_path, dict(labels=labels, stray=stray, molecule=molecule, atom=nodes(stray, key="id",
                                                                                        graph="molecule"), ask=ask))


def holding(tmp_path, bonded):
    atoms = table(tmp_path, "atoms", [dict(drug="d0", atom="a", element="C"), dict(drug="d0", atom="b", element="O"),
                                      dict(drug="d1", atom="a", element="N")],
                  dict(drug="identifier", atom="identifier", element="category"))
    bonds = table(tmp_path, "bonds", bonded, dict(drug="identifier", one="identifier", other="identifier", order="number"))
    drugs = table(tmp_path, "drugs", [dict(drug="d0", mass=1.0), dict(drug="d1", mass=2.0)],
                  dict(drug="identifier", mass="number"))
    drug = nodes(atoms, distinct=["drug"], held="propagated")
    inside = dict(of=drug, key="drug", absent="stop")
    atom = nodes(atoms, key="atom", inside=inside, features=dict(element="onehot"))
    bond = edges(bonds, inside=inside, source=dict(of=atom, key="one", absent="stop"),
                 target=dict(of=atom, key="other", absent="skip"), directed=False, features=dict(order="value"))
    ask = looking(drugs, drug, key="drug", split="'train'",
                  target=dict(kind="number", series="mass", value="mass", known="never", about=None, utility=UTILITY))
    return dict(atoms=atoms, bonds=bonds, drugs=drugs, drug=drug, atom=atom, bond=bond, ask=ask)


def test_rows_inside_a_holder_state_its_graph_as_the_calls_do_and_the_bank_holds_it(tmp_path):
    from graphspec.facts.stored import Facts
    from tuning import build_bank
    from tuning.bank.bank import Bank

    declared = holding(tmp_path, [dict(drug="d0", one="a", other="b", order=2), dict(drug="d1", one="a", other="z",
                                                                                     order=1)])
    found = mapped(tmp_path, declared)
    stated = schema(dict(drug=[]), columns=dict(element=dict(type="category", encoder="onehot"),
                                                order=dict(type="number", encoder="value")),
                    kinds=dict(ask=kind(["at"], number("mass"))), references=dict(at=dict(sort="node", kinds=["drug"])))
    stated["nodes"]["drug"]["holds"] = dict(nodes=dict(atom=dict(columns=["element"])), encoder="propagated", edges=dict(
        bond=dict(endpoints=[["atom", "atom"]], directed=False, columns=["order"])))
    builder = called(tmp_path, stated).node("drug", "d0").node("drug", "d1")
    for owner, key, element in (("d0", "a", "C"), ("d0", "b", "O"), ("d1", "a", "N")):
        builder.held("drug", owner, "atom", key, element=element)
    builder.held_edge("drug", "d0", "bond", "a", "b", order=2.0)
    for key, mass in enumerate((1.0, 2.0)):
        builder.task("ask", key, references=dict(at="d%d" % key), split="train", target=mass, known=None)
    assert found == builder.records()
    for call, message in ((lambda: builder.held("drug", "d0", "atom", "a", element="C"), "is stated twice"),
                          (lambda: builder.held_edge("drug", "d1", "bond", "a", "b", order=1.0), "no held node"),
                          (lambda: builder.held("drug", "d9", "atom", "c", element="C"), "which no fact of graph")):
        with pytest.raises(ValueError, match=message):
            call()
    build_bank(Builder(declared, tmp_path), tmp_path / "bank", 1, ("train",), plan=PLAN)
    bank = Bank(tmp_path / "bank")
    assert Facts.load(bank.truth(0)).held(bank.registry) == dict(drug=dict(
        nodes={"owner": ["d0", "d0", "d1"], "kind": ["atom"] * 3, "key": ["a", "b", "a"], VALUE + "element": ["C", "O", "N"]},
        edges={"owner": ["d0"], "source": ["a"], "target": ["b"], "relation": ["bond"], VALUE + "order": [2.0]}))
    twice = holding(tmp_path, [])
    twice["atom"] = nodes(twice["atoms"], key="'a'", inside=twice["atom"].fields["inside"],
                          features=dict(element="onehot"))
    with pytest.raises(ValueError, match="declaration atom row 1 states the held node a inside drug d0 twice"):
        mapped(tmp_path, twice)
    with pytest.raises(ValueError, match="field inside names under of a nodes statement that states held"):
        mapped(tmp_path, dict(declared, atom=nodes(declared["atoms"], key="atom", inside=dict(
            of=declared["ask"], key="drug", absent="stop"))))


def test_rows_to_tasks_name_references_of_every_sort_and_a_class_target(tmp_path):
    links = table(tmp_path, "links", [dict(id="k1", source="a", target="b"), dict(id="k2", source="b", target="c")],
                  dict(id="identifier", source="identifier", target="identifier"))
    listed = dict(type="list", of="identifier")
    rows = [dict(who="a", group=["a", "b"], left="a", right="b", edge="k2", way=["k1", "k2"], route=["a", "b", "c"],
                 first=["k1"], label="x"),
            dict(who="c", group=["c"], left="b", right="c", edge="k1", way=["k2"], route=["b", "c"], first=[],
                 label="y")]
    asks = table(tmp_path, "asks", rows, dict(who="identifier", group=listed, left="identifier", right="identifier",
                                              edge="identifier", way=listed, route=listed, first=listed,
                                              label="category"))
    person = nodes(links, distinct=["source", "target"])
    link = edges(links, key="id", source=dict(of=person, key="source", absent="stop"),
                 target=dict(of=person, key="target", absent="stop"), directed=True)
    references = dict(
        one=dict(sort="node", of=person, key="who", absent="stop"),
        many=dict(sort="nodes", of=person, keys="group", absent="stop"),
        pair=dict(sort="edge", of=link, source="left", target="right", absent="stop"),
        keyed=dict(sort="edge", of=link, key="edge", absent="stop"),
        way=dict(sort="path", of=link, keys="way", absent="stop"),
        route=dict(sort="path", of=link, nodes="route", absent="stop"),
        sub=dict(sort="subgraph", of=person, keys="group", edges=dict(of=link, keys="first"), absent="stop"),
        whole=dict(sort="graph"))
    ask = tasks(asks, references=references, split="'train'",
                target=dict(kind="class", value="label", known="never", about=None, utility=UTILITY))
    found = mapped(tmp_path, dict(links=links, asks=asks, person=person, link=link, ask=ask))
    sorts = dict(one="node", many="nodes", pair="edge", keyed="edge", way="path", route="path", sub="subgraph")
    stated = {name: dict(sort=sort, kinds=["person"]) for name, sort in sorts.items()}
    stated["whole"] = dict(sort="graph", kinds=[])
    target = dict(kind="class", series=["x", "y"], about=None, utility=weighted("x", "y"))
    builder = called(tmp_path, schema(dict(person=[]), dict(link=edge([("person", "person")])),
                                      kinds=dict(ask=kind(list(references), target)), references=stated))
    for key in ("a", "b", "c"):
        builder.node("person", key)
    builder.edge("link", ("person", "a"), ("person", "b")).edge("link", ("person", "b"), ("person", "c"))
    ab, bc = ["a", "link", "b"], ["b", "link", "c"]
    builder.task("ask", 0, references=dict(one="a", many=["a", "b"], pair=ab, keyed=bc, way=[ab, bc], route=[ab, bc],
                                           sub=dict(nodes=["a", "b"], edges=[ab]), whole=None),
                 split="train", target="x", known=None)
    builder.task("ask", 1, references=dict(one="c", many=["c"], pair=bc, keyed=ab, way=[bc], route=[bc],
                                           sub=dict(nodes=["c"], edges=[]), whole=None),
                 split="train", target="y", known=None)
    assert found == builder.records()


def test_rows_to_decisions_state_their_options_actions_and_the_option_that_does_nothing(tmp_path):
    sessions = table(tmp_path, "sessions", [dict(student="s", step=1, shown=1, given="h1"),
                                            dict(student="s", step=2, shown=2, given="none")],
                     dict(student="identifier", step="number", shown=DAY, given="category"))
    offers = table(tmp_path, "offers", [dict(student="s", step=1, type="h1", words=3, given="h1", shown=1, right=True),
                                        dict(student="s", step=1, type="h2", words=5, given="h1", shown=1, right=True),
                                        dict(student="s", step=2, type="h1", words=4, given="none", shown=2,
                                             right=False)],
                   dict(student="identifier", step="number", type="category", words="number", given="category",
                        shown=DAY, right="flag"))
    student = nodes(sessions, distinct=["student"], holds="always", known=None)
    tutor = nodes(sessions, distinct=["'tutor'"], holds="always", known=None)
    hinting = decisions(
        sessions, key=["student", "step"], at="shown",
        references=dict(student=dict(sort="node", of=student, key="student", absent="stop"),
                        tutor=dict(sort="node", of=tutor, key="'tutor'", absent="stop")),
        actors="tutor", series=["correct"], about="student", utility=UTILITY, split="'train'", ended="step == 2",
        nothing=dict(key="'none'", chosen="given == 'none'", available="True",
                     outcome=dict(correct="0.0 if given == 'none' else missing"), known="shown + 1"),
        options=dict(rows=offers, decision=["student", "step"], key="type", chosen="type == given",
                     available="True", outcome=dict(correct="number(right) if type == given else missing"),
                     known="shown + 1"),
        actions=dict(rows=offers, decision=["student", "step"], option="type", kind="'hint'", actor="'tutor'",
                     kinds=dict(hint=dict(roles=dict(to=dict(sort="node", of=student, keys="student", count=[1, 1])),
                                          parameters=dict(words="standard")))))
    for generators, message in ((None, "states generators as a list of mappings"),
                                ([dict(seed=1)], "generators.0 lacks the required field name")):
        with pytest.raises(ValueError, match=message):
            decisions(sessions, **dict(hinting.fields, generators=generators))
    found = mapped(tmp_path, dict(sessions=sessions, offers=offers, student=student, tutor=tutor, hinting=hinting))
    actions = dict(hint=dict(roles=dict(to=dict(sort="node", kinds=["student"], count=[1, 1])), parameters=["words"]))
    target = dict(kind="decision", series=["correct"], about="student", utility=weighted("correct"), actors="tutor",
                  actions=actions, nothing="none", rule=None, generators=[])
    hinted = dict(kind(["student", "tutor"], target))
    builder = called(tmp_path, schema(
        dict(student=[], tutor=[]), columns=dict(words=dict(type="number", encoder="standard")),
        kinds=dict(hinting=hinted), references=dict(student=dict(sort="node", kinds=["student"]),
                                                    tutor=dict(sort="node", kinds=["tutor"])),
        model="clock", units=["day"]), unit="day")
    builder.node("student", "s", None, None, known=None).node("tutor", "tutor", None, None, known=None)

    def option(key, chosen, outcome, words):
        hints = [] if words is None else [dict(kind="hint", actor="tutor", targets=dict(to=["s"]),
                                               parameters=dict(words=words))]
        return dict(key=key, together="set", available=True, chosen=chosen, order=None, shown=None, logged=None,
                    origin="recorded", outcome=outcome, actions=hints)

    first = dict(options=[option("h1", True, dict(correct=1.0), 3.0), option("h2", False, None, 5.0),
                          option("none", False, None, None)], concurrent=[], next=dict(tutor=1), ended=False,
                 discount=None)
    second = dict(options=[option("h1", False, None, 4.0), option("none", True, dict(correct=0.0), None)],
                  concurrent=[], next={}, ended=True, discount=None)
    references = dict(student="s", tutor="tutor")
    builder.task("hinting", 0, 1, references, "train", first, 2).task("hinting", 1, 2, references, "train", second, 3)
    assert found == builder.records()


def test_a_registered_mapping_and_rule_of_presence_state_their_facts_by_the_calls(tmp_path):
    class Stand:
        version = 1
        creates = False
        keeps = False

        def __call__(self, keys, times, parameters):
            return [parameters["key"]] * len(keys)

    def stating(builder, rows, parameters):
        for key in rows.column("id").to_pylist():
            builder.node(parameters["kind"], key)

    stating.version = 1
    MAPPINGS.register("stating", stating)
    PRESENCE.register("stand", Stand())
    try:
        plain = table(tmp_path, "plain", [dict(id="a", other="z")], dict(id="identifier", other="identifier"))
        thing = nodes(plain, key="id")
        made = mapping(plain, name="stating", schema=dict(nodes=dict(extra=dict(columns=[]))), kind="extra")
        tied = edges(plain, source=dict(of=thing, key="id", absent="stop"),
                     target=dict(of=thing, key="other", absent=dict(name="stand", key="a")), directed=True)
        found = mapped(tmp_path, dict(plain=plain, thing=thing, made=made, tied=tied, ask=looking(plain, thing)))
    finally:
        del MAPPINGS.held["stating"]
        del PRESENCE.held["stand"]
    builder = called(tmp_path, schema(dict(thing=[], extra=[]), dict(tied=edge([("thing", "thing")])),
                                      **asked(["thing"])))
    builder.node("thing", "a").node("extra", "a").edge("tied", ("thing", "a"), ("thing", "a"))
    builder.task("ask", 0, references=dict(at="a"), split=None, target=None, known=None)
    assert found == builder.records()


def test_a_statement_or_a_build_stops_naming_the_declaration_and_the_field(tmp_path):
    rows = table(tmp_path, "rows", [dict(id="a", day=1, late=True), dict(id=None, day=2, late=False)],
                 dict(id=dict(type="identifier", missing="omit"), day=DAY, late=dict(type="flag", outcome="day + 1")))
    for word, fields, message in (
            (nodes, dict(key="id", colour="red"), "nodes holds fields it does not know: colour"),
            (nodes, dict(key="id", distinct=["id"]), "states one of key and distinct"),
            (edges, dict(source=dict(of=None, key="id", absent="stop"), directed=True), "lacks the required field target"),
            (tasks, dict(references={}, target=dict(kind="number", value="1", known="never", about=None,
                                                    utility=UTILITY)), "lacks the required field split"),
            (decisions, dict(key="id", references={}), "lacks the required field actors")):
        with pytest.raises(ValueError, match=message):
            word(rows, **fields)
    thing = nodes(rows, key="id", where="id == 'a'", holds="always", known=None)
    with pytest.raises(ValueError, match="ask field features.late reads the outcome column rows.late"):
        mapped(tmp_path, dict(rows=rows, thing=thing, ask=looking(rows, thing, at="day", where="id == 'a'",
                                                                  features=dict(late="value"))))
    for stated, message in ((dict(holds="always", known=None), "thing field key row 1 reads a missing value"),
                            (dict(), "ask states at, and no mapping|thing lacks holds"),
                            (dict(holds=dict(name="snapshot", at="day"), known=None),
                             "thing field holds.between: the form snapshot states the rule")):
        thing = nodes(rows, key="id", **stated)
        with pytest.raises(ValueError, match=message):
            mapped(tmp_path, dict(rows=rows, thing=thing, ask=looking(rows, thing, where="id == 'a'", at="day")))
