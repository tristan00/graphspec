import json
import re

import pytest

from graphspec import Builder, BuildPlan, Invalid, compute, edges, nodes, source, tasks, validate

pytestmark = pytest.mark.builder
PLAN = BuildPlan(16 << 20, 1)
DAY = dict(type="time", unit="day")
UTILITY = dict(name="weighted", better="higher", weights=1.0)
INTERNAL = re.compile(r"\.rows\d|\.input\d|column:|staging|_")


def table(folder, name, rows, columns):
    (folder / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
    return source(name + ".json", "json", columns)


def script(folder):
    people = table(folder, "people", [dict(id="a", day=1, late=True), dict(id="b", day=2, late=False)],
                   dict(id="identifier", day=DAY, late=dict(type="flag", outcome="day + 1")))
    links = table(folder, "links", [dict(left="a", right="b", day=3), dict(left="a", right="z", day=3)],
                  dict(left="identifier", right="identifier", day=DAY))
    loose = table(folder, "loose", [dict(id="q", note="x")], dict(id="identifier", note=None))
    person = nodes(people, key="id", holds=dict(name="state", since="day", until=None), known=None)
    knew = edges(links, source=dict(of=person, key="left", absent="stop"),
                 target=dict(of=person, key="right", absent="stop"), directed=True,
                 holds=dict(name="event", at="day"), known=None)
    other = nodes(loose, key="id", holds="always", known=None)
    ask = tasks(people, at="day", references=dict(who=dict(sort="node", of=person, key="id", absent="stop")),
                target=dict(kind="number", series="day", value="1", known="day + 1", about=None, utility=UTILITY),
                split="'train'", features=dict(late="value"))
    return Builder(dict(people=people, links=links, loose=loose, person=person, knew=knew, other=other, ask=ask),
                   folder)


def test_a_script_with_three_errors_of_different_kinds_reports_all_three_each_with_its_declaration(tmp_path):
    builder = script(tmp_path)
    with pytest.raises(Invalid) as raised:
        validate(builder, PLAN)
    found = {error.declaration: error for error in raised.value.errors}
    assert sorted(found) == ["ask", "knew", "loose"] and raised.value.unchecked == ("other",)
    assert found["loose"].places == ("loose.json",)
    assert "source loose (loose.json) column note: the column has no stated type" == found["loose"].message
    assert found["knew"].places == ("links.json",)
    assert found["knew"].message.startswith("declaration knew field target row 1 key z: the rule stop meets the key")
    assert found["ask"].places == ("people.json",)
    assert found["ask"].message.startswith("declaration ask field features.late reads the outcome column people.late")
    text = str(raised.value)
    assert text.startswith("the build holds 3 errors\n") and "\ndeclaration knew, reading links.json\n  " in text
    assert text.endswith("not checked, as they read a declaration with an error: other")
    assert not INTERNAL.search(text) and not builder.mapped
    with pytest.raises(Invalid, match="the build holds 3 errors") as staged:
        builder.stage(tmp_path / "staged", "one", PLAN)
    assert staged.value.errors == raised.value.errors and not (tmp_path / "staged").exists()


def test_rules_local_to_one_call_stop_at_the_call_and_a_valid_script_validates(tmp_path):
    rows = table(tmp_path, "rows", [dict(id="a", day=1)], dict(id="identifier", day=DAY))
    with pytest.raises(ValueError, match="the statement nodes holds fields it does not know: colour") as raised:
        nodes(rows, key="id", colour="red")
    assert not isinstance(raised.value, Invalid)
    thing = nodes(rows, key="id", holds="always", known=None)
    ask = tasks(rows, at="day", references=dict(who=dict(sort="node", of=thing, key="id", absent="stop")), split="'train'")
    builder = Builder(dict(rows=rows, thing=thing, ask=ask), tmp_path)
    assert validate(builder, PLAN) is None and not builder.mapped


def test_tables_without_a_name_are_named_by_their_declaration_and_their_field(tmp_path):
    (tmp_path / "people.json").write_text(json.dumps([dict(id="a", day=1, note="x")]), encoding="utf-8")
    loose = dict(id="identifier", day=DAY, note=None)
    derived = compute(source("people.json", "json", loose), dict(more="day + 1"))
    declared = dict(person=nodes(source("people.json", "json", loose), key="id", holds="always", known=None),
                    derived=derived, thing=nodes(derived, key="id", holds="always", known=None),
                    tagged=nodes(source("people.json", "json", dict(loose, note="category")), key="id",
                                 holds="always", known=None, features=dict(note="onehot")),
                    marked=nodes(source("people.json", "json", dict(loose, note="category")), key="id",
                                 holds="always", known=None, features=dict(note="ordinal")))
    with pytest.raises(Invalid) as raised:
        validate(Builder(declared, tmp_path), PLAN)
    assert [(error.declaration, error.places) for error in raised.value.errors] == [
        ("person", ("people.json",)), ("derived", ("people.json",)), ("marked", ("people.json",))]
    messages = [error.message for error in raised.value.errors]
    assert messages[:2] == ["source person (people.json) column note: the column has no stated type",
                            "source derived.inputs.0 (people.json) column note: the column has no stated type"]
    assert messages[2].startswith("the column note is stated by tagged as") and raised.value.unchecked == ("thing",)
    assert not INTERNAL.search(str(raised.value))


def test_two_rows_of_one_node_from_one_time_without_ties_are_reported_in_script_order(tmp_path):
    people = table(tmp_path, "people", [dict(id="a", day=1), dict(id="b", day=1), dict(id="a", day=1),
                                        dict(id="a", day=2)], dict(id="identifier", day=DAY))
    later = table(tmp_path, "later", [dict(id="b", day=1)], dict(id="identifier", day=DAY))
    loose = table(tmp_path, "loose", [dict(id="q", note="x")], dict(id="identifier", note=None))
    state = dict(holds=dict(name="state", since="day", until=None), known=None)
    person = nodes(people, key="id", **state)
    declared = dict(people=people, person=person, again=nodes(later, key="id", of=person, **state), loose=loose,
                    other=nodes(loose, key="id", holds="always", known=None))
    with pytest.raises(Invalid) as raised:
        validate(Builder(declared, tmp_path), PLAN)
    assert [(error.declaration, error.places, error.message) for error in raised.value.errors] == [
        ("person", ("people.json",), "declaration person field ties rows 0 and 2: two facts about the node a of the "
                                     "kind person hold from the time 1 and are known together from 1; the mappings "
                                     "state no rule for ties"),
        ("person", ("people.json", "later.json"), "declaration person field ties row 1 and declaration again row 0: "
                                                  "two facts about the node b of the kind person hold from the time 1 "
                                                  "and are known together from 1; the mappings state no rule for ties"),
        ("loose", ("loose.json",), "source loose (loose.json) column note: the column has no stated type")]
    assert raised.value.unchecked == ("other",) and not INTERNAL.search(str(raised.value))
    stated = dict(declared, again=nodes(later, key="id", of=person, ties="key", **state))
    with pytest.raises(Invalid) as raised:
        validate(Builder(stated, tmp_path), PLAN)
    assert [error.declaration for error in raised.value.errors] == ["loose"]
