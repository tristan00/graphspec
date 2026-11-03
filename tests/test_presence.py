import json

import pytest

from graphspec import Builder, BuildPlan, edges, nodes, source, tasks
from graphspec.schema.schema import VERSION

pytestmark = pytest.mark.builder
PLAN = BuildPlan(16 << 20, 1)
DAY = dict(type="time", unit="day")
SPAN = dict(type="time", unit="day", missing="omit")


def table(folder, name, rows, columns):
    (folder / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
    return source(name + ".json", "json", columns)


def schema(nodes, edges, columns, references, absent):
    kinds = dict(ask=dict(references=list(references), columns=[], target=None, weight=False, absent=absent,
                          event=None, matures=None))
    return dict(version=VERSION, time=dict(model="clock", units=["day"], ties=None), observers=False, columns=columns,
                graph=dict(columns=[]), nodes={kind: dict(columns=held, holds=None) for kind, held in nodes.items()},
                sets={}, edges=edges, tasks=dict(key="key", grouping=None, signals=0, references=references,
                                                 kinds=kinds))


def called(folder, stated):
    return Builder({}, folder).schema(stated).graph("one", "day")


def staged(builder, folder):
    builder.stage(folder / "staged", "one", PLAN)
    return json.loads((folder / "staged" / "one.staging" / "build.json").read_text(encoding="utf-8"))["presence"]


def people(folder):
    return table(folder, "people", [dict(id="a", born=1, left=5), dict(id="b", born=3, left=None)],
                 dict(id="identifier", born=DAY, left=SPAN))


def person_of(rows):
    return nodes(rows, key="id", holds=dict(name="state", since="born", until="left"), known=None)


def test_each_rule_of_presence_gives_the_facts_and_the_counts_written_out_and_a_key_absent_at_its_time_meets_it(
        tmp_path):
    persons = people(tmp_path)
    knows = table(tmp_path, "knows", [dict(one="a", other="b", day=4), dict(one="a", other="b", day=6),
                                      dict(one="b", other="z", day=4), dict(one="b", other="a", day=7)],
                  dict(one="identifier", other="identifier", day=DAY))
    asks = table(tmp_path, "asks", [dict(who="a", group=["a", "b"], day=2), dict(who="a", group=["a", "b"], day=6)],
                 dict(who="identifier", group=dict(type="list", of="identifier"), day=DAY))
    person = person_of(persons)
    knew = edges(knows, source=dict(of=person, key="one", absent="skip"),
                 target=dict(of=person, key="other", absent="create"), directed=True,
                 holds=dict(name="event", at="day"), known=None)
    ask = tasks(asks, at="day", references=dict(who=dict(sort="node", of=person, key="who", absent="keep"),
                                                group=dict(sort="nodes", of=person, keys="group", absent="skip")))
    made = Builder(dict(persons=persons, knows=knows, asks=asks, person=person, knew=knew, ask=ask), tmp_path)
    made.map(PLAN, "one")
    references = dict(who=dict(sort="node", kinds=["person"]), group=dict(sort="nodes", kinds=["person"]))
    builder = called(tmp_path, schema(dict(person=[]), dict(knew=dict(
        endpoints=[["person", "person"]], directed=True, columns=[], relation="declared", about=None)), {},
        references, "keep"))
    builder.node("person", "a", 1, 5, known=None).node("person", "b", 3, None, known=None)
    builder.node("person", "z", 4, 5, known=None).node("person", "a", 7, 8, known=None)
    for one, other, day in (("a", "b", 4), ("b", "z", 4), ("b", "a", 7)):
        builder.event("knew", ("person", one), ("person", other), day, known=None)
    builder.task("ask", 0, 2, dict(who="a", group=["a"]), None, None, None)
    builder.task("ask", 1, 6, dict(who="a", group=["b"]), None, None, None)
    assert made.records() == builder.records()
    assert staged(made, tmp_path) == dict(ask={"references.group": dict(skip=2), "references.who": dict(keep=1)},
                                          knew=dict(source=dict(skip=1), target=dict(create=2)))
    stopping = tasks(asks, at="day", references=dict(who=dict(sort="node", of=person, key="who", absent="stop")))
    with pytest.raises(ValueError, match="declaration ask field references.who row 1 key a: the rule stop meets the "
                                         "key 'a', which names no node at the time 6"):
        Builder(dict(persons=persons, asks=asks, person=person, ask=stopping), tmp_path).map(PLAN, "one")


def sensors(folder, **between):
    folder.mkdir(exist_ok=True)
    rows = [dict(id="s1", day=0, value=1), dict(id="s2", day=0, value=2), dict(id="s1", day=3, value=4),
            dict(id="s1", day=5, value=6), dict(id="s2", day=5, value=7)]
    readings = table(folder, "readings", rows, dict(id="identifier", day=DAY, value="number"))
    probes = table(folder, "probes", [dict(id="s1", day=2), dict(id="s2", day=4), dict(id="s1", day=5)],
                   dict(id="identifier", day=DAY))
    sensor = nodes(readings, key="id", features=dict(value="value"),
                   holds=dict(name="snapshot", at="day", **between), known=None)
    ask = tasks(probes, at="day", references=dict(at=dict(sort="node", of=sensor, key="id", absent="skip")))
    return Builder(dict(readings=readings, probes=probes, sensor=sensor, ask=ask), folder)


REFERENCES = dict(at=dict(sort="node", kinds=["sensor"]))
VALUE = dict(type="number", encoder="value")


def test_the_rule_absent_holds_a_row_at_its_sighting_alone_and_counts_what_misses_it(tmp_path):
    made = sensors(tmp_path, between="absent")
    made.map(PLAN, "one")
    builder = called(tmp_path, schema(dict(sensor=["value"]), {}, dict(value=VALUE), REFERENCES, "stop"))
    for key, day, held in (("s1", 0, 1.0), ("s2", 0, 2.0), ("s1", 3, 4.0), ("s1", 5, 6.0), ("s2", 5, 7.0)):
        builder.node("sensor", key, day, day + 1, known=None, value=held)
    builder.task("ask", 0, 5, dict(at="s1"), None, None, None)
    assert made.records() == builder.records()
    assert staged(made, tmp_path) == dict(ask={"references.at": dict(skip=2)})


def test_the_rule_carried_holds_a_row_until_the_next_sighting_with_its_age_and_counts_what_misses_it(tmp_path):
    made = sensors(tmp_path / "carried", between=dict(name="carried", age="standard"))
    made.map(PLAN, "one")
    columns = {"value": VALUE, "sensor.age": dict(type="number", encoder="standard")}
    builder = called(tmp_path, schema(dict(sensor=["value", "sensor.age"]), {}, columns, REFERENCES, "stop"))
    for key, seen, last, held in (("s1", 0, 3, 1.0), ("s2", 0, 3, 2.0), ("s1", 3, 5, 4.0), ("s1", 5, 6, 6.0),
                                  ("s2", 5, 6, 7.0)):
        for day in range(seen, last):
            builder.node("sensor", key, day, day + 1, known=None,
                         values={"value": held, "sensor.age": float(day - seen)})
    builder.task("ask", 0, 2, dict(at="s1"), None, None, None).task("ask", 1, 5, dict(at="s1"), None, None, None)
    assert made.records() == builder.records()
    assert staged(made, tmp_path / "carried") == dict(ask={"references.at": dict(skip=1)})
    renamed = sensors(tmp_path / "renamed", between=dict(name="carried", age="standard", names=dict(age="stale")))
    assert renamed.map(PLAN, "one").records()["schema"]["nodes"]["sensor"]["columns"] == ["value", "stale"]


def test_an_unstated_or_unknown_rule_stops_naming_the_declaration_and_the_field(tmp_path):
    persons = people(tmp_path)
    person = person_of(persons)
    with pytest.raises(ValueError, match="edges source lacks the required field absent"):
        edges(persons, source=dict(of=person, key="id"), target=dict(of=person, key="id", absent="stop"),
              directed=True)
    with pytest.raises(ValueError, match="tasks references who lacks the required field absent"):
        tasks(persons, references=dict(who=dict(sort="node", of=person, key="id")))
    for rule, message in (("count", "declaration loop field source.absent: no rule of presence is registered under "
                                    "the name 'count'"),
                          ("keep", "declaration loop field source.absent states the rule keep, which keeps a "
                                   "reference only for a reference of tasks")):
        loop = edges(persons, source=dict(of=person, key="id", absent=rule),
                     target=dict(of=person, key="id", absent="stop"), directed=True, holds="always", known=None)
        with pytest.raises(ValueError, match=message):
            Builder(dict(persons=persons, person=person, loop=loop), tmp_path).map(PLAN, "one")
    with pytest.raises(ValueError, match="declaration sensor field holds.between: the form snapshot states the rule"):
        sensors(tmp_path).map(PLAN, "one")
    with pytest.raises(ValueError, match="sensor field holds.between: no rule between sightings is registered under "
                                         "the name 'kept'"):
        sensors(tmp_path, between="kept").map(PLAN, "one")
