import sqlite3

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from test_mappings import DAY, UTILITY, called, edge, kind, mapped, schema, table

from graphspec import (
    FUNCTIONS,
    SCRIPTED,
    Builder,
    BuildPlan,
    compute,
    database,
    edges,
    join,
    nodes,
    scripted,
    sets,
    source,
    tasks,
)

pytestmark = pytest.mark.builder
LATER = dict(type="time", unit="day", missing="omit")
SPAN = dict(first="1", last="5", every=1)
WRITTEN = dict(person=[("a", 1), ("a", 2), ("a", 3), ("a", 4), ("a", 5), ("b", 2), ("c", 3), ("c", 4), ("c", 5)],
               paper=[("p1", 2), ("p2", 4)], link=[("k1", 1), ("k1", 2)], pay=[("a", 2), ("c", 4)])
AIMS = [("a", 1, 6.0, 4.0), ("a", 2, 1.0, 0.0), ("a", 3, 1.0, 0.0), ("a", 4, None, None), ("a", 5, None, None),
        ("b", 2, 7.0, 0.0), ("c", 3, 3.0, 0.0), ("c", 4, 3.0, 0.0), ("c", 5, None, None)]


def things(folder):
    people = table(folder, "people", [dict(id="a", since=1, until=None, seen=1),
                                      dict(id="b", since=2, until=3, seen=2),
                                      dict(id="c", since=1, until=None, seen=3)],
                   dict(id="identifier", since=DAY, until=LATER, seen=DAY))
    works = table(folder, "works", [dict(id="p1", day=2), dict(id="p2", day=4)], dict(id="identifier", day=DAY))
    bylines = table(folder, "bylines", [dict(id="p1", author="a"), dict(id="p2", author="c")],
                    dict(id="identifier", author="identifier"))
    links = table(folder, "links", [dict(id="k1", one="a", other="c", since=1, until=3)],
                  dict(id="identifier", one="identifier", other="identifier", since=DAY, until=DAY))
    paid = table(folder, "paid", [dict(one="a", other="c", day=2), dict(one="c", other="a", day=4)],
                 dict(one="identifier", other="identifier", day=DAY))
    person = nodes(people, key="id", holds=dict(name="state", since="since", until="until"), known="seen")
    paper = sets(works, key="id", holds=dict(name="event", at="day"), known=None,
                 members=[dict(rows=bylines, set="id", node=dict(of=person, key="author", absent="stop"),
                               role="'author'")])
    link = edges(links, key="id", source=dict(of=person, key="one", absent="stop"),
                 target=dict(of=person, key="other", absent="stop"), directed=True,
                 holds=dict(name="state", since="since", until="until"), known=None)
    pay = edges(paid, source=dict(of=person, key="one", absent="stop"), target=dict(of=person, key="other", absent="stop"),
                directed=True, holds=dict(name="event", at="day"), known=None)
    return dict(people=people, works=works, bylines=bylines, links=links, paid=paid, person=person, paper=paper,
                link=link, pay=pay)


def asking(declared, rows):
    person = declared["person"]
    return dict(
        on_person=tasks(rows("person"), key=["id", "moment"], at="moment",
                        references=dict(who=dict(sort="node", of=person, key="id", absent="stop"))),
        on_paper=tasks(rows("paper"), key=["id", "moment"], at="moment",
                       references=dict(work=dict(sort="node", of=declared["paper"], key="id", absent="stop"))),
        on_link=tasks(rows("link"), key=["id", "moment"], at="moment",
                      references=dict(tie=dict(sort="edge", of=declared["link"], key="id", absent="stop"))),
        on_pay=tasks(rows("pay"), key=["one", "moment"], at="moment",
                     references=dict(payer=dict(sort="node", of=person, key="one", absent="stop"))))


def test_tasks_made_by_the_rules_nodes_and_facts_equal_the_tasks_read_from_rows_written_out(tmp_path):
    declared = things(tmp_path)

    def ruled(name):
        return dict(name="facts", of=declared[name]) if name == "pay" else dict(name="nodes", of=declared[name],
                                                                               times=SPAN)

    written = {name + "_rows": table(tmp_path, name, [dict(id=key, one=key, moment=moment) for key, moment in rows],
                                     dict(id="identifier", one="identifier", moment=DAY))
               for name, rows in WRITTEN.items()}
    made = mapped(tmp_path, dict(declared, **asking(declared, ruled)))
    assert made == mapped(tmp_path, dict(declared, **written, **asking(declared, lambda name: written[name + "_rows"])))
    assert len(made["graphs"][0]["tasks"]["key"]) == sum(len(rows) for rows in WRITTEN.values())


def spread(values):
    if not len(values):
        return pa.scalar(None, pa.float64())
    found = pc.min_max(values).as_py()
    return pa.scalar(float(found["max"] - found["min"]), pa.float64())


def aimed(rows, person, value, later):
    target = dict(kind="number", series="spent", value=value, known="moment + 10", about=None, utility=UTILITY)
    if later is not None:
        target["later"] = later
    return tasks(rows, key=["id", "moment"], at="moment", split="'train'", target=target,
                 references=dict(who=dict(sort="node", of=person, key="id", absent="stop")))


def test_a_target_over_later_rows_and_one_by_a_function_of_the_script_equal_the_targets_written_out(
        tmp_path, monkeypatch):
    monkeypatch.setattr(FUNCTIONS, "held", {name: FUNCTIONS.named(name) for name in FUNCTIONS.names()})
    monkeypatch.setattr(SCRIPTED, "held", {})
    reason = "no shipped aggregate gives the spread of a column"
    scripted("function", reason=reason, reads=("number",), gives="number", aggregate=True, version=1)(spread)
    declared = things(tmp_path)
    person = declared["person"]
    spends = table(tmp_path, "spends", [dict(id="a", day=2, amount=5), dict(id="a", day=4, amount=1),
                                        dict(id="b", day=3, amount=7), dict(id="c", day=5, amount=3)],
                   dict(id="identifier", day=DAY, amount="number"))
    later = dict(rows=spends, on=dict(id="id"), at="day", within=None)
    rule = dict(name="nodes", of=person, times=SPAN)
    made = mapped(tmp_path, dict(declared, spends=spends, summed=aimed(rule, person, "sum(later.amount)", later),
                                 spread=aimed(rule, person, "spread(later.amount)", later)))
    aims = table(tmp_path, "aims", [dict(id=key, moment=moment, summed=summed, spread=spread)
                                    for key, moment, summed, spread in AIMS],
                 dict(id="identifier", moment=DAY, summed=dict(type="number", missing="omit"),
                      spread=dict(type="number", missing="omit")))
    assert made == mapped(tmp_path, dict(declared, aims=aims, summed=aimed(aims, person, "summed", None),
                                         spread=aimed(aims, person, "spread", None)))
    assert SCRIPTED.named("spread") == dict(registry="function", reason=reason, version=1)
    with pytest.raises(ValueError, match="the registry 'presence' takes none yet"):
        scripted("presence", reason=reason, version=1)
    with pytest.raises(ValueError, match="states reason, why no declaration serves"):
        scripted("function", reason="", reads=("number",), gives="number", aggregate=True, version=1)


def test_an_outcome_column_is_carried_by_a_fact_from_the_time_it_is_known(tmp_path):
    outages = table(tmp_path, "outages", [dict(id="o1", began=1, ended=3, cause="x", recorded=5),
                                          dict(id="o2", began=2, ended=4, cause="y", recorded=2)],
                    dict(id="identifier", began=DAY, ended=DAY, cause=dict(type="category", outcome="recorded"),
                         recorded=DAY))
    orders = table(tmp_path, "orders", [dict(number="n1", placed=1, shipped=3, cost=2),
                                        dict(number="n2", placed=2, shipped=None, cost=None)],
                   dict(number="identifier", placed=DAY, shipped=dict(LATER, outcome="shipped"),
                        cost=dict(type="number", missing="omit", outcome="shipped")))
    outage = nodes(outages, key="id", features=dict(cause="onehot"),
                   holds=dict(name="state", since="began", until="ended"), known=None)
    order = nodes(orders, key="number", holds=dict(name="state", since="placed", until=None), known=None)
    shipment = edges(orders, where="shipped is not missing", source=dict(of=order, key="number", absent="stop"),
                     target=dict(of=order, key="number", absent="stop"), directed=True, features=dict(cost="value"),
                     holds=dict(name="event", at="shipped"), known=None)
    ask = tasks(orders, at="placed", references=dict(at=dict(sort="node", of=order, key="number", absent="stop")))
    found = mapped(tmp_path, dict(outages=outages, orders=orders, outage=outage, order=order, shipment=shipment,
                                  ask=ask))
    builder = called(tmp_path, schema(
        dict(outage=["cause"], order=[]), dict(shipment=edge([("order", "order")], columns=["cost"])),
        columns=dict(cause=dict(type="category", encoder="onehot"), cost=dict(type="number", encoder="value")),
        model="clocks", units=["day"], kinds=dict(ask=kind(["at"])),
        references=dict(at=dict(sort="node", kinds=["order"]))), unit="day")
    builder.node("outage", "o1", 1, 3, known=None, replaced=5, cause=None)
    builder.node("outage", "o1", 1, 3, known=5, replaced=None, cause="x")
    builder.node("outage", "o2", 2, 4, known=None, replaced=None, cause="y")
    builder.node("order", "n1", 1, None, known=None, replaced=None).node("order", "n2", 2, None, known=None,
                                                                         replaced=None)
    builder.event("shipment", ("order", "n1"), ("order", "n1"), 3, known=None, replaced=None, cost=2.0)
    builder.task("ask", 0, 1, dict(at="n1"), None, None, None).task("ask", 1, 2, dict(at="n2"), None, None, None)
    assert found == builder.records()


def test_a_database_carries_an_outcome_column_known_in_time_as_a_feature_and_never_one_never_known(tmp_path):
    with sqlite3.connect(tmp_path / "games.sqlite") as connection:
        connection.executescript("CREATE TABLE game (id INTEGER PRIMARY KEY, played DATE NOT NULL, score REAL NOT NULL, "
                                 "won INTEGER NOT NULL); INSERT INTO game VALUES (1, '2024-01-01', 3.0, 1);")
    connection.close()
    games = database("games.sqlite", reader="sqlite", tables=["*"], unit="day", times="typed",
                     encoders=dict(number="standard"), missing="omit", absent="stop",
                     outcomes={"game.score": "played + 2", "game.won": "never"})
    ask = tasks(games.game, at="played", references=dict(at=dict(sort="node", of=games.game, key="id", absent="stop")))
    found = mapped(tmp_path, dict(games=games, ask=ask))
    assert found["schema"]["nodes"]["game"]["columns"] == ["score"] and found["schema"]["time"]["model"] == "clocks"


def refusing(folder):
    rows = table(folder, "rows", [dict(id="a", day=1, ended=3, late=True, cost=2),
                                  dict(id="b", day=2, ended=5, late=False, cost=4)],
                 dict(id="identifier", day=DAY, ended=DAY, late=dict(type="flag", outcome="day + 1"),
                      cost=dict(type="number", outcome="never")))
    verdicts = table(folder, "verdicts", [dict(id="a", day=1, verdict=True), dict(id="b", day=2, verdict=False)],
                     dict(id="identifier", day=DAY, verdict=dict(type="flag", outcome="day + 2")))
    thing = nodes(rows, key="id", holds=dict(name="state", since="day", until="ended"), known=None)
    return dict(rows=rows, verdicts=verdicts, thing=thing,
                flagged=compute(rows, columns=dict(flag="not late", left="ended - day")),
                joined=join(rows, verdicts, on=dict(id="id"), take=dict(said="verdict"), unmatched="stop"))


def refused(folder, declared, message, **stated):
    with pytest.raises(ValueError, match=message):
        mapped(folder, dict(declared, **stated))


def asked(table_, declared, **fields):
    return tasks(table_, at="day", references=dict(at=dict(sort="node", of=declared["thing"], key="id",
                                                           absent="stop")), **fields)


def test_an_outcome_is_refused_as_a_feature_of_a_task_in_a_filter_and_in_a_split_directly_and_through_operations(
        tmp_path):
    declared = refusing(tmp_path)
    target = dict(kind="number", series="s", value="1", known="never", about=None, utility=UTILITY)
    for table_, fields, message in (
            ("rows", dict(features=dict(late="value")), "ask field features.late reads the outcome column rows.late"),
            ("flagged", dict(features=dict(flag="value")), "ask field features.flag reads the outcome column rows.late"),
            ("joined", dict(features=dict(said="value")),
             "ask field features.said reads the outcome column verdicts.verdict"),
            ("rows", dict(where="late"), "ask field where reads the outcome column rows.late"),
            ("flagged", dict(where="flag"), "ask field where reads the outcome column rows.late"),
            ("rows", dict(split="'a' if late else 'b'", target=target),
             "ask field split reads the outcome column rows.late"),
            ("joined", dict(split="'a' if said else 'b'", target=target),
             "ask field split reads the outcome column verdicts.verdict"),
            ("rows", dict(eligible="late"), "ask field eligible reads the outcome column rows.late")):
        refused(tmp_path, declared, message, ask=asked(declared[table_], declared, **fields))
    reference = dict(at=dict(sort="node", of=declared["thing"], key="id if late else id", absent="stop"))
    refused(tmp_path, declared, "ask field references.at.key reads the outcome column rows.late",
            ask=tasks(declared["rows"], at="day", references=reference))
    refused(tmp_path, declared, "ask field at reads the outcome column rows.late, known from 3 in row 1, later than "
                                "the time of its task, from 2",
            ask=tasks(declared["rows"], at="day + number(late)", references=dict(at=dict(reference["at"], key="id"))))


def test_an_outcome_is_refused_as_a_fact_or_a_feature_of_a_fact_before_it_is_known(tmp_path):
    declared = refusing(tmp_path)
    rows, thing = declared["rows"], declared["thing"]
    ask = asked(rows, declared)
    end = dict(of=thing, key="id", absent="stop")
    for stated, message in (
            (nodes(rows, key="id", where="late", holds=dict(name="state", since="day", until=None), known=None),
             "other field where reads the outcome column rows.late, known from 2 in row 0, later than its fact is "
             "known, from 1"),
            (edges(rows, source=end, target=end, directed=True, features=dict(late="value"),
                   holds=dict(name="event", at="day"), known=None),
             "other field features.late reads the outcome column rows.late, known from 2 in row 0, later than its "
             "edge is known, from 1"),
            (nodes(rows, of=thing, key="id", features=dict(late="value"), holds=dict(name="change", at="day"),
                   known=None),
             "other field features.late reads the outcome column rows.late, known from 2 in row 0, later than its "
             "fact is known; a fact of the form change is never stated twice"),
            (nodes(rows, of=thing, key="id", features=dict(cost="value"), holds="always", known=None),
             "other field features.cost reads the outcome column rows.cost, which is never known"),
            (nodes(rows, of=thing, key="id", holds=dict(name="event", at="day + number(late)"), known="day"),
             "other field holds.at reads the outcome column rows.late, known from 2 in row 0, later than its fact")):
        refused(tmp_path, declared, message, other=stated, ask=ask)


def test_a_target_known_too_early_and_the_end_of_a_state_read_by_a_task_are_refused(tmp_path):
    declared = refusing(tmp_path)
    rows = declared["rows"]

    def aimed_(table_, value, known, later=None):
        target = dict(kind="number", series="s", value=value, known=known, about=None, utility=UTILITY)
        target.update({} if later is None else dict(later=later))
        return asked(declared[table_], declared, split="'train'", target=target)

    within = source("rows.json", "json", rows.columns, through="time('1970-01-04')")
    for ask, message in (
            (aimed_("rows", "number(late)", "day"), "ask field target.known row 0 gives 1, not later than the task's"),
            (aimed_("joined", "number(said)", "day + 1"),
             "ask field target.known row 0 gives 2, earlier than the outcome column verdicts.verdict that "
             "target.value reads is known, from 3"),
            (aimed_("rows", "count(later.id)", "day + 1", dict(rows=within, on=dict(id="id"), at="day", within=2)),
             "ask field target.known row 0 gives 2, earlier than its later rows end, at 3"),
            (asked(declared["flagged"], declared, features=dict(left="value")),
             r"ask field features.left reads left, the column until of a state reads \(rows.ended\)")):
        refused(tmp_path, dict(declared, within=within), message, ask=ask)
    left = nodes(declared["flagged"], of=declared["thing"], key="id", features=dict(left="value"),
                 holds=dict(name="state", since="day", until=None), known=None)
    refused(tmp_path, declared, "other field features.left reads left, the column until of a state reads",
            other=left, ask=asked(rows, declared))


def test_a_task_made_by_a_rule_refuses_a_fact_known_later_and_a_rule_beyond_the_budget(tmp_path):
    declared = refusing(tmp_path)
    rows = declared["rows"]
    late = nodes(rows, key="id", holds=dict(name="event", at="day"), known="day + 1")
    reference = dict(at=dict(sort="node", of=late, key="id", absent="stop"))
    refused(tmp_path, declared, "ask field at row 0 dates a task at 1, which reads the row of a fact known later, "
                                "from 2",
            late=late, ask=tasks(dict(name="facts", of=late), at="moment", references=reference))
    forever = nodes(rows, key="id", holds=dict(name="state", since="day", until=None), known=None)
    many = dict(name="nodes", of=forever, times=dict(first="1", last="100000000", every=1))
    with pytest.raises(ValueError, match="the tasks the rule nodes makes.* needs .* bytes, above the source worker "
                                         "budget"):
        Builder(dict(declared, forever=forever, ask=tasks(many, at="moment", references=dict(
            at=dict(sort="node", of=forever, key="id", absent="stop")))), tmp_path).map(BuildPlan(16 << 20, 1), "one")
