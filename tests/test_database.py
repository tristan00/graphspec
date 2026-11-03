import csv
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from graphspec import Builder, BuildPlan, Database, database, edges, nodes, source, tasks
from graphspec.build.database_build import typed

pytestmark = pytest.mark.builder
PLAN = BuildPlan(16 << 20, 1)
THROUGH = "time('2024-06-30')"
ENCODERS = dict(number="standard", category="onehot", flag="value")
DEFINITIONS = """
CREATE TABLE region (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE city (id INTEGER PRIMARY KEY, region INTEGER NOT NULL REFERENCES region(id), name TEXT NOT NULL);
CREATE TABLE customer (id INTEGER PRIMARY KEY, city INTEGER REFERENCES city(id), joined DATE NOT NULL,
                       vip BOOLEAN NOT NULL);
CREATE TABLE supplier (id INTEGER PRIMARY KEY, city INTEGER NOT NULL REFERENCES city(id), rating REAL);
CREATE TABLE category (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE item (id INTEGER PRIMARY KEY, category INTEGER NOT NULL REFERENCES category(id),
                   supplier INTEGER NOT NULL REFERENCES supplier(id), price REAL NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id), placed DATE NOT NULL,
                     total REAL NOT NULL);
CREATE TABLE line (orders INTEGER NOT NULL REFERENCES orders(id), place INTEGER NOT NULL,
                   item INTEGER NOT NULL REFERENCES item(id), quantity INTEGER NOT NULL, PRIMARY KEY (orders, place));
CREATE TABLE review (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                     item INTEGER NOT NULL REFERENCES item(id), written DATE NOT NULL, stars INTEGER NOT NULL);
CREATE TABLE employee (id INTEGER PRIMARY KEY, manager INTEGER REFERENCES employee(id), hired DATE NOT NULL, ended DATE);
CREATE TABLE store (id INTEGER PRIMARY KEY, city INTEGER NOT NULL REFERENCES city(id), opened DATE NOT NULL);
CREATE TABLE shift (id INTEGER PRIMARY KEY, employee INTEGER NOT NULL REFERENCES employee(id),
                    store INTEGER NOT NULL REFERENCES store(id), starts DATE NOT NULL);
CREATE TABLE stocks (store INTEGER NOT NULL REFERENCES store(id), item INTEGER NOT NULL REFERENCES item(id),
                     amount REAL NOT NULL, PRIMARY KEY (store, item));
CREATE TABLE follows (follower INTEGER NOT NULL REFERENCES customer(id), followed INTEGER NOT NULL REFERENCES customer(id),
                      strength REAL NOT NULL, PRIMARY KEY (follower, followed));
CREATE VIEW recent AS SELECT id, customer, placed FROM orders;
CREATE VIEW priced AS SELECT id, price FROM item;
"""
ROWS = """
INSERT INTO region VALUES (1, 'north'), (2, 'south');
INSERT INTO city VALUES (1, 1, 'oslo'), (2, 2, 'rome');
INSERT INTO customer VALUES (1, 1, '2024-01-01', 1), (2, NULL, '2024-01-15', 0), (3, 2, '2024-03-15', 0);
INSERT INTO supplier VALUES (1, 1, 4.5), (2, 2, NULL);
INSERT INTO category VALUES (1, 'tools');
INSERT INTO item VALUES (1, 1, 1, 3.5), (2, 1, 2, 9.0);
INSERT INTO orders VALUES (1, 1, '2024-01-20', 10.0), (2, 1, '2024-03-10', 5.0), (3, 2, '2024-02-20', 7.0),
                          (4, 3, '2024-04-20', 12.0);
INSERT INTO line VALUES (1, 1, 1, 2), (1, 2, 2, 1), (2, 1, 2, 3);
INSERT INTO review VALUES (1, 1, 2, '2024-02-02', 4), (2, 2, 1, '2024-03-01', 2);
INSERT INTO employee VALUES (1, NULL, '2023-06-01', NULL), (2, 1, '2023-09-01', '2024-05-01');
INSERT INTO store VALUES (1, 1, '2023-01-01');
INSERT INTO shift VALUES (1, 2, 1, '2024-02-10');
INSERT INTO stocks VALUES (1, 1, 10.0), (1, 2, 4.0);
INSERT INTO follows VALUES (1, 2, 0.5), (3, 1, 0.25);
"""
SUBSET = ("region", "city", "customer", "orders", "follows", "recent")
CLASS = dict(name="weighted", better="lower", weights=dict(leaves=1.0, stays=0.0))
COUNT = dict(name="weighted", better="higher", weights=1.0)
TIMES = dict(first="time('2024-02-01')", last="time('2024-04-01')", every=30)
SPLIT = "'train' if moment < time('2024-03-01') else 'test'"
ID, DAY = "identifier", dict(type="time", unit="day", format="%Y-%m-%d")
STANDARD = dict(type="number", encoder="standard")
COLUMNS = dict(name=dict(type="category", encoder="onehot"), vip=dict(type="flag", encoder="value"), strength=STANDARD,
               price=STANDARD, quantity=STANDARD, total=STANDARD, rating=STANDARD, stars=STANDARD, amount=STANDARD)
KINDS = dict(category=["name"], region=["name"], city=["name"], customer=["vip"], supplier=["rating"], item=["price"],
             orders=["total"], line=["quantity"], store=[], shift=[], staff=[], stocked=["amount"], featured=[])
RELATIONS = {
    "city.region": ("city", "region", []), "customer.city": ("customer", "city", []),
    "supplier.city": ("supplier", "city", []), "item.category": ("item", "category", []),
    "item.supplier": ("item", "supplier", []), "orders.customer": ("orders", "customer", []),
    "line.orders": ("line", "orders", []), "line.item": ("line", "item", []), "store.city": ("store", "city", []),
    "shift.employee": ("shift", "staff", []), "shift.store": ("shift", "store", []), "staff.boss": ("staff", "staff", []),
    "stocked.store": ("stocked", "store", []), "stocked.item": ("stocked", "item", []),
    "follows": ("customer", "customer", ["strength"]), "reviewed": ("customer", "item", ["stars"]),
}


def fact(table, *ends, since=None, until=None, form="state", **values):
    return table, form, ends, since, until, sorted(values.items())


FACTS = sorted([
    fact("category", "1", name="tools"), fact("region", "1", name="north"), fact("region", "2", name="south"),
    fact("city", "1", name="oslo"), fact("city", "2", name="rome"),
    fact("customer", "1", since="2024-01-01", vip=True), fact("customer", "2", since="2024-01-15", vip=False),
    fact("customer", "3", since="2024-03-15", vip=False),
    fact("supplier", "1", rating=4.5), fact("supplier", "2", rating=None),
    fact("item", "1", price=3.5), fact("item", "2", price=9.0),
    fact("orders", "1", since="2024-01-20", total=10.0), fact("orders", "2", since="2024-03-10", total=5.0),
    fact("orders", "3", since="2024-02-20", total=7.0), fact("orders", "4", since="2024-04-20", total=12.0),
    fact("line", "1|1", quantity=2.0), fact("line", "1|2", quantity=1.0), fact("line", "2|1", quantity=3.0),
    fact("store", "1", since="2023-01-01"), fact("shift", "1", since="2024-02-10"),
    fact("staff", "1", since="2023-06-01"), fact("staff", "2", since="2023-09-01", until="2024-05-01"),
    fact("stocked", "1|1", amount=10.0), fact("stocked", "1|2", amount=4.0), fact("featured", "2"),
    fact("city.region", "1", "1"), fact("city.region", "2", "2"),
    fact("customer.city", "1", "1", since="2024-01-01"), fact("customer.city", "3", "2", since="2024-03-15"),
    fact("supplier.city", "1", "1"), fact("supplier.city", "2", "2"),
    fact("item.category", "1", "1"), fact("item.category", "2", "1"),
    fact("item.supplier", "1", "1"), fact("item.supplier", "2", "2"),
    fact("orders.customer", "1", "1", since="2024-01-20"), fact("orders.customer", "2", "1", since="2024-03-10"),
    fact("orders.customer", "3", "2", since="2024-02-20"), fact("orders.customer", "4", "3", since="2024-04-20"),
    fact("line.orders", "1|1", "1"), fact("line.orders", "1|2", "1"), fact("line.orders", "2|1", "2"),
    fact("line.item", "1|1", "1"), fact("line.item", "1|2", "2"), fact("line.item", "2|1", "2"),
    fact("store.city", "1", "1", since="2023-01-01"),
    fact("shift.employee", "1", "2", since="2024-02-10"), fact("shift.store", "1", "1", since="2024-02-10"),
    fact("staff.boss", "2", "1", since="2023-09-01", until="2024-05-01"),
    fact("stocked.store", "1|1", "1"), fact("stocked.store", "1|2", "1"),
    fact("stocked.item", "1|1", "1"), fact("stocked.item", "1|2", "2"),
    fact("follows", "1", "2", strength=0.5), fact("follows", "3", "1", strength=0.25),
    fact("reviewed", "1", "2", since="2024-02-02", form="event", stars=4.0),
    fact("reviewed", "2", "1", since="2024-03-01", form="event", stars=2.0),
], key=repr)
MOMENTS = ["2024-02-01", "2024-03-02", "2024-03-02", "2024-04-01", "2024-04-01"]


def shop(folder, definitions=DEFINITIONS, rows=ROWS):
    place = folder / "shop.sqlite"
    with sqlite3.connect(place) as connection:
        connection.executescript(definitions + rows)
    connection.close()
    return place


def stated(**fields):
    return database("shop.sqlite", reader="sqlite", **dict(dict(
        tables=["*"], unit="day", times="typed", through=THROUGH, encoders=ENCODERS, missing="omit", absent="stop"),
        **fields))


def churned(shop):
    return tasks(
        dict(name="nodes", of=shop.customer, times=TIMES), at="moment",
        earlier=dict(rows=shop.orders, on=dict(customer="id"), at="placed", within=60),
        eligible="count(earlier.id) > 0",
        references=dict(buyer=dict(sort="node", of=shop.customer, key="id", absent="stop")),
        target=dict(kind="class", value="'leaves' if count(later.id) == 0 else 'stays'",
                    later=dict(rows=shop.recent, on=dict(customer="id"), at="placed", within=60),
                    known="moment + 60", about="buyer", utility=CLASS),
        split=SPLIT)


def counted(shop):
    return tasks(
        dict(name="nodes", of=shop.customer, times=TIMES), at="moment", eligible="moment >= joined + 30",
        references=dict(buyer=dict(sort="node", of=shop.customer, key="id", absent="stop")),
        target=dict(kind="number", series="orders", value="count(later.id)",
                    later=dict(rows=shop.orders, on=dict(customer="id"), at="placed", within=120),
                    known="moment + 120", about="buyer", utility=COUNT),
        split=SPLIT)


def looked(shop, table="region"):
    return tasks(getattr(shop, table), at="time('2024-01-01')",
                 references=dict(at=dict(sort="node", of=getattr(shop, table), key="id", absent="stop")))


def six(asked):
    shop = stated()
    return dict(
        shop=shop,
        reviewed=edges(shop.review, source=dict(of=shop.customer, key="customer", absent="stop"),
                       target=dict(of=shop.item, key="item", absent="stop"), directed=True,
                       features=dict(stars="standard"), holds=dict(name="event", at="written"), known=None),
        stocked=nodes(shop.stocks, key=["store", "item"], features=dict(amount="standard"), holds="always", known=None),
        staff=nodes(shop.employee, key="id", links=dict(boss=dict(of=shop.employee, key="manager", absent="skip")),
                    holds=dict(name="state", since="hired", until="ended"), known=None),
        featured=nodes(shop.priced, key="id", where="price > 5", holds="always", known=None),
        asked=asked(shop),
    )


def iso(day):
    return None if day is None else (date(1970, 1, 1) + timedelta(days=day)).isoformat()


def facts(records):
    found = []
    for name in ("values", "edges"):
        for entry in records["graphs"][0][name]:
            rows = entry["rows"]
            ends = ["key"] if name == "values" else ["source", "target"]
            columns = sorted(column for column in rows if column.startswith("column:"))
            for place in range(len(rows[ends[0]])):
                since = rows["at" if entry["form"] == "event" else "since"][place]
                until = None if entry["form"] == "event" else rows["until"][place]
                found.append((entry.get("kind", entry.get("table")), entry["form"],
                              tuple(rows[end][place] for end in ends), iso(since), iso(until),
                              [(column[7:], rows[column][place]) for column in columns]))
    return sorted(found, key=repr)


def converted(folder, declarations):
    return Builder(declarations, folder).map(PLAN, "one").records()


def schema_of(records):
    schema = records["schema"]
    return (schema["columns"], {kind: entry["columns"] for kind, entry in schema["nodes"].items()},
            {name: (*entry["endpoints"][0], entry["columns"]) for name, entry in schema["edges"].items()})


def asked(records):
    tasks = records["graphs"][0]["tasks"]
    return ([iso(day) for day in tasks["time"]], tasks["references"]["buyer"], tasks["split"], tasks["target"],
            [iso(day) for day in tasks["known"]], records["schema"]["tasks"]["kinds"]["asked"]["target"])


def test_a_database_of_twelve_tables_two_link_tables_and_two_views_converts_in_six_statements(tmp_path):
    shop(tmp_path)
    script = six(churned)
    assert len(script) == 6
    found = converted(tmp_path, script)
    assert facts(found) == FACTS
    assert schema_of(found) == (COLUMNS, KINDS, RELATIONS)
    target = dict(kind="class", series=["leaves", "stays"], about="buyer", utility=CLASS)
    assert asked(found) == (MOMENTS, ["1", "1", "2", "1", "2"], ["train", "test", "test", "test", "test"],
                            ["stays", "stays", "leaves", "leaves", "leaves"],
                            ["2024-04-01", "2024-05-01", "2024-05-01", "2024-05-31", "2024-05-31"], target)


def test_the_same_six_statements_with_another_target_and_another_eligibility_convert_it_again(tmp_path):
    shop(tmp_path)
    script = six(counted)
    assert len(script) == 6
    found = converted(tmp_path, script)
    assert facts(found) == FACTS
    assert schema_of(found) == (COLUMNS, KINDS, RELATIONS)
    target = dict(kind="number", series="orders", about="buyer", utility=dict(COUNT, weights=dict(orders=1.0)))
    assert asked(found) == (MOMENTS, ["1", "1", "2", "1", "2"], ["train", "test", "test", "test", "test"],
                            [1.0, 1.0, 0.0, None, None], ["2024-05-31", "2024-06-30", "2024-06-30", None, None],
                            target)


def test_each_rule_of_the_database_holds_where_no_statement_overrides_it(tmp_path):
    shop(tmp_path)
    alone = stated()
    schema = converted(tmp_path, dict(shop=alone, ask=looked(alone)))["schema"]
    assert sorted(schema["nodes"]) == ["category", "city", "customer", "employee", "item", "line", "orders", "region",
                                       "review", "shift", "store", "supplier"]
    assert sorted(schema["edges"]) == [
        "city.region", "customer.city", "employee.manager", "follows", "item.category", "item.supplier", "line.item",
        "line.orders", "orders.customer", "review.customer", "review.item", "shift.employee", "shift.store", "stocks",
        "store.city", "supplier.city"]
    assert schema["edges"]["stocks"]["endpoints"] == [["store", "item"]]
    assert schema["edges"]["stocks"]["columns"] == ["amount"]
    assert schema["edges"]["employee.manager"]["endpoints"] == [["employee", "employee"]]
    assert schema["nodes"]["review"]["columns"] == ["stars"] and schema["nodes"]["employee"]["columns"] == []


def table(called, **columns):
    return source("shop.sqlite", "sqlite", columns, THROUGH, table=called)


def to(of, key):
    return dict(of=of, key=key, absent="stop")


def held(since=None):
    return dict(holds="always" if since is None else dict(name="state", since=since, until=None), known=None)


def normal(records):
    for graph in records["graphs"]:
        for name, value in graph.items():
            if isinstance(value, list):
                graph[name] = sorted(value, key=lambda entry: json.dumps(entry, sort_keys=True, default=str))
    return records


def test_a_database_converts_to_the_graph_its_tables_stated_by_mappings_make(tmp_path):
    shop(tmp_path)
    subset = stated(tables=list(SUBSET))
    found = normal(converted(tmp_path, dict(shop=subset, asked=churned(subset))))
    region = table("region", id=ID, name="category")
    city = table("city", id=ID, region=ID, name="category")
    customer = table("customer", id=ID, city=dict(type="identifier", missing="omit"), joined=DAY,
                     vip=dict(type="flag", true=1, false=0))
    orders = table("orders", id=ID, customer=ID, placed=DAY, total="number")
    follows = table("follows", follower=ID, followed=ID, strength="number")
    recent = table("recent", id=dict(type="number", missing="omit"), customer=dict(type="number", missing="omit"),
                   placed=dict(DAY, missing="omit"))
    region_ = nodes(region, key="id", features=dict(name="onehot"), **held())
    city_ = nodes(city, key="id", features=dict(name="onehot"), links=dict(region=to(region_, "region")), **held())
    customer_ = nodes(customer, key="id", features=dict(vip="value"), links=dict(city=to(city_, "city")), **held("joined"))
    rows = [dict(customer="1", moment="2024-02-01", aim="stays"), dict(customer="1", moment="2024-03-02", aim="stays"),
            dict(customer="1", moment="2024-04-01", aim="leaves"), dict(customer="2", moment="2024-03-02", aim="leaves"),
            dict(customer="2", moment="2024-04-01", aim="leaves")]
    (tmp_path / "asked.json").write_text(json.dumps(rows), encoding="utf-8")
    listed = source("asked.json", "json", dict(customer=ID, moment=DAY, aim="category"))
    expected = dict(
        regions=region, cities=city, customers=customer, placed=orders, following=follows, recently=recent,
        listed=listed, region=region_, city=city_, customer=customer_,
        follows=edges(follows, source=to(customer_, "follower"), target=to(customer_, "followed"), directed=True,
                      features=dict(strength="standard"), **held()),
        orders=nodes(orders, key="id", features=dict(total="standard"), links=dict(customer=to(customer_, "customer")),
                     **held("placed")),
        asked=tasks(listed, at="moment", references=dict(buyer=dict(sort="node", of=customer_, key="customer",
                                                                    absent="stop")),
                    target=dict(kind="class", value="aim", known="moment + 60", about="buyer", utility=CLASS),
                    split=SPLIT))
    assert found == normal(converted(tmp_path, expected))


def test_definitions_in_sql_over_a_folder_of_files_convert_as_the_catalogue_does(tmp_path):
    place = shop(tmp_path)
    folder = tmp_path / "files"
    folder.mkdir()
    with sqlite3.connect(place) as connection:
        for name in SUBSET:
            cursor = connection.execute('SELECT * FROM "%s"' % name)
            with (folder / (name + ".csv")).open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows([["" if value is None else value for value in row] for row in cursor.fetchall()])
    connection.close()
    (tmp_path / "shop.sql").write_text(DEFINITIONS, encoding="utf-8")
    catalogue = stated(times=None, tables=list(SUBSET), leave=["orders.total"])
    catalogued = normal(converted(tmp_path, dict(shop=catalogue, ask=tasks(catalogue.region, references=dict(
        at=dict(sort="node", of=catalogue.region, key="id", absent="stop"))))))
    files = database("files", reader="csv", definitions="shop.sql", tables=list(SUBSET), unit="day", times=None,
                     through=THROUGH, encoders=ENCODERS, missing="omit", absent="stop", leave=["orders.total"])
    defined = normal(converted(tmp_path, dict(shop=files, ask=tasks(files.region, references=dict(
        at=dict(sort="node", of=files.region, key="id", absent="stop"))))))
    assert defined == catalogued and catalogued["schema"]["time"]["model"] == "none"
    assert sorted(catalogued["schema"]["nodes"]) == ["city", "customer", "orders", "region"]
    assert catalogued["schema"]["nodes"]["orders"]["columns"] == []


def test_the_rule_named_states_one_form_for_every_table_that_holds_its_columns(tmp_path):
    shop(tmp_path)
    times = dict(name="named", holds=dict(name="event", at="placed"), known=None)
    named = stated(times=times, tables=["orders", "customer", "city", "region"])
    found = facts(converted(tmp_path, dict(shop=named, ask=looked(named))))
    orders = [entry[2:5] for entry in found if entry[0] == "orders"]
    assert orders == [(("1",), "2024-01-20", "2024-01-21"), (("2",), "2024-03-10", "2024-03-11"),
                      (("3",), "2024-02-20", "2024-02-21"), (("4",), "2024-04-20", "2024-04-21")]
    assert [entry[1] for entry in found if entry[0] == "orders.customer"] == ["event"] * 4
    assert {entry[3:5] for entry in found if not entry[0].startswith("orders")} == {(None, None)}


def test_the_described_database_script_maps_its_tasks_split_by_time_and_stages(tmp_path):
    script = Path(__file__).parent / "described" / "19_database.py"
    shop(tmp_path, """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, joined DATE NOT NULL, vip BOOLEAN NOT NULL);
        CREATE TABLE item (id INTEGER PRIMARY KEY, price REAL NOT NULL, kind TEXT NOT NULL);
        CREATE TABLE review (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                             item INTEGER NOT NULL REFERENCES item(id), written DATE NOT NULL, stars INTEGER NOT NULL);
        CREATE TABLE purchases (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                                item INTEGER NOT NULL REFERENCES item(id), placed DATE NOT NULL,
                                quantity INTEGER NOT NULL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                             placed DATE NOT NULL);
    """, """
        INSERT INTO customer VALUES (1, '2023-12-01', 1), (2, '2024-02-01', 0);
        INSERT INTO item VALUES (1, 2.0, 'tool');
        INSERT INTO review VALUES (1, 1, 1, '2024-01-05', 5);
        INSERT INTO purchases VALUES (1, 1, 1, '2024-01-02', 3);
        INSERT INTO orders VALUES (1, 1, '2024-01-02'), (2, 2, '2024-06-01'), (3, 2, '2024-08-01');
    """)
    builder = Builder(script, tmp_path)
    assert isinstance(builder.declarations["shop"], Database)
    assert sorted(builder.declarations) == ["bought", "leaves", "review", "shop"]
    records = builder.map(PLAN, "one").records()
    tasks = records["graphs"][0]["tasks"]
    assert list(zip([iso(day) for day in tasks["time"]], tasks["references"]["buyer"], tasks["split"],
                    tasks["target"])) == [
        ("2024-01-31", "1", "train", "leaves"), ("2024-03-01", "1", "train", "leaves"),
        ("2024-03-31", "1", "train", "leaves"), ("2024-06-01", "2", "train", "stays"),
        ("2024-07-01", "2", "train", "stays"), ("2024-07-31", "2", "train", "stays"),
        ("2024-08-30", "2", "train", "leaves"), ("2024-09-29", "2", "validation", "leaves"),
        ("2024-10-29", "2", "test", "leaves")]
    built = json.loads((builder.stage(tmp_path, "one", PLAN).parent / "one.staging" / "build.json").read_text())
    assert built["graphs"][0]["counts"]["tasks"] == dict(
        leaves=dict(splits=dict(train=7, validation=1, test=1), without_split=0))


def failing(folder, declarations, message):
    with pytest.raises(ValueError, match=message):
        Builder(declarations, folder).map(PLAN, "one")


def test_a_database_stops_naming_the_statement_the_field_and_the_table(tmp_path):
    shop(tmp_path)
    for fields, message in ((dict(colour=1), "database holds fields it does not know: colour"),
                            (dict(unit=None), "lacks unit"),
                            (dict(leave=["notes"]), "each column as <table>.<column>")):
        with pytest.raises(ValueError, match=message):
            stated(**fields)
    with pytest.raises(ValueError, match="lacks the required field absent"):
        database("shop.sqlite", reader="sqlite", tables=["*"], times=None, encoders={}, missing="omit")
    for fields, message in (
            (dict(tables=["nothing*"]), r"field tables names nothing\*, which no table"),
            (dict(tables=["city"]), r"foreign key city\(region\) names the table region, which the database does not"),
            (dict(encoders=dict(number="standard", category="onehot")), "states no encoder of the type flag"),
            (dict(leave=["customer.id"]), "names customer.id, a column of the primary key"),
            (dict(leave=["*.notes"]), r"names columns no table it reads holds: \*.notes"),
            (dict(times="clock"), "rule of the columns of time")):
        failing(tmp_path, dict(shop=stated(**fields)), message)


def test_a_statement_over_a_database_stops_naming_itself_and_the_table(tmp_path):
    shop(tmp_path)
    whole, shop_ = stated(), stated(tables=list(SUBSET))
    failing(tmp_path, dict(shop=shop_, lost=nodes(shop_.absent, key="id")), "names the table absent, which the database")
    failing(tmp_path, dict(shop=whole, half=edges(whole.review, source=dict(of=whole.customer, key="customer",
                                                                            absent="stop"),
                                                  target=dict(of=whole.customer, key="customer", absent="stop"),
                                                  directed=True, holds="always", known=None)),
            r"half maps the table review and uses not its foreign key review\(item\)")
    failing(tmp_path, dict(shop=shop_, viewed=tasks(shop_.orders, at="placed", references=dict(
        at=dict(sort="node", of=shop_.recent, key="id", absent="stop")))),
            "names the table recent of a database, which becomes no one mapping")
    failing(tmp_path, dict(shop=shop_, peeked=tasks(shop_.orders, at="placed", eligible="count(later.id) > 0",
                                                    references=dict(at=dict(sort="node", of=shop_.orders, key="id",
                                                                            absent="stop")))),
            "peeked field eligible reads later rows, which it may not read there")


def test_the_reader_proposes_a_type_for_each_declared_type_of_a_column():
    day = dict(type="time", unit="day", format="%Y-%m-%d")
    for declared, reader, expected in (
            ("INTEGER", "sqlite", "number"), ("numeric(10,2)", "postgres", "number"),
            ("double precision", "postgres", "number"), ("UNSIGNED BIG INT", "sqlite", "number"),
            ("character varying(20)", "postgres", "category"), ("NVARCHAR(8)", "sqlite", "category"),
            ("BOOLEAN", "sqlite", dict(type="flag", true=1, false=0)), ("boolean", "postgres", "flag"),
            ("DATE", "sqlite", day), ("timestamp without time zone", "postgres", dict(day, format="%Y-%m-%d %H:%M:%S")),
            ("integer[]", "postgres", None), ("point", "postgres", None), ("BLOB", "sqlite", None), ("", "sqlite", None)):
        assert typed(declared, reader, "day") == expected, declared


def test_a_database_stops_on_tables_its_rules_cannot_convert(tmp_path):
    for definitions, message in (
            ("CREATE TABLE trip (id INTEGER PRIMARY KEY, a DATE, b DATE, c DATE);", "holds the columns of time a, b, c"),
            ("CREATE TABLE log (at DATE, what TEXT);", "table log declares no primary key"),
            ("CREATE TABLE blob (id INTEGER PRIMARY KEY, data BLOB);", "blob.data declares the type 'BLOB'"),
            ("CREATE TABLE a (id INTEGER PRIMARY KEY); CREATE TABLE b (id INTEGER PRIMARY KEY);"
             "CREATE TABLE c (id INTEGER PRIMARY KEY);"
             "CREATE TABLE abc (a INTEGER REFERENCES a(id), b INTEGER REFERENCES b(id), c INTEGER REFERENCES c(id),"
             " PRIMARY KEY (a, b, c));", "abc has a key made of foreign keys, and holds 3")):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        shop(folder, definitions, "")
        failing(folder, dict(shop=stated()), message)
    blob = stated(leave=["blob.data"], times=None, through=None)
    built = Builder(dict(shop=blob, ask=tasks(blob.blob, references=dict(
        at=dict(sort="node", of=blob.blob, key="id", absent="stop")))), tmp_path / "2").map(PLAN, "one")
    assert built.records()["schema"]["nodes"] == dict(blob=dict(columns=[], holds=None))
