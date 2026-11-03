import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
START = datetime(2024, 1, 1)
EPOCH = datetime(1970, 1, 1)
MONDAY = date(1969, 12, 29)
SPLITS = ("train", "validation")


@timed("described_sources.csv")
def csv(folder, name, rows):
    names = list(rows[0])
    lines = [",".join(names)] + [",".join("" if row[key] is None else str(row[key]) for key in names) for row in rows]
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@timed("described_sources.documents")
def documents(folder, name, rows):
    (folder / name).write_text(json.dumps(rows), encoding="utf-8")


@timed("described_sources.records")
def records(folder, name, rows):
    (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@timed("described_sources.parquet")
def parquet(folder, name, rows):
    pq.write_table(pa.Table.from_pylist(rows), folder / name)


@timed("described_sources.moment")
def moment(minutes, form="%Y-%m-%d %H:%M"):
    return (START + timedelta(minutes=minutes)).strftime(form)


@timed("described_sources.day")
def day(days):
    return (START + timedelta(days=days)).strftime("%Y-%m-%d")


@timed("described_sources.counted")
def counted(minutes):
    return int((START - EPOCH).total_seconds() // 60) + minutes


@timed("described_sources.week")
def week(days):
    return ((START + timedelta(days=days)).date() - MONDAY).days // 7


@timed("described_sources.sensors")
def sensors(folder):
    names = ["s0", "s1"]
    pairs = [("s0", "s1", 100), ("s1", "s0", 900)]
    csv(folder, "distances.csv", [dict(origin=one, destination=other, metres=metres) for one, other, metres in pairs])
    with h5py.File(folder / "readings.h5", "w") as file:
        matrix = file.create_dataset(
            "speed", data=np.array([[30.0 + (row * 3 + place * 5) % 11 for place in range(2)] for row in range(24)]))
        rows = file.create_dataset("minutes", data=np.arange(24) * 15)
        headers = file.create_dataset("sensors", data=np.array([name.encode() for name in names]))
        rows.make_scale("minutes")
        headers.make_scale("sensors")
        matrix.dims[0].attach_scale(rows)
        matrix.dims[1].attach_scale(headers)


@timed("described_sources.interactions")
def interactions(folder):
    rows = [dict(user="u%d" % (at % 4), item="i%d" % (at * 7 % 20), second=10 * at) for at in range(24)]
    csv(folder, "interactions.csv", rows)


@timed("described_sources.transfers")
def transfers(folder):
    csv(folder, "transfers.csv", [
        dict(id="t%d" % at, sender="a%d" % (at % 5), receiver="a%d" % ((at + 1 + at % 3) % 5), day=day(at // 2),
             amount=10 + at, laundering="true" if at % 4 == 0 else "false")
        for at in range(30)])


@timed("described_sources.papers")
def papers(folder):
    parquet(folder, "papers.parquet", [
        dict(id="p%d" % at, year=2014 + at // 4, words=[float((at * place) % 5) for place in range(16)],
             venue="v%d" % (at % 3))
        for at in range(24)])
    parquet(folder, "citations.parquet", [
        dict(citing="p%d" % at, cited="p%d" % (at - back)) for at in range(3, 24) for back in (1, 3)])


@timed("described_sources.triples")
def triples(folder):
    facts = [dict(id="f%d" % at, head="e%d" % (at % 10), about=None, relation="r%d" % (at % 3),
                  tail="e%d" % ((at + 1 + at % 4) % 10)) for at in range(30)]
    qualifiers = [dict(id="q%d" % at, head=None, about="f%d" % at, relation="since", tail="e%d" % at)
                  for at in range(6)]
    csv(folder, "triples.csv", facts + qualifiers)


@timed("described_sources.molecules")
def molecules(folder):
    csv(folder, "atoms.csv", [dict(molecule="m%d" % graph, element="CNO"[(graph + place) % 3], charge=(place - 1) * 0.5)
                              for graph in range(4) for place in range(3)])
    csv(folder, "bonds.csv", [dict(one=3 * graph + first, other=3 * graph + first + 1, order=1 + first)
                              for graph in range(4) for first in range(2)])
    csv(folder, "molecules.csv", [dict(molecule="m%d" % graph, mutagenic="yes" if graph % 3 else "no")
                                  for graph in range(4)])


@timed("described_sources.events")
def events(folder):
    csv(folder, "objects.csv", [dict(id="o%d" % at, sort=("box", "cart")[at % 2], weight=None if at == 3 else at * 1.5)
                                for at in range(6)])
    csv(folder, "events.csv", [dict(id="e%d" % at, activity=("load", "move", "drop")[at % 3], minute=10 * at)
                               for at in range(12)])
    csv(folder, "touches.csv", [dict(event="e%d" % at, object="o%d" % ((at + shift) % 6), role=role)
                                for at in range(12) for shift, role in ((0, "source"), (3, "target"))])


@timed("described_sources.visits")
def visits(folder):
    stays = [dict(patient="p%d" % patient, visit="v%d_%d" % (patient, number), number=number,
                  ward=("a", "b")[(patient + number) % 2]) for patient in range(6) for number in range(1, 4)]
    csv(folder, "visits.csv", stays)
    csv(folder, "diagnoses.csv", [dict(visit=row["visit"], code="c%d" % (3 + (at + shift) % 5))
                                  for at, row in enumerate(stays) for shift in (0, 2)])
    parents = [("c0", None), ("c1", "c0"), ("c2", "c0"), ("c3", "c1"), ("c4", "c1"), ("c5", "c2"), ("c6", "c2"),
               ("c7", "c2")]
    csv(folder, "parents.csv", [dict(code=code, parent=parent) for code, parent in parents])


@timed("described_sources.trees")
def trees(folder):
    leaf = dict(form="call", leaf=True, children=[])
    root = dict(form="block", leaf=False, children=[leaf, dict(form="loop", leaf=False, children=[leaf])])
    documents(folder, "programs.json", [
        dict(program="g%d" % at, root=root, name=["get", "x", "y"][: 1 + at % 3]) for at in (0, 7)])
    csv(folder, "flows.csv", [dict(program="g%d" % at, written="0.0", read="0.1.0") for at in (0, 7)])
    csv(folder, "tokens.csv", [dict(program="g%d" % at, before="0.0", after="0.1") for at in (0, 7)])


@timed("described_sources.coauthors")
def coauthors(folder):
    documents(folder, "papers.json", [
        dict(id="w%d_%d" % (lab, year), lab="l%d" % lab, year=2010 + year,
             authors=["a%d" % ((lab + year + place) % 8) for place in range(2)])
        for lab in range(3) for year in range(10)])


@timed("described_sources.game")
def game(folder):
    sightings, orders = [], []
    for graph in (0, 3):
        name = "g%d" % graph
        for turn in range(1, 4):
            for unit, owner in (("u0", "a"), ("u1", "a"), ("u2", "b"), ("u3", "b")):
                sightings.append(dict(game=name, turn=turn, unit=unit, owner=owner, strength=float(turn + graph % 3),
                                      watchers=[owner] if unit in ("u0", "u3") else ["a", "b"]))
            for player, unit, enemy in (("a", "u0", "u2"), ("b", "u3", "u1")):
                strike = (turn + graph) % 2 == 0
                orders.append(dict(game=name, turn=turn, player=player, unit=unit,
                                   move="strike" if strike else "hold", onto=enemy if strike else None,
                                   force=float(turn) if strike else None, gain=float((turn * 3 + graph) % 5)))
    parquet(folder, "sightings.parquet", sightings)
    parquet(folder, "orders.parquet", orders)


@timed("described_sources.clicks")
def clicks(folder):
    csv(folder, "clicks.csv", [
        dict(id="k%d_%d" % (user, at), user="u%d" % user, page=("home", "item", "cart")[(user + at) % 3],
             minute=moment(user * 7 + at * (10 if at != 2 else 50) + (40 if at > 2 else 0)))
        for user in range(6) for at in range(4)])


@timed("described_sources.trips")
def trips(folder):
    csv(folder, "segments.csv", [dict(id="r%d" % at, origin="j%d" % at, destination="j%d" % ((at + 1) % 6),
                                      metres=100 + 30 * at) for at in range(6)])
    documents(folder, "trips.json", [
        dict(trip="t%d" % at, left=20 * at, route=["r%d" % ((at + step) % 6) for step in range(1 + at % 3)],
             took=10 + at % 7)
        for at in range(24)])


@timed("described_sources.machines")
def machines(folder):
    csv(folder, "machines.csv", [dict(machine="m%d" % at, model=("x", "y")[at % 2]) for at in range(8)])
    parquet(folder, "logs.parquet", [
        dict(machine="m%d" % at, day=19700 + entry, heat=None if entry == 2 else 50.0 + at + entry,
             shake=float(entry % 3), failed=at % 2 == 0 and entry == 2)
        for at in range(8) for entry in range(3)])


@timed("described_sources.drugs")
def drugs(folder):
    csv(folder, "atoms.csv", [dict(drug="d%d" % drug, atom="a%d_%d" % (drug, at), element="CNO"[(drug + at) % 3])
                              for drug in range(6) for at in range(3)])
    csv(folder, "bonds.csv", [dict(drug="d%d" % drug, one="a%d_%d" % (drug, at), other="a%d_%d" % (drug, at + 1),
                                   order=1 + at) for drug in range(6) for at in range(2)])
    csv(folder, "pairs.csv", [dict(one="d%d" % one, other="d%d" % other, interact="true" if (one + other) % 2 else "false")
                              for one in range(6) for other in range(one + 1, 6)])


@timed("described_sources.figures")
def figures(folder):
    rows = []
    for firm in range(4):
        for quarter in range(6):
            closes = 90 * quarter + firm
            rows.append(dict(firm="f%d" % firm, quarter="q%d" % quarter, closes=day(closes), reported=day(closes + 20),
                             sales=100.0 + 10 * firm + quarter))
            if (firm + quarter) % 2 == 0:
                rows.append(dict(firm="f%d" % firm, quarter="q%d" % quarter, closes=day(closes),
                                 reported=day(closes + 50), sales=101.0 + 10 * firm + quarter))
    csv(folder, "figures.csv", rows)


@timed("described_sources.impressions")
def impressions(folder):
    csv(folder, "impressions.csv", [
        dict(impression="m%d" % at, second=10 * at, policy="p%d" % (at % 2), position=position,
             item="i%d" % ((at + position) % 10), chance=0.2 + 0.1 * position,
             click="true" if (at + position) % 4 == 0 else "false")
        for at in range(12) for position in range(3)])


@timed("described_sources.dispatches")
def dispatches(folder):
    csv(folder, "depots.csv", [dict(depot="d%d" % depot, day=at, stock=10.0 + depot + at)
                               for depot in range(5) for at in range(6)])
    csv(folder, "carriers.csv", [dict(carrier="c%d" % at, capacity=5.0 + at) for at in range(3)])
    csv(folder, "shipments.csv", [
        dict(day=at, carrier="c%d" % carrier, origin="d%d" % ((at + carrier) % 5),
             destination="d%d" % ((at + carrier + 1 + leg) % 5), load=1.0 + leg, profit=float((at * carrier + leg) % 7))
        for at in range(6) for carrier in range(3) for leg in range(1 + (at + carrier) % 2)])


@timed("described_sources.database")
def database(folder):
    customers = ["(%d, '%s', %d)" % (at, day(at * 9 - 30), at % 2) for at in range(1, 4)]
    orders = ["(%d, %d, '%s')" % (at, 1 + at % 3, day(at * 37 % 420)) for at in range(1, 24)]
    purchases = ["(%d, %d, %d, '%s', %d)" % (at, 1 + at % 3, 1 + at % 3, day(at * 41 % 420), 1 + at % 4)
                 for at in range(1, 12)]
    reviews = ["(%d, %d, %d, '%s', %d)" % (at, 1 + at % 3, 1 + at % 3, day(at * 53 % 420), 1 + at % 5)
               for at in range(1, 6)]
    with sqlite3.connect(folder / "shop.sqlite") as connection:
        connection.executescript("""
            CREATE TABLE customer (id INTEGER PRIMARY KEY, joined DATE NOT NULL, vip BOOLEAN NOT NULL);
            CREATE TABLE item (id INTEGER PRIMARY KEY, price REAL NOT NULL, kind TEXT NOT NULL);
            CREATE TABLE review (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                                 item INTEGER NOT NULL REFERENCES item(id), written DATE NOT NULL, stars INTEGER NOT NULL);
            CREATE TABLE purchases (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                                    item INTEGER NOT NULL REFERENCES item(id), placed DATE NOT NULL,
                                    quantity INTEGER NOT NULL);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer INTEGER NOT NULL REFERENCES customer(id),
                                 placed DATE NOT NULL);
            INSERT INTO customer VALUES %s;
            INSERT INTO item VALUES (1, 2.0, 'tool'), (2, 4.0, 'toy'), (3, 1.5, 'tool');
            INSERT INTO orders VALUES %s;
            INSERT INTO purchases VALUES %s;
            INSERT INTO review VALUES %s;
        """ % (", ".join(customers), ", ".join(orders), ", ".join(purchases), ", ".join(reviews)))
    connection.close()


@timed("described_sources.warehouse")
def warehouse(folder):
    orders, lines = [], []
    for at in range(16):
        placed = at * 4 * 1440 + 540
        orders.append(dict(number="o%d" % at, customer="c%d" % (at % 5), placed=moment(placed),
                           shipped=None if at >= 14 else moment(placed + (1 + at % 4) * 1440)))
        for place in range(2):
            lines.append(dict(order="o%d" % at, product="p%d" % ((at + place) % 6), quantity=1 + place,
                              shelf="s%d" % ((at + place) % 4), worker="w%d" % (at % 3),
                              picked=moment(placed + 120 + 10 * place)))
    csv(folder, "orders.csv", orders)
    csv(folder, "lines.csv", lines)
    csv(folder, "shelves.csv", [dict(shelf="s%d" % at, aisle="AB"[at % 2], height=1.0 + at) for at in range(4)])
    csv(folder, "workers.csv", [dict(worker="w%d" % at, shift=("day", "night")[at % 2], started="2023-12-01")
                                for at in range(3)])
    csv(folder, "products.csv", [dict(product="p%d" % at, weight=None if at == 2 else 0.5 * at,
                                      category=("food", "tool")[at % 2]) for at in range(6)])


@timed("described_sources.league")
def league(folder):
    matches, lineups, substitutions, ratings = [], [], [], []
    for at in range(8):
        home, away = at % 4, (at + 1 + at % 3) % 4
        matches.append(dict(match="m%d" % at, home="t%d" % home, away="t%d" % away,
                            kickoff=counted(at * 4 * 1440 + 900), home_goals=float(at % 3), away_goals=float(at % 2),
                            season="2023" if at < 6 else "2024"))
        for team in (home, away):
            for place in range(2):
                lineups.append(dict(match="m%d" % at, player="t%dp%d" % (team, place), team="t%d" % team,
                                    position=("keeper", "forward")[place], started=place == 0))
        substitutions.append(dict(match="m%d" % at, minute=60.0, off="t%dp1" % home, on="t%dp0" % home))
    for team in range(4):
        for place in range(2):
            for weeks in range(5):
                about = week(weeks * 7)
                ratings.append(dict(player="t%dp%d" % (team, place), week=about, published=about,
                                    rating=5.0 + (team + place + weeks) % 4))
                if weeks % 3 == 0:
                    ratings.append(dict(player="t%dp%d" % (team, place), week=about, published=about + 1,
                                        rating=5.5 + (team + place + weeks) % 4))
    parquet(folder, "matches.parquet", matches)
    parquet(folder, "lineups.parquet", lineups)
    parquet(folder, "substitutions.parquet", substitutions)
    parquet(folder, "ratings.parquet", ratings)


@timed("described_sources.dispatch")
def dispatch(folder):
    form = "%Y-%m-%dT%H:%M:%S"
    records(folder, "calls.jsonl", [
        dict(call="c%d" % at, time=moment(480 + (at // 2) * 10, form), place=[float(at % 5), float(at % 3)],
             priority=float(1 + at % 3), dispatcher="k%d" % (at % 2),
             free=[dict(vehicle="v%d" % ((at + place) % 5), station="s%d" % ((at + place) % 5 % 3), sent=place == 0)
                   for place in range(3)])
        for at in range(20)])
    csv(folder, "vehicles.csv", [dict(vehicle="v%d" % at, type=("van", "car")[at % 2], station="s%d" % (at % 3))
                                 for at in range(5)])
    csv(folder, "stations.csv", [dict(station="s%d" % at, place="%d %d" % (at, 2 * at)) for at in range(3)])
    csv(folder, "arrivals.csv", [dict(call="c%d" % at, minutes=5 + at % 7) for at in range(20)])


@timed("described_sources.grid")
def grid(folder):
    regions = ["north", "south", "coast", "east", "south", "east"]
    csv(folder, "substations.csv", [dict(name="n%d" % at, region=region, voltage=("high", "medium")[at % 2])
                                    for at, region in enumerate(regions)])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    csv(folder, "lines.csv", [
        dict(line="l%d" % at, origin="n%d" % one, destination="n%d" % other, length=10.0 + at, capacity=100.0 + 5 * at,
             entered="2024-01-01", retired="2024-01-07" if at == 4 else None)
        for at, (one, other) in enumerate(pairs)])
    parquet(folder, "load.parquet", [
        {"time": counted(at * 360), **{"n%d" % place: 10.0 + (at + place) % 5 for place in range(6)}}
        for at in range(30)])
    csv(folder, "outages.csv", [
        dict(line="l%d" % (0, 3)[at % 2], began=moment(at * 1600 + 60), ended=moment(at * 1600 + 300),
             cause=("storm", "fault")[at % 2], recorded=day(at * 1600 // 1440 + 2))
        for at in range(8)])


@timed("described_sources.tutoring")
def tutoring(folder):
    form = "%Y-%m-%dT%H:%M:%S"
    sessions = []
    for at in range(24):
        start = at * 2 * 1440 + 600
        steps = []
        for place in range(3):
            given = ("short", "long", "none")[(at + place) % 3]
            steps.append(dict(exercise="e%d" % ((at + place) % 5), given=given, correct=(at + place) % 2 == 0,
                              answered=moment(start + place + 1, form),
                              hints=[dict(type="short", words=5.0), dict(type="long", words=20.0)]))
        sessions.append(dict(student="s%d" % (at % 4), start=moment(start, form), steps=steps))
    records(folder, "sessions.jsonl", sessions)
    csv(folder, "exercises.csv", [dict(exercise="e%d" % at, topic="t%d" % (at % 3), difficulty=1.0 + at)
                                  for at in range(5)])
    csv(folder, "prerequisites.csv", [dict(topic="t%d" % at, needed="t%d" % (at - 1)) for at in range(1, 3)])


WRITERS = {
    "01_sensors.py": (sensors, SPLITS),
    "02_interactions.py": (interactions, SPLITS),
    "03_transfers.py": (transfers, SPLITS),
    "04_papers.py": (papers, SPLITS),
    "05_triples.py": (triples, SPLITS),
    "06_molecules.py": (molecules, SPLITS),
    "07_events.py": (events, SPLITS),
    "08_visits.py": (visits, SPLITS),
    "09_trees.py": (trees, SPLITS),
    "10_coauthors.py": (coauthors, SPLITS),
    "11_game.py": (game, SPLITS),
    "12_clicks.py": (clicks, SPLITS),
    "13_trips.py": (trips, SPLITS),
    "14_machines.py": (machines, SPLITS),
    "15_drugs.py": (drugs, SPLITS),
    "16_figures.py": (figures, SPLITS),
    "17_impressions.py": (impressions, SPLITS),
    "18_dispatches.py": (dispatches, SPLITS),
    "19_database.py": (database, SPLITS),
    "20_warehouse.py": (warehouse, ("train", "score")),
    "21_league.py": (league, ("2023", "2024")),
    "22_dispatch.py": (dispatch, ("train", "score")),
    "23_grid.py": (grid, ("train", "score")),
    "24_tutoring.py": (tutoring, ("train", "score")),
}


def written(name, folder):
    started = time.perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    WRITERS[name][0](folder)
    seconds = time.perf_counter() - started
    LOG.debug("written exit script=%s seconds=%.4f", name, seconds)
    return seconds
