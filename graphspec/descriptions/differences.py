import logging
import time
from pathlib import Path

import numpy as np

from graphspec.facts.stored import GROUPS
from graphspec.infra.timing import timed
from graphspec.operations.difference import valued
from graphspec.schema.declaration import Registry

from . import describe as truth

LOG = logging.getLogger(__name__)
SIDES = ("first", "second")
NODES = ("first", "second", "first_anchors", "second_anchors", "valued")
EDGES = ("first", "second", "valued")
STATED = dict(both="", first=", which only the first states", second=", which only the second states")
OWNED = (
    "what the first holds that the second does not, and what the second holds that the first does not, the entities "
    "of the two met by kind and key; a node is held where it is observed other than as an anchor; an anchor, a node "
    "observed only because a reference of a task names it, is counted apart; an edge is two ends a present fact of "
    "its relation joins, in either order for a relation without direction, met as the difference of two graphs meets "
    "them; other values: the edges both hold of which one holds a fact whose values the other holds no fact of, as "
    "that difference tells them under difference/match values, and the nodes both hold of which one holds values the "
    "other does not, bit for bit"
)
UNCOMPARED = "not compared: periods, tasks and their targets, options, the negatives drawn for a link"
VARIED = (
    "; the values of a node are those of the columns both sides hold for its kind, as the encoders of each side give "
    "them, and the columns only one side holds are named once, per kind and per relation both state; "
    + UNCOMPARED
    + ", and the graphs nodes hold"
)
PLACED = "first and second: at a %(one)s, " + OWNED
ROWS = "; a table shows at most %d %%(many)s, as rows: " % truth.SHOWN + truth.CHOSEN
NUMBERED = (
    "; in the record a count over the %(many)s is a list of runs, each the number of a step, the %(many)s of either "
    "side numbered together from 0 in their order, and the count held from it until the step of the next"
)
STEPPED = (
    PLACED + "; the %(many)s are those of either side, and at a %(one)s only one side shows the other holds nothing; "
    "only what differs is listed, %(one)s by %(one)s, and a kind or a relation equal at every %(one)s is counted"
    + VARIED
    + ROWS
    + truth.SEVERAL
    + NUMBERED
)
LAST = (
    "each of the two graphs is read as it holds once every correction is known, a fact a correction replaced "
    "holding nothing and an edge of a table that is about a set edge held as a view holds it, by the rule its table "
    "states for where the set edge it is about holds nothing"
)
HELD = (
    PLACED + "; " + LAST + "; the %(many)s are those of either graph, and before stands for before every one of "
    "them; at each a graph holds what a fact of it holds then, whether or not the %(one)s is one of its own, so what "
    "states no start is held before every %(one)s and on until it ends; only what differs is listed, %(one)s by "
    "%(one)s, and a kind or a relation equal before and at every %(one)s is counted"
)
STATED_VALUES = (
    "; the values of a node are those its facts state, the two graphs being of one bank and holding the same columns; "
    + UNCOMPARED
    + ", the members of set edges and the graphs nodes hold"
)
READING = dict(
    none="first and second: " + OWNED + "; the graphs hold no time and each side shows a graph at one step, where the "
    "two are compared once; only what differs is listed, and a kind or a relation equal there is counted" + VARIED,
    order=STEPPED % truth.NAMED["order"],
    clock=STEPPED % truth.NAMED["clock"],
)
READING["clocks"] = READING["clock"]
TRUTH = dict(
    none="first and second: " + OWNED + "; " + LAST + "; the graphs hold no time, so every fact holds always and the "
    "two are compared once; only what differs is listed, and a kind or a relation equal is counted" + STATED_VALUES,
    order=HELD % truth.NAMED["order"] + STATED_VALUES + (ROWS + truth.RUNS) % truth.NAMED["order"],
    clock=HELD % truth.NAMED["clock"] + STATED_VALUES + (ROWS + truth.RUNS) % truth.NAMED["clock"],
)
TRUTH["clocks"] = TRUTH["clock"]


@timed("differences.tokens")
def tokens(graph):
    return np.ascontiguousarray(np.asarray(graph.entity_token, np.uint64)).view(np.int64)


@timed("differences.numbered")
def numbered(*tables):
    inverse = np.unique(truth.packed(np.concatenate(tables).T), return_inverse=True)[1].reshape(-1)
    bounds = np.cumsum([0] + [len(held) for held in tables])
    return [inverse[low:high] for low, high in zip(bounds[:-1], bounds[1:])]


@timed("differences.distinct")
def distinct(registry, facts):
    return facts[np.unique(truth.met(registry, facts)[0], return_index=True)[1]]


@timed("differences.unmatched")
def unmatched(registry, first, second):
    mine, theirs = truth.met(registry, first, second)
    return ~np.isin(mine, theirs), ~np.isin(theirs, mine)


@timed("differences.compared")
def compared(first, second):
    keys = truth.packed(np.concatenate((first, second)).T)
    mine, theirs = keys[: len(first)], keys[len(first) :]
    return ~np.isin(mine, theirs), ~np.isin(theirs, mine)


@timed("differences.counted")
def counted(keys, steps, size):
    return np.bincount(keys[:, 0] * size + keys[:, 1], minlength=steps * size).reshape(steps, size)


def lasting(registry, facts, moments):
    started = time.perf_counter()
    seen = {group: truth.shown(facts, group, moments) for group in GROUPS}
    begun, ended = seen["obs"]
    stated = np.flatnonzero(ended > begun)
    rows, opened, closed = truth.ruled(registry, facts, moments, seen)
    ends = [np.asarray(facts.arrays[name], np.int64)[rows] for name in ("fact_code", "fact_source", "fact_target")]
    result = dict(
        registry=registry,
        kinds=np.asarray(facts.kinds, np.int64),
        tokens=tokens(facts),
        nodes=np.stack((np.asarray(facts.obs_entity, np.int64)[stated], begun[stated], ended[stated]), 1),
        anchored=np.zeros(len(stated), bool),
        values=np.asarray(facts.obs_values),
        valued=stated,
        facts=np.stack(ends + [opened, closed], 1),
        table=np.asarray(facts.fact_values),
        reference=rows,
    )
    LOG.debug(
        "differences lasting exit graph=%s nodes=%d facts=%d seconds=%.4f",
        facts.name,
        len(stated),
        len(rows),
        time.perf_counter() - started,
    )
    return result


def met(name, first, second):
    started = time.perf_counter()
    shown = [np.asarray(held, np.int64) for held in (first, second)]
    for side, held in zip(SIDES, shown):
        if len(np.unique(held)) != len(held):
            raise ValueError(
                "the %s shows the graph %s at two steps of one time, among %s; a difference meets the steps of the "
                "two by their time" % (side, name, ", ".join(map(str, held.tolist())))
            )
    times = np.union1d(*shown)
    result = times, [np.searchsorted(times, held) for held in shown]
    LOG.debug("differences met exit graph=%s times=%d seconds=%.4f", name, len(times), time.perf_counter() - started)
    return result


@timed("differences.stating")
def stating(holding):
    return "both" if all(holding) else SIDES[holding.index(True)]


def overlaid(first, second):
    started = time.perf_counter()
    (mine, since, until), (theirs, begun, ended) = first, second
    key, place = np.r_[mine, mine, theirs, theirs], np.r_[since, until, begun, ended]
    one, none = np.ones(len(mine), np.int64), np.zeros(2 * len(mine), np.int64)
    other, neither = np.ones(len(theirs), np.int64), np.zeros(2 * len(theirs), np.int64)
    order = np.lexsort((place, key))
    key, place = key[order], place[order]
    last = np.r_[(key[1:] != key[:-1]) | (place[1:] != place[:-1]), True][: len(key)]
    held, met = np.cumsum(np.r_[one, -one, neither][order])[last] > 0, np.cumsum(np.r_[none, other, -other][order])[last] > 0
    key, place = key[last], place[last]
    kept = np.r_[key[1:] == key[:-1], False][: len(key)] & (held | met)
    result = key[kept], place[kept], np.r_[place[1:], 0][: len(key)][kept], held[kept], met[kept]
    LOG.debug(
        "differences overlaid exit spans=%d pieces=%d seconds=%.4f",
        len(mine) + len(theirs),
        len(result[0]),
        time.perf_counter() - started,
    )
    return result


def parted(first, second):
    started = time.perf_counter()
    sides = (first, second)
    whole = [(key[whom], since, until) for (key, _, _, _), (whom, since, until) in zip(sides, [truth.runs(*held[:1], *held[2:]) for held in sides])]
    key, start, stop, mine, theirs = overlaid(*whole)
    width = max(int(value.max()) if len(value) else 0 for _, value, _, _ in sides) + 2
    pairs = [(key * width + value + 1, since, until) for key, value, since, until in sides]
    paired = [(pair[whom], since, until) for (pair, _, _), (whom, since, until) in zip(pairs, [truth.runs(*held) for held in pairs])]
    odd, begun, ended, ours, others = overlaid(*paired)
    unlike, both = ours != others, mine & theirs
    whom, begun, ended = truth.runs(odd[unlike] // width, begun[unlike], ended[unlike])
    valued_at = overlaid(((odd[unlike] // width)[whom], begun, ended), (key[both], start[both], stop[both]))
    told = valued_at[3] & valued_at[4]
    result = [
        (key[mine & ~theirs], start[mine & ~theirs], stop[mine & ~theirs]),
        (key[theirs & ~mine], start[theirs & ~mine], stop[theirs & ~mine]),
        (valued_at[0][told], valued_at[1][told], valued_at[2][told]),
    ]
    LOG.debug("differences parted exit keys=%d seconds=%.4f", len(key), time.perf_counter() - started)
    return result


def tallied(labels, spans, family, names, places):
    started = time.perf_counter()
    totals = [np.bincount(family[key], stop - start, len(names)).astype(np.int64).tolist() for key, start, stop in spans]
    held = [truth.counted(family[key], start, stop, len(names), places) for key, start, stop in spans]
    result = {
        name: dict(zip(labels, [found[at] for found in totals]), at=dict(zip(labels, [found[at] for found in held])))
        for at, name in enumerate(names)
    }
    LOG.debug("differences tallied exit names=%d places=%d seconds=%.4f", len(names), len(places), time.perf_counter() - started)
    return result


def kinded(sides, shared, family, kinds, places):
    started = time.perf_counter()
    values = valued(*[(side["values"], side["valued"]) for side in sides])
    held = [
        (number[side["nodes"][:, 0]], value, side["nodes"][:, 1], side["nodes"][:, 2])
        for side, number, value in zip(sides, shared, values)
    ]
    plain = parted(*[tuple(column[~side["anchored"]] for column in found) for found, side in zip(held, sides)])
    anchors = parted(*[tuple(column[side["anchored"]] for column in found) for found, side in zip(held, sides)])
    told = tallied(NODES, plain[:2] + anchors[:2] + plain[2:], family, kinds, places)
    declared = {name for side in sides for name in side["registry"].declared}
    observed = {kinds[at] for at in np.unique(np.r_[family[held[0][0]], family[held[1][0]]]).tolist()}
    found, equal = {}, 0
    for name in kinds:
        if name not in declared and name not in observed:
            continue
        if any(told[name][label] for label in NODES):
            found[name] = dict(stated=stating([name in side["registry"].nodes for side in sides]), **told[name])
        else:
            equal += 1
    LOG.debug("differences kinded exit differing=%d equal=%d seconds=%.4f", len(found), equal, time.perf_counter() - started)
    return found, equal


def joined(sides, shared, names):
    started = time.perf_counter()
    place = {name: at for at, name in enumerate(names)}
    holder = np.asarray([[name in side["registry"].by_name for side in sides].index(True) for name in names], np.int64)
    coded = np.asarray([sides[first]["registry"].code(name) for first, name in zip(holder.tolist(), names)], np.int64)
    named, tables = [], []
    for side, number in zip(sides, shared):
        own, held = side["registry"], side["facts"]
        index = np.full(own.relations, -1, np.int64)
        for code in np.unique(held[:, 0]).tolist():
            index[code] = place[own.by_code[code].name]
        named.append(index[held[:, 0]])
        tables.append(np.stack((np.zeros(len(held), np.int64), coded[named[-1]], number[held[:, 1]], number[held[:, 2]]), 1))
    keys = [np.zeros(len(table), np.int64) for table in tables]
    for first, side in enumerate(sides):
        chosen = [holder[found] == first for found in named]
        for key, mask, found in zip(keys, chosen, truth.met(side["registry"], *[table[mask] for table, mask in zip(tables, chosen)])):
            key[mask] = found * len(sides) + first
    edges = np.unique(np.concatenate(keys), return_inverse=True)[1].reshape(-1)
    family = np.zeros(int(edges.max()) + 1 if len(edges) else 0, np.int64)
    family[edges] = np.concatenate(named)
    LOG.debug("differences joined exit edges=%d seconds=%.4f", len(family), time.perf_counter() - started)
    return [edges[: len(keys[0])], edges[len(keys[0]) :]], family


def edged(sides, shared, places):
    started = time.perf_counter()
    values = valued(*[(side["table"], side["reference"]) for side in sides])
    stated = [
        name
        for side in sides
        for table in side["registry"].schema.edges
        for name in side["registry"].encoding.relations[table]
    ]
    holding = [side["registry"].by_code[code].name for side in sides for code in np.unique(side["facts"][:, 0]).tolist()]
    names = list(dict.fromkeys(stated + holding))
    for name in names:
        if len({side["registry"].by_name[name].symmetric for side in sides if name in side["registry"].by_name}) != 1:
            raise ValueError(
                "the relation %s has a direction on one side and none on the other; a difference meets the relations of "
                "the two by name" % name
            )
    edges, family = joined(sides, shared, names)
    held = [(key, value, side["facts"][:, 3], side["facts"][:, 4]) for key, value, side in zip(edges, values, sides)]
    told = tallied(EDGES, parted(*held), family, names, places)
    found, equal = {}, 0
    for name in names:
        if any(told[name][label] for label in EDGES):
            found[name] = dict(stated=stating([name in side["registry"].by_name for side in sides]), **told[name])
        else:
            equal += 1
    LOG.debug("differences edged exit differing=%d equal=%d seconds=%.4f", len(found), equal, time.perf_counter() - started)
    return found, equal


def differed(first, second, places):
    started = time.perf_counter()
    sides = (first, second)
    kinds = list(dict.fromkeys(name for side in sides for name in side["registry"].nodes))
    owned = [
        np.asarray([kinds.index(name) for name in side["registry"].nodes], np.int64)[side["kinds"]] for side in sides
    ]
    shared = numbered(*[np.stack((kind, side["tokens"]), 1) for kind, side in zip(owned, sides)])
    family = np.zeros(max([int(number.max()) + 1 for number in shared if len(number)] + [0]), np.int64)
    for kind, number in zip(owned, shared):
        family[number] = kind
    nodes, edges = kinded(sides, shared, family, kinds, places), edged(sides, shared, places)
    result = dict(kinds=nodes[0], relations=edges[0], equal=dict(kinds=nodes[1], relations=edges[1]))
    LOG.info(
        "differences differed exit places=%d kinds=%d relations=%d seconds=%.3f",
        len(places),
        len(nodes[0]),
        len(edges[0]),
        time.perf_counter() - started,
    )
    return result


@timed("differences.unequal")
def unequal(found):
    return bool(found["kinds"] or found["relations"])


def recorded(path, manifest, first, second):
    started = time.perf_counter()
    registry = Registry.from_declaration(manifest["spec"])
    loaded = [truth.stored(path, graph) for graph in (first, second)]
    if loaded[0].unit != loaded[1].unit:
        raise ValueError(
            "the graph %s counts time in %s and the graph %s in %s; times of graphs of different units are never "
            "compared" % (loaded[0].name, loaded[0].unit, loaded[1].name, loaded[1].unit)
        )
    if registry.schema.model == "order" and loaded[0].name != loaded[1].name:
        raise ValueError(
            "the graphs %s and %s state when their facts hold as positions, each in an order of its own; the positions "
            "of two graphs are never met, so under the model order a graph is compared only with itself"
            % (loaded[0].name, loaded[1].name)
        )
    moments = np.union1d(*[truth.clocked(registry, facts) for facts in loaded])
    times = [None] + moments.tolist()
    found = differed(*[lasting(registry, facts, truth.placed(moments)) for facts in loaded], times)
    result = dict(
        version=truth.VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=truth.modelled(registry),
        first=loaded[0].name,
        second=loaded[1].name,
        unit=loaded[0].unit,
        differs=unequal(found),
        times=times,
        **found,
    )
    LOG.info("differences recorded exit times=%d seconds=%.3f", len(times), time.perf_counter() - started)
    return result


@timed("differences.keyed")
def keyed(manifest, first, second, place):
    return dict(truth.keyed(manifest, place), first=dict(graph=first), second=dict(graph=second))


def differences(path, first, second):
    started = time.perf_counter()
    manifest = truth.manifested(path)
    named = {graph["name"]: graph for graph in manifest["graphs"]}
    lacking = [name for name in (first, second) if name not in named]
    if lacking:
        raise ValueError(
            "the bank %s holds no graph named %s; its graphs are %s" % (path, " or ".join(lacking), ", ".join(named))
        )
    place = truth.located(Path(path) / truth.FOLDER, "bank", "differences", manifest["stored_sha256"])
    record = truth.kept(place, keyed(manifest, first, second, place), lambda: recorded(path, manifest, named[first], named[second]), text)
    LOG.info("differences exit first=%s second=%s seconds=%.3f", first, second, time.perf_counter() - started)
    return record


def tabled(named, headers, labels, held, places, heads, kept):
    started = time.perf_counter()
    series = [held[label] for label in labels]
    chosen, total, changing = truth.picked(series, places, True)
    asked = [places[at] for at in chosen]
    rows = np.asarray([truth.counts(found, asked) for found in series], np.int64).T.tolist()
    if named is None:
        result = truth.table(headers, rows)
    else:
        result = truth.table([named["one"]] + headers, [[heads[at]] + row for at, row in zip(chosen, rows)])
    result += truth.omitted(len(chosen), total, changing, "rows", kept)
    LOG.debug("differences tabled exit rows=%d of=%d seconds=%.4f", len(chosen), total, time.perf_counter() - started)
    return result


def found_text(found, named, places, heads, kept):
    started = time.perf_counter()
    equal, lines = "equal%s: kinds %d, relations %d", []
    equal %= ("" if named is None else " at every " + named["one"], found["equal"]["kinds"], found["equal"]["relations"])
    for name, entry in found["kinds"].items():
        lines.append(
            "  kind %s%s: nodes only the first holds %d, only the second %d; anchors only the first holds %d, only the "
            "second %d; nodes both hold with other values %d"
            % (name, STATED[entry["stated"]], *[entry[label] for label in NODES])
        )
        headers = ["first nodes", "second nodes", "first anchors", "second anchors", "other values"]
        lines += tabled(named, headers, NODES, entry["at"], places, heads, kept)
    for name, entry in found["relations"].items():
        lines.append(
            "  relation %s%s: edges only the first holds %d, only the second %d, held by both with other values %d"
            % (name, STATED[entry["stated"]], *[entry[label] for label in EDGES])
        )
        lines += tabled(named, ["first", "second", "other values"], EDGES, entry["at"], places, heads, kept)
    lines.append("  %s" % equal if lines else "  nothing differs of what is compared; %s" % equal)
    LOG.debug("differences found text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    model = record["time"]["model"]
    named = truth.NAMED[model]
    lines = [
        "difference of the truth graphs %s, the first, and %s, the second, of the bank stored as %s, described at "
        "version %d" % (record["first"], record["second"], record["stored_sha256"], record["version"]),
        truth.kept_text(record),
        "%s, %s" % ("no unit" if record["unit"] is None else "unit " + record["unit"], truth.spanned(named, record["times"][1:])),
        TRUTH[model],
    ]
    heads = [truth.BEFORE if moment is None else moment for moment in record["times"]]
    lines += found_text(record, named, record["times"], heads, record["kept"])
    LOG.debug("differences text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def graph_text(graph, named, kept):
    started = time.perf_counter()
    head = "graph %s, %s, %s" % (graph["name"], truth.viewed_text(graph["view"]), truth.spanned(named, graph["times"]))
    if named is not None:
        alone = [truth.several(graph["alone"][side], named) or "none" for side in SIDES]
        head += "; %s only the first shows: %s; %s only the second shows: %s" % (named["many"], alone[0], named["many"], alone[1])
    lines = ["", head]
    lines += found_text(graph, named, list(range(len(graph["times"]))), graph["times"], kept)
    LOG.debug("differences graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines
