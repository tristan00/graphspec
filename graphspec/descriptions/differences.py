import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed
from graphspec.operations.difference import valued
from graphspec.operations.rows import fact_keys
from graphspec.schema.declaration import Registry

from . import describe as truth

LOG = logging.getLogger(__name__)
SIDES = ("first", "second")
NODES = ("first", "second", "first_anchors", "second_anchors")
EDGES = ("first", "second", "valued")
STATED = dict(both="", first=", which only the first states", second=", which only the second states")
OWNED = (
    "what the first holds that the second does not, and what the second holds that the first does not, the entities "
    "of the two met by kind and key; a node is held where it is observed other than as an anchor; an anchor, a node "
    "observed only because a reference of a task names it, is counted apart; an edge is two ends a present fact of "
    "its relation joins, in either order for a relation without direction, met as the difference of two graphs meets "
    "them; other values: the edges both hold of which one holds a fact whose values the other holds no fact of, as "
    "that difference tells them under difference/match values"
)
PLACED = "first and second: at a %(one)s, " + OWNED
STEPPED = (
    PLACED + "; the %(many)s are those of either side, and at a %(one)s only one side shows the other holds nothing; "
    "only what differs is listed, %(one)s by %(one)s, and a kind or a relation equal at every %(one)s is counted"
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
READING = dict(
    none="first and second: " + OWNED + "; the graphs hold no time and each side shows a graph at one step, where the "
    "two are compared once; only what differs is listed, and a kind or a relation equal there is counted",
    order=STEPPED % truth.NAMED["order"],
    clock=STEPPED % truth.NAMED["clock"],
)
READING["clocks"] = READING["clock"]
TRUTH = dict(
    none="first and second: " + OWNED + "; " + LAST + "; the graphs hold no time, so every fact holds always and the "
    "two are compared once; only what differs is listed, and a kind or a relation equal is counted",
    order=HELD % truth.NAMED["order"],
    clock=HELD % truth.NAMED["clock"],
)
TRUTH["clocks"] = TRUTH["clock"]


@timed("differences.tokens")
def tokens(graph):
    return np.ascontiguousarray(np.asarray(graph.entity_token, np.uint64)).view(np.int64)


@timed("differences.numbered")
def numbered(*tables):
    inverse = np.unique(np.concatenate(tables), axis=0, return_inverse=True)[1].reshape(-1)
    bounds = np.cumsum([0] + [len(held) for held in tables])
    return [inverse[low:high] for low, high in zip(bounds[:-1], bounds[1:])]


@timed("differences.distinct")
def distinct(registry, facts):
    return facts[np.unique(fact_keys(registry, facts.T)[0], return_index=True)[1]]


@timed("differences.unmatched")
def unmatched(registry, first, second):
    mine, theirs = fact_keys(registry, first.T, second.T)
    return ~np.isin(mine, theirs), ~np.isin(theirs, mine)


@timed("differences.compared")
def compared(first, second):
    _, inverse = np.unique(np.concatenate((first, second)), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    mine, theirs = inverse[: len(first)], inverse[len(first) :]
    return ~np.isin(mine, theirs), ~np.isin(theirs, mine)


@timed("differences.counted")
def counted(keys, steps, size):
    return np.bincount(keys[:, 0] * size + keys[:, 1], minlength=steps * size).reshape(steps, size)


def lasting(registry, facts, moments):
    started = time.perf_counter()
    rows, steps = truth.spread(*truth.shown(facts, "obs", moments))
    nodes = np.unique(np.stack((steps, np.asarray(facts.obs_entity, np.int64)[rows]), 1), axis=0)
    present = np.flatnonzero(np.asarray(facts.fact_present, bool))
    begun, ended = truth.shown(facts, "fact", moments)
    rows, steps = truth.spread(begun[present], ended[present])
    rows = present[rows]
    kept = truth.lapsing(registry, facts, nodes[:, 0] * len(facts.kinds) + nodes[:, 1], rows, steps, moments)
    rows, steps = rows[kept], steps[kept]
    ends = [np.asarray(facts.arrays[name], np.int64)[rows] for name in ("fact_code", "fact_source", "fact_target")]
    result = dict(
        registry=registry,
        kinds=np.asarray(facts.kinds, np.int64),
        tokens=tokens(facts),
        nodes=nodes,
        anchored=np.zeros(len(nodes), bool),
        facts=np.stack([steps] + ends, 1),
        table=np.asarray(facts.fact_values),
        reference=rows,
    )
    LOG.debug(
        "differences lasting exit graph=%s nodes=%d facts=%d seconds=%.4f",
        facts.name,
        len(nodes),
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


def tallied(labels, counts, times):
    started = time.perf_counter()
    counts = np.asarray(counts, np.int64)
    result = dict(
        zip(labels, counts.sum(1).tolist()),
        at=[
            dict(time=times[at], **dict(zip(labels, counts[:, at].tolist())))
            for at in np.flatnonzero(counts.any(0)).tolist()
        ],
    )
    LOG.debug("differences tallied exit times=%d seconds=%.4f", len(result["at"]), time.perf_counter() - started)
    return result


def kinded(sides, numbers, kinds, times):
    started = time.perf_counter()
    keys = [number[side["nodes"][:, 1]] for side, number in zip(sides, numbers)]
    keys = [np.stack((side["nodes"][:, 0], held[:, 0], held[:, 1]), 1) for side, held in zip(sides, keys)]
    counts = []
    for anchored in (False, True):
        held = [key[side["anchored"] == anchored] for key, side in zip(keys, sides)]
        counts += [counted(table[alone], len(times), len(kinds)) for table, alone in zip(held, compared(*held))]
    declared = {name for side in sides for name in side["registry"].declared}
    observed = set(np.unique(np.concatenate(keys)[:, 1]).tolist())
    found, equal = {}, 0
    for at, name in enumerate(kinds):
        if name not in declared and at not in observed:
            continue
        entry = tallied(NODES, [held[:, at] for held in counts], times)
        if entry["at"]:
            found[name] = dict(stated=stating([name in side["registry"].nodes for side in sides]), **entry)
        else:
            equal += 1
    LOG.debug("differences kinded exit differing=%d equal=%d seconds=%.4f", len(found), equal, time.perf_counter() - started)
    return found, equal


def related(name, sides, facts, times):
    started = time.perf_counter()
    holders = [side["registry"] for side in sides if name in side["registry"].by_name]
    if len({registry.by_name[name].symmetric for registry in holders}) != 1:
        raise ValueError(
            "the relation %s has a direction on one side and none on the other; a difference meets the relations of "
            "the two by name" % name
        )
    registry = holders[0]
    rows = []
    for side, held in zip(sides, facts):
        own = side["registry"]
        chosen = held[held[:, 1] == own.code(name)] if name in own.by_name else held[:0]
        code = np.full((len(chosen), 1), registry.code(name), np.int64)
        rows.append(np.concatenate((chosen[:, :1], code, chosen[:, 2:]), 1))
    keyed = [held[:, :4] for held in rows]
    alone, unlike = unmatched(registry, *keyed), unmatched(registry, *rows)
    edges = [distinct(registry, held[apart]) for held, apart in zip(keyed, alone)]
    both = [held[~apart & other] for held, apart, other in zip(keyed, alone, unlike)]
    edges.append(distinct(registry, np.concatenate(both)))
    result = tallied(EDGES, [np.bincount(held[:, 0], minlength=len(times)) for held in edges], times)
    LOG.debug("differences related exit relation=%s times=%d seconds=%.4f", name, len(result["at"]), time.perf_counter() - started)
    return result


def edged(sides, numbers, times):
    started = time.perf_counter()
    values = valued(*[(side["table"], side["reference"]) for side in sides])
    facts = [
        np.concatenate((side["facts"][:, :2], number[side["facts"][:, 2:], 2], value[:, None]), 1)
        for side, number, value in zip(sides, numbers, values)
    ]
    stated = [
        name
        for side in sides
        for table in side["registry"].schema.edges
        for name in side["registry"].encoding.relations[table]
    ]
    holding = [
        side["registry"].by_code[code].name
        for side, held in zip(sides, facts)
        for code in np.unique(held[:, 1]).tolist()
    ]
    found, equal = {}, 0
    for name in dict.fromkeys(stated + holding):
        entry = related(name, sides, facts, times)
        if entry["at"]:
            found[name] = dict(stated=stating([name in side["registry"].by_name for side in sides]), **entry)
        else:
            equal += 1
    LOG.debug("differences edged exit differing=%d equal=%d seconds=%.4f", len(found), equal, time.perf_counter() - started)
    return found, equal


def differed(first, second, times):
    started = time.perf_counter()
    sides = (first, second)
    kinds = list(dict.fromkeys(name for side in sides for name in side["registry"].nodes))
    owned = [
        np.asarray([kinds.index(name) for name in side["registry"].nodes], np.int64)[side["kinds"]] for side in sides
    ]
    shared = numbered(*[np.stack((kind, side["tokens"]), 1) for kind, side in zip(owned, sides)])
    numbers = [np.stack((kind, side["tokens"], number), 1) for kind, side, number in zip(owned, sides, shared)]
    nodes, edges = kinded(sides, numbers, kinds, times), edged(sides, numbers, times)
    result = dict(kinds=nodes[0], relations=edges[0], equal=dict(kinds=nodes[1], relations=edges[1]))
    LOG.info(
        "differences differed exit times=%d kinds=%d relations=%d seconds=%.3f",
        len(times),
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


def differences(path, first, second):
    started = time.perf_counter()
    manifest = truth.manifested(path)
    named = {graph["name"]: graph for graph in manifest["graphs"]}
    lacking = [name for name in (first, second) if name not in named]
    if lacking:
        raise KeyError(
            "the bank %s holds no graph named %s; its graphs are %s" % (path, " or ".join(lacking), ", ".join(named))
        )
    key = sha256_json(
        dict(stored=manifest["stored_sha256"], version=truth.VERSION, first=dict(graph=first), second=dict(graph=second))
    )
    held = Path(path) / "descriptions" / "differences" / key
    record = truth.kept(held, lambda: recorded(path, manifest, named[first], named[second]), text)
    LOG.info("differences exit first=%s second=%s seconds=%.3f", first, second, time.perf_counter() - started)
    return record


def tabled(named, headers, labels, rows):
    started = time.perf_counter()
    counts = [[held[label] for label in labels] for held in rows]
    if named is None:
        result = truth.table(headers, counts)
    else:
        placed = [[truth.BEFORE if held["time"] is None else held["time"]] + found for held, found in zip(rows, counts)]
        result = truth.table([named["one"]] + headers, placed)
    LOG.debug("differences tabled exit rows=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return result


def found_text(found, named):
    started = time.perf_counter()
    equal, lines = "equal%s: kinds %d, relations %d", []
    equal %= ("" if named is None else " at every " + named["one"], found["equal"]["kinds"], found["equal"]["relations"])
    for name, entry in found["kinds"].items():
        lines.append(
            "  kind %s%s: nodes only the first holds %d, only the second %d; anchors only the first holds %d, only the "
            "second %d" % (name, STATED[entry["stated"]], *[entry[label] for label in NODES])
        )
        lines += tabled(named, ["first nodes", "second nodes", "first anchors", "second anchors"], NODES, entry["at"])
    for name, entry in found["relations"].items():
        lines.append(
            "  relation %s%s: edges only the first holds %d, only the second %d, held by both with other values %d"
            % (name, STATED[entry["stated"]], *[entry[label] for label in EDGES])
        )
        lines += tabled(named, ["first", "second", "other values"], EDGES, entry["at"])
    lines.append("  %s" % equal if lines else "  nothing differs; %s" % equal)
    LOG.debug("differences found text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    model = record["time"]["model"]
    named = truth.NAMED[model]
    lines = [
        "difference of the truth graphs %s, the first, and %s, the second, of the bank stored as %s, described at "
        "version %d" % (record["first"], record["second"], record["stored_sha256"], record["version"]),
        "%s, %s" % ("no unit" if record["unit"] is None else "unit " + record["unit"], truth.spanned(named, record["times"][1:])),
        TRUTH[model],
    ]
    lines += found_text(record, named)
    LOG.debug("differences text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def graph_text(graph, named):
    started = time.perf_counter()
    head = "graph %s, %s, %s" % (graph["name"], truth.viewed_text(graph["view"]), truth.spanned(named, graph["times"]))
    if named is not None:
        alone = [", ".join(map(str, graph["alone"][side])) or "none" for side in SIDES]
        head += "; %s only the first shows: %s; %s only the second shows: %s" % (named["many"], alone[0], named["many"], alone[1])
    lines = ["", head]
    lines += found_text(graph, named)
    LOG.debug("differences graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines
