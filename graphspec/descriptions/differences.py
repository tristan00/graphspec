import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed
from graphspec.operations.difference import valued
from graphspec.operations.rows import counted as offsets
from graphspec.operations.rows import fact_keys
from graphspec.schema.declaration import Registry

from . import describe as truth

LOG = logging.getLogger(__name__)
SIDES = ("first", "second")
NODES = ("first", "second", "first_anchors", "second_anchors")
EDGES = ("first", "second", "valued")
STATED = dict(both="", first=", which only the first states", second=", which only the second states")
READING = (
    "first and second: at a time, what the first holds that the second does not, and what the second holds that the "
    "first does not, the entities of the two met by kind and key; a node is held where it is observed other than as "
    "an anchor; an anchor, a node observed only because a reference of a task names it, is counted apart; an edge is "
    "two ends a present fact of its relation joins, in either order for a relation without direction, met as the "
    "difference of two graphs meets them; other values: the edges both hold of which one holds a fact whose values "
    "the other holds no fact of, as that difference tells them under difference/match values; the times are those of "
    "either side, and at a time only one side shows the other holds nothing; only what differs is listed, time by "
    "time, and a kind or a relation equal at every time is counted"
)
TRUTH = (
    READING + "; each of the two graphs is read as it holds once every correction is known: a fact a correction "
    "replaced holds nothing at any time"
)


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


def spread(begun, ended):
    started = time.perf_counter()
    counts = ended - begun
    rows = np.repeat(np.arange(len(counts)), counts)
    result = rows, begun[rows] + offsets(counts)
    LOG.debug("differences spread exit facts=%d rows=%d seconds=%.4f", len(counts), len(rows), time.perf_counter() - started)
    return result


def lasting(registry, facts, moments):
    started = time.perf_counter()
    rows, steps = spread(*truth.shown(facts, "obs", moments))
    nodes = np.unique(np.stack((steps, np.asarray(facts.obs_entity, np.int64)[rows]), 1), axis=0)
    present = np.flatnonzero(np.asarray(facts.fact_present, bool))
    begun, ended = truth.shown(facts, "fact", moments)
    rows, steps = spread(begun[present], ended[present])
    rows = present[rows]
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
    moments = np.union1d(*[truth.clocked(registry, facts) for facts in loaded])
    sampled = moments if len(moments) else np.zeros(1, np.int64)
    times = moments.tolist() or [None]
    found = differed(*[lasting(registry, facts, sampled) for facts in loaded], times)
    result = dict(
        version=truth.VERSION,
        stored_sha256=manifest["stored_sha256"],
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


@timed("differences.spanned")
def spanned(times):
    return "no time" if times == [None] else "times %d from %d to %d" % (len(times), times[0], times[-1])


@timed("differences.clock")
def clock(moment):
    return "none" if moment is None else moment


def found_text(found):
    started = time.perf_counter()
    equal, lines = "equal at every time: kinds %d, relations %d" % (found["equal"]["kinds"], found["equal"]["relations"]), []
    for name, entry in found["kinds"].items():
        lines.append(
            "  kind %s%s: nodes only the first holds %d, only the second %d; anchors only the first holds %d, only the "
            "second %d" % (name, STATED[entry["stated"]], *[entry[label] for label in NODES])
        )
        lines += truth.table(
            ["time", "first nodes", "second nodes", "first anchors", "second anchors"],
            [[clock(held["time"])] + [held[label] for label in NODES] for held in entry["at"]],
        )
    for name, entry in found["relations"].items():
        lines.append(
            "  relation %s%s: edges only the first holds %d, only the second %d, held by both with other values %d"
            % (name, STATED[entry["stated"]], *[entry[label] for label in EDGES])
        )
        lines += truth.table(
            ["time", "first", "second", "other values"],
            [[clock(held["time"])] + [held[label] for label in EDGES] for held in entry["at"]],
        )
    lines.append("  %s" % equal if lines else "  nothing differs; %s" % equal)
    LOG.debug("differences found text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    lines = [
        "difference of the truth graphs %s, the first, and %s, the second, of the bank stored as %s, described at "
        "version %d" % (record["first"], record["second"], record["stored_sha256"], record["version"]),
        "%s, %s" % ("no unit" if record["unit"] is None else "unit " + record["unit"], spanned(record["times"])),
        TRUTH,
    ]
    lines += found_text(record)
    LOG.debug("differences text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def graph_text(graph):
    started = time.perf_counter()
    lines = [
        "",
        "graph %s, %s; times only the first shows: %s; times only the second shows: %s"
        % (
            graph["name"],
            spanned(graph["times"]),
            *[", ".join(map(str, graph["alone"][side])) or "none" for side in SIDES],
        ),
    ]
    lines += found_text(graph)
    LOG.debug("differences graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines
