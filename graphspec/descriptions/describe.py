import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from graphspec import registries
from graphspec.engine import transform
from graphspec.facts.stored import GROUPS, Facts, admitted
from graphspec.infra.files import parsed, read_json, sha256_json, write_json, write_record
from graphspec.infra.runtime import cache_path, output_path, refusal
from graphspec.infra.timing import timed
from graphspec.operations.operation import OPERATIONS
from graphspec.operations.rows import counted as offsets
from graphspec.operations.rows import fact_keys
from graphspec.representations import groups as representations
from graphspec.schema.decisions import ROLES, roled
from graphspec.schema.declaration import Registry
from graphspec.schema.time import ALWAYS, NEVER, UNITED
from graphspec.views import view as viewing
from graphspec.views import visible
from graphspec.views.view import about

LOG = logging.getLogger(__name__)
VERSION = 28
DIFFERENCE = "graphspec.difference"
CODED = 16
UNWRITTEN = "no file is written for the key"
FOLDER = "descriptions"
BASED = ("part", "beside", "away", "stored")
AWAY = dict(code="inside a code checkout", volume="on another volume than the runtime home")
UNNAMED = -1
LAST = NEVER - 1
LIMIT = 1 << 62
MODELS = dict(none="no time", order="order without a clock", clock="one clock", clocks="two clocks")
BEFORE = "before"
NAMED = dict(
    none=None,
    order=dict(one="position", many="positions", stretch="positions"),
    clock=dict(one="time", many="times", stretch="time"),
)
NAMED["clocks"] = NAMED["clock"]
CORRECTED = (
    "a graph is described as it holds once every correction is known: a fact a correction replaced holds nothing and "
    "is counted apart, as replaced"
)
HELD = (
    "the nodes observed and their observations, and the edges a present fact holds, an edge being its relation and "
    "its two ends, in either order for a relation without direction, as the difference of two graphs meets them"
)
PLACED = (
    CORRECTED + "; at a %(one)s: " + HELD + "; before: before every %(one)s of the graph, where what states no start "
    "holds; a run: a stretch of %(stretch)s over which one edge holds without a break, bounded where it has a start "
    "and an end"
)
APART = (
    "; a node: an entity that a fact no correction replaced is about, the entities no such fact is about being counted "
    "apart, those an action of an option would create and the others; a set edge: counted as a node is, under its own "
    "name, its facts as observations and its members apart; an edge of a table that is about a set edge is held as a "
    "view holds it, by the rule its table states for where the set edge it is about holds nothing, and lapsed counts "
    "the edges that rule leaves unheld where a present fact states them, an edge once for every row of the table below "
    "at which it is; not counted above: the members of set edges, the graphs nodes hold with their held nodes and "
    "edges, and the entities no fact is about; facts naming observers: a task that names another observer is not "
    "shown them, and a task that names none is shown every fact"
)
LATER = (
    "; shown to some tasks only: the counts are what a task that names no observer and looks from after every time is "
    "shown, and a task that looks from before a fact is known is not shown it, so the facts known later than they "
    "hold from are counted, with the most by which one is and the last time from which one is known"
)
SHOWN = 12
CHOSEN = (
    "of more it shows the first, the last and those between at which its counts change most from the one before, by "
    "the sum of the changes over all it counts, the earlier of two that change alike, and a line under it says how "
    "many it shows, at how many of those left out a count changes, and the file of the record, which holds every one"
)
SEVERAL = "; a list of more than %d %%(many)s is given as their number, the first and the last" % SHOWN
RUNS = (
    "; in the record a count over the %(many)s is a list of runs, each a %(one)s and the count held from it until the "
    "%(one)s of the next, the first from before every %(one)s"
)
ROWED = "; a table shows at most %d decisions, as rows: " % SHOWN + CHOSEN
COLUMNED = "; a table shows at most %d %%(many)s, as columns, or decisions, as rows: " % SHOWN + CHOSEN + RUNS
READING = dict(
    none=CORRECTED + "; no fact states a time, so every fact holds always: held always, " + HELD + "; a run: an edge "
    "that holds, without start or end" + APART + ROWED,
    order=PLACED % NAMED["order"]
    + "; nothing is counted between two positions, so no run has a length"
    + APART
    + COLUMNED % NAMED["order"],
    clock=PLACED % NAMED["clock"] + ", lasting in the unit of its graph" + APART + LATER + COLUMNED % NAMED["clock"],
)
READING["clocks"] = READING["clock"]
WORDED = {
    False: dict(kind="kind", nodes="nodes", observations="observations"),
    True: dict(kind="set edge", nodes="set edges", observations="facts"),
}


@timed("describe.coded")
def coded(read):
    return dict(
        description=VERSION,
        views=viewing.VERSION,
        derivation=transform.VERSION,
        representations=representations.VERSION,
        entries=registries.versions(list(read) + [(OPERATIONS, [DIFFERENCE])]),
    )


@timed("describe.merged")
def merged(codes):
    held = coded([])
    entries = {tuple(entry) for code in [held, *codes] for entry in code["entries"]}
    return dict(held, entries=[list(entry) for entry in sorted(entries)])


def located(wanted, beside, part, stored):
    started = time.perf_counter()
    away = refusal(wanted)
    root, shown = (Path(wanted), FOLDER) if away is None else (cache_path(FOLDER, stored), "cache/%s/%s" % (FOLDER, stored))
    result = dict(folder=root / part, part=part, beside=beside, away=away, stored=stored, shown="%s/%s" % (shown, part))
    LOG.debug("describe located exit part=%s away=%s seconds=%.4f", part, away, time.perf_counter() - started)
    return result


@timed("describe.based")
def based(place):
    return {name: place[name] for name in BASED}


@timed("describe.keyed")
def keyed(manifest, place):
    read = Registry.from_declaration(manifest["spec"]).schema.read()
    return dict(based(place), code=coded(read))


@timed("describe.moments")
def moments(*stamps):
    held = np.concatenate([np.asarray(values, np.int64) for values in stamps])
    return np.unique(held[(held > ALWAYS) & (held < NEVER)])


def packed(columns):
    started = time.perf_counter()
    key, size = np.zeros(len(columns[0]), np.int64), 1
    for column in columns:
        column = np.asarray(column, np.int64)
        low = int(column.min()) if len(column) else 0
        span = int(column.max()) - low + 1 if len(column) else 1
        if size * span >= LIMIT:
            found, key = np.unique(key, return_inverse=True)
            key, size = key.reshape(-1), len(found)
        if size * span >= LIMIT:
            found, column = np.unique(column, return_inverse=True)
            column, low, span = column.reshape(-1), 0, len(found)
        key, size = key * span + (column - low), size * span
    LOG.debug("describe packed exit rows=%d columns=%d seconds=%.4f", len(key), len(columns), time.perf_counter() - started)
    return key


def met(registry, *tables):
    started = time.perf_counter()
    whole = np.concatenate(tables)
    found, first, place = np.unique(packed(whole[:, 1:4].T), return_index=True, return_inverse=True)
    (edge,) = fact_keys(registry, (np.zeros(len(first), np.int64), *whole[first, 1:4].T))
    keys = packed([whole[:, 0], edge[place.reshape(-1)], *whole[:, 4:].T])
    bounds = np.cumsum([0] + [len(table) for table in tables])
    LOG.debug("describe met exit rows=%d edges=%d seconds=%.4f", len(whole), len(found), time.perf_counter() - started)
    return [keys[low:high] for low, high in zip(bounds[:-1], bounds[1:])]


def counted(groups, begun, ended, size, places):
    started = time.perf_counter()
    width = len(places) + 1
    keys = np.r_[groups * width + begun, groups * width + ended].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]][: len(keys)])
    steps = np.r_[np.ones(len(groups), np.int64), -np.ones(len(groups), np.int64)][order]
    change = np.add.reduceat(steps, first) if len(first) else steps
    kept = (change != 0) & (keys[first] % width < len(places))
    result = [[[places[0], 0]] for _ in range(size)]
    marks = zip((keys[first][kept] // width).tolist(), (keys[first][kept] % width).tolist(), np.cumsum(change)[kept].tolist())
    for group, place, count in marks:
        if place:
            result[group].append([places[place], count])
        else:
            result[group][0][1] = count
    LOG.debug("describe counted exit spans=%d groups=%d seconds=%.4f", len(groups), size, time.perf_counter() - started)
    return result


@timed("describe.lasted")
def lasted(values, places):
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1][: len(values)]
    return [[places[at], count] for at, count in zip(starts.tolist(), values[starts].tolist())]


def counts(runs, places):
    started = time.perf_counter()
    since = np.asarray([ALWAYS if place is None else place for place, _ in runs], np.int64)
    asked = np.asarray([ALWAYS if place is None else place for place in places], np.int64)
    found = np.searchsorted(since, asked, "right") - 1
    if (found < 0).any():
        raise ValueError(
            "a count is asked at %s, before %s, the first place from which its runs hold it"
            % (places[int(np.flatnonzero(found < 0)[0])], runs[0][0] if runs else "any place, for it holds no run")
        )
    result = np.asarray([count for _, count in runs], np.int64)[found].tolist()
    LOG.debug("describe counts exit runs=%d places=%d seconds=%.4f", len(runs), len(places), time.perf_counter() - started)
    return result


def runs(groups, since, until):
    started = time.perf_counter()
    groups = np.unique(groups, return_inverse=True)[1].astype(np.int64).reshape(-1)
    order = np.lexsort((since, groups))
    groups, since, until = groups[order], since[order], until[order]
    bounds = np.unique(np.concatenate([since, until]))
    lifted = groups * (len(bounds) + 1)
    start = np.searchsorted(bounds, since) + lifted
    stop = np.maximum.accumulate(np.searchsorted(bounds, until) + lifted)
    first = np.r_[True, start[1:] > stop[:-1]][: len(groups)]
    last = np.r_[np.flatnonzero(first)[1:] - 1, len(groups) - 1][: int(first.sum())]
    result = order[first], since[first], bounds[stop[last] - lifted[last]]
    LOG.debug("describe runs exit facts=%d runs=%d seconds=%.4f", len(groups), len(last), time.perf_counter() - started)
    return result


@timed("describe.lasting")
def lasting(since, until, measured):
    lengths = (until - since)[(since > ALWAYS) & (until < NEVER)]
    held = measured and len(lengths) > 0
    return dict(
        runs=len(since),
        bounded=len(lengths),
        least=int(lengths.min()) if held else None,
        most=int(lengths.max()) if held else None,
        mean=float(lengths.mean()) if held else None,
    )


@timed("describe.placed")
def placed(times):
    return np.r_[np.int64(ALWAYS), np.asarray(times, np.int64)]


def shown(facts, group, moments):
    started = time.perf_counter()
    stamps = facts.stamps(group)
    seen = visible.allowed(
        len(stamps["since"]), facts.arrays[group + "_seen_row"], facts.arrays[group + "_seen_observer"], UNNAMED
    )
    result = visible.visible(
        stamps["since"],
        stamps["until"],
        stamps["known"],
        stamps["replaced"],
        seen,
        moments,
        np.full(len(moments), LAST, np.int64),
    )
    LOG.debug(
        "describe shown exit graph=%s group=%s facts=%d seconds=%.4f",
        facts.name,
        group,
        len(seen),
        time.perf_counter() - started,
    )
    return result


def kinds(registry, facts, seen, labels):
    started = time.perf_counter()
    entity = np.asarray(facts.obs_entity, np.int64)
    begun, ended = seen["obs"]
    last = ended > begun
    owner, result = np.asarray(facts.kinds, np.int64), {}
    stated = np.zeros(len(owner), bool)
    stated[entity[last]] = True
    whom, first, stop = runs(entity[last], begun[last], ended[last])
    found = counted(owner[entity[last][whom]], first, stop, len(registry.nodes), labels)
    said = counted(owner[entity[last]], begun[last], ended[last], len(registry.nodes), labels)
    for kind in registry.nodes:
        rows = owner[entity] == registry.kind(kind)
        result[kind] = dict(
            nodes=int((stated & (owner == registry.kind(kind))).sum()),
            observations=int((rows & last).sum()),
            replaced=int((rows & ~last).sum()),
            nodes_held=found[registry.kind(kind)],
            observations_held=said[registry.kind(kind)],
        )
    LOG.debug("describe kinds exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


def spread(begun, ended):
    started = time.perf_counter()
    counts = ended - begun
    rows = np.repeat(np.arange(len(counts)), counts)
    result = rows, begun[rows] + offsets(counts)
    LOG.debug("describe spread exit facts=%d rows=%d seconds=%.4f", len(counts), len(rows), time.perf_counter() - started)
    return result


def lapsing(registry, facts, lit, rows, steps, places):
    started = time.perf_counter()
    found = dict(
        step=steps,
        source=np.asarray(facts.fact_source, np.int64)[rows],
        target=np.asarray(facts.fact_target, np.int64)[rows],
        code=np.asarray(facts.fact_code, np.int64)[rows],
        present=np.asarray(facts.fact_present, bool)[rows],
        moments=places,
    )
    result = about(facts, registry, lit, found)
    LOG.debug(
        "describe lapsing exit graph=%s rows=%d kept=%d seconds=%.4f",
        facts.name,
        len(rows),
        int(result.sum()),
        time.perf_counter() - started,
    )
    return result


def ruled(registry, facts, places, seen):
    started = time.perf_counter()
    begun, ended = seen["fact"]
    code, owner = np.asarray(facts.fact_code, np.int64), np.asarray(facts.kinds, np.int64)
    holding = np.flatnonzero((ended > begun) & np.asarray(facts.fact_present, bool))
    stated = np.isin(code[holding], np.fromiter(registry.about, np.int64, len(registry.about)))
    piece, first, stop = holding[~stated], begun[holding[~stated]], ended[holding[~stated]]
    if stated.any():
        entity = np.asarray(facts.obs_entity, np.int64)
        joined = np.flatnonzero(np.asarray([kind in registry.schema.sets for kind in registry.nodes], bool)[owner[entity]])
        shown_rows, shown_steps = spread(seen["obs"][0][joined], seen["obs"][1][joined])
        lit = np.unique(shown_steps * len(owner) + entity[joined[shown_rows]])
        rows, steps = spread(begun[holding[stated]], ended[holding[stated]])
        rows = holding[stated][rows]
        kept = lapsing(registry, facts, lit, rows, steps, places)
        rows, steps = rows[kept], steps[kept]
        opened = np.flatnonzero(np.r_[True, (rows[1:] != rows[:-1]) | (steps[1:] != steps[:-1] + 1)][: len(rows)])
        closed = np.r_[opened[1:], len(rows)][: len(opened)] - 1
        piece, first, stop = np.r_[piece, rows[opened]], np.r_[first, steps[opened]], np.r_[stop, steps[closed] + 1]
    LOG.debug(
        "describe ruled exit graph=%s facts=%d pieces=%d seconds=%.4f",
        facts.name,
        len(holding),
        len(piece),
        time.perf_counter() - started,
    )
    return piece, first, stop


def relations(registry, facts, places, seen, labels):
    started = time.perf_counter()
    owner = np.asarray(facts.kinds, np.int64)
    code, present = np.asarray(facts.fact_code, np.int64), np.asarray(facts.fact_present, bool)
    source, target = np.asarray(facts.fact_source, np.int64), np.asarray(facts.fact_target, np.int64)
    begun, ended = seen["fact"]
    last, bounds = ended > begun, np.r_[places, np.int64(NEVER)]
    (edge,) = met(registry, np.stack((np.zeros(len(code), np.int64), code, source, target), 1))
    piece, opened, closed = ruled(registry, facts, places, seen)
    names = [(table, name) for table in registry.schema.edges for name in registry.encoding.relations[table]]
    placed_at = np.full(registry.relations, -1, np.int64)
    placed_at[[registry.code(name) for _, name in names]] = np.arange(len(names))
    whom, first, stop = runs(edge[piece], opened, closed)
    which = placed_at[code[piece[whom]]]
    found = counted(which, first, stop, len(names), labels)
    measured, result = registry.schema.model in UNITED, {}
    for at, (table, name) in enumerate(names):
        number = registry.code(name)
        rows = code == number
        said = rows & last
        held = said & present
        own = which == at
        rule = registry.about[number] if number in registry.about else None
        stated = (first[own], stop[own]) if rule is None else runs(edge[held], begun[held], ended[held])[1:]
        ends = sorted(set(zip(owner[source[said]].tolist(), owner[target[said]].tolist())))
        result[name] = dict(
            table=table,
            ends=[[registry.nodes[one], registry.nodes[other]] for one, other in ends],
            columns=list(registry.schema.edges[table]["columns"]),
            about=rule,
            lapsed=int((stated[1] - stated[0]).sum() - (stop[own] - first[own]).sum()),
            edges=len(np.unique(edge[piece[code[piece] == number]])),
            facts=int(held.sum()),
            absent=int((said & ~present).sum()),
            replaced=int((rows & ~last).sum()),
            edges_held=found[at],
            **lasting(bounds[first[own]], bounds[stop[own]], measured),
        )
    LOG.debug(
        "describe relations exit graph=%s relations=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started
    )
    return result


def sized(owner, holder, code):
    started = time.perf_counter()
    sizes = np.bincount(holder[owner[holder] == code], minlength=len(owner))[owner == code]
    result = dict(
        total=int(sizes.sum()),
        least=int(sizes.min()) if len(sizes) else None,
        most=int(sizes.max()) if len(sizes) else None,
    )
    LOG.debug("describe sized exit holders=%d seconds=%.4f", len(sizes), time.perf_counter() - started)
    return len(sizes), result


def unread(registry, facts, seen):
    started = time.perf_counter()
    schema, owner = registry.schema, np.asarray(facts.kinds, np.int64)
    held = {group: ended > begun for group, (begun, ended) in seen.items()}
    counts, gaps, known = {}, [], []
    watched = dict(nodes=0, edges=0, observers={name: {} for name in facts.observers})
    for group, part in zip(GROUPS, ("nodes", "edges")):
        stamps = facts.stamps(group)
        late = held[group] & (stamps["known"] > stamps["since"])
        counts[part] = int(late.sum())
        gaps.append((stamps["known"] - stamps["since"])[late & (stamps["since"] > ALWAYS)])
        known.append(stamps["known"][late])
        rows = np.asarray(facts.arrays[group + "_seen_row"], np.int64)
        observer = np.asarray(facts.arrays[group + "_seen_observer"], np.int64)
        watched[part] = int(held[group][np.unique(rows)].sum())
        for place, name in enumerate(facts.observers):
            watched["observers"][name][part] = int(held[group][rows[observer == place]].sum())
    gaps, known = np.concatenate(gaps), np.concatenate(known)
    joined = {}
    for kind in schema.sets:
        edges, members = sized(owner, np.asarray(facts.set_edge, np.int64), registry.kind(kind))
        joined[kind] = dict(edges=edges, members=members["total"], least=members["least"], most=members["most"])
    inner, source = np.asarray(facts.inner_owner, np.int64), np.asarray(facts.within_source, np.int64)
    holding = {}
    for kind in schema.holds:
        graphs, nodes = sized(owner, inner, registry.kind(kind))
        within = int((owner[inner[source]] == registry.kind(kind)).sum())
        holding[kind] = dict(graphs=graphs, nodes=nodes["total"], edges=within, least=nodes["least"], most=nodes["most"])
    action = np.asarray(facts.role_action, np.int64)
    task = np.asarray(facts.option_task, np.int64)[np.asarray(facts.action_option, np.int64)[action]]
    sort = roled(schema)[
        np.asarray(facts.task_kind, np.int64)[task],
        np.asarray(facts.action_kind, np.int64)[action],
        np.asarray(facts.role_name, np.int64),
    ]
    stated, made = np.zeros(len(owner), bool), np.zeros(len(owner), bool)
    stated[np.asarray(facts.obs_entity, np.int64)[held["obs"]]] = True
    made[np.asarray(facts.role_source, np.int64)[sort == ROLES.index("new")]] = True
    result = dict(
        later=dict(
            counts, gap=int(gaps.max()) if len(gaps) else None, last=int(known.max()) if len(known) else None
        ),
        watched=watched,
        sets=joined,
        held=holding,
        entities={
            kind: dict(
                created=int((~stated & made & (owner == registry.kind(kind))).sum()),
                unstated=int((~stated & ~made & (owner == registry.kind(kind))).sum()),
            )
            for kind in registry.nodes
        },
    )
    LOG.debug("describe unread exit graph=%s seconds=%.4f", facts.name, time.perf_counter() - started)
    return result


def tasks(registry, facts, splits):
    started = time.perf_counter()
    kind, split = np.asarray(facts.task_kind, np.int64), np.asarray(facts.task_split, np.int64)
    result = {}
    for code, name in enumerate(registry.schema.tasks["kinds"]):
        rows = kind == code
        result[name] = dict(
            tasks=int(rows.sum()),
            splits={held: int((rows & (split == at)).sum()) for at, held in enumerate(splits)},
            unsplit=int((rows & (split < 0)).sum()),
        )
    LOG.debug("describe tasks exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


def decisions(registry, facts):
    started = time.perf_counter()
    schema = registry.schema
    clocked, listed = schema.model != "none", list(schema.tasks["kinds"])
    drawn = [None] * len(facts.option_task)
    for option, generator in zip(np.asarray(facts.origin_option).tolist(), np.asarray(facts.origin_generator).tolist()):
        drawn[option] = facts.generators[generator]
    decided, first, count = (
        found.tolist() for found in np.unique(np.asarray(facts.option_task), return_index=True, return_counts=True)
    )
    kind, keys, times = (np.asarray(facts.arrays[name]).tolist() for name in ("task_kind", "task_key", "task_time"))
    result = {}
    for name in schema.decided():
        found = [
            (task, drawn[at : at + size])
            for task, at, size in zip(decided, first, count)
            if kind[task] == listed.index(name)
        ]
        stated = [held["name"] for held in schema.tasks["kinds"][name]["target"]["generators"]]
        named = list(dict.fromkeys(stated + sorted({held for _, options in found for held in options} - {None})))
        each = [
            dict(
                key=keys[task],
                time=times[task] if clocked else None,
                options=len(options),
                recorded=options.count(None),
                regenerated=len(options) - options.count(None),
                generators={held: options.count(held) for held in named},
            )
            for task, options in found
        ]
        result[name] = dict(
            decisions=len(each),
            **{part: sum(entry[part] for entry in each) for part in ("options", "recorded", "regenerated")},
            generators={held: sum(entry["generators"][held] for entry in each) for held in named},
            rule=schema.tasks["kinds"][name]["target"]["rule"],
            each=each,
        )
    LOG.debug("describe decisions exit graph=%s kinds=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


def clocked(registry, facts):
    started = time.perf_counter()
    tasks = facts.task_time if registry.schema.model != "none" else []
    names = [(group, group + "_" + name) for group in GROUPS for name in ("since", "until")]
    every = placed(moments(*[facts.arrays[name] for _, name in names], tasks))
    held = {}
    for group in GROUPS:
        begun, ended = shown(facts, group, every)
        held[group] = ended > begun
    result = moments(*[np.asarray(facts.arrays[name], np.int64)[held[group]] for group, name in names], tasks)
    LOG.debug("describe clocked exit graph=%s times=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started)
    return result


@timed("describe.stored")
def stored(path, graph):
    return Facts.load(Path(path) / "truth" / ("graph_%04d" % graph["index"]))


def graphed(registry, facts, splits):
    started = time.perf_counter()
    times = clocked(registry, facts)
    places, labels = placed(times), [None] + times.tolist()
    seen = {group: shown(facts, group, places) for group in GROUPS}
    result = dict(
        name=facts.name,
        unit=facts.unit,
        times=labels[1:],
        kinds=kinds(registry, facts, seen, labels),
        relations=relations(registry, facts, places, seen, labels),
        tasks=tasks(registry, facts, splits),
        decisions=decisions(registry, facts),
        unread=unread(registry, facts, seen),
    )
    LOG.info("describe graph exit graph=%s times=%d seconds=%.3f", facts.name, len(times), time.perf_counter() - started)
    return result


@timed("describe.modelled")
def modelled(registry):
    return dict(model=registry.schema.model, units=list(registry.schema.units))


def recorded(path, manifest):
    started = time.perf_counter()
    registry = Registry.from_declaration(manifest["spec"])
    result = dict(
        version=VERSION,
        stored_sha256=manifest["stored_sha256"],
        time=modelled(registry),
        splits=list(manifest["splits"]),
        graphs=[graphed(registry, stored(path, graph), manifest["splits"]) for graph in manifest["graphs"]],
    )
    LOG.info("describe recorded exit graphs=%d seconds=%.3f", len(result["graphs"]), time.perf_counter() - started)
    return result


@timed("describe.table")
def table(header, rows):
    cells = [[str(cell) for cell in row] for row in [header, *rows]]
    widths = [max(len(row[at]) for row in cells) for at in range(len(header))]
    return ["    " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells]


def most(change, eligible):
    started = time.perf_counter()
    chosen = eligible
    if len(eligible) > SHOWN:
        inner = eligible[1:-1]
        changing = inner[np.argsort(-change[inner], kind="stable")[: SHOWN - 2]]
        chosen = np.sort(np.r_[eligible[0], changing[change[changing] > 0], eligible[-1]])
    left = int((change[eligible] > 0).sum() - (change[chosen] > 0).sum())
    result = chosen.tolist(), len(eligible), left
    LOG.debug("describe most exit shown=%d of=%d seconds=%.4f", len(chosen), len(eligible), time.perf_counter() - started)
    return result


def picked(series, places, differing):
    started = time.perf_counter()
    index = {place: at for at, place in enumerate(places)}
    change, lit = np.zeros(len(places), np.int64), np.zeros(len(places) + 1, np.int64)
    for held in series:
        starts = np.asarray([index[place] for place, _ in held], np.int64)
        found = np.asarray([count for _, count in held], np.int64)
        np.add.at(change, starts[1:], np.abs(np.diff(found)))
        np.add.at(lit, starts[found != 0], 1)
        np.add.at(lit, np.r_[starts[1:], len(places)][: len(starts)][found != 0], -1)
    result = most(change, np.flatnonzero(np.cumsum(lit[:-1]) > 0) if differing else np.arange(len(places)))
    LOG.debug("describe picked exit series=%d places=%d seconds=%.4f", len(series), len(places), time.perf_counter() - started)
    return result


@timed("describe.omitted")
def omitted(shown, total, changing, laid, kept):
    if shown == total:
        return []
    return [
        "    %d of %d %s shown, the first, the last and those between at which the counts change most; a count changes "
        "at %d of the %d left out; the record holds every one, in the file %s/%s"
        % (shown, total, laid, changing, total - shown, kept["folder"], kept["file"])
    ]


def over(head, series, places, heads, kept):
    started = time.perf_counter()
    chosen, total, changing = picked([held for _, held in series], places, False)
    asked = [places[at] for at in chosen]
    result = table([head] + [heads[at] for at in chosen], [[label] + counts(held, asked) for label, held in series])
    result += omitted(len(chosen), total, changing, "columns", kept)
    LOG.debug("describe over exit series=%d shown=%d of=%d seconds=%.4f", len(series), len(chosen), total, time.perf_counter() - started)
    return result


@timed("describe.several")
def several(moments, named):
    if len(moments) <= SHOWN:
        return ", ".join(map(str, moments))
    return "%d %s from %s to %s" % (len(moments), named["many"], moments[0], moments[-1])


@timed("describe.related")
def related(name, entry):
    return "  relation %s of the table %s: %s; %s; edges %d, facts %d, absent %d, replaced %d; runs %d, bounded %d%s%s" % (
        name,
        entry["table"],
        "ends " + ", ".join("%s to %s" % tuple(pair) for pair in entry["ends"]) if entry["ends"] else "no ends",
        "columns " + ", ".join(entry["columns"]) if entry["columns"] else "no columns",
        entry["edges"],
        entry["facts"],
        entry["absent"],
        entry["replaced"],
        entry["runs"],
        entry["bounded"],
        "" if entry["least"] is None else ", lasting %d to %d, mean %s" % (entry["least"], entry["most"], entry["mean"]),
        ""
        if entry["about"] is None
        else "; about a set edge under the rule %s, lapsed %d" % (entry["about"], entry["lapsed"]),
    )


@timed("describe.ruled_text")
def ruled_text(rule):
    if rule is None:
        return "no rule of options"
    stated = ", ".join("%s %s" % pair for pair in rule["parameters"].items())
    return (
        "its options are given by the rule %s%s, of which those counted here are the options listed or drawn"
        % (rule["name"], " with " + stated if stated else "")
    )


def partial_text(found, named):
    started = time.perf_counter()
    later, watched = found["later"], found["watched"]
    parts = ["no fact known later than it holds from", "no fact naming observers"]
    if later["nodes"] + later["edges"]:
        parts[0] = (
            "facts known later than they hold from, about nodes %d and about edges %d, %s, the last known from %s %d, "
            "before which a task is not shown them"
            % (
                later["nodes"],
                later["edges"],
                "none of them stating from when it holds" if later["gap"] is None else "by %d at most" % later["gap"],
                named["one"],
                later["last"],
            )
        )
    if watched["nodes"] + watched["edges"]:
        parts[1] = (
            "facts naming observers, which a task naming another observer is not shown, about nodes %d and about "
            "edges %d, naming %s"
            % (
                watched["nodes"],
                watched["edges"],
                ", ".join("%s %d and %d" % (name, held["nodes"], held["edges"]) for name, held in watched["observers"].items()),
            )
        )
    result = "  shown to some tasks only: " + "; ".join(parts)
    LOG.debug("describe partial text exit characters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def uncounted_text(found):
    started = time.perf_counter()
    parts = ["no member of a set edge", "no graph a node holds", "no entity without a fact"]
    if found["sets"]:
        parts[0] = "members of set edges, " + ", ".join(
            "%s %d in %d set edges%s"
            % (kind, held["members"], held["edges"], " of %d to %d" % (held["least"], held["most"]) if held["edges"] else "")
            for kind, held in found["sets"].items()
        )
    if found["held"]:
        parts[1] = "graphs nodes hold, " + ", ".join(
            "%s %d%s, %d nodes and %d edges in all"
            % (
                kind,
                held["graphs"],
                " of %d to %d nodes" % (held["least"], held["most"]) if held["graphs"] else "",
                held["nodes"],
                held["edges"],
            )
            for kind, held in found["held"].items()
        )
    lacking = {kind: held for kind, held in found["entities"].items() if held["created"] + held["unstated"]}
    if lacking:
        parts[2] = "entities no fact is about, " + ", ".join(
            "%s %d an option would create and %d other" % (kind, held["created"], held["unstated"])
            for kind, held in lacking.items()
        )
    result = "  not counted above: " + "; ".join(parts)
    LOG.debug("describe uncounted text exit characters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("describe.viewed_text")
def viewed_text(view):
    splits = view["splits"]
    whom = ("the split %s" if len(splits) == 1 else "the splits %s") % ", ".join(splits) if splits else "no split"
    watching = "" if view["observer"] is None else ", those that name the observer %s" % view["observer"]
    return "the graph %s as the tasks of %s see it%s" % (view["graph"], whom, watching)


@timed("describe.spanned")
def spanned(named, times):
    if named is None:
        return "no time"
    return "%s %d from %d to %d" % (named["many"], len(times), times[0], times[-1]) if times else "no " + named["one"]


def decided(name, entry, named, kept):
    started = time.perf_counter()
    lines = [
        "  decisions of %s: decisions %d, options %d, recorded %d, regenerated %d; %s; %s"
        % (
            name,
            entry["decisions"],
            entry["options"],
            entry["recorded"],
            entry["regenerated"],
            ", ".join("%s %d" % pair for pair in entry["generators"].items()) or "no generator",
            ruled_text(entry["rule"]),
        )
    ]
    if entry["each"]:
        rows = [
            [held[part] for part in ("options", "recorded", "regenerated")] + list(held["generators"].values())
            for held in entry["each"]
        ]
        change = np.r_[0, np.abs(np.diff(np.asarray(rows, np.int64), axis=0)).sum(1)]
        chosen, total, changing = most(change, np.arange(len(rows)))
        lines += table(
            ["task", "time" if named is None else named["one"], "options", "recorded", "regenerated"] + list(entry["generators"]),
            [
                [entry["each"][at]["key"], "none" if entry["each"][at]["time"] is None else entry["each"][at]["time"]] + rows[at]
                for at in chosen
            ],
        )
        lines += omitted(len(chosen), total, changing, "rows", kept)
    LOG.debug("describe decided exit kind=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
    return lines


def graph_text(graph, named, kept):
    started = time.perf_counter()
    times = graph["times"]
    lines = [
        "",
        "graph %s, %s, %s"
        % (graph["name"], "no unit" if graph["unit"] is None else "unit " + graph["unit"], spanned(named, times)),
    ]
    worded = {name: WORDED[name in graph["unread"]["sets"]] for name in graph["kinds"]}
    lines += [
        "  %s %s: %s %d, %s %d, replaced %d"
        % (worded[name]["kind"], name, worded[name]["nodes"], entry["nodes"], worded[name]["observations"], entry["observations"], entry["replaced"])
        for name, entry in graph["kinds"].items()
    ]
    lines += [related(name, entry) for name, entry in graph["relations"].items()]
    series = [
        (name + " " + worded[name][part], entry[part + "_held"])
        for name, entry in graph["kinds"].items()
        for part in ("nodes", "observations")
    ] + [(name + " edges", entry["edges_held"]) for name, entry in graph["relations"].items()]
    if named is None:
        lines.append("  held always:")
        lines += over("held", series, [None], ["always"], kept)
    else:
        lines.append("  held before every %(one)s and at each %(one)s:" % named)
        lines += over(named["one"], series, [None] + times, [BEFORE] + times, kept)
    lines += [partial_text(graph["unread"], named), uncounted_text(graph["unread"])]
    lines += [
        "  kind of task %s: tasks %d; %s"
        % (
            name,
            entry["tasks"],
            ", ".join(["%s %d" % pair for pair in entry["splits"].items()] + ["no split %d" % entry["unsplit"]]),
        )
        for name, entry in graph["tasks"].items()
    ]
    for name, entry in graph["decisions"].items():
        lines += decided(name, entry, named, kept)
    LOG.debug("describe graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


@timed("describe.kept_text")
def kept_text(record):
    held = record["kept"]
    if held["away"] is None:
        return "kept beside the %s, in its folder %s" % (held["beside"], held["folder"])
    return "kept under the runtime home, in its folder %s: the library writes nothing beside the %s, which is %s" % (
        held["folder"],
        held["beside"],
        AWAY[held["away"]],
    )


def text(record):
    started = time.perf_counter()
    held = record["time"]
    lines = [
        "truth graphs of the bank stored as %s, described at version %d" % (record["stored_sha256"], record["version"]),
        kept_text(record),
        "time: %s, %s" % (MODELS[held["model"]], "units " + ", ".join(held["units"]) if held["units"] else "no unit"),
        "splits: %s" % (", ".join(record["splits"]) or "none"),
        READING[held["model"]],
    ]
    for graph in record["graphs"]:
        lines += graph_text(graph, NAMED[held["model"]], record["kept"])
    LOG.debug("describe text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def manifested(path):
    started = time.perf_counter()
    found = Path(path) / "manifest.json"
    if not found.is_file():
        raise FileNotFoundError("%s holds no manifest.json; a description reads the folder of a bank" % path)
    result = admitted(path, read_json(found))
    LOG.debug("describe manifested exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return result


@timed("describe.coding")
def coding(key):
    return sha256_json(key["code"])[:CODED]


@timed("describe.named")
def named(key):
    return "%s-%s" % (coding(key), sha256_json(key))


def served(place, key):
    started = time.perf_counter()
    record, unserved = None, UNWRITTEN
    if place.is_file():
        found, unread = parsed(place)
        if unread is not None:
            unserved = "it is no JSON: %s" % unread
        elif type(found) is not dict or sorted(found) != ["key", "record"]:
            unserved = "it holds no key and record"
        elif found["key"] != key:
            unserved = "it holds another key than the one it is named by"
        else:
            record, unserved = found["record"], None
    LOG.debug("describe served exit file=%s served=%s seconds=%.4f", place.name, unserved is None, time.perf_counter() - started)
    return record, unserved


def outdated(code):
    started = time.perf_counter()
    running = coded([])
    result = "its record of the code that made it is not one this code writes"
    if (
        type(code) is dict
        and sorted(code) == sorted(running)
        and type(code["entries"]) is list
        and all(type(entry) is list and len(entry) == 3 and type(entry[0]) is str and type(entry[1]) is str for entry in code["entries"])
    ):
        found = ["%s at version %s, now %s" % (name, code[name], running[name]) for name in running if name != "entries" and code[name] != running[name]]
        found += registries.moved(code["entries"])
        result = ", ".join(found) if found else None
    LOG.debug("describe outdated exit outdated=%s seconds=%.4f", result is not None, time.perf_counter() - started)
    return result


def swept(folder, mine):
    started = time.perf_counter()
    grouped = {}
    for name in sorted(os.listdir(folder)):
        if (folder / name).is_file():
            grouped.setdefault(name.split("-")[0].split(".")[0], []).append(name)
    removed = 0
    for code, names in grouped.items():
        if code == mine:
            continue
        reason = "it holds no record of the code that made it"
        if code + ".json" in names:
            found, unread = parsed(folder / (code + ".json"))
            reason = outdated(found if unread is None else None)
        if reason is not None:
            for name in names:
                (folder / name).unlink(missing_ok=True)
            removed += len(names)
            LOG.info("describe removed folder=%s files=%s made by other code than this: %s", folder, ", ".join(names), reason)
    LOG.info("describe swept exit folder=%s codes=%d files=%d seconds=%.3f", folder, len(grouped), removed, time.perf_counter() - started)
    return removed


def kept(where, key, made, written):
    started = time.perf_counter()
    folder, key = where["folder"], json.loads(json.dumps(key))
    for place in (folder.parent, folder):
        if place.exists() and not place.is_dir():
            raise NotADirectoryError("%s is a file; descriptions are kept in a folder of that name" % place)
    held = folder / named(key)
    place, text = held.with_suffix(".json"), held.with_suffix(".txt")
    record, unserved = served(place, key)
    if unserved is not None:
        record = dict(made(), kept=dict(beside=where["beside"], away=where["away"], folder=where["shown"], file=place.name))
        output_path(folder).mkdir(parents=True, exist_ok=True)
        if not (folder / (coding(key) + ".json")).is_file():
            write_json(folder / (coding(key) + ".json"), key["code"])
            swept(folder, coding(key))
        text.write_text(written(record), encoding="utf-8")
        write_record(place, dict(key=key, record=record))
        if unserved is not UNWRITTEN:
            LOG.warning("describe replaced file=%s because %s", place, unserved)
    elif not text.is_file():
        text.write_text(written(record), encoding="utf-8")
        LOG.warning("describe wrote again text=%s because the record was kept without it", text)
    LOG.info("describe kept exit key=%s served=%s seconds=%.3f", held.name, unserved is None, time.perf_counter() - started)
    return record


def describe(path):
    started = time.perf_counter()
    manifest = manifested(path)
    place = located(Path(path) / FOLDER, "bank", "truth", manifest["stored_sha256"])
    record = kept(place, keyed(manifest, place), lambda: recorded(path, manifest), text)
    LOG.info("describe exit path=%s seconds=%.3f", path, time.perf_counter() - started)
    return record
