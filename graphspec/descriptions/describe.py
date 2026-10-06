import logging
import time
from pathlib import Path

import numpy as np

from graphspec.facts.stored import GROUPS, Facts, admitted
from graphspec.infra.files import read_json, sha256_json, write_json
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.operations.rows import counted as offsets
from graphspec.operations.rows import fact_keys
from graphspec.schema.decisions import ROLES, roled
from graphspec.schema.declaration import Registry
from graphspec.schema.time import ALWAYS, NEVER, UNITED
from graphspec.views import visible
from graphspec.views.view import about

LOG = logging.getLogger(__name__)
VERSION = 17
UNNAMED = -1
LAST = NEVER - 1
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
READING = dict(
    none=CORRECTED + "; no fact states a time, so every fact holds always: held always, " + HELD + "; a run: an edge "
    "that holds, without start or end" + APART,
    order=PLACED % NAMED["order"] + "; nothing is counted between two positions, so no run has a length" + APART,
    clock=PLACED % NAMED["clock"] + ", lasting in the unit of its graph" + APART + LATER,
)
READING["clocks"] = READING["clock"]
WORDED = {
    False: dict(kind="kind", nodes="nodes", observations="observations"),
    True: dict(kind="set edge", nodes="set edges", observations="facts"),
}


@timed("describe.stem")
def stem(path, manifest):
    key = sha256_json(dict(stored=manifest["stored_sha256"], version=VERSION))
    return Path(path) / "descriptions" / "truth" / key


@timed("describe.moments")
def moments(*stamps):
    held = np.concatenate([np.asarray(values, np.int64) for values in stamps])
    return np.unique(held[(held > ALWAYS) & (held < NEVER)])


@timed("describe.counted")
def counted(since, until, times):
    return (
        np.searchsorted(np.sort(since), times, "right") - np.searchsorted(np.sort(until), times, "right")
    ).tolist()


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
    result = since[first], bounds[stop[last] - lifted[last]]
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


def kinds(registry, facts, places, seen):
    started = time.perf_counter()
    entity = np.asarray(facts.obs_entity, np.int64)
    begun, ended = seen["obs"]
    last, steps = ended > begun, np.arange(len(places))
    owner, result = np.asarray(facts.kinds, np.int64), {}
    stated = np.zeros(len(owner), bool)
    stated[entity[last]] = True
    for kind in registry.nodes:
        rows = owner[entity] == registry.kind(kind)
        held = rows & last
        found = counted(*runs(entity[held], begun[held], ended[held]), steps)
        said = counted(begun[held], ended[held], steps)
        result[kind] = dict(
            nodes=int((stated & (owner == registry.kind(kind))).sum()),
            observations=int(held.sum()),
            replaced=int((rows & ~last).sum()),
            nodes_before=found[0],
            observations_before=said[0],
            nodes_at=found[1:],
            observations_at=said[1:],
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


def relations(registry, facts, places, seen):
    started = time.perf_counter()
    owner = np.asarray(facts.kinds, np.int64)
    code, present = np.asarray(facts.fact_code, np.int64), np.asarray(facts.fact_present, bool)
    source, target = np.asarray(facts.fact_source, np.int64), np.asarray(facts.fact_target, np.int64)
    begun, ended = seen["fact"]
    last, steps, bounds = ended > begun, np.arange(len(places)), np.r_[places, np.int64(NEVER)]
    (edge,) = fact_keys(registry, (np.zeros(len(code), np.int64), code, source, target))
    piece, opened, closed = ruled(registry, facts, places, seen)
    measured, result = registry.schema.model in UNITED, {}
    for table, entry in registry.schema.edges.items():
        for name in registry.encoding.relations[table]:
            number = registry.code(name)
            rows = code == number
            said = rows & last
            held = said & present
            own = code[piece] == number
            pairs = edge[piece[own]]
            first, stop = runs(pairs, opened[own], closed[own])
            found = counted(first, stop, steps)
            rule = registry.about[number] if number in registry.about else None
            stated = found if rule is None else counted(*runs(edge[held], begun[held], ended[held]), steps)
            ends = sorted(set(zip(owner[source[said]].tolist(), owner[target[said]].tolist())))
            result[name] = dict(
                table=table,
                ends=[[registry.nodes[one], registry.nodes[other]] for one, other in ends],
                columns=list(entry["columns"]),
                about=rule,
                lapsed=sum(stated) - sum(found),
                edges=len(np.unique(pairs)),
                facts=int(held.sum()),
                absent=int((said & ~present).sum()),
                replaced=int((rows & ~last).sum()),
                edges_before=found[0],
                edges_at=found[1:],
                **lasting(bounds[first], bounds[stop], measured),
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
    places = placed(times)
    seen = {group: shown(facts, group, places) for group in GROUPS}
    result = dict(
        name=facts.name,
        unit=facts.unit,
        times=times.tolist(),
        kinds=kinds(registry, facts, places, seen),
        relations=relations(registry, facts, places, seen),
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
    cells = [list(header)] + [[str(cell) for cell in row] for row in rows]
    widths = [max(len(row[at]) for row in cells) for at in range(len(header))]
    return ["    " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells]


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


@timed("describe.spanned")
def spanned(named, times):
    if named is None:
        return "no time"
    return "%s %d from %d to %d" % (named["many"], len(times), times[0], times[-1]) if times else "no " + named["one"]


def decided(name, entry, named):
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
        lines += table(
            ["task", "time" if named is None else named["one"], "options", "recorded", "regenerated"] + list(entry["generators"]),
            [
                [held["key"], "none" if held["time"] is None else held["time"]]
                + [held[part] for part in ("options", "recorded", "regenerated")]
                + list(held["generators"].values())
                for held in entry["each"]
            ],
        )
    LOG.debug("describe decided exit kind=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
    return lines


def graph_text(graph, named):
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
    columns = [
        (name + " " + worded[name][part], entry[part + "_before"], entry[part + "_at"])
        for name, entry in graph["kinds"].items()
        for part in ("nodes", "observations")
    ] + [(name + " edges", entry["edges_before"], entry["edges_at"]) for name, entry in graph["relations"].items()]
    headers, first = [header for header, _, _ in columns], [before for _, before, _ in columns]
    if named is None:
        lines.append("  held always:")
        lines += table(headers, [first])
    else:
        lines.append("  held before every %(one)s and at each %(one)s:" % named)
        lines += table(
            [named["one"]] + headers,
            [[BEFORE] + first] + [[moment] + [values[at] for _, _, values in columns] for at, moment in enumerate(times)],
        )
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
        lines += decided(name, entry, named)
    LOG.debug("describe graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    held = record["time"]
    lines = [
        "truth graphs of the bank stored as %s, described at version %d" % (record["stored_sha256"], record["version"]),
        "time: %s, %s" % (MODELS[held["model"]], "units " + ", ".join(held["units"]) if held["units"] else "no unit"),
        "splits: %s" % (", ".join(record["splits"]) or "none"),
        READING[held["model"]],
    ]
    for graph in record["graphs"]:
        lines += graph_text(graph, NAMED[held["model"]])
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


def kept(held, made, written):
    started = time.perf_counter()
    served = held.with_suffix(".json").exists()
    if served:
        record = read_json(held.with_suffix(".json"))
    else:
        record = made()
        output_path(held.parent).mkdir(parents=True, exist_ok=True)
        output_path(held.with_suffix(".txt")).write_text(written(record), encoding="utf-8")
        write_json(held.with_suffix(".json"), record)
    LOG.info("describe kept exit key=%s served=%s seconds=%.3f", held.name, served, time.perf_counter() - started)
    return record


def describe(path):
    started = time.perf_counter()
    manifest = manifested(path)
    record = kept(stem(path, manifest), lambda: recorded(path, manifest), text)
    LOG.info("describe exit path=%s seconds=%.3f", path, time.perf_counter() - started)
    return record
