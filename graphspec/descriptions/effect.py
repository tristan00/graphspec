import logging
import time

import numpy as np

from graphspec.engine.segments import Derivation
from graphspec.infra.timing import timed
from graphspec.operations.operation import Stored
from graphspec.operations.sets import PARAMETER

from .describe import NAMED, table, viewed_text
from .differences import compared, counted, distinct, numbered, tokens, unmatched

LOG = logging.getLogger(__name__)
READING = (
    "held at a step: a node observed at the step other than as an anchor; an anchor, a node observed there only "
    "because a reference of a task of the step names it; an edge, two ends a present fact of its relation joins at "
    "the step, in either order for a relation without direction; before, removed, added and after count what is held, "
    "summed over the steps of the view; a share is of what was held before the entry, rounded to four places, and is "
    "nothing where nothing was held; an entry that leaves no node of a kind or no edge of a relation at the steps of "
    "the tasks of a kind of task where one was held before it is named as removing every one; alone: a later entry "
    "named so too, which leaves none where the pipeline is derived without the entries named before it for the same, "
    "or, where the pipeline is refused without one of them, with the one parameter of it re-stated after which the "
    "graph holds the most"
)
ALONE = dict(absent="without the entry", least="with one parameter re-stated of the entry")


def begun(members, beside, registry, steps, spec, groups, seed, limits, partitions, room):
    started = time.perf_counter()
    named = sorted({step.graph for step in steps if step.graph is not None})
    result = Derivation.begun(
        members, {name: beside[name] for name in named}, registry, steps, spec, groups, seed, limits, partitions, room
    )
    LOG.debug("effect begun exit entries=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def ended(members, beside, registry, steps, spec, groups, seed, limits, partitions):
    started = time.perf_counter()
    total = sum(member.steps for member in members)
    start = begun(members, beside, registry, steps, spec, groups, seed, limits, partitions, 0)
    result = start.continued(total, None).derived()
    LOG.debug("effect ended exit entries=%d steps=%d seconds=%.4f", len(steps), total, time.perf_counter() - started)
    return result


@timed("effect.nodes")
def nodes(graph):
    entity = np.asarray(graph.obs_entity, np.int64)
    keys = np.stack(
        (np.asarray(graph.obs_step, np.int64), np.asarray(graph.kinds, np.int64)[entity], tokens(graph)[entity]), 1
    )
    return keys, np.asarray(graph.obs_anchor, bool)


def edges(graph, registry, number):
    started = time.perf_counter()
    present = np.asarray(graph.fact_present, bool)
    step, code, source, target = (
        np.asarray(graph.arrays[name], np.int64)[present]
        for name in ("fact_step", "fact_code", "fact_source", "fact_target")
    )
    result = distinct(registry, np.stack((step, code, number[source], number[target]), 1))
    LOG.debug("effect edges exit facts=%d edges=%d seconds=%.4f", len(step), len(result), time.perf_counter() - started)
    return result


def related(found, registry):
    started = time.perf_counter()
    stated = [name for table_name in registry.schema.edges for name in registry.encoding.relations[table_name]]
    added = {registry.code(name) for _, held in registry.schema.grown for name in held.relations}
    codes = sorted(added | {code for _, _, ends in found for code in np.unique(ends[:, 1]).tolist()})
    names = list(dict.fromkeys(stated + [registry.by_code[code].name for code in codes]))
    places = np.full(registry.relations, -1, np.int64)
    places[[registry.code(name) for name in names]] = np.arange(len(names))
    LOG.debug("effect related exit relations=%d seconds=%.4f", len(names), time.perf_counter() - started)
    return names, places


@timed("effect.within")
def within(keys, low, high):
    chosen = (keys[:, 0] >= low) & (keys[:, 0] < high)
    return np.concatenate((keys[chosen, :1] - low, keys[chosen, 1:]), 1), chosen


@timed("effect.placed")
def placed(facts, places):
    return np.stack((facts[:, 0], places[facts[:, 1]]), 1)


@timed("effect.share")
def share(count, total):
    return round(count / total, 4) if total else None


def held(state, steps, kinds, names, places):
    started = time.perf_counter()
    keys, anchored, ends = state
    found, anchors, edged = (
        counted(keys[~anchored], steps, len(kinds)),
        counted(keys[anchored], steps, len(kinds)),
        counted(placed(ends, places), steps, len(names)),
    )
    result = dict(
        kinds={kind: dict(nodes=found[:, at].tolist(), anchors=anchors[:, at].tolist()) for at, kind in enumerate(kinds)},
        relations={name: dict(edges=edged[:, at].tolist()) for at, name in enumerate(names)},
    )
    LOG.debug("effect held exit steps=%d seconds=%.4f", steps, time.perf_counter() - started)
    return result


def changed(before, after, apart, size):
    started = time.perf_counter()
    removed, added = apart
    result = (
        np.bincount(before[removed, 1], minlength=size).tolist(),
        np.bincount(after[added, 1], minlength=size).tolist(),
    )
    LOG.debug(
        "effect changed exit removed=%d added=%d seconds=%.4f",
        int(removed.sum()),
        int(added.sum()),
        time.perf_counter() - started,
    )
    return result


def tallied(before, after, removed, added, part):
    started = time.perf_counter()
    first = sum(before[part])
    result = dict(
        before=first,
        removed=removed,
        removed_share=share(removed, first),
        added=added,
        added_share=share(added, first),
        after=sum(after[part]),
        **after,
    )
    LOG.debug("effect tallied exit part=%s seconds=%.4f", part, time.perf_counter() - started)
    return result


def cuts(sort, name, before, after, tasks, times):
    started = time.perf_counter()
    before, after, result = np.asarray(before), np.asarray(after), []
    for task, steps in tasks.items():
        steps = np.asarray(steps, np.int64)
        holding = before[steps] > 0
        if holding.any() and not after[steps].any():
            result.append(
                dict(
                    sort=sort,
                    name=name,
                    task=task,
                    times=[times[step] for step in steps[holding].tolist()],
                    everywhere=not after.any(),
                    without=[],
                )
            )
    LOG.debug("effect cuts exit %s=%s cuts=%d seconds=%.4f", sort, name, len(result), time.perf_counter() - started)
    return result


def left(graph, sort, name, task):
    started = time.perf_counter()
    held = graph["held"]["kinds" if sort == "kind" else "relations"]
    steps = [place for place, moment in enumerate(graph["times"]) if moment in graph["tasks"][task]]
    result = name in held and any(held[name]["nodes" if sort == "kind" else "edges"][place] for place in steps)
    LOG.debug("effect left exit %s=%s left=%s seconds=%.4f", sort, name, result, time.perf_counter() - started)
    return result


def effected(step, earlier, later, before, after, kinds, names, places, registry, tasks, times):
    started = time.perf_counter()
    seen = earlier[0][~earlier[1]], later[0][~later[1]]
    removed, added = changed(*seen, compared(*seen), len(kinds))
    lost, made = changed(
        placed(earlier[2], places), placed(later[2], places), unmatched(registry, earlier[2], later[2]), len(names)
    )
    result = dict(
        entry=step.name,
        operation=step.operation.name,
        kinds={
            kind: tallied(before["kinds"][kind], after["kinds"][kind], removed[at], added[at], "nodes")
            for at, kind in enumerate(kinds)
        },
        relations={
            name: tallied(before["relations"][name], after["relations"][name], lost[at], made[at], "edges")
            for at, name in enumerate(names)
        },
        cuts=[
            found
            for kind in kinds
            for found in cuts("kind", kind, before["kinds"][kind]["nodes"], after["kinds"][kind]["nodes"], tasks, times)
        ]
        + [
            found
            for name in names
            for found in cuts(
                "relation", name, before["relations"][name]["edges"], after["relations"][name]["edges"], tasks, times
            )
        ],
    )
    LOG.debug("effect effected exit entry=%s seconds=%.4f", step.name, time.perf_counter() - started)
    return result


def tasked(graph, registry, low, high):
    started = time.perf_counter()
    step, kind = np.asarray(graph.task_step, np.int64), np.asarray(graph.task_kind, np.int64)
    inside = (step >= low) & (step < high)
    result = {name: np.unique(step[inside & (kind == at)] - low).tolist() for at, name in enumerate(registry.tasks)}
    LOG.debug("effect tasked exit tasks=%d seconds=%.4f", int(inside.sum()), time.perf_counter() - started)
    return result


def graphed(name, first, found, steps, registry, low, high, kinds):
    started = time.perf_counter()
    size = high - low
    times = [None] * size if registry.schema.model == "none" else np.asarray(first.step_time, np.int64)[low:high].tolist()
    tasks = tasked(first, registry, low, high)
    own = []
    for keys, anchored, ends in found:
        inside, chosen = within(keys, low, high)
        own.append((inside, anchored[chosen], within(ends, low, high)[0]))
    names, places = related(own, registry)
    states = [held(state, size, kinds, names, places) for state in own]
    result = dict(
        name=name,
        times=times,
        tasks={task: [times[at] for at in held_steps] for task, held_steps in tasks.items()},
        held=states[0],
        effects=[
            effected(
                step, own[at], own[at + 1], states[at], states[at + 1], kinds, names, places, registry, tasks, times
            )
            for at, step in enumerate(steps)
        ],
    )
    LOG.debug("effect graphed exit graph=%s steps=%d seconds=%.4f", name, size, time.perf_counter() - started)
    return result


def told(graphs, members, registry, steps):
    started = time.perf_counter()
    numbers = numbered(*[np.stack((np.asarray(graph.kinds, np.int64), tokens(graph)), 1) for graph in graphs])
    found = [nodes(graph) + (edges(graph, registry, number),) for graph, number in zip(graphs, numbers)]
    kinds, member = list(registry.nodes), np.asarray(graphs[0].step_graph, np.int64)
    result = []
    for at, graph in enumerate(members):
        inside = np.flatnonzero(member == at)
        low, high = int(inside[0]), int(inside[-1]) + 1
        result.append(graphed(graph.name, graphs[0], found, steps, registry, low, high, kinds))
    LOG.debug("effect told exit graphs=%d states=%d seconds=%.4f", len(result), len(graphs), time.perf_counter() - started)
    return result


def entered(step, registry, spec, splits):
    started = time.perf_counter()
    storage = [
        PARAMETER % kind
        for need in step.operation.requires
        if isinstance(need, Stored) and need.parameter in step.parameters
        for kind in need.kinds(registry, step.parameters[need.parameter])
    ]
    result = dict(
        name=step.name,
        operation=step.operation.name,
        version=step.operation.version,
        graph=step.graph,
        parameters=dict(step.parameters),
        reads=dict(step.read),
        splits={part: list(splits[part]) for part in step.operation.splits},
        storage={setting: spec[setting] for setting in dict.fromkeys(storage)},
    )
    LOG.debug("effect entered exit entry=%s seconds=%.4f", step.name, time.perf_counter() - started)
    return result


@timed("effect.listed")
def listed(mapping):
    return ", ".join("%s %s" % pair for pair in mapping.items()) or "none"


@timed("effect.spoken")
def spoken(value):
    return "nothing" if value is None else str(value)


@timed("effect.entry_text")
def entry_text(entry):
    return "entry %s, of the operation %s at version %d: parameters %s; reads %s; splits %s; storage %s; second graph %s" % (
        entry["name"],
        entry["operation"],
        entry["version"],
        listed(entry["parameters"]),
        listed(entry["reads"]),
        listed({part: ", ".join(names) for part, names in entry["splits"].items()}),
        listed(entry["storage"]),
        "none" if entry["graph"] is None else entry["graph"],
    )


def stepped(times, state, named):
    started = time.perf_counter()
    columns = [
        pair
        for kind, entry in state["kinds"].items()
        for pair in ((kind + " nodes", entry["nodes"]), (kind + " anchors", entry["anchors"]))
    ] + [(name + " edges", entry["edges"]) for name, entry in state["relations"].items()]
    counts = [[values[at] for _, values in columns] for at in range(len(times))]
    if named is None:
        result = table([header for header, _ in columns], counts)
    else:
        result = table([named["one"]] + [header for header, _ in columns], [[moment] + row for moment, row in zip(times, counts)])
    LOG.debug("effect stepped exit rows=%d seconds=%.4f", len(times), time.perf_counter() - started)
    return result


@timed("effect.stepping")
def stepping(named):
    return "step" if named is None else named["one"]


@timed("effect.lifted")
def lifted(without):
    return ", ".join("%s %s" % (ALONE[held["how"]], held["entry"]) for held in without)


@timed("effect.cut_text")
def cut_text(effect, found, named):
    one = stepping(named)
    return "  entry %s, of the operation %s, removes every %s %s at the %s of the tasks of %s; %s%s" % (
        effect["entry"],
        effect["operation"],
        "node of the kind" if found["sort"] == "kind" else "edge of the relation",
        found["name"],
        one if named is None else named["many"],
        found["task"] if named is None else "%s: %s" % (found["task"], ", ".join(map(str, found["times"]))),
        ("and at every %s" if found["everywhere"] else "not at every %s") % one,
        "; alone, %s" % lifted(found["without"]) if found["without"] else "",
    )


def effect_text(effect, times, named):
    started = time.perf_counter()
    rows = [(kind + " nodes", entry) for kind, entry in effect["kinds"].items()]
    rows += [(name + " edges", entry) for name, entry in effect["relations"].items()]
    lines = ["  entry %s, of the operation %s:" % (effect["entry"], effect["operation"])]
    lines += table(
        ["held", "before", "removed", "share", "added", "share", "after"],
        [
            [
                label,
                entry["before"],
                entry["removed"],
                spoken(entry["removed_share"]),
                entry["added"],
                spoken(entry["added_share"]),
                entry["after"],
            ]
            for label, entry in rows
        ],
    )
    lines.append("  held at each %s after it:" % stepping(named))
    lines += stepped(times, effect, named)
    lines += [cut_text(effect, found, named) for found in effect["cuts"]]
    LOG.debug(
        "effect entry text exit entry=%s lines=%d seconds=%.4f", effect["entry"], len(lines), time.perf_counter() - started
    )
    return lines


@timed("effect.tasked_text")
def tasked_text(task, moments, named):
    if named is None:
        return "%s at %s" % (task, "the step" if moments else "no step")
    return "%s at %s" % (task, ", ".join(map(str, moments)) or "no " + named["one"])


def graph_text(graph, named):
    started = time.perf_counter()
    times = graph["times"]
    spanned = ", no time" if named is None else " from %s %d to %d" % (named["one"], times[0], times[-1])
    lines = [
        "",
        "graph %s, %s, %d steps%s" % (graph["name"], viewed_text(graph["view"]), len(times), spanned),
        "  tasks: %s" % "; ".join(tasked_text(task, moments, named) for task, moments in graph["tasks"].items()),
        "  held at each %s before the first entry:" % stepping(named),
    ]
    lines += stepped(times, graph["held"], named)
    for effect in graph["effects"]:
        lines += effect_text(effect, times, named)
    LOG.debug("effect graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


def settled(record):
    started = time.perf_counter()
    splits = record["splits"]
    result = [
        "derivation seed %d, device %s" % (record["seed"], record["device"]),
        "splits trained on %s, scored on %s; rules of what each sees: %s"
        % (
            ", ".join(splits["trained"]),
            ", ".join(splits["scored"]),
            listed({name: ", ".join(rules) or "none" for name, rules in splits["sees"].items()}),
        ),
        "settings: %s" % listed(record["settings"]),
        "graphs of the bank the view does not show: %s"
        % (
            "; ".join(
                "%s, which holds %d tasks, %d of them of a split trained on or scored on, and to which the choice of "
                "steps %s gives %d sequences of steps"
                % (held["name"], held["tasks"], held["stated"], record["settings"]["time/steps"], held["sequences"])
                for held in record["unshown"]
            )
            or "none"
        ),
    ]
    LOG.debug("effect settled exit lines=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def text(record):
    started = time.perf_counter()
    lines = [
        "variant %s of the bank stored as %s, described at version %d"
        % (record["variant"], record["stored_sha256"], record["version"]),
        *settled(record),
        READING,
    ]
    lines += [entry_text(entry) for entry in record["entries"]]
    for graph in record["graphs"]:
        lines += graph_text(graph, NAMED[record["time"]["model"]])
    LOG.debug("effect text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
