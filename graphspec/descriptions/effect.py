import logging
import time

import numpy as np

from graphspec.engine.segments import Derivation
from graphspec.infra.timing import timed
from graphspec.operations.operation import Stored
from graphspec.operations.sets import PARAMETER

from .describe import table

LOG = logging.getLogger(__name__)
READING = (
    "held at a step: a node observed at the step other than as an anchor; an anchor, a node observed there only "
    "because a reference of a task of the step names it; an edge, two ends a present fact of its relation joins at "
    "the step, in either order for a relation without direction; before, removed, added and after count what is held, "
    "summed over the steps of the view; a share is of what was held before the entry, rounded to four places, and is "
    "nothing where nothing was held"
)


def begun(members, beside, registry, steps, spec, groups, seed, limits, partitions, room):
    started = time.perf_counter()
    named = sorted({step.graph for step in steps if step.graph is not None})
    result = Derivation.begun(
        members, {name: beside[name] for name in named}, registry, steps, spec, groups, seed, limits, partitions, room
    )
    LOG.debug("effect begun exit entries=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def derived(members, beside, registry, steps, spec, groups, seed, limits, partitions):
    started = time.perf_counter()
    total = sum(member.steps for member in members)
    result = [
        begun(members, beside, registry, steps[:count], spec, groups, seed, limits, partitions, 0).continued(total, None).derived()
        for count in range(len(steps) + 1)
    ]
    LOG.info(
        "effect derived exit graphs=%d states=%d seconds=%.3f", len(members), len(result), time.perf_counter() - started
    )
    return result


@timed("effect.tokens")
def tokens(graph):
    return np.ascontiguousarray(np.asarray(graph.entity_token, np.uint64)).view(np.int64)


@timed("effect.nodes")
def nodes(graph):
    entity = np.asarray(graph.obs_entity, np.int64)
    keys = np.stack(
        (np.asarray(graph.obs_step, np.int64), np.asarray(graph.kinds, np.int64)[entity], tokens(graph)[entity]), 1
    )
    return keys, np.asarray(graph.obs_anchor, bool)


def edges(graph, registry):
    started = time.perf_counter()
    present = np.asarray(graph.fact_present, bool)
    code = np.asarray(graph.fact_code, np.int64)[present]
    kinds, token = np.asarray(graph.kinds, np.int64), tokens(graph)
    source = np.asarray(graph.fact_source, np.int64)[present]
    target = np.asarray(graph.fact_target, np.int64)[present]
    first, second = np.stack((kinds[source], token[source]), 1), np.stack((kinds[target], token[target]), 1)
    symmetric = np.asarray([registry.by_code[place].symmetric for place in range(registry.relations)], bool)
    later = (first[:, 0] > second[:, 0]) | ((first[:, 0] == second[:, 0]) & (first[:, 1] > second[:, 1]))
    swapped = (symmetric[code] & later)[:, None]
    ends = np.concatenate((np.where(swapped, second, first), np.where(swapped, first, second)), 1)
    keys = np.concatenate((np.asarray(graph.fact_step, np.int64)[present][:, None], code[:, None], ends), 1)
    result = np.unique(keys, axis=0)
    LOG.debug("effect edges exit facts=%d edges=%d seconds=%.4f", len(keys), len(result), time.perf_counter() - started)
    return result


def related(found, registry):
    started = time.perf_counter()
    stated = [name for table_name in registry.schema.edges for name in registry.encoding.relations[table_name]]
    codes = sorted({code for _, _, ends in found for code in np.unique(ends[:, 1]).tolist()})
    names = list(dict.fromkeys(stated + [registry.by_code[code].name for code in codes]))
    places = np.full(registry.relations, -1, np.int64)
    places[[registry.code(name) for name in names]] = np.arange(len(names))
    result = names, [
        (keys, anchored, np.concatenate((ends[:, :1], places[ends[:, 1:2]], ends[:, 2:]), 1))
        for keys, anchored, ends in found
    ]
    LOG.debug("effect related exit relations=%d seconds=%.4f", len(names), time.perf_counter() - started)
    return result


@timed("effect.within")
def within(keys, low, high):
    chosen = (keys[:, 0] >= low) & (keys[:, 0] < high)
    return np.concatenate((keys[chosen, :1] - low, keys[chosen, 1:]), 1), chosen


@timed("effect.compared")
def compared(before, after):
    _, inverse = np.unique(np.concatenate((before, after)), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    earlier, later = inverse[: len(before)], inverse[len(before) :]
    return ~np.isin(earlier, later), ~np.isin(later, earlier)


@timed("effect.counted")
def counted(keys, steps, size):
    return np.bincount(keys[:, 0] * size + keys[:, 1], minlength=steps * size).reshape(steps, size)


@timed("effect.share")
def share(count, total):
    return round(count / total, 4) if total else None


def held(state, steps, kinds, names):
    started = time.perf_counter()
    keys, anchored, ends = state
    found, anchors, edged = (
        counted(keys[~anchored], steps, len(kinds)),
        counted(keys[anchored], steps, len(kinds)),
        counted(ends, steps, len(names)),
    )
    result = dict(
        kinds={kind: dict(nodes=found[:, at].tolist(), anchors=anchors[:, at].tolist()) for at, kind in enumerate(kinds)},
        relations={name: dict(edges=edged[:, at].tolist()) for at, name in enumerate(names)},
    )
    LOG.debug("effect held exit steps=%d seconds=%.4f", steps, time.perf_counter() - started)
    return result


def changed(before, after, size):
    started = time.perf_counter()
    removed, added = compared(before, after)
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
                )
            )
    LOG.debug("effect cuts exit %s=%s cuts=%d seconds=%.4f", sort, name, len(result), time.perf_counter() - started)
    return result


def effected(step, earlier, later, before, after, kinds, names, tasks, times):
    started = time.perf_counter()
    removed, added = changed(earlier[0][~earlier[1]], later[0][~later[1]], len(kinds))
    lost, made = changed(earlier[2], later[2], len(names))
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


def graphed(name, first, found, steps, registry, low, high, kinds, names):
    started = time.perf_counter()
    size = high - low
    times = np.asarray(first.step_time, np.int64)[low:high].tolist()
    tasks = tasked(first, registry, low, high)
    own = []
    for keys, anchored, ends in found:
        inside, chosen = within(keys, low, high)
        own.append((inside, anchored[chosen], within(ends, low, high)[0]))
    states = [held(state, size, kinds, names) for state in own]
    result = dict(
        name=name,
        times=times,
        tasks={task: [times[at] for at in held_steps] for task, held_steps in tasks.items()},
        held=states[0],
        effects=[
            effected(step, own[at], own[at + 1], states[at], states[at + 1], kinds, names, tasks, times)
            for at, step in enumerate(steps)
        ],
    )
    LOG.debug("effect graphed exit graph=%s steps=%d seconds=%.4f", name, size, time.perf_counter() - started)
    return result


def described(members, beside, registry, steps, spec, groups, seed, limits, partitions):
    started = time.perf_counter()
    result = told(derived(members, beside, registry, steps, spec, groups, seed, limits, partitions), members, registry, steps)
    LOG.info("effect described exit graphs=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


def told(graphs, members, registry, steps):
    started = time.perf_counter()
    names, found = related([nodes(graph) + (edges(graph, registry),) for graph in graphs], registry)
    kinds, member = list(registry.nodes), np.asarray(graphs[0].step_graph, np.int64)
    result = []
    for at, graph in enumerate(members):
        inside = np.flatnonzero(member == at)
        result.append(
            graphed(graph.name, graphs[0], found, steps, registry, int(inside[0]), int(inside[-1]) + 1, kinds, names)
        )
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


def stepped(times, state):
    started = time.perf_counter()
    columns = [
        pair
        for kind, entry in state["kinds"].items()
        for pair in ((kind + " nodes", entry["nodes"]), (kind + " anchors", entry["anchors"]))
    ] + [(name + " edges", entry["edges"]) for name, entry in state["relations"].items()]
    result = table(
        ["time"] + [header for header, _ in columns],
        [[moment] + [values[at] for _, values in columns] for at, moment in enumerate(times)],
    )
    LOG.debug("effect stepped exit rows=%d seconds=%.4f", len(times), time.perf_counter() - started)
    return result


@timed("effect.cut_text")
def cut_text(effect, found):
    return "  entry %s, of the operation %s, removes every %s %s at the times of the tasks of %s: %s; %s" % (
        effect["entry"],
        effect["operation"],
        "node of the kind" if found["sort"] == "kind" else "edge of the relation",
        found["name"],
        found["task"],
        ", ".join(map(str, found["times"])),
        "and at every time" if found["everywhere"] else "not at every time",
    )


def effect_text(effect, times):
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
    lines.append("  held at each time after it:")
    lines += stepped(times, effect)
    lines += [cut_text(effect, found) for found in effect["cuts"]]
    LOG.debug(
        "effect entry text exit entry=%s lines=%d seconds=%.4f", effect["entry"], len(lines), time.perf_counter() - started
    )
    return lines


def graph_text(graph):
    started = time.perf_counter()
    times = graph["times"]
    lines = [
        "",
        "graph %s, %d steps from time %d to %d" % (graph["name"], len(times), times[0], times[-1]),
        "  tasks: %s"
        % "; ".join(
            "%s at %s" % (task, ", ".join(map(str, moments)) or "no time") for task, moments in graph["tasks"].items()
        ),
        "  held at each time before the first entry:",
    ]
    lines += stepped(times, graph["held"])
    for effect in graph["effects"]:
        lines += effect_text(effect, times)
    LOG.debug("effect graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    splits = record["splits"]
    lines = [
        "variant %s of the bank stored as %s, described at version %d"
        % (record["variant"], record["stored_sha256"], record["version"]),
        "derivation seed %d, device %s" % (record["seed"], record["device"]),
        "splits trained on %s, scored on %s; rules of what each sees: %s"
        % (
            ", ".join(splits["trained"]),
            ", ".join(splits["scored"]),
            listed({name: ", ".join(rules) or "none" for name, rules in splits["sees"].items()}),
        ),
        "settings: %s" % listed(record["settings"]),
        READING,
    ]
    lines += [entry_text(entry) for entry in record["entries"]]
    for graph in record["graphs"]:
        lines += graph_text(graph)
    LOG.debug("effect text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
