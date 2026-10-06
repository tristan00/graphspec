import logging
import time
from pathlib import Path

import numpy as np

from graphspec.descriptions import describe as truth
from graphspec.descriptions import differences as differing
from graphspec.descriptions import effect
from graphspec.engine.segments import padded
from graphspec.infra.timing import timed

from . import variant as varying
from .restated import coded, viewed

LOG = logging.getLogger(__name__)


def ended(deriver, config):
    started = time.perf_counter()
    found = {}
    for indices, given in varying.grouped(deriver, config):
        graph = effect.ended(*given)
        owner = np.asarray(graph.step_graph, np.int64)
        for at, member in enumerate(indices):
            inside = np.flatnonzero(owner == at)
            found[member] = graph, given[2], int(inside[0]), int(inside[-1]) + 1
    result = [(graph["name"], *found[graph["index"]]) for graph in deriver.viewed(config).graphs]
    LOG.info("differences ended exit graphs=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


def shared(registries):
    started = time.perf_counter()
    sized = [
        {name: held.encoding.sizes[name] for name in held.encoding.table if sum(held.encoding.sizes[name])}
        for held in registries
    ]
    names = [name for name in sized[0] if name in sized[1]]
    unlike = [name for name in names if sized[0][name] != sized[1][name]]
    if unlike:
        raise ValueError(
            "the first variant encodes the column %s as %d values and %d flags and the second as %d and %d; the "
            "values of the nodes both hold are compared column by column, each encoded alike on the two sides"
            % (unlike[0], *sized[0][unlike[0]], *sized[1][unlike[0]])
        )
    none = np.zeros(0, np.int64)
    result = [np.concatenate([np.r_[held.encoding.slots(name)] for name in names] + [none]) for held in registries]
    LOG.debug("differences shared exit columns=%d seconds=%.4f", len(names), time.perf_counter() - started)
    return result


def columned(registries):
    started = time.perf_counter()
    result = dict(kinds={}, relations={})
    for kind in [kind for kind in registries[0].nodes if kind in registries[1].nodes]:
        held = [list(registry.schema.entities()[kind]) for registry in registries]
        apart = [[column for column in own if column not in other] for own, other in (held, held[::-1])]
        if apart[0] or apart[1]:
            result["kinds"][kind] = dict(zip(differing.SIDES, apart))
    for name in [name for name in registries[0].by_name if name in registries[1].by_name]:
        held = [list(registry.by_name[name].attributes) for registry in registries]
        apart = [[column for column in own if column not in other] for own, other in (held, held[::-1])]
        if apart[0] or apart[1]:
            result["relations"][name] = dict(zip(differing.SIDES, apart))
    LOG.debug(
        "differences columned exit kinds=%d relations=%d seconds=%.4f",
        len(result["kinds"]),
        len(result["relations"]),
        time.perf_counter() - started,
    )
    return result


def taken(graph, registry, low, high, steps, width, slots):
    started = time.perf_counter()
    moment = np.asarray(graph.obs_step, np.int64)
    seen = (moment >= low) & (moment < high)
    nodes = np.stack((steps[moment[seen] - low], np.asarray(graph.obs_entity, np.int64)[seen]), 1)
    moment = np.asarray(graph.fact_step, np.int64)
    held = np.asarray(graph.fact_present, bool) & (moment >= low) & (moment < high)
    ends = [np.asarray(graph.arrays[name], np.int64)[held] for name in ("fact_code", "fact_source", "fact_target")]
    result = dict(
        registry=registry,
        kinds=np.asarray(graph.kinds, np.int64),
        tokens=differing.tokens(graph),
        nodes=nodes,
        anchored=np.asarray(graph.obs_anchor, bool)[seen],
        values=np.asarray(graph.x_table)[:, slots],
        valued=np.asarray(graph.obs_ref, np.int64)[seen],
        facts=np.stack([steps[moment[held] - low]] + ends, 1),
        table=padded(graph.value_table, width),
        reference=np.asarray(graph.fact_ref, np.int64)[held],
    )
    LOG.debug(
        "differences taken exit graph=%s steps=%d nodes=%d facts=%d seconds=%.4f",
        graph.name,
        high - low,
        len(nodes),
        int(held.sum()),
        time.perf_counter() - started,
    )
    return result


def paired(first, second):
    started = time.perf_counter()
    name, both = first[0], (first, second)
    shown = [np.asarray(graph.step_time, np.int64)[low:high] for _, graph, _, low, high in both]
    times, steps = differing.met(name, *shown)
    width = max(np.asarray(graph.value_table).shape[1] for _, graph, _, _, _ in both)
    slots = shared([registry for _, _, registry, _, _ in both])
    sides = [
        taken(graph, registry, low, high, held, width, columns)
        for (_, graph, registry, low, high), held, columns in zip(both, steps, slots)
    ]
    placed = [None] * len(times) if first[2].schema.model == "none" else times.tolist()
    result = dict(
        name=name,
        times=placed,
        alone=dict(first=np.setdiff1d(*shown).tolist(), second=np.setdiff1d(*shown[::-1]).tolist()),
        **differing.differed(*sides, placed),
    )
    LOG.debug("differences paired exit graph=%s times=%d seconds=%.4f", name, len(times), time.perf_counter() - started)
    return result


def recorded(sides, configs):
    started = time.perf_counter()
    held = [ended(deriver, config) for (deriver, _), config in zip(sides, configs)]
    names = [[graph[0] for graph in graphs] for graphs in held]
    if names[0] != names[1]:
        raise ValueError(
            "the view of the first variant shows the graphs %s and the view of the second %s; a difference reads the "
            "same graphs of the view in both" % (", ".join(names[0]) or "none", ", ".join(names[1]) or "none")
        )
    views = viewed(sides[0][0], configs[0])
    shown = sides[0][0].viewed(configs[0]).graphs
    graphs = [dict(paired(first, second), view=views[graph["index"]]) for first, second, graph in zip(*held, shown)]
    columns = columned([deriver.fitted(config) for (deriver, _), config in zip(sides, configs)])
    result = dict(
        version=truth.VERSION,
        stored_sha256=sides[0][0].source,
        time=truth.modelled(sides[0][0].registry),
        first=varying.headed(sides[0][0], configs[0], sides[0][1]),
        second=varying.headed(sides[1][0], configs[1], sides[1][1]),
        columns=columns,
        differs=bool(columns["kinds"] or columns["relations"]) or any(differing.unequal(graph) for graph in graphs),
        graphs=graphs,
    )
    LOG.info("differences recorded exit graphs=%d seconds=%.3f", len(graphs), time.perf_counter() - started)
    return result


@timed("differences.keyed")
def keyed(sides, configs):
    return dict(
        part="differences",
        first=dict(variant=sides[0][1]),
        second=dict(variant=sides[1][1]),
        code=truth.merged([coded(deriver, config, []) for (deriver, _), config in zip(sides, configs)]),
    )


def differences(bank, config, statement, other, other_statement):
    started = time.perf_counter()
    derivers = {}
    sides = [varying.opened(bank, derivers, *held) for held in ((config, statement), (other, other_statement))]
    held = Path(bank) / "descriptions" / "differences"
    result = truth.kept(held, keyed(sides, (config, other)), lambda: recorded(sides, (config, other)), text)
    LOG.info(
        "differences exit first=%s second=%s seconds=%.3f", sides[0][1][:10], sides[1][1][:10], time.perf_counter() - started
    )
    return result


def text(record):
    started = time.perf_counter()
    model = record["time"]["model"]
    named = truth.NAMED[model]
    lines = [
        "difference of the variants %s, the first, and %s, the second, of the bank stored as %s, described at version %d"
        % (record["first"]["variant"], record["second"]["variant"], record["stored_sha256"], record["version"]),
        differing.READING[model],
    ]
    for side in differing.SIDES:
        lines.append("the %s:" % side)
        lines += effect.settled(record[side])
        lines += [effect.entry_text(entry) for entry in record[side]["entries"]]
    held = [
        "%s %s, %s" % (sort, name, ", ".join("the %s %s" % (side, ", ".join(found[side])) for side in differing.SIDES if found[side]))
        for sort, part in (("kind", "kinds"), ("relation", "relations"))
        for name, found in record["columns"][part].items()
    ]
    lines.append("columns of values only one side holds: %s" % ("; ".join(held) or "none"))
    for graph in record["graphs"]:
        lines += differing.graph_text(graph, named)
    LOG.debug("differences text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
