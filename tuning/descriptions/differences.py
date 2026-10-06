import logging
import time
from pathlib import Path

import numpy as np

from graphspec.descriptions import describe as truth
from graphspec.descriptions import differences as differing
from graphspec.descriptions import effect
from graphspec.engine.segments import padded
from graphspec.infra.files import sha256_json

from . import variant as varying
from .restated import viewed

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


def taken(graph, registry, low, high, steps, width):
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
    sides = [
        taken(graph, registry, low, high, held, width) for (_, graph, registry, low, high), held in zip(both, steps)
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
    result = dict(
        version=truth.VERSION,
        stored_sha256=sides[0][0].source,
        time=truth.modelled(sides[0][0].registry),
        first=varying.headed(sides[0][0], configs[0], sides[0][1]),
        second=varying.headed(sides[1][0], configs[1], sides[1][1]),
        differs=any(differing.unequal(graph) for graph in graphs),
        graphs=graphs,
    )
    LOG.info("differences recorded exit graphs=%d seconds=%.3f", len(graphs), time.perf_counter() - started)
    return result


def differences(bank, config, statement, other, other_statement):
    started = time.perf_counter()
    derivers = {}
    sides = [varying.opened(bank, derivers, *held) for held in ((config, statement), (other, other_statement))]
    key = sha256_json(dict(version=truth.VERSION, first=dict(variant=sides[0][1]), second=dict(variant=sides[1][1])))
    held = Path(bank) / "descriptions" / "differences" / key
    result = truth.kept(held, lambda: recorded(sides, (config, other)), text)
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
    if record["differs"]:
        for graph in record["graphs"]:
            lines += differing.graph_text(graph, named)
    else:
        lines.append(
            "nothing differs: the two hold the same nodes, anchors and edges at %s of each of the %d graphs of the view"
            % ("the step" if named is None else "every " + named["one"], len(record["graphs"]))
        )
    LOG.debug("differences text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
