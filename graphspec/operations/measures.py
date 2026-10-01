import logging
import time

import numpy as np
from numba import njit

from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.schema.schema import ENCODERS

from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import choice, flag
from .rows import made_rows, written

LOG = logging.getLogger(__name__)
MEASURES = ("degree", "components", "cores")
COUNTED = dict(degree="degree", components="components", cores="core numbers")
NUMBER = "number"


@timed("operators.measuring")
def measuring(spec):
    return any(spec["measures/" + name] for name in MEASURES)


def stated(registry):
    started = time.perf_counter()
    reading = tuple(name for name in ENCODERS.names() if NUMBER in ENCODERS.named(name).types)
    result = tuple(flag("measures/" + name, False) for name in MEASURES) + (
        choice("measures/encoder", reading, None, measuring),
    )
    LOG.debug("measures stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.measures_adds")
def adds(registry, settings, entry):
    if not measuring(settings):
        return Added({}, {}, {}, {}, {}, {})
    columns = tuple("%s.%s" % (entry, name) for name in MEASURES if settings["measures/" + name])
    return Added(
        {name: dict(type=NUMBER, encoder=settings["measures/encoder"]) for name in columns},
        {},
        {kind: columns for kind in registry.nodes},
        {},
        {},
        {},
    )


@njit(cache=True)
def structure(obs_entity, obs_step, fact_source, fact_target, fact_step, use, entities):
    observations, facts = len(obs_entity), len(fact_step)
    degree = np.zeros(observations, np.int64)
    size = np.ones(observations, np.int64)
    core = np.zeros(observations, np.int64)
    where = np.full(entities, -1, np.int64)
    stamp = np.full(entities, -1, np.int64)
    low, cursor = 0, 0
    while low < observations:
        moment = np.int64(obs_step[low])
        high = low
        while high < observations and obs_step[high] == moment:
            where[obs_entity[high]] = high
            stamp[obs_entity[high]] = moment
            high += 1
        while cursor < facts and fact_step[cursor] < moment:
            cursor += 1
        end = cursor
        while end < facts and fact_step[end] == moment:
            end += 1
        width = high - low
        left = np.full(end - cursor, -1, np.int64)
        right = np.full(end - cursor, -1, np.int64)
        offsets = np.zeros(width + 1, np.int64)
        parent = np.arange(width)
        for local in range(end - cursor):
            row = cursor + local
            source, target = fact_source[row], fact_target[row]
            if not use[row] or stamp[source] != moment or stamp[target] != moment:
                continue
            a, b = where[source] - low, where[target] - low
            left[local], right[local] = a, b
            degree[low + a] += 1
            offsets[a + 1] += 1
            if b != a:
                degree[low + b] += 1
                offsets[b + 1] += 1
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a < b:
                parent[b] = a
            elif b < a:
                parent[a] = b
        counts = np.zeros(width, np.int64)
        roots = np.empty(width, np.int64)
        for node in range(width):
            root = node
            while parent[root] != root:
                root = parent[root]
            roots[node] = root
            counts[root] += 1
        for node in range(width):
            size[low + node] = counts[roots[node]]
        for node in range(width):
            offsets[node + 1] += offsets[node]
        incident = np.empty(offsets[width], np.int64)
        fill = offsets[:width].copy()
        for local in range(end - cursor):
            a, b = left[local], right[local]
            if a < 0:
                continue
            incident[fill[a]] = b
            fill[a] += 1
            if b != a:
                incident[fill[b]] = a
                fill[b] += 1
        remaining = degree[low:high].copy()
        most = 0
        for node in range(width):
            most = max(most, remaining[node])
        bins = np.zeros(most + 2, np.int64)
        for node in range(width):
            bins[remaining[node]] += 1
        start = 0
        for value in range(most + 1):
            held = bins[value]
            bins[value] = start
            start += held
        position = np.empty(width, np.int64)
        order = np.empty(width, np.int64)
        for node in range(width):
            position[node] = bins[remaining[node]]
            order[position[node]] = node
            bins[remaining[node]] += 1
        for value in range(most, 0, -1):
            bins[value] = bins[value - 1]
        bins[0] = 0
        for place in range(width):
            node = order[place]
            for slot in range(offsets[node], offsets[node + 1]):
                other = incident[slot]
                if other == node or remaining[other] <= remaining[node]:
                    continue
                value = remaining[other]
                first = bins[value]
                swapped = order[first]
                if swapped != other:
                    order[position[other]] = swapped
                    position[swapped] = position[other]
                    order[first] = other
                    position[other] = first
                bins[value] += 1
                remaining[other] -= 1
        for node in range(width):
            core[low + node] = remaining[node]
        low, cursor = high, end
    return degree, size, core


@timed("operators.measures")
def measures(graph, registry, spec, seed, limits, carry, held):
    chosen = [name for name in MEASURES if spec["measures/" + name]]
    if not chosen:
        return graph
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    found = dict(
        zip(
            MEASURES,
            structure(
                entity,
                step,
                np.asarray(graph.fact_source, np.int64),
                np.asarray(graph.fact_target, np.int64),
                np.asarray(graph.fact_step, np.int64),
                np.asarray(graph.fact_present, bool),
                len(graph.kinds),
            ),
        )
    )
    tokens = carry.tokens(entity)
    rows = written(
        registry,
        np.asarray(graph.x_table)[np.asarray(graph.obs_ref, np.int64)],
        tuple("%s.%s" % (carry.entry, name) for name in chosen),
        np.stack([found[name] for name in chosen], 1).astype(np.float32),
        graph.x_table.dtype,
        lambda row: "the node of token %d at the time %d" % (tokens[row], graph.step_time[step[row]]),
    )
    graph, references = made_rows(graph, carry, rows)
    return graph.replace(obs_ref=references.astype(graph.obs_ref.dtype))


MEASURE = OPERATIONS.register(
    "graphspec.measures",
    Operation(
        namespace="graphspec",
        name="measures",
        phases=(Phase("measures", measures, None, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=tuple(
            Count(
                "measures/" + name,
                COUNTED[name],
                lambda registry, spec, name=name: spec["measures/" + name],
                "stored",
                "stored",
            )
            for name in MEASURES
        ),
        changes=("observations",),
        adds=adds,
        kinds=None,
        version=3,
        graphs=1,
    ),
)
