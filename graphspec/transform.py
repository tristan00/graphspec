import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.sources import digest

from . import operators, representations, sets
from .infra.timing import measure, timed
from .truth import conforms, referring, unobserved, unresolved, verify_dense

LOG = logging.getLogger(__name__)
SOURCE = digest((Path(__file__).parent,))
SEED = 0x6772617068
STAGES = (
    ("periods", operators.periods),
    ("time window", operators.window),
    ("nodes", operators.nodes),
    ("derived relations", operators.derive),
    ("relations", operators.relations),
    ("edge filters", operators.edges),
    ("relation merging", operators.merging),
)
CHANGES = {
    operators.periods: ("time",),
    operators.window: ("observations", "facts"),
    operators.nodes: ("observations", "facts", "references"),
    operators.derive: ("facts",),
    operators.relations: ("facts",),
    operators.edges: ("facts",),
    operators.merging: ("facts",),
    sets.node: ("facts",),
    sets.pairs: ("observations", "facts", "references"),
}
REMOVES = "references"
TREATMENTS = ("stored", "refuses")
EDGES = dict(
    sets="the edges joining sets of the kind %s", about="the edges of the table %s, which are about edges"
)


@timed("transform.filtered")
def filtered(mode):
    return lambda spec: any(
        value == mode
        for name, value in spec.items()
        if name.startswith("nodes/") and name.endswith("/filter")
    )


@timed("transform.capped")
def capped(spec):
    return any(
        value > 0 for name, value in spec.items() if name.startswith("edges/") and name.endswith("/cap")
    )


@timed("transform.projecting")
def projecting(spec):
    return any(value is True for name, value in spec.items() if name.startswith("derive/project:"))


@timed("transform.differs")
def differs(name, value):
    return lambda spec: name in spec and spec[name] != value


COUNTS = {
    "nodes/degree": dict(counts="degree", active=filtered("degree"), sets="stored", about="stored"),
    "nodes/total_degree": dict(
        counts="degree", active=filtered("total_degree"), sets="stored", about="stored"
    ),
    "nodes/core": dict(counts="core numbers", active=filtered("core"), sets="stored", about="stored"),
    "derive/hops": dict(counts="paths", active=differs("derive/hops", 1), sets="stored", about="stored"),
    "derive/projection": dict(counts="paths of two edges", active=projecting, sets="stored", about="stored"),
    "derive/diffusion": dict(
        counts="components and paths",
        active=differs("derive/diffusion", "none"),
        sets="stored",
        about="stored",
    ),
    "derive/rewiring": dict(
        counts="components and degree",
        active=differs("derive/rewiring", "none"),
        sets="stored",
        about="stored",
    ),
    "edges/cap": dict(counts="degree", active=capped, sets="stored", about="stored"),
}


def counting(registry, spec):
    started = time.perf_counter()
    met = dict(sets=tuple(registry.schema.sets), about=tuple(registry.schema.about()))
    for name, entry in COUNTS.items():
        if not any(met.values()) or not entry["active"](spec):
            continue
        for part, found in met.items():
            for kind in found:
                if part not in entry or entry[part] not in TREATMENTS:
                    raise ValueError(
                        "the operation %s counts %s and states no treatment of %s"
                        % (name, entry["counts"], EDGES[part] % kind)
                    )
                if entry[part] == "refuses":
                    raise ValueError(
                        "the operation %s counts %s and refuses %s"
                        % (name, entry["counts"], EDGES[part] % kind)
                    )
    LOG.debug("counting exit operations=%d seconds=%.4f", len(COUNTS), time.perf_counter() - started)


def resolving(graph, registry):
    started = time.perf_counter()
    member, entity, moments, _ = referring(graph, registry)
    result = ~unobserved(len(graph.kinds), graph.obs_entity, graph.obs_step, entity, moments)
    LOG.debug(
        "resolving exit graph=%s ends=%d seconds=%.4f", graph.name, len(member), time.perf_counter() - started
    )
    return member, result


def kept(name, stage, before, graph, registry):
    started = time.perf_counter()
    if REMOVES not in CHANGES[stage] and "observations" in CHANGES[stage]:
        _, held = resolving(before, registry)
        member, found = resolving(graph, registry)
        lost = held & ~found
        if lost.any():
            raise ValueError(
                "the stage %s removes the node of %s and does not state that it may remove a referenced node"
                % (name, unresolved(graph, registry, member[np.flatnonzero(lost)[0]]))
            )
    LOG.debug("kept exit stage=%s graph=%s seconds=%.4f", name, graph.name, time.perf_counter() - started)


@timed("graph_transform")
def derive(graph, registry, spec, limits, carry=None, known=None, stages=STAGES, check=True):
    started, given = time.perf_counter(), graph
    counting(registry, spec)
    for name, stage in stages:
        clock = time.perf_counter()
        options = dict(limits=limits) if stage in (operators.window, operators.derive) else {}
        with measure(name):
            before, graph = graph, stage(graph, registry, spec, SEED, carry=carry, **options)
            if graph is not before:
                kept(name, stage, before, graph, registry)
            if check and graph is not before:
                conforms(graph, registry, tuple(part for part in CHANGES[stage] if part != REMOVES))
        LOG.info(
            "stage exit stage=%s graph=%s seconds=%.3f facts=%d observations=%d",
            name,
            graph.name,
            time.perf_counter() - clock,
            len(graph.fact_step),
            len(graph.obs_step),
        )
    if check and graph is not given:
        with measure("verify"):
            verify_dense(graph, registry, known)
    LOG.info("derive exit graph=%s seconds=%.3f", graph.name, time.perf_counter() - started)
    return graph


@timed("materialize")
def materialize(graph, registry, spec, groups, limits):
    started = time.perf_counter()
    derived = derive(graph, registry, spec, limits)
    arrays = representations.build(derived, registry, spec, groups, SEED, limits)
    representations.validate(arrays, derived, registry, spec, groups)
    LOG.info(
        "materialize exit graph=%s seconds=%.3f arrays=%d",
        graph.name,
        time.perf_counter() - started,
        len(arrays),
    )
    return arrays
