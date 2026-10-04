import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.schema.declaration import UNTYPED

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, flag, holds
from .relations import relating
from .rows import FACTS, PARALLEL, collapsed, extend, grouped, lived, reduced, select
from .state import EMPTY

LOG = logging.getLogger(__name__)
LOOPS = "loops"


@timed("operators.reversible")
def reversible(registry):
    return tuple(
        group
        for group, names in registry.groups.items()
        if len(names) == 1 and not registry.by_name[names[0]].symmetric
    )


def stated(registry):
    started = time.perf_counter()
    related = relating(registry)
    result = (
        choice("relation_forms/types", ("typed", "untyped"), "typed", related),
        choice("relation_forms/parallel", PARALLEL, "keep", related),
        flag(
            "relation_forms/reverse",
            False,
            lambda spec: bool(reversible(registry)) and holds(spec, "relation_forms/types", "typed"),
        ),
        flag("relation_forms/loops", False),
    )
    LOG.debug("relation forms stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("relation_forms.cut")
def cut(registry, name, value):
    return tuple(registry.groups) if name == "relation_forms/types" and value == "untyped" else ()


@timed("relation_forms.whole")
def whole(registry, kinds, relations):
    return {"relation_forms/types": ["typed"]} if set(registry.groups) & set(relations) else {}


@timed("operators.relation_forms_adds")
def adds(registry, settings, entry):
    relations = {}
    if "relation_forms/reverse" in settings and settings["relation_forms/reverse"]:
        for group in reversible(registry):
            edge = registry.by_name[registry.groups[group][0]]
            relations["%s.%s" % (entry, edge.name)] = dict(
                pairs=tuple((target, source) for source, target in edge.pairs),
                directed=True,
                columns=edge.attributes,
            )
    if settings["relation_forms/loops"]:
        relations["%s.%s" % (entry, LOOPS)] = dict(
            pairs=tuple((kind, kind) for kind in registry.nodes), directed=False, columns=()
        )
    return Added({}, {}, {}, relations, {}, {})


@timed("operators.untype")
def untype(graph, registry, carry):
    base = np.isin(graph.fact_code, grouped(registry, registry.groups))
    pair = carry.keys(np.zeros(len(graph.fact_source), np.int64), graph.fact_source, graph.fact_target)
    kept = collapsed(graph, carry, "relation_forms/live", base, pair)
    if base.any():
        graph = graph.replace(
            fact_code=np.where(base, registry.code(UNTYPED), graph.fact_code),
            fact_ref=np.where(base, -1, graph.fact_ref).astype(graph.fact_ref.dtype),
        )
    return select(graph, FACTS, kept)


@timed("relation_forms.living")
def living(registry, spec):
    return {"relation_forms/live": lived()}


@timed("operators.untyping")
def untyping(graph, registry, spec, seed, limits, carry, held):
    if holds(spec, "relation_forms/types", "untyped"):
        return untype(graph, registry, carry)
    return graph


@timed("operators.reversed")
def reversed_facts(graph, registry, entry):
    found = []
    for group in reversible(registry):
        edge = registry.by_name[registry.groups[group][0]]
        rows = np.flatnonzero(graph.fact_code == edge.code)
        found.append((rows, np.full(len(rows), registry.code("%s.%s" % (entry, edge.name)), np.int64)))
    rows = np.concatenate([held[0] for held in found] + [EMPTY])
    return extend(
        graph,
        graph.fact_target[rows],
        graph.fact_source[rows],
        np.concatenate([held[1] for held in found] + [EMPTY]),
        graph.fact_step[rows],
        graph.fact_present[rows],
        graph.fact_ref[rows],
    )


@timed("operators.looped")
def looped(graph, registry, entry):
    rows = np.flatnonzero(~np.asarray(graph.obs_anchor, bool))
    entity = graph.obs_entity[rows]
    return extend(
        graph,
        entity,
        entity,
        np.full(len(rows), registry.code("%s.%s" % (entry, LOOPS)), np.int64),
        graph.obs_step[rows],
        np.ones(len(rows), bool),
        np.full(len(rows), -1, np.int64),
    )


@timed("operators.forming")
def forming(graph, registry, spec, seed, limits, carry, held):
    if "relation_forms/parallel" in spec and spec["relation_forms/parallel"] != "keep":
        held = reduced(held, carry, spec["relation_forms/parallel"], np.ones(len(held.fact_step), bool))
    if "relation_forms/reverse" in spec and spec["relation_forms/reverse"]:
        held = reversed_facts(held, registry, carry.entry)
    if spec["relation_forms/loops"]:
        held = looped(held, registry, carry.entry)
    return held


RELATION_FORMS = OPERATIONS.register(
    "graphspec.relation_forms",
    Operation(
        namespace="graphspec",
        name="relation_forms",
        phases=(Phase("untype", untyping, living, None), Phase("forms", forming, None, None)),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        adds=adds,
        kinds=None,
        version=7,
        graphs=1,
    ),
)
