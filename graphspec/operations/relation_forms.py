import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.schema.declaration import UNTYPED

from . import kernels
from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, flag, holds
from .relations import relating
from .rows import FACTS, extend, grouped, made_values, select
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
AGGREGATES = ("sum", "mean", "min", "max")
PARALLEL = ("keep", "first", "last") + AGGREGATES
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
    return Added({}, {}, {}, relations, {})


@timed("operators.untype")
def untype(graph, registry, carry):
    base = np.isin(graph.fact_code, grouped(registry, registry.groups))
    source, target, code, present = graph.fact_source, graph.fact_target, graph.fact_code, graph.fact_present
    live = carry.state["relation_forms/live"]
    pair_keys = carry.keys(np.zeros(len(source), np.int64), source, target)
    chosen = np.flatnonzero(np.isin(live.columns["pair"], pair_keys))
    added = len(chosen)
    source = np.r_[carry.local(live.columns["source"][chosen]), source].astype(source.dtype)
    target = np.r_[carry.local(live.columns["target"][chosen]), target].astype(target.dtype)
    code = np.r_[live.columns["code"][chosen], code].astype(code.dtype)
    present = np.r_[np.ones(added, bool), present]
    base = np.r_[np.ones(added, bool), base]
    typed, typed_count = kernels.keyed(source, target, code)
    pairs, pair_count = kernels.keyed(source, target, np.zeros(len(base), np.int64))
    kept = kernels.untyped(typed, pairs, present, base, typed_count, pair_count)[added:]
    rows = np.flatnonzero(base & graph.entity_connected[source] & graph.entity_connected[target])
    last = np.full(typed_count, -1, np.int64)
    np.maximum.at(last, typed[rows], rows)
    final = last[last >= 0]
    keys = carry.keys(code[final], source[final], target[final])
    state = present[final]
    carry.state["relation_forms/live"] = live.replaced(
        keys,
        keys[state],
        pair=carry.keys(np.zeros(int(state.sum()), np.int64), source[final][state], target[final][state]),
        source=carry.glob[source[final][state]],
        target=carry.glob[target[final][state]],
        code=code[final][state],
    )
    base = base[added:]
    if base.any():
        graph = graph.replace(
            fact_code=np.where(base, registry.code(UNTYPED), graph.fact_code),
            fact_ref=np.where(base, -1, graph.fact_ref).astype(graph.fact_ref.dtype),
        )
    return select(graph, FACTS, kept)


def living(registry, spec):
    return {"relation_forms/live": Table(EMPTY, pair=EMPTY, source=EMPTY, target=EMPTY, code=EMPTY)}


@timed("operators.untyping")
def untyping(graph, registry, spec, seed, limits, carry, held):
    if holds(spec, "relation_forms/types", "untyped"):
        return untype(graph, registry, carry)
    return graph


@timed("operators.parallel")
def parallel(graph, carry, aggregate):
    ids, count = kernels.keyed(
        graph.fact_step.astype(np.int64) * 2 + graph.fact_present,
        graph.fact_code * len(graph.kinds) + graph.fact_source,
        graph.fact_target,
    )
    if aggregate == "first":
        return select(graph, FACTS, kernels.first_rows(ids, count))
    kept = kernels.last_rows(ids, count)
    if aggregate == "last":
        return select(graph, FACTS, kept)
    reference = np.asarray(graph.fact_ref, np.int64)
    valued = np.flatnonzero(np.asarray(graph.fact_present, bool) & (reference >= 0))
    sizes = np.bincount(ids[valued], minlength=count)
    valued = valued[sizes[ids[valued]] > 1]
    if not len(valued):
        return select(graph, FACTS, kept)
    owners = np.unique(ids[valued])
    place = np.searchsorted(owners, ids[valued])
    values = np.asarray(graph.value_table, np.float64)[reference[valued]]
    known = ~np.isnan(values)
    if aggregate in ("sum", "mean"):
        found = np.zeros((len(owners), values.shape[1]))
        np.add.at(found, place, np.where(known, values, 0.0))
        seen = np.zeros(found.shape)
        np.add.at(seen, place, known)
        found = np.divide(found, np.where(aggregate == "mean", seen, 1.0), out=np.full(found.shape, np.nan), where=seen > 0)
    else:
        found = np.full((len(owners), values.shape[1]), np.nan)
        (np.fmin if aggregate == "min" else np.fmax).at(found, place, values)
    graph, rows = made_values(graph, carry, found)
    last = np.flatnonzero(kept)
    chosen = last[np.isin(ids[last], owners)]
    reference = reference.copy()
    reference[chosen] = rows[np.searchsorted(owners, ids[chosen])]
    graph = graph.replace(fact_ref=reference.astype(graph.fact_ref.dtype))
    return select(graph, FACTS, kept)


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
        held = parallel(held, carry, spec["relation_forms/parallel"])
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
        version=1,
        graphs=1,
    ),
)
