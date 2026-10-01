import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.declaration import UNTYPED

from . import kernels
from .operation import OPERATIONS, Operation, Phase
from .parameters import choice, holds
from .relations import relating
from .rows import FACTS, grouped, select
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)


def stated(registry):
    started = time.perf_counter()
    related = relating(registry)
    result = (
        choice("relations/types", ("typed", "untyped"), "typed", related),
        choice("relations/parallel", ("keep", "merge"), "keep", related),
    )
    LOG.debug("merging stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.untype")
def untype(graph, registry, carry):
    base = np.isin(graph.fact_code, grouped(registry, registry.groups))
    source, target, code, present = graph.fact_source, graph.fact_target, graph.fact_code, graph.fact_present
    live = carry.state["merging/live"]
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
    carry.state["merging/live"] = live.replaced(
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
    return {"merging/live": Table(EMPTY, pair=EMPTY, source=EMPTY, target=EMPTY, code=EMPTY)}


@timed("operators.untyping")
def untyping(graph, registry, spec, seed, limits, carry, held):
    if holds(spec, "relations/types", "untyped"):
        return untype(graph, registry, carry)
    return graph


@timed("operators.merging")
def merging(graph, registry, spec, seed, limits, carry, held):
    if not holds(spec, "relations/parallel", "merge"):
        return held
    ids, count = kernels.keyed(
        held.fact_step.astype(np.int64) * 2 + held.fact_present,
        held.fact_code * len(held.kinds) + held.fact_source,
        held.fact_target,
    )
    return select(held, FACTS, kernels.last_rows(ids, count))


MERGING = OPERATIONS.register(
    "graphspec.merging",
    Operation(
        namespace="graphspec",
        name="merging",
        phases=(Phase("untype", untyping, living, None), Phase("merge", merging, None, None)),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        version=3,
        graphs=1,
    ),
)
