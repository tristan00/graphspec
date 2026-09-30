import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.declaration import UNTYPED

from . import kernels
from .operation import OPERATIONS, Operation
from .parameters import choice, holds
from .relations import relating
from .rows import FACTS, grouped, select

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
def untype(graph, registry, carry=None):
    base = np.isin(graph.fact_code, grouped(registry, registry.groups))
    source, target, code, present = graph.fact_source, graph.fact_target, graph.fact_code, graph.fact_present
    added = 0
    if carry is not None:
        live = carry.table("merging/live")
        pair_keys = carry.keys(np.zeros(len(source), np.int64), source, target)
        column = {
            name: live.columns.get(name, np.zeros(0, np.int64))
            for name in ("pair", "source", "target", "code")
        }
        chosen = np.flatnonzero(np.isin(column["pair"], pair_keys))
        added = len(chosen)
        source = np.r_[carry.local(column["source"][chosen]), source].astype(source.dtype)
        target = np.r_[carry.local(column["target"][chosen]), target].astype(target.dtype)
        code = np.r_[column["code"][chosen], code].astype(code.dtype)
        present = np.r_[np.ones(added, bool), present]
        base = np.r_[np.ones(added, bool), base]
    typed, typed_count = kernels.keyed(source, target, code)
    pairs, pair_count = kernels.keyed(source, target, np.zeros(len(base), np.int64))
    kept = kernels.untyped(typed, pairs, present, base, typed_count, pair_count)[added:]
    if carry is not None:
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


@timed("operators.merging")
def merging(graph, registry, spec, seed, limits, carry):
    if holds(spec, "relations/types", "untyped"):
        graph = untype(graph, registry, carry)
    if not holds(spec, "relations/parallel", "merge"):
        return graph
    ids, count = kernels.keyed(
        graph.fact_step.astype(np.int64) * 2 + graph.fact_present,
        graph.fact_code * len(graph.kinds) + graph.fact_source,
        graph.fact_target,
    )
    return select(graph, FACTS, kernels.last_rows(ids, count))


MERGING = OPERATIONS.register(
    "graphspec.merging",
    Operation(
        namespace="graphspec",
        name="merging",
        function=merging,
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        version=1,
    ),
)
