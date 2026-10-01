import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation, Phase
from .parameters import choice
from .rows import fact_keys, prune, pruned, second_of

LOG = logging.getLogger(__name__)
MATCHES = ("keys", "values")


def stated(registry):
    started = time.perf_counter()
    result = (choice("difference/match", MATCHES, "keys"),)
    LOG.debug("difference stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def carried(registry, spec):
    return {"prune/difference": pruned()}


@timed("operators.valued")
def valued(*tables):
    rows = [np.ascontiguousarray(table, np.float32) for table, _ in tables]
    width = rows[0].shape[1] * 4
    found = np.concatenate([held.view(np.uint8).reshape(len(held), width) for held in rows])
    inverse = np.unique(found, axis=0, return_inverse=True)[1].reshape(-1)
    bounds, result = np.cumsum([0] + [len(held) for held in rows]), []
    for (low, high), (_, reference) in zip(zip(bounds[:-1], bounds[1:]), tables):
        reference = np.asarray(reference, np.int64)
        held = np.full(len(reference), -1, np.int64)
        held[reference >= 0] = inverse[low:high][reference[reference >= 0]]
        result.append(held)
    return result


@timed("operators.difference")
def difference(graph, registry, spec, seed, limits, carry, held):
    second = second_of(carry, "graphspec.difference")
    present = np.asarray(second.fact_present, bool)
    source, target = carry.local(second.fact_source[present]), carry.local(second.fact_target[present])
    mine = [graph.fact_step, graph.fact_code, graph.fact_source, graph.fact_target]
    theirs = [second.fact_step[present], second.fact_code[present], source, target]
    if spec["difference/match"] == "values":
        values = valued((graph.value_table, graph.fact_ref), (second.value_table, second.fact_ref[present]))
        mine, theirs = mine + [values[0]], theirs + [values[1]]
    held_mine, held_theirs = fact_keys(registry, mine, theirs)
    removed = np.asarray(graph.fact_present, bool) & np.isin(held_mine, held_theirs)
    return prune(graph, ~removed, carry, "difference")


DIFFERENCE = OPERATIONS.register(
    "graphspec.difference",
    Operation(
        namespace="graphspec",
        name="difference",
        phases=(Phase("difference", difference, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("facts",),
        adds=None,
        version=3,
        graphs=2,
    ),
)
