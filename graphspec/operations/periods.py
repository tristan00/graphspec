import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation
from .parameters import choice, equals, number

LOG = logging.getLogger(__name__)


def stated(registry):
    started = time.perf_counter()
    ungrouped = "truth" if registry.schema.tasks["grouping"] is not None else "steps"
    result = (
        choice("time/periods", (ungrouped, "every"), ungrouped),
        number(
            "time/period_length",
            1 if registry.extras["events"] else 2,
            16,
            4,
            equals("time/periods", "every"),
        ),
    )
    LOG.debug("periods stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("operators.periods")
def periods(graph, registry, spec, seed, limits, carry):
    if spec["time/periods"] != "every":
        return graph
    length = spec["time/period_length"]
    joined = graph.step_graph
    first = np.r_[0, np.flatnonzero(np.diff(joined)) + 1]
    sizes = np.diff(np.r_[first, graph.steps])
    offsets = np.r_[0, np.cumsum(-(-sizes // length))][:-1]
    member = np.repeat(np.arange(len(first)), sizes)
    position = np.arange(graph.steps) - first[member]
    base = 0 if carry is None else carry.period0
    period = (base + offsets[member] + position // length).astype(np.int32)
    if period.dtype == graph.period.dtype and np.array_equal(period, graph.period):
        return graph
    return graph.replace(period=period)


PERIODS = OPERATIONS.register(
    "graphspec.periods",
    Operation(
        namespace="graphspec",
        name="periods",
        function=periods,
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("time",),
        version=1,
    ),
)
