import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .operation import OPERATIONS, Operation, Phase
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


def carried(registry, spec):
    return dict(member=-1, start=0, position=0)


@timed("operators.periods")
def periods(graph, registry, spec, seed, limits, carry, held):
    if spec["time/periods"] != "every":
        return graph
    length, state = spec["time/period_length"], carry.state
    joined = np.asarray(graph.step_graph)
    first = np.r_[0, np.flatnonzero(np.diff(joined)) + 1]
    sizes = np.diff(np.r_[first, graph.steps])
    begins = np.zeros(len(first), np.int64)
    starts = np.zeros(len(first), np.int64)
    if joined[0] == state["member"]:
        begins[0], starts[0] = state["position"], state["start"]
    else:
        starts[0] = state["start"] - (-state["position"] // length)
    for run in range(1, len(first)):
        starts[run] = starts[run - 1] - (-(begins[run - 1] + sizes[run - 1]) // length)
    member = np.repeat(np.arange(len(first)), sizes)
    position = np.arange(graph.steps) - first[member] + begins[member]
    period = (starts[member] + position // length).astype(np.int32)
    carry.state = dict(
        member=int(joined[-1]), start=int(starts[-1]), position=int(begins[-1] + sizes[-1])
    )
    if period.dtype == graph.period.dtype and np.array_equal(period, graph.period):
        return graph
    return graph.replace(period=period)


PERIODS = OPERATIONS.register(
    "graphspec.periods",
    Operation(
        namespace="graphspec",
        name="periods",
        phases=(Phase("periods", periods, carried, None),),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("time",),
        adds=None,
        kinds=None,
        version=5,
        graphs=1,
    ),
)
