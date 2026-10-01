import logging
import time

import numpy as np

from graphspec.operations.state import EMPTY

from .base import INFINITE, localized, resolved
from .groups import boundary_codes, coupling_edges, coupling_setting, observation_boundaries
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


def carries(registry, spec):
    return dict(entity=EMPTY, row=EMPTY, step=EMPTY)


def horizon(spec):
    return 0, spec["coupling/periods"]


def tail_length(spec):
    chosen, mode, width, cap, limit = coupling_setting(spec, 0, 0)
    if mode == 0:
        return None if spec["coupling/scheme"] == "all" else cap
    return int(chosen.max())


def coupled(shown, state):
    own, work, spec, budget, obs0 = shown.own, shown.work, shown.spec, shown.budget, shown.before["obs"]
    entity, step = own["entity"].astype(np.int64), own["step"].astype(np.int64)
    halo = np.flatnonzero(np.isin(state["entity"], entity))
    count = len(halo)
    values, (ext_entity,) = localized(np.r_[state["entity"][halo], entity])
    source, target = coupling_edges(
        ext_entity, np.r_[state["step"][halo], step], spec, INFINITE, INFINITE, len(values), budget
    )
    kept = target >= count
    source = resolved(source[kept], count, state["row"][halo], obs0)
    target = obs0 + target[kept].astype(np.int64) - count
    budget.retain(len(source), "temporal coupling")
    connected = work["connected"]
    merged_entity = np.r_[state["entity"], entity[connected]]
    merged_row = np.r_[state["row"], obs0 + np.flatnonzero(connected)]
    merged_step = np.r_[state["step"], step[connected]]
    order = np.lexsort((merged_row, merged_entity))
    merged_entity, merged_row, merged_step = merged_entity[order], merged_row[order], merged_step[order]
    limit = tail_length(spec)
    if limit is not None and len(order):
        ends = np.r_[np.flatnonzero(merged_entity[1:] != merged_entity[:-1]), len(order) - 1]
        from_end = np.repeat(ends, np.diff(np.r_[-1, ends])) - np.arange(len(order))
        keep = from_end < limit
        merged_entity, merged_row, merged_step = merged_entity[keep], merged_row[keep], merged_step[keep]
    state.update(entity=merged_entity, row=merged_row, step=merged_step)
    return np.stack((source, target), 1).astype(np.int32).reshape(-1, 2)


def bounded(shown, lookback, window):
    own, work, obs0 = shown.own, shown.work, shown.before["obs"]
    recent = shown.recent("obs")
    shift = max(work["group0"] - lookback, 0)
    halo = np.flatnonzero((recent["group"] >= shift) & np.isin(recent["entity"], own["entity"]))
    count = len(halo)
    values, (ext_entity,) = localized(np.r_[recent["entity"][halo], own["entity"]])
    result = observation_boundaries(
        ext_entity,
        np.r_[recent["step"][halo], own["step"]].astype(np.int64),
        np.r_[recent["group"][halo], work["group"]].astype(np.int64) - shift,
        np.r_[recent["period"][halo], own["obs_period"]],
        lookback,
        window,
        len(values),
        work["groups"] - shift,
        shown.registry,
        shown.budget,
    ).astype(np.int64)
    result = result[result[:, 1] >= count]
    result[:, 0] = resolved(result[:, 0], count, recent["row"][halo], obs0)
    result[:, 1] = obs0 + result[:, 1] - count
    return result.astype(np.int32)


def piece(shown, state):
    started = time.perf_counter()
    spec = shown.spec
    result = dict(
        coupling=coupled(shown, state),
        coupling_boundaries=bounded(shown, spec["coupling/periods"], spec["coupling/period_window"])
        if spec["coupling/periods"]
        else np.zeros((0, 4), np.int32),
    )
    LOG.debug("coupling piece exit edges=%d seconds=%.4f", len(result["coupling"]), time.perf_counter() - started)
    return result


def check(arrays, name, registry, spec):
    step, entity, period = arrays["step"], arrays["entity"], arrays["period"]
    source, destination = arrays["coupling"].T
    if np.any(step[source] >= step[destination]) or np.any(entity[source] != entity[destination]):
        raise ValueError("coupling edges of graph %s must link an entity to its own earlier observation" % name)
    source, destination, relation, gaps = arrays["coupling_boundaries"].T
    if (
        np.any(entity[source] != entity[destination])
        or np.any(period[step[source]] >= period[step[destination]])
        or not np.isin(relation, boundary_codes(registry)[1:]).all()
        or not np.array_equal(gaps, period[step[destination]] - period[step[source]])
        or np.any(
            step[destination] - step[source] >= spec["coupling/period_window"]
            if spec["coupling/periods"]
            else False
        )
    ):
        raise ValueError("coupling boundary edges of graph %s are invalid" % name)


COUPLING = REPRESENTATIONS.register(
    "coupling", Representation("coupling", "structure", carries, horizon, piece, None, None, check)
)
