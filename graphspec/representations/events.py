import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from .base import localized, resolved
from .groups import boundary_codes, event_boundaries, event_edges, event_members
from .kernels import check_boundaries
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


@timed("events.carries")
def carries(registry, spec):
    return {}


@timed("events.horizon")
def horizon(spec):
    return spec["events/window"] // 2 + 2, spec["events/periods"]


@timed("events.extended")
def extended(recent, halo, events):
    ext = np.concatenate((recent["events"][halo], events)).astype(np.int32)
    values, (left, right) = localized(ext[:, 0], ext[:, 1])
    ext[:, 0], ext[:, 1] = left, right
    return values, ext


def piece(shown, state):
    started = time.perf_counter()
    own, work, spec, budget, events0 = shown.own, shown.work, shown.spec, shown.budget, shown.before["events"]
    events = own["events"]
    member = event_members(events, spec)
    recent = shown.recent("events")
    known = event_members(recent["events"], spec)
    halo = np.flatnonzero(known & (recent["events"][:, 2] >= 2 * work["first_step"] - spec["events/window"]))
    count = len(halo)
    values, ext = extended(recent, halo, events)
    identity = np.concatenate((recent["identity"][halo], shown.work["identity"]))
    before, after = event_edges(ext, identity, np.concatenate((known[halo], member)), spec, len(values), budget)
    kept = after >= count
    before = resolved(before[kept], count, recent["id"][halo], events0)
    after = events0 + after[kept].astype(np.int64) - count
    budget.retain(len(before), "event graph")
    result = dict(
        event_dag=np.stack((before, after), 1).astype(np.int32).reshape(-1, 2),
        event_member=member,
        event_boundaries=np.zeros((0, 4), np.int32),
    )
    lookback = spec["events/periods"]
    if lookback:
        shift = max(work["group0"] - lookback, 0)
        halo = np.flatnonzero(recent["group"] >= shift)
        count = len(halo)
        values, ext = extended(recent, halo, events)
        edges = event_boundaries(
            ext,
            np.concatenate((known[halo], member)),
            np.concatenate((recent["group"][halo], work["event_group"])).astype(np.int64) - shift,
            np.concatenate((recent["period"][halo], work["event_period"])),
            lookback,
            spec["events/period_window"],
            len(values),
            shown.registry,
            budget,
        ).astype(np.int64)
        edges = edges[edges[:, 1] >= count]
        edges[:, 0] = resolved(edges[:, 0], count, recent["id"][halo], events0)
        edges[:, 1] = events0 + edges[:, 1] - count
        result["event_boundaries"] = edges.astype(np.int32)
    LOG.debug("events piece exit edges=%d seconds=%.4f", len(before), time.perf_counter() - started)
    return result


@timed("events.check")
def check(arrays, name, registry, spec):
    events, member = arrays["events"], arrays["event_member"]
    source, destination = arrays["event_dag"].T
    if np.any(events[source, 2] >= events[destination, 2]) or not (
        member[source].all() and member[destination].all()
    ):
        raise ValueError("event graph of graph %s links a later, simultaneous or excluded event" % name)
    if check_boundaries(
        np.asarray(arrays["event_boundaries"]),
        np.asarray(events),
        np.asarray(member),
        np.asarray(arrays["period"]),
        boundary_codes(registry)[1:],
    ):
        raise ValueError("event boundary edges of graph %s are invalid" % name)


EVENTS = REPRESENTATIONS.register(
    "events", Representation("events", "structure", carries, horizon, piece, None, None, check, 1)
)
