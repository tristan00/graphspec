import logging
import time

import numpy as np

from graphspec.infra.hashing import hashed

from .base import localized, resolved
from .groups import PERIOD_LOOKBACK, neighbor_lists, neighbor_members, walk_arrays
from .neighbors import adjacency
from .registry import REPRESENTATIONS, Representation

LOG = logging.getLogger(__name__)


def carries(registry, spec):
    return {}


def horizon(spec):
    return spec["walks/window"] + 2, PERIOD_LOOKBACK


def piece(shown, state):
    started = time.perf_counter()
    own, work, spec, events0 = shown.own, shown.work, shown.spec, shown.before["events"]
    local = own["member_row"].astype(np.int64) - shown.before["rows"]
    moments = own["row_step"].astype(np.int64)[local]
    roots = own["member_entity"].astype(np.int64)
    first, last = work["starts"], work["ends"]
    prior = work["row_group"][local][:, None].astype(np.int64) - np.arange(1, PERIOD_LOOKBACK + 1)
    floor = np.inf
    if len(moments):
        floor = min(floor, int(np.maximum(0, (moments - spec["walks/window"] + 1) * 2).min()))
    if (prior >= 0).any():
        floor = min(floor, int(2 * first[prior[prior >= 0]].min()))
    recent = shown.recent("events")
    halo = np.flatnonzero(neighbor_members(recent["events"], spec) & (recent["events"][:, 2] >= floor))
    count = len(halo)
    ext = np.r_[recent["events"][halo], own["events"]].astype(np.int32)
    identity = np.r_[recent["identity"][halo], work["identity"]]
    values, (left, right, local_roots) = localized(ext[:, 0], ext[:, 1], roots.reshape(-1))
    ext[:, 0], ext[:, 1] = left, right
    ids, offsets = neighbor_lists(ext, neighbor_members(ext, spec), len(values), identity)
    held = moments - shown.carry.step0
    keys = hashed(
        np.where(roots >= 0, shown.carry.entities.tokens[np.maximum(roots, 0)], 0),
        np.asarray(shown.graph.step_time)[held],
        np.asarray(shown.graph.step_token)[held],
    )
    result = walk_arrays(
        ids,
        offsets,
        ext,
        identity,
        local_roots.reshape(roots.shape),
        keys,
        moments,
        prior,
        first[prior.clip(0)].astype(np.int64),
        last[prior.clip(0)].astype(np.int64),
        spec,
        shown.seed,
        shown.budget,
    )
    for held in ("walks", "period_walks"):
        walked = result[held].astype(np.int64)
        result[held] = np.where(
            walked < 0, -1, resolved(walked, count, recent["id"][halo], events0)
        ).astype(np.int32)
    LOG.debug("walks piece exit roots=%d seconds=%.4f", len(roots), time.perf_counter() - started)
    return result


def check(arrays, name, registry, spec):
    events = arrays["events"]
    moments = np.asarray(arrays["row_step"], np.int64)[np.asarray(arrays["member_row"], np.int64)]
    for held, limit in (("walks", moments[:, None] * 2), ("period_walks", moments[:, None, None] * 2)):
        walks = arrays[held]
        times = np.where(walks >= 0, events[np.maximum(walks, 0), 2], -1)
        if np.any(times[..., 0] > limit) or np.any((np.diff(times, axis=-1) >= 0) & (times[..., 1:] >= 0)):
            raise ValueError("%s of graph %s are not strictly causal" % (held, name))


WALKS = REPRESENTATIONS.register("walks", Representation("walks", carries, horizon, piece, None, adjacency, check))
