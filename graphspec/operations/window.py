import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed
from graphspec.infra.workspace import concatenate, take

from . import kernels
from .operation import OPERATIONS, Operation
from .parameters import choice, holds, number
from .rows import FACTS, OBSERVATIONS, counted, extend, fact_bytes, member_end
from .state import FOREVER

LOG = logging.getLogger(__name__)


def periodic(spec):
    return spec["time/periods"] == "truth" or (
        spec["time/periods"] == "every" and spec["time/period_length"] > 1
    )


def by_period(spec):
    return holds(spec, "time/unit", "period")


def stated(registry):
    started = time.perf_counter()
    result = (
        choice("time/scheme", ("window", "state"), "window"),
        choice(
            "time/unit",
            ("step", "period"),
            "step",
            lambda spec: spec["time/scheme"] == "window" and periodic(spec),
        ),
        number("time/window", 1, 8, 1, lambda spec: spec["time/scheme"] == "window" and not by_period(spec)),
        number("time/past_periods", 0, 3, 0, by_period),
    )
    LOG.debug("window stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def reach(graph, spec, start, lasting):
    if spec["time/scheme"] == "state":
        reached = np.full(len(start), graph.steps - 1, np.int64)
    elif by_period(spec):
        group = np.cumsum(np.r_[0, np.diff(graph.period) != 0])
        reached = np.searchsorted(group, group[start] + spec["time/past_periods"], "right") - 1
    else:
        reached = start + spec["time/window"] - 1
    return np.where(lasting, reached, np.minimum(reached, member_end(graph)[start] - 1))


def groups_of(graph):
    return np.cumsum(np.r_[0, np.diff(graph.period) != 0])


def horizon(graph, spec, carry, start):
    if spec["time/scheme"] == "state":
        return np.full(len(start), FOREVER, np.int64)
    if by_period(spec):
        return carry.group0 + groups_of(graph)[start] + spec["time/past_periods"]
    return carry.step0 + np.asarray(start, np.int64) + spec["time/window"] - 1


def beyond(graph, spec, carry, end):
    if spec["time/scheme"] == "state":
        return np.ones(len(end), bool)
    if by_period(spec):
        return end > carry.group0 + groups_of(graph)[-1]
    return end >= carry.step0 + graph.steps


def carried_reach(graph, spec, carry, end):
    if spec["time/scheme"] == "state":
        return np.full(len(end), graph.steps - 1, np.int64)
    if by_period(spec):
        return np.searchsorted(groups_of(graph), end - carry.group0, "right") - 1
    return end - carry.step0


def held(graph, carry):
    items = carry.state.get("window")
    if items is None:
        return graph, None, None
    obs, facts = items["obs"], items["facts"]
    added_obs, added_facts = len(obs["entity"]), len(facts["source"])
    tables = len(graph.x_table), len(graph.value_table)
    carry.x_global = np.r_[carry.x_global, obs["ref"]]
    carry.value_global = np.r_[carry.value_global, facts["ref"]]

    def prefixed(name, values):
        return np.r_[np.asarray(values), graph.arrays[name]].astype(graph.arrays[name].dtype)

    graph = graph.replace(
        x_table=np.concatenate((graph.x_table, obs["x"].astype(graph.x_table.dtype))),
        value_table=np.concatenate((graph.value_table, facts["values"].astype(graph.value_table.dtype))),
        obs_entity=prefixed("obs_entity", carry.local(obs["entity"])),
        obs_step=prefixed("obs_step", np.full(added_obs, -1)),
        obs_ref=prefixed("obs_ref", tables[0] + np.arange(added_obs)),
        obs_anchor=prefixed("obs_anchor", obs["anchor"]),
        fact_source=prefixed("fact_source", carry.local(facts["source"])),
        fact_target=prefixed("fact_target", carry.local(facts["target"])),
        fact_code=prefixed("fact_code", facts["code"]),
        fact_step=prefixed("fact_step", np.full(added_facts, -1)),
        fact_present=prefixed("fact_present", np.ones(added_facts, bool)),
        fact_ref=prefixed("fact_ref", np.where(facts["valued"], tables[1] + np.arange(added_facts), -1)),
    )
    return graph, obs, facts


@timed("operators.window")
def window(graph, registry, spec, seed, limits, carry):
    if spec["time/scheme"] == "window" and not by_period(spec) and spec["time/window"] == 1:
        return graph
    steps = graph.steps
    carried_obs = carried_facts = None
    if carry is not None:
        graph, carried_obs, carried_facts = held(graph, carry)
    pseudo_obs = 0 if carried_obs is None else len(carried_obs["entity"])
    pseudo_facts = 0 if carried_facts is None else len(carried_facts["source"])
    order = np.lexsort((graph.obs_step, graph.obs_entity))
    entity, step = graph.obs_entity[order], graph.obs_step[order].astype(np.int64)
    later = np.r_[entity[1:] == entity[:-1], False]
    limit = np.where(later, np.r_[step[1:], 0], steps)
    reached = reach(graph, spec, np.maximum(step, 0), graph.entity_connected[entity])
    pseudo = order < pseudo_obs
    if pseudo.any():
        reached[pseudo] = carried_reach(graph, spec, carry, carried_obs["end"][order[pseudo]])
    counts = np.maximum(np.minimum(reached, limit - 1) - step, 0)
    if carry is not None:
        rows = order[~later & graph.entity_connected[entity]]
        end = horizon(graph, spec, carry, np.maximum(graph.obs_step[rows], 0))
        inherited = rows < pseudo_obs
        if inherited.any():
            end[inherited] = carried_obs["end"][rows[inherited]]
        rows, end = rows[beyond(graph, spec, carry, end)], end[beyond(graph, spec, carry, end)]
        kept_obs = dict(
            entity=carry.glob[graph.obs_entity[rows]],
            end=end,
            x=np.asarray(graph.x_table)[graph.obs_ref[rows]],
            ref=carry.x_global[graph.obs_ref[rows]],
            anchor=graph.obs_anchor[rows],
        )
    budget = GraphBudget(limits, fact_bytes(graph))
    budget.pairs(int(counts.sum()), "time window observations")
    budget.workspace(
        int(counts.sum()) * sum(graph.arrays[name].dtype.itemsize + 8 for name in OBSERVATIONS),
        "time window observations",
    )
    origin = np.repeat(order, counts)
    copies = {
        name: concatenate((graph.arrays[name], take(graph.arrays[name], origin))) for name in OBSERVATIONS
    }
    copies["obs_step"][len(graph.obs_step) :] += (counted(counts) + 1).astype(np.int32)
    arrangement = np.lexsort((np.arange(len(copies["obs_step"])), copies["obs_step"]))
    graph = graph.replace(**{name: take(value, arrangement)[pseudo_obs:] for name, value in copies.items()})
    ids, count = kernels.keyed(graph.fact_source, graph.fact_target, graph.fact_code)
    limit = kernels.carried(ids, graph.fact_step, graph.fact_present, count, steps)
    present = np.flatnonzero(graph.fact_present)
    start = graph.fact_step[present].astype(np.int64)
    lasting = (
        graph.entity_connected[graph.fact_source[present]]
        & graph.entity_connected[graph.fact_target[present]]
    )
    reached = reach(graph, spec, np.maximum(start, 0), lasting)
    if pseudo_facts:
        reached[:pseudo_facts] = carried_reach(graph, spec, carry, carried_facts["end"])
    counts = np.maximum(np.minimum(reached, limit[present] - 1) - start, 0)
    if carry is not None:
        final = lasting & (limit[present] == steps)
        rows = present[final]
        end = horizon(graph, spec, carry, np.maximum(start[final], 0))
        stamp, rank = carry.step0 + start[final], rows.astype(np.int64)
        inherited = rows < pseudo_facts
        if inherited.any():
            for values, source in ((end, "end"), (stamp, "stamp"), (rank, "rank")):
                values[inherited] = carried_facts[source][rows[inherited]]
        wanted = beyond(graph, spec, carry, end)
        rows, end, stamp, rank = rows[wanted], end[wanted], stamp[wanted], rank[wanted]
        arranged = np.lexsort((rank, stamp))
        rows, end, stamp, rank = rows[arranged], end[arranged], stamp[arranged], rank[arranged]
        reference = graph.fact_ref[rows]
        values = np.full((len(rows), graph.value_table.shape[1]), np.nan, graph.value_table.dtype)
        values[reference >= 0] = graph.value_table[reference[reference >= 0]]
        carry.state["window"] = dict(
            obs=kept_obs,
            facts=dict(
                source=carry.glob[graph.fact_source[rows]],
                target=carry.glob[graph.fact_target[rows]],
                code=graph.fact_code[rows],
                end=end,
                stamp=stamp,
                rank=rank,
                values=values,
                valued=reference >= 0,
                ref=np.where(reference >= 0, carry.value_global[np.maximum(reference, 0)], -1),
            ),
        )
    budget.pairs(int(counts.sum()), "time window facts")
    origin = np.repeat(present, counts)
    graph = extend(
        graph,
        graph.fact_source[origin],
        graph.fact_target[origin],
        graph.fact_code[origin],
        (graph.fact_step[origin].astype(np.int64) + counted(counts) + 1).astype(np.int32),
        np.ones(len(origin), bool),
        graph.fact_ref[origin],
    )
    if pseudo_facts:
        graph = graph.replace(**{name: graph.arrays[name][pseudo_facts:] for name in FACTS})
    return graph


WINDOW = OPERATIONS.register(
    "graphspec.window",
    Operation(
        namespace="graphspec",
        name="window",
        function=window,
        parameters=stated,
        reads=(),
        splits=(),
        requires=(),
        changes=("observations", "facts"),
        version=1,
    ),
)
