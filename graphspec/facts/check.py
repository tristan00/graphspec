import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema import time as clock

from .stored import GROUPS, SPARSE, ranks, tokens

LOG = logging.getLogger(__name__)


def check(graph, registry, splits, observers):
    started = time.perf_counter()
    name, entities, schema = graph.name, len(graph.kinds), registry.schema
    lengths = dict(
        obs=len(graph.obs_since),
        fact=len(graph.fact_since),
        task=len(graph.task_key),
        member=len(graph.member_task),
        target=len(graph.target_task),
        set=len(graph.set_edge),
        inner=len(graph.inner_owner),
        within=len(graph.within_source),
        next=len(graph.next_task),
        concurrent=len(graph.concurrent_task),
        option=len(graph.option_task),
        action=len(graph.action_option),
        role=len(graph.role_action),
    )
    dimensions = dict(
        obs_values=2,
        fact_values=2,
        task_values=2,
        task_signals=3,
        set_values=2,
        inner_values=2,
        within_values=2,
        action_values=2,
    )
    real = ("fact_present", "graph_values", "task_weight", "target_y", "task_discount")
    real += tuple("option_" + name for name in ("sequence", "available", "chosen", "shown", "logged", "drawn"))
    loose = tuple(group + "_" + part for group in GROUPS for part in SPARSE)
    for key, values in graph.arrays.items():
        if not isinstance(values, np.ndarray) or values.ndim != dimensions.get(key, 1):
            raise ValueError("graph %s has an invalid array shape for %s" % (name, key))
        group = key.split("_", 1)[0]
        if group in lengths and key not in loose and len(values) != lengths[group]:
            raise ValueError("graph %s has inconsistent %s row counts" % (name, group))
        if key not in dimensions and key not in real and values.dtype.kind not in "iu":
            raise ValueError("graph %s requires whole numbers in %s" % (name, key))
    if (schema.model in clock.UNITED) != (graph.unit is not None):
        raise ValueError("graph %s states a unit of time exactly when its schema has a clock" % name)
    if graph.unit is not None and graph.unit not in schema.units:
        raise ValueError(
            "graph %s states the unit of time %r, which its schema does not" % (name, graph.unit)
        )
    if len(graph.keys) != entities or len(graph.entity_token) != entities:
        raise ValueError("graph %s requires one key and token per entity" % name)
    if not np.array_equal(graph.entity_token, tokens(registry, graph.kinds, graph.keys)):
        raise ValueError("entity tokens of graph %s do not match its keys" % name)
    if np.any(graph.kinds < 0) or np.any(graph.kinds >= len(registry.nodes)):
        raise ValueError("graph %s names undeclared node types" % name)
    layout = registry.encoding.layout
    if graph.obs_values.shape[1] != len(layout) or graph.task_values.shape[1] != len(layout):
        raise ValueError("graph %s holds values of a width its schema does not state" % name)
    if graph.task_signals.shape[1:] != (registry.extras["signals"], 2):
        raise ValueError("graph %s signal shape does not match its registry" % name)
    for label, values in (("a node", graph.obs_values), ("a task", graph.task_values)):
        infinite = np.argwhere(np.isinf(np.asarray(values)))
        if len(infinite):
            raise ValueError(
                "graph %s states an infinite value of the column %s of %s"
                % (name, layout[int(infinite[0][1])], label)
            )
    for label, values in (
        ("fact about a node", graph.obs_entity),
        ("fact source", graph.fact_source),
        ("fact target", graph.fact_target),
        ("reference", graph.member_source),
        ("reference", np.maximum(graph.member_target, 0)),
        ("set edge", graph.set_edge),
        ("member of a set edge", graph.set_member),
        ("holder of a graph", graph.inner_owner),
        ("actor of an action", graph.action_actor),
        ("actor of a decision", graph.next_actor),
        ("target of an action", graph.role_source),
        ("target of an action", np.maximum(graph.role_target, 0)),
    ):
        if np.any(values < 0) or np.any(values >= entities):
            raise ValueError("a %s of graph %s names an entity outside 0..%d" % (label, name, entities - 1))
    if len(graph.within_source) and (
        max(graph.within_source.max(), graph.within_target.max()) >= len(graph.inner_owner)
        or min(graph.within_source.min(), graph.within_target.min()) < 0
    ):
        raise ValueError("an edge of a graph held in graph %s names a held node outside it" % name)
    if len(graph.inner) != len(graph.inner_owner):
        raise ValueError("graph %s names its held nodes other than it holds them" % name)
    if graph.graph_values.shape != (len(schema.graph),):
        raise ValueError("graph %s does not hold the attributes its schema states" % name)
    joined = np.asarray([kind in schema.sets for kind in registry.nodes], bool)
    if not joined[graph.kinds[graph.set_edge]].all() or joined[graph.kinds[graph.set_member]].any():
        raise ValueError(
            "a member of a set edge of graph %s joins entities of kinds its schema does not" % name
        )
    found = np.asarray(graph.fact_code)
    if np.any(found < 0) or np.any(found >= len(registry.canonical)) or not registry.canonical[found].all():
        raise ValueError("facts of graph %s carry codes that are not canonical relation codes" % name)
    tasks = len(graph.task_key)
    if np.any(graph.task_kind < 0) or np.any(graph.task_kind >= len(schema.tasks["kinds"])):
        raise ValueError("a task of graph %s is of a kind the schema does not state" % name)
    for label, values in (
        ("reference", graph.member_task),
        ("target", graph.target_task),
        ("option", graph.option_task),
        ("following decision", graph.next_task),
        ("following decision", graph.next_other),
        ("concurrent decision", graph.concurrent_task),
        ("concurrent decision", graph.concurrent_other),
    ):
        if np.any(values < 0) or np.any(values >= tasks):
            raise ValueError("a %s of graph %s names a task outside the graph" % (label, name))
    for label, values, count in (
        ("action", graph.action_option, lengths["option"]),
        ("target of an action", graph.role_action, lengths["action"]),
        ("target", np.maximum(graph.target_option, 0), max(lengths["option"], 1)),
    ):
        if np.any(values < 0) or np.any(values >= count):
            raise ValueError("a %s of graph %s names an option or an action outside the graph" % (label, name))
    if len(graph.options) != lengths["option"] or np.any(graph.option_generator >= len(graph.generators)):
        raise ValueError("graph %s names its options or their generators other than it holds them" % name)
    if np.any(graph.target_series < 0) or np.any(graph.target_series >= len(schema.series())):
        raise ValueError("a target of graph %s names a series its schema does not state" % name)
    if np.any(graph.target_node < -1) or np.any(graph.target_node >= len(graph.kinds)):
        raise ValueError("a target of graph %s names a node outside the graph" % name)
    if np.any(graph.task_observer < -1) or np.any(graph.task_observer >= len(observers)):
        raise ValueError("a task of graph %s names an observer the bank does not hold" % name)
    if np.any(graph.task_split < -1) or np.any(graph.task_split >= len(splits)):
        raise ValueError("a task of graph %s names a split the bank does not hold" % name)
    if np.any(np.diff(graph.task_time) < 0) or np.any(np.diff(graph.target_task) < 0):
        raise ValueError("tasks and targets of graph %s are not ordered by their time" % name)
    if np.any(np.diff(graph.member_task) < 0):
        raise ValueError("references of graph %s are not ordered by their task" % name)
    if np.any(graph.target_known <= graph.task_time[graph.target_task]):
        raise ValueError("a target of graph %s is known no later than the time of its task" % name)
    place = ranks(graph.keys)
    for group in GROUPS:
        stamps = graph.stamps(group)
        if np.any(stamps["until"] <= stamps["since"]) or np.any(stamps["replaced"] <= stamps["known"]):
            raise ValueError("a fact of graph %s holds, or is known, over no time at all" % name)
        rows, codes = graph.arrays[group + "_seen_row"], graph.arrays[group + "_seen_observer"]
        if (
            np.any(rows < 0)
            or np.any(rows >= lengths[group])
            or np.any(codes < 0)
            or np.any(codes >= len(observers))
        ):
            raise ValueError("a fact of graph %s names an observer the bank does not hold" % name)
        if group == "obs":
            owner = np.asarray(graph.obs_entity, np.int64)
            facts = dict(seq=graph.obs_seq, code=graph.kinds[owner], source=place[owner], target=place[owner])

            @timed("check.check.named")
            def named(row):
                entity = int(graph.obs_entity[row])
                return "the node %s of the kind %s" % (graph.keys[entity], registry.nodes[graph.kinds[entity]])

        else:
            identity = np.stack((graph.fact_code, graph.fact_source, graph.fact_target, graph.fact_present), 1)
            owner = np.unique(identity.astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
            facts = dict(
                seq=graph.fact_seq,
                code=graph.fact_code,
                source=place[graph.fact_source],
                target=place[graph.fact_target],
            )

            @timed("check.check.named")
            def named(row):
                return "the edge of %s from %s to %s" % (
                    registry.by_code[int(graph.fact_code[row])].name,
                    graph.keys[int(graph.fact_source[row])],
                    graph.keys[int(graph.fact_target[row])],
                )

        clock.untied(name, named, schema.ties, owner, stamps["since"], facts)
    LOG.debug("check exit graph=%s seconds=%.4f", name, time.perf_counter() - started)
