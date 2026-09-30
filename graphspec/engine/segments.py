import logging
import time

import numpy as np

from graphspec.infra.timing import measure
from graphspec.operations.state import EMPTY, ENTITIES, Entities, Segment
from graphspec.representations import base as representations
from graphspec.representations.registry import chosen as representing
from graphspec.views.graph import Dense

from . import join, transform
from .edits import cut, edited
from .stitch import BUFFERED, Appender, joined, stitched

LOG = logging.getLogger(__name__)


def table_sizes(graph):
    if "x_table" in graph.arrays:
        return len(graph.x_table), len(graph.value_table)
    return len(graph.obs_x), len(graph.fact_values) + len(graph.removal_values)


def features(member):
    return member.x_table if isinstance(member, Dense) else member.obs_x


def alone(member):
    kinds = np.asarray(member.kinds).astype(np.int64)
    connected = np.asarray(member.entity_connected) if isinstance(member, Dense) else np.zeros(len(kinds), bool)
    return [np.arange(len(kinds))], kinds, np.asarray(member.entity_token).astype(np.uint64), connected


def unified(members, registry, spec, seed):
    started = time.perf_counter()
    if spec["graphs/join"] == "single":
        result = join.unify(members, registry, spec, seed)
    elif len(members) != 1:
        raise ValueError("graphs that are not joined are derived one at a time; %d were given" % len(members))
    else:
        result = alone(members[0])
    LOG.debug("unified exit members=%d seconds=%.4f", len(members), time.perf_counter() - started)
    return result


class Plan:
    def __init__(self, members, unify, room):
        started = time.perf_counter()
        if type(room) is not int or room < 0:
            raise ValueError(
                "the room a derivation leaves for the entities of its edits is a whole number, found %r" % (room,)
            )
        maps, kinds, tokens, connected = unify
        self.maps, self.entities, self.base = maps, Entities(kinds, tokens, connected), len(kinds) + room
        self.steps = np.r_[0, np.cumsum([member.steps for member in members])].astype(np.int64)
        self.tasks = np.r_[0, np.cumsum([len(member.task_key) for member in members])].astype(np.int64)
        self.periods = np.r_[0, np.cumsum([int(member.period.max(initial=-1)) + 1 for member in members])]
        sizes = np.asarray([table_sizes(member) for member in members], np.int64).reshape(-1, 2)
        self.x = np.r_[0, np.cumsum(sizes[:, 0])].astype(np.int64)
        self.values = np.r_[0, np.cumsum(sizes[:, 1])].astype(np.int64)
        self.name = members[0].name if len(members) == 1 else "joined"
        LOG.info(
            "plan exit seconds=%.3f graphs=%d entities=%d steps=%d",
            time.perf_counter() - started,
            len(members),
            len(kinds),
            int(self.steps[-1]),
        )

    def spans(self, low, high):
        started = time.perf_counter()
        if not 0 <= low < high <= self.steps[-1]:
            raise ValueError(
                "a derivation continues from the step %d to the step %d and its graphs hold the steps from 0 to %d"
                % (low, high, self.steps[-1])
            )
        result = []
        for member in range(len(self.steps) - 1):
            start, stop = max(low, self.steps[member]), min(high, self.steps[member + 1])
            if start < stop:
                result.append((member, int(start - self.steps[member]), int(stop - self.steps[member])))
        LOG.debug("spans exit spans=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result


def opened(plan, member, graph, low, high, edit, entities, added, seen):
    started = time.perf_counter()
    if (low, high) == (0, graph.steps):
        piece, rows, values, first = graph, np.arange(table_sizes(graph)[0]), np.arange(table_sizes(graph)[1]), 0
    else:
        piece, rows, values, first = cut(graph, low, high)
    ids = plan.maps[member]
    x_global, value_global = plan.x[member] + rows, plan.values[member] + values
    fresh = (piece.x_table[:0], piece.value_table[:0])
    if edit is not None:
        tables = len(piece.x_table), len(piece.value_table)
        piece = edited(piece, edit.moved(-low))
        start = len(ids)
        ids = np.r_[ids, len(entities) + np.arange(len(piece.kinds) - start)]
        entities = entities.added(np.asarray(piece.kinds)[start:], np.asarray(piece.entity_token)[start:])
        if len(entities) > plan.base:
            raise ValueError(
                "the edits of a derivation add %d entities, and it was begun with room for %d"
                % (len(entities) - len(plan.entities), plan.base - len(plan.entities))
            )
        fresh = (piece.x_table[tables[0] :], piece.value_table[tables[1] :])
        x_global = np.r_[x_global, plan.x[-1] + added[0] + np.arange(len(fresh[0]))]
        value_global = np.r_[value_global, plan.values[-1] + added[1] + np.arange(len(fresh[1]))]
    glob = np.unique(ids)
    local = np.searchsorted(glob, ids)
    arrays = dict(piece.arrays)
    for name in ENTITIES:
        held = np.asarray(arrays[name])
        arrays[name] = np.where(held >= 0, local[np.maximum(held, 0)], -1).astype(held.dtype)
    lasting = entities.connected[glob].copy()
    if high < graph.steps:
        lasting[local] = True
    arrays.update(
        kinds=entities.kinds[glob].astype(np.int32),
        entity_token=entities.tokens[glob],
        entity_connected=lasting,
        period=(np.asarray(piece.period) + plan.periods[member]).astype(piece.period.dtype),
        step_graph=(np.asarray(piece.step_graph) + member).astype(np.int32),
    )
    dense = Dense(piece.name, [], piece.steps, **arrays)
    observed = np.zeros(len(glob), bool)
    observed[dense.obs_entity] = True
    carry = Segment(
        glob,
        plan.base,
        int(plan.steps[member] + low),
        int(plan.tasks[member] + first),
        np.isin(glob, seen),
        x_global,
        value_global,
        entities,
    )
    LOG.debug(
        "opened exit member=%d low=%d high=%d entities=%d seconds=%.4f",
        member,
        low,
        high,
        len(glob),
        time.perf_counter() - started,
    )
    return carry, dense, np.union1d(seen, glob[observed & lasting]), entities, fresh


class Source:
    def __init__(self, plan, members, registry, steps, spec, groups, seed, limits, partitions):
        self.plan, self.members, self.registry, self.steps, self.spec = plan, members, registry, steps, spec
        self.chosen, self.seed, self.limits, self.partitions = representing(groups), seed, limits, partitions

    def initial(self):
        started = time.perf_counter()
        result = dict(
            seen=EMPTY,
            steps=transform.initial(self.registry, self.steps),
            representations=representations.initial(self.registry, self.spec, self.chosen),
            entities=self.plan.entities,
            added=(0, 0),
        )
        LOG.debug("source initial exit seconds=%.4f", time.perf_counter() - started)
        return result


class Derived:
    def __init__(self, member, low, high, carry, graph, piece, fresh):
        self.member, self.low, self.high, self.carry, self.graph = member, low, high, carry, graph
        self.piece, self.fresh = piece, fresh


def advanced(source, member, graph, low, high, edit, state):
    started = time.perf_counter()
    with measure("segment"):
        carry, dense, seen, entities, fresh = opened(
            source.plan, member, graph, low, high, edit, state["entities"], state["added"], state["seen"]
        )
    derived, steps = transform.derive(
        dense,
        source.registry,
        source.steps,
        source.seed,
        source.limits,
        source.partitions,
        carry,
        state["steps"],
        edit is not None,
    )
    with measure("represent"):
        piece, held = representations.represented(
            carry, derived, source.registry, source.spec, source.chosen, source.seed, source.limits,
            state["representations"],
        )
    found = dict(
        seen=seen,
        steps=steps,
        representations=held,
        entities=entities,
        added=(state["added"][0] + len(fresh[0]), state["added"][1] + len(fresh[1])),
    )
    LOG.info(
        "segment exit member=%d low=%d high=%d seconds=%.3f entities=%d facts=%d",
        member,
        low,
        high,
        time.perf_counter() - started,
        len(carry.glob),
        len(derived.fact_step),
    )
    return Derived(member, low, high, carry, derived, piece, fresh), found


class Derivation:
    def __init__(self, source, parent, time, state, segments):
        self.source, self.parent, self.time, self.state, self.segments = source, parent, time, state, segments

    @classmethod
    def begun(cls, members, registry, steps, spec, groups, seed, limits, partitions, room):
        started = time.perf_counter()
        strangers = [type(member).__name__ for member in members if not isinstance(member, Dense)]
        if strangers:
            raise TypeError(
                "a derivation begins from graphs in the form the operations work on, Graph.dense(); found %s"
                % ", ".join(strangers)
            )
        plan = Plan(members, unified(members, registry, spec, seed), room)
        source = Source(plan, members, registry, steps, spec, groups, seed, limits, partitions)
        result = cls(source, None, 0, source.initial(), ())
        LOG.info("derivation begun exit graphs=%d seconds=%.3f", len(members), time.perf_counter() - started)
        return result

    def continued(self, high, edit):
        started = time.perf_counter()
        spans = self.source.plan.spans(self.time, high)
        if edit is not None and len(spans) != 1:
            raise ValueError(
                "an edit applies to the steps of one graph; the derivation continues from the step %d to the "
                "step %d across %d graphs" % (self.time, high, len(spans))
            )
        state, segments = self.state, []
        for member, low, stop in spans:
            moved = None if edit is None else edit.moved(-int(self.source.plan.steps[member]))
            segment, state = advanced(self.source, member, self.source.members[member], low, stop, moved, state)
            segments.append(segment)
        result = Derivation(self.source, self, high, state, tuple(segments))
        LOG.info(
            "continued exit time=%d high=%d segments=%d seconds=%.3f",
            self.time,
            high,
            len(segments),
            time.perf_counter() - started,
        )
        return result

    def forked(self, high, edits):
        started = time.perf_counter()
        result = tuple(self.continued(high, edit) for edit in edits)
        LOG.info("forked exit forks=%d seconds=%.3f", len(result), time.perf_counter() - started)
        return result

    def chain(self):
        found, node = [], self
        while node is not None:
            found.append(node)
            node = node.parent
        return found[::-1]

    def derivations(self):
        return [segment for node in self.chain() for segment in node.segments]

    def arrays(self):
        started = time.perf_counter()
        source, segments = self.source, self.derivations()
        arrays = joined([segment.piece for segment in segments])
        for name, value in representations.ended(source.chosen, self.state["representations"]).items():
            arrays[name] = np.r_[arrays[name], value]
        table = np.concatenate(
            [np.asarray(features(member)).astype(np.float16, copy=False) for member in source.members]
            + [np.asarray(segment.fresh[0]).astype(np.float16, copy=False) for segment in segments]
        )
        entities = self.state["entities"]
        result = representations.stitched(
            arrays,
            table,
            entities.kinds,
            entities.tokens,
            source.registry,
            source.spec,
            source.chosen,
            source.limits,
            self.time,
            None,
        )
        representations.validate(result, source.plan.name, source.registry, source.spec, source.chosen)
        LOG.info("derivation arrays exit arrays=%d seconds=%.3f", len(result), time.perf_counter() - started)
        return result

    def derived(self):
        started = time.perf_counter()
        source, segments = self.source, self.derivations()
        result = stitched(
            [(segment.carry, segment.graph) for segment in segments],
            self.state["entities"],
            np.concatenate(
                [np.asarray(member.x_table) for member in source.members]
                + [segment.fresh[0] for segment in segments]
            ),
            np.concatenate(
                [np.asarray(member.value_table) for member in source.members]
                + [segment.fresh[1] for segment in segments]
            ),
            self.time,
            source.plan.name,
        )
        LOG.info("derivation derived exit seconds=%.3f", time.perf_counter() - started)
        return result


def whole(graph, registry, steps, seed, limits, partitions):
    started = time.perf_counter()
    plan = Plan([graph], alone(graph), 0)
    carry, dense, _, _, _ = opened(plan, 0, graph, 0, graph.steps, None, plan.entities, (0, 0), EMPTY)
    states = transform.initial(registry, steps)
    derived, _ = transform.derive(dense, registry, steps, seed, limits, partitions, carry, states, False)
    LOG.info("whole exit graph=%s seconds=%.3f", graph.name, time.perf_counter() - started)
    return derived


def opening(graph):
    started = time.perf_counter()
    plan = Plan([graph], alone(graph), 0)
    carry, dense, _, _, _ = opened(plan, 0, graph, 0, graph.steps, None, plan.entities, (0, 0), EMPTY)
    LOG.debug("opening exit graph=%s seconds=%.4f", graph.name, time.perf_counter() - started)
    return carry, dense


def variant(members, registry, steps, spec, groups, seed, limits, partitions, destination):
    started = time.perf_counter()
    plan = Plan(members, unified(members, registry, spec, seed), 0)
    source = Source(plan, members, registry, steps, spec, groups, seed, limits, partitions)
    state = source.initial()
    appender = Appender(destination, BUFFERED)
    try:
        for member, graph in enumerate(members):
            clock = time.perf_counter()
            with measure("dense"):
                dense = graph.dense()
            segment, state = advanced(source, member, dense, 0, dense.steps, None, state)
            with measure("spill"):
                appender.add(segment.piece)
            LOG.info("segment build exit index=%d seconds=%.3f", member, time.perf_counter() - clock)
        appender.add(representations.ended(source.chosen, state["representations"]))
    except BaseException:
        appender.abandon()
        raise
    written = appender.close()
    with measure("stitch"):
        arrays = representations.stitched(
            written,
            np.concatenate([np.asarray(features(member)).astype(np.float16, copy=False) for member in members]),
            plan.entities.kinds,
            plan.entities.tokens,
            registry,
            spec,
            source.chosen,
            limits,
            int(plan.steps[-1]),
            destination,
        )
    representations.validate(arrays, plan.name, registry, spec, source.chosen)
    LOG.info(
        "variant exit seconds=%.3f segments=%d observations=%d events=%d",
        time.perf_counter() - started,
        len(members),
        state["representations"]["base"]["obs"],
        state["representations"]["base"]["events"],
    )
