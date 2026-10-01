import logging
import time

import numpy as np

from graphspec.infra.timing import measure, timed
from graphspec.infra.workspace import gathered
from graphspec.operations.state import EMPTY, ENTITIES, SECOND, Entities, Second, Segment
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


@timed("segments.observed_rows")
def observed_rows(member):
    entity = np.asarray(member.obs_entity, np.int64)
    if isinstance(member, Dense):
        return entity, np.asarray(member.x_table)[np.asarray(member.obs_ref, np.int64)]
    return entity, np.asarray(member.obs_x)


@timed("segments.padded")
def padded(values, width):
    values = np.asarray(values)
    if values.shape[1] == width:
        return values
    if values.shape[1] > width:
        raise ValueError(
            "rows of values of facts %d wide are read by a registry whose relations hold at most %d values"
            % (values.shape[1], width)
        )
    return np.concatenate((values, np.full((len(values), width - values.shape[1]), np.nan, values.dtype)), 1)


@timed("segments.place")
def place(table_global, table, top):
    table_global = np.asarray(table_global, np.int64)
    made = np.flatnonzero(table_global < 0)
    if not len(made):
        return table_global, np.asarray(table)[:0]
    result = table_global.copy()
    result[made] = top + np.arange(len(made))
    return result, np.asarray(table)[made]


def placed(carry, graph, tops):
    started = time.perf_counter()
    carry.x_global, rows = place(carry.x_global, graph.x_table, tops[0])
    carry.value_global, values = place(carry.value_global, graph.value_table, tops[1])
    LOG.debug(
        "placed exit graph=%s rows=%d values=%d seconds=%.4f",
        graph.name,
        len(rows),
        len(values),
        time.perf_counter() - started,
    )
    return rows, values


def alone(member):
    kinds = np.asarray(member.kinds).astype(np.int64)
    connected = np.asarray(member.entity_connected) if isinstance(member, Dense) else np.zeros(len(kinds), bool)
    return [np.arange(len(kinds))], kinds, np.asarray(member.entity_token).astype(np.uint64), connected, join.apart


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


def paired(steps, beside):
    started = time.perf_counter()
    named = sorted({step.graph for step in steps if step.graph is not None})
    if type(beside) is not dict or sorted(beside) != named:
        raise ValueError(
            "the steps of the derivation take %s as second graphs, and it was given %s"
            % (
                ", ".join(named) or "none",
                (", ".join(sorted(beside)) or "none") if type(beside) is dict else repr(beside),
            )
        )
    LOG.debug("paired exit graphs=%d seconds=%.4f", len(named), time.perf_counter() - started)
    return beside


def matched(kinds, tokens, numbers, second):
    started = time.perf_counter()
    held = np.stack((np.asarray(kinds, np.int64), np.asarray(tokens).astype(np.uint64).view(np.int64)), 1)
    theirs = np.stack(
        (
            np.asarray(second.kinds).astype(np.int64),
            np.asarray(second.entity_token).astype(np.uint64).view(np.int64),
        ),
        1,
    )
    _, inverse = np.unique(np.r_[held, theirs], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    place = np.full(int(inverse.max(initial=-1)) + 1, -1, np.int64)
    place[inverse[: len(held)]] = numbers
    result = place[inverse[len(held) :]]
    LOG.debug(
        "matched exit graph=%s met=%d seconds=%.4f", second.name, int((result >= 0).sum()), time.perf_counter() - started
    )
    return result


@timed("segments.numbered")
def numbered(found, seconds, joins, total):
    fresh = [np.flatnonzero(numbers < 0) for numbers in found]
    kinds = np.concatenate([np.asarray(second.kinds)[place] for second, place in zip(seconds, fresh)] + [EMPTY])
    tokens = np.concatenate(
        [np.asarray(second.entity_token)[place].astype(np.uint64) for second, place in zip(seconds, fresh)]
        + [EMPTY.astype(np.uint64)]
    )
    kinds, shared = kinds.astype(np.int64), joins(kinds, tokens)
    keys = np.stack((np.where(shared, kinds, -1 - np.arange(len(kinds))), np.where(shared, tokens.view(np.int64), 0)), 1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    assigned, offsets, result = total + rank[inverse.reshape(-1)], np.cumsum([0] + [len(place) for place in fresh]), []
    for numbers, place, low in zip(found, fresh, offsets):
        numbers = numbers.copy()
        numbers[place] = assigned[low : low + len(place)]
        result.append(numbers)
    first = np.sort(first)
    return result, kinds[first], tokens[first]


class Plan:
    def __init__(self, members, beside, unify, room, registry):
        started = time.perf_counter()
        if type(room) is not int or room < 0:
            raise ValueError(
                "the room a derivation leaves for the entities of its edits is a whole number, found %r" % (room,)
            )
        maps, kinds, tokens, connected, joins = unify
        self.registry = registry
        self.seconds = {
            name: [self.widened(second if isinstance(second, Dense) else second.dense()) for second in beside[name]]
            for name in sorted(beside)
        }
        made = self.added_entities(members)
        count = sum(len(found) for _, found in made)
        self.made = np.arange(count)
        maps = [np.asarray(own, np.int64) + count for own in maps]
        codes = np.concatenate([np.full(len(found), code, np.int64) for code, found in made] + [EMPTY])
        kinds = [np.r_[codes, np.asarray(kinds, np.int64)]]
        tokens = [np.concatenate([found for _, found in made] + [np.asarray(tokens).astype(np.uint64)])]
        graphs = sum(int(member.step_graph[-1]) + 1 if isinstance(member, Dense) else 1 for member in members)
        connected = np.r_[np.full(count, graphs > 1), np.asarray(connected, bool)]
        self.second_maps, total = {}, len(kinds[0])
        for name, seconds in self.seconds.items():
            if len(seconds) != len(members):
                raise ValueError(
                    "the second graph %s is given beside %d graphs, and the derivation holds %d"
                    % (name, len(seconds), len(members))
                )
            found = []
            for member, second, own in zip(members, seconds, maps):
                if second.steps != member.steps or not np.array_equal(
                    np.asarray(second.step_time), np.asarray(member.step_time)
                ):
                    raise ValueError(
                        "the second graph %s shows other times than the graph %s it is given beside; a second "
                        "graph is shown at the steps of the graph beside it" % (second.name, member.name)
                    )
                found.append(matched(kinds[0][own], tokens[0][own], own, second))
            self.second_maps[name], own_kinds, own_tokens = numbered(found, seconds, joins, total)
            total += len(own_kinds)
            kinds.append(own_kinds.astype(kinds[0].dtype))
            tokens.append(own_tokens)
        kinds, tokens = np.concatenate(kinds), np.concatenate(tokens)
        connected = np.r_[connected, np.full(total - len(connected), graphs > 1)]
        self.maps, self.entities, self.base = maps, Entities(kinds, tokens, connected), len(kinds) + room
        self.steps = np.r_[0, np.cumsum([member.steps for member in members])].astype(np.int64)
        self.tasks = np.r_[0, np.cumsum([len(member.task_key) for member in members])].astype(np.int64)
        self.periods = np.r_[0, np.cumsum([int(member.period.max(initial=-1)) + 1 for member in members])]
        bottom, self.second_x, self.second_values = [0, 0], {}, {}
        for name, seconds in self.seconds.items():
            self.second_x[name], self.second_values[name] = [], []
            for second in seconds:
                self.second_x[name].append(bottom[0])
                self.second_values[name].append(bottom[1])
                bottom = [held + size for held, size in zip(bottom, table_sizes(second))]
        sizes = np.asarray([table_sizes(member) for member in members], np.int64).reshape(-1, 2)
        self.x = bottom[0] + np.r_[0, np.cumsum(sizes[:, 0])].astype(np.int64)
        self.values = bottom[1] + np.r_[0, np.cumsum(sizes[:, 1])].astype(np.int64)
        self.top = int(self.x[-1]), int(self.values[-1])
        self.name = members[0].name if len(members) == 1 else "joined"
        LOG.info(
            "plan exit seconds=%.3f graphs=%d entities=%d steps=%d second=%d",
            time.perf_counter() - started,
            len(members),
            len(kinds),
            int(self.steps[-1]),
            len(self.seconds),
        )

    def added_entities(self, members):
        started = time.perf_counter()
        kinds = [(kind, held) for _, added in self.registry.schema.grown for kind, held in added.kinds.items()]
        tokens = {kind: [np.zeros(0, np.uint64)] for kind, _ in kinds}
        for place, member in enumerate(members if kinds else ()):
            entity, rows = observed_rows(member)
            found = [(member, entity, self.registry.encoding.widened(rows))]
            for held in self.seconds.values():
                found.append((held[place],) + observed_rows(held[place]))
            for kind, held in kinds:
                tokens[kind] += [
                    np.asarray(
                        held["tokens"](
                            np.asarray(graph.kinds)[entity],
                            np.asarray(graph.entity_token).astype(np.uint64)[entity],
                            rows,
                            self.registry,
                        ),
                        np.uint64,
                    )
                    for graph, entity, rows in found
                ]
        result = [(self.registry.kind(kind), np.unique(np.concatenate(tokens[kind]))) for kind, _ in kinds]
        LOG.debug(
            "added entities exit graphs=%d kinds=%d seconds=%.4f", len(members), len(result), time.perf_counter() - started
        )
        return result

    def widened(self, graph):
        started = time.perf_counter()
        encoding = self.registry.encoding
        grown = dict(
            x_table=encoding.widened(graph.x_table),
            task_x=encoding.widened(graph.task_x),
            value_table=padded(graph.value_table, self.registry.attributed),
        )
        changed = {name: value for name, value in grown.items() if value is not graph.arrays[name]}
        result = graph.replace(**changed) if changed else graph
        LOG.debug("plan widened exit graph=%s changed=%d seconds=%.4f", graph.name, len(changed), time.perf_counter() - started)
        return result

    def tables(self, members):
        started = time.perf_counter()
        result = [held.astype(np.float16, copy=False) for held in self.second_tables()[0]] + [
            self.registry.encoding.widened(np.asarray(features(member))).astype(np.float16, copy=False)
            for member in members
        ]
        LOG.debug("plan tables exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def second_tables(self):
        started = time.perf_counter()
        seconds = [second for held in self.seconds.values() for second in held]
        result = [np.asarray(held.x_table) for held in seconds], [np.asarray(held.value_table) for held in seconds]
        LOG.debug("second tables exit tables=%d seconds=%.4f", len(seconds), time.perf_counter() - started)
        return result

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


def second_piece(plan, name, member, low, high):
    started = time.perf_counter()
    second = plan.seconds[name][member]
    if (low, high) == (0, second.steps):
        piece, rows, values = second, np.arange(table_sizes(second)[0]), np.arange(table_sizes(second)[1])
    else:
        piece, rows, values, _ = cut(second, low, high)
    numbers = plan.second_maps[name][member]
    arrays = {field: np.asarray(piece.arrays[field]) for field in SECOND}
    for field in ("obs_entity", "fact_source", "fact_target"):
        arrays[field] = numbers[arrays[field].astype(np.int64)]
    result = Second(
        name,
        arrays,
        piece.x_table,
        plan.second_x[name][member] + rows,
        piece.value_table,
        plan.second_values[name][member] + values,
    )
    LOG.debug(
        "second piece exit graph=%s member=%d low=%d high=%d seconds=%.4f",
        name,
        member,
        low,
        high,
        time.perf_counter() - started,
    )
    return result


def opened(plan, member, graph, low, high, edit, entities, seen):
    started = time.perf_counter()
    if (low, high) == (0, graph.steps):
        piece, rows, values, first = graph, np.arange(table_sizes(graph)[0]), np.arange(table_sizes(graph)[1]), 0
    else:
        piece, rows, values, first = cut(graph, low, high)
    ids = plan.maps[member]
    x_global, value_global = plan.x[member] + rows, plan.values[member] + values
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
        x_global = np.r_[x_global, np.full(len(piece.x_table) - tables[0], -1, np.int64)]
        value_global = np.r_[value_global, np.full(len(piece.value_table) - tables[1], -1, np.int64)]
    piece = plan.widened(piece)
    seconds = {name: second_piece(plan, name, member, low, high) for name in plan.seconds}
    every = np.concatenate([ids] + [plan.second_maps[name][member] for name in plan.seconds] + [plan.made])
    glob = np.unique(every)
    local = np.searchsorted(glob, ids)
    arrays = dict(piece.arrays)
    for name in ENTITIES:
        held = np.asarray(arrays[name])
        arrays[name] = gathered(local, held, -1).astype(held.dtype)
    lasting = entities.connected[glob].copy()
    if high < graph.steps:
        lasting[np.searchsorted(glob, every)] = True
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
    for second in seconds.values():
        observed[np.searchsorted(glob, second.obs_entity)] = True
    carry = Segment(
        glob,
        plan.base,
        int(plan.steps[member] + low),
        int(plan.tasks[member] + first),
        np.isin(glob, seen) | (np.isin(glob, plan.made) & (plan.steps[member] + low > 0)),
        x_global,
        value_global,
        entities,
        seconds,
    )
    LOG.debug(
        "opened exit member=%d low=%d high=%d entities=%d seconds=%.4f",
        member,
        low,
        high,
        len(glob),
        time.perf_counter() - started,
    )
    return carry, dense, np.union1d(seen, glob[observed & lasting]), entities


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
            placed=(0, 0),
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
        carry, dense, seen, entities = opened(
            source.plan, member, graph, low, high, edit, state["entities"], state["seen"]
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
    with measure("place"):
        fresh = placed(carry, derived, tuple(int(top) + count for top, count in zip(source.plan.top, state["placed"])))
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
        placed=(state["placed"][0] + len(fresh[0]), state["placed"][1] + len(fresh[1])),
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
    def begun(cls, members, beside, registry, steps, spec, groups, seed, limits, partitions, room):
        started = time.perf_counter()
        strangers = [type(member).__name__ for member in members if not isinstance(member, Dense)]
        if strangers:
            raise TypeError(
                "a derivation begins from graphs in the form the operations work on, Graph.dense(); found %s"
                % ", ".join(strangers)
            )
        plan = Plan(members, paired(steps, beside), unified(members, registry, spec, seed), room, registry)
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
            source.plan.tables(source.members)
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
        plan = source.plan
        x, values = plan.second_tables()
        result = stitched(
            [(segment.carry, segment.graph) for segment in segments],
            self.state["entities"],
            np.concatenate(
                x
                + [plan.registry.encoding.widened(np.asarray(member.x_table)) for member in source.members]
                + [segment.fresh[0] for segment in segments]
            ),
            np.concatenate(
                values
                + [padded(member.value_table, plan.registry.attributed) for member in source.members]
                + [segment.fresh[1] for segment in segments]
            ),
            self.time,
            source.plan.name,
        )
        LOG.info("derivation derived exit seconds=%.3f", time.perf_counter() - started)
        return result


def single(beside):
    started = time.perf_counter()
    if type(beside) is not dict:
        raise TypeError("the second graphs of one graph are a mapping of their names to graphs, found %r" % (beside,))
    result = {name: [second] for name, second in beside.items()}
    LOG.debug("single exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def apart(graph, beside, registry, steps, seed, limits, partitions):
    started = time.perf_counter()
    plan = Plan([graph], paired(steps, single(beside)), alone(graph), 0, registry)
    carry, dense, _, _ = opened(plan, 0, graph, 0, graph.steps, None, plan.entities, EMPTY)
    states = transform.initial(registry, steps)
    derived, _ = transform.derive(dense, registry, steps, seed, limits, partitions, carry, states, False)
    drawn = carry.drawn()
    candidates = dict(drawn, source=carry.local(drawn["source"]), target=carry.local(drawn["target"]))
    LOG.info(
        "apart exit graph=%s candidates=%d seconds=%.3f", graph.name, len(drawn["step"]), time.perf_counter() - started
    )
    return derived, candidates


def whole(graph, beside, registry, steps, seed, limits, partitions):
    started = time.perf_counter()
    derived, _ = apart(graph, beside, registry, steps, seed, limits, partitions)
    LOG.info("whole exit graph=%s seconds=%.3f", graph.name, time.perf_counter() - started)
    return derived


def opening(graph, beside, registry):
    started = time.perf_counter()
    plan = Plan([graph], single(beside), alone(graph), 0, registry)
    carry, dense, _, _ = opened(plan, 0, graph, 0, graph.steps, None, plan.entities, EMPTY)
    LOG.debug("opening exit graph=%s seconds=%.4f", graph.name, time.perf_counter() - started)
    return carry, dense


def variant(members, beside, registry, steps, spec, groups, seed, limits, partitions, destination):
    started = time.perf_counter()
    plan = Plan(members, paired(steps, beside), unified(members, registry, spec, seed), 0, registry)
    source = Source(plan, members, registry, steps, spec, groups, seed, limits, partitions)
    state = source.initial()
    appender = Appender(destination, BUFFERED)
    made = []
    try:
        for member, graph in enumerate(members):
            clock = time.perf_counter()
            with measure("dense"):
                dense = graph.dense()
            segment, state = advanced(source, member, dense, 0, dense.steps, None, state)
            made.append(np.asarray(segment.fresh[0]).astype(np.float16, copy=False))
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
            np.concatenate(plan.tables(members) + made),
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
