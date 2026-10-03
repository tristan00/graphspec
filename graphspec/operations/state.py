import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.infra.workspace import gathered
from graphspec.views.spans import encode

LOG = logging.getLogger(__name__)
FOREVER = np.int64(1) << np.int64(62)
EMPTY = np.zeros(0, np.int64)
ENTITIES = ("obs_entity", "fact_source", "fact_target", "member_source", "member_target")
SECOND = (
    "obs_entity",
    "obs_step",
    "obs_ref",
    "fact_source",
    "fact_target",
    "fact_code",
    "fact_step",
    "fact_present",
    "fact_ref",
)


class Table:
    def __init__(self, keys=(), **columns):
        keys = np.asarray(keys, np.int64)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.columns = {name: np.asarray(value)[order] for name, value in columns.items()}

    @timed("state.Table.__len__")
    def __len__(self):
        return len(self.keys)

    def find(self, keys):
        keys = np.asarray(keys, np.int64)
        position = np.searchsorted(self.keys, keys)
        inside = np.flatnonzero(position < len(self.keys))
        named = inside[self.keys[position[inside]] == keys[inside]]
        result = np.full(len(keys), -1, np.int64)
        result[named] = position[named]
        return result

    def lookup(self, name, keys, default):
        return gathered(self.columns[name], self.find(keys), default)

    @timed("carry.table.replaced")
    def replaced(self, touched, keys, **columns):
        kept = ~np.isin(self.keys, np.asarray(touched, np.int64))
        return Table(
            np.concatenate((self.keys[kept], np.asarray(keys, np.int64))),
            **{
                name: np.concatenate((self.columns[name][kept], np.asarray(value)))
                for name, value in columns.items()
            },
        )


class Entities:
    @timed("state.Entities.__init__")
    def __init__(self, kinds, tokens, connected):
        self.kinds, self.tokens, self.connected = kinds, tokens, connected

    @timed("state.Entities.__len__")
    def __len__(self):
        return len(self.kinds)

    @timed("carry.entities.added")
    def added(self, kinds, tokens):
        return Entities(
            np.concatenate((self.kinds, np.asarray(kinds, self.kinds.dtype))),
            np.concatenate((self.tokens, np.asarray(tokens, self.tokens.dtype))),
            np.concatenate((self.connected, np.zeros(len(kinds), bool))),
        )


class Second:
    def __init__(self, name, arrays, x_table, x_global, value_table, value_global):
        started = time.perf_counter()
        lacking = [field for field in SECOND if field not in arrays]
        if lacking:
            raise ValueError("the second graph %s of a segment lacks %s" % (name, ", ".join(lacking)))
        self.name, self.arrays = name, {field: np.asarray(arrays[field]) for field in SECOND}
        self.x_table, self.x_global = np.asarray(x_table), np.asarray(x_global, np.int64)
        self.value_table, self.value_global = np.asarray(value_table), np.asarray(value_global, np.int64)
        LOG.debug("second exit graph=%s seconds=%.4f", name, time.perf_counter() - started)

    @timed("state.Second.__getattr__")
    def __getattr__(self, key):
        if key in SECOND:
            return self.__dict__["arrays"][key]
        raise AttributeError(key)


class Segment:
    @timed("state.Segment.__init__")
    def __init__(self, glob, base, step0, task0, known, x_global, value_global, entities, seconds):
        self.glob, self.base, self.step0, self.task0 = glob, base, step0, task0
        self.known, self.x_global, self.value_global = known, x_global, value_global
        self.entities, self.state, self.part = entities, None, None
        self.seconds, self.second, self.entry, self.candidates = seconds, None, None, []

    @timed("state.Segment.__getstate__")
    def __getstate__(self):
        return dict(self.__dict__, entities=None, state=None, part=None, second=None, entry=None)

    def named(self, name, operation, entry):
        self.entry = entry
        if name is None:
            self.second = None
        elif name not in self.seconds:
            raise ValueError(
                "the operation %s takes the graph %s as its second graph, and the derivation was given %s"
                % (operation, name, ", ".join(sorted(self.seconds)) or "no second graph")
            )
        else:
            self.second = self.seconds[name]

    @timed("state.Segment.keys")
    def keys(self, code, source, target):
        return encode(code, self.glob[source], self.glob[target], self.base)

    @timed("carry.segment.candidate")
    def candidate(self, code, step, source, target):
        self.candidates.append(
            (
                np.asarray(code, np.int64),
                np.asarray(step, np.int64),
                self.glob[np.asarray(source, np.int64)],
                self.glob[np.asarray(target, np.int64)],
            )
        )

    @timed("carry.segment.drawn")
    def drawn(self):
        code, step, source, target = (
            np.concatenate([held[place] for held in self.candidates] + [EMPTY]) for place in range(4)
        )
        order = np.lexsort((target, source, code, step))
        return dict(code=code[order], step=step[order], source=source[order], target=target[order])

    @timed("carry.segment.tokens")
    def tokens(self, entity):
        return self.entities.tokens[self.glob[np.asarray(entity, np.int64)]].astype(np.uint64)

    @timed("carry.segment.local")
    def local(self, entity):
        entity = np.asarray(entity, np.int64)
        position = np.searchsorted(self.glob, entity)
        if len(entity) and (
            np.any(position >= len(self.glob))
            or np.any(self.glob[np.minimum(position, len(self.glob) - 1)] != entity)
        ):
            raise ValueError("a carried entity is missing from its segment")
        return position

    def widened(self, graph, names):
        started = time.perf_counter()
        names = np.setdiff1d(np.asarray(names, np.int64), self.glob)
        if not len(names):
            return graph, np.arange(len(self.glob))
        glob = np.union1d(self.glob, names)
        position = np.searchsorted(glob, self.glob)
        added = np.searchsorted(glob, names)
        arrays = {}
        for name in ENTITIES:
            values = np.asarray(graph.arrays[name])
            arrays[name] = gathered(position, values, -1).astype(values.dtype)
        for name, table in (("kinds", "kinds"), ("entity_token", "tokens"), ("entity_connected", "connected")):
            values = np.asarray(graph.arrays[name])
            grown = np.empty(len(glob), values.dtype)
            grown[position], grown[added] = values, getattr(self.entities, table)[names]
            arrays[name] = grown
        known = np.ones(len(glob), bool)
        known[position] = self.known
        self.glob, self.known = glob, known
        LOG.debug(
            "widened exit entities=%d added=%d seconds=%.4f", len(glob), len(names), time.perf_counter() - started
        )
        return graph.replace(**arrays), position
