import numpy as np

from graphspec.infra.timing import timed
from graphspec.views.spans import encode

FOREVER = np.int64(1) << np.int64(62)


class Table:
    @timed("carry.table")
    def __init__(self, keys=(), **columns):
        keys = np.asarray(keys, np.int64)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.columns = {name: np.asarray(value)[order] for name, value in columns.items()}

    def __len__(self):
        return len(self.keys)

    def find(self, keys):
        keys = np.asarray(keys, np.int64)
        if not len(self.keys):
            return np.full(len(keys), -1, np.int64)
        position = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[position] == keys, position, -1)

    def lookup(self, name, keys, default):
        found = self.find(keys)
        if not len(self.keys):
            return np.full(len(found), default, np.int64)
        return np.where(found >= 0, self.columns[name][np.maximum(found, 0)], default)

    @timed("carry.table.replaced")
    def replaced(self, touched, keys, **columns):
        kept = ~np.isin(self.keys, np.asarray(touched, np.int64))
        return Table(
            np.r_[self.keys[kept], np.asarray(keys, np.int64)],
            **{
                name: np.concatenate((self.columns[name][kept], np.asarray(value)))
                if name in self.columns
                else np.asarray(value)
                for name, value in columns.items()
            },
        )


class Segment:
    def __init__(self, glob, entities, step0, group0, period0, state, x_global, value_global):
        self.glob, self.entities = glob, entities
        self.step0, self.group0, self.period0 = step0, group0, period0
        self.state = state
        self.x_global, self.value_global = x_global, value_global

    def keys(self, code, source, target):
        return encode(code, self.glob[source], self.glob[target], self.entities)

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

    def table(self, name):
        return self.state.get(name, Table())
