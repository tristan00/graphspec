import logging
import struct
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import Arrays
from graphspec.operations.state import ENTITIES
from graphspec.views.graph import DENSE, Dense

LOG = logging.getLogger(__name__)
MOMENTS = ("obs_step", "fact_step", "task_step")
TASKED = ("member_task", "label_task")
GROUP = ("kinds", "entity_token", "entity_connected", "x_table", "value_table")
HEADER = 128
BUFFERED = 64 << 20


def header(dtype, shape):
    text = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (
        np.lib.format.dtype_to_descr(dtype),
        shape,
    )
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", HEADER - 10)
        + (text.ljust(HEADER - 11) + "\n").encode("latin1")
    )


class Appender:
    def __init__(self, directory, limit):
        self.directory, self.limit, self.size = Path(directory), limit, 0
        self.kinds, self.rows, self.held, self.streams = {}, {}, {}, None

    def add(self, piece):
        started = time.perf_counter()
        for name, values in piece.items():
            values = np.ascontiguousarray(values)
            if name not in self.kinds:
                self.kinds[name], self.rows[name] = (values.dtype, values.shape[1:]), 0
                if self.streams is None:
                    self.held[name] = []
                else:
                    self.opened(name)
            if values.dtype != self.kinds[name][0] or values.shape[1:] != self.kinds[name][1]:
                raise ValueError("segment array %s changed dtype or shape" % name)
            self.rows[name] += len(values)
            if self.streams is None:
                self.held[name].append(values)
                self.size += values.nbytes
            elif values.size:
                self.streams[name].write(values.data)
        if self.streams is None and self.size > self.limit:
            self.spill()
        LOG.debug("appender add exit arrays=%d seconds=%.4f", len(piece), time.perf_counter() - started)

    def opened(self, name):
        self.streams[name] = open(self.directory / (name + ".npy"), "wb")
        self.streams[name].write(bytes(HEADER))

    def spill(self):
        started = time.perf_counter()
        self.streams = {}
        for name, parts in self.held.items():
            self.opened(name)
            for part in parts:
                if part.size:
                    self.streams[name].write(part.data)
        self.held = {}
        LOG.debug("appender spill exit arrays=%d seconds=%.4f", len(self.streams), time.perf_counter() - started)

    def abandon(self):
        if self.streams is not None:
            for stream in self.streams.values():
                stream.close()
        self.streams, self.held = {}, {}

    def close(self):
        started = time.perf_counter()
        if self.streams is None:
            result = {name: np.concatenate(parts) for name, parts in self.held.items()}
            for name, values in result.items():
                np.save(self.directory / (name + ".npy"), values, allow_pickle=False)
            self.held = {}
        else:
            for name, stream in self.streams.items():
                dtype, trailing = self.kinds[name]
                stream.seek(0)
                stream.write(header(dtype, (self.rows[name],) + trailing))
                stream.close()
            self.streams, result = {}, Arrays(self.directory)
        LOG.debug("appender close exit arrays=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result


def joined(pieces):
    started = time.perf_counter()
    names = [name for name in pieces[0]]
    result = {name: np.concatenate([piece[name] for piece in pieces]) for name in names}
    LOG.debug("joined exit pieces=%d arrays=%d seconds=%.4f", len(pieces), len(names), time.perf_counter() - started)
    return result


def globalized(carry, name, values):
    values = np.asarray(values)
    if name in ENTITIES:
        return np.where(values >= 0, carry.glob[np.maximum(values, 0)], -1).astype(values.dtype)
    if name in MOMENTS:
        return (values + carry.step0).astype(values.dtype)
    if name in TASKED:
        return (values + carry.task0).astype(values.dtype)
    if name == "obs_ref":
        return carry.x_global[values].astype(values.dtype)
    if name == "fact_ref":
        table = np.r_[carry.value_global, -1]
        return np.where(values >= 0, table[np.maximum(values, 0)], -1).astype(values.dtype)
    return values


def stitched(segments, entities, tables, values, steps, name):
    started = time.perf_counter()
    columns = {
        held: np.concatenate([globalized(carry, held, graph.arrays[held]) for carry, graph in segments])
        for held in DENSE
        if held not in GROUP
    }
    result = Dense(
        name,
        [],
        steps,
        kinds=np.asarray(entities.kinds).astype(np.int32),
        entity_token=entities.tokens,
        entity_connected=entities.connected,
        x_table=tables,
        value_table=values,
        **columns,
    )
    LOG.info("stitched derived exit seconds=%.3f segments=%d", time.perf_counter() - started, len(segments))
    return result
