import logging
import struct
import time
from pathlib import Path

import numpy as np

from graphspec.operations.state import ENTITIES
from graphspec.views.graph import DENSE, Dense

LOG = logging.getLogger(__name__)
MOMENTS = ("obs_step", "fact_step", "task_step")
TASKED = ("member_task", "label_task")
GROUP = ("kinds", "entity_token", "entity_connected", "x_table", "value_table")
HEADER = 128


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
    def __init__(self, directory):
        self.directory, self.streams, self.kinds, self.rows = Path(directory), {}, {}, {}

    def add(self, piece):
        started = time.perf_counter()
        for name, values in piece.items():
            if name not in self.streams:
                values = np.asarray(values)
                self.kinds[name], self.rows[name] = (values.dtype, values.shape[1:]), 0
                self.streams[name] = open(self.directory / (name + ".npy"), "wb")
                self.streams[name].write(bytes(HEADER))
            dtype, trailing = self.kinds[name]
            if np.asarray(values).dtype != dtype or np.shape(values)[1:] != trailing:
                raise ValueError("segment array %s changed dtype or shape" % name)
            values = np.ascontiguousarray(values)
            self.rows[name] += len(values)
            if values.size:
                self.streams[name].write(values.data)
        LOG.debug("appender add exit arrays=%d seconds=%.4f", len(piece), time.perf_counter() - started)

    def close(self):
        for name, stream in self.streams.items():
            dtype, trailing = self.kinds[name]
            stream.seek(0)
            stream.write(header(dtype, (self.rows[name],) + trailing))
            stream.close()
        self.streams = {}


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
