import numpy as np
import pyarrow as pa

from graphspec.clock import NEVER
from graphspec.infra.timing import timed
from graphspec.references import Sort
from graphspec.sets import Form, node
from graphspec.view import Tasks

NAME = "registered"


class Count:
    accepts = ("INT32",)
    written = pa.int32()
    symbols = False
    ordered = True

    def read(self, column):
        return np.asarray(column.to_numpy(zero_copy_only=False), np.float32)

    def missing(self, stated):
        return np.isnan(stated)

    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else int(value) for value in codes]


class Halved:
    types = ("count",)

    def fit(self, codes, dictionary, column):
        return dict(most=float(np.nanmax(codes)))

    def widths(self, state):
        return 1, 1

    def encode(self, codes, state, dictionary):
        known = ~np.isnan(np.asarray(codes, np.float32))
        held = np.where(known, codes, 0.0).astype(np.float32)
        return (held / (2 * state["most"]))[:, None], known[:, None].astype(np.float32)


class Counted:
    @timed("registered.counted.width")
    def width(self, encoding, held):
        return 1

    @timed("registered.counted.encode")
    def encode(self, encoding, held, graphs):
        return np.bincount(graphs["owner"], minlength=graphs["owners"]).astype(np.float32)[:, None]


class Rank:
    columns = dict(seq="INT32")

    @timed("registered.rank.keys")
    def keys(self, facts):
        return (-np.asarray(facts["seq"], np.int64),)


class Kept:
    @timed("registered.kept.kept")
    def kept(self, lacking, named):
        return np.ones(len(lacking), bool)


class Own(Tasks):
    @timed("registered.own.serves")
    def serves(self, registry):
        return registry.name == NAME


@timed("registered.middle")
def middle(starts, stops):
    return (starts + stops - 1) // 2


@timed("registered.everything")
def everything(facts, registry, held):
    return np.full(len(facts.obs_entity), NEVER, np.int64), np.full(len(facts.fact_source), NEVER, np.int64)


@timed("registered.spanned")
def spanned(table, length):
    since = np.asarray(table.column("at").to_numpy(zero_copy_only=False), np.int64)
    result = table.drop_columns(["at"]).append_column("since", pa.array(since, pa.int64()))
    return result.append_column("until", pa.array(since + length, pa.int64()))


@timed("registered.ready")
def ready():
    return None


TRIPLE = Sort(True, False, 3, 3, True)
STORED = Form(node, True, True)
