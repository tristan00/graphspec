import numpy as np
import pyarrow as pa

from graphspec import Operation, Phase, Representation
from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.operations.sets import Form, node
from graphspec.operations.state import EMPTY, Table
from graphspec.schema.references import Sort
from graphspec.schema.time import NEVER
from graphspec.views.view import Tasks

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


@timed("registered.tallied")
def tallied(graph, registry, settings, seed, limits, carry, held):
    total = carry.state["total"] + np.cumsum(np.bincount(graph.obs_step, minlength=graph.steps))
    carry.state = dict(total=int(total[-1]))
    return graph.replace(period=total.astype(graph.period.dtype))


def untallied(registry, settings):
    return dict(total=0)


def unstated(registry):
    return ()


def uncounted(registry, spec):
    return dict(seen=Table(EMPTY, count=EMPTY))


def unlooking(spec):
    return 0, 0


@timed("registered.running")
def running(shown, state):
    entity = shown.own["entity"].astype(np.int64)
    order = np.argsort(entity, kind="stable")
    before = np.empty(len(entity), np.int64)
    before[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    seen = state["seen"]
    held, totals = np.unique(entity, return_counts=True)
    state["seen"] = seen.replaced(held, held, count=seen.lookup("count", held, 0) + totals)
    return dict(running=seen.lookup("count", entity, 0) + before)


def unchecked(arrays, name, registry, spec):
    if np.any(arrays["running"] < 0):
        raise ValueError("a running count of graph %s is negative" % name)


def unlinked(registry, spec):
    return dict(last=Table(EMPTY, row=EMPTY))


@timed("registered.linked")
def linked(shown, state):
    entity = shown.own["entity"].astype(np.int64)
    rows = shown.before["obs"] + np.arange(len(entity), dtype=np.int64)
    order = np.lexsort((rows, entity))
    same = np.r_[False, entity[order][1:] == entity[order][:-1]]
    previous = np.full(len(entity), -1, np.int64)
    previous[order[same]] = rows[order[np.flatnonzero(same) - 1]]
    first = order[~same]
    previous[first] = state["last"].lookup("row", entity[first], -1)
    ends = order[np.r_[~same[1:], True]] if len(order) else order
    state["last"] = state["last"].replaced(entity[ends], entity[ends], row=rows[ends])
    return dict(linked=previous)


def checked_links(arrays, name, registry, spec):
    linked, entity = np.asarray(arrays["linked"]), np.asarray(arrays["entity"])
    chosen = np.flatnonzero(linked >= 0)
    if np.any(linked[chosen] >= chosen) or np.any(entity[linked[chosen]] != entity[chosen]):
        raise ValueError("a link of graph %s does not lead to an earlier observation of its entity" % name)


def unkept(registry, spec):
    return {}


@timed("registered.kept")
def kept(shown, state):
    entity = shown.own["entity"].astype(np.int64)
    local = shown.own["step"].astype(np.int64) - shown.carry.step0
    drawn = unit(
        shown.seed,
        "test.kept",
        shown.carry.entities.tokens[entity].astype(np.uint64),
        np.asarray(shown.graph.step_time, np.int64)[local],
        np.asarray(shown.graph.step_token, np.uint64)[local],
    )
    return dict(kept=drawn < 0.5)


def checked_kept(arrays, name, registry, spec):
    if len(arrays["kept"]) != len(arrays["entity"]):
        raise ValueError("graph %s does not draw once for every observation" % name)


TRIPLE = Sort(True, False, 3, 3, True)
STORED = Form(node, True, True)
TALLY = Operation("test", "tally", (Phase("tally", tallied, untallied, None),), unstated, (), (), (), ("time",), 1)
RUNNING = Representation("test.running", "encode", uncounted, unlooking, running, None, None, unchecked)
LINKED = Representation("test.linked", "structure", unlinked, unlooking, linked, None, None, checked_links)
KEPT = Representation("test.kept", "sample", unkept, unlooking, kept, None, None, checked_kept)
