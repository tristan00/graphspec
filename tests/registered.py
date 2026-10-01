import functools

import numpy as np
import pyarrow as pa
import torch

from graphspec import Operation, Phase, Representation
from graphspec.infra.hashing import token, unit
from graphspec.infra.timing import timed
from graphspec.operations.negatives import Rule
from graphspec.operations.parameters import flag
from graphspec.operations.rows import extend, made_rows, made_values, observed, written
from graphspec.operations.sets import Form, node
from graphspec.operations.state import EMPTY, Table
from graphspec.schema.added import Added
from graphspec.schema.references import Sort
from graphspec.schema.time import NEVER
from graphspec.views.view import Tasks
from tuning.families.heads import FIELD, Attached

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


@timed("registered.hubs")
def hubs(entry, kinds, tokens, rows, registry):
    return np.asarray([token(entry)], np.uint64)


@timed("registered.adding")
def adding(registry, settings, entry):
    count = entry + ".count"
    return Added(
        {count: dict(type="number", encoder="standard")},
        {entry: dict(columns=(count,), tokens=functools.partial(hubs, entry))},
        {},
        {entry + ".in": dict(pairs=(("place", entry),), directed=True, columns=(count,))},
        {},
        {},
    )


@timed("registered.hubbed")
def hubbed(graph, registry, settings, seed, limits, carry, held):
    entry, kinds = carry.entry, np.asarray(graph.kinds)
    (hub,) = np.flatnonzero(kinds == registry.kind(entry))
    entity, step = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    places = np.flatnonzero(~np.asarray(graph.obs_anchor, bool) & (kinds[entity] == registry.kind("place")))
    counts = np.bincount(step[places], minlength=graph.steps)
    moments = np.flatnonzero(counts)
    rows = written(
        registry,
        np.zeros((len(moments), registry.features), np.float32),
        (entry + ".count",),
        counts[moments],
        graph.x_table.dtype,
        lambda row: "the hub at the step %d" % moments[row],
    )
    graph, references = made_rows(graph, carry, rows)
    graph = observed(graph, np.full(len(moments), hub), moments, references, np.zeros(len(moments), bool))
    values = np.full((len(places), graph.value_table.shape[1]), np.nan, np.float32)
    values[:, 0] = counts[step[places]]
    graph, references = made_values(graph, carry, values)
    return extend(
        graph,
        entity[places],
        np.full(len(places), hub),
        np.full(len(places), registry.code(entry + ".in")),
        step[places],
        np.ones(len(places), bool),
        references,
    )


@timed("registered.parities")
def parities(registry):
    return (flag("negatives/test.parity/odd", True),)


@timed("registered.parity")
def parity(near, steps, anchors, candidates, settings):
    return (near.tokens[candidates] % np.uint64(2) == settings["negatives/test.parity/odd"]).astype(np.float64)


class Share:
    scaled = False
    matures = False
    columns = dict(trials="FLOAT")
    fields = ("trials",)

    @timed("registered.share.series")
    def series(self, stated, where):
        if type(stated) is not str or not stated:
            raise ValueError("%s names its one series, found %r" % (where, stated))
        return (stated,)

    @timed("registered.share.written")
    def written(self, series):
        return series[0]

    @timed("registered.share.points")
    def points(self, rows, series, where):
        share = np.asarray(rows["y"], np.float64)
        trials = np.asarray([np.nan if value is None else value for value in rows["trials"]], np.float64)
        if np.any((share < 0) | (share > 1)):
            raise ValueError("a share of %s lies outside 0 and 1" % where)
        if not np.all(trials >= 1) or len(np.unique(rows["task"])) != len(share):
            raise ValueError("%s holds one share of one or more trials for a task" % where)
        return dict(series=[series[0]] * len(share), trials=trials.tolist())

    @timed("registered.share.staged")
    def staged(self, stated, series, where):
        share, trials = stated
        return [dict(y=share, trials=trials)]

    @timed("registered.share.read")
    def read(self, rows):
        (row,) = rows
        return [row["y"], row["trials"]]


class Expected:
    reads = ("share",)
    parameters = ()

    @timed("registered.expected.checked")
    def checked(self, stated, series, where):
        return {}

    @timed("registered.expected.value")
    def value(self, stated, points):
        tasks, owner = np.unique(np.asarray(points["task"], np.int64), return_inverse=True)
        counts = np.asarray(points["y"], np.float64) * np.asarray(points["trials"], np.float64)
        return tasks, np.bincount(owner.reshape(-1), counts, minlength=len(tasks))


class ShareHead(torch.nn.Module):
    @timed("registered.sharehead")
    def __init__(self, hidden):
        super().__init__()
        self.output = torch.nn.Linear(hidden * 2 + 1, 1)

    def forward(self, h, series, b, points):
        trials = torch.log1p(b[FIELD + "trials"][points])[:, None]
        return torch.sigmoid(self.output(torch.cat((h, series, trials), -1)).squeeze(-1))


@timed("registered.shared")
def shared(hidden, spec, registry):
    return ShareHead(hidden)


class Crossed:
    reads = ("share",)

    def __call__(self, predictions, batch, bank):
        held, share, trials = predictions.clamp(1e-6, 1 - 1e-6), batch["target"], batch[FIELD + "trials"]
        return -(trials * (share * held.log() + (1 - share) * (1 - held).log())).sum() / trials.sum()


class Absolute:
    reads = ("share",)
    better = "lower"
    label, short = "Validation absolute error", "MAE"

    @timed("registered.absolute")
    def __call__(self, held):
        error = np.abs(held.prediction - held.target)
        return dict(
            value=float(error.mean()),
            series={name: float(error[held.series == code].mean()) for code, name in enumerate(held.names)},
        )


TRIPLE = Sort(True, False, 3, 3, True)
SHARE_HEAD = Attached((), shared)
PARITY = Rule(parities, parity)
STORED = Form(node, True, True)
TALLY = Operation(
    "test", "tally", (Phase("tally", tallied, untallied, None),), unstated, (), (), (), ("time",), None, None, 1, 1
)
HUB = Operation(
    "test",
    "hub",
    (Phase("hub", hubbed, None, None),),
    unstated,
    (),
    (),
    (),
    ("observations", "facts"),
    adding,
    None,
    1,
    1,
)
RUNNING = Representation("test.running", "encode", uncounted, unlooking, running, None, None, unchecked)
LINKED = Representation("test.linked", "structure", unlinked, unlooking, linked, None, None, checked_links)
KEPT = Representation("test.kept", "sample", unkept, unlooking, kept, None, None, checked_kept)
