import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.time import ALWAYS, NEVER, UNITED
from graphspec.views import spans

LOG = logging.getLogger(__name__)
FORMS = Names("form")
GIVEN = ("since", "until")


@timed("forms.whole")
def whole(table, name, form):
    if name not in table.column_names:
        raise ValueError("the form %s reads the column %s, which the rows do not state" % (form, name))
    column = table.column(name)
    if not pa.types.is_integer(column.type) or column.null_count:
        raise ValueError(
            "the column %s of the form %s holds whole numbers and no missing value" % (name, form)
        )
    return np.asarray(column.to_numpy(zero_copy_only=False), np.int64)


@timed("forms.placed")
def placed(table, names, since, until, known):
    held = table.drop_columns([name for name in names if name in table.column_names])
    for name in GIVEN:
        if name in held.column_names:
            raise ValueError("rows stated in a form do not also state the column %s" % name)
    held = held.append_column("since", pa.array(since, pa.int64()))
    return held.append_column("until", pa.array(until, pa.int64(), mask=~np.asarray(known, bool)))


class Form:
    @timed("forms.form")
    def __init__(self, reduce, clock):
        self.reduce, self.clock = reduce, clock

    @timed("forms.form.call")
    def __call__(self, table, **parameters):
        return self.reduce(table, **parameters)


def state(table):
    started = time.perf_counter()
    lacking = [name for name in GIVEN if name not in table.column_names]
    if lacking:
        raise ValueError("the form state reads the columns %s" % ", ".join(lacking))
    result = table
    for name in GIVEN:
        place = result.schema.get_field_index(name)
        result = result.set_column(place, name, result.column(name).cast(pa.int64()))
    LOG.debug("state exit rows=%d seconds=%.4f", table.num_rows, time.perf_counter() - started)
    return result


def event(table):
    started = time.perf_counter()
    at = whole(table, "at", "event")
    result = placed(table, ("at",), at, at + 1, np.ones(len(at), bool))
    LOG.debug("event exit rows=%d seconds=%.4f", table.num_rows, time.perf_counter() - started)
    return result


def signal(table, start, stride):
    started = time.perf_counter()
    if type(start) is not int or type(stride) is not int or stride < 1:
        raise ValueError(
            "a signal states its first time and the time between two readings as whole numbers, found %r and %r"
            % (start, stride)
        )
    index = whole(table, "index", "signal")
    if np.any(index < 0):
        raise ValueError("the readings of a signal are numbered from zero")
    since = start + index * stride
    result = placed(table, ("index",), since, since + stride, np.ones(len(index), bool))
    LOG.debug("signal exit rows=%d seconds=%.4f", table.num_rows, time.perf_counter() - started)
    return result


def codes(table, names):
    started = time.perf_counter()
    columns = []
    for name in names:
        column = table.column(name).combine_chunks()
        try:
            found = column.dictionary_encode().indices
            found = np.asarray(found.fill_null(-1).to_numpy(zero_copy_only=False), np.int64)
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
            seen = {}
            found = np.asarray(
                [seen.setdefault(repr(value), len(seen)) for value in column.to_pylist()], np.int64
            )
        columns.append(found.reshape(-1))
    rows = np.stack(columns, 1) if columns else np.zeros((table.num_rows, 0), np.int64)
    result = spans.identity(np.ascontiguousarray(rows))[0] if rows.shape[1] else np.zeros(len(rows), np.int64)
    LOG.debug(
        "codes exit rows=%d columns=%d seconds=%.4f", len(rows), len(names), time.perf_counter() - started
    )
    return result


def change(table, keys):
    started = time.perf_counter()
    keys = list(keys)
    if not keys or any(name not in table.column_names for name in keys):
        raise ValueError(
            "the form change names the columns that say what changes, found %s" % ", ".join(map(str, keys))
        )
    at = whole(table, "at", "change")
    changed = codes(table, keys)
    order = np.lexsort((at, changed))
    same = changed[order][1:] == changed[order][:-1]
    if np.any(same & (at[order][1:] == at[order][:-1])):
        moment = at[order][1:][same & (at[order][1:] == at[order][:-1])][0]
        raise ValueError("the form change meets two changes of one thing at the time %d" % moment)
    until, known = np.zeros(len(at), np.int64), np.zeros(len(at), bool)
    until[order[:-1][same]] = at[order][1:][same]
    known[order[:-1][same]] = True
    result = placed(table, ("at",), at, until, known)
    LOG.debug("change exit rows=%d seconds=%.4f", table.num_rows, time.perf_counter() - started)
    return result


def ranked(span, heads, count, position, seq, times):
    started = time.perf_counter()
    order = np.lexsort((seq, position))
    together = position[order][1:] == position[order][:-1]
    if np.any(together & (seq[order][1:] == seq[order][:-1])):
        moment = position[order][1:][together & (seq[order][1:] == seq[order][:-1])][0]
        raise ValueError("two rows of the snapshot at the time %d share one seq" % times[moment])
    before, after = span[order][:-1][together], span[order][1:][together]
    rank = spans.topological(before, after, count, np.argsort(heads, kind="stable"))
    wrong = rank[before] >= rank[after]
    if wrong.any():
        moment = position[order][1:][together][wrong][0]
        raise ValueError(
            "the snapshot at the time %d orders two facts against the order of an earlier snapshot; "
            "facts stated in snapshots keep one order, or the schema states another rule for ties"
            % times[moment]
        )
    LOG.debug("ranked exit spans=%d seconds=%.4f", count, time.perf_counter() - started)
    return rank


def together(tables, times, sequence):
    started = time.perf_counter()
    times = np.asarray(times, np.int64)
    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("the times of snapshots are stated once each, in order")
    parts = []
    for place, table in enumerate(tables):
        at = whole(table, "at", "snapshot")
        position = np.searchsorted(times, at)
        if np.any(position >= len(times)) or not np.array_equal(
            times[np.minimum(position, len(times) - 1)], at
        ):
            raise ValueError(
                "rows of a snapshot are stated at a time that is not among the times of snapshots"
            )
        content = [name for name in table.column_names if name not in ("at", "seq")]
        parts.append(
            dict(
                table=np.full(len(at), place, np.int64),
                row=np.arange(len(at)),
                position=position,
                content=codes(table, content),
                seq=whole(table, "seq", "snapshot") if sequence else np.zeros(len(at), np.int64),
            )
        )
    held = {
        name: np.concatenate([part[name] for part in parts] + [np.zeros(0, np.int64)])
        for name in ("table", "row", "position", "content", "seq")
    }
    if not len(held["row"]):
        empty = np.zeros(0, np.int64)
        results = [placed(table, ("at", "seq"), empty, empty, np.zeros(0, bool)) for table in tables]
        if sequence:
            results = [table.append_column("seq", pa.array(empty, pa.int32())) for table in results]
        LOG.debug(
            "snapshots together exit tables=%d rows=0 seconds=%.4f",
            len(tables),
            time.perf_counter() - started,
        )
        return results
    ident, _ = spans.identity(np.ascontiguousarray(np.stack((held["table"], held["content"]), 1)))
    rows = np.arange(len(ident))
    order = np.lexsort((rows, held["position"], ident))
    grouped = np.r_[
        True,
        (ident[order][1:] != ident[order][:-1])
        | (held["position"][order][1:] != held["position"][order][:-1]),
    ][: len(order)]
    first = np.maximum.accumulate(np.where(grouped, np.arange(len(order)), 0))
    occurrence = np.empty(len(order), np.int64)
    occurrence[order] = np.arange(len(order)) - first
    member, _ = spans.identity(np.ascontiguousarray(np.stack((ident, occurrence), 1)))
    span, heads, count = spans.runs(member, held["position"])
    tail = np.zeros(count, np.int64)
    np.maximum.at(tail, span, held["position"])
    rank = (
        ranked(span, heads, count, held["position"], held["seq"], times)
        if sequence
        else np.zeros(count, np.int64)
    )
    arrangement = np.lexsort((rank, heads, held["position"][heads]))
    since = times[held["position"][heads]]
    known = tail + 1 < len(times)
    until = times[np.minimum(tail + 1, len(times) - 1)]
    results = []
    for place, table in enumerate(tables):
        chosen = arrangement[held["table"][heads][arrangement] == place]
        taken = table.take(pa.array(held["row"][heads][chosen]))
        result = placed(taken, ("at", "seq"), since[chosen], until[chosen], known[chosen])
        if sequence:
            result = result.append_column("seq", pa.array(rank[chosen], pa.int32()))
        results.append(result)
    LOG.debug(
        "snapshots together exit tables=%d rows=%d facts=%d seconds=%.4f",
        len(tables),
        len(ident),
        count,
        time.perf_counter() - started,
    )
    return results


def snapshot(table, times):
    started = time.perf_counter()
    result = together([table], times, "seq" in table.column_names)[0]
    LOG.debug("snapshot exit rows=%d seconds=%.4f", table.num_rows, time.perf_counter() - started)
    return result


FORMS.register("state", Form(state, False))
FORMS.register("event", Form(event, False))
FORMS.register("snapshot", Form(snapshot, False))
FORMS.register("change", Form(change, False))
FORMS.register("signal", Form(signal, True))


def reduced(name, model, table, **parameters):
    started = time.perf_counter()
    form = FORMS.named(name)
    if model == "none":
        raise ValueError("the graphs of the schema have no time, and the form %s states times" % name)
    if form.clock and model not in UNITED:
        raise ValueError("the form %s counts on a clock, and the graphs of the schema have none" % name)
    result = form(table, **parameters)
    for column in GIVEN:
        if column not in result.column_names or result.column(column).type != pa.int64():
            raise ValueError("the form %s gives facts without the whole column %s" % (name, column))
    since = result.column("since").fill_null(ALWAYS).to_numpy(zero_copy_only=False)
    until = result.column("until").fill_null(NEVER).to_numpy(zero_copy_only=False)
    if np.any(until <= since):
        raise ValueError("the form %s gives a fact that holds until no later than it holds from" % name)
    LOG.debug(
        "reduced exit form=%s rows=%d facts=%d seconds=%.4f",
        name,
        table.num_rows,
        result.num_rows,
        time.perf_counter() - started,
    )
    return result
