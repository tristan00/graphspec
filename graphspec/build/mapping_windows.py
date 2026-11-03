import logging
import math
import time
from collections.abc import Mapping as Held

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .database import Part, ruled
from .expression_types import Kind
from .mapping_rows import Reader, joined
from .mappings import Mapping
from .table_types import Frame, generated, take

LOG = logging.getLogger(__name__)
EVERY = Names("rule of the rows of tasks", True)
WINDOW = ("rows", "on", "at", "until", "within")


@timed("mapping_windows.length")
def length(context, stated, where):
    value = context.constant(stated, where)
    found = value.values.as_py() if hasattr(value.values, "as_py") else value.values
    if value.type != "number" or found is None or not float(found).is_integer() or found < 1:
        raise ValueError("%s states a whole number of units of the graph from 1, found %r" % (where, stated))
    return int(found)


class Window:
    @timed("mapping_windows.Window")
    def __init__(self, reader, session, stated, field, moments, later):
        where = "declaration %s field %s" % (reader.name, field)
        if not isinstance(stated, Held):
            raise ValueError("%s states a mapping of rows, on, at, until and within" % where)
        unknown = sorted(set(stated) - set(WINDOW))
        lacking = [name for name in ("rows", "on", "at", "within") if name not in stated]
        if unknown or lacking:
            raise ValueError("%s holds fields it does not know or lacks: %s" % (where, ", ".join(unknown + lacking)))
        if not isinstance(stated["on"], Held) or not stated["on"]:
            raise ValueError("%s states on as a mapping from a column of its rows to an expression of the task" % where)
        if None in moments:
            raise ValueError("%s reads rows by the time of the task, which states at" % where)
        rows = Reader(session, stated["rows"], reader.name, reader.plan)
        self.frame = rows.frame
        columns = list(stated["on"])
        found = rows.texts(["column(%r)" % name for name in columns], field + ".on", missing="omit", outcome=later)
        asked = reader.texts([stated["on"][name] for name in columns], field + ".on", missing="omit")
        at = rows.moments(stated["at"], field + ".at", outcome=later)
        until = None if stated.get("until") is None else rows.moments(stated["until"], field + ".until", "omit", later)
        within = None if stated["within"] is None else length(rows.context, stated["within"], where + " within")
        through = None
        if later and within is not None:
            complete = self.frame.complete
            if not isinstance(complete, pa.Scalar) or not complete.is_valid:
                raise ValueError("%s states within, and the source of its rows states no through" % where)
            through = complete.as_py()
        index = {}
        for place, key in enumerate(found):
            if key is not None:
                index.setdefault(key, []).append(place)
        self.places, self.open = [], []
        for key, moment in zip(asked, moments):
            low = moment if later else -math.inf if within is None else moment - within
            high = (math.inf if within is None else moment + within) if later else moment
            self.places.append([place for place in index.get(key, []) if at[place] <= high and (
                low < at[place] if until is None else until[place] is None or until[place] > low)])
            self.open.append(through is not None and high > through)


class Nodes:
    version = 1

    @timed("mapping_windows.Nodes")
    def __call__(self, facts, parameters):
        every, best = parameters["every"], {}
        names = ("row", "key", "since", "until", "first", "last")
        for row, key, since, until, first, last in zip(*(facts[name].to_pylist() for name in names)):
            moment = first
            while moment <= last:
                if (since is None or since <= moment) and (until is None or moment < until):
                    held = best.get((key, moment))
                    start = -math.inf if since is None else since
                    if held is None or held[1] <= start:
                        best[(key, moment)] = (row, start)
                moment += every
        chosen = sorted((row, moment) for (_, moment), (row, _) in best.items())
        return pa.table(dict(row=pa.array([row for row, _ in chosen], pa.int64()),
                             moment=pa.array([moment for _, moment in chosen], pa.int64())))


EVERY.register("nodes", Nodes())


@timed("mapping_windows.bounds")
def bounds(session):
    found = []
    for frame in session.frames.values():
        for name, kind in frame.kinds.items():
            if kind.name == "time" and frame.table.num_rows:
                extremes = pc.min_max(frame.table[name]).as_py()
                found += [value for value in extremes.values() if value is not None]
    if not found:
        raise ValueError("the sources of the script state no time")
    return min(found), max(found)


@timed("mapping_windows.thing")
def thing(mapped, stated, where):
    from .mapping_build import NodeStated

    target = stated.get("of")
    if isinstance(target, Part):
        target = mapped.part(target, where + " of")
    if not isinstance(target, Mapping) or id(target) not in mapped.stated:
        raise ValueError("%s names under of a mapping stated above it, or a table of a database" % where)
    found = mapped.stated[id(target)]
    if not isinstance(found, NodeStated) or "key" not in target.fields:
        raise ValueError("%s names under of a nodes statement that states key, or a kind of a database" % where)
    return found


@timed("mapping_windows.every")
def every(mapped, label, rule, session):
    started = time.perf_counter()
    where = "declaration %s" % label
    name, stated = ruled(rule, where + " rows")
    try:
        entry = EVERY.named(name)
    except KeyError as error:
        raise ValueError("%s rows: %s" % (where, error)) from error
    found = thing(mapped, stated, where + " rows")
    times = stated.get("times")
    if not isinstance(times, Held) or set(times) != {"first", "last", "every"}:
        raise ValueError("%s rows states times as a mapping of first, last and every" % where)
    unknown = sorted(set(stated) - {"of", "times", "names"})
    if unknown:
        raise ValueError("%s rows holds fields it does not know: %s" % (where, ", ".join(unknown)))
    reader = found.reader
    since = [row[2]["since"] for row in found.rows]
    until = [row[2]["until"] for row in found.rows]
    edges = bounds(session) if None in (times["first"], times["last"]) else None
    first = [max(edges[0], -math.inf if value is None else value) for value in since] if times["first"] is None \
        else reader.moments(times["first"], "rows.times.first")
    last = [min(edges[1], math.inf if value is None else value - 1) for value in until] if times["last"] is None \
        else reader.moments(times["last"], "rows.times.last")
    facts = pa.table(dict(row=pa.array(range(reader.count), pa.int64()),
                          key=[joined(row[0]) + "|" + joined(row[1]) for row in found.rows],
                          since=pa.array(since, pa.int64()), until=pa.array(until, pa.int64()),
                          first=pa.array(first, pa.int64()), last=pa.array(last, pa.int64())))
    made = entry(facts, dict(every=length(reader.context, times["every"], where + " rows.times.every")))
    if not isinstance(made, pa.Table) or set(made.column_names) != {"row", "moment"}:
        raise ValueError("%s rows: the rule %s gives a table of row and moment" % (where, name))
    moment = generated(reader.frame, dict(moment=None), dict(stated.get("names") or {}))["moment"]
    taken = take(reader.frame, made["row"].to_pylist())
    unit = reader.frame.unit
    frame = Frame(taken.table.append_column(moment, made["moment"].cast(pa.int64())),
                  dict(taken.kinds, **{moment: Kind("time", unit)}), unit, taken.through, taken.outcomes,
                  taken.lineage, taken.provenance, taken.complete, taken.positions)
    LOG.debug("rows exit declaration=%s tasks=%d seconds=%.6f", label, frame.table.num_rows,
              time.perf_counter() - started)
    return reader.framed(frame), moment
