import logging
import math
import time
from collections.abc import Mapping as Held

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .database import Part
from .expression_types import Kind
from .mapping_rows import Reader, joined, learned
from .mappings import Mapping
from .table_types import Frame, generated, guard, ruled, take

LOG = logging.getLogger(__name__)
EVERY = Names("rule of the rows of tasks", True)
WINDOW = ("rows", "on", "at", "until", "within")
CANDIDATE = 256


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
        self.within, self.at = within, at
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


@timed("mapping_windows.made")
def made(chosen):
    return pa.table(dict(row=pa.array([row for row, _ in chosen], pa.int64()),
                         moment=pa.array([moment for _, moment in chosen], pa.int64())))


class Nodes:
    version = 2

    @timed("mapping_windows.Nodes")
    def __call__(self, facts, parameters):
        if "every" not in parameters:
            raise ValueError("the rule nodes states times, as a mapping of first, last and every")
        if set(parameters) != {"every"}:
            raise ValueError("the rule nodes holds fields it does not know: %s"
                             % ", ".join(sorted(set(parameters) - {"every"})))
        every, best, spans = parameters["every"], {}, []
        names = ("row", "key", "since", "until", "known", "replaced", "first", "last")
        for row, key, since, until, known, replaced, first, last in zip(*(facts[name].to_pylist() for name in names)):
            low = first if since is None else max(first, since)
            start = first + -(-(low - first) // every)
            end = last if until is None else min(last, until - 1)
            spans.append((row, key, since, known, replaced, start, end))
        guard(sum(max(0, (end - start) // every + 1) for *_, start, end in spans) * CANDIDATE,
              "the tasks the rule nodes makes")
        for row, key, since, known, replaced, start, end in spans:
            seen = -math.inf if known is None and since is None else since if known is None else known
            for moment in range(start, end + 1, every):
                if seen <= moment and (replaced is None or moment < replaced):
                    held = best.get((key, moment))
                    begun = -math.inf if since is None else since
                    if held is None or held[1] <= begun:
                        best[(key, moment)] = (row, begun)
        return made(sorted((row, moment) for (_, moment), (row, _) in best.items()))


class Facts:
    version = 1

    @timed("mapping_windows.Facts")
    def __call__(self, facts, parameters):
        if parameters:
            raise ValueError("the rule facts states of alone, and names its rows' times as names states")
        guard(facts.num_rows * CANDIDATE, "the tasks the rule facts makes")
        chosen, places = [], set()
        for row, place, since in zip(*(facts[name].to_pylist() for name in ("row", "place", "since"))):
            if place in places:
                continue
            if since is None:
                raise ValueError("the fact of row %d holds from before every time, and dates no task" % place)
            places.add(place)
            chosen.append((row, since))
        return made(chosen)


EVERY.register("nodes", Nodes())
EVERY.register("facts", Facts())


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
    from .mapping_build import EdgeStated, NodeStated, SetStated

    target = stated.get("of")
    if isinstance(target, Part):
        target = mapped.part(target, where + " of")
    if not isinstance(target, Mapping) or id(target) not in mapped.stated:
        raise ValueError("%s names under of a mapping stated above it, or a table of a database" % where)
    found = mapped.stated[id(target)]
    if isinstance(found, NodeStated) and "key" in target.fields:
        return found, [(place, joined(kind) + "|" + joined(key), fact)
                       for (kind, key, fact, _), place in zip(found.rows, found.places)]
    if isinstance(found, SetStated):
        return found, [(place, found.label + "|" + joined(key), fact)
                       for key, fact, place in zip(found.keys, found.facts, found.places)]
    if isinstance(found, EdgeStated):
        if "times" in stated and any(fact["form"] == "change" for fact in found.facts):
            raise ValueError("%s states times over edges of the form change, whose ends are found only after their "
                             "nodes resolve; state them in the form state" % where)
        keys = found.keys or [None] * len(found.facts)
        return found, [(place, None if key is None else found.label + ".edge|" + joined(key), fact)
                       for place, (key, fact) in enumerate(zip(keys, found.facts))]
    raise ValueError("%s names under of a nodes statement that states key, a sets, an edges, or a kind of a database"
                     % where)


@timed("mapping_windows.every")
def every(mapped, label, rule, session):
    started = time.perf_counter()
    where = "declaration %s" % label
    name, stated = ruled(rule, where + " rows")
    try:
        entry = EVERY.named(name)
    except KeyError as error:
        raise ValueError("%s rows: %s" % (where, error)) from error
    if "of" not in stated:
        raise ValueError("%s rows lacks of, the mapping whose facts it reads" % where)
    statement, found = thing(mapped, stated, where + " rows")
    reader = statement.reader
    since =[fact["since"] for _, _, fact in found]
    until = [fact["until"] for _, _, fact in found]
    columns = dict(row=pa.array(range(len(found)), pa.int64()), place=pa.array([row[0] for row in found], pa.int64()),
                   key=pa.array([row[1] for row in found], pa.string()), since=pa.array(since, pa.int64()),
                   until=pa.array(until, pa.int64()), known=pa.array([fact["known"] for _, _, fact in found], pa.int64()),
                   replaced=pa.array([fact["replaced"] for _, _, fact in found], pa.int64()))
    parameters = {field: value for field, value in stated.items() if field not in ("of", "names", "times")}
    if "times" in stated:
        times = stated["times"]
        if not isinstance(times, Held) or set(times) != {"first", "last", "every"}:
            raise ValueError("%s rows states times as a mapping of first, last and every" % where)
        if None in [key for _, key, _ in found]:
            raise ValueError("%s rows states times over the facts of edges that state no key" % where)
        edges = bounds(session) if None in (times["first"], times["last"]) else None
        places = [row[0] for row in found]
        if times["first"] is None:
            first = [max(edges[0], -math.inf if value is None else value) for value in since]
        else:
            stated_first = reader.moments(times["first"], "rows.times.first")
            first = [stated_first[place] for place in places]
        if times["last"] is None:
            last = [min(edges[1], math.inf if value is None else value - 1) for value in until]
        else:
            stated_last = reader.moments(times["last"], "rows.times.last")
            last = [stated_last[place] for place in places]
        columns.update(first=pa.array(first, pa.int64()), last=pa.array(last, pa.int64()))
        parameters["every"] = length(reader.context, times["every"], where + " rows.times.every")
    try:
        result = entry(pa.table(columns), parameters)
    except ValueError as error:
        raise ValueError("%s rows: %s" % (where, error)) from error
    if not isinstance(result, pa.Table) or set(result.column_names) != {"row", "moment"}:
        raise ValueError("%s rows: the rule %s gives a table of row and moment" % (where, name))
    chosen = result["row"].to_pylist()
    moment = generated(reader.frame, dict(moment=None), dict(stated.get("names") or {}))["moment"]
    taken = take(reader.frame, [found[row][0] for row in chosen])
    unit = reader.frame.unit
    frame = Frame(taken.table.append_column(moment, result["moment"].cast(pa.int64())),
                  dict(taken.kinds, **{moment: Kind("time", unit)}), unit, taken.through, taken.outcomes,
                  taken.lineage, taken.provenance, taken.complete, taken.positions)
    known = [learned(dict(since=[found[row][2]["since"]], known=[found[row][2]["known"]]))[0] for row in chosen]
    LOG.debug("rows exit declaration=%s rule=%s tasks=%d seconds=%.6f", label, name, frame.table.num_rows,
              time.perf_counter() - started)
    return reader.framed(frame), moment, known
