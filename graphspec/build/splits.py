import math
from collections import Counter
from collections.abc import Mapping as Held

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .expression_temporal import shift
from .table_types import ruled

SPLITS = Names("form of split", True)
GIVEN = ("time", "graph")


@timed("splits.where")
def where(reader, field="split"):
    return "declaration %s field %s" % (reader.name, field)


@timed("splits.fields")
def fields(reader, parameters, required, optional=()):
    unknown = sorted(set(parameters) - set(required) - set(optional))
    lacking = [name for name in required if name not in parameters]
    if unknown or lacking:
        raise ValueError("%s holds fields it does not know or lacks: %s" % (where(reader), ", ".join(unknown + lacking)))


@timed("splits.salted")
def salted(reader, parameters):
    salt = parameters["salt"]
    if type(salt) is not str or not salt:
        raise ValueError("%s states salt as a nonempty text, found %r" % (where(reader, "split.salt"), salt))
    return salt


@timed("splits.parted")
def parted(reader, parts):
    if not isinstance(parts, Held) or not parts:
        raise ValueError("%s states a mapping from each split to its part" % where(reader, "split.parts"))
    for name in parts:
        if type(name) is not str or not name:
            raise ValueError("%s names each split by a nonempty text, found %r" % (where(reader, "split.parts"), name))
    return list(parts.items())


@timed("splits.shares")
def shares(reader, parts):
    total, bounds = 0.0, []
    for name, share in parted(reader, parts):
        if type(share) not in (int, float) or not share > 0:
            raise ValueError("%s states the share of %s as a number above zero, found %r"
                             % (where(reader, "split.parts"), name, share))
        total += share
        bounds.append([name, total])
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("%s states shares that sum to 1, found %r" % (where(reader, "split.parts"), total))
    bounds[-1][1] = math.inf
    return bounds


@timed("splits.shared")
def shared(fraction, bounds):
    return next(name for name, bound in bounds if fraction < bound)


class Hash:
    version = 1
    field = "key"

    @timed("splits.Hash.read")
    def read(self, reader, parameters):
        fields(reader, parameters, (self.field, "parts", "salt"))
        field, stated = "split." + self.field, parameters[self.field]
        keys = list(stated) if isinstance(stated, (list, tuple)) else [stated]
        if not keys or any(type(key) is not str or not key for key in keys):
            raise ValueError("%s states an expression or a list of expressions" % where(reader, field))
        text = "hash(%s, salt=%r)" % (", ".join("(%s)" % key for key in keys), salted(reader, parameters))
        values, _ = reader.values(text, field)
        for place, value in enumerate(values):
            if value is None:
                raise ValueError("%s row %d reads a missing value as a key" % (where(reader, field), reader.row(place)))
        return values, shares(reader, parameters["parts"])

    @timed("splits.Hash.call")
    def __call__(self, prepared, tasks):
        values, bounds = prepared
        return [shared(values[row], bounds) for row in tasks["row"]]


class Group(Hash):
    version = 1
    field = "by"


class Graph:
    version = 1

    @timed("splits.Graph.read")
    def read(self, reader, parameters):
        fields(reader, parameters, ("parts", "salt"))
        return reader.context, salted(reader, parameters), shares(reader, parameters["parts"])

    @timed("splits.Graph.call")
    def __call__(self, prepared, tasks):
        context, salt, bounds = prepared
        units = {graph: context.constant("hash(%r, salt=%r)" % (graph, salt), "split").values.as_py()
                 for graph in set(tasks["graph"])}
        return [shared(units[graph], bounds) for graph in tasks["graph"]]


class Time:
    version = 1

    @timed("splits.Time.read")
    def read(self, reader, parameters):
        fields(reader, parameters, ("parts",), ("by",))
        by = None
        if "by" in parameters:
            if type(parameters["by"]) is not str or not parameters["by"]:
                raise ValueError("%s states an expression giving a time" % where(reader, "split.by"))
            by = reader.moments(parameters["by"], "split.by")
        parts = parted(reader, parameters["parts"])
        if parts[0][1] is not None:
            return by, shares(reader, parameters["parts"]), None
        lengths, unit = [], reader.context.unit
        for name, stated in parts[1:]:
            field = "split.parts." + name
            if stated is None:
                raise ValueError("%s: only the first split states None and takes the rest" % where(reader, field))
            value = reader.context.constant(stated, field)
            amount = value.values.as_py() if hasattr(value.values, "as_py") else value.values
            if value.type not in ("number", "duration") or amount is None or not amount > 0:
                raise ValueError("%s states a length of time above zero, found %r" % (where(reader, field), stated))
            lengths.append((name, amount, unit if value.type == "number" else value.unit))
        return by, parts[0][0], (lengths, unit)

    @timed("splits.Time.call")
    def __call__(self, prepared, tasks):
        by, rule, lengths = prepared
        times = tasks["time"] if by is None else [by[row] for row in tasks["row"]]
        if None in times:
            raise ValueError("the form time orders the tasks by their time, and a task has none; a kind of task "
                             "without at states by")
        counts, named = Counter(times), {}
        if not times:
            return []
        if lengths is None:
            before = 0
            for moment in sorted(counts):
                named[moment] = shared(before / len(times), rule)
                before += counts[moment]
            return [named[moment] for moment in times]
        held, unit = lengths
        lowers, upper = [], max(times)
        for name, amount, own in reversed(held):
            upper = shift(upper, -amount, own, unit)
            lowers.append((name, upper))
        for moment in counts:
            named[moment] = next((name for name, lower in lowers if moment > lower), rule)
        return [named[moment] for moment in times]


class Divided:
    @timed("splits.Divided")
    def __init__(self, function, version):
        self.function, self.version = function, version

    @timed("splits.Divided.read")
    def read(self, reader, parameters):
        frame, ending = reader.frame, reader.knowing.ending
        columns = [name for name in frame.table.column_names
                   if not frame.outcomes.get(name) and not set(frame.lineage.get(name, ())) & ending]
        clash = [name for name in GIVEN if name in columns]
        if clash:
            raise ValueError("%s: its rows hold the column %s, which the table a function of split is given names"
                             % (where(reader), ", ".join(clash)))
        return frame.table.select(columns), dict(parameters)

    @timed("splits.Divided.call")
    def __call__(self, prepared, tasks):
        rows, parameters = prepared
        table = rows.take(pa.array(tasks["row"], pa.int64()))
        table = table.append_column("time", pa.array(tasks["time"], pa.int64()))
        return self.function(table.append_column("graph", pa.array(tasks["graph"], pa.string())), parameters)


SPLITS.register("hash", Hash())
SPLITS.register("group", Group())
SPLITS.register("graph", Graph())
SPLITS.register("time", Time())


class Splitting:
    @timed("splits.Splitting")
    def __init__(self, reader, stated):
        self.label, self.row, self.form = reader.name, reader.row, None
        self.named = [None] * reader.count
        if stated is None:
            return
        if type(stated) is str:
            self.named = reader.texts(stated, "split")
            return
        self.form, parameters = ruled(stated, where(reader))
        try:
            self.entry = SPLITS.named(self.form)
        except KeyError as error:
            raise ValueError("%s: %s" % (where(reader), error.args[0])) from error
        self.prepared = self.entry.read(reader, parameters)

    @timed("splits.Splitting.members")
    def members(self, rows, times, graphs):
        if self.form is None:
            return [self.named[row] for row in rows]
        try:
            found = self.entry(self.prepared, dict(row=list(rows), time=list(times), graph=list(graphs)))
        except ValueError as error:
            raise ValueError("declaration %s field split: %s" % (self.label, error)) from error
        found = found.to_pylist() if isinstance(found, (pa.Array, pa.ChunkedArray)) else list(found)
        if len(found) != len(rows):
            raise ValueError("declaration %s field split: the form %s gives %d names for %d tasks"
                             % (self.label, self.form, len(found), len(rows)))
        for name, row in zip(found, rows):
            if type(name) is not str or not name:
                raise ValueError("declaration %s field split row %d: the form %s gives %r, not the name of a split"
                                 % (self.label, self.row(row), self.form, name))
        return found
