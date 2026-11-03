import math
from collections import deque
from collections.abc import Mapping
from fractions import Fraction

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.schema.fields import ColumnValueError

from .expression_temporal import fixed, shift, time_count
from .expression_types import NEVER_TIME, Kind, common, datatype
from .table_types import (
    TABLES,
    broadcast,
    carry,
    column,
    combine_provenance,
    dependencies,
    fresh,
    generated,
    take,
    working,
)


@timed("table_relations.mapping")
def mapping(value, field):
    if not isinstance(value, Mapping) or any(type(name) is not str or not name for name in value):
        raise ValueError("%s is a mapping with nonempty column names" % field)
    return value


@timed("table_relations.expression")
def expression(value, field):
    if type(value) is not str or not value:
        raise ValueError("%s is a nonempty expression" % field)


@timed("table_relations.length")
def length(value, field):
    if type(value) is str:
        expression(value, field)
    elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("%s is a nonnegative finite number or a length expression" % field)


@timed("table_relations.key")
def key(value):
    if isinstance(value, (list, tuple)):
        return ("list", tuple(key(member) for member in value))
    if isinstance(value, dict):
        return ("record", tuple((name, key(member)) for name, member in value.items()))
    return ("value", value)


@timed("table_relations.marks")
def marks(*values):
    return tuple(dict.fromkeys(item for group in values for item in group))


@timed("table_relations.length_values")
def length_values(frame, value, field, context):
    result = context.evaluate(frame, value, field) if type(value) is str else context.constant(value, field)
    if result.kind.name not in ("number", "ordered", "duration"):
        raise ValueError("%s is a number or duration" % field)
    values = broadcast(result, frame.table.num_rows).to_pylist()
    for row, amount in enumerate(values, 1):
        if amount is not None and (not math.isfinite(amount) or amount < 0):
            raise ColumnValueError(row, "%s is a nonnegative finite length" % field)
    return values, result.kind


@timed("table_relations.columns")
def columns(value, field, empty=True):
    if not isinstance(value, (list, tuple)) or not empty and not value \
            or any(type(name) is not str or not name for name in value) or len(set(value)) != len(value):
        raise ValueError("%s states distinct column names%s" % (field, "" if empty else " and is nonempty"))


@timed("table_relations.renames")
def renames(parameters, allowed):
    stated = mapping(parameters.get("names", {}), "names")
    if set(stated) - set(allowed) or any(type(value) is not str or not value for value in stated.values()):
        raise ValueError("names renames only generated columns with nonempty column names")
    resolved = [stated.get(name, name) for name in allowed]
    if len(set(resolved)) != len(resolved):
        raise ValueError("names gives distinct names to generated columns")


@timed("table_relations.gap")
def gap(kind, stated, context):
    value = context.constant(stated, "gap")
    if value.kind.name not in ("number", "ordered", "duration"):
        raise ValueError("order gap is one number or duration")
    amount = value.values.as_py()
    if amount is None or not math.isfinite(amount) or amount < 0:
        raise ValueError("order gap is a nonnegative finite length")
    if kind.name not in ("number", "ordered", "duration", "time"):
        raise ValueError("order gap reads a numeric or temporal first by column")
    if value.kind.name == "duration":
        if kind.name == "time":
            shift(NEVER_TIME, amount, value.kind.unit, kind.unit)
            if value.kind.unit in ("month", "year") and value.kind.unit != kind.unit:
                return "calendar", amount, value.kind.unit
            amount = fixed(amount, value.kind.unit, kind.unit)
        elif kind.name == "duration":
            if value.kind.unit in ("month", "year") or kind.unit in ("month", "year"):
                raise ValueError("order gap cannot compare calendar durations as fixed lengths")
            amount = fixed(amount, value.kind.unit, kind.unit)
        else:
            raise ValueError("a duration gap requires a declared temporal by column")
    else:
        amount = Fraction(str(amount))
    if kind.name == "time" and amount.denominator != 1:
        raise ValueError("order gap is a whole count of the graph unit %r" % kind.unit)
    return "number", amount, None


@timed("table_relations.grows")
def grows(previous, current, kind, boundary):
    mode, amount, unit = boundary
    if kind.name == "time":
        previous, current = time_count(previous), time_count(current)
        if mode == "calendar" and previous != NEVER_TIME:
            return current > shift(previous, amount, unit, kind.unit)
        if NEVER_TIME in (previous, current):
            return current != previous
        return current - previous > amount
    return Fraction(str(current)) - Fraction(str(previous)) > amount


class Join:
    version = 1
    fields = frozenset(("on", "take", "unmatched", "where", "window", "outcomes"))
    required = ("on", "take", "unmatched")
    input_count = (2, 2)

    @timed("tables.Join.validate")
    def validate(self, parameters):
        for name, text in mapping(parameters["on"], "join on").items():
            expression(text, "join on " + name)
        for name, held in mapping(parameters["take"], "join take").items():
            expression(held, "join take " + name)
        if parameters["unmatched"] not in ("drop", "keep", "stop"):
            raise ValueError("join unmatched is drop, keep or stop")
        if "where" in parameters:
            expression(parameters["where"], "join where")
        if "window" in parameters:
            window = mapping(parameters["window"], "join window")
            if set(window) != {"at", "other", "before", "after"}:
                raise ValueError("join window states exactly at, other, before and after")
            for field in ("at", "other"):
                expression(window[field], "join window " + field)
            for field in ("before", "after"):
                length(window[field], "join window " + field)
        for name, text in mapping(parameters.get("outcomes", {}), "join outcomes").items():
            if name not in parameters["take"]:
                raise ValueError("join outcomes names only columns brought by take: " + name)
            expression(text, "join outcome " + name)

    @timed("tables.Join.apply")
    def apply(self, frames, parameters, context):
        with working(context):
            self.validate(parameters)
            left, right = frames
            fresh(left, parameters["take"])
            kinds = dict(left.kinds)
            for name, source in parameters["take"].items():
                kinds[name] = column(right, source)
            context.check((left.nbytes + right.nbytes) * 8 + right.table.num_rows * 128
                          + (left.table.num_rows + right.table.num_rows) * len(parameters["on"]) * 96, "join key index")
            left_keys, right_keys = [], []
            for name, text in parameters["on"].items():
                result = context.evaluate(left, text, "on." + name)
                common(result.kind, column(right, name))
                left_keys.append(broadcast(result, left.table.num_rows).to_pylist())
                right_keys.append(right.table[name].to_pylist())
            index = {}
            for row in range(right.table.num_rows):
                values = tuple(values[row] for values in right_keys)
                if any(value is None for value in values):
                    continue
                index.setdefault(tuple(key(value) for value in values), []).append(row)
            window = self.window(left, right, parameters.get("window"), context)
            paired_left, paired_right, size = [], [], 0
            left_sizes, right_sizes = {}, {}
            taken = right.table.select(list(dict.fromkeys(parameters["take"].values())))
            considered = right.table if "where" in parameters else taken
            for row in range(left.table.num_rows):
                values = tuple(values[row] for values in left_keys)
                candidates = [] if any(value is None for value in values) else index.get(tuple(key(value) for value in values), [])
                if window is not None:
                    starts, stops, times = window
                    candidates = [] if starts[row] is None or stops[row] is None else [
                        candidate for candidate in candidates
                        if times[candidate] is not None and starts[row] <= times[candidate] <= stops[row]
                    ]
                if candidates and row not in left_sizes:
                    left_sizes[row] = left.table.slice(row, 1).nbytes + sum(value.slice(row, 1).nbytes for value in left.provenance.values())
                for candidate in candidates:
                    if candidate not in right_sizes:
                        right_sizes[candidate] = considered.slice(candidate, 1).nbytes + sum(
                            value.slice(candidate, 1).nbytes for value in right.provenance.values()
                        )
                    size += left_sizes[row] + right_sizes[candidate] + 32
                    context.check(size * 8 + (left.nbytes + right.nbytes) * 8, "join candidate rows")
                    paired_left.append(row)
                    paired_right.append(candidate)
            if "where" in parameters:
                repeated, other = take(left, paired_left), take(right, paired_right)
                predicate = context.evaluate(repeated, parameters["where"], "where", namespace=dict(other=other), positions=paired_left)
                if predicate.kind.name not in ("flag", "missing"):
                    raise ValueError("join where is a flag predicate")
                kept = [index for index, value in enumerate(broadcast(predicate, len(paired_left)).to_pylist()) if value is True]
                paired_left = [paired_left[index] for index in kept]
                paired_right = [paired_right[index] for index in kept]
                del repeated, other, predicate, kept
            left_rows, right_rows, position, size = [], [], 0, 0
            for row in range(left.table.num_rows):
                start = position
                while position < len(paired_left) and paired_left[position] == row:
                    position += 1
                candidates = paired_right[start:position]
                if not candidates:
                    if parameters["unmatched"] == "stop":
                        raise ColumnValueError(row + 1, "join has no match and unmatched is stop")
                    candidates = [None] if parameters["unmatched"] == "keep" else []
                for candidate in candidates:
                    size += left.table.slice(row, 1).nbytes + right_sizes.get(candidate, 0) + 32
                    context.check(size * 4 + (left.nbytes + right.nbytes) * 8, "join output rows")
                    left_rows.append(row)
                    right_rows.append(candidate)
            table = pc.take(left.table, pa.array(left_rows, pa.int64()))
            selected = pa.array(right_rows, pa.int64())
            for name, source in parameters["take"].items():
                table = table.append_column(name, pc.take(right.table[source], selected))
            context.check(table.nbytes * 4 + len(left_rows) * 64, "join output")
            outcomes, lineage = dict(left.outcomes), dict(left.lineage)
            for name, source in parameters["take"].items():
                lineage[name] = marks(right.lineage.get(source, ()), ((context.name, "other." + source),))
                inherited = right.outcomes.get(source, ())
                stated = ((context.name, parameters["outcomes"][name]),) if name in parameters.get("outcomes", {}) else ()
                if inherited or stated:
                    outcomes[name] = marks(inherited, stated)
            provenance = combine_provenance(frames, [
                [[row] for row in left_rows], [[] if row is None else [row] for row in right_rows],
            ])
            return carry(left, table, kinds, indices=left_rows, outcomes=outcomes, lineage=lineage, provenance=provenance)

    @timed("tables.Join.window")
    def window(self, left, right, parameters, context):
        if parameters is None:
            return None
        at = context.evaluate(left, parameters["at"], "window.at")
        other = context.evaluate(right, parameters["other"], "window.other")
        if at.kind.name != "time" or other.kind.name != "time":
            raise ValueError("join window at and other are declared times")
        common(at.kind, other.kind)
        before, before_kind = length_values(left, parameters["before"], "window.before", context)
        after, after_kind = length_values(left, parameters["after"], "window.after", context)
        starts, stops = [], []
        for row, (moment, past, future) in enumerate(zip(
            broadcast(at, left.table.num_rows).to_pylist(), before, after, strict=True
        ), 1):
            try:
                if moment is None or past is None or future is None:
                    starts.append(None)
                    stops.append(None)
                    continue
                starts.append(shift(moment, -past, before_kind.unit if before_kind.name == "duration" else at.kind.unit, at.kind.unit))
                stops.append(shift(moment, future, after_kind.unit if after_kind.name == "duration" else at.kind.unit, at.kind.unit))
            except (ValueError, TypeError, OverflowError) as error:
                raise ColumnValueError(row, "join window: %s" % error) from error
        return starts, stops, broadcast(other, right.table.num_rows).to_pylist()


class Closure:
    version = 1
    fields = frozenset(("source", "target", "most", "names"))
    required = ("source", "target", "most")
    input_count = (1, 1)

    @timed("tables.Closure.validate")
    def validate(self, parameters):
        for field in ("source", "target"):
            expression(parameters[field], "closure " + field)
        if parameters["source"] == parameters["target"]:
            raise ValueError("closure names distinct source and target columns")
        most = parameters["most"]
        if most is not None and (type(most) is not int or most < 0):
            raise ValueError("closure most is a nonnegative whole number or None")
        renames(parameters, ("steps",))

    @timed("tables.Closure.apply")
    def apply(self, frames, parameters, context):
        with working(context):
            self.validate(parameters)
            frame, = frames
            source, target = parameters["source"], parameters["target"]
            common(column(frame, source), column(frame, target))
            names = generated(frame, ("steps",), parameters.get("names"))
            context.check(frame.nbytes * 8 + frame.table.num_rows * 256, "closure adjacency")
            sources, targets = frame.table[source].to_pylist(), frame.table[target].to_pylist()
            nodes, adjacency, indegree, roots = {}, {}, {}, {}
            for row, (one, other) in enumerate(zip(sources, targets, strict=True)):
                if one is None:
                    raise ColumnValueError(row + 1, "closure source is missing")
                first = key(one)
                nodes.setdefault(first, one)
                adjacency.setdefault(first, {})
                indegree.setdefault(first, 0)
                roots.setdefault(first, row)
                if other is None:
                    continue
                last = key(other)
                nodes.setdefault(last, other)
                adjacency.setdefault(last, {})
                indegree.setdefault(last, 0)
                if last not in adjacency[first]:
                    adjacency[first][last] = row
                    indegree[last] += 1
            pending = deque(node for node in nodes if indegree[node] == 0)
            visited = 0
            while pending:
                node = pending.popleft()
                visited += 1
                for neighbor in adjacency[node]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        pending.append(neighbor)
            if visited != len(nodes):
                remaining = next(node for node in nodes if indegree[node])
                raise ValueError("closure contains a cycle; remaining key %r" % (nodes[remaining],))
            output_sources, output_targets, steps, parents, source_rows = [], [], [], [], []
            path_bytes = 0
            for root in roots:
                pending, seen = deque([(root, 0, ())]), {root}
                while pending:
                    node, distance, path = pending.popleft()
                    if parameters["most"] is not None and distance >= parameters["most"]:
                        continue
                    for neighbor, row in adjacency[node].items():
                        if neighbor in seen:
                            continue
                        seen.add(neighbor)
                        walked = (*path, row)
                        path_bytes += len(walked) * 8
                        context.check(frame.nbytes * 8 + (len(steps) + 1) * 192 + path_bytes * 8, "closure output paths")
                        output_sources.append(sources[roots[root]])
                        output_targets.append(nodes[neighbor])
                        steps.append(float(distance + 1))
                        parents.append(walked)
                        source_rows.append(roots[root])
                        pending.append((neighbor, distance + 1, walked))
            table = pa.Table.from_arrays([
                pa.array(output_sources, frame.table[source].type), pa.array(output_targets, frame.table[target].type),
                pa.array(steps, pa.float64()),
            ], names=[source, target, names["steps"]]).replace_schema_metadata(frame.table.schema.metadata)
            context.check(table.nbytes * 4 + path_bytes * 8, "closure output")
            kinds = {source: frame.kinds[source], target: frame.kinds[target], names["steps"]: Kind("number")}
            lineage = {name: frame.lineage.get(name, ()) for name in (source, target)}
            lineage[names["steps"]] = ((context.name, "closure steps"),)
            provenance = combine_provenance([frame], [parents])
            return carry(frame, table, kinds, indices=source_rows, lineage=lineage, provenance=provenance)


class Order:
    version = 1
    fields = frozenset(("within", "by", "gap", "columns", "names"))
    required = ("within", "by", "gap")
    input_count = (1, 1)

    @timed("tables.Order.validate")
    def validate(self, parameters):
        columns(parameters["within"], "order within")
        columns(parameters["by"], "order by", empty=False)
        if parameters["gap"] is not None:
            length(parameters["gap"], "order gap")
        for name, text in mapping(parameters.get("columns", {}), "order columns").items():
            expression(text, "order column " + name)
        renames(parameters, ("run", "place"))

    @timed("tables.Order.apply")
    def apply(self, frames, parameters, context):
        with working(context):
            self.validate(parameters)
            frame, = frames
            names = generated(frame, ("run", "place"), parameters.get("names"))
            additions = parameters.get("columns", {})
            fresh(frame, [*names.values(), *additions])
            for name in parameters["within"]:
                column(frame, name)
            ordered = {name: column(frame, name) for name in parameters["by"]}
            if any(kind.name not in ("number", "ordered", "duration", "time", "category", "identifier", "text", "flag")
                   for kind in ordered.values()):
                raise ValueError("order by columns have scalar ordered types")
            count = frame.table.num_rows
            context.check(frame.nbytes * 8 + count * (len(parameters["within"]) + len(ordered) + 8) * 96, "order groups")
            by = {name: frame.table[name].to_pylist() for name in ordered}
            for name, values in by.items():
                for row, value in enumerate(values, 1):
                    if value is None:
                        raise ColumnValueError(row, "order by column %s is missing" % name)
                    if ordered[name].name in ("number", "ordered", "duration") and not math.isfinite(value):
                        raise ColumnValueError(row, "order by column %s is finite" % name)
                    if ordered[name].name == "time":
                        try:
                            time_count(value)
                        except (TypeError, ValueError, OverflowError) as error:
                            raise ColumnValueError(row, "order by column %s: %s" % (name, error)) from error
            first = parameters["by"][0]
            boundary = None if parameters["gap"] is None else gap(ordered[first], parameters["gap"], context)
            within = [frame.table[name].to_pylist() for name in parameters["within"]]
            groups = {}
            for row in range(count):
                groups.setdefault(tuple(key(values[row]) for values in within), []).append(row)
            runs, places = [0.0] * count, [0.0] * count
            neighbors = dict(previous=[None] * count, following=[None] * count)
            for group in groups.values():
                arranged = sorted(group, key=lambda row: tuple(values[row] for values in by.values()))
                run, place, previous = 0, 0, None
                for row in arranged:
                    try:
                        new = previous is not None and boundary is not None and grows(by[first][previous], by[first][row], ordered[first], boundary)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise ColumnValueError(row + 1, "order gap: %s" % error) from error
                    if new:
                        run, place = run + 1, 0
                    elif previous is not None:
                        neighbors["previous"][row] = previous
                        neighbors["following"][previous] = row
                    runs[row], places[row] = float(run), float(place)
                    place += 1
                    previous = row
            table = frame.table.append_column(names["run"], pa.array(runs, pa.float64()))
            table = table.append_column(names["place"], pa.array(places, pa.float64()))
            kinds = dict(frame.kinds, **{name: Kind("number") for name in names.values()})
            lineage, outcomes = dict(frame.lineage), dict(frame.outcomes)
            for field, name in names.items():
                lineage[name] = ((context.name, field),)
            original = carry(frame, table, kinds, lineage=lineage, outcomes=outcomes)
            for name, text in additions.items():
                result = context.evaluate(original, text, "columns." + name, ordered=True, neighbors=neighbors)
                values = broadcast(result, count).cast(datatype(result.kind), safe=True)
                table = table.append_column(name, values)
                kinds[name] = result.kind
                inherited, known = dependencies(original, text)
                lineage[name] = marks(inherited, ((context.name, text),))
                if known:
                    outcomes[name] = known
                context.check(table.nbytes * 4 + frame.nbytes * 4 + count * 64, "order output columns")
            return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage)


TABLES.register("join", Join())
TABLES.register("closure", Closure())
TABLES.register("order", Order())
