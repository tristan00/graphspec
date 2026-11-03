from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.schema.fields import ColumnValueError

from .aggregates import key
from .expression_types import Kind, datatype
from .table_types import TABLES, carry, column, combine_provenance, dependencies, generated, take


@timed("table_reshape.names")
def names(values, field):
    if not isinstance(values, (list, tuple)) or any(type(value) is not str or not value for value in values):
        raise ValueError("%s is a list of nonempty column names" % field)
    if len(set(values)) != len(values):
        raise ValueError("%s states each column once" % field)


@timed("table_reshape.expression")
def expression(value, field):
    if type(value) is not str or not value:
        raise ValueError("%s is a nonempty expression" % field)


@timed("table_reshape.columns")
def columns(values, field):
    if not isinstance(values, Mapping) or any(type(name) is not str or not name for name in values):
        raise ValueError("%s is a mapping of column names to expressions" % field)
    for name, value in values.items():
        expression(value, "%s.%s" % (field, name))


@timed("table_reshape.joined")
def joined(*values):
    return tuple(dict.fromkeys(item for value in values for item in value))


@timed("table_reshape.materialization")
def materialization(values):
    if isinstance(values, pa.ChunkedArray):
        return sum(materialization(chunk) for chunk in values.chunks)
    kind = values.type
    size = values.nbytes * 4 + len(values) * 64
    if pa.types.is_dictionary(kind):
        return size + len(values) * materialization(values.dictionary)
    if pa.types.is_struct(kind):
        return size + sum(materialization(values.field(index)) for index in range(kind.num_fields))
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind) or pa.types.is_map(kind):
        return size + materialization(values.values)
    return size


@timed("table_reshape.materialized")
def materialized(frame, fields, context, operation):
    context.check(frame.table.nbytes + sum(materialization(frame.table[name]) for name in fields), operation)


@timed("table_reshape.provenance")
def provenance(frame, parent, context):
    size = sum(sum(values.slice(row, 1).nbytes for row in parent) for values in frame.provenance.values())
    context.check(size * 4 + len(parent) * 16, "repeating row provenance")
    selected = pa.array(parent, pa.int64())
    return {name: pc.take(values, selected) for name, values in frame.provenance.items()}


@timed("table_reshape.groups")
def groups(frame, by, context):
    for name in by:
        column(frame, name)
    count = frame.table.num_rows
    context.check(frame.table.nbytes * 4 + count * (len(by) + 2) * 64, "grouping complete table rows")
    if not by:
        return [list(range(count))]
    materialized(frame, by, context, "materializing group keys")
    values = [frame.table[name].to_pylist() for name in by]
    seen, result = {}, []
    for row in range(count):
        identity = tuple(key(values[position][row]) for position in range(len(by)))
        if identity not in seen:
            seen[identity] = len(result)
            result.append([])
        result[seen[identity]].append(row)
    return result


@timed("table_reshape.metadata")
def metadata(frame, kept, computed, context, controls=()):
    lineage = {name: frame.lineage.get(name, ()) for name in kept}
    outcomes = {name: frame.outcomes[name] for name in kept if name in frame.outcomes}
    control_lineage = joined(*(frame.lineage.get(name, ()) for name in controls))
    control_marks = joined(*(frame.outcomes.get(name, ()) for name in controls))
    for name, text in computed.items():
        inherited, marks = dependencies(frame, text)
        lineage[name] = joined(inherited, control_lineage, ((context.name, text),))
        marks = joined(marks, control_marks)
        if marks:
            outcomes[name] = marks
    return lineage, outcomes


@timed("table_reshape.scalar")
def scalar(context, frame, text, field, kind):
    value = context.evaluate(frame, text, field)
    if not isinstance(value.values, pa.Scalar):
        raise ValueError("%s is an aggregate giving one value per group" % field)
    if value.kind != kind or value.values.type != datatype(kind):
        raise ValueError("%s keeps one declared type in every group" % field)
    return value.values


@timed("table_reshape.collected")
def collected(values, kind, context):
    parts, size = [], 0
    for value in values:
        part = pa.repeat(value, 1)
        size += part.nbytes
        context.check(size * 4 + len(values) * 64, "assembling aggregate values")
        parts.append(part)
    return pa.concat_arrays(parts) if parts else pa.array([], type=datatype(kind))


class Group:
    version = 1
    fields = frozenset(("by", "columns"))
    required = ("by", "columns")
    input_count = (1, 1)

    @timed("table_reshape.Group.validate")
    def validate(self, parameters):
        names(parameters["by"], "by")
        columns(parameters["columns"], "columns")
        if not parameters["by"] and not parameters["columns"]:
            raise ValueError("group states at least one key or aggregate column")

    @timed("table_reshape.Group.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        by, computed = parameters["by"], parameters["columns"]
        generated(frame, tuple(computed), None)
        partitions = groups(frame, by, context)
        kinds = {name: column(frame, name) for name in by}
        kinds.update({name: context.infer(frame, text, "columns." + name, aggregate=True) for name, text in computed.items()})
        first = pa.array([rows[0] for rows in partitions], pa.int64()) if by else None
        output = {name: pc.take(frame.table[name], first) for name in by}
        reduced = {name: [] for name in computed}
        for rows in partitions:
            selected = take(frame, rows)
            for name, text in computed.items():
                reduced[name].append(scalar(context, selected, text, "columns." + name, kinds[name]))
        for name in computed:
            output[name] = collected(reduced[name], kinds[name], context)
        table = pa.table(output)
        context.check(frame.table.nbytes + table.nbytes * 4, "grouped table")
        lineage, outcomes = metadata(frame, by, computed, context, by)
        return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage,
                     provenance=combine_provenance(frames, [partitions]))


class Pivot:
    version = 1
    fields = frozenset(("by", "names", "value"))
    required = ("by", "names", "value")
    input_count = (1, 1)

    @timed("table_reshape.Pivot.validate")
    def validate(self, parameters):
        names(parameters["by"], "by")
        expression(parameters["names"], "names")
        expression(parameters["value"], "value")

    @timed("table_reshape.Pivot.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        by, field, text = parameters["by"], parameters["names"], parameters["value"]
        if column(frame, field).name not in ("identifier", "category", "text"):
            raise ValueError("pivot names column %s is text, category or identifier" % field)
        materialized(frame, [field], context, "materializing pivot names")
        values = frame.table[field].to_pylist()
        for row, value in enumerate(values, 1):
            if type(value) is not str or not value:
                raise ColumnValueError(row, "pivot names column %s holds nonempty strings" % field)
        labels = list(dict.fromkeys(values))
        generated(frame, labels, None)
        partitions = groups(frame, by, context)
        if not frame.table.num_rows and not by:
            partitions = []
        kind = context.infer(frame, text, "value", aggregate=True)
        kinds = {name: column(frame, name) for name in by}
        kinds.update({label: kind for label in labels})
        first = pa.array([rows[0] for rows in partitions], pa.int64()) if by else None
        output = {name: pc.take(frame.table[name], first) for name in by}
        for label in labels:
            results = []
            for rows in partitions:
                selected = take(frame, [row for row in rows if values[row] == label])
                results.append(scalar(context, selected, text, "value." + label, kind))
            output[label] = collected(results, kind, context)
        table = pa.table(output)
        context.check(frame.table.nbytes + table.nbytes * 4, "pivoted table")
        lineage, outcomes = metadata(frame, by, {label: text for label in labels}, context, [*by, field])
        return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage,
                     provenance=combine_provenance(frames, [partitions]))


class Unpivot:
    version = 1
    fields = frozenset(("keep", "name", "value"))
    required = ("keep", "name", "value")
    input_count = (1, 1)

    @timed("table_reshape.Unpivot.validate")
    def validate(self, parameters):
        names(parameters["keep"], "keep")
        for field in ("name", "value"):
            expression(parameters[field], field)
        if parameters["name"] == parameters["value"]:
            raise ValueError("unpivot name and value are distinct new columns")

    @timed("table_reshape.Unpivot.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        keep, name, value = parameters["keep"], parameters["name"], parameters["value"]
        for kept in keep:
            column(frame, kept)
        generated(frame, (name, value), None)
        spread = [field for field in frame.table.column_names if field not in keep]
        if not spread:
            raise ValueError("unpivot reads at least one column not kept")
        kind = column(frame, spread[0])
        if any(column(frame, field) != kind for field in spread[1:]):
            raise ValueError("unpivot columns not kept have one exact declared type")
        count, width = frame.table.num_rows, len(spread)
        context.check(frame.table.nbytes * 4 + count * width * 80, "unpivoting table rows")
        parent = [row for row in range(count) for _ in spread]
        indices = pa.array([position * count + row for row in range(count) for position in range(width)], pa.int64())
        stacked = pa.concat_arrays([frame.table[field].combine_chunks() for field in spread])
        output = {field: pc.take(frame.table[field], pa.array(parent, pa.int64())) for field in keep}
        output[name] = pa.array([field for _ in range(count) for field in spread], pa.string())
        output[value] = pc.take(stacked, indices)
        kinds = {field: column(frame, field) for field in keep}
        kinds.update({name: Kind("identifier"), value: kind})
        lineage = {field: frame.lineage.get(field, ()) for field in keep}
        lineage[name] = ((context.name, name),)
        lineage[value] = joined(*(frame.lineage.get(field, ()) for field in spread), ((context.name, value),))
        outcomes = {field: frame.outcomes[field] for field in keep if field in frame.outcomes}
        marks = joined(*(frame.outcomes.get(field, ()) for field in spread))
        if marks:
            outcomes[value] = marks
        return carry(frame, pa.table(output), kinds, lineage=lineage, outcomes=outcomes,
                     provenance=provenance(frame, parent, context))


TABLES.register("group", Group())
TABLES.register("pivot", Pivot())
TABLES.register("unpivot", Unpivot())
