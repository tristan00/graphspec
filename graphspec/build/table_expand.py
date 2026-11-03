from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from graphspec.infra.timing import timed
from graphspec.schema.columns import checked as checked_column
from graphspec.schema.columns import statement
from graphspec.schema.fields import ColumnValueError
from graphspec.schema.temporal import whole

from .expression_types import Kind, Value
from .expressions import column_kind
from .table_reshape import expression, joined, materialized, provenance
from .table_types import (
    TABLES,
    broadcast,
    carry,
    column,
    combine_provenance,
    dependencies,
    fresh,
    generated,
    typed,
)


@timed("table_expand.fields")
def fields(values):
    if not isinstance(values, Mapping) or any(type(name) is not str or not name for name in values):
        raise ValueError("columns states a mapping of record fields to source column declarations")
    for name, given in values.items():
        try:
            declared, _, _ = checked_column(given)
            if "outcome" in declared:
                expression(declared["outcome"], "outcome")
        except Exception as error:
            raise ValueError("columns.%s: %s" % (name, error)) from error


@timed("table_expand.names")
def names(defaults, given):
    if len(set(defaults)) != len(defaults):
        raise ValueError("record fields and generated columns have distinct names")
    if given is None:
        return
    if not isinstance(given, Mapping) or set(given) - set(defaults):
        raise ValueError("names renames only the fields and columns this operation generates")
    output = [given.get(name, name) for name in defaults]
    if any(type(name) is not str or not name for name in output) or len(set(output)) != len(output):
        raise ValueError("names states distinct nonempty output names")


@timed("table_expand.counts")
def counts(frame, text, context, field):
    value = context.evaluate(frame, text, field)
    if value.kind.name not in ("number", "ordered"):
        raise ValueError("%s gives numeric counts" % field)
    result = []
    for row, scalar in enumerate(broadcast(value, frame.table.num_rows), 1):
        try:
            count = whole(scalar.as_py())
            if count < 0:
                raise ValueError("counts are nonnegative")
        except (TypeError, ValueError, OverflowError) as error:
            raise ColumnValueError(row, "%s is a nonmissing nonnegative whole count: %s" % (field, error)) from error
        result.append(count)
    return result


@timed("table_expand.budget")
def budget(frame, kept, lengths, context, operation):
    table = frame.table.select(kept)
    size = sum(table.slice(row, 1).nbytes * count for row, count in enumerate(lengths))
    context.check(frame.nbytes + size * 4 + sum(lengths) * 256, operation)


@timed("table_expand.outer")
def outer(frame, kept, parent):
    indices = pa.array(parent, pa.int64())
    return {name: pc.take(frame.table[name], indices) for name in kept}


@timed("table_expand.metadata")
def metadata(frame, kept):
    return ({name: frame.lineage.get(name, ()) for name in kept},
            {name: frame.outcomes[name] for name in kept if name in frame.outcomes})


@timed("table_expand.field_values")
def field_values(frame, declarations, raw, renamed, parent, locations, inherited, context, converted=None):
    output, kinds, lineage, outcomes = {}, {}, {}, {}
    for field, given in declarations.items():
        name = renamed[field]
        try:
            value = converted[field] if converted is not None and field in converted else typed(raw[field], given, context, name)
        except ColumnValueError as error:
            position = error.row - 1
            raise ColumnValueError(parent[position] + 1, "%s field %s: %s" %
                                   (locations[position], name, error.detail)) from error
        output[name], kinds[name] = value.values, value.kind
        lineage[name] = joined(frame.lineage.get(inherited, ()), ((context.name, name),))
        marks = frame.outcomes.get(inherited, ())
        declared = statement(given)
        if "outcome" in declared:
            marks = joined(marks, ((context.name, declared["outcome"]),))
        if marks:
            outcomes[name] = marks
    return output, kinds, lineage, outcomes


@timed("table_expand.combined")
def combined(parts, field):
    try:
        dtype = pa.unify_schemas([pa.schema([(field, part.type)]) for part in parts]).field(field).type
        return pa.concat_arrays([part if part.type == dtype else part.cast(dtype, safe=True) for part in parts])
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise ValueError("columns.%s has incompatible native record field types: %s" % (field, error)) from error


class Expand:
    version = 1
    fields = frozenset(("column", "count", "into", "columns", "names"))
    required = ()
    input_count = (1, 1)

    @timed("table_expand.Expand.validate")
    def validate(self, parameters):
        if ("column" in parameters) == ("count" in parameters):
            raise ValueError("expand states exactly one of column and count")
        if ("into" in parameters) == ("columns" in parameters):
            raise ValueError("expand states exactly one of into and columns")
        field = "column" if "column" in parameters else "count"
        expression(parameters[field], field)
        if "count" in parameters and "columns" in parameters:
            raise ValueError("count expansion states into, not record columns")
        if "into" in parameters:
            expression(parameters["into"], "into")
        else:
            fields(parameters["columns"])
        defaults = [*parameters.get("columns", {}), "place"] if "column" in parameters else []
        names(defaults, parameters.get("names"))

    @timed("table_expand.Expand.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        counted = "count" in parameters
        source = parameters.get("column")
        kept = [name for name in frame.table.column_names if name != source]
        defaults = [*parameters.get("columns", {}), "place"] if not counted else []
        renamed = generated(frame, defaults, parameters.get("names"))
        fresh(frame, [*renamed.values(), *([parameters["into"]] if "into" in parameters else [])])
        if counted:
            lengths = counts(frame, parameters["count"], context, "count")
            inherited, marks = dependencies(frame, parameters["count"])
        else:
            kind = column(frame, source)
            if kind.name != "list":
                raise ValueError("column %s is a list" % source)
            lengths = pc.list_value_length(frame.table[source]).to_pylist()
            for row, length in enumerate(lengths, 1):
                if length is None:
                    raise ColumnValueError(row, "column %s is a nonmissing list" % source)
            inherited, marks = frame.lineage.get(source, ()), frame.outcomes.get(source, ())
        budget(frame, kept, lengths, context, "expanding rows")
        parent = [row for row, count in enumerate(lengths) for _ in range(count)]
        places = [place for count in lengths for place in range(count)]
        output = outer(frame, kept, parent)
        kinds = {name: column(frame, name) for name in kept}
        lineage, outcomes = metadata(frame, kept)
        if counted:
            name = parameters["into"]
            output[name], kinds[name] = pa.array(places, pa.float64()), Kind("number")
            lineage[name] = joined(inherited, ((context.name, parameters["count"]),))
            if marks:
                outcomes[name] = marks
        else:
            values = pc.list_flatten(frame.table[source])
            if "into" in parameters:
                name = parameters["into"]
                output[name], kinds[name] = values, kind.of
                lineage[name] = joined(inherited, ((context.name, name),))
                if marks:
                    outcomes[name] = marks
            else:
                if kind.of.name != "nested" or not pa.types.is_struct(values.type):
                    raise ValueError("columns expansion reads a list of nested records")
                for position, missing in enumerate(values.is_null().to_pylist()):
                    if missing:
                        raise ColumnValueError(parent[position] + 1, "column %s member %d is a nonmissing record" %
                                               (source, places[position]))
                raw = {field: pc.struct_field(values, field) if field in values.type.names else pa.nulls(len(values))
                       for field in parameters["columns"]}
                added = field_values(frame, parameters["columns"], raw, renamed, parent,
                                     ["member %d" % place for place in places], source, context)
                output.update(added[0])
                kinds.update(added[1])
                lineage.update(added[2])
                outcomes.update(added[3])
            name = renamed["place"]
            output[name], kinds[name] = pa.array(places, pa.float64()), Kind("number")
            lineage[name] = joined(inherited, ((context.name, name),))
            if marks:
                outcomes[name] = marks
        table = pa.table(output)
        context.check(table.nbytes * 4 + frame.nbytes, "expanded table")
        return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage,
                     provenance=provenance(frame, parent, context))


class Nested:
    version = 1
    fields = frozenset(("record", "children", "columns", "depth", "names"))
    required = ("record", "children", "columns", "depth")
    input_count = (1, 1)

    @timed("table_expand.Nested.validate")
    def validate(self, parameters):
        for field in ("record", "children"):
            expression(parameters[field], field)
        fields(parameters["columns"])
        depth = parameters["depth"]
        if depth is not None and (type(depth) is not int or depth < 0):
            raise ValueError("depth is None or a nonnegative whole number")
        names([*parameters["columns"], "path", "parent", "place", "depth"], parameters.get("names"))

    @timed("table_expand.Nested.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        record, children, maximum = parameters["record"], parameters["children"], parameters["depth"]
        if column(frame, record).name != "nested" or not pa.types.is_struct(frame.table[record].type):
            raise ValueError("record %s holds nested records" % record)
        renamed = generated(frame, [*parameters["columns"], "path", "parent", "place", "depth"], parameters.get("names"))
        kept = [name for name in frame.table.column_names if name != record]
        materialized(frame, [record], context, "materializing nested records")
        root_type = frame.table[record].type
        parents, records, paths, ancestors, places, depths = [], [], [], [], [], []
        declared_children = parameters["columns"].get(children)
        converted_children = []
        lengths = [0] * frame.table.num_rows
        row = 0
        for chunk in frame.table[record].chunks:
            for position in range(len(chunk)):
                stack = [(chunk.slice(position, 1), "0", None, 0, 0)]
                while stack:
                    value, path, parent, place, depth = stack.pop()
                    while pa.types.is_dictionary(value.type):
                        value = pc.take(value.dictionary, value.indices)
                    if not pa.types.is_struct(value.type) or value.null_count:
                        raise ColumnValueError(row + 1, "record %s path %s is a nonmissing record" % (record, path))
                    child = value.field(children) if children in value.type.names else pa.nulls(1)
                    if declared_children is not None:
                        try:
                            converted = typed(child, declared_children, context, renamed[children])
                        except ColumnValueError as error:
                            raise ColumnValueError(row + 1, "children %s at path %s: %s" %
                                                   (children, path, error.detail)) from error
                        if converted.kind.name != "list" or converted.kind.of.name != "nested":
                            raise ValueError("columns.%s declares children as a list of nested records" % children)
                        child = converted.values.combine_chunks() if isinstance(converted.values, pa.ChunkedArray) else converted.values
                        converted_children.append(child)
                    child_type = child.type
                    if child.null_count or not (pa.types.is_list(child_type) or pa.types.is_large_list(child_type)
                                                or pa.types.is_fixed_size_list(child_type)):
                        raise ColumnValueError(row + 1, "children %s at path %s is an explicit nonmissing list; leaves state []" %
                                               (children, path))
                    parents.append(row)
                    records.append(value)
                    paths.append(path)
                    ancestors.append(parent)
                    places.append(place)
                    depths.append(depth)
                    lengths[row] += 1
                    if maximum is None or depth < maximum:
                        members = child[0].values
                        for index in range(len(members) - 1, -1, -1):
                            stack.append((members.slice(index, 1), path + "." + str(index), path, index, depth + 1))
                row += 1
        budget(frame, kept, lengths, context, "expanding nested records")
        raw = {}
        for field in parameters["columns"]:
            if field == children and declared_children is not None:
                continue
            parts = [value.field(field) if field in value.type.names else pa.nulls(1) for value in records]
            if parts:
                raw[field] = combined(parts, field)
            else:
                dtype = root_type.field(field).type if field in root_type.names else pa.null()
                raw[field] = pa.array([], type=dtype)
        converted = {}
        if declared_children is not None:
            if converted_children:
                values = combined(converted_children, children)
                converted[children] = Value(values, column_kind(declared_children, values.type, context.unit))
            else:
                dtype = root_type.field(children).type if children in root_type.names else pa.null()
                converted[children] = typed(pa.array([], dtype), declared_children, context, renamed[children])
        output = outer(frame, kept, parents)
        kinds = {name: column(frame, name) for name in kept}
        lineage, outcomes = metadata(frame, kept)
        added = field_values(frame, parameters["columns"], raw, renamed, parents,
                             ["path %s" % path for path in paths], record, context, converted)
        output.update(added[0])
        kinds.update(added[1])
        lineage.update(added[2])
        outcomes.update(added[3])
        for field, values, kind, dtype in (
            ("path", paths, Kind("identifier"), pa.string()),
            ("parent", ancestors, Kind("identifier"), pa.string()),
            ("place", places, Kind("number"), pa.float64()),
            ("depth", depths, Kind("number"), pa.float64()),
        ):
            name = renamed[field]
            output[name], kinds[name] = pa.array(values, dtype), kind
            lineage[name] = joined(frame.lineage.get(record, ()), ((context.name, name),))
            if record in frame.outcomes:
                outcomes[name] = frame.outcomes[record]
        table = pa.table(output)
        context.check(frame.nbytes + table.nbytes * 4, "nested table")
        return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage,
                     provenance=provenance(frame, parents, context))


class Cut:
    version = 1
    fields = frozenset(("count", "into", "names"))
    required = ("count", "into")
    input_count = (2, 2)

    @timed("table_expand.Cut.validate")
    def validate(self, parameters):
        expression(parameters["count"], "count")
        expression(parameters["into"], "into")
        names(["place"], parameters.get("names"))

    @timed("table_expand.Cut.apply")
    def apply(self, frames, parameters, context):
        frame, sizes = frames
        renamed = generated(frame, ["place"], parameters.get("names"))
        into, place = parameters["into"], renamed["place"]
        fresh(frame, (into, place))
        lengths = counts(sizes, parameters["count"], context, "count")
        if sum(lengths) != frame.table.num_rows:
            raise ValueError("cut count lengths sum to %d, but the table has %d rows" % (sum(lengths), frame.table.num_rows))
        context.check(frame.nbytes + sizes.nbytes + frame.table.num_rows * 160, "cutting flat rows")
        owners = [row for row, count in enumerate(lengths) for _ in range(count)]
        places = [position for count in lengths for position in range(count)]
        table = frame.table.append_column(into, pa.array(owners, pa.float64())).append_column(place, pa.array(places, pa.float64()))
        kinds = dict(frame.kinds, **{into: Kind("number"), place: Kind("number")})
        lineage, outcomes = metadata(frame, frame.table.column_names)
        inherited, marks = dependencies(sizes, parameters["count"])
        for name in (into, place):
            lineage[name] = joined(inherited, ((context.name, name),))
            if marks:
                outcomes[name] = marks
        origins = combine_provenance(frames, [[[row] for row in range(frame.table.num_rows)], [[row] for row in owners]])
        return carry(frame, table, kinds, outcomes=outcomes, lineage=lineage, provenance=origins)


TABLES.register("expand", Expand())
TABLES.register("nested", Nested())
TABLES.register("cut", Cut())
