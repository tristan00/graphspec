import copy
from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow as pa

from graphspec.infra.timing import timed

from . import table_expand as table_expand
from . import table_relations as table_relations
from . import table_reshape as table_reshape
from .expression_types import Kind, common, datatype
from .readers import Source
from .table_types import TABLES, broadcast, carry, combine_provenance, dependencies, fresh, take


@timed("tables.expression")
def expression(value, field):
    if type(value) is not str or not value:
        raise ValueError("%s states a nonempty expression" % field)


@timed("tables.mapping")
def mapping(value, field):
    if not isinstance(value, Mapping) or any(type(name) is not str or not name for name in value):
        raise ValueError("%s states a mapping with nonempty column names" % field)


@timed("tables.united_kind")
def united_kind(left, right):
    if left == right:
        return left
    if left.name == "missing":
        return right
    if right.name == "missing":
        return left
    if left.name != right.name:
        raise ValueError("union columns have incompatible semantic types %s and %s" % (left.name, right.name))
    if left.name == "list":
        return Kind("list", of=united_kind(left.of, right.of))
    if left.unit == right.unit and left.width == right.width and datatype(left) == datatype(right):
        return left
    return common(left, right)


@dataclass(frozen=True, init=False, eq=False)
class Table:
    operation: str
    inputs: tuple
    _parameters: dict
    _columns: object

    @timed("tables.Table")
    def __init__(self, operation, inputs, parameters, columns=None):
        entry = TABLES.named(operation)
        if not isinstance(inputs, (tuple, list)) or not inputs or any(not isinstance(value, (Source, Table)) for value in inputs):
            raise ValueError("%s reads a nonempty list of source or table declarations" % operation)
        if hasattr(entry, "apply"):
            if columns is not None:
                raise ValueError("a shipped table operation infers its output columns")
            allowed, required = set(entry.fields), set(entry.required)
            if set(parameters) - allowed:
                raise ValueError("%s has unknown fields %s" % (operation, ", ".join(sorted(set(parameters) - allowed))))
            if required - set(parameters):
                raise ValueError("%s requires fields %s" % (operation, ", ".join(sorted(required - set(parameters)))))
            least, most = entry.input_count
            if len(inputs) < least or most is not None and len(inputs) > most:
                raise ValueError("%s has the wrong number of input tables" % operation)
            entry.validate(parameters)
        else:
            if not callable(entry):
                raise ValueError("a custom table entry is callable")
            mapping(columns, "custom table columns")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "_parameters", copy.deepcopy(dict(parameters)))
        object.__setattr__(self, "_columns", copy.deepcopy(columns))

    @property
    @timed("tables.Table.parameters")
    def parameters(self):
        return copy.deepcopy(self._parameters)

    @property
    @timed("tables.Table.columns")
    def columns(self):
        return copy.deepcopy(self._columns)


@timed("tables.table")
def table(tables, name, columns, **parameters):
    return Table(name, tables, parameters, columns)


@timed("tables.select")
def select(table, where):
    return Table("select", [table], dict(where=where))


@timed("tables.compute")
def compute(table, columns, outcomes=None):
    parameters = dict(columns=columns)
    if outcomes is not None:
        parameters["outcomes"] = outcomes
    return Table("compute", [table], parameters)


@timed("tables.align")
def align(tables):
    return Table("align", tables, {})


@timed("tables.union")
def union(tables):
    return Table("union", tables, {})


@timed("tables.join")
def join(table, other, on, take, unmatched, where=None, window=None, outcomes=None):
    parameters = dict(on=on, take=take, unmatched=unmatched)
    parameters.update({name: value for name, value in dict(where=where, window=window, outcomes=outcomes).items() if value is not None})
    return Table("join", [table, other], parameters)


@timed("tables.group")
def group(table, by, columns):
    return Table("group", [table], dict(by=by, columns=columns))


@timed("tables.pivot")
def pivot(table, by, names, value):
    return Table("pivot", [table], dict(by=by, names=names, value=value))


@timed("tables.unpivot")
def unpivot(table, keep, name, value):
    return Table("unpivot", [table], dict(keep=keep, name=name, value=value))


@timed("tables.expand")
def expand(table, column=None, count=None, into=None, columns=None, names=None):
    parameters = {name: value for name, value in dict(column=column, count=count, into=into, columns=columns, names=names).items()
                  if value is not None}
    return Table("expand", [table], parameters)


@timed("tables.nested")
def nested(table, record, children, columns, depth, names=None):
    parameters = dict(record=record, children=children, columns=columns, depth=depth)
    if names is not None:
        parameters["names"] = names
    return Table("nested", [table], parameters)


@timed("tables.cut")
def cut(table, counts, count, into, names=None):
    parameters = dict(count=count, into=into)
    if names is not None:
        parameters["names"] = names
    return Table("cut", [table, counts], parameters)


@timed("tables.order")
def order(table, within, by, gap, columns=None, names=None):
    parameters = dict(within=within, by=by, gap=gap)
    parameters.update({name: value for name, value in dict(columns=columns, names=names).items() if value is not None})
    return Table("order", [table], parameters)


@timed("tables.closure")
def closure(table, source, target, most, names=None):
    parameters = dict(source=source, target=target, most=most)
    if names is not None:
        parameters["names"] = names
    return Table("closure", [table], parameters)


class Select:
    version = 1
    fields = frozenset(("where",))
    required = fields
    input_count = (1, 1)

    @timed("tables.Select.validate")
    def validate(self, parameters):
        expression(parameters["where"], "select.where")

    @timed("tables.Select.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        predicate = context.evaluate(frame, parameters["where"], "where")
        if predicate.type not in ("flag", "missing"):
            raise ValueError("select.where gives a flag")
        mask = broadcast(predicate, frame.table.num_rows).to_pylist()
        selected = [index for index, value in enumerate(mask) if value is True]
        return take(frame, selected)


class Compute:
    version = 1
    fields = frozenset(("columns", "outcomes"))
    required = frozenset(("columns",))
    input_count = (1, 1)

    @timed("tables.Compute.validate")
    def validate(self, parameters):
        mapping(parameters["columns"], "compute.columns")
        for name, text in parameters["columns"].items():
            expression(text, "compute.columns." + name)
        marks = parameters.get("outcomes", {})
        mapping(marks, "compute.outcomes")
        if set(marks) - set(parameters["columns"]):
            raise ValueError("compute.outcomes names only newly computed columns")
        for name, text in marks.items():
            expression(text, "compute.outcomes." + name)

    @timed("tables.Compute.apply")
    def apply(self, frames, parameters, context):
        frame = frames[0]
        fresh(frame, parameters["columns"])
        values, kinds = list(frame.table.columns), dict(frame.kinds)
        outcomes, lineage = dict(frame.outcomes), dict(frame.lineage)
        held, added = context.retained, 0
        try:
            for name, text in parameters["columns"].items():
                context.retained = held + added
                value = context.evaluate(frame, text, "columns." + name)
                column = broadcast(value, frame.table.num_rows)
                context.check(column.nbytes * 4, "computed column " + name)
                values.append(column)
                added += column.nbytes * 4
                kinds[name] = value.kind
                origins, marks = dependencies(frame, text)
                lineage[name] = tuple(dict.fromkeys((*origins, (context.name, text))))
                if name in parameters.get("outcomes", {}):
                    marks = tuple(dict.fromkeys((*marks, (context.name, parameters["outcomes"][name]))))
                if marks:
                    outcomes[name] = marks
        finally:
            context.retained = held
        result = pa.Table.from_arrays(values, names=list(kinds))
        context.check(result.nbytes * 4, "computed table")
        return carry(frame, result, kinds, outcomes=outcomes, lineage=lineage)


class Align:
    version = 1
    fields = frozenset()
    required = fields
    input_count = (1, None)

    @timed("tables.Align.validate")
    def validate(self, parameters):
        return None

    @timed("tables.Align.apply")
    def apply(self, frames, parameters, context):
        count = frames[0].table.num_rows
        if any(frame.table.num_rows != count for frame in frames):
            raise ValueError("align requires equal input row counts")
        names, arrays, kinds, outcomes, lineage = [], [], {}, {}, {}
        for frame in frames:
            if set(names) & set(frame.kinds):
                raise ValueError("align input column names collide: " + ", ".join(sorted(set(names) & set(frame.kinds))))
            names.extend(frame.table.column_names)
            arrays.extend(frame.table.columns)
            kinds.update(frame.kinds)
            outcomes.update(frame.outcomes)
            lineage.update(frame.lineage)
        context.check(sum(frame.nbytes for frame in frames) * 4 + len(frames) * count * 64, "align result")
        groups = [[[row] for row in range(count)] for _ in frames]
        return carry(frames[0], pa.Table.from_arrays(arrays, names=names), kinds, outcomes=outcomes, lineage=lineage,
                     provenance=combine_provenance(frames, groups))


class Union:
    version = 1
    fields = frozenset()
    required = fields
    input_count = (1, None)

    @timed("tables.Union.validate")
    def validate(self, parameters):
        return None

    @timed("tables.Union.apply")
    def apply(self, frames, parameters, context):
        names, kinds = frames[0].table.column_names, dict(frames[0].kinds)
        if any(set(frame.kinds) != set(names) for frame in frames):
            raise ValueError("union requires the same columns in every input")
        for frame in frames[1:]:
            for name in names:
                try:
                    kinds[name] = united_kind(kinds[name], frame.kinds[name])
                except ValueError as error:
                    raise ValueError("union column %s: %s" % (name, error)) from error
        count = sum(frame.table.num_rows for frame in frames)
        context.check(sum(frame.nbytes for frame in frames) * 4 + len(frames) * count * 64, "union result")
        tables, groups, start = [], [], 0
        outcomes, lineage = {}, {}
        for frame in frames:
            tables.append(pa.Table.from_arrays([frame.table[name].cast(datatype(kinds[name]), safe=True) for name in names], names=names))
            groups.append([[] for _ in range(start)] + [[row] for row in range(frame.table.num_rows)]
                          + [[] for _ in range(count - start - frame.table.num_rows)])
            start += frame.table.num_rows
            for name in names:
                outcomes[name] = tuple(dict.fromkeys((*outcomes.get(name, ()), *frame.outcomes.get(name, ()))))
                lineage[name] = tuple(dict.fromkeys((*lineage.get(name, ()), *frame.lineage.get(name, ()))))
        return carry(frames[0], pa.concat_tables(tables), kinds, outcomes={name: marks for name, marks in outcomes.items() if marks},
                     lineage=lineage, provenance=combine_provenance(frames, groups))


TABLES.register("select", Select())
TABLES.register("compute", Compute())
TABLES.register("align", Align())
TABLES.register("union", Union())
