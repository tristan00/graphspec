import copy
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.schema.fields import fields, named
from graphspec.schema.references import SORTS
from graphspec.schema.schema import VALUE, Schema
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import ALWAYS, NEVER, UNITED

from . import records
from .readers import Source, SourceSession, declarations, namespace, typed

LOG = logging.getLogger(__name__)
UNSTATED = object()


@timed("builder.key")
def keyed(value, where):
    if type(value) is not str or not value:
        raise ValueError("%s needs a nonempty string key, found %r" % (where, value))
    return value


@timed("builder.supplied")
def supplied(values, columns, where):
    if values is None:
        return columns
    if not isinstance(values, Mapping):
        raise ValueError("%s states values as a mapping from column name to value" % where)
    repeated = sorted(set(values) & set(columns))
    if repeated:
        raise ValueError("%s states the columns twice: %s" % (where, ", ".join(repeated)))
    return dict(values, **columns)


class Builder:
    @timed("builder.init")
    def __init__(self, script, sources):
        if isinstance(script, (str, Path)):
            script = declarations(namespace(script, sources))
        if not isinstance(script, Mapping):
            raise ValueError("a builder takes a script path or a mapping from names to declarations")
        self.declarations = dict(script)
        self.sources = Path(sources).resolve()
        self.stated, self.graphs, self.changes = None, {}, {}

    @timed("builder.read")
    def read(self, declaration, plan):
        name = declaration if isinstance(declaration, str) else None
        if name is not None:
            if name not in self.declarations:
                raise ValueError("the builder has no source named %s" % name)
            declaration = self.declarations[name]
        if not isinstance(declaration, Source):
            raise ValueError("read names a source declaration")
        held = dict(self.declarations)
        if not any(value is declaration for value in held.values()):
            held[str(declaration.place)] = declaration
        return typed(declaration, self.sources, plan, name, held)

    @timed("builder.schema")
    def schema(self, mapping):
        if self.stated is not None:
            raise ValueError("the builder schema is already stated")
        self.stated = Schema(copy.deepcopy(mapping))
        return self

    @timed("builder.required")
    def required(self):
        if self.stated is None:
            raise ValueError("declare the builder schema before stating a graph or fact")
        return self.stated

    @timed("builder.values")
    def values(self, values, columns, where, partial):
        extra = sorted(set(values) - set(columns))
        missing = sorted(set(columns) - set(values))
        if extra or (missing and not partial):
            raise ValueError("%s states invalid columns; unknown %s, missing %s" % (where, extra, missing))
        result = copy.deepcopy(values)
        for column, value in result.items():
            try:
                pa.array([value], self.required().kind(column).written)
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, OverflowError) as error:
                raise ValueError("%s has an invalid value for column %s: %r" % (where, column, value)) from error
        return result

    @timed("builder.graph")
    def graph(self, name, unit, *, values=None, **columns):
        schema = self.required()
        named(name, "a graph")
        if name in self.graphs:
            raise ValueError("the graph %s is already declared" % name)
        if (unit is not None) != (schema.model in UNITED) or (unit is not None and unit not in schema.units):
            raise ValueError("the graph %s states one of its schema's units, or None without a clock" % name)
        columns = self.values(supplied(values, columns, "the graph " + name), schema.graph, "the graph " + name, False)
        tasks = {field: [] for field in records.ASKED if field != "references"}
        tasks["references"] = {reference: [] for reference in schema.tasks["references"]}
        tasks.update({VALUE + column: [] for column in schema.asked_columns()})
        if schema.model != "none":
            tasks["time"] = []
        if schema.observers:
            tasks["observer"] = []
        if schema.tasks["grouping"] is not None:
            tasks["grouping"] = []
        if schema.tasks["signals"]:
            tasks.update(signal=[], signal_observed=[])
        graph = dict(name=name, unit=unit, columns=columns, nodes={kind: [] for kind in schema.entities()},
                     values=[], edges=[], tasks=tasks)
        if schema.sets:
            graph["sets"] = {kind: dict(set=[], member=[], role=[], **{column: [] for column in entry["members"]})
                             for kind, entry in schema.sets.items()}
        if schema.holds:
            raise ValueError("the base graph declaration needs held graph contents for %s" % ", ".join(schema.holds))
        self.graphs[name] = graph
        return self

    @timed("builder.chosen")
    def chosen(self, graph):
        self.required()
        if graph is None:
            if len(self.graphs) != 1:
                raise ValueError("name the graph of the fact, or declare exactly one graph first")
            return next(iter(self.graphs.values()))
        if graph not in self.graphs:
            raise ValueError("the graph %r has not been declared" % graph)
        return self.graphs[graph]

    @timed("builder.when")
    def when(self, form, since, until, known, replaced, seen, seq, where):
        schema = self.required()
        if schema.model == "none":
            if form not in (None, "state") or since is not UNSTATED or until is not None or known not in (UNSTATED, None) or replaced is not None:
                raise ValueError("%s states no times because its schema has no time" % where)
            result = {}
        else:
            if since is UNSTATED:
                raise ValueError("%s lacks its time" % where)
            if since is not None and (type(since) is not int or not ALWAYS < since < NEVER):
                raise ValueError("%s states a whole time in the supported range" % where)
            if form in ("event", "change") and since is None:
                raise ValueError("%s states the time of its %s" % (where, form))
            if until is not None and (type(until) is not int or not ALWAYS < until < NEVER or (since is not None and until <= since)):
                raise ValueError("%s states an end later than its beginning" % where)
            result = dict(since=since, until=until) if form == "state" else dict(at=since)
            if known is UNSTATED:
                raise ValueError("%s states known, or None for no separate time of knowledge" % where)
            if schema.model == "clocks":
                for field, value in (("known", known), ("replaced", replaced)):
                    if value is not None and (type(value) is not int or not ALWAYS < value < NEVER):
                        raise ValueError("%s states %s as a whole time or None" % (where, field))
                learned = known if known is not None else since if since is not None else ALWAYS
                if replaced is not None and replaced <= learned:
                    raise ValueError("%s is replaced after it is known" % where)
                result.update(known=known, replaced=replaced)
            elif known is not None or replaced is not None:
                raise ValueError("%s needs the clocks time model to state known or replaced" % where)
        if schema.observers:
            if seen is not None and (not isinstance(seen, (list, tuple)) or any(type(name) is not str or not name for name in seen)
                                     or len(set(seen)) != len(seen)):
                raise ValueError("%s states seen as a list of observer names or None" % where)
            result["observers"] = None if seen is None else sorted(seen)
        elif seen is not None:
            raise ValueError("%s names observers its schema does not declare" % where)
        if "seq" in schema.stamps():
            if type(seq) is not int or not 0 <= seq < 1 << 31:
                raise ValueError("%s needs seq, a nonnegative int32, for its tie rule" % where)
            result["seq"] = seq
        elif seq is not UNSTATED:
            raise ValueError("%s states seq without a sequence tie rule" % where)
        return result

    @timed("builder.node")
    def node(self, kind, key, since=UNSTATED, until=None, *, graph=None, known=UNSTATED,
             replaced=None, seen=None, seq=UNSTATED, values=None, **columns):
        schema, held = self.required(), self.chosen(graph)
        if kind not in schema.nodes:
            raise ValueError("node names an undeclared kind of node %r" % kind)
        keyed(key, "node")
        where = "node %s:%s of graph %s" % (kind, key, held["name"])
        values = self.values(supplied(values, columns, where), schema.nodes[kind], where, True)
        when = self.when("state", since, until, known, replaced, seen, seq, where)
        if key not in held["nodes"][kind]:
            held["nodes"][kind].append(key)
        rows = {name: [value] for name, value in dict(key=key, **when, **{VALUE + name: value for name, value in values.items()}).items()}
        held["values"].append(dict(kind=kind, form=None if schema.model == "none" else "state", parameters={}, rows=rows))
        return self

    @timed("builder.end")
    def end(self, value, graph, where):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("%s states each end as (kind, key)" % where)
        kind, key = value
        if kind not in graph["nodes"] or key not in graph["nodes"][kind]:
            raise ValueError("%s names the undeclared node %r" % (where, value))
        return kind, key

    @timed("builder.edge_fact")
    def edge_fact(self, relation, source, target, form, since, until, graph, known, replaced, seen, seq, present, values):
        schema, held = self.required(), self.chosen(graph)
        table, label = relation if isinstance(relation, tuple) and len(relation) == 2 else (relation, None)
        if type(table) is not str or table not in schema.edges:
            raise ValueError("edge names an undeclared relation %r" % (relation,))
        where, entry = "edge %s of graph %s" % (table, held["name"]), schema.edges[table]
        left, right = self.end(source, held, where), self.end(target, held, where)
        if (left[0], right[0]) not in entry["endpoints"]:
            if entry["directed"] or (right[0], left[0]) not in entry["endpoints"]:
                raise ValueError("%s does not join %s to %s" % (where, left[0], right[0]))
            left, right = right, left
        if type(present) is not bool:
            raise ValueError("%s states present as true or false" % where)
        if (label is None) != (entry["column"] is None):
            raise ValueError("%s names (table, relation) exactly when the table takes relations from a column" % where)
        values = self.values(values, entry["columns"], where, False)
        when = self.when(form, since, until, known, replaced, seen, seq, where)
        rows = dict(source=left[1], target=right[1], present=present, **when,
                    **{VALUE + name: value for name, value in values.items()})
        if label is not None:
            rows["relation"] = keyed(label, where)
        held["edges"].append(dict(table=table, ends=(left[0], right[0]),
                                  form=None if schema.model == "none" else form, parameters={},
                                  rows={name: [value] for name, value in rows.items()}))
        return self

    @timed("builder.edge")
    def edge(self, relation, source, target, since=UNSTATED, until=None, *, graph=None, known=UNSTATED,
             replaced=None, seen=None, seq=UNSTATED, present=True, values=None, **columns):
        return self.edge_fact(relation, source, target, "state", since, until, graph, known, replaced, seen, seq,
                              present, supplied(values, columns, "edge %r" % (relation,)))

    @timed("builder.event")
    def event(self, relation, source, target, at, *, graph=None, known=UNSTATED,
              replaced=None, seen=None, seq=UNSTATED, present=True, values=None, **columns):
        return self.edge_fact(relation, source, target, "event", at, None, graph, known, replaced, seen, seq,
                              present, supplied(values, columns, "event %r" % (relation,)))

    @timed("builder.change")
    def change(self, kind, key, at, *, graph=None, known=UNSTATED, replaced=None, seen=None, seq=UNSTATED,
               values=None, **columns):
        schema, held = self.required(), self.chosen(graph)
        if kind not in schema.nodes:
            raise ValueError("change names an undeclared kind of node %r" % kind)
        keyed(key, "change")
        where = "change of %s:%s in graph %s" % (kind, key, held["name"])
        values = self.values(supplied(values, columns, where), schema.nodes[kind], where, True)
        when = self.when("change", at, None, known, replaced, seen, seq, where)
        if key not in held["nodes"][kind]:
            held["nodes"][kind].append(key)
        for column in values or (None,):
            observer_scope = None if seen is None else tuple(sorted(seen))
            identity = (held["name"], kind, column, observer_scope)
            row = dict(key=key, **when, **({VALUE + column: values[column]} if column is not None else {}))
            if identity not in self.changes:
                statement = dict(kind=kind, form="change", parameters=dict(keys=["key"]), rows={name: [] for name in row})
                self.changes[identity] = statement
                held["values"].append(statement)
            for name, value in row.items():
                self.changes[identity]["rows"][name].append(value)
        return self

    @timed("builder.task")
    def task(self, kind, key, at=UNSTATED, references=UNSTATED, split=UNSTATED, target=UNSTATED,
             known=UNSTATED, *, graph=None, weight=None, observer=None, grouping=UNSTATED,
             signal=UNSTATED, signal_observed=UNSTATED, values=None, **columns):
        schema, held = self.required(), self.chosen(graph)
        if kind not in schema.tasks["kinds"]:
            raise ValueError("task names an undeclared kind %r" % kind)
        where, entry = "task %s of kind %s in graph %s" % (key, kind, held["name"]), schema.tasks["kinds"][kind]
        if type(key) is not int:
            raise ValueError("%s needs a distinct integer key" % where)
        if any(value is UNSTATED for value in (references, split, target, known)):
            raise ValueError("%s states references, split, target and known" % where)
        fields(references, entry["references"], where + " references")
        if schema.model == "none":
            if at is not UNSTATED or known is not None:
                raise ValueError("%s states no time or time of knowledge" % where)
        elif type(at) is not int or not ALWAYS < at < NEVER:
            raise ValueError("%s needs a whole time in the supported range" % where)
        if known is not None and (type(known) is not int or not at < known < NEVER):
            raise ValueError("%s needs a time of knowledge later than its time" % where)
        if split is not None:
            named(split, where + " split")
        if target is not None:
            if entry["target"] is None:
                raise ValueError("%s states a target its kind does not declare" % where)
            TARGETS.named(entry["target"]["kind"]).staged(target, entry["target"], where)
        if (weight is not None) != entry["weight"]:
            raise ValueError("%s states weight exactly when its kind requires it" % where)
        if not schema.observers and observer is not None:
            raise ValueError("%s names an observer its schema does not declare" % where)
        values = self.values(supplied(values, columns, where), entry["columns"], where, False)
        row = dict(key=key, kind=kind, split=split, target=copy.deepcopy(target), known=known, weight=weight)
        row.update({VALUE + column: values[column] if column in values else None for column in schema.asked_columns()})
        if schema.model != "none":
            row["time"] = at
        if schema.observers:
            row["observer"] = observer
        if schema.tasks["grouping"] is not None:
            if type(grouping) is not int:
                raise ValueError("%s needs its integer grouping" % where)
            row["grouping"] = grouping
        elif grouping is not UNSTATED:
            raise ValueError("%s states a grouping its schema does not declare" % where)
        if schema.tasks["signals"]:
            if any(not isinstance(value, (list, tuple)) or len(value) != schema.tasks["signals"]
                   for value in (signal, signal_observed)):
                raise ValueError("%s needs signal and signal_observed of its declared width" % where)
            row.update(signal=list(signal), signal_observed=list(signal_observed))
        elif signal is not UNSTATED or signal_observed is not UNSTATED:
            raise ValueError("%s states a signal its schema does not declare" % where)
        for reference, stated in references.items():
            if SORTS.named(schema.tasks["references"][reference]["sort"]).whole:
                if stated is not None:
                    raise ValueError("%s names the whole graph without a node under %s" % (where, reference))
            elif stated is None:
                raise ValueError("%s lacks its reference %s" % (where, reference))
        if key in held["tasks"]["key"]:
            position = held["tasks"]["key"].index(key)
            previous = held["tasks"]
            if (previous["target"][position] is not None and kind not in schema.decided()) or target is None:
                raise ValueError("%s needs a distinct integer key unless completing a missing target" % where)
            if any(previous[name][position] != value for name, value in row.items() if name not in ("target", "known")) or any(
                previous["references"][name][position] != references.get(name) for name in previous["references"]
            ):
                raise ValueError("%s cannot change an existing task while completing its target" % where)
            if schema.model != "none":
                from .appending import target_delta

                target_delta(schema, dict(target=previous["target"][position], known=previous["known"][position]), row, at)
            previous["target"][position], previous["known"][position] = row["target"], known
            return self
        for name, value in row.items():
            held["tasks"][name].append(value)
        for reference in held["tasks"]["references"]:
            held["tasks"]["references"][reference].append(copy.deepcopy(references[reference]) if reference in references else None)
        return self

    @timed("builder.records")
    def records(self, *, snapshot=True):
        schema = self.required()
        if not self.graphs:
            raise ValueError("the builder holds no declared graph")
        pending = [name for name, declaration in self.declarations.items() if not isinstance(declaration, Source)]
        if pending:
            raise ValueError("the declarations need handlers before they can be built: %s" % ", ".join(pending))
        if not snapshot:
            return dict(schema=schema.mapping(), graphs=list(self.graphs.values()))
        graphs = copy.deepcopy(self.graphs)
        for graph in graphs.values():
            tasks = graph["tasks"]
            order = sorted(range(len(tasks["key"])), key=lambda at: (tasks["time"][at], tasks["key"][at])) if schema.model != "none" else list(range(len(tasks["key"])))
            for name in tasks:
                if name != "references":
                    tasks[name] = [tasks[name][at] for at in order]
            tasks["references"] = {name: [values[at] for at in order] for name, values in tasks["references"].items()}
        return dict(schema=schema.mapping(), graphs=list(graphs.values()))

    @timed("builder.stage")
    def stage(self, output, name, plan, *, reuse=False):
        from .writing import stage

        started = time.perf_counter()
        self.check_sources(plan)
        result = stage(self.records(snapshot=False), output, name, plan, reuse=reuse)
        LOG.info("builder staged graphs=%d seconds=%.4f", len(self.graphs), time.perf_counter() - started)
        return result

    @timed("builder.append")
    def append(self, task_path, plan):
        from .appending import append_stage

        self.check_sources(plan)
        return append_stage(self, task_path, plan)

    @timed("builder.check_sources")
    def check_sources(self, plan):
        sources = {name: declaration for name, declaration in self.declarations.items() if isinstance(declaration, Source)}
        if not sources:
            return
        with SourceSession(sources, self.sources, plan) as session:
            for name, declaration in sources.items():
                for _ in typed(declaration, self.sources, plan, name, sources, session):
                    pass
