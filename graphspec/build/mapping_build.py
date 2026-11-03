import logging
import time
from collections.abc import Mapping as Held

import pyarrow as pa

from graphspec.infra.timing import timed
from graphspec.schema.fields import named
from graphspec.schema.schema import DECLARED, VERSION
from graphspec.schema.time import UNITED

from .database import Part, owner
from .database_build import expanded
from .expression_types import UNSTATED
from .expressions import custom_argument
from .mapping_rows import Reader, following, joined, when
from .mappings import MAPPINGS, Linked, Mapping, tabled
from .presence import PRESENCE, ruled
from .readers import Source
from .table_session import Session
from .table_types import Context, working
from .tables import Table

LOG = logging.getLogger(__name__)
PHASES = ("nodes", "created", "custom", "edges", "tasks")
FACTS = ("nodes", "edges", "sets")


@timed("mapping_build.facts")
def facts(reader, fields, mapped):
    stated = when(reader, fields, mapped.timed, mapped.clocks)
    graphs = reader.texts(fields["graph"], "graph") if "graph" in fields else [None] * reader.count
    for graph in graphs:
        if graph is not None:
            named(graph, "declaration %s field graph" % reader.name)
    return [dict({name: stated[name][place] for name in stated}, graph=graphs[place], row=reader.row(place))
            for place in range(reader.count)]


@timed("mapping_build.inputs")
def inputs(statement):
    found = [statement.table] if tabled(statement.table) else []
    fields = statement.fields
    found += [member["rows"] for member in fields.get("members") or ()]
    found += [fields[name]["rows"] for name in ("options", "actions") if name in fields]
    target = fields.get("target") or {}
    found += [stated["rows"] for stated in (fields.get("earlier"), target.get("later"))
              if isinstance(stated, Held) and "rows" in stated]
    return found


class End:
    @timed("mapping_build.End")
    def __init__(self, reader, stated, where, missing="stop", creatable=False):
        self.where, self.creatable = "declaration %s field %s" % (reader.name, where), creatable
        self.targets = list(stated["of"]) if isinstance(stated["of"], (list, tuple)) else [stated["of"]]
        self.keys = reader.texts(stated["key"], where + ".key", missing=missing)
        self.chosen = reader.texts(stated["kind"], where + ".kind") if "kind" in stated else None
        self.rule, self.parameters = ruled(stated["absent"], self.where + ".absent")
        self.rows = [reader.row(place) for place in range(reader.count)]

    @timed("mapping_build.End.kinds")
    def kinds(self, mapped, place):
        targets = self.targets
        if self.chosen is not None:
            targets = [target for target in targets if mapped.label(target) == self.chosen[place]]
            if not targets:
                raise ValueError("%s row %d names under kind %r none of the mappings under of"
                                 % (self.where, self.rows[place], self.chosen[place]))
        return [kind for target in targets for kind in mapped.kinds(target, self.where)]

    @timed("mapping_build.End.resolve")
    def resolve(self, mapped, place, graph, fact):
        key = self.keys[place]
        if key is None:
            return None
        kinds = self.kinds(mapped, place)
        found = mapped.find(graph, kinds, key)
        if len(found) > 1:
            raise ValueError("%s row %d names the key %s, which the kinds %s all hold"
                             % (self.where, self.rows[place], joined(key), ", ".join(found)))
        if found:
            mapped.note(graph, found[0], joined(key))
            return found[0], joined(key)
        entry = PRESENCE.named(self.rule)
        try:
            (stand,) = entry([joined(key)], [fact["since"]], self.parameters)
        except ValueError as error:
            raise ValueError("%s row %d key %s: %s" % (self.where, self.rows[place], joined(key), error)) from error
        mapped.count(self.where, self.rule)
        if stand is None:
            return None
        if entry.creates:
            if not self.creatable or len(kinds) != 1 or graph is None:
                raise ValueError("%s states the rule %s, which makes a node only for an end of edges or a member of "
                                 "sets naming one kind in one graph" % (self.where, self.rule))
            mapped.create(graph, kinds[0], stand, fact, self.targets[0])
            return kinds[0], stand
        found = mapped.find(graph, kinds, stand)
        if len(found) != 1:
            raise ValueError("%s row %d: the rule %s gives the key %s, which names no one node"
                             % (self.where, self.rows[place], self.rule, stand))
        mapped.note(graph, found[0], stand)
        return found[0], stand


class Stated:
    graphs = ()
    owners = ()

    @timed("mapping_build.Stated.made")
    def made(self, mapped):
        return None

    @timed("mapping_build.Stated.resolve")
    def resolve(self, mapped):
        return None

    @timed("mapping_build.Stated.schema")
    def schema(self, mapped):
        return None

    @timed("mapping_build.Stated.issue")
    def issue(self, mapped, phase):
        return None

    @timed("mapping_build.Stated.kinds")
    def kinds(self, where):
        raise ValueError("%s names under of a mapping that makes no node" % where)


class NodeStated(Stated):
    @timed("mapping_build.NodeStated")
    def __init__(self, mapped, label, statement, session):
        fields = self.fields = statement.fields
        self.label = label
        for name in ("held", "inside"):
            if name in fields:
                raise ValueError("declaration %s states %s; the base layer states no held graphs" % (label, name))
        reader = Reader(session, statement.table, label, mapped.plan).where(fields.get("where"))
        stated = facts(reader, fields, mapped)
        self.columns, values = reader.features(fields.get("features"))
        self.graphs = [fact["graph"] for fact in stated]
        if "distinct" in fields:
            found, rows = set(), []
            for column in (reader.texts(text, "distinct", missing="omit") for text in fields["distinct"]):
                for place, key in enumerate(column):
                    held = dict(stated[place], row=None, seq=0)
                    identity = (key, repr(sorted(held.items())))
                    if key is not None and identity not in found:
                        found.add(identity)
                        rows.append((label, key, held, {}))
            self.rows = rows
        else:
            keys = reader.texts(fields["key"], "key")
            if "of" in fields:
                kinds = [mapped.label(fields["of"])] * reader.count
                if "kind" in fields["of"].fields:
                    raise ValueError("declaration %s names under of a nodes statement that makes several kinds"
                                     % label)
            elif "kind" in fields:
                kinds = reader.texts(fields["kind"], "kind")
            else:
                kinds = [label] * reader.count
            self.rows = [(kinds[place], keys[place], stated[place], values[place]) for place in range(reader.count)]
        self.made_kinds = list(dict.fromkeys(row[0] for row in self.rows)) if "kind" in fields else [
            mapped.label(fields["of"]) if "of" in fields else label]
        self.links = {}
        for name, link in (fields.get("links") or {}).items():
            if type(link) is str:
                self.links[name] = reader.texts(link, "links." + name, missing="omit")
            else:
                self.links[name] = End(reader, link, "links." + name, missing="omit")
        self.owners = tuple(self.made_kinds)
        self.reader = reader
        changing = [place for place, row in enumerate(self.rows) if row[2]["form"] == "change"]
        if changing:
            groups = [(self.rows[place][2]["graph"], self.rows[place][0], joined(self.rows[place][1]),
                       repr(self.rows[place][2]["seen"])) for place in changing]
            ends = following(groups, [self.rows[place][2]["since"] for place in changing], "declaration " + self.label)
            for place, until in zip(changing, ends):
                self.rows[place][2]["until"] = until

    @timed("mapping_build.NodeStated.kinds")
    def kinds(self, where):
        if "of" in self.fields:
            raise ValueError("%s names under of the nodes %s, which states of" % (where, self.label))
        return self.made_kinds

    @timed("mapping_build.NodeStated.made")
    def made(self, mapped):
        where, held = "declaration %s" % self.label, {}
        for kind, key, fact, _ in self.rows:
            mapped.make(fact["graph"], kind, key, where)
            if held.setdefault((fact["graph"], joined(key)), kind) != kind:
                raise ValueError("%s states the key %s under the kinds %s and %s; its keys are unique across them"
                                 % (where, joined(key), held[(fact["graph"], joined(key))], kind))
        for name, link in self.links.items():
            if type(link) is list:
                if name in mapped.names():
                    raise ValueError("declaration %s makes under links the kind %s, which a declaration of the script "
                                     "also bears" % (self.label, name))
                for (_, _, fact, _), key in zip(self.rows, link):
                    if key is not None:
                        mapped.make(fact["graph"], name, key, where)

    @timed("mapping_build.NodeStated.resolve")
    def resolve(self, mapped):
        self.linked = []
        for name, link in self.links.items():
            for place, (kind, key, fact, _) in enumerate(self.rows):
                for graph in mapped.graphed(fact["graph"]):
                    if type(link) is list:
                        found = None if link[place] is None else (name, joined(link[place]))
                    else:
                        found = link.resolve(mapped, place, graph, fact)
                    if found is not None:
                        self.linked.append(("%s.%s" % (kind, name), (kind, joined(key)), found, dict(fact, graph=graph)))

    @timed("mapping_build.NodeStated.schema")
    def schema(self, mapped):
        for kind in self.made_kinds:
            mapped.kind(kind, self.columns, self.label)
        for name, link in self.links.items():
            if type(link) is list:
                mapped.kind(name, {}, self.label)
        for relation, source, target, _ in self.linked:
            mapped.relation(relation, (source[0], target[0]), True, {}, None, None, self.label)

    @timed("mapping_build.NodeStated.issue")
    def issue(self, mapped, phase):
        if phase == "nodes":
            for kind, key, fact, values in self.rows:
                mapped.node(kind, joined(key), fact, values)
            for name, link in self.links.items():
                if type(link) is list:
                    for row, key in zip(self.rows, link):
                        if key is not None:
                            for graph in mapped.graphed(row[2]["graph"]):
                                mapped.always(name, joined(key), graph)
        elif phase == "edges":
            for relation, source, target, fact in self.linked:
                mapped.edge(relation, source, target, fact, True, {})


class EdgeStated(Stated):
    @timed("mapping_build.EdgeStated")
    def __init__(self, mapped, label, statement, session):
        fields = self.fields = statement.fields
        self.label, self.named, self.index = label, [], {}
        if "inside" in fields:
            raise ValueError("declaration %s states inside; the base layer states no held graphs" % label)
        reader = Reader(session, statement.table, label, mapped.plan).where(fields.get("where"))
        self.facts = facts(reader, fields, mapped)
        self.columns, self.values = reader.features(fields.get("features"))
        self.graphs = [fact["graph"] for fact in self.facts]
        self.source = End(reader, fields["source"], "source", creatable=True)
        self.target = End(reader, fields["target"], "target", creatable=True)
        self.relations = reader.texts(fields["relation"], "relation") if "relation" in fields else None
        self.present = [True] * reader.count
        if "present" in fields:
            values, kind = reader.values(fields["present"], "present")
            if kind.name not in ("flag", "missing") or None in values:
                raise ValueError("declaration %s field present gives a flag in every row" % label)
            self.present = values
        self.keys = reader.texts(fields["key"], "key") if "key" in fields else None
        self.owners = (label,)

    @timed("mapping_build.EdgeStated.kinds")
    def kinds(self, where):
        if self.keys is None:
            raise ValueError("%s names under of the edges %s, which states no key" % (where, self.label))
        return [self.label + ".edge"]

    @timed("mapping_build.EdgeStated.resolve")
    def resolve(self, mapped):
        self.rows = []
        for place, fact in enumerate(self.facts):
            graphs = mapped.graphed(fact["graph"])
            if fact["graph"] is None and len(graphs) > 1:
                graphs = [graph for graph in graphs if all(
                    end.keys[place] is not None and len(mapped.find(graph, end.kinds(mapped, place), end.keys[place])) == 1
                    for end in (self.source, self.target))] or [None]
            for graph in graphs:
                held = dict(fact, graph=graph)
                source = self.source.resolve(mapped, place, graph, held)
                target = self.target.resolve(mapped, place, graph, held) if source is not None else None
                if target is not None:
                    self.rows.append((place, held, source, target))
        changing = [row for row in self.rows if row[1]["form"] == "change"]
        if changing:
            groups = [(row[1]["graph"], self.relation(row[0]), row[2], row[3], repr(row[1]["seen"])) for row in changing]
            for row, until in zip(changing, following(groups, [row[1]["since"] for row in changing],
                                                      "declaration " + self.label)):
                row[1]["until"] = until
        self.between = {}
        for place, fact, source, target in self.rows:
            found = self.between.setdefault((fact["graph"], source[1], target[1]), [])
            if self.named_relation(place) not in found:
                found.append(self.named_relation(place))
            if not self.fields["directed"]:
                self.between.setdefault((fact["graph"], target[1], source[1]), found)
        if self.keys is not None:
            mapped.keyed[self.label + ".edge"] = self
            mapped.setkinds.add(self.label + ".edge")
            for place, fact, source, target in self.rows:
                key = joined(self.keys[place])
                entries = self.index.setdefault((fact["graph"], key), [])
                if entries and entries[0][1:4] != (source, target, self.named_relation(place)):
                    raise ValueError("declaration %s states the key %s of two different edges" % (self.label, key))
                entries.append((fact, source, target, self.named_relation(place)))
                mapped.make(fact["graph"], self.label + ".edge", key, "declaration " + self.label)

    @timed("mapping_build.EdgeStated.relation")
    def relation(self, place):
        return self.label if self.relations is None else (self.label, self.relations[place])

    @timed("mapping_build.EdgeStated.named_relation")
    def named_relation(self, place):
        return self.label if self.relations is None else self.relations[place]

    @timed("mapping_build.EdgeStated.sources")
    def sources(self):
        return list(dict.fromkeys(source[0] for _, _, source, _ in self.rows))

    @timed("mapping_build.EdgeStated.schema")
    def schema(self, mapped):
        about = self.fields.get("about")
        for place, _, source, target in self.rows:
            if (source[0] in mapped.setkinds or target[0] in mapped.setkinds) and about is None:
                raise ValueError("declaration %s joins an edge or a set edge and lacks about" % self.label)
            mapped.relation(self.label, (source[0], target[0]), self.fields["directed"], self.columns,
                            None if self.relations is None else "relation",
                            about if source[0] in mapped.setkinds or target[0] in mapped.setkinds else None, self.label)
        if self.named:
            kinds = list(dict.fromkeys(kind for entries in self.index.values() for _, source, target, _ in entries
                                       for kind in (source[0], target[0])))
            mapped.sets[self.label + ".edge"] = dict(kinds=kinds, roles=["source", "target"], members=[], columns=[])

    @timed("mapping_build.EdgeStated.issue")
    def issue(self, mapped, phase):
        if phase == "created":
            kind = self.label + ".edge"
            for identity in dict.fromkeys(self.named):
                for fact, _, _, _ in self.index[identity]:
                    mapped.node(kind, identity[1], dict(fact, form="state"), {})
            for identity in dict.fromkeys(self.named):
                _, source, target, _ = self.index[identity][0]
                mapped.builder.member(kind, identity[1], source, "source", graph=identity[0])
                mapped.builder.member(kind, identity[1], target, "target", graph=identity[0])
        elif phase == "edges":
            for place, fact, source, target in self.rows:
                mapped.edge(self.relation(place), source, target, fact, self.present[place], self.values[place])


class SetStated(Stated):
    @timed("mapping_build.SetStated")
    def __init__(self, mapped, label, statement, session):
        fields = self.fields = statement.fields
        self.label = label
        reader = Reader(session, statement.table, label, mapped.plan).where(fields.get("where"))
        self.facts = facts(reader, fields, mapped)
        self.columns, self.values = reader.features(fields.get("features"))
        self.keys = reader.texts(fields["key"], "key")
        self.graphs = [fact["graph"] for fact in self.facts]
        self.members, self.member_columns = [], {}
        for place, member in enumerate(fields["members"]):
            rows = Reader(session, member["rows"], label, mapped.plan)
            columns, values = rows.features(member.get("features"), "members.%d.features" % place)
            self.member_columns.update(columns)
            self.members.append((rows.texts(member["set"], "members.%d.set" % place),
                                 End(rows, member["node"], "members.%d.node" % place, creatable=True),
                                 rows.texts(member["role"], "members.%d.role" % place), values))
        self.owners = (label,)

    @timed("mapping_build.SetStated.kinds")
    def kinds(self, where):
        return [self.label]

    @timed("mapping_build.SetStated.made")
    def made(self, mapped):
        mapped.setkinds.add(self.label)
        for fact, key in zip(self.facts, self.keys):
            mapped.make(fact["graph"], self.label, key, "declaration " + self.label)

    @timed("mapping_build.SetStated.resolve")
    def resolve(self, mapped):
        self.rows = []
        for sets, end, roles, values in self.members:
            for place, key in enumerate(sets):
                graphs = [graph for graph in mapped.graphs if joined(key) in mapped.keys.get((graph, self.label), {})]
                if not graphs:
                    raise ValueError("declaration %s member row %d names the set edge %s, which no row states"
                                     % (self.label, end.rows[place], joined(key)))
                for graph in graphs:
                    fact = next(fact for fact, stated in zip(self.facts, self.keys)
                                if joined(stated) == joined(key) and fact["graph"] in (None, graph))
                    node = end.resolve(mapped, place, graph, dict(fact, graph=graph))
                    if node is not None:
                        self.rows.append((graph, joined(key), node, roles[place], values[place]))

    @timed("mapping_build.SetStated.schema")
    def schema(self, mapped):
        for name, spec in self.member_columns.items():
            mapped.column(name, spec, self.label)
        for name, spec in self.columns.items():
            mapped.column(name, spec, self.label)
        mapped.sets[self.label] = dict(kinds=list(dict.fromkeys(row[2][0] for row in self.rows)),
                                       roles=list(dict.fromkeys(row[3] for row in self.rows)),
                                       members=list(self.member_columns), columns=list(self.columns))

    @timed("mapping_build.SetStated.issue")
    def issue(self, mapped, phase):
        if phase == "nodes":
            for fact, key, values in zip(self.facts, self.keys, self.values):
                mapped.node(self.label, joined(key), fact, values)
        elif phase == "edges":
            for graph, key, node, role, values in self.rows:
                mapped.builder.member(self.label, key, node, role, graph=graph,
                                      values={name: values.get(name) for name in self.member_columns})


class GraphStated(Stated):
    @timed("mapping_build.GraphStated")
    def __init__(self, mapped, label, statement, session):
        reader = Reader(session, statement.table, label, mapped.plan).where(statement.fields.get("where"))
        self.columns, values = reader.features(statement.fields.get("features"))
        self.graphs = reader.texts(statement.fields["key"], "key")
        self.label = label
        for place, (graph, held) in enumerate(zip(self.graphs, values)):
            if graph in mapped.attributes:
                raise ValueError("declaration %s row %d states the graph %s twice" % (label, reader.row(place), graph))
            mapped.attributes[named(graph, "declaration %s field key" % label)] = held

    @timed("mapping_build.GraphStated.schema")
    def schema(self, mapped):
        for name, spec in self.columns.items():
            mapped.column(name, spec, self.label)
        mapped.graph_columns.extend(name for name in self.columns if name not in mapped.graph_columns)


class CustomStated(Stated):
    @timed("mapping_build.CustomStated")
    def __init__(self, mapped, label, statement, session):
        reader = Reader(session, statement.table, label, mapped.plan)
        frame = reader.frame
        self.rows = pa.Table.from_arrays([custom_argument(frame.table[name], frame.kinds[name])
                                             for name in frame.table.column_names], names=frame.table.column_names)
        self.label, self.word = label, statement.word
        self.parameters = {name: value for name, value in statement.fields.items() if name != "schema"}
        self.stated = statement.fields["schema"]

    @timed("mapping_build.CustomStated.schema")
    def schema(self, mapped):
        stated = self.stated
        for name, spec in (stated.get("columns") or {}).items():
            mapped.column(name, dict(spec), self.label)
        for kind, entry in (stated.get("nodes") or {}).items():
            mapped.kind(kind, {name: None for name in entry["columns"]}, self.label)
        for kind, entry in (stated.get("sets") or {}).items():
            mapped.sets[kind] = {name: list(value) for name, value in entry.items()}
        for table, entry in (stated.get("edges") or {}).items():
            for pair in entry["endpoints"]:
                mapped.relation(table, tuple(pair), entry["directed"], {name: None for name in entry["columns"]},
                                None, entry.get("about"), self.label)

    @timed("mapping_build.CustomStated.issue")
    def issue(self, mapped, phase):
        if phase == "custom":
            MAPPINGS.named(self.word)(mapped.builder, self.rows, dict(self.parameters))


class Mapped:
    @timed("mapping_build.Mapped")
    def __init__(self, builder, plan, name):
        self.builder, self.plan, self.name = builder, plan, name
        declared = [(label, value) for label, value in builder.declarations.items() if isinstance(value, Mapping)]
        generated, self.represented, aliases, follows = expanded(builder.declarations, declared, builder.sources)
        self.statements = []
        for label, value in builder.declarations.items():
            if isinstance(value, Mapping):
                self.statements += [(label, value)] + follows.get(id(value), [])
            self.statements += generated.get(label, [])
        self.labels = {id(value): label for label, value in builder.declarations.items()}
        for label, value in self.statements:
            if id(value) not in self.labels and label in builder.declarations:
                raise ValueError("a database makes the kind or relation %s, which a declaration of the script also "
                                 "bears" % label)
            self.labels[id(value)] = label
        if len({label for label, _ in self.statements}) != len(self.statements):
            raise ValueError("the statements and the databases of the script name two declarations alike")
        for identity, (_, statement) in self.represented.items():
            if statement is not None:
                self.labels[identity] = self.labels[id(statement)]
        self.aliases = {identity: source for identity, (_, source) in aliases.items()}
        self.aliased = dict(aliases.values())
        self.order = {id(value): place for place, (_, value) in enumerate(self.statements)}
        stated = [statement.fields for _, statement in self.statements if statement.word in FACTS]
        asked = [statement.fields for _, statement in self.statements if statement.word in ("tasks", "decisions")]
        self.timed = any("holds" in fields for fields in stated) or any("at" in fields for fields in asked)
        self.clocks = any(fields.get("known") is not None or fields.get("replaced") is not None for fields in stated)
        self.observers = any("seen" in fields for fields in stated) or any("observer" in fields for fields in asked)
        self.stated, self.keys, self.attributes, self.created, self.setkinds = {}, {}, {}, {}, set()
        self.columns, self.kinded, self.sets, self.edges, self.graph_columns = {}, {}, {}, {}, []
        self.references, self.tasks, self.grouping, self.signals = {}, {}, None, 0
        self.graphs, self.unit, self.model, self.ties, self.current = [], None, "none", None, None
        self.made, self.keyed, self.numbers = set(), {}, {}

    @timed("mapping_build.Mapped.names")
    def names(self):
        return set(self.labels.values())

    @timed("mapping_build.Mapped.label")
    def label(self, declaration):
        if id(declaration) not in self.labels:
            raise ValueError("a mapping names a declaration that is bound to no name")
        return self.labels[id(declaration)]

    @timed("mapping_build.Mapped.tables")
    def tables(self):
        found = {name: value for name, value in self.builder.declarations.items() if isinstance(value, (Source, Table))}
        found.update(self.aliased)
        held = {id(value) for value in found.values()}
        for label, statement in self.statements:
            for place, value in enumerate(inputs(statement)):
                value = self.aliases.get(id(value), value)
                if isinstance(value, Part):
                    raise ValueError("declaration %s reads a table of a database that is not stated" % label)
                if id(value) not in held:
                    held.add(id(value))
                    found["%s.rows%d" % (label, place)] = value
        return found

    @timed("mapping_build.Mapped.run")
    def run(self):
        started = time.perf_counter()
        tables = self.tables()
        if not tables:
            raise ValueError("the mappings of the builder read no table")
        with Session(tables, self.builder.sources, self.plan, aliases=self.aliases) as session:
            for value in tables.values():
                session.get(value)
            self.unit = None if session.unit is UNSTATED else session.unit
            if self.timed:
                self.model = "order" if self.unit is None else "clocks" if self.clocks else "clock"
            if self.clocks and self.model != "clocks":
                raise ValueError("the mappings state known or replaced, which needs graphs counted on a clock")
            for label, statement in self.statements:
                entry = MAPPINGS.named(statement.word)
                with working(Context(label, self.plan, session.unit, session.retained())):
                    self.stated[id(statement)] = entry.evaluate(self, label, statement, session) \
                        if hasattr(entry, "evaluate") else CustomStated(self, label, statement, session)
        self.declared_graphs()
        for stated in self.stated.values():
            stated.made(self)
        for _, statement in self.statements:
            self.current = statement
            self.stated[id(statement)].resolve(self)
        self.builder.schema(self.schema())
        for graph in self.graphs:
            columns = self.attributes.get(graph, {}) if self.graph_columns else {}
            if self.graph_columns and graph not in self.attributes:
                raise ValueError("a mapping names the graph %s, which no row of graphs states" % graph)
            self.builder.graph(graph, self.unit if self.model in UNITED else None, values=columns)
        for phase in PHASES:
            if phase == "created":
                for (graph, kind, key), fact in self.created.items():
                    self.node(kind, key, fact, {})
            for _, statement in self.statements:
                self.stated[id(statement)].issue(self, phase)
        LOG.info("mapped exit statements=%d graphs=%d seconds=%.3f", len(self.statements), len(self.graphs),
                 time.perf_counter() - started)

    @timed("mapping_build.Mapped.declared_graphs")
    def declared_graphs(self):
        found = list(self.attributes)
        for stated in self.stated.values():
            found += [graph for graph in stated.graphs if graph is not None]
        self.graphs = list(dict.fromkeys(found)) or [self.name]

    @timed("mapping_build.Mapped.graphed")
    def graphed(self, graph):
        return list(self.graphs) if graph is None else [graph]

    @timed("mapping_build.Mapped.make")
    def make(self, graph, kind, key, where):
        for held in self.graphed(graph):
            found = self.keys.setdefault((held, kind), {})
            text = joined(key)
            if text in found and found[text] != key:
                raise ValueError("%s states two keys of the kind %s, %r and %r, that the bank cannot tell apart"
                                 % (where, kind, found[text], key))
            found[text] = key

    @timed("mapping_build.Mapped.find")
    def find(self, graph, kinds, key):
        text = joined(key)
        return [kind for kind in kinds if text in self.keys.get((graph, kind), {})]

    @timed("mapping_build.Mapped.kinds")
    def kinds(self, target, where):
        if isinstance(target, Linked):
            return [target.name]
        if isinstance(target, Part):
            return self.stated[id(self.part(target, where))].kinds(where)
        if id(target) not in self.stated or self.order[id(target)] >= self.order[id(self.current)]:
            raise ValueError("%s names under of a declaration that is not a mapping stated above it" % where)
        return self.stated[id(target)].kinds(where)

    @timed("mapping_build.Mapped.part")
    def part(self, target, where):
        table, statement = self.represented.get(id(target), (owner(target)[1], None))
        if statement is None:
            raise ValueError("%s names the table %s of a database, which becomes no one mapping; a view, a table "
                             "without a key and a table several statements map make none" % (where, table))
        return statement

    @timed("mapping_build.Mapped.note")
    def note(self, graph, kind, key):
        if kind in self.keyed:
            self.keyed[kind].named.append((graph, key))

    @timed("mapping_build.Mapped.create")
    def create(self, graph, kind, key, fact, target):
        self.make(graph, kind, key, "a node made by create")
        always = isinstance(target, Linked)
        held = dict(fact, form="state", seq=0, until=None if always else fact["until"],
                    since=None if always else fact["since"])
        if always:
            held.update(known=None, replaced=None, seen=None)
        self.created.setdefault((graph, kind, key), held)

    @timed("mapping_build.Mapped.absent")
    def absent(self, where, rule, parameters, key, since, row):
        entry = PRESENCE.named(rule)
        if entry.creates:
            raise ValueError("%s states the rule %s, which makes a node only for an end of edges or a member of sets"
                             % (where, rule))
        try:
            (stand,) = entry([key], [since], parameters)
        except ValueError as error:
            raise ValueError("%s row %d key %s: %s" % (where, row, key, error)) from error
        self.count(where, rule)
        return stand

    @timed("mapping_build.Mapped.numbered")
    def numbered(self, graph, kind, key, where):
        held = self.numbers.setdefault(graph, {})
        if (kind, key) in held:
            raise ValueError("%s names two tasks of the graph %s by the key %s" % (where, graph, key))
        held[(kind, key)] = len(held)
        return held[(kind, key)]

    @timed("mapping_build.Mapped.count")
    def count(self, where, rule):
        held = self.builder.counts.setdefault(where, {})
        held[rule] = held.get(rule, 0) + 1

    @timed("mapping_build.Mapped.column")
    def column(self, name, spec, label):
        if name in self.columns and self.columns[name][0] != spec:
            raise ValueError("the column %s is stated by %s as %s and by %s as %s"
                             % (name, self.columns[name][1], self.columns[name][0], label, spec))
        self.columns.setdefault(name, (spec, label))

    @timed("mapping_build.Mapped.kind")
    def kind(self, kind, columns, label):
        held = self.kinded.setdefault(kind, [])
        for name, spec in columns.items():
            if spec is not None:
                self.column(name, spec, label)
            if name not in held:
                held.append(name)

    @timed("mapping_build.Mapped.relation")
    def relation(self, table, pair, directed, columns, column, about, label):
        for name, spec in columns.items():
            if spec is not None:
                self.column(name, spec, label)
        entry = self.edges.setdefault(table, dict(endpoints=[], directed=directed, columns=list(columns),
                                                  relation=DECLARED if column is None else dict(column=column),
                                                  about=about))
        if entry["directed"] != directed or entry["columns"] != list(columns):
            raise ValueError("the relation %s is stated twice with different directions or columns, by %s" % (table, label))
        if list(pair) not in entry["endpoints"]:
            entry["endpoints"].append(list(pair))

    @timed("mapping_build.Mapped.tied")
    def tied(self):
        rules = {}
        for label, statement in self.statements:
            ties = statement.fields.get("ties")
            if ties is not None:
                rules[label] = ties if type(ties) is str else ties["name"]
        if len(set(rules.values())) > 1:
            raise ValueError("the mappings state different rules for ties: %s" % ", ".join(
                "%s %s" % pair for pair in sorted(rules.items())))
        rule = next(iter(rules.values()), None)
        if rule == "sequence":
            owned = {owner for label, statement in self.statements if label in rules
                     for owner in self.stated[id(statement)].owners}
            for label, statement in self.statements:
                shared = owned & set(self.stated[id(statement)].owners)
                if label not in rules and shared:
                    raise ValueError("declaration %s states no ties, and facts of %s carry the rule sequence; every "
                                     "mapping of their facts states the rule" % (label, ", ".join(sorted(shared))))
        return rule

    @timed("mapping_build.Mapped.schema")
    def schema(self):
        for _, statement in self.statements:
            self.current = statement
            self.stated[id(statement)].schema(self)
        self.ties = self.tied()
        result = dict(
            version=VERSION,
            time=dict(model=self.model, units=[self.unit] if self.model in UNITED else [], ties=self.ties),
            observers=self.observers,
            columns={name: dict(spec) for name, (spec, _) in self.columns.items()},
            graph=dict(columns=list(self.graph_columns)),
            nodes={kind: dict(columns=list(columns), holds=None) for kind, columns in self.kinded.items()},
            sets={kind: dict(entry) for kind, entry in self.sets.items()},
            edges={table: dict(entry) for table, entry in self.edges.items()},
            tasks=dict(key="key", grouping=self.grouping, signals=self.signals, references=self.references,
                       kinds=self.tasks),
        )
        return result

    @timed("mapping_build.Mapped.stamps")
    def stamps(self, fact):
        result = dict(known=fact["known"], replaced=fact["replaced"])
        if self.observers:
            result["seen"] = fact["seen"]
        if self.ties == "sequence":
            result["seq"] = fact["seq"]
        return result

    @timed("mapping_build.Mapped.node")
    def node(self, kind, key, fact, values):
        for graph in self.graphed(fact["graph"]):
            if self.model == "none":
                self.builder.node(kind, key, graph=graph, values=values)
            elif fact["form"] == "change":
                self.builder.change(kind, key, fact["since"], graph=graph, values=values, **self.stamps(fact))
            else:
                self.builder.node(kind, key, fact["since"], fact["until"], graph=graph, values=values,
                                  **self.stamps(fact))

    @timed("mapping_build.Mapped.always")
    def always(self, kind, key, graph):
        if (graph, kind, key) in self.made:
            return
        self.made.add((graph, kind, key))
        self.node(kind, key, dict(graph=graph, form="state", since=None, until=None, known=None, replaced=None,
                                  seen=None, seq=0), {})

    @timed("mapping_build.Mapped.edge")
    def edge(self, relation, source, target, fact, present, values):
        graph = fact["graph"]
        if self.model == "none":
            self.builder.edge(relation, source, target, graph=graph, present=present, values=values)
        elif fact["form"] == "event":
            self.builder.event(relation, source, target, fact["since"], graph=graph, present=present, values=values,
                               **self.stamps(fact))
        else:
            self.builder.edge(relation, source, target, fact["since"], fact["until"], graph=graph, present=present,
                              values=values, **self.stamps(fact))
