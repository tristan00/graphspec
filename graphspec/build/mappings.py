from collections.abc import Mapping as Held
from dataclasses import dataclass
from importlib import import_module

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .database import Part
from .readers import Source
from .table_types import frozen
from .tables import Table

MAPPINGS = Names("mapping", True)
TIMES = ("holds", "known", "replaced", "seen", "ties")
FACT = ("where", "graph", "features") + TIMES
ASKED = ("key", "at", "observer", "grouping", "where", "graph", "earlier", "eligible", "references", "features",
         "signals", "weight", "split")
END = ("of", "key", "kind", "absent")


@dataclass(frozen=True, eq=False)
class Linked:
    mapping: object
    name: str


@dataclass(frozen=True, init=False, eq=False)
class Mapping:
    word: str
    table: object
    _fields: object

    @timed("mappings.Mapping")
    def __init__(self, word, table, fields):
        entry = MAPPINGS.named(word)
        where = "the statement %s" % word
        fields = dict(fields)
        if hasattr(entry, "evaluate"):
            unknown = sorted(set(fields) - set(entry.fields))
            if unknown:
                raise ValueError("%s holds fields it does not know: %s" % (where, ", ".join(unknown)))
            if not tabled(table) and not (entry.ruled and isinstance(table, Held)):
                raise ValueError("%s reads a source or table declaration as its first argument" % where)
            entry.validate(fields, where)
        elif not tabled(table):
            raise ValueError("%s reads a source or table declaration as its first argument" % where)
        object.__setattr__(self, "word", word)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "_fields", frozen(fields))

    @property
    @timed("mappings.Mapping.fields")
    def fields(self):
        return self._fields

    @timed("mappings.Mapping.getattr")
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        links = self._fields.get("links") if self.word == "nodes" else None
        if links is not None and type(links.get(name)) is str:
            return Linked(self, name)
        raise AttributeError("the statement %s makes no kind %s under links" % (self.word, name))


@timed("mappings.tabled")
def tabled(value):
    return isinstance(value, (Source, Table, Part))


@timed("mappings.expression")
def expression(value, where, listed=False):
    if type(value) is str and value:
        return
    if listed and isinstance(value, (list, tuple)) and value and all(type(item) is str and item for item in value):
        return
    raise ValueError("%s states %s" % (where, "an expression or a list of them" if listed else "an expression"))


@timed("mappings.stated")
def stated(fields, names, where):
    lacking = [name for name in names if name not in fields]
    if lacking:
        raise ValueError("%s lacks the required field %s" % (where, ", ".join(lacking)))


@timed("mappings.encoded")
def encoded(value, where):
    if value is None:
        return
    if not isinstance(value, Held) or any(type(name) is not str or not name or type(encoder) is not str
                                          for name, encoder in value.items()):
        raise ValueError("%s states a mapping from a column to the name of its encoder" % where)


@timed("mappings.ended")
def ended(value, where):
    if not isinstance(value, Held):
        raise ValueError("%s states an end as a mapping of of, key and absent" % where)
    stated(value, ("of", "key", "absent"), where)
    unknown = sorted(set(value) - set(END))
    if unknown:
        raise ValueError("%s holds fields it does not know: %s" % (where, ", ".join(unknown)))
    targets = value["of"] if isinstance(value["of"], (list, tuple)) else [value["of"]]
    if not targets or any(not isinstance(target, (Mapping, Linked, Part)) for target in targets):
        raise ValueError("%s names under of a mapping stated above, or a list of them" % where)
    if isinstance(value["of"], (list, tuple)) and "kind" not in value:
        raise ValueError("%s names a list under of and states kind" % where)
    expression(value["key"], where + " key", True)


@timed("mappings.timing")
def timing(fields, where):
    holds = fields.get("holds")
    if holds is not None and holds != "always" and not (isinstance(holds, Held) and type(holds.get("name")) is str):
        raise ValueError("%s states holds as always or as a mapping of a form and its fields" % where)
    for name in ("known", "replaced"):
        if fields.get(name) is not None:
            expression(fields[name], "%s %s" % (where, name))
    if fields.get("seen") is not None:
        expression(fields["seen"], where + " seen")
    ties = fields.get("ties")
    if ties is not None and ties != "key" and not (isinstance(ties, Held) and type(ties.get("name")) is str):
        raise ValueError("%s states ties by name, or as a mapping of name and its fields" % where)


class Entry:
    stated = None

    @timed("mappings.Entry.evaluate")
    def evaluate(self, mapped, label, statement, session):
        module, name = self.stated
        return getattr(import_module("graphspec.build." + module), name)(mapped, label, statement, session)


class Nodes(Entry):
    version = 1
    ruled = False
    stated = ("mapping_build", "NodeStated")
    fields = frozenset(("key", "distinct", "of", "kind", "links", "held", "inside") + FACT)

    @timed("mappings.Nodes.validate")
    def validate(self, fields, where):
        if ("key" in fields) == ("distinct" in fields):
            raise ValueError("%s states one of key and distinct" % where)
        if "key" in fields:
            expression(fields["key"], where + " key", True)
        else:
            expression(fields["distinct"], where + " distinct", True)
            for name in ("features", "of", "kind", "links"):
                if name in fields:
                    raise ValueError("%s states distinct and takes no %s" % (where, name))
        if "of" in fields:
            if not isinstance(fields["of"], Mapping) or fields["of"].word != "nodes" or "of" in fields["of"].fields \
                    or "key" not in fields["of"].fields:
                raise ValueError("%s names under of a nodes statement that states key and no of" % where)
            for name in ("distinct", "kind", "held", "inside"):
                if name in fields:
                    raise ValueError("%s states of and takes no %s" % (where, name))
        encoded(fields.get("features"), where + " features")
        for name, link in (fields.get("links") or {}).items():
            if type(link) is not str:
                ended(link, "%s links %s" % (where, name))
        timing(fields, where)


class Edges(Entry):
    version = 1
    ruled = False
    stated = ("mapping_build", "EdgeStated")
    fields = frozenset(("source", "target", "directed", "relation", "key", "present", "about", "inside") + FACT)

    @timed("mappings.Edges.validate")
    def validate(self, fields, where):
        stated(fields, ("source", "target", "directed"), where)
        ended(fields["source"], where + " source")
        ended(fields["target"], where + " target")
        if type(fields["directed"]) is not bool:
            raise ValueError("%s states directed as True or False" % where)
        for name in ("relation", "present"):
            if name in fields:
                expression(fields[name], "%s %s" % (where, name))
        if "key" in fields:
            expression(fields["key"], where + " key", True)
        encoded(fields.get("features"), where + " features")
        timing(fields, where)


class Sets(Entry):
    version = 1
    ruled = False
    stated = ("mapping_build", "SetStated")
    fields = frozenset(("key", "members") + FACT)

    @timed("mappings.Sets.validate")
    def validate(self, fields, where):
        stated(fields, ("key", "members"), where)
        expression(fields["key"], where + " key", True)
        members = fields["members"]
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError("%s states members as a nonempty list" % where)
        for place, member in enumerate(members):
            inner = "%s members %d" % (where, place)
            if not isinstance(member, Held):
                raise ValueError("%s states a mapping of rows, set, node, role and features" % inner)
            unknown = sorted(set(member) - {"rows", "set", "node", "role", "features"})
            if unknown:
                raise ValueError("%s holds fields it does not know: %s" % (inner, ", ".join(unknown)))
            stated(member, ("rows", "set", "node", "role"), inner)
            if not tabled(member["rows"]):
                raise ValueError("%s rows names a source or table declaration" % inner)
            expression(member["set"], inner + " set", True)
            ended(member["node"], inner + " node")
            expression(member["role"], inner + " role")
            encoded(member.get("features"), inner + " features")
        encoded(fields.get("features"), where + " features")
        timing(fields, where)


class Graphs(Entry):
    version = 1
    ruled = False
    stated = ("mapping_build", "GraphStated")
    fields = frozenset(("key", "features", "where"))

    @timed("mappings.Graphs.validate")
    def validate(self, fields, where):
        stated(fields, ("key",), where)
        expression(fields["key"], where + " key")
        encoded(fields.get("features"), where + " features")


@timed("mappings.asked")
def asked(fields, where):
    references = fields.get("references")
    if not isinstance(references, Held):
        raise ValueError("%s states references as a mapping from a name to what it names" % where)
    for name, reference in references.items():
        if not isinstance(reference, Held) or type(reference.get("sort")) is not str:
            raise ValueError("%s references %s states its sort" % (where, name))
        if reference["sort"] != "graph":
            stated(reference, ("absent",), "%s references %s" % (where, name))
    if "at" in fields:
        expression(fields["at"], where + " at")
    if "key" in fields:
        expression(fields["key"], where + " key", True)
    encoded(fields.get("features"), where + " features")


class Tasks(Entry):
    version = 1
    ruled = True
    stated = ("mapping_tasks", "TaskStated")
    fields = frozenset(ASKED + ("event", "matures", "target"))

    @timed("mappings.Tasks.validate")
    def validate(self, fields, where):
        stated(fields, ("references",), where)
        asked(fields, where)
        target = fields.get("target")
        if target is not None:
            if not isinstance(target, Held):
                raise ValueError("%s states target as a mapping" % where)
            stated(target, ("kind", "value", "known", "about", "utility"), where + " target")
            stated(fields, ("split",), where)


class Decisions(Entry):
    version = 2
    ruled = True
    stated = ("mapping_tasks", "DecisionStated")
    fields = frozenset(ASKED + ("actors", "series", "about", "utility", "nothing", "rule", "generators", "ended",
                                "discount", "concurrent", "options", "actions"))

    @timed("mappings.Decisions.validate")
    def validate(self, fields, where):
        stated(fields, ("key", "references", "actors", "series", "about", "utility", "split", "options", "actions"),
               where)
        asked(fields, where)
        for name, required in (("options", ("rows", "decision", "key", "chosen", "available")),
                               ("actions", ("rows", "decision", "option", "kind", "actor", "kinds"))):
            if not isinstance(fields[name], Held):
                raise ValueError("%s states %s as a mapping" % (where, name))
            stated(fields[name], required, "%s %s" % (where, name))
            if not tabled(fields[name]["rows"]):
                raise ValueError("%s %s rows names a source or table declaration" % (where, name))
        if "generators" in fields:
            generators = fields["generators"]
            if not isinstance(generators, (list, tuple)) or not all(isinstance(entry, Held) for entry in generators):
                raise ValueError("%s states generators as a list of mappings" % where)
            for place, entry in enumerate(generators):
                stated(entry, ("name",), "%s generators.%d" % (where, place))
        nothing = fields.get("nothing")
        if nothing is not None and type(nothing) is not str:
            if not isinstance(nothing, Held):
                raise ValueError("%s states nothing as a quoted key or a mapping" % where)
            stated(nothing, ("key", "chosen", "available"), where + " nothing")


@timed("mappings.nodes")
def nodes(table, **fields):
    return Mapping("nodes", table, fields)


@timed("mappings.edges")
def edges(table, **fields):
    return Mapping("edges", table, fields)


@timed("mappings.sets")
def sets(table, **fields):
    return Mapping("sets", table, fields)


@timed("mappings.graphs")
def graphs(table, **fields):
    return Mapping("graphs", table, fields)


@timed("mappings.tasks")
def tasks(rows, **fields):
    return Mapping("tasks", rows, fields)


@timed("mappings.decisions")
def decisions(rows, **fields):
    return Mapping("decisions", rows, fields)


@timed("mappings.mapping")
def mapping(table, name, schema, **parameters):
    if not isinstance(schema, Held):
        raise ValueError("the mapping %s states schema, the kinds, relations and columns its calls name" % name)
    return Mapping(name, table, dict(parameters, schema=schema))


MAPPINGS.register("nodes", Nodes())
MAPPINGS.register("edges", Edges())
MAPPINGS.register("sets", Sets())
MAPPINGS.register("graphs", Graphs())
MAPPINGS.register("tasks", Tasks())
MAPPINGS.register("decisions", Decisions())
