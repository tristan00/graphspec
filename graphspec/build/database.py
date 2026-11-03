import weakref
from collections.abc import Mapping as Held
from pathlib import Path

from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.columns import statement

from .table_types import frozen, references

TIMES = Names("rule of the columns of time", True)
FIELDS = ("place", "reader", "tables", "unit", "times", "through", "encoders", "missing", "absent", "graph", "leave",
          "outcomes", "definitions")
REQUIRED = ("place", "reader", "tables", "times", "encoders", "missing", "absent")
STATED = weakref.WeakKeyDictionary()


class Part:
    @timed("database.Part")
    def __init__(self, database, name):
        STATED[self] = (database, name)

    @timed("database.Part.getattr")
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        database, held = STATED[self]
        return part(database, held + "." + name)


class Database:
    @timed("database.Database")
    def __init__(self, fields):
        where = "the statement database"
        unknown = sorted(set(fields) - set(FIELDS))
        if unknown:
            raise ValueError("%s holds fields it does not know: %s" % (where, ", ".join(unknown)))
        lacking = [name for name in REQUIRED if name not in fields]
        if lacking:
            raise ValueError("%s lacks the required field %s" % (where, ", ".join(lacking)))
        if not isinstance(fields["place"], (str, Path)) or not str(fields["place"]):
            raise ValueError("%s states its place as a nonempty path or connection string" % where)
        if type(fields["reader"]) is not str or not fields["reader"]:
            raise ValueError("%s states the name of its reader" % where)
        for name in ("tables", "leave"):
            value = fields.get(name, ["*"])
            if not isinstance(value, (list, tuple)) or not value or any(type(item) is not str or not item for item in value):
                raise ValueError("%s states %s as a nonempty list of names or patterns" % (where, name))
        if any("." not in pattern for pattern in fields.get("leave", ())):
            raise ValueError("%s states under leave each column as <table>.<column>" % where)
        if fields["times"] is not None:
            ruled(fields["times"], where + " field times")
            if fields.get("unit", None) is None:
                raise ValueError("%s states times and lacks unit, the unit the graph counts in" % where)
        for name in ("encoders", "outcomes"):
            value = fields.get(name, {})
            if not isinstance(value, Held) or any(type(key) is not str or not key or type(item) is not str or not item
                                                  for key, item in value.items()):
                raise ValueError("%s states %s as a mapping of names" % (where, name))
        if any("." not in name for name in fields.get("outcomes", {})):
            raise ValueError("%s states under outcomes each column as <table>.<column>" % where)
        if "graph" in fields and (type(fields["graph"]) is not str or not fields["graph"]):
            raise ValueError("%s states graph as the name of a column" % where)
        if "definitions" in fields and (not isinstance(fields["definitions"], (str, Path)) or not str(fields["definitions"])):
            raise ValueError("%s states definitions as the path of a file of definitions in SQL" % where)
        STATED[self] = (frozen(dict(fields)), {})

    @timed("database.Database.getattr")
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return part(self, name)


@timed("database.database")
def database(place, **fields):
    return Database(dict(fields, place=place))


@timed("database.part")
def part(database, name):
    held = STATED[database][1]
    if name not in held:
        held[name] = Part(database, name)
    return held[name]


@timed("database.stated")
def stated(database):
    return STATED[database][0]


@timed("database.owner")
def owner(held):
    return STATED[held]


@timed("database.ruled")
def ruled(stated, where):
    if type(stated) is str and stated:
        return stated, {}
    if isinstance(stated, Held) and type(stated.get("name")) is str:
        return stated["name"], {name: value for name, value in stated.items() if name != "name"}
    raise ValueError("%s states a rule by name, or as a mapping of name and its fields" % where)


class Typed:
    version = 1

    @timed("database.Typed")
    def __call__(self, table, columns, parameters):
        found = [name for name, given in columns.items() if statement(given).get("type") == "time"]
        if len(found) > 2:
            raise ValueError("the rule typed meets the table %s, which holds the columns of time %s; state its times by "
                             "a mapping of the table" % (table, ", ".join(found)))
        if not found:
            return dict(holds="always", known=None)
        return dict(holds=dict(name="state", since=found[0], until=found[1] if len(found) == 2 else None), known=None)


class Named:
    version = 1

    @timed("database.Named")
    def __call__(self, table, columns, parameters):
        if set(parameters) != {"holds", "known"}:
            raise ValueError("the rule named states holds and known")
        holds, known = parameters["holds"], parameters["known"]
        texts = [value for name, value in (holds.items() if isinstance(holds, Held) else ()) if name != "name"]
        names = {name for text in texts + [known] if type(text) is str for _, name in references(text)}
        if all(name in columns for name in names):
            return dict(holds=holds, known=known)
        return dict(holds="always", known=None)


TIMES.register("typed", Typed())
TIMES.register("named", Named())
