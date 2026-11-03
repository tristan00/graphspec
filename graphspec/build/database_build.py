import logging
import re
import sqlite3
import time
from collections.abc import Mapping as Held
from fnmatch import fnmatchcase
from pathlib import Path

from graphspec.infra.timing import timed

from .database import TIMES, Database, Part, owner, part, stated
from .mappings import Mapping, edges, nodes
from .readers import READERS, Source
from .table_types import ruled

LOG = logging.getLogger(__name__)
FEATURED = ("number", "flag", "category")
OVERRIDING = ("nodes", "edges", "sets", "graphs")
DAY, MOMENT = "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"
INTEGERS = ("int", "integer", "tinyint", "smallint", "mediumint", "bigint", "int2", "int4", "int8")
REALS = ("real", "float", "float4", "float8", "double", "numeric", "decimal", "number")


@timed("database_build.typed")
def typed(declared, reader, unit):
    name = re.sub(r"\(.*\)", "", declared.lower()).strip()
    words = name.split()
    if "[" in name or not words:
        return None
    if name in ("bool", "boolean"):
        return dict(type="flag", true=1, false=0) if reader == "sqlite" else "flag"
    if name == "date":
        return dict(type="time", unit=unit, format=DAY)
    if words[0] in ("datetime", "timestamp", "timestamptz"):
        return dict(type="time", unit=unit, format=MOMENT)
    if set(words) & set(INTEGERS) or words[0] in REALS:
        return "number"
    if any("char" in word or word in ("text", "clob", "string") for word in words):
        return "category"
    return None


@timed("database_build.definitions")
def definitions(path, where):
    try:
        return READERS.named("sqlite").definitions(Path(path).read_text(encoding="utf-8"))
    except (OSError, sqlite3.Error) as error:
        raise ValueError("%s field definitions: %s" % (where, error)) from error


@timed("database_build.walked")
def walked(value):
    if isinstance(value, Part):
        return [value]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Held):
        return [part for item in value.values() for part in walked(item)]
    if isinstance(value, (list, tuple)):
        return [part for item in value for part in walked(item)]
    return []


@timed("database_build.listed")
def listed(key):
    return [key] if type(key) is str else list(key)


class Expansion:
    @timed("database_build.Expansion")
    def __init__(self, database, label, statements, sources):
        self.database, self.label, self.fields = database, label, stated(database)
        self.where = "declaration %s" % label
        self.reader = self.fields["reader"]
        self.catalogue = self.read(sources)
        self.read_tables = self.chosen()
        self.overrides, self.referenced = {}, set()
        for name, statement in statements:
            for held in walked(dict(statement.fields)) + walked([statement.table]):
                if owner(held)[0] is database:
                    self.referenced.add(owner(held)[1])
            if statement.word not in ("tasks", "decisions") and isinstance(statement.table, Part) \
                    and owner(statement.table)[0] is database:
                self.overrides.setdefault(owner(statement.table)[1], []).append((name, statement))
        unread = sorted(self.referenced - set(self.read_tables))
        if unread:
            raise ValueError("%s: a statement names the table %s, which the database does not read"
                             % (self.where, ", ".join(unread)))
        self.columns = self.typed_columns()
        self.sources = {table: self.source(table, sources) for table in self.read_tables}

    @timed("database_build.Expansion.read")
    def read(self, sources):
        reader = READERS.named(self.reader)
        if "definitions" in self.fields:
            path = Path(self.fields["definitions"])
            return definitions(path if path.is_absolute() else Path(sources) / path, self.where)
        if not hasattr(reader, "catalogue") or self.reader == "layout":
            raise ValueError("%s: the reader %s declares no catalogue of tables and keys; state definitions"
                             % (self.where, self.reader))
        return reader.catalogue(self.placed(sources))

    @timed("database_build.Expansion.placed")
    def placed(self, sources):
        place = self.fields["place"]
        if getattr(READERS.named(self.reader), "file", True) and not Path(place).is_absolute():
            return (Path(sources) / place).resolve()
        return place

    @timed("database_build.Expansion.chosen")
    def chosen(self):
        found = []
        for pattern in self.fields["tables"]:
            matched = [name for name in self.catalogue if fnmatchcase(name, pattern)]
            if not matched:
                raise ValueError("%s field tables names %s, which no table of the database matches" % (self.where, pattern))
            found += [name for name in matched if name not in found]
        return found

    @timed("database_build.Expansion.typed_columns")
    def typed_columns(self):
        leave, outcomes = list(self.fields.get("leave", ())), dict(self.fields.get("outcomes", {}))
        unit, missing, graph = self.fields.get("unit"), self.fields["missing"], self.fields.get("graph")
        used, result = set(), {}
        for table in self.read_tables:
            entry = self.catalogue[table]
            keys = set(entry["primary_key"])
            foreign = {column for key in entry["foreign_keys"] for column in key["columns"]}
            columns = {}
            for column, described in entry["columns"].items():
                left = [pattern for pattern in leave if fnmatchcase("%s.%s" % (table, column), pattern)]
                used.update(left)
                if left:
                    if column in keys:
                        raise ValueError("%s field leave names %s.%s, a column of the primary key"
                                         % (self.where, table, column))
                    continue
                if column in keys or column in foreign:
                    given = "identifier"
                else:
                    given = typed(described["type"], self.reader, unit)
                    if given is None:
                        raise ValueError("%s: the column %s.%s declares the type %r, which the reader %s proposes no "
                                         "type for; leave it" % (self.where, table, column, described["type"], self.reader))
                given = dict(type=given) if type(given) is str else dict(given)
                if given["type"] == "time" and given["unit"] is None:
                    raise ValueError("%s lacks unit, which the column of time %s.%s needs" % (self.where, table, column))
                outcome = outcomes.pop("%s.%s" % (table, column), None)
                if outcome is not None:
                    given["outcome"] = outcome
                if described["nullable"] and column not in keys:
                    plain = given["type"] in FEATURED and column not in foreign and column != graph and outcome is None
                    given["missing"] = missing if plain else "omit"
                columns[column] = given
            for key in entry["foreign_keys"]:
                kept = [column for column in key["columns"] if column in columns]
                if kept and len(kept) != len(key["columns"]):
                    raise ValueError("%s field leave names some columns of the foreign key %s(%s); leave all or none"
                                     % (self.where, table, ", ".join(key["columns"])))
            result[table] = columns
        unused = [pattern for pattern in leave if pattern not in used]
        if unused or outcomes:
            raise ValueError("%s names columns no table it reads holds: %s"
                             % (self.where, ", ".join(sorted(unused + list(outcomes)))))
        return result

    @timed("database_build.Expansion.source")
    def source(self, table, sources):
        reader = READERS.named(self.reader)
        through = self.fields.get("through")
        if hasattr(reader, "catalogue"):
            return Source(self.fields["place"], self.reader, self.columns[table], through, table=table)
        folder = Path(self.fields["place"])
        folder = folder if folder.is_absolute() else Path(sources) / folder
        files = [path for path in sorted(folder.iterdir()) if path.name.split(".")[0] == table] if folder.is_dir() else []
        if len(files) != 1:
            raise ValueError("%s: the folder %s holds %d files of the table %s, not one"
                             % (self.where, folder, len(files), table))
        return Source(files[0], self.reader, self.columns[table], through)

    @timed("database_build.Expansion.foreign")
    def foreign(self, table):
        entry, result = self.catalogue[table], []
        for key in entry["foreign_keys"]:
            if key["columns"][0] not in self.columns[table]:
                continue
            target = key["table"]
            if target not in self.read_tables:
                raise ValueError("%s: the foreign key %s(%s) names the table %s, which the database does not read; "
                                 "read it or leave the columns" % (self.where, table, ", ".join(key["columns"]), target))
            if list(key["references"]) != list(self.catalogue[target]["primary_key"]):
                raise ValueError("%s: the foreign key %s(%s) names columns of %s other than its primary key"
                                 % (self.where, table, ", ".join(key["columns"]), target))
            result.append(dict(name="_".join(key["columns"]), columns=list(key["columns"]), table=target))
        return result

    @timed("database_build.Expansion.timing")
    def timing(self, table):
        if self.fields["times"] is None:
            return {}
        name, parameters = ruled(self.fields["times"], self.where + " field times")
        columns = {column: given for column, given in self.columns[table].items() if "outcome" not in given}
        try:
            result = TIMES.named(name)(table, columns, parameters)
        except (KeyError, ValueError) as error:
            raise ValueError("%s field times: %s" % (self.where, error)) from error
        if not isinstance(result, Held) or set(result) != {"holds", "known"}:
            raise ValueError("%s field times: the rule %s gives holds and known" % (self.where, name))
        return dict(result)

    @timed("database_build.Expansion.features")
    def features(self, table, taken):
        encoders, result = self.fields["encoders"], {}
        for column, given in self.columns[table].items():
            if column in taken or given["type"] not in FEATURED or "outcome" in given or column == self.fields.get("graph"):
                continue
            if given["type"] not in encoders:
                raise ValueError("%s field encoders states no encoder of the type %s, which the column %s.%s has"
                                 % (self.where, given["type"], table, column))
            result[column] = encoders[given["type"]]
        return result

    @timed("database_build.Expansion.common")
    def common(self, table):
        fields = self.timing(table)
        if self.fields.get("graph") in self.columns[table]:
            fields["graph"] = self.fields["graph"]
        return fields

    @timed("database_build.Expansion.ended")
    def ended(self, key):
        return dict(of=part(self.database, key["table"]), key=key["columns"][0] if len(key["columns"]) == 1
                    else key["columns"], absent=self.fields["absent"])

    @timed("database_build.Expansion.made")
    def made(self, table):
        entry, held = self.catalogue[table], part(self.database, table)
        if entry["kind"] == "view":
            return None
        keys, foreign = list(entry["primary_key"]), self.foreign(table)
        if not keys:
            if table in self.referenced:
                return None
            raise ValueError("%s: the table %s declares no primary key; map it by a statement or read it from none"
                             % (self.where, table))
        within = [key for key in foreign if set(key["columns"]) <= set(keys)]
        if set(keys) <= {column for key in within for column in key["columns"]} and len(within) > 1:
            if len(within) != 2 or len(foreign) != 2:
                raise ValueError("%s: the table %s has a key made of foreign keys, and holds %d foreign keys; an edge "
                                 "joins two, so map it by a statement" % (self.where, table, len(foreign)))
            first, second = sorted(within, key=lambda key: keys.index(key["columns"][0]))
            return edges(held, source=self.ended(first), target=self.ended(second), directed=True,
                         features=self.features(table, set(keys)), **self.common(table))
        taken = set(keys) | {column for key in foreign for column in key["columns"]}
        links = {key["name"]: self.ended(key) for key in foreign}
        fields = self.common(table)
        if links:
            fields["links"] = links
        return nodes(held, key=keys[0] if len(keys) == 1 else keys, features=self.features(table, taken), **fields)

    @timed("database_build.Expansion.kept")
    def kept(self, table, label, statement):
        fields = statement.fields
        ends = [fields[name] for name in ("source", "target") if name in fields]
        ends += [link for link in (fields.get("links") or {}).values() if isinstance(link, Held)]
        ends += [member["node"] for member in fields.get("members") or ()]
        result = []
        for key in self.foreign(table):
            if any(isinstance(end["of"], Part) and owner(end["of"])[1] == key["table"] and listed(end["key"]) == key["columns"]
                   for end in ends):
                continue
            if statement.word != "nodes" or "key" not in fields:
                raise ValueError("declaration %s maps the table %s and uses not its foreign key %s(%s) as an end; "
                                 "only nodes that state key keep it, so leave its columns"
                                 % (label, table, table, ", ".join(key["columns"])))
            if isinstance(fields.get("holds"), Held) and fields["holds"]["name"] == "change":
                raise ValueError("declaration %s maps the table %s in the form change and keeps its foreign key %s; "
                                 "state it under links" % (label, table, key["name"]))
            present = " and ".join("%s is not missing" % column for column in key["columns"])
            where = present if fields.get("where") is None else "(%s) and %s" % (fields["where"], present)
            kept = {name: fields[name] for name in ("graph", "holds", "known", "replaced", "seen", "ties") if name in fields}
            source = dict(of=fields.get("of", statement), key=fields["key"], absent="stop")
            result.append(("%s.%s" % (label, key["name"]), edges(
                part(self.database, table), where=where, source=source, target=self.ended(key), directed=True, **kept)))
        return result

    @timed("database_build.Expansion.expand")
    def expand(self):
        generated, represented, after = [], {}, {}
        for table in self.read_tables:
            if table in self.overrides:
                held = self.overrides[table]
                represented[table] = held[0][1] if len(held) == 1 else None
                for label, statement in held:
                    after[id(statement)] = self.kept(table, label, statement)
                continue
            made = self.made(table)
            if made is not None:
                generated.append((table, made))
                represented[table] = made
        return generated, represented, after


@timed("database_build.expanded")
def expanded(declarations, statements, sources):
    started = time.perf_counter()
    result, represented, aliases, follows = [], {}, {}, {}
    for label, value in declarations.items():
        if isinstance(value, Database):
            expansion = Expansion(value, label, statements, sources)
            generated, made, after = expansion.expand()
            result.append((label, generated))
            follows.update(after)
            for table, statement in made.items():
                represented[id(part(value, table))] = (table, statement)
            for name, source in expansion.sources.items():
                aliases[id(part(value, name))] = ("%s.%s" % (label, name), source)
    LOG.debug("expanded exit databases=%d seconds=%.6f", len(result), time.perf_counter() - started)
    return dict(result), represented, aliases, follows
