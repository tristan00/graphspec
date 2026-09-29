import copy
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.files import sha256_json

LOG = logging.getLogger(__name__)
VERSION = 1
EDGE = "edge_"
OBSERVED = "observed_"
QUERY = "query_"
FACTS = ("seq", "step", "present")
TABLES = ("graph", "observation", "query")
PREFIXES = (EDGE, OBSERVED, QUERY)
LEVELS = ("node", "query", "graph", "path")
FIELDS = ("version", "time", "columns", "graph", "nodes", "edges", "queries")
DECLARED = "declared"
ASKED = ("key", "grouping", "columns", "references", "signals")
GIVEN = ("id", "graph", "step", "category", "signal", "signal_observed", "y", "tau", "series")
VALUE = "column:"
ENDS = ("from", "to")
REFERENCES = 3


class Names:
    def __init__(self, label):
        self.label, self.held = label, {}

    def register(self, name, entry):
        started = time.perf_counter()
        if type(name) is not str or not name:
            raise ValueError("a %s is registered under a name, found %r" % (self.label, name))
        if name in self.held:
            raise ValueError("the %s %s is registered twice" % (self.label, name))
        self.held[name] = entry
        LOG.debug(
            "register exit label=%s name=%s seconds=%.4f", self.label, name, time.perf_counter() - started
        )
        return entry

    def named(self, name):
        if type(name) is not str or name not in self.held:
            raise KeyError("no %s is registered under the name %r" % (self.label, name))
        return self.held[name]

    def names(self):
        return tuple(self.held)


TYPES = Names("column type")
ENCODERS = Names("encoder")


def absent(column):
    return np.asarray(column.is_null().to_numpy(zero_copy_only=False), bool)


class Number:
    accepts = ("FLOAT",)
    written = pa.float32()
    symbols = False
    ordered = True

    def read(self, column):
        return np.asarray(column.to_numpy(zero_copy_only=False), np.float32)

    def missing(self, stated):
        return np.isnan(stated)

    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else float(value) for value in np.asarray(codes, np.float32)]


class Flag:
    accepts = ("BOOL", "BOOLEAN")
    written = pa.bool_()
    symbols = False
    ordered = False

    def read(self, column):
        values = np.asarray(column.fill_null(False).to_numpy(zero_copy_only=False), np.float32)
        values[absent(column)] = np.nan
        return values

    def missing(self, stated):
        return np.isnan(stated)

    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else bool(value) for value in np.asarray(codes, np.float32)]


class Category:
    accepts = ("STRING",)
    written = pa.string()
    symbols = True
    ordered = False

    def read(self, column):
        return np.asarray(column.to_pylist(), object)

    def missing(self, stated):
        return np.asarray([value is None for value in stated], bool)

    def codes(self, stated, dictionary):
        position = {value: index for index, value in enumerate(dictionary)}
        unknown = sorted({value for value in stated if value is not None and value not in position})
        if unknown:
            raise ValueError("values outside the dictionary of their column: " + ", ".join(unknown))
        return np.asarray([np.nan if value is None else position[value] for value in stated], np.float32)

    def stated(self, codes, dictionary):
        return [
            None if np.isnan(value) else dictionary[int(value)] for value in np.asarray(codes, np.float32)
        ]


def parted(codes):
    codes = np.asarray(codes, np.float32)
    known = ~np.isnan(codes)
    return codes, known


class Value:
    types = ("number", "flag")

    def fit(self, codes, dictionary, column):
        return {}

    def widths(self, state):
        return 1, 1

    def encode(self, codes, state):
        codes, known = parted(codes)
        return np.where(known, codes, 0)[:, None].astype(np.float32), known[:, None].astype(np.float32)


class Logarithm:
    types = ("number",)

    def fit(self, codes, dictionary, column):
        return {}

    def widths(self, state):
        return 1, 1

    def encode(self, codes, state):
        codes, known = parted(codes)
        held = np.where(known, codes, 0)
        return (np.sign(held) * np.log1p(np.abs(held)))[:, None].astype(np.float32), known[:, None].astype(
            np.float32
        )


class Standard:
    types = ("number",)

    def fit(self, codes, dictionary, column):
        values = np.asarray(codes, np.float64)
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError(
                "the column %s has fewer than two differing values in the rows trained on and cannot be standardised"
                % column
            )
        return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)))

    def widths(self, state):
        return 1, 1

    def encode(self, codes, state):
        codes, known = parted(codes)
        scaled = (np.where(known, codes, state["mean"]).astype(np.float64) - state["mean"]) / state["sd"]
        return scaled[:, None].astype(np.float32), known[:, None].astype(np.float32)


class OneHot:
    types = ("category",)

    def fit(self, codes, dictionary, column):
        found = np.unique(np.asarray(codes, np.int64))
        if not len(found):
            raise ValueError("the column %s has no value in the rows trained on" % column)
        return dict(codes=found.tolist(), values=[dictionary[code] for code in found.tolist()])

    def widths(self, state):
        return len(state["codes"]) + 1, 1

    def encode(self, codes, state):
        codes, known = parted(codes)
        seen = np.asarray(state["codes"], np.int64)
        held = np.where(known, codes, -1).astype(np.int64)
        position = np.minimum(np.searchsorted(seen, held), len(seen) - 1)
        position = np.where(seen[position] == held, position, len(seen))
        values = np.zeros((len(codes), len(seen) + 1), np.float32)
        rows = np.flatnonzero(known)
        values[rows, position[rows]] = 1
        return values, known[:, None].astype(np.float32)


TYPES.register("number", Number())
TYPES.register("flag", Flag())
TYPES.register("category", Category())
ENCODERS.register("value", Value())
ENCODERS.register("logarithm", Logarithm())
ENCODERS.register("standard", Standard())
ENCODERS.register("onehot", OneHot())


def fields(mapping, names, where):
    if not isinstance(mapping, dict):
        raise ValueError("%s is stated as a mapping, found %r" % (where, mapping))
    lacking = [name for name in names if name not in mapping]
    if lacking:
        raise ValueError("%s lacks %s" % (where, ", ".join(lacking)))
    unknown = sorted(set(mapping) - set(names))
    if unknown:
        raise ValueError("%s holds fields it does not know: %s" % (where, ", ".join(unknown)))
    return mapping


def named(name, where):
    if type(name) is not str or not name or ":" in name:
        raise ValueError("%s needs a name without colons, found %r" % (where, name))
    return name


def listed(names, known, where):
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ValueError("the columns of %s are a list of distinct names" % where)
    unstated = [name for name in names if name not in known]
    if unstated:
        raise ValueError("the columns of %s have no stated type: %s" % (where, ", ".join(map(str, unstated))))
    return tuple(names)


class Schema:
    def __init__(self, mapping):
        started = time.perf_counter()
        held = copy.deepcopy(fields(mapping, FIELDS, "the schema"))
        if type(held["version"]) is not int or held["version"] != VERSION:
            raise ValueError(
                "the schema has version %r; this code reads version %d" % (held["version"], VERSION)
            )
        unit = fields(held["time"], ("unit",), "the time of the schema")["unit"]
        if type(unit) is not str or not unit:
            raise ValueError("the unit of time of the schema is a name, found %r" % (unit,))
        self.version, self.unit = held["version"], unit
        if not isinstance(held["columns"], dict):
            raise ValueError("the columns of the schema are a mapping from name to type and encoder")
        self.columns = {}
        for name, entry in held["columns"].items():
            where = "the column " + named(name, "a column")
            entry = fields(entry, ("type", "encoder"), where)
            TYPES.named(entry["type"])
            if entry["type"] not in ENCODERS.named(entry["encoder"]).types:
                raise ValueError(
                    "%s has the type %s, which the encoder %s does not read"
                    % (where, entry["type"], entry["encoder"])
                )
            self.columns[name] = dict(type=entry["type"], encoder=entry["encoder"])
        self.graph = listed(
            fields(held["graph"], ("columns",), "the graph of the schema")["columns"],
            self.columns,
            "the graph",
        )
        if not isinstance(held["nodes"], dict) or not held["nodes"]:
            raise ValueError("the schema states at least one kind of node")
        self.nodes = {}
        for kind, entry in held["nodes"].items():
            where = "the kind of node " + named(kind, "a kind of node")
            if kind in TABLES or kind == "step" or kind.startswith(PREFIXES):
                raise ValueError("%s collides with the name of a table of the contract" % where)
            self.nodes[kind] = listed(fields(entry, ("columns",), where)["columns"], self.columns, where)
        if not isinstance(held["edges"], dict):
            raise ValueError("the edges of the schema are a mapping from table to statement")
        self.edges = {}
        for table, entry in held["edges"].items():
            where = "the edge table " + named(table, "an edge table")
            entry = fields(entry, ("endpoints", "directed", "columns", "relation"), where)
            pairs = entry["endpoints"]
            if (
                not isinstance(pairs, list)
                or not pairs
                or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
                or len({tuple(pair) for pair in pairs}) != len(pairs)
            ):
                raise ValueError("%s states its endpoint kinds as distinct pairs" % where)
            strangers = sorted({kind for pair in pairs for kind in pair} - set(self.nodes))
            if strangers:
                raise ValueError("%s joins kinds of node not stated: %s" % (where, ", ".join(strangers)))
            if type(entry["directed"]) is not bool:
                raise ValueError("%s states whether it has a direction as true or false" % where)
            relation = entry["relation"]
            if relation != DECLARED:
                relation = fields(relation, ("column",), "the relation of " + where)["column"]
                if named(relation, "the relation column of " + where) in FACTS + ENDS:
                    raise ValueError("the relation column of %s collides with a column of facts" % where)
            columns = listed(entry["columns"], self.columns, where)
            if relation in columns:
                raise ValueError("the relation column of %s is also one of its columns" % where)
            self.edges[table] = dict(
                endpoints=tuple((source, target) for source, target in pairs),
                directed=entry["directed"],
                columns=columns,
                column=None if relation == DECLARED else relation,
            )
        self.queries = None
        if held["queries"] is not None:
            entry = fields(held["queries"], ASKED, "the queries")
            references = entry["references"]
            if not isinstance(references, dict) or len(references) != REFERENCES:
                raise ValueError(
                    "the queries state %d named references until tasks replace them, found %r"
                    % (REFERENCES, references)
                )
            for name, kinds in references.items():
                where = "the reference " + named(name, "a reference")
                if not isinstance(kinds, list) or not kinds or not set(kinds) <= set(self.nodes):
                    raise ValueError("%s names kinds of node that are stated" % where)
            if type(entry["signals"]) is not int or entry["signals"] < 0:
                raise ValueError("the queries state the width of their signals as a whole number")
            for part in ("key", "grouping"):
                if named(entry[part], "the %s of the queries" % part) in GIVEN:
                    raise ValueError(
                        "the %s of the queries collides with a column of the contract: %s"
                        % (part, entry[part])
                    )
            if entry["key"] == entry["grouping"]:
                raise ValueError("the key and the grouping of the queries are two columns")
            self.queries = dict(
                key=entry["key"],
                grouping=entry["grouping"],
                columns=listed(entry["columns"], self.columns, "the queries"),
                references={name: tuple(kinds) for name, kinds in references.items()},
                signals=entry["signals"],
            )
        LOG.debug(
            "schema exit kinds=%d edges=%d columns=%d seconds=%.4f",
            len(self.nodes),
            len(self.edges),
            len(self.columns),
            time.perf_counter() - started,
        )

    def mapping(self):
        started = time.perf_counter()
        result = dict(
            version=self.version,
            time=dict(unit=self.unit),
            columns={name: dict(entry) for name, entry in self.columns.items()},
            graph=dict(columns=list(self.graph)),
            nodes={kind: dict(columns=list(columns)) for kind, columns in self.nodes.items()},
            edges={
                table: dict(
                    endpoints=[list(pair) for pair in entry["endpoints"]],
                    directed=entry["directed"],
                    columns=list(entry["columns"]),
                    relation=DECLARED if entry["column"] is None else dict(column=entry["column"]),
                )
                for table, entry in self.edges.items()
            },
            queries=None
            if self.queries is None
            else dict(
                key=self.queries["key"],
                grouping=self.queries["grouping"],
                columns=list(self.queries["columns"]),
                references={name: list(kinds) for name, kinds in self.queries["references"].items()},
                signals=self.queries["signals"],
            ),
        )
        LOG.debug("schema mapping exit seconds=%.4f", time.perf_counter() - started)
        return result

    def fingerprint(self):
        return sha256_json(self.mapping())

    def layout(self):
        groups = list(self.nodes.values()) + ([] if self.queries is None else [self.queries["columns"]])
        return tuple(dict.fromkeys(name for columns in groups for name in columns))

    def kind(self, name):
        return TYPES.named(self.columns[name]["type"])

    def encoder(self, name):
        return ENCODERS.named(self.columns[name]["encoder"])

    def symbolic(self):
        return tuple(name for name in self.columns if self.kind(name).symbols)

    def tables(self):
        started = time.perf_counter()
        keys = dict(id="INT64", graph="INT32")
        result = {"graph": dict(kind="NODE", required=dict(id="INT32", name="STRING"), columns=self.graph)}
        result["observation"] = dict(
            kind="NODE", required=dict(id="INT64", seq="INT32", step="INT32"), columns=()
        )
        for kind, columns in self.nodes.items():
            result[kind] = dict(kind="NODE", required=dict(keys, entity="INT32", key="STRING"), columns=())
            result[OBSERVED + kind] = dict(
                kind="REL", required={}, columns=columns, pairs=((kind, "observation"),)
            )
        for table, entry in self.edges.items():
            required = dict(seq="INT32", step="INT32", present=("BOOL", "BOOLEAN"))
            if entry["column"] is not None:
                required[entry["column"]] = "STRING"
            result[EDGE + table] = dict(
                kind="REL", required=required, columns=entry["columns"], pairs=entry["endpoints"]
            )
        if self.queries is not None:
            required = dict(keys, step="INT32", category="STRING")
            required.update({self.queries["grouping"]: "INT32", self.queries["key"]: "INT64"})
            if self.queries["signals"]:
                vector = "FLOAT[%d]" % self.queries["signals"]
                required.update(signal=vector, signal_observed=vector)
            result["query"] = dict(kind="NODE", required=required, columns=self.queries["columns"])
            for name, kinds in self.queries["references"].items():
                result[QUERY + name] = dict(
                    kind="REL", required={}, columns=(), pairs=tuple(("query", kind) for kind in kinds)
                )
        LOG.debug("schema tables exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def conform(self, source):
        started = time.perf_counter()
        expected = self.tables()
        for table, entry in expected.items():
            if source.kinds.get(table) != entry["kind"]:
                raise ValueError(
                    "the schema states a %s table named %s, which the source does not hold"
                    % (entry["kind"].lower(), table)
                )
            found = source.columns(table)
            wanted = dict(entry["required"])
            for name in entry["columns"]:
                if name in wanted or name in ENDS:
                    raise ValueError(
                        "the column %s of %s collides with a column of the contract" % (name, table)
                    )
                wanted[name] = self.kind(name).accepts
            lacking = [name for name in wanted if name not in found]
            if lacking:
                raise ValueError("%s lacks the columns %s" % (table, ", ".join(lacking)))
            unstated = [name for name in found if name not in wanted]
            if unstated:
                raise ValueError("the columns of %s have no stated type: %s" % (table, ", ".join(unstated)))
            for name, accepted in wanted.items():
                accepted = (accepted,) if type(accepted) is str else tuple(accepted)
                if found[name] not in accepted:
                    raise ValueError(
                        "the column %s of %s is stored as %s and is stated as %s"
                        % (name, table, found[name], " or ".join(accepted))
                    )
            if entry["kind"] == "REL":
                pairs = set(source.pairs(table))
                if pairs != set(entry["pairs"]):
                    raise ValueError(
                        "%s joins %s and is stated to join %s"
                        % (
                            table,
                            ", ".join("%s to %s" % pair for pair in sorted(pairs)),
                            ", ".join("%s to %s" % pair for pair in sorted(entry["pairs"])),
                        )
                    )
        strangers = sorted(
            table
            for table in source.kinds
            if table not in expected and (table in TABLES or table.startswith(PREFIXES))
        )
        if strangers:
            raise ValueError("the source holds tables the schema does not state: " + ", ".join(strangers))
        LOG.debug("schema conform exit tables=%d seconds=%.4f", len(expected), time.perf_counter() - started)

    def text(self, encoding=None):
        started = time.perf_counter()

        def described(columns):
            if not columns:
                return "no columns"
            return ", ".join(
                "%s %s by %s" % (name, self.columns[name]["type"], self.columns[name]["encoder"])
                for name in columns
            )

        def wide(columns):
            return "" if encoding is None else ", width %d" % encoding.width(columns)

        lines = ["schema version %d" % self.version, "unit of time: %s" % self.unit]
        lines.append("graph: %s" % described(self.graph))
        lines.append("kinds of node:")
        lines += [
            "  %s: %s%s" % (kind, described(columns), wide(columns)) for kind, columns in self.nodes.items()
        ]
        lines.append("edge tables:")
        for table, entry in self.edges.items():
            lines.append(
                "  %s: %s, %s, relation %s, %s%s"
                % (
                    table,
                    "; ".join("%s to %s" % pair for pair in entry["endpoints"]),
                    "directed" if entry["directed"] else "without direction",
                    "declared" if entry["column"] is None else "taken from the column " + entry["column"],
                    described(entry["columns"]),
                    wide(entry["columns"]),
                )
            )
        if self.queries is None:
            lines.append("queries: none")
        else:
            lines.append(
                "queries: keyed by %s, grouped by %s, %s%s, signals of width %d"
                % (
                    self.queries["key"],
                    self.queries["grouping"],
                    described(self.queries["columns"]),
                    wide(self.queries["columns"]),
                    self.queries["signals"],
                )
            )
            lines += [
                "  reference %s: %s" % (name, ", ".join(kinds))
                for name, kinds in self.queries["references"].items()
            ]
        if encoding is not None:
            lines.append("relations:")
            lines += [
                "  %s: from %s" % (name, table)
                for table, names in encoding.relations.items()
                for name in names
            ]
        LOG.debug("schema text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
        return "\n".join(lines)


class Encoding:
    def __init__(self, schema, dictionaries, states, relations):
        started = time.perf_counter()
        self.schema, self.dictionaries, self.states = schema, dictionaries, states
        self.relations = {table: tuple(names) for table, names in relations.items()}
        for name in schema.symbolic():
            if name not in dictionaries:
                raise ValueError("the column %s has no dictionary of its values" % name)
        lacking = [name for name in schema.columns if name not in states]
        if lacking:
            raise ValueError("the columns %s have no fitted encoder" % ", ".join(lacking))
        lacking = [table for table in schema.edges if table not in self.relations]
        if lacking:
            raise ValueError("the edge tables %s have no relations" % ", ".join(lacking))
        self.layout = schema.layout()
        self.sizes = {name: schema.encoder(name).widths(states[name]) for name in schema.columns}
        self.values = sum(self.sizes[name][0] for name in self.layout)
        self.features = self.values + sum(self.sizes[name][1] for name in self.layout)
        LOG.debug(
            "encoding exit columns=%d features=%d seconds=%.4f",
            len(self.layout),
            self.features,
            time.perf_counter() - started,
        )

    def declaration(self):
        return dict(
            dictionaries=self.dictionaries,
            states=self.states,
            relations={table: list(names) for table, names in self.relations.items()},
        )

    def width(self, columns):
        return sum(sum(self.sizes[name]) for name in columns)

    def positions(self, columns):
        return [self.layout.index(name) for name in columns]

    def encoded(self, columns, codes):
        parts = [
            self.schema.encoder(name).encode(codes[:, position], self.states[name])
            for position, name in enumerate(columns)
        ]
        return [values for values, _ in parts], [flags for _, flags in parts]

    def nodes(self, codes):
        started = time.perf_counter()
        codes = np.asarray(codes, np.float32).reshape(-1, len(self.layout))
        values, flags = self.encoded(self.layout, codes)
        result = np.concatenate(values + flags + [np.zeros((len(codes), 0), np.float32)], 1)
        LOG.debug("encoding nodes exit rows=%d seconds=%.4f", len(codes), time.perf_counter() - started)
        return result

    def edges(self, table, codes):
        columns = self.schema.edges[table]["columns"]
        codes = np.asarray(codes, np.float32).reshape(-1, len(columns))
        values, flags = self.encoded(columns, codes)
        return np.concatenate(values + flags + [np.zeros((len(codes), 0), np.float32)], 1)


def fitted(schema, gathered, relations):
    started = time.perf_counter()
    dictionaries, states = {}, {}
    for name in schema.columns:
        kind, found = schema.kind(name), gathered[name]
        trained = found["trained"]
        dictionary = sorted(set(found["distinct"])) if kind.symbols else None
        if kind.symbols:
            dictionaries[name] = dictionary
        if not len(trained):
            raise ValueError("the column %s has no value in the rows trained on" % name)
        states[name] = schema.encoder(name).fit(kind.codes(trained, dictionary), dictionary, name)
    result = Encoding(schema, dictionaries, states, relations)
    LOG.info(
        "encoders fitted exit columns=%d features=%d seconds=%.3f",
        len(states),
        result.features,
        time.perf_counter() - started,
    )
    return result


class Tables:
    def destinations(self, table):
        return tuple(dict.fromkeys(target for _, target in self.pairs(table)))

    def target(self, name):
        started = time.perf_counter()
        if self.kinds.get(name) != "NODE" or self.kinds.get(name + "_of") != "REL":
            raise ValueError("graph source requires target table %s with a %s_of relationship" % (name, name))
        if any(source != name for source, _ in self.pairs(name + "_of")):
            raise ValueError("%s_of relationships must start at %s" % (name, name))
        destinations = set(self.destinations(name + "_of"))
        typed = self.columns(name)
        LOG.debug("target exit table=%s seconds=%.4f", name, time.perf_counter() - started)
        if destinations == {"query"}:
            if "tau" not in typed:
                return "query"
            if typed["tau"] not in ("FLOAT", "DOUBLE"):
                raise ValueError("%s.tau must be a FLOAT or DOUBLE column, found %s" % (name, typed["tau"]))
            if typed.get("series") != "STRING":
                raise ValueError("%s is a path target and requires a series STRING column" % name)
            return "path"
        if "tau" in typed or "series" in typed:
            raise ValueError("%s carries tau or series columns, which only path targets may have" % name)
        if destinations == {"graph"}:
            return "graph"
        if not destinations or not destinations <= set(self.nodes):
            raise ValueError("%s_of must point to query, graph, or observed node types" % name)
        if typed.get("step") != "INT32":
            raise ValueError("%s labels node types and requires a step INT32 column" % name)
        return "node"
