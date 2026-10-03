import copy
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.files import sha256_json
from graphspec.infra.timing import timed
from graphspec.registries import Names

from .references import ABSENT, SORTS
from .targets import BETTER, TARGETS, UTILITIES, UTILITY
from .time import ABOUT, MODELS, STAMPS, TIES, UNITED

LOG = logging.getLogger(__name__)
VERSION = 6
EDGE = "edge_"
OBSERVED = "observed_"
TASK = "task_"
MEMBER = "member_"
INNER = "held_"
WITHIN = "held_edge_"
TARGET = "target"
LINK = "target_of"
FACTS = ("present",)
STAMP = "INT64"
SEQUENCE = dict(seq="INT32")
WATCHERS = "STRING[]"
WRITTEN = {
    STAMP: pa.int64(),
    "INT32": pa.int32(),
    "FLOAT": pa.float32(),
    "DOUBLE": pa.float64(),
    "BOOL": pa.bool_(),
    "STRING": pa.string(),
    WATCHERS: pa.list_(pa.string()),
}
NAMED = {kind: name for name, kind in WRITTEN.items()}
TABLES = ("graph", "observation", "task", TARGET, LINK)
PREFIXES = (EDGE, OBSERVED, TASK, MEMBER, INNER)
FIELDS =("version", "time", "observers", "columns", "graph", "nodes", "sets", "edges", "tasks")
TIMED = ("model", "units", "ties")
KINDED = ("columns", "holds")
HOLDS = ("nodes", "edges", "encoder")
INWARD = ("endpoints", "directed", "columns")
SETS = ("kinds", "roles", "members", "columns")
EDGED = ("endpoints", "directed", "columns", "relation", "about")
OWNED = dict(id="INT64", graph="INT32", owner="STRING", kind="STRING", key="STRING")
DECLARED = "declared"
ASKED = ("key", "grouping", "signals", "references", "kinds")
REFERRED = ("sort", "kinds")
KIND = ("references", "columns", "target", "weight", "absent", "event", "matures")
AIMED = ("kind", "series", "about", "utility")
OWNED_TARGET = ("id", "y", "known", "task", "owner")
MATURED = ("reference", "column")
GIVEN = ("id", "graph", "time", "observer", "kind", "split", "weight", "signal", "signal_observed")
VALUE = "column:"
ENDS = ("from", "to")
FITTED = ("sizes", "values", "held", "inner", "features")


TYPES = Names("column type")
ENCODERS = Names("encoder")


@timed("schema.absent")
def absent(column):
    return np.asarray(column.is_null().to_numpy(zero_copy_only=False), bool)


def typename(kind):
    if pa.types.is_fixed_size_list(kind) and kind.value_type == pa.float32():
        return "FLOAT[%d]" % kind.list_size
    if pa.types.is_list(kind) and kind.value_type == pa.string():
        return WATCHERS
    if kind not in NAMED:
        raise ValueError("unsupported graph property type: " + str(kind))
    return NAMED[kind]


class Number:
    accepts = ("FLOAT",)
    written = pa.float32()
    symbols = False
    ordered = True

    @timed("schema.Number.read")
    def read(self, column):
        return np.asarray(column.to_numpy(zero_copy_only=False), np.float32)

    @timed("schema.Number.missing")
    def missing(self, stated):
        return np.isnan(stated)

    @timed("schema.Number.codes")
    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    @timed("schema.Number.stated")
    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else float(value) for value in np.asarray(codes, np.float32)]


class Flag:
    accepts = ("BOOL", "BOOLEAN")
    written = pa.bool_()
    symbols = False
    ordered = False

    @timed("schema.Flag.read")
    def read(self, column):
        values = np.asarray(column.fill_null(False).to_numpy(zero_copy_only=False), np.float32)
        values[absent(column)] = np.nan
        return values

    @timed("schema.Flag.missing")
    def missing(self, stated):
        return np.isnan(stated)

    @timed("schema.Flag.codes")
    def codes(self, stated, dictionary):
        return np.asarray(stated, np.float32)

    @timed("schema.Flag.stated")
    def stated(self, codes, dictionary):
        return [None if np.isnan(value) else bool(value) for value in np.asarray(codes, np.float32)]


class Category:
    accepts = ("STRING",)
    written = pa.string()
    symbols = True
    ordered = False

    @timed("schema.Category.read")
    def read(self, column):
        return np.asarray(column.to_pylist(), object)

    @timed("schema.Category.missing")
    def missing(self, stated):
        return np.asarray([value is None for value in stated], bool)

    @timed("schema.Category.codes")
    def codes(self, stated, dictionary):
        position = {value: index for index, value in enumerate(dictionary)}
        unknown = sorted({value for value in stated if value is not None and value not in position})
        if unknown:
            raise ValueError("values outside the dictionary of their column: " + ", ".join(unknown))
        return np.asarray([np.nan if value is None else position[value] for value in stated], np.float32)

    @timed("schema.Category.stated")
    def stated(self, codes, dictionary):
        return [
            None if np.isnan(value) else dictionary[int(value)] for value in np.asarray(codes, np.float32)
        ]


@timed("schema.parted")
def parted(codes):
    codes = np.asarray(codes, np.float32)
    known = ~np.isnan(codes)
    return codes, known


class Value:
    types = ("number", "flag")

    @timed("schema.Value.fit")
    def fit(self, codes, dictionary, column):
        return {}

    @timed("schema.Value.widths")
    def widths(self, state):
        return 1, 1

    @timed("schema.Value.encode")
    def encode(self, codes, state, dictionary):
        codes, known = parted(codes)
        return np.where(known, codes, 0)[:, None].astype(np.float32), known[:, None].astype(np.float32)


class Logarithm:
    types = ("number",)

    @timed("schema.Logarithm.fit")
    def fit(self, codes, dictionary, column):
        return {}

    @timed("schema.Logarithm.widths")
    def widths(self, state):
        return 1, 1

    @timed("schema.Logarithm.encode")
    def encode(self, codes, state, dictionary):
        codes, known = parted(codes)
        held = np.where(known, codes, 0)
        return (np.sign(held) * np.log1p(np.abs(held)))[:, None].astype(np.float32), known[:, None].astype(
            np.float32
        )


class Standard:
    types = ("number",)

    @timed("schema.Standard.fit")
    def fit(self, codes, dictionary, column):
        values = np.asarray(codes, np.float64)
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError(
                "the column %s has fewer than two differing values in what the views of the splits trained on "
                "show, and cannot be standardised" % column
            )
        return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)))

    @timed("schema.Standard.widths")
    def widths(self, state):
        return 1, 1

    @timed("schema.Standard.encode")
    def encode(self, codes, state, dictionary):
        codes, known = parted(codes)
        scaled = (np.where(known, codes, state["mean"]).astype(np.float64) - state["mean"]) / state["sd"]
        return scaled[:, None].astype(np.float32), known[:, None].astype(np.float32)


class OneHot:
    types = ("category",)
    choices = ("stop", "shared")

    def __init__(self, unseen):
        started = time.perf_counter()
        if unseen not in self.choices:
            raise ValueError(
                "the encoder states what it does with a value unseen in training as one of %s, found %r"
                % (", ".join(self.choices), unseen)
            )
        self.unseen = unseen
        LOG.debug("onehot exit unseen=%s seconds=%.4f", unseen, time.perf_counter() - started)

    @timed("schema.OneHot.fit")
    def fit(self, codes, dictionary, column):
        return dict(values=[dictionary[code] for code in np.unique(np.asarray(codes, np.int64)).tolist()])

    @timed("schema.OneHot.widths")
    def widths(self, state):
        return len(state["values"]) + 1, 1

    @timed("schema.OneHot.encode")
    def encode(self, codes, state, dictionary):
        codes, known = parted(codes)
        place = {value: code for code, value in enumerate(dictionary)}
        seen = np.asarray([place[value] for value in state["values"]], np.int64)
        held = np.where(known, codes, -1).astype(np.int64)
        position = np.searchsorted(seen, held)
        inside = np.flatnonzero(position < len(seen))
        found = np.zeros(len(held), bool)
        found[inside] = seen[position[inside]] == held[inside]
        position = np.where(found, position, len(seen))
        strangers = known & ~found
        if self.unseen == "stop" and strangers.any():
            raise ValueError(
                "the encoder stops on a value unseen in training and met the values "
                + ", ".join(dictionary[code] for code in np.unique(held[strangers]).tolist())
            )
        values = np.zeros((len(codes), len(seen) + 1), np.float32)
        rows = np.flatnonzero(known)
        values[rows, position[rows]] = 1
        return values, known[:, None].astype(np.float32)


@timed("schema.inward")
def inward(held):
    return tuple(dict.fromkeys(name for columns in held["nodes"].values() for name in columns))


@timed("schema.outward")
def outward(held):
    return tuple(dict.fromkeys(name for entry in held["edges"].values() for name in entry["columns"]))


class Pooled:
    @timed("schema.pooled.width")
    def width(self, encoding, held):
        return encoding.width(inward(held)) + 1

    def encode(self, encoding, held, graphs):
        started = time.perf_counter()
        own = encoding.vectors(inward(held), graphs["values"])
        node = np.concatenate((own, np.ones((len(own), 1), np.float32)), 1)
        result = np.zeros((graphs["owners"], node.shape[1]), np.float32)
        np.add.at(result, graphs["owner"], node)
        LOG.debug(
            "pooled exit graphs=%d nodes=%d seconds=%.4f",
            graphs["owners"],
            len(own),
            time.perf_counter() - started,
        )
        return result


class Propagated:
    @timed("schema.propagated.width")
    def width(self, encoding, held):
        return 2 * encoding.width(inward(held)) + 1

    def encode(self, encoding, held, graphs):
        started = time.perf_counter()
        own = encoding.vectors(inward(held), graphs["values"])
        ends = np.r_[graphs["source"], graphs["target"]].astype(np.int64)
        other = np.r_[graphs["target"], graphs["source"]].astype(np.int64)
        summed = np.zeros_like(own)
        np.add.at(summed, ends, own[other])
        degree = np.bincount(ends, minlength=len(own)).astype(np.float32)
        mean = np.divide(summed, degree[:, None], out=np.zeros_like(summed), where=degree[:, None] > 0)
        node = np.concatenate((own, mean, np.ones((len(own), 1), np.float32)), 1)
        result = np.zeros((graphs["owners"], node.shape[1]), np.float32)
        np.add.at(result, graphs["owner"], node)
        LOG.debug(
            "propagated exit graphs=%d nodes=%d edges=%d seconds=%.4f",
            graphs["owners"],
            len(own),
            len(graphs["source"]),
            time.perf_counter() - started,
        )
        return result


HELD = Names("encoder of held graphs")
TYPES.register("number", Number())
TYPES.register("flag", Flag())
TYPES.register("category", Category())
ENCODERS.register("value", Value())
ENCODERS.register("logarithm", Logarithm())
ENCODERS.register("standard", Standard())
ENCODERS.register("onehot", OneHot("shared"))
HELD.register("pooled", Pooled())
HELD.register("propagated", Propagated())


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


@timed("schema.reserved")
def reserved(name, where):
    named(name, where)
    if name in TABLES or name == "step" or name.startswith(PREFIXES):
        raise ValueError(
            "the name %s of %s collides with the name of a table of the contract" % (name, where)
        )
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
        self.grown = ()
        if type(held["version"]) is not int or held["version"] != VERSION:
            raise ValueError(
                "the schema has version %r; this code reads version %d" % (held["version"], VERSION)
            )
        stated = fields(held["time"], TIMED, "the time of the schema")
        if stated["model"] not in MODELS:
            raise ValueError(
                "the time of the schema states its model as one of %s, found %r"
                % (", ".join(MODELS), stated["model"])
            )
        units = stated["units"]
        if (
            not isinstance(units, list)
            or len(set(units)) != len(units)
            or any(type(unit) is not str or not unit for unit in units)
        ):
            raise ValueError(
                "the units of time of the schema are a list of distinct names, found %r" % (units,)
            )
        if bool(units) != (stated["model"] in UNITED):
            raise ValueError(
                "a schema states units of time when its graphs have a clock and none when they have not; "
                "the model is %s and the units are %r" % (stated["model"], units)
            )
        if stated["ties"] is not None:
            read = TIES.named(stated["ties"]).columns
            if not set(read.items()) <= set(SEQUENCE.items()):
                raise ValueError(
                    "the rule %s for ties reads the columns %s; a bank keeps %s for a rule to read"
                    % (stated["ties"], ", ".join(read), ", ".join(SEQUENCE))
                )
        if type(held["observers"]) is not bool:
            raise ValueError("the schema states whether facts and tasks name observers as true or false")
        self.version, self.model, self.units = held["version"], stated["model"], tuple(units)
        self.ties, self.observers = stated["ties"], held["observers"]
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
        self.nodes, self.holds = {}, {}
        for kind, entry in held["nodes"].items():
            where = "the kind of node " + reserved(kind, "a kind of node")
            entry = fields(entry, KINDED, where)
            self.nodes[kind] = listed(entry["columns"], self.columns, where)
            if entry["holds"] is not None:
                self.holds[kind] = self.holding(entry["holds"], where)
        if not isinstance(held["sets"], dict):
            raise ValueError("the sets of the schema are a mapping from kind of set edge to statement")
        self.sets = {}
        for kind, entry in held["sets"].items():
            where = "the kind of set edge " + reserved(kind, "a kind of set edge")
            if kind in self.nodes:
                raise ValueError("%s is also a kind of node" % where)
            self.sets[kind] = self.joined(fields(entry, SETS, where), where)
        if not isinstance(held["edges"], dict):
            raise ValueError("the edges of the schema are a mapping from table to statement")
        self.edges = {}
        for table, entry in held["edges"].items():
            where = "the edge table " + named(table, "an edge table")
            if table in self.nodes or table in self.sets:
                raise ValueError("%s is also a kind of node or of set edge" % where)
            entry = fields(entry, EDGED, where)
            pairs = entry["endpoints"]
            if (
                not isinstance(pairs, list)
                or not pairs
                or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
                or len({tuple(pair) for pair in pairs}) != len(pairs)
            ):
                raise ValueError("%s states its endpoint kinds as distinct pairs" % where)
            ends = {kind for pair in pairs for kind in pair}
            strangers = sorted(ends - set(self.nodes) - set(self.sets))
            if strangers:
                raise ValueError("%s joins kinds of node not stated: %s" % (where, ", ".join(strangers)))
            if type(entry["directed"]) is not bool:
                raise ValueError("%s states whether it has a direction as true or false" % where)
            about = entry["about"]
            if ends & set(self.sets):
                if type(about) is not str:
                    raise ValueError(
                        "%s joins a kind of set edge and states by name the rule for a time at which the edge "
                        "it is about does not hold, found %r" % (where, about)
                    )
                ABOUT.named(about)
            elif about is not None:
                raise ValueError(
                    "%s joins kinds of node only and states no rule for an edge about an edge, found %r"
                    % (where, about)
                )
            relation = entry["relation"]
            if relation != DECLARED:
                relation = fields(relation, ("column",), "the relation of " + where)["column"]
                if named(relation, "the relation column of " + where) in FACTS + ENDS + tuple(self.stamps()):
                    raise ValueError("the relation column of %s collides with a column of facts" % where)
            columns = listed(entry["columns"], self.columns, where)
            if relation in columns:
                raise ValueError("the relation column of %s is also one of its columns" % where)
            self.edges[table] = dict(
                endpoints=tuple((source, target) for source, target in pairs),
                directed=entry["directed"],
                columns=columns,
                column=None if relation == DECLARED else relation,
                about=about,
            )
        self.tasks = self.asked(held["tasks"])
        self.target_columns()
        self.united()
        both =[name for name in self.graph if name in self.asked_columns()]
        if both:
            raise ValueError(
                "the columns %s are columns of the graph and of a kind of task, and a task reads the columns of "
                "its graph in the same places as its own" % ", ".join(both)
            )
        used = set(self.graph) | set(self.layout()) | set(self.member_columns())
        used |= set(self.inner_columns()) | set(self.within_columns())
        used |= {name for entry in self.edges.values() for name in entry["columns"]}
        unused = [name for name in self.columns if name not in used]
        if unused:
            raise ValueError(
                "the columns %s are declared and used by no kind of node or of set edge, no edge table, no "
                "graph held by a node, no kind of task and not by the graph" % ", ".join(unused)
            )
        LOG.debug(
            "schema exit kinds=%d sets=%d edges=%d columns=%d seconds=%.4f",
            len(self.nodes),
            len(self.sets),
            len(self.edges),
            len(self.columns),
            time.perf_counter() - started,
        )

    def joined(self, entry, where):
        started = time.perf_counter()
        kinds, roles = entry["kinds"], entry["roles"]
        if (
            not isinstance(kinds, list)
            or not kinds
            or len(set(kinds)) != len(kinds)
            or not set(kinds) <= set(self.nodes)
        ):
            raise ValueError("%s joins kinds of node that are stated, each once" % where)
        if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
            raise ValueError("%s states the roles of its members as a list of distinct names" % where)
        for role in roles:
            named(role, "a role of " + where)
        members = listed(entry["members"], self.columns, "the members of " + where)
        if "role" in members:
            raise ValueError("a column of the members of %s collides with the column role" % where)
        result = dict(
            kinds=tuple(kinds),
            roles=tuple(roles),
            members=members,
            columns=listed(entry["columns"], self.columns, where),
        )
        LOG.debug("schema joined exit members=%d seconds=%.4f", len(kinds), time.perf_counter() - started)
        return result

    def holding(self, stated, where):
        started = time.perf_counter()
        stated = fields(stated, HOLDS, "the graphs held by " + where)
        if not isinstance(stated["nodes"], dict) or not stated["nodes"]:
            raise ValueError("the graphs held by %s state at least one kind of node" % where)
        if not isinstance(stated["edges"], dict):
            raise ValueError(
                "the edges of the graphs held by %s are a mapping from relation to statement" % where
            )
        nodes = {}
        for kind, entry in stated["nodes"].items():
            inner = "the kind %s of the graphs held by %s" % (named(kind, "a kind of held node"), where)
            nodes[kind] = listed(fields(entry, ("columns",), inner)["columns"], self.columns, inner)
        edges = {}
        for relation, entry in stated["edges"].items():
            inner = "the relation %s of the graphs held by %s" % (named(relation, "a held relation"), where)
            entry = fields(entry, INWARD, inner)
            pairs = entry["endpoints"]
            if (
                not isinstance(pairs, list)
                or not pairs
                or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
                or len({tuple(pair) for pair in pairs}) != len(pairs)
                or not {kind for pair in pairs for kind in pair} <= set(nodes)
            ):
                raise ValueError("%s states its endpoint kinds as distinct pairs of held kinds" % inner)
            if type(entry["directed"]) is not bool:
                raise ValueError("%s states whether it has a direction as true or false" % inner)
            edges[relation] = dict(
                endpoints=tuple((source, target) for source, target in pairs),
                directed=entry["directed"],
                columns=listed(entry["columns"], self.columns, inner),
            )
        held = dict(nodes=nodes, edges=edges, encoder=stated["encoder"])
        HELD.named(held["encoder"])
        clashes = sorted(set(inward(held) + outward(held)) & (set(OWNED) | {"relation"}))
        if clashes:
            raise ValueError(
                "the columns %s of the graphs held by %s collide with columns of the contract"
                % (", ".join(clashes), where)
            )
        LOG.debug("schema holding exit kinds=%d seconds=%.4f", len(nodes), time.perf_counter() - started)
        return held

    def declared(self):
        return dict(self.nodes, **{kind: entry["columns"] for kind, entry in self.sets.items()})

    def entities(self):
        result = self.declared()
        for _, added in self.grown:
            for kind, columns in added.placed.items():
                result[kind] = tuple(result[kind]) + columns
            result.update({kind: entry["columns"] for kind, entry in added.kinds.items()})
        return result

    def added_columns(self):
        return tuple(name for _, added in self.grown for name in added.columns)

    def grew(self, entry, added):
        started = time.perf_counter()
        result = copy.copy(self)
        result.columns = dict(self.columns, **added.columns)
        result.grown = self.grown + ((entry, added),)
        LOG.debug("schema grew exit entry=%s seconds=%.4f", entry, time.perf_counter() - started)
        return result

    @timed("schema.member_columns")
    def member_columns(self):
        return tuple(dict.fromkeys(name for entry in self.sets.values() for name in entry["members"]))

    @timed("schema.inner_columns")
    def inner_columns(self):
        return tuple(dict.fromkeys(name for held in self.holds.values() for name in inward(held)))

    @timed("schema.within_columns")
    def within_columns(self):
        return tuple(dict.fromkeys(name for held in self.holds.values() for name in outward(held)))

    def about(self):
        return {table: entry["about"] for table, entry in self.edges.items() if entry["about"] is not None}

    def asked(self, stated):
        started = time.perf_counter()
        entry = fields(stated, ASKED, "the tasks")
        if type(entry["signals"]) is not int or entry["signals"] < 0:
            raise ValueError("the tasks state the width of their signals as a whole number")
        for part in ("key", "grouping"):
            if part == "grouping" and entry[part] is None:
                continue
            if named(entry[part], "the %s of the tasks" % part) in GIVEN:
                raise ValueError(
                    "the %s of the tasks collides with a column of the contract: %s" % (part, entry[part])
                )
        if entry["key"] == entry["grouping"]:
            raise ValueError("the key and the grouping of the tasks are two columns")
        if not isinstance(entry["references"], dict):
            raise ValueError("the references of the tasks are a mapping from name to sort and kinds of node")
        references = {}
        for name, found in entry["references"].items():
            where = "the reference " + named(name, "a reference")
            found = fields(found, REFERRED, where)
            sort = SORTS.named(found["sort"])
            kinds = found["kinds"]
            if (
                not isinstance(kinds, list)
                or len(set(kinds)) != len(kinds)
                or not set(kinds) <= set(self.entities())
            ):
                raise ValueError("%s names kinds of node or of set edge that are stated, each once" % where)
            if bool(kinds) == sort.whole:
                raise ValueError(
                    "%s is of the sort %s and names %s"
                    % (
                        where,
                        found["sort"],
                        "kinds of node, which the sort does not hold" if kinds else "no kind of node",
                    )
                )
            references[name] = dict(sort=found["sort"], kinds=tuple(kinds))
        if not isinstance(entry["kinds"], dict) or not entry["kinds"]:
            raise ValueError("the tasks state at least one kind of task")
        kinds = {}
        for kind, found in entry["kinds"].items():
            where = "the kind of task " + named(kind, "a kind of task")
            found = fields(found, KIND, where)
            names = found["references"]
            if not isinstance(names, list) or len(set(names)) != len(names):
                raise ValueError("the references of %s are a list of distinct names" % where)
            unstated = [name for name in names if name not in references]
            if unstated:
                raise ValueError(
                    "%s names references not stated: %s" % (where, ", ".join(map(str, unstated)))
                )
            if type(found["weight"]) is not bool:
                raise ValueError("%s states whether its tasks carry a weight as true or false" % where)
            if found["absent"] not in ABSENT:
                raise ValueError(
                    "%s states what a reference to a node not observed resolves to as one of %s, found %r"
                    % (where, ", ".join(ABSENT), found["absent"])
                )
            target = self.aimed(found["target"], names, references, where)
            kinds[kind] = dict(
                references=tuple(names),
                columns=listed(found["columns"], self.columns, where),
                target=target,
                weight=found["weight"],
                absent=found["absent"],
                event=self.seen(found["event"], names, references, found["absent"], where),
                matures=self.matured(found["matures"], target, names, references, where),
            )
        owners = {}
        for held in kinds.values():
            if held["target"] is None:
                continue
            for name in held["target"]["series"]:
                other = owners.setdefault(name, held["target"]["kind"])
                if other != held["target"]["kind"]:
                    raise ValueError(
                        "the series %s is named by targets of the kinds %s and %s; a series belongs to targets of "
                        "one kind" % (name, other, held["target"]["kind"])
                    )
        result = dict(
            key=entry["key"],
            grouping=entry["grouping"],
            signals=entry["signals"],
            references=references,
            kinds=kinds,
        )
        LOG.debug(
            "schema asked exit kinds=%d references=%d seconds=%.4f",
            len(kinds),
            len(references),
            time.perf_counter() - started,
        )
        return result

    @timed("schema.aimed")
    def aimed(self, stated, names, references, where):
        if stated is None:
            return None
        if not isinstance(stated, dict) or "kind" not in stated:
            raise ValueError(
                "the target of %s states the name of a registered kind of target, found %r" % (where, stated)
            )
        kind = TARGETS.named(stated["kind"])
        stated = fields(stated, AIMED + kind.parameters, "the target of " + where)
        if stated["about"] is not None and stated["about"] not in names:
            raise ValueError(
                "the target of %s is about one of its references, %s, or about none of them, found %r"
                % (where, ", ".join(names) or "none", stated["about"])
            )
        series = kind.series(stated["series"], "the target of " + where)
        return dict(
            kind=stated["kind"],
            series=series,
            about=stated["about"],
            utility=self.valued(stated["utility"], stated["kind"], series, where),
            **kind.checked(stated, self, references, where),
        )

    @timed("schema.valued")
    def valued(self, stated, kind, series, where):
        where = "the utility of the target of " + where
        if not isinstance(stated, dict) or "name" not in stated:
            raise ValueError("%s states the name of a registered utility, found %r" % (where, stated))
        entry = UTILITIES.named(stated["name"])
        fields(stated, UTILITY + entry.parameters, where)
        if stated["better"] not in BETTER:
            raise ValueError(
                "%s states which value is better as one of %s, found %r" % (where, ", ".join(BETTER), stated["better"])
            )
        if kind not in entry.reads:
            raise ValueError(
                "%s, %s, reads targets of the kinds %s and the target is of the kind %s"
                % (where, stated["name"], ", ".join(entry.reads), kind)
            )
        return dict(name=stated["name"], better=stated["better"], **entry.checked(stated, series, where))

    @timed("schema.seen")
    def seen(self, stated, names, references, absent, where):
        if stated is None:
            return None
        if (
            not isinstance(stated, list)
            or len(stated) != 2
            or any(name not in names or references[name]["sort"] != "node" for name in stated)
        ):
            raise ValueError(
                "%s is seen by later tasks as an event between two of its references of the sort node, "
                "found %r" % (where, stated)
            )
        if absent != "stop":
            raise ValueError("%s is seen as an event, and its references are observed at its time" % where)
        return tuple(stated)

    @timed("schema.matured")
    def matured(self, stated, target, names, references, where):
        if stated is None:
            return None
        stated = fields(stated, MATURED, "what matures of " + where)
        kind = None if target is None else TARGETS.named(target["kind"])
        if kind is None or kind.matures is None:
            raise ValueError(
                "%s states what matures, and only a target of a kind that matures into the values of columns "
                "matures" % where
            )
        name = stated["reference"]
        if name not in names or references[name]["sort"] != "node":
            raise ValueError(
                "what matures of %s is a value of one of its references of the sort node, found %r"
                % (where, name)
            )
        columns = kind.ripe(stated["column"], target["series"], "what matures of " + where)
        entities = self.entities()
        for column in dict.fromkeys(columns.values()):
            lacking = [held for held in references[name]["kinds"] if column not in entities[held]]
            if lacking or column not in self.columns or self.columns[column]["type"] != kind.matures:
                raise ValueError(
                    "what matures of %s is a column of the type %s of every kind of node of the reference %s, "
                    "found %r" % (where, kind.matures, name, column)
                )
        return dict(reference=name, column=copy.deepcopy(stated["column"]), columns=columns)

    @timed("schema.ripened")
    def ripened(self):
        result = {}
        for entry in self.tasks["kinds"].values():
            if entry["matures"] is None:
                continue
            for name, column in entry["matures"]["columns"].items():
                if self.kind(column).symbols:
                    result.setdefault(column, []).append(name)
        return result

    def mapping(self):
        started = time.perf_counter()
        result = dict(
            version=self.version,
            time=dict(model=self.model, units=list(self.units), ties=self.ties),
            observers=self.observers,
            columns={name: dict(entry) for name, entry in self.columns.items()},
            graph=dict(columns=list(self.graph)),
            nodes={
                kind: dict(
                    columns=list(columns),
                    holds=None
                    if kind not in self.holds
                    else dict(
                        nodes={
                            inner: dict(columns=list(held))
                            for inner, held in self.holds[kind]["nodes"].items()
                        },
                        edges={
                            relation: dict(
                                endpoints=[list(pair) for pair in entry["endpoints"]],
                                directed=entry["directed"],
                                columns=list(entry["columns"]),
                            )
                            for relation, entry in self.holds[kind]["edges"].items()
                        },
                        encoder=self.holds[kind]["encoder"],
                    ),
                )
                for kind, columns in self.nodes.items()
            },
            sets={
                kind: dict(
                    kinds=list(entry["kinds"]),
                    roles=list(entry["roles"]),
                    members=list(entry["members"]),
                    columns=list(entry["columns"]),
                )
                for kind, entry in self.sets.items()
            },
            edges={
                table: dict(
                    endpoints=[list(pair) for pair in entry["endpoints"]],
                    directed=entry["directed"],
                    columns=list(entry["columns"]),
                    relation=DECLARED if entry["column"] is None else dict(column=entry["column"]),
                    about=entry["about"],
                )
                for table, entry in self.edges.items()
            },
            tasks=dict(
                key=self.tasks["key"],
                grouping=self.tasks["grouping"],
                signals=self.tasks["signals"],
                references={
                    name: dict(sort=entry["sort"], kinds=list(entry["kinds"]))
                    for name, entry in self.tasks["references"].items()
                },
                kinds={
                    kind: dict(
                        references=list(entry["references"]),
                        columns=list(entry["columns"]),
                        target=None
                        if entry["target"] is None
                        else dict(
                            kind=entry["target"]["kind"],
                            series=TARGETS.named(entry["target"]["kind"]).written(entry["target"]["series"]),
                            about=entry["target"]["about"],
                            utility=copy.deepcopy(entry["target"]["utility"]),
                            **{
                                name: entry["target"][name]
                                for name in TARGETS.named(entry["target"]["kind"]).parameters
                            },
                        ),
                        weight=entry["weight"],
                        absent=entry["absent"],
                        event=None if entry["event"] is None else list(entry["event"]),
                        matures=None
                        if entry["matures"] is None
                        else dict(
                            reference=entry["matures"]["reference"],
                            column=copy.deepcopy(entry["matures"]["column"]),
                        ),
                    )
                    for kind, entry in self.tasks["kinds"].items()
                },
            ),
        )
        LOG.debug("schema mapping exit seconds=%.4f", time.perf_counter() - started)
        return result

    @timed("schema.Schema.fingerprint")
    def fingerprint(self):
        return sha256_json(self.mapping())

    @timed("schema.stamps")
    def stamps(self):
        held = {name: STAMP for name in STAMPS[self.model]}
        if self.ties is not None:
            held.update(TIES.named(self.ties).columns)
        if self.observers:
            held["observers"] = WATCHERS
        return held

    @timed("schema.Schema.layout")
    def layout(self):
        groups = list(self.declared().values()) + [self.asked_columns()]
        return tuple(dict.fromkeys(name for columns in groups for name in columns))

    @timed("schema.asked_columns")
    def asked_columns(self):
        return tuple(
            dict.fromkeys(name for entry in self.tasks["kinds"].values() for name in entry["columns"])
        )

    @timed("schema.series")
    def series(self):
        return tuple(
            dict.fromkeys(
                name
                for entry in self.tasks["kinds"].values()
                if entry["target"] is not None
                for name in entry["target"]["series"]
            )
        )

    def targets(self):
        return tuple(
            dict.fromkeys(
                entry["target"]["kind"] for entry in self.tasks["kinds"].values() if entry["target"] is not None
            )
        )

    @timed("schema.measured")
    def measured(self):
        return {
            name: entry["target"]["kind"]
            for entry in self.tasks["kinds"].values()
            if entry["target"] is not None
            for name in entry["target"]["series"]
        }

    def target_columns(self):
        started = time.perf_counter()
        result, owner = {}, {}
        for kind in self.targets():
            for name, written in TARGETS.named(kind).columns.items():
                if name in OWNED_TARGET:
                    raise ValueError(
                        "the kind of target %s stores the column %s, a column of the contract of the table of "
                        "targets" % (kind, name)
                    )
                if result.setdefault(name, written) != written:
                    raise ValueError(
                        "the kinds of target %s and %s store the column %s of the table of targets as %s and as %s"
                        % (owner[name], kind, name, result[name], written)
                    )
                owner.setdefault(name, kind)
        LOG.debug("schema target columns exit columns=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def united(self):
        started = time.perf_counter()
        held = {
            kind: entry["target"]["kind"]
            for kind, entry in self.tasks["kinds"].items()
            if entry["target"] is not None and TARGETS.named(entry["target"]["kind"]).times
        }
        if held and len(self.units) > 1:
            raise ValueError(
                "the kinds of task %s state targets of the kinds %s, whose values are times, and the schema states "
                "the units %s; times of graphs of different units are never compared, so a schema holding such a "
                "target states one unit"
                % (", ".join(held), ", ".join(dict.fromkeys(held.values())), ", ".join(self.units))
            )
        LOG.debug("schema united exit timed=%d seconds=%.4f", len(held), time.perf_counter() - started)

    @timed("schema.events")
    def events(self):
        return tuple(kind for kind, entry in self.tasks["kinds"].items() if entry["event"] is not None)

    @timed("schema.aimed_kinds")
    def aimed_kinds(self):
        return tuple(kind for kind, entry in self.tasks["kinds"].items() if entry["target"] is not None)

    @timed("schema.revealing")
    def revealing(self):
        return tuple(
            dict.fromkeys(
                entry["target"]["about"]
                for entry in self.tasks["kinds"].values()
                if entry["target"] is not None and entry["target"]["about"] is not None
            )
        )

    def kind(self, name):
        return TYPES.named(self.columns[name]["type"])

    def encoder(self, name):
        return ENCODERS.named(self.columns[name]["encoder"])

    @timed("schema.Schema.symbolic")
    def symbolic(self):
        return tuple(name for name in self.columns if self.kind(name).symbols)

    def tables(self):
        started = time.perf_counter()
        keys = dict(id="INT64", graph="INT32")
        described = dict(id="INT32", name="STRING")
        if self.model in UNITED:
            described["unit"] = "STRING"
        result = {"graph": dict(kind="NODE", required=described, columns=self.graph)}
        result["observation"] = dict(kind="NODE", required=dict(id="INT64", **self.stamps()), columns=())
        for kind, columns in self.entities().items():
            result[kind] = dict(kind="NODE", required=dict(keys, entity="INT32", key="STRING"), columns=())
            result[OBSERVED + kind] = dict(
                kind="REL", required={}, columns=columns, pairs=((kind, "observation"),)
            )
        for kind, entry in self.sets.items():
            result[MEMBER + kind] = dict(
                kind="REL",
                required=dict(role="STRING"),
                columns=entry["members"],
                pairs=tuple((kind, member) for member in entry["kinds"]),
            )
        for kind, held in self.holds.items():
            result[INNER + kind] = dict(kind="NODE", required=dict(OWNED), columns=inward(held))
            result[WITHIN + kind] = dict(
                kind="REL",
                required=dict(relation="STRING"),
                columns=outward(held),
                pairs=((INNER + kind, INNER + kind),),
            )
        for table, entry in self.edges.items():
            required = dict(present=("BOOL", "BOOLEAN"), **self.stamps())
            if entry["column"] is not None:
                required[entry["column"]] = "STRING"
            result[EDGE + table] = dict(
                kind="REL", required=required, columns=entry["columns"], pairs=entry["endpoints"]
            )
        asked = self.tasks
        required = dict(keys, kind="STRING")
        if self.model != "none":
            required["time"] = STAMP
        if self.observers:
            required["observer"] = "STRING"
        required.update(split="STRING", weight="FLOAT")
        if asked["grouping"] is not None:
            required[asked["grouping"]] = "INT32"
        required[asked["key"]] = "INT64"
        if asked["signals"]:
            vector = "FLOAT[%d]" % asked["signals"]
            required.update(signal=vector, signal_observed=vector)
        result["task"] = dict(kind="NODE", required=required, columns=self.asked_columns())
        for name, entry in asked["references"].items():
            stored = SORTS.named(entry["sort"]).columns()
            if stored is not None:
                result[TASK + name] = dict(
                    kind="REL",
                    required=stored,
                    columns=(),
                    pairs=tuple(("task", kind) for kind in entry["kinds"]),
                )
        required = dict(id="INT64", y="DOUBLE")
        if self.model != "none":
            required["known"] = STAMP
        required.update(self.target_columns())
        result[TARGET] =dict(kind="NODE", required=required, columns=())
        result[LINK] = dict(kind="REL", required={}, columns=(), pairs=((TARGET, "task"),))
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

    def text(self, encoding):
        started = time.perf_counter()

        @timed("schema.Schema.text.described")
        def described(columns):
            if not columns:
                return "no columns"
            return ", ".join(
                "%s %s by %s" % (name, self.columns[name]["type"], self.columns[name]["encoder"])
                for name in columns
            )

        fitted = encoding is not None and encoding.states is not None

        @timed("schema.Schema.text.wide")
        def wide(columns):
            return ", width %d" % encoding.width(columns) if fitted else ""

        lines = ["schema version %d" % self.version]
        lines.append(
            "time: %s, %s, %s"
            % (
                dict(none="none", order="order without a clock", clock="one clock", clocks="two clocks")[
                    self.model
                ],
                "units " + ", ".join(self.units) if self.units else "no unit",
                "no rule for ties" if self.ties is None else "ties by " + self.ties,
            )
        )
        lines.append("observers: %s" % ("named by facts and tasks" if self.observers else "none"))
        lines.append("graph: %s" % described(self.graph))
        lines.append("kinds of node:")
        for kind, columns in self.nodes.items():
            lines.append("  %s: %s%s" % (kind, described(columns), wide(columns)))
            if kind in self.holds:
                held = self.holds[kind]
                lines.append(
                    "    holds graphs encoded by %s%s: %s; %s"
                    % (
                        held["encoder"],
                        ", width %d" % encoding.held[kind][1] if fitted else "",
                        "; ".join(
                            "%s of %s" % (inner, described(owned)) for inner, owned in held["nodes"].items()
                        ),
                        "; ".join(
                            "%s from %s, %s, %s"
                            % (
                                relation,
                                " or ".join("%s to %s" % pair for pair in entry["endpoints"]),
                                "directed" if entry["directed"] else "without direction",
                                described(entry["columns"]),
                            )
                            for relation, entry in held["edges"].items()
                        )
                        or "no relations",
                    )
                )
        lines.append("kinds of set edge:")
        lines += [
            "  %s: members of %s as %s, per member %s; %s%s"
            % (
                kind,
                ", ".join(entry["kinds"]),
                ", ".join(entry["roles"]),
                described(entry["members"]),
                described(entry["columns"]),
                wide(entry["columns"]),
            )
            for kind, entry in self.sets.items()
        ]
        lines.append("edge tables:")
        for table, entry in self.edges.items():
            lines.append(
                "  %s: %s, %s, relation %s, %s%s%s"
                % (
                    table,
                    "; ".join("%s to %s" % pair for pair in entry["endpoints"]),
                    "directed" if entry["directed"] else "without direction",
                    "declared" if entry["column"] is None else "taken from the column " + entry["column"],
                    described(entry["columns"]),
                    wide(entry["columns"]),
                    "" if entry["about"] is None else ", about an edge by the rule " + entry["about"],
                )
            )
        asked = self.tasks
        lines.append(
            "tasks: keyed by %s, %s, signals of width %d"
            % (
                asked["key"],
                "not grouped" if asked["grouping"] is None else "grouped by " + asked["grouping"],
                asked["signals"],
            )
        )
        lines += [
            "  reference %s: %s%s"
            % (name, entry["sort"], " of " + ", ".join(entry["kinds"]) if entry["kinds"] else "")
            for name, entry in asked["references"].items()
        ]
        for kind, entry in asked["kinds"].items():
            lines.append(
                "  kind %s: references %s; %s%s; target %s; %s; absent %s; %s; %s"
                % (
                    kind,
                    ", ".join(entry["references"]) or "none",
                    described(entry["columns"]),
                    wide(entry["columns"]),
                    "none"
                    if entry["target"] is None
                    else "%s of %s%s%s, valued by %s, %s better"
                    % (
                        entry["target"]["kind"],
                        ", ".join(entry["target"]["series"]),
                        "" if entry["target"]["about"] is None else " about " + entry["target"]["about"],
                        "".join(
                            ", %s %s" % (name, entry["target"][name])
                            for name in TARGETS.named(entry["target"]["kind"]).parameters
                        ),
                        entry["target"]["utility"]["name"],
                        entry["target"]["utility"]["better"],
                    ),
                    "weighted" if entry["weight"] else "not weighted",
                    entry["absent"],
                    "not seen as an event"
                    if entry["event"] is None
                    else "seen as an event between %s and %s" % entry["event"],
                    "nothing matures"
                    if entry["matures"] is None
                    else "matures as %s of %s"
                    % (
                        entry["matures"]["column"]
                        if type(entry["matures"]["column"]) is str
                        else ", ".join(
                            "%s for %s" % (column, name) for name, column in entry["matures"]["columns"].items()
                        ),
                        entry["matures"]["reference"],
                    ),
                )
            )
        if self.grown:
            lines.append("added by the pipeline:")
            lines += ["  " + added.described(entry) for entry, added in self.grown]
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
        lacking = [table for table in schema.edges if table not in self.relations]
        if lacking:
            raise ValueError("the edge tables %s have no relations" % ", ".join(lacking))
        self.layout = schema.layout()
        self.table = tuple(dict.fromkeys(self.layout + schema.graph + schema.added_columns()))
        if states is not None:
            lacking = [name for name in schema.columns if name not in states]
            if lacking:
                raise ValueError("the columns %s have no fitted encoder" % ", ".join(lacking))
            self.sizes = {
                name: (0, 0) if states[name] is None else schema.encoder(name).widths(states[name])
                for name in schema.columns
            }
            self.values = sum(self.sizes[name][0] for name in self.table)
            self.held, self.inner = {}, 0
            for kind, held in schema.holds.items():
                width = HELD.named(held["encoder"]).width(self, held)
                self.held[kind] = (self.inner, width)
                self.inner += width
            self.features = self.values + sum(self.sizes[name][1] for name in self.table) + self.inner
        LOG.debug(
            "encoding exit columns=%d fitted=%s seconds=%.4f",
            len(self.layout),
            states is not None,
            time.perf_counter() - started,
        )

    @timed("schema.Encoding.__getattr__")
    def __getattr__(self, key):
        if key in FITTED:
            raise ValueError(
                "the encoders of the columns are fitted when the views of a statement of splits are built, and "
                "this encoding holds the dictionaries of a bank only; %s needs the fitted encoders" % key
            )
        raise AttributeError(key)

    @timed("schema.Encoding.declaration")
    def declaration(self):
        return dict(
            dictionaries=self.dictionaries,
            states=self.states,
            relations={table: list(names) for table, names in self.relations.items()},
        )

    @timed("schema.Encoding.width")
    def width(self, columns):
        return sum(sum(self.sizes[name]) for name in columns)

    @timed("schema.encoding.slots")
    def slots(self, name):
        start, flagged = 0, self.values
        for held in self.table:
            width, flag = self.sizes[held]
            if held == name:
                return np.arange(start, start + width), np.arange(flagged, flagged + flag)
            start, flagged = start + width, flagged + flag
        raise KeyError("the encoding holds no column %s" % name)

    def widened(self, rows):
        added = self.schema.added_columns()
        if not added:
            return rows
        started = time.perf_counter()
        stem = [name for name in self.table if name not in added]
        values, flags = (sum(self.sizes[name][part] for name in stem) for part in (0, 1))
        kept = np.r_[
            np.arange(values), self.values + np.arange(flags), self.features - self.inner + np.arange(self.inner)
        ]
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != len(kept):
            raise ValueError(
                "rows of encoded values of the shape %s are widened from the encoding of %d values the pipeline "
                "grew from" % (rows.shape, len(kept))
            )
        result = np.zeros((len(rows), self.features), rows.dtype)
        result[:, kept] = rows
        LOG.debug("encoding widened exit rows=%d seconds=%.4f", len(rows), time.perf_counter() - started)
        return result

    @timed("schema.Encoding.positions")
    def positions(self, columns):
        return [self.layout.index(name) for name in columns]

    @timed("schema.Encoding.encoded")
    def encoded(self, columns, codes):
        parts = [
            (np.zeros((len(codes), 0), np.float32),) * 2
            if self.states[name] is None
            else self.schema.encoder(name).encode(
                codes[:, position],
                self.states[name],
                self.dictionaries[name] if self.schema.kind(name).symbols else None,
            )
            for position, name in enumerate(columns)
        ]
        return [values for values, _ in parts], [flags for _, flags in parts]

    def placed(self, codes, graph):
        started = time.perf_counter()
        codes = np.asarray(codes, np.float32).reshape(-1, len(self.layout))
        result = np.full((len(codes), len(self.table)), np.nan, np.float32)
        result[:, : len(self.layout)] = codes
        if graph is not None:
            result[:, [self.table.index(name) for name in self.schema.graph]] = np.asarray(graph, np.float32)
        LOG.debug("encoding placed exit rows=%d seconds=%.4f", len(codes), time.perf_counter() - started)
        return result

    def nodes(self, codes, graph):
        started = time.perf_counter()
        codes = self.placed(codes, graph)
        values, flags = self.encoded(self.table, codes)
        result = np.concatenate(values + flags + [np.zeros((len(codes), 0), np.float32)], 1)
        LOG.debug("encoding nodes exit rows=%d seconds=%.4f", len(codes), time.perf_counter() - started)
        return result

    def inputs(self, codes, held, graph):
        started = time.perf_counter()
        held = np.asarray(held, np.float32)
        codes = np.asarray(codes, np.float32).reshape(-1, len(self.layout))
        if held.shape != (len(codes), self.inner):
            raise ValueError(
                "the encoded held graphs of %d rows are of the shape %s, and the schema states a width of %d"
                % (len(codes), held.shape, self.inner)
            )
        result = np.concatenate((self.nodes(codes, graph), held), 1)
        LOG.debug("encoding inputs exit rows=%d seconds=%.4f", len(codes), time.perf_counter() - started)
        return result

    def origin(self, place, codes, graph):
        started = time.perf_counter()
        edge, codes = 0, self.placed(codes, graph)[0]
        for part in (0, 1):
            for position, name in enumerate(self.table):
                edge += self.sizes[name][part]
                if place < edge:
                    kind = self.schema.kind(name)
                    dictionary = self.dictionaries[name] if kind.symbols else None
                    value = kind.stated(codes[[position]], dictionary)[0]
                    LOG.debug(
                        "encoding origin exit place=%d seconds=%.4f", place, time.perf_counter() - started
                    )
                    return "the value %r of the column %s, encoded by %s," % (
                        value,
                        name,
                        self.schema.columns[name]["encoder"],
                    )
        for kind, (start, width) in self.held.items():
            if place < edge + start + width:
                LOG.debug("encoding origin exit place=%d seconds=%.4f", place, time.perf_counter() - started)
                return "the graph it holds, encoded by %s," % self.schema.holds[kind]["encoder"]
        raise ValueError("the encoded values hold no column at the place %d" % place)

    @timed("schema.encoding.vectors")
    def vectors(self, columns, codes):
        codes = np.asarray(codes, np.float32).reshape(-1, len(columns))
        values, flags = self.encoded(columns, codes)
        return np.concatenate(values + flags + [np.zeros((len(codes), 0), np.float32)], 1)

    def graphs(self, kind, graphs):
        started = time.perf_counter()
        held = self.schema.holds[kind]
        result = HELD.named(held["encoder"]).encode(self, held, graphs)
        if result.shape != (graphs["owners"], self.held[kind][1]):
            raise ValueError(
                "the encoder %s of the graphs held by %s gave the shape %s, and states the width %d"
                % (held["encoder"], kind, result.shape, self.held[kind][1])
            )
        LOG.debug(
            "encoding graphs exit kind=%s graphs=%d seconds=%.4f",
            kind,
            graphs["owners"],
            time.perf_counter() - started,
        )
        return result


def fitted(encoding, samples):
    started = time.perf_counter()
    schema, states = encoding.schema, {}
    for name in schema.columns:
        if samples[name] is None:
            states[name] = None
            continue
        codes = np.concatenate(samples[name] + [np.zeros(0, np.float32)])
        dictionary = encoding.dictionaries[name] if schema.kind(name).symbols else None
        states[name] = schema.encoder(name).fit(codes, dictionary, name)
    result = Encoding(schema, encoding.dictionaries, states, encoding.relations)
    LOG.info(
        "encoders fitted exit columns=%d unshown=%d features=%d seconds=%.3f",
        len(states),
        sum(state is None for state in states.values()),
        result.features,
        time.perf_counter() - started,
    )
    return result
