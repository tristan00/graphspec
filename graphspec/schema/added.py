import logging
import time

from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
PROVISIONAL = "value"


def paired(stated, where):
    if (
        type(stated) is not tuple
        or not stated
        or len(set(stated)) != len(stated)
        or not all(type(pair) is tuple and len(pair) == 2 and all(type(kind) is str for kind in pair) for pair in stated)
    ):
        raise ValueError("%s states its pairs of kinds as a tuple of distinct pairs of names, found %r" % (where, stated))
    return stated


@timed("added.named")
def named(stated, where):
    if type(stated) is not tuple or len(set(stated)) != len(stated) or not all(type(name) is str for name in stated):
        raise ValueError("%s states its columns as a tuple of distinct names, found %r" % (where, stated))
    return stated


def mapping(stated, where):
    if type(stated) is not dict or not all(type(name) is str and name for name in stated):
        raise ValueError("%s is a mapping from names, found %r" % (where, stated))
    return stated


class Added:
    def __init__(self, columns, kinds, placed, relations, joins, candidates):
        started = time.perf_counter()
        self.columns = {}
        for name, entry in mapping(columns, "the columns an operation adds").items():
            if type(entry) is not dict or set(entry) != {"type", "encoder"}:
                raise ValueError("the column %s an operation adds states its type and its encoder, found %r" % (name, entry))
            self.columns[name] = dict(type=entry["type"], encoder=entry["encoder"])
        self.kinds = {}
        for kind, entry in mapping(kinds, "the kinds of node an operation adds").items():
            if type(entry) is not dict or set(entry) != {"columns", "tokens"} or not callable(entry["tokens"]):
                raise ValueError(
                    "the kind %s an operation adds states its columns and a function from the kinds, the tokens and "
                    "the encoded rows of the observations of a graph to the tokens of its nodes in that graph, found %r"
                    % (kind, entry)
                )
            self.kinds[kind] = dict(columns=named(entry["columns"], "the kind " + kind), tokens=entry["tokens"])
        self.placed = {
            kind: named(held, "the columns placed on " + kind)
            for kind, held in mapping(placed, "the columns an operation places on kinds of node").items()
        }
        self.relations = {}
        for name, entry in mapping(relations, "the relations an operation adds").items():
            if type(entry) is not dict or set(entry) != {"pairs", "directed", "columns"} or type(entry["directed"]) is not bool:
                raise ValueError(
                    "the relation %s an operation adds states its pairs of kinds, whether it has a direction and its "
                    "columns, found %r" % (name, entry)
                )
            self.relations[name] = dict(
                pairs=paired(entry["pairs"], "the relation " + name),
                directed=entry["directed"],
                columns=named(entry["columns"], "the relation " + name),
            )
        self.joins = {
            name: paired(held, "the pairs joined to " + name)
            for name, held in mapping(joins, "the pairs an operation joins to relations").items()
        }
        self.candidates = {}
        for name, entry in mapping(candidates, "the relations an operation adds as candidates").items():
            if type(entry) is not dict or set(entry) != {"pairs", "directed"} or type(entry["directed"]) is not bool:
                raise ValueError(
                    "the relation %s an operation adds as candidates states its pairs of kinds and whether it has a "
                    "direction, found %r" % (name, entry)
                )
            self.candidates[name] = dict(pairs=paired(entry["pairs"], "the relation " + name), directed=entry["directed"])
        LOG.debug(
            "added exit columns=%d kinds=%d relations=%d candidates=%d seconds=%.4f",
            len(self.columns),
            len(self.kinds),
            len(self.relations),
            len(self.candidates),
            time.perf_counter() - started,
        )

    @timed("added.holds")
    def holds(self):
        return any((self.columns, self.kinds, self.placed, self.relations, self.joins, self.candidates))

    @timed("added.provisional")
    def provisional(self):
        return Added(
            {name: dict(entry, encoder=PROVISIONAL) for name, entry in self.columns.items()},
            self.kinds,
            self.placed,
            self.relations,
            self.joins,
            self.candidates,
        )

    @timed("added.declaration")
    def declaration(self):
        return dict(
            columns={name: dict(entry) for name, entry in self.columns.items()},
            kinds={kind: dict(columns=list(entry["columns"])) for kind, entry in self.kinds.items()},
            placed={kind: list(columns) for kind, columns in self.placed.items()},
            relations={
                name: dict(pairs=[list(pair) for pair in entry["pairs"]], directed=entry["directed"], columns=list(entry["columns"]))
                for name, entry in self.relations.items()
            },
            joins={name: [list(pair) for pair in pairs] for name, pairs in self.joins.items()},
            candidates={
                name: dict(pairs=[list(pair) for pair in entry["pairs"]], directed=entry["directed"])
                for name, entry in self.candidates.items()
            },
        )

    @timed("added.described")
    def described(self, entry):
        found = ["the column %s, %s by %s" % (name, held["type"], held["encoder"]) for name, held in self.columns.items()]
        found += [
            "the kind of node %s with %s" % (kind, ", ".join(held["columns"]) or "no columns")
            for kind, held in self.kinds.items()
        ]
        found += ["%s on the kind %s" % (", ".join(columns), kind) for kind, columns in self.placed.items()]
        found += [
            "the relation %s from %s, %s, with %s"
            % (
                name,
                " or ".join("%s to %s" % pair for pair in held["pairs"]),
                "directed" if held["directed"] else "without direction",
                ", ".join(held["columns"]) or "no columns",
            )
            for name, held in self.relations.items()
        ]
        found += [
            "%s to the relation %s" % (" and ".join("%s to %s" % pair for pair in pairs), name)
            for name, pairs in self.joins.items()
        ]
        found += [
            "the relation %s of candidates from %s, %s"
            % (
                name,
                " or ".join("%s to %s" % pair for pair in held["pairs"]),
                "directed" if held["directed"] else "without direction",
            )
            for name, held in self.candidates.items()
        ]
        return "%s adds %s" % (entry, "; ".join(found))
