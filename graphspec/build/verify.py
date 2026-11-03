import json
import logging
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from graphspec.infra.files import read_json
from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.schema.schema import VALUE
from graphspec.schema.time import ALWAYS, NEVER

from .builder import Builder
from .database import Database
from .mapping_build import EdgeStated, NodeStated
from .mapping_tasks import Asked, DecisionStated
from .mappings import Mapping
from .planning import stated
from .writing import ENCODER, canonical_records, verified

LOG = logging.getLogger(__name__)
SALT = "verify"
Fact = namedtuple("Fact", ("identity", "content", "first", "end"))
Difference = namedtuple("Difference", ("declaration", "places", "rows", "times", "message"))
OWN = dict(node=("key", "since", "at"), edge=("source", "target", "relation", "since", "at"),
           member=("set", "member", "role"), task=("key", "kind", "time", "references"))
TIMELESS = ("graph", "member")


@timed("verify.moment")
def moment(value):
    return ALWAYS if value is None else value


@timed("verify.shown")
def shown(value):
    return "before every time" if value == ALWAYS else str(value)


@timed("verify.when")
def when(value):
    return "before every time" if value is None else "the time %d" % value


@timed("verify.spelled")
def spelled(value):
    return "missing" if value is None else ENCODER.encode(value)


@timed("verify.columns")
def columns(rows):
    return [{name: values[at] for name, values in rows.items()} for at in range(len(next(iter(rows.values()))))]


@timed("verify.fact")
def fact(identity, row, family, first, end):
    content = {name: value for name, value in row.items() if name not in OWN[family]}
    return Fact(identity, ENCODER.encode(content), first, end)


@timed("verify.timed_fact")
def timed_fact(identity, row, family, since, until):
    first = moment(since) if "known" not in row or row["known"] is None else max(moment(since), row["known"])
    ends = [value for value in (until, row["replaced"] if "replaced" in row else None) if value is not None]
    return fact(identity, row, family, first, min(ends) if ends else NEVER)


@timed("verify.changed")
def changed(rows):
    following, order = [None] * len(rows), {}
    for at, row in enumerate(rows):
        order.setdefault(row["key"], []).append(at)
    for places in order.values():
        places.sort(key=lambda at: rows[at]["at"])
        for at, later in zip(places, places[1:]):
            following[at] = rows[later]["at"]
    return following


@timed("verify.values")
def values(graph, name):
    found = []
    for statement in graph["values"]:
        rows, kind, form = columns(statement["rows"]), statement["kind"], statement["form"]
        ends = changed(rows) if form == "change" else [row["until"] if form == "state" else None for row in rows]
        for row, until in zip(rows, ends):
            since = row["at"] if form == "change" else row["since"] if form == "state" else None
            found.append(timed_fact(("node", name, kind, row["key"], since), dict(row, until=until), "node", since,
                                    until))
    return found


@timed("verify.edges")
def edges(graph, name):
    found = []
    for statement in graph["edges"]:
        table, ends, form = statement["table"], statement["ends"], statement["form"]
        for row in columns(statement["rows"]):
            since = row["at"] if form == "event" else row["since"] if form == "state" else None
            until = since + 1 if form == "event" else row["until"] if form == "state" else None
            identity = ("edge", name, table, row["relation"] if "relation" in row else None,
                        (ends[0], row["source"]), (ends[1], row["target"]), since)
            found.append(timed_fact(identity, row, "edge", since, until))
    return found


@timed("verify.asked")
def asked(graph, name, schema):
    tasks, found = graph["tasks"], []
    references = tasks["references"]
    for at, row in enumerate(columns({name: value for name, value in tasks.items() if name != "references"})):
        named = schema["tasks"]["kinds"][row["kind"]]["references"]
        held = ENCODER.encode({reference: references[reference][at] for reference in named})
        time_ = row["time"] if "time" in row else None
        found.append(fact(("task", name, row["kind"], time_, held), row, "task", moment(time_), moment(time_)))
    return found


@timed("verify.held")
def held(case):
    found = []
    for graph in case["graphs"]:
        name = graph["name"]
        found.append(Fact(("graph", name), ENCODER.encode(graph["columns"]), ALWAYS, NEVER))
        found += values(graph, name) + edges(graph, name) + asked(graph, name, case["schema"])
        for kind, rows in (graph["sets"] if "sets" in graph else {}).items():
            found += [fact(("member", name, kind, row["set"], row["member"], row["role"]), row, "member", ALWAYS,
                           NEVER) for row in columns(rows) if rows["set"]]
    return found


@timed("verify.sampled")
def sampled(facts, times, seed):
    moments = np.unique(np.asarray([found.first for found in facts if found.identity[0] not in TIMELESS], np.int64))
    order = np.lexsort((moments, unit(seed, SALT, moments)))
    return sorted(moments[order][:times].tolist())


@timed("verify.compared")
def compared(sides, sample):
    found = {}
    for at in sample:
        shown_ = [{}, {}]
        for side, facts in zip(shown_, sides):
            for held_ in facts:
                if held_.first <= at and (at < held_.end or at == held_.first):
                    side.setdefault(held_.identity, []).append(held_.content)
        for identity in shown_[0].keys() | shown_[1].keys():
            pair = tuple(tuple(sorted(side.get(identity, ()))) for side in shown_)
            if pair[0] != pair[1]:
                found.setdefault((identity, pair), []).append(at)
    return found


class Origins:
    @timed("verify.Origins.init")
    def __init__(self, mapped):
        self.rows, self.owners, self.order = {}, {}, list(mapped.errors.order)
        self.places = {label: list(dict.fromkeys(mapped.placed(statement))) for label, statement in mapped.statements}
        for label, statement in mapped.statements:
            stated_ = mapped.stated[id(statement)]
            self.owners.setdefault(label, label)
            if statement.word == "graphs":
                self.owners.setdefault("graph", label)
            for identity, _, _, held_ in stated_.held(mapped):
                if len(identity) == 3:
                    self.add(("node",) + identity + (held_["since"],), label, held_["row"])
            if isinstance(stated_, NodeStated):
                self.linked(label, stated_)
            if isinstance(stated_, EdgeStated):
                self.edged(label, stated_)
            if isinstance(stated_, Asked):
                tasks = [(task["graph"], task["place"], task["references"]) for task in stated_.decided] \
                    if isinstance(stated_, DecisionStated) else [(task[0], task[2], task[3]) for task in stated_.tasks]
                for graph, place, references in tasks:
                    self.add(("task", graph, label, stated_.at[place], ENCODER.encode(references)), label,
                             stated_.rows[place])

    @timed("verify.Origins.add")
    def add(self, identity, label, row):
        self.owners.setdefault(identity[2], label)
        found = self.rows.setdefault(identity, [])
        if (label, row) not in found:
            found.append((label, row))

    @timed("verify.Origins.linked")
    def linked(self, label, stated_):
        for name in stated_.links:
            self.owners.setdefault(name, label)
        for relation, source, target, held_ in stated_.linked:
            self.add(("edge", held_["graph"], relation, None, source, target, held_["since"]), label, held_["row"])

    @timed("verify.Origins.edged")
    def edged(self, label, stated_):
        for place, held_, source, target in stated_.rows:
            relation = None if stated_.relations is None else stated_.relations[place]
            ends = [(source, target)] if stated_.fields["directed"] else [(source, target), (target, source)]
            for first, second in ends:
                self.add(("edge", held_["graph"], label, relation, first, second, held_["since"]), label,
                         held_["row"])
        for (graph, key), entries in stated_.index.items():
            for held_, _, _, _ in entries:
                self.add(("node", graph, label + ".edge", key, held_["since"]), label, held_["row"])

    @timed("verify.Origins.rank")
    def rank(self, label):
        return self.order.index(label) if label in self.order else len(self.order)

    @timed("verify.Origins.found")
    def found(self, identity):
        rows = self.rows.get(identity, [])
        if rows:
            label = min((found[0] for found in rows), key=self.rank)
            return label, tuple(sorted(row for owner, row in rows if owner == label and row is not None))
        name = identity[1] if identity[0] == "graph" else identity[2]
        return self.owners.get(name), ()


@timed("verify.named")
def named(identity, graphs, model):
    family, graph = identity[:2]
    where = " of the graph %s" % graph if graphs > 1 else ""
    if family == "graph":
        return "the graph %s" % graph
    if family == "member":
        return "the member %s of the set edge %s of the kind %s in the role %s%s" % (
            identity[4], identity[3], identity[2], identity[5], where)
    if family == "task":
        text = "the task of the kind %s%s naming %s" % (identity[2], where, identity[4])
        return text if model == "none" else "%s at %s" % (text, when(identity[3]))
    if family == "node":
        text = "the node %s of the kind %s%s" % (identity[3], identity[2], where)
    else:
        relation = identity[2] if identity[3] is None else "%s %s" % (identity[2], identity[3])
        text = "the edge of %s from %s to %s%s" % (relation, identity[4][1], identity[5][1], where)
    return text if model == "none" else "%s from %s" % (text, when(identity[-1]))


@timed("verify.told")
def told(text, pair):
    source, build = pair
    if not build:
        return "the source states %s, which the build does not hold" % text
    if not source:
        return "the build holds %s, which no row of the source states" % text
    if len(source) != len(build):
        return "the source states %s %d times and the build holds it %d times" % (text, len(source), len(build))
    parts = []
    for first, second in zip(source, build):
        first, second = json.loads(first), json.loads(second)
        parts += ["%s is %s in the source and %s in the build" % (
            "the column " + name[len(VALUE):] if name.startswith(VALUE) else name, spelled(first.get(name)),
            spelled(second.get(name)))
            for name in sorted(first.keys() | second.keys()) if first.get(name) != second.get(name)]
    return "%s: %s" % (text, "; ".join(parts))


@timed("verify.parts")
def parts(source, build, prefix=""):
    found = []
    for name in sorted(source.keys() | build.keys()):
        if name in source and name in build and isinstance(source[name], dict) and isinstance(build[name], dict) \
                and not prefix:
            found += parts(source[name], build[name], name + ".")
        elif name not in source or name not in build or source[name] != build[name]:
            found.append(prefix + name)
    return found


class Verified(namedtuple("Verified", ("bank", "model", "times", "differences"))):
    @timed("verify.Verified.str")
    def __str__(self):
        sampled_ = "the graphs hold no time and are compared once" if self.model == "none" else \
            "at the %d sampled times %s" % (len(self.times), ", ".join(shown(at) for at in self.times))
        if not self.differences:
            return "the build of the bank %s holds what its sources hold, %s" % (self.bank, sampled_)
        lines = ["the sources differ from the build of the bank %s in %d %s, %s" % (
            self.bank, len(self.differences), "fact" if len(self.differences) == 1 else "facts", sampled_)]
        held_ = {}
        for difference in self.differences:
            held_.setdefault(difference.declaration, []).append(difference)
        for declaration, found in held_.items():
            places = list(dict.fromkeys(place for difference in found for place in difference.places))
            lines.append("%s%s" % ("the script" if declaration is None else "declaration " + declaration,
                                   ", reading " + ", ".join(places) if places else ""))
            for difference in found:
                head = ["row %s" % ", ".join(map(str, difference.rows))] if difference.rows else []
                if difference.times:
                    head.append("seen at %s" % ", ".join(shown(at) for at in difference.times))
                lines.append("  " + ": ".join([", ".join(head), difference.message] if head else [difference.message]))
        return "\n".join(lines)


@timed("verify.built")
def built(bank):
    root = Path(bank).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["source_fingerprint"] is None:
        raise ValueError("the bank %s was built from no staged output, and verify compares the declarations with "
                         "the staged graph a bank is built from" % root)
    staging = Path(manifest["database"])
    staged = verified(staging, manifest["source_fingerprint"])
    graphs = [read_json(staging / graph["records"]) for graph in staged["graphs"]]
    return root, manifest["task"]["name"], stated(staged["plan"]), dict(schema=staged["schema"], graphs=graphs)


@timed("verify.verify")
def verify(builder, bank, times, seed):
    started = time.perf_counter()
    if type(times) is not int or times < 1:
        raise ValueError("verify states times, the number of times of the graph it samples, as a whole number above "
                         "zero, found %r" % (times,))
    if type(seed) is not int:
        raise ValueError("verify states seed, the whole number its sample of times is keyed by, found %r" % (seed,))
    if not any(isinstance(value, (Mapping, Database)) for value in builder.declarations.values()):
        raise ValueError("verify reads the declarations of the builder again, and the builder holds no mapping; a "
                         "graph stated by calls has no source to read again")
    root, name, plan, build = built(bank)
    fresh = Builder(builder.declarations, builder.sources)
    origins = Origins(fresh.mapping(plan, name))
    source = json.loads(ENCODER.encode(canonical_records(fresh.records(snapshot=False))))
    model, sides = source["schema"]["time"]["model"], (held(source), held(build))
    sample = [ALWAYS] if model == "none" else sampled(sides[0] + sides[1], times, seed)
    differences = []
    if source["schema"] != build["schema"]:
        differences.append(Difference(None, (), (), (), "the declarations make a schema other than the build's, in "
                                      + ", ".join(parts(source["schema"], build["schema"]))))
    count = len({graph["name"] for case in (source, build) for graph in case["graphs"]})
    for (identity, pair), moments in compared(sides, sample).items():
        declaration, rows = origins.found(identity)
        differences.append(Difference(declaration, tuple(origins.places.get(declaration, ())), rows,
                                      tuple(moments) if model != "none" else (),
                                      told(named(identity, count, model), pair)))
    differences.sort(key=lambda found: (-1 if found.declaration is None else origins.rank(found.declaration),
                                        found.rows, found.message))
    result = Verified(str(root), model, tuple(sample) if model != "none" else (), tuple(differences))
    LOG.info("verify exit times=%d differences=%d seconds=%.4f", len(result.times), len(differences),
             time.perf_counter() - started)
    return result
