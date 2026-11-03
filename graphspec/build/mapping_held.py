import logging
import time

from graphspec.infra.timing import timed
from graphspec.schema.fields import named

from .mapping_build import End, Stated
from .mapping_rows import Reader, joined
from .mappings import Mapping

LOG = logging.getLogger(__name__)


@timed("mapping_held.holders")
def holders(fields):
    found = fields["inside"]["of"]
    return tuple(id(target) for target in (found if isinstance(found, (list, tuple)) else [found]))


class Inside(Stated):
    @timed("mapping_held.Inside.read")
    def read(self, mapped, label, statement, session):
        self.fields, self.label, self.statement = statement.fields, label, statement
        reader = Reader(session, statement.table, label, mapped.plan).where(self.fields.get("where"))
        self.graphs = reader.texts(self.fields["graph"], "graph") if "graph" in self.fields else [None] * reader.count
        for graph in self.graphs:
            if graph is not None:
                named(graph, "declaration %s field graph" % label)
        self.inside = End(mapped, reader, self.fields["inside"], "inside", "inside")
        for target in self.inside.targets:
            if not isinstance(target, Mapping) or target.word != "nodes" or "held" not in target.fields:
                raise ValueError("declaration %s field inside names under of a nodes statement that states held" % label)
        self.columns, self.values = reader.features(self.fields.get("features"))
        self.reader = reader
        return reader

    @timed("mapping_held.Inside.owned")
    def owned(self, mapped, place):
        graphs = mapped.graphed(self.graphs[place])
        if self.graphs[place] is None and len(graphs) > 1:
            graphs = [graph for graph in graphs if len(
                mapped.find(graph, self.inside.kinds(mapped, place), self.inside.keys[place])) == 1] or [None]
        for graph in graphs:
            found = self.inside.resolve(mapped, place, graph, dict(since=None, graph=graph))
            if found is not None:
                yield graph, found

    @timed("mapping_held.Inside.holding")
    def holding(self, mapped):
        return [kind for target in self.inside.targets for kind in mapped.kinds(target, self.inside.where)]


class HeldNodes(Inside):
    @timed("mapping_held.HeldNodes")
    def __init__(self, mapped, label, statement, session):
        reader = self.read(mapped, label, statement, session)
        self.keys = reader.texts(self.fields["key"], "key")
        chosen = reader.texts(self.fields["kind"], "kind") if "kind" in self.fields else [label] * reader.count
        self.chosen = [named(kind, "declaration %s field kind" % label) for kind in chosen]
        self.made_kinds = list(dict.fromkeys(self.chosen)) if "kind" in self.fields else [label]

    @timed("mapping_held.HeldNodes.kinds")
    def kinds(self, where):
        raise ValueError("%s names under of the nodes %s, held inside a holder; only the edges inside the same holder "
                         "name them" % (where, self.label))

    @timed("mapping_held.HeldNodes.resolve")
    def resolve(self, mapped):
        started = time.perf_counter()
        self.rows = []
        for place in range(self.reader.count):
            key = joined(self.keys[place])
            for graph, (holder, owner) in self.owned(mapped, place):
                inner = mapped.inner.setdefault((graph, holder, owner), {})
                if key in inner:
                    raise ValueError("declaration %s row %d states the held node %s inside %s %s twice; a held node's "
                                     "key is unique within its holder"
                                     % (self.label, self.reader.row(place), key, holder, owner))
                inner[key] = self.chosen[place]
                self.rows.append((graph, holder, owner, self.chosen[place], key, self.values[place]))
        LOG.debug("held nodes resolve exit declaration=%s rows=%d seconds=%.4f", self.label, len(self.rows),
                  time.perf_counter() - started)

    @timed("mapping_held.HeldNodes.schema")
    def schema(self, mapped):
        for name, spec in self.columns.items():
            mapped.column(name, spec, self.label)
        for holder in self.holding(mapped):
            for kind in self.made_kinds:
                columns = mapped.holdings[holder]["nodes"].setdefault(kind, dict(columns=[]))["columns"]
                columns += [name for name in self.columns if name not in columns]

    @timed("mapping_held.HeldNodes.issue")
    def issue(self, mapped, phase):
        if phase != "edges":
            return
        holds = mapped.builder.required().holds
        for graph, holder, owner, kind, key, values in self.rows:
            mapped.builder.held(holder, owner, kind, key, graph=graph,
                                values={name: values.get(name) for name in holds[holder]["nodes"][kind]})


class HeldEdges(Inside):
    @timed("mapping_held.HeldEdges")
    def __init__(self, mapped, label, statement, session):
        reader = self.read(mapped, label, statement, session)
        self.ends = [End(mapped, reader, self.fields[name], name, "held ends") for name in ("source", "target")]

    @timed("mapping_held.HeldEdges.allowed")
    def allowed(self, mapped, end, place):
        kinds = []
        for target in end.targets:
            stated = mapped.stated.get(id(target))
            if not isinstance(stated, HeldNodes) or mapped.order[id(target)] >= mapped.order[id(self.statement)] \
                    or holders(stated.fields) != holders(self.fields):
                raise ValueError("%s names under of a nodes statement stated above it inside the same holder"
                                 % end.where)
            if end.chosen is None or mapped.label(target) == end.chosen[place]:
                kinds += stated.made_kinds
        return kinds

    @timed("mapping_held.HeldEdges.found")
    def found(self, mapped, end, place, inner):
        kinds, key = self.allowed(mapped, end, place), joined(end.keys[place])
        if inner.get(key) in kinds:
            return key, inner[key]
        stand = mapped.absent(end, key, None, end.rows[place])
        if stand is None:
            return None
        if inner.get(stand) not in kinds:
            raise ValueError("%s row %d: the rule %s gives the key %s, which names no held node of its holder"
                             % (end.where, end.rows[place], end.rule, stand))
        return stand, inner[stand]

    @timed("mapping_held.HeldEdges.resolve")
    def resolve(self, mapped):
        started = time.perf_counter()
        self.rows = []
        for place in range(self.reader.count):
            for graph, (holder, owner) in self.owned(mapped, place):
                inner = mapped.inner.get((graph, holder, owner), {})
                source = self.found(mapped, self.ends[0], place, inner)
                target = self.found(mapped, self.ends[1], place, inner) if source is not None else None
                if target is not None:
                    self.rows.append((graph, holder, owner, source, target, self.values[place]))
        LOG.debug("held edges resolve exit declaration=%s rows=%d seconds=%.4f", self.label, len(self.rows),
                  time.perf_counter() - started)

    @timed("mapping_held.HeldEdges.schema")
    def schema(self, mapped):
        for name, spec in self.columns.items():
            mapped.column(name, spec, self.label)
        directed = self.fields["directed"]
        for _, holder, _, source, target, _ in self.rows:
            entry = mapped.holdings[holder]["edges"].setdefault(
                self.label, dict(endpoints=[], directed=directed, columns=list(self.columns)))
            pair = [source[1], target[1]]
            if pair not in entry["endpoints"] and (directed or pair[::-1] not in entry["endpoints"]):
                entry["endpoints"].append(pair)

    @timed("mapping_held.HeldEdges.issue")
    def issue(self, mapped, phase):
        if phase != "edges":
            return
        for graph, holder, owner, source, target, values in self.rows:
            mapped.builder.held_edge(holder, owner, self.label, source[0], target[0], graph=graph, values=values)
