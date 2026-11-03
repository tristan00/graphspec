from collections.abc import Mapping as Held

from graphspec.infra.timing import timed
from graphspec.schema.targets import TARGETS

from .mapping_build import EdgeStated, End, SetStated, Stated
from .mapping_rows import Reader, joined, spelled
from .mappings import Mapping, tabled
from .presence import ruled

SKIP = object()
POINTS = ("tau", "series", "y")


@timed("mapping_tasks.plain")
def plain(value):
    if isinstance(value, Mapping):
        raise ValueError("a mapping is named here only where a field states it")
    if isinstance(value, Held):
        return {name: plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@timed("mapping_tasks.weighed")
def weighed(utility, series):
    result = plain(utility)
    if type(result.get("weights")) in (int, float):
        result["weights"] = {name: float(result["weights"]) for name in series}
    return result


@timed("mapping_tasks.agreed")
def agreed(values, where):
    if len({repr(value) for value in values}) > 1:
        raise ValueError("%s: the rows of one decision or option state different values" % where)
    return values[0] if values else None


@timed("mapping_tasks.one")
def one(reader, text, field, outcome=False):
    values, kind = reader.values(text, field, outcome)
    return agreed(values, "declaration %s field %s" % (reader.name, field)), kind


class Reference:
    @timed("mapping_tasks.Reference")
    def __init__(self, reader, name, stated):
        self.name, self.sort, self.stated = name, stated["sort"], stated
        self.where = "declaration %s field references.%s" % (reader.name, name)
        self.rows = [reader.row(place) for place in range(reader.count)]
        self.raw = [()] * reader.count
        if self.sort == "graph":
            return
        field = "references.%s." % name
        self.rule, self.parameters = ruled(stated["absent"], self.where + ".absent")
        if self.sort == "node":
            self.end = End(reader, stated, "references." + name)
            self.raw = self.end.keys
        elif self.sort in ("nodes", "subgraph"):
            self.keys = reader.listed(stated["keys"], field + "keys")
            self.raw = self.keys
            if self.sort == "subgraph":
                self.edged = reader.listed(stated["edges"]["keys"], field + "edges.keys")
        elif self.sort == "edge":
            self.keys = reader.texts(stated["key"], field + "key") if "key" in stated else None
            self.pairs = None if "key" in stated else list(zip(reader.texts(stated["source"], field + "source"),
                                                                reader.texts(stated["target"], field + "target")))
            self.raw = self.keys if self.keys is not None else self.pairs
        elif self.sort == "path":
            self.keys = reader.listed(stated["keys"], field + "keys") if "keys" in stated else None
            self.nodes = None if "keys" in stated else reader.listed(stated["nodes"], field + "nodes")
            self.raw = self.keys if self.keys is not None else self.nodes
        else:
            raise ValueError("%s is of the sort %s, which mappings do not state" % (self.where, self.sort))

    @timed("mapping_tasks.Reference.edges")
    def edges(self, mapped, of):
        stated = mapped.stated.get(id(of))
        if not isinstance(stated, EdgeStated):
            raise ValueError("%s names under of an edges statement" % self.where)
        return stated

    @timed("mapping_tasks.Reference.kinds")
    def kinds(self, mapped):
        if self.sort == "graph":
            return []
        if self.sort == "node":
            return [kind for target in self.end.targets for kind in mapped.kinds(target, self.where)]
        if self.sort in ("edge", "path"):
            return self.edges(mapped, self.stated["of"]).sources()
        return mapped.kinds(self.stated["of"], self.where)

    @timed("mapping_tasks.Reference.found")
    def found(self, mapped, place, graph, kinds, key, since):
        held = mapped.find(graph, kinds, key)
        if len(held) == 1:
            mapped.note(graph, held[0], key)
            return key
        if held:
            raise ValueError("%s row %d names the key %s, which several kinds hold" % (self.where, self.rows[place], key))
        stand = mapped.absent(self.where, self.rule, self.parameters, key, since, self.rows[place])
        if stand is not None and len(mapped.find(graph, kinds, stand)) != 1:
            raise ValueError("%s row %d: the rule %s gives %s, which names no one node"
                             % (self.where, self.rows[place], self.rule, stand))
        return stand

    @timed("mapping_tasks.Reference.between")
    def between(self, mapped, place, graph, edges, source, target, since):
        relations = edges.between.get((graph, source, target), [])
        if len(relations) > 1:
            raise ValueError("%s row %d names the edges from %s to %s of several relations"
                             % (self.where, self.rows[place], source, target))
        if relations:
            return [source, relations[0], target]
        stand = mapped.absent(self.where, self.rule, self.parameters, "%s-%s" % (source, target), since,
                              self.rows[place])
        if stand is not None:
            raise ValueError("%s: the rule %s stands no other edge for an edge" % (self.where, self.rule))
        return None

    @timed("mapping_tasks.Reference.keyed")
    def keyed(self, mapped, place, graph, edges, key, since):
        entries = edges.index.get((graph, key))
        if entries is None:
            if mapped.absent(self.where, self.rule, self.parameters, key, since, self.rows[place]) is not None:
                raise ValueError("%s: the rule %s stands no other edge for an edge" % (self.where, self.rule))
            return None
        _, source, target, relation = entries[0]
        return [source[1], relation, target[1]]

    @timed("mapping_tasks.Reference.resolve")
    def resolve(self, mapped, place, graph, since):
        if self.sort == "graph":
            return None
        if self.sort == "node":
            found = self.end.resolve(mapped, place, graph, dict(since=since))
            return SKIP if found is None else found[1]
        kinds = self.kinds(mapped)
        if self.sort in ("nodes", "subgraph"):
            if self.keys[place] is None:
                raise ValueError("%s row %d reads a missing value as keys" % (self.where, self.rows[place]))
            nodes = [self.found(mapped, place, graph, kinds, key, since) for key in self.keys[place]]
            nodes = [key for key in nodes if key is not None]
            if self.sort == "nodes":
                return nodes
            edges = self.edges(mapped, self.stated["edges"]["of"])
            found = [self.keyed(mapped, place, graph, edges, key, since) for key in self.edged[place] or ()]
            return dict(nodes=nodes, edges=[edge for edge in found if edge is not None])
        edges = self.edges(mapped, self.stated["of"])
        if self.sort == "edge":
            if self.keys is not None:
                found = self.keyed(mapped, place, graph, edges, joined(self.keys[place]), since)
            else:
                found = self.between(mapped, place, graph, edges, *self.pairs[place], since)
            return SKIP if found is None else found
        if self.keys is not None:
            found = [self.keyed(mapped, place, graph, edges, key, since) for key in self.keys[place]]
        else:
            stops = self.nodes[place]
            found = [self.between(mapped, place, graph, edges, source, target, since)
                     for source, target in zip(stops, stops[1:])]
        return SKIP if None in found or not found else found


class Asked(Stated):
    @timed("mapping_tasks.Asked.read")
    def read(self, mapped, label, statement, session):
        fields = self.fields = statement.fields
        self.label, self.where = label, "declaration " + label
        if not tabled(statement.table):
            raise ValueError("%s states its rows by a rule, which mappings do not yet serve" % self.where)
        for name in ("earlier", "eligible"):
            if name in fields:
                raise ValueError("%s states %s, which mappings do not yet serve" % (self.where, name))
        if fields.get("split") is not None and type(fields["split"]) is not str:
            raise ValueError("%s states its split by a form, which mappings do not yet serve; an expression serves"
                             % self.where)
        reader = self.reader = Reader(session, statement.table, label, mapped.plan).where(fields.get("where"))
        count = reader.count
        self.rows = [reader.row(place) for place in range(count)]
        self.graphs = reader.texts(fields["graph"], "graph") if "graph" in fields else [None] * count
        if mapped.timed and "at" not in fields:
            raise ValueError("%s lacks at, which every kind of task states where the graphs have time" % self.where)
        self.at = reader.moments(fields["at"], "at") if mapped.timed else [None] * count
        self.split = reader.texts(fields["split"], "split") if fields.get("split") else [None] * count
        self.observer = reader.texts(fields["observer"], "observer", "omit") if "observer" in fields else [None] * count
        self.grouping = reader.moments(fields["grouping"], "grouping") if "grouping" in fields else [None] * count
        self.weight = [None] * count
        if "weight" in fields:
            values, _ = reader.values(fields["weight"], "weight")
            if any(value is None or not value > 0 for value in values):
                raise ValueError("%s field weight gives a number above zero in every row" % self.where)
            self.weight = [float(value) for value in values]
        self.width, self.signal = 0, [None] * count
        if "signals" in fields:
            values, kind = reader.values(fields["signals"], "signals")
            if kind.name != "vector":
                raise ValueError("%s field signals gives a vector" % self.where)
            self.width = kind.width
            self.signal = [[None] * kind.width if value is None else list(value) for value in values]
        self.columns, self.values = reader.features(fields.get("features"))
        self.references = {name: Reference(reader, name, stated) for name, stated in fields["references"].items()}

    @timed("mapping_tasks.Asked.placed")
    def placed(self, mapped, place):
        graphs = mapped.graphed(self.graphs[place])
        if self.graphs[place] is None and len(graphs) > 1:
            nodes = [reference for reference in self.references.values() if reference.sort == "node"]
            graphs = [graph for graph in graphs if all(
                len(mapped.find(graph, reference.kinds(mapped), reference.end.keys[place])) == 1 for reference in nodes)]
        return graphs

    @timed("mapping_tasks.Asked.referred")
    def referred(self, mapped, place, graph):
        found = {}
        for name, reference in self.references.items():
            found[name] = reference.resolve(mapped, place, graph, self.at[place])
            if found[name] is SKIP:
                return None
        return found

    @timed("mapping_tasks.Asked.declared")
    def declared(self, mapped, target):
        for name, reference in self.references.items():
            entry = mapped.references.setdefault(name, dict(sort=reference.sort, kinds=[]))
            if entry["sort"] != reference.sort:
                raise ValueError("%s names the reference %s of the sort %s, which another kind of task names of the "
                                 "sort %s" % (self.where, name, reference.sort, entry["sort"]))
            entry["kinds"].extend(kind for kind in self.kinds[name] if kind not in entry["kinds"])
        for name, spec in self.columns.items():
            mapped.column(name, spec, self.label)
        asked = [statement.fields for _, statement in mapped.statements if statement.word in ("tasks", "decisions")]
        if any("grouping" in fields for fields in asked) != ("grouping" in self.fields):
            raise ValueError("%s: every kind of task states grouping, or none" % self.where)
        if "grouping" in self.fields:
            mapped.grouping = "grouping"
        if mapped.signals and self.width and mapped.signals != self.width:
            raise ValueError("%s states signals of a width other kinds of task do not" % self.where)
        mapped.signals = mapped.signals or self.width
        mapped.tasks[self.label] = dict(
            references=list(self.references), columns=list(self.columns), target=target, weight="weight" in self.fields,
            absent="stop", event=plain(self.fields.get("event")), matures=plain(self.fields.get("matures")))

    @timed("mapping_tasks.Asked.task")
    def task(self, mapped, graph, key, place, references, target, known):
        builder, extra = mapped.builder, {}
        if mapped.model != "none":
            extra["at"] = self.at[place]
        if mapped.observers:
            extra["observer"] = self.observer[place]
        if mapped.grouping is not None:
            extra["grouping"] = self.grouping[place]
        if mapped.signals:
            signal = self.signal[place] if self.width else [None] * mapped.signals
            extra.update(signal=[0.0 if value is None else float(value) for value in signal],
                         signal_observed=[value is not None for value in signal])
        builder.task(self.label, key, references=references, split=self.split[place], target=target, known=known,
                     graph=graph, weight=self.weight[place], values=self.values[place], **extra)


class TaskStated(Asked):
    @timed("mapping_tasks.TaskStated")
    def __init__(self, mapped, label, statement, session):
        self.read(mapped, label, statement, session)
        reader, target = self.reader, self.fields.get("target")
        self.keys = reader.texts(self.fields["key"], "key") if "key" in self.fields else [str(row) for row in self.rows]
        self.target = None if target is None else plain({name: value for name, value in target.items()
                                                         if not isinstance(value, Mapping)})
        self.aims = [None] * reader.count
        self.known = [None] * reader.count
        if target is None:
            return
        if "later" in target:
            raise ValueError("%s field target states later, which mappings do not yet serve" % self.where)
        self.kind = target["kind"]
        self.target.update({name: mapped.label(value) for name, value in target.items() if isinstance(value, Mapping)})
        self.sets = target.get("sets")
        self.aims = self.aimed(reader, target)
        if target["known"] != "never" and mapped.model != "none":
            self.known = reader.moments(target["known"], "target.known", outcome=True)

    @timed("mapping_tasks.TaskStated.aimed")
    def aimed(self, reader, target):
        value, field, kind = target["value"], "target.value", self.kind
        if kind == "survival":
            lasted = reader.moments(value["duration"], field + ".duration", "omit", True)
            happened, _ = reader.values(value["happened"], field + ".happened", True)
            return [None if None in pair else dict(duration=pair[0], happened=pair[1]) for pair in zip(lasted, happened)]
        if kind == "set":
            roles = {role: reader.listed(text, "%s.%s" % (field, role), True) for role, text in value.items()}
            return [None if any(keys[place] is None for keys in roles.values())
                    else {role: keys[place] for role, keys in roles.items()} for place in range(reader.count)]
        if kind in ("classes", "sequence", "nodes"):
            return reader.listed(value, field, True)
        values, _ = reader.values(value, field, True)
        if kind == "number" or kind == "ranking":
            return [None if item is None else float(item) if kind == "number" else [float(part) for part in item]
                    for item in values]
        if kind == "proportions":
            return [None if item is None else {part["name"]: part["weight"] for part in item} for item in values]
        if kind == "path":
            return [None if item is None else [[part[name] for name in POINTS] for part in item] for item in values]
        return [spelled(item) for item in values]

    @timed("mapping_tasks.TaskStated.adapted")
    def adapted(self, mapped, graph, aim, references):
        if aim is None or self.kind not in ("nodes", "ranking", "set"):
            return aim
        if self.kind == "set":
            stated = mapped.stated[id(self.sets)]
            if not isinstance(stated, SetStated):
                raise ValueError("%s field target.sets names a sets statement" % self.where)
            kinds = list(dict.fromkeys(row[2][0] for row in stated.rows))
            result = {}
            for role, keys in aim.items():
                for key in keys:
                    found = mapped.find(graph, kinds, key)
                    if len(found) != 1:
                        raise ValueError("%s field target.value names the member %s, which no one node is"
                                         % (self.where, key))
                    result.setdefault(role, []).append([found[0], key])
            return result
        listed = references[self.target["about"]]
        if self.kind == "nodes":
            return {key: key in aim for key in listed}
        if len(aim) != len(listed):
            raise ValueError("%s field target.value gives a relevance for every key of %s"
                             % (self.where, self.target["about"]))
        return dict(zip(listed, aim))

    @timed("mapping_tasks.TaskStated.resolve")
    def resolve(self, mapped):
        self.kinds = {name: reference.kinds(mapped) for name, reference in self.references.items()}
        self.tasks = []
        for place in range(len(self.rows)):
            for graph in self.placed(mapped, place):
                references = self.referred(mapped, place, graph)
                if references is None:
                    continue
                key = mapped.numbered(graph, self.label, joined(self.keys[place]), self.where)
                self.tasks.append((graph, key, place, references,
                                   self.adapted(mapped, graph, self.aims[place], references)))

    @timed("mapping_tasks.TaskStated.schema")
    def schema(self, mapped):
        target = None
        if self.target is not None:
            entry = TARGETS.named(self.kind)
            target = dict(kind=self.kind, about=self.target["about"], utility=self.target["utility"],
                          **{name: self.target[name] for name in entry.parameters})
            if self.kind in ("class", "classes", "proportions", "sequence"):
                names = set()
                for aim in self.aims:
                    if aim is not None:
                        names.update(aim if isinstance(aim, (list, dict)) else [aim])
                target["series"] = sorted(names)
            else:
                if "series" not in self.target:
                    raise ValueError("%s field target lacks series" % self.where)
                target["series"] = self.target["series"]
            series = target["series"]
            target["utility"] = weighed(target["utility"], [series] if type(series) is str else series)
        self.declared(mapped, target)

    @timed("mapping_tasks.TaskStated.issue")
    def issue(self, mapped, phase):
        if phase == "tasks":
            for graph, key, place, references, aim in self.tasks:
                known = self.known[place] if aim is not None else None
                self.task(mapped, graph, key, place, references, aim, known)


class DecisionStated(Asked):
    @timed("mapping_tasks.DecisionStated")
    def __init__(self, mapped, label, statement, session):
        self.read(mapped, label, statement, session)
        fields, reader = self.fields, self.reader
        self.series = [fields["series"]] if type(fields["series"]) is str else list(fields["series"])
        keys = reader.texts(fields["key"], "key")
        self.ended = reader.values(fields["ended"], "ended")[0] if "ended" in fields else None
        self.discount = reader.values(fields["discount"], "discount")[0] if "discount" in fields else [None] * reader.count
        self.concurrent = reader.values(fields["concurrent"], "concurrent")[0] if "concurrent" in fields else None
        self.groups = reader.grouped([joined(key) for key in keys])
        for key, places in self.groups.items():
            where = "%s decision %s" % (self.where, key)
            for name, values in (("at", self.at), ("graph", self.graphs), ("split", self.split),
                                 ("observer", self.observer), ("grouping", self.grouping), ("weight", self.weight),
                                 ("signals", self.signal), ("features", self.values), ("discount", self.discount),
                                 ("ended", self.ended or [None] * reader.count),
                                 ("concurrent", self.concurrent or [None] * reader.count)):
                agreed([values[place] for place in places], "%s field %s" % (where, name))
            for name, reference in self.references.items():
                agreed([reference.raw[place] for place in places], "%s field references.%s" % (where, name))
        self.options = self.optioned(session, mapped)
        self.nothing = self.nothinged(mapped)
        self.actions = self.acted(session, mapped)

    @timed("mapping_tasks.DecisionStated.option")
    def option(self, reader, stated, field, key):
        chosen, _ = one(reader, stated["chosen"], field + ".chosen", True)
        available, _ = one(reader, stated["available"], field + ".available")
        if type(chosen) is not bool or type(available) is not bool:
            raise ValueError("%s field %s states chosen and available as flags in every row" % (self.where, field))
        held = dict(key=key, chosen=chosen, available=available, together=None, order=None, shown=None, logged=None)
        if stated.get("together") is not None:
            held["together"] = one(reader, stated["together"], field + ".together")[0]
        if stated.get("order") is not None:
            order = one(reader, stated["order"], field + ".order", True)[0]
            held["order"] = None if order is None else int(order)
        for name in ("shown", "logged"):
            if stated.get(name) is not None:
                value = one(reader, stated[name], "%s.%s" % (field, name), True)[0]
                held[name] = None if value is None else float(value)
        outcome = {series: one(reader, text, "%s.outcome.%s" % (field, series), True)[0]
                   for series, text in (stated.get("outcome") or {}).items()}
        held["outcome"] = None if not outcome or None in outcome.values() else {
            series: float(value) for series, value in outcome.items()}
        held["known"] = None
        if stated.get("known") and reader.count:
            held["known"] = agreed(reader.moments(stated["known"], field + ".known", "omit", True),
                                   "declaration %s field %s.known" % (self.where, field))
        return held

    @timed("mapping_tasks.DecisionStated.optioned")
    def optioned(self, session, mapped):
        stated = self.fields["options"]
        reader = Reader(session, stated["rows"], self.label, mapped.plan)
        decided = [joined(key) for key in reader.texts(stated["decision"], "options.decision")]
        keys = reader.texts(stated["key"], "options.key")
        groups = reader.grouped(list(zip(decided, keys)))
        return {identity: self.option(reader.taken(places), stated, "options", identity[1])
                for identity, places in groups.items()}

    @timed("mapping_tasks.DecisionStated.nothinged")
    def nothinged(self, mapped):
        stated = self.fields.get("nothing")
        if stated is None:
            return None, {}
        if type(stated) is str:
            return spelled(one(self.reader, stated, "nothing")[0]), {}
        found = {}
        for key, places in self.groups.items():
            reader = self.reader.taken(places)
            found[key] = self.option(reader, stated, "nothing", spelled(one(reader, stated["key"], "nothing.key")[0]))
        return next(iter(found.values()))["key"] if found else None, found

    @timed("mapping_tasks.DecisionStated.acted")
    def acted(self, session, mapped):
        stated = self.fields["actions"]
        reader = Reader(session, stated["rows"], self.label, mapped.plan)
        decided = [joined(key) for key in reader.texts(stated["decision"], "actions.decision")]
        options = reader.texts(stated["option"], "actions.option")
        kinds = reader.texts(stated["kind"], "actions.kind")
        actors = reader.texts(stated["actor"], "actions.actor")
        places = reader.moments(stated["place"], "actions.place") if stated.get("place") else [None] * reader.count
        outcomes = {series: reader.values(text, "actions.outcome." + series, True)[0]
                    for series, text in (stated.get("outcome") or {}).items()}
        self.parameters, found = {}, [None] * reader.count
        for kind, rows in reader.grouped(kinds).items():
            if kind not in stated["kinds"]:
                raise ValueError("%s field actions.kind gives %s, which kinds does not state" % (self.where, kind))
            spec = stated["kinds"][kind]
            sub = reader.taken(rows)
            columns, values = sub.features(spec.get("parameters"), "actions.kinds.%s.parameters" % kind)
            self.parameters[kind] = columns
            roles = {}
            for role, aim in (spec.get("roles") or {}).items():
                field = "actions.kinds.%s.roles.%s" % (kind, role)
                if aim["sort"] == "edge":
                    roles[role] = list(zip(sub.listed(aim["sources"], field + ".sources"),
                                           sub.listed(aim["targets"], field + ".targets")))
                else:
                    roles[role] = sub.listed(aim["keys"], field + ".keys")
            for at, row in enumerate(rows):
                found[row] = dict(kind=kind, actor=actors[row], parameters=values[at],
                                  targets={role: held[at] or [] for role, held in roles.items()},
                                  place=places[row], outcome=None if not outcomes or None in [
                                      values[row] for values in outcomes.values()] else {
                                      series: float(values[row]) for series, values in outcomes.items()})
        return reader.grouped(list(zip(decided, options))), found

    @timed("mapping_tasks.DecisionStated.targets")
    def targets(self, mapped, graph, action):
        result, spec = {}, self.fields["actions"]["kinds"][action["kind"]]
        for role, held in action["targets"].items():
            aim = spec["roles"][role]
            where = "%s field actions.kinds.%s.roles.%s" % (self.where, action["kind"], role)
            if aim["sort"] == "new":
                result[role] = list(held)
            elif aim["sort"] == "edge":
                edges = mapped.stated[id(aim["of"])]
                found = []
                sources, targets = held if held else ((), ())
                for source, target in zip(sources or (), targets or ()):
                    relations = edges.between.get((graph, source, target), [])
                    if len(relations) != 1:
                        raise ValueError("%s names the edge from %s to %s, which does not resolve"
                                         % (where, source, target))
                    found.append([source, relations[0], target])
                result[role] = found
            else:
                kinds = mapped.kinds(aim["of"], where)
                for key in held:
                    if len(mapped.find(graph, kinds, key)) != 1:
                        raise ValueError("%s names the target %s, which does not resolve" % (where, key))
                result[role] = list(held)
        return result

    @timed("mapping_tasks.DecisionStated.resolve")
    def resolve(self, mapped):
        self.kinds = {name: reference.kinds(mapped) for name, reference in self.references.items()}
        groups, actions = self.actions
        nothing, made = self.nothing
        self.decided = []
        for key, places in self.groups.items():
            place = places[0]
            for graph in self.placed(mapped, place):
                references = self.referred(mapped, place, graph)
                if references is None:
                    continue
                actors = references[self.fields["actors"]]
                actors = actors if isinstance(actors, list) else [actors]
                options = []
                for (decided, option), held in self.options.items():
                    if decided != key:
                        continue
                    listed = [actions[row] for row in groups.get((decided, option), [])]
                    if nothing is not None and option == nothing and listed:
                        raise ValueError("%s: an action names the option %s, which does nothing" % (self.where, option))
                    options.append(self.completed(mapped, graph, held, listed, actors))
                if key in made:
                    options.append(self.completed(mapped, graph, made[key], [], actors))
                number = mapped.numbered(graph, self.label, key, self.where)
                self.decided.append(dict(graph=graph, key=number, place=place, references=references, actors=actors,
                                         options=options))
        self.sequenced()

    @timed("mapping_tasks.DecisionStated.completed")
    def completed(self, mapped, graph, held, listed, actors):
        together = held["together"]
        if together is None:
            if len(listed) > 1:
                raise ValueError("%s: an option of several actions states together" % self.where)
            together = "set"
        if together == "sequence":
            listed = sorted(listed, key=lambda action: action["place"])
        for action in listed:
            if action["actor"] not in actors:
                raise ValueError("%s: an action names the actor %s, who is no actor of its decision"
                                 % (self.where, action["actor"]))
        parts = [action["outcome"] for action in listed]
        outcome = held["outcome"]
        if any(part is not None for part in parts):
            if outcome is not None:
                raise ValueError("%s states an outcome of an option and of its actions" % self.where)
            outcome = parts
        return dict(key=held["key"], together=together, available=held["available"], chosen=held["chosen"],
                    order=held["order"], shown=held["shown"], logged=held["logged"], origin="recorded",
                    outcome=outcome, known=held["known"],
                    actions=[dict(kind=action["kind"], actor=action["actor"], targets=self.targets(mapped, graph, action),
                                  parameters=dict(action["parameters"])) for action in listed])

    @timed("mapping_tasks.DecisionStated.sequenced")
    def sequenced(self):
        order = sorted(self.decided, key=lambda decision: (decision["graph"], self.at[decision["place"]] or 0,
                                                           decision["key"]))
        for decision in self.decided:
            place = decision["place"]
            decision["next"] = {}
            if self.ended is not None and self.ended[place] is not True:
                for actor in decision["actors"]:
                    later = [other for other in order[order.index(decision) + 1:]
                             if other["graph"] == decision["graph"] and actor in other["actors"]]
                    if later:
                        decision["next"][actor] = later[0]["key"]
            decision["concurrent"] = [] if self.concurrent is None else [
                other["key"] for other in order if other is not decision and other["graph"] == decision["graph"]
                and self.at[other["place"]] == self.at[place]
                and repr(self.concurrent[other["place"]]) == repr(self.concurrent[place])
                and not set(other["actors"]) & set(decision["actors"])]

    @timed("mapping_tasks.DecisionStated.schema")
    def schema(self, mapped):
        actions = {}
        for kind, spec in self.fields["actions"]["kinds"].items():
            roles = {}
            for role, aim in (spec.get("roles") or {}).items():
                kinds = mapped.stated[id(aim["of"])].sources() if aim["sort"] == "edge" else \
                    mapped.kinds(aim["of"], "%s field actions.kinds.%s.roles.%s" % (self.where, kind, role))
                roles[role] = dict(sort=aim["sort"], kinds=kinds, count=plain(aim["count"]))
            for name, column in self.parameters.get(kind, {}).items():
                mapped.column(name, column, self.label)
            actions[kind] = dict(roles=roles, parameters=list(self.parameters.get(kind, {})))
        fields = self.fields
        target = dict(kind="decision", series=self.series, about=fields["about"],
                      utility=weighed(fields["utility"], self.series),
                      actors=fields["actors"], actions=actions, nothing=self.nothing[0],
                      rule=plain(fields.get("rule")), generators=plain(fields.get("generators") or []))
        self.declared(mapped, target)

    @timed("mapping_tasks.DecisionStated.issue")
    def issue(self, mapped, phase):
        if phase != "tasks":
            return
        for decision in self.decided:
            place = decision["place"]
            known = [option.pop("known") for option in decision["options"]]
            known = [moment for moment, option in zip(known, decision["options"])
                     if moment is not None and option["outcome"] is not None]
            target = dict(options=decision["options"], concurrent=decision["concurrent"], next=decision["next"],
                          ended=None if self.ended is None else self.ended[place],
                          discount=None if self.discount[place] is None else float(self.discount[place]))
            self.task(mapped, decision["graph"], decision["key"], place, decision["references"], target,
                      max(known) if known and mapped.model != "none" else None)
