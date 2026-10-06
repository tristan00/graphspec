import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from graphspec.descriptions import describe as truth
from graphspec.descriptions import effect
from graphspec.facts.stored import GROUPS, Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import load_arrays, sha256_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import sets
from graphspec.schema.decisions import ROLES
from graphspec.views import visible
from graphspec.views.view import sequences
from tuning.bank import forks
from tuning.bank.bank import Deriver, editing
from tuning.bank.options import offered
from tuning.bank.routes import routed
from tuning.batches.inputs import aimed
from tuning.families.heads import REALISATION, fielded, realisation
from tuning.families.model_spec import realised
from tuning.statement import agreed, state

from . import restated
from .restated import ABSENT, KEPT, REFUSED, STOPPED, STOPS, Derived, halted, masked, values

LOG = logging.getLogger(__name__)
HELD, PUT, JOINED, UNJOINED, UNHELD, CREATED = "held", "put", "joined", "unjoined", "unheld", "created"
OPTION, ROLE, TARGET, FACT = "option", "role", "target", "fact"
OBSERVATION, EDIT, ACTOR = "observation", "edit", "actor"
BACK, STILL, UNTRIED = "back", "still", "untried"
KNOWN, OBSERVER, SEES = "known", "observer", "sees"
ROUTES = {
    OPTION: "with the option",
    ROLE: "with the option, by its role alone",
    TARGET: "as the targets the variant holds",
    FACT: "as the facts its edit makes of it",
}
SITES = {HELD: "held by a fit", PUT: "put before the model", JOINED: "joined to a reference"}
UNPUT = "not put before the model, its fork holding none of the facts its edit makes of it"
REFUSAL = (
    "a fit of this configuration is refused before it starts, by the check of the routes from the options to the "
    "prediction: %s; what follows is what such a fit would have been given had it not been refused"
)
READ = (
    "described: the decisions of a graph of the view of which an option holds a point there, an outcome a fit reads, "
    "which are the decisions whose options a realisation puts before a model; a decision holding none is counted "
    "with its graph and its options are not described; "
    "held by a fit: the batches a fit of this configuration packs hold the option, by the rule that packs them: beside "
    "the variant every node its actions target is observed at the step of its decision, every edge they target has "
    "both ends observed there and a present fact of its relation joining them, and where the realisation reads the "
    "actor the actor of each action is observed there; under an edit the option holds no action, or its own fork, "
    "the history continued one step with its edit, shows a fact or an observation that the history continued one step "
    "without it does not; "
    "left out by a fit: the option is put before no model, so nothing is said of the fields it states, which are "
    "listed, and what it lacks is named: beside the variant each target or actor the variant does not hold, with its "
    "nodes not observed at the step or, where they all are, no present fact of its relation joining its ends; under an "
    "edit each fact its edit makes, not in its fork, or in its fork and in the variant without the edit; "
    "put before the model: the first of two statements about a field of an option a fit holds, whether the realisation "
    "hands the field to the model, by its route: beside the variant always, a kind and a parameter with the option "
    "itself, an element an action creates with the option by its role alone, a node or an edge as the target the "
    "variant holds; under an edit only as the facts the edit makes of it, where the option's own fork holds them at "
    "the step after the whole pipeline: a target as a present fact of the relation its role becomes between the actor "
    "and a node it gives, a parameter as a value of such a fact of its action whose relation carries its column, a "
    "kind as the relation of such a fact of its actions; "
    "joined to a reference: the second statement, about a target alone, how the graph the family reads, the variant "
    "or under an edit the option's own fork, joins it at the step to the references of its task: by present facts, "
    "each taken in either direction, at the least number of hops, by the relations at each hop of every such shortest "
    "path; by no path where the graph holds the target and nothing joins it; not held where the graph does not hold "
    "it there; an element an option creates is no node of the variant; the second statement tells how far the target "
    "lies from the references, which matters to a family that reads the graph by hops from them, and is not needed "
    "for a model to read the field beside the variant, where a field put before the model is read whatever joins it; "
    "the walk reads the facts of the step of the decision alone, so a target that a fact joined to a reference at an "
    "earlier step of the history a family reads, and none joins at this one, reads as joined by no path; and it takes "
    "each fact either way, which no family does under relations/direction directed, where a relation with a direction "
    "is read from its source to its target alone: so a joined target also gives the least number of hops along the "
    "direction of its facts, of a path from the target to a reference that takes each fact of a relation with a "
    "direction from its source to its target, as what the target holds is passed on towards the reference, and a "
    "fact of a relation without direction either way, or says that no such path joins it; "
    "held at the step: a node is held where it is observed at the step other than as an anchor, as the description "
    "of a variant counts it; an anchor, a node observed there only because a reference of a task of the step names "
    "it, is read by a family all the same, so a target that is one is said to be an anchor at the step and is joined "
    "or not as any other, and a target is not held only where it is observed there neither as a node nor as an anchor; "
    "features: the columns of the kind of a target's node that hold a value at the step; of an edge, those of each of "
    "its two ends, named, and of the edge the columns of its relation of which a present fact joining its ends at the "
    "step holds a value; "
    "a cut is written once, numbered, under its graph, and every option, field and target it is of names it; it is of "
    "an option a fit leaves out, for being held by a fit, of a field or a target not put before the model, for being "
    "put before it, or of a target joined by no path, for being joined to a reference; it names every entry of the "
    "pipeline that cuts, the first being the entry after the last state of the pipeline, read from the last back, in "
    "which what it is of holds; the parameters of that entry are re-stated one by one with each other value, every "
    "option of a choice and the two ends of a number's range, a value being stated without the parameters it makes "
    "inactive and with every value of each parameter it makes active, and the whole pipeline is derived again under "
    "each statement, one derivation of a graph read by every option, field and target of it; the statements that give "
    "it back are named, each by its parameter and value and the parameters stated with it, and the rest are counted, "
    "those that do not give it back, those the pipeline refuses with every parameter stated, those whose derivation "
    "the library stops with an error of its own, and those not tried, at most %(bound)d being tried for a value, all "
    "of which the record holds; the parameters of the entries it reads are re-stated only where none of its own gives "
    "it back, and the text says which were and which were not; where no statement "
    "gives it back the pipeline is derived without the entry or, where the pipeline is refused without it, with the "
    "one parameter of it re-stated after which the graph holds the most, and the entry after which it is still "
    "missing there is named and its parameters re-stated in turn, each entry so named cutting alone; a statement not "
    "tried is tried by stating it in the pipeline of the configuration and asking the reach of that configuration; "
    "by no entry: it holds in no state, the view before the first entry included, and the truth graph is read "
    "%(when)s as the description of the bank reads it: the truth holds none of it there, or the truth holds it and "
    "the view does not show it to the task, by the rule named, what its split may see, that it became known later, "
    "or that another observer saw it; an end of a reference that the pipeline replaced by a node of a kind it adds "
    "is named by that kind and the node of the view in whose place it stands"
)
WHEN = dict(none="as it holds always", order="at the position of the task", clock="at the time of the task")
WHEN["clocks"] = WHEN["clock"]
READING = {model: READ % dict(bound=restated.BOUND, when=when) for model, when in WHEN.items()}
HIDDEN = {KNOWN: "it became known later", OBSERVER: "another observer saw it"}
WIDTH = 200


def looked(graph, step):
    started = time.perf_counter()
    tokens = masked(graph.entity_token)
    order = np.argsort(tokens, kind="stable")
    entity, moment = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    rows, anchors = np.full(len(tokens), -1, np.int64), np.zeros(len(tokens), bool)
    at = np.flatnonzero(moment == step)
    rows[entity[at]] = at
    anchors[entity[at]] = np.asarray(graph.obs_anchor, bool)[at]
    present = np.asarray(graph.fact_present, bool) & (np.asarray(graph.fact_step, np.int64) == step)
    facts = tuple(np.asarray(graph.arrays[name], np.int64)[present] for name in ("fact_source", "fact_target", "fact_code"))
    table, refs = np.asarray(graph.value_table, np.float64), np.asarray(graph.fact_ref, np.int64)[present]
    valued = np.full((len(refs), table.shape[1]), np.nan)
    valued[refs >= 0] = table[refs[refs >= 0]]
    result = dict(tokens=tokens[order], order=order, rows=rows, anchors=anchors, facts=facts, values=valued)
    LOG.debug("reach looked exit step=%d facts=%d seconds=%.4f", step, len(facts[0]), time.perf_counter() - started)
    return result


@timed("reach.entity")
def entity(seen, token):
    place = int(np.searchsorted(seen["tokens"], token))
    return int(seen["order"][place]) if place < len(seen["tokens"]) and seen["tokens"][place] == token else -1


def hopped(start, end, count, sources):
    started = time.perf_counter()
    hops = np.full(count, -1, np.int64)
    frontier, depth = np.unique(np.asarray(sources, np.int64)), 0
    hops[frontier] = 0
    while len(frontier):
        depth += 1
        reached = end[np.isin(start, frontier)]
        frontier = np.unique(reached[hops[reached] < 0])
        hops[frontier] = depth
    LOG.debug("reach hopped exit sources=%d hops=%d seconds=%.4f", len(sources), depth - 1, time.perf_counter() - started)
    return hops


def walked(seen, sources):
    started = time.perf_counter()
    source, target, code = seen["facts"]
    start, end = np.r_[source, target], np.r_[target, source]
    result = hopped(start, end, len(seen["rows"]), sources), (start, end, np.r_[code, code])
    LOG.debug("reach walked exit sources=%d seconds=%.4f", len(sources), time.perf_counter() - started)
    return result


def directed(seen, registry, sources):
    started = time.perf_counter()
    source, target, code = seen["facts"]
    either = registry.flipped[code] == code
    result = hopped(np.r_[target, source[either]], np.r_[source, target[either]], len(seen["rows"]), sources)
    LOG.debug("reach directed exit sources=%d seconds=%.4f", len(sources), time.perf_counter() - started)
    return result


def owned(seen, registry, ends, relation):
    started = time.perf_counter()
    source, target, code = seen["facts"]
    joins = (source == ends[0]) & (target == ends[1])
    if registry.flipped[relation] == relation:
        joins |= (source == ends[1]) & (target == ends[0])
    rows = np.flatnonzero(joins & (code == relation)) if min(ends) >= 0 else np.zeros(0, np.int64)
    place = int(registry.value_group[relation])
    columns = registry.attributes(registry.valued[place]) if place >= 0 else ()
    result = rows, [column for at, column in enumerate(columns) if not np.isnan(seen["values"][rows, at]).all()]
    LOG.debug("reach owned exit facts=%d columns=%d seconds=%.4f", len(rows), len(result[1]), time.perf_counter() - started)
    return result


def traced(walk, names, starts):
    started = time.perf_counter()
    hops, (start, end, code) = walk
    layer, found = np.asarray(starts, np.int64), []
    for depth in range(int(hops[layer[0]]), 0, -1):
        chosen = np.isin(end, layer) & (hops[start] == depth - 1)
        found.append(sorted({names[held] for held in np.unique(code[chosen]).tolist()}))
        layer = np.unique(start[chosen])
    LOG.debug("reach traced exit hops=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found[::-1]


class Reached(Derived):
    def __init__(self, deriver, config, name):
        started = time.perf_counter()
        super().__init__(deriver, config)
        self.name, self.entry = name, realisation(name)
        schema = deriver.registry.schema
        self.found, self.columns = offered(deriver), schema.parameters()
        every = {task: fielded(schema, task) for task in schema.decided()}
        self.fields = {task: [field for field in fields if field["sort"] in self.entry.reads] for task, fields in every.items()}
        self.roled = {field["code"]: field["name"] for fields in every.values() for field in fields if field["sort"] == "targets"}
        self.plains, self.forks, self.plans, self.walks = {}, {}, {}, {}
        self.holds, self.aims, self.placings, self.options, self.truths = {}, {}, {}, {}, {}
        self.briefs = [self.plains, self.forks, self.plans, self.walks, self.holds, self.placings, self.options]
        self.caches += self.briefs
        LOG.debug("reached exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)

    def planned(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, option) not in self.plans:
            part = forks.Part(decision["group"], -1, decision["step"], option, forks.UNMOVED)
            order = forks.ordered(self.deriver, option)
            self.plans[key, option] = forks.planned(self.deriver, self.variants[key][0], self.history(key, count, decision), part, order)
        LOG.debug("reached planned exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return self.plans[key, option]

    def forked(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, count, option) not in self.forks:
            part = forks.Part(decision["group"], -1, decision["step"], option, forks.UNMOVED)
            edit = forks.edited(self.deriver, self.variants[key][0], part, self.planned(key, count, decision, option))
            fork = self.history(key, count, decision).continued(decision["step"] + 1, edit)
            self.forks[key, count, option] = fork, fork.derived()
        LOG.debug("reached forked exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return self.forks[key, count, option]

    def plain(self, key, count, decision):
        started = time.perf_counter()
        if (key, count, decision["decision"]) not in self.plains:
            history = self.history(key, count, decision)
            self.plains[key, count, decision["decision"]] = forks.listed(history.continued(decision["step"] + 1, None).arrays())
        LOG.debug("reached plain exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return self.plains[key, count, decision["decision"]]

    def showing(self, key, count, decision, option):
        started = time.perf_counter()
        result = forks.differing(forks.listed(self.forked(key, count, decision, option)[0].arrays()), self.plain(key, count, decision))
        LOG.debug("reached showing exit option=%d showing=%s seconds=%.4f", option, result, time.perf_counter() - started)
        return result

    def decided(self, graph, group):
        started = time.perf_counter()
        rows, tasks = np.asarray(graph.label_row, np.int64), np.asarray(graph.label_task, np.int64)
        decision = self.found.decision[rows]
        chosen = np.flatnonzero(decision >= 0)
        picked = chosen[np.unique(decision[chosen], return_index=True)[1]]
        picked = picked[np.argsort(tasks[picked], kind="stable")]
        indices = self.loaded(group)[0]
        titles = {held["index"]: held["name"] for held in self.deriver.viewed(self.config).graphs}
        timed_steps = self.deriver.registry.schema.model != "none"
        result = []
        for at in picked.tolist():
            task = int(tasks[at])
            step = int(graph.task_step[task])
            view = indices[int(graph.step_graph[step])]
            result.append(
                dict(
                    decision=int(decision[at]),
                    group=group,
                    view=view,
                    graph=titles[view],
                    step=step,
                    time=int(graph.step_time[step]) if timed_steps else None,
                    key=int(graph.task_key[task]),
                    kind=int(graph.task_kind[task]),
                )
            )
        LOG.debug("reached decided exit group=%d decisions=%d seconds=%.4f", group, len(result), time.perf_counter() - started)
        return result

    def keyed(self, token, where):
        started = time.perf_counter()
        if token not in self.names:
            raise ValueError(
                "%s names the node of the token %d, which no graph of the view holds; reach names each node by its key "
                "in the view" % (where, token)
            )
        LOG.debug("reached keyed exit token=%d seconds=%.6f", token, time.perf_counter() - started)
        return self.names[token]

    def referred(self, graph, decision):
        started = time.perf_counter()
        task = np.flatnonzero(
            (np.asarray(graph.task_step) == decision["step"])
            & (np.asarray(graph.task_key) == decision["key"])
            & (np.asarray(graph.task_kind) == decision["kind"])
        )
        if len(task) != 1:
            raise ValueError(
                "reach reads the references of the decision %d of the kind %s of graph %s from its task at the step %d, "
                "and the graph derived for it holds %d tasks of that key and kind there"
                % (decision["key"], self.registry.tasks[decision["kind"]], decision["graph"], decision["step"], len(task))
            )
        rows = np.flatnonzero(
            (np.asarray(graph.member_task) == task[0]) & (np.asarray(graph.member_name) < len(self.registry.references))
        )
        result = [
            (int(graph.member_name[row]), int(graph.member_source[row]), int(graph.member_target[row]), int(graph.member_code[row]))
            for row in rows.tolist()
        ]
        LOG.debug("reached referred exit members=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def ended(self, graph, decision, members, place, side, where):
        started = time.perf_counter()
        end = members[place][side]
        token, kind = int(masked(graph.entity_token)[end]), self.registry.nodes[int(graph.kinds[end])]
        if token in self.names or kind in self.deriver.registry.nodes:
            result = self.keyed(token, where)
        else:
            view = self.derived(None, 0, decision["group"])
            stated = self.referred(view, decision)
            if [held[0] for held in stated] != [held[0] for held in members]:
                raise ValueError(
                    "%s names a node of the kind %s, which the pipeline adds, and the references of its task hold %d "
                    "members after the pipeline and %d in the view, or the same in another order, so reach cannot say "
                    "in place of which node of the view that node stands" % (where, kind, len(members), len(stated))
                )
            result = "%s in place of %s" % (kind, self.keyed(int(masked(view.entity_token)[stated[place][side]]), where))
        LOG.debug("reached ended exit end=%s seconds=%.4f", result, time.perf_counter() - started)
        return result

    def references(self, graph, decision):
        started = time.perf_counter()
        members, result = self.referred(graph, decision), {}
        for place, (name, _, target, code) in enumerate(members):
            where = "the reference %s of the decision %d of the kind %s of graph %s" % (
                self.registry.references[name],
                decision["key"],
                self.registry.tasks[decision["kind"]],
                decision["graph"],
            )
            spoken = self.ended(graph, decision, members, place, 1, where)
            if target >= 0:
                second = self.ended(graph, decision, members, place, 2, where)
                spoken = "%s %s %s" % (spoken, self.registry.by_code[code].name, second)
            result.setdefault(self.registry.references[name], []).append(spoken)
        LOG.debug("reached references exit references=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def looking(self, graph, registry, decision, seen):
        started = time.perf_counter()
        sources = [end for _, source, target, _ in self.referred(graph, decision) for end in (source, target) if end >= 0]
        names = [registry.by_code[code].name for code in range(registry.relations)]
        result = seen, walked(seen, sources), names, directed(seen, registry, sources)
        LOG.debug("reached looking exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return result

    def featured(self, graph, registry, seen, ends, keys):
        started = time.perf_counter()
        entities, table, refs = registry.schema.entities(), np.asarray(graph.x_table), np.asarray(graph.obs_ref, np.int64)
        found = []
        for end, key in zip(ends, keys):
            row, held = table[refs[seen["rows"][end]]], []
            for column in entities[registry.nodes[int(graph.kinds[end])]]:
                flags = registry.encoding.slots(column)[1]
                if len(flags) and row[flags[0]] > 0:
                    held.append(column)
            found.append(dict(node=key, columns=held))
        LOG.debug("reached featured exit nodes=%d seconds=%.4f", len(found), time.perf_counter() - started)
        return found

    def placed(self, graph, registry, looked_at, ends, target):
        started = time.perf_counter()
        seen, walk, names, toward = looked_at
        valued = None if target["relation"] is None else owned(seen, registry, ends, registry.code(target["relation"]))[1]
        result = dict(state=UNHELD, hops=None, relations=None, along=None, anchor=False, features=[], values=valued)
        if all(end >= 0 and seen["rows"][end] >= 0 for end in ends):
            hops, along = walk[0][list(ends)], toward[list(ends)]
            reached = hops[hops >= 0]
            held = dict(
                along=int(along[along >= 0].min()) if (along >= 0).any() else None,
                anchor=bool(seen["anchors"][list(ends)].any()),
                features=self.featured(graph, registry, seen, ends, target["nodes"]),
                values=valued,
            )
            result = dict(held, state=UNJOINED, hops=None, relations=None, along=None)
            if len(reached):
                least = int(reached.min())
                starts = [end for end, hop in zip(ends, hops.tolist()) if hop == least]
                result = dict(held, state=JOINED, hops=least, relations=traced(walk, names, starts))
        LOG.debug("reached placed exit state=%s seconds=%.4f", result["state"], time.perf_counter() - started)
        return result

    def aiming(self, option):
        started = time.perf_counter()
        if option not in self.aims:
            found, bank = self.found, self.deriver.registry.by_code
            first, count = int(found.actions[0][option]), int(found.actions[1][option])
            rows = [
                (action, role)
                for action in range(first, first + count)
                for role in range(int(found.roles[0][action]), int(found.roles[0][action] + found.roles[1][action]))
            ]
            sorts = {role: ROLES[int(found.sorts[found.role["code"][role]])] for _, role in rows}
            elements = {}
            for _, role in rows:
                if sorts[role] == "new":
                    source, element = int(found.role["source"][role]), int(found.role["element"][role])
                    elements[source] = min(elements[source], element) if source in elements else element
            result, where = [], "a target of the option %s" % found.keys[option]
            for action, role in rows:
                source, target = int(found.role["source"][role]), int(found.role["target"][role])
                spoken, relation = self.keyed(source, where), None
                nodes = [spoken]
                if sorts[role] == "edge":
                    relation = bank[int(found.role["relation"][role])].name
                    nodes.append(self.keyed(target, where))
                    spoken = "%s %s %s" % (nodes[0], relation, nodes[1])
                ends = (int(masked([elements[source]])[0]),) if sorts[role] == "new" else (source, target)[: 1 + (sorts[role] == "edge")]
                held = dict(action=action, role=role, code=int(found.role["code"][role]), sort=sorts[role], target=spoken, ends=ends)
                result.append(dict(held, nodes=nodes, relation=relation))
            self.aims[option] = sorted(result, key=lambda held: (held["action"], held["code"], held["target"], held["role"]))
        LOG.debug("reached aiming exit option=%d targets=%d seconds=%.4f", option, len(self.aims[option]), time.perf_counter() - started)
        return self.aims[option]

    def stating(self, decision, option):
        started = time.perf_counter()
        found, task = self.found, self.registry.tasks[decision["kind"]]
        first, many = int(found.actions[0][option]), int(found.actions[1][option])
        aims, result = self.aiming(option), []
        for field in self.fields[task]:
            if field["sort"] == "kind":
                held = list(range(first, first + many)) or None
            elif field["sort"] == "targets":
                held = [target for target in aims if target["code"] == field["code"]] or None
            else:
                column = found.action["values"][first : first + many, self.columns.index(field["column"])]
                held = [
                    first + at
                    for at in range(many)
                    if int(found.action["kind"][first + at]) == field["code"] and not np.isnan(column[at])
                ] or None
            if held is not None:
                result.append((field, held))
        LOG.debug("reached stating exit option=%d fields=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
        return result

    def lacked(self, aims, row, where):
        started = time.perf_counter()
        code, sort = int(aims.code[row]), ROLES[int(aims.sort[row])]
        ends = [int(end) for end in aims.ends[row].tolist() if end >= 0]
        keys = [self.keyed(end, where) for end in ends]
        target, wanted = keys[0], [dict(target=keys[0], ends=tuple(ends), relation=None)]
        if sort == "edge":
            target = "%s %s %s" % (keys[0], self.deriver.registry.by_code[int(aims.relation[row])].name, keys[1])
            wanted = [dict(target=target, ends=tuple(ends), relation=int(aims.relation[row]))]
        unobserved = [(key, end) for key, end, seen in zip(keys, ends, aims.observed[row].tolist()) if seen < 0]
        if unobserved:
            wanted = [dict(target=key, ends=(end,), relation=None) for key, end in unobserved]
        result = dict(
            field=self.roled[code] if code in self.roled else ACTOR,
            target=target,
            lacks=OBSERVATION if unobserved else FACT,
            fact=None,
            unobserved=[key for key, _ in unobserved],
        )
        LOG.debug("reached lacked exit target=%s lacks=%s seconds=%.4f", target, result["lacks"], time.perf_counter() - started)
        return result, wanted

    def kept(self, key, count, decision):
        started = time.perf_counter()
        if (key, count, decision["decision"]) not in self.holds:
            arrays = self.shown(key, count, decision["group"])
            options = self.optioned(decision)
            steps = np.full(len(options), decision["step"], np.int64)
            aims = aimed(self.found, self.name, arrays, 0, len(arrays["snapshot_offsets"]) - 1, np.asarray(options, np.int64), steps)
            owner, result = aims.acting[aims.roling], {}
            for place, option in enumerate(options):
                where = "a target of the option %s" % self.found.keys[option]
                rows = np.flatnonzero((owner == place) & aims.unheld).tolist()
                lacking = [self.lacked(aims, row, where) for row in rows]
                result[option] = (
                    not bool(aims.left[place]),
                    [held for held, _ in lacking],
                    [wanted for _, found in lacking for wanted in found],
                )
            self.holds[key, count, decision["decision"]] = result
        LOG.debug("reached kept exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return self.holds[key, count, decision["decision"]]

    def beside(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, count, option) not in self.placings:
            registry, graph = self.variants[key][4], self.derived(key, count, decision["group"])
            if (key, count, decision["decision"]) not in self.walks:
                self.walks[key, count, decision["decision"]] = self.looking(graph, registry, decision, looked(graph, decision["step"]))
            looked_at, result = self.walks[key, count, decision["decision"]], {}
            for field, held in self.stating(decision, option):
                if field["sort"] != "targets":
                    result[field["name"]] = dict(put=True, route=OPTION, facts=[])
                    continue
                created = field["role"] == "new"
                targets = [
                    dict(
                        target=target["target"],
                        put=True,
                        facts=[],
                        **(
                            dict(state=CREATED, hops=None, relations=None, along=None, anchor=False, features=[], values=None)
                            if created
                            else self.placed(
                                graph, registry, looked_at, tuple(entity(looked_at[0], end) for end in target["ends"]), target
                            )
                        ),
                    )
                    for target in held
                ]
                result[field["name"]] = dict(put=True, route=ROLE if created else TARGET, targets=targets)
            self.placings[key, count, option] = result
        LOG.debug("reached beside exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return self.placings[key, count, option]

    def edits(self, key, count, decision, option):
        started = time.perf_counter()
        registry, graph = self.variants[key][4], self.forked(key, count, decision, option)[1]
        plan, seen = self.planned(key, count, decision, option), looked(graph, decision["step"])
        marks = np.r_[masked(plan.graph.entity_token), masked([found for found, _ in plan.made])]
        created = {target["ends"][0]: target["target"] for target in self.aiming(option) if target["sort"] == "new"}
        where, result = "a fact of the edit of the option %s" % self.found.keys[option], []
        for relation, first, second, row, action, role in plan.facts:
            ends = [int(marks[first]), int(marks[second])]
            keys = [created[end] if end in created else self.keyed(end, where) for end in ends]
            edge = registry.by_code[relation]
            rows, valued = owned(seen, registry, [entity(seen, end) for end in ends], relation)
            place = int(registry.value_group[relation])
            columns = registry.attributes(registry.valued[place]) if place >= 0 else ()
            result.append(
                dict(
                    action=action,
                    role=role,
                    fact="%s %s %s" % (keys[0], edge.name, keys[1]),
                    held=bool(len(rows)),
                    carried=[column for at, column in enumerate(columns) if not np.isnan(row[at])],
                    valued=valued,
                    keys=keys,
                    ends=ends,
                    relation=edge.name,
                )
            )
        LOG.debug("reached edits exit option=%d facts=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
        return result, seen

    @timed("reached.told")
    def told(self, facts):
        found = {}
        for fact in facts:
            found[fact["fact"]] = found.get(fact["fact"], False) or fact["held"]
        return [dict(fact=fact, held=held) for fact, held in sorted(found.items())]

    def edited(self, key, count, decision, option):
        started = time.perf_counter()
        registry, graph = self.variants[key][4], self.forked(key, count, decision, option)[1]
        facts, seen = self.edits(key, count, decision, option)
        looked_at, result = self.looking(graph, registry, decision, seen), {}
        for field, held in self.stating(decision, option):
            if field["sort"] == "kind":
                own = self.told(facts)
                result[field["name"]] = dict(put=any(fact["held"] for fact in own), route=FACT, facts=own)
            elif field["sort"] == "parameters":
                carrying = [fact for fact in facts if fact["action"] in held and field["column"] in fact["carried"]]
                own = self.told([dict(fact, held=fact["held"] and field["column"] in fact["valued"]) for fact in carrying])
                result[field["name"]] = dict(put=any(fact["held"] for fact in own), route=FACT, facts=own)
            else:
                targets = []
                for target in held:
                    own = [dict(fact=fact["fact"], held=fact["held"]) for fact in facts if fact["role"] == target["role"]]
                    ends = tuple(entity(seen, end) for end in target["ends"])
                    targets.append(
                        dict(
                            target=target["target"],
                            put=any(fact["held"] for fact in own),
                            facts=own,
                            **self.placed(graph, registry, looked_at, ends, target),
                        )
                    )
                result[field["name"]] = dict(put=any(target["put"] for target in targets), route=FACT, targets=targets)
        LOG.debug("reached edited exit option=%d fields=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
        return result, facts

    def unshown(self, decision, option, facts):
        started = time.perf_counter()
        aims = {target["role"]: target for target in self.aiming(option)}
        result = [
            dict(
                field=self.roled[aims[fact["role"]]["code"]],
                target=aims[fact["role"]]["target"],
                lacks=EDIT if fact["held"] else FACT,
                fact=fact["fact"],
                unobserved=[],
            )
            for fact in facts
        ]
        result.sort(key=lambda held: (held["field"], held["target"], held["fact"]))
        LOG.debug("reached unshown exit option=%d facts=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
        return result

    def read(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, count, option) not in self.options:
            if self.entry.back:
                fields, facts = self.edited(key, count, decision, option)
                held = not forks.left(self.found, option, partial(self.showing, key, count, decision, option))
                lacking = [] if held else self.unshown(decision, option, facts)
            else:
                held, lacking, _ = self.kept(key, count, decision)[option]
                fields = self.beside(key, count, decision, option) if held else {}
            if not held:
                fields = {field["name"]: None for field, _ in self.stating(decision, option)}
            self.options[key, count, option] = dict(held=held, lacking=lacking, fields=fields)
        LOG.debug("reached read exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return self.options[key, count, option]

    def holding(self, site, key, count, decision, option):
        started = time.perf_counter()
        if site[0] == JOINED and not self.entry.back:
            targets = self.beside(key, count, decision, option)[site[1]]["targets"]
            result = targets[site[2]]["state"] == JOINED
        else:
            held = self.read(key, count, decision, option)
            result = held["held"]
            if result and site[0] != HELD:
                field = held["fields"][site[1]]
                part = field if site[2] is None else field["targets"][site[2]]
                result = part["put"] if site[0] == PUT else part["state"] == JOINED
        LOG.debug("reached holding exit site=%s holding=%s seconds=%.4f", site[0], result, time.perf_counter() - started)
        return result

    def stopped(self, key, decision, option, site):
        started = time.perf_counter()
        counts = range(len(self.variants[key][1]) - 1, -1, -1)
        result = next((count for count in counts if self.holding(site, key, count, decision, option)), None)
        LOG.debug("reached stopped exit site=%s entry=%s seconds=%.4f", site[0], result, time.perf_counter() - started)
        return result

    def wanted(self, decision, option, site):
        started = time.perf_counter()
        last = len(self.steps)
        if site[0] == JOINED:
            (aims,) = [held for field, held in self.stating(decision, option) if field["name"] == site[1]]
            result = [dict(target=None, ends=aims[site[2]]["ends"], relation=None)]
        elif self.entry.back:
            facts, bank = self.edits(None, 0, decision, option)[0], self.deriver.registry.by_name
            if site[0] == PUT:
                field = self.read(None, last, decision, option)["fields"][site[1]]
                made = [held["fact"] for held in (field if site[2] is None else field["targets"][site[2]])["facts"]]
                facts = [fact for fact in facts if fact["fact"] in made]
            nodes = {key: end for fact in facts if not fact["held"] for key, end in zip(fact["keys"], fact["ends"])}
            result = [dict(target=key, ends=(end,), relation=None) for key, end in sorted(nodes.items())]
            result += [
                dict(target=fact["fact"], ends=tuple(fact["ends"]), relation=bank[fact["relation"]].code)
                for fact in facts
                if site[0] == HELD and fact["held"] and fact["relation"] in bank
            ]
        else:
            result = self.kept(None, 0, decision)[option][2]
        result = [held for place, held in enumerate(result) if held not in result[:place]]
        LOG.debug("reached wanted exit site=%s wanted=%d seconds=%.4f", site[0], len(result), time.perf_counter() - started)
        return result

    def asking(self, decision, option, site, into):
        started = time.perf_counter()
        last = self.stopped(None, decision, option, site)
        wanted = self.wanted(decision, option, site) if last is None else []
        result = dict(decision=decision, option=option, site=site, into=into, last=last, wanted=wanted, entries=[], truth=None)
        LOG.debug("reached asking exit site=%s entry=%s seconds=%.4f", site[0], last, time.perf_counter() - started)
        return result

    @timed("reached.sited")
    def sited(self, name, place, target):
        if not target["put"]:
            return (PUT, name, place)
        return (JOINED, name, place) if target["state"] == UNJOINED else None

    def option(self, decision, option, asked):
        started = time.perf_counter()
        held = self.read(None, len(self.steps), decision, option)
        result = dict(key=self.found.keys[option], held=held["held"], lacking=held["lacking"], cut=None, fields={})
        if not held["held"]:
            result["fields"] = dict(held["fields"])
            asked.append(self.asking(decision, option, (HELD,), result))
        for name, field in held["fields"].items() if held["held"] else ():
            if "targets" in field:
                targets = [dict(target, cut=None) for target in field["targets"]]
                result["fields"][name] = dict(field, targets=targets)
                sites = [self.sited(name, place, target) for place, target in enumerate(targets)]
                asked += [self.asking(decision, option, site, target) for site, target in zip(sites, targets) if site is not None]
            else:
                result["fields"][name] = dict(field, cut=None)
                if not field["put"]:
                    asked.append(self.asking(decision, option, (PUT, name, None), result["fields"][name]))
        LOG.debug("reached option exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return result

    def judged(self, key, asked):
        started = time.perf_counter()
        count = len(self.variants[key][1])
        try:
            result = [
                (BACK if self.holding(held["site"], key, count, held["decision"], held["option"]) else STILL, {})
                for held in asked
            ]
        except STOPS as error:
            result = [halted(error)] * len(asked)
        LOG.debug("reached judged exit sites=%d seconds=%.4f", len(asked), time.perf_counter() - started)
        return result

    def tried(self, base, place, asked):
        started = time.perf_counter()
        step, holders = self.variants[base][1][place], self.settable(base, place)
        left = [
            (held, dict(entry=step.name, operation=step.operation.name, restated=[], spared=[], settings=[], without=None))
            for held in asked
        ]
        for held, found in left:
            held["entries"].append(found)
        for stage in (holders[:1], holders[1:]):
            for holder, parameters in stage if left else ():
                for _, found in left:
                    found["restated"].append(holder.name)
                for parameter in parameters:
                    value = holder.parameters[parameter["name"]]
                    settings = [
                        dict(entry=holder.name, parameter=parameter["name"], value=value, back=[], still=[], refused=[], stopped=[], untried=[])
                        for _ in left
                    ]
                    for other in [other for other in values(parameter) if other != value]:
                        tried, untried = self.restating(base, holder, parameter, other)
                        for added, key, why in tried:
                            outcomes = [why] * len(left) if key is None else self.judged(key, [held for held, _ in left])
                            self.released(key)
                            for setting, (part, said) in zip(settings, outcomes):
                                setting[part].append(dict(value=other, stating=added, **said))
                        for setting in settings:
                            setting[UNTRIED] += [dict(value=other, stating=added) for added in untried]
                    for (_, found), setting in zip(left, settings):
                        found["settings"].append(setting)
            for _, found in left:
                if any(setting[BACK] for setting in found["settings"]):
                    found["spared"] = [holder.name for holder, _ in holders if holder.name not in found["restated"]]
            left = [(held, found) for held, found in left if not any(setting[BACK] for setting in found["settings"])]
        LOG.debug("reached tried exit entry=%s sites=%d left=%d seconds=%.4f", step.name, len(asked), len(left), time.perf_counter() - started)
        return left

    def searched(self, group, asked, cuts):
        started = time.perf_counter()
        waiting, places = {}, {}
        for held in asked:
            waiting.setdefault((None, held["last"]), []).append(held)
        while waiting:
            base, place = next(iter(waiting))
            left = [] if place is None else self.tried(base, place, waiting[base, place])
            del waiting[base, place]
            lifted = self.lifted(base, place, group) if left else None
            outcomes = [(STILL, {})] * len(left)
            if left and lifted["key"] is not None:
                outcomes = self.judged(lifted["key"], [held for held, _ in left])
            for (held, found), (part, _) in zip(left, outcomes):
                how = KEPT if part == STOPPED else lifted["how"]
                found["without"] = dict(how=how, parameter=lifted["parameter"], value=lifted["value"], stating=lifted["stating"], back=part == BACK)
                after = None
                if part == STILL and how != KEPT:
                    after = self.stopped(lifted["key"], held["decision"], held["option"], held["site"])
                if after is not None and (how == ABSENT or after != place):
                    waiting.setdefault((lifted["key"], after), []).append(held)
        for held in asked:
            if held["last"] is None:
                found = self.truthful(held["decision"])
                held["truth"] = [self.told_truly(found, held, wanted) for wanted in held["wanted"]]
            view = held["decision"]["view"]
            cut = dict(on=held["site"][0], entries=held["entries"], truth=held["truth"])
            mark = view, json.dumps(cut, sort_keys=True)
            if mark not in places:
                places[mark] = len(cuts.setdefault(view, []))
                cuts[view].append(cut)
            held["into"]["cut"] = places[mark]
        LOG.debug("reached searched exit group=%d sites=%d seconds=%.4f", group, len(asked), time.perf_counter() - started)

    def truthful(self, decision):
        started = time.perf_counter()
        viewed, registry = self.deriver.viewed(self.config), self.deriver.registry
        (shown,) = [held for held in viewed.graphs if held["index"] == decision["view"]]
        if shown["source"] not in self.truths:
            facts = sets.stored(Facts.load(self.deriver.truth(shown["source"])), registry, viewed.storage)
            moments = truth.placed(truth.clocked(registry, facts))
            spans = {group: truth.shown(facts, group, moments) for group in GROUPS}
            self.truths[shown["source"]] = facts, moments, spans, sequences(facts, registry, viewed.steps, viewed.looking)
        facts, moments, spans, shows = self.truths[shown["source"]]
        kind = list(registry.schema.tasks["kinds"]).index(self.registry.tasks[decision["kind"]])
        task = int(np.flatnonzero((np.asarray(facts.task_kind) == kind) & (np.asarray(facts.task_key) == decision["key"]))[0])
        (sequence,) = [held for held in shows if (held["observer"], held["split"]) == (shown["observer"], shown["split"])]
        place = 0 if registry.schema.model == "none" else int(np.searchsorted(moments, facts.task_time[task]))
        tokens = masked(facts.entity_token)
        order = np.argsort(tokens, kind="stable")
        result = dict(
            facts=facts,
            sequence=sequence,
            looking=int(sequence["looking"][sequence["task_step"][sequence["tasks"] == task][0]]),
            tokens=tokens[order],
            order=order,
            held={group: (begun <= place) & (place < ended) for group, (begun, ended) in spans.items()},
        )
        LOG.debug("reached truthful exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return result

    def hidden(self, held, group, rows):
        started = time.perf_counter()
        facts, sequence, looking = held["facts"], held["sequence"], held["looking"]
        stamps = facts.stamps(group)
        allowed = visible.allowed(
            len(stamps["since"]), facts.arrays[group + "_seen_row"], facts.arrays[group + "_seen_observer"], sequence["observer"]
        )
        result = {OBSERVER: ~allowed[rows], KNOWN: np.asarray(stamps["known"])[rows] > looking}
        for name in sequence["rules"]:
            kept, since, until = visible.SEES.named(name)(facts, self.deriver.registry, sequence)[GROUPS.index(group)]
            result["%s %s" % (SEES, name)] = np.isin(rows, np.asarray(kept)[(np.asarray(since) <= looking) & (looking < np.asarray(until))])
        LOG.debug("reached hidden exit group=%s rows=%d seconds=%.4f", group, len(rows), time.perf_counter() - started)
        return result

    def told_truly(self, held, asked, wanted):
        started = time.perf_counter()
        facts, flipped = held["facts"], self.deriver.registry.flipped
        ends = [entity(held, end) for end in wanted["ends"]]
        source, target, code = (np.asarray(facts.arrays[name], np.int64) for name in ("fact_source", "fact_target", "fact_code"))
        present = held["held"]["fact"] & np.asarray(facts.fact_present, bool)
        observed = held["held"]["obs"] & np.isin(np.asarray(facts.obs_entity, np.int64), ends)
        if asked["site"][0] == JOINED:
            decision = asked["decision"]
            view = self.derived(None, 0, decision["group"])
            marks = masked(view.entity_token)
            named = [entity(held, int(marks[end])) for _, first, second, _ in self.referred(view, decision) for end in (first, second) if end >= 0]
            seen = dict(rows=held["tokens"], facts=(source[present], target[present], code[present]))
            hops = walked(seen, [end for end in named if end >= 0])[0]
            found = len(np.unique(np.asarray(facts.obs_entity, np.int64)[observed])) == len(set(ends)) and any(hops[end] >= 0 for end in ends)
            reasons = [self.hidden(held, "obs", np.flatnonzero(held["held"]["obs"])), self.hidden(held, "fact", np.flatnonzero(present))]
        else:
            group, rows = "obs", np.flatnonzero(observed)
            if wanted["relation"] is not None:
                joins = (source == ends[0]) & (target == ends[1])
                if flipped[wanted["relation"]] == wanted["relation"]:
                    joins |= (source == ends[1]) & (target == ends[0])
                group, rows = "fact", np.flatnonzero(present & (code == wanted["relation"]) & joins)
            reasons, found = [self.hidden(held, group, rows)], bool(len(rows))
            if not np.logical_or.reduce(list(reasons[0].values())).all():
                reasons = []
        hidden = sorted({name for flags in reasons for name, flagged in flags.items() if flagged.any()}) if found else []
        result = dict(target=wanted["target"], held=bool(found), hidden=hidden)
        LOG.debug("reached told truly exit site=%s held=%s seconds=%.4f", asked["site"][0], result["held"], time.perf_counter() - started)
        return result

    def optioned(self, decision):
        started = time.perf_counter()
        first, count = int(self.found.first[decision["decision"]]), int(self.found.count[decision["decision"]])
        result = sorted(range(first, first + count), key=lambda option: self.found.keys[option])
        LOG.debug("reached optioned exit options=%d seconds=%.4f", count, time.perf_counter() - started)
        return result

    @timed("reached.decision_done")
    def decision_done(self):
        for done in self.briefs:
            done.clear()

    @timed("reached.group_done")
    def group_done(self):
        super().group_done()
        self.truths.clear()

    def counted(self, graph, group):
        started = time.perf_counter()
        sources = {held["index"]: held["source"] for held in self.deriver.viewed(self.config).graphs}
        member = np.asarray(graph.step_graph, np.int64)[np.asarray(graph.task_step, np.int64)]
        kind, key = np.asarray(graph.task_kind, np.int64), np.asarray(graph.task_key, np.int64)
        result = {}
        for at, view in enumerate(self.loaded(group)[0]):
            held = load_arrays(self.deriver.truth(sources[view]), ("option_task", "task_kind", "task_key"))
            optioned = np.unique(np.asarray(held["option_task"], np.int64))
            stated = set(zip(np.asarray(held["task_kind"])[optioned].tolist(), np.asarray(held["task_key"])[optioned].tolist()))
            shown = [code for code, found in zip(kind[member == at].tolist(), key[member == at].tolist()) if (code, found) in stated]
            result[view] = {task: shown.count(self.registry.tasks.index(task)) for task in self.fields}
        LOG.debug("reached counted exit group=%d graphs=%d seconds=%.4f", group, len(result), time.perf_counter() - started)
        return result

    def graphed(self):
        started = time.perf_counter()
        held, stated, cuts = {}, {}, {}
        for group in self.deriver.graphs(self.config):
            trunk, asked = self.derived(None, len(self.steps), group), []
            stated.update(self.counted(trunk, group))
            for decision in self.decided(trunk, group):
                held.setdefault(decision["view"], []).append(
                    dict(
                        task=decision["key"],
                        kind=self.registry.tasks[decision["kind"]],
                        time=decision["time"],
                        references=self.references(trunk, decision),
                        options=[self.option(decision, option, asked) for option in self.optioned(decision)],
                    )
                )
                self.decision_done()
            self.searched(group, asked, cuts)
            self.group_done()
        result, views = [], restated.viewed(self.deriver, self.config)
        for graph in self.deriver.viewed(self.config).graphs:
            decisions = held[graph["index"]] if graph["index"] in held else []
            described = [decision["kind"] for decision in decisions]
            decided = {
                task: dict(decisions=count, described=described.count(task)) for task, count in stated[graph["index"]].items()
            }
            found = cuts[graph["index"]] if graph["index"] in cuts else []
            view = views[graph["index"]]
            result.append(dict(name=graph["name"], view=view, decided=decided, cuts=found, decisions=decisions))
        LOG.info("reached graphed exit graphs=%d seconds=%.3f", len(result), time.perf_counter() - started)
        return result


def stated(name):
    started = time.perf_counter()
    entry = realisation(name)
    result = dict(
        name=name,
        version=entry.version,
        reads=list(entry.reads),
        back=entry.back,
        structure=entry.structure,
        actor=entry.actor,
        mapping=entry.mapping if entry.back else None,
        relations=editing(name)[1].declaration()["relations"] if entry.back else None,
    )
    LOG.debug("reach stated exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)
    return result


def recorded(deriver, config, key, name, refused):
    started = time.perf_counter()
    reached = Reached(deriver, config, name)
    schema, entry = deriver.registry.schema, reached.entry
    every = {task: fielded(schema, task) for task in schema.decided()}
    result = dict(
        version=truth.VERSION,
        variant=key,
        stored_sha256=deriver.source,
        time=truth.modelled(deriver.registry),
        realisation=stated(name),
        refused=refused,
        seed=deriver.seed,
        device=deriver.device,
        splits=config["splits"],
        settings=specs.materialized(reached.spec),
        unshown=restated.unviewed(deriver, config),
        entries=[effect.entered(step, deriver.registry, reached.spec, config["splits"]) for step in reached.steps],
        tasks={
            task: dict(
                fields=[field["name"] for field in fields if field["sort"] in entry.reads],
                unread=[field["name"] for field in fields if field["sort"] not in entry.reads],
            )
            for task, fields in every.items()
        },
        undecided=[kind for kind in schema.tasks["kinds"] if kind not in every],
        graphs=reached.graphed(),
    )
    LOG.info("reach recorded exit realisation=%s seconds=%.3f", name, time.perf_counter() - started)
    return result


def realising(deriver, config):
    started = time.perf_counter()
    schema = deriver.registry.schema
    if not schema.decided():
        raise ValueError(
            "the bank %s holds no kind of task that decides; reach describes the options of decisions" % deriver.directory
        )
    name = realised(config["model"], schema)
    if name is None:
        raise ValueError(
            "the model of the configuration states %s none, which puts no option before it; reach describes the "
            "options of decisions as a realisation puts them before a model" % REALISATION
        )
    reason = realisation(name).refusal(deriver.registry)
    if reason is not None:
        raise ValueError("the realisation of options %s cannot read the options of this bank: %s" % (name, reason))
    LOG.debug("reach realising exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)
    return name


def unrouted(deriver, config):
    started = time.perf_counter()
    result = None
    try:
        routed(deriver, config)
    except ValueError as error:
        result = str(error)
    LOG.debug("reach unrouted exit refused=%s seconds=%.4f", result is not None, time.perf_counter() - started)
    return result


@timed("reach.stemmed")
def stemmed(key, name, refused):
    return sha256_json(dict(variant=key, realisation=stated(name), refused=refused, version=truth.VERSION))


def reach(bank, config, statement):
    started = time.perf_counter()
    truth.manifested(bank)
    stated = state(statement)
    agreed(config, stated)
    deriver = Deriver(bank, stated.device, stated.derivation_seed, RESIDENT)
    name = realising(deriver, config)
    key = specs.digest(deriver.identity(config))
    refused = unrouted(deriver, config)
    held = Path(bank) / "descriptions" / "reach" / stemmed(key, name, refused)
    result = truth.kept(held, lambda: recorded(deriver, config, key, name, refused), text)
    LOG.info("reach exit variant=%s realisation=%s seconds=%.3f", key[:10], name, time.perf_counter() - started)
    return result


@timed("reach.listed")
def listed(values):
    return ", ".join(map(str, values)) or "none"


@timed("reach.spoken")
def spoken(held):
    anchored = "an anchor at the step, " if held["anchor"] else ""
    if held["state"] == JOINED:
        return "%sjoined to a reference at hop %d by %s and %s along the direction of its facts" % (
            anchored,
            held["hops"],
            " then ".join(" or ".join(names) for names in held["relations"]) or "nothing",
            "by no path" if held["along"] is None else "at hop %d" % held["along"],
        )
    if held["state"] == UNJOINED:
        return anchored + "joined to a reference by no path"
    return "an element the option creates, no node of the variant" if held["state"] == CREATED else "not held at the step"


@timed("reach.features_text")
def features_text(target):
    if target["values"] is None:
        return "features %s" % listed([column for held in target["features"] for column in held["columns"]])
    ends = ["of %s %s" % (held["node"], listed(held["columns"])) for held in target["features"]]
    return "features %s" % ", ".join(ends + ["of the edge %s" % listed(target["values"])])


@timed("reach.worded")
def worded(setting, held, own):
    return "%s %s%s%s" % (
        setting["parameter"],
        held["value"],
        "" if setting["entry"] == own else " of the entry %s" % setting["entry"],
        "".join(
            " %s %s %s%s"
            % (
                "and" if place else "with",
                stated["parameter"],
                stated["value"],
                "" if stated["entry"] == setting["entry"] else " of the entry %s" % stated["entry"],
            )
            for place, stated in enumerate(held["stating"])
        ),
    )


@timed("reach.lifted_text")
def lifted_text(found):
    held = found["without"]
    if held["how"] == KEPT:
        return (
            "the pipeline is refused or stops without the entry and no one parameter of it re-stated leaves more in "
            "the graph, so the entries after it are not examined"
        )
    given = "the pipeline gives it back" if held["back"] else "it is still missing"
    if held["how"] == ABSENT:
        return "without the entry %s" % given
    stated = worded(dict(entry=found["entry"], parameter=held["parameter"]), held, found["entry"])
    return "the pipeline is refused or stops without the entry; under %s, after which the graph holds the most, %s" % (stated, given)


def wrapped(lead, items):
    started = time.perf_counter()
    lines = []
    for item in items:
        if lines and len(lines[-1]) + len(item) < WIDTH:
            lines[-1] += ", " + item
        else:
            lines.append("      %s %s" % (lead, item))
    LOG.debug("reach wrapped exit items=%d lines=%d seconds=%.4f", len(items), len(lines), time.perf_counter() - started)
    return lines


def found_text(found, under):
    started = time.perf_counter()
    back = [worded(setting, held, found["entry"]) for setting in found["settings"] for held in setting[BACK]]
    counts = [sum(len(setting[part]) for setting in found["settings"]) for part in (STILL, REFUSED, STOPPED, UNTRIED)]
    parts = ["statements giving it back %d, not giving it back %d, refused %d, stopped %d, not tried %d" % (len(back), *counts)]
    if found["spared"]:
        parts.append("the parameters of %s, which it reads, not re-stated" % " and ".join(found["spared"]))
    if found["without"] is not None:
        parts.append(lifted_text(found))
    untried = dict.fromkeys(
        "%s %s" % (setting["parameter"], held["value"]) for setting in found["settings"] for held in setting[UNTRIED]
    )
    lines = [
        "    %s%s, its parameters%s re-stated: %s"
        % (
            found["entry"],
            "".join(", %s" % held for held in under),
            "".join(" and those of %s" % name for name in found["restated"][1:]),
            "; ".join(parts),
        )
    ]
    lines += wrapped("given back by", back)
    lines += wrapped("not tried past the first %d statements of" % restated.BOUND, list(untried))
    LOG.debug("reach found text exit entry=%s lines=%d seconds=%.4f", found["entry"], len(lines), time.perf_counter() - started)
    return lines


@timed("reach.truth_text")
def truth_text(on, held, named):
    kept = ", ".join(HIDDEN[reason] if reason in HIDDEN else "the rule %s of what its split may see" % reason.split(" ", 1)[1] for reason in held["hidden"])
    unkept = "none of what its split may see, when it became known and who observed it"
    when = "" if named is None else " at the %s of the task" % named["one"]
    if on == JOINED and held["held"]:
        said = "    the truth joins it to a reference%s, and the view keeps facts%s from the task by: %s"
        return said % (when, "" if named is None else " of that %s" % named["one"], kept or unkept)
    if on == JOINED:
        return "    the truth joins it to no reference%s" % when
    if not held["held"]:
        return "    %s: the truth holds none of it%s" % (held["target"], when)
    if not held["hidden"]:
        return "    %s: the truth holds it%s, and %s keeps it from the task" % (held["target"], when, unkept)
    return "    %s: the truth holds it%s and the view does not show it to the task: %s" % (held["target"], when, kept)


def cut_text(place, cut, timed_by):
    started = time.perf_counter()
    site, entries = SITES[cut["on"]], cut["entries"]
    if not entries:
        lines = ["  cut %d, of being %s: by no entry, %s in no state, the view before the first entry included" % (place + 1, site, site)]
        lines += [truth_text(cut["on"], held, timed_by) for held in cut["truth"]]
    else:
        named = [
            found["entry"] if found["entry"] == found["operation"] else "%s, of the operation %s" % (found["entry"], found["operation"])
            for found in entries
        ]
        lines = [
            "  cut %d, of being %s: by the %s %s%s"
            % (place + 1, site, "entry" if len(named) == 1 else "entries", " and ".join(named), ", each alone" if len(named) > 1 else "")
        ]
        under = []
        for found in entries:
            lines += found_text(found, under)
            if found["without"] is not None:
                held = found["without"]
                stated = "at %s" % worded(dict(entry=found["entry"], parameter=held["parameter"]), held, found["entry"])
                under = under + ["with %s %s" % (found["entry"], "absent" if held["how"] == ABSENT else stated)]
    LOG.debug("reach cut text exit entries=%d lines=%d seconds=%.4f", len(entries), len(lines), time.perf_counter() - started)
    return lines


@timed("reach.cited")
def cited(cuts, place):
    return "cut %d, by %s" % (place + 1, " and ".join(found["entry"] for found in cuts[place]["entries"]) or "no entry")


@timed("reach.facts_text")
def facts_text(facts):
    parts = [
        "%s %s %s its fork" % ("fact" if len(found) == 1 else "facts", ", ".join(found), "in" if held else "not in")
        for held in (True, False)
        for found in [[fact["fact"] for fact in facts if fact["held"] is held]]
        if found
    ]
    return ", ".join(parts)


@timed("reach.lacking_text")
def lacking_text(held):
    where = "%s, %s" % (held["target"], held["field"]) if held["fact"] is None else "%s of %s, %s" % (held["fact"], held["target"], held["field"])
    if held["lacks"] == OBSERVATION:
        return "%s: %s not observed at the step" % (where, ", ".join(held["unobserved"]))
    if held["fact"] is None:
        return "%s: no present fact of its relation joins its ends at the step" % where
    return "%s: %s" % (where, "not in its fork" if held["lacks"] == FACT else "in its fork and in the variant without the edit")


def target_text(target, cuts):
    started = time.perf_counter()
    parts = [facts_text(target["facts"])] if target["facts"] else []
    parts.append("%s, %s" % (spoken(target), features_text(target)) if target["state"] in (JOINED, UNJOINED) else spoken(target))
    if target["cut"] is not None:
        parts.append(cited(cuts, target["cut"]))
    result = "        %s: %s" % (target["target"], "; ".join(parts))
    LOG.debug("reach target text exit target=%s seconds=%.4f", target["target"], time.perf_counter() - started)
    return result


def field_text(name, field, cuts):
    started = time.perf_counter()
    head = "put before the model %s" % ROUTES[field["route"]] if field["put"] else UNPUT
    if "targets" in field:
        lines = ["      %s: %s" % (name, head)] + [target_text(target, cuts) for target in field["targets"]]
    else:
        parts = [head] + ([facts_text(field["facts"])] if field["facts"] else [])
        lines = ["      %s: %s" % (name, "; ".join(parts + ([] if field["cut"] is None else [cited(cuts, field["cut"])])))]
    LOG.debug("reach field text exit field=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
    return lines


def option_text(option, cuts):
    started = time.perf_counter()
    if option["held"]:
        lines = ["    option %s, held by a fit" % option["key"]]
        for name, field in option["fields"].items():
            lines += field_text(name, field, cuts)
    else:
        lines = [
            "    option %s, left out by a fit, lacking %s; %s; nothing is put before the model of the fields it states, %s"
            % (
                option["key"],
                "; ".join(lacking_text(held) for held in option["lacking"]),
                cited(cuts, option["cut"]),
                listed(option["fields"]),
            )
        ]
    LOG.debug("reach option text exit option=%s lines=%d seconds=%.4f", option["key"], len(lines), time.perf_counter() - started)
    return lines


def graph_text(graph, named):
    started = time.perf_counter()
    lines = [
        "",
        "graph %s, %s, %s"
        % (
            graph["name"],
            truth.viewed_text(graph["view"]),
            "; ".join(
                "decisions of %s %d, described %d%s"
                % (
                    task,
                    held["decisions"],
                    held["described"],
                    ", %d holding no point a fit reads" % (held["decisions"] - held["described"])
                    if held["decisions"] > held["described"]
                    else "",
                )
                for task, held in graph["decided"].items()
            ),
        ),
    ]
    for place, cut in enumerate(graph["cuts"]):
        lines += cut_text(place, cut, named)
    for decision in graph["decisions"]:
        lines.append(
            "  decision %s of %s%s, references %s"
            % (
                decision["task"],
                decision["kind"],
                "" if named is None else " at %s %d" % (named["one"], decision["time"]),
                "; ".join("%s %s" % (name, ", ".join(held)) for name, held in decision["references"].items()) or "none",
            )
        )
        for option in decision["options"]:
            lines += option_text(option, graph["cuts"])
    LOG.debug("reach graph text exit graph=%s lines=%d seconds=%.4f", graph["name"], len(lines), time.perf_counter() - started)
    return lines


def realised_text(held):
    started = time.perf_counter()
    if held["back"]:
        placed = "each option an edit of the graph, read in its own fork"
    else:
        placed = "options put into the graph of its family as structure" if held["structure"] else "options read"
        placed += " beside the variant, %s the actor of each action" % ("with" if held["actor"] else "without")
    lines = [
        "realisation of options %s at version %d, reading %s; %s"
        % (held["name"], held["version"], ", ".join(held["reads"]), placed)
    ]
    if held["back"]:
        made = [
            "%s of %s of %s a fact of %s %s the actor" % (role, action, task, found["relation"], "from" if found["actor"] == "source" else "to")
            for task, actions in held["mapping"].items()
            for action, roles in actions.items()
            for role, found in roles.items()
        ]
        added = [
            "%s from %s, %s, with %s"
            % (
                name,
                " or ".join("%s to %s" % tuple(pair) for pair in found["pairs"]),
                "directed" if found["directed"] else "without direction",
                ", ".join(found["columns"]) or "no columns",
            )
            for name, found in held["relations"].items()
        ]
        lines.append("the edit makes %s; it adds the relations %s" % ("; ".join(made), "; ".join(added) or "none"))
    LOG.debug("reach realised text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return lines


def text(record):
    started = time.perf_counter()
    lines = [
        "reach of the options of the variant %s of the bank stored as %s, described at version %d"
        % (record["variant"], record["stored_sha256"], record["version"]),
        *([] if record["refused"] is None else [REFUSAL % record["refused"]]),
        *realised_text(record["realisation"]),
        *effect.settled(record),
        READING[record["time"]["model"]],
    ]
    lines += [effect.entry_text(entry) for entry in record["entries"]]
    lines += [
        "kind of task %s: fields %s; not read by the realisation: %s" % (task, listed(entry["fields"]), listed(entry["unread"]))
        for task, entry in record["tasks"].items()
    ]
    lines.append("kinds of task that decide nothing, which hold no option: %s" % listed(record["undecided"]))
    for graph in record["graphs"]:
        lines += graph_text(graph, truth.NAMED[record["time"]["model"]])
    LOG.debug("reach text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
