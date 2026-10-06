import logging
import time
from collections import ChainMap
from functools import partial
from pathlib import Path

import numpy as np

from graphspec.descriptions import describe as truth
from graphspec.descriptions import effect
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import load_arrays, sha256_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline as pipelines
from graphspec.operations.operation import Refused
from graphspec.schema.decisions import ROLES
from tuning.bank import forks
from tuning.bank.bank import Deriver, editing
from tuning.bank.options import TOKEN, offered
from tuning.batches.inputs import aimed
from tuning.families.heads import REALISATION, fielded, realisation
from tuning.families.model_spec import realised
from tuning.statement import agreed, state

LOG = logging.getLogger(__name__)
HELD, PUT, JOINED, UNJOINED, UNHELD, CREATED = "held", "put", "joined", "unjoined", "unheld", "created"
OPTION, ROLE, TARGET, FACT = "option", "role", "target", "fact"
OBSERVATION, EDIT, ACTOR = "observation", "edit", "actor"
BOUND = 16
ROUTES = {
    OPTION: "with the option",
    ROLE: "with the option, by its role alone",
    TARGET: "as the targets the variant holds",
    FACT: "as the facts its edit makes of it",
}
SITES = {HELD: "held by a fit", PUT: "put before the model", JOINED: "joined to a reference"}
UNPUT = "not put before the model, its fork holding none of the facts its edit makes of it"
READING = (
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
    "features: the columns of the kind of a target's nodes holding a value at the step; "
    "a cut names the last entry of the pipeline after which what it is about stops: for an option a fit leaves out, "
    "being held by a fit; for a field or a target not put before the model, being put before it; for a target joined "
    "by no path, being joined to a reference; no entry where it holds in no state, the view before the first entry "
    "included; and, for each setting of that entry and each it reads, the other values under which the whole pipeline "
    "gives it back, those under which it does not, and those it refuses with every parameter stated; a value is "
    "stated without the parameters it makes inactive and with every value of each parameter it makes active, every "
    "option of a choice and the two ends of a number's range, named after it; at most %d statements are tried for a "
    "value, and those past them are listed as not tried; an end of a reference that the pipeline replaced by a node "
    "of a kind it adds is named by that kind and the node of the view in whose place it stands" % BOUND
)


class Shown(dict):
    @timed("reach.Shown.numpy")
    def numpy(self, name):
        return np.asarray(self[name])


@timed("reach.masked")
def masked(tokens):
    return (np.asarray(tokens).astype(np.uint64) & np.uint64(TOKEN)).astype(np.int64)


@timed("reach.values")
def values(parameter):
    if parameter["kind"] in ("bool", "choice"):
        return list(parameter["options"])
    return list(dict.fromkeys((parameter["low"], parameter["high"])))


@timed("reach.named")
def named(entry):
    return entry["label"] if "label" in entry else entry["operation"]


@timed("reach.stating")
def stating(config, name, stated, dropped):
    pipeline = [
        dict(entry, parameters={key: value for key, value in {**entry["parameters"], **stated}.items() if key not in dropped})
        if named(entry) == name
        else entry
        for entry in config["pipeline"]
    ]
    return dict(config, pipeline=pipeline)


def unsettled(registry, entries):
    started = time.perf_counter()
    found = pipelines.checked(entries, pipelines.STATED)
    reads, places = pipelines.readers(found), {name: place for place, (name, _, _, _) in enumerate(found)}
    stated, growing, result = {}, pipelines.Growing(registry, found), None
    for name in pipelines.ordered(found, reads, "adds"):
        _, operation, _, entry = found[places[name]]
        held = growing.before(places[name], partial(pipelines.specified, stated, reads))
        if operation.lacking(held) is not None:
            break
        own, lacking = {}, None
        view = ChainMap(own, *[stated[read] for read in reads[name]])
        for parameter in operation.stated(held):
            if not parameter.active(view):
                continue
            if parameter.name not in entry["parameters"]:
                lacking = parameter.declaration()
                break
            own[parameter.name] = entry["parameters"][parameter.name]
        unread = sorted(set(entry["parameters"]) - set(own))
        if lacking is not None or unread:
            result = name, lacking, unread
            break
        if operation.refused(held, own) is not None:
            break
        stated[name] = own
    LOG.debug("reach unsettled exit entries=%d settled=%s seconds=%.4f", len(found), result is None, time.perf_counter() - started)
    return result


def looked(graph, step):
    started = time.perf_counter()
    tokens = masked(graph.entity_token)
    order = np.argsort(tokens, kind="stable")
    entity, moment = np.asarray(graph.obs_entity, np.int64), np.asarray(graph.obs_step, np.int64)
    rows = np.full(len(tokens), -1, np.int64)
    at = np.flatnonzero(moment == step)
    rows[entity[at]] = at
    present = np.asarray(graph.fact_present, bool) & (np.asarray(graph.fact_step, np.int64) == step)
    facts = tuple(np.asarray(graph.arrays[name], np.int64)[present] for name in ("fact_source", "fact_target", "fact_code"))
    table, refs = np.asarray(graph.value_table, np.float64), np.asarray(graph.fact_ref, np.int64)[present]
    valued = np.full((len(refs), table.shape[1]), np.nan)
    valued[refs >= 0] = table[refs[refs >= 0]]
    result = dict(tokens=tokens[order], order=order, rows=rows, facts=facts, values=valued)
    LOG.debug("reach looked exit step=%d facts=%d seconds=%.4f", step, len(facts[0]), time.perf_counter() - started)
    return result


@timed("reach.entity")
def entity(seen, token):
    place = int(np.searchsorted(seen["tokens"], token))
    return int(seen["order"][place]) if place < len(seen["tokens"]) and seen["tokens"][place] == token else -1


def walked(seen, sources):
    started = time.perf_counter()
    source, target, code = seen["facts"]
    start, end = np.r_[source, target], np.r_[target, source]
    hops = np.full(len(seen["rows"]), -1, np.int64)
    frontier, depth = np.unique(np.asarray(sources, np.int64)), 0
    hops[frontier] = 0
    while len(frontier):
        depth += 1
        reached = end[np.isin(start, frontier)]
        frontier = np.unique(reached[hops[reached] < 0])
        hops[frontier] = depth
    result = hops, (start, end, np.r_[code, code])
    LOG.debug("reach walked exit sources=%d hops=%d seconds=%.4f", len(sources), depth - 1, time.perf_counter() - started)
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


class Reached:
    def __init__(self, deriver, config, name):
        started = time.perf_counter()
        self.deriver, self.config, self.name, self.entry = deriver, config, name, realisation(name)
        schema = deriver.registry.schema
        self.found, self.columns = offered(deriver), schema.parameters()
        every = {task: fielded(schema, task) for task in schema.decided()}
        self.fields = {task: [field for field in fields if field["sort"] in self.entry.reads] for task, fields in every.items()}
        self.roled = {field["code"]: field["name"] for fields in every.values() for field in fields if field["sort"] == "targets"}
        self.variants, self.derivations, self.graphs, self.arrays = {}, {}, {}, {}
        self.histories, self.plains, self.forks, self.plans, self.walks = {}, {}, {}, {}, {}
        self.holds, self.aims, self.placings, self.options = {}, {}, {}, {}
        self.members, self.besides, self.names, self.restated = {}, {}, {}, {}
        held = self.entered(None, config)
        self.steps, self.spec, self.registry = held[1], held[2], held[4]
        LOG.debug("reached exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)

    def entered(self, key, cfg):
        started = time.perf_counter()
        _, steps, spec, chosen = self.deriver.stated(cfg)
        self.variants[key] = (cfg, steps, spec, chosen, self.deriver.fitted(cfg))
        LOG.debug("reached entered exit entries=%d seconds=%.4f", len(steps), time.perf_counter() - started)
        return self.variants[key]

    def restating(self, holder, parameter, other):
        started = time.perf_counter()
        name = (holder.name, parameter["name"], other)
        if name not in self.restated:
            pending, tried, untried = [(stating(self.config, holder.name, {parameter["name"]: other}, ()), [])], [], []
            while pending:
                cfg, added = pending.pop(0)
                if len(tried) == BOUND:
                    untried.append(added)
                    continue
                held, key = unsettled(self.deriver.registry, cfg["pipeline"]), name + (len(tried),)
                if held is None:
                    try:
                        self.entered(key, cfg)
                    except Refused as error:
                        tried.append((added, None, str(error)))
                    else:
                        tried.append((added, key, None))
                    continue
                entry, lacking, unread = held
                if lacking is None:
                    pending.insert(0, (stating(cfg, entry, {}, unread), added))
                    continue
                pending[:0] = [
                    (
                        stating(cfg, entry, {lacking["name"]: value}, ()),
                        added + [dict(entry=entry, parameter=lacking["name"], value=value)],
                    )
                    for value in values(lacking)
                ]
            self.restated[name] = tried, untried
        LOG.debug(
            "reached restating exit entry=%s parameter=%s statements=%d seconds=%.4f",
            holder.name,
            parameter["name"],
            len(self.restated[name][0]),
            time.perf_counter() - started,
        )
        return self.restated[name]

    def loaded(self, group):
        started = time.perf_counter()
        if group not in self.members:
            self.members[group] = forks.members(self.deriver, self.config, group)
            for view in self.members[group][1]:
                self.names.update(zip(masked(view.entity_token).tolist(), view.keys))
        LOG.debug("reached loaded exit group=%d seconds=%.4f", group, time.perf_counter() - started)
        return self.members[group]

    def begun(self, key, count, group, space):
        started = time.perf_counter()
        cfg, steps, spec, chosen, registry = self.variants[key]
        indices, members = self.loaded(group)
        if (key, group) not in self.besides:
            self.besides[key, group] = self.deriver.seconds(steps, indices, cfg, False)
        result = effect.begun(
            members,
            self.besides[key, group],
            registry,
            steps[:count],
            spec,
            chosen,
            self.deriver.seed,
            self.deriver.limits(),
            self.deriver.partitions(cfg),
            space,
        )
        LOG.debug("reached begun exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return result

    def derivation(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.derivations:
            total = sum(member.steps for member in self.loaded(group)[1])
            self.derivations[key, count, group] = self.begun(key, count, group, 0).continued(total, None)
        LOG.debug("reached derivation exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.derivations[key, count, group]

    def derived(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.graphs:
            self.graphs[key, count, group] = self.derivation(key, count, group).derived()
        LOG.debug("reached derived exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.graphs[key, count, group]

    def shown(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.arrays:
            self.arrays[key, count, group] = Shown(self.derivation(key, count, group).arrays())
        LOG.debug("reached shown exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.arrays[key, count, group]

    def history(self, key, count, decision):
        started = time.perf_counter()
        group, step = decision["group"], decision["step"]
        if (key, count, group, step) not in self.histories:
            begun = self.begun(key, count, group, forks.room(self.deriver))
            self.histories[key, count, group, step] = begun.continued(step, None) if step else begun
        LOG.debug("reached history exit group=%d step=%d seconds=%.4f", group, step, time.perf_counter() - started)
        return self.histories[key, count, group, step]

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
        if (key, count) not in self.plains:
            self.plains[key, count] = forks.listed(self.history(key, count, decision).continued(decision["step"] + 1, None).arrays())
        LOG.debug("reached plain exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return self.plains[key, count]

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
                    time=int(graph.step_time[step]),
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
        result = seen, walked(seen, sources), [registry.by_code[code].name for code in range(registry.relations)]
        LOG.debug("reached looking exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return result

    def featured(self, graph, registry, seen, ends):
        started = time.perf_counter()
        entities, table, refs = registry.schema.entities(), np.asarray(graph.x_table), np.asarray(graph.obs_ref, np.int64)
        found = []
        for end in ends:
            row = table[refs[seen["rows"][end]]]
            for column in entities[registry.nodes[int(graph.kinds[end])]]:
                flags = registry.encoding.slots(column)[1]
                if len(flags) and row[flags[0]] > 0 and column not in found:
                    found.append(column)
        LOG.debug("reached featured exit columns=%d seconds=%.4f", len(found), time.perf_counter() - started)
        return found

    def placed(self, graph, registry, looked_at, ends):
        started = time.perf_counter()
        seen, walk, names = looked_at
        result = dict(state=UNHELD, hops=None, relations=None, features=[])
        if all(end >= 0 and seen["rows"][end] >= 0 for end in ends):
            hops = walk[0][list(ends)]
            reached = hops[hops >= 0]
            features = self.featured(graph, registry, seen, ends)
            result = dict(state=UNJOINED, hops=None, relations=None, features=features)
            if len(reached):
                least = int(reached.min())
                starts = [end for end, hop in zip(ends, hops.tolist()) if hop == least]
                result = dict(state=JOINED, hops=least, relations=traced(walk, names, starts), features=features)
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
                spoken = self.keyed(source, where)
                if sorts[role] == "edge":
                    spoken = "%s %s %s" % (spoken, bank[int(found.role["relation"][role])].name, self.keyed(target, where))
                ends = (int(masked([elements[source]])[0]),) if sorts[role] == "new" else (source, target)[: 1 + (sorts[role] == "edge")]
                result.append(dict(action=action, role=role, code=int(found.role["code"][role]), sort=sorts[role], target=spoken, ends=ends))
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
        keys = [self.keyed(int(end), where) for end in aims.ends[row].tolist() if end >= 0]
        target = keys[0]
        if sort == "edge":
            target = "%s %s %s" % (keys[0], self.deriver.registry.by_code[int(aims.relation[row])].name, keys[1])
        unobserved = [key for key, seen in zip(keys, aims.observed[row].tolist()) if seen < 0]
        result = dict(
            field=self.roled[code] if code in self.roled else ACTOR,
            target=target,
            lacks=OBSERVATION if unobserved else FACT,
            fact=None,
            unobserved=unobserved,
        )
        LOG.debug("reached lacked exit target=%s lacks=%s seconds=%.4f", target, result["lacks"], time.perf_counter() - started)
        return result

    def kept(self, key, count, decision):
        started = time.perf_counter()
        if (key, count) not in self.holds:
            arrays = self.shown(key, count, decision["group"])
            options = self.optioned(decision)
            steps = np.full(len(options), decision["step"], np.int64)
            aims = aimed(self.found, self.name, arrays, 0, len(arrays["snapshot_offsets"]) - 1, np.asarray(options, np.int64), steps)
            owner, result = aims.acting[aims.roling], {}
            for place, option in enumerate(options):
                where = "a target of the option %s" % self.found.keys[option]
                rows = np.flatnonzero((owner == place) & aims.unheld).tolist()
                result[option] = not bool(aims.left[place]), [self.lacked(aims, row, where) for row in rows]
            self.holds[key, count] = result
        LOG.debug("reached kept exit decision=%d seconds=%.4f", decision["decision"], time.perf_counter() - started)
        return self.holds[key, count]

    def beside(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, count, option) not in self.placings:
            registry, graph = self.variants[key][4], self.derived(key, count, decision["group"])
            if (key, count) not in self.walks:
                self.walks[key, count] = self.looking(graph, registry, decision, looked(graph, decision["step"]))
            looked_at, result = self.walks[key, count], {}
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
                            dict(state=CREATED, hops=None, relations=None, features=[])
                            if created
                            else self.placed(graph, registry, looked_at, tuple(entity(looked_at[0], end) for end in target["ends"]))
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
        source, target, code = seen["facts"]
        for relation, first, second, row, action, role in plan.facts:
            ends = [int(marks[first]), int(marks[second])]
            keys = [created[end] if end in created else self.keyed(end, where) for end in ends]
            held = [entity(seen, end) for end in ends]
            edge = registry.by_code[relation]
            joined = (source == held[0]) & (target == held[1])
            if edge.symmetric:
                joined |= (source == held[1]) & (target == held[0])
            rows = np.flatnonzero(joined & (code == relation)) if min(held) >= 0 else np.zeros(0, np.int64)
            place = int(registry.value_group[relation])
            columns = registry.attributes(registry.valued[place]) if place >= 0 else ()
            result.append(
                dict(
                    action=action,
                    role=role,
                    fact="%s %s %s" % (keys[0], edge.name, keys[1]),
                    held=bool(len(rows)),
                    carried=[column for at, column in enumerate(columns) if not np.isnan(row[at])],
                    valued=[column for at, column in enumerate(columns) if not np.isnan(seen["values"][rows, at]).all()],
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
                            **self.placed(graph, registry, looked_at, ends),
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
                held, lacking = self.kept(key, count, decision)[option]
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

    def settable(self, place):
        started = time.perf_counter()
        step, result = self.steps[place], []
        for holder, names in [(step, list(step.parameters))] + [
            (other, [name for name in step.read if name in other.parameters])
            for other in self.steps
            if other.label is None and other.operation.name in step.operation.reads
        ]:
            stated = holder.operation.stated(pipelines.grown(self.deriver.registry, self.steps[: self.steps.index(holder)], None))
            result += [(holder, parameter.declaration()) for name in names for parameter in stated if parameter.name == name]
        LOG.debug("reached settable exit entry=%s settings=%d seconds=%.4f", step.name, len(result), time.perf_counter() - started)
        return result

    def tried(self, decision, option, site, holder, parameter):
        started = time.perf_counter()
        value = holder.parameters[parameter["name"]]
        result = dict(entry=holder.name, parameter=parameter["name"], value=value, back=[], still=[], refused=[], untried=[])
        for other in values(parameter):
            if other == value:
                continue
            tried, untried = self.restating(holder, parameter, other)
            for added, key, reason in tried:
                if key is None:
                    result["refused"].append(dict(value=other, stating=added, reason=reason))
                else:
                    back = self.holding(site, key, len(self.variants[key][1]), decision, option)
                    result["back" if back else "still"].append(dict(value=other, stating=added))
            result["untried"] += [dict(value=other, stating=added) for added in untried]
        LOG.debug(
            "reached tried exit parameter=%s back=%d seconds=%.4f",
            parameter["name"],
            len(result["back"]),
            time.perf_counter() - started,
        )
        return result

    def stopped(self, decision, option, site):
        started = time.perf_counter()
        result = next(
            (count for count in range(len(self.steps) - 1, -1, -1) if self.holding(site, None, count, decision, option)),
            None,
        )
        LOG.debug("reached stopped exit site=%s entry=%s seconds=%.4f", site[0], result, time.perf_counter() - started)
        return result

    def cut(self, decision, option, site):
        started = time.perf_counter()
        last = self.stopped(decision, option, site)
        result = dict(on=site[0], entry=None, operation=None, settings=[])
        if last is not None:
            step = self.steps[last]
            result = dict(
                on=site[0],
                entry=step.name,
                operation=step.operation.name,
                settings=[self.tried(decision, option, site, holder, parameter) for holder, parameter in self.settable(last)],
            )
        LOG.debug("reached cut exit site=%s entry=%s seconds=%.4f", site[0], result["entry"], time.perf_counter() - started)
        return result

    @timed("reached.sited")
    def sited(self, name, place, target):
        if not target["put"]:
            return (PUT, name, place)
        return (JOINED, name, place) if target["state"] == UNJOINED else None

    def option(self, decision, option):
        started = time.perf_counter()
        held = self.read(None, len(self.steps), decision, option)
        result = dict(key=self.found.keys[option], held=held["held"], lacking=held["lacking"], cut=None, fields={})
        if not held["held"]:
            result.update(cut=self.cut(decision, option, (HELD,)), fields=dict(held["fields"]))
        for name, field in held["fields"].items() if held["held"] else ():
            if "targets" in field:
                sites = [self.sited(name, place, target) for place, target in enumerate(field["targets"])]
                targets = [
                    dict(target, cut=None if site is None else self.cut(decision, option, site))
                    for target, site in zip(field["targets"], sites)
                ]
                result["fields"][name] = dict(field, targets=targets)
            else:
                result["fields"][name] = dict(field, cut=None if field["put"] else self.cut(decision, option, (PUT, name, None)))
        LOG.debug("reached option exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return result

    def optioned(self, decision):
        started = time.perf_counter()
        first, count = int(self.found.first[decision["decision"]]), int(self.found.count[decision["decision"]])
        result = sorted(range(first, first + count), key=lambda option: self.found.keys[option])
        LOG.debug("reached optioned exit options=%d seconds=%.4f", count, time.perf_counter() - started)
        return result

    @timed("reached.decision_done")
    def decision_done(self):
        for done in (self.histories, self.plains, self.forks, self.plans, self.walks, self.holds, self.placings, self.options):
            done.clear()

    @timed("reached.group_done")
    def group_done(self):
        for done in (self.derivations, self.graphs, self.arrays, self.besides, self.members):
            done.clear()

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
        held, stated = {}, {}
        for group in self.deriver.graphs(self.config):
            trunk = self.derived(None, len(self.steps), group)
            stated.update(self.counted(trunk, group))
            for decision in self.decided(trunk, group):
                held.setdefault(decision["view"], []).append(
                    dict(
                        task=decision["key"],
                        kind=self.registry.tasks[decision["kind"]],
                        time=decision["time"],
                        references=self.references(trunk, decision),
                        options=[self.option(decision, option) for option in self.optioned(decision)],
                    )
                )
                self.decision_done()
            self.group_done()
        result = []
        for graph in self.deriver.viewed(self.config).graphs:
            decisions = held[graph["index"]] if graph["index"] in held else []
            described = [decision["kind"] for decision in decisions]
            decided = {
                task: dict(decisions=count, described=described.count(task)) for task, count in stated[graph["index"]].items()
            }
            result.append(dict(name=graph["name"], decided=decided, decisions=decisions))
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


def recorded(deriver, config, key, name):
    started = time.perf_counter()
    reached = Reached(deriver, config, name)
    schema, entry = deriver.registry.schema, reached.entry
    every = {task: fielded(schema, task) for task in schema.decided()}
    result = dict(
        version=truth.VERSION,
        variant=key,
        stored_sha256=deriver.source,
        realisation=stated(name),
        seed=deriver.seed,
        device=deriver.device,
        splits=config["splits"],
        settings=specs.materialized(reached.spec),
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


@timed("reach.stemmed")
def stemmed(key, name):
    return sha256_json(dict(variant=key, realisation=stated(name), version=truth.VERSION))


def reach(bank, config, statement):
    started = time.perf_counter()
    truth.manifested(bank)
    stated = state(statement)
    agreed(config, stated)
    deriver = Deriver(bank, stated.device, stated.derivation_seed, RESIDENT)
    name = realising(deriver, config)
    key = specs.digest(deriver.identity(config))
    held = Path(bank) / "descriptions" / "reach" / stemmed(key, name)
    result = truth.kept(held, lambda: recorded(deriver, config, key, name), text)
    LOG.info("reach exit variant=%s realisation=%s seconds=%.3f", key[:10], name, time.perf_counter() - started)
    return result


@timed("reach.listed")
def listed(values):
    return ", ".join(map(str, values)) or "none"


@timed("reach.spoken")
def spoken(held):
    if held["state"] == JOINED:
        return "joined to a reference at hop %d by %s" % (
            held["hops"],
            " then ".join(" or ".join(names) for names in held["relations"]) or "nothing",
        )
    if held["state"] == UNJOINED:
        return "joined to a reference by no path"
    return "an element the option creates, no node of the variant" if held["state"] == CREATED else "not held at the step"


@timed("reach.trials")
def trials(setting, part):
    return listed(
        "%s%s%s"
        % (
            held["value"],
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
            " (%s)" % held["reason"] if "reason" in held else "",
        )
        for held in setting[part]
    )


def cut_text(cut):
    started = time.perf_counter()
    if cut["entry"] is None:
        result = "cut by no entry: %s in no state, the view before the first entry included" % SITES[cut["on"]]
    else:
        result = "cut by the entry %s, of the operation %s, before which it is %s" % (
            cut["entry"],
            cut["operation"],
            SITES[cut["on"]],
        )
        result += "".join(
            "; %s %s of the entry %s gives it back under %s, not under %s, refused under %s%s"
            % (
                setting["parameter"],
                setting["value"],
                setting["entry"],
                trials(setting, "back"),
                trials(setting, "still"),
                trials(setting, "refused"),
                ", not tried under %s" % trials(setting, "untried") if setting["untried"] else "",
            )
            for setting in cut["settings"]
        )
    LOG.debug("reach cut text exit settings=%d seconds=%.4f", len(cut["settings"]), time.perf_counter() - started)
    return result


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


def target_text(target):
    started = time.perf_counter()
    parts = [facts_text(target["facts"])] if target["facts"] else []
    parts.append("%s, features %s" % (spoken(target), listed(target["features"])) if target["state"] in (JOINED, UNJOINED) else spoken(target))
    if target["cut"] is not None:
        parts.append(cut_text(target["cut"]))
    result = "        %s: %s" % (target["target"], "; ".join(parts))
    LOG.debug("reach target text exit target=%s seconds=%.4f", target["target"], time.perf_counter() - started)
    return result


def field_text(name, field):
    started = time.perf_counter()
    head = "put before the model %s" % ROUTES[field["route"]] if field["put"] else UNPUT
    if "targets" in field:
        lines = ["      %s: %s" % (name, head)] + [target_text(target) for target in field["targets"]]
    else:
        parts = [head] + ([facts_text(field["facts"])] if field["facts"] else [])
        lines = ["      %s: %s" % (name, "; ".join(parts + ([] if field["cut"] is None else [cut_text(field["cut"])])))]
    LOG.debug("reach field text exit field=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
    return lines


def option_text(option):
    started = time.perf_counter()
    if option["held"]:
        lines = ["    option %s, held by a fit" % option["key"]]
        for name, field in option["fields"].items():
            lines += field_text(name, field)
    else:
        lines = [
            "    option %s, left out by a fit, lacking %s; %s; nothing is put before the model of the fields it states, %s"
            % (
                option["key"],
                "; ".join(lacking_text(held) for held in option["lacking"]),
                cut_text(option["cut"]),
                listed(option["fields"]),
            )
        ]
    LOG.debug("reach option text exit option=%s lines=%d seconds=%.4f", option["key"], len(lines), time.perf_counter() - started)
    return lines


def graph_text(graph):
    started = time.perf_counter()
    lines = [
        "",
        "graph %s, %s"
        % (
            graph["name"],
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
    for decision in graph["decisions"]:
        lines.append(
            "  decision %s of %s at time %d, references %s"
            % (
                decision["task"],
                decision["kind"],
                decision["time"],
                "; ".join("%s %s" % (name, ", ".join(held)) for name, held in decision["references"].items()) or "none",
            )
        )
        for option in decision["options"]:
            lines += option_text(option)
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
        *realised_text(record["realisation"]),
        *effect.settled(record),
        READING,
    ]
    lines += [effect.entry_text(entry) for entry in record["entries"]]
    lines += [
        "kind of task %s: fields %s; not read by the realisation: %s" % (task, listed(entry["fields"]), listed(entry["unread"]))
        for task, entry in record["tasks"].items()
    ]
    lines.append("kinds of task that decide nothing, which hold no option: %s" % listed(record["undecided"]))
    for graph in record["graphs"]:
        lines += graph_text(graph)
    LOG.debug("reach text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
