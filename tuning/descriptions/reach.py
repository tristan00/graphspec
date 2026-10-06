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
from tuning.bank.bank import Deriver, editing
from tuning.bank.forks import UNMOVED, Part, made, members, ordered, room
from tuning.bank.options import TOKEN, offered
from tuning.families.heads import REALISATION, fielded, realisation
from tuning.families.model_spec import realised
from tuning.statement import agreed, state

LOG = logging.getLogger(__name__)
CONNECTED, APART, ABSENT = "connected", "apart", "absent"
BOUND = 16
READING = (
    "described: the decisions of a graph of the view of which an option holds a point there, an outcome a fit reads, "
    "which are the decisions whose options a realisation puts before a model; a decision holding none is counted "
    "with its graph and its options are not described; "
    "connected: joined to a reference of the task at its step by present facts, each taken in either direction, at "
    "the least number of hops, by the relations at each hop of every such shortest path; apart: held at the step and "
    "joined to no reference; absent: not held at the step; a node is held where it is observed, an edge where both "
    "its ends are and, unless options are edits, a present fact of its relation joins them, an element an option "
    "creates only in the option's own fork; a kind or a parameter is where the targets of its actions are; features: "
    "the columns of a target's kind holding a value at the step; a cut names the last entry after which the field is "
    "no longer connected and, for each setting of that entry and each it reads, the other values under which the "
    "whole pipeline gives the field back, those under which it does not, and those it refuses with every parameter "
    "stated; a value is stated without the parameters it makes inactive and with every value of each parameter it "
    "makes active, every option of a choice and the two ends of a number's range, named after it; at most %d "
    "statements are tried for a value, and those past them are listed as not tried; an end of a reference that the "
    "pipeline replaced by a node of a kind it adds is named by that kind and the node of the view in whose place it "
    "stands" % BOUND
)


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
    result = dict(tokens=tokens[order], order=order, rows=rows, facts=facts)
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


def summed(results):
    started = time.perf_counter()
    connected = [held for held in results if held["state"] == CONNECTED]
    if connected:
        least = min(held["hops"] for held in connected)
        nearest = [held["relations"] for held in connected if held["hops"] == least]
        result = dict(
            state=CONNECTED,
            hops=least,
            relations=[sorted({name for found in nearest for name in found[at]}) for at in range(least)],
        )
    else:
        result = dict(state=APART if any(held["state"] == APART for held in results) else ABSENT, hops=None, relations=None)
    LOG.debug("reach summed exit targets=%d state=%s seconds=%.4f", len(results), result["state"], time.perf_counter() - started)
    return result


class Reached:
    def __init__(self, deriver, config, name):
        started = time.perf_counter()
        self.deriver, self.config, self.entry = deriver, config, realisation(name)
        schema = deriver.registry.schema
        self.found, self.columns = offered(deriver), schema.parameters()
        self.fields = {task: [field for field in fielded(schema, task) if field["sort"] in self.entry.reads] for task in schema.decided()}
        self.variants, self.graphs, self.histories, self.walks, self.options = {}, {}, {}, {}, {}
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
            self.members[group] = members(self.deriver, self.config, group)
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

    def derived(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.graphs:
            total = sum(member.steps for member in self.loaded(group)[1])
            self.graphs[key, count, group] = self.begun(key, count, group, 0).continued(total, None).derived()
        LOG.debug("reached derived exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.graphs[key, count, group]

    def forked(self, key, count, group, step, option):
        started = time.perf_counter()
        if (key, count, group, step) not in self.histories:
            begun = self.begun(key, count, group, room(self.deriver))
            self.histories[key, count, group, step] = begun.continued(step, None) if step else begun
        history = self.histories[key, count, group, step]
        edit = made(self.deriver, self.variants[key][0], history, Part(group, -1, step, option, UNMOVED), ordered(self.deriver, option))
        result = history.continued(step + 1, edit).derived()
        LOG.debug("reached forked exit option=%d seconds=%.4f", option, time.perf_counter() - started)
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

    def looking(self, graph, registry, decision):
        started = time.perf_counter()
        seen = looked(graph, decision["step"])
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

    def placed(self, graph, registry, looked_at, ends, relation):
        started = time.perf_counter()
        seen, walk, names = looked_at
        held = all(end >= 0 and seen["rows"][end] >= 0 for end in ends)
        if held and relation >= 0:
            source, target, code = seen["facts"]
            joined = (source == ends[0]) & (target == ends[1])
            if registry.by_code[relation].symmetric:
                joined |= (source == ends[1]) & (target == ends[0])
            held = bool((joined & (code == relation)).any())
        result = dict(state=ABSENT, hops=None, relations=None, features=[])
        if held:
            hops = walk[0][list(ends)]
            reached = hops[hops >= 0]
            features = self.featured(graph, registry, seen, ends)
            result = dict(state=APART, hops=None, relations=None, features=features)
            if len(reached):
                least = int(reached.min())
                starts = [end for end, hop in zip(ends, hops.tolist()) if hop == least]
                result = dict(state=CONNECTED, hops=least, relations=traced(walk, names, starts), features=features)
        LOG.debug("reached placed exit state=%s seconds=%.4f", result["state"], time.perf_counter() - started)
        return result

    def targets(self, graph, registry, looked_at, option):
        started = time.perf_counter()
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
        result, where = [], "a target of the option %s" % self.found.keys[option]
        for action, role in rows:
            source, target = int(found.role["source"][role]), int(found.role["target"][role])
            relation, sort = int(found.role["relation"][role]), sorts[role]
            spoken = self.keyed(source, where)
            if sort == "edge":
                spoken = "%s %s %s" % (spoken, bank[relation].name, self.keyed(target, where))
            if sort == "new" and not self.entry.back:
                placed = dict(state=ABSENT, hops=None, relations=None, features=[])
            elif sort == "new":
                placed = self.placed(graph, registry, looked_at, (entity(looked_at[0], int(masked([elements[source]])[0])),), -1)
            else:
                ends = (source, target) if sort == "edge" else (source,)
                joining = relation if sort == "edge" and not self.entry.back else -1
                placed = self.placed(graph, registry, looked_at, tuple(entity(looked_at[0], end) for end in ends), joining)
            result.append(dict(action=action, code=int(found.role["code"][role]), target=spoken, **placed))
        LOG.debug("reached targets exit option=%d targets=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
        return result

    def read(self, key, count, decision, option):
        started = time.perf_counter()
        if (key, count, option) not in self.options:
            registry = self.variants[key][4]
            if self.entry.back:
                graph = self.forked(key, count, decision["group"], decision["step"], option)
                looked_at = self.looking(graph, registry, decision)
            else:
                graph = self.derived(key, count, decision["group"])
                if (key, count) not in self.walks:
                    self.walks[key, count] = self.looking(graph, registry, decision)
                looked_at = self.walks[key, count]
            targets = self.targets(graph, registry, looked_at, option)
            found, task = self.found, self.registry.tasks[decision["kind"]]
            first, many = int(found.actions[0][option]), int(found.actions[1][option])
            result = {}
            for field in self.fields[task]:
                if field["sort"] == "kind":
                    held = targets if many else None
                elif field["sort"] == "targets":
                    held = [target for target in targets if target["code"] == field["code"]] or None
                else:
                    values = found.action["values"][first : first + many, self.columns.index(field["column"])]
                    acting = [
                        first + at
                        for at in range(many)
                        if int(found.action["kind"][first + at]) == field["code"] and not np.isnan(values[at])
                    ]
                    held = [target for target in targets if target["action"] in acting] if acting else None
                if held is None:
                    continue
                result[field["name"]] = summed(held)
                if field["sort"] == "targets":
                    result[field["name"]]["targets"] = [
                        {name: target[name] for name in ("target", "state", "hops", "relations", "features")} for target in held
                    ]
            self.options[key, count, option] = result
        LOG.debug("reached read exit option=%d seconds=%.4f", option, time.perf_counter() - started)
        return self.options[key, count, option]

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

    def tried(self, decision, option, name, holder, parameter):
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
                    state = self.read(key, len(self.variants[key][1]), decision, option)[name]["state"]
                    result["back" if state == CONNECTED else "still"].append(dict(value=other, stating=added))
            result["untried"] += [dict(value=other, stating=added) for added in untried]
        LOG.debug(
            "reached tried exit parameter=%s back=%d seconds=%.4f",
            parameter["name"],
            len(result["back"]),
            time.perf_counter() - started,
        )
        return result

    def stopped(self, decision, option, name):
        started = time.perf_counter()
        result = next(
            (
                count
                for count in range(len(self.steps) - 1, -1, -1)
                if self.read(None, count, decision, option)[name]["state"] == CONNECTED
            ),
            None,
        )
        LOG.debug("reached stopped exit field=%s entry=%s seconds=%.4f", name, result, time.perf_counter() - started)
        return result

    def cut(self, decision, option, name):
        started = time.perf_counter()
        last = self.stopped(decision, option, name)
        result = dict(entry=None, operation=None, settings=[])
        if last is not None:
            step = self.steps[last]
            result = dict(
                entry=step.name,
                operation=step.operation.name,
                settings=[self.tried(decision, option, name, holder, parameter) for holder, parameter in self.settable(last)],
            )
        LOG.debug("reached cut exit field=%s entry=%s seconds=%.4f", name, result["entry"], time.perf_counter() - started)
        return result

    def option(self, decision, option):
        started = time.perf_counter()
        fields = {}
        for name, field in self.read(None, len(self.steps), decision, option).items():
            fields[name] = dict(field, cut=None if field["state"] == CONNECTED else self.cut(decision, option, name))
        result = dict(key=self.found.keys[option], fields=fields)
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
        for done in (self.histories, self.walks, self.options):
            done.clear()

    @timed("reached.group_done")
    def group_done(self):
        for done in (self.graphs, self.besides, self.members):
            done.clear()

    def stating(self, graph, group):
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
        LOG.debug("reached stating exit group=%d graphs=%d seconds=%.4f", group, len(result), time.perf_counter() - started)
        return result

    def graphed(self):
        started = time.perf_counter()
        held, stated = {}, {}
        for group in self.deriver.graphs(self.config):
            trunk = self.derived(None, len(self.steps), group)
            stated.update(self.stating(trunk, group))
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
    if held["state"] != CONNECTED:
        return held["state"]
    return "connected at hop %d by %s" % (held["hops"], " then ".join(" or ".join(names) for names in held["relations"]) or "nothing")


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
        result = "cut by no entry: connected in no state, the view before the first entry included"
    else:
        result = "cut by the entry %s, of the operation %s" % (cut["entry"], cut["operation"])
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


def field_text(name, field):
    started = time.perf_counter()
    lines = ["      %s: %s%s" % (name, spoken(field), "" if field["cut"] is None else "; " + cut_text(field["cut"]))]
    for target in field["targets"] if "targets" in field else ():
        lines.append("        %s: %s, features %s" % (target["target"], spoken(target), listed(target["features"])))
    LOG.debug("reach field text exit field=%s lines=%d seconds=%.4f", name, len(lines), time.perf_counter() - started)
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
            lines.append("    option %s" % option["key"])
            for name, field in option["fields"].items():
                lines += field_text(name, field)
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
