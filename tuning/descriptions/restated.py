import logging
import time
from collections import ChainMap
from functools import partial

import numpy as np

from graphspec.budgets import GraphCapacityError
from graphspec.descriptions import effect
from graphspec.facts.stored import Facts
from graphspec.infra.timing import timed
from graphspec.operations import pipeline as pipelines
from graphspec.operations import sets
from graphspec.operations.operation import Refused
from graphspec.views.view import stepped
from tuning.bank import forks
from tuning.bank.options import TOKEN

LOG = logging.getLogger(__name__)
REFUSED, STOPPED = "refused", "stopped"
ABSENT, LEAST, KEPT = "absent", "least", "kept"
STOPS = (GraphCapacityError, Refused)
BOUND = 16


class Shown(dict):
    @timed("restated.Shown.numpy")
    def numpy(self, name):
        return np.asarray(self[name])


@timed("restated.masked")
def masked(tokens):
    return (np.asarray(tokens).astype(np.uint64) & np.uint64(TOKEN)).astype(np.int64)


@timed("restated.values")
def values(parameter):
    if parameter["kind"] in ("bool", "choice"):
        return list(parameter["options"])
    return list(dict.fromkeys((parameter["low"], parameter["high"])))


@timed("restated.named")
def named(entry):
    return entry["label"] if "label" in entry else entry["operation"]


@timed("restated.stating")
def stating(config, name, stated, dropped):
    pipeline = [
        dict(entry, parameters={key: value for key, value in {**entry["parameters"], **stated}.items() if key not in dropped})
        if named(entry) == name
        else entry
        for entry in config["pipeline"]
    ]
    return dict(config, pipeline=pipeline)


@timed("restated.halted")
def halted(error):
    return STOPPED, dict(error=type(error).__name__, reason=str(error))


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
    LOG.debug("restated unsettled exit entries=%d settled=%s seconds=%.4f", len(found), result is None, time.perf_counter() - started)
    return result


def viewed(deriver, config):
    started = time.perf_counter()
    held = deriver.viewed(config)
    looking, observers = held.looking, deriver.manifest["observers"]
    names = {graph["index"]: graph["name"] for graph in deriver.manifest["graphs"]}
    ruled = [code for code, rules in looking.rules.items() if rules]
    shared = [name for code, name in enumerate(looking.splits) if code not in ruled]
    result = {
        graph["index"]: dict(
            graph=names[graph["source"]],
            observer=None if graph["observer"] < 0 else observers[graph["observer"]],
            splits=list(shared) if graph["split"] < 0 else [looking.splits[graph["split"]]],
        )
        for graph in held.graphs
    }
    LOG.debug("restated viewed exit views=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def unviewed(deriver, config):
    started = time.perf_counter()
    held = deriver.viewed(config)
    looking, shown, result = held.looking, {graph["source"] for graph in held.graphs}, []
    for graph in deriver.manifest["graphs"]:
        if graph["index"] in shown:
            continue
        facts = sets.stored(Facts.load(deriver.truth(graph["index"])), deriver.registry, held.storage)
        split = np.asarray(facts.task_split, np.int64)
        result.append(
            dict(
                name=graph["name"],
                tasks=len(split),
                stated=int(np.isin(split, looking.trained + looking.scored).sum()),
                sequences=len(stepped(facts, deriver.registry, held.steps, np.arange(len(split)))),
            )
        )
    LOG.debug("restated unviewed exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


class Derived:
    def __init__(self, deriver, config):
        started = time.perf_counter()
        self.deriver, self.config = deriver, config
        self.variants, self.restated, self.lifts, self.members, self.names = {}, {}, {}, {}, {}
        self.derivations, self.graphs, self.arrays, self.histories, self.besides = {}, {}, {}, {}, {}
        self.caches = [self.derivations, self.graphs, self.arrays, self.histories, self.besides]
        held = self.entered(None, config)
        self.steps, self.spec, self.registry = held[1], held[2], held[4]
        LOG.debug("derived exit entries=%d seconds=%.4f", len(self.steps), time.perf_counter() - started)

    def entered(self, key, cfg):
        started = time.perf_counter()
        _, steps, spec, chosen = self.deriver.stated(cfg)
        self.variants[key] = (cfg, steps, spec, chosen, self.deriver.fitted(cfg))
        LOG.debug("derived entered exit entries=%d seconds=%.4f", len(steps), time.perf_counter() - started)
        return self.variants[key]

    def settled(self, name, cfg):
        started = time.perf_counter()
        if name not in self.restated:
            pending, tried, untried = [(cfg, [])], [], []
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
                        tried.append((added, None, (REFUSED, dict(reason=str(error)))))
                    except GraphCapacityError as error:
                        tried.append((added, None, halted(error)))
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
        LOG.debug("derived settled exit statements=%d seconds=%.4f", len(self.restated[name][0]), time.perf_counter() - started)
        return self.restated[name]

    @timed("derived.restating")
    def restating(self, base, holder, parameter, other):
        cfg = stating(self.variants[base][0], holder.name, {parameter["name"]: other}, ())
        return self.settled((base, holder.name, parameter["name"], other), cfg)

    def settable(self, base, place):
        started = time.perf_counter()
        steps = self.variants[base][1]
        step, result = steps[place], []
        for holder, names in [(step, list(step.parameters))] + [
            (other, [name for name in step.read if name in other.parameters])
            for other in steps
            if other.label is None and other.operation.name in step.operation.reads
        ]:
            stated = holder.operation.stated(pipelines.grown(self.deriver.registry, steps[: steps.index(holder)], None))
            result.append((holder, [parameter.declaration() for name in names for parameter in stated if parameter.name == name]))
        LOG.debug("derived settable exit entry=%s holders=%d seconds=%.4f", step.name, len(result), time.perf_counter() - started)
        return result

    def loaded(self, group):
        started = time.perf_counter()
        if group not in self.members:
            self.members[group] = forks.members(self.deriver, self.config, group)
            for view in self.members[group][1]:
                self.names.update(zip(masked(view.entity_token).tolist(), view.keys))
        LOG.debug("derived loaded exit group=%d seconds=%.4f", group, time.perf_counter() - started)
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
        LOG.debug("derived begun exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return result

    def derivation(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.derivations:
            total = sum(member.steps for member in self.loaded(group)[1])
            self.derivations[key, count, group] = self.begun(key, count, group, 0).continued(total, None)
        LOG.debug("derived derivation exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.derivations[key, count, group]

    def derived(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.graphs:
            self.graphs[key, count, group] = self.derivation(key, count, group).derived()
        LOG.debug("derived derived exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.graphs[key, count, group]

    def shown(self, key, count, group):
        started = time.perf_counter()
        if (key, count, group) not in self.arrays:
            self.arrays[key, count, group] = Shown(self.derivation(key, count, group).arrays())
        LOG.debug("derived shown exit group=%d entries=%d seconds=%.4f", group, count, time.perf_counter() - started)
        return self.arrays[key, count, group]

    def history(self, key, count, decision):
        started = time.perf_counter()
        group, step = decision["group"], decision["step"]
        held = self.histories[key, count, group] if (key, count, group) in self.histories else None
        if held is None or held[0] > step:
            held = 0, self.begun(key, count, group, forks.room(self.deriver))
        if held[0] < step:
            held = step, held[1].continued(step, None)
        self.histories[key, count, group] = held
        LOG.debug("derived history exit group=%d step=%d seconds=%.4f", group, step, time.perf_counter() - started)
        return held[1]

    def weighed(self, key, count, group):
        started = time.perf_counter()
        try:
            graph = self.derived(key, count, group)
            result = len(graph.obs_entity) + int(np.asarray(graph.fact_present, bool).sum())
        except STOPS:
            result = None
        LOG.debug("derived weighed exit group=%d weight=%s seconds=%.4f", group, result, time.perf_counter() - started)
        return result

    def lifted(self, base, place, group):
        started = time.perf_counter()
        if (base, place, group) not in self.lifts:
            cfg, steps = self.variants[base][0], self.variants[base][1]
            step = steps[place]
            without = dict(cfg, pipeline=[entry for entry in cfg["pipeline"] if named(entry) != step.name])
            absent = [(added, key) for added, key, _ in self.settled((base, step.name, ABSENT), without)[0] if key is not None]
            found = next(((added, key) for added, key in absent if self.weighed(key, len(steps) - 1, group) is not None), None)
            result = dict(how=KEPT, key=None, parameter=None, value=None, stating=[])
            if found is not None:
                result = dict(how=ABSENT, key=found[1], parameter=None, value=None, stating=found[0])
            most = None if found is not None else self.weighed(base, place + 1, group)
            for parameter in () if found is not None else self.settable(base, place)[0][1]:
                for other in [other for other in values(parameter) if other != step.parameters[parameter["name"]]]:
                    for added, key, _ in self.restating(base, step, parameter, other)[0]:
                        weight = None if key is None else self.weighed(key, place + 1, group)
                        if weight is not None and weight > most:
                            result, most = dict(how=LEAST, key=key, parameter=parameter["name"], value=other, stating=added), weight
                        self.released(key)
            self.lifts[base, place, group] = result
        LOG.debug("derived lifted exit entry=%d how=%s seconds=%.4f", place, self.lifts[base, place, group]["how"], time.perf_counter() - started)
        return self.lifts[base, place, group]

    def effected(self, key, group, low, high):
        started = time.perf_counter()
        _, steps, _, _, registry = self.variants[key]
        states = [self.derived(key, count, group) for count in range(low, high + 1)]
        result = effect.told(states, self.loaded(group)[1], registry, steps[low:high])
        LOG.debug("derived effected exit group=%d entries=%d seconds=%.4f", group, high - low, time.perf_counter() - started)
        return result

    def effects(self, group):
        started = time.perf_counter()
        result, cutting = self.effected(None, group, 0, len(self.steps)), {}
        for at, graph in enumerate(result):
            for held in graph["effects"]:
                for cut in held["cuts"]:
                    cutting.setdefault(held["entry"], []).append((at, cut["sort"], cut["name"], cut["task"]))
        waiting = [(None, [], name, marks) for name, marks in cutting.items()]
        while waiting:
            base, without, name, carried = waiting.pop(0)
            place = [step.name for step in self.variants[base][1]].index(name)
            lifted = self.lifted(base, place, group)
            if lifted["key"] is None:
                continue
            steps, found = self.variants[lifted["key"]][1], {}
            under = without + [dict(entry=name, how=lifted["how"])]
            ended = self.effected(lifted["key"], group, len(steps), len(steps))
            carried = [mark for mark in carried if not effect.left(ended[mark[0]], *mark[1:])]
            for count in range(len(steps) - 1, place - (lifted["how"] == ABSENT), -1):
                told = self.effected(lifted["key"], group, count, count + 1) if carried else []
                for at, graph in enumerate(told):
                    for cut in graph["effects"][0]["cuts"]:
                        mark = (at, cut["sort"], cut["name"], cut["task"])
                        if mark in carried:
                            carried.remove(mark)
                            found.setdefault(steps[count].name, []).append(mark)
                            (alone,) = [held for held in result[at]["effects"] if held["entry"] == steps[count].name]
                            alone["cuts"].append(dict(cut, without=under))
            waiting += [(lifted["key"], under, entry, marks) for entry, marks in found.items()]
        LOG.debug("derived effects exit group=%d graphs=%d seconds=%.4f", group, len(result), time.perf_counter() - started)
        return result

    @timed("derived.released")
    def released(self, key):
        for done in self.caches if key is not None else ():
            for name in [name for name in done if name[0] == key]:
                del done[name]

    @timed("derived.group_done")
    def group_done(self):
        for done in self.caches + [self.members, self.lifts]:
            done.clear()
