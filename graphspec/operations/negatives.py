import functools
import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.hashing import hashed, token, unit
from graphspec.infra.timing import timed
from graphspec.registries import Names
from graphspec.schema.added import Added
from graphspec.schema.declaration import membership, paired
from graphspec.schema.references import SORTS

from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import Parameter, choice, number
from .rows import counted, extend, fact_bytes, made_rows, observed

LOG = logging.getLogger(__name__)
NEGATIVES = Names("rule of negative sampling")
LINK, SET = "link", "set"
SALTS = dict(link="graphspec.negatives/links", set="graphspec.negatives/sets")
TINY = 2.0**-54
EMPTY = np.zeros(0, np.int64)


class Rule:
    def __init__(self, parameters, weights):
        started = time.perf_counter()
        if not callable(parameters) or not callable(weights):
            raise ValueError(
                "a rule of negative sampling states a function from a registry to its parameters, each named "
                "negatives/<rule>/<name>, and a function from the step, the nodes kept, the candidates drawn for "
                "them and the settings to the weight of every candidate"
            )
        self.parameters, self.weights = parameters, weights
        LOG.debug("rule exit seconds=%.4f", time.perf_counter() - started)


class Near:
    def __init__(self, entities, source, target, degree, tokens):
        started = time.perf_counter()
        ends, others = np.r_[source, target].astype(np.int64), np.r_[target, source].astype(np.int64)
        order = np.argsort(ends, kind="stable")
        self.entities, self.degree, self.tokens = entities, degree, tokens
        self.offsets, self.columns = np.searchsorted(ends[order], np.arange(entities + 1)), others[order]
        LOG.debug("near exit facts=%d seconds=%.4f", len(source), time.perf_counter() - started)

    @timed("negatives.near.within")
    def within(self, anchors, candidates, hops):
        count, anchors = self.entities, np.asarray(anchors, np.int64)
        seeds = np.unique(anchors)
        reached, origin, frontier = seeds * count + seeds, seeds, seeds
        for _ in range(hops):
            counts = self.offsets[frontier + 1] - self.offsets[frontier]
            following = self.columns[np.repeat(self.offsets[frontier], counts) + counted(counts)]
            keys = np.unique(np.repeat(origin, counts) * count + following)
            fresh = keys[~np.isin(keys, reached)]
            reached = np.union1d(reached, fresh)
            origin, frontier = fresh // count, fresh % count
        return np.isin(anchors * count + np.asarray(candidates, np.int64), reached)


def unstated(registry):
    return ()


def hopped(registry):
    return (number("negatives/hops/within", 1, 3, 2),)


@timed("negatives.uniform")
def uniform(near, anchors, candidates, settings):
    return np.ones(len(candidates))


@timed("negatives.degree")
def degree(near, anchors, candidates, settings):
    return near.degree[candidates].astype(np.float64)


@timed("negatives.hops")
def hops(near, anchors, candidates, settings):
    return near.within(anchors, candidates, settings["negatives/hops/within"]).astype(np.float64)


NEGATIVES.register("uniform", Rule(unstated, uniform))
NEGATIVES.register("degree", Rule(unstated, degree))
NEGATIVES.register("hops", Rule(hopped, hops))


@timed("negatives.parsed")
def parsed(stated):
    parts = stated.split(":")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


@timed("negatives.sources")
def sources(registry, table):
    edge = registry.named[table]
    return set(edge.sources) | (set(edge.targets) if edge.symmetric else set())


def offered(registry):
    started = time.perf_counter()
    stated = registry.schema.tasks["references"]
    nodes = [
        name
        for name in registry.references
        if SORTS.named(stated[name]["sort"]).nodes and not SORTS.named(stated[name]["sort"]).edges
    ]
    result = []
    for table in dict.fromkeys(registry.tables.values()):
        result.append("%s:%s" % (LINK, table))
        result += [
            "%s:%s:%s" % (LINK, table, name) for name in nodes if set(stated[name]["kinds"]) & sources(registry, table)
        ]
    for kind in registry.schema.sets:
        result.append("%s:%s" % (SET, kind))
        result += ["%s:%s:%s" % (SET, kind, name) for name in nodes if kind in stated[name]["kinds"]]
    LOG.debug("negatives offered exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


def ruled(name, parameter):
    return lambda spec: spec["negatives/rule"] == name and parameter.active(spec)


def stated(registry):
    started = time.perf_counter()
    options, result = offered(registry), []
    if options:
        result = [
            choice("negatives/of", options, options[0]),
            choice("negatives/rule", NEGATIVES.names(), "uniform"),
            number("negatives/count", 1, 16, 1),
        ]
        for name in NEGATIVES.names():
            for parameter in NEGATIVES.named(name).parameters(registry):
                if not parameter.name.startswith("negatives/%s/" % name):
                    raise ValueError(
                        "the rule of negative sampling %s states the parameter %s, which is not named "
                        "negatives/%s/<name>" % (name, parameter.name, name)
                    )
                result.append(
                    Parameter(
                        parameter.name,
                        parameter.kind,
                        parameter.default,
                        parameter.low,
                        parameter.high,
                        parameter.log,
                        parameter.options,
                        ruled(name, parameter),
                    )
                )
    LOG.debug("negatives stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


def named(registry, settings):
    started = time.perf_counter()
    result = {}
    if registry.tables or registry.schema.sets:
        form, name, _ = parsed(settings["negatives/of"])
        if form == SET:
            result[name] = "names"
    LOG.debug("negatives named exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("negatives.set_tokens")
def set_tokens(entry, kind, count, kinds, tokens, rows, registry):
    held = np.unique(np.asarray(tokens, np.uint64)[np.asarray(kinds) == registry.kind(kind)])
    return np.concatenate(
        [hashed(np.full(len(held), token(entry), np.uint64), held, np.full(len(held), at)) for at in range(count)]
        + [np.zeros(0, np.uint64)]
    )


@timed("negatives.adds")
def adds(registry, settings, entry):
    if not offered(registry):
        raise ValueError(
            "graphspec.negatives draws absent edges of a relation or absent sets of a kind of set edge, and the "
            "bank %s states neither" % registry.name
        )
    form, name, _ = parsed(settings["negatives/of"])
    if form == LINK:
        edge = registry.named[name]
        pairs = edge.pairs + (tuple((second, first) for first, second in edge.pairs) if edge.symmetric else ())
        return Added({}, {}, {}, {entry: dict(pairs=tuple(dict.fromkeys(pairs)), directed=True, columns=())}, {})
    held = registry.schema.sets[name]
    tokens = functools.partial(set_tokens, entry, name, settings["negatives/count"])
    relations = {
        "%s.%s" % (entry, role): dict(pairs=tuple((entry, kind) for kind in held["kinds"]), directed=True, columns=())
        for role in held["roles"]
    }
    return Added({}, {entry: dict(columns=(), tokens=tokens)}, {}, relations, {})


@timed("negatives.drawn")
def drawn(seed, salt, weights, groups, quota, ordered, keys):
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("a rule of negative sampling gave a weight that is negative or not finite")
    kept = np.flatnonzero(weights > 0)
    draws = np.maximum(unit(seed, salt, *(key[kept] for key in keys)), TINY)
    scores = np.log(weights[kept]) - np.log(-np.log(draws))
    order = kept[np.lexsort((ordered[kept], -scores, groups[kept]))]
    first = np.r_[0, np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1] if len(order) else EMPTY
    rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    chosen = rank < quota[groups[order]]
    return order[chosen], rank[chosen]


@timed("negatives.crossed")
def crossed(kinds, choices):
    counts = np.asarray([len(choices[kind]) for kind in kinds.tolist()], np.int64)
    return np.repeat(np.arange(len(kinds)), counts), np.concatenate([choices[kind] for kind in kinds.tolist()] + [EMPTY])


@timed("negatives.expanded")
def expanded(keys, owners, wanted):
    order = np.argsort(owners, kind="stable")
    owners, keys = owners[order], keys[order]
    low, high = np.searchsorted(owners, wanted, "left"), np.searchsorted(owners, wanted, "right")
    counts = high - low
    return np.repeat(np.arange(len(wanted)), counts), keys[np.repeat(low, counts) + counted(counts)]


class Step:
    def __init__(self, graph, moment, rows, shown, tokens):
        started = time.perf_counter()
        self.moment, self.shown, self.tokens = moment, shown, tokens
        self.source = np.asarray(graph.fact_source, np.int64)[rows]
        self.target = np.asarray(graph.fact_target, np.int64)[rows]
        self.code = np.asarray(graph.fact_code, np.int64)[rows]
        self.mark, self.time = np.uint64(graph.step_token[moment]), np.int64(graph.step_time[moment])
        LOG.debug("negatives step exit step=%d facts=%d seconds=%.4f", moment, len(rows), time.perf_counter() - started)

    @timed("negatives.step.near")
    def near(self, entities, codes):
        chosen = np.isin(self.code, codes)
        degree = np.bincount(np.r_[self.source[chosen], self.target[chosen]], minlength=entities)
        return Near(entities, self.source, self.target, degree, self.tokens)


@timed("negatives.referenced")
def referenced(graph, registry, tasks, reference, moment):
    task = np.asarray(graph.member_task, np.int64)
    held = np.isin(task, tasks) & (np.asarray(graph.task_step, np.int64)[task] == moment)
    named = held & (np.asarray(graph.member_name) == list(registry.references).index(reference))
    ends = np.r_[np.asarray(graph.member_source, np.int64)[held], np.asarray(graph.member_target, np.int64)[held]]
    owners = np.r_[task[held], task[held]]
    return np.asarray(graph.member_source, np.int64)[named], task[named], ends[ends >= 0], owners[ends >= 0]


@timed("negatives.links")
def links(graph, registry, spec, seed, step, tasks, rule, budget, entry):
    _, table, reference = parsed(spec["negatives/of"])
    entities, kinds, tokens = len(graph.kinds), np.asarray(graph.kinds, np.int64), step.tokens
    edge, codes = registry.named[table], registry.coded[table]
    chosen = np.isin(step.code, codes)
    first, second = step.source[chosen], step.target[chosen]
    if edge.symmetric:
        first, second = np.r_[first, second], np.r_[second, first]
    excluded = np.unique(first * entities + second)
    if reference is None:
        anchors, quota = np.unique(excluded // entities, return_counts=True)
        quota = quota * spec["negatives/count"]
    else:
        named, owner, ends, owners = referenced(graph, registry, tasks, reference, step.moment)
        anchors = np.unique(named)
        quota = np.full(len(anchors), spec["negatives/count"], np.int64)
        at, others = expanded(ends, owners, owner)
        excluded = np.r_[excluded, named[at] * entities + others]
    kept = np.isin(anchors, step.shown)
    anchors, quota = anchors[kept], quota[kept]
    pairs = edge.pairs + (tuple((second, first) for first, second in edge.pairs) if edge.symmetric else ())
    targets = {}
    for first_kind, second_kind in pairs:
        targets.setdefault(registry.kind(first_kind), []).append(registry.kind(second_kind))
    choices = {
        kind: step.shown[np.isin(kinds[step.shown], targets[kind])] if kind in targets else EMPTY
        for kind in np.unique(kinds[anchors]).tolist()
    }
    group, candidates = crossed(kinds[anchors], choices)
    budget.pairs(len(candidates), "negative candidates")
    origin = anchors[group]
    allowed = (candidates != origin) & ~np.isin(origin * entities + candidates, excluded)
    group, candidates, origin = group[allowed], candidates[allowed], origin[allowed]
    weights = np.asarray(rule.weights(step.near(entities, codes), origin, candidates, spec), np.float64)
    count = len(candidates)
    keys = (
        np.full(count, token(table), np.uint64),
        np.full(count, step.mark, np.uint64),
        np.full(count, step.time, np.int64),
        tokens[origin],
        tokens[candidates],
    )
    picked, _ = drawn(seed, SALTS[LINK], weights, group, quota, tokens[candidates], keys)
    budget.retain(len(picked), "negatives")
    return origin[picked], candidates[picked], np.full(len(picked), registry.code(entry), np.int64)


@timed("negatives.sets")
def sets(graph, registry, spec, seed, step, tasks, rule, budget, nodes, entry):
    _, kind, reference = parsed(spec["negatives/of"])
    entities, kinds, tokens = len(graph.kinds), np.asarray(graph.kinds, np.int64), step.tokens
    roles = registry.schema.sets[kind]["roles"]
    codes = np.asarray([registry.code(membership(kind, role)) for role in roles], np.int64)
    made = np.asarray([registry.code("%s.%s" % (entry, role)) for role in roles], np.int64)
    position = np.full(registry.relations, -1, np.int64)
    position[codes] = np.arange(len(codes))
    marks = np.asarray([token(role) for role in roles], np.uint64)
    if np.any(step.code == registry.code(paired(kind))):
        raise ValueError(
            "graphspec.negatives draws negative sets of %s among the sets stored as nodes, and the graph %s stores "
            "them as pairs; state for %s a form of storage that makes each set edge a node" % (kind, graph.name, kind)
        )
    chosen = np.isin(step.code, codes) & np.isin(step.source, step.shown) & np.isin(step.target, step.shown)
    slots = np.unique(
        np.stack((step.source[chosen], position[step.code[chosen]], step.target[chosen]), 1).reshape(-1, 3), axis=0
    )
    owner, role, member = slots[:, 0], slots[:, 1], slots[:, 2]
    part = hashed(marks[role], tokens[member])
    held, place = np.unique(owner, return_inverse=True)
    keys = np.zeros(len(held), np.uint64)
    np.add.at(keys, place.reshape(-1), part)
    anchors = held if reference is None else np.intersect1d(held, referenced(graph, registry, tasks, reference, step.moment)[0])
    quota = np.full(len(anchors), spec["negatives/count"], np.int64)
    within = np.flatnonzero(np.isin(owner, anchors))
    choices = {value: step.shown[kinds[step.shown] == value] for value in np.unique(kinds[member[within]]).tolist()}
    slot, candidates = crossed(kinds[member[within]], choices)
    slot = within[slot]
    budget.pairs(len(candidates), "negative candidates")
    origin = owner[slot]
    after = keys[np.searchsorted(held, origin)] - part[slot] + hashed(marks[role[slot]], tokens[candidates])
    allowed = ~np.isin(origin * entities + candidates, owner * entities + member) & ~np.isin(after, keys)
    slot, candidates, origin = slot[allowed], candidates[allowed], origin[allowed]
    weights = np.asarray(rule.weights(step.near(entities, codes), origin, candidates, spec), np.float64)
    count = len(candidates)
    drawing = (
        np.full(count, step.mark, np.uint64),
        np.full(count, step.time, np.int64),
        tokens[origin],
        marks[role[slot]],
        tokens[member[slot]],
        tokens[candidates],
    )
    picked, rank = drawn(seed, SALTS[SET], weights, np.searchsorted(anchors, origin), quota, tokens[candidates], drawing)
    wanted = hashed(np.full(len(picked), token(entry), np.uint64), tokens[origin[picked]], rank)
    found = np.searchsorted(nodes[0], wanted)
    lost = np.flatnonzero((found >= len(nodes[0])) | (nodes[0][np.minimum(found, len(nodes[0]) - 1)] != wanted))
    if len(lost):
        raise ValueError(
            "graphspec.negatives draws a negative of the set of %s of token %d, which no graph given to the "
            "derivation held; the nodes of negative sets are numbered from those graphs before the derivation "
            "begins, and an edit or an earlier operation brought this set" % (kind, int(tokens[origin[picked[lost[0]]]]))
        )
    negative = nodes[1][found]
    which, index = expanded(np.arange(len(owner)), owner, origin[picked])
    target = member[index].copy()
    swap = index == slot[picked][which]
    target[swap] = candidates[picked][which][swap]
    budget.retain(len(picked) + len(index), "negative sets")
    return negative, negative[which], target, made[role[index]]


@timed("operators.negatives")
def negatives(graph, registry, spec, seed, limits, carry, held):
    form, _, reference = parsed(spec["negatives/of"])
    rule, budget = NEGATIVES.named(spec["negatives/rule"]), GraphBudget(limits, fact_bytes(graph))
    tasks = EMPTY if reference is None else np.union1d(spec.split("trained"), spec.split("scored"))
    entities, structural = len(graph.kinds), ~np.asarray(graph.obs_anchor, bool)
    tokens = carry.tokens(np.arange(entities))
    present = np.flatnonzero(np.asarray(graph.fact_present, bool))
    fact_bounds = np.searchsorted(np.asarray(graph.fact_step)[present], np.arange(graph.steps + 1))
    obs_bounds = np.searchsorted(graph.obs_step, np.arange(graph.steps + 1))
    nodes = None
    if form == SET:
        made = np.flatnonzero(np.asarray(graph.kinds) == registry.kind(carry.entry))
        order = np.argsort(tokens[made], kind="stable")
        nodes = tokens[made][order], made[order]
    observations, facts = [], []
    for moment in range(graph.steps):
        lo, hi = obs_bounds[moment], obs_bounds[moment + 1]
        shown = np.unique(np.asarray(graph.obs_entity, np.int64)[lo:hi][structural[lo:hi]])
        step = Step(graph, moment, present[fact_bounds[moment] : fact_bounds[moment + 1]], shown, tokens)
        if form == LINK:
            facts.append((moment,) + links(graph, registry, spec, seed, step, tasks, rule, budget, carry.entry))
        else:
            found = sets(graph, registry, spec, seed, step, tasks, rule, budget, nodes, carry.entry)
            observations.append((moment, found[0]))
            facts.append((moment,) + found[1:])
    if observations and sum(len(part[1]) for part in observations):
        graph, references = made_rows(graph, carry, np.zeros((1, graph.x_table.shape[1]), graph.x_table.dtype))
        entity = np.concatenate([part[1] for part in observations])
        moment = np.concatenate([np.full(len(part[1]), part[0], np.int64) for part in observations])
        graph = observed(graph, entity, moment, np.full(len(entity), references[0], np.int64), np.zeros(len(entity), bool))
    source, target, code = (np.concatenate([part[place] for part in facts] + [EMPTY]) for place in (1, 2, 3))
    moment = np.concatenate([np.full(len(part[1]), part[0], np.int64) for part in facts] + [EMPTY])
    LOG.debug("negatives exit entry=%s facts=%d", carry.entry, len(source))
    return extend(graph, source, target, code, moment, np.ones(len(source), bool), np.full(len(source), -1, np.int64))


def drawing(rule):
    return lambda registry, spec: spec["negatives/rule"] == rule


NEGATIVE = OPERATIONS.register(
    "graphspec.negatives",
    Operation(
        namespace="graphspec",
        name="negatives",
        phases=(Phase("negatives", negatives, None, None),),
        parameters=stated,
        reads=(),
        splits=("trained", "scored"),
        requires=(
            Count("negatives/degree", "degree", drawing("degree"), "stored", "stored"),
            Count("negatives/hops", "paths", drawing("hops"), "stored", "stored"),
        ),
        changes=("observations", "facts"),
        adds=adds,
        kinds=named,
        version=1,
        graphs=1,
    ),
)
