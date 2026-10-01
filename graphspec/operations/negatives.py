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
from .rows import counted, fact_bytes

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
                "negatives/<rule>/<name>, and a function from the facts of a segment, the steps, the nodes kept and "
                "the candidates drawn for them, and the settings, to the weight of every candidate"
            )
        self.parameters, self.weights = parameters, weights
        LOG.debug("rule exit seconds=%.4f", time.perf_counter() - started)


class Near:
    def __init__(self, entities, step, source, target, held, counts, tokens):
        started = time.perf_counter()
        ends, others = np.r_[step * entities + source, step * entities + target], np.r_[step * entities + target, step * entities + source]
        order = np.argsort(ends, kind="stable")
        self.entities, self.tokens, self.held, self.counts = entities, tokens, held, counts
        self.ends, self.others = ends[order], others[order]
        LOG.debug("near exit facts=%d seconds=%.4f", len(source), time.perf_counter() - started)

    @timed("negatives.near.degree")
    def degree(self, steps, nodes):
        keys = np.asarray(steps, np.int64) * self.entities + np.asarray(nodes, np.int64)
        position = np.searchsorted(self.held, keys)
        found = np.flatnonzero(position < len(self.held))
        found = found[self.held[position[found]] == keys[found]]
        result = np.zeros(len(keys), np.int64)
        result[found] = self.counts[position[found]]
        return result

    @timed("negatives.near.within")
    def within(self, steps, anchors, candidates, hops):
        count = self.entities
        seeds = np.unique(np.asarray(steps, np.int64) * count + np.asarray(anchors, np.int64))
        reached, origin, frontier = seeds * count + seeds % count, seeds, seeds
        for _ in range(hops):
            low, high = np.searchsorted(self.ends, frontier, "left"), np.searchsorted(self.ends, frontier, "right")
            counts = high - low
            following = self.others[np.repeat(low, counts) + counted(counts)]
            keys = np.unique(np.repeat(origin, counts) * count + following % count)
            fresh = keys[~np.isin(keys, reached)]
            reached = np.union1d(reached, fresh)
            origin = fresh // count
            frontier = origin // count * count + fresh % count
        wanted = (np.asarray(steps, np.int64) * count + np.asarray(anchors, np.int64)) * count + candidates
        return np.isin(wanted, reached)


def unstated(registry):
    return ()


def hopped(registry):
    return (number("negatives/hops/within", 1, 3, 2),)


@timed("negatives.uniform")
def uniform(near, steps, anchors, candidates, settings):
    return np.ones(len(candidates))


@timed("negatives.degree")
def degree(near, steps, anchors, candidates, settings):
    return near.degree(steps, candidates).astype(np.float64)


@timed("negatives.hops")
def hops(near, steps, anchors, candidates, settings):
    return near.within(steps, anchors, candidates, settings["negatives/hops/within"]).astype(np.float64)


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
        return Added({}, {}, {}, {}, {}, {entry: dict(pairs=tuple(dict.fromkeys(pairs)), directed=True)})
    held = registry.schema.sets[name]
    tokens = functools.partial(set_tokens, entry, name, settings["negatives/count"])
    candidates = {
        "%s.%s" % (entry, role): dict(pairs=tuple((entry, kind) for kind in held["kinds"]), directed=True)
        for role in held["roles"]
    }
    return Added({}, {entry: dict(columns=(), tokens=tokens)}, {}, {}, {}, candidates)


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


@timed("negatives.expanded")
def expanded(keys, owners, wanted):
    order = np.argsort(owners, kind="stable")
    owners, keys = owners[order], keys[order]
    low, high = np.searchsorted(owners, wanted, "left"), np.searchsorted(owners, wanted, "right")
    counts = high - low
    return np.repeat(np.arange(len(wanted)), counts), keys[np.repeat(low, counts) + counted(counts)]


class Frame:
    def __init__(self, graph, tokens, width):
        started = time.perf_counter()
        count, present = len(graph.kinds), np.flatnonzero(np.asarray(graph.fact_present, bool))
        self.entities, self.tokens, self.kinds, self.width = count, tokens, np.asarray(graph.kinds, np.int64), width
        self.step = np.asarray(graph.fact_step, np.int64)[present]
        self.source = np.asarray(graph.fact_source, np.int64)[present]
        self.target = np.asarray(graph.fact_target, np.int64)[present]
        self.code = np.asarray(graph.fact_code, np.int64)[present]
        structural = ~np.asarray(graph.obs_anchor, bool)
        entity = np.asarray(graph.obs_entity, np.int64)[structural]
        self.shown = np.unique(np.asarray(graph.obs_step, np.int64)[structural] * count + entity)
        nodes = self.shown % count
        placed = (self.shown // count) * width + self.kinds[nodes]
        order = np.argsort(placed, kind="stable")
        self.placed, self.nodes = placed[order], nodes[order]
        self.marks = np.asarray(graph.step_token, np.uint64)
        self.times = np.asarray(graph.step_time, np.int64)
        LOG.debug("frame exit facts=%d shown=%d seconds=%.4f", len(present), len(self.shown), time.perf_counter() - started)

    @timed("negatives.frame.near")
    def near(self, codes):
        chosen = np.isin(self.code, codes)
        ends = np.r_[self.step[chosen] * self.entities + self.source[chosen], self.step[chosen] * self.entities + self.target[chosen]]
        held, counts = np.unique(ends, return_counts=True)
        return Near(self.entities, self.step, self.source, self.target, held, counts, self.tokens)

    @timed("negatives.frame.candidates")
    def candidates(self, steps, kinds, targets):
        lengths = np.asarray([len(targets.get(kind, ())) for kind in range(self.width)], np.int64)
        offsets = np.r_[0, np.cumsum(lengths)]
        flat = np.asarray([held for kind in range(self.width) for held in targets.get(kind, ())], np.int64)
        counts = lengths[kinds]
        rows = np.repeat(np.arange(len(kinds)), counts)
        wanted = steps[rows] * self.width + flat[np.repeat(offsets[kinds], counts) + counted(counts)]
        low, high = np.searchsorted(self.placed, wanted, "left"), np.searchsorted(self.placed, wanted, "right")
        counts = high - low
        return np.repeat(rows, counts), self.nodes[np.repeat(low, counts) + counted(counts)]


@timed("negatives.referenced")
def referenced(graph, registry, tasks, reference):
    task = np.asarray(graph.member_task, np.int64)
    held = np.isin(task, tasks)
    named = held &(np.asarray(graph.member_name) == list(registry.references).index(reference))
    ends = np.r_[np.asarray(graph.member_source, np.int64)[held], np.asarray(graph.member_target, np.int64)[held]]
    owners = np.r_[task[held], task[held]]
    moments = np.asarray(graph.task_step, np.int64)[task[named]]
    return moments, np.asarray(graph.member_source, np.int64)[named], task[named], ends[ends >= 0], owners[ends >= 0]


@timed("negatives.links")
def links(graph, registry, spec, seed, frame, tasks, rule, budget):
    _, table, reference = parsed(spec["negatives/of"])
    count, tokens = frame.entities, frame.tokens
    edge, codes = registry.named[table], registry.coded[table]
    chosen = np.isin(frame.code, codes)
    moment, first, second = frame.step[chosen], frame.source[chosen], frame.target[chosen]
    if edge.symmetric:
        moment, first, second = np.r_[moment, moment], np.r_[first, second], np.r_[second, first]
    excluded = np.unique((moment * count + first) * count + second)
    if reference is None:
        anchors, quota = np.unique(excluded // count, return_counts=True)
        quota = quota * spec["negatives/count"]
    else:
        moments, named, owner, ends, owners = referenced(graph, registry, tasks, reference)
        anchors = np.unique(moments * count + named)
        quota = np.full(len(anchors), spec["negatives/count"], np.int64)
        at, others = expanded(ends, owners, owner)
        excluded = np.r_[excluded, (moments[at] * count + named[at]) * count + others]
    kept = np.isin(anchors, frame.shown)
    anchors, quota = anchors[kept], quota[kept]
    pairs = edge.pairs + (tuple((second, first) for first, second in edge.pairs) if edge.symmetric else ())
    targets = {}
    for first_kind, second_kind in pairs:
        targets.setdefault(registry.kind(first_kind), []).append(registry.kind(second_kind))
    group, candidates = frame.candidates(anchors // count, frame.kinds[anchors % count], targets)
    budget.pairs(len(candidates), "negative candidates")
    moments, origin = anchors[group] // count, anchors[group] % count
    allowed = (candidates != origin) & ~np.isin(anchors[group] * count + candidates, excluded)
    group, candidates, origin, moments = group[allowed], candidates[allowed], origin[allowed], moments[allowed]
    weights = np.asarray(rule.weights(frame.near(codes), moments, origin, candidates, spec), np.float64)
    keys = (
        np.full(len(candidates), token(table), np.uint64),
        frame.marks[moments],
        frame.times[moments],
        tokens[origin],
        tokens[candidates],
    )
    picked, _ = drawn(seed, SALTS[LINK], weights, group, quota, tokens[candidates], keys)
    budget.retain(len(picked), "negatives")
    return moments[picked], origin[picked], candidates[picked]


@timed("negatives.sets")
def sets(graph, registry, spec, seed, frame, tasks, rule, budget, entry):
    _, kind, reference = parsed(spec["negatives/of"])
    count, kinds, tokens = frame.entities, frame.kinds, frame.tokens
    roles = registry.schema.sets[kind]["roles"]
    codes = np.asarray([registry.code(membership(kind, role)) for role in roles], np.int64)
    made = np.asarray([registry.by_candidate["%s.%s" % (entry, role)].code for role in roles], np.int64)
    position = np.full(registry.relations, -1, np.int64)
    position[codes] = np.arange(len(codes))
    marks = np.asarray([token(role) for role in roles], np.uint64)
    if np.any(frame.code == registry.code(paired(kind))):
        raise ValueError(
            "graphspec.negatives draws negative sets of %s among the sets stored as nodes, and the graph %s stores "
            "them as pairs; state for %s a form of storage that makes each set edge a node" % (kind, graph.name, kind)
        )
    chosen = np.isin(frame.code, codes)
    chosen &= np.isin(frame.step * count + frame.source, frame.shown) & np.isin(frame.step * count + frame.target, frame.shown)
    slots = np.unique(
        np.stack(
            (frame.step[chosen], frame.source[chosen], position[frame.code[chosen]], frame.target[chosen]), 1
        ).reshape(-1, 4),
        axis=0,
    )
    moment, owner, role, member = slots[:, 0], slots[:, 1], slots[:, 2], slots[:, 3]
    owned = moment * count + owner
    part = hashed(marks[role], tokens[member])
    held, place = np.unique(owned, return_inverse=True)
    keys = np.zeros(len(held), np.uint64)
    np.add.at(keys, place.reshape(-1), part)
    anchors = held
    if reference is not None:
        moments, named = referenced(graph, registry, tasks, reference)[:2]
        anchors = np.intersect1d(held, moments * count + named)
    quota = np.full(len(anchors), spec["negatives/count"], np.int64)
    within = np.flatnonzero(np.isin(owned, anchors))
    slot, candidates = frame.candidates(moment[within], kinds[member[within]], {value: (value,) for value in range(frame.width)})
    slot = within[slot]
    budget.pairs(len(candidates), "negative candidates")
    origin, moments = owner[slot], moment[slot]
    after = keys[np.searchsorted(held, owned[slot])] - part[slot] + hashed(marks[role[slot]], tokens[candidates])
    allowed = ~np.isin(owned[slot] * count + candidates, owned * count + member)
    allowed &= ~np.isin(hashed(moments, after), hashed(held // count, keys))
    slot, candidates, origin, moments = slot[allowed], candidates[allowed], origin[allowed], moments[allowed]
    weights = np.asarray(rule.weights(frame.near(codes), moments, origin, candidates, spec), np.float64)
    drawing = (
        frame.marks[moments],
        frame.times[moments],
        tokens[origin],
        marks[role[slot]],
        tokens[member[slot]],
        tokens[candidates],
    )
    picked, rank = drawn(seed, SALTS[SET], weights, np.searchsorted(anchors, owned[slot]), quota, tokens[candidates], drawing)
    nodes = np.flatnonzero(kinds == registry.kind(entry))
    nodes = nodes[np.argsort(tokens[nodes], kind="stable")]
    wanted = hashed(np.full(len(picked), token(entry), np.uint64), tokens[origin[picked]], rank)
    found = np.searchsorted(tokens[nodes], wanted)
    lost = np.flatnonzero((found >= len(nodes)) | (tokens[nodes][np.minimum(found, len(nodes) - 1)] != wanted))
    if len(lost):
        raise ValueError(
            "graphspec.negatives draws a negative of the set of %s of token %d, which no graph given to the "
            "derivation held; the nodes of negative sets are numbered from those graphs before the derivation "
            "begins, and an edit or an earlier operation brought this set" % (kind, int(tokens[origin[picked[lost[0]]]]))
        )
    negative = nodes[found]
    which, index = expanded(np.arange(len(owned)), owned, owned[slot][picked])
    target = member[index].copy()
    swap = index == slot[picked][which]
    target[swap] = candidates[picked][which][swap]
    budget.retain(len(picked) + len(index), "negative sets")
    return moments[picked][which], negative[which], target, made[role[index]]


@timed("operators.negatives")
def negatives(graph, registry, spec, seed, limits, carry, held):
    form, _, reference = parsed(spec["negatives/of"])
    rule, budget = NEGATIVES.named(spec["negatives/rule"]), GraphBudget(limits, fact_bytes(graph))
    tasks = EMPTY if reference is None else np.union1d(spec.split("trained"), spec.split("scored"))
    frame = Frame(graph, carry.tokens(np.arange(len(graph.kinds))), len(registry.nodes))
    if form == LINK:
        moment, source, target = links(graph, registry, spec, seed, frame, tasks, rule, budget)
        code = np.full(len(source), registry.by_candidate[carry.entry].code, np.int64)
    else:
        moment, source, target, code = sets(graph, registry, spec, seed, frame, tasks, rule, budget, carry.entry)
    carry.candidate(code, moment, source, target)
    LOG.debug("negatives exit entry=%s candidates=%d", carry.entry, len(source))
    return graph


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
        changes=(),
        adds=adds,
        kinds=named,
        version=5,
        graphs=1,
    ),
)
