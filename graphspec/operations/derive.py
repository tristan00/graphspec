import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.hashing import unit
from graphspec.infra.timing import timed
from graphspec.schema.declaration import HOPS, completion, hop

from . import kernels, paths
from .operation import OPERATIONS, Count, Operation, Phase
from .parameters import among, choice, enabled, equals, flag, number, real
from .relations import joined, relating
from .rows import REWIRE, fact_bytes, grouped, prune, pruned, runs

LOG = logging.getLogger(__name__)
TINY = 2.0**-54
SERIES = dict(ppr=("geometric", "derive/ppr_alpha"), heat=("poisson", "derive/heat_time"))


def anchoring(registry):
    started = time.perf_counter()
    nodes, pairs = registry.declared, joined(registry)
    result = {kind: tuple(other for other in nodes if frozenset((kind, other)) in pairs) for kind in nodes}
    LOG.debug("anchoring exit kinds=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def stated(registry):
    started = time.perf_counter()
    nodes, anchors = registry.declared, anchoring(registry)
    projected = [edge for edge in registry.edges if edge.origin[0] == "project"]
    result = [flag("derive/" + edge.name, False) for edge in projected]
    result.append(
        number(
            "derive/project_cap",
            0,
            64,
            0,
            lambda spec: any(enabled(spec, "derive/" + edge.name) for edge in projected),
        )
    )
    result += [flag("derive/complete/" + kind, False) for kind in nodes]
    result += [
        flag(
            "derive/complete_within/" + kind,
            False,
            lambda spec, kind=kind: any(enabled(spec, "derive/complete/" + other) for other in anchors[kind]),
        )
        for kind in nodes
    ]
    related = relating(registry)
    result += [
        number("derive/hops", 1, HOPS[-1], 1, related),
        choice("derive/diffusion", ("none", "ppr", "heat"), "none", related),
        number("derive/diffusion_top_k", 1, 16, 4, among("derive/diffusion", ("ppr", "heat"))),
        real("derive/ppr_alpha", 0.05, 0.5, 0.15, equals("derive/diffusion", "ppr")),
        real("derive/heat_time", 0.5, 5.0, 2.0, equals("derive/diffusion", "heat")),
        choice("derive/rewiring", ("none", "sdrf", "fosr"), "none", related),
        number("derive/rewiring_budget", 1, 64, 8, among("derive/rewiring", ("sdrf", "fosr"))),
        real("derive/sdrf_temperature", 1.0, 500.0, 115.0, equals("derive/rewiring", "sdrf"), log=True),
        real("derive/sdrf_threshold", 0.1, 20.0, 1.0, equals("derive/rewiring", "sdrf"), log=True),
    ]
    LOG.debug("derive stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return tuple(result)


def named(registry, settings):
    result = {}
    for edge in registry.edges:
        if edge.origin[0] == "project" and enabled(settings, "derive/" + edge.name):
            result.update(dict.fromkeys(edge.origin[3:], "names"))
    for kind in registry.declared:
        if enabled(settings, "derive/complete/" + kind) or enabled(settings, "derive/complete_within/" + kind):
            result[kind] = "names"
    return result


@timed("derive.closing")
def closing(spec):
    return "derive/hops" in spec and spec["derive/hops"] > 1


@timed("derive.diffusing")
def diffusing(spec):
    return "derive/diffusion" in spec and spec["derive/diffusion"] != "none"


@timed("derive.rewiring")
def rewiring(spec):
    return "derive/rewiring" in spec and spec["derive/rewiring"] != "none"


@timed("transform.projecting")
def projecting(registry, spec):
    return any(enabled(spec, "derive/" + edge.name) for edge in registry.edges if edge.origin[0] == "project")


@timed("derive.undirected")
def undirected(source, target, entities):
    use = source != target
    key = np.unique(
        np.minimum(source[use], target[use]).astype(np.int64) * entities
        + np.maximum(source[use], target[use])
    )
    return (key // entities).astype(np.int32), (key % entities).astype(np.int32)


@timed("derive.recalled")
def recalled(memo, name, inputs, budget, compute):
    held = memo.get(name)
    if held is not None and all(np.array_equal(old, new) for old, new in zip(held[0], inputs)):
        budget.retain(held[2], name)
        return held[1]
    before = budget.facts
    result = compute()
    memo[name] = (inputs, result, budget.facts - before)
    return result


@timed("operators.project")
def project(graph, registry, edge, source, target, code, recency, cap, tokens, budget, memo):
    group, side, first, second = edge.origin[1:]
    use = np.flatnonzero(np.isin(code, registry.coded[group]))
    members, owners = (source[use], target[use]) if side == "sources" else (target[use], source[use])
    ages = recency[use]

    @timed("derive.project.compute")
    def compute():
        kept = paths.latest(owners, members, ages, tokens, cap)
        held, at, kinds = members[kept], owners[kept], np.asarray(graph.kinds)[members[kept]]
        left, right = kinds == registry.kind(first), kinds == registry.kind(second)
        source, target, _ = paths.product(
            ((held[left], at[left], None), (at[right], held[right], None)), len(graph.kinds), None, budget, "projection"
        )
        if first == second:
            source, target = source[source < target], target[source < target]
        return source.astype(np.int32), target.astype(np.int32)

    return recalled(memo, edge.name, (members, owners, ages), budget, compute)


@timed("operators.complete")
def complete(graph, registry, selected, within, present, u, v, budget):
    members = present[np.isin(graph.kinds[present], selected)]
    if not len(within):
        flat, offsets = members, np.asarray([0, len(members)], np.int64)
    else:
        anchors, linked = np.concatenate((u, v)), np.concatenate((v, u))
        use = np.isin(graph.kinds[anchors], within) & np.isin(linked, members)
        anchors, linked = anchors[use], linked[use]
        order = np.argsort(anchors, kind="stable")
        anchors, flat = anchors[order], linked[order]
        offsets = (
            np.r_[0, np.flatnonzero(np.diff(anchors)) + 1, len(anchors)]
            if len(order)
            else np.zeros(1, np.int64)
        )
    source, target, _, _ = paths.reduced(
        *kernels.within_pairs(flat, offsets, graph.kinds, len(registry.nodes), budget.pair_limit), None, None
    )
    table = np.full((len(registry.nodes), len(registry.nodes)), -1, np.int64)
    for i, left in enumerate(registry.declared):
        for right in registry.declared[i:]:
            table[i, registry.kind(right)] = registry.code(completion(left, right))
    return source, target, table[graph.kinds[source], graph.kinds[target]]


@timed("operators.closure")
def closure(registry, u, v, limit, entities, budget):
    return [
        (a, b, np.full(len(a), registry.code(hop(distance)), np.int64))
        for distance, (a, b) in enumerate(paths.closure(u, v, limit, entities, budget, True, "hop closure"), 2)
    ]


@timed("derive.blocks")
def blocks(labels):
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(labels[order]) != 0, True])
    return [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:]) if hi - lo > 1]


@timed("derive.diffusion_scores")
def diffusion_scores(adjacency, spec, device):
    series, parameter = SERIES[spec["derive/diffusion"]]
    scores = paths.summed(adjacency, series, spec[parameter], device)
    scores[adjacency] = -np.inf
    np.fill_diagonal(scores, -np.inf)
    return scores


@timed("operators.ranked")
def ranked(present, u, v, tokens):
    order = np.argsort(tokens[present], kind="stable")
    canonical, rank = present[order], np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    first, second = rank[np.searchsorted(present, u)], rank[np.searchsorted(present, v)]
    low, high = np.minimum(first, second), np.maximum(first, second)
    pairs = np.lexsort((high, low))
    return canonical, low[pairs], high[pairs]


@timed("operators.diffuse")
def diffuse(registry, spec, present, u, v, tokens, budget):
    from graphspec.infra import compute

    count = len(present)
    if count < 2:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    canonical, low, high = ranked(present, u, v, tokens)
    sources, targets, retained = [], [], 0
    for members, source, target in component_edges(np.arange(count), low, high):
        budget.dense(len(members), "diffusion")
        budget.pairs(
            retained + len(members) * min(spec["derive/diffusion_top_k"], len(members) - 1),
            "diffusion",
        )
        adjacency = np.zeros((len(members), len(members)), bool)
        adjacency[source, target] = adjacency[target, source] = True
        scores = diffusion_scores(adjacency, spec, budget.limits.device)
        width = min(spec["derive/diffusion_top_k"], len(members) - 1)
        chosen, scored = compute.top(scores, width, budget.limits.device)
        valid = np.isfinite(scored)
        rows = np.broadcast_to(np.arange(len(members))[:, None], chosen.shape)
        sources.append(canonical[members[rows[valid]]])
        targets.append(canonical[members[chosen[valid]]])
        retained += int(valid.sum())
    source = np.concatenate(sources + [np.zeros(0, np.int32)]).astype(np.int32)
    target = np.concatenate(targets + [np.zeros(0, np.int32)]).astype(np.int32)
    return source, target, np.full(len(source), registry.code("diffusion"), np.int64)


@timed("derive.neighbours")
def neighbours(adjacency):
    return np.r_[0, np.cumsum(adjacency.sum(1))].astype(np.int64), np.nonzero(adjacency)[1].astype(np.int64)


@timed("derive.forman")
def forman(adjacency, sources, targets):
    offsets, columns = neighbours(adjacency)
    return kernels.forman(adjacency, offsets, columns, sources.astype(np.int64), targets.astype(np.int64))


@timed("derive.spectral")
def spectral(adjacency, members, device):
    from graphspec.infra import compute

    local = adjacency[np.ix_(members, members)]
    if local.sum() == len(members) * (len(members) - 1):
        return None
    scale = 1 / np.sqrt(local.sum(1) + 1.0)
    values, vectors = compute.eigh(scale[:, None] * (local + np.eye(len(members))) * scale[None, :], device)
    fiedler = vectors[:, -2] * scale
    scores = np.outer(fiedler, fiedler)
    scores[local] = np.inf
    np.fill_diagonal(scores, np.inf)
    i, j = divmod(int(np.argmin(scores)), len(members))
    return float(values[-1] - values[-2]), members[i], members[j]


@timed("derive.fosr")
def fosr(adjacencies, found, device):
    best = None
    for label, adjacency in enumerate(adjacencies):
        if label not in found:
            found[label] = spectral(adjacency, np.arange(len(adjacency)), device)
        candidate = found[label]
        if candidate is not None and (best is None or candidate[0] < best[1][0] - 1e-9):
            best = label, candidate
    return best


def component_edges(present, u, v):
    started = time.perf_counter()
    local_u, local_v = np.searchsorted(present, u), np.searchsorted(present, v)
    labels = kernels.components(local_u.astype(np.int64), local_v.astype(np.int64), len(present))
    groups = blocks(labels)
    edge_labels = labels[local_u]
    order = np.argsort(edge_labels, kind="stable")
    edge_labels = edge_labels[order]
    for members in groups:
        lo = np.searchsorted(edge_labels, labels[members[0]], side="left")
        hi = np.searchsorted(edge_labels, labels[members[0]], side="right")
        rows = order[lo:hi]
        source, target = np.searchsorted(members, local_u[rows]), np.searchsorted(members, local_v[rows])
        yield members, source, target
    LOG.debug("component edges exit components=%d seconds=%.4f", len(groups), time.perf_counter() - started)


@timed("derive.component_adjacencies")
def component_adjacencies(present, u, v, budget, operation):
    edges = list(component_edges(present, u, v))
    cells = sum(len(members) ** 2 for members, _, _ in edges)
    budget.workspace(
        2 * cells + max((len(members) ** 2 * 78 for members, _, _ in edges), default=0), operation
    )
    groups, adjacencies = [], []
    for members, source, target in edges:
        adjacency = np.zeros((len(members), len(members)), bool)
        adjacency[source, target] = adjacency[target, source] = True
        groups.append(members)
        adjacencies.append(adjacency)
    return groups, adjacencies


@timed("derive.curvature_extreme")
def curvature_extreme(groups, adjacencies, found, maximum=False):
    best = None
    for label, adjacency in enumerate(adjacencies):
        if label not in found:
            edges = np.argwhere(np.triu(adjacency, 1))
            found[label] = edges, forman(adjacency, edges[:, 0], edges[:, 1])
        edges, values = found[label]
        if not len(edges):
            continue
        row = int(np.argmax(values) if maximum else np.argmin(values))
        i, j = edges[row]
        key = (-values[row] if maximum else values[row], groups[label][i], groups[label][j])
        if best is None or key < best[0]:
            best = key, label, i, j, values[row]
    return best


@timed("operators.canonical_rewire")
def canonical_rewire(spec, present, u, v, tokens, moment, seed, budget):
    canonical, low, high = ranked(present, u, v, tokens)
    size, width = len(tokens), len(canonical)

    @timed("derive.canonical_rewire.drawn")
    def drawn(iteration, left, right):
        left, right = tokens[canonical[left]], tokens[canonical[right]]
        count = len(left)
        return unit(
            seed,
            REWIRE,
            np.full(count, moment[0], np.uint64),
            np.full(count, moment[1], np.int64),
            np.full(count, iteration, np.int64),
            np.minimum(left, right),
            np.maximum(left, right),
        )

    (a, b), cut = rewire(spec, np.arange(width), low, high, width, drawn, budget)
    a, b = canonical[a], canonical[b]
    added = np.sort(np.minimum(a, b).astype(np.int64) * size + np.maximum(a, b))
    a, b = canonical[cut // width], canonical[cut % width]
    cut = np.sort(np.minimum(a, b).astype(np.int64) * size + np.maximum(a, b))
    return ((added // size).astype(present.dtype), (added % size).astype(present.dtype)), cut


@timed("operators.rewire")
def rewire(spec, present, u, v, entities, drawn, budget):
    budget.pairs(spec["derive/rewiring_budget"], "rewiring")
    groups, adjacencies = component_adjacencies(present, u, v, budget, spec["derive/rewiring"] + " rewiring")
    originals = [adjacency.copy() for adjacency in adjacencies]
    found = {}
    for iteration in range(spec["derive/rewiring_budget"]):
        if spec["derive/rewiring"] == "fosr":
            best = fosr(adjacencies, found, budget.limits.device)
            if best is None:
                break
            label, (_, i, j) = best
            adjacency = adjacencies[label]
            adjacency[i, j] = adjacency[j, i] = True
            del found[label]
            continue
        best = curvature_extreme(groups, adjacencies, found)
        if best is None:
            break
        _, label, i, j, value = best
        adjacency = adjacencies[label]
        left = np.concatenate(([i], np.flatnonzero(adjacency[i]))).astype(np.int64)
        right = np.concatenate(([j], np.flatnonzero(adjacency[j]))).astype(np.int64)
        k, partner = np.repeat(left, len(right)), np.tile(right, len(left))
        open_pair = (k != partner) & ~adjacency[k, partner]
        chosen = np.stack((k[open_pair], partner[open_pair]), 1)
        if len(chosen):
            offsets, columns = neighbours(adjacency)
            gains = (
                kernels.forman_with(adjacency, offsets, columns, int(i), int(j), chosen[:, 0], chosen[:, 1])
                - value
            )
            scaled = np.asarray(gains) * spec["derive/sdrf_temperature"]
            members = groups[label]
            draws = np.maximum(drawn(iteration, members[chosen[:, 0]], members[chosen[:, 1]]), TINY)
            k, partner = chosen[int(np.argmax(scaled - np.log(-np.log(draws))))]
            adjacency[k, partner] = adjacency[partner, k] = True
            del found[label]
        best = curvature_extreme(groups, adjacencies, found, maximum=True)
        if best is not None and best[4] > spec["derive/sdrf_threshold"]:
            _, label, p, q, _ = best
            adjacencies[label][p, q] = adjacencies[label][q, p] = False
            del found[label]
    additions, removals = [], []
    for members, adjacency, original in zip(groups, adjacencies, originals):
        for changed, result in ((adjacency & ~original, additions), (original & ~adjacency, removals)):
            rows = np.argwhere(np.triu(changed, 1))
            result.append(
                present[members[rows[:, 0]]].astype(np.int64) * entities + present[members[rows[:, 1]]]
            )
    added = np.sort(np.concatenate(additions + [np.zeros(0, np.int64)]))
    cut = np.sort(np.concatenate(removals + [np.zeros(0, np.int64)]))
    return ((added // entities).astype(present.dtype), (added % entities).astype(present.dtype)), cut


@timed("derive.derivation")
def derivation(registry, spec):
    projected = [
        edge
        for edge in registry.edges
        if edge.origin[0] == "project" and enabled(spec, "derive/" + edge.name)
    ]
    selected = np.asarray(
        [registry.kind(kind) for kind in registry.declared if enabled(spec, "derive/complete/" + kind)], np.int64
    )
    within = np.asarray(
        [registry.kind(kind) for kind in registry.declared if enabled(spec, "derive/complete_within/" + kind)],
        np.int64,
    )
    wanted = projected or len(selected) or closing(spec) or diffusing(spec) or rewiring(spec)
    return (projected, selected, within) if wanted else None


@timed("operators.derivable")
def derivable(graph, registry):
    return graph.fact_present & np.isin(graph.fact_code, grouped(registry, registry.groups))


@timed("operators.derive_steps")
def derive_steps(graph, registry, spec, seed, recency, carry, limits):
    projected, selected, within = derivation(registry, spec)
    steps, entities = graph.steps, len(graph.kinds)
    stated = derivable(graph, registry)
    fact_bounds = np.searchsorted(graph.fact_step, np.arange(steps + 1))
    observed_bounds = np.searchsorted(graph.obs_step, np.arange(steps + 1))
    budget = GraphBudget(limits, fact_bytes(graph))
    chunks, removed, memo = [], np.zeros(len(graph.fact_step), bool), {}
    closed, diffusion, rewired = closing(spec), diffusing(spec), rewiring(spec)
    tokens = carry.tokens(np.arange(entities))
    for step in range(steps):
        span = np.arange(fact_bounds[step], fact_bounds[step + 1])
        rows = span[stated[span]]
        source, target = graph.fact_source[rows], graph.fact_target[rows]
        lo, hi = observed_bounds[step], observed_bounds[step + 1]
        present = np.sort(graph.obs_entity[lo:hi][~graph.obs_anchor[lo:hi]])
        u, v = undirected(source, target, entities)
        found = []
        for edge in projected:
            a, b = project(
                graph,
                registry,
                edge,
                source,
                target,
                graph.fact_code[rows],
                recency[rows],
                spec["derive/project_cap"],
                tokens,
                budget,
                memo,
            )
            budget.retain(len(a), edge.name)
            found.append((a, b, np.full(len(a), edge.code, np.int64)))
        if len(selected):
            completed = recalled(
                memo,
                "completion",
                (present, u, v),
                budget,
                lambda: complete(graph, registry, selected, within, present, u, v, budget),
            )
            budget.retain(len(completed[0]), "completion")
            found.append(completed)
        if closed:
            found.extend(
                recalled(
                    memo,
                    "hop closure",
                    (u, v),
                    budget,
                    lambda: closure(registry, u, v, spec["derive/hops"], entities, budget),
                )
            )
        if diffusion:
            diffused = recalled(
                memo,
                "diffusion",
                (present, u, v),
                budget,
                lambda: diffuse(registry, spec, present, u, v, tokens, budget),
            )
            budget.retain(len(diffused[0]), "diffusion")
            found.append(diffused)
        if rewired:
            (a, b), cut = canonical_rewire(
                spec, present, u, v, tokens, (graph.step_token[step], graph.step_time[step]), seed, budget
            )
            budget.retain(len(a), "rewiring")
            found.append((a, b, np.full(len(a), registry.code("rewire"), np.int64)))
            keys = np.minimum(source, target).astype(np.int64) * entities + np.maximum(source, target)
            removed[rows[np.isin(keys, cut)]] = True
        chunks.extend((a, b, c, np.full(len(a), step, np.int32)) for a, b, c in found)
    source, target, code, step = (
        np.concatenate([chunk[i] for chunk in chunks] + [np.zeros(0, dtype)])
        for i, dtype in enumerate((np.int32, np.int32, np.int64, np.int32))
    )
    return (source, target, code, step), removed, budget


@timed("operators.derive_begin")
def derive_begin(graph, registry, spec, seed, limits, carry, held):
    if derivation(registry, spec) is None:
        return None
    return paths.recency(graph, carry, "derive/recency", derivable(graph, registry))


@timed("operators.derive_each")
def derive_each(graph, registry, spec, seed, limits, carry, held):
    if held is None:
        return None
    found, removed, budget = derive_steps(graph, registry, spec, seed, held, carry, limits)
    return found, removed, budget.facts


@timed("operators.derive_finish")
def derive_finish(graph, registry, spec, seed, limits, carry, held):
    if held is None:
        return graph, None
    found, removed, facts = held
    lived = len(carry.state["derive/live"]["source"])
    graph, found = paths.widened(graph, carry, "derive/live", found)
    graph = prune(graph, ~removed, carry, "derive")
    if not facts and not lived:
        carry.state["derive/live"] = paths.unlived()
        return graph, None
    return graph, (found, paths.handoff(carry, "derive/live", found, graph.steps, graph.entity_connected), facts)


@timed("operators.derive_assemble")
def derive_assemble(graph, registry, spec, seed, limits, carry, held):
    graph, rest = held
    if rest is None:
        return graph
    found, stop, facts = rest
    budget = GraphBudget(limits, fact_bytes(graph))
    budget.facts = facts
    return paths.assembled(graph, found + (np.full(len(found[0]), -1, np.int64),), stop, budget)


@timed("derive.recent")
def recent(registry, spec):
    return {"derive/recency": runs()}


@timed("derive.living")
def living(registry, spec):
    return {"prune/derive": pruned(), "derive/live": paths.unlived()}


DERIVE = OPERATIONS.register(
    "graphspec.derive",
    Operation(
        namespace="graphspec",
        name="derive",
        phases=(
            Phase("begin", derive_begin, recent, None),
            Phase("each", derive_each, None, None),
            Phase("finish", derive_finish, living, None),
            Phase("assemble", derive_assemble, None, None),
        ),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(
            Count("derive/hops", "paths", lambda registry, spec: closing(spec), "stored", "stored"),
            Count("derive/projection", "paths of two edges", projecting, "stored", "stored"),
            Count(
                "derive/diffusion", "components and paths", lambda registry, spec: diffusing(spec), "stored", "stored"
            ),
            Count(
                "derive/rewiring", "components and degree", lambda registry, spec: rewiring(spec), "stored", "stored"
            ),
        ),
        changes=("facts",),
        adds=None,
        kinds=named,
        version=11,
        graphs=1,
    ),
)
