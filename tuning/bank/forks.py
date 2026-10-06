import logging
import time
from dataclasses import dataclass

import numpy as np

from graphspec.engine.edits import Edit
from graphspec.engine.segments import Derivation, padded
from graphspec.facts.stored import encoded, unheld
from graphspec.infra.arrays import Resident
from graphspec.infra.hashing import hashed
from graphspec.infra.timing import timed
from graphspec.infra.workspace import MEMORY_BYTES
from graphspec.schema.decisions import ROLES
from graphspec.views.graph import Graph
from tuning.families.heads import coded, realisation

from .options import TOKEN, offered

LOG = logging.getLogger(__name__)
UNMOVED = ()
SHOWN = ("entity", "spatial", "spatial_step", "step")


@dataclass(frozen=True)
class Part:
    graph: int
    place: int
    step: int
    option: int
    move: tuple


class Held(dict):
    @timed("forks.Held.read")
    def read(self, key):
        return self[key]


class Forks:
    @timed("forks.Forks.__init__")
    def __init__(self, bank):
        self.bank, self.chains, self.held, self.outer = bank, {}, {}, None

    @timed("forks.Forks.__enter__")
    def __enter__(self):
        self.outer = self.bank.forking
        if self.outer is None:
            self.bank.forking = self
        return self.bank.forking

    def __exit__(self, *exception):
        started = time.perf_counter()
        if self.outer is None:
            self.bank.forking = None
        LOG.debug(
            "forks exit released=%s chains=%d forks=%d seconds=%.4f",
            self.outer is None,
            len(self.chains),
            len(self.held),
            time.perf_counter() - started,
        )


@timed("forks.opened")
def opened(bank):
    if bank.forking is None:
        raise ValueError(
            "the forks of a derivation are held by the work that reads them, and no work holds them here; a fork "
            "is derived within tuning.bank.forks.Forks(bank), which releases its forks and their histories at its end"
        )
    return bank.forking


@timed("forks.widened")
def widened(graph, registry):
    return graph.replace(value_table=padded(graph.value_table, registry.attributed))


def editing(bank, cfg):
    started = time.perf_counter()
    name = bank.stated(cfg)[0][3]
    if name is None:
        raise ValueError(
            "an option is derived as an edit of the graph only under a realisation of options that states that "
            "messages pass back; the model states %s" % cfg["model"].get("options/realisation")
        )
    LOG.debug("forks editing exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return name


@timed("forks.room")
def room(bank):
    found = offered(bank)
    created = found.sorts[found.role["code"]] == ROLES.index("new")
    return int(np.bincount(found.action["option"][found.role["action"][created]], minlength=1).max())


def members(bank, cfg, graph):
    started = time.perf_counter()
    indices = bank.groups(cfg)[graph] if bank.joining(cfg) else [graph]
    registry = bank.fitted(cfg)
    result = indices, [widened(Graph.load(bank.view(member, cfg)).dense(), registry) for member in indices]
    LOG.debug("forks members exit graph=%d members=%d seconds=%.4f", graph, len(indices), time.perf_counter() - started)
    return result


def begun(bank, cfg, graph):
    started = time.perf_counter()
    _, steps, spec, chosen = bank.variant(cfg)
    indices, held = members(bank, cfg, graph)
    beside = bank.seconds(steps, indices, cfg, False)
    result = Derivation.begun(
        held, beside, bank.fitted(cfg), steps, spec, chosen, bank.seed, bank.limits(), bank.partitions(cfg), room(bank)
    )
    LOG.debug("forks begun exit graph=%d members=%d seconds=%.4f", graph, len(held), time.perf_counter() - started)
    return result


def trunk(bank, cfg, graph, step):
    started = time.perf_counter()
    chains = opened(bank).chains
    key = (bank.variant(cfg)[0], graph)
    if key not in chains:
        chains[key] = {0: begun(bank, cfg, graph)}
    held = chains[key]
    if step not in held:
        low = max(moment for moment in held if moment < step)
        held[step] = held[low].continued(step, None)
    LOG.debug("forks trunk exit graph=%d step=%d seconds=%.4f", graph, step, time.perf_counter() - started)
    return held[step]


def entities(graph, tokens):
    started = time.perf_counter()
    held = np.asarray(graph.entity_token).astype(np.uint64) & np.uint64(TOKEN)
    order = np.argsort(held, kind="stable")
    wanted = np.asarray(tokens, np.int64).astype(np.uint64)
    place = np.minimum(np.searchsorted(held[order], wanted), len(held) - 1)
    lost = np.flatnonzero(held[order][place] != wanted)
    if len(lost):
        raise ValueError(
            "an option of graph %s names the entity of the token %d, which the graph does not hold"
            % (graph.name, int(wanted[lost[0]]))
        )
    result = order[place].astype(np.int64)
    LOG.debug("forks entities exit graph=%s tokens=%d seconds=%.4f", graph.name, len(result), time.perf_counter() - started)
    return result


def valued(registry, relation, action, values, own, move):
    started = time.perf_counter()
    row = np.full(registry.attributed, np.nan)
    place = registry.value_group[relation]
    held = registry.attributes(registry.valued[place]) if place >= 0 else ()
    columns = registry.schema.parameters()
    for at, column in enumerate(held):
        if column in own and move != ("parameter", action, column):
            row[at] = values[columns.index(column)]
    LOG.debug("forks valued exit columns=%d seconds=%.6f", len(held), time.perf_counter() - started)
    return row


def stated(bank, cfg, option, order):
    started = time.perf_counter()
    found = offered(bank)
    first, count = int(found.actions[0][option]), int(found.actions[1][option])
    if sorted(order) != list(range(first, first + count)):
        raise ValueError("the actions of the option %d are %d from %d, found %r" % (option, count, first, order))
    result = []
    for action in order:
        low, many = int(found.roles[0][action]), int(found.roles[1][action])
        roles = [
            tuple(int(found.role[name][role]) for name in ("code", "source", "target", "element"))
            for role in range(low, low + many)
        ]
        result.append(
            (int(found.action["kind"][action]), int(found.action["actor"][action]), found.action["values"][action], roles)
        )
    LOG.debug("forks stated exit option=%d actions=%d seconds=%.4f", option, len(result), time.perf_counter() - started)
    return result


def other(registry, entry, relation, kinds):
    started = time.perf_counter()
    result = relation
    for name in [name for name in list(registry.extras["base"]) + list(entry.relations) if name != relation.name]:
        edge = registry.by_name[name]
        pairs = {(registry.kind(first), registry.kind(second)) for first, second in edge.pairs}
        pairs |= {(second, first) for first, second in pairs} if edge.symmetric else set()
        if (int(kinds[0]), int(kinds[1])) in pairs:
            result = edge
            break
    LOG.debug("forks other exit relation=%s other=%s seconds=%.6f", relation.name, result.name, time.perf_counter() - started)
    return result


def made(bank, cfg, history, part, order):
    started = time.perf_counter()
    registry, view = bank.fitted(cfg), bank.viewed(cfg).registry
    entry, (acts, roles) = realisation(editing(bank, cfg)), coded(registry.schema)
    plan = history.source.plan
    graph = history.source.members[int(np.searchsorted(plan.steps, part.step, "right")) - 1]
    actions, created = stated(bank, cfg, part.option, order), {}
    for _, _, _, held in actions:
        for code, source, _, element in held:
            if roles[code][3] == "new" and part.move != ("target", code):
                created[source] = (min(created[source][0], element) if source in created else element, registry.kind(roles[code][4][0]))
    made = sorted(set(created.values()))
    numbered = {source: len(graph.kinds) + made.index(found) for source, found in created.items()}
    every = np.r_[np.asarray(graph.kinds, np.int64), np.asarray([kind for _, kind in made], np.int64)]
    facts = []
    for kind, actor, values, held in actions:
        task, action = acts[kind]
        own = registry.schema.tasks["kinds"][task]["target"]["actions"][action]["parameters"]
        tokens = [actor] + [token for code, source, target, _ in held if roles[code][3] != "new" for token in (source, target) if token >= 0]
        known = dict(zip(tokens, entities(graph, tokens).tolist()))
        for code, source, target, _ in held:
            if part.move == ("target", code):
                continue
            _, _, role, sort, _ = roles[code]
            mapped = entry.mapping[task][action][role]
            first = numbered[source] if sort == "new" else known[source]
            for end in [first] + ([known[target]] if sort == "edge" else []):
                pair = (known[actor], end) if mapped["actor"] == "source" else (end, known[actor])
                kinds = every[list(pair)]
                relation = registry.by_name[mapped["relation"]]
                if part.move == ("kind",):
                    relation = other(registry, entry, relation, kinds)
                if relation.symmetric and kinds[0] == kinds[1] and pair[0] > pair[1]:
                    pair = pair[::-1]
                facts.append((relation.code, pair[0], pair[1], valued(registry, relation.code, kind, values, own, part.move)))
    facts.sort(key=lambda fact: (fact[0], fact[1], fact[2], tuple(np.nan_to_num(fact[3], nan=np.inf).tolist())))
    observed = len(graph.kinds) + np.arange(len(made), dtype=np.int64)
    missing = np.full((len(made), len(view.encoding.layout)), np.nan, np.float32)
    x = encoded(view, missing, unheld(view, len(made)), None, lambda row: "the element an option creates")
    result = Edit(
        dict(kind=every[observed], token=np.asarray([found for found, _ in made], np.uint64)),
        dict(observations=dict(entity=[], step=[]), facts=dict(code=[], source=[], target=[], step=[])),
        dict(
            observations=dict(entity=observed, step=np.full(len(made), part.step, np.int64), x=x),
            facts=dict(
                source=np.asarray([fact[1] for fact in facts], np.int64),
                target=np.asarray([fact[2] for fact in facts], np.int64),
                code=np.asarray([fact[0] for fact in facts], np.int64),
                step=np.full(len(facts), part.step, np.int64),
                present=np.ones(len(facts), bool),
                values=np.asarray([fact[3] for fact in facts], np.float64).reshape(len(facts), registry.attributed),
            ),
        ),
    )
    LOG.debug(
        "forks made exit option=%d facts=%d created=%d seconds=%.4f",
        part.option,
        len(facts),
        len(made),
        time.perf_counter() - started,
    )
    return result


@timed("forks.ordered")
def ordered(bank, option):
    found = offered(bank)
    first = int(found.actions[0][option])
    return list(range(first, first + int(found.actions[1][option])))


def derived(bank, cfg, part):
    started = time.perf_counter()
    history = trunk(bank, cfg, part.graph, part.step)
    result = Held(history.continued(part.step + 1, made(bank, cfg, history, part, ordered(bank, part.option))).arrays())
    LOG.info(
        "forks derived exit graph=%d step=%d option=%d move=%s seconds=%.4f",
        part.graph,
        part.step,
        part.option,
        part.move,
        time.perf_counter() - started,
    )
    return result


def forked(bank, part, cfg, device):
    started = time.perf_counter()
    if part.option < 0:
        result = bank.resident(part.graph, cfg, device)
    else:
        held, key = opened(bank).held, (bank.variant(cfg)[0], editing(bank, cfg), part, str(device))
        if key not in held:
            held[key] = Resident(derived(bank, cfg, part), device, bank.residents.dtypes, MEMORY_BYTES)
        result = held[key]
    LOG.debug("forks forked exit option=%d seconds=%.4f", part.option, time.perf_counter() - started)
    return result


def listed(arrays):
    started = time.perf_counter()
    entity, spatial = np.asarray(arrays["entity"], np.int64), np.asarray(arrays["spatial"], np.int64)
    facts = hashed(np.asarray(arrays["spatial_step"], np.int64), entity[spatial[:, 0]], entity[spatial[:, 1]], spatial[:, 2])
    result = np.sort(facts), np.sort(hashed(np.asarray(arrays["step"], np.int64), entity))
    LOG.debug("forks listed exit facts=%d seconds=%.4f", len(facts), time.perf_counter() - started)
    return result


def shown(bank, cfg, part):
    started = time.perf_counter()
    held, key = opened(bank).held, (bank.variant(cfg)[0], "unedited", part.graph, part.step)
    if key not in held:
        held[key] = listed(trunk(bank, cfg, part.graph, part.step).continued(part.step + 1, None).arrays())
    fork = bank.resident(part, cfg, "cpu")
    found = listed({name: fork.numpy(name) for name in SHOWN})
    result = any(not np.array_equal(mine, plain) for mine, plain in zip(found, held[key]))
    LOG.debug("forks shown exit option=%d shown=%s seconds=%.4f", part.option, result, time.perf_counter() - started)
    return result


def carried(bank, part, rows):
    started = time.perf_counter()
    found = offered(bank)
    result = found.row_option[rows] == part.option if part.option >= 0 else found.decision[rows] < 0
    LOG.debug("forks carried exit rows=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return result
