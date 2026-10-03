import logging
import time
from weakref import WeakKeyDictionary

import numpy as np

from graphspec.facts.stored import HALF
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.infra.workspace import allocate, scatter, take

from . import kernels
from .state import EMPTY, Table

LOG = logging.getLogger(__name__)
AGGREGATES = ("sum", "mean", "min", "max")
PARALLEL = ("keep", "first", "last") + AGGREGATES
FACTS = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
OBSERVATIONS = ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
SAMPLE = "graphspec.nodes/sample"
DROP = "graphspec.edges/drop"
RANDOM = "graphspec.edges/cap_order"
REWIRE = "graphspec.derive/rewiring"
KEYED = WeakKeyDictionary()


def keyed_relations(registry):
    if registry not in KEYED:
        started = time.perf_counter()
        tokens, symmetric = np.zeros(registry.relations, np.uint64), np.zeros(registry.relations, bool)
        for edge in registry.edges:
            tokens[edge.code], symmetric[edge.code] = token(edge.name), edge.symmetric
        KEYED[registry] = tokens, symmetric
        LOG.debug(
            "keyed relations exit relations=%d seconds=%.4f", len(registry.edges), time.perf_counter() - started
        )
    return KEYED[registry]


@timed("rows.counted")
def counted(counts):
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


@timed("operators.second_of")
def second_of(carry, operation):
    if carry.second is None:
        raise ValueError("the operation %s takes a second graph and was given none" % operation)
    return carry.second


@timed("operators.fact_keys")
def fact_keys(registry, *graphs):
    both = keyed_relations(registry)[1]
    stacked = []
    for columns in graphs:
        step, code, source, target = (np.asarray(values, np.int64) for values in columns[:4])
        swap = both[code] & (source > target)
        stacked.append(
            np.stack(
                (step, code, np.where(swap, target, source), np.where(swap, source, target))
                + tuple(np.asarray(extra, np.int64) for extra in columns[4:]),
                1,
            )
        )
    _, inverse = np.unique(np.concatenate(stacked), axis=0, return_inverse=True)
    inverse, bounds = inverse.reshape(-1), np.cumsum([0] + [len(held) for held in stacked])
    return tuple(inverse[low:high] for low, high in zip(bounds[:-1], bounds[1:]))


@timed("operators.shown")
def shown(graph, step, source, target):
    count = len(graph.kinds)
    structural = ~np.asarray(graph.obs_anchor, bool)
    lit = np.unique(
        np.asarray(graph.obs_step, np.int64)[structural] * count + np.asarray(graph.obs_entity, np.int64)[structural]
    )
    moment = np.asarray(step, np.int64) * count
    return np.isin(moment + np.asarray(source, np.int64), lit) & np.isin(moment + np.asarray(target, np.int64), lit)


@timed("rows.fact_bytes")
def fact_bytes(graph):
    return sum(graph.arrays[name].dtype.itemsize for name in FACTS)


@timed("operators.made_rows")
def made_rows(graph, carry, rows):
    start = len(graph.x_table)
    carry.x_global = np.concatenate((carry.x_global, np.full(len(rows), -1, np.int64)))
    grown = graph.replace(x_table=np.concatenate((graph.x_table, np.asarray(rows, graph.x_table.dtype))))
    return grown, start + np.arange(len(rows))


@timed("operators.made_values")
def made_values(graph, carry, values):
    start = len(graph.value_table)
    carry.value_global = np.concatenate((carry.value_global, np.full(len(values), -1, np.int64)))
    grown = graph.replace(value_table=np.concatenate((graph.value_table, np.asarray(values, graph.value_table.dtype))))
    return grown, start + np.arange(len(values))


@timed("operators.written")
def written(registry, rows, columns, codes, dtype, named):
    encoding = registry.encoding
    rows = np.array(rows, np.float32)
    codes = np.asarray(codes, np.float32).reshape(len(rows), len(columns))
    slots = {}
    for place, name in enumerate(columns):
        values, flags = encoding.slots(name)
        found = encoding.vectors((name,), codes[:, place])
        rows[:, values], rows[:, flags] = found[:, : len(values)], found[:, len(values) :]
        slots.update({int(slot): name for slot in np.r_[values, flags]})
    with np.errstate(over="ignore"):
        result = rows.astype(dtype)
    lost = ~np.isfinite(result)
    if lost.any():
        row, slot = (int(found[0]) for found in np.nonzero(lost))
        raise ValueError(
            "%s holds in the column %s the value %r, which gives %r; model inputs are stored at 16 bits, which hold "
            "magnitudes up to %d"
            % (named(row), slots[slot], float(codes[row, columns.index(slots[slot])]), float(rows[row, slot]), HALF)
        )
    return result


@timed("rows.select")
def select(graph, names, mask):
    if mask.all():
        return graph
    return graph.replace(**{name: take(graph.arrays[name], mask) for name in names})


@timed("rows.extend")
def extend(graph, source, target, code, step, present, references):
    if not len(source):
        return graph
    added = dict(
        fact_source=source,
        fact_target=target,
        fact_code=code,
        fact_step=step,
        fact_present=present,
        fact_ref=references,
    )
    added = {name: np.asarray(value, graph.arrays[name].dtype) for name, value in added.items()}
    order = np.argsort(added["fact_step"].astype(np.int64) * 2 + ~added["fact_present"], kind="stable")
    key = graph.fact_step.astype(np.int64) * 2 + ~graph.fact_present
    position = np.searchsorted(
        key, added["fact_step"][order].astype(np.int64) * 2 + ~added["fact_present"][order], "right"
    ) + np.arange(len(order))
    total = len(key) + len(order)
    existing = np.ones(total, bool)
    existing[position] = False
    merged = {}
    for name in FACTS:
        column = allocate((total,) + graph.arrays[name].shape[1:], graph.arrays[name].dtype)
        scatter(column, existing, graph.arrays[name])
        scatter(column, position, added[name], order)
        merged[name] = column
    return graph.replace(**merged)


@timed("operators.observed")
def observed(graph, entity, step, reference, anchor):
    if not len(entity):
        return graph
    added = dict(obs_entity=entity, obs_step=step, obs_ref=reference, obs_anchor=anchor)
    added = {name: np.asarray(value, graph.arrays[name].dtype) for name, value in added.items()}
    order = np.argsort(added["obs_step"], kind="stable")
    position = np.searchsorted(graph.obs_step, added["obs_step"][order], "right") + np.arange(len(order))
    total = len(graph.obs_step) + len(order)
    existing = np.ones(total, bool)
    existing[position] = False
    merged = {}
    for name in OBSERVATIONS:
        column = allocate((total,), graph.arrays[name].dtype)
        scatter(column, existing, graph.arrays[name])
        scatter(column, position, added[name], order)
        merged[name] = column
    return graph.replace(**merged)


@timed("operators.prune")
def prune(graph, kept, carry, name):
    return select(
        graph,
        FACTS,
        prune_mask(
            graph.fact_source,
            graph.fact_target,
            graph.fact_code,
            graph.fact_present,
            kept,
            graph.entity_connected,
            carry,
            name,
        ),
    )


@timed("rows.pruned")
def pruned():
    return Table(EMPTY, state=np.zeros(0, np.int8))


@timed("rows.kept")
def kept():
    return Table(EMPTY, value=EMPTY)


@timed("rows.lived")
def lived():
    return Table(EMPTY, pair=EMPTY, source=EMPTY, target=EMPTY, code=EMPTY)


@timed("operators.collapsed")
def collapsed(graph, carry, name, base, pair):
    live = carry.state[name]
    chosen = np.flatnonzero(np.isin(live.columns["pair"], pair))
    added = len(chosen)
    source = np.concatenate((carry.local(live.columns["source"][chosen]), graph.fact_source)).astype(np.int64)
    target = np.concatenate((carry.local(live.columns["target"][chosen]), graph.fact_target)).astype(np.int64)
    code = np.concatenate((live.columns["code"][chosen], graph.fact_code)).astype(np.int64)
    pair = np.concatenate((live.columns["pair"][chosen], pair))
    present = np.concatenate((np.ones(added, bool), graph.fact_present))
    base = np.concatenate((np.ones(added, bool), base))
    typed, typed_count = kernels.keyed(source, target, code)
    pairs, pair_count = kernels.keyed(pair, np.zeros(len(pair), np.int64), np.zeros(len(pair), np.int64))
    result = kernels.collapsing(typed, pairs, present, base, typed_count, pair_count)[added:]
    rows = np.flatnonzero(base & graph.entity_connected[source] & graph.entity_connected[target])
    last = np.full(typed_count, -1, np.int64)
    np.maximum.at(last, typed[rows], rows)
    final = last[last >= 0]
    keys = carry.keys(code[final], source[final], target[final])
    held = present[final]
    carry.state[name] = live.replaced(
        keys,
        keys[held],
        pair=pair[final][held],
        source=carry.glob[source[final][held]],
        target=carry.glob[target[final][held]],
        code=code[final][held],
    )
    return result


@timed("operators.reduced")
def reduced(graph, carry, aggregate, chosen):
    ids, count = kernels.keyed(
        graph.fact_step.astype(np.int64) * 2 + graph.fact_present,
        graph.fact_code * len(graph.kinds) + graph.fact_source,
        graph.fact_target,
    )
    ids = np.where(chosen, ids, count + np.arange(len(ids)))
    count += len(ids)
    if aggregate == "first":
        return select(graph, FACTS, kernels.first_rows(ids, count))
    remains = kernels.last_rows(ids, count)
    if aggregate == "last":
        return select(graph, FACTS, remains)
    reference = np.asarray(graph.fact_ref, np.int64)
    valued = np.flatnonzero(np.asarray(graph.fact_present, bool) & (reference >= 0))
    sizes = np.bincount(ids[valued], minlength=count)
    valued = valued[sizes[ids[valued]] > 1]
    if not len(valued):
        return select(graph, FACTS, remains)
    owners = np.unique(ids[valued])
    place = np.searchsorted(owners, ids[valued])
    values = np.asarray(graph.value_table, np.float64)[reference[valued]]
    known = ~np.isnan(values)
    if aggregate in ("sum", "mean"):
        found = np.zeros((len(owners), values.shape[1]))
        np.add.at(found, place, np.where(known, values, 0.0))
        seen = np.zeros(found.shape)
        np.add.at(seen, place, known)
        divisor = seen if aggregate == "mean" else np.ones(found.shape)
        found = np.divide(found, divisor, out=np.full(found.shape, np.nan), where=seen > 0)
    else:
        found = np.full((len(owners), values.shape[1]), np.nan)
        (np.fmin if aggregate == "min" else np.fmax).at(found, place, values)
    graph, rows = made_values(graph, carry, found)
    last = np.flatnonzero(remains)
    picked = last[np.isin(ids[last], owners)]
    reference = reference.copy()
    reference[picked] = rows[np.searchsorted(owners, ids[picked])]
    graph = graph.replace(fact_ref=reference.astype(graph.fact_ref.dtype))
    return select(graph, FACTS, remains)


@timed("operators.first_kept")
def first_kept(carry, name, graph, entity, step, alive):
    first, _ = kernels.first_observed(entity, step, alive, len(graph.kinds), graph.steps)
    table = carry.state[name]
    first[table.lookup("value", carry.glob, 0) > 0] = -1
    lasting = np.flatnonzero((first < graph.steps) & np.asarray(graph.entity_connected, bool))
    carry.state[name] = table.replaced(
        carry.glob[lasting], carry.glob[lasting], value=np.ones(len(lasting), np.int64)
    )
    return first


@timed("rows.runs")
def runs():
    return Table(EMPTY, last=EMPTY, current=EMPTY, begun=EMPTY)


@timed("rows.prune_mask")
def prune_mask(fact_source, fact_target, fact_code, fact_present, kept, connected, carry, name):
    table = carry.state["prune/" + name]
    keys = carry.keys(fact_code, fact_source, fact_target)
    state = table.lookup("state", keys, 0)
    carried = np.flatnonzero(state > 0)
    first = carried[np.unique(keys[carried], return_index=True)[1]]
    source, target, code = (
        np.concatenate((column[first], column)) for column in (fact_source, fact_target, fact_code)
    )
    present = np.concatenate((np.ones(len(first), bool), fact_present))
    everything = np.concatenate((state[first] == 2, kept))
    ids, count = kernels.keyed(source, target, code)
    result = kernels.orphans(ids, present, everything, count)[len(first) :]
    stated = np.bincount(ids[present], minlength=count) > 0
    surviving = np.bincount(ids[present & everything], minlength=count) > 0
    group_key = np.zeros(count, np.int64)
    group_key[ids] = np.concatenate((keys[first], keys))
    lasting = np.zeros(count, bool)
    lasting[ids[connected[source] & connected[target]]] = True
    chosen = lasting & stated
    carry.state["prune/" + name] = table.replaced(
        group_key[chosen], group_key[chosen], state=np.where(surviving[chosen], 2, 1).astype(np.int8)
    )
    return result


@timed("rows.run_state")
def run_state(carry, name, graph, ids, count, step, keys):
    last, current, begun = np.full(count, -2, np.int64), np.zeros(count, np.int64), np.zeros(count, np.int64)
    group_key = np.zeros(count, np.int64)
    group_key[ids] = keys
    table = carry.state[name]
    found = table.find(group_key)
    use = found >= 0
    if use.any():
        last[use] = table.columns["last"][found[use]] - carry.step0
        current[use] = table.columns["current"][found[use]]
        begun[use] = table.columns["begun"][found[use]] - carry.step0
    result = kernels.runs(ids, step, last, current, begun)
    active = last == graph.steps - 1
    return result, (
        group_key[active],
        last[active] + carry.step0,
        current[active],
        begun[active] + carry.step0,
    )


@timed("rows.open_runs")
def open_runs(carry, name, pieces):
    carry.state[name] = Table(
        np.concatenate([piece[0] for piece in pieces] + [np.zeros(0, np.int64)]),
        **{
            column: np.concatenate([piece[index] for piece in pieces] + [np.zeros(0, np.int64)])
            for index, column in enumerate(("last", "current", "begun"), 1)
        },
    )


@timed("rows.member_end")
def member_end(graph):
    member = graph.step_graph
    first = np.r_[0, np.flatnonzero(np.diff(member)) + 1]
    return np.r_[first[1:], graph.steps][np.cumsum(np.r_[0, np.diff(member) != 0])]


@timed("operators.grouped")
def grouped(registry, groups):
    return np.concatenate([registry.coded[group] for group in groups] + [np.zeros(0, np.int64)])
