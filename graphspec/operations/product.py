import logging
import time

import numpy as np

from graphspec.budgets import GraphBudget
from graphspec.infra.timing import timed
from graphspec.schema.added import Added
from graphspec.schema.schema import ENCODERS

from . import paths
from .operation import OPERATIONS, Count, Need, Operation, Phase
from .parameters import choice, flag, number
from .rows import fact_bytes, grouped, made_values, runs

LOG = logging.getLogger(__name__)
NUMBER = "number"
LONGEST = 3


@timed("operators.reading")
def reading():
    return tuple(name for name in ENCODERS.names() if NUMBER in ENCODERS.named(name).types)


@timed("operators.product_path")
def path(spec):
    return paths.parsed(spec["product/path"])


@timed("operators.product_length")
def length(spec):
    return spec["product/length"] if "product/length" in spec else 1


@timed("operators.product_aggregate")
def aggregate(spec):
    if paths.valued(path(spec)):
        return spec["product/aggregate"]
    return "paths" if spec["product/paths"] else None


@timed("operators.product_columns")
def columns(spec, entry):
    found = () if aggregate(spec) is None else (entry + (".paths" if aggregate(spec) == "paths" else ".value"),)
    return found + ((entry + ".length",) if length(spec) > 1 else ())


def stated(registry):
    started = time.perf_counter()
    options = paths.offered(registry)
    result = ()
    if options:
        result = (
            choice("product/path", options, options[0]),
            number("product/length", 1, LONGEST, 1, lambda spec: paths.chains(registry, path(spec))),
            flag("product/paths", False, lambda spec: not paths.valued(path(spec))),
            choice("product/aggregate", paths.AGGREGATES, "product", lambda spec: paths.valued(path(spec))),
            number("product/cap", 0, 64, 0, lambda spec: len(path(spec)) > 1),
            choice("product/encoder", reading(), None, lambda spec: bool(columns(spec, ""))),
        )
    LOG.debug("product stated exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("product.lacking")
def lacking(registry, value):
    if registry.groups:
        return None
    return "needs a relation, and the bank %s states none" % registry.name


@timed("operators.product_adds")
def adds(registry, settings, entry):
    steps, held = path(settings), columns(settings, entry)
    return Added(
        {name: dict(type=NUMBER, encoder=settings["product/encoder"]) for name in held},
        {},
        {},
        {
            entry: dict(
                pairs=paths.joined(registry, steps, length(settings)),
                directed=not paths.symmetric(steps),
                columns=held,
            )
        },
        {},
        {},
    )


@timed("operators.product_codes")
def codes(registry, table):
    return grouped(registry, registry.groups) if table == paths.UNION else registry.coded[table]


@timed("operators.product_edges")
def edges(graph, registry, rows, recency, step):
    table, direction, column = step
    rows = rows[np.isin(graph.fact_code[rows], codes(registry, table))]
    values = None
    if column is not None:
        reference = np.asarray(graph.fact_ref, np.int64)[rows]
        values = np.full(len(rows), np.nan)
        values[reference >= 0] = np.asarray(graph.value_table, np.float64)[
            reference[reference >= 0], registry.attributes(table).index(column)
        ]
        rows, values = rows[~np.isnan(values)], values[~np.isnan(values)]
    source, target, picked = paths.oriented(graph.fact_source[rows], graph.fact_target[rows], direction)
    return source, target, None if values is None else values[picked], recency[rows][picked]


@timed("operators.product_step")
def product_step(graph, registry, spec, rows, recency, tokens, budget):
    steps, entities, chosen = path(spec), len(graph.kinds), aggregate(spec)
    found = [edges(graph, registry, rows, recency, step) for step in steps]
    if len(found) == 2 and spec["product/cap"]:
        first = paths.latest(found[0][1], found[0][0], found[0][3], tokens, spec["product/cap"])
        second = paths.latest(found[1][0], found[1][1], found[1][3], tokens, spec["product/cap"])
        found = [tuple(None if part is None else part[kept] for part in held) for held, kept in zip(found, (first, second))]
    source, target, values = paths.product([held[:3] for held in found], entities, chosen, budget, "product")
    both, limit = paths.symmetric(steps), length(spec)
    lengths = np.ones(len(source), np.int64)
    if limit > 1 and chosen is None:
        if both:
            source, target = source[source < target], target[source < target]
        reached = [(source, target)] + paths.closure(source, target, limit, entities, budget, both, "product closure")
        lengths = np.concatenate([np.full(len(pair[0]), at, np.int64) for at, pair in enumerate(reached, 1)])
        source = np.concatenate([pair[0] for pair in reached])
        target = np.concatenate([pair[1] for pair in reached])
    elif limit > 1:
        source, target, values, lengths = paths.powers(source, target, values, limit, entities, chosen, budget, "product closure")
    if both:
        kept = source < target
        source, target, lengths = source[kept], target[kept], lengths[kept]
        values = None if values is None else values[kept]
    budget.retain(len(source), "product")
    held = [] if values is None else [values]
    held += [lengths] if limit > 1 else []
    return source.astype(np.int32), target.astype(np.int32), np.stack(held, 1) if held else None


@timed("operators.product_begin")
def product_begin(graph, registry, spec, seed, limits, carry, held):
    if "product/cap" not in spec or not spec["product/cap"]:
        return None
    stated = np.asarray(graph.fact_present, bool) & np.isin(
        graph.fact_code, np.concatenate([codes(registry, table) for table, _, _ in path(spec)])
    )
    return paths.recency(graph, carry, "product/recency", stated)


@timed("operators.product_each")
def product_each(graph, registry, spec, seed, limits, carry, held):
    entities, code = len(graph.kinds), registry.code(carry.entry)
    recency = np.zeros(len(graph.fact_step), np.int64) if held is None else held
    budget, tokens = GraphBudget(limits, fact_bytes(graph)), carry.tokens(np.arange(entities))
    bounds = np.searchsorted(graph.fact_step, np.arange(graph.steps + 1))
    present, chunks = np.asarray(graph.fact_present, bool), []
    for step in range(graph.steps):
        rows = np.arange(bounds[step], bounds[step + 1])
        source, target, values = product_step(graph, registry, spec, rows[present[rows]], recency, tokens, budget)
        chunks.append((source, target, np.full(len(source), code, np.int64), np.full(len(source), step, np.int32), values))
    found = tuple(np.concatenate([chunk[place] for chunk in chunks]) for place in range(4))
    rows = None if chunks[0][4] is None else np.concatenate([chunk[4] for chunk in chunks])
    return found, rows, budget.facts


@timed("operators.product_finish")
def product_finish(graph, registry, spec, seed, limits, carry, held):
    found, rows, facts = held
    lived = len(carry.state["product/live"]["source"])
    graph, found = paths.widened(graph, carry, "product/live", found)
    if not facts and not lived:
        carry.state["product/live"] = paths.unlived()
        return graph, None
    return graph, (found, rows, paths.handoff(carry, "product/live", found, graph.steps, graph.entity_connected), facts)


@timed("operators.product_assemble")
def product_assemble(graph, registry, spec, seed, limits, carry, held):
    graph, rest = held
    if rest is None:
        return graph
    found, rows, stop, facts = rest
    references = np.full(len(found[0]), -1, np.int64)
    if rows is not None:
        table = np.full((len(rows), graph.value_table.shape[1]), np.nan)
        table[:, : rows.shape[1]] = rows
        graph, references = made_values(graph, carry, table)
    budget = GraphBudget(limits, fact_bytes(graph))
    budget.facts = facts
    return paths.assembled(graph, found + (np.asarray(references, np.int64),), stop, budget)


@timed("product.recent")
def recent(registry, spec):
    return {"product/recency": runs()}


@timed("product.living")
def living(registry, spec):
    return {"product/live": paths.unlived()}


PRODUCT = OPERATIONS.register(
    "graphspec.product",
    Operation(
        namespace="graphspec",
        name="product",
        phases=(
            Phase("begin", product_begin, recent, None),
            Phase("each", product_each, None, None),
            Phase("finish", product_finish, living, None),
            Phase("assemble", product_assemble, None, None),
        ),
        parameters=stated,
        reads=(),
        splits=(),
        requires=(
            Need(None, lacking),
            Count("product/path", "paths", lambda registry, spec: bool(registry.groups), "stored", "stored"),
        ),
        changes=("facts",),
        adds=adds,
        kinds=None,
        version=4,
        graphs=1,
    ),
)
