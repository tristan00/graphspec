import argparse
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.infra.files import write_json
from graphspec.infra.log import configure
from graphspec.infra.runtime import exclusive, output_path
from graphspec.infra.staging import Stage

from .schema import DECLARED, EDGE, LEVELS, OBSERVED, QUERY, VERSION, Schema

LOG = logging.getLogger(__name__)
CATEGORIES = ("buy", "sell", "move")
TARGET = "outcome"
SERIES = ("level", "trend")
SETTINGS = ("entities", "steps", "degree", "max_degree", "alive", "lifetime", "period")
SHARES = dict(person=0.5, place=0.2, item=0.2, team=0.1)
REFERENCES = dict(person=["person"], place=["place"], team=["team"])


def edge(source, target, columns, directed=True, relation=DECLARED):
    return dict(endpoints=[[source, target]], directed=directed, columns=list(columns), relation=relation)


def uniform(columns, signals):
    started = time.perf_counter()
    names = ["v%d" % index for index in range(columns)]
    stated = {name: dict(type="number", encoder="value") for name in names}
    stated.update(
        weight=dict(type="number", encoder="logarithm"), distance=dict(type="number", encoder="logarithm")
    )
    result = dict(
        schema=dict(
            version=VERSION,
            time=dict(unit="step"),
            columns=stated,
            graph=dict(columns=[]),
            nodes={kind: dict(columns=names) for kind in SHARES},
            edges=dict(
                visits=edge("person", "place", ["weight"]),
                owns=edge("person", "item", []),
                member=edge("person", "team", []),
                near=edge("place", "place", []),
                located=edge("team", "place", ["distance"]),
            ),
            queries=dict(key="key", grouping="period", columns=names, references=REFERENCES, signals=signals),
        ),
        shares=SHARES,
        symbols={},
    )
    LOG.debug("uniform shape exit columns=%d seconds=%.4f", columns, time.perf_counter() - started)
    return result


def varied():
    started = time.perf_counter()
    result = dict(
        schema=dict(
            version=VERSION,
            time=dict(unit="step"),
            columns=dict(
                size=dict(type="number", encoder="standard"),
                speed=dict(type="number", encoder="standard"),
                colour=dict(type="category", encoder="onehot"),
                open=dict(type="flag", encoder="value"),
                weight=dict(type="number", encoder="logarithm"),
                mode=dict(type="category", encoder="onehot"),
                distance=dict(type="number", encoder="standard"),
                season=dict(type="category", encoder="onehot"),
            ),
            graph=dict(columns=["season"]),
            nodes=dict(
                person=dict(columns=["size", "speed", "colour", "open"]),
                place=dict(columns=["size"]),
                item=dict(columns=["speed", "colour"]),
                team=dict(columns=["open"]),
            ),
            edges=dict(
                visits=edge("person", "place", ["weight", "mode"]),
                owns=edge("person", "item", []),
                member=edge("person", "team", []),
                near=edge("place", "place", [], directed=False),
                located=edge("team", "place", ["distance"]),
                links=edge("person", "item", ["mode"], relation=dict(column="predicate")),
            ),
            queries=dict(
                key="ticket", grouping="week", columns=["size", "speed"], references=REFERENCES, signals=2
            ),
        ),
        shares=SHARES,
        symbols=dict(
            colour=["red", "green", "blue"],
            mode=["foot", "road", "rail", "air"],
            season=["dry", "wet"],
            predicate=["likes", "lends", "sells", "seeks", "sends", "shows"],
        ),
    )
    LOG.debug("varied shape exit seconds=%.4f", time.perf_counter() - started)
    return result


SHAPES = dict(uniform=uniform, varied=varied)


def vectors(rng, rows, columns):
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    present = (rng.random((rows, columns)) < 0.9).astype(np.float32)
    return np.concatenate((values * present, present), 1)


def fixed(values):
    values = np.ascontiguousarray(values, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def steps_of(draws):
    return np.floor(np.abs(np.asarray(draws, np.float64)) * 1000).astype(np.int64)


def number(draws, present, symbols):
    return pa.array(np.asarray(draws, np.float32), pa.float32(), mask=~present)


def flag(draws, present, symbols):
    return pa.array(steps_of(draws) % 2 == 1, pa.bool_(), mask=~present)


def category(draws, present, symbols):
    chosen = np.asarray(symbols, object)[steps_of(draws) % len(symbols)]
    return pa.array(chosen.tolist(), pa.string(), mask=~present)


STATED = dict(number=number, flag=flag, category=category)


def stated(shape, name, draws, present):
    kind = shape["schema"]["columns"][name]["type"]
    if kind not in STATED:
        raise KeyError("the synthetic source states no values of the column type %s" % kind)
    symbols = None
    if kind == "category":
        if name not in shape["symbols"]:
            raise KeyError("the synthetic shape names no values for the column %s" % name)
        symbols = shape["symbols"][name]
    return STATED[kind](draws, np.asarray(present, bool), symbols)


class Counter:
    def __init__(self):
        self.next = {}

    def take(self, table, count):
        start = self.next.get(table, 0)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def graph_rows(rng, index, name, shape, settings, counter, level):
    started = time.perf_counter()
    schema = Schema(shape["schema"])
    entities, steps = settings["entities"], settings["steps"]
    nodes, layout = tuple(schema.nodes), schema.layout()
    width, signals = len(layout), schema.queries["signals"]
    counts = np.maximum(
        np.floor(np.asarray([shape["shares"][kind] for kind in nodes]) * entities).astype(np.int64), 2
    )
    offsets = np.r_[0, np.cumsum(counts)]
    total = int(offsets[-1])
    kinds = np.repeat(np.arange(len(nodes)), counts)
    birth = rng.integers(0, max(1, steps // 4), total)
    observed = (rng.random((steps, total)) < settings["alive"]) & (
        np.arange(steps)[:, None] >= birth[None, :]
    )
    roles = {}
    for role, allowed in schema.queries["references"].items():
        place = nodes.index(allowed[0])
        roles[role] = rng.choice(np.arange(offsets[place], offsets[place + 1]), steps)
        observed[np.arange(steps), roles[role]] = True
    first = roles[next(iter(roles))]
    facts = {}
    for table, entry in schema.edges.items():
        source_kind, target_kind = entry["endpoints"][0]
        sources = np.arange(offsets[nodes.index(source_kind)], offsets[nodes.index(source_kind) + 1])
        targets = np.arange(offsets[nodes.index(target_kind)], offsets[nodes.index(target_kind) + 1])
        degree = np.minimum(rng.poisson(settings["degree"], len(sources)), settings["max_degree"])
        source = np.repeat(sources, degree)
        target = rng.choice(targets, len(source))
        keep = source != target
        source, target = source[keep], target[keep]
        pairs = np.unique(source.astype(np.int64) * total + target)
        source, target = pairs // total, pairs % total
        begin = rng.integers(0, steps, len(source))
        length = np.minimum(rng.geometric(1.0 / settings["lifetime"], len(source)), steps - begin)
        pair = np.repeat(np.arange(len(source)), length)
        step = begin[pair] + np.arange(len(pair)) - np.repeat(np.cumsum(length) - length, length)
        both = observed[step, source[pair]] & observed[step, target[pair]]
        present = np.stack((source[pair][both], target[pair][both], step[both]), 1)
        held = np.zeros(len(source), bool)
        held[pair[both]] = True
        ended = held & (begin + length < steps)
        absent = np.stack((source[ended], target[ended], (begin + length)[ended]), 1)
        facts[table] = (present, absent)
    obs_step, obs_entity = np.nonzero(observed)
    order = np.lexsort((obs_entity, obs_step))
    obs_step, obs_entity = obs_step[order], obs_entity[order]
    seq = np.arange(len(obs_step)) - np.searchsorted(obs_step, obs_step)
    base = vectors(rng, total, width)
    x = base[obs_entity] + np.concatenate(
        (
            rng.normal(scale=0.3, size=(len(obs_step), width)).astype(np.float32),
            np.zeros((len(obs_step), width), np.float32),
        ),
        1,
    )
    x[:, :width] *= x[:, width:]
    node_ids = {kind: counter.take(kind, int(counts[place])) for place, kind in enumerate(nodes)}
    entity_id = np.concatenate([node_ids[kind] for kind in nodes])
    observation_ids = counter.take("observation", len(obs_step))
    query_ids = counter.take("query", steps)
    described = dict(id=pa.array([index], pa.int32()), name=pa.array([name]))
    for column in schema.graph:
        described[column] = stated(shape, column, np.asarray([index / 1000]), np.ones(1, bool))
    tables = {("graph", None, None): pa.table(described)}
    for place, kind in enumerate(nodes):
        members = np.arange(offsets[place], offsets[place + 1])
        tables[(kind, None, None)] = pa.table(
            dict(
                id=pa.array(node_ids[kind]),
                graph=pa.array(np.full(len(members), index), pa.int32()),
                entity=pa.array(members, pa.int32()),
                key=pa.array(["%s_%d" % (kind, member) for member in members]),
            )
        )
        picked = kinds[obs_entity] == place
        links = {"from": pa.array(entity_id[obs_entity[picked]]), "to": pa.array(observation_ids[picked])}
        for column in schema.nodes[kind]:
            position = layout.index(column)
            links[column] = stated(shape, column, x[picked, position], x[picked, width + position] > 0)
        tables[(OBSERVED + kind, kind, "observation")] = pa.table(links)
    tables[("observation", None, None)] = pa.table(
        dict(
            id=pa.array(observation_ids),
            seq=pa.array(seq, pa.int32()),
            step=pa.array(obs_step, pa.int32()),
        )
    )
    chosen = rng.integers(0, len(CATEGORIES), steps)
    asked = vectors(rng, steps, width)
    query = dict(
        id=pa.array(query_ids),
        graph=pa.array(np.full(steps, index), pa.int32()),
        step=pa.array(np.arange(steps), pa.int32()),
    )
    query[schema.queries["grouping"]] = pa.array(np.arange(steps) // settings["period"], pa.int32())
    query[schema.queries["key"]] = pa.array(query_ids * 7 + 3)
    query["category"] = pa.array([CATEGORIES[value] for value in chosen])
    for column in schema.queries["columns"]:
        position = layout.index(column)
        query[column] = stated(shape, column, asked[:, position], asked[:, width + position] > 0)
    if signals:
        query.update(
            signal=fixed(rng.normal(size=(steps, signals))),
            signal_observed=fixed(rng.random((steps, signals)) < 0.7),
        )
    tables[("query", None, None)] = pa.table(query)
    for role, allowed in schema.queries["references"].items():
        tables[(QUERY + role, "query", allowed[0])] = pa.table(
            {"from": pa.array(query_ids), "to": pa.array(entity_id[roles[role]])}
        )
    fact_seq = {}
    for table, entry in schema.edges.items():
        present, absent = facts[table]
        rows = np.concatenate((present, absent))
        flags = np.r_[np.ones(len(present), bool), np.zeros(len(absent), bool)]
        sequence = np.empty(len(rows), np.int64)
        for held in (True, False):
            picked = np.flatnonzero(flags == held)
            for step in np.unique(rows[picked, 2]):
                members = picked[rows[picked, 2] == step]
                start = fact_seq.get((held, step), 0)
                sequence[members] = np.arange(start, start + len(members))
                fact_seq[(held, step)] = start + len(members)
        columns = {
            "from": pa.array(entity_id[rows[:, 0]]),
            "to": pa.array(entity_id[rows[:, 1]]),
            "seq": pa.array(sequence, pa.int32()),
            "step": pa.array(rows[:, 2], pa.int32()),
            "present": pa.array(flags),
        }
        if entry["column"] is not None:
            columns[entry["column"]] = category(
                (rows[:, 0] + rows[:, 1]) / 1000, np.ones(len(rows), bool), shape["symbols"][entry["column"]]
            )
        for column in entry["columns"]:
            columns[column] = stated(shape, column, rng.random(len(rows)).astype(np.float32), flags)
        tables[(EDGE + table,) + entry["endpoints"][0]] = pa.table(columns)
    first_x = base[first, :width]
    if level == "node":
        moments = np.arange(settings["period"] - 1, steps, settings["period"])
        members = np.arange(offsets[0], offsets[1])
        label_step, label_member = np.nonzero(observed[moments][:, members])
        label_step, label_member = moments[label_step], members[label_member]
        y = base[label_member, :3].sum(1) + 0.1 * label_step + rng.normal(scale=0.5, size=len(label_step))
        split = np.where(label_step >= steps * 0.8, "validation", "train")
        label_ids = counter.take(TARGET, len(y))
        tables[(TARGET, None, None)] = pa.table(
            dict(
                id=pa.array(label_ids),
                y=pa.array(y, pa.float64()),
                split=pa.array(split),
                step=pa.array(label_step, pa.int32()),
            )
        )
        tables[(TARGET + "_of", TARGET, nodes[0])] = pa.table(
            {"from": pa.array(label_ids), "to": pa.array(entity_id[label_member])}
        )
    elif level == "query":
        y = first_x[:, :3].sum(1) + (chosen == 0) + rng.normal(scale=0.5, size=steps)
        split = np.where(np.arange(steps) >= steps * 0.8, "validation", "train")
        label_ids = counter.take(TARGET, steps)
        tables[(TARGET, None, None)] = pa.table(
            dict(id=pa.array(label_ids), y=pa.array(y, pa.float64()), split=pa.array(split))
        )
        tables[(TARGET + "_of", TARGET, "query")] = pa.table(
            {"from": pa.array(label_ids), "to": pa.array(query_ids)}
        )
    elif level == "path":
        source, later = path_points(steps, settings["period"])
        tau = (later - source).astype(np.float64)
        drift = first_x[source, :3].sum(1)
        y = np.concatenate(
            (
                0.05 * tau * drift + (chosen[source] == 0) + rng.normal(scale=0.5, size=len(source)),
                -2.0 * np.log1p(tau) * drift + rng.normal(scale=0.2, size=len(source)),
            )
        )
        source, tau = np.tile(source, len(SERIES)), np.tile(tau, len(SERIES))
        label_ids = counter.take(TARGET, len(source))
        tables[(TARGET, None, None)] = pa.table(
            dict(
                id=pa.array(label_ids),
                y=pa.array(y, pa.float64()),
                split=pa.array(["validation" if index % 2 == 1 else "train"] * len(source)),
                tau=pa.array(tau, pa.float32()),
                series=pa.array(np.repeat(SERIES, len(source) // len(SERIES)).tolist(), pa.string()),
            )
        )
        tables[(TARGET + "_of", TARGET, "query")] = pa.table(
            {"from": pa.array(label_ids), "to": pa.array(query_ids[source])}
        )
    else:
        label_ids = counter.take(TARGET, 1)
        tables[(TARGET, None, None)] = pa.table(
            dict(
                id=pa.array(label_ids),
                y=pa.array([float(first_x[:, 0].mean() + rng.normal(scale=0.2))], pa.float64()),
                split=pa.array(["validation" if index % 2 == 1 else "train"]),
            )
        )
        tables[(TARGET + "_of", TARGET, "graph")] = pa.table(
            {"from": pa.array(label_ids), "to": pa.array([index], pa.int64())}
        )
    LOG.info(
        "synthetic graph exit graph=%d seconds=%.3f entities=%d observations=%d facts=%d",
        index,
        time.perf_counter() - started,
        total,
        len(obs_step),
        sum(len(present) + len(absent) for present, absent in facts.values()),
    )
    return tables


def path_points(steps, period):
    moments = np.arange(period - 1, steps, period)
    source = np.repeat(np.arange(steps), [int((moments > step).sum()) for step in range(steps)])
    later = np.concatenate([moments[moments > step] for step in range(steps)] + [np.zeros(0, np.int64)])
    return source, later.astype(np.int64)


def labels(graphs, level, settings):
    steps, period = settings["steps"], settings["period"]
    if level == "query":
        return steps - (steps + 4) // 5, (steps + 4) // 5
    if level == "node":
        moments = np.arange(period - 1, steps, period)
        return 2 * int((moments < steps * 0.8).sum()), 2 * int((moments >= steps * 0.8).sum())
    if level == "path":
        points = len(path_points(steps, period)[0]) * len(SERIES)
        return points * (graphs - graphs // 2), points * (graphs // 2)
    return graphs - graphs // 2, graphs // 2


@exclusive
def generate(output, name, graphs, level, seed, shape, **settings):
    started = time.perf_counter()
    if graphs < 1 or level not in LEVELS:
        raise ValueError("synthetic data requires graphs and a target level among " + ", ".join(LEVELS))
    lacking = [key for key in SETTINGS if key not in settings]
    if lacking or len(settings) != len(SETTINGS):
        raise ValueError(
            "synthetic data states exactly the settings %s; lacking %s"
            % (", ".join(SETTINGS), ", ".join(lacking))
        )
    schema = Schema(shape["schema"])
    if schema.queries is None or set(shape["shares"]) != set(schema.nodes):
        raise ValueError("a synthetic shape states queries and a share of the entities for every kind")
    if settings["entities"] < len(schema.nodes) * 2 or settings["steps"] < 1 or settings["period"] < 1:
        raise ValueError("synthetic data requires two entities for every kind and positive steps and period")
    if (
        not 0 <= settings["alive"] <= 1
        or settings["degree"] < 0
        or settings["max_degree"] < 0
        or settings["lifetime"] < 1
    ):
        raise ValueError("synthetic graph sampling settings are outside their valid ranges")
    train, validation = labels(graphs, level, settings)
    if train < 2 or validation < 2:
        raise ValueError(
            "%s-level targets with %d graphs, %d steps and period %d give %d training and %d validation label groups; each needs two"
            % (level, graphs, settings["steps"], settings["period"], train, validation)
        )
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.joinpath(name + ".staging").exists():
        raise FileExistsError("synthetic staging exists; choose another name or output")
    rng = np.random.default_rng(seed)
    counter = Counter()
    with Stage(output / (name + ".staging")) as stage:
        for index in range(graphs):
            stage.write(graph_rows(rng, index, "graph_%03d" % index, shape, settings, counter, level))
    write_json(
        output / (name + ".task.json"),
        dict(name=name, staging=name + ".staging", target=TARGET, schema=schema.mapping()),
    )
    LOG.info(
        "synthetic exit seconds=%.3f staging=%s graphs=%d level=%s",
        time.perf_counter() - started,
        stage.directory,
        graphs,
        level,
    )
    return output / (name + ".task.json")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--shape", required=True, choices=sorted(SHAPES))
    parser.add_argument("--columns", type=int)
    parser.add_argument("--signals", type=int)
    parser.add_argument("--graphs", type=int, required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--degree", type=float, required=True)
    parser.add_argument("--max-degree", type=int, required=True)
    parser.add_argument("--alive", type=float, required=True)
    parser.add_argument("--lifetime", type=float, required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    configure()
    stated_widths = (args.columns, args.signals)
    if args.shape == "uniform" and None in stated_widths:
        parser.error("the uniform shape needs --columns and --signals")
    if args.shape != "uniform" and stated_widths != (None, None):
        parser.error("--columns and --signals belong to the uniform shape")
    shape = uniform(args.columns, args.signals) if args.shape == "uniform" else SHAPES[args.shape]()
    task = generate(
        args.output,
        args.name,
        args.graphs,
        args.level,
        args.seed,
        shape,
        entities=args.entities,
        steps=args.steps,
        degree=args.degree,
        max_degree=args.max_degree,
        alive=args.alive,
        lifetime=args.lifetime,
        period=args.period,
    )
    LOG.info("synthetic command exit seconds=%.3f", time.perf_counter() - started)
    print(task)


if __name__ == "__main__":
    main()
