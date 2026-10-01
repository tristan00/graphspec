import argparse
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.build import forms
from graphspec.build.output import Stage
from graphspec.infra.files import write_json
from graphspec.infra.log import configure
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.references import SORTS
from graphspec.schema.schema import (
    DECLARED,
    EDGE,
    INNER,
    LINK,
    MEMBER,
    OBSERVED,
    TARGET,
    TASK,
    VERSION,
    WITHIN,
    WRITTEN,
    Schema,
    inward,
    outward,
)
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import UNITED

LOG = logging.getLogger(__name__)
KINDS = ("buy", "sell", "move")
OUTCOME = "outcome"
SERIES = ("level", "trend")
LEVELS = ("node", "task", "graph", "path")
SETTINGS = ("entities", "steps", "degree", "max_degree", "alive", "lifetime", "period")
SHARES = dict(person=0.5, place=0.2, item=0.2, team=0.1)
REFERENCES = dict(
    person=dict(sort="node", kinds=["person"]),
    place=dict(sort="node", kinds=["place"]),
    team=dict(sort="node", kinds=["team"]),
)
HOLDER = "holder"
UNIT = "step"
TIME = dict(model="clock", units=[UNIT], ties="sequence")


def edge(source, target, columns, directed=True, relation=DECLARED):
    return dict(
        endpoints=[[source, target]], directed=directed, columns=list(columns), relation=relation, about=None
    )


@timed("synthetic.valued")
def valued(series):
    return dict(name="weighted", better="higher", weights={name: 1.0 for name in series})


@timed("synthetic.kind")
def kind(references, columns, target, event):
    return dict(
        references=list(references),
        columns=list(columns),
        target=target,
        weight=False,
        absent="stop",
        event=event,
        matures=None,
    )


def asked(key, grouping, columns, signals, level):
    started = time.perf_counter()
    if level not in LEVELS:
        raise ValueError("a synthetic shape states what holds its targets as one of " + ", ".join(LEVELS))
    target = dict(kind="number", series=OUTCOME, about=None, utility=valued([OUTCOME]))
    if level == "path":
        target = dict(kind="path", series=list(SERIES), about=None, utility=valued(SERIES))
    references = dict(REFERENCES)
    kinds = {
        name: kind(REFERENCES, columns, None if level == "node" else target, ["person", "place"])
        for name in sorted(KINDS)
    }
    if level == "node":
        references[HOLDER] = dict(sort="node", kinds=["person"])
        kinds[OUTCOME] = kind([HOLDER], [], target, None)
    result = dict(
        key=key,
        grouping=grouping,
        signals=0 if level == "node" else signals,
        references=references,
        kinds=kinds,
    )
    LOG.debug("asked exit level=%s kinds=%d seconds=%.4f", level, len(kinds), time.perf_counter() - started)
    return result


def uniform(columns, signals, level):
    started = time.perf_counter()
    names = ["v%d" % index for index in range(columns)]
    stated = {name: dict(type="number", encoder="value") for name in names}
    stated.update(
        weight=dict(type="number", encoder="logarithm"), distance=dict(type="number", encoder="logarithm")
    )
    result = dict(
        schema=dict(
            version=VERSION,
            time=TIME,
            observers=False,
            columns=stated,
            graph=dict(columns=[]),
            nodes={kind: dict(columns=names, holds=None) for kind in SHARES},
            sets={},
            edges=dict(
                visits=edge("person", "place", ["weight"]),
                owns=edge("person", "item", []),
                member=edge("person", "team", []),
                near=edge("place", "place", []),
                located=edge("team", "place", ["distance"]),
            ),
            tasks=asked("key", "period", names, signals, level),
        ),
        shares=SHARES,
        symbols={},
        level=level,
        signals=signals,
    )
    LOG.debug("uniform shape exit columns=%d seconds=%.4f", columns, time.perf_counter() - started)
    return result


def varied(level):
    started = time.perf_counter()
    result = dict(
        schema=dict(
            version=VERSION,
            time=TIME,
            observers=False,
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
                person=dict(columns=["size", "speed", "colour", "open"], holds=None),
                place=dict(columns=["size"], holds=None),
                item=dict(columns=["speed", "colour"], holds=None),
                team=dict(columns=["open"], holds=None),
            ),
            sets={},
            edges=dict(
                visits=edge("person", "place", ["weight", "mode"]),
                owns=edge("person", "item", []),
                member=edge("person", "team", []),
                near=edge("place", "place", [], directed=False),
                located=edge("team", "place", ["distance"]),
                links=edge("person", "item", ["mode"], relation=dict(column="predicate")),
            ),
            tasks=asked("ticket", "week", ["size", "speed"], 2, level),
        ),
        shares=SHARES,
        symbols=dict(
            colour=["red", "green", "blue"],
            mode=["foot", "road", "rail", "air"],
            season=["dry", "wet"],
            predicate=["likes", "lends", "sells", "seeks", "sends", "shows"],
        ),
        level=level,
        signals=2,
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


def graph_rows(rng, index, name, shape, settings, counter):
    started = time.perf_counter()
    schema, level = Schema(shape["schema"]), shape["level"]
    entities, steps = settings["entities"], settings["steps"]
    nodes, layout = tuple(schema.nodes), schema.layout()
    width, signals = len(layout), shape["signals"]
    asked_columns = schema.asked_columns()
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
    for role, allowed in REFERENCES.items():
        place = nodes.index(allowed["kinds"][0])
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
    moments = np.arange(steps)
    described = dict(id=pa.array([index], pa.int32()), name=pa.array([name]), unit=pa.array([UNIT]))
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
        links = {
            "from": pa.array(entity_id[obs_entity[picked]]),
            "at": pa.array(obs_step[picked], pa.int64()),
            "seq": pa.array(seq[picked], pa.int32()),
        }
        for column in schema.nodes[kind]:
            position = layout.index(column)
            links[column] = stated(shape, column, x[picked, position], x[picked, width + position] > 0)
        tables[(OBSERVED + kind, kind, "observation")] = pa.table(links)
    tables.update(node_facts(schema, tables, moments, counter))
    chosen = rng.integers(0, len(KINDS), steps)
    given = vectors(rng, steps, width)
    drawn = dict(signal=rng.normal(size=(steps, signals)), signal_observed=rng.random((steps, signals)) < 0.7)
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
            "present": pa.array(flags),
            "at": pa.array(rows[:, 2], pa.int64()),
            "seq": pa.array(sequence, pa.int32()),
        }
        if entry["column"] is not None:
            columns[entry["column"]] = category(
                (rows[:, 0] + rows[:, 1]) / 1000, np.ones(len(rows), bool), shape["symbols"][entry["column"]]
            )
        for column in entry["columns"]:
            columns[column] = stated(shape, column, rng.random(len(rows)).astype(np.float32), flags)
        tables[(EDGE + table,) + entry["endpoints"][0]] = pa.table(columns)
    tables.update(edge_facts(schema, tables, moments))
    first_x = base[first, :width]
    held = np.zeros(0, np.int64)
    if level == "node":
        times = np.arange(settings["period"] - 1, steps, settings["period"])
        members = np.arange(offsets[0], offsets[1])
        label_step, label_member = np.nonzero(observed[times][:, members])
        label_step, held = times[label_step], members[label_member]
        y = base[held, :3].sum(1) + 0.1 * label_step + rng.normal(scale=0.5, size=len(label_step))
        owner = steps + np.arange(len(y))
        split = [None] * steps + np.where(label_step >= steps * 0.8, "validation", "train").tolist()
        points = {}
    elif level == "task":
        y = first_x[:, :3].sum(1) + (chosen == 0) + rng.normal(scale=0.5, size=steps)
        owner = np.arange(steps)
        split = np.where(np.arange(steps) >= steps * 0.8, "validation", "train").tolist()
        label_step, points = np.zeros(0, np.int64), {}
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
        owner, tau = np.tile(source, len(SERIES)), np.tile(tau, len(SERIES))
        aimed = np.isin(np.arange(steps), source)
        split = np.where(aimed, "validation" if index % 2 == 1 else "train", None).tolist()
        label_step = np.zeros(0, np.int64)
        points = dict(
            tau=pa.array(tau, pa.float32()),
            series=pa.array(np.repeat(SERIES, len(source)).tolist(), pa.string()),
        )
    else:
        y = np.asarray([float(first_x[:, 0].mean() + rng.normal(scale=0.2))])
        owner = np.asarray([steps - 1])
        split = [None] * (steps - 1) + ["validation" if index % 2 == 1 else "train"]
        label_step, points = np.zeros(0, np.int64), {}
    count = steps + len(held)
    ids, numbers = counter.take("task", count), counter.take(TARGET, len(y))
    keys = np.r_[counter.take("key", steps) * 7 + 3, -(numbers[: len(held)] + 1)]
    tasks = dict(
        id=pa.array(ids),
        graph=pa.array(np.full(count, index), pa.int32()),
        kind=pa.array([KINDS[value] for value in chosen] + [OUTCOME] * len(held)),
        time=pa.array(np.r_[np.arange(steps), label_step], pa.int64()),
        split=pa.array(split, pa.string()),
        weight=pa.array([None] * count, pa.float32()),
    )
    tasks[schema.tasks["grouping"]] = pa.array(
        np.r_[np.arange(steps), label_step] // settings["period"], pa.int32()
    )
    tasks[schema.tasks["key"]] = pa.array(keys)
    stated_by = np.r_[np.ones(steps, bool), np.zeros(len(held), bool)]
    for column in asked_columns:
        position = layout.index(column)
        tasks[column] = stated(
            shape,
            column,
            np.r_[given[:, position], np.zeros(len(held), np.float32)],
            np.r_[given[:, width + position] > 0, np.zeros(len(held), bool)] & stated_by,
        )
    if schema.tasks["signals"]:
        tasks.update({part: fixed(values) for part, values in drawn.items()})
    tables[("task", None, None)] = pa.table(tasks)
    for role, allowed in REFERENCES.items():
        tables[(TASK + role, "task", allowed["kinds"][0])] = pa.table(
            {"from": pa.array(ids[:steps]), "to": pa.array(entity_id[roles[role]])}
        )
    if level == "node":
        tables[(TASK + HOLDER, "task", nodes[0])] = pa.table(
            {"from": pa.array(ids[steps:]), "to": pa.array(entity_id[held])}
        )
    tables[(TARGET, None, None)] = pa.table(
        dict(
            id=pa.array(numbers),
            y=pa.array(y, pa.float64()),
            known=pa.array([None] * len(y), pa.int64()),
            **points,
        )
    )
    tables[(LINK, TARGET, "task")] = pa.table({"from": pa.array(numbers), "to": pa.array(ids[owner])})
    LOG.info(
        "synthetic graph exit graph=%d seconds=%.3f entities=%d observations=%d facts=%d",
        index,
        time.perf_counter() - started,
        total,
        len(obs_step),
        sum(len(present) + len(absent) for present, absent in facts.values()),
    )
    return tables


def node_facts(schema, tables, moments, counter):
    started = time.perf_counter()
    keys = [(OBSERVED + kind, kind, "observation") for kind in schema.nodes]
    sequenced = "seq" in schema.stamps()
    facts = forms.together([tables[key] for key in keys], moments, sequenced)
    stamps = [name for name in schema.stamps() if name != "observers"]
    result, parts = {}, []
    for key, table in zip(keys, facts):
        ids = counter.take("observation", table.num_rows)
        parts.append(pa.table(dict(id=pa.array(ids), **{name: table.column(name) for name in stamps})))
        result[key] = table.drop_columns(stamps).add_column(1, "to", pa.array(ids))
    result[("observation", None, None)] = pa.concat_tables(parts)
    LOG.debug(
        "observed exit kinds=%d facts=%d seconds=%.4f",
        len(keys),
        result[("observation", None, None)].num_rows,
        time.perf_counter() - started,
    )
    return result


def edge_facts(schema, tables, moments):
    started = time.perf_counter()
    keys = [key for key in tables if key[0].startswith(EDGE)]
    sequenced = "seq" in schema.stamps()
    parts = {key: [] for key in keys}
    for flag in (True, False):
        chosen = [
            tables[key].filter(pa.array(np.asarray(tables[key].column("present")) == flag)) for key in keys
        ]
        for key, table in zip(keys, forms.together(chosen, moments, sequenced)):
            parts[key].append(table)
    result = {}
    for key in keys:
        table = pa.concat_tables(parts[key])
        names = [name for name in tables[key].column_names if name not in ("at", "seq")]
        names[3:3] = [name for name in schema.stamps() if name != "observers"]
        result[key] = table.select(names)
    LOG.debug("held exit tables=%d seconds=%.4f", len(keys), time.perf_counter() - started)
    return result


def path_points(steps, period):
    moments = np.arange(period - 1, steps, period)
    source = np.repeat(np.arange(steps), [int((moments > step).sum()) for step in range(steps)])
    later = np.concatenate([moments[moments > step] for step in range(steps)] + [np.zeros(0, np.int64)])
    return source, later.astype(np.int64)


def labels(graphs, level, settings):
    steps, period = settings["steps"], settings["period"]
    if level == "task":
        return steps - (steps + 4) // 5, (steps + 4) // 5
    if level == "node":
        moments = np.arange(period - 1, steps, period)
        return 2 * int((moments < steps * 0.8).sum()), 2 * int((moments >= steps * 0.8).sum())
    if level == "path":
        points = len(path_points(steps, period)[0]) * len(SERIES)
        return points * (graphs - graphs // 2), points * (graphs // 2)
    return graphs - graphs // 2, graphs // 2


@exclusive
def generate(output, name, graphs, seed, shape, **settings):
    started = time.perf_counter()
    level = shape["level"]
    if graphs < 1 or level not in LEVELS:
        raise ValueError("synthetic data requires graphs and targets held by one of " + ", ".join(LEVELS))
    lacking = [key for key in SETTINGS if key not in settings]
    if lacking or len(settings) != len(SETTINGS):
        raise ValueError(
            "synthetic data states exactly the settings %s; lacking %s"
            % (", ".join(SETTINGS), ", ".join(lacking))
        )
    schema = Schema(shape["schema"])
    if set(shape["shares"]) != set(schema.nodes):
        raise ValueError("a synthetic shape states a share of the entities for every kind")
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
            "targets held by %s with %d graphs, %d steps and period %d give %d training and %d validation label groups; each needs two"
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
            stage.write(graph_rows(rng, index, "graph_%03d" % index, shape, settings, counter))
    write_json(
        output / (name + ".task.json"),
        dict(name=name, staging=name + ".staging", schema=schema.mapping()),
    )
    LOG.info(
        "synthetic exit seconds=%.3f staging=%s graphs=%d level=%s",
        time.perf_counter() - started,
        stage.directory,
        graphs,
        level,
    )
    return output / (name + ".task.json")


GIVEN = dict(
    key=pa.string(),
    source=pa.string(),
    target=pa.string(),
    relation=pa.string(),
    present=pa.bool_(),
    at=pa.int64(),
    index=pa.int64(),
    since=pa.int64(),
    until=pa.int64(),
    known=pa.int64(),
    replaced=pa.int64(),
    seq=pa.int32(),
    observers=pa.list_(pa.string()),
)
OPTIONAL = ("known", "replaced", "observers")
PARTS = ("schema", "graphs")
GRAPH = ("name", "unit", "columns", "nodes", "values", "edges", "tasks")
ASKED = ("key", "kind", "split", "weight", "references", "target", "known")
STATEMENT = ("form", "parameters", "rows")


@timed("synthetic.parted")
def parted(held, names, where):
    lacking = [name for name in names if name not in held]
    unknown = sorted(set(held) - set(names))
    if lacking or unknown:
        raise ValueError(
            "%s states exactly %s; it lacks %s and holds %s besides"
            % (where, ", ".join(names), ", ".join(lacking) or "nothing", ", ".join(unknown) or "nothing")
        )
    return held


def table(schema, columns, where):
    started = time.perf_counter()
    held = {}
    for name, values in columns.items():
        if name in GIVEN:
            kind = GIVEN[name]
        elif name in schema.columns:
            kind = schema.kind(name).written
        else:
            raise ValueError("%s states the column %s, which its schema does not" % (where, name))
        held[name] = pa.array(list(values), kind)
    result = pa.table(held)
    LOG.debug("table exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
    return result


def facts(schema, statement, given, values, where):
    started = time.perf_counter()
    parted(
        statement, STATEMENT + tuple(name for name in ("kind", "table", "ends") if name in statement), where
    )
    stated_rows = table(schema, statement["rows"], where)
    if schema.model == "none":
        if statement["form"] is not None:
            raise ValueError("%s names a form, and the graphs of its schema have no time" % where)
        result = stated_rows
    else:
        result = forms.reduced(statement["form"], schema.model, stated_rows, **statement["parameters"])
    stamps = schema.stamps()
    for name, kind in stamps.items():
        if name not in result.column_names:
            if name not in OPTIONAL:
                raise ValueError("%s states no column %s" % (where, name))
            result = result.append_column(name, pa.nulls(result.num_rows, WRITTEN[kind]))
    wanted = list(given) + list(stamps) + list(values)
    if sorted(result.column_names) != sorted(wanted):
        raise ValueError(
            "%s states the columns %s and is to state %s"
            % (where, ", ".join(result.column_names), ", ".join(wanted))
        )
    LOG.debug("facts exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
    return result.select(wanted)


@timed("synthetic.none")
def none(schema, given, values):
    names = list(given) + list(schema.stamps()) + list(values)
    kinds = dict(GIVEN, **{name: schema.kind(name).written for name in values})
    kinds.update({name: WRITTEN[kind] for name, kind in schema.stamps().items()})
    return pa.table({name: pa.array([], kinds[name]) for name in names})


@timed("synthetic.gathered")
def gathered(schema, statements, given, values, where):
    found = [facts(schema, statement, given, values, where) for statement in statements]
    return pa.concat_tables(found) if found else none(schema, given, values)


@timed("synthetic.fields")
def fields(schema):
    return GRAPH + (("sets",) if schema.sets else ()) + (("held",) if schema.holds else ())


def case_rows(schema, index, graph, counter):
    started = time.perf_counter()
    where = "the graph %s" % parted(graph, fields(schema), "a graph of a synthetic case")["name"]
    united = schema.model in UNITED
    if united != (graph["unit"] is not None):
        raise ValueError("%s states a unit of time exactly when its schema has a clock" % where)
    described = dict(id=pa.array([index], pa.int32()), name=pa.array([graph["name"]]))
    if united:
        described["unit"] = pa.array([graph["unit"]])
    described.update(
        table(schema, {name: [graph["columns"][name]] for name in schema.graph}, where).to_pydict()
    )
    tables = {("graph", None, None): table_of(schema, described)}
    lookup, entity, stamps = {}, 0, list(schema.stamps())
    for kind in schema.entities():
        keys = list(graph["nodes"][kind])
        ids = counter.take(kind, len(keys))
        lookup[kind] = dict(zip(keys, ids.tolist()))
        if len(lookup[kind]) != len(keys):
            raise ValueError("%s names a node of the kind %s twice" % (where, kind))
        tables[(kind, None, None)] = pa.table(
            dict(
                id=pa.array(ids),
                graph=pa.array(np.full(len(keys), index), pa.int32()),
                entity=pa.array(np.arange(entity, entity + len(keys)), pa.int32()),
                key=pa.array(keys, pa.string()),
            )
        )
        entity += len(keys)

    def found(kind, keys):
        strangers = sorted({key for key in keys if key not in lookup[kind]})
        if strangers:
            raise ValueError("%s names nodes of the kind %s it does not hold: %s" % (where, kind, strangers))
        return pa.array([lookup[kind][key] for key in keys], pa.int64())

    parts = []
    for kind, columns in schema.entities().items():
        held_rows = gathered(
            schema,
            [statement for statement in graph["values"] if statement["kind"] == kind],
            ("key",),
            columns,
            "%s, values of %s" % (where, kind),
        )
        ids = counter.take("observation", held_rows.num_rows)
        parts.append(pa.table(dict(id=pa.array(ids), **{name: held_rows.column(name) for name in stamps})))
        tables[(OBSERVED + kind, kind, "observation")] = pa.table(
            {
                "from": found(kind, held_rows.column("key").to_pylist()),
                "to": pa.array(ids),
                **{name: held_rows.column(name) for name in columns},
            }
        )
    tables[("observation", None, None)] = pa.concat_tables(parts)
    for name, entry in schema.edges.items():
        given = ("source", "target", "present") + (() if entry["column"] is None else ("relation",))
        for ends in entry["endpoints"]:
            held_rows = gathered(
                schema,
                [
                    statement
                    for statement in graph["edges"]
                    if statement["table"] == name and tuple(statement["ends"]) == ends
                ],
                given,
                entry["columns"],
                "%s, edges of %s" % (where, name),
            )
            columns = {
                "from": found(ends[0], held_rows.column("source").to_pylist()),
                "to": found(ends[1], held_rows.column("target").to_pylist()),
                "present": held_rows.column("present"),
            }
            columns.update({stamp: held_rows.column(stamp) for stamp in stamps})
            if entry["column"] is not None:
                columns[entry["column"]] = held_rows.column("relation")
            columns.update({column: held_rows.column(column) for column in entry["columns"]})
            tables[(EDGE + name,) + ends] = pa.table(columns)
    for kind, entry in schema.sets.items():
        tables.update(case_members(schema, kind, graph["sets"][kind], lookup, found, where))
    for kind in schema.holds:
        tables.update(case_held(schema, index, kind, graph["held"][kind], counter, lookup, where))
    tables.update(case_tasks(schema, index, graph["tasks"], counter, lookup, where))
    LOG.info(
        "synthetic case graph exit graph=%s tables=%d seconds=%.3f",
        graph["name"],
        len(tables),
        time.perf_counter() - started,
    )
    return tables


def case_members(schema, kind, rows, lookup, found, where):
    started = time.perf_counter()
    entry = schema.sets[kind]
    where = "%s, members of %s" % (where, kind)
    parted(rows, ("set", "member", "role") + entry["members"], where)
    owners = []
    for key in rows["member"]:
        holders = [member for member in entry["kinds"] if key in lookup[member]]
        if len(holders) != 1:
            raise ValueError(
                "%s names the node %s, which %d of its kinds of node hold" % (where, key, len(holders))
            )
        owners.append(holders[0])
    tables = {}
    for member in entry["kinds"]:
        chosen = [row for row, owner in enumerate(owners) if owner == member]
        tables[(MEMBER + kind, kind, member)] = pa.table(
            {
                "from": found(kind, [rows["set"][row] for row in chosen]),
                "to": found(member, [rows["member"][row] for row in chosen]),
                "role": pa.array([rows["role"][row] for row in chosen], pa.string()),
                **{
                    column: pa.array([rows[column][row] for row in chosen], schema.kind(column).written)
                    for column in entry["members"]
                },
            }
        )
    LOG.debug(
        "case members exit kind=%s rows=%d seconds=%.4f", kind, len(owners), time.perf_counter() - started
    )
    return tables


def case_held(schema, index, kind, stated, counter, lookup, where):
    started = time.perf_counter()
    held = schema.holds[kind]
    where = "%s, graphs held by %s" % (where, kind)
    parted(stated, ("nodes", "edges"), where)
    nodes = parted(stated["nodes"], ("owner", "kind", "key") + inward(held), where + ", nodes")
    edges = parted(
        stated["edges"], ("owner", "source", "target", "relation") + outward(held), where + ", edges"
    )
    strangers = sorted({str(owner) for owner in nodes["owner"] + edges["owner"] if owner not in lookup[kind]})
    if strangers:
        raise ValueError("%s names owners the graph does not hold: %s" % (where, ", ".join(strangers)))
    ids = counter.take(INNER + kind, len(nodes["key"]))
    position = dict(zip(zip(nodes["owner"], nodes["key"]), ids.tolist()))
    if len(position) != len(ids):
        raise ValueError("%s names one held node twice" % where)
    tables = {
        (INNER + kind, None, None): pa.table(
            dict(
                id=pa.array(ids),
                graph=pa.array(np.full(len(ids), index), pa.int32()),
                owner=pa.array(nodes["owner"], pa.string()),
                kind=pa.array(nodes["kind"], pa.string()),
                key=pa.array(nodes["key"], pa.string()),
                **{column: pa.array(nodes[column], schema.kind(column).written) for column in inward(held)},
            )
        ),
        (WITHIN + kind, INNER + kind, INNER + kind): pa.table(
            {
                "from": pa.array(
                    [position[pair] for pair in zip(edges["owner"], edges["source"])], pa.int64()
                ),
                "to": pa.array([position[pair] for pair in zip(edges["owner"], edges["target"])], pa.int64()),
                "relation": pa.array(edges["relation"], pa.string()),
                **{column: pa.array(edges[column], schema.kind(column).written) for column in outward(held)},
            }
        ),
    }
    LOG.debug("case held exit kind=%s nodes=%d seconds=%.4f", kind, len(ids), time.perf_counter() - started)
    return tables


def ended(schema, lookup, reference, member, where):
    started = time.perf_counter()
    source, relation, target = member
    if relation is None:
        target_kind, target = None, None
    else:
        found = [kind for kind in schema.nodes if target in lookup[kind]]
        if len(found) != 1:
            raise ValueError(
                "%s names the node %s as the end of an edge, which %d kinds of node hold"
                % (where, target, len(found))
            )
        target_kind = found[0]
    found = [kind for kind in schema.tasks["references"][reference]["kinds"] if source in lookup[kind]]
    if len(found) != 1:
        raise ValueError(
            "%s names the node %s under the reference %s, which %d of its kinds of node hold"
            % (where, source, reference, len(found))
        )
    LOG.debug("ended exit reference=%s seconds=%.4f", reference, time.perf_counter() - started)
    return found[0], source, relation, target_kind, target


def members(sort, stated):
    started = time.perf_counter()
    if sort.nodes and not sort.edges:
        held = [(key, None, None) for key in (stated if isinstance(stated, list) else [stated])]
    elif sort.edges and not sort.nodes:
        held = [tuple(member) for member in (stated if isinstance(stated, list) else [stated])]
    else:
        held = [(key, None, None) for key in stated["nodes"]] + [tuple(member) for member in stated["edges"]]
    LOG.debug("members exit members=%d seconds=%.4f", len(held), time.perf_counter() - started)
    return held


def case_tasks(schema, index, tasks, counter, lookup, where):
    started = time.perf_counter()
    asked = schema.tasks
    names = list(ASKED) + list(schema.asked_columns())
    names += ["time"] if schema.model != "none" else []
    names += ["observer"] if schema.observers else []
    names += ["grouping"] if asked["grouping"] is not None else []
    names += ["signal", "signal_observed"] if asked["signals"] else []
    parted(tasks, names, where + ", tasks")
    count = len(tasks["key"])
    ids = counter.take("task", count)
    held = dict(
        id=pa.array(ids),
        graph=pa.array(np.full(count, index), pa.int32()),
        kind=pa.array(tasks["kind"], pa.string()),
    )
    if schema.model != "none":
        held["time"] = pa.array(tasks["time"], pa.int64())
    if schema.observers:
        held["observer"] = pa.array(tasks["observer"], pa.string())
    held["split"] = pa.array(tasks["split"], pa.string())
    held["weight"] = pa.array(tasks["weight"], pa.float32())
    if asked["grouping"] is not None:
        held[asked["grouping"]] = pa.array(tasks["grouping"], pa.int32())
    held[asked["key"]] = pa.array(tasks["key"], pa.int64())
    held.update(
        table(schema, {name: tasks[name] for name in schema.asked_columns()}, where + ", tasks").to_pydict()
    )
    if asked["signals"]:
        held.update(signal=fixed(tasks["signal"]), signal_observed=fixed(tasks["signal_observed"]))
    tables = {("task", None, None): table_of(schema, held)}
    for name, entry in asked["references"].items():
        sort = SORTS.named(entry["sort"])
        stored = sort.columns()
        if stored is None:
            continue
        rows = {
            kind: dict(task=[], source=[], relation=[], to_kind=[], to_key=[], place=[])
            for kind in entry["kinds"]
        }
        for task, stated in enumerate(tasks["references"][name]):
            if stated is None:
                continue
            for place, member in enumerate(members(sort, stated)):
                kind, source, relation, to_kind, to_key = ended(schema, lookup, name, member, where)
                part = rows[kind]
                part["task"].append(int(ids[task]))
                part["source"].append(lookup[kind][source])
                part["relation"].append(relation)
                part["to_kind"].append(to_kind)
                part["to_key"].append(to_key)
                part["place"].append(place)
        for kind, part in rows.items():
            columns = {"from": pa.array(part["task"], pa.int64()), "to": pa.array(part["source"], pa.int64())}
            for column in stored:
                columns[column] = pa.array(part[column], pa.int32() if column == "place" else pa.string())
            tables[(TASK + name, "task", kind)] = pa.table(columns)
    stored = schema.target_columns()
    aimed = dict(task=[], y=[], known=[], **{name: [] for name in stored})
    for task, (target, known) in enumerate(zip(tasks["target"], tasks["known"])):
        if target is None:
            continue
        stated = asked["kinds"][tasks["kind"][task]]["target"]
        if stated is None:
            raise ValueError("%s, tasks: a task of the kind %s states a target" % (where, tasks["kind"][task]))
        kind = TARGETS.named(stated["kind"])
        for row in kind.staged(target, stated["series"], "%s, the task %r" % (where, tasks["key"][task])):
            aimed["task"].append(int(ids[task]))
            aimed["y"].append(row["y"])
            aimed["known"].append(known)
            for name in stored:
                aimed[name].append(row[name] if name in kind.columns else None)
    numbers = counter.take(TARGET, len(aimed["y"]))
    columns = dict(id=pa.array(numbers), y=pa.array(aimed["y"], pa.float64()))
    if schema.model != "none":
        columns["known"] = pa.array(aimed["known"], pa.int64())
    columns.update({name: pa.array(aimed[name], WRITTEN[written]) for name, written in stored.items()})
    tables[(TARGET, None, None)] = pa.table(columns)
    tables[(LINK, TARGET, "task")] = pa.table(
        {"from": pa.array(numbers), "to": pa.array(aimed["task"], pa.int64())}
    )
    LOG.debug("case tasks exit tasks=%d seconds=%.4f", count, time.perf_counter() - started)
    return tables


@timed("synthetic.table_of")
def table_of(schema, columns):
    return pa.table(
        {
            name: pa.array(values, schema.kind(name).written)
            if isinstance(values, list) and name in schema.columns
            else values
            for name, values in columns.items()
        }
    )


def restated(schema, statement, given, values, times, where):
    started = time.perf_counter()
    found = facts(schema, statement, given, values, where)
    since = np.asarray(found.column("since").to_numpy(zero_copy_only=False), np.int64)
    until = found.column("until").to_pylist()
    rows, moments = [], []
    for row, (start, stop) in enumerate(zip(since.tolist(), until)):
        chosen = [moment for moment in times if start <= moment and (stop is None or moment < stop)]
        rows += [row] * len(chosen)
        moments += chosen
    taken = found.take(pa.array(rows, pa.int64())).drop_columns(["since", "until"])
    columns = dict(taken.to_pydict(), at=moments)
    result = dict(
        {name: statement[name] for name in statement if name not in STATEMENT},
        form="snapshot",
        parameters=dict(times=list(times)),
        rows={
            name: held
            for name, held in columns.items()
            if name not in OPTIONAL or any(value is not None for value in held)
        },
    )
    LOG.debug(
        "restated exit facts=%d rows=%d seconds=%.4f",
        found.num_rows,
        len(rows),
        time.perf_counter() - started,
    )
    return result


def snapshots(case):
    started = time.perf_counter()
    schema = Schema(parted(case, PARTS, "a synthetic case")["schema"])
    graphs = []
    for graph in case["graphs"]:
        where = "the graph %s" % graph["name"]
        stated_facts = [
            (statement, ("key",), schema.entities()[statement["kind"]], "values")
            for statement in graph["values"]
        ] + [
            (
                statement,
                ("source", "target", "present")
                + (() if schema.edges[statement["table"]]["column"] is None else ("relation",)),
                schema.edges[statement["table"]]["columns"],
                "edges",
            )
            for statement in graph["edges"]
        ]
        times = set(graph["tasks"]["time"])
        for statement, given, values, _ in stated_facts:
            found = facts(schema, statement, given, values, where)
            for name in ("since", "until"):
                times |= {moment for moment in found.column(name).to_pylist() if moment is not None}
        times = sorted(times)
        held_parts = dict(values=[], edges=[])
        for statement, given, values, part in stated_facts:
            held_parts[part].append(restated(schema, statement, given, values, times, where))
        graphs.append(dict(graph, **held_parts))
    LOG.info("synthetic snapshots exit graphs=%d seconds=%.3f", len(graphs), time.perf_counter() - started)
    return dict(case, graphs=graphs)


@exclusive
def stage(output, name, case):
    started = time.perf_counter()
    schema = Schema(parted(case, PARTS, "a synthetic case")["schema"])
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.joinpath(name + ".staging").exists():
        raise FileExistsError("synthetic staging exists; choose another name or output")
    counter = Counter()
    with Stage(output / (name + ".staging")) as held_stage:
        for index, graph in enumerate(case["graphs"]):
            held_stage.write(case_rows(schema, index, graph, counter))
    write_json(
        output / (name + ".task.json"),
        dict(name=name, staging=name + ".staging", schema=schema.mapping()),
    )
    LOG.info(
        "synthetic case exit seconds=%.3f staging=%s graphs=%d",
        time.perf_counter() - started,
        held_stage.directory,
        len(case["graphs"]),
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
    shape = (
        uniform(args.columns, args.signals, args.level)
        if args.shape == "uniform"
        else SHAPES[args.shape](args.level)
    )
    task = generate(
        args.output,
        args.name,
        args.graphs,
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
