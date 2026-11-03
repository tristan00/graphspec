import argparse
import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.build import forms
from graphspec.build.output import Stage
from graphspec.build.records import Counter as Counter
from graphspec.build.records import fixed
from graphspec.build.records import snapshots as snapshots
from graphspec.build.records import stage as stage
from graphspec.infra.files import write_json
from graphspec.infra.log import configure
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.decisions import RECORDED
from graphspec.schema.schema import (
    DECLARED,
    EDGE,
    LINK,
    OBSERVED,
    TARGET,
    TASK,
    VERSION,
    Schema,
)

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


@timed("synthetic.edge")
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


@timed("synthetic.vectors")
def vectors(rng, rows, columns):
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    present = (rng.random((rows, columns)) < 0.9).astype(np.float32)
    return np.concatenate((values * present, present), 1)


@timed("synthetic.steps_of")
def steps_of(draws):
    return np.floor(np.abs(np.asarray(draws, np.float64)) * 1000).astype(np.int64)


@timed("synthetic.number")
def number(draws, present, symbols):
    return pa.array(np.asarray(draws, np.float32), pa.float32(), mask=~present)


@timed("synthetic.flag")
def flag(draws, present, symbols):
    return pa.array(steps_of(draws) % 2 == 1, pa.bool_(), mask=~present)


@timed("synthetic.category")
def category(draws, present, symbols):
    chosen = np.asarray(symbols, object)[steps_of(draws) % len(symbols)]
    return pa.array(chosen.tolist(), pa.string(), mask=~present)


STATED = dict(number=number, flag=flag, category=category)


@timed("synthetic.stated")
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
        owner, tau = np.tile(source, len(SERIES)), np.tile(later - source, len(SERIES))
        aimed = np.isin(np.arange(steps), source)
        split = np.where(aimed, "validation" if index % 2 == 1 else "train", None).tolist()
        label_step = np.zeros(0, np.int64)
        points = dict(
            tau=pa.array(tau, pa.int64()),
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


@timed("synthetic.path_points")
def path_points(steps, period):
    moments = np.arange(period - 1, steps, period)
    source = np.repeat(np.arange(steps), [int((moments > step).sum()) for step in range(steps)])
    later = np.concatenate([moments[moments > step] for step in range(steps)] + [np.zeros(0, np.int64)])
    return source, later.astype(np.int64)


@timed("synthetic.labels")
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


AGENTS = ("a0", "a1", "a2")
SPOTS = ("s0", "s1", "s2", "s3", "s4")
GOALS = ("gain", "cost")
TURNS = (1, 2, 3, 4, 5, 6)
MODES = ("slow", "fast")
BLIND = (1, TURNS[-1])


@timed("synthetic.chooser")
def chooser(nothing):
    actions = dict(
        go=dict(roles=dict(to=dict(sort="node", kinds=["spot"], count=[1, 1])), parameters=["speed"]),
        cut=dict(roles=dict(along=dict(sort="edge", kinds=["spot"], count=[1, None])), parameters=["mode"]),
        put=dict(
            roles=dict(
                at=dict(sort="node", kinds=["spot"], count=[1, 2]), made=dict(sort="new", kinds=["mark"], count=[0, 1])
            ),
            parameters=["lit"],
        ),
    )
    target = dict(
        kind="decision",
        series=list(GOALS),
        about=None,
        utility=dict(name="weighted", better="higher", weights=dict(gain=1.0, cost=-1.0)),
        actors="agents",
        actions=actions,
        nothing=nothing,
        rule=None,
        generators=[],
    )
    return dict(references=["agents"], columns=[], target=target, weight=False, absent="stop", event=None, matures=None)


@timed("synthetic.decision_schema")
def decision_schema():
    number = dict(type="number", encoder="value")
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(load=number, speed=number, mode=dict(type="category", encoder="onehot"), lit=dict(type="flag", encoder="value")),
        graph=dict(columns=[]),
        nodes=dict(
            agent=dict(columns=["load"], holds=None),
            spot=dict(columns=["load"], holds=None),
            mark=dict(columns=[], holds=None),
        ),
        sets={},
        edges=dict(near=edge("spot", "spot", [], directed=False), base=edge("agent", "spot", [])),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=1,
            references=dict(
                agents=dict(sort="nodes", kinds=["agent"]),
                who=dict(sort="node", kinds=["agent"]),
                at=dict(sort="node", kinds=["spot"]),
            ),
            kinds=dict(turn=chooser("wait"), show=chooser(None), visit=kind(["at", "who"], [], None, ["at", "who"])),
        ),
    )


@timed("synthetic.acting")
def acting(kind, actor, targets, parameters):
    return dict(kind=kind, actor=actor, targets=targets, parameters=parameters)


@timed("synthetic.offered")
def offered(key, actions, together, origin):
    return dict(
        key=key,
        actions=actions,
        together=together,
        available=True,
        chosen=False,
        order=None,
        shown=None,
        logged=None,
        origin=origin,
        outcome=None,
    )


@timed("synthetic.worth")
def worth(moment, place, number):
    return dict(gain=0.5 * moment + 0.25 * number - place, cost=0.1 * ((moment * 3 + number + place) % 5))


def turned(moment, place, actors, generator):
    started = time.perf_counter()
    spot, edge_at = SPOTS[(moment + place) % 5], (moment + place) % 4
    along = [SPOTS[edge_at], "near", SPOTS[edge_at + 1]]
    across = [SPOTS[(edge_at + 3) % 4 + 1], "near", SPOTS[(edge_at + 3) % 4]]
    cuts = [along[::-1] if moment % 2 else along] + ([across] if moment % 3 == 0 else [])
    beside = SPOTS[(moment + place + 2) % 5]
    actor = actors[0]
    made = "m0" if (place, moment, actor) == (0, 3, "a0") else "m0_%d_%s" % (moment, actor)
    options = [
        offered("wait", [], "set", RECORDED),
        offered("go", [acting("go", actor, dict(to=[spot]), dict(speed=0.5 * moment + place))], "set", RECORDED),
        offered("cut", [acting("cut", actor, dict(along=cuts), dict(mode=MODES[moment % 2]))], "set", RECORDED),
        offered(
            "put",
            [
                acting("go", actor, dict(to=[spot]), dict(speed=1.5)),
                acting(
                    "put",
                    actor,
                    dict(at=[beside, spot] if (moment + place) % 2 else [spot], made=[] if moment % 3 == 2 else [made]),
                    dict(lit=moment % 2 == 0),
                ),
            ],
            "sequence",
            RECORDED,
        ),
    ]
    if len(actors) > 1:
        pair = [acting("go", other, dict(to=[SPOTS[(moment + at) % 5]]), dict(speed=None)) for at, other in enumerate(actors)]
        options.append(offered("pair", pair, "set", RECORDED))
    drawn = 0.25 if moment % 2 else None
    options.append(
        offered(
            "drawn",
            [acting("go", actor, dict(to=[SPOTS[(moment + place + 2) % 5]]), dict(speed=2.0))],
            "set",
            dict(generator=generator, seed=7 * moment + place, drawn=drawn),
        )
    )
    chosen = (moment + place) % (len(options) - 1)
    options[chosen]["chosen"] = True
    options[2]["available"] = moment % 3 != 0 or chosen == 2
    blind = (place, moment) == BLIND
    for number, option in enumerate(options):
        if (number != chosen) if blind else (moment % 2 or number == chosen):
            held = worth(moment, place, number)
            option["outcome"] = [held, worth(moment + 1, place, number)] if option["together"] == "sequence" else held
        if moment % 2 and option["origin"] == RECORDED:
            option["shown"] = 1.0 / (len(options) - 1)
            option["logged"] = 0.5 if number == chosen else 0.5 / (len(options) - 2)
    LOG.debug("turned exit options=%d seconds=%.4f", len(options), time.perf_counter() - started)
    return options


def shown(moment, place):
    started = time.perf_counter()
    options = [
        offered("show%d" % at, [acting("go", "a2", dict(to=[SPOTS[at]]), dict(speed=1.0))], "set", RECORDED)
        for at in range(4)
    ]
    blind = (place, moment) == BLIND
    for order, at in enumerate(((moment + place) % 4, (moment + place + 1) % 4)):
        options[at].update(chosen=True, order=order, outcome=None if blind else worth(moment, place, at + order))
    for at, option in enumerate(options):
        if blind and not option["chosen"]:
            option["outcome"] = worth(moment, place, at)
        option.update(shown=0.25, logged=0.5 if option["chosen"] else 0.0)
    LOG.debug("shown exit options=%d seconds=%.4f", len(options), time.perf_counter() - started)
    return options


def decision_graph(place, generator):
    started = time.perf_counter()
    tasks = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[])
    tasks.update(references=dict(agents=[], who=[], at=[]), signal=[], signal_observed=[])
    for moment in TURNS:
        last = moment == TURNS[-1]
        actors = [["a0", "a1"]] if last else [["a0"], ["a1"]]
        keys = [10 * moment + at for at in range(len(actors))]
        shows = moment % 2 == 0
        together = keys + ([10 * moment + 2] if shows else [])
        for actors_of, key in zip(actors, keys):
            following = {} if last else {actor: 10 * (moment + 1) + (0 if moment + 1 == TURNS[-1] else int(actor[1])) for actor in actors_of}
            target = dict(
                options=turned(moment, place, actors_of, generator),
                concurrent=[other for other in together if other != key],
                next=following,
                ended=last,
                discount=0.9,
            )
            tasks["key"].append(key)
            tasks["kind"].append("turn")
            tasks["references"]["agents"].append(actors_of)
            tasks["target"].append(target)
        if shows:
            tasks["key"].append(10 * moment + 2)
            tasks["kind"].append("show")
            tasks["references"]["agents"].append(["a2"])
            following = {} if last else {"a2": 10 * (moment + 2) + 2}
            tasks["target"].append(
                dict(
                    options=shown(moment, place),
                    concurrent=keys,
                    next=following,
                    ended=last,
                    discount=None,
                )
            )
        added = len(tasks["key"]) - len(tasks["time"])
        tasks["references"]["who"] += [None] * added
        tasks["references"]["at"] += [None] * added
        tasks["split"] += ["train" if moment <= 3 else "validation"] * added
        tasks["known"] += [moment + 1] * added
        tasks["key"].append(10 * moment + 9)
        tasks["kind"].append("visit")
        tasks["references"]["agents"].append(None)
        tasks["references"]["who"].append(AGENTS[(moment + place) % 3])
        tasks["references"]["at"].append(SPOTS[moment % 5])
        tasks["target"].append(None)
        tasks["split"].append(None)
        tasks["known"].append(None)
        added += 1
        tasks["time"] += [moment] * added
        tasks["weight"] += [None] * added
        tasks["signal"] += [[0.1 * moment + place + 0.01 * at] for at in range(added)]
        tasks["signal_observed"] += [[True]] * added
    agents = dict(key=list(AGENTS), since=[0] * 3, until=[None] * 3, load=[1.0 + at + place for at in range(3)])
    spots = dict(key=list(SPOTS), since=[0] * 5, until=[None] * 5, load=[2.0 * at - place for at in range(5)])
    near = dict(
        source=list(SPOTS[:-1]), target=list(SPOTS[1:]), present=[True] * 4, since=[0] * 4, until=[None] * 4
    )
    base = dict(source=list(AGENTS), target=["s0", "s2", "s4"], present=[True] * 3, since=[0] * 3, until=[None] * 3)
    result = dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(agent=list(AGENTS), spot=list(SPOTS), mark=["m0"]),
        values=[
            dict(form="state", parameters={}, rows=agents, kind="agent"),
            dict(form="state", parameters={}, rows=spots, kind="spot"),
            dict(form="state", parameters={}, rows=dict(key=["m0"], since=[5], until=[None]), kind="mark"),
        ],
        edges=[
            dict(form="state", parameters={}, rows=near, table="near", ends=("spot", "spot")),
            dict(form="state", parameters={}, rows=base, table="base", ends=("agent", "spot")),
        ],
        tasks=tasks,
    )
    LOG.debug("decision graph exit graph=%d tasks=%d seconds=%.4f", place, len(tasks["key"]), time.perf_counter() - started)
    return result


@timed("synthetic.decisions")
def decisions(generator):
    return dict(schema=decision_schema(), graphs=[decision_graph(place, generator) for place in range(2)])


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
