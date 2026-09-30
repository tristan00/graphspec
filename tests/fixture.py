import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.schema import EDGE, LINK, OBSERVED, TARGET, TASK
from graphspec.sources import synthetic

LOG = logging.getLogger(__name__)
VERSION = 7
ROLES = ("actor", "subject", "context")
QUERY = "query_"
LABELS = "outcome"
JOINED = "step_campaign"
SCALING = ("reward_mean", "reward_sd")
UNSPLIT = "spread_indices"
ROW_KEYS = dict(decision="key", turn="grouping")
QUERY_COLUMNS = dict(key="key", period="grouping")
UNIT = "step"
VIEW = {"time/steps": "tasks", "nodes/referenced": "keep"}
MOMENT = "step"
LEVELS = dict(query="task", node="node", graph="graph", path="path")
KINDS = ("events/kinds", "neighbors/kinds")
DERIVED = dict(
    actor="first",
    subject="second",
    query_context="starts",
    query_step="row_step",
    decision="key",
    decision_actor="event_first",
    query="task",
    turn="obs_period",
    turn_starts="period_starts",
    turn_ends="period_ends",
    turn_membership="period_membership",
    turn_actor_keys="period_first_keys",
    turn_actor_rows="period_first_rows",
    turn_action_category="period_kind",
    turn_numbers="period_numbers",
    turn_observation_keys="period_observation_keys",
    turn_walks="period_walks",
)
RELATIONS_FIRST = dict(
    configurations=("09_fosr_drop_cap_walks",),
    order=(
        "graphspec.periods",
        "graphspec.window",
        "graphspec.nodes",
        "graphspec.relations",
        "graphspec.derive",
        "graphspec.edges",
        "graphspec.merging",
    ),
)


def read(place):
    started = time.perf_counter()
    place = Path(place)
    if not (place / "fixture.json").exists():
        raise FileNotFoundError("the fixture of the ground is absent: %s" % place)
    held = read_json(place / "fixture.json")
    if held["truth_version"] != VERSION:
        raise ValueError(
            "the fixture holds version %s; its reader reads %d" % (held["truth_version"], VERSION)
        )
    LOG.debug("read exit seconds=%.4f", time.perf_counter() - started)
    return held


def configurations(place):
    held = read(place)
    return held["configurations"], held["joined"]


def labels(place):
    return tuple(read(place)["stages"])


def order(entry):
    from support import SEVEN

    started = time.perf_counter()
    result = RELATIONS_FIRST["order"] if entry["name"] in RELATIONS_FIRST["configurations"] else SEVEN
    LOG.debug("order exit name=%s seconds=%.6f", entry["name"], time.perf_counter() - started)
    return result


def holds(names, count):
    from support import SEVEN

    started = time.perf_counter()
    result = set(names[:count]) == set(SEVEN[:count])
    LOG.debug("holds exit count=%d held=%s seconds=%.6f", count, result, time.perf_counter() - started)
    return result


def pipeline(registry, entry):
    from support import pipelined

    started = time.perf_counter()
    values = spec(entry["values"])
    if "relations/direction" in values:
        values["edges/direction"] = values["relations/direction"]
    result = pipelined(registry, values, order(entry))
    LOG.debug("pipeline exit name=%s seconds=%.4f", entry["name"], time.perf_counter() - started)
    return result


def shaped(source):
    started = time.perf_counter()
    settings = dict(source["settings"])
    shape = synthetic.uniform(settings.pop("features") // 2, settings.pop("signals"), LEVELS[source["level"]])
    LOG.debug("shaped exit level=%s seconds=%.4f", source["level"], time.perf_counter() - started)
    return shape, settings


def columns(vectors, names, layout, stated):
    values = np.asarray(vectors.flatten().to_numpy(), np.float32).reshape(len(vectors), 2 * len(layout))
    return {
        name: pa.array(
            values[:, layout.index(name)],
            pa.float32(),
            mask=(values[:, len(layout) + layout.index(name)] == 0) | ~stated,
        )
        for name in names
    }


@timed("fixture.moved")
def moved(table, old, new, kind):
    place = table.schema.get_field_index(old)
    return table.set_column(place, new, table.column(old).cast(kind))


def restated(schema, found, layout):
    started = time.perf_counter()
    observation = found[("observation", None, None)]
    vectors = observation.column("x").combine_chunks()
    place = {number: row for row, number in enumerate(observation.column("id").to_pylist())}
    result = {}
    for (table, source, target), held in found.items():
        if table == "observed":
            rows = pa.array([place[number] for number in held.column("to").to_pylist()], pa.int64())
            held = pa.table(
                {
                    "from": held.column("from"),
                    "at": observation.column(MOMENT).take(rows).cast(pa.int64()),
                    "seq": observation.column("seq").take(rows),
                    **columns(vectors.take(rows), schema.nodes[source], layout, np.ones(len(rows), bool)),
                }
            )
            result[(OBSERVED + source, source, target)] = held
        elif table.startswith(EDGE):
            for name in schema.edges[table[len(EDGE) :]]["columns"]:
                values = np.asarray(held.column(name).to_numpy(), np.float32)
                held = held.set_column(
                    held.schema.get_field_index(name),
                    name,
                    pa.array(values, pa.float32(), mask=np.isnan(values)),
                )
            names = ["from", "to", "present", MOMENT, "seq"]
            names += [name for name in held.column_names if name not in names]
            result[(table, source, target)] = moved(held.select(names), MOMENT, "at", pa.int64())
    LOG.debug("restated exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def asked(schema, level, found, layout):
    started = time.perf_counter()
    stated = schema.tasks
    query, labels = found[("query", None, None)], found[(LABELS, None, None)]
    (link,) = [held for key, held in found.items() if key[0] == LABELS + "_of"]
    owner = np.asarray(query.column("graph").to_numpy(), np.int64)
    old = np.asarray(query.column("id").to_numpy(), np.int64)
    moments = np.asarray(query.column(MOMENT).to_numpy(), np.int64)
    grouping = np.asarray(query.column("period").to_numpy(), np.int64)
    numbers = np.asarray(labels.column("id").to_numpy(), np.int64)
    place = np.empty(len(numbers), np.int64)
    place[np.searchsorted(numbers, np.asarray(link.column("from").to_numpy(), np.int64))] = np.asarray(
        link.column("to").to_numpy(), np.int64
    )
    graphs = np.unique(owner)
    split = np.asarray(labels.column("split").to_pylist(), object)
    count = len(old)
    held = dict(
        graph=owner,
        kind=np.asarray(query.column("category").to_pylist(), object),
        time=moments,
        split=np.full(count, None, object),
        grouping=grouping,
        key=np.asarray(query.column("key").to_numpy(), np.int64),
        stated=np.ones(count, bool),
        row=np.arange(count),
        holder=np.full(count, -1, np.int64),
    )
    if level == "node":
        person = found[("person", None, None)]
        home = dict(zip(person.column("id").to_pylist(), person.column("graph").to_pylist()))
        times = np.asarray(labels.column(MOMENT).to_numpy(), np.int64)
        homes = np.asarray([home[number] for number in place.tolist()], np.int64)
        groups = {
            (graph, moment): group
            for graph, moment, group in zip(owner.tolist(), moments.tolist(), grouping.tolist())
        }
        extra = dict(
            graph=homes,
            kind=np.full(len(numbers), LABELS, object),
            time=times,
            split=split,
            grouping=np.asarray([groups[pair] for pair in zip(homes.tolist(), times.tolist())], np.int64),
            key=-(numbers + 1),
            stated=np.zeros(len(numbers), bool),
            row=np.zeros(len(numbers), np.int64),
            holder=place,
        )
        order = np.lexsort(
            (np.r_[np.zeros(count, np.int64), np.ones(len(numbers), np.int64)], np.r_[owner, homes])
        )
        held = {name: np.r_[values, extra[name]][order] for name, values in held.items()}
        origin = np.r_[np.arange(count), count + np.arange(len(numbers))][order]
        ids = np.empty(len(origin), np.int64)
        ids[origin] = np.arange(len(origin))
        aimed = ids[count:]
    else:
        ids = np.arange(count)
        if level == "graph":
            last = {graph: row for row, graph in enumerate(owner.tolist())}
            aimed = np.asarray([last[graph] for graph in place.tolist()], np.int64)
        else:
            aimed = np.searchsorted(old, place)
        held["split"][aimed] = split
    total = len(held["graph"])
    task = dict(
        id=pa.array(np.arange(total), pa.int64()),
        graph=pa.array(held["graph"], pa.int32()),
        kind=pa.array(held["kind"].tolist(), pa.string()),
        time=pa.array(held["time"], pa.int64()),
        split=pa.array(held["split"].tolist(), pa.string()),
        weight=pa.array([None] * total, pa.float32()),
    )
    task[stated["grouping"]] = pa.array(held["grouping"], pa.int32())
    task[stated["key"]] = pa.array(held["key"], pa.int64())
    vectors = query.column("x").combine_chunks().take(pa.array(held["row"], pa.int64()))
    task.update(columns(vectors, schema.asked_columns(), layout, held["stated"]))
    if stated["signals"]:
        task.update(signal=query.column("signal"), signal_observed=query.column("signal_observed"))
    result = {("task", None, None): pa.table(task)}
    names = list(stated["references"])
    for (table, source, target), links in found.items():
        if table.startswith(QUERY) and table[len(QUERY) :] in ROLES:
            name = names[ROLES.index(table[len(QUERY) :])]
            sources = np.searchsorted(old, np.asarray(links.column("from").to_numpy(), np.int64))
            result[(TASK + name, "task", target)] = pa.table(
                {"from": pa.array(ids[sources], pa.int64()), "to": links.column("to")}
            )
    if level == "node":
        chosen = np.flatnonzero(held["holder"] >= 0)
        result[(TASK + synthetic.HOLDER, "task", "person")] = pa.table(
            {"from": pa.array(chosen, pa.int64()), "to": pa.array(held["holder"][chosen], pa.int64())}
        )
    target = dict(
        id=labels.column("id"), y=labels.column("y"), known=pa.array([None] * len(numbers), pa.int64())
    )
    if level == "path":
        target.update(tau=labels.column("tau"), series=labels.column("series"))
    result[(TARGET, None, None)] = pa.table(target)
    result[(LINK, TARGET, "task")] = pa.table(
        {"from": labels.column("id"), "to": pa.array(aimed, pa.int64())}
    )
    LOG.debug(
        "asked exit level=%s tasks=%d graphs=%d seconds=%.4f",
        level,
        total,
        len(graphs),
        time.perf_counter() - started,
    )
    return result


def tables(folder, schema, level):
    started = time.perf_counter()
    folder, layout = Path(folder), list(schema.layout())
    found = {}
    for entry in read_json(folder / "stage.json"):
        key = (entry["table"], entry["source"], entry["target"])
        found[key] = pa.concat_tables([pq.read_table(folder / name) for name in entry["files"]])
    graphs = found[("graph", None, None)]
    owner = {}
    for kind in schema.nodes:
        held = found[(kind, None, None)]
        owner[kind] = dict(zip(held.column("id").to_pylist(), held.column("graph").to_pylist()))
    query = found[("query", None, None)]
    snapshots, counter, parts = restated(schema, found, layout), synthetic.Counter(), {}
    for graph in graphs.column("id").to_pylist():
        moments = np.sort(
            np.asarray(query.column(MOMENT))[np.asarray(query.column("graph")) == graph].astype(np.int64)
        )
        own = {
            key: held.filter(
                pa.array(
                    [owner[key[1]][number] == graph for number in held.column("from").to_pylist()], pa.bool_()
                )
            )
            for key, held in snapshots.items()
        }
        facts = synthetic.node_facts(schema, own, moments, counter)
        facts.update(synthetic.edge_facts(schema, own, moments))
        for key, held in facts.items():
            parts.setdefault(key, []).append(held)
    result = {key: pa.concat_tables(held) for key, held in parts.items()}
    for (table, source, target), held in found.items():
        if table == "graph":
            result[(table, source, target)] = held.add_column(
                2, "unit", pa.array([UNIT] * held.num_rows, pa.string())
            )
        elif table in schema.nodes:
            result[(table, source, target)] = held
    result.update(asked(schema, LEVELS[level], found, layout))
    LOG.debug("tables exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def spec(held):
    started = time.perf_counter()
    result = dict(held, **VIEW)
    for name in KINDS:
        if name in result:
            result[name] = result[name].replace("query", "task")
    LOG.debug("spec exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def tasks(arrays, level):
    started = time.perf_counter()
    result = dict(arrays)
    held = {name: result.pop(QUERY + name) for name in ROLES + ("key", "category", "x", "signals")}
    labels = {name: result.pop("label_" + name) for name in ("entity", "step", "row", "id")}
    steps, marks = len(held["key"]), len(labels["row"])
    references = np.stack([held[role] for role in ROLES], 1)
    if LEVELS[level] == "node":
        order = np.lexsort(
            (
                np.r_[np.zeros(steps, np.int64), np.ones(marks, np.int64)],
                np.r_[np.arange(steps), labels["step"]],
            )
        )
        place = np.empty(len(order), np.int64)
        place[order] = np.arange(len(order))
        result.update(
            task_step=np.r_[np.arange(steps), labels["step"]][order].astype(np.int32),
            task_key=np.r_[held["key"], -(labels["id"] + 1)][order],
            task_kind=np.r_[held["category"], np.full(marks, 3)][order].astype(np.int32),
            task_x=np.concatenate((held["x"], np.zeros((marks, held["x"].shape[1]), held["x"].dtype)))[order],
            task_signals=np.zeros((steps + marks, 0, 2), np.float32),
            label_task=place[steps:].astype(np.int32),
        )
        owner = np.r_[np.repeat(place[:steps], 3), place[steps:]]
        name = np.r_[np.tile(np.arange(3), steps), np.full(marks, 3)]
        source = np.r_[references.reshape(-1), labels["entity"]]
        order = np.lexsort((name, owner))
        owner, name, source = owner[order], name[order], source[order]
    else:
        result.update(
            task_step=np.arange(steps, dtype=np.int32),
            task_key=held["key"],
            task_kind=held["category"].astype(np.int32),
            task_x=held["x"],
            task_signals=held["signals"],
            label_task=labels["step"].astype(np.int32),
        )
        owner, name = np.repeat(np.arange(steps), 3), np.tile(np.arange(3), steps)
        source = references.reshape(-1)
    result.update(
        label_row=labels["row"],
        member_task=owner.astype(np.int32),
        member_name=name.astype(np.int16),
        member_place=np.zeros(len(owner), np.int32),
        member_source=source.astype(np.int32),
        member_target=np.full(len(owner), -1, np.int32),
        member_code=np.full(len(owner), -1, np.int16),
    )
    LOG.debug(
        "tasks exit level=%s tasks=%d seconds=%.4f",
        level,
        len(result["task_key"]),
        time.perf_counter() - started,
    )
    return result


def renamed(arrays, level):
    started = time.perf_counter()
    result = dict(arrays)
    if JOINED in result:
        result["step_graph"] = result.pop(JOINED)
    if all(QUERY + role in result for role in ROLES):
        result = tasks(result, level)
    result = {DERIVED[name] if name in DERIVED else name: values for name, values in result.items()}
    LOG.debug("renamed exit arrays=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def content(arrays):
    result = dict(arrays)
    for reference, table, name in (
        ("obs_ref", "x_table", "obs_x"),
        ("fact_ref", "value_table", "fact_values"),
        ("x_ref", "x_table", "x"),
    ):
        if reference in result and table in result:
            rows, values = np.asarray(result.pop(reference)), np.asarray(result[table])
            found = np.full((len(rows),) + values.shape[1:], np.nan, np.float32)
            found[rows >= 0] = values[rows[rows >= 0]]
            result[name] = found
    for table in ("x_table", "value_table"):
        result.pop(table, None)
    return result


def differing(left, right):
    started = time.perf_counter()
    left, right = content(left), content(right)
    found = sorted(set(left) ^ set(right))
    for name in sorted(set(left) & set(right)):
        one, other = np.asarray(left[name]), np.asarray(right[name])
        if (
            one.dtype != other.dtype
            or one.shape != other.shape
            or not np.array_equal(one, other, equal_nan=one.dtype.kind == "f")
        ):
            found.append(name)
    LOG.debug("differing exit arrays=%d seconds=%.4f", len(left), time.perf_counter() - started)
    return found


def arrays(path, level):
    started = time.perf_counter()
    graphs = {}
    with np.load(path, allow_pickle=False) as held:
        for key in held.files:
            graph, name = key.split("/", 1)
            graphs.setdefault(graph, {})[name] = held[key]
    result = {graph: renamed(found, level) for graph, found in graphs.items()}
    LOG.debug("arrays exit path=%s graphs=%d seconds=%.4f", path, len(result), time.perf_counter() - started)
    return result


def joined(path, level, members):
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as held:
        found = {name: held[name] for name in held.files}
    steps = np.bincount(found[JOINED])
    if len(members) != len(steps) or LEVELS[level] == "node":
        raise ValueError("the joined input of the fixture holds one task at every step of every graph")
    result = renamed(found, level)
    LOG.debug("joined exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return result


def truth(bank, level):
    started = time.perf_counter()
    bank, result = Path(bank), {}
    manifest = read_json(bank / "manifest.json")
    if manifest["version"] != VERSION:
        raise ValueError("the bank %s holds version %s" % (bank, manifest["version"]))
    for graph in manifest["graphs"]:
        folder = bank / "truth" / ("graph_%04d" % graph["index"])
        meta = read_json(folder / "graph.json")
        found = {path.stem: np.load(path, allow_pickle=False) for path in sorted(folder.glob("*.npy"))}
        result[meta["name"]] = dict(keys=meta["keys"], steps=meta["steps"], arrays=renamed(found, level))
    LOG.debug("truth exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def supervision(bank, schema, level, staged):
    started = time.perf_counter()
    held = read_json(Path(bank) / "supervision.json")
    if held["split"][UNSPLIT]:
        raise ValueError("the bank %s holds rows of a split that version 8 does not state" % bank)
    task, link = staged[("task", None, None)], staged[(LINK, TARGET, "task")]
    owner = np.asarray(link.column("to").to_numpy(), np.int64)
    kinds = np.asarray(task.column("kind").to_pylist(), object)[owner]
    keys = np.asarray(task.column(schema.tasks["key"]).to_numpy(), np.int64)[owner]
    groups = np.asarray(task.column(schema.tasks["grouping"]).to_numpy(), np.int64)[owner]
    names = dict({MOMENT: "time"}, **{name: schema.tasks[part] for name, part in ROW_KEYS.items()})
    rows = []
    for place, row in enumerate(held["rows"]):
        found = {"graph": row["graph"], "kind": kinds[place], schema.tasks["key"]: int(keys[place])}
        found["time"] = row[MOMENT]
        found[schema.tasks["grouping"]] = int(groups[place])
        for name, value in row.items():
            if name in names and found[names[name]] != value:
                raise ValueError("the row %d of the bank %s differs from its task" % (place, bank))
            if name not in names and name not in ("graph", "node"):
                found[name] = value
        rows.append(found)
    if all(name in held["split"] for name in SCALING):
        mean, sd = (held["split"][name] for name in SCALING)
        stored = [dict(name=LABELS, mean=mean, sd=sd)]
    else:
        stored = held["split"]["series"]
    indices = dict(train=held["split"]["train_indices"], validation=held["split"]["validation_indices"])
    series = []
    for entry in stored:
        values = np.asarray(
            [
                rows[place]["y"]
                for place in indices["train"]
                if "series" not in rows[place] or rows[place]["series"] == entry["name"]
            ],
            np.float64,
        )
        found = dict(name=entry["name"], mean=float(values.mean()), sd=float(values.std(ddof=1)))
        if not np.allclose([found["mean"], found["sd"]], [entry["mean"], entry["sd"]], rtol=1e-6, atol=0):
            raise ValueError("the series %s of the bank %s is scaled by other rows" % (entry["name"], bank))
        series.append(found)
    LOG.debug("supervision exit rows=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return dict(rows=rows, split=dict(indices=indices, series=series))
