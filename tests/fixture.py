import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import read_json
from graphspec.schema import EDGE, OBSERVED, QUERY

LOG = logging.getLogger(__name__)
VERSION = 7
ROLES = ("actor", "subject", "context")
JOINED = "step_campaign"
SCALING = ("reward_mean", "reward_sd")
UNSPLIT = "spread_indices"
ROW_KEYS = dict(decision="key", turn="grouping")
QUERY_COLUMNS = dict(key="key", period="grouping")


def configurations(place):
    started = time.perf_counter()
    place = Path(place)
    if not (place / "fixture.json").exists():
        raise FileNotFoundError("the fixture of the ground is absent: %s" % place)
    held = read_json(place / "fixture.json")
    if held["truth_version"] != VERSION:
        raise ValueError(
            "the fixture holds version %s; its reader reads %d" % (held["truth_version"], VERSION)
        )
    LOG.debug("configurations exit seconds=%.4f", time.perf_counter() - started)
    return held["configurations"], held["joined"]


def columns(vectors, names, layout):
    values = np.asarray(vectors.flatten().to_numpy(), np.float32).reshape(len(vectors), 2 * len(layout))
    return {
        name: pa.array(
            values[:, layout.index(name)], pa.float32(), mask=values[:, len(layout) + layout.index(name)] == 0
        )
        for name in names
    }


def tables(folder, schema):
    started = time.perf_counter()
    folder, layout = Path(folder), list(schema.layout())
    found = {}
    for entry in read_json(folder / "stage.json"):
        key = (entry["table"], entry["source"], entry["target"])
        found[key] = pa.concat_tables([pq.read_table(folder / name) for name in entry["files"]])
    observation = found[("observation", None, None)]
    vectors = observation.column("x").combine_chunks()
    place = {number: row for row, number in enumerate(observation.column("id").to_pylist())}
    result = {}
    for (table, source, target), held in found.items():
        if table == "observation":
            result[(table, source, target)] = held.drop_columns(["x"])
        elif table == "observed":
            rows = pa.array([place[number] for number in held.column("to").to_pylist()], pa.int64())
            for name, values in columns(vectors.take(rows), schema.nodes[source], layout).items():
                held = held.append_column(name, values)
            result[(OBSERVED + source, source, target)] = held
        elif table == "query":
            asked = columns(held.column("x").combine_chunks(), schema.queries["columns"], layout)
            position = held.schema.get_field_index("x")
            held = held.remove_column(position)
            for offset, (name, values) in enumerate(asked.items()):
                held = held.add_column(position + offset, name, values)
            result[(table, source, target)] = held.rename_columns(
                [
                    schema.queries[QUERY_COLUMNS[name]] if name in QUERY_COLUMNS else name
                    for name in held.column_names
                ]
            )
        elif table.startswith(QUERY) and table[len(QUERY) :] in ROLES:
            name = list(schema.queries["references"])[ROLES.index(table[len(QUERY) :])]
            result[(QUERY + name, source, target)] = held
        elif table.startswith(EDGE):
            for name in schema.edges[table[len(EDGE) :]]["columns"]:
                values = np.asarray(held.column(name).to_numpy(), np.float32)
                held = held.set_column(
                    held.schema.get_field_index(name),
                    name,
                    pa.array(values, pa.float32(), mask=np.isnan(values)),
                )
            result[(table, source, target)] = held
        else:
            result[(table, source, target)] = held
    LOG.debug("tables exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def renamed(arrays):
    result = dict(arrays)
    if JOINED in result:
        result["step_graph"] = result.pop(JOINED)
    roles = [QUERY + role for role in ROLES]
    if all(role in result for role in roles):
        result["query_references"] = np.stack([result.pop(role) for role in roles], 1)
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


def arrays(path):
    started = time.perf_counter()
    graphs = {}
    with np.load(path, allow_pickle=False) as held:
        for key in held.files:
            graph, name = key.split("/", 1)
            graphs.setdefault(graph, {})[name] = held[key]
    result = {graph: renamed(found) for graph, found in graphs.items()}
    LOG.debug("arrays exit path=%s graphs=%d seconds=%.4f", path, len(result), time.perf_counter() - started)
    return result


def joined(path):
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as held:
        result = renamed({name: held[name] for name in held.files})
    LOG.debug("joined exit path=%s seconds=%.4f", path, time.perf_counter() - started)
    return result


def truth(bank):
    started = time.perf_counter()
    bank, result = Path(bank), {}
    manifest = read_json(bank / "manifest.json")
    if manifest["version"] != VERSION:
        raise ValueError("the bank %s holds version %s" % (bank, manifest["version"]))
    for graph in manifest["graphs"]:
        folder = bank / "truth" / ("graph_%04d" % graph["index"])
        meta = read_json(folder / "graph.json")
        found = {path.stem: np.load(path, allow_pickle=False) for path in sorted(folder.glob("*.npy"))}
        result[meta["name"]] = dict(keys=meta["keys"], steps=meta["steps"], arrays=renamed(found))
    LOG.debug("truth exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def supervision(bank, target, schema):
    started = time.perf_counter()
    held = read_json(Path(bank) / "supervision.json")
    if held["split"][UNSPLIT]:
        raise ValueError("the bank %s holds rows of a split that version 8 does not state" % bank)
    split = {name: value for name, value in held["split"].items() if name not in SCALING + (UNSPLIT,)}
    if all(name in held["split"] for name in SCALING):
        mean, sd = (held["split"][name] for name in SCALING)
        split["series"] = [dict(name=target, mean=mean, sd=sd)]
    rows = [
        {schema.queries[ROW_KEYS[name]] if name in ROW_KEYS else name: value for name, value in row.items()}
        for row in held["rows"]
    ]
    LOG.debug("supervision exit rows=%d seconds=%.4f", len(rows), time.perf_counter() - started)
    return dict(rows=rows, split=split)
