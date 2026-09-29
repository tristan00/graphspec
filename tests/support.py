import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec.infra.files import read_json
from graphspec.registry import Registry
from graphspec.schema import DECLARED, EDGE, OBSERVED, QUERY, VALUE, VERSION, Encoding, Schema

LOG = logging.getLogger(__name__)
KEYS = ("id", "from", "to", "since", "seq", "present")
STAMPS = ("since", "until", "known", "replaced", "observers")


def staged(folder):
    started = time.perf_counter()
    folder = Path(folder)
    result = {
        (entry["table"], entry["source"], entry["target"]): pa.concat_tables(
            [pq.read_table(folder / name) for name in entry["files"]]
        )
        for entry in read_json(folder / "stage.json")
    }
    LOG.debug("staged exit tables=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def sorted_by_key(table):
    return table.sort_by([(name, "ascending") for name in KEYS if name in table.column_names])


def rows(columns):
    return sorted(zip(*columns.values()), key=lambda row: tuple(map(repr, row)))


def alike(left, right):
    if isinstance(left, dict) and isinstance(right, dict) and left and right:
        held = all(isinstance(value, list) for value in list(left.values()) + list(right.values()))
        if held:
            return list(left) == list(right) and rows(left) == rows(right)
        return left.keys() == right.keys() and all(alike(left[key], right[key]) for key in left)
    return left == right


def values(table, names):
    return {VALUE + name: table.column(name).to_pylist() for name in names}


def timed(table, schema):
    count = table.num_rows
    found = {
        name: table.column(name).to_pylist() if name in table.column_names else [None] * count
        for name in STAMPS
    }
    found["known"] = [
        None if known == since else known for known, since in zip(found["known"], found["since"])
    ]
    found["observers"] = [None if names is None else sorted(names) for names in found["observers"]]
    if "seq" in schema.stamps():
        found["seq"] = table.column("seq").to_pylist()
    return found


def written(folder, schema):
    started = time.perf_counter()
    tables = staged(folder)
    graphs = tables[("graph", None, None)]
    units = graphs.column("unit").to_pylist() if "unit" in graphs.column_names else [None] * graphs.num_rows
    result = {
        name: dict(graph=dict(unit=unit), nodes={}, observations={}, edges={}, queries=None)
        for name, unit in zip(graphs.column("name").to_pylist(), units)
    }
    names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
    for column in schema.graph:
        for name, value in zip(graphs.column("name").to_pylist(), graphs.column(column).to_pylist()):
            result[name]["graph"][column] = value
    owner, key, entity = {}, {}, {}
    for kind in schema.nodes:
        held = tables[(kind, None, None)]
        for name in result:
            result[name]["nodes"][kind] = []
        for number, graph, label, member in zip(
            *(held.column(name).to_pylist() for name in ("id", "graph", "key", "entity"))
        ):
            owner[(kind, number)], key[(kind, number)], entity[(kind, number)] = names[graph], label, member
            result[names[graph]]["nodes"][kind].append(label)
    observation = tables[("observation", None, None)]
    place = {number: row for row, number in enumerate(observation.column("id").to_pylist())}

    def split(graph_of, columns):
        found = {}
        for position, graph in enumerate(graph_of):
            for name, held in columns.items():
                found.setdefault(graph, {name: [] for name in columns})[name].append(held[position])
        return found

    for kind, stated_columns in schema.nodes.items():
        held = tables[(OBSERVED + kind, kind, "observation")]
        sources, targets = held.column("from").to_pylist(), held.column("to").to_pylist()
        moments = observation.take(pa.array([place[number] for number in targets], pa.int64()))
        columns = dict(
            key=[key[(kind, number)] for number in sources],
            **timed(moments, schema),
            **values(held, stated_columns),
        )
        for name in result:
            result[name]["observations"][kind] = {column: [] for column in columns}
        for graph, found in split([owner[(kind, number)] for number in sources], columns).items():
            result[graph]["observations"][kind] = found
    for table, entry in schema.edges.items():
        parts = {}
        for first, second in entry["endpoints"]:
            held = tables[(EDGE + table, first, second)]
            sources, targets = held.column("from").to_pylist(), held.column("to").to_pylist()
            ends = [((first, source), (second, target)) for source, target in zip(sources, targets)]
            if not entry["directed"] and first == second:
                ends = [tuple(sorted(pair, key=lambda end: entity[end])) for pair in ends]
            columns = dict(
                source=[key[source] for source, _ in ends],
                target=[key[target] for _, target in ends],
                present=held.column("present").to_pylist(),
                relation=[table] * held.num_rows
                if entry["column"] is None
                else held.column(entry["column"]).to_pylist(),
                **timed(held, schema),
                **values(held, entry["columns"]),
            )
            for name in result:
                parts.setdefault(name, {column: [] for column in columns})
            for graph, found in split([owner[source] for source, _ in ends], columns).items():
                for column, more in found.items():
                    parts[graph][column] += more
        for name in result:
            result[name]["edges"][table] = parts[name]
    if schema.queries is not None:
        held = tables[("query", None, None)]
        place = {number: row for row, number in enumerate(held.column("id").to_pylist())}
        references = {}
        for name, kinds in schema.queries["references"].items():
            references[name] = [None] * held.num_rows
            for kind in kinds:
                links = tables[(QUERY + name, "query", kind)]
                for source, target in zip(links.column("from").to_pylist(), links.column("to").to_pylist()):
                    references[name][place[source]] = key[(kind, target)]
        asked, count = schema.queries, held.num_rows
        columns = dict(
            time=held.column("time").to_pylist() if "time" in held.column_names else [None] * count,
            observer=held.column("observer").to_pylist()
            if "observer" in held.column_names
            else [None] * count,
            key=held.column(asked["key"]).to_pylist(),
            category=held.column("category").to_pylist(),
        )
        columns.update(values(held, schema.queries["columns"]))
        if asked["grouping"] is not None:
            columns["grouping"] = held.column(asked["grouping"]).to_pylist()
        owners = [names[graph] for graph in held.column("graph").to_pylist()]
        found, named = split(owners, columns), split(owners, references)
        for graph in found:
            moments = [0 if value is None else value for value in found[graph]["time"]]
            order = np.argsort(moments, kind="stable").tolist()
            result[graph]["queries"] = dict(
                {name: [held[row] for row in order] for name, held in found[graph].items()},
                references={name: [held[row] for row in order] for name, held in named[graph].items()},
            )
    LOG.debug("written exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def stated(nodes, edges, columns=None, queries=None):
    started = time.perf_counter()
    columns = dict(value=dict(type="number", encoder="value")) if columns is None else columns
    result = Schema(
        dict(
            version=VERSION,
            time=dict(model="clock", units=["step"], ties="sequence"),
            observers=False,
            columns=columns,
            graph=dict(columns=[]),
            nodes={kind: dict(columns=held) for kind, held in nodes.items()},
            edges={
                table: dict(
                    endpoints=[list(pair) for pair in entry["endpoints"]],
                    directed=entry["directed"],
                    columns=entry["columns"],
                    relation=DECLARED,
                )
                for table, entry in edges.items()
            },
            queries=queries,
        )
    )
    LOG.debug("stated exit seconds=%.4f", time.perf_counter() - started)
    return result


def registry(name, nodes, edges, extras):
    started = time.perf_counter()
    base = [edge for edge in edges if edge.origin[0] == "base"]
    schema = stated(
        {kind: ["value"] for kind in nodes},
        {
            edge.name: dict(endpoints=edge.pairs, directed=not edge.symmetric, columns=list(edge.attributes))
            for edge in base
        },
    )
    encoding = Encoding(schema, {}, dict(value={}), {edge.name: [edge.name] for edge in base})
    result = Registry(name, encoding, edges, [], extras)
    LOG.debug("registry exit seconds=%.4f", time.perf_counter() - started)
    return result


def shown(opened, index):
    from graphspec.truth import Graph
    from tuning.model_spec import configuration

    started = time.perf_counter()
    cfg = configuration(opened.registry, dict(view="snapshot"), {}, 5)
    result = Graph.load(opened.view(index, cfg))
    LOG.debug("shown exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
    return result
