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
KEYS = ("id", "from", "to", "step", "seq", "present")


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


def written(folder, schema):
    started = time.perf_counter()
    tables = staged(folder)
    graphs = tables[("graph", None, None)]
    result = {
        name: dict(graph={}, nodes={}, observations={}, edges={}, queries=None)
        for name in graphs.column("name").to_pylist()
    }
    names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
    for column in schema.graph:
        for name, value in zip(graphs.column("name").to_pylist(), graphs.column(column).to_pylist()):
            result[name]["graph"][column] = value
    owner, key, entity = {}, {}, {}
    for kind in schema.nodes:
        held = tables[(kind, None, None)]
        for number, graph, label, member in zip(
            *(held.column(name).to_pylist() for name in ("id", "graph", "key", "entity"))
        ):
            owner[(kind, number)], key[(kind, number)], entity[(kind, number)] = names[graph], label, member
            result[names[graph]]["nodes"].setdefault(kind, []).append(label)
    observation = tables[("observation", None, None)]
    moment = dict(zip(observation.column("id").to_pylist(), observation.column("step").to_pylist()))

    def split(table, graph_of, columns):
        found = {}
        for position, graph in enumerate(graph_of):
            for name, held in columns.items():
                found.setdefault(graph, {name: [] for name in columns})[name].append(held[position])
        return found

    for kind, stated_columns in schema.nodes.items():
        held = tables[(OBSERVED + kind, kind, "observation")]
        sources, targets = held.column("from").to_pylist(), held.column("to").to_pylist()
        columns = dict(
            key=[key[(kind, number)] for number in sources],
            step=[moment[number] for number in targets],
            **values(held, stated_columns),
        )
        for name in result:
            result[name]["observations"][kind] = {column: [] for column in columns}
        for graph, found in split(held, [owner[(kind, number)] for number in sources], columns).items():
            result[graph]["observations"][kind] = found
    for table, entry in schema.edges.items():
        for name in result:
            result[name]["edges"][table] = None
        for first, second in entry["endpoints"]:
            held = tables[(EDGE + table, first, second)]
            sources, targets = held.column("from").to_pylist(), held.column("to").to_pylist()
            ends = [((first, source), (second, target)) for source, target in zip(sources, targets)]
            if not entry["directed"] and first == second:
                ends = [tuple(sorted(pair, key=lambda end: entity[end])) for pair in ends]
            columns = dict(
                source=[key[source] for source, _ in ends],
                target=[key[target] for _, target in ends],
                step=held.column("step").to_pylist(),
                present=held.column("present").to_pylist(),
                relation=[table] * held.num_rows
                if entry["column"] is None
                else held.column(entry["column"]).to_pylist(),
                **values(held, entry["columns"]),
            )
            for name in result:
                result[name]["edges"][table] = {column: [] for column in columns}
            for graph, found in split(held, [owner[source] for source, _ in ends], columns).items():
                result[graph]["edges"][table] = found
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
        asked = schema.queries
        columns = dict(
            step=held.column("step").to_pylist(),
            period=held.column(asked["grouping"]).to_pylist(),
            key=held.column(asked["key"]).to_pylist(),
            category=held.column("category").to_pylist(),
        )
        columns.update(values(held, schema.queries["columns"]))
        found = split(held, [names[graph] for graph in held.column("graph").to_pylist()], columns)
        named = split(held, [names[graph] for graph in held.column("graph").to_pylist()], references)
        for graph in found:
            order = np.argsort(found[graph]["step"], kind="stable").tolist()
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
            time=dict(unit="step"),
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
