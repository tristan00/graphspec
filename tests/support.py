import logging
import time
from collections import ChainMap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec import OPERATIONS
from graphspec.infra.files import read_json
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.pipeline import resolved
from graphspec.schema.declaration import Registry
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
    VALUE,
    VERSION,
    WITHIN,
    Encoding,
    Schema,
    inward,
    outward,
)

LOG = logging.getLogger(__name__)
SEVEN = (
    "graphspec.periods",
    "graphspec.window",
    "graphspec.nodes",
    "graphspec.derive",
    "graphspec.relations",
    "graphspec.edges",
    "graphspec.merging",
)
ADDED = (
    "graphspec.merge",
    "graphspec.difference",
    "graphspec.nearby",
    "graphspec.consecutive",
    "graphspec.neighbourhood",
)
KEYS = ("id", "from", "to", "since", "seq", "present")
VALUES = dict(value=dict(type="number", encoder="value"))
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
        name: dict(
            graph=dict(unit=unit),
            nodes={},
            observations={},
            edges={},
            sets={},
            held={},
            tasks=None,
            targets=None,
        )
        for name, unit in zip(graphs.column("name").to_pylist(), units)
    }
    names = dict(zip(graphs.column("id").to_pylist(), graphs.column("name").to_pylist()))
    for column in schema.graph:
        for name, value in zip(graphs.column("name").to_pylist(), graphs.column(column).to_pylist()):
            result[name]["graph"][column] = value
    owner, key, entity = {}, {}, {}
    for kind in schema.entities():
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

    for kind, stated_columns in schema.entities().items():
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
    for kind, entry in schema.sets.items():
        columns = ["set", "member", "role"] + [VALUE + name for name in entry["members"]]
        for name in result:
            result[name]["sets"][kind] = {column: [] for column in columns}
        for member in entry["kinds"]:
            held = tables[(MEMBER + kind, kind, member)]
            for row in zip(
                *(held.column(name).to_pylist() for name in ["from", "to", "role"] + list(entry["members"]))
            ):
                found = result[owner[(kind, row[0])]]["sets"][kind]
                for column, value in zip(columns, (key[(kind, row[0])], key[(member, row[1])]) + row[2:]):
                    found[column].append(value)
    for kind, entry in schema.holds.items():
        nodes = tables[(INNER + kind, None, None)]
        edges = tables[(WITHIN + kind, INNER + kind, INNER + kind)]
        inner = ["owner", "kind", "key"] + [VALUE + name for name in inward(entry)]
        links = ["owner", "source", "target", "relation"] + [VALUE + name for name in outward(entry)]
        for name in result:
            result[name]["held"][kind] = dict(
                nodes={column: [] for column in inner}, edges={column: [] for column in links}
            )
        within = {}
        for row in zip(
            *(
                nodes.column(name).to_pylist()
                for name in ["id", "graph", "owner", "kind", "key"] + list(inward(entry))
            )
        ):
            within[row[0]] = (names[row[1]], row[2], row[4])
            for column, value in zip(inner, row[2:]):
                result[names[row[1]]]["held"][kind]["nodes"][column].append(value)
        for row in zip(
            *(edges.column(name).to_pylist() for name in ["from", "to", "relation"] + list(outward(entry)))
        ):
            graph, held_owner, source = within[row[0]]
            for column, value in zip(links, (held_owner, source, within[row[1]][2]) + row[2:]):
                result[graph]["held"][kind]["edges"][column].append(value)
    asked, held = schema.tasks, tables[("task", None, None)]
    count = held.num_rows
    place = {number: row for row, number in enumerate(held.column("id").to_pylist())}
    kinds = held.column("kind").to_pylist()
    references = {}
    for name, entry in asked["references"].items():
        sort = SORTS.named(entry["sort"])
        references[name] = [[] if name in asked["kinds"][kind]["references"] else None for kind in kinds]
        if sort.columns() is None:
            continue
        for kind in entry["kinds"]:
            links = tables[(TASK + name, "task", kind)]
            stored = {
                column: links.column(column).to_pylist()
                if column in links.column_names
                else [None] * links.num_rows
                for column in ("relation", "to_kind", "to_key", "place")
            }
            for row, (source, target) in enumerate(
                zip(links.column("from").to_pylist(), links.column("to").to_pylist())
            ):
                first, relation = (kind, target), stored["relation"][row]
                second = None
                if relation is not None:
                    number = {
                        label: found
                        for (other, found), label in key.items()
                        if other == stored["to_kind"][row] and owner[(other, found)] == owner[first]
                    }[stored["to_key"][row]]
                    second = (stored["to_kind"][row], number)
                    table = [
                        held_table
                        for held_table, entry_of in schema.edges.items()
                        if relation == held_table or entry_of["column"] is not None
                    ][0]
                    if not schema.edges[table]["directed"] and first[0] == second[0]:
                        first, second = sorted((first, second), key=lambda end: entity[end])
                references[name][place[source]].append(
                    (
                        stored["place"][row],
                        key[first],
                        relation,
                        None if second is None else key[second],
                    )
                )
    for name, entry in asked["references"].items():
        ordered = SORTS.named(entry["sort"]).ordered
        references[name] = [
            None
            if members is None
            else [member[1:] for member in sorted(members, key=lambda member: member[0])]
            if ordered
            else sorted((member[1:] for member in members), key=repr)
            for members in references[name]
        ]
    columns = dict(
        time=held.column("time").to_pylist() if "time" in held.column_names else [None] * count,
        observer=held.column("observer").to_pylist() if "observer" in held.column_names else [None] * count,
        key=held.column(asked["key"]).to_pylist(),
        kind=kinds,
        split=held.column("split").to_pylist(),
        weight=held.column("weight").to_pylist(),
    )
    columns.update(values(held, schema.asked_columns()))
    if asked["grouping"] is not None:
        columns["grouping"] = held.column(asked["grouping"]).to_pylist()
    owners = [names[graph] for graph in held.column("graph").to_pylist()]
    found, named = split(owners, columns), split(owners, references)
    aimed, link = tables[(TARGET, None, None)], tables[(LINK, TARGET, "task")]
    belongs = dict(zip(link.column("from").to_pylist(), link.column("to").to_pylist()))
    known = aimed.column("known").to_pylist() if "known" in aimed.column_names else [None] * aimed.num_rows
    first = {}
    for row, graph in enumerate(owners):
        first.setdefault(graph, row)
    for graph in found:
        moments = [0 if value is None else value for value in found[graph]["time"]]
        order = np.argsort(moments, kind="stable").tolist()
        result[graph]["tasks"] = dict(
            {name: [held[row] for row in order] for name, held in found[graph].items()},
            references={name: [held[row] for row in order] for name, held in named[graph].items()},
        )
        position = {first[graph] + row: at for at, row in enumerate(order)}
        rows = sorted(
            (position[place[belongs[number]]], row, y, moment)
            for row, (number, y, moment) in enumerate(
                zip(aimed.column("id").to_pylist(), aimed.column("y").to_pylist(), known)
            )
            if owners[place[belongs[number]]] == graph
        )
        result[graph]["targets"] = dict(
            task=[row[0] for row in rows],
            row=[row[1] for row in rows],
            y=[row[2] for row in rows],
            known=[row[3] for row in rows],
        )
    LOG.debug("written exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def asked(nodes):
    started = time.perf_counter()
    result = dict(
        key="key",
        grouping=None,
        signals=0,
        references=dict(holder=dict(sort="node", kinds=list(nodes))),
        kinds=dict(
            outcome=dict(
                references=["holder"],
                columns=[],
                target=dict(kind="number", series="outcome", about="holder"),
                weight=False,
                absent="stop",
                event=None,
                matures=None,
            )
        ),
    )
    LOG.debug("asked exit seconds=%.4f", time.perf_counter() - started)
    return result


def stated(nodes, edges, columns, tasks):
    started = time.perf_counter()
    result = Schema(
        dict(
            version=VERSION,
            time=dict(model="clock", units=["step"], ties="sequence"),
            observers=False,
            columns=columns,
            graph=dict(columns=[]),
            nodes={kind: dict(columns=held, holds=None) for kind, held in nodes.items()},
            sets={},
            edges={
                table: dict(
                    endpoints=[list(pair) for pair in entry["endpoints"]],
                    directed=entry["directed"],
                    columns=entry["columns"],
                    relation=DECLARED,
                    about=None,
                )
                for table, entry in edges.items()
            },
            tasks=tasks,
        )
    )
    LOG.debug("stated exit seconds=%.4f", time.perf_counter() - started)
    return result


def registry(name, nodes, edges):
    started = time.perf_counter()
    base = [edge for edge in edges if edge.origin[0] == "base"]
    schema = stated(
        {kind: ["value"] for kind in nodes},
        {
            edge.name: dict(endpoints=edge.pairs, directed=not edge.symmetric, columns=list(edge.attributes))
            for edge in base
        },
        VALUES,
        asked(nodes),
    )
    encoding = Encoding(schema, {}, dict(value={}), {edge.name: [edge.name] for edge in base})
    extras = dict(
        target="number",
        series=["outcome"],
        signals=0,
        base=[edge.name for edge in base],
        events=False,
        several=False,
        untasked=[],
    )
    result = Registry(name, encoding, edges, [], extras)
    LOG.debug("registry exit seconds=%.4f", time.perf_counter() - started)
    return result


def normal(tasks, schema):
    started = time.perf_counter()
    references = {
        name: [
            members
            if members is None or SORTS.named(schema.tasks["references"][name]["sort"]).ordered
            else sorted(members, key=repr)
            for members in held
        ]
        for name, held in tasks["references"].items()
    }
    LOG.debug("normal exit references=%d seconds=%.4f", len(references), time.perf_counter() - started)
    return dict(tasks, references=references)


def several(registry):
    started = time.perf_counter()
    result = {"time/several": "last"} if registry.extras["several"] else {}
    LOG.debug("several exit stated=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def shown(opened, index):
    from conftest import SPLITS
    from graphspec.views.graph import Graph

    started = time.perf_counter()
    cfg = configuration(opened.registry, dict(view="snapshot"), several(opened.registry), 5, SPLITS)
    result = Graph.load(opened.view(index, cfg))
    LOG.debug("shown exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
    return result


def beside(opened, index, name):
    from conftest import SPLITS
    from graphspec.views.graph import Graph

    started = time.perf_counter()
    cfg = configuration(opened.registry, dict(view="snapshot"), several(opened.registry), 5, SPLITS)
    result = Graph.load(opened.viewed(cfg).second(opened, index, name)).dense()
    LOG.debug("beside exit graph=%s index=%d seconds=%.4f", name, index, time.perf_counter() - started)
    return result


def fitted(opened):
    from conftest import SPLITS

    started = time.perf_counter()
    cfg = configuration(opened.registry, dict(view="snapshot"), several(opened.registry), 5, SPLITS)
    result = opened.fitted(cfg)
    LOG.debug("fitted exit features=%d seconds=%.4f", result.features, time.perf_counter() - started)
    return result


def pipelined(registry, values, names=SEVEN, labels=None, graphs=None):
    started = time.perf_counter()
    stated = [
        dict(
            operation=name,
            **({} if labels is None or labels[place] is None else dict(label=labels[place])),
            **({} if graphs is None or name not in graphs else dict(graph=graphs[name])),
        )
        for place, name in enumerate(names)
    ]
    found = pipeline.checked(stated, pipeline.TEMPLATE)
    reads, remaining, held = pipeline.readers(found), dict(values), {}
    for name in pipeline.drawn(found, reads):
        (operation,) = [operation for key, operation, _, _ in found if key == name]
        own = {}
        view = ChainMap(own, *[held[read] for read in reads[name]])
        for parameter in operation.stated(registry):
            if parameter.active(view):
                own[parameter.name] = remaining.pop(parameter.name) if parameter.name in remaining else parameter.default
        held[name] = own
    entries = [dict(entry, parameters=held[name]) for name, _, _, entry in found]
    LOG.debug("pipelined exit entries=%d seconds=%.4f", len(entries), time.perf_counter() - started)
    return entries, remaining


def configuration(registry, values, graph_values, seed, splits, names=SEVEN):
    from tuning.families.model_spec import configuration as configure
    from tuning.families.model_spec import configured, groups

    started = time.perf_counter()
    model = configured(registry, values)
    pipeline, rest = pipelined(registry, graph_values, names)
    graph = specs.configured(specs.settings(registry, groups(model)), rest)
    result = configure(registry, model, pipeline, graph, seed, splits)
    LOG.debug("configuration exit entries=%d seconds=%.4f", len(pipeline), time.perf_counter() - started)
    return result


def template(names=SEVEN):
    return [dict(operation=name) for name in names]


def parameters(registry, groups, names=SEVEN):
    started = time.perf_counter()
    result = specs.settings(registry, groups) + tuple(
        parameter for name in names for parameter in OPERATIONS.named(name).stated(registry)
    )
    LOG.debug("parameters exit parameters=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def steps(registry, values, groups=(), names=SEVEN):
    started = time.perf_counter()
    pipeline, rest = pipelined(registry, values, names)
    result = resolved(registry, pipeline), specs.configured(specs.settings(registry, groups), rest)
    LOG.debug("steps exit steps=%d seconds=%.4f", len(result[0]), time.perf_counter() - started)
    return result
