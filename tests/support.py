import logging
import time
from collections import ChainMap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from graphspec import OPERATIONS
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.pipeline import resolved
from graphspec.schema.declaration import Registry
from graphspec.schema.references import SORTS
from graphspec.schema.schema import (
    DECLARED,
    VALUE,
    VERSION,
    Encoding,
    Schema,
    inward,
    outward,
)
from graphspec.schema.targets import TARGETS

LOG = logging.getLogger(__name__)
SEED = 0x6772617068
SEVEN = (
    "graphspec.periods",
    "graphspec.window",
    "graphspec.nodes",
    "graphspec.derive",
    "graphspec.relations",
    "graphspec.edges",
    "graphspec.relation_forms",
)
ADDED = (
    "graphspec.merge",
    "graphspec.difference",
    "graphspec.nearby",
    "graphspec.consecutive",
    "graphspec.neighbourhood",
    "graphspec.measures",
    "graphspec.contract",
    "graphspec.product",
    "graphspec.negatives",
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


def stored_as(schema, name, stated):
    started = time.perf_counter()
    written = schema.kind(name).written
    if written == pa.float32():
        result = [None if value is None else float(np.float32(value)) for value in stated]
    elif written == pa.bool_():
        result = [None if value is None else bool(value) for value in stated]
    else:
        result = list(stated)
    LOG.debug("stored as exit column=%s seconds=%.6f", name, time.perf_counter() - started)
    return result


def snapshotted(rows, times, count):
    started = time.perf_counter()
    place, seen, result = {moment: at for at, moment in enumerate(times)}, {}, []
    for row in range(count):
        content = repr([(name, rows[name][row]) for name in rows if name != "at"])
        seen.setdefault((content, place[rows["at"][row]]), []).append(row)
    for (content, first), members in seen.items():
        for occurrence, row in enumerate(members):
            if occurrence < len(seen.get((content, first - 1), ())):
                continue
            last = first
            while occurrence < len(seen.get((content, last + 1), ())):
                last += 1
            result.append((row, times[first], times[last + 1] if last + 1 < len(times) else None))
    LOG.debug("snapshotted exit rows=%d facts=%d seconds=%.6f", count, len(result), time.perf_counter() - started)
    return result


def spans(statement, count):
    started = time.perf_counter()
    form, rows, parameters = statement["form"], statement["rows"], statement["parameters"]
    if form is None:
        result = [(row, None, None) for row in range(count)]
    elif form == "state":
        result = list(zip(range(count), rows["since"], rows["until"]))
    elif form == "event":
        result = [(row, at, at + 1) for row, at in enumerate(rows["at"])]
    elif form == "signal":
        since = [parameters["start"] + index * parameters["stride"] for index in rows["index"]]
        result = [(row, moment, moment + parameters["stride"]) for row, moment in enumerate(since)]
    elif form == "change":
        things = list(zip(*(rows[name] for name in parameters["keys"])))
        result = [
            (row, at, min((later for other, later in zip(things, rows["at"]) if other == thing and later > at), default=None))
            for row, (thing, at) in enumerate(zip(things, rows["at"]))
        ]
    elif form == "snapshot" and "seq" not in rows:
        result = snapshotted(rows, parameters["times"], count)
    else:
        raise ValueError("a case states facts in the form %s, which the reading of a case does not know" % form)
    LOG.debug("spans exit form=%s facts=%d seconds=%.6f", form, len(result), time.perf_counter() - started)
    return result


def laid(schema, given, columns):
    started = time.perf_counter()
    names = list(given) + list(STAMPS) + (["seq"] if "seq" in schema.stamps() else [])
    result = {name: [] for name in names + [VALUE + name for name in columns]}
    LOG.debug("laid exit columns=%d seconds=%.6f", len(result), time.perf_counter() - started)
    return result


def stated_facts(schema, statement, given, columns):
    started = time.perf_counter()
    rows = statement["rows"]
    count = len(next(iter(rows.values())))
    held = {name: list(rows[name]) if name in rows else [None] * count for name in ("known", "replaced", "observers")}
    values = {name: stored_as(schema, name, rows[name]) for name in columns}
    result = laid(schema, given, columns)
    for row, since, until in spans(statement, count):
        known, watchers = held["known"][row], held["observers"][row]
        for name, found in given.items():
            result[name].append(found[row])
        result["since"].append(since)
        result["until"].append(until)
        result["known"].append(None if known == since else known)
        result["replaced"].append(held["replaced"][row])
        result["observers"].append(None if watchers is None else sorted(watchers))
        if "seq" in result:
            result["seq"].append(rows["seq"][row])
        for name in columns:
            result[VALUE + name].append(values[name][row])
    LOG.debug("stated facts exit facts=%d seconds=%.6f", len(result["since"]), time.perf_counter() - started)
    return result


def observed(schema, graph):
    started = time.perf_counter()
    result = {}
    for kind, columns in schema.entities().items():
        result[kind] = laid(schema, ("key",), columns)
        for statement in graph["values"]:
            if statement["kind"] == kind:
                found = stated_facts(schema, statement, dict(key=list(statement["rows"]["key"])), columns)
                for name, held in found.items():
                    result[kind][name] += held
    LOG.debug("observed exit graph=%s seconds=%.6f", graph["name"], time.perf_counter() - started)
    return result


def edged(schema, graph, entity):
    started = time.perf_counter()
    result = {}
    for table, entry in schema.edges.items():
        result[table] = laid(schema, ("source", "target", "present", "relation"), entry["columns"])
        for statement in graph["edges"]:
            if statement["table"] != table:
                continue
            rows, ends = statement["rows"], tuple(statement["ends"])
            pairs = list(zip(rows["source"], rows["target"]))
            if not entry["directed"] and ends[0] == ends[1]:
                pairs = [tuple(sorted(pair, key=lambda key: entity[ends[0], key])) for pair in pairs]
            given = dict(
                source=[source for source, _ in pairs],
                target=[target for _, target in pairs],
                present=list(rows["present"]),
                relation=[table] * len(pairs) if entry["column"] is None else list(rows["relation"]),
            )
            for name, held in stated_facts(schema, statement, given, entry["columns"]).items():
                result[table][name] += held
    LOG.debug("edged exit graph=%s seconds=%.6f", graph["name"], time.perf_counter() - started)
    return result


def joined(schema, graph):
    started = time.perf_counter()
    result = {}
    for kind, entry in schema.sets.items():
        rows = graph["sets"][kind]
        result[kind] = dict(
            set=list(rows["set"]),
            member=list(rows["member"]),
            role=list(rows["role"]),
            **{VALUE + name: stored_as(schema, name, rows[name]) for name in entry["members"]},
        )
    LOG.debug("joined exit graph=%s seconds=%.6f", graph["name"], time.perf_counter() - started)
    return result


def holding(schema, graph):
    started = time.perf_counter()
    result = {}
    for kind, entry in schema.holds.items():
        nodes, edges = graph["held"][kind]["nodes"], graph["held"][kind]["edges"]
        result[kind] = dict(
            nodes=dict(
                {name: list(nodes[name]) for name in ("owner", "kind", "key")},
                **{VALUE + name: stored_as(schema, name, nodes[name]) for name in inward(entry)},
            ),
            edges=dict(
                {name: list(edges[name]) for name in ("owner", "source", "target", "relation")},
                **{VALUE + name: stored_as(schema, name, edges[name]) for name in outward(entry)},
            ),
        )
    LOG.debug("holding exit graph=%s seconds=%.6f", graph["name"], time.perf_counter() - started)
    return result


def kind_of(graph, kinds, key):
    started = time.perf_counter()
    found = [kind for kind in kinds if key in graph["nodes"][kind]]
    if len(found) != 1:
        raise ValueError("the graph %s holds the node %s under %d of the kinds %s" % (graph["name"], key, len(found), kinds))
    LOG.debug("kind of exit key=%s seconds=%.6f", key, time.perf_counter() - started)
    return found[0]


def ended(schema, graph, entity, kinds, member):
    started = time.perf_counter()
    source, relation, target = member
    result = member
    if relation is not None:
        tables = [relation] if relation in schema.edges else [name for name, entry in schema.edges.items() if entry["column"]]
        if len(tables) != 1:
            raise ValueError("the relation %s is held by the tables %s, and a case names it in one" % (relation, tables))
        first, second = kind_of(graph, kinds, source), kind_of(graph, list(schema.entities()), target)
        if not schema.edges[tables[0]]["directed"] and first == second:
            low, high = sorted((source, target), key=lambda key: entity[first, key])
            result = (low, relation, high)
    LOG.debug("ended exit relation=%s seconds=%.6f", relation, time.perf_counter() - started)
    return result


def referred(schema, graph, entity, name, stated):
    started = time.perf_counter()
    entry = schema.tasks["references"][name]
    sort = SORTS.named(entry["sort"])
    if sort.columns() is None:
        members = []
    elif sort.nodes and not sort.edges:
        members = [(key, None, None) for key in (stated if isinstance(stated, list) else [stated])]
    elif sort.edges and not sort.nodes:
        members = [tuple(member) for member in (stated if isinstance(stated, list) else [stated])]
    else:
        members = [(key, None, None) for key in stated["nodes"]] + [tuple(member) for member in stated["edges"]]
    members = [ended(schema, graph, entity, entry["kinds"], member) for member in members]
    LOG.debug("referred exit reference=%s seconds=%.6f", name, time.perf_counter() - started)
    return members if sort.ordered else sorted(members, key=repr)


def asked_of(schema, graph, entity):
    started = time.perf_counter()
    tasks, kinds = graph["tasks"], schema.tasks["kinds"]
    count = len(tasks["key"])
    moments = list(tasks["time"]) if "time" in tasks else [None] * count
    order = sorted(range(count), key=lambda at: 0 if moments[at] is None else moments[at])
    found = dict(
        time=moments,
        observer=list(tasks["observer"]) if schema.observers else [None] * count,
        key=list(tasks["key"]),
        kind=list(tasks["kind"]),
        split=list(tasks["split"]),
        weight=[None if value is None else float(np.float32(value)) for value in tasks["weight"]],
        **{VALUE + name: stored_as(schema, name, tasks[name]) for name in schema.asked_columns()},
    )
    if schema.tasks["grouping"] is not None:
        found["grouping"] = list(tasks["grouping"])
    references = {}
    for name, entry in schema.tasks["references"].items():
        given = [None] * count if SORTS.named(entry["sort"]).columns() is None else tasks["references"][name]
        references[name] = [
            None if name not in kinds[kind]["references"] else referred(schema, graph, entity, name, stated)
            for kind, stated in zip(tasks["kind"], given)
        ]
    result = {name: [held[at] for at in order] for name, held in found.items()}
    result["references"] = {name: [held[at] for at in order] for name, held in references.items()}
    LOG.debug("asked of exit graph=%s tasks=%d seconds=%.6f", graph["name"], count, time.perf_counter() - started)
    return result, order


def aimed_of(schema, graph, order, row):
    started = time.perf_counter()
    tasks, found = graph["tasks"], []
    place = {at: position for position, at in enumerate(order)}
    for at, (kind, target, known) in enumerate(zip(tasks["kind"], tasks["target"], tasks["known"])):
        if target is None:
            continue
        stated = schema.tasks["kinds"][kind]["target"]
        where = "the task %r of the graph %s" % (tasks["key"][at], graph["name"])
        for point in TARGETS.named(stated["kind"]).staged(target, stated, where):
            found.append((place[at], row, float(point["y"]), known))
            row += 1
    found.sort(key=lambda point: point[:2])
    result = {name: [point[at] for point in found] for at, name in enumerate(("task", "row", "y", "known"))}
    LOG.debug("aimed of exit graph=%s points=%d seconds=%.6f", graph["name"], len(found), time.perf_counter() - started)
    return result, row


def made(schema, graph):
    started = time.perf_counter()
    result = {kind: set() for kind in schema.entities()}
    for kind, target in zip(graph["tasks"]["kind"], graph["tasks"]["target"]):
        if kind not in schema.decided() or target is None:
            continue
        roles = {name: entry["roles"] for name, entry in schema.tasks["kinds"][kind]["target"]["actions"].items()}
        for option in target["options"]:
            for action in option["actions"]:
                for role, key in action["targets"].items():
                    declared = roles[action["kind"]][role]
                    if declared["sort"] == "new" and key not in graph["nodes"][declared["kinds"][0]]:
                        result[declared["kinds"][0]].add(key)
    LOG.debug("made exit graph=%s seconds=%.6f", graph["name"], time.perf_counter() - started)
    return {kind: sorted(keys) for kind, keys in result.items()}


@timed("support.probable")
def probable(value):
    return None if value is None else float(value)


def chosen_of(schema, graph, entity, decision, stated):
    started = time.perf_counter()
    options = []
    for option in stated["options"]:
        actions = []
        for action in option["actions"]:
            roles = decision["actions"][action["kind"]]["roles"]
            targets = {
                role: list(ended(schema, graph, entity, roles[role]["kinds"], tuple(key))) if isinstance(key, list) else key
                for role, key in action["targets"].items()
            }
            parameters = {name: stored_as(schema, name, [value])[0] for name, value in action["parameters"].items()}
            actions.append(dict(action, targets=targets, parameters=parameters))
        outcome = option["outcome"]
        valued = [outcome] if isinstance(outcome, dict) else outcome or []
        valued = [None if held is None else {name: float(value) for name, value in held.items()} for held in valued]
        origin = option["origin"]
        options.append(
            dict(
                option,
                actions=actions,
                shown=probable(option["shown"]),
                logged=probable(option["logged"]),
                origin=origin if isinstance(origin, str) else dict(origin, drawn=probable(origin["drawn"])),
                outcome=valued[0] if isinstance(outcome, dict) else valued if outcome is not None else None,
            )
        )
    result = dict(stated, options=options, concurrent=sorted(stated["concurrent"]), discount=probable(stated["discount"]))
    LOG.debug("chosen of exit options=%d seconds=%.6f", len(options), time.perf_counter() - started)
    return result


def decided_of(schema, graph, entity, order):
    started = time.perf_counter()
    tasks, kinds = graph["tasks"], schema.tasks["kinds"]
    result = [
        chosen_of(schema, graph, entity, kinds[tasks["kind"][at]]["target"], tasks["target"][at])
        if tasks["kind"][at] in schema.decided() and tasks["target"][at] is not None
        else None
        for at in order
    ]
    LOG.debug("decided of exit graph=%s tasks=%d seconds=%.4f", graph["name"], len(result), time.perf_counter() - started)
    return result


def meant(case):
    started = time.perf_counter()
    schema, result, row = Schema(case["schema"]), {}, 0
    for graph in case["graphs"]:
        listed = [(kind, key) for kind in schema.entities() for key in graph["nodes"][kind]]
        entity = {pair: at for at, pair in enumerate(listed)}
        tasks, order = asked_of(schema, graph, entity)
        targets, row = aimed_of(schema, graph, order, row)
        created = made(schema, graph)
        result[graph["name"]] = dict(
            graph=dict(
                unit=graph["unit"],
                **{name: stored_as(schema, name, [graph["columns"][name]])[0] for name in schema.graph},
            ),
            nodes={kind: list(graph["nodes"][kind]) + created[kind] for kind in schema.entities()},
            observations=observed(schema, graph),
            edges=edged(schema, graph, entity),
            sets=joined(schema, graph),
            held=holding(schema, graph),
            tasks=tasks,
            targets=targets,
            decisions=decided_of(schema, graph, entity, order),
        )
    LOG.debug("meant exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def read_back(held):
    from graphspec.facts.stored import Facts

    started = time.perf_counter()
    result = {}
    for entry in held.manifest["graphs"]:
        truth = Facts.load(held.truth(entry["index"]))
        result[truth.name] = truth.stated(held.registry, held.splits, held.manifest["observers"])
    LOG.debug("read back exit graphs=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def differing(found, wanted, schema):
    started = time.perf_counter()
    result = [part for part in ("graph", "nodes", "targets", "decisions") if found[part] != wanted[part]]
    result += [part for part in ("observations", "edges", "sets", "held") if not alike(found[part], wanted[part])]
    if normal(found["tasks"], schema) != wanted["tasks"]:
        result.append("tasks")
    LOG.debug("differing exit parts=%d seconds=%.6f", len(result), time.perf_counter() - started)
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
                target=dict(
                    kind="number",
                    series="outcome",
                    about="holder",
                    utility=dict(name="weighted", better="higher", weights=dict(outcome=1.0)),
                ),
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
    return columned(name, nodes, edges, VALUES)


def columned(name, nodes, edges, columns):
    started = time.perf_counter()
    base = [edge for edge in edges if edge.origin[0] == "base"]
    schema = stated(
        {kind: list(columns) for kind in nodes},
        {
            edge.name: dict(endpoints=edge.pairs, directed=not edge.symmetric, columns=list(edge.attributes))
            for edge in base
        },
        columns,
        asked(nodes),
    )
    encoding = Encoding(schema, {}, {column: {} for column in columns}, {edge.name: [edge.name] for edge in base})
    extras = dict(
        signals=0,
        base=[edge.name for edge in base],
        events=False,
        several=False,
        untasked=[],
        unvalued={},
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


@timed("support.entries")
def entries(names):
    return {name: names.named(name) for name in names.names()}


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


def unoffered(registry, grown):
    started = time.perf_counter()
    result = [
        name
        for name in OPERATIONS.names()
        if [parameter.declaration() for parameter in OPERATIONS.named(name).stated(grown)]
        != [parameter.declaration() for parameter in OPERATIONS.named(name).stated(registry)]
    ]
    LOG.debug("unoffered exit operations=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


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


def same(left, right):
    import torch

    started = time.perf_counter()
    if isinstance(left, torch.Tensor):
        result = (
            isinstance(right, torch.Tensor)
            and (left.dtype, left.shape) == (right.dtype, right.shape)
            and torch.equal(left.isnan(), right.isnan())
            and torch.equal(left.nan_to_num(), right.nan_to_num())
        )
    elif isinstance(left, np.ndarray):
        result = isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        result = isinstance(right, dict) and sorted(left) == sorted(right) and all(same(left[key], right[key]) for key in left)
    elif isinstance(left, (list, tuple)):
        result = type(left) is type(right) and len(left) == len(right) and all(map(same, left, right))
    elif hasattr(left, "__dict__"):
        result = type(left) is type(right) and same(vars(left), vars(right))
    else:
        result = type(left) is type(right) and left == right
    LOG.debug("same exit equal=%s seconds=%.6f", result, time.perf_counter() - started)
    return result


def outcomeless(held, cfg):
    from graphspec.schema.targets import TARGETS
    from tuning.batches.batching import batches, pack_batch
    from tuning.families.heads import FIELD
    from tuning.families.periods import blocks as period_blocks

    started = time.perf_counter()
    named = {name for kind in held.registry.schema.targets() for name in TARGETS.named(kind).outcomes}
    history = cfg["graph"]["time/history"]
    blocks = period_blocks(held, [block for partition in (0, 1) for block in held.blocks(partition, history, cfg)], cfg)
    found = [[block for block in blocks if block[0] == graph] for graph in sorted({block[0] for block in blocks})]
    groups = [group for chosen in found for group in batches(chosen, len(chosen), held.sizes(cfg), None, frame_budget=1024)]
    stated, targets, fields = [pack_batch(held, group, cfg, "cpu") for group in groups], held.targets, held.fields
    held.targets = targets[::-1] + 1
    held.fields = {name: values + 1 if name in named else values for name, values in fields.items()}
    try:
        changed = [pack_batch(held, group, cfg, "cpu") for group in groups]
    finally:
        held.targets, held.fields = targets, fields
    read = sorted({key for b in stated for key in b if key == "target" or key.startswith(FIELD)})
    result = read, same(stated, changed)
    LOG.debug("outcomeless exit batches=%d seconds=%.4f", len(stated), time.perf_counter() - started)
    return result
