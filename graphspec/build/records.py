import logging
import time

import numpy as np
import pyarrow as pa

from graphspec.build import forms
from graphspec.build.output import Stage
from graphspec.infra.files import write_json
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema.decisions import RECORDED, decision
from graphspec.schema.references import SORTS
from graphspec.schema.schema import (
    ACTIONS,
    EDGE,
    FACT_COLUMNS,
    FOLLOWS,
    INNER,
    LINK,
    MEMBER,
    OBSERVED,
    OF,
    OPTIONS,
    ROLED,
    TARGET,
    TASK,
    TOGETHER,
    VALUE,
    WITHIN,
    WRITTEN,
    Schema,
    inward,
    outward,
)
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import UNITED

LOG = logging.getLogger(__name__)


@timed("records.fixed")
def fixed(values):
    values = np.ascontiguousarray(values, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


class Counter:
    @timed("records.Counter.__init__")
    def __init__(self):
        self.next = {}

    @timed("records.Counter.take")
    def take(self, table, count):
        start = self.next.get(table, 0)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


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


@timed("records.parted")
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
        if name.startswith(VALUE) and name[len(VALUE):] in schema.columns:
            kind = schema.kind(name[len(VALUE):]).written
        elif name in GIVEN:
            kind = GIVEN[name]
        elif name in schema.columns:
            kind = schema.kind(name).written
        else:
            raise ValueError("%s states the column %s, which its schema does not" % (where, name))
        held[name] = pa.array(list(values), kind)
    result = pa.table(held)
    LOG.debug("table exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
    return result


@timed("records.feature_names")
def feature_names(rows, columns, metadata, where, partial):
    found = {}
    for name in columns:
        prefixed = VALUE + name
        if prefixed in rows:
            if name in rows and name not in metadata:
                raise ValueError("%s states the feature column %s twice" % (where, name))
            found[name] = prefixed
        elif name in rows and name not in metadata:
            found[name] = name
        elif not partial:
            raise ValueError("%s lacks feature column %s; use values for a name shared with metadata" % (where, name))
    return found


@timed("records.feature_table")
def feature_table(schema, columns):
    return pa.table({name: pa.array(values, schema.kind(name).written) for name, values in columns.items()})


def facts(schema, statement, given, values, where):
    started = time.perf_counter()
    parted(
        statement, STATEMENT + tuple(name for name in ("kind", "table", "ends") if name in statement), where
    )
    rows = dict(statement["rows"])
    feature = feature_names(rows, values, set(given) | set(schema.stamps()) | {"at", "index"}, where, False)
    for name, field in feature.items():
        rows[VALUE + name] = rows.pop(field)
    stated_rows = table(schema, rows, where)
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
    wanted = list(given) + list(stamps) + [VALUE + name for name in values]
    if sorted(result.column_names) != sorted(wanted):
        raise ValueError(
            "%s states the columns %s and is to state %s"
            % (where, ", ".join(result.column_names), ", ".join(wanted))
        )
    LOG.debug("facts exit rows=%d seconds=%.4f", result.num_rows, time.perf_counter() - started)
    return result.select(wanted)


@timed("records.none")
def none(schema, given, values):
    names = list(given) + list(schema.stamps()) + [VALUE + name for name in values]
    kinds = dict(GIVEN, **{VALUE + name: schema.kind(name).written for name in values})
    kinds.update({name: WRITTEN[kind] for name, kind in schema.stamps().items()})
    return pa.table({name: pa.array([], kinds[name]) for name in names})


@timed("records.gathered")
def gathered(schema, statements, given, values, where):
    found = [facts(schema, statement, given, values, where) for statement in statements]
    return pa.concat_tables(found) if found else none(schema, given, values)


@timed("records.observations")
def observations(schema, statements, columns, where):
    found = []
    for statement in statements:
        rows = statement["rows"]
        names = list(feature_names(rows, columns, {"key", "at", "index"} | set(schema.stamps()), where, True))
        count = len(rows["key"])
        supplied = dict(rows, **{VALUE + name: [None] * count for name in columns if name not in names})
        part = facts(schema, dict(statement, rows=supplied), ("key",), columns, where)
        found.append(part.append_column(FACT_COLUMNS, pa.array([names] * part.num_rows, pa.list_(pa.string()))))
    if found:
        return pa.concat_tables(found)
    return none(schema, ("key",), columns).append_column(FACT_COLUMNS, pa.array([], pa.list_(pa.string())))


@timed("records.fields")
def fields(schema):
    return GRAPH + (("sets",) if schema.sets else ()) + (("held",) if schema.holds else ())


def case_rows(schema, index, graph, counter):
    started = time.perf_counter()
    where = "the graph %s" % parted(graph, fields(schema), "a graph of graph records")["name"]
    united = schema.model in UNITED
    if united != (graph["unit"] is not None):
        raise ValueError("%s states a unit of time exactly when its schema has a clock" % where)
    described = dict(id=pa.array([index], pa.int32()), name=pa.array([graph["name"]]))
    if united:
        described["unit"] = pa.array([graph["unit"]])
    described.update(
        feature_table(schema, {name: [graph["columns"][name]] for name in schema.graph}).to_pydict()
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

    @timed("records.case_rows.found")
    def found(kind, keys):
        strangers = sorted({key for key in keys if key not in lookup[kind]})
        if strangers:
            raise ValueError("%s names nodes of the kind %s it does not hold: %s" % (where, kind, strangers))
        return pa.array([lookup[kind][key] for key in keys], pa.int64())

    parts = []
    for kind, columns in schema.entities().items():
        held_rows = observations(
            schema,
            [statement for statement in graph["values"] if statement["kind"] == kind],
            columns,
            "%s, values of %s" % (where, kind),
        )
        ids = counter.take("observation", held_rows.num_rows)
        parts.append(pa.table(dict(id=pa.array(ids), **{name: held_rows.column(name) for name in stamps})))
        tables[(OBSERVED + kind, kind, "observation")] = pa.table(
            {
                "from": found(kind, held_rows.column("key").to_pylist()),
                "to": pa.array(ids),
                **{name: held_rows.column(VALUE + name) for name in columns},
                FACT_COLUMNS: held_rows.column(FACT_COLUMNS),
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
            columns.update({column: held_rows.column(VALUE + column) for column in entry["columns"]})
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
        single = isinstance(stated, tuple) or bool(stated) and not isinstance(stated[0], (list, tuple))
        held = [tuple(member) for member in ([stated] if single else stated)]
    else:
        held = [(key, None, None) for key in stated["nodes"]] + [tuple(member) for member in stated["edges"]]
    LOG.debug("members exit members=%d seconds=%.4f", len(held), time.perf_counter() - started)
    return held


def case_tasks(schema, index, tasks, counter, lookup, where, ids=None, task_ids=None):
    started = time.perf_counter()
    asked = schema.tasks
    feature = feature_names(tasks, schema.asked_columns(), set(ASKED) | {"time", "observer", "grouping", "signal", "signal_observed"}, where, False)
    names = list(ASKED) + list(feature.values())
    names += ["time"] if schema.model != "none" else []
    names += ["observer"] if schema.observers else []
    names += ["grouping"] if asked["grouping"] is not None else []
    names += ["signal", "signal_observed"] if asked["signals"] else []
    parted(tasks, names, where + ", tasks")
    count = len(tasks["key"])
    ids = counter.take("task", count) if ids is None else np.asarray(ids, np.int64)
    if len(ids) != count:
        raise ValueError("task IDs and task records have different lengths")
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
        feature_table(schema, {name: tasks[field] for name, field in feature.items()}).to_pydict()
    )
    if asked["signals"]:
        held.update(signal=fixed(tasks["signal"]), signal_observed=fixed(tasks["signal_observed"]))
    deciding = schema.decided()
    if deciding:
        stated = [
            decision(target, asked["kinds"][kind]["target"], "%s, the task %r" % (where, key))
            if kind in deciding and target is not None
            else dict(ended=None, discount=None)
            for kind, key, target in zip(tasks["kind"], tasks["key"], tasks["target"])
        ]
        held.update(
            ended=pa.array([target["ended"] for target in stated], pa.bool_()),
            discount=pa.array([target["discount"] for target in stated], pa.float64()),
        )
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
        for row in kind.staged(target, stated, "%s, the task %r" % (where, tasks["key"][task])):
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
    if deciding:
        tables.update(case_decisions(schema, index, tasks, ids, counter, where, task_ids))
    LOG.debug("case tasks exit tasks=%d seconds=%.4f", count, time.perf_counter() - started)
    return tables


@timed("records.linked")
def linked(source, target):
    return pa.table({"from": pa.array(source, pa.int64()), "to": pa.array(target, pa.int64())})


@timed("records.fresh_decisions")
def fresh_decisions(tables, task_ids):
    if (OPTIONS, None, None) not in tables:
        return tables
    result, owners = dict(tables), set(task_ids)
    for table, parent in ((OPTIONS, "task"), (ACTIONS, OPTIONS), (ROLED, ACTIONS)):
        identity = table + OF, table, parent
        links = result[identity]
        mask = pa.array([owner in owners for owner in links.column("to").to_pylist()], pa.bool_())
        result[identity] = links.filter(mask)
        owners = set(result[identity].column("from").to_pylist())
        identity = table, None, None
        result[identity] = result[identity].filter(pa.array([value in owners for value in result[identity].column("id").to_pylist()], pa.bool_()))
    return result


def case_decisions(schema, index, tasks, ids, counter, where, task_ids=None):
    started = time.perf_counter()
    deciding, columns = schema.decided(), schema.parameters()
    keyed = {key: int(ids[at]) for at, key in enumerate(tasks["key"])} if task_ids is None else task_ids
    found = dict(
        next=dict(task=[], other=[], actor=[]),
        concurrent=dict(task=[], other=[]),
        option=dict(owner=[], key=[], sequence=[], available=[], chosen=[], order=[], shown=[], logged=[]),
        action=dict(owner=[], kind=[], actor=[], **{name: [] for name in columns}),
        role=dict(owner=[], role=[], key=[], relation=[], to_key=[]),
    )
    found["option"].update(generator=[], seed=[], drawn=[])
    for at, (kind, stated) in enumerate(zip(tasks["kind"], tasks["target"])):
        if kind not in deciding or stated is None:
            continue
        task, inner = int(ids[at]), "%s, the decision %r" % (where, tasks["key"][at])
        strangers = sorted(key for key in stated["concurrent"] + list(stated["next"].values()) if key not in keyed)
        if strangers:
            raise ValueError("%s names tasks the graph does not hold: %s" % (inner, strangers))
        for actor, key in stated["next"].items():
            found["next"]["task"].append(task)
            found["next"]["other"].append(keyed[key])
            found["next"]["actor"].append(actor)
        found["concurrent"]["task"] += [task] * len(stated["concurrent"])
        found["concurrent"]["other"] += [keyed[key] for key in stated["concurrent"]]
        actions = schema.tasks["kinds"][kind]["target"]["actions"]
        for option in stated["options"]:
            origin = option["origin"]
            held = found["option"]
            held["owner"].append(task)
            held["sequence"].append(option["together"] == "sequence")
            held["generator"].append(None if origin == RECORDED else origin["generator"])
            held["seed"].append(None if origin == RECORDED else origin["seed"])
            held["drawn"].append(None if origin == RECORDED else origin["drawn"])
            for name in ("key", "available", "chosen", "order", "shown", "logged"):
                held[name].append(option[name])
            for action in option["actions"]:
                if action["kind"] not in actions or set(action["parameters"]) != set(actions[action["kind"]]["parameters"]):
                    raise ValueError(
                        "%s states an action of the kind %s with the parameters %s; a kind of action of its kind of "
                        "task states every parameter it lists"
                        % (inner, action["kind"], ", ".join(sorted(action["parameters"])) or "none")
                    )
                if set(action["targets"]) != set(actions[action["kind"]]["roles"]):
                    raise ValueError(
                        "%s states an action of the kind %s with the roles %s; an action states every role of its "
                        "kind, %s, with the list of its targets, empty where it holds none"
                        % (
                            inner,
                            action["kind"],
                            ", ".join(sorted(action["targets"])) or "none",
                            ", ".join(actions[action["kind"]]["roles"]) or "none",
                        )
                    )
                found["action"]["owner"].append(len(held["owner"]) - 1)
                found["action"]["kind"].append(action["kind"])
                found["action"]["actor"].append(action["actor"])
                for name in columns:
                    found["action"][name].append(action["parameters"][name] if name in action["parameters"] else None)
                for role, targets in action["targets"].items():
                    for target in targets:
                        edged = isinstance(target, list)
                        found["role"]["owner"].append(len(found["action"]["owner"]) - 1)
                        found["role"]["role"].append(role)
                        found["role"]["key"].append(target[0] if edged else target)
                        found["role"]["relation"].append(target[1] if edged else None)
                        found["role"]["to_key"].append(target[2] if edged else None)
    numbers = {name: counter.take(table, len(found[name]["owner"])) for name, table in (
        ("option", OPTIONS), ("action", ACTIONS), ("role", ROLED)
    )}
    graphs = {name: pa.array(np.full(len(found[name]["owner"]), index), pa.int32()) for name in numbers}
    option, action, role = found["option"], found["action"], found["role"]
    kinds = dict(order=pa.int32(), shown=pa.float64(), logged=pa.float64(), seed=pa.int64(), drawn=pa.float64())
    tables = {
        (FOLLOWS, "task", "task"): linked(found["next"]["task"], found["next"]["other"]).append_column(
            "actor", pa.array(found["next"]["actor"], pa.string())
        ),
        (TOGETHER, "task", "task"): linked(found["concurrent"]["task"], found["concurrent"]["other"]),
        (OPTIONS, None, None): pa.table(
            dict(
                id=pa.array(numbers["option"]),
                graph=graphs["option"],
                key=pa.array(option["key"], pa.string()),
                **{name: pa.array(option[name], pa.bool_()) for name in ("sequence", "available", "chosen")},
                generator=pa.array(option["generator"], pa.string()),
                **{name: pa.array(option[name], kind) for name, kind in kinds.items()},
            )
        ),
        (OPTIONS + OF, OPTIONS, "task"): linked(numbers["option"], option["owner"]),
        (ACTIONS, None, None): pa.table(
            dict(
                id=pa.array(numbers["action"]),
                graph=graphs["action"],
                kind=pa.array(action["kind"], pa.string()),
                actor=pa.array(action["actor"], pa.string()),
                **{name: pa.array(action[name], schema.kind(name).written) for name in columns},
            )
        ),
        (ACTIONS + OF, ACTIONS, OPTIONS): linked(
            numbers["action"], numbers["option"][np.asarray(action["owner"], np.int64)]
        ),
        (ROLED, None, None): pa.table(
            dict(
                id=pa.array(numbers["role"]),
                graph=graphs["role"],
                **{name: pa.array(role[name], pa.string()) for name in ("role", "key", "relation", "to_key")},
            )
        ),
        (ROLED + OF, ROLED, ACTIONS): linked(numbers["role"], numbers["action"][np.asarray(role["owner"], np.int64)]),
    }
    LOG.debug(
        "case decisions exit options=%d actions=%d seconds=%.4f",
        len(option["owner"]),
        len(action["owner"]),
        time.perf_counter() - started,
    )
    return tables


@timed("records.table_of")
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
    feature = feature_names(statement["rows"], values, set(given) | set(schema.stamps()) | {"at", "index"}, where, False)
    renamed = {VALUE + name: field for name, field in feature.items()}
    columns = dict({renamed.get(name, name): held for name, held in taken.to_pydict().items()}, at=moments)
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
    schema = Schema(parted(case, PARTS, "graph records")["schema"])
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
    schema = Schema(parted(case, PARTS, "graph records")["schema"])
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.joinpath(name + ".staging").exists():
        raise FileExistsError("staging exists; choose another name or output")
    counter = Counter()
    with Stage(output / (name + ".staging")) as held_stage:
        for index, graph in enumerate(case["graphs"]):
            held_stage.write(case_rows(schema, index, graph, counter))
    write_json(
        output / (name + ".task.json"),
        dict(name=name, staging=name + ".staging", schema=schema.mapping()),
    )
    LOG.info(
        "records stage exit seconds=%.3f staging=%s graphs=%d",
        time.perf_counter() - started,
        held_stage.directory,
        len(case["graphs"]),
    )
    return output / (name + ".task.json")
