import logging
import time

import numpy as np

from graphspec.facts.check import check
from graphspec.facts.stored import Facts, codes, column, matrix, sparse, tokens
from graphspec.infra.timing import timed
from graphspec.schema import time as clock
from graphspec.schema.decisions import WHOLE
from graphspec.schema.references import SORTS
from graphspec.schema.schema import VALUE, absent, inward, outward
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import ALWAYS, NEVER

from .decisions import decided, undecided

LOG = logging.getLogger(__name__)


def distinct(schema, tables):
    started = time.perf_counter()
    places = [(tables["graph"], schema.graph)]
    places += [(tables["observations"][kind], columns) for kind, columns in schema.entities().items()]
    places += [(tables["edges"][held], entry["columns"]) for held, entry in schema.edges.items()]
    places += [(tables["sets"][kind], entry["members"]) for kind, entry in schema.sets.items()]
    for kind, held in schema.holds.items():
        places += [
            (tables["held"][kind]["nodes"], inward(held)),
            (tables["held"][kind]["edges"], outward(held)),
        ]
    if "tasks" in tables:
        places.append((tables["tasks"], schema.asked_columns()))
    if schema.decided():
        places.append((tables["decisions"]["action"], schema.parameters()))
    result = {name: set() for name in schema.symbolic()}
    for table, columns in places:
        for name in columns:
            if name in result:
                values = schema.kind(name).read(table.column(VALUE + name))
                result[name].update(values[~schema.kind(name).missing(values)].tolist())
    LOG.debug("distinct exit columns=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def gathered(parts, width):
    started = time.perf_counter()
    offsets = np.r_[0, np.cumsum([len(part["since"]) for part in parts])].astype(np.int64)
    result = {
        name: np.concatenate([part[name] for part in parts] + [np.zeros(0, np.int64)])
        for name in ("since", "until", "known", "replaced", "seq")
    }
    result["values"] = np.concatenate([part["values"] for part in parts] + [np.zeros((0, width), np.float32)])
    result["watched"] = np.concatenate(
        [part["watched"][0] + offset for part, offset in zip(parts, offsets)] + [np.zeros(0, np.int64)]
    )
    result["watchers"] = np.concatenate([part["watched"][1] for part in parts] + [np.zeros(0, np.int32)])
    LOG.debug("gathered exit parts=%d seconds=%.4f", len(parts), time.perf_counter() - started)
    return result


def arranged(held, order):
    started = time.perf_counter()
    place = np.empty(len(order), np.int64)
    place[order] = np.arange(len(order))
    result = {name: values[order] for name, values in held.items() if name not in ("watched", "watchers")}
    result["watched"], result["watchers"] = place[held["watched"]], held["watchers"]
    LOG.debug("arranged exit rows=%d seconds=%.4f", len(order), time.perf_counter() - started)
    return result


def read(schema, observers, table, label, name):
    started = time.perf_counter()
    sequenced = schema.ties is not None and "seq" in clock.TIES.named(schema.ties).columns
    result = clock.stamped(table, schema.model, label, name)
    result["seq"] = column(table, "seq", np.int64) if sequenced else np.zeros(table.num_rows, np.int64)
    result["watched"] = clock.watched(table, observers, label, name)
    LOG.debug("read exit graph=%s rows=%d seconds=%.4f", name, table.num_rows, time.perf_counter() - started)
    return result


def asked(registry, splits, observers, tables, name):
    started = time.perf_counter()
    schema, encoding = registry.schema, registry.encoding
    stated, tasks, layout = schema.tasks, tables["tasks"], encoding.layout
    count, timed = tasks.num_rows, schema.model != "none"
    if timed and tasks.column("time").null_count:
        raise ValueError("a task of graph %s states no time" % name)
    moments = column(tasks, "time", np.int64) if timed else np.zeros(count, np.int64)
    if np.any((moments <= ALWAYS) | (moments >= NEVER)):
        raise ValueError("a task of graph %s states a time outside the range of times" % name)
    order = np.argsort(moments, kind="stable")
    place = np.empty(count, np.int64)
    place[order] = np.arange(count)
    listed = {kind: code for code, kind in enumerate(stated["kinds"])}
    found = tasks.column("kind").to_pylist()
    strangers = sorted({str(value) for value in found if value not in listed})
    if strangers:
        raise ValueError(
            "tasks of graph %s are of kinds the schema does not state: %s" % (name, ", ".join(strangers))
        )
    kinds = np.asarray([listed[value] for value in found], np.int32).reshape(-1)
    looking = np.full(count, -1, np.int32)
    if schema.observers:
        known = {held: code for code, held in enumerate(observers)}
        named = tasks.column("observer").to_pylist()
        strangers = sorted({value for value in named if value is not None and value not in known})
        if strangers:
            raise ValueError(
                "tasks of graph %s name observers the bank does not hold: %s" % (name, ", ".join(strangers))
            )
        looking = np.asarray([-1 if value is None else known[value] for value in named], np.int32).reshape(-1)
    known = {held: code for code, held in enumerate(splits)}
    named = tasks.column("split").to_pylist()
    strangers = sorted({value for value in named if value is not None and value not in known})
    if strangers:
        raise ValueError(
            "tasks of graph %s name splits the bank does not hold: %s" % (name, ", ".join(strangers))
        )
    split = np.asarray([-1 if value is None else known[value] for value in named], np.int32).reshape(-1)
    weight = np.asarray(tasks.column("weight").to_numpy(zero_copy_only=False), np.float32)
    keys = column(tasks, "key", np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("two tasks of graph %s state one key" % name)
    columns = schema.asked_columns()
    values = codes(registry, tasks, columns, encoding.positions(columns), len(layout))
    for kind, entry in stated["kinds"].items():
        members = kinds == listed[kind]
        weighted = ~np.isnan(weight[members])
        wrong = np.flatnonzero(members & ~(np.isfinite(weight) & (weight > 0)))
        if entry["weight"] and len(wrong):
            raise ValueError(
                "the task %d of the kind %s of graph %s states the weight %s; its kind states a weight above "
                "zero and finite at float32 precision" % (keys[wrong[0]], kind, name, weight[wrong[0]])
            )
        if not entry["weight"] and weighted.any():
            raise ValueError(
                "a task of graph %s of the kind %s states a weight, which its kind does not" % (name, kind)
            )
        for other in columns:
            if (
                other not in entry["columns"]
                and not np.isnan(values[members][:, encoding.positions([other])]).all()
            ):
                raise ValueError(
                    "a task of graph %s of the kind %s states the column %s, which its kind does not"
                    % (name, kind, other)
                )
    ended, discount = np.full(count, -1, np.int8), np.full(count, np.nan, np.float64)
    if schema.decided():
        given = tasks.column("ended")
        ended = np.where(absent(given), -1, np.asarray(given.fill_null(False).to_numpy(zero_copy_only=False), np.int8))
        discount = np.asarray(tasks.column("discount").to_numpy(zero_copy_only=False), np.float64)
    result = dict(
        task_ended=ended.astype(np.int8)[order],
        task_discount=discount[order],
        task_time=moments[order],
        task_observer=looking[order],
        task_grouping=column(tasks, "period", np.int32)[order]
        if stated["grouping"] is not None
        else np.zeros(count, np.int32),
        task_key=keys[order],
        task_kind=kinds[order],
        task_split=split[order],
        task_weight=weight[order],
        task_values=values[order],
        task_signals=np.stack(
            (matrix(tasks, "signal", np.float32), matrix(tasks, "signal_observed", np.float32)), 2
        )[order]
        if stated["signals"]
        else np.zeros((count, 0, 2), np.float32),
    )
    LOG.debug("asked exit graph=%s tasks=%d seconds=%.4f", name, count, time.perf_counter() - started)
    return result, place


def referred(registry, tables, tasks, place, local, kinds, name):
    started = time.perf_counter()
    stated = registry.schema.tasks
    listed = list(stated["kinds"])
    parts = dict(task=[], name=[], place=[], source=[], target=[], code=[])
    for number, (reference, entry) in enumerate(stated["references"].items()):
        sort = SORTS.named(entry["sort"])
        holders = np.asarray(
            [reference in stated["kinds"][listed[kind]]["references"] for kind in tasks["task_kind"]], bool
        ).reshape(-1)
        where = "the reference %s of graph %s" % (reference, name)
        if sort.columns() is None:
            continue
        table = tables["references"][reference]
        owner = place[column(table, "task", np.int64)]
        source = local(column(table, "source", np.int64))
        allowed = [registry.kind(kind) for kind in entry["kinds"]]
        target, code = np.full(len(owner), -1, np.int64), np.full(len(owner), -1, np.int64)
        if "relation" in table.column_names:
            relations = table.column("relation").to_pylist()
            edge = np.asarray([value is not None for value in relations], bool).reshape(-1)
            strangers = sorted({value for value in relations if value is not None} - set(registry.tables))
            if strangers:
                raise ValueError(
                    "%s names relations the schema does not state: %s" % (where, ", ".join(strangers))
                )
            ended = np.asarray(table.column("target").is_valid().to_numpy(zero_copy_only=False), bool)
            if not np.array_equal(edge, ended):
                raise ValueError("%s states both ends and the relation of every edge it names" % where)
            target[edge] = local(
                np.asarray(table.column("target").fill_null(0).to_numpy(zero_copy_only=False), np.int64)[edge]
            )
            code[edge] = [registry.code(value) for value in relations if value is not None]
            swap = np.zeros(len(owner), bool)
            for row in np.flatnonzero(edge).tolist():
                relation = registry.by_code[int(code[row])]
                pair = (registry.nodes[kinds[source[row]]], registry.nodes[kinds[target[row]]])
                pairs = [tuple(held) for held in relation.pairs]
                if pair not in pairs and not (relation.symmetric and pair[::-1] in pairs):
                    raise ValueError(
                        "%s names an edge of %s between kinds of node the relation does not join"
                        % (where, relation.name)
                    )
                swap[row] = relation.symmetric and pair[0] == pair[1] and source[row] > target[row]
            source, target = np.where(swap, target, source), np.where(swap, source, target)
        if len(owner) and not np.isin(kinds[source[code < 0]], allowed).all():
            raise ValueError("%s names nodes of kinds the reference does not state" % where)
        if np.any(~holders[owner]):
            raise ValueError("%s is stated by a task whose kind does not state it" % where)
        position = (
            column(table, "place", np.int64)
            if "place" in table.column_names
            else np.zeros(len(owner), np.int64)
        )
        order = np.lexsort((target, code, source, position, owner))
        owner, source, target, code, position = (
            values[order] for values in (owner, source, target, code, position)
        )
        if not sort.ordered:
            same = np.r_[False, owner[1:] == owner[:-1]]
            same[1:] &= ((source[1:] == source[:-1]) & (target[1:] == target[:-1]) & (code[1:] == code[:-1]))[
                : len(same) - 1
            ]
            if same.any():
                raise ValueError("%s names a member twice for one task" % where)
            start = np.maximum.accumulate(
                np.where(np.r_[True, owner[1:] != owner[:-1]][: len(owner)], np.arange(len(owner)), 0)
            )
            position = np.arange(len(owner)) - start
        counts = np.bincount(owner, minlength=len(holders))
        wrong = np.flatnonzero(holders & sort.miscounted(counts))
        if len(wrong):
            raise ValueError(
                "the kind %s requires of each of its tasks %s under the reference %s, which is of the sort %s, "
                "and the task %d of graph %s names %d"
                % (
                    listed[tasks["task_kind"][wrong[0]]],
                    sort.required(),
                    reference,
                    entry["sort"],
                    tasks["task_key"][wrong[0]],
                    name,
                    counts[wrong[0]],
                )
            )
        sort.check(
            where,
            counts[holders],
            owner,
            source,
            target,
            code,
            position,
        )
        parts["task"].append(owner)
        parts["name"].append(np.full(len(owner), number, np.int64))
        parts["place"].append(position)
        parts["source"].append(source)
        parts["target"].append(target)
        parts["code"].append(code)
    held = {key: np.concatenate(values + [np.zeros(0, np.int64)]) for key, values in parts.items()}
    rank = np.zeros(len(held["task"]), np.int64)
    for kind, entry in stated["kinds"].items():
        order = {list(stated["references"]).index(found): at for at, found in enumerate(entry["references"])}
        members = np.flatnonzero(tasks["task_kind"][held["task"]] == listed.index(kind))
        rank[members] = [order[value] for value in held["name"][members].tolist()]
    order = np.lexsort((held["place"], rank, held["task"]))
    result = dict(
        member_task=held["task"][order].astype(np.int32),
        member_name=held["name"][order].astype(np.int16),
        member_place=held["place"][order].astype(np.int32),
        member_source=held["source"][order].astype(np.int32),
        member_target=held["target"][order].astype(np.int32),
        member_code=held["code"][order].astype(np.int16),
    )
    LOG.debug(
        "referred exit graph=%s members=%d seconds=%.4f", name, len(order), time.perf_counter() - started
    )
    return result


def aimed(registry, tables, tasks, place, rows, series, referred, keys, kinds, name):
    started = time.perf_counter()
    stated = registry.schema.tasks
    listed = list(stated["kinds"])
    found = tables["targets"]
    owner = place[column(found, "task", np.int64)]
    rows, series = np.asarray(rows, np.int64), np.asarray(series, np.int32)
    if len(rows) != len(owner) or len(series) != len(owner):
        raise ValueError("graph %s reads different targets than its supervision" % name)
    known = np.full(len(owner), NEVER, np.int64)
    if "known" in found.column_names:
        given = found.column("known")
        held = np.asarray(given.fill_null(0).to_numpy(zero_copy_only=False), np.int64)
        if np.any((held <= ALWAYS) | (held >= NEVER)):
            raise ValueError("a target of graph %s is known from a time outside the range of times" % name)
        known = np.where(absent(given), NEVER, held)
    if np.any(known <= tasks["task_time"][owner]):
        raise ValueError(
            "a target of graph %s is known no later than the time of its task, whose view it would enter"
            % name
        )
    counts = np.bincount(owner, minlength=len(tasks["task_key"]))
    for kind, entry in stated["kinds"].items():
        members = tasks["task_kind"] == listed.index(kind)
        if entry["target"] is None and counts[members].any():
            raise ValueError(
                "a task of graph %s of the kind %s holds a target, which its kind does not state"
                % (name, kind)
            )
    if np.any(tasks["task_split"][owner] < 0):
        raise ValueError("a task of graph %s holds a target and states no split" % name)
    nodes = found.column("node").to_pylist() if "node" in found.column_names else [None] * len(owner)
    node = noded(registry, tasks, referred, keys, kinds, owner, nodes, name)
    order = np.lexsort((rows, owner))
    result = dict(
        target_task=owner[order].astype(np.int32),
        target_row=rows[order],
        target_known=known[order],
        target_y=column(found, "y", np.float64)[order],
        target_series=series[order],
        target_node=node[order].astype(np.int32),
    )
    aims = dict(task=owner[order], option=np.full(len(order), WHOLE), part=np.full(len(order), WHOLE))
    aims["chosen"] = np.zeros(len(order), bool)
    if registry.schema.decided():
        for part in ("option", "part"):
            given = found.column(part)
            aims[part] = np.where(absent(given), WHOLE, np.asarray(given.fill_null(0).to_numpy(), np.int64))[order]
        aims["chosen"] = np.asarray(found.column("chosen").fill_null(False).to_numpy(zero_copy_only=False), bool)[order]
    LOG.debug("aimed exit graph=%s targets=%d seconds=%.4f", name, len(order), time.perf_counter() - started)
    return result, aims


def noded(registry, tasks, members, keys, kinds, owner, nodes, name):
    started = time.perf_counter()
    schema, result = registry.schema, np.full(len(owner), -1, np.int64)
    references, count = list(schema.tasks["references"]), len(tasks["task_key"])
    names, code = np.unique(np.asarray(keys, str), return_inverse=True)
    code = code.reshape(-1)
    for number, (kind, entry) in enumerate(schema.tasks["kinds"].items()):
        target = entry["target"]
        chosen = np.flatnonzero(tasks["task_kind"][owner] == number)
        if target is None or TARGETS.named(target["kind"]).names is None or not len(chosen):
            continue
        where = "the target of the kind of task %s of graph %s" % (kind, name)
        stated = np.asarray([nodes[at] for at in chosen.tolist()], str)
        wanted = np.searchsorted(names, stated)
        lost = (wanted >= len(names)) | (names[np.minimum(wanted, len(names) - 1)] != stated)
        if lost.any():
            raise ValueError("%s names the node %s, which the graph does not hold" % (where, stated[np.argmax(lost)]))
        held = np.flatnonzero(members["member_name"] == references.index(target["about"]))
        task, source = members["member_task"][held].astype(np.int64), members["member_source"][held].astype(np.int64)
        if TARGETS.named(target["kind"]).names == "members":
            listed = task * len(names) + code[source]
            order = np.argsort(listed, kind="stable")
            asked = owner[chosen] * len(names) + wanted
            low = np.searchsorted(listed[order], asked, "left")
            ones = np.searchsorted(listed[order], asked, "right") - low == 1
            if not ones.all():
                raise ValueError(
                    "%s names the node %s, which is not one member of the reference %s of its task"
                    % (where, stated[np.argmin(ones)], target["about"])
                )
            result[chosen] = source[order[low]]
            points, listing = np.bincount(owner[chosen], minlength=count), np.bincount(task, minlength=count)
            unnamed = np.flatnonzero((points > 0) & (points != listing))
            if len(unnamed):
                raise ValueError(
                    "%s names %d of the %d members of the reference %s of the task %d; it names every member once"
                    % (where, points[unnamed[0]], listing[unnamed[0]], target["about"], tasks["task_key"][unnamed[0]])
                )
            continue
        anchor = np.full(count, -1, np.int64)
        anchor[task] = source
        edge = schema.edges[target["relation"]]
        pairs = {(registry.kind(left), registry.kind(right)) for left, right in edge["endpoints"]}
        if not edge["directed"]:
            pairs |= {(right, left) for left, right in pairs}
        within = np.argsort(code, kind="stable")
        low, high = np.searchsorted(code[within], wanted, "left"), np.searchsorted(code[within], wanted, "right")
        for at, point in enumerate(chosen.tolist()):
            ends = [
                int(found)
                for found in within[low[at] : high[at]].tolist()
                if (int(kinds[anchor[owner[point]]]), int(kinds[found])) in pairs
            ]
            if len(ends) != 1:
                raise ValueError(
                    "%s names the node %s for the task %d; the graph holds %d nodes of that key of a kind the "
                    "relation %s joins to the node of %s, and a link names one"
                    % (
                        where,
                        stated[at],
                        tasks["task_key"][owner[point]],
                        len(ends),
                        target["relation"],
                        target["about"],
                    )
                )
            result[point] = ends[0]
    LOG.debug("noded exit graph=%s points=%d seconds=%.4f", name, len(owner), time.perf_counter() - started)
    return result


def parse(registry, splits, observers, tables, rows, series, name):
    started = time.perf_counter()
    schema, encoding = registry.schema, registry.encoding
    unit = None
    if schema.model in clock.UNITED:
        unit = tables["graph"].column("unit").to_pylist()[0]
        if unit not in schema.units:
            raise ValueError(
                "graph %s states the unit of time %r; the schema states %s"
                % (name, unit, ", ".join(schema.units))
            )
    nodes = tables["nodes"]
    numbers = column(nodes, "entity", np.int64)
    order = np.argsort(numbers, kind="stable")
    ordered_numbers = numbers[order]
    if np.any(np.diff(ordered_numbers) == 0):
        raise ValueError("two nodes of graph %s share an entity number" % name)

    @timed("parse.parse.local")
    def local(values):
        position = np.searchsorted(ordered_numbers, values)
        if np.any(position >= len(ordered_numbers)) or not np.array_equal(ordered_numbers[position], values):
            raise ValueError("a row of graph %s names an entity outside the graph" % name)
        return position.astype(np.int32)

    kinds = np.asarray([registry.kind(kind) for kind in nodes.column("kind").to_pylist()], np.int32)[order]
    names = nodes.column("key").to_pylist()
    keys = [names[i] for i in order.tolist()]
    layout, entities = encoding.layout, schema.entities()
    parts = []
    for kind in registry.nodes:
        table = tables["observations"][kind]
        owners = local(column(table, "entity", np.int64))
        if np.any(kinds[owners] != registry.kind(kind)):
            raise ValueError(
                "an observation of graph %s of kind %s names a node of another kind" % (name, kind)
            )
        part = read(schema, observers, table, "fact about a node", name)
        part["entity"] = owners
        part["values"] = codes(
            registry, table, entities[kind], encoding.positions(entities[kind]), len(layout)
        )
        parts.append(part)
    seen = gathered(parts, len(layout))
    seen["entity"] = np.concatenate([part["entity"] for part in parts] + [np.zeros(0, np.int32)])
    seen = arranged(seen, np.lexsort((seen["seq"], seen["until"], seen["entity"], seen["since"])))
    width = max([len(registry.attributes(group)) for group in registry.groups] + [0])
    parts = []
    for held, entry in schema.edges.items():
        table = tables["edges"][held]
        count = table.num_rows
        if entry["column"] is None:
            relation = np.full(count, registry.code(held), np.int64)
        else:
            known = {found: registry.code(found) for found in encoding.relations[held]}
            found = table.column("relation").to_pylist()
            strangers = sorted({str(value) for value in found if value not in known})
            if strangers:
                raise ValueError(
                    "graph %s names relations of %s that the bank does not hold: %s"
                    % (name, held, ", ".join(strangers))
                )
            relation = np.asarray([known[value] for value in found], np.int64)
        source = local(column(table, "source", np.int64))
        target = local(column(table, "target", np.int64))
        if not entry["directed"]:
            swap = (kinds[source] == kinds[target]) & (source > target)
            source, target = np.where(swap, target, source), np.where(swap, source, target)
        part = read(schema, observers, table, "fact about an edge", name)
        part.update(source=source, target=target, code=relation, present=column(table, "present", bool))
        part["values"] = codes(registry, table, entry["columns"], range(len(entry["columns"])), width)
        parts.append(part)
    facts = gathered(parts, width)
    for key, kind in (("source", np.int32), ("target", np.int32), ("code", np.int64), ("present", bool)):
        facts[key] = np.concatenate([part[key] for part in parts] + [np.zeros(0, kind)])
    facts = arranged(
        facts,
        np.lexsort(
            (
                facts["until"],
                facts["target"],
                facts["source"],
                facts["code"],
                facts["seq"],
                ~facts["present"],
                facts["since"],
            )
        ),
    )
    for group in registry.valued:
        chosen = facts["present"] & np.isin(facts["code"], registry.coded[group])
        if not np.isfinite(facts["values"][chosen, : len(registry.attributes(group))]).all():
            raise ValueError("graph %s has a missing or infinite %s value on a present fact" % (name, group))
    if len(facts["code"]) and (facts["code"].min() < 0 or facts["code"].max() > np.iinfo(np.int16).max):
        raise ValueError("fact codes must fit 0..%d" % np.iinfo(np.int16).max)
    tasks, place = asked(registry, splits, observers, tables, name)
    members = referred(registry, tables, tasks, place, local, kinds, name)
    targets, aims = aimed(registry, tables, tasks, place, rows, series, members, keys, kinds, name)
    joined = belonging(registry, tables, kinds, local, name)
    inner, held = holding(registry, tables, kinds, local, name)
    choices, options, generators = undecided(tasks, aims), [], []
    if schema.decided():
        choices, keys, kinds, options, generators = decided(
            registry, tables["decisions"], tasks, place, members, keys, kinds, seen, aims, name
        )
    graph = Facts(
        name,
        keys,
        unit,
        inner,
        observers,
        options,
        generators,
        kinds=kinds,
        entity_token=tokens(registry, kinds, keys),
        **joined,
        **held,
        obs_entity=seen["entity"].astype(np.int32),
        obs_since=seen["since"],
        obs_until=seen["until"],
        obs_seq=seen["seq"].astype(np.int32),
        obs_values=seen["values"],
        **sparse("obs", seen, seen["watched"], seen["watchers"]),
        fact_source=facts["source"].astype(np.int32),
        fact_target=facts["target"].astype(np.int32),
        fact_code=facts["code"].astype(np.int16),
        fact_present=facts["present"],
        fact_since=facts["since"],
        fact_until=facts["until"],
        fact_seq=facts["seq"].astype(np.int32),
        fact_values=facts["values"],
        **sparse("fact", facts, facts["watched"], facts["watchers"]),
        graph_values=codes(
            registry, tables["graph"], schema.graph, range(len(schema.graph)), len(schema.graph)
        )[0],
        **tasks,
        **members,
        **targets,
        **choices,
    )
    check(graph, registry, splits, observers)
    LOG.info(
        "truth exit graph=%s seconds=%.3f observations=%d facts=%d tasks=%d labels=%d",
        name,
        time.perf_counter() - started,
        len(graph.obs_since),
        len(graph.fact_since),
        len(graph.task_time),
        len(graph.target_task),
    )
    return graph


def belonging(registry, tables, kinds, local, name):
    started = time.perf_counter()
    schema, columns = registry.schema, registry.schema.member_columns()
    parts = dict(edge=[], member=[], role=[], values=[])
    for kind, entry in schema.sets.items():
        table, where = tables["sets"][kind], "a member of a set edge of %s in graph %s" % (kind, name)
        edge, member = local(column(table, "set", np.int64)), local(column(table, "member", np.int64))
        if np.any(kinds[edge] != registry.kind(kind)):
            raise ValueError("%s belongs to an entity of another kind" % where)
        if not np.isin(kinds[member], [registry.kind(held) for held in entry["kinds"]]).all():
            raise ValueError("%s is a node of a kind the set edge does not join" % where)
        roles = table.column("role").to_pylist()
        strangers = sorted({str(role) for role in roles if role not in entry["roles"]})
        if strangers:
            raise ValueError("%s takes roles its kind does not state: %s" % (where, ", ".join(strangers)))
        positions = [columns.index(held) for held in entry["members"]]
        values = codes(registry, table, entry["members"], positions, len(columns))
        if not np.isfinite(values[:, positions]).all():
            raise ValueError("%s carries no value in a column of its members" % where)
        lonely = np.setdiff1d(np.flatnonzero(kinds == registry.kind(kind)), edge)
        if len(lonely):
            raise ValueError("the set edge of %s in graph %s has no member" % (kind, name))
        parts["edge"].append(edge.astype(np.int64))
        parts["member"].append(member.astype(np.int64))
        parts["role"].append(np.asarray([entry["roles"].index(role) for role in roles], np.int64).reshape(-1))
        parts["values"].append(values)
    edge, member, role = (
        np.concatenate(parts[part] + [np.zeros(0, np.int64)]) for part in ("edge", "member", "role")
    )
    values = np.concatenate(parts["values"] + [np.zeros((0, len(columns)), np.float32)])
    order = np.lexsort((role, member, edge))
    edge, member, role, values = edge[order], member[order], role[order], values[order]
    if np.any((edge[1:] == edge[:-1]) & (member[1:] == member[:-1]) & (role[1:] == role[:-1])):
        raise ValueError("a node of graph %s is a member of one set edge twice in one role" % name)
    result = dict(
        set_edge=edge.astype(np.int32),
        set_member=member.astype(np.int32),
        set_role=role.astype(np.int16),
        set_values=values,
    )
    LOG.debug(
        "belonging exit graph=%s members=%d seconds=%.4f", name, len(edge), time.perf_counter() - started
    )
    return result


def holding(registry, tables, kinds, local, name):
    started = time.perf_counter()
    schema = registry.schema
    nodes, links = schema.inner_columns(), schema.within_columns()
    parts = dict(owner=[], kind=[], keys=[], values=[], source=[], target=[], code=[], links=[])
    offset = 0
    for kind, held in schema.holds.items():
        where = "a graph held by a node of %s in graph %s" % (kind, name)
        table, edges = tables["held"][kind]["nodes"], tables["held"][kind]["edges"]
        owner = local(column(table, "owner", np.int64)).astype(np.int64)
        if np.any(kinds[owner] != registry.kind(kind)):
            raise ValueError("%s belongs to a node of another kind" % where)
        names, relations = list(held["nodes"]), list(held["edges"])
        found, keys = table.column("kind").to_pylist(), table.column("key").to_pylist()
        strangers = sorted({str(value) for value in found if value not in names})
        if strangers or None in keys:
            raise ValueError(
                "%s names held nodes of kinds it does not state, %s, or a held node without a key"
                % (where, ", ".join(strangers) or "none")
            )
        inner = np.asarray([names.index(value) for value in found], np.int64).reshape(-1)
        paired = sorted(zip(owner.tolist(), keys))
        if any(first == second for first, second in zip(paired[1:], paired[:-1])):
            raise ValueError("%s names one held node twice" % where)
        values = codes(
            registry, table, inward(held), [nodes.index(part) for part in inward(held)], len(nodes)
        )
        for code, part in enumerate(names):
            others = [nodes.index(other) for other in inward(held) if other not in held["nodes"][part]]
            if not np.isnan(values[inner == code][:, others]).all():
                raise ValueError("%s states a column that its held kind %s does not" % (where, part))
        source, target = column(edges, "source", np.int64), column(edges, "target", np.int64)
        if np.any(owner[source] != owner[target]):
            raise ValueError("%s holds an edge between the graphs of two nodes" % where)
        found = edges.column("relation").to_pylist()
        strangers = sorted({str(value) for value in found if value not in relations})
        if strangers:
            raise ValueError(
                "%s holds edges of relations it does not state: %s" % (where, ", ".join(strangers))
            )
        code = np.asarray([relations.index(value) for value in found], np.int64).reshape(-1)
        joints = codes(
            registry, edges, outward(held), [links.index(part) for part in outward(held)], len(links)
        )
        for number, relation in enumerate(relations):
            entry, chosen = held["edges"][relation], code == number
            pairs = {(names.index(first), names.index(second)) for first, second in entry["endpoints"]}
            if not entry["directed"]:
                pairs |= {(second, first) for first, second in pairs}
            ends = zip(inner[source[chosen]].tolist(), inner[target[chosen]].tolist())
            if not all(pair in pairs for pair in ends):
                raise ValueError(
                    "%s holds an edge of %s between held kinds it does not join" % (where, relation)
                )
            others = [links.index(other) for other in outward(held) if other not in entry["columns"]]
            if not np.isnan(joints[chosen][:, others]).all():
                raise ValueError("%s holds an edge of %s with a column it does not state" % (where, relation))
        order = np.argsort(owner, kind="stable")
        place = np.empty(len(order), np.int64)
        place[order] = np.arange(len(order))
        parts["owner"].append(owner[order])
        parts["kind"].append(inner[order])
        parts["keys"] += [keys[row] for row in order.tolist()]
        parts["values"].append(values[order])
        parts["source"].append(place[source] + offset)
        parts["target"].append(place[target] + offset)
        parts["code"].append(code)
        parts["links"].append(joints)
        offset += len(order)
    empty = np.zeros(0, np.int64)
    result = dict(
        inner_owner=np.concatenate(parts["owner"] + [empty]).astype(np.int32),
        inner_kind=np.concatenate(parts["kind"] + [empty]).astype(np.int16),
        inner_values=np.concatenate(parts["values"] + [np.zeros((0, len(nodes)), np.float32)]),
        within_source=np.concatenate(parts["source"] + [empty]).astype(np.int32),
        within_target=np.concatenate(parts["target"] + [empty]).astype(np.int32),
        within_code=np.concatenate(parts["code"] + [empty]).astype(np.int16),
        within_values=np.concatenate(parts["links"] + [np.zeros((0, len(links)), np.float32)]),
    )
    LOG.debug("holding exit graph=%s held=%d seconds=%.4f", name, offset, time.perf_counter() - started)
    return parts["keys"], result
