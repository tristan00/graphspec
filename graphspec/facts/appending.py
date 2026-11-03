import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.time import NEVER

from .stored import Facts

LOG = logging.getLogger(__name__)
ENTITIES = ("obs_entity", "fact_source", "fact_target", "member_source", "member_target", "target_node",
            "set_edge", "set_member", "inner_owner", "next_actor", "action_actor", "role_source", "role_target")
TASKS = ("member_task", "target_task", "option_task", "next_task", "next_other", "concurrent_task", "concurrent_other")
SPARSE = ("known_row", "known_time", "replaced_row", "replaced_time", "seen_row", "seen_observer")


@timed("facts.append.lineage")
def lineage(directory, reference):
    directory = Path(directory).resolve()
    roots = [parent for parent in directory.parents if (parent / "manifest.json").is_file()]
    if not roots:
        raise ValueError("an appended truth graph must belong to a bank")
    target = (directory / reference).resolve()
    if not target.is_relative_to(roots[0]) or target == directory:
        raise ValueError("an appended truth layer leaves its bank lineage or refers to itself")
    return target


@timed("facts.append.remapped")
def remapped(values, mapping):
    values = np.asarray(values)
    found = values >= 0
    result = values.copy()
    mapping = np.asarray(mapping)
    if mapping.ndim != 1 or np.any(values[found] >= len(mapping)):
        raise ValueError("an appended row map does not cover its referenced rows")
    result[found] = mapping[values[found]].astype(values.dtype)
    return result


@timed("facts.append.same")
def same(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return first.shape == second.shape and np.array_equal(first, second, equal_nan=first.dtype.kind in "fc")


@timed("facts.append.rows")
def rows(first, second, fields, identity, where):
    lookup = {tuple(np.asarray(first[name])[at].item() for name in identity): at for at in range(len(first[identity[0]]))}
    added, mapping = [], []
    for at in range(len(second[identity[0]])):
        key = tuple(np.asarray(second[name])[at].item() for name in identity)
        if key in lookup:
            old = lookup[key]
            for name in fields:
                before = first[name][old] if old < len(first[name]) else second[name][added[old - len(first[name])]]
                if not same(before, second[name][at]):
                    raise ValueError("append changes an existing %s in %s" % (where, name))
        else:
            old = lookup[key] = len(first[identity[0]]) + len(added)
            added.append(at)
        mapping.append(old)
    chosen = np.asarray(added, np.int64)
    return {name: np.concatenate((first[name], second[name][chosen]), axis=0) for name in fields}, np.asarray(mapping, np.int64)


@timed("facts.append.ordered")
def ordered(arrays, group, order):
    count = len(order)
    position = np.empty(count, np.int64)
    position[order] = np.arange(count)
    sparse = {group + "_" + name for name in SPARSE}
    for name in tuple(arrays):
        if not name.startswith(group + "_"):
            continue
        if name in sparse:
            if name.endswith("_row"):
                arrays[name] = position[arrays[name]].astype(arrays[name].dtype)
        else:
            arrays[name] = arrays[name][order]
    for prefix in ("known", "replaced", "seen"):
        row = group + "_" + prefix + "_row"
        at = np.argsort(arrays[row], kind="stable")
        arrays[row] = arrays[row][at]
        value = group + "_" + prefix + ("_observer" if prefix == "seen" else "_time")
        arrays[value] = arrays[value][at]


def combined(first, second, plan):
    started = time.perf_counter()
    if first.name != second.name or first.unit != second.unit:
        raise ValueError("an appended graph keeps its name and its unit")
    a = {name: np.asarray(value) for name, value in first.arrays.items()}
    b = {name: np.asarray(value) for name, value in second.arrays.items()}
    for patch in plan["patches"]:
        name = patch["array"]
        if name not in ("obs_until", "fact_until", "obs_replaced_time", "fact_replaced_time"):
            raise ValueError("append cannot patch the stored array %s" % name)
        positions = np.asarray(patch["rows"], np.int64)
        values = np.asarray(patch["values"], np.int64)
        count = len(a["obs_since" if name.startswith("obs_") else "fact_since"])
        if positions.ndim != 1 or values.shape != positions.shape or np.any(positions < 0) or np.any(positions >= count):
            raise ValueError("append patch rows and values must identify existing fact rows")
        if len(np.unique(positions)) != len(positions) or np.any(values >= NEVER):
            raise ValueError("append patches state distinct rows and finite endpoints")
        if name.endswith("_replaced_time"):
            row_name = name.removesuffix("_time") + "_row"
            prior = dict(zip(a[row_name].tolist(), a[name].tolist()))
            if any(prior.get(row, NEVER) != NEVER for row in positions.tolist()):
                raise ValueError("append only closes previously open replacement endpoints")
            prior.update(zip(positions.tolist(), values.tolist()))
            ordered_rows = sorted(prior)
            a[row_name] = np.asarray(ordered_rows, a[row_name].dtype)
            a[name] = np.asarray([prior[row] for row in ordered_rows], a[name].dtype)
        else:
            if np.any(a[name][positions] != NEVER):
                raise ValueError("append only closes previously open interval endpoints")
            a[name] = a[name].copy()
            a[name][positions] = values.astype(a[name].dtype)
    keys, kinds = list(first.keys), a["kinds"].tolist()
    node_index = {(kind, key): at for at, (kind, key) in enumerate(zip(kinds, keys))}
    new_nodes, nodes = [], []
    for at, (kind, key) in enumerate(zip(b["kinds"].tolist(), second.keys)):
        if (kind, key) not in node_index:
            node_index[kind, key] = len(keys)
            keys.append(key)
            kinds.append(kind)
            new_nodes.append(at)
        nodes.append(node_index[kind, key])
    for name in ENTITIES:
        b[name] = remapped(b[name], nodes)
    task_fields = tuple(name for name in a if name.startswith("task_"))
    tasks, task_map = rows(a, b, task_fields, ("task_key",), "task input")
    for name in TASKS:
        b[name] = remapped(b[name], task_map)
    if isinstance(plan["rows"], str):
        if plan["rows"] != "identity":
            raise ValueError("an append row map is identity or an explicit index array")
    else:
        a["target_row"] = remapped(a["target_row"], plan["rows"])
    result = dict(tasks)
    selected = np.asarray(new_nodes, np.int64)
    result["kinds"] = np.asarray(kinds, a["kinds"].dtype)
    result["entity_token"] = np.r_[a["entity_token"], b["entity_token"][selected]]
    if not same(a["graph_values"], b["graph_values"]):
        raise ValueError("append changes the attributes of graph %s" % first.name)
    result["graph_values"] = a["graph_values"]
    for group in ("obs", "fact"):
        count = len(a[group + ("_entity" if group == "obs" else "_source")])
        for name in a:
            if name.startswith(group + "_"):
                later = b[name] + count if name.endswith("_row") else b[name]
                result[name] = np.concatenate((a[name], later), axis=0)
        if group == "obs":
            order = np.lexsort((result["obs_seq"], result["obs_until"], result["obs_entity"], result["obs_since"]))
        else:
            order = np.lexsort((result["fact_until"], result["fact_target"], result["fact_source"], result["fact_code"],
                                result["fact_seq"], ~result["fact_present"], result["fact_since"]))
        ordered(result, group, order)
    member_fields = tuple(name for name in a if name.startswith("member_"))
    members, _ = rows(a, b, member_fields, member_fields, "reference")
    result.update(members)
    member_order = np.argsort(result["member_task"], kind="stable")
    for name in member_fields:
        result[name] = result[name][member_order]
    set_fields = tuple(name for name in a if name.startswith("set_"))
    sets, _ = rows(a, b, set_fields, ("set_edge", "set_member", "set_role"), "set membership")
    result.update(sets)
    inner_first = dict(a, inner_key=np.asarray(first.inner, str))
    inner_second = dict(b, inner_key=np.asarray(second.inner, str))
    inner_fields = ("inner_owner", "inner_kind", "inner_values", "inner_key")
    inner, inner_map = rows(inner_first, inner_second, inner_fields, ("inner_owner", "inner_kind", "inner_key"), "held node")
    inner_keys = inner.pop("inner_key").tolist()
    result.update(inner)
    for name in ("within_source", "within_target"):
        b[name] = remapped(b[name], inner_map)
    within_fields = tuple(name for name in a if name.startswith("within_"))
    within, _ = rows(a, b, within_fields, ("within_source", "within_target", "within_code"), "held edge")
    result.update(within)
    option_first = dict(a, option_key=np.asarray(first.options, str))
    option_second = dict(b, option_key=np.asarray(second.options, str))
    option_fields = tuple(name for name in option_first if name.startswith("option_"))
    options, option_map = rows(option_first, option_second, option_fields, ("option_task", "option_key"), "option")
    option_keys = options.pop("option_key").tolist()
    result.update(options)
    for name in ("action_option", "origin_option", "target_option"):
        b[name] = remapped(b[name], option_map)
    generators = list(first.generators)
    for name in second.generators:
        if name not in generators:
            generators.append(name)
    b["origin_generator"] = remapped(b["origin_generator"], [generators.index(name) for name in second.generators])
    origin_fields = tuple(name for name in a if name.startswith("origin_"))
    origins, _ = rows(a, b, origin_fields, ("origin_option",), "option origin")
    result.update(origins)
    action_fields = tuple(name for name in a if name.startswith("action_"))
    counts = {}
    positions = []
    for owned in (a["action_option"], b["action_option"]):
        counts.clear()
        values = []
        for owner in owned.tolist():
            counts[owner] = counts.get(owner, -1) + 1
            values.append(counts[owner])
        positions.append(np.asarray(values, np.int64))
    actions, action_map = rows(dict(a, action_place=positions[0]), dict(b, action_place=positions[1]),
                               action_fields + ("action_place",), ("action_option", "action_place"), "action")
    actions.pop("action_place")
    result.update(actions)
    b["role_action"] = remapped(b["role_action"], action_map)
    role_fields = tuple(name for name in a if name.startswith("role_"))
    roles, _ = rows(a, b, role_fields, role_fields, "action role")
    result.update(roles)
    for group in ("next", "concurrent"):
        names = tuple(name for name in a if name.startswith(group + "_"))
        links, _ = rows(a, b, names, names, group + " decision")
        result.update(links)
    for name in tuple(name for name in a if name.startswith("target_")):
        result[name] = np.concatenate((a[name], b[name]), axis=0)
    target_order = np.lexsort((result["target_row"], result["target_task"]))
    for name in tuple(name for name in result if name.startswith("target_")):
        result[name] = result[name][target_order]
    for name, value in plan["replace"].items():
        if name not in ("next_task", "next_actor", "next_other", "concurrent_task", "concurrent_other"):
            raise ValueError("append cannot replace %s" % name)
        result[name] = np.asarray(value, result[name].dtype)
    found = Facts(first.name, keys, first.unit, inner_keys, second.observers, option_keys, generators, **result)
    LOG.debug("facts combined exit graph=%s observations=%d seconds=%.4f", found.name, len(found.obs_entity), time.perf_counter() - started)
    return found
