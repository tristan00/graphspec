import logging
import os
import shutil
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pyarrow as pa

from graphspec import Registry
from graphspec.build.decisions import Decided
from graphspec.build.generators import Frame, World, appended, looking
from graphspec.build.parse import distinct, parse, parsed
from graphspec.build.planning import stated as build_plan
from graphspec.build.writing import verified
from graphspec.engine.transform import shown as viewed
from graphspec.engine.transform import validate
from graphspec.facts.appending import combined
from graphspec.facts.check import check
from graphspec.facts.stored import Facts, admitted
from graphspec.infra.files import read_json, save_array, sha256_file, sha256_json, write_json
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed
from graphspec.schema import declaration
from graphspec.schema.decisions import generating
from graphspec.schema.schema import Encoding, Schema
from graphspec.schema.time import ALWAYS, NEVER

from .build import (
    budget,
    counts,
    crowded,
    inspection,
    largest,
    nameable,
    observing,
    open_source,
    supervision,
    target_ids,
    within_budget,
)

LOG = logging.getLogger(__name__)


@timed("bank.append.inside")
def inside(root, name):
    root = Path(root).resolve()
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError("an appended bank path leaves its bank: " + str(name))
    return result


@timed("bank.append.selected")
def selected(table, mask):
    return table.filter(pa.array(np.asarray(mask, bool)))


@timed("bank.append.mapped")
def mapped(table, field, places):
    values = np.asarray(table.column(field).to_numpy(), np.int64)
    table = selected(table, places[values] >= 0)
    values = places[np.asarray(table.column(field).to_numpy(), np.int64)]
    return table.set_column(table.schema.get_field_index(field), field, pa.array(values, table.schema.field(field).type))


@timed("bank.append.positions")
def positions(mask):
    result = np.full(len(mask), -1, np.int64)
    result[mask] = np.arange(np.count_nonzero(mask))
    return result


@timed("bank.append.later")
def later(table, watermark):
    since = np.asarray(table.column("since").fill_null(ALWAYS).to_numpy(), np.int64)
    known = np.asarray(table.column("known").fill_null(ALWAYS).to_numpy(), np.int64) if "known" in table.column_names else since
    known = np.where(known == ALWAYS, since, known)
    return selected(table, np.maximum(since, known) > watermark)


@timed("bank.append.delta_tables")
def delta_tables(tables, entry, keys, previous_ids):
    if entry["previous"] is None:
        return tables
    keep = np.isin(tables["tasks"].column("key").to_numpy(), keys)
    place = positions(keep)
    result = dict(tables)
    result["tasks"] = selected(tables["tasks"], keep)
    result["observations"] = {kind: later(table, entry["previous_watermark"]) for kind, table in tables["observations"].items()}
    result["edges"] = {kind: later(table, entry["previous_watermark"]) for kind, table in tables["edges"].items()}
    result["references"] = {name: mapped(table, "task", place) for name, table in tables["references"].items()}
    targets = tables["targets"]
    targets = selected(targets, ~np.isin(targets.column("id").to_numpy(), previous_ids))
    if np.any(place[np.asarray(targets.column("task").to_numpy(), np.int64)] < 0):
        raise ValueError("an appended target belongs to no new or completed task")
    result["targets"] = mapped(targets, "task", place)
    if "decisions" in tables:
        found = tables["decisions"]
        option_keep = place[np.asarray(found["option"].column("task").to_numpy(), np.int64)] >= 0
        option_place = positions(option_keep)
        action_keep = option_place[np.asarray(found["action"].column("option").to_numpy(), np.int64)] >= 0
        action_place = positions(action_keep)
        result["decisions"] = dict(option=mapped(found["option"], "task", place),
                                   action=mapped(found["action"], "option", option_place),
                                   role=mapped(found["role"], "action", action_place))
        for name in ("next", "concurrent"):
            table = found[name]
            mask = np.all(np.column_stack([place[np.asarray(table.column(field).to_numpy(), np.int64)] >= 0
                                           for field in ("task", "other")]), axis=1)
            table = selected(table, mask)
            for field in ("task", "other"):
                table = mapped(table, field, place)
            result["decisions"][name] = table
    return result


class Supervision:
    @timed("bank.append.Supervision.init")
    def __init__(self, source, previous):
        self.source, self.stated, self.previous = source, source.stated, previous

    @timed("bank.append.Supervision.asked")
    def asked(self):
        return self.source.asked()

    @timed("bank.append.Supervision.aims")
    def aims(self):
        found = self.source.aims()
        return selected(found, ~np.isin(found.column("id").to_numpy(), self.previous))


@timed("bank.append.extended")
def extended(previous, found):
    return list(previous) + sorted(set(found) - set(previous))


@timed("bank.append.registry")
def registry(task, old, found):
    schema = Schema(task["schema"])
    symbols = [entry["symbols"] for entry in found]
    dictionaries = {name: extended(old.encoding.dictionaries[name], set().union(*(entry[name] for entry in symbols)))
                    for name in schema.symbolic()}
    for name, entry in schema.edges.items():
        if entry["column"] is not None and any(None in graph["relations"][name] for graph in found):
            raise ValueError("an appended edge states no relation in table " + name)
    relations = {name: extended(old.encoding.relations[name],
                               set().union(*(entry["relations"][name] for entry in found)))
                 if entry["column"] is not None else list(old.encoding.relations[name])
                 for name, entry in schema.edges.items()}
    encoding = Encoding(schema, dictionaries, None, relations)
    extras = old.extras
    result = declaration.registry(task, encoding, extras["several"], extras["untasked"], extras["unvalued"], extras["acted"])
    edges = [result.by_name[edge.name] for edge in old.edges]
    edges += [edge for edge in result.edges if edge.name not in old.by_name]
    return Registry(old.name, encoding, edges, old.markers, result.extras)


@timed("bank.append.first_seen")
def first_seen(truth):
    first = np.full(len(truth.keys), NEVER, np.int64)
    np.minimum.at(first, np.asarray(truth.obs_entity, np.int64), np.asarray(truth.obs_since, np.int64))
    return {(int(kind), key): int(moment) for kind, key, moment in zip(truth.kinds, truth.keys, first)}


@timed("bank.append.patches")
def patches(truth, registry, descriptors):
    result = []
    identities = {(registry.nodes[int(kind)], key): at for at, (kind, key) in enumerate(zip(truth.kinds, truth.keys))}
    for patch in descriptors:
        group = patch["group"]
        stamps = truth.stamps(group)
        chosen = (stamps["since"] == patch["since"]) & (stamps["known"] == (patch["since"] if patch["known"] is None else patch["known"]))
        chosen &= np.asarray(truth.arrays[group + "_seq"]) == patch["seq"]
        if group == "obs":
            owner = identities[(patch["kind"], patch["key"])]
            chosen &= np.asarray(truth.obs_entity) == owner
            scope = np.zeros(len(registry.encoding.layout), bool)
            scope[registry.encoding.positions(patch["scope"])] = True
            chosen &= np.all(np.asarray(truth.obs_scope) == scope, axis=1)
            columns, values = registry.schema.entities()[patch["kind"]], np.asarray(truth.obs_values)
            columns_at = registry.encoding.positions(columns)
        elif group == "fact":
            source, target = [identities[(kind, patch[end])] for kind, end in zip(patch["ends"], ("source", "target"))]
            edge = registry.by_name[patch["relation"]]
            if edge.symmetric and truth.kinds[source] == truth.kinds[target] and source > target:
                source, target = target, source
            chosen &= (np.asarray(truth.fact_source) == source) & (np.asarray(truth.fact_target) == target)
            chosen &= (np.asarray(truth.fact_code) == edge.code) & (np.asarray(truth.fact_present) == patch["present"])
            columns, values = registry.schema.edges[patch["table"]]["columns"], np.asarray(truth.fact_values)
            columns_at = range(len(columns))
        else:
            raise ValueError("an append patch names an unknown fact group")
        for column, position in zip(columns, columns_at):
            kind = registry.schema.kind(column)
            value = kind.codes([patch["values"].get(column)], registry.encoding.dictionaries[column] if kind.symbols else None)[0]
            chosen &= np.isnan(values[:, position]) if np.isnan(value) else values[:, position] == value
        rows = np.flatnonzero(chosen).tolist()
        watchers = truth.arrays[group + "_seen_observer"]
        watched = truth.arrays[group + "_seen_row"]
        wanted = sorted(patch["observers"] or [])
        rows = [row for row in rows if sorted(truth.observers[int(code)] for code in watchers[watched == row]) == wanted]
        if len(rows) != 1:
            raise ValueError("an append patch of graph %s matches %d prior facts" % (truth.name, len(rows)))
        column = patch["column"]
        if column not in ("until", "replaced"):
            raise ValueError("an append patch may only close a fact endpoint")
        name = group + ("_until" if column == "until" else "_replaced_time")
        result.append(dict(array=name, rows=rows, values=[patch["value"]]))
    return result


@timed("bank.append.linked")
def linked(truth, registry, tables):
    if not registry.schema.decided():
        return {}
    lookup = {int(key): at for at, key in enumerate(truth.task_key)}
    place = np.asarray([lookup[key] for key in tables["tasks"].column("key").to_pylist()], np.int64)
    held = Decided(registry, truth.arrays, truth.arrays, truth.keys, truth.kinds,
                   dict(entity=np.zeros(0, np.int64), since=np.zeros(0, np.int64)), truth.name, first_seen(truth))
    options = dict(option_task=np.asarray(truth.option_task))
    return dict(**held.followed(tables["decisions"]["next"], options, place),
                **held.together(tables["decisions"]["concurrent"], options, place))


@timed("bank.append.generated")
def generated(truth, registry, splits, trained, keys):
    world, result = World(truth, registry), []
    for held, view in viewed(truth, registry, looking(splits, trained))[1]:
        frame = Frame(world, held, view)
        for task in frame.tasks.tolist():
            if int(truth.task_key[task]) in keys:
                result += frame.generated(task)
    return result


@timed("bank.append.characteristics")
def characteristics(truth, registry, trained):
    values, scopes = np.asarray(truth.obs_values), np.asarray(truth.obs_scope)
    entity, since, until = (np.asarray(truth.arrays["obs_" + name]) for name in ("entity", "since", "until"))
    known = truth.stamps("obs")["known"]
    lacking = {}
    for kind in nameable(registry.schema):
        rows = np.asarray(truth.kinds)[entity] == registry.kind(kind)
        for column in registry.schema.nodes[kind]:
            count = len(np.unique(entity[rows & np.isnan(values[:, registry.encoding.positions([column])[0]])]))
            if count:
                lacking[kind + ":" + column] = count
    scopes = np.column_stack((scopes, ~scopes.any(1)))
    rows, columns = np.nonzero(scopes)
    entity, since, until, known = entity[rows] * scopes.shape[1] + columns, since[rows], until[rows], known[rows]
    times, splits, observers = (np.asarray(truth.arrays["task_" + name]) for name in ("time", "split", "observer"))
    common = (splits < 0) | np.isin(splits, trained)
    sittings = []
    for observer in np.unique(observers):
        mine = observers == observer
        masks = [mine] + [mine & (common | (splits == split)) for split in np.unique(splits[mine]) if split >= 0]
        sittings += [np.unique(times[mask]) for mask in masks if mask.any()]
    learned = np.maximum(since, known)
    several = crowded(entity, since, until, until == since + 1, sittings) or crowded(
        entity, learned, np.maximum(until, learned + 1), (until <= learned + 1) | (known > since), sittings)
    return lacking, several


@timed("bank.append.layer")
def layer(directory, previous, delta, truth, plan):
    directory.mkdir(parents=True)
    write_json(directory / "graph.json", dict(name=truth.name, keys=truth.keys, unit=truth.unit, inner=truth.inner,
               observers=truth.observers, options=truth.options, generators=truth.generators,
               append=dict(base=os.path.relpath(previous, directory), delta=os.path.relpath(delta, directory), plan=plan)))


@exclusive
def append(task_path, output, workers):
    started = time.perf_counter()
    output, task_path = output_path(output), Path(task_path).resolve()
    manifest = admitted(output, read_json(output / "manifest.json"))
    task = read_json(task_path)
    if "staging" not in task:
        raise ValueError("bank append requires staged output")
    staging = (task_path.parent / task["staging"]).resolve()
    source_manifest = read_json(staging / "build.json")
    if "append" not in source_manifest:
        raise ValueError("bank append requires a staged append journal")
    journal = source_manifest["append"]
    if manifest["source_fingerprint"] != journal["previous_source"]:
        raise ValueError("the staged append does not continue the source fingerprint of this bank")
    if source_manifest["source_fingerprint"] != journal["source"]:
        raise ValueError("the staged append source fingerprint does not match its journal")
    verified(staging, journal["source"])
    plan = build_plan(source_manifest["plan"])
    if workers != plan.workers:
        raise ValueError("bank append workers must equal the build plan workers")
    limit = budget(task_path, task, plan, (output, manifest))
    previous = Registry.from_declaration(manifest["spec"])
    if previous.fingerprint() != manifest["spec_sha256"]:
        raise ValueError("bank declaration does not reproduce its pinned fingerprint")
    if Schema(task["schema"]).mapping() != previous.schema.mapping():
        raise ValueError("bank append keeps the resolved schema unchanged")
    old_ids = np.load(inside(output, manifest["source_target_ids"]), allow_pickle=False)
    prior_supervision = read_json(inside(output, manifest["supervision"]))
    if len(old_ids) != len(prior_supervision["rows"]) or len(np.unique(old_ids)) != len(old_ids):
        raise ValueError("bank source target IDs do not match its supervision")
    generation = int(manifest.get("generation", 0)) + 1
    directory = inside(output, "generations/%d" % generation)
    if directory.exists():
        raise FileExistsError("the append generation already exists: " + str(directory))
    directory.mkdir(parents=True)
    try:
        with closing(open_source(task_path, task)) as source:
            named = source.graphs()
            names = dict(zip(named.column("id").to_pylist(), named.column("name").to_pylist()))
            if any(names.get(entry["index"]) != entry["name"] for entry in manifest["graphs"]):
                raise ValueError("bank append preserves every existing graph index and name")
            rows, groups, found_splits, bare = supervision(Supervision(source, old_ids), manifest["trained"], names,
                                                          (prior_supervision, old_ids))
            splits = extended(manifest["splits"], found_splits)
            entries = {entry["index"]: entry for entry in journal["graphs"]}
            if set(entries) != set(names):
                raise ValueError("the append journal must describe every graph")
            reader, found = source.reader(), []
            for index in names:
                entry = entries[index]
                keys = read_json(staging / entry["delta"] / "records.json")["tasks"]["key"]
                tables = reader.graph(index)
                within_budget(tables, limit)
                delta = delta_tables(tables, entry, keys, old_ids)
                found.append(dict(symbols=distinct(source.stated, delta), observers=observing(delta),
                                  relations={name: set(table.column("relation").to_pylist()) for name, table in delta["edges"].items()
                                             if source.stated.edges[name]["column"] is not None}))
            held = registry(task, previous, found)
            observers = extended(manifest["observers"], set().union(*(set(entry["observers"]) for entry in found)))
            code = {name: at for at, name in enumerate(held.schema.series())}
            series = np.asarray([code[row["series"]] for row in rows["rows"]], np.int32)
            metrics, lacking, repeated, distinct_nodes = [], Counter(), False, {}
            for index, ids, members, _ in groups:
                entry = entries[index]
                keys = read_json(staging / entry["delta"] / "records.json")["tasks"]["key"]
                full = reader.graph(index)
                within_budget(full, limit)
                tables = delta_tables(full, entry, keys, old_ids)
                old_path = inside(output, manifest["truth_root"]) / ("graph_%04d" % index)
                old = None if entry["previous"] is None else Facts.load(old_path)
                targets = np.asarray(tables["targets"].column("id").to_numpy(), np.int64)
                lookup = dict(zip(ids.tolist(), members.tolist()))
                target_rows = np.asarray([lookup[value] for value in targets], np.int64)
                parts = parsed(held, splits, observers, tables, target_rows, series[target_rows], names[index])
                first = None if old is None else first_seen(old)
                delta = parse(held, tables, parts, first, old is None)
                merge = dict(patches=[] if old is None else patches(old, held, entry["patches"]), rows="identity", replace={})
                truth = delta if old is None else combined(old, delta, merge)
                missing, several = characteristics(truth, held, [splits.index(name) for name in manifest["trained"]])
                held.extras["several"] = repeated or several
                if generating(held.schema):
                    keys = set(delta.task_key.tolist()) - (set() if old is None else set(old.task_key.tolist()))
                    drawn = generated(truth, held, splits, manifest["trained"], keys)
                    delta = parse(held, dict(tables, decisions=appended(held, tables, drawn)), parts, first, old is None)
                    truth = delta if old is None else combined(old, delta, merge)
                replacements = linked(truth, held, full)
                merge["replace"] = {name: values.tolist() for name, values in replacements.items()}
                if replacements:
                    truth = truth.replace(**replacements)
                lacking.update(missing)
                repeated |= several
                held.extras["several"] = repeated
                check(truth, held, splits, observers)
                validate(truth, held, splits)
                destination = directory / "truth" / ("graph_%04d" % index)
                if old is None:
                    truth.save(destination)
                else:
                    delta_path = directory / "delta" / ("graph_%04d" % index)
                    delta.save(delta_path)
                    layer(destination, old_path, delta_path, truth, merge)
                for kind, token, key in zip(np.asarray(truth.kinds).tolist(), np.asarray(truth.entity_token).tolist(), truth.keys):
                    identity = kind, token
                    if identity in distinct_nodes and distinct_nodes[identity] != key:
                        raise ValueError("two entity keys of one node type share a token; entity identity would merge them")
                    distinct_nodes[identity] = key
                metrics.append(dict(index=index, name=names[index], **counts(truth), inspection=inspection(truth, held, splits),
                                    acted=largest(truth.action_option), seconds=0.0))
            held.extras.update(several=repeated, unvalued=dict(sorted(lacking.items())), untasked=bare,
                               acted=max((entry["acted"] for entry in metrics), default=0))
        write_json(directory / "supervision.json", rows)
        save_array(directory / "source-target-ids.npy", target_ids(groups, len(rows["rows"])))
        result = dict(manifest, generation=generation, truth_root=(directory / "truth").relative_to(output).as_posix(),
                      supervision=(directory / "supervision.json").relative_to(output).as_posix(),
                      source_target_ids=(directory / "source-target-ids.npy").relative_to(output).as_posix(),
                      source_fingerprint=journal["source"], task=task, database=str(staging), spec=held.declaration(),
                      spec_sha256=held.fingerprint(), graphs=metrics, rows=len(rows["rows"]), splits=splits, observers=observers,
                      distinct_entities=len(distinct_nodes), split_rows={name: len(indices) for name, indices in rows["split"]["indices"].items()},
                      supervision_sha256=sha256_file(directory / "supervision.json"))
        result["stored_sha256"] = sha256_json(dict(previous=manifest["stored_sha256"], source=journal["source"],
                 spec=result["spec_sha256"], files=sorted((path.relative_to(directory).as_posix(), sha256_file(path))
                 for path in directory.rglob("*") if path.is_file())))
        result["seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", result)
    except BaseException:
        if directory.resolve().is_relative_to((output / "generations").resolve()):
            shutil.rmtree(directory)
        raise
    LOG.info("bank append exit generation=%d graphs=%d seconds=%.3f", generation, len(result["graphs"]), time.perf_counter() - started)
    return result
