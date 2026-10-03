import logging
import pickle
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import pyarrow as pa
import torch
from numba import njit

from graphspec import Registry, registries
from graphspec.budgets import GraphLimits
from graphspec.engine import join, schedule, segments, transform
from graphspec.engine.edits import sliced
from graphspec.facts.stored import PACKED, Facts, admitted, inputs
from graphspec.infra.arrays import Residents, between
from graphspec.infra.files import (
    ArrayStore,
    load_arrays,
    read_json,
    read_pack,
    sha256_json,
    write_json,
)
from graphspec.infra.runtime import cache_path, output_path
from graphspec.infra.timing import timed
from graphspec.infra.workspace import MEMORY_BYTES, Workspace
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline, sets
from graphspec.operations.pipeline import agreed, resolved
from graphspec.representations import groups as representations
from graphspec.representations.registry import REPRESENTATIONS
from graphspec.schema.schema import ENCODERS, WRITTEN, Encoding, fitted
from graphspec.schema.targets import TARGETS
from graphspec.schema.time import SEVERAL
from graphspec.views.graph import Graph
from graphspec.views.view import STEPS, beside, finished, sampled, sequences, view
from graphspec.views.visible import SEES, STATEMENT, Looking
from tuning.statement import DERIVATIONS, derivation

LOG = logging.getLogger(__name__)
PACKAGES = (Path(__file__).parents[1], Path(__file__).parents[2] / "graphspec")
ROWS = 1024
JOINED_RESIDENT = 4 << 30
JOINED_BUDGET = 12 << 30
DTYPES = dict(entity_kind=torch.long)
VIEW = 62
TOKEN = (1 << 62) - 1


def layout(index, steps, labels, partitions, partition, history, windowed):
    started = time.perf_counter()
    held, places = labels
    chosen = np.unique(places[partitions[held] == partition])
    moments = steps[chosen]
    result = []
    for moment in np.unique(moments).tolist():
        start, stop = max(0, moment - history + 1), moment + 1
        if not windowed:
            members = chosen[moments == moment]
            for piece in range(0, len(members), ROWS):
                part = members[piece : piece + ROWS]
                result.append((index, start, stop, moment, partition, int(part[0]), int(part[-1]) + 1))
        else:
            first, last = between(steps, start, stop)
            result.append((index, start, stop, moment, partition, int(first), int(last)))
    LOG.debug(
        "layout exit graph=%d blocks=%d seconds=%.4f", index, len(result), time.perf_counter() - started
    )
    return result


def pointed(schema, rows):
    started = time.perf_counter()
    targets = schema.targets()
    code = {
        kind: targets.index(entry["target"]["kind"])
        for kind, entry in schema.tasks["kinds"].items()
        if entry["target"] is not None
    }
    aims = np.asarray([code[row["kind"]] for row in rows], np.int64)
    fields, inputs, columns = {}, {}, schema.target_columns()
    for place, kind in enumerate(targets):
        chosen = np.flatnonzero(aims == place).tolist()
        entry = TARGETS.named(kind)
        for name in entry.inputs + entry.outcomes:
            if name not in fields:
                stored = pa.array([], WRITTEN[columns[name]]).to_numpy(zero_copy_only=False).dtype
                fields[name] = np.full(len(rows), np.nan if stored.kind == "f" else 0, stored)
            fields[name][chosen] = [rows[row][name] for row in chosen]
        for name in entry.inputs:
            inputs.setdefault(name, np.full(len(rows), np.nan, np.float32))[chosen] = fields[name][chosen]
    LOG.debug(
        "pointed exit points=%d fields=%d inputs=%d seconds=%.4f",
        len(rows),
        len(fields),
        len(inputs),
        time.perf_counter() - started,
    )
    return aims, fields, inputs


def grouped(graphs, keys, size):
    started = time.perf_counter()
    found = {}
    for key, graph in sorted(
        zip(keys, graphs), key=lambda pair: (pair[0], pair[1]["name"], pair[1]["observer"], pair[1]["split"])
    ):
        found.setdefault(key, []).append(graph["index"])
    result = [members[start : start + size] for members in found.values() for start in range(0, len(members), size)]
    LOG.debug("grouped exit graphs=%d groups=%d seconds=%.4f", len(graphs), len(result), time.perf_counter() - started)
    return result


@njit(cache=True)
def distinct(values, lo, hi, size):
    order = np.argsort(hi)
    last = np.full(size, -1, np.int64)
    tree = np.zeros(len(values) + 1, np.int64)
    result = np.empty(len(lo), np.int64)
    position = 0
    for query in order:
        while position < hi[query]:
            value = values[position]
            if last[value] >= 0:
                index = last[value] + 1
                while index <= len(values):
                    tree[index] -= 1
                    index += index & -index
            index = position + 1
            while index <= len(values):
                tree[index] += 1
                index += index & -index
            last[value] = position
            position += 1
        total = 0
        index = hi[query]
        while index > 0:
            total += tree[index]
            index -= index & -index
        index = lo[query]
        while index > 0:
            total -= tree[index]
            index -= index & -index
        result[query] = total
    return result


class Windows:
    @timed("bank.Windows.__init__")
    def __init__(self, arrays, members):
        self.entity, self.offsets = arrays["entity"], arrays["snapshot_offsets"]
        self.size, self.members, self.known = len(arrays["entity_kind"]), members, {}

    def __call__(self, starts, stops):
        started = time.perf_counter()
        wanted = [pair for pair in dict.fromkeys(zip(starts, stops)) if pair not in self.known]
        if wanted:
            offsets = np.asarray(self.offsets)
            counts = distinct(
                np.asarray(self.entity, np.int64),
                offsets[np.asarray([pair[0] for pair in wanted])].astype(np.int64),
                offsets[np.asarray([pair[1] for pair in wanted])].astype(np.int64),
                self.size,
            )
            self.known.update(zip(wanted, counts.tolist()))
        result = np.asarray([self.known[pair] for pair in zip(starts, stops)], np.int64)
        LOG.info(
            "windows exit seconds=%.3f windows=%d counted=%d",
            time.perf_counter() - started,
            len(result),
            len(wanted),
        )
        return result


def written(folder, build):
    started = time.perf_counter()
    folder = output_path(folder)
    if not folder.exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=folder.name + ".", suffix=".tmp", dir=folder.parent) as name:
            build(Path(name))
            try:
                Path(name).replace(folder)
            except (FileExistsError, PermissionError):
                if not folder.is_dir():
                    raise
    LOG.debug("written exit folder=%s seconds=%.4f", folder.name, time.perf_counter() - started)
    return folder


@timed("bank.coded")
def coded():
    return dict(views=VIEW, derivation=transform.VERSION, representations=representations.VERSION)


def readings(registry, steps, several, storage, sees):
    started = time.perf_counter()
    result = registry.schema.read() + [
        (STEPS, [steps]),
        (SEVERAL, [] if several is None else [several]),
        (sets.STORAGE, list(storage.values())),
        (SEES, [rule for rules in sees.values() for rule in rules]),
    ]
    LOG.debug("readings exit seconds=%.4f", time.perf_counter() - started)
    return result


class Viewed:
    def __init__(self, bank, steps, several, storage, looking):
        started = time.perf_counter()
        self.steps, self.several, self.storage, self.looking = steps, several, storage, looking
        splits = looking.mapping()
        self.folder = (
            bank.cache
            / "views"
            / specs.digest(
                dict(
                    steps=steps,
                    several=several,
                    storage=storage,
                    splits=splits,
                    entries=registries.versions(readings(bank.registry, steps, several, storage, splits["sees"])),
                )
            )
        )
        if not (self.folder / "index.json").exists():
            self.build(bank, steps, several, storage, looking)
        self.graphs = read_json(self.folder / "index.json")
        stated = bank.registry.encoding
        self.registry = bank.registry.fitted(
            Encoding(
                stated.schema,
                stated.dictionaries,
                read_json(self.folder / "encoders.json")["states"],
                stated.relations,
            )
        )
        self.row_index, self.row_steps, self.labels, self.entities = {}, {}, {}, {}
        for graph in self.graphs:
            index = graph["index"]
            held = read_pack(self.path(index) / PACKED, True)
            aimed = bank.registry.aimed[held["task_kind"].astype(np.int64)]
            place = np.cumsum(aimed) - 1
            self.row_steps[index] = held["task_step"][aimed].astype(np.int32)
            self.row_index[index] = np.full(int(aimed.sum()), -1, np.int64)
            self.row_index[index][place[held["label_task"]]] = held["label_row"]
            self.labels[index] = (np.array(held["label_row"]), place[held["label_task"]])
            self.entities[index] = graph["entities"]
        LOG.info(
            "viewed exit seconds=%.3f steps=%s several=%s graphs=%d",
            time.perf_counter() - started,
            steps,
            several,
            len(self.graphs),
        )

    @timed("viewed.path")
    def path(self, index):
        return self.folder / ("graph_%04d" % index)

    def second(self, bank, index, name):
        started = time.perf_counter()
        entry, graphs = self.graphs[index], {graph["name"]: graph for graph in bank.manifest["graphs"]}
        if name not in graphs:
            raise KeyError("the bank holds no graph named %s to show beside graph %s" % (name, entry["name"]))
        named, source = graphs[name], bank.manifest["graphs"][entry["source"]]
        if named["unit"] != source["unit"]:
            raise ValueError(
                "the graph %s counts time in %s and the graph %s it would be shown beside in %s"
                % (name, named["unit"], source["name"], source["unit"])
            )
        path = self.folder / "beside" / ("graph_%04d" % named["index"]) / ("graph_%04d" % index)
        if not path.exists():
            registry = bank.registry
            facts = sets.stored(Facts.load(bank.truth(entry["source"])), registry, self.storage)
            (held,) = [
                held
                for held in sequences(facts, registry, self.steps, self.looking)
                if (held["observer"], held["split"]) == (entry["observer"], entry["split"])
            ]
            other = sets.stored(Facts.load(bank.truth(named["index"])), registry, self.storage)
            shown = finished(
                beside(other, registry, held, self.several), self.registry, inputs(self.registry, other.kinds, other.arrays)
            )
            written(path, shown.save)
        LOG.debug("viewed second exit graph=%s beside=%d seconds=%.4f", name, index, time.perf_counter() - started)
        return path

    def build(self, bank, steps, several, storage, looking):
        started = time.perf_counter()
        registry, entries = bank.registry, []
        columns = sets.shown(registry, storage)
        samples = {name: [] if name in columns else None for name in registry.schema.columns}
        rows = [row for code in looking.trained for row in bank.split["indices"][looking.splits[code]]]
        trained = {bank.rows[row]["graph"] for row in rows}
        for stored in (graph for graph in bank.manifest["graphs"] if graph["name"] in trained):
            facts = sets.stored(Facts.load(bank.truth(stored["index"])), registry, storage)
            for held in sequences(facts, registry, steps, looking):
                found = sampled(facts, registry, held, several)
                for name in columns:
                    samples[name] += found[name]
        encoding = fitted(registry.encoding, samples)
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json(self.folder / "encoders.json", dict(states=encoding.states))
        chosen = registry.fitted(encoding)
        for stored in bank.manifest["graphs"]:
            facts = sets.stored(Facts.load(bank.truth(stored["index"])), registry, storage)
            inner = inputs(chosen, facts.kinds, facts.arrays)
            for held in sequences(facts, registry, steps, looking):
                graph = finished(view(facts, registry, held, several), chosen, inner)
                written(self.path(len(entries)), graph.save)
                entries.append(
                    dict(
                        index=len(entries),
                        name=graph.name,
                        source=stored["index"],
                        observer=held["observer"],
                        split=held["split"],
                        steps=graph.steps,
                        entities=len(graph.kinds),
                        observations=int((graph.obs_stop.astype(np.int64) - graph.obs_start).sum()),
                        facts=int((graph.fact_stop.astype(np.int64) - graph.fact_start).sum())
                        + len(graph.removal_step),
                        labels=len(graph.label_task),
                    )
                )
        write_json(output_path(self.folder / "index.json"), entries)
        LOG.info("views built exit seconds=%.3f graphs=%d", time.perf_counter() - started, len(entries))


class Bank:
    def __init__(self, directory):
        started = time.perf_counter()
        self.directory = Path(directory)
        self.manifest = admitted(self.directory, read_json(self.directory / "manifest.json"))
        self.source = self.manifest["stored_sha256"]
        self.cache = cache_path("banks", sha256_json(dict(stored=self.source, version=VIEW)))
        self.registry = Registry.from_declaration(self.manifest["spec"])
        if self.registry.fingerprint() != self.manifest["spec_sha256"]:
            raise ValueError("bank declaration does not reproduce its pinned fingerprint")
        self.splits = self.manifest["splits"]
        supervision = read_json(self.directory / "supervision.json")
        self.rows, self.split = supervision["rows"], supervision["split"]
        self.targets = np.asarray([row["y"] for row in self.rows], np.float64)
        self.weights = np.asarray(
            [row["weight"] if "weight" in row else 1.0 for row in self.rows], np.float32
        )
        schema = self.registry.schema
        self.series_names = list(schema.series())
        measured, scaled = schema.measured(), {entry["name"]: entry for entry in self.split["series"]}
        wrong = [name for name in self.series_names if TARGETS.named(measured[name]).scaled != (name in scaled)]
        if wrong or len(scaled) != sum(TARGETS.named(measured[name]).scaled for name in self.series_names):
            raise ValueError(
                "the supervision of the bank scales the series %s, and its kinds of target scale the series %s"
                % (
                    ", ".join(scaled) or "none",
                    ", ".join(name for name in self.series_names if TARGETS.named(measured[name]).scaled) or "none",
                )
            )
        code = {name: index for index, name in enumerate(self.series_names)}
        self.series = np.asarray([code[row["series"]] for row in self.rows], np.int64)
        owners = {graph["name"]: graph["index"] for graph in self.manifest["graphs"]}
        asked = self.registry.schema.tasks["key"]
        keys = np.asarray([(owners[row["graph"]], row[asked]) for row in self.rows], np.int64).reshape(-1, 2)
        self.task_of = np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)
        self.aims, self.fields, self.inputs = pointed(schema, self.rows)
        self.nodes = self.named()
        self.center = np.asarray(
            [scaled[name]["mean"] if name in scaled else 0.0 for name in self.series_names], np.float32
        )
        self.scale = np.asarray([scaled[name]["sd"] if name in scaled else 1.0 for name in self.series_names], np.float32)
        self.variants = {}
        self.digests = {}
        self.choices = {}
        self._groups = {}
        self._viewed = {}
        self._looking = {}
        self._shared_kinds = None
        self._tokens = WeakKeyDictionary()
        LOG.info(
            "bank exit seconds=%.3f graphs=%d rows=%d targets=%s",
            time.perf_counter() - started,
            len(self.manifest["graphs"]),
            len(self.rows),
            ",".join(schema.targets()),
        )

    def named(self):
        started = time.perf_counter()
        result = np.full(len(self.rows), -1, np.int64)
        if any(TARGETS.named(kind).names is not None for kind in self.registry.schema.targets()):
            for graph in self.manifest["graphs"]:
                held = load_arrays(self.truth(graph["index"]), ("target_row", "target_node", "entity_token"))
                chosen = held["target_node"] >= 0
                tokens = held["entity_token"][held["target_node"][chosen]] & np.uint64(TOKEN)
                result[held["target_row"][chosen]] = tokens.astype(np.int64)
        LOG.debug("bank named exit named=%d seconds=%.4f", int((result >= 0).sum()), time.perf_counter() - started)
        return result

    def digested(self, kind, value, *owners):
        pickled = pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
        key = (kind, *map(id, owners))
        held = self.digests.get(key)
        if held is None or held[1] != pickled:
            held = self.digests[key] = (owners, pickled, specs.digest(value))
        return held[2]

    def looking(self, cfg):
        started = time.perf_counter()
        if "splits" not in cfg:
            raise KeyError("the configuration states no splits to train on and to score on")
        stated = cfg["splits"]
        key = self.digested("splits", stated, stated)
        if key not in self._looking:
            lacking = [name for name in STATEMENT if name not in stated]
            if lacking or len(stated) != len(STATEMENT):
                raise KeyError("the splits of the configuration state exactly " + ", ".join(STATEMENT))
            held = Looking(self.splits, stated["trained"], stated["scored"], stated["sees"])
            if sorted(stated["trained"]) != sorted(self.manifest["trained"]):
                raise ValueError(
                    "the scaling of the targets of the bank was fitted on the splits %s, and the fit trains on %s"
                    % (", ".join(self.manifest["trained"]), ", ".join(stated["trained"]))
                )
            partitions = np.full(len(self.rows), -1, np.int8)
            for partition, names in enumerate((stated["trained"], stated["scored"])):
                for name in names:
                    partitions[self.split["indices"][name]] = partition
            self._looking[key] = (held, partitions)
        LOG.debug("bank looking exit seconds=%.4f", time.perf_counter() - started)
        return self._looking[key]

    @timed("bank.Bank.partitions")
    def partitions(self, cfg):
        return self.looking(cfg)[1]

    @timed("bank.members")
    def members(self, cfg, partition):
        return np.flatnonzero(self.partitions(cfg) == partition)

    @timed("bank.Bank.truth")
    def truth(self, index):
        return self.directory / "truth" / ("graph_%04d" % index)

    def viewed(self, cfg):
        spec = self.stated(cfg)[2]
        storage = sets.chosen(self.registry, spec)
        key = (
            spec["time/steps"],
            self.several(spec),
            self.digested("storage", storage, spec),
            self.digested("splits", cfg["splits"], cfg["splits"]),
            registries.GENERATION,
        )
        if key not in self._viewed:
            self._viewed[key] = Viewed(self, *key[:2], storage, self.looking(cfg)[0])
        return self._viewed[key]

    def several(self, spec):
        return spec["time/several"] if self.registry.extras["several"] else None

    @timed("bank.view")
    def view(self, index, cfg):
        return self.viewed(cfg).path(index)

    @timed("shared_kinds")
    def shared_kinds(self):
        if self._shared_kinds is not None:
            return self._shared_kinds
        path = self.cache / "shared_kinds.json"
        if not path.exists():
            started = time.perf_counter()
            kinds, tokens, owners = [], [], []
            for graph in self.manifest["graphs"]:
                loaded = load_arrays(self.truth(graph["index"]), ("kinds", "entity_token"))
                kinds.append(loaded["kinds"].astype(np.int64))
                tokens.append(loaded["entity_token"])
                owners.append(np.full(len(loaded["kinds"]), graph["index"], np.int64))
            kinds, tokens, owners = np.concatenate(kinds), np.concatenate(tokens), np.concatenate(owners)
            order = np.lexsort((owners, tokens, kinds))
            kinds, tokens, owners = kinds[order], tokens[order], owners[order]
            key = np.r_[True, (kinds[1:] != kinds[:-1]) | (tokens[1:] != tokens[:-1])]
            fresh = key | np.r_[True, owners[1:] != owners[:-1]]
            graphs = np.add.reduceat(fresh, np.flatnonzero(key)) if len(key) else np.zeros(0, np.int64)
            first = kinds[key]
            distinct = np.bincount(first, minlength=len(self.registry.nodes))
            recurring = np.bincount(first[graphs > 1], minlength=len(self.registry.nodes))
            shared = [
                bool(
                    len(self.manifest["graphs"]) > 1
                    and distinct[kind] > 0
                    and recurring[kind] * 2 >= distinct[kind]
                )
                for kind in range(len(self.registry.nodes))
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, shared)
            LOG.info(
                "shared kinds exit seconds=%.3f shared=%s",
                time.perf_counter() - started,
                ",".join(self.registry.nodes[kind] for kind in range(len(shared)) if shared[kind]),
            )
        self._shared_kinds = np.asarray(read_json(path), bool)
        return self._shared_kinds

    def tokens(self, held):
        started = time.perf_counter()
        if held not in self._tokens:
            stated, kinds = self.shared_kinds(), held.numpy("entity_kind")
            shared = np.zeros(len(kinds), bool)
            inside = kinds < len(stated)
            shared[inside] = stated[kinds[inside]]
            self._tokens[held] = np.where(shared, held.numpy("entity_token"), -1)
        result = self._tokens[held]
        LOG.debug("bank tokens exit entities=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def joining(self, cfg):
        return self.stated(cfg)[2]["graphs/join"] == "single"

    def groups(self, cfg):
        started = time.perf_counter()
        spec, viewed = self.stated(cfg)[2], self.viewed(cfg)
        size = (
            spec["time/steps"],
            self.several(spec),
            self.digested("storage", sets.chosen(self.registry, spec), spec),
            self.digested("splits", cfg["splits"], cfg["splits"]),
            spec["graphs/join_size"],
        )
        if size in self._groups:
            LOG.debug("bank groups exit held seconds=%.4f", time.perf_counter() - started)
            return self._groups[size]
        partitions = self.partitions(cfg)
        keys = [tuple(np.unique(partitions[viewed.labels[graph["index"]][0]]).tolist()) for graph in viewed.graphs]
        self._groups[size] = grouped(viewed.graphs, keys, size[-1])
        LOG.info(
            "bank groups exit seconds=%.3f size=%d groups=%d",
            time.perf_counter() - started,
            size[-1],
            len(self._groups[size]),
        )
        return self._groups[size]

    @timed("bank.Bank.graphs")
    def graphs(self, cfg):
        if self.joining(cfg):
            return list(range(len(self.groups(cfg))))
        return [graph["index"] for graph in self.viewed(cfg).graphs]

    @timed("bank.Bank.points")
    def points(self, index, cfg):
        viewed = self.viewed(cfg)
        if not self.joining(cfg):
            return viewed.row_index[index], viewed.row_steps[index]
        steps = {graph["index"]: graph["steps"] for graph in viewed.graphs}
        return join.layout(
            [
                (steps[member], viewed.row_index[member], viewed.row_steps[member])
                for member in self.groups(cfg)[index]
            ]
        )

    def labels(self, index, cfg):
        started = time.perf_counter()
        viewed = self.viewed(cfg)
        if not self.joining(cfg):
            LOG.debug("bank labels exit graph=%d seconds=%.4f", index, time.perf_counter() - started)
            return viewed.labels[index]
        members = self.groups(cfg)[index]
        offsets = np.r_[0, np.cumsum([len(viewed.row_steps[member]) for member in members])]
        result = (
            np.concatenate([viewed.labels[member][0] for member in members]),
            np.concatenate([viewed.labels[member][1] + offset for member, offset in zip(members, offsets)]),
        )
        LOG.debug("bank labels exit group=%d seconds=%.4f", index, time.perf_counter() - started)
        return result

    def chosen(self, model):
        from tuning.families.model_spec import derived

        held = self.choices.get(id(model))
        if held is None or held[0] != model or held[1] != registries.GENERATION:
            held = self.choices[id(model)] = dict(model), registries.GENERATION, derived(model)
        return held[2]

    def stated(self, cfg):
        if "pipeline" not in cfg:
            raise KeyError("the configuration states no pipeline")
        chosen = self.chosen(cfg["model"])
        stated = dict(pipeline=cfg["pipeline"], graph=cfg["graph"])
        key = (chosen, self.digested("stated", stated, cfg["pipeline"], cfg["graph"]), registries.GENERATION)
        if key not in self.variants:
            self.variants[key] = self.settled(cfg, chosen)
        return key, *self.variants[key]

    def settled(self, cfg, chosen):
        from tuning.families.model_spec import groups

        started = time.perf_counter()
        result = (
            resolved(self.registry, cfg["pipeline"]),
            specs.canonical(specs.settings(self.registry, groups(cfg["model"])), cfg["graph"]),
            chosen,
        )
        agreed(self.registry, *result[:2])
        LOG.debug("bank stated exit seconds=%.4f", time.perf_counter() - started)
        return result

    @timed("blocks")
    def blocks(self, partition, history, cfg):
        from tuning.families.model_spec import windowed

        started = time.perf_counter()
        partitions = self.partitions(cfg)
        result = []
        for index in self.graphs(cfg):
            moments = self.points(index, cfg)[1]
            result.extend(
                layout(
                    index, moments, self.labels(index, cfg), partitions, partition, history, windowed(cfg["model"])
                )
            )
        LOG.info(
            "bank blocks exit seconds=%.3f partition=%d history=%d blocks=%d",
            time.perf_counter() - started,
            partition,
            history,
            len(result),
        )
        return result


class Deriver(Bank):
    def __init__(self, directory, device, seed, resident):
        started = time.perf_counter()
        if not derivation(seed):
            raise ValueError(
                "the seed of derivation is a whole number from 0 to %d, found %r" % (DERIVATIONS - 1, seed)
            )
        self.held = GraphLimits(device)
        self.device, self.seed = device, seed
        super().__init__(directory)
        self.store, self.residents = ArrayStore(), Residents(DTYPES, resident)
        self.grown = {}
        LOG.debug("deriver exit device=%s seconds=%.4f", device, time.perf_counter() - started)

    @timed("bank.Deriver.limits")
    def limits(self):
        return self.held

    @timed("deriver.fitted")
    def fitted(self, cfg):
        viewed, (key, steps, spec, _) = self.viewed(cfg), self.stated(cfg)
        if (viewed.folder, key) not in self.grown:
            self.grown[viewed.folder, key] = self.growing(viewed, steps, spec, cfg)
        return self.grown[viewed.folder, key]

    def reads(self, steps, spec, cfg):
        started = time.perf_counter()
        storage = sets.chosen(self.registry, spec)
        added, read = pipeline.read(self.registry, steps)
        read = readings(self.registry, spec["time/steps"], self.several(spec), storage, cfg["splits"]["sees"]) + read
        LOG.debug(
            "deriver reads exit added=%d read=%d seconds=%.4f", len(added), len(read), time.perf_counter() - started
        )
        return added, read

    def growing(self, viewed, steps, spec, cfg):
        started = time.perf_counter()
        provisional = pipeline.grown(viewed.registry, steps, None)
        if not provisional.schema.added_columns():
            LOG.debug("deriver growing exit columns=0 seconds=%.4f", time.perf_counter() - started)
            return provisional
        named = specs.digest(
            dict(
                pipeline=[step.keyed() for step in steps],
                device=self.device,
                seed=self.seed,
                versions=coded(),
                entries=registries.versions(self.reads(steps, spec, cfg)[1]),
            )
        )
        path = viewed.folder / "added" / (named + ".json")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, dict(states=self.trained(viewed, steps, provisional, cfg)))
        result = pipeline.grown(viewed.registry, steps, read_json(path)["states"])
        LOG.info("deriver growing exit columns=%d seconds=%.3f", len(result.schema.added_columns()), time.perf_counter() - started)
        return result

    def trained(self, viewed, steps, provisional, cfg):
        started = time.perf_counter()
        partitions = self.partitions(cfg)
        stated = {
            name: held for _, added in pipeline.read(viewed.registry, steps)[0] for name, held in added.columns.items()
        }
        samples = {name: [] for name in stated}
        for graph in viewed.graphs:
            index = graph["index"]
            rows, places = viewed.labels[index]
            moments = viewed.row_steps[index][places[partitions[rows] == 0]]
            if not len(moments):
                continue
            bound = int(moments.max()) + 1
            member = sliced(Graph.load(viewed.path(index)).dense(), bound)
            beside = {
                name: sliced(held[0].dense(), bound)
                for name, held in self.seconds(steps, [index], cfg, False).items()
            }
            derived = segments.whole(member, beside, provisional, steps, self.seed, self.limits(), partitions)
            for name, codes in pipeline.sampled(derived, provisional).items():
                samples[name].append(codes)
        result = {
            name: ENCODERS.named(held["encoder"]).fit(np.concatenate(samples[name] + [np.zeros(0, np.float32)]), None, name)
            for name, held in stated.items()
        }
        LOG.info("deriver trained exit columns=%d seconds=%.3f", len(result), time.perf_counter() - started)
        return result

    @timed("seconds")
    def seconds(self, steps, indices, cfg, mapped):
        viewed = self.viewed(cfg)
        return {
            name: [Graph.load(viewed.second(self, index, name), mapped=mapped) for index in indices]
            for name in sorted({step.graph for step in steps if step.graph is not None})
        }

    @timed("segmented")
    def segmented(self, indices, cfg, workers):
        started = time.perf_counter()
        root = self.stamp(cfg)
        _, steps, spec, chosen = self.variant(cfg)
        jobs, folders = [], []
        try:
            for index in indices:
                final = output_path(root / ("graph_%04d" % index))
                temporary = Path(tempfile.mkdtemp(prefix=final.name + ".", suffix=".tmp", dir=final.parent))
                folders.append((temporary, final))
                spill = temporary / "segments"
                spill.mkdir()
                paths = [self.view(member, cfg) for member in self.groups(cfg)[index]]
                members = [Graph.load(path, mapped=True) for path in paths]
                beside = self.seconds(steps, self.groups(cfg)[index], cfg, True)
                jobs.append(dict(members=members, beside=beside, paths=paths, spill=spill, destination=temporary))
            schedule.variants(
                jobs, self.fitted(cfg), steps, spec, chosen, self.seed, workers, self.limits(), self.partitions(cfg)
            )
            for (temporary, final), job in zip(folders, jobs):
                shutil.rmtree(job["spill"])
                temporary.replace(final)
        finally:
            for temporary, _ in folders:
                if temporary.exists():
                    shutil.rmtree(temporary)
        LOG.info(
            "bank segmented exit seconds=%.3f groups=%d workers=%d",
            time.perf_counter() - started,
            len(indices),
            workers,
        )

    @timed("sizes")
    def sizes(self, cfg):
        graphs = {graph["index"]: graph for graph in self.viewed(cfg).graphs}
        if not self.joining(cfg):
            return {
                index: (graph["entities"], graph["facts"] / graph["steps"]) for index, graph in graphs.items()
            }
        return {
            group: (
                Windows(self.arrays(group, cfg), len(members)),
                sum(graphs[member]["facts"] for member in members)
                / sum(graphs[member]["steps"] for member in members),
            )
            for group, members in enumerate(self.groups(cfg))
        }

    def identity(self, cfg):
        started = time.perf_counter()
        _, steps, spec, chosen = self.stated(cfg)
        added, read = self.reads(steps, spec, cfg)
        result = dict(
            schema=self.manifest["spec_sha256"],
            source=self.source,
            pipeline=[step.keyed() for step in steps],
            added=[dict(entry=name, **held.declaration()) for name, held in added],
            settings=specs.materialized(spec),
            groups=list(chosen),
            splits=cfg["splits"],
            device=self.device,
            seeds=dict(derivation=self.seed),
            versions=coded(),
            entries=registries.versions(read + [(REPRESENTATIONS, chosen)]),
        )
        LOG.debug("deriver identity exit device=%s seconds=%.4f", self.device, time.perf_counter() - started)
        return result

    @timed("bank.Deriver.variant")
    def variant(self, cfg):
        _, steps, spec, chosen = self.stated(cfg)
        return self.cache / "variants" / specs.digest(self.identity(cfg)), steps, spec, chosen

    @timed("bank.Deriver.stamp")
    def stamp(self, cfg):
        root = self.variant(cfg)[0]
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "variant.json").exists():
            write_json(root / "variant.json", self.identity(cfg))
        return root

    @timed("arrays")
    def arrays(self, index, cfg):
        root, steps, spec, chosen = self.variant(cfg)
        joining = self.joining(cfg)
        indices = self.groups(cfg)[index] if joining else [index]

        def build(destination):
            started = time.perf_counter()
            self.stamp(cfg)
            with Workspace(destination):
                try:
                    members = [Graph.load(self.view(member, cfg), mapped=joining) for member in indices]
                    segments.variant(
                        members,
                        self.seconds(steps, indices, cfg, joining),
                        self.fitted(cfg),
                        steps,
                        spec,
                        chosen,
                        self.seed,
                        self.limits(),
                        self.partitions(cfg),
                        destination,
                    )
                except BaseException as error:
                    traceback.clear_frames(error.__traceback__)
                    raise
            LOG.info("variant exit graph=%d seconds=%.3f", index, time.perf_counter() - started)

        return self.store.open_written(root / ("graph_%04d" % index), build)

    @timed("bank.Deriver.resident")
    def resident(self, index, cfg, device):
        return self.residents.get(
            self.variant(cfg)[0] / ("graph_%04d" % index),
            device,
            lambda: self.arrays(index, cfg),
            JOINED_RESIDENT if self.joining(cfg) else MEMORY_BYTES,
            JOINED_BUDGET if self.joining(cfg) else None,
        )
