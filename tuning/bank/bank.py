import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import torch
from numba import njit

from graphspec import Registry
from graphspec.budgets import GraphLimits
from graphspec.engine import join, schedule, segments, transform
from graphspec.engine.transform import SEED
from graphspec.facts.stored import Facts, admitted, inputs
from graphspec.infra.arrays import Residents, between
from graphspec.infra.files import ArrayStore, load_arrays, read_json, sha256_file, sha256_json, write_json
from graphspec.infra.runtime import cache_path, output_path
from graphspec.infra.timing import timed
from graphspec.infra.workspace import MEMORY_BYTES, Workspace
from graphspec.operations import parameters as specs
from graphspec.operations import sets
from graphspec.operations.pipeline import resolved
from graphspec.representations import groups as representations
from graphspec.schema.schema import Encoding, fitted
from graphspec.views.graph import Graph
from graphspec.views.view import finished, sampled, sequences, view
from graphspec.views.visible import STATEMENT, Looking

LOG = logging.getLogger(__name__)
PACKAGES = (Path(__file__).parents[1], Path(__file__).parents[2] / "graphspec")
ROWS = 1024
JOINED_RESIDENT = 4 << 30
JOINED_BUDGET = 12 << 30
DTYPES = dict(entity_kind=torch.long)
VIEW = 4


def layout(index, rows, steps, labels, partitions, partition, history, windowed):
    started = time.perf_counter()
    if labels is None:
        chosen = np.flatnonzero(rows >= 0)
        chosen = chosen[partitions[rows[chosen]] == partition]
        moments = steps[chosen]
    else:
        held, places = labels
        moments = steps[places[partitions[held] == partition]]
    result = []
    for moment in np.unique(moments).tolist():
        start, stop = max(0, moment - history + 1), moment + 1
        if labels is None and not windowed:
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


class Viewed:
    def __init__(self, bank, steps, several, storage, looking):
        started = time.perf_counter()
        self.folder = (
            bank.cache
            / "views"
            / specs.digest(dict(steps=steps, several=several, storage=storage, splits=looking.mapping()))
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
            held = load_arrays(self.path(index), ("task_kind", "task_step", "label_task", "label_row"))
            aimed = bank.registry.aimed[held["task_kind"].astype(np.int64)]
            place = np.cumsum(aimed) - 1
            self.row_steps[index] = held["task_step"][aimed].astype(np.int32)
            self.row_index[index] = np.full(int(aimed.sum()), -1, np.int64)
            self.row_index[index][place[held["label_task"]]] = held["label_row"]
            self.labels[index] = (held["label_row"], place[held["label_task"]])
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
        self.source = sha256_file(self.directory / "manifest.json")
        self.cache = cache_path("banks", sha256_json(dict(manifest=self.source, version=VIEW)))
        self.registry = Registry.from_declaration(self.manifest["spec"])
        if self.registry.fingerprint() != self.manifest["spec_sha256"]:
            raise ValueError("bank declaration does not reproduce its pinned fingerprint")
        self.path = self.registry.extras["target"] == "path"
        self.splits = self.manifest["splits"]
        supervision = read_json(self.directory / "supervision.json")
        self.rows, self.split = supervision["rows"], supervision["split"]
        self.targets = np.asarray([row["y"] for row in self.rows], np.float32)
        self.weights = np.asarray(
            [row["weight"] if "weight" in row else 1.0 for row in self.rows], np.float32
        )
        self.series_names = [entry["name"] for entry in self.split["series"]]
        if self.series_names != self.registry.extras["series"]:
            raise ValueError("bank supervision and declaration name different series")
        code = {name: index for index, name in enumerate(self.series_names)}
        kinds = self.registry.schema.tasks["kinds"]
        self.series = np.asarray(
            [
                code[row["series"] if self.path else kinds[row["kind"]]["target"]["series"][0]]
                for row in self.rows
            ],
            np.int64,
        )
        owners = {graph["name"]: graph["index"] for graph in self.manifest["graphs"]}
        asked = self.registry.schema.tasks["key"]
        keys = np.asarray([(owners[row["graph"]], row[asked]) for row in self.rows], np.int64).reshape(-1, 2)
        self.task_of = np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)
        if self.path:
            self.taus = np.asarray([row["tau"] for row in self.rows], np.float32)
        self.center = np.asarray([entry["mean"] for entry in self.split["series"]], np.float32)
        self.scale = np.asarray([entry["sd"] for entry in self.split["series"]], np.float32)
        self.variants = {}
        self._groups = {}
        self._viewed = {}
        self._looking = {}
        self._shared_kinds = None
        self._tokens = WeakKeyDictionary()
        LOG.info(
            "bank exit seconds=%.3f graphs=%d rows=%d target=%s",
            time.perf_counter() - started,
            len(self.manifest["graphs"]),
            len(self.rows),
            self.registry.extras["target"],
        )

    def looking(self, cfg):
        if "splits" not in cfg:
            raise KeyError("the configuration states no splits to train on and to score on")
        stated = cfg["splits"]
        key = specs.digest(stated)
        if key not in self._looking:
            started = time.perf_counter()
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

    def partitions(self, cfg):
        return self.looking(cfg)[1]

    @timed("bank.members")
    def members(self, cfg, partition):
        return np.flatnonzero(self.partitions(cfg) == partition)

    def truth(self, index):
        return self.directory / "truth" / ("graph_%04d" % index)

    @timed("bank.viewed")
    def viewed(self, cfg):
        spec = self.stated(cfg)[2]
        storage = sets.chosen(self.registry, spec)
        key = (spec["time/steps"], self.several(spec), specs.digest(storage), specs.digest(cfg["splits"]))
        if key not in self._viewed:
            self._viewed[key] = Viewed(self, *key[:2], storage, self.looking(cfg)[0])
        return self._viewed[key]

    @timed("bank.fitted")
    def fitted(self, cfg):
        return self.viewed(cfg).registry

    @timed("bank.several")
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
        if held not in self._tokens:
            self._tokens[held] = np.where(
                self.shared_kinds()[held.numpy("entity_kind")], held.numpy("entity_token"), -1
            )
        return self._tokens[held]

    def joining(self, cfg):
        return self.stated(cfg)[2]["graphs/join"] == "single"

    def groups(self, cfg):
        spec, viewed = self.stated(cfg)[2], self.viewed(cfg)
        size = (
            spec["time/steps"],
            self.several(spec),
            specs.digest(sets.chosen(self.registry, spec)),
            specs.digest(cfg["splits"]),
            spec["graphs/join_size"],
        )
        if size not in self._groups:
            started = time.perf_counter()
            splits, partitions = {}, self.partitions(cfg)
            for graph in viewed.graphs:
                rows = viewed.labels[graph["index"]][0]
                key = tuple(np.unique(partitions[rows]).tolist())
                splits.setdefault(key, []).append(graph["index"])
            self._groups[size] = [
                members[start : start + size[-1]]
                for _, members in sorted(splits.items())
                for start in range(0, len(members), size[-1])
            ]
            LOG.info(
                "bank groups exit seconds=%.3f size=%d groups=%d",
                time.perf_counter() - started,
                size[-1],
                len(self._groups[size]),
            )
        return self._groups[size]

    def graphs(self, cfg):
        if self.joining(cfg):
            return list(range(len(self.groups(cfg))))
        return [graph["index"] for graph in self.viewed(cfg).graphs]

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
            return viewed.labels[index]
        members = self.groups(cfg)[index]
        offsets = np.r_[0, np.cumsum([len(viewed.row_steps[member]) for member in members])]
        result = (
            np.concatenate([viewed.labels[member][0] for member in members]),
            np.concatenate([viewed.labels[member][1] + offset for member, offset in zip(members, offsets)]),
        )
        LOG.debug("bank labels exit group=%d seconds=%.4f", index, time.perf_counter() - started)
        return result

    def stated(self, cfg):
        from tuning.families.model_spec import groups

        if "pipeline" not in cfg:
            raise KeyError("the configuration states no pipeline")
        chosen = groups(cfg["model"])
        key = (chosen, specs.digest(dict(pipeline=cfg["pipeline"], graph=cfg["graph"])))
        if key not in self.variants:
            started = time.perf_counter()
            self.variants[key] = (
                resolved(self.registry, cfg["pipeline"]),
                specs.canonical(specs.settings(self.registry, chosen), cfg["graph"]),
                chosen,
            )
            LOG.debug("bank stated exit seconds=%.4f", time.perf_counter() - started)
        return key, *self.variants[key]

    @timed("blocks")
    def blocks(self, partition, history, cfg):
        from tuning.families.model_spec import windowed

        started = time.perf_counter()
        partitions = self.partitions(cfg)
        result = []
        for index in self.graphs(cfg):
            rows, moments = self.points(index, cfg)
            result.extend(
                layout(
                    index,
                    rows,
                    moments,
                    self.labels(index, cfg) if self.path else None,
                    partitions,
                    partition,
                    history,
                    windowed(cfg["model"]),
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
    def __init__(self, directory, device, resident):
        started = time.perf_counter()
        self.held = GraphLimits(device)
        self.device = device
        super().__init__(directory)
        self.roots, self.store, self.residents = {}, ArrayStore(), Residents(DTYPES, resident)
        LOG.debug("deriver exit device=%s seconds=%.4f", device, time.perf_counter() - started)

    def limits(self):
        return self.held

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
                jobs.append(dict(members=members, paths=paths, spill=spill, destination=temporary))
            schedule.built(
                jobs, self.fitted(cfg), steps, spec, chosen, SEED, workers, self.limits(), self.partitions(cfg)
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
        result = dict(
            schema=self.registry.fingerprint(),
            source=self.source,
            pipeline=[step.keyed() for step in steps],
            settings=specs.materialized(spec),
            groups=list(chosen),
            splits=cfg["splits"],
            device=self.device,
            seeds=dict(derivation=SEED),
            versions=dict(views=VIEW, derivation=transform.VERSION, representations=representations.VERSION),
        )
        LOG.debug("deriver identity exit device=%s seconds=%.4f", self.device, time.perf_counter() - started)
        return result

    def variant(self, cfg):
        key, steps, spec, chosen = self.stated(cfg)
        if key not in self.roots:
            self.roots[key] = self.cache / "variants" / specs.digest(self.identity(cfg))
        return self.roots[key], steps, spec, chosen

    def stamp(self, cfg):
        root = self.variant(cfg)[0]
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "variant.json").exists():
            write_json(root / "variant.json", self.identity(cfg))
        return root

    @timed("arrays")
    def arrays(self, index, cfg):
        root, steps, spec, chosen = self.variant(cfg)
        indices = self.groups(cfg)[index] if self.joining(cfg) else [index]

        def build(destination):
            started = time.perf_counter()
            self.stamp(cfg)
            with Workspace(destination):
                try:
                    members = [Graph.load(self.view(member, cfg), mapped=True) for member in indices]
                    segments.build(
                        members,
                        self.fitted(cfg),
                        steps,
                        spec,
                        chosen,
                        SEED,
                        self.limits(),
                        self.partitions(cfg),
                        destination,
                    )
                except BaseException as error:
                    traceback.clear_frames(error.__traceback__)
                    raise
            LOG.info("variant exit graph=%d seconds=%.3f", index, time.perf_counter() - started)

        return self.store.open_written(root / ("graph_%04d" % index), build)

    def resident(self, index, cfg, device):
        return self.residents.get(
            self.variant(cfg)[0] / ("graph_%04d" % index),
            device,
            lambda: self.arrays(index, cfg),
            JOINED_RESIDENT if self.joining(cfg) else MEMORY_BYTES,
            JOINED_BUDGET if self.joining(cfg) else None,
        )
