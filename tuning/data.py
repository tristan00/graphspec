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

from graphspec import Registry, join, segments
from graphspec import spec as specs
from graphspec.infra.arrays import RESIDENT, Residents, between
from graphspec.infra.files import ArrayStore, load_arrays, read_json, save_arrays, sha256_file, write_json
from graphspec.infra.runtime import cache_path, output_path
from graphspec.infra.timing import measure, timed
from graphspec.infra.workspace import MEMORY_BYTES, Workspace
from graphspec.resources import GraphLimits
from graphspec.transform import SEED, SOURCE, materialize
from graphspec.truth import Graph, admitted

LOG = logging.getLogger(__name__)
ROWS = 1024
JOINED_RESIDENT = 4 << 30
JOINED_BUDGET = 12 << 30
DTYPES = dict(entity_kind=torch.long)


def layout(index, rows, steps, partitions, partition, history, level):
    chosen = np.flatnonzero(rows >= 0)
    chosen = chosen[partitions[rows[chosen]] == partition]
    result = []
    for moment in np.unique(steps[chosen]).tolist():
        start, stop = max(0, moment - history + 1), moment + 1
        if level == "node":
            members = chosen[steps[chosen] == moment]
            for piece in range(0, len(members), ROWS):
                part = members[piece : piece + ROWS]
                result.append((index, start, stop, moment, partition, int(part[0]), int(part[-1]) + 1))
        elif level == "path":
            result.append((index, start, stop, moment, partition, start, stop))
        else:
            first, last = between(steps, start, stop)
            result.append((index, start, stop, moment, partition, int(first), int(last)))
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


class Bank:
    def __init__(self, directory, resident=RESIDENT, device=None):
        started = time.perf_counter()
        self.directory = Path(directory)
        self.device = device
        self.manifest = admitted(self.directory, read_json(self.directory / "manifest.json"))
        self.cache = cache_path("banks", sha256_file(self.directory / "manifest.json"))
        self.registry = Registry.from_declaration(self.manifest["spec"])
        if self.registry.fingerprint() != self.manifest["spec_sha256"]:
            raise ValueError("bank declaration does not reproduce its pinned fingerprint")
        self.level = self.registry.extras["level"]
        self.categories = self.manifest["categories"]
        supervision = read_json(self.directory / "supervision.json")
        self.rows, self.split = supervision["rows"], supervision["split"]
        self.targets = np.asarray([row["y"] for row in self.rows], np.float32)
        self.partitions = np.zeros(len(self.rows), np.int8)
        self.partitions[self.split["validation_indices"]] = 1
        if self.level == "path":
            self.taus = np.asarray([row["tau"] for row in self.rows], np.float32)
            self.series_names = [entry["name"] for entry in self.split["series"]]
            if self.series_names != self.registry.extras["series"]:
                raise ValueError("bank supervision and declaration name different series")
            code = {name: index for index, name in enumerate(self.series_names)}
            self.series = np.asarray([code[row["series"]] for row in self.rows], np.int64)
            owners = {graph["name"]: graph["index"] for graph in self.manifest["graphs"]}
            keys = np.asarray([(owners[row["graph"]], row["step"]) for row in self.rows], np.int64)
            self.query_of = np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)
        self.center = np.asarray([entry["mean"] for entry in self.split["series"]], np.float32)
        self.scale = np.asarray([entry["sd"] for entry in self.split["series"]], np.float32)
        self.row_index, self.row_steps, self.entities = {}, {}, {}
        for graph in self.manifest["graphs"]:
            index = graph["index"]
            labels = load_arrays(self.truth(index), ("label_row", "label_step"))
            if self.level in ("node", "path"):
                self.row_index[index], self.row_steps[index] = labels["label_row"], labels["label_step"]
            else:
                self.row_index[index] = np.full(graph["steps"], -1, np.int64)
                self.row_index[index][labels["label_step"]] = labels["label_row"]
                self.row_steps[index] = np.arange(graph["steps"], dtype=np.int32)
            self.entities[index] = graph["entities"]
        self.variants, self.store, self.residents = {}, ArrayStore(), Residents(DTYPES, resident)
        self.roots = {}
        self._groups = {}
        self._shared_kinds = None
        self._tokens = WeakKeyDictionary()
        LOG.info(
            "bank exit seconds=%.3f graphs=%d rows=%d level=%s",
            time.perf_counter() - started,
            len(self.manifest["graphs"]),
            len(self.rows),
            self.level,
        )

    def truth(self, index):
        return self.directory / "truth" / ("graph_%04d" % index)

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
        return self.stated(cfg)[1]["graphs/join"] == "single"

    def groups(self, cfg):
        size = self.stated(cfg)[1]["graphs/join_size"]
        if size not in self._groups:
            started = time.perf_counter()
            splits = {}
            for graph in self.manifest["graphs"]:
                rows = self.row_index[graph["index"]]
                key = tuple(np.unique(self.partitions[rows[rows >= 0]]).tolist())
                splits.setdefault(key, []).append(graph["index"])
            self._groups[size] = [
                members[start : start + size]
                for _, members in sorted(splits.items())
                for start in range(0, len(members), size)
            ]
            LOG.info(
                "bank groups exit seconds=%.3f size=%d groups=%d",
                time.perf_counter() - started,
                size,
                len(self._groups[size]),
            )
        return self._groups[size]

    def graphs(self, cfg):
        if self.joining(cfg):
            return list(range(len(self.groups(cfg))))
        return [graph["index"] for graph in self.manifest["graphs"]]

    def points(self, index, cfg):
        if not self.joining(cfg):
            return self.row_index[index], self.row_steps[index]
        steps = {graph["index"]: graph["steps"] for graph in self.manifest["graphs"]}
        return join.layout(
            [
                (steps[member], self.row_index[member], self.row_steps[member])
                for member in self.groups(cfg)[index]
            ],
            self.level,
        )

    def limits(self):
        started = time.perf_counter()
        if self.device is None:
            raise ValueError(
                "bank %s was opened without a device and cannot derive a variant" % self.directory
            )
        result = GraphLimits(self.device)
        LOG.debug("bank limits exit device=%s seconds=%.4f", self.device, time.perf_counter() - started)
        return result

    @timed("segmented")
    def segmented(self, indices, cfg, workers):
        started = time.perf_counter()
        root = self.stamp(cfg)
        _, spec, chosen = self.variant(cfg)
        jobs, folders = [], []
        try:
            for index in indices:
                final = output_path(root / ("graph_%04d" % index))
                temporary = Path(tempfile.mkdtemp(prefix=final.name + ".", suffix=".tmp", dir=final.parent))
                folders.append((temporary, final))
                spill = temporary / "segments"
                spill.mkdir()
                paths = [self.truth(member) for member in self.groups(cfg)[index]]
                members = [Graph.load(path, mapped=True) for path in paths]
                jobs.append(dict(members=members, paths=paths, spill=spill, destination=temporary))
            segments.build_parallel(
                jobs, self.registry, spec, chosen, self.categories, SEED, workers, self.limits()
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
        graphs = {graph["index"]: graph for graph in self.manifest["graphs"]}
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

    def stated(self, cfg):
        from .model_spec import groups

        chosen = groups(cfg["model"])
        key = (chosen, specs.digest(cfg["graph"]))
        if key not in self.variants:
            started = time.perf_counter()
            self.variants[key] = (specs.canonical(specs.space(self.registry, chosen), cfg["graph"]), chosen)
            LOG.debug("bank stated exit seconds=%.4f", time.perf_counter() - started)
        return key, *self.variants[key]

    def identity(self, cfg):
        started = time.perf_counter()
        _, spec, chosen = self.stated(cfg)
        if self.device is None:
            raise ValueError(
                "the key of a variant of bank %s holds the device of dense algebra and no device was stated"
                % self.directory
            )
        result = dict(spec=specs.materialized(spec), groups=list(chosen), source=SOURCE, device=self.device)
        LOG.debug("bank identity exit device=%s seconds=%.4f", self.device, time.perf_counter() - started)
        return result

    def variant(self, cfg):
        key, spec, chosen = self.stated(cfg)
        if key not in self.roots:
            self.roots[key] = self.cache / "variants" / specs.digest(self.identity(cfg))
        return self.roots[key], spec, chosen

    def stamp(self, cfg):
        root = self.variant(cfg)[0]
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "variant.json").exists():
            write_json(root / "variant.json", self.identity(cfg))
        return root

    @timed("arrays")
    def arrays(self, index, cfg):
        root, spec, chosen = self.variant(cfg)
        joining = spec["graphs/join"] == "single"

        def materialized():
            started = time.perf_counter()
            self.stamp(cfg)
            with measure("truth_load"):
                truth = Graph.load(self.truth(index)).dense()
            arrays = materialize(truth, self.registry, spec, chosen, self.categories, self.limits())
            with measure("feature_deduplication"):
                linked = []
                for name, held, source in (("query", "query_x", "query_x"), ("x_table", "x_table", "obs_x")):
                    if identical(arrays[name], truth.arrays[held]):
                        arrays[name] = self.truth(index) / (source + ".npy")
                        linked.append(name)
            LOG.info(
                "variant exit graph=%d seconds=%.3f arrays=%d linked=%s",
                index,
                time.perf_counter() - started,
                len(arrays),
                ",".join(linked),
            )
            return arrays

        def build(destination):
            if joining:
                self.stamp(cfg)
                with (
                    Workspace(destination),
                    tempfile.TemporaryDirectory(prefix="segments-", dir=destination) as spill,
                ):
                    try:
                        _, spec, chosen = self.variant(cfg)
                        paths = [self.truth(member) for member in self.groups(cfg)[index]]
                        members = [Graph.load(path, mapped=True) for path in paths]
                        segments.build(
                            members,
                            self.registry,
                            spec,
                            chosen,
                            self.categories,
                            SEED,
                            spill,
                            self.limits(),
                            destination,
                        )
                    except BaseException as error:
                        traceback.clear_frames(error.__traceback__)
                        raise
            else:
                arrays = materialized()
                with measure("array_save"):
                    save_arrays(destination, arrays)

        return self.store.open_written(root / ("graph_%04d" % index), build)

    def resident(self, index, cfg, device):
        return self.residents.get(
            self.variant(cfg)[0] / ("graph_%04d" % index),
            device,
            lambda: self.arrays(index, cfg),
            JOINED_RESIDENT if self.joining(cfg) else MEMORY_BYTES,
            JOINED_BUDGET if self.joining(cfg) else None,
        )

    @timed("blocks")
    def blocks(self, partition, history, cfg):
        started = time.perf_counter()
        if self.joining(cfg):
            steps = {graph["index"]: graph["steps"] for graph in self.manifest["graphs"]}
            result = []
            for group, members in enumerate(self.groups(cfg)):
                rows, moments = join.layout(
                    [(steps[member], self.row_index[member], self.row_steps[member]) for member in members],
                    self.level,
                )
                result.extend(layout(group, rows, moments, self.partitions, partition, history, self.level))
        else:
            result = []
            for graph in self.manifest["graphs"]:
                index = graph["index"]
                result.extend(
                    layout(
                        index,
                        self.row_index[index],
                        self.row_steps[index],
                        self.partitions,
                        partition,
                        history,
                        self.level,
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


def identical(left, right):
    return left is right or (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )
