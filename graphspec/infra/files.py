import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .runtime import output_path

if os.name == "nt":
    from .windows_files import open_read, replace
else:
    from os import replace

    def open_read(path):
        return Path(path).open("r", encoding="utf-8")


LOG = logging.getLogger(__name__)
BUDGET = 300 << 30
SIZES, SWEEPERS = {}, {}
PUBLISH = threading.Lock()


def write_json(path, value):
    with PUBLISH:
        path = output_path(path)
        stream = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        )
        temporary = Path(stream.name)
        try:
            with stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path):
    with open_read(path) as stream:
        return json.load(stream)


def write_jsonl(path, rows):
    with output_path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def write_csv(path, columns, rows):
    with output_path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_json(value):
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


def save_arrays(directory, arrays):
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        if isinstance(value, Path):
            shutil.copyfile(value, directory / (name + ".npy"))
        else:
            np.save(directory / (name + ".npy"), value, allow_pickle=False)


def load_arrays(directory, names, mmap_mode=None):
    directory = Path(directory)
    return {
        name: np.load(directory / (name + ".npy"), mmap_mode=mmap_mode, allow_pickle=False) for name in names
    }


class Mapped(Mapping):
    def __init__(self, directory, names):
        self.directory, self.names, self.loaded = Path(directory), tuple(names), {}

    def __getitem__(self, key):
        if key not in self.loaded:
            if key not in self.names:
                raise KeyError(key)
            self.loaded[key] = np.load(self.directory / (key + ".npy"), mmap_mode="r", allow_pickle=False)
        return self.loaded[key]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


class Arrays(Mapping):
    def __init__(self, directory):
        self.sources = {path.stem: path for path in Path(directory).glob("*.npy")}
        self.loaded = {}

    def __getitem__(self, key):
        if key not in self.loaded:
            self.loaded[key] = np.load(self.sources[key], mmap_mode="r", allow_pickle=False)
        return self.loaded[key]

    def read(self, key):
        return np.load(self.sources[key], allow_pickle=False)

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def __contains__(self, key):
        return key in self.sources


class ArrayStore:
    def __init__(self, limit=128):
        if type(limit) is not int or limit < 1:
            raise ValueError("array cache entry limit must be a positive integer")
        self.limit, self.loaded = limit, OrderedDict()

    def open(self, folder, build):
        return self.open_written(folder, lambda destination: save_arrays(destination, build()))

    def open_written(self, folder, build):
        folder = Path(folder)
        if folder in self.loaded:
            self.loaded.move_to_end(folder)
            return self.loaded[folder]
        folder = output_path(folder)
        if folder in self.loaded:
            self.loaded.move_to_end(folder)
            return self.loaded[folder]
        if not folder.exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=folder.name + ".", suffix=".tmp", dir=folder.parent
            ) as temporary_name:
                temporary = Path(temporary_name)
                build(temporary)
                try:
                    temporary.replace(folder)
                except (FileExistsError, PermissionError):
                    if not folder.is_dir():
                        raise
        self.loaded[folder] = Arrays(folder)
        while len(self.loaded) > self.limit:
            self.loaded.popitem(last=False)
        return self.loaded[folder]


class JsonCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def read(self, path):
        path = Path(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            with self.lock:
                self.cache.pop(path, None)
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if path not in self.cache or self.cache[path][0] != key:
                self.cache[path] = (key, read_json(path))
            return self.cache[path][1]

    def require(self, path):
        value = self.read(path)
        if value is None:
            raise FileNotFoundError("required JSON record is missing: " + str(path))
        return value


def folder_size(folder):
    total, stack = 0, [str(folder)]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                else:
                    total += entry.stat(follow_symlinks=False).st_size
    return total


def cached_size(folder):
    entries = len(os.listdir(folder))
    held = SIZES.get(folder)
    if held is None or held[0] != entries:
        started = time.perf_counter()
        held = SIZES[folder] = (entries, folder_size(folder))
        LOG.debug(
            "folder size exit folder=%s entries=%d bytes=%d seconds=%.3f",
            folder.name,
            entries,
            held[1],
            time.perf_counter() - started,
        )
    return held[1]


def sweep(trash):
    started, removed = time.perf_counter(), 0
    while True:
        folders = list(trash.iterdir())
        if not folders:
            break
        for folder in folders:
            shutil.rmtree(folder)
            removed += 1
    LOG.debug("sweep exit trash=%s removed=%d seconds=%.3f", trash, removed, time.perf_counter() - started)


def prune(root, limit, keep, stamp):
    started = time.perf_counter()
    root = Path(root)
    trash = root.parent / "trash"
    folders = [folder for folder in root.iterdir() if folder.is_dir() and folder != keep]
    unstamped = [folder for folder in folders if not (folder / stamp).exists()]
    if unstamped:
        raise ValueError(
            "folders without %s under %s: %s" % (stamp, root, ", ".join(folder.name for folder in unstamped))
        )
    sizes = {folder: cached_size(folder) for folder in folders}
    total = sum(sizes.values()) + cached_size(keep)
    removed = 0
    for folder in sorted(folders, key=lambda folder: (folder / stamp).stat().st_mtime):
        if total <= limit:
            break
        trash.mkdir(exist_ok=True)
        folder.replace(trash / ("%s.%d" % (folder.name, time.time_ns())))
        SIZES.pop(folder, None)
        total -= sizes[folder]
        removed += 1
    sweeper = SWEEPERS.get(trash)
    if trash.exists() and any(trash.iterdir()) and not (sweeper and sweeper.is_alive()):
        SWEEPERS[trash] = threading.Thread(target=sweep, args=(trash,), daemon=True)
        SWEEPERS[trash].start()
    LOG.debug(
        "prune exit root=%s folders=%d removed=%d bytes=%d seconds=%.3f",
        root,
        len(folders) + 1,
        removed,
        total,
        time.perf_counter() - started,
    )
    return total, removed
