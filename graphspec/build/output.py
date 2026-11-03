import logging
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from graphspec.infra.files import read_json, sha256_file, write_json
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
DIGEST = re.compile("[0-9a-f]{64}")


@timed("output.manifest")
def manifest(directory):
    directory = Path(directory).resolve()
    entries = read_json(directory / "stage.json")
    seen = set()
    for entry in entries:
        key = (entry["table"], entry["source"], entry["target"])
        if key in seen:
            raise ValueError(f"duplicate stage table partition: {key}")
        seen.add(key)
        if (entry["source"] is None) != (entry["target"] is None):
            raise ValueError(f"relationship needs both endpoint types: {key}")
        if type(entry["rows"]) is not int or entry["rows"] < 0:
            raise ValueError(f"invalid stage row count: {key}")
        if "files" not in entry:
            raise ValueError("stage manifest requires a files list; convert legacy file entries explicitly")
        names = entry["files"]
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError(f"stage needs a nonempty list of distinct files: {key}")
        for name in names:
            if not (directory / name).resolve().is_relative_to(directory):
                raise ValueError(f"stage file escapes its directory: {name}")
        digests = entry["sha256"] if "sha256" in entry else None
        if (
            type(digests) is not list
            or len(digests) != len(names)
            or not all(type(digest) is str and DIGEST.fullmatch(digest) for digest in digests)
        ):
            raise ValueError(
                f"the staging {directory} records no sha256 of each file of {key}; a staging records the digest "
                "of every file it writes, and one written before it did is written again"
            )
        patches = entry.get("patches", [])
        if not isinstance(patches, list):
            raise ValueError("stage patches are a list")
        seen_patches = set()
        for patch in patches:
            if not isinstance(patch, dict) or set(patch) != {"file", "row", "column", "value"}:
                raise ValueError("stage patches state file, row, column and value")
            identity = (patch["file"], patch["row"], patch["column"])
            if (
                patch["file"] not in names
                or type(patch["row"]) is not int
                or patch["row"] < 0
                or patch["column"] not in ("until", "replaced")
                or (patch["value"] is not None and type(patch["value"]) is not int)
                or identity in seen_patches
            ):
                raise ValueError("invalid or repeated stage patch: %r" % (patch,))
            seen_patches.add(identity)
    return entries


@dataclass(frozen=True)
class StageFiles:
    directory: Path
    files: dict
    rows: dict
    digests: dict
    patches: dict = field(default_factory=dict)

    @timed("output.StageFiles.record")
    def record(self):
        entries = [
            dict(
                table=key[0],
                source=key[1],
                target=key[2],
                files=[path.relative_to(self.directory).as_posix() for path in paths],
                sha256=list(self.digests[key]),
                rows=self.rows[key],
                **({"patches": self.patches[key]} if key in self.patches and self.patches[key] else {}),
            )
            for key, paths in self.files.items()
        ]
        write_json(self.directory / "stage.json", entries)


class Stage:
    @timed("output.Stage.__init__")
    def __init__(self, directory):
        self.directory = output_path(directory)
        self.directory.mkdir(parents=True)
        self.writers = {}
        self.files = {}
        self.rows = {}
        self.resources = ExitStack()
        self.closed = False

    @timed("output.Stage.__enter__")
    def __enter__(self):
        return self

    @timed("output.Stage.__exit__")
    def __exit__(self, kind, error, traceback):
        self.close(commit=kind is None)

    @timed("output.Stage.write")
    def write(self, units):
        if self.closed:
            raise RuntimeError("stage is closed")
        for key, table in units.items():
            if key not in self.writers:
                path = self.directory / f"{len(self.writers):05d}.parquet"
                self.writers[key] = pq.ParquetWriter(
                    path, table.schema, compression="zstd", use_dictionary=False
                )
                self.resources.callback(self.writers[key].close)
                self.files[key] = (path,)
                self.rows[key] = 0
            if table.schema != self.writers[key].schema:
                raise ValueError(f"inconsistent table schema: {key}")
            if table.num_rows:
                self.writers[key].write_table(table)
                self.rows[key] += table.num_rows

    def close(self, commit=True):
        started = time.perf_counter()
        if self.closed:
            LOG.debug("stage close exit closed before seconds=%.4f", time.perf_counter() - started)
            return
        try:
            self.resources.close()
            if commit:
                digests = {key: tuple(sha256_file(path) for path in paths) for key, paths in self.files.items()}
                StageFiles(self.directory, self.files, self.rows, digests).record()
        finally:
            self.closed = True
        LOG.info(
            "stage close tables=%d committed=%s seconds=%.4f",
            len(self.files),
            commit,
            time.perf_counter() - started,
        )

    @staticmethod
    @timed("output.Stage.open")
    def open(directory):
        directory = Path(directory).resolve()
        files, rows, digests, patches = {}, {}, {}, {}
        for entry in manifest(directory):
            key = (entry["table"], entry["source"], entry["target"])
            files[key] = tuple(directory / name for name in entry["files"])
            rows[key], digests[key] = entry["rows"], tuple(entry["sha256"])
            if "patches" in entry:
                patches[key] = entry["patches"]
        return StageFiles(directory, files, rows, digests, patches)

    @classmethod
    def merge(cls, directory, shards):
        started = time.perf_counter()
        directory = output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files, rows, digests, patches = {}, {}, {}, {}
        seen_shards, seen_files, schemas = set(), set(), {}
        for shard in shards:
            stage = cls.open(shard)
            if stage.directory in seen_shards:
                raise ValueError("stage merge names a shard twice: " + str(stage.directory))
            seen_shards.add(stage.directory)
            if not stage.directory.is_relative_to(directory):
                raise ValueError(f"shard must be inside merged stage: {stage.directory}")
            for key, paths in stage.files.items():
                for path in paths:
                    resolved = path.resolve()
                    if resolved in seen_files:
                        raise ValueError("stage merge names a file twice: " + str(path))
                    seen_files.add(resolved)
                    schema = pq.read_schema(path)
                    if key in schemas and schemas[key] != schema:
                        raise ValueError("stage merge has incompatible schemas for %r" % (key,))
                    schemas[key] = schema
                files.setdefault(key, []).extend(paths)
                digests.setdefault(key, []).extend(stage.digests[key])
                rows[key] = rows.get(key, 0) + stage.rows[key]
                for patch in stage.patches.get(key, ()):
                    patches.setdefault(key, []).append(dict(
                        patch, file=(stage.directory / patch["file"]).relative_to(directory).as_posix()
                    ))
        result = StageFiles(
            directory,
            {key: tuple(paths) for key, paths in files.items()},
            rows,
            {key: tuple(found) for key, found in digests.items()},
            patches,
        )
        result.record()
        LOG.info("stage merge tables=%d seconds=%.4f", len(files), time.perf_counter() - started)
        return result
