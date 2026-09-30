import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from graphspec.infra.files import read_json, write_json
from graphspec.infra.runtime import output_path

LOG = logging.getLogger(__name__)


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
    return entries


@dataclass(frozen=True)
class StageFiles:
    directory: Path
    files: dict
    rows: dict

    def record(self):
        entries = [
            dict(
                table=key[0],
                source=key[1],
                target=key[2],
                files=[path.relative_to(self.directory).as_posix() for path in paths],
                rows=self.rows[key],
            )
            for key, paths in self.files.items()
        ]
        write_json(self.directory / "stage.json", entries)


class Stage:
    def __init__(self, directory):
        self.directory = output_path(directory)
        self.directory.mkdir(parents=True)
        self.writers = {}
        self.files = {}
        self.rows = {}
        self.resources = ExitStack()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.close(commit=kind is None)

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
        if self.closed:
            return
        started = time.perf_counter()
        try:
            self.resources.close()
            if commit:
                StageFiles(self.directory, self.files, self.rows).record()
        finally:
            self.closed = True
        LOG.info(
            "stage close tables=%d committed=%s seconds=%.4f",
            len(self.files),
            commit,
            time.perf_counter() - started,
        )

    @staticmethod
    def open(directory):
        directory = Path(directory).resolve()
        files, rows = {}, {}
        for entry in manifest(directory):
            key = (entry["table"], entry["source"], entry["target"])
            files[key] = tuple(directory / name for name in entry["files"])
            rows[key] = entry["rows"]
        return StageFiles(directory, files, rows)

    @classmethod
    def merge(cls, directory, shards):
        started = time.perf_counter()
        directory = output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files, rows = {}, {}
        for shard in shards:
            stage = cls.open(shard)
            if not stage.directory.is_relative_to(directory):
                raise ValueError(f"shard must be inside merged stage: {stage.directory}")
            for key, paths in stage.files.items():
                files.setdefault(key, []).extend(paths)
                rows[key] = rows.get(key, 0) + stage.rows[key]
        result = StageFiles(directory, {key: tuple(paths) for key, paths in files.items()}, rows)
        result.record()
        LOG.info("stage merge tables=%d seconds=%.4f", len(files), time.perf_counter() - started)
        return result
