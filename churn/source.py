import csv as text_csv
import logging
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv

from .schema import DATES, EMPTY

LOG = logging.getLogger(__name__)


@contextmanager
def stream(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
            ]
            if len(names) != 1:
                raise ValueError("archive must contain exactly one CSV")
            with archive.open(names[0]) as source:
                yield source
    else:
        with path.open("rb") as source:
            yield source


def read(path):
    started = time.perf_counter()
    with stream(path) as source:
        header = source.readline().decode("utf-8-sig")
        if not header.strip():
            raise ValueError("CSV must contain a header")
        names = next(text_csv.reader([header]))
        if len(set(names)) != len(names):
            raise ValueError("CSV column names must be unique")
    with stream(path) as source:
        table = csv.read_csv(
            source,
            convert_options=csv.ConvertOptions(
                column_types={name: pa.string() for name in names}, strings_can_be_null=False
            ),
        )
    required = {"policy_key", "client_key", *DATES}
    if required - set(table.column_names):
        raise ValueError("CSV lacks columns: " + ", ".join(sorted(required - set(table.column_names))))
    for key in ("policy_key", "client_key"):
        if any(value.strip().lower() in EMPTY for value in table[key].to_pylist()):
            raise ValueError(key + " must be present")
    if len(set(table["policy_key"].to_pylist())) != table.num_rows:
        raise ValueError(
            "policy_key must be unique; provide policy versions with historical availability explicitly"
        )
    LOG.info("churn read exit rows=%d seconds=%.3f", table.num_rows, time.perf_counter() - started)
    return table
