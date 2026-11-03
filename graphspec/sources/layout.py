import logging
import time
from pathlib import Path

from graphspec.build.readers import READERS
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)
COMPRESSED = (".gz", ".zst", ".bz2", ".xz")


@timed("layout.name")
def named(path):
    name = path.name
    if path.is_file():
        while Path(name).suffix.lower() in COMPRESSED:
            name = name[:-len(Path(name).suffix)]
        if name.lower().endswith(".parquet"):
            name = name[:-8]
        elif Path(name).suffix:
            raise ValueError("a fixed layout holds Parquet files, found " + str(path))
    if not name:
        raise ValueError("a fixed layout table has a nonempty name")
    return name


@timed("layout.catalogue")
def catalogue(place):
    place = Path(place).resolve()
    if not place.is_dir():
        raise ValueError("a fixed layout source is a directory: " + str(place))
    result = {}
    for part in ("nodes", "edges"):
        directory = place / part
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError("the fixed layout %s is a directory" % part)
        for path in sorted(directory.iterdir()):
            name = named(path)
            table = part + "/" + name
            if table in result:
                raise ValueError("the fixed layout declares the table %s twice" % table)
            result[table] = dict(kind=part, name=name, place=path)
    for path in sorted(place.iterdir()):
        if path.name in ("nodes", "edges"):
            continue
        if path.name == "tasks" or path.name.startswith("tasks.parquet"):
            if named(path) != "tasks":
                raise ValueError("the fixed layout task table is named tasks")
            if "tasks" in result:
                raise ValueError("the fixed layout declares the task table twice")
            result["tasks"] = dict(kind="tasks", name="tasks", place=path)
    if not result:
        raise ValueError("the fixed layout holds no nodes, edges or tasks")
    return result


@timed("layout.mapping")
def mapping(place):
    result = {}
    for table, entry in catalogue(place).items():
        result[table] = dict(
            word=entry["kind"], name=entry["name"],
            rows=dict(place=Path(place).resolve(), reader="layout", parameters=dict(table=table)),
        )
    return result


def layout(place, columns, parameters):
    started = time.perf_counter()
    try:
        if set(parameters) != {"table"} or type(parameters["table"]) is not str:
            raise ValueError("a fixed layout reader states the name of its table")
        tables = catalogue(place)
        if parameters["table"] not in tables:
            raise ValueError("the fixed layout has no table named " + parameters["table"])
        yield from READERS.named("parquet")(tables[parameters["table"]]["place"], columns, {})
    finally:
        LOG.debug("layout reader exit seconds=%.4f", time.perf_counter() - started)


layout.catalogue = catalogue
layout.mapping = mapping
READERS.register("layout", layout)
