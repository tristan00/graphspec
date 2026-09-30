import argparse
import logging
import time
from pathlib import Path

from graphspec.facts.stored import admitted
from graphspec.infra.files import read_json
from graphspec.schema.declaration import Registry
from graphspec.schema.schema import Schema

LOG = logging.getLogger(__name__)


def inspect(path):
    started = time.perf_counter()
    path = Path(path)
    if path.is_dir():
        manifest = admitted(path, read_json(path / "manifest.json"))
        registry = Registry.from_declaration(manifest["spec"])
        result = registry.schema.text(registry.encoding)
    else:
        task = read_json(path)
        if "schema" not in task:
            raise ValueError("the task %s states no schema" % path)
        result = Schema(task["schema"]).text(None)
    LOG.info("inspect exit path=%s seconds=%.3f", path, time.perf_counter() - started)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="a task file or the directory of a bank")
    args = parser.parse_args()
    print(inspect(args.path))
    LOG.info("inspect command exit seconds=%.3f", time.perf_counter() - started)


if __name__ == "__main__":
    main()
