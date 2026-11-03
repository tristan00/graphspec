import argparse
import json
import logging
import time
from pathlib import Path

from graphspec.facts.stored import admitted
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.declaration import Registry
from graphspec.schema.schema import Schema

LOG = logging.getLogger(__name__)


@timed("inspect.builder")
def builder_text(builder, plan=None):
    from graphspec.build.planning import stated
    from graphspec.build.readers import Source, proposals
    from graphspec.build.writing import canonical_records

    sources = [(name, declaration) for name, declaration in builder.declarations.items() if isinstance(declaration, Source)]
    if sources and plan is None:
        raise ValueError("inspect of source declarations requires an explicit BuildPlan")
    if sources:
        plan = stated(plan)
    result = ["source %s: %s" % (name, json.dumps(proposals(declaration, builder.sources, plan, name), sort_keys=True))
              for name, declaration in sources]
    if builder.stated is None and sources:
        return "\n".join(result)
    case = canonical_records(builder.records(snapshot=False))
    schema = Schema(case["schema"])
    result.append(schema.text(None))
    for graph in case["graphs"]:
        counts = dict(nodes={kind: len(keys) for kind, keys in graph["nodes"].items()},
                      observations={kind: 0 for kind in schema.entities()},
                      edges={kind: 0 for kind in schema.edges},
                      tasks={kind: dict(splits={}, without_split=0) for kind in schema.tasks["kinds"]})
        for field, name in (("values", "kind"), ("edges", "table")):
            target = counts["observations" if field == "values" else "edges"]
            for statement in graph[field]:
                kind = statement[name]
                target[kind] = target.get(kind, 0) + len(next(iter(statement["rows"].values())))
        for kind, split in zip(graph["tasks"]["kind"], graph["tasks"]["split"]):
            target = counts["tasks"][kind]
            if split is None:
                target["without_split"] += 1
            else:
                target["splits"][split] = target["splits"].get(split, 0) + 1
        result.append("graph %s counts: %s" % (graph["name"], json.dumps(counts, sort_keys=True)))
    result.append("skipped references: 0")
    return "\n".join(result)


def inspect(path, *, plan=None):
    from graphspec.build.builder import Builder

    started = time.perf_counter()
    if isinstance(path, Builder):
        result = builder_text(path, plan)
        LOG.info("inspect builder exit seconds=%.3f", time.perf_counter() - started)
        return result
    path = Path(path)
    if path.is_dir():
        manifest = admitted(path, read_json(path / "manifest.json"))
        registry = Registry.from_declaration(manifest["spec"])
        result = registry.schema.text(registry.encoding)
        result += "\n" + "\n".join("graph %s counts: %s" % (graph["name"], json.dumps(graph["inspection"], sort_keys=True))
                                      for graph in manifest["graphs"])
        result += "\nskipped references: %d" % manifest["skipped_references"]
    else:
        task = read_json(path)
        if "schema" not in task:
            raise ValueError("the task %s states no schema" % path)
        result = Schema(task["schema"]).text(None)
        if "staging" in task:
            manifest = path.parent / task["staging"] / "build.json"
            if manifest.exists():
                built = read_json(manifest)
                result += "\n" + "\n".join("graph %s counts: %s" % (graph["name"], json.dumps(graph["counts"], sort_keys=True))
                                              for graph in built["graphs"])
                result += "\nskipped references: %d" % built["skipped_references"]
    LOG.info("inspect exit path=%s seconds=%.3f", path, time.perf_counter() - started)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="a task file or the directory of a bank")
    parser.add_argument("--sources", help="source folder for a Python file exporting builder")
    parser.add_argument("--memory-bytes", type=int, help="explicit memory budget for source inspection")
    parser.add_argument("--workers", type=int, default=1, help="workers sharing the source inspection budget")
    args = parser.parse_args()
    if args.sources is not None:
        from graphspec.build.command import script

        given = script(args.path, args.sources)
    else:
        given = args.path
    from graphspec.build.planning import BuildPlan

    plan = None if args.memory_bytes is None else BuildPlan(args.memory_bytes, args.workers)
    print(inspect(given, plan=plan))
    LOG.info("inspect command exit seconds=%.3f", time.perf_counter() - started)


if __name__ == "__main__":
    main()
