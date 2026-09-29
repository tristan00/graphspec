import argparse
import logging
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .infra.database import connect, copy
from .infra.runtime import exclusive, output_path
from .infra.staging import Stage

LOG = logging.getLogger(__name__)
TYPES = {
    pa.int64(): "INT64",
    pa.int32(): "INT32",
    pa.float32(): "FLOAT",
    pa.float64(): "DOUBLE",
    pa.bool_(): "BOOLEAN",
    pa.string(): "STRING",
}


def typename(dtype):
    if pa.types.is_fixed_size_list(dtype) and dtype.value_type == pa.float32():
        return "FLOAT[%d]" % dtype.list_size
    if pa.types.is_list(dtype) and dtype.value_type == pa.string():
        return "STRING[]"
    if dtype not in TYPES:
        raise ValueError("unsupported graph property type: " + str(dtype))
    return TYPES[dtype]


def identifier(name):
    return "`" + name.replace("`", "``") + "`"


def declarations(stage):
    tables = {}
    for (name, source, target), paths in stage.files.items():
        schema = pq.read_schema(paths[0])
        properties = [
            (field.name, typename(field.type)) for field in schema if field.name not in ("from", "to")
        ]
        entry = tables.setdefault(name, dict(node=source is None, properties=properties, pairs=[]))
        if entry["properties"] != properties or entry["node"] != (source is None):
            raise ValueError("inconsistent table schema for " + name)
        if source is not None:
            entry["pairs"].append((source, target))
    for name, entry in sorted(tables.items(), key=lambda item: not item[1]["node"]):
        fields = [
            identifier(key) + " " + dtype + (" PRIMARY KEY" if entry["node"] and key == "id" else "")
            for key, dtype in entry["properties"]
        ]
        if not entry["node"]:
            fields = [
                "FROM %s TO %s" % (identifier(source), identifier(target))
                for source, target in entry["pairs"]
            ] + fields
        yield "CREATE %s TABLE %s(%s)" % (
            "NODE" if entry["node"] else "REL",
            identifier(name),
            ", ".join(fields),
        )


@exclusive
def export(staging, output, buffer_pool_size=1 << 30):
    started = time.perf_counter()
    output = output_path(output)
    if output.exists():
        raise FileExistsError("database exists: " + str(output))
    stage = Stage.open(staging)
    statements = list(declarations(stage))
    output.parent.mkdir(parents=True, exist_ok=True)
    store, connection = connect(output, read_only=False, buffer_pool_size=buffer_pool_size)
    try:
        for statement in statements:
            connection.execute(statement)
        copy(connection, stage)
    finally:
        try:
            connection.close()
        finally:
            store.close()
    LOG.info("database export exit tables=%d seconds=%.3f", len(statements), time.perf_counter() - started)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.staging, args.output))


if __name__ == "__main__":
    main()
