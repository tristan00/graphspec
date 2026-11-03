import csv as csv_library
import json as json_library
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as arrow_parquet

from graphspec.build.readers import READERS
from graphspec.build.streams import capacity, checked, loaded, opened, yielded
from graphspec.infra.timing import timed

LOG = logging.getLogger(__name__)


@timed("readers.parameters")
def accepted(parameters, names, reader):
    unknown = sorted(set(parameters) - set(names) - {"member"})
    if unknown:
        raise ValueError("the reader %s does not take %s" % (reader, ", ".join(unknown)))


@timed("readers.parquet_budget")
def parquet_budget(metadata, columns):
    widths = dict(BOOLEAN=1, INT32=4, INT64=8, INT96=12, FLOAT=4, DOUBLE=8)
    for at in range(metadata.num_row_groups):
        group = metadata.row_group(at)
        size = 0
        for position in range(group.num_columns):
            column = group.column(position)
            if "*" in columns or any(column.path_in_schema == name or column.path_in_schema.startswith(name + ".")
                                     for name in columns):
                encoded = column.total_uncompressed_size
                if column.physical_type in widths:
                    expanded = column.num_values * widths[column.physical_type]
                elif any(encoding in ("RLE_DICTIONARY", "PLAIN_DICTIONARY") for encoding in column.encodings):
                    expanded = encoded * column.num_values
                else:
                    expanded = encoded
                levels = metadata.schema.column(position).max_definition_level + 1
                size += encoded + expanded + (column.num_values + group.num_rows) * 8 * levels
        checked(size * 4, "decoding a Parquet row group")


def header_lines(stream):
    started = time.perf_counter()
    consumed = 0
    try:
        while True:
            line = stream.readline(capacity() // 64 - consumed + 1)
            if not line:
                break
            consumed += len(line)
            checked(consumed * 64, "reading CSV column names")
            yield line.decode("utf-8-sig" if consumed == len(line) else "utf-8")
    finally:
        LOG.debug("header lines exit bytes=%d seconds=%.6f", consumed, time.perf_counter() - started)


def parquet(place, columns, parameters):
    started = time.perf_counter()
    accepted(parameters, (), "parquet")
    projection = None if "*" in columns else list(columns)
    empty = True
    try:
        if Path(place).is_dir():
            if "member" in parameters:
                raise ValueError("a Parquet folder is not an archive")
            dataset = arrow_dataset.dataset(place, format="parquet")
            schemas = []
            for fragment in dataset.get_fragments():
                parquet_budget(fragment.metadata, columns)
                schemas.append(fragment.physical_schema)
            checked(sum(schema.serialize().size for schema in schemas) * 4, "Parquet folder schemas")
            if schemas:
                dataset = arrow_dataset.dataset(place, format="parquet",
                                                 schema=pa.unify_schemas(schemas, promote_options="permissive"))
            scanner = dataset.scanner(columns=projection, batch_size=min(65536, max(1, capacity() // 256)),
                                      use_threads=False, batch_readahead=0, fragment_readahead=0)
            for batch in scanner.to_batches():
                empty = False
                yield from yielded(pa.Table.from_batches([batch]), columns)
            if empty:
                yield pa.Table.from_batches([], schema=scanner.projected_schema)
        else:
            with opened(place, parameters.get("member")) as stream:
                file = arrow_parquet.ParquetFile(stream)
                parquet_budget(file.metadata, columns)
                for batch in file.iter_batches(batch_size=min(65536, max(1, capacity() // 256)),
                                               columns=projection, use_threads=False):
                    empty = False
                    yield from yielded(pa.Table.from_batches([batch]), columns)
                if empty:
                    yield from yielded(pa.Table.from_batches([], schema=file.schema_arrow), columns)
    finally:
        LOG.debug("parquet exit place=%s seconds=%.6f", place, time.perf_counter() - started)


def separated(place, columns, parameters, delimiter, names):
    started = time.perf_counter()
    if type(delimiter) is not str or len(delimiter) != 1:
        raise ValueError("a delimited source states a one-character delimiter")
    if names is not None and (not isinstance(names, (list, tuple)) or not names
                              or any(type(name) is not str or not name for name in names)
                              or len(set(names)) != len(names)):
        raise ValueError("a headerless source states distinct nonempty column names in order")
    options = arrow_csv.ReadOptions(use_threads=False, block_size=max(1, min(1 << 20, capacity() // 8)),
                                    column_names=[] if names is None else list(names))
    try:
        with opened(place, parameters.get("member")) as stream:
            fields = names
            if fields is None:
                fields = next(csv_library.reader(header_lines(stream), delimiter=delimiter), None)
                stream.seek(0)
                if not fields:
                    raise ValueError("a CSV source has a header row naming its columns")
            if len(set(fields)) != len(fields) or any(not name for name in fields):
                raise ValueError("a CSV source has distinct nonempty column names")
            convert = arrow_csv.ConvertOptions(include_columns=None if "*" in columns else list(columns),
                                               column_types={name: pa.string() for name in fields},
                                               null_values=[""], strings_can_be_null=True)
            reader = arrow_csv.open_csv(stream, read_options=options,
                                        parse_options=arrow_csv.ParseOptions(delimiter=delimiter),
                                        convert_options=convert)
            empty = True
            for batch in reader:
                empty = False
                yield from yielded(pa.Table.from_batches([batch]), columns)
            if empty:
                yield pa.Table.from_batches([], schema=reader.schema)
    finally:
        LOG.debug("separated exit place=%s seconds=%.6f", place, time.perf_counter() - started)


@timed("readers.csv")
def csv(place, columns, parameters):
    accepted(parameters, (), "csv")
    return separated(place, columns, parameters, ",", None)


@timed("readers.delimited")
def delimited(place, columns, parameters):
    accepted(parameters, ("delimiter", "names"), "delimited")
    if "delimiter" not in parameters or "names" not in parameters:
        raise ValueError("a delimited source requires delimiter and names, with names=None for a header")
    return separated(place, columns, parameters, parameters["delimiter"], parameters["names"])


@timed("readers.json_records")
def json_records(text, path):
    decoder, offset, documents = json_library.JSONDecoder(), 0, []
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        record, offset = decoder.raw_decode(text, offset)
        documents.append(record)
    if path is not None:
        if len(documents) != 1:
            raise ValueError("a JSON records path names a list inside one document")
        if type(path) is not str or not path:
            raise ValueError("a JSON records path is a nonempty dotted field name")
        records = documents[0]
        for name in path.split("."):
            if not isinstance(records, dict) or name not in records:
                raise ValueError("the JSON records path %s does not exist" % path)
            records = records[name]
    elif len(documents) == 1 and isinstance(documents[0], list):
        records = documents[0]
    else:
        records = documents
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("a JSON source holds a list of records or one record per line")
    return records


def json(place, columns, parameters):
    started = time.perf_counter()
    accepted(parameters, ("records",), "json")
    try:
        with opened(place, parameters.get("member")) as stream:
            content = loaded(stream, "reading a JSON document")
        checked(len(content) * 64, "decoding a JSON document")
        records = json_records(content.decode("utf-8-sig"), parameters.get("records"))
        names = list(dict.fromkeys(name for record in records for name in record)) if "*" in columns else list(columns)
        missing = [name for name in columns if name != "*" and records and not any(name in record for record in records)]
        if missing:
            raise ValueError("the JSON source lacks the declared columns %s" % ", ".join(missing))
        checked(len(content) * 4, "converting JSON records")
        table = pa.table({name: pa.array([record.get(name) for record in records]) for name in names})
        yield from yielded(table, columns)
    finally:
        LOG.debug("json exit place=%s seconds=%.6f", place, time.perf_counter() - started)


READERS.register("parquet", parquet)
READERS.register("csv", csv)
READERS.register("delimited", delimited)
READERS.register("json", json)
