import argparse
import logging
import time
from datetime import date
from pathlib import Path

from graphspec import Stage, write_task
from graphspec.infra.features import token
from graphspec.infra.files import sha256_file, write_json
from graphspec.infra.log import configure
from graphspec.infra.runtime import exclusive, output_path

from .schema import KNOWN_FIELDS, NUMBERS, TARGET
from .snapshot import active_clients, anniversary, outcomes
from .source import read
from .truth import truth

LOG = logging.getLogger(__name__)


def split(key, fraction, seed):
    return "validation" if token(str(seed) + ":client:" + key) / 2**64 < fraction else "train"


@exclusive
def build(
    source,
    output,
    prediction,
    data_through,
    known_fields=KNOWN_FIELDS,
    name="churn",
    features=32,
    validation_fraction=0.2,
    seed=1,
):
    started = time.perf_counter()
    prediction, data_through = date.fromisoformat(str(prediction)), date.fromisoformat(str(data_through))
    evaluation = anniversary(prediction)
    if data_through < evaluation:
        raise ValueError(
            "data-through date must cover evaluation; future scheduled policies do not establish outcome coverage"
        )
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    output = output_path(output)
    stage_path = output / (name + ".staging")
    if stage_path.exists() or (output / (name + ".task.json")).exists():
        raise FileExistsError("churn output already exists; choose a new name or output")
    table = read(source)
    graph = truth(table, prediction, known_fields, features)
    cohort = sorted(active_clients(table, prediction))
    y = outcomes(table, evaluation, cohort)
    partitions = [split(key, validation_fraction, seed) for key in cohort]
    for partition in ("train", "validation"):
        chosen = [value for value, member in zip(y, partitions) if member == partition]
        if len(chosen) < 2 or len(set(chosen)) < 2:
            raise ValueError(
                partition + " needs at least two clients and both outcomes for regression scoring"
            )
    graph.labels("client", cohort, y, partitions, target=TARGET)
    output.mkdir(parents=True, exist_ok=True)
    with Stage(stage_path) as stage:
        graph.write(stage)
    task = write_task(output / (name + ".task.json"), name, stage_path, TARGET)
    write_json(
        output / (name + ".build.json"),
        dict(
            source=str(Path(source).resolve()),
            source_sha256=sha256_file(source),
            prediction=str(prediction),
            evaluation=str(evaluation),
            data_through=str(data_through),
            known_fields=list(known_fields),
            rows=table.num_rows,
            nodes={kind: len(members) for kind, members in graph.lookup.items()},
            facts=graph.sequence,
            cohort=len(cohort),
            retained=int(sum(y)),
            train=partitions.count("train"),
            validation=partitions.count("validation"),
            validation_fraction=validation_fraction,
            seed=seed,
            features=features,
            seconds=time.perf_counter() - started,
            numeric_features=NUMBERS,
            excluded_features=["raw cancellation flag", "future end dates", "future cancellation outcomes"],
            availability="descriptive snapshot attributes accepted; cancellation and ended-history features require an end strictly before prediction",
        ),
    )
    LOG.info(
        "churn build exit target=%s rows=%d seconds=%.3f", TARGET, len(cohort), time.perf_counter() - started
    )
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--data-through", required=True)
    parser.add_argument(
        "--known-fields",
        nargs="+",
        default=["all"],
        help="descriptive graph fields; all uses the designed TX schema, none keeps client and policy nodes",
    )
    parser.add_argument("--name", default="churn")
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    configure()
    selections = {("all",): KNOWN_FIELDS, ("none",): ()}
    requested = tuple(args.known_fields)
    fields = selections.get(requested, requested)
    print(
        build(
            args.source,
            args.output,
            args.prediction,
            args.data_through,
            fields,
            args.name,
            args.features,
            args.validation_fraction,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
