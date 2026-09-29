import argparse
import logging
import time

import numpy as np

from graphspec.infra.files import JsonCache, read_json, write_json
from graphspec.infra.log import configure
from graphspec.infra.runtime import output_path

from . import measures
from .catalog import TRACKED
from .data import Bank
from .records import Run, Status

LOG = logging.getLogger(__name__)


def track(bank, indices, values):
    started = time.perf_counter()
    span = float(bank.taus[bank.split["train_indices"]].max())
    order = np.lexsort((bank.taus[indices], bank.series[indices], bank.query_of[indices]))
    rows, values = indices[order], values[order]
    key = np.stack((bank.query_of[rows], bank.series[rows]), 1)
    start = np.r_[np.flatnonzero(np.r_[True, (key[1:] != key[:-1]).any(1)]), len(rows)]
    found = {name: {key: [] for key in TRACKED} for name in bank.series_names}
    for first, last in zip(start[:-1], start[1:]):
        series = bank.series[rows[first]]
        scale = float(bank.scale[series])
        points = rows[first:last]
        truth = measures.Truth(
            bank.taus[points].astype(np.float64), bank.targets[points].astype(np.float64) / scale, span
        )
        scored = measures.compare(truth, values[first:last] / scale, None)[0]
        for key in TRACKED:
            if key in scored:
                found[bank.series_names[series]][key].append(scored[key])
    LOG.info("track exit seconds=%.3f paths=%d", time.perf_counter() - started, len(start) - 1)
    return {
        name: {key: float(np.nanmean(entries)) for key, entries in kept.items()}
        for name, kept in found.items()
    }


def backfill(directory):
    started = time.perf_counter()
    run = Run(output_path(directory), JsonCache())
    bank = Bank(run.plan["bank"])
    done = 0
    for trial in run.trials(selection=False):
        if trial.status != Status.COMPLETE:
            continue
        result = read_json(trial.directory / "result.json")
        if all("measures" in entry for entry in result["series"].values()):
            continue
        with np.load(trial.directory / "predictions.npz") as saved:
            found = track(bank, saved["row_index"], saved["prediction"])
        for name, entry in result["series"].items():
            entry["measures"] = found[name]
        write_json(trial.directory / "result.json", result)
        done += 1
    LOG.info("backfill exit seconds=%.3f trials=%d", time.perf_counter() - started, done)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    configure()
    backfill(args.run)


if __name__ == "__main__":
    main()
