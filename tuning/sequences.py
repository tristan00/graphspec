import logging
import time
from pathlib import Path

import numpy as np

from graphspec.infra.files import JsonCache, read_json, write_json
from graphspec.infra.runtime import output_path

from . import objective
from .data import Bank
from .records import Run, Status

LOG = logging.getLogger(__name__)
POINTS = ("graph", "step", "turn", "series", "tau", "target", "held")


def ranked(run):
    started = time.perf_counter()
    complete = [trial for trial in run.trials(selection=False) if trial.status == Status.COMPLETE]
    complete.sort(key=lambda trial: objective.value(trial.outcome.data, run.plan))
    LOG.info("ranked exit seconds=%.3f complete=%d", time.perf_counter() - started, len(complete))
    return complete


def points(bank):
    started = time.perf_counter()
    names = sorted({row["graph"] for row in bank.rows})
    code = {name: index for index, name in enumerate(names)}
    graph = np.asarray([code[row["graph"]] for row in bank.rows], np.int64)
    step = np.asarray([row["step"] for row in bank.rows], np.int64)
    turn = np.asarray([row["turn"] for row in bank.rows], np.int64)
    order = np.lexsort((bank.taus, bank.series, step, graph))
    order = order[bank.partitions[order] < 2]
    table = dict(
        graph=graph[order],
        step=step[order],
        turn=turn[order],
        series=bank.series[order],
        tau=bank.taus[order].astype(np.float64),
        target=bank.targets[order].astype(np.float64),
        held=bank.partitions[order] == 1,
    )
    LOG.info(
        "points exit seconds=%.3f rows=%d graphs=%d", time.perf_counter() - started, len(order), len(names)
    )
    return names, order, table


def collect(run_directory, output, top):
    started = time.perf_counter()
    run = Run(output_path(run_directory), JsonCache())
    output = output_path(output)
    output.mkdir(parents=True)
    bank = Bank(run.plan["bank"])
    if bank.level != "path":
        raise ValueError("sequences need a bank of path targets")
    chosen = ranked(run)[:top]
    if not chosen:
        raise ValueError("the run has no completed trial")
    names, order, table = points(bank)
    held = order[table["held"]]
    if not np.array_equal(np.sort(held), np.asarray(bank.split["validation_indices"])):
        raise ValueError("held points differ from the frozen validation partition")
    predictions = np.empty((len(chosen), len(held)), np.float64)
    for index, trial in enumerate(chosen):
        with np.load(trial.directory / "predictions.npz") as saved:
            rows, values, targets = saved["row_index"], saved["prediction"], saved["target"]
        if not np.array_equal(rows, np.sort(held)) or not np.array_equal(targets, bank.targets[rows]):
            raise ValueError("saved predictions differ from the bank: " + str(trial.directory))
        predictions[index] = values[np.searchsorted(rows, held)]
    np.savez_compressed(output / "sequences.npz", prediction=predictions, **table)
    write_json(
        output / "sequences.json",
        dict(
            run=str(run.directory),
            bank=str(bank.directory),
            supervision_sha256=run.plan["supervision_sha256"],
            graphs=names,
            series=bank.series_names,
            center=bank.center.tolist(),
            scale=bank.scale.tolist(),
            weights=run.plan["objective"],
            models=[
                dict(
                    name=trial.item["directory"],
                    description=trial.item["description"],
                    selector=trial.item["selector"],
                    view=trial.config["model"]["view"],
                    objective=objective.value(trial.outcome.data, run.plan),
                    series={name: entry["val_loss"] for name, entry in trial.outcome.data["series"].items()},
                )
                for trial in chosen
            ],
        ),
    )
    LOG.info(
        "collect exit seconds=%.3f models=%d held_points=%d output=%s",
        time.perf_counter() - started,
        len(chosen),
        len(held),
        output,
    )
    return output


class Sequences:
    def __init__(self, directory):
        started = time.perf_counter()
        directory = Path(directory)
        self.about = read_json(directory / "sequences.json")
        with np.load(directory / "sequences.npz") as saved:
            table = {name: saved[name] for name in POINTS}
            self.prediction = saved["prediction"]
        self.series = self.about["series"]
        self.scale = np.asarray(self.about["scale"], np.float64)
        self.models = [model["name"] for model in self.about["models"]]
        held = table.pop("held")
        self.train = {name: values[~held] for name, values in table.items()}
        self.held = {name: values[held] for name, values in table.items()}
        for part in (self.train, self.held):
            part.update(paths(part))
        LOG.info(
            "sequences exit seconds=%.3f models=%d paths=%d decisions=%d",
            time.perf_counter() - started,
            len(self.models),
            len(self.held["start"]) - 1,
            int(self.held["decision"].max()) + 1,
        )


def paths(part):
    key = np.stack((part["graph"], part["step"], part["series"]), 1)
    first = np.r_[True, (key[1:] != key[:-1]).any(1)]
    start = np.r_[np.flatnonzero(first), len(key)]
    moment = key[start[:-1], :2]
    decision = np.unique(moment, axis=0, return_inverse=True)[1].reshape(-1)
    return dict(
        start=start,
        decision=decision,
        path_graph=key[start[:-1], 0],
        path_step=key[start[:-1], 1],
        path_series=key[start[:-1], 2],
        path_turn=part["turn"][start[:-1]],
    )
