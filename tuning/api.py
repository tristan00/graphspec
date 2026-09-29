import logging
import time

from graphspec.infra.locks import exclusive
from graphspec.infra.runtime import output_path

LOG = logging.getLogger(__name__)


def build_bank(task, output, workers, trained):
    from .build import build

    started = time.perf_counter()
    result = build(task, output, workers, trained)
    LOG.info("build_bank exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result


def open_bank(path):
    from .data import Bank

    started = time.perf_counter()
    result = Bank(path)
    LOG.info("open_bank exit seconds=%.3f path=%s", time.perf_counter() - started, path)
    return result


def configuration(bank, model=None, graph=None, training=None, *, seed, splits):
    from graphspec.spec import configured

    from .model_spec import TRAINING
    from .model_spec import configuration as configure

    started = time.perf_counter()
    registry = open_bank(bank).registry
    result = configure(registry, {} if model is None else model, {} if graph is None else graph, seed, splits)
    result.update(configured(TRAINING, {} if training is None else training))
    LOG.info("configuration exit seconds=%.3f", time.perf_counter() - started)
    return result


@exclusive
def fit(bank, config, output, statement):
    from graphspec.infra import compute

    from .statement import state
    from .training import fit as train

    started = time.perf_counter()
    stated = state(statement)
    compute.require(stated.device)
    output = output_path(output)
    if output.exists():
        raise FileExistsError("fit output already exists: " + str(output))
    opened = open_bank(bank)
    blocks = opened.blocks(0, config["graph"]["time/history"], config)
    output.mkdir(parents=True)
    deadline = time.perf_counter() + stated.seconds
    result = train(bank, output, config, blocks, stated.batch_sequences, deadline, stated)
    LOG.info("fit exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result
