import logging
import time

from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path

LOG = logging.getLogger(__name__)


def build_bank(task, output, workers, trained):
    from .bank.build import build

    started = time.perf_counter()
    result = build(task, output, workers, trained)
    LOG.info("build_bank exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result


def open_bank(path):
    from .bank.bank import Bank

    started = time.perf_counter()
    result = Bank(path)
    LOG.info("open_bank exit seconds=%.3f path=%s", time.perf_counter() - started, path)
    return result


def configuration(bank, model=None, training=None, *, pipeline, graph, seed, splits):
    from graphspec.operations.parameters import configured

    from .families.model_spec import TRAINING
    from .families.model_spec import configuration as configure

    started = time.perf_counter()
    registry = open_bank(bank).registry
    result = configure(registry, {} if model is None else model, pipeline, graph, seed, splits)
    result.update(configured(TRAINING, {} if training is None else training))
    LOG.info("configuration exit seconds=%.3f", time.perf_counter() - started)
    return result


def describe(bank, config, statement):
    from .descriptions.variant import describe as described

    started = time.perf_counter()
    result = described(bank, config, statement)
    LOG.info("describe exit seconds=%.3f bank=%s", time.perf_counter() - started, bank)
    return result


@exclusive
def fit(bank, config, output, statement):
    from graphspec.infra import compute

    from .fits.training import fit as train
    from .statement import state

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
