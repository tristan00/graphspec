import logging
import math
import time

from .infra.runtime import exclusive, output_path

LOG = logging.getLogger(__name__)


def build_bank(task, output, workers=1):
    from tuning.build import build

    return build(task, output, workers)


def open_bank(path):
    from tuning.data import Bank

    return Bank(path)


def configuration(bank, model=None, graph=None, training=None, seed=1):
    from tuning.model_spec import TRAINING
    from tuning.model_spec import configuration as configure

    from .spec import configured

    registry = open_bank(bank).registry
    result = configure(registry, {} if model is None else model, {} if graph is None else graph, seed)
    result.update(configured(TRAINING, {} if training is None else training))
    return result


@exclusive
def fit(bank, config, output, workers=1, seconds=600, batch_sequences=64):
    if type(workers) is not int or workers < 1 or type(batch_sequences) is not int or batch_sequences < 1:
        raise ValueError("fit workers and batch_sequences must be positive integers")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("fit seconds must be a finite positive number")
    from tuning.training import fit as train

    started = time.perf_counter()
    output = output_path(output)
    if output.exists():
        raise FileExistsError("fit output already exists: " + str(output))
    opened = open_bank(bank)
    blocks = opened.blocks(0, config["graph"]["time/history"], config)
    output.mkdir(parents=True)
    result = train(bank, output, config, blocks, batch_sequences, time.perf_counter() + seconds, workers)
    LOG.info("fit exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result
