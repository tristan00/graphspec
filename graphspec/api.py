import logging
import time

from .infra.runtime import exclusive, output_path

LOG = logging.getLogger(__name__)


def build_bank(task, output, workers=1):
    from tuning.build import build

    return build(task, output, workers)


def open_bank(path, device=None):
    from tuning.data import Bank

    return Bank(path, device=device)


def configuration(bank, model=None, graph=None, training=None, *, seed):
    from tuning.model_spec import TRAINING
    from tuning.model_spec import configuration as configure

    from .spec import configured

    registry = open_bank(bank).registry
    result = configure(registry, {} if model is None else model, {} if graph is None else graph, seed)
    result.update(configured(TRAINING, {} if training is None else training))
    return result


@exclusive
def fit(bank, config, output, statement, batch_sequences=64):
    from tuning.statement import state
    from tuning.training import fit as train

    from .infra import compute

    started = time.perf_counter()
    stated = state(statement)
    compute.require(stated.device)
    if type(batch_sequences) is not int or batch_sequences < 1:
        raise ValueError("fit batch_sequences must be a positive integer")
    output = output_path(output)
    if output.exists():
        raise FileExistsError("fit output already exists: " + str(output))
    opened = open_bank(bank, stated.device)
    blocks = opened.blocks(0, config["graph"]["time/history"], config)
    output.mkdir(parents=True)
    deadline = time.perf_counter() + stated.seconds
    result = train(bank, output, config, blocks, batch_sequences, deadline, statement)
    LOG.info("fit exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result
