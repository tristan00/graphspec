import logging
import time

from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path

LOG = logging.getLogger(__name__)


def build_main():
    import argparse
    import json

    from graphspec.build.command import script
    from graphspec.build.planning import BuildPlan
    from graphspec.infra.log import configure

    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task")
    source.add_argument("--script")
    parser.add_argument("--sources")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trained", nargs="+", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--memory-bytes", type=int)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.script is not None and (args.sources is None or args.memory_bytes is None):
        parser.error("--script requires --sources and --memory-bytes")
    configure()
    given = script(args.script, args.sources) if args.script is not None else args.task
    plan = None if args.memory_bytes is None else BuildPlan(args.memory_bytes, args.workers, args.shards)
    result = build_bank(given, args.output, args.workers, args.trained, plan=plan, reuse=args.reuse)
    print(json.dumps({key: value for key, value in result.items() if key not in ("graphs", "spec")}, indent=2))
    LOG.info("build_main exit seconds=%.3f", time.perf_counter() - started)


def append_bank(task, output, workers):
    from .bank.append import append

    started = time.perf_counter()
    result = append(task, output, workers)
    LOG.info("append_bank exit seconds=%.3f output=%s", time.perf_counter() - started, output)
    return result


def build_bank(task, output, workers, trained, *, plan=None, reuse=False):
    from graphspec.build.builder import Builder
    from graphspec.build.planning import stated

    from .bank.build import build

    started = time.perf_counter()
    if isinstance(task, Builder):
        plan = stated(plan)
        if workers != plan.workers:
            raise ValueError("bank workers must match the build plan")
        output = output_path(output)
        if output.exists():
            raise FileExistsError("bank output already exists: " + str(output))
        task = task.stage(output.parent, output.name + "-source", plan, reuse=reuse)
    result = build(task, output, workers, trained, plan=plan) if plan is not None else build(task, output, workers, trained)
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


def differences(bank, config, statement, other, other_statement):
    from .descriptions.differences import differences as differed

    started = time.perf_counter()
    result = differed(bank, config, statement, other, other_statement)
    LOG.info("differences exit seconds=%.3f bank=%s", time.perf_counter() - started, bank)
    return result


def reach(bank, config, statement):
    from .descriptions.reach import reach as reached

    started = time.perf_counter()
    result = reached(bank, config, statement)
    LOG.info("reach exit seconds=%.3f bank=%s", time.perf_counter() - started, bank)
    return result


def reach_over(bank, configs, statement):
    from .descriptions.reaches import over

    started = time.perf_counter()
    result = over(bank, configs, statement)
    LOG.info("reach_over exit seconds=%.3f bank=%s", time.perf_counter() - started, bank)
    return result


def reach_search(bank, search):
    from .descriptions.reaches import search as searched

    started = time.perf_counter()
    result = searched(bank, search)
    LOG.info("reach_search exit seconds=%.3f bank=%s search=%s", time.perf_counter() - started, bank, search)
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
