import argparse
import logging
import time
from pathlib import Path

from graphspec.descriptions import describe as truth
from graphspec.descriptions import effect
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, sha256_json
from graphspec.operations import parameters as specs
from graphspec.views.graph import Graph
from tuning.bank.bank import Deriver
from tuning.statement import agreed, state

from . import reach as reaching

LOG = logging.getLogger(__name__)


def graphed(deriver, config):
    started = time.perf_counter()
    _, steps, spec, chosen = deriver.stated(config)
    registry, joining, found = deriver.fitted(config), deriver.joining(config), {}
    for index in deriver.graphs(config):
        indices = deriver.groups(config)[index] if joining else [index]
        members = [Graph.load(deriver.view(member, config)).dense() for member in indices]
        found.update(
            zip(
                indices,
                effect.described(
                    members,
                    deriver.seconds(steps, indices, config, False),
                    registry,
                    steps,
                    spec,
                    chosen,
                    deriver.seed,
                    deriver.limits(),
                    deriver.partitions(config),
                ),
            )
        )
    result = [found[graph["index"]] for graph in deriver.viewed(config).graphs]
    LOG.info("variant graphed exit graphs=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


def recorded(deriver, config, key):
    started = time.perf_counter()
    _, steps, spec, _ = deriver.stated(config)
    result = dict(
        version=truth.VERSION,
        variant=key,
        stored_sha256=deriver.source,
        seed=deriver.seed,
        device=deriver.device,
        splits=config["splits"],
        settings=specs.materialized(spec),
        entries=[effect.entered(step, deriver.registry, spec, config["splits"]) for step in steps],
        graphs=graphed(deriver, config),
    )
    LOG.info("variant recorded exit entries=%d seconds=%.3f", len(steps), time.perf_counter() - started)
    return result


def describe(bank, config, statement):
    started = time.perf_counter()
    truth.manifested(bank)
    stated = state(statement)
    agreed(config, stated)
    deriver = Deriver(bank, stated.device, stated.derivation_seed, RESIDENT)
    key = specs.digest(deriver.identity(config))
    held = Path(bank) / "descriptions" / "variants" / sha256_json(dict(variant=key, version=truth.VERSION))
    result = truth.kept(held, lambda: recorded(deriver, config, key), effect.text)
    LOG.info("variant describe exit variant=%s seconds=%.3f", key[:10], time.perf_counter() - started)
    return result


def read(parser, name):
    started = time.perf_counter()
    if not Path(name).is_file():
        parser.error("%s is no file; describe reads the configuration and the statement of a fit from JSON files" % name)
    result = read_json(Path(name))
    LOG.debug("variant read exit file=%s seconds=%.4f", name, time.perf_counter() - started)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(prog="graphspec describe")
    parser.add_argument("path", help="the directory of a bank")
    parser.add_argument(
        "variant",
        nargs="*",
        metavar="FILE",
        help="the configuration and the statement of a fit, JSON files, which name the variant described; "
        "without them the truth graphs of the bank are described",
    )
    parser.add_argument(
        "--reach",
        action="store_true",
        help="describe the reach of the options of the decisions of the variant, as the realisation the model of "
        "the configuration states puts them before it",
    )
    args = parser.parse_args()
    if len(args.variant) not in (0, 2) or (args.reach and not args.variant):
        parser.error(
            "describe takes a bank, or a bank, the configuration and the statement of a fit, and --reach the "
            "latter; found %d files" % len(args.variant)
        )
    if args.reach:
        written = reaching.text(reaching.reach(args.path, *[read(parser, name) for name in args.variant]))
    elif args.variant:
        written = effect.text(describe(args.path, *[read(parser, name) for name in args.variant]))
    else:
        written = truth.text(truth.describe(args.path))
    print(written, end="")
    LOG.info("variant command exit files=%d seconds=%.3f", len(args.variant), time.perf_counter() - started)
