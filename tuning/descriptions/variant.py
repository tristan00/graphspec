import argparse
import logging
import time
from pathlib import Path

from graphspec.descriptions import describe as truth
from graphspec.descriptions import differences as differing
from graphspec.descriptions import effect
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.views.graph import Graph

from . import reach as reaching
from . import reaches
from .restated import Derived, coded, opened, unviewed, viewed

LOG = logging.getLogger(__name__)


def grouped(deriver, config):
    started = time.perf_counter()
    _, steps, spec, _ = deriver.stated(config)
    registry, joining, groups = deriver.fitted(config), deriver.joining(config), deriver.graphs(config)
    for index in groups:
        indices = deriver.groups(config)[index] if joining else [index]
        members = [Graph.load(deriver.view(member, config)).dense() for member in indices]
        beside = deriver.seconds(steps, indices, config, False)
        yield indices, (members, beside, registry, steps, spec, (), deriver.seed, deriver.limits(), deriver.partitions(config))
    LOG.debug("variant grouped exit groups=%d seconds=%.4f", len(groups), time.perf_counter() - started)


def graphed(deriver, config):
    started = time.perf_counter()
    held, found = Derived(deriver, config, False), {}
    for group in deriver.graphs(config):
        found.update(zip(held.loaded(group)[0], held.effects(group)))
        held.group_done()
    views = viewed(deriver, config)
    result = [dict(found[graph["index"]], view=views[graph["index"]]) for graph in deriver.viewed(config).graphs]
    LOG.info("variant graphed exit graphs=%d seconds=%.3f", len(result), time.perf_counter() - started)
    return result


def headed(deriver, config, key):
    started = time.perf_counter()
    _, steps, spec, _ = deriver.stated(config)
    result = dict(
        variant=key,
        seed=deriver.seed,
        device=deriver.device,
        splits=config["splits"],
        settings=specs.materialized(spec),
        unshown=unviewed(deriver, config),
        entries=[effect.entered(step, deriver.registry, spec, config["splits"]) for step in steps],
    )
    LOG.debug("variant headed exit entries=%d seconds=%.4f", len(steps), time.perf_counter() - started)
    return result


def recorded(deriver, config, key):
    started = time.perf_counter()
    result = dict(
        version=truth.VERSION,
        stored_sha256=deriver.source,
        time=truth.modelled(deriver.registry),
        **headed(deriver, config, key),
        graphs=graphed(deriver, config),
    )
    LOG.info("variant recorded exit graphs=%d seconds=%.3f", len(result["graphs"]), time.perf_counter() - started)
    return result


@timed("variant.keyed")
def keyed(deriver, config, key, place):
    return dict(truth.based(place), variant=key, code=coded(deriver, config, []))


def describe(bank, config, statement):
    started = time.perf_counter()
    deriver, key = opened(bank, {}, config, statement)
    place = truth.located(Path(bank) / truth.FOLDER, "bank", "variants", deriver.source)
    result = truth.kept(place, keyed(deriver, config, key, place), lambda: recorded(deriver, config, key), effect.text)
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
    from . import differences

    started = time.perf_counter()
    parser = argparse.ArgumentParser(prog="graphspec describe")
    parser.add_argument("path", help="the directory of a bank")
    parser.add_argument(
        "variant",
        nargs="*",
        metavar="FILE",
        help="the configuration and the statement of a fit, JSON files, which name the variant described, or with "
        "--reach the folder of a search, whose trials name the variants described, or with --difference the names "
        "of two graphs of the bank, or the configuration and the statement of a fit and then those of another; "
        "without them the truth graphs of the bank are described",
    )
    parser.add_argument(
        "--reach",
        action="store_true",
        help="describe the reach of the options of the decisions of the variant, as the realisation the model of "
        "the configuration states puts them before it, or, given the folder of a search, of every trial of it in "
        "one table",
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help="describe what the first of two truth graphs of the bank, or of the variants of two fits, holds that "
        "the second does not, and the second that the first does not, per kind, per relation and per time",
    )
    args = parser.parse_args()
    if args.reach and args.difference:
        parser.error("describe takes --reach or --difference, not both")
    if len(args.variant) not in ((2, 4) if args.difference else (1, 2) if args.reach else (0, 2)):
        parser.error(
            "describe takes a bank, or a bank, the configuration and the statement of a fit, with --reach the "
            "latter or a bank and the folder of a search, and with --difference a bank and the names of two of its "
            "graphs or the configuration and the statement of each of two fits; found %d paths after the bank"
            % len(args.variant)
        )
    if args.difference and len(args.variant) == 2:
        written = differing.text(differing.differences(args.path, *args.variant))
    elif args.difference:
        written = differences.text(differences.differences(args.path, *[read(parser, name) for name in args.variant]))
    elif args.reach and len(args.variant) == 1:
        if not Path(args.variant[0]).is_dir():
            parser.error("%s is no folder; describe --reach reads the folder of a search" % args.variant[0])
        written = reaches.text(reaches.search(args.path, args.variant[0]))
    elif args.reach:
        written = reaching.text(reaching.reach(args.path, *[read(parser, name) for name in args.variant]))
    elif args.variant:
        written = effect.text(describe(args.path, *[read(parser, name) for name in args.variant]))
    else:
        written = truth.text(truth.describe(args.path))
    print(written, end="")
    LOG.info("variant command exit files=%d seconds=%.3f", len(args.variant), time.perf_counter() - started)
