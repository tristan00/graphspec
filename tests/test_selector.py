import argparse
import importlib
import logging
import re
import shutil
import sys
import time
import weakref
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import support
from optuna.distributions import IntDistribution

from graphspec import OPERATIONS, Operation, Phase
from graphspec.infra.files import read_json, write_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.operation import Refused
from graphspec.representations.groups import lags
from tuning.bank.bank import Bank
from tuning.families.model_spec import space
from tuning.searches import selector
from tuning.searches.run import run
from tuning.searches.selector import distribution
from tuning.searches.space import Space
from tuning.statement import state

LOG = logging.getLogger(__name__)
TEMPLATE = support.template()
PIPELINE = str(Path(__file__).parent / "pipeline.json")
LATER = ("contract", "derive", "nearby", "consecutive", "neighbourhood", "nodes")
GROUPED = "pipeline/graphspec.contract"
NEARBY = {"pipeline/graphspec.nearby": True, "pipeline/graphspec.nearby/nearby/kinds": "item:item"}
CONTRACTED = {GROUPED: True, GROUPED + "/contract/grouping": "item:v0"}
FORCED = (
    "the bounds hold the entry graphspec.contract of the template present, and it removes the kind item, which "
    "the entry graphspec.nearby after it, held present, names"
)


DEFERRING = "try:\n    import graphspec_absent_module\nexcept ImportError as error:\n    DEFERRED = error\n"


class Owner:
    pass


@timed("selector_tests.reached")
def reached(importing, name, owner):
    return importing(name)


def random_suggest(*arguments):
    return selector.random_suggest(*arguments, TEMPLATE)


def suggest(*arguments):
    return selector.suggest(*arguments, TEMPLATE)


def test_the_template_of_the_suite_is_the_seven_in_their_order():
    assert read_json(PIPELINE) == TEMPLATE


def test_the_search_imports_optuna_on_a_thread_of_its_own_so_no_frame_of_its_caller_outlives_the_import(tmp_path, monkeypatch):
    from tuning.searches import tpe

    started, kept, ways = time.perf_counter(), {}, dict(plainly=importlib.import_module, apart=tpe.imported)
    for way in ways:
        (tmp_path / ("deferring_" + way + ".py")).write_text(DEFERRING, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for way, importing in ways.items():
        owner = Owner()
        held = weakref.ref(owner)
        assert reached(importing, "deferring_" + way, owner).DEFERRED.name == "graphspec_absent_module"
        del owner
        kept[way] = held() is not None
        sys.modules.pop("deferring_" + way)
    assert kept == dict(plainly=True, apart=False)
    assert tpe.imported("optuna") is sys.modules["optuna"]
    LOG.info("optuna imported apart exit seconds=%.3f", time.perf_counter() - started)


def test_numeric_parameters_use_full_integer_domains(bank):
    registry = Bank(bank).registry
    parameters = space(registry) + support.parameters(registry, specs.GROUPS)
    expected = {
        "snapshot/heads": (1, 4),
        "sequence/heads": (1, 4),
        "memory/neighbors": (4, 64),
        "projection/neighbors": (4, 64),
        "sequence/neighbors": (4, 64),
        "projection/width": (16, 256),
        "sequence/patch": (1, 16),
        "sequence/state": (4, 64),
        "tree/neighbors": (4, 20),
        "walks/count": (1, 32),
        "affinity/buffer": (1, 32),
        "keys/table": (4096, 65536),
        "derive/project_cap": (0, 64),
        "coupling/max_lag": (2, 64),
        "events/max_lag": (2, 16),
    }
    assert not [p.name for p in parameters if p.kind == "choice" and type(p.options[0]) in (int, float)]
    checked = set()
    for parameter in parameters:
        if parameter.name not in expected:
            continue
        checked.add(parameter.name)
        low, high = expected[parameter.name]
        domain = distribution(parameter)
        assert isinstance(domain, IntDistribution)
        assert (domain.low, domain.high, domain.step, domain.log) == (low, high, 1, False)
        assert all(parameter.check(value) == value for value in range(low, high + 1))
        rng = np.random.default_rng(81)
        samples = [parameter.uniform(rng) for _ in range(256)]
        assert all(type(value) is int and low <= value <= high for value in samples)
        assert any(value & (value - 1) for value in samples)
    assert checked == expected.keys()


def test_random_and_tpe_accept_non_power_of_two_settings(bank):
    registry = Bank(bank).registry
    bounds = {
        "view": "sequence",
        "sequence/operator": "transformer",
        "sequence/heads": 3,
        "sequence/neighbors": 13,
        "sequence/patch": 7,
        "keys/embedding": True,
        "keys/table": 12345,
    }
    random, _ = random_suggest(registry, 10, 11, bounds, 600)
    records = []
    for index in range(12):
        config, _ = random_suggest(registry, index, index, bounds, 600)
        records.append(dict(config=config, status="complete", trial=str(index), objective=float(index)))
    tpe, metadata = suggest(registry, records, [], 12, 13, bounds, 600)
    assert metadata["phase"] == "tpe" and metadata["observations"] == 12
    assert all(config["model"][key] == value for config in (random, tpe) for key, value in bounds.items())


def test_numeric_lag_limits_are_used_exactly():
    for maximum in range(2, 65):
        selected = lags(maximum)
        assert selected[-1] == maximum
        assert tuple(sorted(set(selected))) == selected
        assert selected[:-1] == tuple(2**i for i in range((maximum - 1).bit_length()))


def arguments(bank, output, **overrides):
    values = dict(
        bank=str(bank),
        output=str(output),
        device="cpu",
        seed=3,
        derivation_seed=support.SEED,
        preparation_seconds=300,
        batch_sequences=8,
        continuous=False,
        trials=None,
        seconds=10,
        selector=None,
        random_trials=None,
        random_params=None,
        mix=None,
        objective=None,
        bounds=None,
        pipeline=None,
        workers=1,
        variant_gb=1,
        trained=["train"],
        scored=["validation"],
        sees=["train=", "validation="],
        loss=["number=squared"],
        measures=["mse", "r2"],
        alike=1.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


FRESH = dict(selector="random", random_trials=10, mix=["random=100"], random_params=[], pipeline=PIPELINE)


def test_hidden_covers_every_integer_and_uses_integer_tpe_distribution(bank):
    registry = Bank(bank).registry
    hidden = next(parameter for parameter in space(registry) if parameter.name == "hidden")
    domain = distribution(hidden)
    assert isinstance(domain, IntDistribution)
    assert (domain.low, domain.high, domain.step, domain.log) == (1, 128, 1, False)
    rng = np.random.default_rng(73)
    assert {hidden.uniform(rng) for _ in range(4096)} == set(range(1, 129))
    bounds = {"hidden": {"low": 37, "high": 37}}
    random, _ = random_suggest(registry, 10, 11, bounds, 600)
    tpe, _ = suggest(registry, [], [], 12, 13, bounds, 600)
    assert random["model"]["hidden"] == tpe["model"]["hidden"] == 37


def test_tpe_learns_from_every_recorded_trial_inside_the_bounds(bank):
    registry = Bank(bank).registry
    records = []
    for index in range(14):
        config, _ = random_suggest(registry, 100 + index, 200 + index, {}, 600)
        config["epochs"] = 1 + index % 4
        status = "complete" if index % 5 else "failed"
        records.append(
            dict(config=config, status=status, trial="trial_%06d_random" % index, objective=1.0 + index)
        )
    bounds = {"epochs": {"low": 1, "high": 2}}
    completed = [entry for entry in records if entry["status"] == "complete"]
    failures = [dict(entry, objective=None) for entry in records if entry["status"] != "complete"]
    config, selection = suggest(registry, completed, failures, 7, 8, bounds, 600)
    inside = [entry for entry in records if entry["config"]["epochs"] <= 2]
    assert selection["observations"] == sum(entry["status"] == "complete" for entry in inside)
    assert selection["failed_observations"] == sum(entry["status"] != "complete" for entry in inside)
    assert sorted(entry["trial"] for entry in selection["excluded"]) == sorted(
        entry["trial"] for entry in records if entry["config"]["epochs"] > 2
    )
    assert all(entry["parameters"] == ["epochs"] for entry in selection["excluded"])
    assert 1 <= config["epochs"] <= 2
    assert selection["phase"] == "random_startup" and len(inside) < 10

    config, selection = suggest(registry, completed, failures, 7, 8, {}, 600)
    assert (
        selection["excluded"] == []
        and selection["observations"] == len(completed)
        and selection["failed_observations"] == len(failures)
    )
    assert selection["phase"] == "tpe"


def test_tpe_excludes_recorded_trials_that_lack_an_active_parameter(bank):
    registry = Bank(bank).registry
    records = []
    for index in range(3):
        config, _ = random_suggest(registry, 300 + index, 400 + index, {"graphs/join": "single"}, 600)
        records.append(dict(config=config, status="complete", trial="trial_%d" % index, objective=1.0))
    del records[0]["config"]["graph"]["graphs/join_size"]
    _, selection = suggest(registry, records, [], 7, 8, {}, 600)
    assert selection["excluded"] == [
        dict(trial="trial_0", status="complete", parameters=["graphs/join_size"])
    ]
    assert selection["observations"] == 2


def contracting(opened):
    started = time.perf_counter()
    second = opened.manifest["graphs"][1]["name"]
    result = [dict(operation="graphspec." + name) for name in LATER] + [
        dict(operation="graphspec.contract", label="again"),
        dict(operation="graphspec.merge", graph=second),
        dict(operation="graphspec.nearby", label="late"),
    ]
    LOG.debug("contracting exit entries=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def test_a_search_offers_no_pipeline_its_statement_refuses_and_bounds_that_leave_nothing_stop_naming_the_entry(
    bank,
):
    started = time.perf_counter()
    opened = Bank(bank)
    registry, template = opened.registry, contracting(opened)
    rng, offered = np.random.default_rng(3), Space(registry, {}, 600, [], template)
    drawn = [offered.draw(rng, 1, {})["pipeline"] for _ in range(60)]
    records = []
    for seed in range(12):
        config, _ = selector.random_suggest(registry, seed, seed, {}, 600, template)
        records.append(dict(config=config, status="complete", trial="trial_%d" % seed, objective=float(seed)))
    refused = dict(records[0], config=dict(records[0]["config"]), trial="refused")
    refused["config"]["pipeline"] = support.pipelined(
        registry,
        {"contract/grouping": "item:v0", "nearby/kinds": "item:item", "nearby/encoder": "standard"},
        ("graphspec.contract", "graphspec.nearby"),
    )[0]
    with pytest.raises(ValueError, match="which the entry graphspec.contract before it removed"):
        pipeline.resolved(registry, refused["config"]["pipeline"])
    config, selection = selector.suggest(registry, records + [refused], [], 7, 8, {}, 600, template)
    assert selection["phase"] == "tpe" and selection["excluded"] == [
        dict(trial="refused", status="complete", parameters=["pipeline"])
    ]
    drawn += [entry["config"]["pipeline"] for entry in records] + [config["pipeline"]]
    named = [[step.name for step in pipeline.resolved(registry, stated)] for stated in drawn]
    assert sum(held[0] == "graphspec.contract" and "graphspec.nearby" in held for held in named) > 3
    bounded = Space(registry, NEARBY, 600, [], template)
    groupings = set()
    for _ in range(30):
        stated = bounded.draw(rng, 1, {})["pipeline"]
        assert [step.name for step in pipeline.resolved(registry, stated)].count("graphspec.nearby") == 1
        groupings |= {
            entry["parameters"]["contract/grouping"].split(":")[0]
            for entry in stated
            if entry["operation"] == "graphspec.contract" and "label" not in entry
        }
    assert "item" not in groupings and len(groupings) > 1
    with pytest.raises(ValueError, match=re.escape(FORCED)):
        selector.random_suggest(registry, 1, 1, dict(NEARBY, **CONTRACTED), 600, template)
    LOG.info("offered pipelines exit drawn=%d seconds=%.3f", len(drawn), time.perf_counter() - started)


@timed("selector.broken")
def broken(registry, settings):
    raise ValueError("a fault in the code of an operation")


@timed("selector.unchanged")
def unchanged(graph, registry, settings, seed, limits, carry, held):
    return graph


def test_only_what_a_statement_refuses_leaves_a_trial_out_and_a_fault_in_the_code_it_runs_propagates(bank):
    started = time.perf_counter()
    faulty = Operation(
        "test", "faulty", (Phase("faulty", unchanged, None, None),), lambda registry: (), (), (), (), (), None, broken,
        1, 1,
    )
    OPERATIONS.register(faulty.name, faulty)
    try:
        registry = Bank(bank).registry
        config = random_suggest(registry, 0, 0, {}, 600)[0]
        window = dict(operation="graphspec.window", label="again", parameters={"time/scheme": "nothing"})
        assert selector.refused(registry, dict(config, pipeline=config["pipeline"] + [window])) == ["pipeline"]
        faulted = dict(config, pipeline=config["pipeline"] + [dict(operation=faulty.name, parameters={})])
        with pytest.raises(ValueError, match="a fault in the code of an operation") as raised:
            selector.refused(registry, faulted)
        assert not isinstance(raised.value, Refused)
    finally:
        del OPERATIONS.held[faulty.name]
    LOG.info("refused trials exit seconds=%.3f", time.perf_counter() - started)


@timed("selector.present")
def present(rng, parameter):
    return parameter.options[-1] if parameter.name.count("/") == 1 else parameter.uniform(rng)


def test_a_search_draws_a_reader_before_what_it_reads_with_entries_that_name_kinds_between_them(bank):
    started = time.perf_counter()
    registry, rng, named = Bank(bank).registry, np.random.default_rng(11), []
    for between in ("graphspec.nodes", "graphspec.contract"):
        template = support.template(("graphspec.edges", between, "graphspec.derive"))
        space = pipeline.space(registry, template)
        for _ in range(12):
            flat = pipeline.draw(registry, template, space, partial(present, rng), {})
            named.append([step.name for step in pipeline.resolved(registry, pipeline.entries(template, flat))])
    assert named.count(["graphspec.edges", "graphspec.contract", "graphspec.derive"]) == 12
    assert named.count(["graphspec.edges", "graphspec.nodes", "graphspec.derive"]) == 12
    LOG.info("reader before its read exit drawn=%d seconds=%.3f", len(named), time.perf_counter() - started)


def test_running_again_continues_and_may_change_selector_random_phase_and_bounds(bank, tmp_path):
    output = tmp_path / "run"
    run(arguments(bank, output, **FRESH))
    plan = read_json(output / "plan.json")
    assert plan["selector"] == "random" and plan["random_trials"] == 10 and plan["bounds"] == {}
    assert plan["mix"] == {"random": 100} and plan["random_params"] == []

    run(arguments(bank, output))
    plan = read_json(output / "plan.json")
    assert plan["selector"] == "random" and plan["random_trials"] == 10 and plan["bounds"] == {}

    bounds = tmp_path / "bounds.json"
    write_json(bounds, {"epochs": {"low": 1, "high": 2}})
    run(arguments(bank, output, selector="tpe", random_trials=5, bounds=str(bounds), trials=20))
    plan = read_json(output / "plan.json")
    assert (
        plan["selector"] == "tpe"
        and plan["random_trials"] == 5
        and plan["bounds"] == {"epochs": {"low": 1, "high": 2}}
        and plan["limit"] == 20
    )

    run(arguments(bank, output, selector="random"))
    assert read_json(output / "plan.json")["selector"] == "random"

    rebuilt = tmp_path / "rebuilt"
    shutil.copytree(bank, rebuilt)
    manifest = read_json(rebuilt / "manifest.json")
    write_json(rebuilt / "manifest.json", dict(manifest, seconds=manifest["seconds"] + 1, database="elsewhere"))
    run(arguments(rebuilt, output))
    plan = read_json(output / "plan.json")
    assert plan["bank"] == str(rebuilt.resolve()) and plan["bank_stored_sha256"] == manifest["stored_sha256"]
    write_json(rebuilt / "manifest.json", dict(manifest, stored_sha256="0" * 64))
    differing = "stored %s, and the bank %s stored %s" % (manifest["stored_sha256"], rebuilt, "0" * 64)
    with pytest.raises(ValueError, match=re.escape(differing)):
        run(arguments(rebuilt, output))
    assert read_json(output / "plan.json") == plan

    with pytest.raises(ValueError, match="a new search states --selector, --startup-trials, --mix"):
        run(arguments(bank, tmp_path / "fresh"))
    assert not (tmp_path / "fresh").exists()


def test_random_parameter_settings_persist_and_can_be_cleared(bank, tmp_path):
    output = tmp_path / "run"
    run(
        arguments(
            bank,
            output,
            selector="botorch",
            random_trials=100,
            mix=["botorch=100"],
            random_params=["model/view"],
            pipeline=PIPELINE,
        )
    )
    plan = read_json(output / "plan.json")
    assert plan["random_params"] == ["model/view"]
    run(arguments(bank, output))
    assert read_json(output / "plan.json")["random_params"] == ["model/view"]
    run(arguments(bank, output, random_params=[]))
    assert read_json(output / "plan.json")["random_params"] == []


def test_random_epoch_ranges_persist_and_use_inclusive_integer_draws(bank, tmp_path):
    registry = Bank(bank).registry
    bounds = {"epochs": {"low": 1, "high": 5}}
    bounds_path = tmp_path / "bounds.json"
    write_json(bounds_path, bounds)
    output = tmp_path / "run"
    patterns = ["model/view", "epochs", "max_seconds"]
    fresh = dict(FRESH, selector="botorch", mix=["botorch=100"], random_params=patterns)
    run(arguments(bank, output, bounds=str(bounds_path), seconds=1200, **fresh))
    run(arguments(bank, output, seconds=1200))
    plan = read_json(output / "plan.json")
    assert plan["bounds"] == bounds
    assert plan["random_params"] == patterns
    configs = [random_suggest(registry, seed, 1, bounds, 2000)[0] for seed in range(40)]
    assert {config["epochs"] for config in configs} == {1, 2, 3, 4, 5}
    assert all(config["max_seconds"] == 2000 for config in configs)


@pytest.mark.parametrize(
    "bound", [0, -1, True, 1.5, 600, {"low": 1200, "high": 600}, {"low": 600, "high": 1200}]
)
def test_bounds_that_hold_the_seconds_of_a_trial_are_refused_naming_them(bank, tmp_path, bound):
    from tuning.searches.selector import check_bounds

    registry = Bank(bank).registry
    with pytest.raises(ValueError, match="the bounds hold max_seconds"):
        check_bounds(registry, {"max_seconds": bound}, TEMPLATE)
    with pytest.raises(ValueError, match="the bounds hold max_seconds"):
        random_suggest(registry, 1, 2, {"epochs": 2, "max_seconds": bound}, 600)
    with pytest.raises(ValueError, match="the bounds hold max_seconds"):
        suggest(registry, [], [], 1, 2, {"max_seconds": bound}, 600)
    path = tmp_path / "bounds.json"
    write_json(path, {"max_seconds": bound})
    with pytest.raises(ValueError, match="the bounds hold max_seconds"):
        run(arguments(bank, tmp_path / "run", bounds=str(path), **FRESH))
    assert not (tmp_path / "run").exists()


def test_a_search_ranks_by_a_first_measure_lower_being_better_that_reads_the_kinds_of_the_bank(bank, tmp_path):
    with pytest.raises(ValueError, match="the first measure of its statement, .* r2 states that higher is better"):
        run(arguments(bank, tmp_path / "run", measures=["r2", "mse"], **FRESH))
    with pytest.raises(KeyError, match="no measure is registered under the name 'spread'"):
        run(arguments(bank, tmp_path / "run", measures=["mse", "spread"], **FRESH))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("seconds", [0, -1, True, 1.5, None])
def test_seconds_of_a_trial_outside_their_meaning_are_refused(seconds):
    from tuning.searches.selector import runtime_parameter

    with pytest.raises(ValueError, match="max_seconds is %r and must be a positive integer" % (seconds,)):
        runtime_parameter(seconds)
    found = runtime_parameter(600)
    assert (found.low, found.high) == (600, 600)


@pytest.mark.parametrize(
    "flag,value,field",
    [
        ("seed", 4, "seed"),
        ("derivation_seed", 4, "derivation_seed"),
        ("seconds", 1800, "seconds"),
        ("workers", 2, "workers"),
        ("variant_gb", 2, "variant_bytes"),
        ("batch_sequences", 16, "batch_sequences"),
        ("measures", ["balanced"], "measures"),
        ("alike", 0.5, "alike"),
    ],
)
def test_a_copy_in_the_plan_that_differs_from_the_statement_stops_naming_the_field(
    bank, tmp_path, flag, value, field
):
    output = tmp_path / "run"
    run(arguments(bank, output, seconds=1200, **FRESH))
    before = read_json(output / "plan.json")
    with pytest.raises(ValueError, match=re.escape("plan.json holds %s %r and" % (field, before[field]))):
        run(arguments(bank, output, **{"seconds": 1200, flag: value}))
    with pytest.raises(KeyError, match="the statement lacks"):
        run(arguments(bank, output, **{"seconds": 1200, flag: None}))
    assert read_json(output / "plan.json") == before
    run(arguments(bank, output, seconds=1200))
    assert read_json(output / "plan.json")[field] == before[field]


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "derivation_seed",
        "seconds",
        "workers",
        "variant_bytes",
        "batch_sequences",
        "selector",
        "random_trials",
        "mix",
        "random_params",
        "bounds",
        "objective",
        "pipeline",
        "loss",
        "alike",
        "bank_stored_sha256",
    ],
)
def test_a_plan_on_disk_that_lacks_a_field_stops_naming_it(bank, tmp_path, field):
    output = tmp_path / "run"
    run(arguments(bank, output, **FRESH))
    plan = read_json(output / "plan.json")
    del plan[field]
    write_json(output / "plan.json", plan)
    with pytest.raises(KeyError, match="plan.json lacks " + field):
        run(arguments(bank, output))
    assert read_json(output / "plan.json") == plan


def test_new_trial_limits_are_frozen_and_epochs_can_reach_five(bank, tmp_path, statement):
    from tuning.searches.selector import append_trial

    registry = Bank(bank).registry
    plan = dict(
        selector="random",
        seed=5,
        trials=[],
        random_trials=100,
        mix={"random": 100},
        seconds=1200,
        bounds={"epochs": {"low": 1, "high": 5}},
        bank_stored_sha256="bank",
        supervision_sha256="labels",
        version=9,
        bank=str(bank),
        train_rows=1,
        holdout_rows=1,
        objective=None,
        limit=None,
        pipeline=TEMPLATE,
    )
    write_json(tmp_path / "plan.json", plan)
    append_trial(tmp_path, plan, registry, state(dict(statement, seconds=1200)))
    first_path = tmp_path / plan["trials"][0]["directory"] / "config.json"
    assert read_json(first_path)["max_seconds"] == 1200
    plan["seconds"] = 1800
    write_json(tmp_path / "plan.json", plan)
    append_trial(tmp_path, plan, registry, state(dict(statement, seconds=1800)))
    second = read_json(tmp_path / plan["trials"][1]["directory"] / "config.json")
    assert second["max_seconds"] == 1800
    assert read_json(first_path)["max_seconds"] == 1200
    values = {random_suggest(registry, seed, 1, plan["bounds"], 600)[0]["epochs"] for seed in range(40)}
    assert values == {1, 2, 3, 4, 5}


def test_tpe_good_set_is_the_best_hundred_trials():
    from tuning.searches import tpe
    from tuning.searches.selector import SETTINGS

    study = tpe.study(1, SETTINGS, [])
    assert study.sampler._gamma(410) == 100
    assert study.sampler._gamma(40) == 40
    assert SETTINGS["good_set"] == 100


@pytest.mark.parametrize(
    "field,flag",
    [
        ("selector", "--selector"),
        ("random_trials", "--startup-trials"),
        ("mix", "--mix"),
        ("random_params", "--random-params"),
        ("pipeline", "--pipeline"),
    ],
)
def test_a_new_run_that_leaves_a_choice_of_the_search_unstated_stops_naming_it(bank, tmp_path, field, flag):
    with pytest.raises(ValueError, match="a new search states " + flag + "$"):
        run(arguments(bank, tmp_path / "unstated", **dict(FRESH, **{field: None})))
    assert not (tmp_path / "unstated").exists()


def test_a_new_run_stores_the_choices_of_the_search_as_stated(bank, tmp_path):
    mix = ["random=25", "tpe=25", "forest=25", "botorch=25"]
    run(
        arguments(
            bank,
            tmp_path / "mixed",
            selector="botorch",
            random_trials=10,
            mix=mix,
            random_params=[],
            pipeline=PIPELINE,
        )
    )
    plan = read_json(tmp_path / "mixed" / "plan.json")
    assert plan["selector"] == "botorch" and plan["random_trials"] == 10 and plan["random_params"] == []
    assert plan["pipeline"] == TEMPLATE
    assert plan["mix"] == {"random": 25, "tpe": 25, "forest": 25, "botorch": 25}
    assert plan["batch_sequences"] == 8 and plan["seconds"] == 10 and plan["seed"] == 3


def test_score_histograms_survive_nearly_identical_predictions():
    from tuning.searches.space import histogram

    nearly = [0.25, 0.25 + 5e-17, 0.25 + 1e-16]
    assert histogram(nearly)["counts"] == [3]
    assert sum(histogram([0.0, 1.0, 2.0])["counts"]) == 3 and len(histogram([0.0, 1.0])["counts"]) == 30
