import fnmatch
import math
import time

import numpy as np

from graphspec.infra.files import JsonCache, read_json, sha256_file, write_json
from graphspec.infra.log import emit
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.views.view import STEPS

from . import objective, tpe
from .model_spec import TRAINING, describe, groups
from .model_spec import space as model_space
from .records import Run, Status

SURROGATES = ("botorch", "forest")
SETTINGS = dict(n_startup_trials=10, n_ei_candidates=64, multivariate=True, group=True, good_set=100)
FAILURE_POLICY = dict(
    version=1,
    outcomes=["timeout", "failed"],
    objective="positive_infinity",
    interpretation="Failure ranks worse than every finite completed score; no prediction metric is assigned.",
)


def distribution(parameter):
    if parameter.kind in ("bool", "choice"):
        return tpe.categorical(parameter.options)
    if parameter.kind == "int":
        return tpe.integer(parameter.low, parameter.high)
    return tpe.real(parameter.low, parameter.high, log=parameter.log)


@timed("selector.stepped")
def stepped(registry, parameters):
    (held,) = [parameter for parameter in parameters if parameter.name == "time/steps"]
    refusals = {name: STEPS.named(name).refusal(registry) for name in held.options}
    accepted = [name for name, found in refusals.items() if found is None]
    if not accepted:
        raise ValueError(
            "the bank %s holds a graph that no choice of steps among %s shows: %s"
            % (
                registry.name,
                ", ".join(held.options),
                "; ".join("%s %s" % (name, found) for name, found in refusals.items()),
            )
        )
    return specs.narrow(parameters, {held.name: accepted})


def graph_space(registry, model, bounds):
    return stepped(registry, narrow_group(specs.space(registry, groups(model)), bounds))


def spaces(registry, bounds):
    return narrow_group(model_space(registry), bounds), narrow_group(TRAINING, bounds)


def runtime_parameter(seconds):
    if type(seconds) is not int or seconds < 1:
        raise ValueError("max_seconds is %r and must be a positive integer" % (seconds,))
    return specs.number("max_seconds", seconds, seconds, seconds)


def narrow_group(parameters, bounds):
    applicable = {
        key: value
        for key, value in bounds.items()
        if any(fnmatch.fnmatchcase(parameter.name, key) for parameter in parameters)
    }
    return specs.narrow(parameters, applicable)


def check_bounds(registry, bounds):
    if "max_seconds" in bounds:
        raise ValueError(
            "the bounds hold max_seconds; the seconds of a trial are those of the statement and no other"
        )
    known = {
        parameter.name for parameter in model_space(registry) + TRAINING + specs.space(registry, specs.GROUPS)
    }
    unknown = sorted(
        key
        for key in bounds
        if not (key in known if "*" not in key else any(fnmatch.fnmatchcase(name, key) for name in known))
    )
    if unknown:
        raise ValueError("bounds name unknown parameters: " + ", ".join(unknown))
    specs.narrow(model_space(registry) + TRAINING + specs.space(registry, specs.GROUPS), bounds)
    stepped(registry, narrow_group(specs.space(registry, ()), bounds))


def training(config):
    return {parameter.name: config[parameter.name] for parameter in TRAINING}


def observation(registry, config, value, bounds, **attrs):
    params, distributions = {}, {}
    models, trainings = spaces(registry, bounds)

    def enter(prefix, parameters, spec):
        lookup = {parameter.name: parameter for parameter in parameters}
        for name, setting in specs.canonical(parameters, spec).items():
            params[prefix + name], distributions[prefix + name] = setting, distribution(lookup[name])

    enter("model/", models, config["model"])
    enter("", trainings, training(config))
    enter("graph/", graph_space(registry, config["model"], bounds), config["graph"])
    if "max_seconds" in config:
        enter("", (runtime_parameter(config["max_seconds"]),), {"max_seconds": config["max_seconds"]})
    return tpe.record(value, params, distributions, config=config, **attrs)


def outside(parameters, spec):
    names = []

    def read(parameter):
        if parameter.name not in spec:
            names.append(parameter.name)
            return parameter.default
        value = spec[parameter.name]
        try:
            parameter.check(value)
        except ValueError:
            names.append(parameter.name)
        return value

    specs.draw(parameters, read)
    return names


def excluded(registry, config, bounds):
    models, trainings = spaces(registry, bounds)
    return (
        outside(models, config["model"])
        + outside(trainings, training(config))
        + outside(graph_space(registry, config["model"], bounds), config["graph"])
        + (outside((runtime_parameter(config["max_seconds"]),), config) if "max_seconds" in config else [])
    )


def metadata(registry, config, seed, training_seed, bounds, **extra):
    model, graph = spaces(registry, bounds)[0], graph_space(registry, config["model"], bounds)
    return dict(
        extra,
        model_digest=specs.digest(config["model"]),
        description=describe(config["model"]),
        model_summary=specs.summary(model, config["model"]),
        model_space=[parameter.declaration() for parameter in model],
        graph_digest=specs.digest(config["graph"]),
        graph_summary=specs.summary(graph, config["graph"]),
        graph_space=[parameter.declaration() for parameter in graph],
        selector_seed=seed,
        training_seed=training_seed,
    )


def configuration(registry, choose, training_seed, bounds, seconds):
    models, trainings = spaces(registry, bounds)
    model = specs.draw(models, lambda parameter: choose("model/", parameter))
    config = dict(
        seed=training_seed, model=model, **specs.draw(trainings, lambda parameter: choose("", parameter))
    )
    config["graph"] = specs.draw(
        graph_space(registry, model, bounds), lambda parameter: choose("graph/", parameter)
    )
    config["max_seconds"] = choose("", runtime_parameter(seconds))
    return config


def suggest(registry, observations, failures, seed, training_seed, bounds, seconds):
    started = time.perf_counter()
    check_bounds(registry, bounds)
    records, left_out, used = [], [], dict(complete=0, failed=0)
    for entry in observations + failures:
        names = excluded(registry, entry["config"], bounds)
        if names:
            left_out.append(dict(trial=entry["trial"], status=entry["status"], parameters=names))
            continue
        complete = entry["status"] == "complete"
        used["complete" if complete else "failed"] += 1
        records.append(
            observation(
                registry,
                entry["config"],
                entry["objective"] if complete else math.inf,
                bounds,
                outcome=entry["status"],
            )
        )
    study = tpe.study(seed, SETTINGS, records)
    trial = study.ask()
    config = configuration(
        registry,
        lambda prefix, parameter: tpe.propose(trial, prefix + parameter.name, distribution(parameter)),
        training_seed,
        bounds,
        seconds,
    )
    return config, metadata(
        registry,
        config,
        seed,
        training_seed,
        bounds,
        sampler="TPESampler",
        settings=SETTINGS,
        params=trial.params,
        failure_policy=FAILURE_POLICY,
        observations=used["complete"],
        failed_observations=used["failed"],
        excluded=left_out,
        optuna_version=tpe.VERSION,
        phase="tpe" if len(records) >= SETTINGS["n_startup_trials"] else "random_startup",
        seconds=time.perf_counter() - started,
    )


def random_suggest(registry, seed, training_seed, bounds, seconds):
    started = time.perf_counter()
    check_bounds(registry, bounds)
    rng = np.random.default_rng(seed)
    config = configuration(
        registry, lambda prefix, parameter: parameter.uniform(rng), training_seed, bounds, seconds
    )
    return config, metadata(
        registry,
        config,
        seed,
        training_seed,
        bounds,
        sampler="numpy.random.Generator",
        phase="random",
        seconds=time.perf_counter() - started,
    )


def load_history(directory, plan):
    observations, failures, history, failure_history = [], [], [], []
    run = Run(directory, JsonCache())
    for trial in run.trials():
        outcome = trial.outcome
        if outcome is None or trial.status == Status.INTERRUPTED:
            continue
        if trial.status != Status.COMPLETE:
            failures.append(dict(config=trial.parameters, status=trial.status, trial=trial.item["directory"]))
            failure_history.append(
                dict(
                    trial=trial.item["directory"],
                    status=trial.status,
                    record=outcome.path.name,
                    failure_sha256=sha256_file(outcome.path),
                    config_sha256=sha256_file(trial.directory / "config.json"),
                )
            )
            continue
        result = outcome.data
        score = objective.value(result, plan)
        observations.append(
            dict(result, objective=score, config=trial.parameters, trial=trial.item["directory"])
        )
        history.append(
            dict(
                trial=trial.item["directory"],
                objective=score,
                result_sha256=sha256_file(outcome.path),
            )
        )
    return observations, failures, history, failure_history


def seeds(seed, index):
    started = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    result = tuple(int(value) for value in rng.integers(1, 2**31, size=2))
    emit("trial_seeds", index=index, seconds=time.perf_counter() - started)
    return result


def roll():
    return np.random.default_rng().random()


def check_mix(mix):
    unknown = sorted(set(mix) - {"random", *SURROGATES, "tpe"})
    if unknown:
        raise ValueError("unknown selectors in the mix: " + ", ".join(unknown))
    if any(share < 0 for share in mix.values()) or abs(sum(mix.values()) - 100) > 1e-9:
        raise ValueError("selector mix shares must be nonnegative and sum to 100")
    return mix


def pick(mix):
    point, total = roll() * 100, 0.0
    for name, share in mix.items():
        total += share
        if point < total:
            return name
    return list(mix)[-1]


def append_trial(directory, plan, registry, stated):
    started = time.perf_counter()
    observations, failures, history, failure_history = load_history(directory, plan)
    index = len(plan["trials"])
    selector_seed, training_seed = seeds(stated.seed, index)
    bounds = plan["bounds"]
    finished = len(observations) + len(failures)
    startup = (
        finished < plan["random_trials"] if plan["selector"] in SURROGATES else index < plan["random_trials"]
    )
    selector = "random" if startup else pick(plan["mix"])
    if selector == "random":
        config, selection = random_suggest(registry, selector_seed, training_seed, bounds, stated.seconds)
    elif selector in SURROGATES:
        from .surrogates import suggest_botorch, suggest_forest

        surrogate_suggest = dict(botorch=suggest_botorch, forest=suggest_forest)[selector]
        progress_started = time.time()

        def progress(value):
            write_json(
                directory / "selection_progress.json",
                dict(value, trial_number=index + 1, started=progress_started, recorded_at=time.time()),
            )

        try:
            config, selection = surrogate_suggest(
                registry,
                observations,
                failures,
                selector_seed,
                training_seed,
                bounds,
                plan["random_params"],
                stated.device,
                stated.seconds,
                progress=progress,
            )
        except BaseException as error:
            progress(dict(stage="Selection failed", error=str(error)))
            raise
        selection["failure_history"] = failure_history
    else:
        config, selection = suggest(
            registry, observations, failures, selector_seed, training_seed, bounds, stated.seconds
        )
        selection["failure_history"] = failure_history
    selection.update(
        startup_trials=plan["random_trials"],
        startup_trials_finished=finished,
        mix=plan["mix"],
    )
    name = "trial_%06d_%s_%s_%s" % (
        index,
        selector,
        selection["model_digest"][:10],
        selection["graph_digest"][:10],
    )
    config["splits"] = stated.splits()
    trial = directory / name
    trial.mkdir(exist_ok=True)
    if (trial / "config.json").exists() and read_json(trial / "config.json") != config:
        raise ValueError("proposal changed on continuation: " + name)
    write_json(trial / "config.json", config)
    write_json(
        trial / "selection.json",
        dict(
            selection,
            history=history,
            bounds=bounds,
            bank_manifest_sha256=plan["bank_manifest_sha256"],
            supervision_sha256=plan["supervision_sha256"],
            spec_sha256=registry.fingerprint(),
            seconds=time.perf_counter() - started,
        ),
    )
    plan["trials"].append(
        dict(
            directory=name,
            model=selection["model_digest"],
            description=selection["description"],
            graph=selection["graph_digest"],
            selector=selector,
        )
    )
    write_json(directory / "plan.json", plan)
    emit(
        "parameter_selection",
        trial=name,
        observations=len(observations),
        excluded=len(selection.get("excluded", ())),
        model=selection["description"],
        graph=selection["graph_digest"][:10],
        phase=selection["phase"],
        seconds=time.perf_counter() - started,
    )
