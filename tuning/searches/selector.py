import fnmatch
import math
import time

import numpy as np

from graphspec.infra.files import JsonCache, read_json, sha256_file, write_json
from graphspec.infra.log import emit
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.operations.negatives import needed
from graphspec.operations.operation import Refused
from graphspec.views.view import STEPS
from tuning.families.model_spec import TRAINING, describe, drawn, groups, mapped, placed, realisable, turned
from tuning.families.model_spec import space as model_space
from tuning.measures import objective

from . import tpe
from .records import Run, Status

SURROGATES = ("botorch", "forest")
SETTINGS = dict(n_startup_trials=10, n_ei_candidates=64, multivariate=True, group=True, good_set=100)
FAILURE_POLICY = dict(
    version=1,
    outcomes=["timeout", "failed"],
    objective="positive_infinity",
    interpretation="Failure ranks worse than every finite completed score; no prediction metric is assigned.",
)


@timed("selector.distribution")
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


@timed("selector.graph_space")
def graph_space(registry, model, bounds):
    return turned(registry, model, stepped(registry, narrow_group(specs.settings(registry, groups(model)), bounds)))


@timed("selector.pipeline_space")
def pipeline_space(registry, template, bounds):
    return kept(registry, template, needing(registry, template, narrow_group(pipeline.space(registry, template), bounds)))


@timed("selector.kept")
def kept(registry, template, parameters):
    kinds, groups = pipeline.decided(registry)
    stated = pipeline.whole(registry, template, set(kinds), set(groups) | mapped(registry))
    names = {parameter.name for parameter in parameters}
    return specs.narrow(parameters, {name: value for name, value in stated.items() if name in names})


@timed("selector.needing")
def needing(registry, template, parameters):
    found = [(name, operation.name) for name, operation, _, _ in pipeline.checked(template, pipeline.TEMPLATE)]
    named, taken, held = {parameter.name: parameter for parameter in parameters}, {}, {}
    for kind, (operation, values) in needed(registry.schema).items():
        wanted = (operation, tuple(sorted(values.items())))
        if wanted in taken:
            continue
        free = [
            name
            for name, entry in found
            if entry == operation
            and name not in taken.values()
            and True in named[pipeline.PRESENT % name].options
            and all(value in named[pipeline.NAMED % name + key].options for key, value in values.items())
        ]
        if not free:
            raise ValueError(
                "the kind of task %s states a target that needs an entry of %s with %s, and the template holds none "
                "that its bounds leave free to be it" % (kind, operation, ", ".join("%s %s" % pair for pair in wanted[1]))
            )
        taken[wanted] = free[0]
        held[pipeline.PRESENT % free[0]] = [True]
        held.update({pipeline.NAMED % free[0] + key: [value] for key, value in values.items()})
    for name, entry in found:
        if name in taken.values():
            continue
        refused = {}
        for operation, pairs in taken:
            for key, value in pairs if entry == operation else ():
                refused.setdefault(pipeline.NAMED % name + key, set()).add(value)
        left = {key: [option for option in named[key].options if option not in values] for key, values in refused.items()}
        if all(left.values()):
            held.update(left)
        elif False in named[pipeline.PRESENT % name].options:
            held[pipeline.PRESENT % name] = [False]
        else:
            raise ValueError(
                "the bounds hold the entry %s of the template present, and every value it may take repeats an entry "
                "a target needs" % name
            )
    return specs.narrow(parameters, held)


@timed("selector.keyed")
def keyed(prefix, parameter, static):
    name = prefix + parameter.name
    if name not in static or static[name] == parameter.options:
        return name
    return name + "|" + "|".join(map(str, parameter.options))


@timed("selector.recorded")
def recorded(registry, template, parameters, flat, record, settings):
    @timed("selector.recorded.read")
    def read(parameter):
        if parameter.name not in flat:
            raise ValueError("spec lacks the active parameter " + parameter.name)
        return record(parameter, parameter.check(flat[parameter.name]))

    extra = sorted(set(flat) - set(pipeline.draw(registry, template, parameters, read, settings)))
    if extra:
        raise ValueError("spec sets inactive or unknown parameters: " + ", ".join(extra))


@timed("selector.refused")
def refused(registry, config):
    try:
        pipeline.agreed(registry, pipeline.resolved(registry, config["pipeline"]), config["graph"])
    except Refused:
        return ["pipeline"]
    return []


@timed("selector.spaces")
def spaces(registry, bounds):
    return narrow_group(realisable(registry, model_space(registry)), bounds), narrow_group(TRAINING, bounds)


@timed("selector.runtime_parameter")
def runtime_parameter(seconds):
    if type(seconds) is not int or seconds < 1:
        raise ValueError("max_seconds is %r and must be a positive integer" % (seconds,))
    return specs.number("max_seconds", seconds, seconds, seconds)


@timed("selector.narrow_group")
def narrow_group(parameters, bounds):
    applicable = {
        key: value
        for key, value in bounds.items()
        if any(fnmatch.fnmatchcase(parameter.name, key) for parameter in parameters)
    }
    return specs.narrow(parameters, applicable)


@timed("selector.check_bounds")
def check_bounds(registry, bounds, template):
    if "max_seconds" in bounds:
        raise ValueError(
            "the bounds hold max_seconds; the seconds of a trial are those of the statement and no other"
        )
    parameters = (
        model_space(registry)
        + TRAINING
        + specs.settings(registry, specs.GROUPS)
        + pipeline.space(registry, template)
    )
    known = {parameter.name for parameter in parameters}
    unknown = sorted(
        key
        for key in bounds
        if not (key in known if "*" not in key else any(fnmatch.fnmatchcase(name, key) for name in known))
    )
    if unknown:
        raise ValueError("bounds name unknown parameters: " + ", ".join(unknown))
    specs.narrow(parameters, bounds)
    placed(spaces(registry, bounds)[0], None)
    graph = stepped(registry, narrow_group(specs.settings(registry, ()), bounds))
    held = pipeline_space(registry, template, bounds)
    pipeline.housing(registry, template, held, graph)
    pipeline.draw(registry, template, held, pipeline.settled, specs.draw(graph, pipeline.settled))


@timed("selector.training")
def training(config):
    return {parameter.name: config[parameter.name] for parameter in TRAINING}


@timed("selector.statics")
def statics(models, held):
    return {
        **{"model/" + parameter.name: parameter.options for parameter in models},
        **{parameter.name: parameter.options for parameter in held},
    }


@timed("selector.observation")
def observation(registry, config, value, bounds, template, **attrs):
    params, distributions = {}, {}
    models, trainings = spaces(registry, bounds)
    held = pipeline_space(registry, template, bounds)
    static = statics(models, held)

    @timed("selector.observation.enter")
    def enter(prefix, parameters, spec, drawing):
        @timed("selector.observation.enter.read")
        def read(parameter):
            if parameter.name not in spec:
                raise ValueError("spec lacks the active parameter " + parameter.name)
            return record(prefix, parameter, parameter.check(spec[parameter.name]))

        extra = sorted(set(spec) - set(drawing(parameters, read)))
        if extra:
            raise ValueError("spec sets inactive or unknown parameters: " + ", ".join(extra))

    @timed("selector.observation.record")
    def record(prefix, parameter, value):
        name = keyed(prefix, parameter, static)
        params[name], distributions[name] = value, distribution(parameter)
        return value

    enter("model/", models, config["model"], drawn)
    enter("", trainings, training(config), specs.draw)
    enter("graph/", graph_space(registry, config["model"], bounds), config["graph"], specs.draw)
    flat = pipeline.flattened(template, config["pipeline"])
    recorded(registry, template, held, flat, lambda parameter, value: record("", parameter, value), config["graph"])
    if "max_seconds" in config:
        enter("", (runtime_parameter(config["max_seconds"]),), {"max_seconds": config["max_seconds"]}, specs.draw)
    return tpe.record(value, params, distributions, config=config, **attrs)


@timed("selector.outside")
def outside(parameters, spec, drawing):
    names = []

    @timed("selector.outside.read")
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

    drawing(parameters, read)
    return names


@timed("selector.excluded")
def excluded(registry, config, bounds, template):
    models, trainings = spaces(registry, bounds)
    flat = pipeline.flattened(template, config["pipeline"])
    return (
        (outside(models, config["model"], specs.draw) or outside(models, config["model"], drawn))
        + outside(trainings, training(config), specs.draw)
        + outside(graph_space(registry, config["model"], bounds), config["graph"], specs.draw)
        + outside(pipeline_space(registry, template, bounds), flat, specs.draw)
        + refused(registry, config)
        + (outside((runtime_parameter(config["max_seconds"]),), config, specs.draw) if "max_seconds" in config else [])
    )


@timed("selector.metadata")
def metadata(registry, config, seed, training_seed, bounds, template, **extra):
    model, graph = spaces(registry, bounds)[0], graph_space(registry, config["model"], bounds)
    held = pipeline_space(registry, template, bounds)
    return dict(
        extra,
        model_digest=specs.digest(config["model"]),
        description=describe(config["model"]),
        model_summary=specs.summary(model, config["model"]),
        model_space=[parameter.declaration() for parameter in model],
        graph_digest=specs.digest(config["graph"]),
        graph_summary=specs.summary(graph, config["graph"]),
        graph_space=[parameter.declaration() for parameter in graph],
        pipeline_digest=specs.digest(config["pipeline"]),
        pipeline_summary=specs.summary(held, pipeline.flattened(template, config["pipeline"])),
        pipeline_space=[parameter.declaration() for parameter in held],
        selector_seed=seed,
        training_seed=training_seed,
    )


@timed("selector.configuration")
def configuration(registry, choose, training_seed, bounds, seconds, template):
    models, trainings = spaces(registry, bounds)
    model = drawn(models, lambda parameter: choose("model/", parameter))
    config = dict(
        seed=training_seed, model=model, **specs.draw(trainings, lambda parameter: choose("", parameter))
    )
    config["graph"] = specs.draw(
        graph_space(registry, model, bounds), lambda parameter: choose("graph/", parameter)
    )
    config["pipeline"] = pipeline.entries(
        template,
        pipeline.draw(
            registry,
            template,
            pipeline_space(registry, template, bounds),
            lambda parameter: choose("", parameter),
            config["graph"],
        ),
    )
    config["max_seconds"] = choose("", runtime_parameter(seconds))
    return config


@timed("selector.suggest")
def suggest(registry, observations, failures, seed, training_seed, bounds, seconds, template):
    started = time.perf_counter()
    check_bounds(registry, bounds, template)
    records, left_out, used = [], [], dict(complete=0, failed=0)
    for entry in observations + failures:
        names = excluded(registry, entry["config"], bounds, template)
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
                template,
                outcome=entry["status"],
            )
        )
    study = tpe.study(seed, SETTINGS, records)
    trial = study.ask()
    static = statics(spaces(registry, bounds)[0], pipeline_space(registry, template, bounds))
    config = configuration(
        registry,
        lambda prefix, parameter: tpe.propose(trial, keyed(prefix, parameter, static), distribution(parameter)),
        training_seed,
        bounds,
        seconds,
        template,
    )
    return config, metadata(
        registry,
        config,
        seed,
        training_seed,
        bounds,
        template,
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


@timed("selector.random_suggest")
def random_suggest(registry, seed, training_seed, bounds, seconds, template):
    started = time.perf_counter()
    check_bounds(registry, bounds, template)
    rng = np.random.default_rng(seed)
    config = configuration(
        registry, lambda prefix, parameter: parameter.uniform(rng), training_seed, bounds, seconds, template
    )
    return config, metadata(
        registry,
        config,
        seed,
        training_seed,
        bounds,
        template,
        sampler="numpy.random.Generator",
        phase="random",
        seconds=time.perf_counter() - started,
    )


@timed("selector.load_history")
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
        if result["unscored"]:
            history.append(
                dict(trial=trial.item["directory"], unscored=result["unscored"], result_sha256=sha256_file(outcome.path))
            )
            continue
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


@timed("selector.seeds")
def seeds(seed, index):
    started = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    result = tuple(int(value) for value in rng.integers(1, 2**31, size=2))
    emit("trial_seeds", index=index, seconds=time.perf_counter() - started)
    return result


@timed("selector.roll")
def roll():
    return np.random.default_rng().random()


@timed("selector.check_mix")
def check_mix(mix):
    unknown = sorted(set(mix) - {"random", *SURROGATES, "tpe"})
    if unknown:
        raise ValueError("unknown selectors in the mix: " + ", ".join(unknown))
    if any(share < 0 for share in mix.values()) or abs(sum(mix.values()) - 100) > 1e-9:
        raise ValueError("selector mix shares must be nonnegative and sum to 100")
    return mix


@timed("selector.pick")
def pick(mix):
    point, total = roll() * 100, 0.0
    for name, share in mix.items():
        total += share
        if point < total:
            return name
    return list(mix)[-1]


@timed("selector.append_trial")
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
    template = plan["pipeline"]
    if selector == "random":
        config, selection = random_suggest(
            registry, selector_seed, training_seed, bounds, stated.seconds, template
        )
    elif selector in SURROGATES:
        from .surrogates import suggest_botorch, suggest_forest

        surrogate_suggest = dict(botorch=suggest_botorch, forest=suggest_forest)[selector]
        progress_started = time.time()

        @timed("selector.append_trial.progress")
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
                template,
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
            registry, observations, failures, selector_seed, training_seed, bounds, stated.seconds, template
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
            bank_stored_sha256=plan["bank_stored_sha256"],
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
