import fnmatch
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import numpy as np

from graphspec import registries
from graphspec.infra.log import emit
from graphspec.infra.processes.processes import guarded, processors, spawned
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations import pipeline
from graphspec.registries import carried, restored
from tuning.families.heads import DEPTH, REALISATION
from tuning.families.model_spec import TRAINING, drawn, edited, groups, placed
from tuning.families.model_spec import space as model_space

from .selector import (
    check_bounds,
    graph_space,
    metadata,
    narrow_group,
    pipeline_space,
    runtime_parameter,
    spaces,
    stepped,
)

CANDIDATES = 10000
MODEL = "model/"
RANKED = 25
IMPORTANT = 20
LOWER = {"predicted_mse"}
FAILURE_POLICY = dict(
    version=2,
    outcomes=["timeout", "failed"],
    objective="objective",
    interpretation="Completed trials train the score model; all completed and failed trials train feasibility.",
)


@timed("space.histogram")
def histogram(values):
    values = np.asarray(values, np.float64)
    low, high = float(values.min()), float(values.max())
    resolvable = high - low > 64 * np.spacing(max(abs(low), abs(high)))
    counts, edges = np.histogram(values, bins=30 if resolvable else 1)
    return dict(edges=edges.tolist(), counts=counts.tolist())


@timed("space.flatten")
def flatten(config, template):
    return {
        **{"model/" + name: value for name, value in config["model"].items()},
        **{"graph/" + name: value for name, value in config["graph"].items()},
        **pipeline.flattened(template, config["pipeline"]),
        **{parameter.name: config[parameter.name] for parameter in TRAINING},
        **({"max_seconds": config["max_seconds"]} if "max_seconds" in config else {}),
    }


class Space:
    @timed("space.made")
    def __init__(self, registry, bounds, seconds, history, template):
        self.registry, self.bounds, self.template = registry, bounds, template
        self.models, self.training = spaces(registry, bounds)
        self.pipeline = pipeline_space(registry, template, bounds)
        self.runtime = runtime_parameter(seconds)
        limits = [self.runtime.low, self.runtime.high]
        limits.extend(
            runtime_parameter(config["max_seconds"]).low for config in history if "max_seconds" in config
        )
        runtime = specs.number("max_seconds", min(limits), max(limits), self.runtime.default)
        self.graphs = {}
        graph = specs.settings(registry, specs.GROUPS)
        self.parameters = {
            **{"model/" + p.name: p for p in model_space(registry)},
            **{p.name: p for p in TRAINING},
            **{"graph/" + p.name: p for p in graph},
            **{p.name: p for p in pipeline.space(registry, template)},
            runtime.name: runtime,
        }
        self.proposal_parameters = {
            **{"model/" + p.name: p for p in self.models},
            **{p.name: p for p in self.training},
            **{"graph/" + p.name: p for p in stepped(registry, narrow_group(graph, bounds))},
            **{p.name: p for p in self.pipeline},
            self.runtime.name: self.runtime,
        }
        self.width = sum(len(p.options) if p.options else 2 for p in self.parameters.values())
        self.named, self.fixings, self.modellings = {p.name for p in self.pipeline}, {}, {}

    @timed("space.fixing")
    def fixing(self, fixed):
        held = {name: value for name, value in fixed.items() if name in self.named}
        key = json.dumps(held, sort_keys=True)
        if key not in self.fixings:
            self.fixings[key] = specs.narrow(self.pipeline, held)
        return self.fixings[key]

    @timed("space.modelling")
    def modelling(self, fixed):
        held = {name[len(MODEL) :]: value for name, value in fixed.items() if name.startswith(MODEL)}
        key = json.dumps(held, sort_keys=True)
        if key not in self.modellings:
            self.modellings[key] = specs.narrow(self.models, held)
        return self.modellings[key]

    @timed("space.Space.importance")
    def importance(self, weights):
        column, totals = 0, {}
        for name, parameter in self.parameters.items():
            width = len(parameter.options) if parameter.options else 2
            totals[name] = float(weights[column : column + width].sum())
            column += width
        ranked = sorted(totals.items(), key=lambda item: -item[1])[:IMPORTANT]
        return [dict(name=name, value=value) for name, value in ranked if value > 0]

    @timed("space.Space.random_names")
    def random_names(self, patterns):
        unknown = [p for p in patterns if not any(fnmatch.fnmatchcase(n, p) for n in self.parameters)]
        if unknown:
            raise ValueError("random parameter names unknown parameters: " + ", ".join(unknown))
        names = [n for n in self.parameters if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
        fixed = [name for name in names if name in (MODEL + REALISATION, MODEL + DEPTH)]
        if fixed:
            placed(self.models, fixed, self.registry.schema)
        return names

    @timed("space.Space.encode")
    def encode(self, configs):
        encoded = np.zeros((len(configs), self.width), dtype=np.float64)
        for row, config in enumerate(configs):
            values, column = flatten(config, self.template), 0
            unknown = sorted(set(values) - self.parameters.keys())
            if unknown:
                raise ValueError("historical configuration has unknown parameters: " + ", ".join(unknown))
            for name, parameter in self.parameters.items():
                width = len(parameter.options) if parameter.options else 2
                if name in values:
                    value = parameter.check(values[name])
                    if parameter.options:
                        encoded[row, column + parameter.options.index(value)] = 1.0
                    else:
                        low, high = parameter.low, parameter.high
                        if parameter.log:
                            low, high, value = math.log(low), math.log(high), math.log(value)
                        encoded[row, column] = (value - low) / (high - low) if high > low else 0.0
                        encoded[row, column + 1] = 1.0
                column += width
        return encoded

    @timed("space.Space.draw")
    def draw(self, rng, training_seed, fixed):
        def choose(prefix, parameter):
            name = prefix + parameter.name
            if name in fixed:
                return parameter.check(fixed[name])
            return parameter.uniform(rng)

        model = drawn(self.modelling(fixed), lambda p: choose(MODEL, p), self.registry.schema)
        key = groups(model), edited(model, self.registry.schema)
        if key not in self.graphs:
            self.graphs[key] = graph_space(self.registry, model, self.bounds)
        graph = specs.draw(self.graphs[key], lambda p: choose("graph/", p))
        return dict(
            seed=training_seed,
            model=model,
            graph=graph,
            pipeline=pipeline.entries(
                self.template,
                pipeline.draw(self.registry, self.template, self.fixing(fixed), lambda p: choose("", p), graph),
            ),
            **specs.draw(self.training, lambda p: choose("", p)),
            max_seconds=choose("", self.runtime),
        )


POOL = {}


@timed("space.pool")
def pool():
    if "executor" in POOL and POOL["generation"] != registries.GENERATION:
        POOL.pop("executor").shutdown(wait=False, cancel_futures=True)
    if "executor" not in POOL:
        started, workers = time.perf_counter(), processors()
        executor = ProcessPoolExecutor(workers, **spawned(None, (), carried, restored))
        POOL.update(executor=executor, generation=registries.GENERATION)
        emit("candidate_pool_started", workers=workers, seconds=time.perf_counter() - started)
    return POOL["executor"]


@timed("space.draw_chunk")
def draw_chunk(registry, bounds, seconds, history, template, seed, training_seed, fixed, count):
    space = Space(registry, bounds, seconds, history, template)
    rng = np.random.default_rng(seed)
    return [space.draw(rng, training_seed, fixed) for _ in range(count)]


@timed("space.draw_candidates")
def draw_candidates(registry, bounds, seconds, history, template, seed, training_seed, fixed, count):
    started = time.perf_counter()
    workers = min(processors(), count)
    counts = [count // workers + (index < count % workers) for index in range(workers)]
    seeds = np.random.SeedSequence(seed).spawn(workers)
    chunks = pool().map(
        partial(guarded, draw_chunk),
        repeat(registry),
        repeat(bounds),
        repeat(seconds),
        repeat(history),
        repeat(template),
        seeds,
        repeat(training_seed),
        repeat(fixed),
        counts,
    )
    candidates = [candidate for chunk in chunks for candidate in chunk]
    emit("candidates_drawn", candidates=count, workers=workers, seconds=time.perf_counter() - started)
    return candidates


@timed("space.propose")
def propose(
    build,
    registry,
    observations,
    failures,
    seed,
    training_seed,
    bounds,
    template,
    random_params,
    device,
    seconds,
    progress=None,
):
    started = time.perf_counter()
    check_bounds(registry, bounds, template)
    records = observations + failures
    space = Space(registry, bounds, seconds, [entry["config"] for entry in records], template)
    rng = np.random.default_rng(seed)
    fixed = {name: space.proposal_parameters[name].uniform(rng) for name in space.random_names(random_params)}
    complete = [entry["status"] == "complete" for entry in records]
    losses = [entry["objective"] if ok else 0.0 for entry, ok in zip(records, complete)]

    @timed("space.propose.update")
    def update(stage):
        if progress is not None:
            progress(
                dict(
                    stage=stage,
                    fixed_params=fixed,
                    observations=sum(complete),
                    failed_observations=len(complete) - sum(complete),
                    seconds=time.perf_counter() - started,
                )
            )

    update("Fitting models")
    history = [entry["config"] for entry in records]
    model = build(space.encode(history), losses, complete, seed, device)
    candidates = draw_candidates(
        registry, bounds, seconds, history, template, seed, training_seed, fixed, CANDIDATES
    )
    update("Ranking candidates")
    index, value = model.choose(space.encode(candidates))
    config = candidates[index]
    update("Proposal ready")
    order = np.argsort(-model.scores["acquisition"])[:RANKED]
    ranked = [
        dict(
            {key: float(values[i]) for key, values in model.scores.items()},
            rank={
                key: int((values < values[i]).sum() if key in LOWER else (values > values[i]).sum()) + 1
                for key, values in model.scores.items()
            },
            config=candidates[i],
        )
        for i in order
    ]
    params = flatten(config, template)
    if any(params[name] != value for name, value in fixed.items() if name in params):
        raise RuntimeError(model.sampler + " proposal changed a preset random parameter")
    elapsed = time.perf_counter() - started
    emit(model.phase + "_selection", fixed_params=fixed, observations=len(records), seconds=elapsed)
    return config, metadata(
        registry,
        config,
        seed,
        training_seed,
        bounds,
        template,
        sampler=model.sampler,
        phase=model.phase,
        params=params,
        random_params=list(random_params),
        fixed_params=fixed,
        observations=sum(complete),
        failed_observations=len(complete) - sum(complete),
        training_trials=[entry["trial"] for entry in records],
        excluded=[],
        failure_policy=FAILURE_POLICY,
        acquisition="LogConstrainedExpectedImprovement" if not all(complete) else "LogExpectedImprovement",
        acquisition_value=value,
        diagnostics=dict(
            prediction=model.diagnostics,
            dimensions=space.width,
            best_observed_mse=model.best,
            candidates=ranked,
            distribution={key: histogram(values) for key, values in model.scores.items()},
            **(
                dict(
                    importance={
                        k: None if v is None else space.importance(v) for k, v in model.importance.items()
                    }
                )
                if hasattr(model, "importance")
                else {}
            ),
        ),
        settings=dict(model.settings, candidates=CANDIDATES),
        seconds=elapsed,
    )
