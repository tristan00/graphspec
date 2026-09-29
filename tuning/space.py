import atexit
import fnmatch
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import numpy as np

from graphspec import spec as specs
from graphspec.infra.log import emit
from graphspec.infra.processes import guarded, spawned

from .model_spec import TRAINING, groups
from .model_spec import space as model_space
from .selector import check_bounds, graph_space, metadata, narrow_group, runtime_parameter, spaces, stepped

CANDIDATES = 10000
RANKED = 25
IMPORTANT = 20
LOWER = {"predicted_mse"}
FAILURE_POLICY = dict(
    version=2,
    outcomes=["timeout", "failed"],
    objective="objective",
    interpretation="Completed trials train the score model; all completed and failed trials train feasibility.",
)


def histogram(values):
    values = np.asarray(values, np.float64)
    low, high = float(values.min()), float(values.max())
    resolvable = high - low > 64 * np.spacing(max(abs(low), abs(high)))
    counts, edges = np.histogram(values, bins=30 if resolvable else 1)
    return dict(edges=edges.tolist(), counts=counts.tolist())


def flatten(config):
    return {
        **{"model/" + name: value for name, value in config["model"].items()},
        **{"graph/" + name: value for name, value in config["graph"].items()},
        **{parameter.name: config[parameter.name] for parameter in TRAINING},
        **({"max_seconds": config["max_seconds"]} if "max_seconds" in config else {}),
    }


class Space:
    def __init__(self, registry, bounds, seconds, history):
        self.registry, self.bounds = registry, bounds
        self.models, self.training = spaces(registry, bounds)
        self.runtime = runtime_parameter(seconds)
        limits = [self.runtime.low, self.runtime.high]
        limits.extend(
            runtime_parameter(config["max_seconds"]).low for config in history if "max_seconds" in config
        )
        runtime = specs.number("max_seconds", min(limits), max(limits), self.runtime.default)
        self.graphs = {}
        graph = specs.space(registry, specs.GROUPS)
        self.parameters = {
            **{"model/" + p.name: p for p in model_space(registry)},
            **{p.name: p for p in TRAINING},
            **{"graph/" + p.name: p for p in graph},
            runtime.name: runtime,
        }
        self.proposal_parameters = {
            **{"model/" + p.name: p for p in self.models},
            **{p.name: p for p in self.training},
            **{"graph/" + p.name: p for p in stepped(registry, narrow_group(graph, bounds))},
            self.runtime.name: self.runtime,
        }
        self.width = sum(len(p.options) if p.options else 2 for p in self.parameters.values())

    def importance(self, weights):
        column, totals = 0, {}
        for name, parameter in self.parameters.items():
            width = len(parameter.options) if parameter.options else 2
            totals[name] = float(weights[column : column + width].sum())
            column += width
        ranked = sorted(totals.items(), key=lambda item: -item[1])[:IMPORTANT]
        return [dict(name=name, value=value) for name, value in ranked if value > 0]

    def random_names(self, patterns):
        unknown = [p for p in patterns if not any(fnmatch.fnmatchcase(n, p) for n in self.parameters)]
        if unknown:
            raise ValueError("random parameter names unknown parameters: " + ", ".join(unknown))
        return [n for n in self.parameters if any(fnmatch.fnmatchcase(n, p) for p in patterns)]

    def encode(self, configs):
        encoded = np.zeros((len(configs), self.width), dtype=np.float64)
        for row, config in enumerate(configs):
            values, column = flatten(config), 0
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

    def draw(self, rng, training_seed, fixed, base=None, probability=1.0):
        base = {} if base is None else base

        def choose(prefix, parameter):
            name = prefix + parameter.name
            if name in fixed:
                return parameter.check(fixed[name])
            if name in base and rng.random() >= probability:
                return parameter.check(base[name])
            return parameter.uniform(rng)

        model = specs.draw(self.models, lambda p: choose("model/", p))
        key = groups(model)
        if key not in self.graphs:
            self.graphs[key] = graph_space(self.registry, model, self.bounds)
        return dict(
            seed=training_seed,
            model=model,
            graph=specs.draw(self.graphs[key], lambda p: choose("graph/", p)),
            **specs.draw(self.training, lambda p: choose("", p)),
            max_seconds=choose("", self.runtime),
        )


POOL = {}


def pool():
    if "executor" not in POOL:
        started = time.perf_counter()
        POOL["executor"] = ProcessPoolExecutor(os.cpu_count(), **spawned(None, ()))
        atexit.register(POOL["executor"].shutdown, wait=False, cancel_futures=True)
        emit("candidate_pool_started", workers=os.cpu_count(), seconds=time.perf_counter() - started)
    return POOL["executor"]


def draw_chunk(registry, bounds, seconds, history, seed, training_seed, fixed, count):
    space = Space(registry, bounds, seconds, history)
    rng = np.random.default_rng(seed)
    return [space.draw(rng, training_seed, fixed) for _ in range(count)]


def draw_candidates(registry, bounds, seconds, history, seed, training_seed, fixed, count):
    started = time.perf_counter()
    workers = min(os.cpu_count(), count)
    counts = [count // workers + (index < count % workers) for index in range(workers)]
    seeds = np.random.SeedSequence(seed).spawn(workers)
    chunks = pool().map(
        partial(guarded, draw_chunk),
        repeat(registry),
        repeat(bounds),
        repeat(seconds),
        repeat(history),
        seeds,
        repeat(training_seed),
        repeat(fixed),
        counts,
    )
    candidates = [candidate for chunk in chunks for candidate in chunk]
    emit("candidates_drawn", candidates=count, workers=workers, seconds=time.perf_counter() - started)
    return candidates


def propose(
    build,
    registry,
    observations,
    failures,
    seed,
    training_seed,
    bounds,
    random_params,
    device,
    seconds,
    progress=None,
):
    started = time.perf_counter()
    check_bounds(registry, bounds)
    records = observations + failures
    space = Space(registry, bounds, seconds, [entry["config"] for entry in records])
    rng = np.random.default_rng(seed)
    fixed = {name: space.proposal_parameters[name].uniform(rng) for name in space.random_names(random_params)}
    complete = [entry["status"] == "complete" for entry in records]
    losses = [entry["objective"] if ok else 0.0 for entry, ok in zip(records, complete)]

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
    candidates = draw_candidates(registry, bounds, seconds, history, seed, training_seed, fixed, CANDIDATES)
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
    params = flatten(config)
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
