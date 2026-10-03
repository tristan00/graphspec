import time
from functools import partial

import numpy as np
import torch

from graphspec.infra import compute
from graphspec.infra.log import emit
from graphspec.infra.timing import timed

from .space import propose

FIT_ITERATIONS = 1000
BATCH = 64
TREES = 100
COMPLETION_WEIGHT = 3.0


@timed("surrogates.fit_model")
def fit_model(x, y):
    from botorch.models.map_saas import get_map_saas_model
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.core import OptimizationStatus
    from botorch.optim.fit import fit_gpytorch_mll_scipy
    from gpytorch.mlls import ExactMarginalLogLikelihood

    started = time.perf_counter()
    model = get_map_saas_model(x, y, outcome_transform=Standardize(m=1))
    result = fit_gpytorch_mll_scipy(
        ExactMarginalLogLikelihood(model.likelihood, model),
        options=dict(maxiter=FIT_ITERATIONS, ftol=1e-6),
    )
    if result.status in (OptimizationStatus.FAILURE, OptimizationStatus.RUNNING):
        raise RuntimeError("BoTorch model fitting failed: " + str(result.message))
    model.eval()
    emit(
        "botorch_fit",
        observations=len(x),
        dimensions=x.shape[-1],
        iterations=result.step,
        seconds=time.perf_counter() - started,
    )
    return model


class Search:
    @timed("surrogates.Search.__init__")
    def __init__(self, x, losses, completed, device):
        from botorch.acquisition.analytic import LogConstrainedExpectedImprovement, LogExpectedImprovement
        from botorch.models import ModelListGP

        started = time.perf_counter()
        self.device = torch.device(device)
        x = torch.as_tensor(np.asarray(x), dtype=torch.double, device=self.device)
        completed = torch.as_tensor(completed, dtype=torch.bool, device=self.device)
        losses = torch.as_tensor(losses, dtype=torch.double, device=self.device)
        if completed.sum() < 2:
            raise ValueError("BoTorch requires at least two completed trials with finite scores")
        if not torch.isfinite(x).all() or not torch.isfinite(losses[completed]).all():
            raise ValueError("BoTorch received nonfinite parameters or completed scores")
        objective = fit_model(x[completed], -losses[completed, None])
        self.objective = objective
        self.feasibility = None
        self.best = float(losses[completed].min())
        self.improvement = LogExpectedImprovement(objective, best_f=-self.best)
        if completed.all():
            self.acquisition = self.improvement
        else:
            feasibility = fit_model(x, completed.to(torch.double).unsqueeze(-1))
            self.feasibility = feasibility
            self.acquisition = LogConstrainedExpectedImprovement(
                ModelListGP(objective, feasibility),
                best_f=-self.best,
                objective_index=0,
                constraints={1: (0.5, None)},
            )
        emit(
            "botorch_model",
            device=self.device.type,
            observations=int(completed.sum()),
            failed_observations=int((~completed).sum()),
            seconds=time.perf_counter() - started,
        )

    @timed("surrogates.Search.evaluate")
    def evaluate(self, x):
        posterior = self.objective.posterior(x)
        mean, deviation = -posterior.mean.reshape(-1), posterior.variance.clamp_min(0).sqrt().reshape(-1)
        scores = dict(
            acquisition=self.acquisition(x),
            log_improvement=self.improvement(x),
            beat_probability=torch.special.ndtr((self.best - mean) / deviation.clamp_min(1e-9)),
            predicted_mse=mean,
            uncertainty=deviation,
        )
        if self.feasibility is not None:
            feasible = self.feasibility.posterior(x)
            z = (feasible.mean - 0.5) / feasible.variance.clamp_min(1e-12).sqrt()
            scores["completion_probability"] = torch.special.ndtr(z).reshape(-1)
        return scores

    @timed("surrogates.Search.choose")
    def choose(self, candidates):
        started = time.perf_counter()
        x = torch.as_tensor(np.asarray(candidates), dtype=torch.double, device=self.device).unsqueeze(-2)
        with torch.no_grad():
            batches = [self.evaluate(batch) for batch in x.split(BATCH)]
        self.scores = {key: torch.cat([b[key] for b in batches]).cpu().numpy() for key in batches[0]}
        if not np.isfinite(self.scores["acquisition"]).all():
            raise RuntimeError("BoTorch produced a nonfinite acquisition value")
        index = int(self.scores["acquisition"].argmax())
        self.diagnostics = dict(
            predicted_mse=float(self.scores["predicted_mse"][index]),
            uncertainty=float(self.scores["uncertainty"][index]),
            completion_probability=None
            if self.feasibility is None
            else float(self.scores["completion_probability"][index]),
        )
        emit("botorch_acquisition", candidates=len(x), seconds=time.perf_counter() - started)
        return index, float(self.scores["acquisition"][index])


@timed("surrogates.log_expected_improvement")
def log_expected_improvement(best, mean, deviation):
    from scipy.stats import norm

    deviation = np.maximum(deviation, 1e-9)
    z = (best - mean) / deviation
    far = z < -6
    body = np.where(far, 0.0, z * norm.cdf(z) + norm.pdf(z))
    tail = norm.logpdf(z) - 2 * np.log(np.maximum(-z, 1e-9))
    return np.log(deviation) + np.where(far, tail, np.log(np.maximum(body, 1e-300)))


class Forest:
    @timed("surrogates.Forest.__init__")
    def __init__(self, x, losses, completed, seed):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        completed = np.asarray(completed, dtype=bool)
        losses = np.asarray(losses, dtype=np.float64)
        if completed.sum() < 2:
            raise ValueError("the forest requires at least two completed trials with finite scores")
        if not np.isfinite(x).all() or not np.isfinite(losses[completed]).all():
            raise ValueError("the forest received nonfinite parameters or completed scores")
        if (losses[completed] <= 0).any():
            raise ValueError("the forest fits log scores, which requires positive completed scores")
        self.trees = TREES
        self.best = float(losses[completed].min())
        self.objective = RandomForestRegressor(
            n_estimators=TREES, max_features="sqrt", n_jobs=-1, random_state=seed
        ).fit(x[completed], np.log(losses[completed]))
        self.feasibility = None
        if not completed.all():
            self.feasibility = RandomForestClassifier(
                n_estimators=TREES, max_features="sqrt", n_jobs=-1, random_state=seed
            ).fit(x, completed)
        self.importance = dict(
            score=self.objective.feature_importances_,
            completion=None if self.feasibility is None else self.feasibility.feature_importances_,
        )
        emit(
            "forest_model",
            observations=int(completed.sum()),
            failed_observations=int((~completed).sum()),
            dimensions=x.shape[1],
            seconds=time.perf_counter() - started,
        )

    @timed("surrogates.Forest.choose")
    def choose(self, candidates):
        from joblib import Parallel, delayed
        from scipy.stats import norm

        started = time.perf_counter()
        x = np.asarray(candidates, dtype=np.float64)
        leaves = np.ascontiguousarray(x, dtype=np.float32)
        per_tree = np.stack(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(tree.predict)(leaves, check_input=False) for tree in self.objective.estimators_
            )
        )
        mean, deviation = per_tree.mean(0), per_tree.std(0)
        scores = dict(
            log_improvement=log_expected_improvement(np.log(self.best), mean, deviation),
            beat_probability=norm.cdf((np.log(self.best) - mean) / np.maximum(deviation, 1e-9)),
            predicted_mse=np.exp(mean),
            uncertainty=np.exp(per_tree).std(0),
        )
        scores["acquisition"] = scores["log_improvement"]
        if self.feasibility is not None:
            completion = self.feasibility.predict_proba(x)[:, list(self.feasibility.classes_).index(True)]
            scores["completion_probability"] = completion
            scores["acquisition"] = scores["log_improvement"] + COMPLETION_WEIGHT * np.log(
                np.maximum(completion, 1e-9)
            )
        self.scores = scores
        if not np.isfinite(scores["acquisition"]).all():
            raise RuntimeError("the forest produced a nonfinite acquisition value")
        index = int(scores["acquisition"].argmax())
        self.diagnostics = dict(
            predicted_mse=float(scores["predicted_mse"][index]),
            uncertainty=float(scores["uncertainty"][index]),
            completion_probability=None
            if self.feasibility is None
            else float(scores["completion_probability"][index]),
        )
        emit("forest_acquisition", candidates=len(x), seconds=time.perf_counter() - started)
        return index, float(scores["acquisition"][index])


@timed("surrogates.botorch_model")
def botorch_model(x, losses, complete, seed, device):
    import botorch

    started = time.perf_counter()
    compute.require(device)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Search(x, losses, complete, device)
    model.sampler, model.phase = "BoTorch", "botorch"
    model.settings = dict(model="MAP-SAAS GP", device=device, botorch_version=botorch.__version__)
    emit("botorch_build", device=device, seconds=time.perf_counter() - started)
    return model


@timed("surrogates.forest_model")
def forest_model(x, losses, complete, seed, device):
    import sklearn

    started = time.perf_counter()
    model = Forest(x, losses, complete, seed)
    model.sampler, model.phase = "SMAC", "forest"
    model.settings = dict(
        model="random forest",
        trees=model.trees,
        transform="log",
        completion_weight=COMPLETION_WEIGHT,
        sklearn_version=sklearn.__version__,
    )
    emit("forest_build", trees=model.trees, seconds=time.perf_counter() - started)
    return model


suggest_botorch = partial(propose, botorch_model)
suggest_forest = partial(propose, forest_model)
