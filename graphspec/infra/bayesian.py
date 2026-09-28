import time

import numpy as np
import torch

from .log import emit

FIT_ITERATIONS = 1000
BATCH = 64


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
    def __init__(self, x, losses, completed, device="cpu"):
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
