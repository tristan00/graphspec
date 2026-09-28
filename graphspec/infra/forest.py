import time

import numpy as np

from .log import emit

TREES = 100
COMPLETION_WEIGHT = 3.0


def log_expected_improvement(best, mean, deviation):
    from scipy.stats import norm

    deviation = np.maximum(deviation, 1e-9)
    z = (best - mean) / deviation
    far = z < -6
    body = np.where(far, 0.0, z * norm.cdf(z) + norm.pdf(z))
    tail = norm.logpdf(z) - 2 * np.log(np.maximum(-z, 1e-9))
    return np.log(deviation) + np.where(far, tail, np.log(np.maximum(body, 1e-300)))


class Forest:
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
