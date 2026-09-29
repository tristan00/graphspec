from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graphspec.infra import compute, sources
from graphspec.infra.files import sha256_bytes, sha256_file, write_json
from graphspec.infra.log import emit
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import measure, timed
from graphspec.spec import digest

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer

    from .budget import FitBudget
    from .data import Deriver
    from .packing import Packer

PACKAGES = (Path(__file__).parent, Path(__file__).parent.parent / "graphspec")
Block = Sequence[int]


def score(targets, predictions):
    started = time.perf_counter()
    targets = np.asarray(targets, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or targets.shape != values.shape:
        raise ValueError("targets and predictions must be matching vectors")
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("validation predictions must be nonempty and finite")
    mse = float(np.mean((values - targets) ** 2))
    emit("score", seconds=time.perf_counter() - started, rows=len(values), raw_mse=mse)
    variance = float(targets.var())
    if variance == 0:
        raise ValueError("R2 requires varying validation targets")
    return dict(raw_val_mse=mse, val_r2=1 - mse / variance)


def path_score(targets, predictions, queries, series, names, scale):
    started = time.perf_counter()
    targets = np.asarray(targets, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or not targets.shape == values.shape == queries.shape == series.shape:
        raise ValueError("every point needs a target, a prediction, a query and a series")
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("validation predictions must be nonempty and finite")
    result = {}
    for code, name in enumerate(names):
        member = series == code
        if not member.any():
            raise ValueError("series %s has no validation points" % name)
        owner = np.unique(queries[member], return_inverse=True)[1].reshape(-1)
        weight = 1.0 / np.bincount(owner)[owner]
        mse = float((weight * (values[member] - targets[member]) ** 2).sum() / weight.sum())
        result[name] = dict(
            val_loss=mse / float(scale[code]) ** 2,
            raw_val_mse=mse,
            points=int(member.sum()),
            queries=int(owner.max()) + 1,
        )
    emit(
        "score",
        seconds=time.perf_counter() - started,
        rows=len(values),
        **{name + "_loss": entry["val_loss"] for name, entry in result.items()},
    )
    return dict(val_loss=float(np.mean([entry["val_loss"] for entry in result.values()])), series=result)


def path_loss(predictions, targets, point_query, series, queries, count):
    import torch

    error = (predictions - targets).square()
    group = point_query * count + series
    total = error.new_zeros(queries * count).index_add_(0, group, error)
    points = error.new_zeros(queries * count).index_add_(0, group, error.new_ones(len(error)))
    present = (points > 0).to(error.dtype)
    kind = torch.arange(queries * count, device=error.device) % count
    series_total = error.new_zeros(count).index_add_(0, kind, total / points.clamp_min(1))
    series_queries = error.new_zeros(count).index_add_(0, kind, present)
    return (series_total / series_queries.clamp_min(1)).sum() / (series_queries > 0).sum()


def standardisation(predictions, batch, bank):
    center, scale = predictions.new_tensor(bank.center), predictions.new_tensor(bank.scale)
    if bank.level == "path":
        return center[batch["series"]], scale[batch["series"]]
    return center[0], scale[0]


def loss(predictions, batch, bank):
    center, scale = standardisation(predictions, batch, bank)
    targets = (batch["target"] - center) / scale
    if bank.level == "path":
        return path_loss(
            predictions, targets, batch["point_query"], batch["series"], batch["queries"], len(bank.center)
        )
    return (predictions - targets).square().mean()


@dataclass(frozen=True)
class Schedule:
    training: list[tuple[int, list[Block]]]
    ends: set[int]
    sizes: dict[int, tuple[int, float]]
    batch_size: int


class TrainingSession:
    def __init__(self, bank_directory, directory, config: dict, budget, stated):
        import torch

        from graphspec.infra.arrays import RESIDENT

        from .data import Deriver
        from .model_spec import describe
        from .models import build
        from .variants import prepare

        self.entered = time.perf_counter()
        self.directory = output_path(directory)
        self.stated = stated
        self.device = stated.device
        self.cuda = compute.require(self.device).type == "cuda"
        self.budget: FitBudget = budget
        with self.budget.measure("graph_prepare"):
            prepare(bank_directory, config, stated.workers, self.device, self.directory / "timing_workers")
        self.budget.check()
        with measure("source_digest"):
            self.revision = sources.digest(PACKAGES)
        self.config = config
        self.graph_digest = digest(config["graph"])
        self.history = config["graph"]["time/history"]
        self.description = describe(config["model"])
        with measure("device_setup"):
            torch.manual_seed(stated.seed)
            np.random.seed(stated.seed)
            compute.reserve(self.device)
        with measure("bank_load_and_verify"):
            self.bank: Deriver = Deriver(bank_directory, self.device, RESIDENT)
            self.supervision_digest = sha256_file(Path(bank_directory) / "supervision.json")
            if self.supervision_digest != self.bank.manifest["supervision_sha256"]:
                raise ValueError("frozen supervision changed")
        with measure("model_build"):
            self.net: nn.Module = build(config, self.bank.registry, len(self.bank.categories)).to(self.device)
        with measure("optimizer_setup"):
            self.optimizer: Optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                fused=self.cuda,
            )
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.timing = self.budget.timings
        self.timing.update(
            packer=0.0,
            train_pack=0.0,
            train_compute=0.0,
            validation_pack=0.0,
            validation_compute=0.0,
        )
        self.losses: list[float] = []
        self.trained_rows: list[int] = []
        self.epoch_losses: list[float] = []

    @timed("schedule")
    def schedule(self, training_blocks: Sequence[Block], batch_size: int) -> Schedule:
        from .batching import batches
        from .turns import blocks as turn_blocks

        if sorted(map(tuple, training_blocks)) != sorted(self.bank.blocks(0, self.history, self.config)):
            raise ValueError("training blocks must cover the entire frozen training partition")
        epochs = self.config["epochs"]
        if epochs < 1:
            raise ValueError("each fit requires at least one complete training epoch")
        with measure("turn_context"):
            context = turn_blocks(self.bank, training_blocks, self.config)
        with measure("graph_sizes"):
            sizes = self.bank.sizes(self.config)
        with measure("batch_schedule"):
            training, ends = [], set()
            for epoch in range(epochs):
                training.extend(
                    (epoch, group)
                    for group in batches(
                        context, batch_size, sizes, self.stated.seed + epoch, frame_budget=1024
                    )
                )
                ends.add(len(training))
        emit(
            "fit_enter",
            model=self.description,
            graph=self.graph_digest[:10],
            updates=len(training),
            epochs=epochs,
            training_rows=len(self.bank.split["train_indices"]),
            batch_sequences=batch_size,
            config=self.config,
        )
        return Schedule(training, ends, sizes, batch_size)

    def optimize(self, batch: dict) -> float:
        import torch

        with self.budget.measure("train_compute"):
            with measure("zero_grad"):
                self.optimizer.zero_grad(set_to_none=True)
            with measure("forward"):
                predictions = self.net(batch)
            with measure("loss"):
                value = loss(predictions, batch, self.bank)
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite training loss")
            with measure("backward"):
                value.backward()
            with measure("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.config["clip"], error_if_nonfinite=True
                )
            with measure("optimizer_step"):
                self.optimizer.step()
            if self.cuda:
                with measure("device_wait"):
                    torch.cuda.synchronize(self.device)
            return float(value.detach())

    @timed("epoch_bookkeeping")
    def complete_epoch(self, epoch: int, rows: list[int], loss_sum: float, started: float) -> float:
        if not np.array_equal(np.sort(rows), np.asarray(self.bank.split["train_indices"])):
            raise ValueError("epoch omitted or duplicated training rows")
        self.epoch_losses.append(loss_sum / len(rows))
        finished = time.perf_counter()
        self.budget.epoch_seconds.append(finished - started)
        emit(
            "epoch_complete",
            epoch=epoch + 1,
            rows=len(rows),
            loss=self.epoch_losses[-1],
            epoch_seconds=self.budget.epoch_seconds[-1],
            mean_epoch_seconds=float(np.mean(self.budget.epoch_seconds)),
            seconds=finished - self.entered,
        )
        return finished

    @timed("training")
    def train(self, packer: Packer, schedule: Schedule):
        stream = self.budget.batches(packer.stream([blocks for _, blocks in schedule.training]), "train_pack")
        epoch_rows, epoch_sum = [], 0.0
        epoch_started = time.perf_counter()
        self.budget.pulse("training")
        for update, ((epoch, _), batch) in enumerate(zip(schedule.training, stream, strict=True)):
            self.budget.check()
            if np.any(self.bank.partitions[batch["rows"]] != 0):
                raise ValueError("holdout row reached the optimizer")
            self.trained_rows.extend(batch["rows"].tolist())
            epoch_rows.extend(batch["rows"].tolist())
            self.losses.append(self.optimize(batch))
            epoch_sum += self.losses[-1] * len(batch["rows"])
            self.budget.check()
            if update + 1 in schedule.ends:
                epoch_started = self.complete_epoch(epoch, epoch_rows, epoch_sum, epoch_started)
                epoch_rows, epoch_sum = [], 0.0
                self.budget.pulse("training")
            if (update + 1) % 16 == 0:
                emit(
                    "training_progress",
                    updates=update + 1,
                    total_updates=len(schedule.training),
                    epoch=epoch + 1,
                    epoch_rows=len(epoch_rows),
                    loss=float(np.mean(self.losses[-16:])),
                    seconds=time.perf_counter() - self.entered,
                )
            del batch
        if epoch_rows or len(self.epoch_losses) != self.config["epochs"]:
            raise ValueError("evaluation requires every requested epoch to complete")
        self.budget.check()

    @timed("checkpoint_save")
    def checkpoint(self, training_blocks: Sequence[Block]):
        import torch

        self.budget.pulse("checkpoint")
        self.net.eval()
        checkpoint = dict(
            config=self.config,
            state_dict=self.net.state_dict(),
            center=self.bank.center.tolist(),
            scale=self.bank.scale.tolist(),
            supervision_sha256=self.supervision_digest,
            source_sha256=self.revision,
            train_blocks=training_blocks,
            train_rows=sorted(set(self.trained_rows)),
            epochs=len(self.epoch_losses),
            requested_epochs=self.config["epochs"],
            epoch_seconds=self.budget.epoch_seconds,
        )
        with measure("checkpoint_write"):
            torch.save(checkpoint, self.directory / "checkpoint.pt")

    @timed("checkpoint_reload")
    def verify_checkpoint(self, predictions: Tensor, batch: dict):
        import torch

        from .models import build

        with measure("checkpoint_read"):
            loaded = torch.load(
                self.directory / "checkpoint.pt", map_location=self.device, weights_only=False
            )
        clone = build(loaded["config"], self.bank.registry, len(self.bank.categories)).to(self.device).eval()
        clone.load_state_dict(loaded["state_dict"], strict=True)
        torch.testing.assert_close(predictions, clone(batch), atol=2e-5, rtol=2e-5)

    @timed("validation")
    def evaluate(self, packer: Packer, schedule: Schedule) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.budget.pulse("evaluation")

        from .batching import batches
        from .turns import blocks as turn_blocks

        values, indices = [], []
        with measure("validation_schedule"):
            validation = batches(
                turn_blocks(self.bank, self.bank.blocks(1, self.history, self.config), self.config),
                schedule.batch_size,
                schedule.sizes,
                frame_budget=1024,
            )
        stream = self.budget.batches(packer.stream(validation), "validation_pack")
        with torch.inference_mode():
            for index, (_, batch) in enumerate(zip(validation, stream, strict=True)):
                self.budget.check()
                with self.budget.measure("validation_compute"):
                    with measure("forward"):
                        predictions = self.net(batch)
                    if index == 0:
                        self.verify_checkpoint(predictions, batch)
                    with measure("prediction_collect"):
                        if not torch.isfinite(predictions).all():
                            raise FloatingPointError("nonfinite holdout predictions")
                        center, scale = standardisation(predictions, batch, self.bank)
                        raw = predictions * scale + center
                        values.extend(raw.cpu().tolist())
                        indices.extend(batch["rows"].tolist())
                if (index + 1) % 256 == 0:
                    emit(
                        "validation_progress",
                        blocks=index + 1,
                        total_blocks=len(validation),
                        rows=len(indices),
                        seconds=time.perf_counter() - self.entered,
                    )
                del batch, predictions
        return np.asarray(indices, np.int64), np.asarray(values, np.float64)

    @timed("score_and_save")
    def save_result(self, indices: np.ndarray, values: np.ndarray, batch_size: int) -> dict:
        import torch

        from .budget import policy

        self.budget.pulse("scoring")
        ordering = np.argsort(indices)
        indices, values = indices[ordering], values[ordering]
        if not np.array_equal(indices, np.asarray(self.bank.split["validation_indices"])):
            raise ValueError("holdout predictions do not exactly cover the frozen partition")
        path = self.bank.level == "path"
        with measure("metrics"):
            targets = self.bank.targets[indices].astype(np.float64)
            if path:
                metrics = path_score(
                    targets,
                    values,
                    self.bank.query_of[indices],
                    self.bank.series[indices],
                    self.bank.series_names,
                    self.bank.scale,
                )
            else:
                metrics = score(targets, values)
        with measure("prediction_save"):
            np.savez_compressed(
                self.directory / "predictions.npz",
                row_index=indices,
                prediction=values,
                target=self.bank.targets[indices],
                **(dict(tau=self.bank.taus[indices], series=self.bank.series[indices]) if path else {}),
            )
        self.budget.check()
        result = dict(
            status="complete",
            model=digest(self.config["model"]),
            description=self.description,
            graph=self.graph_digest,
            config=self.config,
            **metrics,
            seconds=time.perf_counter() - self.entered,
            timings=self.timing,
            parameters=sum(p.numel() for p in self.net.parameters()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(self.device) if self.cuda else 0,
            optimizer_updates=len(self.losses),
            epochs=len(self.epoch_losses),
            requested_epochs=self.config["epochs"],
            epoch_seconds=self.budget.epoch_seconds,
            budget_policy=policy(self.stated.preparation_seconds),
            statement=self.stated.mapping(),
            epoch_losses=self.epoch_losses,
            batch_sequences=batch_size,
            training_coverage_complete=True,
            training_examples_seen=len(self.trained_rows),
            train_unique_rows=len(set(self.trained_rows)),
            train_available_rows=len(self.bank.split["train_indices"]),
            train_graphs=len({self.bank.rows[i]["graph"] for i in self.trained_rows}),
            holdout_rows=len(indices),
            training_losses=self.losses,
            checkpoint_reload=True,
            supervision_sha256=self.supervision_digest,
            source_sha256=self.revision,
            prediction_rows_sha256=sha256_bytes(indices.tobytes()),
            completed=datetime.now(timezone.utc).isoformat(),
        )
        self.budget.pulse("complete")
        result["fit_phases"] = dict(self.budget.phase_seconds)
        emit(
            "fit_exit",
            model=self.description,
            graph=self.graph_digest[:10],
            loss=metrics["val_loss"] if path else metrics["raw_val_mse"],
            seconds=result["seconds"],
            train_rows=result["train_unique_rows"],
        )
        with measure("result_write"):
            write_json(self.directory / "result.json", result)
        return result


def fit(bank_directory, directory, config, training_blocks, batch_size, deadline, statement):
    from .budget import FitBudget
    from .packing import Packer
    from .statement import stated as checked

    stated = checked(statement)
    compute.require(stated.device)
    if config["seed"] != stated.seed:
        raise ValueError(
            "the seed of the configuration is %r and the seed of the statement is %r"
            % (config["seed"], stated.seed)
        )
    with FitBudget(directory, deadline, stated.preparation_seconds) as budget:
        with budget.measure("prepare"):
            session = TrainingSession(bank_directory, directory, config, budget, stated)
        budget.pulse("scheduling")
        schedule = session.schedule(training_blocks, batch_size)
        session.net.train()
        budget.pulse("packer_setup")
        with budget.measure("packer"):
            packer = Packer(
                session.bank, config, stated.device, stated.workers, session.directory / "timing_workers"
            )
        try:
            session.train(packer, schedule)
            session.checkpoint(training_blocks)
            indices, values = session.evaluate(packer, schedule)
        finally:
            budget.pulse("cleanup")
            with measure("packer_cleanup"):
                packer.close()
        result = session.save_result(indices, values, batch_size)
    result["timeline"] = budget.timeline.snapshot()
    write_json(Path(directory) / "result.json", result)
    return result
