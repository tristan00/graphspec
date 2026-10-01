from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graphspec.infra import compute
from graphspec.infra.code import digest as code_digest
from graphspec.infra.files import read_json, sha256_bytes, sha256_file, write_json
from graphspec.infra.log import emit
from graphspec.infra.runtime import output_path
from graphspec.infra.timing import measure, timed
from graphspec.operations.parameters import digest
from graphspec.schema.schema import Schema
from tuning.bank.bank import PACKAGES
from tuning.measures.measures import measured, scored

from .losses import LOSSES, admitted

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer

    from tuning.bank.bank import Deriver
    from tuning.batches.packing import Packer

    from .budget import FitBudget

Block = Sequence[int]


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
        from tuning.bank.bank import Deriver
        from tuning.bank.variants import prepare
        from tuning.batches.packing import Packer
        from tuning.families.model_spec import describe
        from tuning.families.models import build

        self.entered = time.perf_counter()
        self.directory = output_path(directory)
        self.stated = stated
        self.loss = LOSSES.named(stated.loss)
        self.device = stated.device
        self.cuda = compute.require(self.device).type == "cuda"
        self.budget: FitBudget = budget
        with self.budget.measure("graph_prepare"):
            prepare(bank_directory, config, stated.workers, self.device, self.directory / "timing_workers")
        self.budget.check()
        with measure("source_digest"):
            self.revision = code_digest(PACKAGES)
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
        self.timing = self.budget.timings
        self.timing.update(
            packer=0.0,
            train_pack=0.0,
            train_compute=0.0,
            validation_pack=0.0,
            validation_compute=0.0,
        )
        with self.budget.measure("packer"):
            self.packer: Packer = Packer(
                self.bank, config, self.device, stated.workers, self.directory / "timing_workers"
            )
        try:
            with measure("model_build"):
                self.net: nn.Module = build(config, self.bank.fitted(config)).to(self.device)
            with measure("optimizer_setup"):
                self.optimizer: Optimizer = torch.optim.AdamW(
                    self.net.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    fused=self.cuda,
                )
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
        except BaseException:
            self.packer.close()
            raise
        self.losses: list[float] = []
        self.trained_rows: list[int] = []
        self.epoch_losses: list[float] = []

    @timed("schedule")
    def schedule(self, training_blocks: Sequence[Block], batch_size: int) -> Schedule:
        from tuning.batches.batching import Shuffle, batches, identities
        from tuning.families.periods import blocks as period_blocks

        if sorted(map(tuple, training_blocks)) != sorted(self.bank.blocks(0, self.history, self.config)):
            raise ValueError("training blocks must cover the entire frozen training partition")
        self.trained = self.bank.members(self.config, 0)
        self.scored = self.bank.members(self.config, 1)
        epochs = self.config["epochs"]
        if epochs < 1:
            raise ValueError("each fit requires at least one complete training epoch")
        with measure("period_context"):
            context = period_blocks(self.bank, training_blocks, self.config)
        with measure("graph_sizes"):
            sizes = self.bank.sizes(self.config)
        with measure("batch_schedule"):
            training, ends, keys = [], set(), identities(self.bank, self.config)
            for epoch in range(epochs):
                training.extend(
                    (epoch, group)
                    for group in batches(
                        context, batch_size, sizes, Shuffle(self.stated.seed, epoch, keys), frame_budget=1024
                    )
                )
                ends.add(len(training))
        emit(
            "fit_enter",
            model=self.description,
            graph=self.graph_digest[:10],
            updates=len(training),
            epochs=epochs,
            training_rows=len(self.trained),
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
                value = self.loss(predictions, batch, self.bank)
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
        if not np.array_equal(np.sort(rows), self.trained):
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
            if np.any(self.bank.partitions(self.config)[batch["rows"]] != 0):
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

        from tuning.families.models import build

        with measure("checkpoint_read"):
            loaded = torch.load(
                self.directory / "checkpoint.pt", map_location=self.device, weights_only=False
            )
        clone = build(loaded["config"], self.bank.fitted(loaded["config"])).to(self.device).eval()
        clone.load_state_dict(loaded["state_dict"], strict=True)
        torch.testing.assert_close(predictions, clone(batch), atol=2e-5, rtol=2e-5)

    @timed("validation")
    def evaluate(self, packer: Packer, schedule: Schedule) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.budget.pulse("evaluation")

        from tuning.batches.batching import batches
        from tuning.families.periods import blocks as period_blocks

        values, indices = [], []
        with measure("validation_schedule"):
            validation = batches(
                period_blocks(self.bank, self.bank.blocks(1, self.history, self.config), self.config),
                schedule.batch_size,
                schedule.sizes,
                None,
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
                        series = batch["series"]
                        raw = predictions * predictions.new_tensor(self.bank.scale)[series]
                        raw = raw + predictions.new_tensor(self.bank.center)[series]
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
        if not np.array_equal(indices, self.scored):
            raise ValueError("holdout predictions do not exactly cover the frozen partition")
        with measure("metrics"):
            held = scored(
                self.bank.targets[indices],
                values,
                self.bank.task_of[indices],
                self.bank.series[indices],
                self.bank.series_names,
                self.bank.scale,
            )
            metrics = measured(self.stated.measures, held)
        with measure("prediction_save"):
            np.savez_compressed(
                self.directory / "predictions.npz",
                row_index=indices,
                prediction=values,
                target=self.bank.targets[indices],
                series=self.bank.series[indices],
                **({} if not self.bank.path else dict(tau=self.bank.taus[indices])),
            )
        self.budget.check()
        result = dict(
            status="complete",
            model=digest(self.config["model"]),
            description=self.description,
            graph=self.graph_digest,
            config=self.config,
            loss=self.stated.loss,
            measures=metrics,
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
            train_available_rows=len(self.trained),
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
            measure=self.stated.measures[0],
            value=metrics[self.stated.measures[0]]["value"],
            seconds=result["seconds"],
            train_rows=result["train_unique_rows"],
        )
        with measure("result_write"):
            write_json(self.directory / "result.json", result)
        return result


def fit(bank_directory, directory, config, training_blocks, batch_size, deadline, statement):
    from tuning.statement import stated as checked

    from .budget import FitBudget

    stated = checked(statement)
    compute.require(stated.device)
    if config["seed"] != stated.seed:
        raise ValueError(
            "the seed of the configuration is %r and the seed of the statement is %r"
            % (config["seed"], stated.seed)
        )
    if "splits" not in config:
        raise KeyError("the configuration states no splits to train on and to score on")
    if config["splits"] != stated.splits():
        raise ValueError(
            "the splits of the configuration are %r and those of the statement are %r"
            % (config["splits"], stated.splits())
        )
    if "pipeline" not in config:
        raise KeyError("the configuration states no pipeline")
    declared = read_json(Path(bank_directory) / "manifest.json")["spec"]["schema"]
    admitted(Schema(declared).targets(), stated.loss, stated.measures)
    with FitBudget(directory, deadline, stated.preparation_seconds) as budget:
        with budget.measure("prepare"):
            session = TrainingSession(bank_directory, directory, config, budget, stated)
        try:
            budget.pulse("scheduling")
            schedule = session.schedule(training_blocks, batch_size)
            session.net.train()
            session.train(session.packer, schedule)
            session.checkpoint(training_blocks)
            indices, values = session.evaluate(session.packer, schedule)
        finally:
            budget.pulse("cleanup")
            with measure("packer_cleanup"):
                session.packer.close()
        result = session.save_result(indices, values, batch_size)
    result["timeline"] = budget.timeline.snapshot()
    write_json(Path(directory) / "result.json", result)
    return result
