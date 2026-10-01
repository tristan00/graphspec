import argparse
import time
from pathlib import Path

import torch

from graphspec.infra import compute
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure, emit
from graphspec.infra.processes.locks import exclusive
from graphspec.infra.runtime import output_path
from graphspec.operations import parameters as specs
from tuning.bank.bank import Deriver
from tuning.batches.batching import pack_batch
from tuning.families.model_spec import configuration, describe, groups, samples
from tuning.families.models import build
from tuning.statement import paired, seen

from .losses import Summed


def probe(bank, block, cfg, device, loss):
    from tuning.families.periods import blocks as period_blocks

    started = time.perf_counter()
    b = pack_batch(bank, period_blocks(bank, [block], cfg), cfg, device)
    packed = time.perf_counter()
    net = build(cfg, bank.fitted(cfg)).to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    values = net(b)
    if values.shape != b["series"].shape or not torch.isfinite(values).all():
        raise ValueError("model output must be finite and match the target shape")
    loss(values, b, bank).backward()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("model must produce finite gradients")
    net.eval()
    with torch.no_grad():
        before = net(b)
        clone = build(cfg, bank.fitted(cfg)).to(device).eval()
        clone.load_state_dict(net.state_dict(), strict=True)
        torch.testing.assert_close(before, clone(b))
    return dict(
        rows=len(b["series"]),
        pack_seconds=packed - started,
        total_seconds=time.perf_counter() - started,
        parameters=sum(p.numel() for p in net.parameters()),
        unused_parameters=[name for name, p in net.named_parameters() if p.grad is None],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else 0,
        ok=True,
    )


def settled(registry, model, given):
    started = time.perf_counter()

    def read(parameter):
        if parameter.name not in given:
            raise ValueError(
                "the settings of the check state no %s, which the model %s reads" % (parameter.name, describe(model))
            )
        return parameter.check(given[parameter.name])

    result = specs.draw(specs.settings(registry, groups(model)), read)
    emit("check_settled", settings=len(result), seconds=time.perf_counter() - started)
    return result


@exclusive
def check(directory, output, device, seed, splits, loss, pipeline_path, settings_path, graph=None):
    entered = time.perf_counter()
    output = output_path(output)
    compute.reserve(device)
    bank = Deriver(directory, device, RESIDENT)
    loss = Summed(loss, bank.registry.schema.targets())
    pipeline, given = read_json(pipeline_path), read_json(settings_path)
    models = samples(bank.registry)
    unknown = sorted(
        set(given)
        - {parameter.name for model in models for parameter in specs.settings(bank.registry, groups(model))}
    )
    if unknown:
        raise ValueError("the settings of the check name unknown parameters: " + ", ".join(unknown))
    first = settled(bank.registry, models[0], given)
    shown = bank.viewed(configuration(bank.registry, models[0], pipeline, first, seed, splits)).graphs
    chosen = max(shown, key=lambda entry: entry["observations"])["index"] if graph is None else graph
    results = []
    for model in models:
        cfg = configuration(bank.registry, model, pipeline, settled(bank.registry, model, given), seed, splits)
        candidates = sorted(
            block
            for partition in (0, 1)
            for block in bank.blocks(partition, cfg["graph"]["time/history"], cfg)
            if block[0] == chosen
        )
        if not candidates:
            raise ValueError("graph %d has no labels to check" % chosen)
        block = candidates[-1]
        outcome = probe(bank, block, cfg, device, loss)
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        result = dict(
            model=describe(model), graph=specs.digest(cfg["graph"]), config=cfg, block=list(block), **outcome
        )
        results.append(result)
        emit("check_model", **{k: v for k, v in result.items() if k not in ("config", "unused_parameters")})
    write_json(Path(output), dict(check="forward_backward_checkpoint_reload", graph=chosen, results=results))
    emit("check_exit", models=len(results), seconds=time.perf_counter() - entered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trained", nargs="+", required=True)
    parser.add_argument("--scored", nargs="+", required=True)
    parser.add_argument("--sees", nargs="+", required=True)
    parser.add_argument("--graph", type=int)
    parser.add_argument(
        "--loss",
        nargs="+",
        required=True,
        help="the registered loss of every kind of target of the bank, as kind=loss pairs, whose sum every model is "
        "checked with",
    )
    parser.add_argument("--pipeline", required=True, help="the pipeline every model derives its variant by")
    parser.add_argument("--settings", required=True, help="every setting of the graph any model reads")
    args = parser.parse_args()
    configure()
    splits = dict(trained=args.trained, scored=args.scored, sees=seen(args.sees))
    check(
        args.directory,
        args.output,
        args.device,
        args.seed,
        splits,
        paired(args.loss),
        args.pipeline,
        args.settings,
        args.graph,
    )


if __name__ == "__main__":
    main()
