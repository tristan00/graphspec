import argparse
import time
from pathlib import Path

import torch

from graphspec import spec as specs
from graphspec.infra import compute
from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure, emit
from graphspec.infra.runtime import exclusive, output_path

from .batching import pack_batch
from .data import Bank
from .model_spec import configuration, describe, groups, samples
from .models import build
from .training import loss


def probe(bank, block, cfg, device):
    from .turns import blocks as turn_blocks

    started = time.perf_counter()
    b = pack_batch(bank, turn_blocks(bank, [block], cfg), cfg, device)
    packed = time.perf_counter()
    net = build(cfg, bank.registry, len(bank.categories)).to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    values = net(b)
    if values.shape != b["target"].shape or not torch.isfinite(values).all():
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
        clone = build(cfg, bank.registry, len(bank.categories)).to(device).eval()
        clone.load_state_dict(net.state_dict(), strict=True)
        torch.testing.assert_close(before, clone(b))
    return dict(
        rows=len(b["target"]),
        pack_seconds=packed - started,
        total_seconds=time.perf_counter() - started,
        parameters=sum(p.numel() for p in net.parameters()),
        unused_parameters=[name for name, p in net.named_parameters() if p.grad is None],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else 0,
        ok=True,
    )


@exclusive
def check(directory, output, device, graph=None, spec_path=None):
    entered = time.perf_counter()
    output = output_path(output)
    compute.reserve(device)
    bank = Bank(directory, device=device)
    chosen = (
        max(bank.manifest["graphs"], key=lambda entry: entry["observations"])["index"]
        if graph is None
        else graph
    )
    given = read_json(spec_path) if spec_path else {}
    models = samples(bank.registry)
    unknown = sorted(
        set(given)
        - {parameter.name for model in models for parameter in specs.space(bank.registry, groups(model))}
    )
    if unknown:
        raise ValueError("spec file names unknown parameters: " + ", ".join(unknown))
    results = []
    for model in models:
        cfg = configuration(bank.registry, model, given, 771)
        candidates = sorted(
            block
            for partition in (0, 1)
            for block in bank.blocks(partition, cfg["graph"]["time/history"], cfg)
            if block[0] == chosen
        )
        if not candidates:
            raise ValueError("graph %d has no labels to check" % chosen)
        block = candidates[-1]
        outcome = probe(bank, block, cfg, device)
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
    parser.add_argument("--graph", type=int)
    parser.add_argument("--spec")
    args = parser.parse_args()
    configure()
    check(args.directory, args.output, args.device, args.graph, args.spec)


if __name__ == "__main__":
    main()
