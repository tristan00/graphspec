import argparse
from pathlib import Path
import time

import torch

from graphspec import spec as specs
from graph_tuning_infra import compute
from graph_tuning_infra.files import read_json, write_json
from graph_tuning_infra.log import configure, emit
from .data import Bank, pack
from .model_spec import configuration, describe, groups, samples
from .models import build


def probe(bank, block, cfg, device):
    from .turn_inputs import capture_schedule
    from .turns import blocks as turn_blocks, enabled

    started = time.perf_counter()
    context = turn_blocks(bank, [block], cfg)[0]
    b = pack(bank, context, cfg, device)
    if enabled(cfg):
        capture_schedule(b)
    packed = time.perf_counter()
    net = build(cfg, bank.registry, len(bank.categories)).to(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    values = net(b)
    target = (b['target'] - bank.split['reward_mean']) / bank.split['reward_sd']
    loss = (values - target).square().mean()
    loss.backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    assert values.shape == target.shape and torch.isfinite(values).all()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    net.eval()
    with torch.no_grad():
        before = net(b)
        clone = build(cfg, bank.registry, len(bank.categories)).to(device).eval()
        clone.load_state_dict(net.state_dict(), strict=True)
        torch.testing.assert_close(before, clone(b))
    return dict(rows=len(target), pack_seconds=packed - started, total_seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in net.parameters()), unused_parameters=[name for name, p in net.named_parameters() if p.grad is None],
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0, ok=True)


def check(directory, output, device, graph=None, spec_path=None):
    entered = time.perf_counter()
    if device == 'cuda':
        compute.reserve()
    bank = Bank(directory)
    chosen = max(bank.manifest['graphs'], key=lambda entry: entry['observations'])['index'] if graph is None else graph
    given = read_json(spec_path) if spec_path else {}
    models = samples(bank.registry)
    unknown = sorted(set(given) - {parameter.name for model in models for parameter in specs.space(bank.registry, groups(model))})
    if unknown:
        raise ValueError('spec file names unknown parameters: ' + ', '.join(unknown))
    results = []
    for model in models:
        cfg = configuration(bank.registry, model, given, 771)
        candidates = sorted(block for partition in (0, 1) for block in bank.blocks(partition, cfg['graph']['time/history'], cfg) if block[0] == chosen)
        if not candidates:
            raise ValueError('graph %d has no labels to check' % chosen)
        block = candidates[-1]
        started = time.perf_counter()
        try:
            outcome = probe(bank, block, cfg, device)
        except torch.OutOfMemoryError:
            outcome = dict(ok=False, error='out of memory', total_seconds=time.perf_counter() - started)
        if device == 'cuda':
            torch.cuda.empty_cache()
        result = dict(model=describe(model), graph=specs.digest(cfg['graph']), config=cfg, block=list(block), **outcome)
        results.append(result)
        emit('check_model', **{k: v for k, v in result.items() if k not in ('config', 'unused_parameters')})
    write_json(Path(output), dict(check='forward_backward_checkpoint_reload', graph=chosen, results=results))
    failed = [result['model'] for result in results if not result['ok']]
    emit('check_exit', models=len(results), failed=len(failed), seconds=time.perf_counter() - entered)
    if failed:
        raise SystemExit('%d models ran out of GPU memory: %s' % (len(failed), '; '.join(failed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default=compute.DEVICE)
    parser.add_argument('--graph', type=int)
    parser.add_argument('--spec')
    args = parser.parse_args()
    configure()
    check(args.directory, args.output, args.device, args.graph, args.spec)


if __name__ == '__main__':
    main()
