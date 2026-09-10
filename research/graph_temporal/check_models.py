import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache import write_json
from .data import Bank, pack
from .registry import DESIGNS, create, sample


def check(directory, output, device='cuda', turn_context='none'):
    from .turns import blocks as turn_blocks

    torch.set_num_threads(2)
    bank = Bank(directory)
    campaign = max(bank.manifest['campaigns'], key=lambda c: c['nodes'])
    start = min(16, campaign['decisions'] - 1)
    if turn_context == 'end':
        a = bank.campaign(campaign['index'])
        start = int(a['turn_starts'][min(2, len(a['turn_starts']) - 1)])
    block = campaign['index'], start, min(start + 16, campaign['decisions'])
    results = []
    for model, designs in DESIGNS.items():
        for design in designs:
            cfg = sample(model, design, np.random.default_rng(871), 771)
            cfg.update(hidden=16, layers=2, dropout=0.0)
            cfg.update(turn_context=turn_context, turn_lookback=3)
            started = time.perf_counter()
            context = turn_blocks(bank, [block], cfg)[0]
            b = pack(bank, context, cfg, device)
            packed = time.perf_counter()
            net = create(cfg, len(bank.manifest['categories'])).to(device)
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
                clone = create(cfg, len(bank.manifest['categories'])).to(device).eval()
                clone.load_state_dict(net.state_dict(), strict=True)
                torch.testing.assert_close(before, clone(b))
            result = dict(model=model, design=design, config=cfg, block=list(block), rows=len(target),
                pack_seconds=packed - started, total_seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in net.parameters()), unused_parameters=[name for name, p in net.named_parameters() if p.grad is None],
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0, ok=True)
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'unused_parameters')}), flush=True)
            del net, clone, b, values, loss
    write_json(Path(output), dict(check='forward_backward_checkpoint_reload', results=results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--turn-context', choices=('none', 'end'), default='none')
    args = parser.parse_args()
    check(args.directory, args.output, args.device, args.turn_context)


if __name__ == '__main__':
    main()
