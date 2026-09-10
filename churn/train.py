import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import spec


LOG = logging.getLogger(__name__)


def load_campaign(directory, index, names):
    path = Path(directory) / ('campaign_%04d' % index)
    return {name: np.load(path / (name + '.npy'), allow_pickle=False) for name in names}


class BookNet(nn.Module):

    def __init__(self, features, hidden, layers, kinds, relations, dropout):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.kind = nn.Embedding(kinds, hidden)
        self.relation = nn.Embedding(relations, hidden)
        self.message = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, kind, edges, relation, query):
        h = self.input(x) + self.kind(kind)
        source, target = edges
        gate = torch.sigmoid(self.relation(relation))
        for message, norm in zip(self.message, self.norm):
            carried = message(h)[source] * gate
            aggregate = h.new_zeros(h.shape).index_add_(0, target, carried)
            count = h.new_zeros(len(h)).index_add_(0, target, h.new_ones(len(target)))
            h = norm(h + self.dropout(torch.nn.functional.silu(aggregate / count.clamp_min(1)[:, None])))
        return self.head(h[query]).squeeze(-1)


def campaign_tensors(arrays, recipe, device):
    spatial = np.asarray(arrays['spatial'])
    kinds = np.asarray(arrays['entity_kind'])[np.asarray(arrays['entity'])]
    allowed = recipe.kind_mask()
    keep = np.isin(spatial[:, 2], recipe.codes()) & allowed[kinds[spatial[:, 0]]] & allowed[kinds[spatial[:, 1]]]
    spatial = spatial[keep]
    x = torch.as_tensor(np.asarray(arrays['x'], np.float32), device=device)
    return dict(x=x, kind=torch.as_tensor(kinds.astype(np.int64), device=device),
                edges=torch.as_tensor(spatial[:, :2].T.astype(np.int64), device=device),
                relation=torch.as_tensor(spatial[:, 2].astype(np.int64), device=device),
                query=torch.as_tensor(np.asarray(arrays['query_context'])[:, 0].astype(np.int64), device=device))


def metrics(targets, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive, negative = targets.sum(), len(targets) - targets.sum()
    auc = (ranks[targets == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    probability = 1 / (1 + np.exp(-scores))
    return dict(auc=float(auc), brier=float(np.mean((probability - targets) ** 2)),
                positives=int(positive), rows=len(targets))


def fit(directory, design='P-full', hidden=48, layers=2, dropout=0.1, epochs=12, lr=0.003, seed=0, device='cuda'):
    started = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    supervision = json.loads((directory / 'supervision.json').read_text())
    if manifest['spec_sha256'] != spec.fingerprint():
        raise ValueError('cache was built under a different graph spec')
    recipe = spec.resolve(design)
    names = spec.VIEWS['snapshot']
    holdout, trained = set(manifest['holdout_moments']), set(manifest['train_moments'])
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    targets = np.asarray([row['y'] for row in supervision['rows']], np.float32)
    train, validate = [], []
    for campaign in manifest['campaigns']:
        arrays = load_campaign(directory, campaign['index'], names)
        batch = campaign_tensors(arrays, recipe, device)
        rows = np.asarray(np.load(directory / ('campaign_%04d' % campaign['index']) / 'row_index.npy'))
        batch['target'] = torch.as_tensor(targets[rows], device=device)
        batch['name'] = campaign['name']
        if campaign['name'] in holdout:
            validate.append(batch)
        elif campaign['name'] in trained:
            train.append(batch)
    base = float(np.concatenate([b['target'].cpu().numpy() for b in train]).mean())
    model = BookNet(spec.REGISTRY.features, hidden, layers, len(spec.NODES),
                    spec.REGISTRY.relation_slots, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train:
            optimiser.zero_grad()
            loss = loss_function(model(batch['x'], batch['kind'], batch['edges'], batch['relation'], batch['query']),
                                 batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * len(batch['target'])
        LOG.info('churn epoch exit epoch=%d loss=%.5f seconds=%.1f', epoch,
                 total / sum(len(b['target']) for b in train), time.perf_counter() - started)
    model.eval()
    scores, actual = [], []
    with torch.no_grad():
        for batch in validate:
            scores.append(model(batch['x'], batch['kind'], batch['edges'], batch['relation'], batch['query']).cpu().numpy())
            actual.append(batch['target'].cpu().numpy())
    scores, actual = np.concatenate(scores), np.concatenate(actual)
    result = metrics(actual, scores)
    result.update(design=design, hidden=hidden, layers=layers, epochs=epochs, seed=seed,
                  train_base_rate=base, holdout_base_rate=float(actual.mean()),
                  flat_brier=float(np.mean((base - actual) ** 2)), flat_auc=0.5,
                  seconds=time.perf_counter() - started, holdout_moments=sorted(holdout),
                  train_moments=sorted(trained), embargo=manifest['embargo'])
    LOG.info('churn fit exit design=%s auc=%.4f brier=%.5f flat_brier=%.5f seconds=%.1f',
             design, result['auc'], result['brier'], result['flat_brier'], result['seconds'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--design', default='P-full')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--output')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = fit(args.cache, design=args.design, epochs=args.epochs)
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
