import math

from .history_models import AnonymousWalk, HistoryModel, TemporalAttention
from .memory_models import MemoryModel, VirtualState, WalkProjection
from .snapshot_models import EventDAG, MixedModel, SignalGraph, SnapshotModel


NAMES = (
    'Mapgraph-GRU', 'Mapgraph-LSTM', 'Mapgraph-Transformer', 'GCN-GRU', 'GraphSAGE-GRU', 'RGCN-GRU',
    'RGCN-TCN', 'T-GCN', 'DCRNN', 'EvolveGCN-H', 'DySAT', 'Temporal-graph MPN', 'Temporal-edge gated',
    'Temporal-edge attention', 'Event-DAG gated', 'Event-DAG attention', 'TGAT', 'Typed TGN',
    'JODIE reward', 'DyRep reward', 'GraphMixer', 'DyGFormer', 'HyperEvent', 'TPNet', 'CAWN', 'TNCN',
    'DyGMamba', 'NAViS reward', 'CTAN', 'MTGNN reward')
DESIGNS = {f'M{i:02d}': ['S-full'] for i in range(1, 12)}
DESIGNS.update(M08=['S-core', 'S-full'], M12=['O-mixed'], M13=['X-chain', 'X-skip', 'X-allpast'],
    M14=['X-chain', 'X-skip', 'X-allpast'], M15=['D-event'], M16=['D-event'], M17=['E-entity'],
    M18=['E-entity'], M19=['B-role'], M20=['E-entity'], M21=['H-neighbor'], M22=['H-neighbor'],
    M23=['H-neighbor'], M24=['W-temporal'], M25=['W-temporal'], M26=['H-neighbor'], M27=['H-neighbor'],
    M28=['N-affinity'], M29=['E-entity'], M30=['V-signals'])
for i in range(1, 12):
    DESIGNS[f'M{i:02d}'].extend(design + '+colocated' for design in tuple(DESIGNS[f'M{i:02d}']))
for model in ('M13', 'M14'):
    DESIGNS[model].extend(design + '+turn-boundary' for design in tuple(DESIGNS[model]))
for model in ('M15', 'M16'):
    DESIGNS[model].append('D-event+turn-boundary')


def sample(model, design, rng, seed):
    cfg = dict(model=model, design=design, hidden=int(rng.choice([16, 24, 32])), layers=int(rng.choice([1, 2])),
        heads=int(rng.choice([1, 2])), dropout=float(rng.choice([0.0, 0.1])), history=16,
        neighbors=int(rng.choice([4, 8])), seed=int(seed), lr=float(math.exp(rng.uniform(math.log(0.0003), math.log(0.002)))),
        weight_decay=float(math.exp(rng.uniform(math.log(1e-6), math.log(1e-3)))), clip=1.0,
        diffusion_hops=1, patch=2, state_size=4, walks=4, walk_hops=3,
        projection_hops=2, projection_width=16, decay=0.05, affinity_buffer=4, signal_topk=3,
        integration_steps=2, damping=0.1)
    if model in ('M09', 'M30'):
        cfg['diffusion_hops'] = int(rng.choice([1, 2]))
    if model == 'M24':
        cfg.update(projection_hops=int(rng.choice([1, 2])), projection_width=int(rng.choice([16, 32])), decay=float(rng.choice([0.01, 0.05, 0.1])))
    if model == 'M25':
        cfg.update(walks=int(rng.choice([2, 4])), walk_hops=int(rng.choice([2, 3])))
    if model == 'M27':
        cfg['state_size'] = int(rng.choice([4, 8]))
    return cfg


def create(cfg, categories):
    family = int(cfg['model'][1:])
    if cfg['design'] not in DESIGNS[cfg['model']]:
        raise ValueError('model and cache design are incompatible')
    if family in (*range(1, 12), 13, 14):
        return SnapshotModel(cfg)
    if family == 12:
        return MixedModel(cfg)
    if family in (15, 16):
        return EventDAG(cfg)
    if family == 17:
        return TemporalAttention(cfg)
    if family in (18, 19, 20, 26, 29):
        return MemoryModel(cfg)
    if family in (21, 22, 23, 27):
        return HistoryModel(cfg)
    if family == 24:
        return WalkProjection(cfg)
    if family == 25:
        return AnonymousWalk(cfg)
    if family == 28:
        return VirtualState(cfg, categories)
    if family == 30:
        return SignalGraph(cfg)
    raise ValueError('unknown model ' + cfg['model'])
