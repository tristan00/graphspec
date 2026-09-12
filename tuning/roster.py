import math

from graphspec.spec import GROUPS

from .history_models import AnonymousWalk, HistoryModel, TemporalAttention
from .memory_models import MemoryModel, VirtualState, WalkProjection
from .snapshot_models import EventDAG, MixedModel, SignalGraph, SnapshotModel


NAMES = (
    'Relational snapshot GRU', 'Relational snapshot LSTM', 'Relational snapshot Transformer', 'GCN-GRU', 'GraphSAGE-GRU', 'RGCN-GRU',
    'RGCN-TCN', 'T-GCN', 'DCRNN', 'EvolveGCN-H', 'DySAT', 'MPNN over derived nodes', 'Entity-coupling gated',
    'Entity-coupling attention', 'Event-graph gated', 'Event-graph attention', 'TGAT', 'Typed TGN',
    'JODIE', 'DyRep', 'GraphMixer', 'DyGFormer', 'HyperEvent', 'TPNet', 'CAWN', 'TNCN',
    'DyGMamba', 'NAViS', 'CTAN', 'MTGNN')
MODELS = tuple('M%02d' % i for i in range(1, len(NAMES) + 1))
INPUTS = {
    'snapshot': tuple('M%02d' % i for i in list(range(1, 17)) + [20, 29]),
    'lifted': ('M13', 'M14'),
    'mixed': ('M12',),
    'events': ('M15', 'M16', 'M18', 'M19', 'M20', 'M24', 'M26', 'M28', 'M29'),
    'dag': ('M15', 'M16'),
    'tree': ('M17',),
    'history': ('M18', 'M20', 'M21', 'M22', 'M23', 'M24', 'M26', 'M27'),
    'structure': ('M23',),
    'history_targets': ('M21', 'M22', 'M23', 'M27'),
    'turn_history': ('M17', 'M21', 'M22', 'M23', 'M27'),
    'walks': ('M25',),
}
BLOCKS = {model: frozenset(name for name, members in INPUTS.items() if model in members) for model in MODELS}
REPRESENTATIONS = dict(lifted='coupling', dag='events', tree='neighbors', history='neighbors', walks='walks', mixed='mixed')


def groups(model):
    wanted = {REPRESENTATIONS[block] for block in BLOCKS[model] if block in REPRESENTATIONS}
    return tuple(group for group in GROUPS if group in wanted)


def models(registry):
    extras = registry.extras
    excluded = set() if extras['signals'] else {'M30'}
    if extras['level'] == 'node':
        excluded.add('M28')
    if not extras['queries']:
        excluded.add('M19')
    return tuple(model for model in MODELS if model not in excluded)


def sample(model, rng, seed):
    cfg = dict(model=model, hidden=int(rng.choice([16, 24, 32])), layers=int(rng.choice([1, 2])),
        heads=int(rng.choice([1, 2])), dropout=float(rng.choice([0.0, 0.1])),
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


def create(cfg, registry, categories):
    family = int(cfg['model'][1:])
    if cfg['model'] not in models(registry):
        raise ValueError('model %s cannot run on this bank' % cfg['model'])
    if family in (*range(1, 12), 13, 14):
        return SnapshotModel(cfg, registry)
    if family == 12:
        return MixedModel(cfg, registry)
    if family in (15, 16):
        return EventDAG(cfg, registry)
    if family == 17:
        return TemporalAttention(cfg, registry)
    if family in (18, 19, 20, 26, 29):
        return MemoryModel(cfg, registry)
    if family in (21, 22, 23, 27):
        return HistoryModel(cfg, registry)
    if family == 24:
        return WalkProjection(cfg, registry)
    if family == 25:
        return AnonymousWalk(cfg, registry)
    if family == 28:
        return VirtualState(cfg, registry, categories)
    return SignalGraph(cfg, registry)
