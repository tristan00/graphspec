from graphspec import Domain
from . import spec


NAMES = ('Book RGCN', 'Book GraphSAGE', 'Renewal-chain gated', 'Renewal-chain attention')
GRAPHS = dict(C01='book_graph', C02='P-full', C03='R-chain', C04='R-chain')

INPUTS = {
    'snapshot': ('C01', 'C02', 'C03', 'C04'),
    'lifted': ('C03', 'C04'),
}
BLOCKS = {model: frozenset(name for name, models in INPUTS.items() if model in models) for model in GRAPHS}

VIEWS = spec.VIEWS

DOMAIN = Domain(spec.REGISTRY, GRAPHS, BLOCKS, (), VIEWS)
DESIGNS = DOMAIN.designs
