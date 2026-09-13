import pytest

from graphspec.synthetic import generate
from tuning.build import build
from tuning.data import Bank, pack
from tuning.model_spec import views

from .support import config


def test_generated_databases_follow_the_contract_at_every_level(tmp_path):
    root = tmp_path
    with pytest.raises(ValueError):
        generate(root / 'few', 'few', 3, 'graph', 5, entities=40, steps=10, features=8, signals=0, degree=2.0, max_degree=4, alive=0.8, lifetime=3.0, period=3)
    with pytest.raises(ValueError):
        generate(root / 'short', 'short', 5, 'node', 5, entities=40, steps=5, features=8, signals=0, degree=2.0, max_degree=4, alive=0.8, lifetime=3.0, period=4)
    if True:
        for level in ('query', 'node', 'graph'):
            task = generate(root / level, level, 5, level, 5, entities=60, steps=10, features=8, signals=2 if level == 'query' else 0,
                            degree=2.0, max_degree=4, alive=0.8, lifetime=3.0, period=3)
            manifest = build(task, root / level / 'bank', 1)
            bank = Bank(root / level / 'bank')
            assert manifest['target']['level'] == level and len(manifest['graphs']) == 5 and manifest['rows'] >= 3
            assert ('signal' in views(bank.registry)) == (level == 'query') and ('affinity' in views(bank.registry)) == (level != 'node')
            for values in ({'view': 'snapshot'}, {'view': 'event_graph'}, {'view': 'tree'}, {'view': 'sequence', 'sequence/operator': 'mixer'}):
                cfg = config(bank.registry, values, {'time/history': 4})
                block = bank.blocks(0, 4)[-1]
                b = pack(bank, block, cfg, 'cpu')
                assert b['predictions'] >= 1 and int(b['selected'].sum()) >= 1, (level, values)
