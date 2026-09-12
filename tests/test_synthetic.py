from pathlib import Path
import shutil
import tempfile

import numpy as np

from graphspec.synthetic import generate
from tuning.build import build
from tuning.data import Bank, pack
from tuning.roster import groups, models, sample

from .support import configured


def test_generated_databases_follow_the_contract_at_every_level():
    root = Path(tempfile.mkdtemp())
    try:
        for level in ('query', 'node', 'graph'):
            task = generate(root / level, level, 5, level, 5, entities=60, steps=10, features=8, signals=2 if level == 'query' else 0,
                            degree=2.0, max_degree=4, alive=0.8, lifetime=3.0, period=3)
            manifest = build(task, root / level / 'bank', 1)
            bank = Bank(root / level / 'bank')
            assert manifest['target']['level'] == level and len(manifest['graphs']) == 5 and manifest['rows'] >= 3
            assert ('M30' in models(bank.registry)) == (level == 'query') and ('M19' in models(bank.registry))
            for model in ('M01', 'M15', 'M17', 'M21'):
                cfg = sample(model, np.random.default_rng(0), 5)
                cfg['graph'] = configured(bank.registry, {'time/history': 4}, groups(model))
                block = bank.blocks(0, 4)[-1]
                b = pack(bank, block, cfg, 'cpu')
                assert b['predictions'] >= 1 and int(b['selected'].sum()) >= 1, (level, model)
    finally:
        shutil.rmtree(root, ignore_errors=True)
