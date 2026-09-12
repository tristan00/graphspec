import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from tuning.build import build
from tuning.data import Bank
from tuning.roster import groups, sample
from tuning.variants import prepare, prune, size

from .support import configured
from .test_bank import database


def test_prepare_builds_every_graph_and_prune_keeps_the_current_variant():
    root = Path(tempfile.mkdtemp())
    try:
        database(root / 'book.lbug')
        (root / 'book.task.json').write_text(json.dumps(dict(name='book', database='book.lbug', target='label')))
        build(root / 'book.task.json', root / 'bank', 1)
        bank = Bank(root / 'bank')
        first = sample('M04', np.random.default_rng(0), 5)
        first['graph'] = configured(bank.registry, {'time/history': 3}, groups('M04'))
        second = dict(first, graph=configured(bank.registry, {'time/history': 3, 'time/window': 2}, groups('M04')))
        older = prepare(root / 'bank', first, 1)
        newer = prepare(root / 'bank', second, 1)
        assert older != newer and all((folder / 'graph_0000' / 'x.npy').exists() for folder in (older, newer))
        assert prepare(root / 'bank', second, 1) == newer
        total = size(older) + size(newer)
        assert prune(root / 'bank', total, newer) == total and older.exists()
        assert prune(root / 'bank', size(newer), newer) == size(newer) and not older.exists() and newer.exists()
        assert prune(root / 'bank', 0, newer) == size(newer) and newer.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)
