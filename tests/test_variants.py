import shutil

from tuning.data import Bank
from graph_tuning_infra.files import folder_size as size
from tuning.variants import prepare, prune

from .support import config, configured


def test_prepare_builds_every_graph_and_prune_keeps_the_current_variant(book, tmp_path):
    root = tmp_path
    shutil.copytree(book, root / 'bank')
    bank = Bank(root / 'bank')
    first = config(bank.registry, {'view': 'snapshot'}, {'time/history': 3})
    second = dict(first, graph=configured(bank.registry, {'time/history': 3, 'time/window': 2}, ()))
    older = prepare(root / 'bank', first, 1)
    newer = prepare(root / 'bank', second, 1)
    assert older != newer and all((folder / 'graph_0000' / 'x.npy').exists() for folder in (older, newer))
    assert prepare(root / 'bank', second, 1) == newer
    total = size(older) + size(newer)
    assert prune(root / 'bank', total, newer) == total and older.exists()
    assert prune(root / 'bank', size(newer), newer) == size(newer) and not older.exists() and newer.exists()
    assert prune(root / 'bank', 0, newer) == size(newer) and newer.exists()
