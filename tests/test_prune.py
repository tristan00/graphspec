import os
import time

import pytest

from graph_tuning_infra import files


def variant(root, name, graphs, size, age):
    folder = root / name
    for index in range(graphs):
        (folder / ('graph_%04d' % index)).mkdir(parents=True)
        (folder / ('graph_%04d' % index) / 'x.npy').write_bytes(b'x' * size)
    (folder / 'variant.json').write_text('{}')
    os.utime(folder / 'variant.json', (time.time() - age, time.time() - age))
    return folder


def test_prune_moves_oldest_to_trash_and_sweeps(tmp_path):
    root = tmp_path / 'variants'
    old, middle, keep = variant(root, 'old', 3, 100, 300), variant(root, 'middle', 2, 100, 200), variant(root, 'keep', 4, 100, 100)
    assert files.prune(root, 10000, keep, 'variant.json') == (906, 0)
    assert files.prune(root, 650, keep, 'variant.json') == (604, 1)
    assert not old.exists() and middle.exists() and keep.exists()
    files.SWEEPERS[tmp_path / 'trash'].join(10)
    assert list((tmp_path / 'trash').iterdir()) == []
    assert files.prune(root, 0, keep, 'variant.json') == (402, 1)
    files.SWEEPERS[tmp_path / 'trash'].join(10)
    assert sorted(path.name for path in root.iterdir()) == ['keep'] and not list((tmp_path / 'trash').iterdir())


def test_cached_size_follows_new_entries(tmp_path):
    folder = variant(tmp_path / 'variants', 'grown', 2, 50, 10)
    assert files.cached_size(folder) == 102
    (folder / 'graph_0000' / 'x.npy').write_bytes(b'x' * 500)
    assert files.cached_size(folder) == 102
    (folder / 'graph_0002').mkdir()
    (folder / 'graph_0002' / 'x.npy').write_bytes(b'x' * 25)
    assert files.cached_size(folder) == 577


def test_prune_rejects_unstamped_folders(tmp_path):
    root = tmp_path / 'variants'
    keep = variant(root, 'keep', 1, 10, 0)
    (root / 'stray').mkdir()
    with pytest.raises(ValueError, match='stray'):
        files.prune(root, 0, keep, 'variant.json')
