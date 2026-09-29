import numpy as np
import pytest
import torch

from graphspec.infra import workspace
from graphspec.infra.arrays import MappedRows, join_tensors
from graphspec.infra.files import ArrayStore, save_arrays
from graphspec.representations import turns
from graphspec.resources import GraphBudget, GraphLimits
from tuning.turn_inputs import prior_rows


def test_sparse_turn_lookup_matches_dense_history():
    from types import SimpleNamespace

    entity = np.array([0, 2, 0, 1, 2, 0, 3], np.int32)
    step = np.array([0, 0, 1, 2, 3, 4, 5], np.int32)
    arrays = dict(
        period=np.array([0, 0, 1, 1, 2, 2]),
        observation_ids=np.argsort(entity, kind="stable").astype(np.int32),
        entity=entity,
        step=step,
        events=np.array([[2, 0, 0, 2, 0, 0, 1], [0, 0, 6, 2, 0, 0, 2]]),
    )
    arrays.update(turns(arrays, GraphBudget(GraphLimits("cpu"))))
    held = SimpleNamespace(numpy=arrays.__getitem__)
    for actor in (False, True):
        expected = np.full((3, 5), -1, np.int32)
        for period in range(3):
            if actor:
                for row in arrays["events"]:
                    if row[2] // 2 <= arrays["turn_ends"][period]:
                        expected[period, row[0]] = row[2] // 2
            else:
                for row, (owner, moment) in enumerate(zip(entity, step)):
                    if moment <= arrays["turn_ends"][period]:
                        expected[period, owner] = row
        actual = prior_rows(held, np.arange(3)[:, None], np.arange(5)[None, :], actor)
        np.testing.assert_array_equal(actual, expected)


def test_disk_workspace_preserves_row_operations_and_cleans_files(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "BLOCK_BYTES", 24)
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    with workspace.Workspace(tmp_path, memory_bytes=0):
        joined = workspace.concatenate((values[:4], values[4:]))
        assert isinstance(joined, np.memmap)
        np.testing.assert_array_equal(joined, values)
        for indices in (np.array([8, 0, 8, 1]), np.arange(10) % 2 == 0, np.array([], dtype=np.int64)):
            np.testing.assert_array_equal(workspace.take(joined, indices), values[indices])
        np.testing.assert_array_equal(
            workspace.changed(joined, np.array([0, 2]), np.array([0, 1])), [False, True]
        )
        assert workspace.finite(joined)
        joined[-1, 0] = np.inf
        assert not workspace.finite(joined)
        for indices in (np.array([8, 0, 8, 1]), np.arange(10) % 2 == 0):
            count = int(indices.sum()) if indices.dtype == np.bool_ else len(indices)
            source = values[:count]
            order = np.arange(count)[::-1]
            expected = np.zeros_like(values)
            actual = np.zeros_like(values)
            expected[indices] = source[order]
            workspace.scatter(actual, indices, source, order)
            np.testing.assert_array_equal(actual, expected)
    assert not list(tmp_path.iterdir())


def test_disk_workspace_closes_mappings_on_failure(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        with workspace.Workspace(tmp_path, memory_bytes=0):
            workspace.allocate((10, 3), np.float32)[:] = 1
            raise RuntimeError("interrupted")
    assert not list(tmp_path.iterdir())


def test_written_array_store_publishes_after_workspace_cleanup(tmp_path):
    def write(destination):
        with workspace.Workspace(destination, memory_bytes=0):
            array = workspace.allocate((7, 2), np.int64)
            array[:] = np.arange(14).reshape(7, 2)
            save_arrays(destination, {"x": array})

    arrays = ArrayStore().open_written(tmp_path / "result", write)
    np.testing.assert_array_equal(arrays["x"], np.arange(14).reshape(7, 2))
    assert [path.name for path in (tmp_path / "result").iterdir()] == ["x.npy"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_mapped_rows_and_mixed_tables_preserve_order_duplicates_and_dtype(device):
    values = np.arange(21, dtype=np.float16).reshape(7, 3)
    mapped = MappedRows(values, device)
    resident = torch.as_tensor(values + 100, device=device)
    table = join_tensors([mapped, resident], device)
    indices = torch.tensor([13, 1, 8, 1, 0], device=device)
    expected = torch.cat([torch.as_tensor(values, device=device), resident]).index_select(0, indices)
    torch.testing.assert_close(table.index_select(0, indices), expected, rtol=0, atol=0)
    assert table.index_select(0, indices[:0]).shape == (0, 3)
    with pytest.raises(IndexError):
        mapped.index_select(0, torch.tensor([-1], device=device))
    with pytest.raises(IndexError):
        table.index_select(0, torch.tensor([14], device=device))
