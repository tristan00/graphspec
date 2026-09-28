import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graphspec.infra.arrays import Bundle, Slot, shared, slot_bytes, views
from tuning.batching import batches, recipe_batch
from tuning.packing import Packer
from tuning.tensors import materialise
from tuning.turns import blocks as turn_blocks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefetch_window_matches_distinct_graph_budget(device):
    members = [{0, 1}, {1}, {2, 3}, {0, 2}, {4}, {1, 4}, {0, 3}, {2}, {2, 4}]
    groups = [[(graph,) for graph in sorted(group)] for group in members]
    packer = Packer.__new__(Packer)
    packer.position, packer.device, packer.cfg = 0, device, {}
    copied = []
    packer.bank = SimpleNamespace(
        residents=SimpleNamespace(limit=6 if device == "cuda" else 3),
        resident=lambda graph, cfg, device: SimpleNamespace(tensor=lambda key: copied.append((graph, key))),
    )

    class Gate:
        def __init__(self):
            self.index = -1

        def __enter__(self):
            self.index += 1

        def __exit__(self, *args):
            return False

        def wait_for(self, ready):
            while True:
                expected = (
                    self.index <= packer.position
                    or len(set().union(*members[packer.position : self.index + 1])) <= 3
                )
                assert ready() == expected
                if expected:
                    return
                packer.position += 1

    packer.condition = Gate()
    failures = []
    packer.prefetch(groups, {"x", "entity_kind"}, threading.Event(), failures)
    assert not failures and copied and packer.position > 0


def test_turn_context_preserves_interleaved_windows():
    class ContextBank:
        def __init__(self):
            self.calls = []

        def arrays(self, graph, cfg):
            self.calls.append(graph)
            return dict(turn_membership=np.array([0, 0, 1, 1, 2]), turn_starts=np.array([0, 2, 4]))

    bank = ContextBank()
    original = [(0, 4, 5, 4, 0, 3, 5), (1, 2, 5, 4, 1, 0, 2), (0, 0, 3, 2, 0, 1, 3)]
    for context in (1, 2, 8):
        bank.calls.clear()
        actual = turn_blocks(bank, original, {"graph": {"time/context": context}})
        assert actual == [
            (0, 2 if context == 1 else 0, 5, 4, 0, 3, 5),
            (1, 0, 5, 4, 1, 0, 2),
            (0, 0, 3, 2, 0, 1, 3),
        ]
        assert bank.calls == [0, 1]
    assert turn_blocks(bank, original, {"graph": {"time/context": 0}}) is original


def same(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def held_arrays():
    return [
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.array([2**40], np.int64),
        np.zeros((0, 4), np.float32),
        np.float64(1.5),
        np.array([True, False]),
        np.arange(5, dtype=np.int32),
        np.arange(3, dtype=np.float32) / 7,
    ]


def test_bundle_write_matches_pack():
    bundle = Bundle("cpu")
    for array in held_arrays():
        bundle.hold(array)
    arrays, spans, total = bundle.layout()
    assert all(offset % 8 == 0 for offset, _ in arrays) and total == arrays[-1][0] + arrays[-1][1].nbytes + 8
    buffer = np.full(total + 100, 255, np.uint8)
    bundle.write(buffer, arrays)
    for original, tensor in zip(held_arrays(), views(torch.from_numpy(buffer), spans)):
        expected = torch.as_tensor(original)
        assert torch.equal(tensor, expected.long() if expected.dtype == torch.int32 else expected)
    packed = Bundle("cpu")
    for array in held_arrays():
        packed.hold(array)
    packed_buffer, packed_spans = packed.pack()
    assert packed_spans == spans and len(packed_buffer) == total
    assert all(
        same(x, y)
        for x, y in zip(views(torch.from_numpy(packed_buffer), spans), views(torch.from_numpy(buffer), spans))
    )
    assert (
        spans[0][2] == torch.int32
        and spans[0][4] == torch.long
        and spans[1][2] == torch.long
        and spans[5][2] == torch.int32
        and spans[5][4] == torch.long
    )


def test_slot_transfer_and_growth():
    assert slot_bytes(1) == 1 << 20 and slot_bytes(100 << 20) == 125 << 20
    tensor = shared(1 << 20)
    tensor.numpy()[:] = np.arange(1 << 20, dtype=np.uint32).view(np.uint8)[: 1 << 20]
    slot = Slot(tensor, "cuda")
    moved = slot.transfer(4096, "cuda")
    assert moved.device.type == "cuda" and torch.equal(moved.cpu(), tensor[:4096])
    tensor.numpy()[:8] = 7
    assert int(moved[0]) != 7 and int(slot.transfer(8, "cuda")[0]) == 7
    slot.release()


@pytest.mark.parametrize("view", ("snapshot", "event_graph"))
def test_packer_matches_single_process(packers, view):
    cfg, packer = packers[view]
    opened = packer.bank
    blocks = turn_blocks(opened, opened.blocks(0, cfg["graph"]["time/history"], cfg), cfg)
    groups = batches(blocks, 2, opened.sizes(cfg), 5, frame_budget=1024)[:5]
    assert len(groups) == 5
    for group, streamed in zip(groups, packer.stream(groups), strict=True):
        assert same(materialise(opened, cfg, recipe_batch(opened, group, cfg), "cuda"), streamed)
    packer.close()
    assert len(packer.slots) == 2 and packer.slots[0].tensor.numel() >= 1 << 20
