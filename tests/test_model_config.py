import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import SPLITS
from graphspec import spec as specs
from graphspec.infra.arrays import RESIDENT
from tuning.batching import pack_batch
from tuning.data import Deriver
from tuning.layers import CausalAttention, TimeEncoding
from tuning.model_spec import configuration, configured, samples, space, views
from tuning.models import build
from tuning.packing import Packer
from tuning.periods import blocks as period_blocks


def registry(events=True, signals=2):
    kinds = dict(ask=dict(target=dict(kind="number"), references=("unit",), absent="stop"))
    return SimpleNamespace(
        name="stated",
        extras=dict(events=events, signals=signals, target="number"),
        schema=SimpleNamespace(tasks=dict(kinds=kinds, references=dict(unit=dict(sort="node")))),
    )


@pytest.fixture(scope="module")
def opened(bank):
    return Deriver(bank, "cpu", RESIDENT)


@pytest.mark.parametrize(
    "values",
    (
        {"view": "invalid"},
        {"view": "tree", "tree/depth": 0},
        {"view": "tree", "tree/depth": True},
        {"view": "tree", "sequence/operator": "mixer"},
        {"view": "tree", "unknown": 1},
        {"view": "snapshot", "dropout": float("nan")},
        {"view": "tree", "hidden": 0},
        {"view": "tree", "hidden": 129},
        {"view": "tree", "hidden": 1.5},
        {"view": "tree", "hidden": True},
    ),
)
def test_model_overrides_reject_invalid_and_inactive_values(values):
    with pytest.raises(ValueError):
        configured(registry(), values)


def test_all_model_samples_are_canonical_and_cover_each_available_view():
    declaration = registry()
    offered = samples(declaration)
    assert {model["view"] for model in offered} == set(views(declaration))
    assert all(specs.canonical(space(declaration), model) == model for model in offered)
    assert {model["memory/update"] for model in offered if model["view"] == "memory"} == {
        "mailbox_gru",
        "mutual_rnn",
        "intensity",
        "antisymmetric",
    }
    assert {model["sequence/operator"] for model in offered if model["view"] == "sequence"} == {
        "mixer",
        "transformer",
        "ssm",
    }


def test_signal_is_offered_where_tasks_carry_signals_and_affinity_where_tasks_are_seen_as_events():
    assert {"signal", "affinity"} <= set(views(registry()))
    assert "affinity" not in views(registry(False)) and "signal" in views(registry(False))
    assert "signal" not in views(registry(True, 0)) and "affinity" in views(registry(True, 0))


def test_antisymmetric_state_readout_has_no_unused_time_encoding():
    model = configured(
        registry(), {"view": "memory", "memory/update": "antisymmetric", "memory/embedding": "state"}
    )
    assert "time/encoding" not in model


def test_variant_cache_reads_mutated_configuration(monkeypatch, tmp_path):
    bank = Deriver.__new__(Deriver)
    bank.registry = registry()
    bank.cache = Path(tmp_path)
    bank.variants, bank.roots, bank.device, bank.directory = {}, {}, "cpu", Path(tmp_path)
    monkeypatch.setattr(
        specs, "space", lambda declaration, groups: (specs.number("time/history", 1, 32, 16),)
    )
    cfg = {"model": {"view": "signal"}, "graph": {"time/history": 1}, "splits": SPLITS}
    assert bank.variant(cfg)[1]["time/history"] == 1
    cfg["graph"]["time/history"] = 2
    assert bank.variant(cfg)[1]["time/history"] == 2


def test_encoding_and_attention_reject_invalid_parameters_at_construction():
    with pytest.raises(ValueError, match="unknown time encoding"):
        TimeEncoding(8, "invalid")
    with pytest.raises(ValueError, match="positive"):
        CausalAttention(7, heads=0)


@pytest.mark.parametrize(
    "model",
    [
        {"view": name}
        for name in (
            "snapshot",
            "mixed",
            "event_graph",
            "memory",
            "projection",
            "sequence",
            "tree",
            "walks",
            "affinity",
            "signal",
        )
    ]
    + [
        {"view": "snapshot", "snapshot/temporal": "coupling_edges", "keys/embedding": True},
        {"view": "tree", "keys/embedding": True},
    ],
)
@pytest.mark.parametrize("hidden", [1, 7, 128])
def test_each_model_produces_finite_predictions_for_its_labels(opened, model, hidden):
    cfg = configuration(
        opened.registry, dict(model, hidden=hidden), {"time/history": 2, "time/context": 1}, 11, SPLITS
    )
    blocks = opened.blocks(0, 2, cfg)
    batch = pack_batch(opened, period_blocks(opened, [blocks[-1]], cfg), cfg, "cpu")
    net = build(cfg, opened.fitted(cfg)).eval()
    with torch.inference_mode():
        prediction = net(batch)
    assert prediction.shape == batch["target"].shape
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("pending", (0, 1))
def test_packer_close_joins_owned_processes_before_releasing_slots(pending):
    events = []
    packer = Packer.__new__(Packer)
    packer.closed = False
    packer.pending = pending
    packer.condition = threading.Condition()
    packer.loading_done = SimpleNamespace(set=lambda: events.append("prefetch_cancel"))
    packer.loader = SimpleNamespace(join=lambda: events.append("prefetch_join"))
    packer.processes = [
        SimpleNamespace(
            is_alive=lambda: True,
            kill=lambda: events.append("kill"),
            join=lambda: events.append("join"),
            close=lambda: events.append("process_close"),
        )
    ]
    packer.jobs = SimpleNamespace(
        put=lambda value: events.append(("put", value)),
        cancel_join_thread=lambda: events.append("cancel_feeder"),
        close=lambda: events.append("queue_close"),
        join_thread=lambda: events.append("join_feeder"),
    )
    packer.results = [SimpleNamespace(close=lambda: events.append("channel_close"))]
    packer.slots = {
        0: SimpleNamespace(
            release=lambda: events.append("release"),
            tensor=SimpleNamespace(numel=lambda: 8),
        )
    }
    packer.close()
    expected = ["prefetch_cancel", "prefetch_join"]
    expected += ["kill", "cancel_feeder"] if pending else [("put", None)]
    expected += ["join", "process_close", "queue_close"]
    if not pending:
        expected.append("join_feeder")
    expected += ["channel_close", "release"]
    assert events == expected
    packer.close()
    assert events == expected


def test_packer_waits_on_worker_exit_events_without_polling(monkeypatch):
    packer = Packer.__new__(Packer)
    packer.results = []
    packer.processes = [SimpleNamespace(sentinel=41, pid=12, exitcode=3)]
    observed = []
    monkeypatch.setattr("tuning.packing.wait", lambda handles: observed.append(handles) or [41])
    with pytest.raises(RuntimeError, match="exited before returning"):
        packer.receive()
    assert observed == [[41]]
