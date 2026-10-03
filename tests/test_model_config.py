import copy
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from support import SEED, configuration

from conftest import SPLITS
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from tuning.bank.bank import Deriver
from tuning.batches.batching import pack_batch
from tuning.batches.packing import Packer
from tuning.families.heads import NONE, REALISATION, REALISATIONS
from tuning.families.layers import CausalAttention, TimeEncoding
from tuning.families.model_spec import configured, samples, space, views
from tuning.families.models import build
from tuning.families.periods import blocks as period_blocks

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("model_config")


class Stated:
    version = 1
    reads = ("kind", "targets", "parameters")
    together = False
    back = False

    @timed("model_config.Stated.refusal")
    def refusal(self, schema):
        return None

    @timed("model_config.Stated.module")
    def module(self, hidden, width, spec, registry):
        return torch.nn.Identity()


def registry(events=True, signals=2, decided=()):
    picking = dict(pick=dict(roles=dict(at=dict(sort="node", kinds=["unit"])), parameters=[]))
    target = dict(kind="number", actions=picking, nothing=None)
    kinds = dict(ask=dict(target=target, references=("unit",), absent="stop"))
    return SimpleNamespace(
        name="stated",
        extras=dict(events=events, signals=signals),
        schema=SimpleNamespace(
            tasks=dict(kinds=kinds, references=dict(unit=dict(sort="node"))),
            targets=lambda: ("number",),
            decided=lambda: decided,
        ),
    )


@pytest.fixture(scope="module")
def opened(bank):
    return Deriver(bank, "cpu", SEED, RESIDENT)


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


def test_the_check_states_every_value_of_a_choice_without_a_default_and_starts_its_switches_from_each():
    started = time.perf_counter()
    REALISATIONS.register("stated", Stated())
    try:
        offered = samples(registry(decided=("ask",)))
    finally:
        del REALISATIONS.held["stated"]
    starts = {}
    for model in offered:
        rest = tuple(sorted((name, value) for name, value in model.items() if name != REALISATION))
        starts.setdefault(model[REALISATION], set()).add(rest)
    assert sorted(starts) == sorted([NONE, "stated", *REALISATIONS.shipped])
    assert all(found == starts[NONE] for found in starts.values())
    LOG.info("samples of a choice without a default exit seconds=%.3f", time.perf_counter() - started)


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
    stated = registry()
    stated.extras["several"] = False
    stated.schema.columns, stated.schema.holds, stated.schema.edges, stated.schema.sets = {}, {}, {}, {}
    stated.schema.ties, stated.schema.read = None, list
    bank.registry, bank.manifest = SimpleNamespace(**vars(stated)), dict(spec_sha256="stated")
    bank.cache = Path(tmp_path)
    bank.variants, bank.device, bank.directory, bank.digests = {}, "cpu", Path(tmp_path), {}
    bank.source, bank.choices, bank.seed = "source", {}, SEED
    monkeypatch.setattr(
        specs,
        "settings",
        lambda declaration, groups: (
            specs.number("time/history", 1, 32, 16),
            specs.choice("time/steps", ("tasks",), "tasks"),
        ),
    )
    cfg = {
        "model": {"view": "signal"},
        "pipeline": [],
        "graph": {"time/history": 1, "time/steps": "tasks"},
        "splits": SPLITS,
    }
    assert bank.variant(cfg)[2]["time/history"] == 1
    cfg["graph"]["time/history"] = 2
    assert bank.variant(cfg)[2]["time/history"] == 2


def counted(monkeypatch):
    started = time.perf_counter()
    found, original = [], specs.digest

    def digest(value):
        found.append(value)
        return original(value)

    monkeypatch.setattr(specs, "digest", digest)
    LOG.debug("counted exit seconds=%.4f", time.perf_counter() - started)
    return found


def test_a_bank_digests_each_part_of_a_configuration_once_and_again_only_when_it_changes(bank, monkeypatch):
    started = time.perf_counter()
    held = Deriver(bank, "cpu", SEED, RESIDENT)
    cfg = configuration(held.registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS)
    key, _, spec, _ = held.stated(cfg)
    folder, labels = held.viewed(cfg).folder, held.labels(0, cfg)
    assert key[1] == specs.digest(dict(pipeline=cfg["pipeline"], graph=cfg["graph"]))
    found = counted(monkeypatch)
    for _ in range(20):
        assert held.stated(cfg)[0] == key and held.viewed(cfg).folder == folder and held.labels(0, cfg) is labels
        assert held.looking(cfg) is held.looking(dict(cfg))
    assert found == []
    copied = copy.deepcopy(cfg)
    assert held.stated(copied)[0] == key and held.viewed(copied).folder == folder and len(found) > 0
    counts = len(found)
    held.stated(copied)
    assert len(found) == counts
    cfg["graph"]["time/history"] = 5
    assert held.stated(cfg)[0] == (key[0], specs.digest(dict(pipeline=cfg["pipeline"], graph=cfg["graph"])), key[2])
    assert held.stated(cfg)[2]["time/history"] == 5 and held.stated(cfg)[0] != key
    LOG.info("digested once exit seconds=%.4f", time.perf_counter() - started)


def recorded(module, name, monkeypatch):
    started = time.perf_counter()
    found, original = [], getattr(module, name)

    def wrapped(*arguments):
        clock = time.perf_counter()
        result = original(*arguments)
        found.append((arguments, result))
        LOG.debug("recorded %s exit seconds=%.6f", name, time.perf_counter() - clock)
        return result

    monkeypatch.setattr(module, name, wrapped)
    LOG.debug("recorded exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return found


def test_a_model_space_is_built_once_and_a_bank_derives_the_representations_of_a_model_once(bank, monkeypatch):
    from graphspec import registries
    from tuning.families import model_spec

    started = time.perf_counter()
    held = Deriver(bank, "cpu", SEED, RESIDENT)
    assert space(held.registry) is space(held.registry) is space(Deriver(bank, "cpu", SEED, RESIDENT).registry)
    cfg = configuration(held.registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS)
    found = recorded(model_spec, "derived", monkeypatch)
    chosen = [held.stated(cfg)[0][0] for _ in range(20)]
    assert len(found) == 1 and chosen == [found[0][1]] * 20
    cfg["model"]["snapshot/temporal"] = "coupling_edges"
    assert held.chosen(cfg["model"]) == ("coupling",) and len(found) == 2
    registries.changed()
    assert held.chosen(cfg["model"]) == ("coupling",) and len(found) == 3
    LOG.info("derived once exit seconds=%.4f", time.perf_counter() - started)


def test_probing_what_an_operation_does_with_kinds_derives_each_specification_of_its_parameters_once(
    bank, monkeypatch
):
    from support import template

    from graphspec.operations import pipeline
    from tuning.bank.bank import Bank

    started = time.perf_counter()
    registry = Bank(bank).registry
    stated = template()
    offered = pipeline.Offered(registry, stated, pipeline.space(registry, stated))
    names = [name for name in offered.names if offered.probes[name] is not None and not offered.reads[name]]
    assert names
    for name in names:
        bounded = {held.name[len(pipeline.NAMED % name) :]: held for held in offered.owned[name]}
        del pipeline.PROBED[registry]
        found = recorded(pipeline, "made", monkeypatch)
        operation = offered.operations[name]
        assert pipeline.probed(registry, operation, bounded, {}) == offered.probes[name]
        assigned = [tuple(map(arguments[3], operation.stated(registry))) for arguments, _ in found]
        assert len(assigned) == len(set(assigned)) > 0
        monkeypatch.undo()
    LOG.info("probed once exit seconds=%.4f", time.perf_counter() - started)


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
    assert prediction.shape == batch["series"].shape
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
    monkeypatch.setattr("tuning.batches.packing.wait", lambda handles: observed.append(handles) or [41])
    with pytest.raises(RuntimeError, match="exited before returning"):
        packer.receive()
    assert observed == [[41]]
